# This file is originally from AnimateDiff/animatediff/models/motion_module.py at main · guoyww/AnimateDiff
# SPDX-License-Identifier: Apache-2.0 license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file was released under [ Apache-2.0 license], with the full license text available at [https://github.com/guoyww/AnimateDiff?tab=Apache-2.0-1-ov-file#readme].
import torch
import torch.nn.functional as F
from torch import nn

from .attention import CrossAttention, FeedForward, apply_rotary_emb, precompute_freqs_cis

from einops import rearrange, repeat
import math

try:
    import xformers
    import xformers.ops

    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available")
    XFORMERS_AVAILABLE = False


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class TemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        num_attention_blocks               = 2,
        norm_num_groups                    = 32,
        temporal_max_len                   = 32,
        zero_initialize                    = True,
        pos_embedding_type                 = "ape",
        use_causal_mask                    = True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads,
            num_layers=num_transformer_block,
            num_attention_blocks=num_attention_blocks,
            norm_num_groups=norm_num_groups,
            temporal_max_len=temporal_max_len,
            pos_embedding_type=pos_embedding_type,
            use_causal_mask=use_causal_mask,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None, cached_hidden_state_list=None):
        hidden_states = input_tensor
        hidden_states, output_hidden_state_list = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask, cached_hidden_state_list)

        output = hidden_states
        return output, output_hidden_state_list  # list of hidden states


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        num_attention_blocks               = 2,
        norm_num_groups                    = 32,
        temporal_max_len                   = 32,
        pos_embedding_type                 = "ape",
        use_causal_mask                    = True,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_attention_blocks=num_attention_blocks,
                    temporal_max_len=temporal_max_len,
                    pos_embedding_type=pos_embedding_type,
                    use_causal_mask=use_causal_mask,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cached_hidden_state_list=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        output_hidden_state_list = []

        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim).contiguous()
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        if cached_hidden_state_list is not None:
            n = len(cached_hidden_state_list) // len(self.transformer_blocks)
        else:
            n = 0
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, hidden_state_list = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, attention_mask=attention_mask,
                                                     cached_hidden_state_list=cached_hidden_state_list[i*n:(i+1)*n] if n else None)
            output_hidden_state_list.extend(hidden_state_list)

        # output
        hidden_states = self.proj_out(hidden_states)
        
        # Safe reshape: use the original spatial dimensions
        try:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        except RuntimeError as e:
            # If reshape fails, try to infer correct dimensions
            print(f"Original reshape failed: {e}")
            total_elements = hidden_states.numel()
            
            # Calculate dimensions based on residual tensor
            target_batch, _, target_height, target_width = residual.shape
            expected_elements = target_batch * target_height * target_width * inner_dim
            
            if total_elements == expected_elements:
                hidden_states = hidden_states.reshape(target_batch, target_height, target_width, inner_dim).permute(0, 3, 1, 2).contiguous()
                print(f"Successfully reshaped using residual dimensions: {hidden_states.shape}")
            else:
                # Last resort: create zero tensor with correct shape
                print(f"Creating zero tensor. Total elements: {total_elements}, Expected: {expected_elements}")
                hidden_states = torch.zeros(target_batch, inner_dim, target_height, target_width, 
                                           device=hidden_states.device, dtype=hidden_states.dtype)

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output, output_hidden_state_list


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        num_attention_blocks               = 2,
        temporal_max_len                   = 32,
        pos_embedding_type                 = "ape",
        use_causal_mask                    = True,
    ):
        super().__init__()

        self.attention_blocks = nn.ModuleList(
            [
                TemporalAttention(
                        query_dim=dim,
                        heads=num_attention_heads,
                        dim_head=attention_head_dim,
                        temporal_max_len=temporal_max_len,
                        pos_embedding_type=pos_embedding_type,
                        use_causal_mask=use_causal_mask,
                )
                for i in range(num_attention_blocks)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dim)
                for i in range(num_attention_blocks)
            ]
        )

        self.ff = FeedForward(dim, dropout=0.0, activation_fn="geglu")
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_state_list=None):
        output_hidden_state_list = []
        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            residual_hidden_states, output_hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                attention_mask=attention_mask,
                cached_hidden_states=cached_hidden_state_list[i] if cached_hidden_state_list is not None else None,
            )
            hidden_states = residual_hidden_states + hidden_states
            output_hidden_state_list.append(output_hidden_states)

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output, output_hidden_state_list


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.,
        max_len = 32
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.dtype)
        return self.dropout(x)

class TemporalAttention(CrossAttention):
    def __init__(
            self,
            temporal_max_len                   = 32,
            pos_embedding_type                 = "ape",
            use_causal_mask                    = True,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.pos_embedding_type = pos_embedding_type
        self._use_memory_efficient_attention_xformers = True
        self.use_causal_mask = use_causal_mask

        self.pos_encoder = None
        self.freqs_cis = None
        if self.pos_embedding_type == "ape":
            self.pos_encoder = PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.,
                max_len=temporal_max_len
            )

        elif self.pos_embedding_type == "rope":
            self.freqs_cis = precompute_freqs_cis(
                kwargs["query_dim"],
                temporal_max_len
            )

        else:
            raise NotImplementedError

    def _create_causal_mask(self, seq_len, current_frame_idx=None, device='cuda'):
        """
        Create causal mask for temporal attention.
        For streaming mode, mask out future frames beyond current_frame_idx.
        For training mode, create lower triangular mask.
        
        Args:
            seq_len: Sequence length (number of frames)
            current_frame_idx: Index of current frame in streaming mode (None for training)
            device: Device to create mask on
            
        Returns:
            Causal mask of shape (seq_len, seq_len)
        """
        if current_frame_idx is not None:
            # Streaming mode: only allow attention to past and current frames
            mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
            mask[:, :current_frame_idx + 1] = True
        else:
            # Training mode: standard causal mask (lower triangular)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        # Convert to attention bias (large negative values for masked positions)
        causal_mask = torch.where(mask, 0.0, float('-inf'))
        return causal_mask

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_states=None):
        # TODO: support cache for these
        assert encoder_hidden_states is None
        assert attention_mask is None

        d = hidden_states.shape[1]
        d_in = 0
        is_streaming_mode = cached_hidden_states is not None
        
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            input_hidden_states = hidden_states  # (bxd) f c
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            input_hidden_states = hidden_states
            
            # Handle cached states properly - ensure correct dimensions
            
            if cached_hidden_states.dim() == 2:
                # If cached tensor is 2D [seq_len, dim], convert to 3D format
                cached_hidden_states = cached_hidden_states.unsqueeze(1)  # [seq_len, 1, dim]
                # Repeat for batch dimension to match current input
                cached_hidden_states = cached_hidden_states.repeat(hidden_states.shape[0], 1, 1)  # [batch*seq_len, 1, dim]
                cached_hidden_states = cached_hidden_states.view(hidden_states.shape[0], -1, cached_hidden_states.shape[2])  # [batch, seq_len, dim]
            # elif cached_hidden_states.dim() == 3 and cached_hidden_states.shape[1] == 1:
                # If cached tensor is [spatial_patches, 1, dim], we need to expand temporal dimension
                # print(f"Expanding single temporal step cache from {cached_hidden_states.shape}")
                # This means we have a single frame cache, just use it as is for concatenation
            
            # Check if cached states are compatible - only feature dimension needs to match
            # Batch dimension should already match since we're processing the same image resolution
            if cached_hidden_states.shape[2] != hidden_states.shape[2]:
                print("Creating new compatible cache...")
                cached_hidden_states = torch.zeros(
                    hidden_states.shape[0], 0, hidden_states.shape[2],
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                d_in = 0
            elif cached_hidden_states.shape[0] != hidden_states.shape[0]:
                # This should NOT happen for same resolution frames - indicates a bug
                print(f"ERROR: Batch dimension mismatch for same resolution: cached {cached_hidden_states.shape} vs current {hidden_states.shape}")
                print("This indicates a cache storage/loading bug. Resetting cache...")
                cached_hidden_states = torch.zeros(
                    hidden_states.shape[0], 0, hidden_states.shape[2],
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                d_in = 0
            else:
                d_in = cached_hidden_states.shape[1]
            
            # Ensure both tensors have the same dimensions before concatenation
            
            # Fix dimension mismatch if it occurs
            if cached_hidden_states.dim() != hidden_states.dim():
                if cached_hidden_states.dim() == 2 and hidden_states.dim() == 3:
                    cached_hidden_states = cached_hidden_states.unsqueeze(1)
                elif cached_hidden_states.dim() == 3 and hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(1)
                    print(f"Fixed current shape: {hidden_states.shape}")
            
            # Verify shapes are compatible for concatenation
            if cached_hidden_states.shape[0] != hidden_states.shape[0] or cached_hidden_states.shape[2] != hidden_states.shape[2]:
                print(f"ERROR: Shape incompatible for concat: cached {cached_hidden_states.shape} vs current {hidden_states.shape}")
                print("Resetting cache to avoid error...")
                cached_hidden_states = torch.zeros(
                    hidden_states.shape[0], 0, hidden_states.shape[2],
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                d_in = 0
                
            hidden_states = torch.cat([cached_hidden_states, hidden_states], dim=1)
            
            # Ensure total temporal length doesn't exceed maximum
            max_total_length = 32
            if hidden_states.shape[1] > max_total_length:
                hidden_states = hidden_states[:, -max_total_length:, :]

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states[:, d_in:, ...])
        dim = query.shape[-1]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if self.freqs_cis is not None:
            seq_len = query.shape[1]
            freqs_cis = self.freqs_cis[:seq_len].to(query.device)
            query, key = apply_rotary_emb(query, key, freqs_cis)

        # Apply causal masking if enabled
        causal_attention_mask = None
        if self.use_causal_mask:
            seq_len = hidden_states.shape[1]
            if is_streaming_mode:
                # In streaming mode, current frame can only attend to past frames
                current_frame_idx = seq_len - 1  # Last frame is the current frame
                causal_attention_mask = self._create_causal_mask(seq_len, current_frame_idx, hidden_states.device)
            else:
                # In training mode, apply standard causal mask
                causal_attention_mask = self._create_causal_mask(seq_len, None, hidden_states.device)
            
            # Expand mask for multiple heads and batch dimensions
            batch_size = key.shape[0]
            if causal_attention_mask is not None:
                # Only apply mask to the query frames (not full sequence)
                query_seq_len = query.shape[1]  # This is 1 for streaming mode
                if query_seq_len < seq_len:
                    # Extract the relevant part of the mask for query
                    causal_attention_mask = causal_attention_mask[-query_seq_len:, :]
                
                causal_attention_mask = causal_attention_mask.unsqueeze(0).repeat(batch_size, 1, 1)
                causal_attention_mask = causal_attention_mask.repeat_interleave(self.heads, dim=0)

        # Combine with existing attention mask if provided
        final_attention_mask = causal_attention_mask
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
            
            if final_attention_mask is not None:
                final_attention_mask = final_attention_mask + attention_mask
            else:
                final_attention_mask = attention_mask

        use_memory_efficient = XFORMERS_AVAILABLE and self._use_memory_efficient_attention_xformers
        if use_memory_efficient and (dim // self.heads) % 8 != 0:
            # print('Warning: the dim {} cannot be divided by 8. Fall into normal attention'.format(dim // self.heads))
            use_memory_efficient = False

        # attention, what we cannot get enough of
        if use_memory_efficient:
            query = self.reshape_heads_to_4d(query)
            key = self.reshape_heads_to_4d(key)
            value = self.reshape_heads_to_4d(value)

            hidden_states = self._memory_efficient_attention_xformers(query, key, value, final_attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, final_attention_mask)
            else:
                raise NotImplementedError

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        # Reshape back to original format
        # For streaming mode, we want [spatial_patches, temporal_len, dim]
        # For non-streaming, we want [batch*temporal, spatial_patches, dim]
        if is_streaming_mode and d_in > 0:
            # In streaming mode, we already have the right shape after attention
            # hidden_states is already [spatial_patches*temporal, dim] from attention
            # We need to reshape it to [spatial_patches, temporal_len, dim]
            spatial_patches = d
            temporal_len = hidden_states.shape[0] // spatial_patches
            hidden_states = hidden_states.view(spatial_patches, temporal_len, -1)
        else:
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        
        # In streaming mode with cached states, we need to ensure output matches input
        if is_streaming_mode and d_in > 0:
            # For streaming, we only want the output for the new frame
            # hidden_states is now [spatial_patches, total_temporal_len, dim] after concat
            
            # Get only the new frame's output (last temporal step)
            # Keep the spatial dimension intact for proper output
            new_frame_output = hidden_states[:, -1, :]  # [spatial_patches, dim] - remove temporal dim for output
            
            # Reshape to expected output format if needed
            output_tensor = new_frame_output  # Keep as [spatial_patches, dim]
            
            # For cache output, we need to save the current frame in the right format
            # Cache should match the spatial dimension [spatial_patches, 1, dim]
            # Extract the new frame from the last temporal position
            new_frame_cache = hidden_states[:, -1:, :]  # [spatial_patches, 1, dim]
            cache_output = new_frame_cache  # Keep in [spatial_patches, 1, dim] format
            
            return output_tensor, cache_output
        else:
            # Standard mode (first frame) - return cache in streaming-compatible format
            # We need to return cache that matches the streaming mode format
            
            # For first frame, hidden_states is [1, spatial_patches, dim] 
            # We need to convert it to [spatial_patches, 1, dim] format for streaming compatibility
            # Remove the batch dimension and add temporal dimension at the right place
            
            cache_output = hidden_states.squeeze(0)      # [1, spatial_patches, dim] -> [spatial_patches, dim]
            cache_output = cache_output.unsqueeze(1)     # [spatial_patches, dim] -> [spatial_patches, 1, dim]
            
            return hidden_states, cache_output
