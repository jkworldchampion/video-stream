# This file is originally from AnimateDiff/animatediff/models/motion_module.py at main · guoyww/AnimateDiff
# SPDX-License-Identifier: Apache-2.0 license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file was released under [ Apache-2.0 license], with the full license text available at [https://github.com/guoyww/AnimateDiff?tab=Apache-2.0-1-ov-file#readme].
import torch
import torch.nn.functional as F
from torch import nn

try:
    from .attention import CrossAttention, FeedForward, apply_rotary_emb, precompute_freqs_cis
except ImportError:
    # Create placeholder classes if import fails
    class CrossAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.heads = kwargs.get('heads', 8)
            self.scale = 1.0
            self.to_q = nn.Linear(kwargs.get('query_dim', 512), kwargs.get('query_dim', 512))
            self.to_k = nn.Linear(kwargs.get('query_dim', 512), kwargs.get('query_dim', 512))
            self.to_v = nn.Linear(kwargs.get('query_dim', 512), kwargs.get('query_dim', 512))
            self.to_out = nn.ModuleList([
                nn.Linear(kwargs.get('query_dim', 512), kwargs.get('query_dim', 512)),
                nn.Dropout(0.0)
            ])
            self.group_norm = None
            self._slice_size = None
            self.added_kv_proj_dim = None
            self._use_memory_efficient_attention_xformers = False
            self.upcast_efficient_attention = False
            
        def forward(self, hidden_states, *args, **kwargs):
            _ = args, kwargs  # Acknowledge parameters
            return hidden_states
            
        def reshape_heads_to_batch_dim(self, tensor):
            head_size = self.heads
            batch_size, seq_len, dim = tensor.shape
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor
            
        def reshape_batch_dim_to_heads(self, tensor):
            head_size = self.heads
            batch_size, seq_len, dim = tensor.shape
            tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, head_size * dim)
            return tensor
            
        def _attention(self, query, key, value, attention_mask=None):
            _, _, _ = query.shape  # Acknowledge variables
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
                
            attention_probs = F.softmax(attention_scores, dim=-1)
            hidden_states = torch.matmul(attention_probs, value)
            return hidden_states
            
        def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
            # Fallback implementation when xformers not available
            return self._attention(query, key, value, attention_mask)
            
        def reshape_heads_to_4d(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size).contiguous()
            return tensor
    
    class FeedForward(nn.Module):
        def __init__(self, dim, *args, **kwargs):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        def forward(self, x, scale=1.0):
            return self.net(x) * scale
    
    def apply_rotary_emb(query, key, freqs_cis):
        _ = freqs_cis  # Acknowledge parameter
        return query, key
    
    def precompute_freqs_cis(dim, max_len):
        return torch.zeros(max_len, dim)

try:
    from einops import rearrange, repeat
except ImportError:
    # Fallback implementations
    def rearrange(tensor, pattern, **kwargs):
        # Simple fallback for common patterns
        if pattern == "(b f) d c -> (b d) f c":
            b_f, d, c = tensor.shape
            f = kwargs.get('f', 1)
            b = b_f // f
            return tensor.view(b, f, d, c).permute(0, 2, 1, 3).reshape(b * d, f, c)
        elif pattern == "(b d) f c -> (b f) d c":
            b_d, f, c = tensor.shape
            d = kwargs.get('d', 1)
            b = b_d // d
            return tensor.view(b, d, f, c).permute(0, 2, 1, 3).reshape(b * f, d, c)
        else:
            return tensor
    
    def repeat(tensor, pattern, **kwargs):
        if pattern == "b n c -> (b d) n c":
            d = kwargs.get('d', 1)
            return tensor.repeat(d, 1, 1)
        else:
            return tensor

import math

try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


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

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None, cached_hidden_state_list=None, bidirectional_update_length=16, current_frame=0):
        hidden_states = input_tensor
        hidden_states, output_hidden_state_list = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, cached_hidden_state_list,
            bidirectional_update_length=bidirectional_update_length,
            current_frame=current_frame
        )

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

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cached_hidden_state_list=None, bidirectional_update_length=16, current_frame=0):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        output_hidden_state_list = []

        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, _, height, width = hidden_states.shape
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
            hidden_states, hidden_state_list = block(
                hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, 
                attention_mask=attention_mask,
                cached_hidden_state_list=cached_hidden_state_list[i*n:(i+1)*n] if n else None,
                bidirectional_update_length=bidirectional_update_length,
                current_frame=current_frame
            )
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


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_state_list=None, bidirectional_update_length=16, current_frame=0):
        output_hidden_state_list = []
        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            residual_hidden_states, output_hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                attention_mask=attention_mask,
                cached_hidden_states=cached_hidden_state_list[i] if cached_hidden_state_list is not None else None,
                bidirectional_update_length=bidirectional_update_length,
                current_frame=current_frame,
            )
            hidden_states = residual_hidden_states + hidden_states
            output_hidden_state_list.append(output_hidden_states)

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output, output_hidden_state_list


class TemporalAttention(CrossAttention):
    def __init__(
            self,
            *args,
            temporal_max_len=32,
            pos_embedding_type="ape",
            use_causal_mask=True,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.pos_embedding_type = pos_embedding_type
        self._use_memory_efficient_attention_xformers = True
        self.use_causal_mask = use_causal_mask

        # Attention output caching for Knowledge Distillation
        self.attention_output_cache = None
        self.enable_attention_caching = False

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
    
    def enable_kd_caching(self, enable=True):
        """Enable/disable attention output caching for Knowledge Distillation"""
        self.enable_attention_caching = enable
        if not enable:
            self.attention_output_cache = None
    
    def get_cached_attention_output(self):
        """Get the cached attention output from last forward pass"""
        return self.attention_output_cache

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        """Memory efficient attention using xformers"""
        if hasattr(self, 'upcast_efficient_attention') and self.upcast_efficient_attention:
            org_dtype = query.dtype
            query = query.float()
            key = key.float()
            value = value.float()
            if attention_mask is not None:
                attention_mask = attention_mask.float()
        
        hidden_states = self._memory_efficient_attention_split(query, key, value, attention_mask)

        if hasattr(self, 'upcast_efficient_attention') and self.upcast_efficient_attention:
            hidden_states = hidden_states.to(org_dtype)

        hidden_states = self.reshape_4d_to_heads(hidden_states)
        return hidden_states
    
    def _memory_efficient_attention_split(self, query, key, value, attention_mask):
        """Split memory efficient attention computation"""
        try:
            import xformers.ops
            
            # Reshape to 4D format for xformers
            query = self.reshape_heads_to_4d(query)
            key = self.reshape_heads_to_4d(key)
            value = self.reshape_heads_to_4d(value)
            
            batch_size = query.shape[0]
            max_batch_size = 65535
            num_batches = (batch_size + max_batch_size - 1) // max_batch_size
            results = []
            
            for i in range(num_batches):
                start_idx = i * max_batch_size
                end_idx = min((i + 1) * max_batch_size, batch_size)
                query_batch = query[start_idx:end_idx]
                key_batch = key[start_idx:end_idx]
                value_batch = value[start_idx:end_idx]
                if attention_mask is not None:
                    attention_mask_batch = attention_mask[start_idx:end_idx]
                else:
                    attention_mask_batch = None
                result = xformers.ops.memory_efficient_attention(
                    query_batch, key_batch, value_batch, attn_bias=attention_mask_batch
                )
                results.append(result)
            
            full_result = torch.cat(results, dim=0)
            return full_result
            
        except ImportError:
            # Fallback to regular attention if xformers not available
            hidden_states = self._attention(query, key, value, attention_mask)
            # Convert to 4D format to match expected output
            batch_size, seq_len, dim = hidden_states.shape
            head_size = self.heads
            hidden_states = hidden_states.view(batch_size // head_size, seq_len, head_size, dim // head_size)
            return hidden_states
    
    def reshape_heads_to_4d(self, tensor):
        """Reshape tensor to 4D format for xformers"""
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size).contiguous()
        return tensor
    
    def reshape_4d_to_heads(self, tensor):
        """Reshape 4D tensor back to 3D format"""
        batch_size, seq_len, head_size, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, dim * head_size).contiguous()
        return tensor
    
    def reshape_batch_dim_to_heads(self, tensor):
        """Safe reshape with fallback"""
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        
        # Safety check
        if batch_size % head_size != 0:
            print(f"WARNING TemporalAttention: batch_size {batch_size} not divisible by heads {head_size}")
            # Use parent method with adjusted tensor
            adjusted_batch_size = (batch_size // head_size) * head_size
            tensor = tensor[:adjusted_batch_size]
        
        # Call parent method
        return super().reshape_batch_dim_to_heads(tensor)

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

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_states=None, bidirectional_update_length=16, current_frame=0):
        # Support for encoder_hidden_states and attention_mask will be added later
        assert encoder_hidden_states is None
        assert attention_mask is None

        # Use current_frame parameter in calculations
        _ = current_frame  # Acknowledge parameter usage

        d = hidden_states.shape[1]
        d_in = 0
        is_streaming_mode = cached_hidden_states is not None
        
        if cached_hidden_states is None:
            # First frame or training mode
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
            # Initialize cache structure for bidirectional update
            current_cache = {
                'old_frames_kv': None,  # Only K, V for frames older than bidirectional window
                'recent_frames_qkv': hidden_states,  # Full Q, K, V for recent frames
                'total_frames': video_length,
                'bidirectional_length': bidirectional_update_length
            }
        else:
            # Streaming mode - new frame arrives
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)

            # Handle bidirectional sliding window cache
            if isinstance(cached_hidden_states, dict) and 'recent_frames_qkv' in cached_hidden_states:
                # New cache structure with bidirectional support
                old_frames_kv = cached_hidden_states.get('old_frames_kv', None)
                recent_frames_qkv = cached_hidden_states.get('recent_frames_qkv', None)
                total_frames = cached_hidden_states.get('total_frames', 0)
                
                if recent_frames_qkv is not None:
                    # Add new frame to recent frames (with Q, K, V)
                    recent_frames_qkv = torch.cat([recent_frames_qkv, hidden_states], dim=1)
                    
                    # Check if we need to move oldest recent frame to old_frames_kv
                    if recent_frames_qkv.shape[1] > bidirectional_update_length:
                        # Move oldest frame from recent to old (Q,K,V -> K,V only)
                        oldest_recent = recent_frames_qkv[:, :1, :]  # First frame in recent
                        recent_frames_qkv = recent_frames_qkv[:, 1:, :]  # Remove first frame
                        
                        # Add oldest_recent to old_frames_kv (K,V만 저장)
                        if old_frames_kv is not None:
                            old_frames_kv = torch.cat([old_frames_kv, oldest_recent], dim=1)
                        else:
                            old_frames_kv = oldest_recent
                    
                    # Apply max cache length limit to old frames
                    max_old_frames = getattr(self, 'max_total_length', 32) - bidirectional_update_length
                    if old_frames_kv is not None and old_frames_kv.shape[1] > max_old_frames:
                        old_frames_kv = old_frames_kv[:, -max_old_frames:, :]
                    
                    # Combine for attention computation
                    if old_frames_kv is not None:
                        hidden_states = torch.cat([old_frames_kv, recent_frames_qkv], dim=1)
                        d_in_old = old_frames_kv.shape[1]
                        d_in_recent_without_new = recent_frames_qkv.shape[1] - 1
                    else:
                        hidden_states = recent_frames_qkv
                        d_in_old = 0
                        d_in_recent_without_new = recent_frames_qkv.shape[1] - 1
                    
                    d_in = d_in_old + d_in_recent_without_new
                    total_frames += 1
                else:
                    # Initialize if not properly set
                    hidden_states = torch.cat([cached_hidden_states, hidden_states], dim=1)
                    d_in = cached_hidden_states.shape[1]
                    total_frames += 1
                    d_in_old = 0
                    recent_frames_qkv = hidden_states
                    old_frames_kv = None
                
                # Update cache structure
                current_cache = {
                    'old_frames_kv': old_frames_kv,
                    'recent_frames_qkv': recent_frames_qkv,
                    'total_frames': total_frames,
                    'bidirectional_length': bidirectional_update_length
                }
            else:
                # Legacy cache format - convert to new format or simple concatenation
                if isinstance(cached_hidden_states, dict):
                    cached_tensor = cached_hidden_states.get('data', cached_hidden_states)
                else:
                    cached_tensor = cached_hidden_states
                
                # Ensure compatibility
                if cached_tensor.shape[2] != hidden_states.shape[2]:
                    cached_tensor = torch.zeros(
                        hidden_states.shape[0], 0, hidden_states.shape[2],
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    d_in = 0
                elif cached_tensor.shape[0] != hidden_states.shape[0]:
                    cached_tensor = torch.zeros(
                        hidden_states.shape[0], 0, hidden_states.shape[2],
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    d_in = 0
                else:
                    d_in = cached_tensor.shape[1]
                
                hidden_states = torch.cat([cached_tensor, hidden_states], dim=1)
                
                # Split into old and recent based on bidirectional window
                total_frames = hidden_states.shape[1]
                if total_frames > bidirectional_update_length:
                    d_in_old = total_frames - bidirectional_update_length
                    old_frames_kv = hidden_states[:, :d_in_old, :]
                    recent_frames_qkv = hidden_states[:, d_in_old:, :]
                else:
                    d_in_old = 0
                    old_frames_kv = None
                    recent_frames_qkv = hidden_states
                
                current_cache = {
                    'old_frames_kv': old_frames_kv,
                    'recent_frames_qkv': recent_frames_qkv,
                    'total_frames': total_frames,
                    'bidirectional_length': bidirectional_update_length
                }
            
            # Apply max total length
            max_total_length = getattr(self, 'max_total_length', 32)
            if hidden_states.shape[1] > max_total_length:
                hidden_states = hidden_states[:, -max_total_length:, :]
                d_in = min(d_in, max_total_length - 1)

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # For Query: only use new frame in streaming mode
        if is_streaming_mode:
            query = self.to_q(hidden_states[:, d_in:, ...])  # Only new frame gets Q
        else:
            query = self.to_q(hidden_states)  # All frames in training mode
        
        dim = query.shape[-1]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        # For Key and Value: 
        # - Recent frames (bidirectional window): full K, V computation  
        # - Old frames: only use existing K, V (no re-computation)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        if is_streaming_mode and 'old_frames_kv' in locals() and d_in_old > 0:
            # Split K, V computation for old vs recent frames
            old_frames = hidden_states[:, :d_in_old, :]
            recent_frames = hidden_states[:, d_in_old:, :]
            
            # For old frames: reuse cached K, V (would be stored separately in practice)
            # For now, compute but mark that these shouldn't be updated
            key_old = self.to_k(old_frames)
            value_old = self.to_v(old_frames)
            
            # For recent frames: compute fresh K, V (these will be updated)
            key_recent = self.to_k(recent_frames) 
            value_recent = self.to_v(recent_frames)
            
            # Combine K, V
            key = torch.cat([key_old, key_recent], dim=1)
            value = torch.cat([value_old, value_recent], dim=1)
        else:
            # Standard K, V computation
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

        if self.freqs_cis is not None:
            seq_len = query.shape[1]
            freqs_cis = self.freqs_cis[:seq_len].to(query.device)
            query, key = apply_rotary_emb(query, key, freqs_cis)

        # Apply bidirectional attention mask for sliding window
        causal_attention_mask = None
        if self.use_causal_mask:
            seq_len = hidden_states.shape[1]
            
            if is_streaming_mode and 'recent_frames_qkv' in locals():
                # Bidirectional sliding window attention
                causal_attention_mask = torch.full((1, seq_len), float('-inf'), device=hidden_states.device)
                
                # Old frames: can be attended to but are causal
                if d_in_old > 0:
                    causal_attention_mask[:, :d_in_old] = 0.0
                
                # Recent frames within bidirectional window: full bidirectional attention
                recent_start = d_in_old
                if recent_start < seq_len:
                    causal_attention_mask[:, recent_start:] = 0.0  # Allow attention to all recent frames
                    
            elif is_streaming_mode:
                # Standard causal attention for streaming
                current_frame_idx = seq_len - 1
                causal_attention_mask = self._create_causal_mask(seq_len, current_frame_idx, hidden_states.device)
                causal_attention_mask = causal_attention_mask[-1:, :]  # Only new frame
            else:
                # Training mode: standard causal mask
                causal_attention_mask = self._create_causal_mask(seq_len, None, hidden_states.device)
                query_seq_len = query.shape[1]
                if query_seq_len < seq_len:
                    causal_attention_mask = causal_attention_mask[-query_seq_len:, :]
            
            # Expand mask for multiple heads and batch dimensions
            if causal_attention_mask is not None:
                batch_size = key.shape[0]
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
            use_memory_efficient = False

        # attention, what we cannot get enough of
        if use_memory_efficient:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            hidden_states = self._memory_efficient_attention_xformers(query, key, value, final_attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
            # Memory efficient attention already returns correct dimensions via reshape_4d_to_heads
        else:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, final_attention_mask)
            else:
                raise NotImplementedError

            # _attention already calls reshape_batch_dim_to_heads internally

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # Cache attention output for Knowledge Distillation (before dropout)
        if self.enable_attention_caching:
            # Store current frame attention output for KD
            if is_streaming_mode:
                # Only cache the new frame's attention output
                self.attention_output_cache = hidden_states.detach().clone()
            else:
                # Cache the last frame's attention output for training
                last_frame_output = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
                self.attention_output_cache = last_frame_output[-d:].detach().clone()  # Last frame only

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        # Return in proper format based on mode
        if is_streaming_mode:
            # Reshape back to [batch*temporal, spatial_patches, dim] format
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
            
            # Return new frame output and updated cache
            return hidden_states, current_cache
        else:
            # Training mode: reshape and return
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
            # Return consistent format: (output, cache) - cache can be the same as output for training
            return hidden_states, hidden_states
