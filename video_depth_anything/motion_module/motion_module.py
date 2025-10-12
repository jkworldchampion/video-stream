# This file is originally from AnimateDiff/animatediff/models/motion_module.py at main · guoyww/AnimateDiff
# SPDX-License-Identifier: Apache-2.0 license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file was released under [ Apache-2.0 license], with the full license text available at [https://github.com/guoyww/AnimateDiff?tab=Apache-2.0-1-ov-file#readme].
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from .attention import CrossAttention, FeedForward

# --- xFormers는 버전별 텐서 규격 차이가 있어 shape 오류를 유발할 수 있음.
# 본 구현에서는 수동 SDPA(softmax(QK^T)V)만 사용해서 차원 정합을 보장한다.
XFORMERS_AVAILABLE = False

def zero_module(module):
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
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None, cached_hidden_state_list=None):
        hidden_states = input_tensor
        hidden_states, output_hidden_state_list = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, cached_hidden_state_list
        )
        return hidden_states, output_hidden_state_list


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
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
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
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cached_hidden_state_list=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, got {hidden_states.dim()}."
        output_hidden_state_list = []

        video_length = hidden_states.shape[2]  # time
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner).contiguous()
        hidden_states = self.proj_in(hidden_states)  # -> [B, HW, inner_dim]

        # cache split per block
        n = (len(cached_hidden_state_list) // len(self.transformer_blocks)) if cached_hidden_state_list is not None else 0

        for i, block in enumerate(self.transformer_blocks):
            sub_cache = cached_hidden_state_list[i*n:(i+1)*n] if n else None
            hidden_states, hidden_state_list = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                attention_mask=attention_mask,
                cached_hidden_state_list=sub_cache,
            )
            output_hidden_state_list.extend(hidden_state_list)

        hidden_states = self.proj_out(hidden_states)  # [B, HW, inner_dim]
        hidden_states = hidden_states.reshape(batch, height, width, inner).permute(0, 3, 1, 2).contiguous()

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
                )
                for _ in range(num_attention_blocks)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_attention_blocks)])

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
            # 캐시 비활성 베이스라인: 빈 값은 모으지 않음
            if output_hidden_states:
                output_hidden_state_list.append(output_hidden_states)

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        return hidden_states, output_hidden_state_list


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=32):
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
    """
    수동 SDPA만 사용.
    - 항상: [B*H, q, dh] → (merge heads) → [B, q, H*dh] → to_out
    """
    def __init__(self, temporal_max_len=32, pos_embedding_type="ape", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pos_embedding_type = pos_embedding_type
        self.temporal_max_len = temporal_max_len

        self.pos_encoder = None
        assert self.pos_embedding_type == "ape", "Baseline removes RoPE; use APE only."
        self.pos_encoder = PositionalEncoding(kwargs["query_dim"], dropout=0., max_len=temporal_max_len)
 

        # xformers는 사용하지 않음 (shape 안전성)
        self._use_memory_efficient_attention_xformers = False

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        cached_hidden_states=None,
    ):
        assert encoder_hidden_states is None
        assert attention_mask is None

        d = hidden_states.shape[1]  # tokens per frame

        # (b*f, d, c) -> (b*d, f, c)
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            now = hidden_states
            past = None
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            now = hidden_states
            past = cached_hidden_states

        # Positional
        if self.pos_encoder is not None:
            now_pos  = self.pos_encoder(now)
            past_pos = self.pos_encoder(past) if past is not None else None
        else:
            now_pos, past_pos = now, past

        # Q/K/V
        q_now = self.to_q(now_pos)
        k_now = self.to_k(now_pos)
        v_now = self.to_v(now_pos)

        # 스트리밍: 과거가 있으면 전부 concat, 없으면 현재만 사용
        if past_pos is not None:
            k_past_all = self.to_k(past_pos)
            v_past_all = self.to_v(past_pos)
            key   = torch.cat([k_past_all, k_now], dim=1)  # (b*d, K, C)
            value = torch.cat([v_past_all, v_now], dim=1)
            query = q_now
        else:
            key, value, query = k_now, v_now, q_now

        # ==== Manual SDPA (always) ====
        # reshape to [B*H, L, Dh]
        def _to_bh(x):
            B_, L_, C_ = x.shape
            H = self.heads
            assert C_ % H == 0, f"Last dim {C_} not divisible by heads {H}"
            Dh = C_ // H
            x = x.view(B_, L_, H, Dh).permute(0, 2, 1, 3).reshape(B_ * H, L_, Dh).contiguous()
            return x, Dh

        q_bh, Dh_q = _to_bh(query)
        k_bh, Dh_k = _to_bh(key)
        v_bh, Dh_v = _to_bh(value)
        assert Dh_q == Dh_k == Dh_v, f"Head dims mismatch: {Dh_q}, {Dh_k}, {Dh_v}"

        scale = (Dh_q ** -0.5)
        scores = torch.bmm(q_bh, k_bh.transpose(1, 2)) * scale  # [B*H, q, k]
        attn = torch.softmax(scores, dim=-1)                    # [B*H, q, k]
        ctx_bh = torch.bmm(attn, v_bh)                          # [B*H, q, Dh]

        # merge heads: [B*H, q, Dh] -> [B, q, H*Dh]
        BxH, q_len, Dh = ctx_bh.shape
        H = self.heads
        assert BxH % H == 0, f"Batch*heads {BxH} not divisible by heads {H}"
        B0 = BxH // H
        out = ctx_bh.view(B0, H, q_len, Dh).permute(0, 2, 1, 3).reshape(B0, q_len, H * Dh).contiguous()

        # to_out: Linear(H*Dh -> C), Dropout
        hidden_states = self.to_out[1](self.to_out[0](out))     # [B0, q, C]

        # back to (b*f, d, c)
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d).contiguous()

        return hidden_states, []
