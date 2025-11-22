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

    def forward(self, input_tensor, encoder_hidden_states=None, attention_mask=None, cached_hidden_state_list=None, return_qkv: bool = False,):
        hidden_states = input_tensor
        if return_qkv:
            hidden_states, output_hidden_state_list, qkv = self.temporal_transformer(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cached_hidden_state_list,
                return_qkv=True,   # 반드시 True
            )
            return hidden_states, output_hidden_state_list, qkv
        else:
            hidden_states, output_hidden_state_list = self.temporal_transformer(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cached_hidden_state_list,
                return_qkv=False,  # 반드시 False
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

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        cached_hidden_state_list=None,
        return_qkv: bool = False,   # ★ 추가
    ):
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

        # ★ qkv 수집 용 컨테이너 (마지막 블록/어텐션 기준으로 OK)
        q_export = k_export = v_export = None

        for i, block in enumerate(self.transformer_blocks):
            sub_cache = cached_hidden_state_list[i*n:(i+1)*n] if n else None
            if return_qkv:
                hidden_states, hidden_state_list, qkv_blk = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    video_length=video_length,
                    attention_mask=attention_mask,
                    cached_hidden_state_list=sub_cache,
                    return_qkv=True,  # ★ 전달
                )
                if qkv_blk is not None:  # dict {"Q","K","V"} [B,H,T,Dh]
                    q_export, k_export, v_export = qkv_blk["Q"], qkv_blk["K"], qkv_blk["V"]
            else:
                hidden_states, hidden_state_list = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    video_length=video_length,
                    attention_mask=attention_mask,
                    cached_hidden_state_list=sub_cache,
                    return_qkv=False,  # ★
                )
            output_hidden_state_list.extend(hidden_state_list)

        hidden_states = self.proj_out(hidden_states)  # [B, HW, inner_dim]
        hidden_states = hidden_states.reshape(batch, height, width, inner).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        if return_qkv:
            return output, output_hidden_state_list, {"Q": q_export, "K": k_export, "V": v_export}
        else:
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

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        cached_hidden_state_list=None,
        return_qkv: bool = False,    # ★ 추가
    ):
        output_hidden_state_list = []
        q_export = k_export = v_export = None  # ★

        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            if return_qkv:
                residual_hidden_states, output_hidden_states, qkv_attn = attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    video_length=video_length,
                    attention_mask=attention_mask,
                    cached_hidden_states=cached_hidden_state_list[i] if cached_hidden_state_list is not None else None,
                    return_qkv=True,   # ★ 전달
                )
                if qkv_attn is not None:
                    q_export, k_export, v_export = qkv_attn["Q"], qkv_attn["K"], qkv_attn["V"]
            else:
                residual_hidden_states, output_hidden_states = attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    video_length=video_length,
                    attention_mask=attention_mask,
                    cached_hidden_states=cached_hidden_state_list[i] if cached_hidden_state_list is not None else None,
                    return_qkv=False,  # ★
                )

            hidden_states = residual_hidden_states + hidden_states
            if output_hidden_states is not None:
                output_hidden_state_list.append(output_hidden_states)

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        if return_qkv:
            return hidden_states, output_hidden_state_list, {"Q": q_export, "K": k_export, "V": v_export}
        else:
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
        return_qkv: bool = False,   # ★ 추가
    ):
        assert encoder_hidden_states is None
        assert attention_mask is None

        d = hidden_states.shape[1]  # tokens per frame

        # (b*f, d, c) -> (b*d, f, c)  or cache path -> (b*d, 1, c)
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            now = hidden_states
            past = None
            B_tokens = now.shape[0]  # == b*d
            T_seq = now.shape[1]     # == f
            input_hidden_states = now
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            now = hidden_states
            past = cached_hidden_states
            B_tokens = now.shape[0]  # == b*d
            T_seq = now.shape[1]     # == 1
            input_hidden_states = now

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

        if past_pos is not None:
            q_past = self.to_q(past_pos)  # ★ export용
            k_past = self.to_k(past_pos)
            v_past = self.to_v(past_pos)

            key   = torch.cat([k_past, k_now], dim=1)   # (b*d, L, C)
            value = torch.cat([v_past, v_now], dim=1)
            query = q_now

            if return_qkv:
                Q_all = torch.cat([q_past, q_now], dim=1)  # (b*d, L, C)
                K_all = key
                V_all = value
        else:
            key, value, query = k_now, v_now, q_now
            if return_qkv:
                Q_all, K_all, V_all = q_now, k_now, v_now

        # ==== Manual SDPA ====
        def _to_bh(x):
            B_, L_, C_ = x.shape
            H = self.heads
            Dh = C_ // H
            x = x.view(B_, L_, H, Dh).permute(0, 2, 1, 3).reshape(B_ * H, L_, Dh).contiguous()
            return x, Dh

        q_bh, Dh_q = _to_bh(query)
        k_bh, Dh_k = _to_bh(key)
        v_bh, Dh_v = _to_bh(value)
        assert Dh_q == Dh_k == Dh_v

        scale = (Dh_q ** -0.5)
        scores = torch.bmm(q_bh, k_bh.transpose(1, 2)) * scale  # [B*H, q, k]
        attn = torch.softmax(scores, dim=-1)
        ctx_bh = torch.bmm(attn, v_bh)

        # merge heads back
        BxH, q_len, Dh = ctx_bh.shape
        H = self.heads
        B0 = BxH // H
        out = ctx_bh.view(B0, H, q_len, Dh).permute(0, 2, 1, 3).reshape(B0, q_len, H * Dh).contiguous()

        hidden_states = self.to_out[1](self.to_out[0](out))     # [B0, q, C]
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d).contiguous()

        # ===== Q/K/V export =====
        qkv_export = None
        if return_qkv:
            # Q_all/K_all/V_all: (b*d, L, C) -> [B, H, L, Dh] by averaging over spatial tokens d
            B_est = B0 // d if d > 0 else 1  # b
            Hh = self.heads
            Dh2 = Q_all.shape[-1] // Hh

            def _avg_over_tokens(x):
                # (b*d, L, C) -> [b, d, L, H, Dh] -> avg over d -> [b, H, L, Dh]
                x = x.view(B_est, d, x.shape[1], Hh, Dh2).permute(0, 3, 2, 1, 4).contiguous().mean(dim=3)
                return x  # [b, H, L, Dh]

            Q_export = _avg_over_tokens(Q_all)
            K_export = _avg_over_tokens(K_all)
            V_export = _avg_over_tokens(V_all)
            qkv_export = {"Q": Q_export, "K": K_export, "V": V_export}

        if return_qkv:
            return hidden_states, input_hidden_states, qkv_export
        else:
            return hidden_states, input_hidden_states
