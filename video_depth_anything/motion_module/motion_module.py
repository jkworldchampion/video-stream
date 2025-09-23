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

from .attention import CrossAttention, FeedForward, apply_rotary_emb, precompute_freqs_cis

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

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None, cached_hidden_state_list=None, **kwargs):
        """
        kwargs (선택):
          stream_mode: bool
          rope_dt: float or Tensor
          select_top_r: int
          update_top_u: int (자리만)
          return_attn: bool
          return_qkv: bool
        """
        hidden_states = input_tensor
        hidden_states, output_hidden_state_list = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, cached_hidden_state_list, **kwargs
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

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cached_hidden_state_list=None, **kwargs):
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
                **kwargs,
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

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_state_list=None, **kwargs):
        output_hidden_state_list = []
        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            residual_hidden_states, output_hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                attention_mask=attention_mask,
                cached_hidden_states=cached_hidden_state_list[i] if cached_hidden_state_list is not None else None,
                **kwargs,
            )
            hidden_states = residual_hidden_states + hidden_states
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
    - KD 훅: enable_kd_caching(True)일 때 정확한 attn/q/k/v/context를 self._kd_cache에 저장
    """
    def __init__(self, temporal_max_len=32, pos_embedding_type="ape", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pos_embedding_type = pos_embedding_type
        self.temporal_max_len = temporal_max_len

        self.pos_encoder = None
        self.freqs_cis = None
        if self.pos_embedding_type == "ape":
            self.pos_encoder = PositionalEncoding(kwargs["query_dim"], dropout=0., max_len=temporal_max_len)
        elif self.pos_embedding_type == "rope":
            self.freqs_cis = precompute_freqs_cis(kwargs["query_dim"], temporal_max_len)
        else:
            raise NotImplementedError

        # xformers는 사용하지 않음 (shape 안전성)
        self._use_memory_efficient_attention_xformers = False

        # KD 훅 상태
        self._kd_cache_enabled = False
        self._kd_cache = None
        
        self.enable_refiner = True
        self.refiner_alpha = 0.5   # norm clamp 비율(예)
        c = self.heads * self.dim_head
        hidden = max(128, c // 2)
        self.kv_refiner = nn.Sequential(
            nn.Linear(3 * c, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * c + 2)  # ΔK(c) | ΔV(c) | log_s(1) | logit_g(1)
        )

    # ===== KD hook API =====
    def enable_kd_caching(self, flag: bool = True):
        self._kd_cache_enabled = bool(flag)
        if not flag:
            self._kd_cache = None

    def clear_attention_cache(self):
        self._kd_cache = None

    def get_cached_attention_output(self):
        return self._kd_cache

    # ===== Rope scaling =====
    def _scaled_rope(self, seq_len, device, rope_dt=None):
        assert self.freqs_cis is not None, "RoPE not initialized"
        base = self.freqs_cis.to(device)
        if rope_dt is None:
            return base[:seq_len]
        if isinstance(rope_dt, (float, int)):
            idx = torch.clamp(
                (torch.arange(seq_len, device=device).float() * float(rope_dt)).round().long(),
                0, self.temporal_max_len - 1
            )
            return base.index_select(0, idx)
        elif torch.is_tensor(rope_dt):
            if rope_dt.numel() == 1:
                s = float(rope_dt.item())
                idx = torch.clamp(
                    (torch.arange(seq_len, device=device).float() * s).round().long(),
                    0, self.temporal_max_len - 1
                )
                return base.index_select(0, idx)
            else:
                idx = torch.clamp(rope_dt.round().long(), 0, self.temporal_max_len - 1)
                return base.index_select(0, idx)
        else:
            return base[:seq_len]

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        cached_hidden_states=None,
        *,
        stream_mode: bool = False,
        rope_dt=None,
        select_top_r: int = None,
        update_top_u: int = None,
        return_attn: bool = False,
        return_qkv: bool = False,
    ):
        assert encoder_hidden_states is None
        assert attention_mask is None

        d = hidden_states.shape[1]  # tokens-per-frame (spatial tokens)

        # (b*f, d, c) -> (b*d, f, c)
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            now = hidden_states
            past = None
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            now = hidden_states
            past = cached_hidden_states

        # Positional(APE or RoPE-prep)
        if self.pos_encoder is not None:
            now_pos  = self.pos_encoder(now)
            past_pos = self.pos_encoder(past) if past is not None else None
        else:
            now_pos, past_pos = now, past

        # Q/K/V (concat-head 차원 C = heads*dim_head)
        q_now = self.to_q(now_pos)   # [(b*d), q(=1), C]
        k_now = self.to_k(now_pos)   # [(b*d), 1, C]
        v_now = self.to_v(now_pos)   # [(b*d), 1, C]

        sel_idx = None          # for logging
        ref_stats = {}          # dK/dV/g/s/clamp_violation 저장용
        if stream_mode and (past_pos is not None):
            k_past_all = self.to_k(past_pos)   # [(b*d), Tp, C]
            v_past_all = self.to_v(past_pos)   # [(b*d), Tp, C]

            # --- Top-R selector ---
            if (select_top_r is not None) and (select_top_r > 0) and (k_past_all.shape[1] > select_top_r):
                # 점수: [(b*d), 1, Tp]
                score = torch.matmul(q_now, k_past_all.transpose(-1, -2))
                sel_idx = torch.topk(score.squeeze(1), k=select_top_r, dim=-1, largest=True, sorted=False).indices  # [(b*d), R]
                gidx = sel_idx.unsqueeze(-1).expand(-1, -1, k_past_all.shape[-1])                                   # [(b*d), R, C]
                k_sel = torch.gather(k_past_all, dim=1, index=gidx)  # [(b*d), R, C]
                v_sel = torch.gather(v_past_all, dim=1, index=gidx)  # [(b*d), R, C]
            else:
                k_sel, v_sel = k_past_all, v_past_all                # [(b*d), Tp, C]

            # --- 🔧 KV-Refiner (선택된 과거 토큰에만 적용) ---
            if self.enable_refiner and (k_sel is not None) and (k_sel.numel() > 0):
                B0, R, C = k_sel.shape
                # q_len=1 가정 → q를 R에 맞춰 broadcast
                q_rep = q_now.expand(B0, R, C)                       # [(b*d), R, C]
                inp   = torch.cat([q_rep, k_sel, v_sel], dim=-1)     # [(b*d), R, 3C]
                out   = self.kv_refiner(inp)                         # [(b*d), R, 2C+2]
                dK, dV, log_s, logit_g = torch.split(out, [C, C, 1, 1], dim=-1)
                s = F.softplus(log_s) + 1e-5                         # ≥ 0
                g = torch.sigmoid(logit_g)                           # 0~1 gate

                k_ref = k_sel + g * dK * s
                v_ref = v_sel + g * dV * s

                # norm clamp 위반 페널티(위반분만 제곱 평균)
                eps = 1e-6
                k_base = k_sel.norm(dim=-1, keepdim=True) + eps
                v_base = v_sel.norm(dim=-1, keepdim=True) + eps
                k_over = torch.clamp(dK.norm(dim=-1, keepdim=True) - self.refiner_alpha * k_base, min=0.0)
                v_over = torch.clamp(dV.norm(dim=-1, keepdim=True) - self.refiner_alpha * v_base, min=0.0)
                clamp_violation = (k_over.pow(2) + v_over.pow(2)).mean()

                # refined를 사용
                k_sel, v_sel = k_ref, v_ref

                # 통계 저장(캐시에 넣기 위해)
                ref_stats = {"dK": dK, "dV": dV, "g": g, "s": s, "clamp_violation": clamp_violation}

            # 현재 프레임과 concat
            key   = torch.cat([k_sel, k_now], dim=1)     # [(b*d), K, C]
            value = torch.cat([v_sel, v_now], dim=1)     # [(b*d), K, C]
            query = q_now                                # [(b*d), q, C]
        else:
            # 과거 없음(offline/단일)
            key, value, query = k_now, v_now, q_now      # [(b*d), 1, C]

        # RoPE (원한다면 q/k에 적용)
        if self.freqs_cis is not None:
            k_len = key.shape[1]
            q_len = query.shape[1]
            freqs_k = self._scaled_rope(k_len, device=key.device, rope_dt=rope_dt)
            freqs_q = self._scaled_rope(q_len, device=query.device, rope_dt=rope_dt)
            query, key = apply_rotary_emb(query, key, (freqs_q, freqs_k))

        # ===== Manual SDPA (항상) =====
        def _to_bh(x):
            B_, L_, C_ = x.shape
            H = self.heads
            assert C_ % H == 0, f"Last dim {C_} not divisible by heads {H}"
            Dh = C_ // H
            x = x.view(B_, L_, H, Dh).permute(0, 2, 1, 3).reshape(B_ * H, L_, Dh).contiguous()
            return x, Dh

        q_bh, Dh = _to_bh(query)         # [B*H, q, Dh]
        k_bh, _  = _to_bh(key)           # [B*H, k, Dh]
        v_bh, _  = _to_bh(value)         # [B*H, k, Dh]

        scale = (Dh ** -0.5)
        scores = torch.bmm(q_bh, k_bh.transpose(1, 2)) * scale  # [B*H, q, k]
        attn   = torch.softmax(scores, dim=-1)                  # [B*H, q, k]
        ctx_bh = torch.bmm(attn, v_bh)                          # [B*H, q, Dh]

        # merge heads → [B0, q, C]
        BxH, q_len, Dh = ctx_bh.shape
        H = self.heads
        assert BxH % H == 0
        B0 = BxH // H
        ctx_merged = ctx_bh.view(B0, H, q_len, Dh).permute(0, 2, 1, 3).reshape(B0, q_len, H * Dh).contiguous()
        hidden_states = self.to_out[1](self.to_out[0](ctx_merged))  # [B0, q, C]

        # back to (b*f, d, c)
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d).contiguous()

        # ===== 정확 KD 캐시 =====
        if self._kd_cache_enabled:
            # head-avg attn: [B0, q, k]
            attn_avg = attn.view(B0, H, q_len, -1).mean(dim=1).contiguous()

            # K/V를 concat-head 공간으로 병합해 저장: [B0, k, C]
            k_merged = k_bh.view(B0, H, -1, Dh).permute(0, 2, 1, 3).reshape(B0, -1, H * Dh).contiguous()
            v_merged = v_bh.view(B0, H, -1, Dh).permute(0, 2, 1, 3).reshape(B0, -1, H * Dh).contiguous()

            cache_dict = {
                "attn":    attn_avg,     # [B0, q(=1), k]
                "K":       k_merged,     # [B0, k, C]
                "V":       v_merged,     # [B0, k, C]
                "context": ctx_merged,   # [B0, q(=1), C]
            }
            if sel_idx is not None:
                cache_dict["selected_indices"] = sel_idx  # [(b*d), R]
            # Refiner 통계(있을 때만)
            cache_dict.update({k: v for k, v in ref_stats.items()})

            self._kd_cache = cache_dict

        # ===== 선택 리턴 =====
        if return_attn or return_qkv:
            extra = {"attn": attn_avg if self._kd_cache_enabled else None}
        else:
            extra = None

        return hidden_states, extra

