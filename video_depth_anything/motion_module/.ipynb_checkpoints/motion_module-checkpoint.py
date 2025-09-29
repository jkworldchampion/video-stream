# SPDX-License-Identifier: Apache-2.0 license
#
# Adapted from AnimateDiff (guoyww/AnimateDiff)
# Modified for RoSA-KV KD caching (history accumulation)

import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from .attention import CrossAttention, FeedForward
from .time_pe import TemporalPE
from .kv_selector import KVSelector, SelectorOutput
from .kv_refiner import KVRefiner
from .rw_memory import RWMemory

XFORMERS_AVAILABLE = False

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class TemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        num_attention_blocks=2,
        norm_num_groups=32,
        temporal_max_len=32,
        zero_initialize=True,
        pos_embedding_type="ape",
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
        hidden_states, output_hidden_state_list = self.temporal_transformer(
            input_tensor, encoder_hidden_states, attention_mask, cached_hidden_state_list, **kwargs
        )
        return hidden_states, output_hidden_state_list


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        num_attention_blocks=2,
        norm_num_groups=32,
        temporal_max_len=32,
        pos_embedding_type="ape",
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
        assert hidden_states.dim() == 5
        output_hidden_state_list = []

        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner).contiguous()
        hidden_states = self.proj_in(hidden_states)

        n = (len(cached_hidden_state_list) // len(self.transformer_blocks)) if cached_hidden_state_list is not None else 0

        for i, block in enumerate(self.transformer_blocks):
            sub_cache = cached_hidden_state_list[i * n:(i + 1) * n] if n else None
            hidden_states, hidden_state_list = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                attention_mask=attention_mask,
                cached_hidden_state_list=sub_cache,
                **kwargs,
            )
            output_hidden_state_list.extend(hidden_state_list)

        hidden_states = self.proj_out(hidden_states)
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
        num_attention_blocks=2,
        temporal_max_len=32,
        pos_embedding_type="ape",
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
    오케스트레이터:
      - Q/K/V 추출 (pre-PE)
      - RW-Memory READ로 q 강화
      - Selector로 read/write 인덱스 산출 (없으면 full-pass)
      - RoPE(+Δt)/ALiBi 적용
      - 수동 SDPA
      - Refiner로 선택 토큰 ΔK/ΔV 갱신(옵션)
      - RW-Memory WRITE (Top-U 있으면 저랭크, 없으면 전역평균)
      - KD 캐시(K_all/V_all/attn_hist/Δreg) 저장
      - 캐시로 사용할 now(pre-PE)를 반환
    """
    def __init__(self, temporal_max_len=32, pos_embedding_type="ape", dim_head=64, mem_slots=16, *args, **kwargs):
        # CrossAttention은 heads/ to_q/to_k/to_v/to_out 등을 초기화함
        super().__init__(dim_head=dim_head, *args, **kwargs)

        self.temporal_max_len   = int(temporal_max_len)
        self.pos_embedding_type = pos_embedding_type
        self.dim_head           = int(dim_head)

        qdim = int(kwargs["query_dim"])  # 기존 코드 호환 (CrossAttention 생성 시 넘겨줌)

        # --- 하위 컴포넌트 ---
        self.time_pe  = TemporalPE(
            query_dim=qdim,
            temporal_max_len=self.temporal_max_len,
            pos_embedding_type=("rope" if pos_embedding_type == "rope" else "ape"),
            alibi_slope=0.0,
        )
        self.selector = KVSelector()
        self.refiner  = KVRefiner(Cfull=qdim, dim_head=self.dim_head, tau=3.0)
        self.memory   = RWMemory(slots=mem_slots, Dh=self.dim_head,
                                 alpha_init=0.5, gamma_init=0.1, use_slot_slot_decorr=True)

        # q에 메모리 읽기를 주입하는 게이트
        self.mem_read_alpha = nn.Parameter(torch.tensor(0.5))

        # KD(옵션) 캐시
        self._kd_cache_enabled = False
        self._kd_cache_role = "off"   # "off" | "teacher" | "student"
        self._hist = None

    # --- KD hooks ---
    def enable_kd_caching(self, flag: bool = True, role: str = "teacher"):
        self._kd_cache_enabled = bool(flag)
        self._kd_cache_role = role
        self._hist = {} if flag else None

    def get_cached_attention_output(self):
        # collect_kd_caches()가 이 메서드를 호출
        if not self._kd_cache_enabled:
            return None
        return self._hist

    def clear_attention_cache(self):
        self._hist = None

    # --- 본연의 forward ---
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
        teacher_sig=None,
    ):
        assert encoder_hidden_states is None and attention_mask is None

        d = hidden_states.shape[1]  # tokens per frame (=P)
        # (b*f, d, c) → (b*d, f, c)
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            now  = hidden_states                 # [(B*P), f_now(=T_offline or 1), C]
            past = None
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            now  = hidden_states                 # [(B*P), 1, C]
            past = cached_hidden_states          # [(B*P), T_past, C] (pre-PE로 저장된 것)

        # ---- Q/K/V (pre-PE) ----
        q_now = self.to_q(now)  # [(B*P), f_now, C]
        k_now = self.to_k(now)
        v_now = self.to_v(now)

        # ---- RW-Memory READ → q 주입 ----
        BxP, f_now, Cfull = q_now.shape
        P = d
        assert BxP % P == 0, f"invalid batch*pos split: BxP={BxP}, P={P}"
        B0 = BxP // P
        H  = self.heads
        Dh = Cfull // H
        # 머리 분해 일치성 보증
        assert Cfull == H * Dh, f"head split mismatch: C={Cfull}, H={H}, Dh={Dh}, heads*Dh={H*Dh}"

        # 전역 q (마지막 프레임, 위치 평균)
        q_bpfc   = q_now.view(B0, P, f_now, Cfull)
        q_global = q_bpfc[:, :, -1].mean(dim=1)          # [B0,C]
        qg       = q_global.view(B0, H, Dh).mean(dim=1)  # [B0,Dh]
        r_t      = self.memory.read(qg)                  # [B0,Dh]

        # q에 주입: (B0,Dh) → (B0,P,f_now,C)
        r_full = r_t.unsqueeze(1).repeat(1, H, 1)              # [B0,H,Dh]
        r_full = r_full.reshape(B0, H * Dh)                    # [B0,C]
        alpha  = torch.tanh(self.mem_read_alpha).clamp(-1, 1)  # 스칼라
        r_add  = r_full[:, None, None, :].expand(B0, P, f_now, Cfull).reshape(B0 * P, f_now, Cfull)
        q_now  = q_now + alpha * r_add

        # ---- 과거 토큰 준비 ----
        if stream_mode and (past is not None):
            # past는 pre-PE hidden_states로 들어온다고 가정 (우리 설계)
            k_past = self.to_k(past)  # [(B*P), T_past, C]
            v_past = self.to_v(past)
            K_hist = k_past.view(B0, P, -1, Cfull)       # [B0,P,Tp,C]
        else:
            k_past = None
            v_past = None
            K_hist = None

        # ---- Selector (Top-R/U) ----
        q_t_flat    = q_now[:, -1:].squeeze(1).view(B0, P, Cfull)  # [B0,P,C]
        time_lag    = getattr(teacher_sig, "time_lag", None) if teacher_sig is not None else None
        motion_score= getattr(teacher_sig, "motion_score", None)   if teacher_sig is not None else None

        sel: SelectorOutput = self.selector(
            q_t=q_t_flat, K_hist=K_hist,
            top_r=select_top_r, top_u=update_top_u,
            time_lag=time_lag, motion_score=motion_score
        )

        # ---- Read 세트 구성 (과거 + 현재) ----
        if (k_past is not None):
            if (sel.read_idx is not None):                 # Top-R
                k_read = self.selector.gather_time(k_past, sel.read_idx, B0, P, Cfull)  # [(B*P), R, C]
                v_read = self.selector.gather_time(v_past, sel.read_idx, B0, P, Cfull)
                key    = torch.cat([k_read, k_now], dim=1)
                value  = torch.cat([v_read, v_now], dim=1)
            else:
                # full past 사용
                key    = torch.cat([k_past, k_now], dim=1)
                value  = torch.cat([v_past, v_now], dim=1)
        else:
            key, value = k_now, v_now

        # ---- RoPE(+ALiBi) ----
        q_rope, k_rope, alibi_bias = self.time_pe.apply(q_now, key, rope_dt)

        # ---- 수동 SDPA ----
        def _to_bh(x):
            B_, L_, C_ = x.shape
            H = self.heads
            Dh = C_ // H
            x = x.view(B_, L_, H, Dh).permute(0, 2, 1, 3).reshape(B_ * H, L_, Dh).contiguous()
            return x, Dh

        q_bh, _ = _to_bh(q_rope)
        k_bh, _ = _to_bh(k_rope)
        v_bh, _ = _to_bh(value)

        scores = torch.bmm(q_bh, k_bh.transpose(1, 2)) * (self.dim_head ** -0.5)
        if alibi_bias is not None:
            # alibi_bias: [BxH, q_len, k_len]
            scores = scores + alibi_bias.to(scores.dtype)
        attn   = torch.softmax(scores, dim=-1)  # [BxH,q,k]
        ctx_bh = torch.bmm(attn, v_bh)          # [BxH,q,Dh]

        BxH, q_len, Dh_chk = ctx_bh.shape
        assert Dh_chk == Dh, "context head-dim mismatch"
        H   = self.heads
        B0b = BxH // H
        out = ctx_bh.view(B0b, H, q_len, Dh).permute(0, 2, 1, 3).reshape(B0b, q_len, H * Dh).contiguous()
        hidden_states = self.to_out[1](self.to_out[0](out))                      # proj→dropout
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)  # [(B*P), f, C] → [(B*f), d, C]

        # ---- Refiner: write 후보 Δ 갱신 (옵션) ----
        aux      = {}
        V_u_ref  = None  # ← 미리 초기화(에러 방지)
        if (k_past is not None) and (sel.write_idx is not None):
            K_u, V_u, sc = self.selector.gather_kv_for_update(k_past, v_past, sel.write_idx, B0, P, Cfull)
            K_u_ref, V_u_ref, aux = self.refiner(q_t=q_t_flat, K_u=K_u, V_u=V_u, r_t=r_t, teacher_sig=teacher_sig)
            # in-place 업데이트
            self.selector.scatter_kv_updated(k_past, v_past, sel.write_idx, K_u_ref, V_u_ref, B0, P, Cfull)

        # ---- RW-Memory WRITE ----
        if V_u_ref is not None:
            # Top-U 기반 저랭크 업데이트
            self.memory.write_topu_lowrank(V_u_ref, H=self.heads, Dh=Dh, write_idx=sel.write_idx, teacher_sig=teacher_sig)
        else:
            # MVP 경로: 현재 프레임 전역 평균
            self.memory.write_from_frame(v_now=v_now, B0=B0, P=P, Cfull=Cfull, H=self.heads, Dh=Dh, teacher_sig=teacher_sig)

        # ---- KD 캐시 저장 ----
        if self._kd_cache_enabled:
            attn_b = attn.view(B0b, H, attn.shape[1], attn.shape[2])
        
            if self._kd_cache_role == "student":
                # 학생: 그래프 유지 (detach 금지)
                if (k_past is not None) and (v_past is not None):
                    K_all = torch.cat([k_past, k_now], dim=1)  # [(B*P), T_total, C]
                    V_all = torch.cat([v_past, v_now], dim=1)
                else:
                    K_all, V_all = k_now, v_now
        
                K_all = K_all.view(B0, P, -1, Cfull)    # no detach
                V_all = V_all.view(B0, P, -1, Cfull)
                attn_b = attn_b.detach()                # 분포 로깅만
                delta_reg = aux.get("delta_reg", None)
                if torch.is_tensor(delta_reg):
                    delta_reg = delta_reg.detach()
        
            else:  # role == "teacher" (또는 기타)
                # 교사: detach로 고정
                if (k_past is not None) and (v_past is not None):
                    K_all = torch.cat([k_past, k_now], dim=1)
                    V_all = torch.cat([v_past, v_now], dim=1)
                else:
                    K_all, V_all = k_now, v_now
        
                K_all = K_all.view(B0, P, -1, Cfull).detach()
                V_all = V_all.view(B0, P, -1, Cfull).detach()
                attn_b = attn_b.detach()
                delta_reg = aux.get("delta_reg", None)
                if torch.is_tensor(delta_reg):
                    delta_reg = delta_reg.detach()
        
            self._hist = {
                "K_all_pre": K_all,      # [B0, P, T, C]
                "V_all_pre": V_all,      # [B0, P, T, C]
                "attn_hist": attn_b,     # [B0, H, q_len, k_len]
            }
            if delta_reg is not None:
                self._hist["delta_reg"] = delta_reg

        # 캐시로 pre-PE now 반환 ([(B*P), f_now, C]) — 상위에서 past로 재사용
        return hidden_states, now