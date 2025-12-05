# Copyright (2025)
# Auxiliary Non-streaming Layer for KD:
#  - Proj (C_s -> C_t)
#  - 1-layer Transformer (bidirectional) with "hole" mask for APC (mask future [t+1..t+N])
#  - Uni-directional LSTM on top (for APC target prediction proxy)
#  - Optional Q/K/V export as [B, A, T, Dh]
#
# Usage:
#   aux = AuxBlock(c_in=Cs, c_teacher=Ct, nhead=8, return_qkv=True)
#   z, r, qkv = aux(s_feat, hole_mask_n=2)
#   # z: [B,T,Ct]  (KD feature similarity target)
#   # r: [B,T,Ct]  (KD APC head output)
#   # qkv: {"Q","K","V"} each [B,A,T,Dh]  (Self-Attn relation KL)

from typing import Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# experiment: mamba
try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except Exception:
    _MAMBA_AVAILABLE = False

class SinusoidalPE1D(nn.Module):
    """Simple 1D sinusoidal positional embedding (absolute)."""
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        pe = torch.zeros(max_len, d_model)  # [L, C]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        T = x.shape[1]
        return self.dropout(x + self.pe[:T].to(dtype=x.dtype, device=x.device))


class MHAttention1D(nn.Module):
    """
    Minimal MHA with explicit Q/K/V projections and optional 'hole' mask for APC.
    Operates on sequences: input [B,T,C], outputs [B,T,C] and (optionally) Q/K/V as [B,A,T,Dh]
    """
    def __init__(self, dim: int, nhead: int, dropout: float = 0.0, return_qkv: bool = False):
        super().__init__()
        assert dim % nhead == 0, f"dim({dim}) must be divisible by nhead({nhead})"
        self.dim = dim
        self.nhead = nhead
        self.dh = dim // nhead
        self.return_qkv = return_qkv

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

        self.out = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> [B,A,T,Dh]
        B, T, _ = x.shape
        x = x.view(B, T, self.nhead, self.dh).permute(0, 2, 1, 3).contiguous()
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,A,T,Dh] -> [B,T,C]
        batch, heads, seq_len, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, heads * head_dim)

    @staticmethod
    def _build_hole_mask(seq_len: int, hole: int, device, dtype) -> torch.Tensor:
        """
        Build attention mask that forbids attending to the next N future frames:
          mask[t, k] = -inf if k in (t+1, ..., t+N); 0 otherwise.
        Shape returned: [T, T] added to logits before softmax.
        """
        if hole <= 0:
            return torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        m = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        idx = torch.arange(seq_len, device=device)
        for offset in range(1, hole + 1):
            k = idx + offset
            valid = k < seq_len
            m[idx[valid], k[valid]] = float("-inf")
        return m

    def forward(self, x: torch.Tensor, hole_mask_n: int = 0) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        x: [B,T,C]
        hole_mask_n: N>0 -> forbid attention to [t+1 .. t+N]
        returns:
          y: [B,T,C]
          qkv: dict or None  (Q/K/V each [B,A,T,Dh])
        """
        _, seq_len, _ = x.shape
        q = self._split_heads(self.to_q(x))  # [B,A,T,Dh]
        k = self._split_heads(self.to_k(x))  # [B,A,T,Dh]
        v = self._split_heads(self.to_v(x))  # [B,A,T,Dh]

        # Scaled dot-product attention (manual), with 'hole' mask
        scale = 1.0 / math.sqrt(self.dh)
        # [B,A,T,T]
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale

        if hole_mask_n > 0:
            mask = self._build_hole_mask(seq_len, hole_mask_n, attn_logits.device, attn_logits.dtype)  # [T,T]
            attn_logits = attn_logits + mask.view(1, 1, seq_len, seq_len)

        attn = F.softmax(attn_logits, dim=-1)
        ctx = torch.matmul(attn, v)  # [B,A,T,Dh]
        y = self._merge_heads(ctx)   # [B,T,C]
        y = self.drop(self.out(y))   # [B,T,C]

        qkv = {"Q": q, "K": k, "V": v} if self.return_qkv else None
        return y, qkv

    def forward_stream(
        self,
        x_t: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        *,
        hole_mask_n: int = 0,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Streaming variant that consumes a single timestep.

        Args:
            x_t: [B,C] or [B,1,C]
            cache: previous K/V cache dict with tensors shaped [B,A,L,Dh]
            hole_mask_n: kept for API parity (no-op in streaming as no future tokens yet)
            max_len: optional cap on stored cache length

        Returns:
            y_t: [B,C]
            qkv_t: optional dict with per-head projections for the current step [B,A,1,Dh]
            new_cache: updated cache with detached K/V tensors (length clipped to max_len if provided)
        """
        if x_t.dim() == 3:
            if x_t.size(1) != 1:
                raise ValueError(f"Expected singleton time dimension, got shape {x_t.shape}")
            x_t = x_t[:, 0, :]

        cache = cache or {}
        k_cached = cache.get("k")
        v_cached = cache.get("v")

        q = self._split_heads(self.to_q(x_t.unsqueeze(1)))  # [B,A,1,Dh]
        k_new = self._split_heads(self.to_k(x_t.unsqueeze(1)))
        v_new = self._split_heads(self.to_v(x_t.unsqueeze(1)))

        if k_cached is not None:
            k_all = torch.cat([k_cached, k_new], dim=2)
            v_all = torch.cat([v_cached, v_new], dim=2)
        else:
            k_all = k_new
            v_all = v_new

        scale = 1.0 / math.sqrt(self.dh)
        attn_logits = torch.matmul(q, k_all.transpose(-1, -2)) * scale  # [B,A,1,L]
        attn = torch.softmax(attn_logits, dim=-1)
        ctx = torch.matmul(attn, v_all)  # [B,A,1,Dh]
        y = self._merge_heads(ctx).squeeze(1)  # [B,C]
        y = self.drop(self.out(y))

        if max_len is not None and k_all.shape[2] > max_len:
            k_all = k_all[:, :, -max_len:, :].contiguous()
            v_all = v_all[:, :, -max_len:, :].contiguous()

        new_cache = {
            "k": k_all.detach(),
            "v": v_all.detach(),
        }

        qkv = {"Q": q, "K": k_new, "V": v_new} if self.return_qkv else None
        return y, qkv, new_cache


class TransformerBlock1D(nn.Module):
    """1-layer Transformer block with LayerNorm + MHA + FFN."""
    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0, return_qkv: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mha = MHAttention1D(dim, nhead, dropout=dropout, return_qkv=return_qkv)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim, bias=True),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, hole_mask_n: int = 0):
        # x: [B,T,C]
        h = self.ln1(x)
        y, qkv = self.mha(h, hole_mask_n=hole_mask_n)  # [B,T,C], dict or None
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x, qkv

    def forward_stream(
        self,
        x_t: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        *,
        hole_mask_n: int = 0,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Streaming single-step forward returning the updated cache."""
        cache = cache.copy() if cache else {}

        h_t = self.ln1(x_t)
        mha_cache = cache.get("mha")
        y_t, qkv, mha_cache = self.mha.forward_stream(
            h_t,
            cache=mha_cache,
            hole_mask_n=hole_mask_n,
            max_len=max_len,
        )
        x_res = x_t + y_t
        ff_out = self.ff(self.ln2(x_res))
        out = x_res + ff_out
        cache["mha"] = mha_cache
        return out, qkv, cache


class AuxBlock(nn.Module):
    """
    Student temporal features -> (Aux non-streaming layer) -> z (feature KD) + r (APC head)
    Optionally export per-head Q/K/V of the transformer.
    """
    def __init__(
        self,
        c_in: int,
        c_teacher: int,
        nhead: int = 8,
        dropout: float = 0.0,
        return_attn: bool = False,  # kept for compatibility (unused)
        return_qkv: bool = True,
        rnn_type: str = "lstm",     # "lstm" / "mamba"
        mamba_d_state: int = 16,    # placeholders for API compatibility
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_pos_enc: bool = True,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_t = c_teacher
        self.nhead = nhead
        self.return_qkv = return_qkv
        self.use_pos_enc = use_pos_enc

        # like TransformerEncoderLayer but with our MHAttention1D and APC uni-LSTM
        # 1) projection to teacher dim
        self.proj_in = nn.Linear(c_in, c_teacher, bias=True)

        # 2) optional APE (positional encoding)
        self.pos = SinusoidalPE1D(c_teacher, max_len=8192, dropout=0.0) if use_pos_enc else nn.Identity()

        # 3) single-layer transformer (bidirectional)
        self.tr = TransformerBlock1D(
            dim=c_teacher, nhead=nhead, mlp_ratio=mlp_ratio, dropout=dropout, return_qkv=return_qkv
        )

        # 4) uni-directional predictor (APC head)
        rnn_type = rnn_type.lower()
        if rnn_type == "lstm":
            self.predictor = nn.LSTM(input_size=c_teacher, hidden_size=c_teacher, num_layers=1,
                                     batch_first=True, bidirectional=False)
            self._pred_is_mamba = False
        elif rnn_type == "mamba":
            assert _MAMBA_AVAILABLE, "Install mamba-ssm to use rnn_type='mamba'."
            # Mamba는 기본적으로 causal(단방향)이며 입력/출력 차원을 d_model로 맞춤
            self.predictor = Mamba(
                d_model=c_teacher,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
            self._pred_is_mamba = True
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        # 5) (optional) output norm
        self.ln_out = nn.LayerNorm(c_teacher)

    def forward(self, s_feat: torch.Tensor, hole_mask_n: int = 0):
        """
        s_feat: [B, T, C_in]  (student temporal features pooled to [B,T,C_in])
    hole_mask_n: for the transformer attention — block [t+1..t+N] to avoid APC leakage
        returns:
          z:   [B, T, C_t]  (to match teacher feature h for feature loss)
          r:   [B, T, C_t]  (uni-LSTM outputs, for APC to predict h_{t+N})
          qkv: dict or None; {"Q","K","V"} each [B, A, T, Dh]
        """
        assert s_feat.dim() == 3, f"s_feat must be [B,T,C_in], got {s_feat.shape}"
        x = self.proj_in(s_feat)        # [B,T,Ct]

        if self.use_pos_enc:
            x = self.pos(x)                 # [B,T,Ct]

        # z: feature KD용 출력, qkv: self-attn Q/K/V
        z, qkv = self.tr(x, hole_mask_n=hole_mask_n)  # z: [B,T,Ct] Transformer 출력, qkv: dict or None

        # Uni-LSTM over z
        if self._pred_is_mamba:
            # Mamba는 입력 [B,T,C] → [B,T,C] (causal)
            r = self.predictor(z)
        else:
            r, _ = self.predictor(z)
        z = self.ln_out(z)
        r = self.ln_out(r)

        return z, r, qkv

    def forward_stream(
        self,
        s_feat_one: torch.Tensor,
        *,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        hole_mask_n: int = 0,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Streaming variant consuming a single timestep feature.

        Args:
            s_feat_one: [B,C_in] or [B,1,C_in] tensor for the current frame.
            cache: dict holding internal streaming state (K/V cache, RNN hidden states, position index).
            hole_mask_n: APC hole mask horizon (kept for parity with batched path).
            max_len: optional max cache length for attention history (e.g., KD window).

        Returns:
            z_t: [B,C_t]    – feature projection for KD (last frame)
            r_t: [B,C_t]    – APC predictor output for the same frame
            qkv_t: optional dict with per-head Q/K/V [B,A,1,Dh]
            new_cache: updated cache to feed the next timestep
        """
        if s_feat_one.dim() == 3:
            if s_feat_one.size(1) != 1:
                raise ValueError(f"Expected single-frame feature, got shape {s_feat_one.shape}")
            s_feat_one = s_feat_one[:, 0, :]

        cache = dict(cache) if cache else {}

        x = self.proj_in(s_feat_one)
        if self.use_pos_enc:
            pos_idx = cache.get("pos_idx", 0)
            pe_len = self.pos.pe.size(0)
            idx = pos_idx % pe_len
            pe = self.pos.pe[idx:idx + 1].to(dtype=x.dtype, device=x.device)
            x = x + pe
            x = self.pos.dropout(x)
            cache["pos_idx"] = pos_idx + 1

        tr_cache = cache.get("tr")
        z_raw, qkv, tr_cache = self.tr.forward_stream(
            x,
            cache=tr_cache,
            hole_mask_n=hole_mask_n,
            max_len=max_len,
        )
        cache["tr"] = tr_cache

        if self._pred_is_mamba:
            r_raw = self.predictor(z_raw.unsqueeze(1)).squeeze(1)
        else:
            lstm_state = cache.get("rnn_state")
            r_seq, lstm_state = self.predictor(z_raw.unsqueeze(1), lstm_state)
            cache["rnn_state"] = tuple(h.detach() for h in lstm_state)
            r_raw = r_seq.squeeze(1)

        z = self.ln_out(z_raw)
        r = self.ln_out(r_raw)

        return z, r, qkv, cache
