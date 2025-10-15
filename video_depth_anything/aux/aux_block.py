# Copyright (2025)
# Auxiliary Non-streaming Layer for KD:
#  - Proj (C_s -> C_t)
#  - 1-layer Transformer (bidirectional) with "hole" mask for APC (mask future [t+1..t+N])
#  - Uni-directional LSTM on top (for APC target prediction proxy)
#  - Optional Q/K/V export as [B, A, T, Dh]
#
# Usage:
#   aux = AuxBlock(c_in=Cs, c_teacher=Ct, nhead=8, return_qkv=True)
#   z, r, qkv = aux(s_feat, hole_mask_N=2)
#   # z: [B,T,Ct]  (KD feature similarity target)
#   # r: [B,T,Ct]  (KD APC head output)
#   # qkv: {"Q","K","V"} each [B,A,T,Dh]  (Self-Attn relation KL)

from typing import Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        B, T, C = x.shape
        x = x.view(B, T, self.nhead, self.dh).permute(0, 2, 1, 3).contiguous()
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,A,T,Dh] -> [B,T,C]
        B, A, T, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, A * Dh)

    @staticmethod
    def _build_hole_mask(T: int, N: int, device, dtype) -> torch.Tensor:
        """
        Build attention mask that forbids attending to the next N future frames:
          mask[t, k] = -inf if k in (t+1, ..., t+N); 0 otherwise.
        Shape returned: [T, T] added to logits before softmax.
        """
        if N <= 0:
            return torch.zeros(T, T, device=device, dtype=dtype)
        m = torch.zeros(T, T, device=device, dtype=dtype)
        idx = torch.arange(T, device=device)
        for offset in range(1, N + 1):
            k = idx + offset
            valid = k < T
            m[idx[valid], k[valid]] = float("-inf")
        return m

    def forward(self, x: torch.Tensor, hole_mask_N: int = 0) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        x: [B,T,C]
        hole_mask_N: N>0 -> forbid attention to [t+1 .. t+N]
        returns:
          y: [B,T,C]
          qkv: dict or None  (Q/K/V each [B,A,T,Dh])
        """
        B, T, C = x.shape
        q = self._split_heads(self.to_q(x))  # [B,A,T,Dh]
        k = self._split_heads(self.to_k(x))  # [B,A,T,Dh]
        v = self._split_heads(self.to_v(x))  # [B,A,T,Dh]

        # Scaled dot-product attention (manual), with 'hole' mask
        scale = 1.0 / math.sqrt(self.dh)
        # [B,A,T,T]
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale

        if hole_mask_N > 0:
            mask = self._build_hole_mask(T, hole_mask_N, attn_logits.device, attn_logits.dtype)  # [T,T]
            attn_logits = attn_logits + mask.view(1, 1, T, T)

        attn = F.softmax(attn_logits, dim=-1)
        ctx = torch.matmul(attn, v)  # [B,A,T,Dh]
        y = self._merge_heads(ctx)   # [B,T,C]
        y = self.drop(self.out(y))   # [B,T,C]

        qkv = {"Q": q, "K": k, "V": v} if self.return_qkv else None
        return y, qkv


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

    def forward(self, x: torch.Tensor, hole_mask_N: int = 0):
        # x: [B,T,C]
        h = self.ln1(x)
        y, qkv = self.mha(h, hole_mask_N=hole_mask_N)  # [B,T,C], dict or None
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x, qkv


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
        rnn_type: str = "lstm",     # "lstm" only in this minimal implementation
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

        # 1) projection to teacher dim
        self.proj_in = nn.Linear(c_in, c_teacher, bias=True)

        # 2) optional APE
        self.pos = SinusoidalPE1D(c_teacher, max_len=8192, dropout=0.0) if use_pos_enc else nn.Identity()

        # 3) single-layer transformer (bidirectional)
        self.tr = TransformerBlock1D(
            dim=c_teacher, nhead=nhead, mlp_ratio=mlp_ratio, dropout=dropout, return_qkv=return_qkv
        )

        # 4) uni-directional LSTM (APC head)
        assert rnn_type.lower() == "lstm", "This AuxBlock implements only LSTM for rnn_type."
        self.rnn = nn.LSTM(input_size=c_teacher, hidden_size=c_teacher, num_layers=1, batch_first=True, bidirectional=False)

        # 5) (optional) output norm
        self.ln_out = nn.LayerNorm(c_teacher)

    def forward(self, s_feat: torch.Tensor, hole_mask_N: int = 0):
        """
        s_feat: [B, T, C_in]  (student temporal features pooled to [B,T,C_in])
        hole_mask_N: for the transformer attention — block [t+1..t+N] to avoid APC leakage
        returns:
          z:   [B, T, C_t]  (to match teacher feature h for feature loss)
          r:   [B, T, C_t]  (uni-LSTM outputs, for APC to predict h_{t+N})
          qkv: dict or None; {"Q","K","V"} each [B, A, T, Dh]
        """
        assert s_feat.dim() == 3, f"s_feat must be [B,T,C_in], got {s_feat.shape}"
        B, T, _ = s_feat.shape
        x = self.proj_in(s_feat)        # [B,T,Ct]
        x = self.pos(x)                 # [B,T,Ct]

        z, qkv = self.tr(x, hole_mask_N=hole_mask_N)  # z: [B,T,Ct]

        # Uni-LSTM over z
        r, _ = self.rnn(z)              # [B,T,Ct]
        z = self.ln_out(z)
        r = self.ln_out(r)

        return z, r, qkv
