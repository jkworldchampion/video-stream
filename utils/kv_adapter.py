from typing import Dict, Tuple

import torch
from torch import nn


class KVAdapter(nn.Module):
    """Head-wise residual adapter that perturbs K/V with gated deltas."""

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        assert d_model % heads == 0, f"d_model ({d_model}) must be divisible by heads ({heads})"
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.delta_k = nn.Linear(d_model, d_model, bias=False)
        self.delta_v = nn.Linear(d_model, d_model, bias=False)

        gate_init = -3.0  # sigmoid(-3) ~= 0.047
        self.gk = nn.Parameter(torch.full((heads,), gate_init))
        self.gv = nn.Parameter(torch.full((heads,), gate_init))

    def forward(self, r_seq: torch.Tensor, qkv: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            r_seq: [B, T, C] residual features (e.g. AuxBlock future predictor output).
            qkv:   dict with optional tensors for "Q", "K", "V" each shaped [B, H, T, Dh].
        Returns:
            updated_qkv: dict with K/V perturbed by gated residuals.
            reg_term:    scalar tensor for L2 penalty on raw deltas.
        """
        if qkv is None:
            raise ValueError("qkv must not be None")
        k = qkv.get("K", None)
        v = qkv.get("V", None)
        if k is None or v is None:
            return qkv, r_seq.new_zeros(())

        if r_seq.dim() != 3:
            raise ValueError(f"Expected r_seq with shape [B,T,C], got {r_seq.shape}")

        b, t, c = r_seq.shape
        if c != self.d_model:
            raise ValueError(f"Adapter d_model {self.d_model} does not match r_seq last dim {c}")

        if k.shape[0] != b:
            raise ValueError("Batch size mismatch between r_seq and K")
        if k.shape[1] != self.heads or v.shape[1] != self.heads:
            raise ValueError("Head count mismatch between adapter and qkv")

        if k.shape[2] != t or v.shape[2] != t:
            raise ValueError("Temporal length mismatch between r_seq and qkv")
        if k.shape[3] != self.head_dim or v.shape[3] != self.head_dim:
            raise ValueError("Head dim mismatch between adapter and qkv")

        flat = r_seq.reshape(b * t, c)
        delta_k = self.delta_k(flat).reshape(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,T,Dh]
        delta_v = self.delta_v(flat).reshape(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)

        gate_k = torch.sigmoid(self.gk).view(1, self.heads, 1, 1)
        gate_v = torch.sigmoid(self.gv).view(1, self.heads, 1, 1)

        k_prime = k + gate_k * delta_k
        v_prime = v + gate_v * delta_v

        reg = (delta_k.pow(2).mean() + delta_v.pow(2).mean()) * 0.5

        out = {
            "Q": qkv.get("Q", None),
            "K": k_prime,
            "V": v_prime,
        }
        return out, reg
