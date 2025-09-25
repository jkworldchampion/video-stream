# motion_module/time_pe.py
import torch
from torch import nn
from .attention import precompute_freqs_cis, apply_rotary_emb

class TemporalPE(nn.Module):
    def __init__(self, query_dim, temporal_max_len=32, pos_embedding_type="rope", alibi_slope=0.0):
        super().__init__()
        self.kind = pos_embedding_type
        self.temporal_max_len = temporal_max_len
        self.alibi_slope = float(alibi_slope)
        self.freqs_cis = precompute_freqs_cis(query_dim, temporal_max_len) if self.kind == "rope" else None

    def _scaled_rope(self, seq_len, device, rope_dt=None):
        base = self.freqs_cis.to(device)
        if rope_dt is None:
            return base[:seq_len]
        if isinstance(rope_dt, (float, int)):
            idx = torch.clamp((torch.arange(seq_len, device=device).float() * float(rope_dt)).round().long(),
                              0, self.temporal_max_len - 1)
            return base.index_select(0, idx)
        elif torch.is_tensor(rope_dt):
            if rope_dt.numel() == 1:
                s = float(rope_dt.item())
                idx = torch.clamp((torch.arange(seq_len, device=device).float() * s).round().long(),
                                  0, self.temporal_max_len - 1)
                return base.index_select(0, idx)
            else:
                idx = torch.clamp(rope_dt.round().long(), 0, self.temporal_max_len - 1)
                return base.index_select(0, idx)
        else:
            return base[:seq_len]

    def _alibi_bias(self, q_len, k_len, device, dtype):
        if self.alibi_slope <= 0.0:
            return None
        q_idx = torch.arange(q_len, device=device).unsqueeze(1)
        k_idx = torch.arange(k_len, device=device).unsqueeze(0)
        bias = (q_idx - k_idx).clamp_min(0).to(dtype) * (-self.alibi_slope)  # [q,k]
        # [1,q,k]로 반환하여 브로드캐스트로 사용
        return bias.unsqueeze(0)

    def apply(self, q, k, rope_dt):
        # RoPE
        if self.kind == "rope" and self.freqs_cis is not None:
            freqs_k = self._scaled_rope(k.shape[1], k.device, rope_dt)
            freqs_q = self._scaled_rope(q.shape[1], q.device, rope_dt)
            q, k = apply_rotary_emb(q, k, (freqs_q, freqs_k))
        # ALiBi
        alibi = self._alibi_bias(q_len=q.shape[1], k_len=k.shape[1], device=q.device, dtype=q.dtype)
        return q, k, alibi
