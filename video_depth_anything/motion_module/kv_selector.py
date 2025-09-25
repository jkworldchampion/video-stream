# motion_module/kv_selector.py
from dataclasses import dataclass
from typing import Optional   # ✅ 추가
import torch
from torch import nn

@dataclass
class SelectorOutput:
    read_idx: Optional[torch.Tensor]   # ✅ 변경 (torch.Tensor | None -> Optional[torch.Tensor])
    write_idx: Optional[torch.Tensor]  # ✅ 변경

class KVSelector(nn.Module):
    def __init__(self, time_prior_weight=0.1, motion_prior_weight=0.1, temperature=1.0, same_pos_only=True):
        super().__init__()
        self.w_t = float(time_prior_weight)
        self.w_m = float(motion_prior_weight)
        self.tau = float(temperature)
        self.same_pos_only = bool(same_pos_only)

    @torch.no_grad()
    def forward(
        self,
        q_t,
        K_hist,
        *,
        top_r: Optional[int],   # ✅ 변경 (int | None -> Optional[int])
        top_u: Optional[int],   # ✅ 변경
        time_lag=None,
        motion_score=None
    ) -> SelectorOutput:
        if K_hist is None:
            return SelectorOutput(None, None)
        B, P, T, C = K_hist.shape

        # base score (cosine)
        qn = q_t / (q_t.norm(dim=-1, keepdim=True) + 1e-6)        # [B,P,C]
        Kn = K_hist / (K_hist.norm(dim=-1, keepdim=True) + 1e-6)  # [B,P,T,C]
        base = torch.einsum("bpc, bpTc -> bpT", qn, Kn) / max(self.tau, 1e-6)

        # time prior: newer is better
        if time_lag is not None:
            prior_t = torch.exp(-time_lag.clamp_min(0.0))          # [B,P,T]
            base = base + self.w_t * prior_t

        # motion prior
        if motion_score is not None:
            base = base + self.w_m * motion_score                  # [B,P,T]

        score = base

        read_idx = (torch.topk(score, k=min(top_r, T), dim=-1).indices
                    if (top_r is not None and top_r > 0) else None)
        write_idx = (torch.topk(score, k=min(top_u, T), dim=-1).indices
                     if (top_u is not None and top_u > 0) else None)
        return SelectorOutput(read_idx, write_idx)

    @staticmethod
    def gather_time(K_or_V_flat, idx, B0, P, C):
        if idx is None:
            return None
        T = K_or_V_flat.shape[1]
        K = K_or_V_flat.view(B0, P, T, C)
        ridx = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # [B,P,R,C]
        out = torch.gather(K, 2, ridx)                  # [B,P,R,C]
        return out.view(B0 * P, -1, C)

    @staticmethod
    def gather_kv_for_update(K_flat, V_flat, idx_u, B0, P, C):
        K_u = KVSelector.gather_time(K_flat, idx_u, B0, P, C)
        V_u = KVSelector.gather_time(V_flat, idx_u, B0, P, C)
        return K_u, V_u, {"B0": B0, "P": P, "C": C}

    @staticmethod
    def scatter_kv_updated(K_flat, V_flat, idx_u, K_new, V_new, B0, P, C):
        T = K_flat.shape[1]
        K = K_flat.view(B0, P, T, C)
        V = V_flat.view(B0, P, T, C)
        ridx = idx_u.unsqueeze(-1).expand(-1, -1, -1, C)   # [B,P,U,C]
        K.scatter_(2, ridx, K_new.view(B0, P, -1, C))
        V.scatter_(2, ridx, V_new.view(B0, P, -1, C))
