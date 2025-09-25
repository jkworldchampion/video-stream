# motion_module/kv_refiner.py
import torch
from torch import nn
import torch.nn.functional as F

class KVRefiner(nn.Module):
    def __init__(self, Cfull, dim_head, tau=3.0, mode="concat_head", lora_rank=None,
                 beta_mem=0.2, lora_gate_reduce: str = "none", teacher_delta_scale: float = 0.1):
        """
        lora_gate_reduce: {"none", "mean"}  # LoRA 게이트를 U축으로 평균할지 여부
        """
        super().__init__()
        self.tau = float(tau)
        self.mode = mode
        self.beta_mem = float(beta_mem)
        self.teacher_delta_scale = float(teacher_delta_scale)
        self.Cfull = int(Cfull)
        self.Dh = int(dim_head)
        self.lora_gate_reduce = lora_gate_reduce
        hidden = max(64, self.Dh)

        in_dim = self.Cfull * 3 + self.Dh  # 항상 r_t 포함(없으면 0으로 채움)

        if lora_rank is None:
            self.norm = nn.LayerNorm(in_dim)
            self.mlp  = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 2 * self.Cfull + 2)  # [ΔK(C), ΔV(C), s(1), g(1)]
            )
        else:
            r = int(lora_rank)
            self.norm = nn.LayerNorm(in_dim)
            self.proj = nn.Linear(in_dim, r)
            self.Uk   = nn.Linear(r, self.Cfull, bias=False)
            self.Uv   = nn.Linear(r, self.Cfull, bias=False)
            self.gate = nn.Linear(r, 2)  # [s, g]

    @staticmethod
    def _l2_clamp(x, tau, dim=-1, eps=1e-6):
        n = x.norm(dim=dim, keepdim=True).clamp_min(eps)
        return x * (tau / n).clamp_max(1.0)

    def forward(self, q_t, K_u, V_u, r_t=None, teacher_sig=None):
        """
        q_t: [B,P,C],  K_u/V_u: [(B*P),U,C]
        r_t: [B,Dh] or None
        """
        B, P, C = q_t.shape
        BP, U, Ckv = K_u.shape
        assert BP == B * P and Ckv == C, f"shape mismatch: q_t {q_t.shape}, K_u {K_u.shape}"

        q_rep = q_t.reshape(B * P, 1, C).expand(-1, U, -1)  # [(BP),U,C]

        # r_t 채널 준비
        if r_t is not None:
            assert r_t.shape[-1] == self.Dh, f"r_t Dh mismatch: {r_t.shape[-1]} vs {self.Dh}"
            r_rep = r_t.repeat_interleave(P, dim=0).unsqueeze(1).expand(-1, U, -1)  # [(BP),U,Dh]
        else:
            r_rep = torch.zeros(B * P, U, self.Dh, device=K_u.device, dtype=K_u.dtype)

        inp = torch.cat([q_rep, K_u, V_u, r_rep], dim=-1)  # [(BP),U,3C+Dh]

        aux = {}
        if hasattr(self, "mlp"):
            inp_n = self.norm(inp)
            out = self.mlp(inp_n)  # [(BP),U,2C+2]
            dK, dV, s, g = torch.split(out, [C, C, 1, 1], dim=-1)
        else:
            z = self.proj(self.norm(inp)).tanh()  # [(BP),U,r]
            dK = self.Uk(z)
            dV = self.Uv(z)
            sg = self.gate(z)  # [(BP),U,2]
            if self.lora_gate_reduce == "mean":
                sg = sg.mean(dim=1, keepdim=True)  # [(BP),1,2]
            s  = torch.tanh(sg[..., :1])
            g  = torch.sigmoid(sg[..., 1:2])
            aux["lora_reg"] = z.pow(2).mean(dtype=torch.float32)

        K_new = self._l2_clamp(K_u + g * dK, self.tau, dim=-1)
        V_new = self._l2_clamp(V_u * (1.0 + s) + g * dV, self.tau, dim=-1)

        # RW-Mem 간접 반영
        if r_t is not None:
            assert C % self.Dh == 0, f"C must be multiple of Dh; got C={C}, Dh={self.Dh}"
            H = C // self.Dh
            r_full = r_rep.repeat(1, 1, H)  # [(BP),U,Dh*H] = [(BP),U,C]
            V_new = self._l2_clamp(V_new + (self.beta_mem * g) * r_full, self.tau, dim=-1)

        # Teacher Δ̄ 주입
        if (teacher_sig is not None) and (getattr(teacher_sig, "delta_bar", None) is not None):
            db = teacher_sig.delta_bar
            if db.dim() == 2:
                db = db.unsqueeze(1)  # [B,1,C]
            db = db.repeat_interleave(P, 0).unsqueeze(1).expand(-1, U, -1)  # [(BP),U,C]
            V_new = self._l2_clamp(V_new + self.teacher_delta_scale * db, self.tau, dim=-1)

        aux["delta_reg"] = (dK.pow(2).mean(dtype=torch.float32) + dV.pow(2).mean(dtype=torch.float32))
        return K_new, V_new, aux
