# motion_module/rw_memory.py
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from .teacher_signals import TeacherSignals

class RWMemory(nn.Module):
    def __init__(self, slots: int, Dh: int, alpha_init=0.5, gamma_init=0.1, use_slot_slot_decorr=True):
        super().__init__()
        self.slots = int(slots)
        self.Dh = int(Dh)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))  # (선택) read 게이트
        self.gamma = nn.Parameter(torch.tensor(gamma_init))  # write 강도
        self.use_slot_slot_decorr = bool(use_slot_slot_decorr)
        self._M = None  # [B,S,Dh]

        # 저랭크 write용 작은 투영
        self.proj_v = nn.Linear(Dh, Dh, bias=False)
        self.proj_u = nn.Linear(Dh, Dh, bias=False)

    # --- lifecycle ---
    def reset(self):
        self._M = None

    def _ensure_mem(self, B, device, dtype):
        if (self._M is None) or (self._M.shape[0] != B) or (self._M.shape[2] != self.Dh):
            M = torch.empty(B, self.slots, self.Dh, device=device, dtype=dtype)
            nn.init.xavier_uniform_(M)
            self._M = M.detach()

    # --- read ---
    def read(self, qg):  # qg:[B,Dh]
        self._ensure_mem(qg.shape[0], qg.device, qg.dtype)
        M = self._M
        logits = torch.bmm(M, qg.unsqueeze(-1)).squeeze(-1)   # [B,S]
        w = torch.softmax(logits, dim=-1)
        r_t = torch.bmm(w.unsqueeze(1), M).squeeze(1)         # [B,Dh]
        # r_t = torch.tanh(self.alpha).clamp(-1, 1) * r_t  # 필요 시 사용
        return r_t

    # --- reg ---
    @staticmethod
    def _decorr_slot_slot(M, eps=1e-6):
        B, S, Dh = M.shape
        Mt = F.normalize(M, dim=-1, eps=eps)
        G = torch.bmm(Mt, Mt.transpose(1, 2))   # [B,S,S]
        I = torch.eye(S, device=M.device, dtype=M.dtype).unsqueeze(0)
        return ((G - I) ** 2).mean(dtype=torch.float32)

    # --- simple write (MVP) ---
    def write_from_frame(self, v_now, B0, P, Cfull, H, Dh, teacher_sig: Optional[TeacherSignals] = None):
        self._ensure_mem(B0, v_now.device, v_now.dtype)

        v_global = v_now[:, -1].reshape(B0, P, Cfull).mean(dim=1)  # [B,C]
        vg = v_global.reshape(B0, H, Dh).mean(dim=1)               # [B,Dh]
        vg = self.proj_v(vg)

        gmem = torch.sigmoid(self.gamma).clamp(1e-4, 1 - 1e-4)
        new_M = ((1.0 - gmem).unsqueeze(-1) * self._M
                 + gmem.unsqueeze(-1) * vg.unsqueeze(1))
        self._M = new_M.detach()

        loss = self._decorr_slot_slot(self._M) if self.use_slot_slot_decorr else None
        return {"mem_decorr": loss}

    # --- low-rank slot-wise write ---
    def write_topu_lowrank(self, V_u_ref, H, Dh, write_idx=None, teacher_sig=None):
        if V_u_ref is None or V_u_ref.numel() == 0:
            return {"mem_decorr": self._decorr_slot_slot(self._M) if (self._M is not None and self.use_slot_slot_decorr) else None}

        BP, U, C = V_u_ref.shape
        assert C == H * Dh, f"C must equal H*Dh; got C={C}, H={H}, Dh={Dh}"
        if self._M is None:
            self._ensure_mem(B=1, device=V_u_ref.device, dtype=V_u_ref.dtype)
        B = self._M.shape[0]

        vg_full = V_u_ref.mean(dim=1)                   # [(BP), C]
        vg = vg_full.reshape(BP, H, Dh).mean(dim=1)     # [(BP), Dh]
        if BP % B != 0:
            vg = vg.mean(dim=0, keepdim=True).expand(B, -1)  # [B,Dh]
        else:
            P_eff = BP // B
            vg = vg.reshape(B, P_eff, Dh).mean(dim=1)        # [B,Dh]

        vg = self.proj_v(vg)                                 # [B,Dh]

        q = self.proj_u(vg).unsqueeze(-1)                    # [B,Dh,1]
        logits = torch.bmm(self._M, q).squeeze(-1)           # [B,S]
        u = torch.softmax(logits, dim=-1)                    # [B,S]

        gmem = torch.sigmoid(self.gamma).clamp(1e-4, 1 - 1e-4)
        outer = u.unsqueeze(-1) * vg.unsqueeze(1)            # [B,S,Dh]
        new_M = ((1.0 - gmem).unsqueeze(-1) * self._M
                 + gmem.unsqueeze(-1) * outer)
        self._M = new_M.detach()

        loss = self._decorr_slot_slot(self._M) if self.use_slot_slot_decorr else None
        return {"mem_decorr": loss}
