# motion_module/teacher_signals.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class TeacherSignals:
    attn_window: Optional[torch.Tensor] = None   # [B,H,T,T] or rows
    delta_bar:  Optional[torch.Tensor] = None    # [B,C] or [B,1,C]
    top_r:      Optional[torch.Tensor] = None    # [B,P,R]
    top_u:      Optional[torch.Tensor] = None    # [B,P,U]
    time_lag:   Optional[torch.Tensor] = None    # [B,P,T]
    motion_score: Optional[torch.Tensor] = None  # [B,P,T]
