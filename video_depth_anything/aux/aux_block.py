# requirements: pip install mamba-ssm
from mamba_ssm import Mamba
import torch
import torch.nn as nn

class AuxBlock(nn.Module):
    """
    Proj -> Transformer(1L, maskable) -> {Uni-LSTM or Mamba}
    x: [B, T, C_in]  -> z:[B, T, C_t], r:[B, T, C_t]
    """
    def __init__(
        self,
        c_in: int,
        c_teacher: int,
        nhead: int = 8,
        dropout: float = 0.0,
        return_attn: bool = False,
        return_qkv: bool = False,
        rnn_type: str = "lstm",      # "lstm" | "mamba"
        # --- mamba hparams ---
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.proj = nn.Linear(c_in, c_teacher)

        # Transformer 1L (maskable)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=c_teacher, num_heads=nhead, batch_first=True, dropout=dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(c_teacher, 4*c_teacher),
            nn.GELU(),
            nn.Linear(4*c_teacher, c_teacher),
        )
        self.norm1 = nn.LayerNorm(c_teacher)
        self.norm2 = nn.LayerNorm(c_teacher)
        self.dropout = nn.Dropout(dropout)

        # Sequence model head
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            self.seq_model = nn.LSTM(
                input_size=c_teacher, hidden_size=c_teacher,
                num_layers=1, batch_first=True, bidirectional=False
            )
        elif self.rnn_type == "mamba":
            # Mamba: causal SSM block (B,T,C) -> (B,T,C)
            self.seq_model = Mamba(
                d_model=c_teacher,
                d_state=mamba_d_state,   # memory size per channel
                d_conv=mamba_d_conv,     # conv kernel for short-range mixing
                expand=mamba_expand,     # channel expansion
            )
        else:
            raise ValueError(f"Unknown rnn_type: {self.rnn_type}")

        self.return_attn = return_attn
        self.return_qkv = return_qkv

    @staticmethod
    def build_mask(T: int, mode: str, N: int, device):
        """
        mode='causal' | 'bandN' | 'none'
        - 'causal': mask all future j>t
        - 'bandN' : mask j in [t+1..t+N] only
        - 'none'  : no mask
        returns attn_mask [T,T] with True=masked
        """
        if mode == "none":
            return None
        m = torch.zeros(T, T, dtype=torch.bool, device=device)
        if mode == "causal":
            m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        elif mode == "bandN":
            if N > 0:
                for t in range(T):
                    j0, j1 = t+1, min(t+N, T-1)
                    if j0 <= j1:
                        m[t, j0:j1+1] = True
        else:
            raise ValueError(mode)
        return m

    def _compute_qkv(self, x):  # identical to earlier (omitted here for brevity)
        # ... (same as I gave before)
        raise NotImplementedError

    def forward(
        self, x_bt_c: torch.Tensor,
        apc_N: int = 0,
        transformer_mask_mode: str = "causal",   # "causal" (A) | "bandN" (B) | "none"
    ):
        """
        transformer_mask_mode:
          - "causal": 완전 인과 (추천)
          - "bandN":  논문 parity ([t+1..t+N]만 차단)
          - "none":  마스크 없음 (권장X)
        """
        B, T, _ = x_bt_c.shape
        x = self.proj(x_bt_c)  # [B,T,C_t]

        # Transformer 1L with mask
        _x = self.norm1(x)
        attn_mask = self.build_mask(T, transformer_mask_mode, apc_N, x.device)
        z, attn = self.self_attn(_x, _x, _x, need_weights=True,
                                 attn_mask=attn_mask, average_attn_weights=False)
        z = x + self.dropout(z)
        z = z + self.dropout(self.ffn(self.norm2(z)))   # [B,T,C_t]

        # Sequence model head
        if self.rnn_type == "lstm":
            r, _ = self.seq_model(z)                    # [B,T,C_t]
        else:  # mamba
            r = self.seq_model(z)                       # [B,T,C_t]

        aux = {}
        if self.return_attn:
            aux["attn"] = attn  # [B, A, T, T]
        if self.return_qkv:
            # Optional: implement _compute_qkv similarly to earlier if you need Q/K/V
            pass

        return z, r, aux
