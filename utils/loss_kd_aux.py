# utils/loss_kd_aux.py
# Copyright (2025)
# Losses for KD with auxiliary non-streaming layer:
#  - MiniLM-v2 style relation KL over Q/Q, K/K, V/V
# Note: DIS (feature similarity) and APC (future prediction) removed for better performance

from typing import Dict, Optional
import torch
import torch.nn.functional as F
import math


def _safe_mean(x: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """Mean with zero-safe denominator."""
    return (x / denom.clamp(min=1e-8)).sum()


# -------------------------------------
# Self-Attention relation KL loss
# -------------------------------------
def _relation_row_from_proj(
    proj: torch.Tensor,     # [B, A, T, Dh]
    eps: float = 1e-8
) -> torch.Tensor:
    """
    행 분포 R(a,t,:) = Softmax_k ( <proj[a,t], proj[a,k]> / sqrt(Dh) )
    반환: prob [B, A, T, T]
    """
    B, A, T, Dh = proj.shape
    scale = Dh ** -0.5

    # 유사도 S[b,a,t,k] = dot(proj[b,a,t,:], proj[b,a,k,:]) / sqrt(Dh)
    # 구현: (B*A, T, Dh) @ (B*A, Dh, T) -> (B*A, T, T)
    x = proj.reshape(B * A, T, Dh)
    sim = torch.bmm(x, x.transpose(1, 2)) * scale  # [B*A, T, T]
    prob = F.softmax(sim, dim=-1)                   # row-wise softmax over k
    prob = prob.reshape(B, A, T, T).clamp_min(eps)
    return prob


def _masked_row_softmax_from_proj(
    proj: torch.Tensor,     # [B, A, T, Dh]
    mask: Optional[torch.Tensor],  # [B, T] in {0,1}
    eps: float = 1e-8
) -> torch.Tensor:
    """
    마스크를 열축(k)에 적용한 소프트맥스 분포. 행(t) 자체가 invalid면 이후 평균 시 제외됨.
    반환: prob [B, A, T, T]
    """
    B, A, T, Dh = proj.shape
    scale = Dh ** -0.5
    x = proj.reshape(B * A, T, Dh)                    # [BA, T, Dh]
    sim = torch.bmm(x, x.transpose(1, 2)) * scale     # [BA, T, T]

    if mask is not None:
        m = mask.to(sim.dtype)                        # [B, T]
        m = m.unsqueeze(1)                            # [B,1,T]
        m = m.repeat(1, A, 1).reshape(B * A, 1, T)    # [BA,1,T]
        # invalid columns -> -inf
        sim = sim.masked_fill(m == 0, float("-inf"))

    prob = F.softmax(sim, dim=-1)                     # [BA, T, T]
    prob = prob.reshape(B, A, T, T).clamp_min(eps)
    return prob


def _kl_rows(
    p: torch.Tensor,    # [B, A, T, T]
    q: torch.Tensor,    # [B, A, T, T]
    row_mask: Optional[torch.Tensor] = None,  # [B, T] (해당 행 유효 여부)
) -> torch.Tensor:
    """
    KL(p||q) = sum_k p * (log p - log q)  (행 단위)
    평균은 (유효 행 수 * A)로 나눔.
    """
    # 안정성
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)

    kl = (p * (p.log() - q.log())).sum(dim=-1)   # [B, A, T]

    if row_mask is not None:
        m = row_mask.to(kl.dtype).unsqueeze(1)   # [B,1,T]
        kl = kl * m
        denom = (m.sum(dim=-1) * 1.0).sum()      # sum over B and A: (B,A)
    else:
        denom = kl.numel() / kl.shape[-1]        # equals B*A*T -> divide by A*T per batch

    return kl.sum() / denom.clamp(min=1e-8)


def attention_relation_kl(t_qkv, s_qkv, mask=None, eps=1e-8, q_mask=None, k_mask=None):
    """
    Distill Query-Query, Key-Key, Value-Value attention relation (MiniLM-v2 style).
    - t_qkv/s_qkv: dict with {"Q","K","V"} each [B,A,T,Dh]
    - q_mask: [B,T] -> which t rows (anchors) to average over (e.g., mask_last)
    - k_mask: [B,T] -> which k columns to keep inside the softmax (e.g., frame_valid)
    - mask (legacy): if provided and q_mask/k_mask are None, use it for BOTH rows & cols (backward compat)
    returns: scalar tensor
    """
    def _rel_kl(qT, qS, row_mask, col_mask):
        # qT,qS: [B,A,T,Dh]
        B, A, T, Dh = qT.shape
        scale = 1.0 / math.sqrt(Dh)
        # logits over (t,k): [B,A,T,T]
        logitT = torch.einsum('batd,bakd->batk', qT, qT) * scale
        logitS = torch.einsum('batd,bakd->batk', qS, qS) * scale

        # column mask only on k-axis
        if col_mask is not None:
            col = col_mask[:, None, None, :].to(dtype=logitT.dtype)
            neginf = torch.finfo(logitT.dtype).min
            logitT = logitT.masked_fill(col == 0, neginf)
            logitS = logitS.masked_fill(col == 0, neginf)

        PT = torch.softmax(logitT, dim=-1)
        PS = torch.softmax(logitS, dim=-1).clamp_min(eps)

        # KL over k: [B,A,T]
        kl = (PT * (PT.clamp_min(eps).log() - PS.log())).sum(dim=-1)

        # average over t with row_mask
        if row_mask is not None:
            rm = row_mask[:, None, :].to(dtype=kl.dtype)  # [B,1,T]
            num = (kl * rm).sum()
            den = rm.sum().clamp_min(1.0)
            kl_mean = num / den
        else:
            kl_mean = kl.mean()

        # average over A, B
        return kl_mean / A

    # backward compatibility
    if (q_mask is None) and (k_mask is None) and (mask is not None):
        q_mask = mask
        k_mask = mask

    QT_t, QT_s = t_qkv["Q"], s_qkv["Q"]
    KT_t, KT_s = t_qkv["K"], s_qkv["K"]
    VT_t, VT_s = t_qkv["V"], s_qkv["V"]

    Lq = _rel_kl(QT_t, QT_s, q_mask, k_mask)
    Lk = _rel_kl(KT_t, KT_s, q_mask, k_mask)
    Lv = _rel_kl(VT_t, VT_s, q_mask, k_mask)
    return Lq + Lk + Lv
