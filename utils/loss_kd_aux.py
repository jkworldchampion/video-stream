# utils/loss_kd_aux.py
# Copyright (2025)
# Losses for KD with auxiliary non-streaming layer:
#  - DistilHuBERT-like feature similarity
#  - MiniLM-v2 style relation KL over Q/Q, K/K, V/V
#  - APC future prediction loss

from typing import Dict, Optional
import torch
import torch.nn.functional as F


def _safe_mean(x: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """Mean with zero-safe denominator."""
    return (x / denom.clamp(min=1e-8)).sum()


# -------------------------------
# 3.3.1 Feature similarity loss
# -------------------------------
def distilhubert_feature_loss(
    h: torch.Tensor,   # [B, T, C]
    z: torch.Tensor,   # [B, T, C]
    mask: Optional[torch.Tensor] = None,  # [B, T] in {0,1}
) -> torch.Tensor:
    """
    L_DIS = sum_t ( 1/D * ||h_t - z_t||_1 - log(sigmoid(cos(h_t, z_t))) )
    평균은 유효 프레임 마스크 기준으로 산정.
    """
    assert h.shape == z.shape, f"shape mismatch: {h.shape} vs {z.shape}"
    B, T, C = h.shape

    # L1 term
    l1 = (h - z).abs().sum(dim=-1) / float(C)  # [B, T]

    # cosine term
    # F.cosine_similarity는 마지막 dim 기준. 안정성을 위해 eps 사용.
    cos = F.cosine_similarity(h, z, dim=-1, eps=1e-8).clamp(min=-1.0, max=1.0)  # [B, T]
    term = l1 - torch.log(torch.sigmoid(cos) + 1e-8)  # [B, T]

    if mask is None:
        return term.mean()

    # mask: [B,T] -> 평균
    mask = mask.to(term.dtype)
    num = (term * mask).sum()
    den = mask.sum()
    return num / den.clamp(min=1e-8)


# -------------------------------------
# 3.3.2 Self-Attention relation KL loss
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


def attention_relation_kl(
    t_qkv: Dict[str, torch.Tensor],   # {"Q","K","V"} each [B, A, T, Dh]
    s_qkv: Dict[str, torch.Tensor],   # {"Q","K","V"} each [B, A, T, Dh]
    mask: Optional[torch.Tensor] = None,  # [B, T] valid frames
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    전체 프레임 범위(0..T-1)에 대해 Q/Q + K/K + V/V KL을 평균.
    - 열 마스크(mask)는 열(k) 소프트맥스에 적용되어 invalid frame을 무시
    - 행 마스크(mask)는 KL 평균에서 제외
    """
    # Teacher 분포
    pQ = _masked_row_softmax_from_proj(t_qkv["Q"], mask, eps)  # [B,A,T,T]
    pK = _masked_row_softmax_from_proj(t_qkv["K"], mask, eps)
    pV = _masked_row_softmax_from_proj(t_qkv["V"], mask, eps)

    # Student 분포
    qQ = _masked_row_softmax_from_proj(s_qkv["Q"], mask, eps)
    qK = _masked_row_softmax_from_proj(s_qkv["K"], mask, eps)
    qV = _masked_row_softmax_from_proj(s_qkv["V"], mask, eps)

    # KL(T || S) — row-wise 평균
    lq = _kl_rows(pQ, qQ, row_mask=mask)
    lk = _kl_rows(pK, qK, row_mask=mask)
    lv = _kl_rows(pV, qV, row_mask=mask)
    return lq + lk + lv


# -------------------------------
# 3.3.3 APC future prediction loss
# -------------------------------
def apc_loss(
    h: torch.Tensor,   # [B, T, C]  (teacher feature sequence)
    r: torch.Tensor,   # [B, T, C]  (student's uni-RNN outputs)
    N: int = 2,
    mask: Optional[torch.Tensor] = None,  # [B, T] valid frames
) -> torch.Tensor:
    """
    LAPC = sum_t ( 1/D * || h_{t+N} - r_t ||_1 - log(sigmoid(cos(h_{t+N}, r_t))) )
    - 유효 쌍 (t, t+N)만 평균
    """
    assert h.shape == r.shape, f"shape mismatch: {h.shape} vs {r.shape}"
    B, T, C = h.shape
    if N <= 0:
        return h.new_zeros(())

    # 유효한 t 범위: 0..T-1-N
    T_eff = T - N
    if T_eff <= 0:
        return h.new_zeros(())

    h_future = h[:, N:, :]          # [B, T-N, C]
    r_now    = r[:, :T_eff, :]      # [B, T-N, C]

    # L1
    l1 = (h_future - r_now).abs().sum(dim=-1) / float(C)  # [B, T-N]

    # cosine
    cos = F.cosine_similarity(h_future, r_now, dim=-1, eps=1e-8).clamp(min=-1.0, max=1.0)  # [B, T-N]
    term = l1 - torch.log(torch.sigmoid(cos) + 1e-8)

    if mask is None:
        return term.mean()

    # mask 쌍: m_pair[b,t] = mask[b,t] AND mask[b,t+N]
    m1 = mask[:, :T_eff]
    m2 = mask[:, N:]
    m_pair = (m1 > 0) & (m2 > 0)
    m_pair = m_pair.to(term.dtype)

    num = (term * m_pair).sum()
    den = m_pair.sum()
    return num / den.clamp(min=1e-8)
