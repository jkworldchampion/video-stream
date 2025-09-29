
# utils/loss_video_depth.py
# Author-style losses (TrimmedProcrustes, TemporalGradientMatching) + thin wrappers
# to match our training loop. Accepts either [B,T,H,W] or [B,T,1,H,W] and bool masks.

import torch
import torch.nn as nn

# ----------------------------- Reductions -----------------------------

# loss를 batch_size와 무관하게 consistency를 보장하기 위함
def _reduction_batch_based(image_loss, M):
    # average over all valid pixels in the batch
    divisor = torch.sum(M)
    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    else:
        return torch.sum(image_loss) / divisor

def _reduction_image_based(image_loss, M):
    # mean of per-image averages
    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]
    return torch.mean(image_loss)

# ----------------------------- Core pieces -----------------------------

def _gradient_loss(prediction, target, mask, reduction=_reduction_batch_based, frame_id_mask=None):
    # prediction/target/mask: [B,H,W]
    valid_id_mask_x = torch.ones_like(mask[:, :, 1:])
    valid_id_mask_y = torch.ones_like(mask[:, 1:, :])
    if frame_id_mask is not None:
        valid_id_mask_x = ((frame_id_mask[:, :, 1:] - frame_id_mask[:, :, :-1]) == 0).to(mask.dtype)
        valid_id_mask_y = ((frame_id_mask[:, 1:, :] - frame_id_mask[:, :-1, :]) == 0).to(mask.dtype)

    M = torch.sum(mask, (1, 2))

    diff  = prediction - target
    diff  = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(torch.mul(mask[:, :, 1:], mask[:, :, :-1]), valid_id_mask_x)
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(torch.mul(mask[:, 1:, :], mask[:, :-1, :]), valid_id_mask_y)
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    return reduction(image_loss, M)

def _normalize_prediction_robust(target, mask, ms=None, eps=1e-6):
    """
    target: [B,1,H,W] (float)
    mask:   [B,1,H,W] (bool/float)
    반환: (normalized_target, (median, scale))  where median/scale: [B]
    """
    # 보장된 dtype/shape
    if mask.dtype != torch.bool:
        mask = mask.bool()
    assert target.dim() == 4 and target.size(1) == 1, f"target must be [B,1,H,W], got {target.shape}"
    assert mask.dim()   == 4 and mask.size(1)   == 1, f"mask must be [B,1,H,W], got {mask.shape}"

    B, _, H, W = target.shape
    P = H * W

    # [B, P]
    t_flat = target.view(B, P)
    m_flat = mask.view(B, P)

    # 각 배치별 유효픽셀 수 [B]
    ssum = m_flat.sum(dim=1)                 # float
    valid = ssum > 0                         # [B] bool

    # median/scale 초기값
    m = torch.zeros(B, device=target.device, dtype=target.dtype)
    s = torch.ones (B, device=target.device, dtype=target.dtype)

    if ms is not None:
        # 외부에서 median/scale 주어진 경우 그대로 사용
        m, s = ms
        # 안전장치
        if m.dim() == 0: m = m.expand(B).clone()
        if s.dim() == 0: s = s.expand(B).clone()
        s = s.clamp_min(eps)
    else:
        if valid.any():
            idx = valid.nonzero(as_tuple=False).squeeze(1)   # [Nv]
            # Nv>0 인 경우에만 median/scale 계산
            tv = (m_flat[idx] * t_flat[idx])                 # [Nv, P] (마스크 적용)
            # median of masked 값들: 마스크 밖은 0이라 median에 영향 줄 수 있으므로
            # 완전히 마스킹된 픽셀은 제외하도록 boolean 인덱싱으로 다시 뽑음
            # -> 각 배치마다 따로 median을 구해야 하므로 loop (Nv는 작아서 OK)
            med_list = []
            scale_list = []
            for r in range(idx.numel()):
                rr = idx[r].item()
                vv = t_flat[rr][m_flat[rr]]                  # 1D valid 값만 모음
                if vv.numel() == 0:
                    med_list.append(target.new_tensor(0.0))
                    scale_list.append(target.new_tensor(1.0))
                else:
                    med = vv.median()
                    mad = (vv - med).abs().mean() + eps      # robust scale (L1)
                    med_list.append(med)
                    scale_list.append(mad)
            m[idx] = torch.stack(med_list)
            s[idx] = torch.stack(scale_list).clamp_min(eps)

    # 정규화
    target_centered = target - m.view(B, 1, 1, 1)
    target_norm     = target_centered / s.view(B, 1, 1, 1)

    return target_norm, (m.detach(), s.detach())

def _compute_scale_and_shift(prediction, target, mask):
    # all [B,H,W]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det   = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1

class _TrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, reduction="batch-based"):
        super().__init__()
        self.trim = trim
        if reduction == "batch-based":
            self._reduction = _reduction_batch_based
        else:
            self._reduction = _reduction_image_based

    def forward(self, prediction, target, mask, weight_mask=None):
        # prediction/target/mask: [B,H,W]
        if torch.sum(mask) == 0:
            return torch.sum(prediction) * 0.0
        M   = torch.sum(mask, (1, 2))
        res = prediction - target
        if weight_mask is not None:
            res = res * weight_mask
        res = res[mask.bool()].abs()

        trimmed, _ = torch.sort(res.view(-1), descending=False)
        keep_num   = int(len(res) * (1.0 - self.trim))
        if keep_num <= 0:
            return torch.sum(prediction) * 0.0
        trimmed = trimmed[:keep_num]
        return self._reduction(trimmed, M)

class _GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()
        self._reduction = _reduction_batch_based if reduction == "batch-based" else _reduction_image_based
        self._scales = scales

    def forward(self, prediction, target, mask, num_frame_h=1):
        # all [B,H,W]
        total = 0
        frame_id_mask = None
        if num_frame_h > 1:
            # not used in our per-frame wrapper (kept for completeness)
            frame_h = mask.shape[1] // num_frame_h
            frame_id_mask = torch.zeros_like(mask)
            for i in range(num_frame_h):
                frame_id_mask[:, i*frame_h:(i+1)*frame_h, :] = i+1

        for s in range(self._scales):
            step = pow(2, s)
            total += _gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self._reduction,
                frame_id_mask=frame_id_mask[:, ::step, ::step] if frame_id_mask is not None else None,
            )
        return total

class TrimmedProcrustesLoss(nn.Module):
    """
    Author spatial loss (per-frame): normalize + trimmed MAE + gradient regularization.
    Expects [B,H,W] tensors.
    """
    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super().__init__()
        self.data_loss = _TrimmedMAELoss(trim=trim, reduction=reduction)
        self.reg_loss  = _GradientLoss(scales=scales, reduction=reduction)
        self.alpha     = alpha

    def forward(self, prediction, target, mask, pred_ms=None, tar_ms=None, no_norm=False):
        # all [B,H,W]
        if no_norm:
            pred_ssi, pred_ms_ = prediction, (0, 1)
            tar_ssi,  tar_ms_  = target,     (0, 1)
        else:
            pred_ssi, pred_ms_ = _normalize_prediction_robust(prediction, mask, ms=pred_ms)
            tar_ssi,  tar_ms_  = _normalize_prediction_robust(target,     mask, ms=tar_ms)

        total = self.data_loss(pred_ssi, tar_ssi, mask)
        if self.alpha > 0:
            total = total + self.alpha * self.reg_loss(pred_ssi, tar_ssi, mask)
        return total

class TemporalGradientMatchingLoss(nn.Module):
    """
    Author temporal loss: operates on **depth** sequences [B,T,H,W]
    """
    def __init__(self, trim=0.2, temp_grad_scales=4, temp_grad_decay=0.5, reduction="batch-based", diff_depth_th=0.05):
        super().__init__()
        self.data_loss        = _TrimmedMAELoss(trim=trim, reduction=reduction)
        self.temp_grad_scales = temp_grad_scales
        self.temp_grad_decay  = temp_grad_decay
        self.diff_depth_th    = diff_depth_th

    def forward(self, prediction, target, mask):
        # prediction/target/mask: [B,T,H,W] (depth, not disparity)
        total = 0.0
        cnt   = 0

        min_target = torch.where(mask.bool(), target, torch.inf).amin(dim=(-1,-2)).amin(dim=1)  # [B]
        max_target = torch.where(mask.bool(), target, -torch.inf).amax(dim=(-1,-2)).amax(dim=1) # [B]
        target_th  = (max_target - min_target) * self.diff_depth_th                              # [B]

        for s in range(self.temp_grad_scales):
            temp_stride = pow(2, s)
            if temp_stride < prediction.shape[1]:
                pred_tg   = torch.diff(prediction[:, ::temp_stride, ...], dim=1)   # [B,T',H,W]
                targ_tg   = torch.diff(target    [:, ::temp_stride, ...], dim=1)   # [B,T',H,W]
                temp_mask = mask      [:, ::temp_stride, ...]
                temp_mask = temp_mask[:, 1:, ...] & temp_mask[:, :-1, ...]

                # build per-b stride thresholds
                th = target_th.view(-1, 1, 1, 1)[:, ::temp_stride, ...]            # [B,1,1,1] → stride
                if th.shape[1] < targ_tg.shape[1]:  # safe crop if shorter
                    th = th.repeat(1, targ_tg.shape[1], 1, 1)
                valid_by_th = (targ_tg.abs() < th[:, :targ_tg.shape[1], ...])
                temp_mask   = temp_mask & valid_by_th

                # flatten frames to batch for trimmed MAE
                B, Tp, H, W = pred_tg.shape
                loss_s = self.data_loss(
                    prediction=pred_tg.flatten(0,1),     # [B*Tp,H,W]
                    target=targ_tg.flatten(0,1),         # [B*Tp,H,W]
                    mask=temp_mask.flatten(0,1).float(), # [B*Tp,H,W]
                ) * pow(self.temp_grad_decay, s)
                total += loss_s
                cnt   += 1

        if cnt == 0:
            return prediction.sum() * 0.0
        return total / cnt

# ----------------------------- Shape helpers -----------------------------

def _to_BTHW(x):
    """
    Accept [B,T,H,W] or [B,T,1,H,W] -> [B,T,H,W]
    Accept [B,1,H,W]                 -> [B,1,H,W]
    """
    if x.dim() == 5:
        if x.size(2) == 1:
            return x.squeeze(2).contiguous()
        else:
            # if C>1, average across C
            return x.mean(dim=2).contiguous()
    elif x.dim() == 4:
        return x.contiguous()
    else:
        raise ValueError(f"_to_BTHW expects 4D/5D, got {tuple(x.shape)}")

def _to_mask_BTHW(m):
    """
    Mask to bool [B,T,H,W]
    Accept [B,T,H,W] (any dtype) or [B,T,1,H,W] or [B,1,H,W]
    """
    if m.dim() == 5 and m.size(2) == 1:
        m = m.squeeze(2)
    elif m.dim() == 5 and m.size(2) != 1:
        m = m.any(dim=2)
    elif m.dim() == 4:
        pass
    else:
        raise ValueError(f"_to_mask_BTHW expects 4D/5D, got {tuple(m.shape)}")
    return m.bool().contiguous()

# ----------------------------- Thin wrappers (drop-in for train.py) -----------------------------

class SpatialLossWrapper(nn.Module):
    """
    Drop-in replacement for old SSI.
    forward(pred_depth, gt_depth, mask) where inputs can be [B,1,H,W] or [B,T,H,W] (T usually 1).
    Internally uses TrimmedProcrustesLoss per frame and averages over valid frames.
    """
    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super().__init__()
        self.base = TrimmedProcrustesLoss(alpha=alpha, scales=scales, trim=trim, reduction=reduction)

    def forward(self, pred_depth, gt_depth, mask):
        # 표준화: [B,T,H,W]
        P = _to_BTHW(pred_depth)       # [B,T,H,W]
        G = _to_BTHW(gt_depth)         # [B,T,H,W]
        M = _to_mask_BTHW(mask).bool() # [B,T,H,W]

        if P.shape != G.shape or P.shape != M.shape:
            raise RuntimeError(f"SpatialLossWrapper shape mismatch: pred={P.shape}, gt={G.shape}, mask={M.shape}")

        B, T, H, W = P.shape

        total = None
        valid_cnt = 0

        for t in range(T):
            p = P[:, t]    # [B,H,W]
            g = G[:, t]    # [B,H,W]
            m = M[:, t]    # [B,H,W] (bool)

            # 유효 픽셀 없는 프레임은 스킵 (그래프는 유지)
            if m.sum() == 0:
                continue

            # TrimmedProcrustesLoss는 [B,1,H,W]를 기대하므로 채널 차원 추가
            p4 = p.unsqueeze(1)              # [B,1,H,W]
            g4 = g.unsqueeze(1)              # [B,1,H,W]
            m4 = m.unsqueeze(1).float()      # [B,1,H,W] float mask

            loss_t = self.base(p4, g4, m4)

            total = loss_t if (total is None) else (total + loss_t)
            valid_cnt += 1

        if valid_cnt == 0:
            # 전 프레임 무효 → 0-loss (그래프 유지)
            return (P * 0.0).sum()

        return total / valid_cnt

class TemporalLossWrapper(nn.Module):
    """
    Drop-in replacement for our old TGM:
    forward(pred_depth_seq, gt_depth_seq, mask_seq) with [B,T,H,W] (T>=2).
    Internally uses TemporalGradientMatchingLoss (author).
    """
    def __init__(self, trim=0.2, temp_grad_scales=4, temp_grad_decay=0.5, reduction="batch-based", diff_depth_th=0.05):
        super().__init__()
        self.base = TemporalGradientMatchingLoss(
            trim=trim, temp_grad_scales=temp_grad_scales, temp_grad_decay=temp_grad_decay,
            reduction=reduction, diff_depth_th=diff_depth_th
        )

    def forward(self, pred_depth_seq, gt_depth_seq, mask_seq):
        P = _to_BTHW(pred_depth_seq)
        G = _to_BTHW(gt_depth_seq)
        M = _to_mask_BTHW(mask_seq)

        if P.shape != G.shape or P.shape != M.shape:
            raise RuntimeError(f"TemporalLossWrapper shape mismatch: pred={P.shape}, gt={G.shape}, mask={M.shape}")

        return self.base(P, G, M)
