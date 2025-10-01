import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based, frame_id_mask=None):
    # mask for distinguish different frames
    valid_id_mask_x = torch.ones_like(mask[:, :, 1:])
    valid_id_mask_y = torch.ones_like(mask[:, 1:, :])
    if frame_id_mask is not None:
        valid_id_mask_x = ((frame_id_mask[:, :, 1:] - frame_id_mask[:, :, :-1]) == 0).to(mask.dtype)
        valid_id_mask_y = ((frame_id_mask[:, 1:, :] - frame_id_mask[:, :-1, :]) == 0).to(mask.dtype)
    
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(torch.mul(mask[:, :, 1:], mask[:, :, :-1]), valid_id_mask_x)
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(torch.mul(mask[:, 1:, :], mask[:, :-1, :]), valid_id_mask_y)
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

def normalize_prediction_robust(target, mask, ms=None):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    if ms is None:
        m = torch.zeros_like(ssum)
        s = torch.ones_like(ssum)

        # m[valid] = torch.median((mask[valid] * target[valid]).view(valid.sum(), -1), dim=1).values
        mv = (mask * target)                # [N,H,W]
        N = target.shape[0]
        for i in range(N):
            if valid[i]:
                vi = mv[i][mask[i].bool()].view(-1)
                if vi.numel() > 0:
                    m[i] = vi.median()
                else:
                    m[i] = 0.0
    else:
        m, s = ms

    target = target - m.view(-1, 1, 1)

    if ms is None:
        sq = torch.sum(mask * target.abs(), (1, 2))
        # s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)
        # --- (B) boolean 인덱싱 → where로 대체 ---
        s_candidate = torch.clamp(sq / ssum.clamp_min(1e-6), min=1e-6)  # [N]
        s = torch.where(valid, s_candidate, s)  # [N]

    return target / (s.view(-1, 1, 1)), (m.detach(), s.detach())


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1

def _masked_quantile_1d(x, m, q):
    # x: [N], m: [N] bool
    vals = x[m]
    if vals.numel() == 0:
        return None
    return torch.quantile(vals, q)

def _robust_range(gt_pair, mask_pair):
    # gt_pair, mask_pair: [B,2,H,W] (mask: bool)
    B, _, H, W = gt_pair.shape
    rr = torch.empty(B, device=gt_pair.device, dtype=gt_pair.dtype)
    for b in range(B):
        g = gt_pair[b].reshape(-1)
        m = mask_pair[b].reshape(-1)
        lo = _masked_quantile_1d(g, m, 0.05)
        hi = _masked_quantile_1d(g, m, 0.95)
        if (lo is None) or (hi is None) or not torch.isfinite(lo) or not torch.isfinite(hi) or (hi - lo) <= 0:
            rr[b] = torch.tensor(1.0, device=gt_pair.device, dtype=gt_pair.dtype)  # fallback
        else:
            rr[b] = (hi - lo).clamp_min(1e-6)
    return rr  # [B]

def _trimmed_huber(residual, trim=0.4, delta=0.03):
    # residual: [N] (>=0), 트림 후 허버 평균
    if residual.numel() == 0:
        return residual.sum() * 0.0
    k = int((1.0 - trim) * residual.numel())
    if k <= 0:
        return residual.sum() * 0.0
    # 작은 값 우선 유지
    sorted_vals, _ = torch.sort(residual, descending=False)
    kept = sorted_vals[:k]
    # Huber
    absr = kept
    quad = 0.5 * (absr.clamp(max=delta) ** 2) / delta
    lin  = absr - 0.5 * delta
    hub  = torch.where(absr <= delta, quad, lin)
    return hub.mean()

class StreamingTemporalConsistencyLoss(nn.Module):
    """
    CRTH-TGM: Causal, Range-Normalized, Trimmed, Huber Temporal Consistency for streaming (T=2)
    입력:
      pred_pair   : [B,2,H,W] (raw pred depth; 외부 정렬 없음)
      gt_pair     : [B,2,H,W]
      mask_pair   : [B,2,H,W] (float or bool)
      prev_pred_grad: Optional [B,1,H,W] (직전 스텝의 Δpred; 같은 정렬 규칙 하 동일 도메인) 
    반환:
      total_loss, aux, curr_pred_grad(=Δpred aligned, detach)
    """
    def __init__(self, diff_ratio=0.01, trim=0.4, huber_delta=0.03, lambda_accel=0.2, range_norm=True):
        super().__init__()
        self.diff_ratio   = diff_ratio
        self.trim         = trim
        self.huber_delta  = huber_delta
        self.lambda_accel = lambda_accel
        self.range_norm   = range_norm

    def forward(self, pred_pair, gt_pair, mask_pair, prev_pred_grad=None):
        # dtypes & shapes
        if mask_pair.dtype != torch.bool:
            mask_bool = mask_pair > 0.5
        else:
            mask_bool = mask_pair

        B, T, H, W = pred_pair.shape
        assert T == 2, f"Streaming loss expects T=2, got T={T}"

        # 공통 스케일·시프트 정렬(두 프레임 concat 후 한 번): pred → gt
        # compute_scale_and_shift expects [B,H,W], 합치기 위해 [B,2*H,W]로 reshape
        from loss.loss import compute_scale_and_shift  # 기존 함수 재사용
        P_flat = pred_pair.reshape(B, 2 * H, W)
        G_flat = gt_pair.reshape(B, 2 * H, W)
        M_flat = mask_bool.reshape(B, 2 * H, W).float()
        s, t = compute_scale_and_shift(P_flat, G_flat, M_flat)  # [B], [B]
        s = s.view(B, 1, 1, 1); t = t.view(B, 1, 1, 1)
        pred_aligned = s * pred_pair + t  # [B,2,H,W]

        # 시간차 & 유효 마스크(두 프레임 모두 유효)
        dpred = (pred_aligned[:, 1] - pred_aligned[:, 0])                      # [B,H,W]
        dgt   = (gt_pair[:, 1]       - gt_pair[:, 0])
        m12   = mask_bool[:, 1] & mask_bool[:, 0]                              # [B,H,W]

        # 로버스트 범위 & 정적 임계값
        rr = _robust_range(gt_pair, mask_bool)                                 # [B]
        th = (self.diff_ratio * rr).view(B, 1, 1)                              # [B,1,1]
        static = m12 & (dgt.abs() < th)                                        # [B,H,W]

        # 범위 정규화(선택)
        if self.range_norm:
            scale = rr.view(B, 1, 1).clamp_min(1e-6)
            r_data = (dpred - dgt).abs() / scale                               # [B,H,W]
        else:
            r_data = (dpred - dgt).abs()

        # 데이터 항: 정적&유효 픽셀 중 trimmed huber
        loss_data = []
        for b in range(B):
            rb = r_data[b][static[b]]
            loss_data.append(_trimmed_huber(rb, trim=self.trim, delta=self.huber_delta))
        loss_data = torch.stack(loss_data).mean() if len(loss_data) > 0 else r_data.sum() * 0.0

        # 가속도 항(선택): Δpred_t vs prev Δpred_{t-1}
        if (prev_pred_grad is not None) and (self.lambda_accel > 0):
            prev = prev_pred_grad  # [B,1,H,W] or [B,H,W]
            if prev.dim() == 4 and prev.size(1) == 1:
                prev = prev[:, 0]
            if self.range_norm:
                r_acc = (dpred - prev).abs() / scale
            else:
                r_acc = (dpred - prev).abs()
            loss_acc = []
            for b in range(B):
                rb = r_acc[b][static[b]]  # 동일 static gating
                loss_acc.append(_trimmed_huber(rb, trim=self.trim, delta=self.huber_delta))
            loss_acc = torch.stack(loss_acc).mean() if len(loss_acc) > 0 else r_acc.sum() * 0.0
        else:
            loss_acc = r_data.sum() * 0.0

        total = loss_data + self.lambda_accel * loss_acc
        aux = {"data": loss_data.detach(), "accel": loss_acc.detach(), "rr_mean": rr.mean().detach()}

        # 다음 스텝용 현재 Δpred 저장(정렬된 기준으로)
        curr_grad = dpred.detach().unsqueeze(1)  # [B,1,H,W]
        return total, aux, curr_grad

class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None
        self.__prediction_median_scale = None
        self.__target_median_scale = None

    def forward(self, prediction, target, mask, pred_ms=None, tar_ms=None, num_frame_h=1, no_norm=False):
        if no_norm:
            self.__prediction_ssi, self.__prediction_median_scale = prediction, (0, 1)
            target_, self.__target_median_scale = target, (0, 1)
        else:
            self.__prediction_ssi, self.__prediction_median_scale = normalize_prediction_robust(prediction, mask, ms=pred_ms)
            target_, self.__target_median_scale = normalize_prediction_robust(target, mask, ms=tar_ms)

        total = self.__data_loss(self.__prediction_ssi, target_, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask, num_frame_h=num_frame_h
            )

        return total

    def get_median_scale(self):
        return self.__prediction_median_scale, self.__target_median_scale

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class TrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.trim = trim

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask, weight_mask=None):
        if torch.sum(mask) == 0:
            return torch.sum(prediction) * 0.0
        M = torch.sum(mask, (1, 2))
        res = prediction - target
        if weight_mask is not None:
            res = res * weight_mask
        res = res[mask.bool()].abs()
        trimmed, _ = torch.sort(res.view(-1), descending=False)
        keep_num = int(len(res) * (1.0 - self.trim))
        if keep_num <= 0:
            return torch.sum(prediction) * 0.0
        trimmed = trimmed[: keep_num]

        return self.__reduction(trimmed, M)

    
class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask, num_frame_h=1):
        total = 0

        frame_id_mask = None
        if num_frame_h > 1:
            frame_h = mask.shape[1] // num_frame_h
            frame_id_mask = torch.zeros_like(mask)
            for i in range(num_frame_h):
                frame_id_mask[:, i*frame_h:(i+1)*frame_h, :] = i+1

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
                frame_id_mask=frame_id_mask[:, ::step, ::step] if num_frame_h > 1 else None,
            )

        return total


class TemporalGradientMatchingLoss(nn.Module):
    def __init__(self, trim=0.2, temp_grad_scales=4, temp_grad_decay=0.5, reduction="batch-based", diff_depth_th=0.01):
        super().__init__()
        self.data_loss = TrimmedMAELoss(trim=trim, reduction=reduction)
        self.temp_grad_scales = temp_grad_scales
        self.temp_grad_decay = temp_grad_decay
        self.diff_depth_th = diff_depth_th

    def forward(self, prediction, target, mask):
        """
        prediction: [B, T, H, W]
        target    : [B, T, H, W]
        mask      : [B, T, H, W]  (float 또는 bool)
        """
        # ---- dtype 정리 ----
        if mask.dtype != torch.bool:
            mask_bool = mask > 0.5
        else:
            mask_bool = mask

        B, T, H, W = prediction.shape

        # (A) 프레임 2개 미만이면 temporal 항 없음
        if T < 2:
            return prediction.sum() * 0.0

        total = 0.0
        cnt = 0

        # (B) 타깃 기반 임계값 (배치 스칼라) 계산
        #     마스크가 0인 곳은 +inf/-inf로 제외
        min_target = torch.where(mask_bool, target, torch.inf).amin(dim=(1, 2, 3))   # [B]
        max_target = torch.where(mask_bool, target, -torch.inf).amax(dim=(1, 2, 3))  # [B]
        target_th = (max_target - min_target) * self.diff_depth_th                    # [B]

        for scale in range(self.temp_grad_scales):
            temp_stride = 2 ** scale
            if temp_stride >= T:
                continue

            # 시간 축 다운샘플링 후 차분
            pred_sub   = prediction[:, ::temp_stride, ...]  # [B, T_s, H, W]
            target_sub = target[:, ::temp_stride, ...]
            mask_sub   = mask_bool[:, ::temp_stride, ...]

            if pred_sub.size(1) < 2:
                continue

            pred_temp_grad   = torch.diff(pred_sub,   dim=1)     # [B, T_s-1, H, W]
            target_temp_grad = torch.diff(target_sub, dim=1)     # [B, T_s-1, H, W]
            # 두 시점 모두 유효한 위치만
            temp_mask = mask_sub[:, 1:, ...] & mask_sub[:, :-1, ...]  # bool [B, T_s-1, H, W]

            # 타깃 변화량 기반 정적 영역 필터: |Δgt| < th
            # target_th: [B] → [B, T_s-1, H, W]로 브로드캐스트
            th = target_th.view(B, 1, 1, 1)
            valid_from_th = (target_temp_grad.abs() < th)           # [B, T_s-1, H, W]
            temp_mask = temp_mask & valid_from_th                   # bool

            # 이 스케일에서 유효한 픽셀이 없으면 스킵
            if not temp_mask.any():
                continue

            # Trimmed MAE는 mask=float 기대 → 마지막에만 float 캐스팅
            total += self.data_loss(
                prediction=pred_temp_grad.flatten(0, 1),
                target=target_temp_grad.flatten(0, 1),
                mask=temp_mask.flatten(0, 1).float()
            ) * (self.temp_grad_decay ** scale)
            cnt += 1

        # (C) 어떤 스케일에서도 기여가 없으면 0 반환(스트리밍 초반 방어)
        if cnt == 0:
            return prediction.sum() * 0.0

        return total / cnt


class VideoDepthLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, trim=0.0, stable_scale=10, reduction="batch-based"):
        super().__init__()
        self.spatial_loss = TrimmedProcrustesLoss(alpha=alpha, scales=scales, trim=trim, reduction=reduction)
        self.stable_loss = TemporalGradientMatchingLoss(trim=trim, reduction=reduction, temp_grad_decay=0.5, temp_grad_scales=1)
        self.stable_scale = stable_scale

    def forward(self, prediction, target, mask):
        '''
            prediction: Shape(B, T, H, W)
            target: Shape(B, T, H, W)
            mask: Shape(B, T, H, W)
        '''
        loss_dict = {}
        total = 0
        loss_dict['spatial_loss'] = self.spatial_loss(prediction=prediction.flatten(0, 1), target=target.flatten(0, 1), mask=mask.flatten(0, 1).float())
        total += loss_dict['spatial_loss']
        scale, shift = compute_scale_and_shift(prediction.flatten(1,2), target.flatten(1,2), mask.flatten(1,2))
        prediction = scale.view(-1, 1, 1, 1) * prediction + shift.view(-1, 1, 1, 1)
        loss_dict['stable_loss'] = self.stable_loss(prediction=prediction, target=target, mask=mask) * self.stable_scale
        total += loss_dict['stable_loss']

        loss_dict['total_loss'] = total
        return loss_dict
