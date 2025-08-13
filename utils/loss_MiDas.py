# This version is different from the Depth Anything V2 loss function.
# It is same as the MiDaS loss function. It uses the MSE(not MAE) for the loss calculation.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_ssi_basic(nn.Module):
    # DA와는 다르게, 들어오는 차원이 B x N x 1 x H x W 임 !!
    # mask 차원 : [B, T, H, W]
    def __init__(self, eps=1e-8, trim=0.2):
        super().__init__()
        self.eps = eps
        self.trim = trim # trim 비율 추가함

    def _d_hat(self, d, mask):
        ## 자 지금 d input : B , N , H , W 임 .
        ## -> 일단 끝에 2차원에다가 각 마스크를 씌워줘야함
        B, N, H, W = mask.shape

        flat_d    = d.view(B * N, H * W)
        flat_mask = mask.contiguous().view(B * N, H * W)

        medians = torch.zeros((B * N,), device=d.device, dtype=d.dtype)
        scales  = torch.zeros((B * N,), device=d.device, dtype=d.dtype)

        for idx in range(B * N):
            vec       = flat_d[idx] # H*W 짜리 1d 텐서
            vec_mask  = flat_mask[idx]

            valid_vals = vec[vec_mask]   # 이러면 1차원 텐서 나옴

            if valid_vals.numel() == 0:
                # 만약 valid 없을때 처리
                med  = vec.new_tensor(0.0)
                sc   = vec.new_tensor(self.eps)
            else:
                # t(d)
                med = torch.median(valid_vals)  # -> 지금 valid_vals는 1d 텐서 여기서 median
                # s(d)
                abs_diff = torch.abs(valid_vals - med)
                sc = abs_diff.mean() + self.eps  

            medians[idx] = med  # (B*N) 차원 텐서
            scales[idx]  = sc

        median_matrix = medians.unsqueeze(1).expand(-1, H * W)  # (B*N, H*W)로 복원
        scale_matrix  = scales.unsqueeze(1).expand(-1, H * W)

        d_hat_flat = (flat_d - median_matrix) / scale_matrix

        d_hat = d_hat_flat.view(B, N, H, W) ## 이러고 나면, scale과 median이 valid 기준으로 만들어짐 ! !

        return d_hat
    
    def _rho(self, pred, y, mask):
        diff = torch.abs(self._d_hat(pred, mask) - self._d_hat(y, mask))
        return diff 

    def _grad_loss(self, pred, target, mask):
        total = 0.0
        ## R이 scale map -> 4 scale level로 나눴다. 즉 4면 2^4 = 16
        for s in range(4):
            if s > 0:
                k = 2 ** s
                p = F.avg_pool2d(pred, k, stride=k)
                t = F.avg_pool2d(target, k, stride=k)
                m = F.avg_pool2d(mask.float(), k, stride=k).bool()
            else:
                p, t, m = pred, target, mask

            diff = p - t
            gx = (diff[..., :, 1:] - diff[..., :, :-1]).abs()
            gy = (diff[..., 1:, :] - diff[..., :-1, :]).abs()

            mx = m[..., :, 1:] & m[..., :, :-1]
            my = m[..., 1:, :] & m[..., :-1, :]

            ## 둘이 혹시나 다를 수도 있나? 일단 쪼개서
            lx = (gx * mx).sum() / mx.sum().clamp(min=1.0)
            ly = (gy * my).sum() / my.sum().clamp(min=1.0)
            
            total += lx + ly

        return total / 4
        
    def forward(self, pred, y, masks_squeezed):
        # mask 차원 : [B, T, H, W]
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)

        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)
            
        masks_squeezed = masks_squeezed.bool()  

        rho = self._rho(pred, y, masks_squeezed)        ## 리턴차원 : B T H W
        rho = rho * masks_squeezed.float()  # 표현 수정

        # trimm MAE 계산 루프
        B, T, H, W = rho.shape
        loss_frames = []
    
        for b in range(B):
            for t in range(T):
                # (5-1) 프레임 b,t 의 valid 오차들만 추출
                errors = rho[b, t][masks_squeezed[b, t]]
                if errors.numel() == 0:
                    continue
    
                # (5-2) 에러 오름차순 정렬
                sorted_err, _ = torch.sort(errors)
                # (5-3) 상위 trim 비율만큼 제거
                keep_num = int(sorted_err.numel() * (1.0 - self.trim))
                if keep_num <= 0:
                    continue
                trimmed = sorted_err[:keep_num]
    
                # (5-4) 나머지에 대한 평균
                loss_frames.append(trimmed.mean())
            
        # 배치·프레임 전체 평균
        if len(loss_frames) == 0:
            loss_ssi = torch.tensor(0.0, device=pred.device)
        else:
            loss_ssi = torch.stack(loss_frames).mean()                    
        
        loss_grad = self._grad_loss(
            self._d_hat(pred,masks_squeezed), 
            self._d_hat(y,masks_squeezed), 
            masks_squeezed
        )

        total = loss_ssi + 0.5 * loss_grad
        # print("SSI Loss per batch:", loss_ssi.item())

        return total



class Loss_ssi(nn.Module):
    # DA와는 다르게, 들어오는 차원이 B x N x 1 x H x W 임 !!
    # mask 차원 : [B, T, H, W]
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _d_hat(self, d, ref, mask):  # 정렬해야할 ref 추가
        ## 자 지금 d input : B , N , H , W 임 .
        ## -> 일단 끝에 2차원에다가 각 마스크를 씌워줘야함
        B, N, H, W = mask.shape

        # 1) 각 프레임별로 flatten: (B*N, H*W)
        flat_d    = d.view(B * N, H * W)
        flat_ref = ref.view(B * N, H * W)  # ref도 flatten
        flat_mask = mask.contiguous().view(B * N, H * W)
        
        # 2) valid pixel 개수 per frame: shape = (B*N, 1)
        valid_counts = flat_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # valid 픽셀 수 (B*N, 1)

        # 3) masked means μ_d, μ_ref per frame
        sum_d   = (flat_d   * flat_mask).sum(dim=1, keepdim=True)  # (B*N, 1)
        sum_ref = (flat_ref * flat_mask).sum(dim=1, keepdim=True)  # (B*N, 1)
        mu_d    = sum_d   / valid_counts  # (B*N, 1)
        mu_ref  = sum_ref / valid_counts  # (B*N, 1)

        # 4) centered differences
        d_diff   = flat_d   - mu_d    # (B*N, M)
        ref_diff = flat_ref - mu_ref  # (B*N, M)

        # 5) numerator, denom for least squares per frame
        numerator = torch.sum((d_diff * ref_diff) * flat_mask, dim=1, keepdim=True)  # (B*N, 1)
        denom     = torch.sum((d_diff * d_diff)   * flat_mask, dim=1, keepdim=True)  # (B*N, 1)

        # 6) scale s and shift t per frame
        s = numerator / (denom + self.eps)      # (B*N, 1)
        t = mu_ref - s * mu_d                   # (B*N, 1)

        # 7) aligned prediction: d_hat_flat = s * d_flat + t
        d_hat_flat = s * flat_d + t             # (B*N, M)

        # 8) reshape back to (B, N, H, W)
        d_hat = d_hat_flat.view(B, N, H, W)     # (B, N, H, W)
        return d_hat

    def _rho(self, pred, y, mask):
        # pred를 y에 맞춰서 정렬
        aligned_pred = self._d_hat(pred, y, mask)  # (B, N, H, W)
        return (aligned_pred - y) ** 2

    def forward(self, pred, y, masks_squeezed):
        # mask 차원 : [B, T, H, W]
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)

        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)
            
        masks_squeezed = masks_squeezed.bool()  

        rho = self._rho(pred, y, masks_squeezed)        ## 리턴차원 : B T H W
        rho[~masks_squeezed] = 0

        # valid_counts = masks_squeezed.sum(dim=-1).clamp_min(1.0)
        # loss_per_image = rho.sum(dim=-1) / valid_counts
        # loss_ssi = loss_per_image.mean()
        
        valid_counts = masks_squeezed.sum(dim=(2, 3)).clamp_min(1.0)  # shape = (B, N)
        sum_rho = rho.sum(dim=(2, 3))  # shape = (B, N)
        loss_per_frame = sum_rho / valid_counts  # shape = (B, N)

        loss_ssi = loss_per_frame.mean()  # scalar
        # print("SSI Loss per batch:", loss_ssi.item())

        return loss_ssi



class LossTGMVector(nn.Module):
    """
    pred_disp : [B, T, H, W]  predicted disparity
    gt_depth  : [B, T, H, W]  ground-truth depth (m)
    mask      : [B, T, H, W]  valid mask (bool)
    """
    def __init__(self, diff_depth_th=0.05, trim_ratio=0.2, eps=1e-6):
        super().__init__()
        self.diff_depth_th  = diff_depth_th
        self.trim_ratio = trim_ratio
        self.eps        = eps

    def forward(self, pred_disp, gt_depth, mask):
        # remove channel=1 dimension if present
        if pred_disp.dim() == 5 and pred_disp.size(2) == 1:
            pred_disp = pred_disp.squeeze(2)   # [B,T,H,W]
        if gt_depth.dim() == 5 and gt_depth.size(2) == 1:
            gt_depth = gt_depth.squeeze(2)
        if mask.dim() == 5 and mask.size(2) == 1:
            mask = mask.squeeze(2)
        
        B, T, H, W = pred_disp.shape
        if T < 2:
            return torch.tensor(0., device=pred_disp.device)

        # 1) raw pred disparity & gt disparity
        raw_pred_disp = pred_disp.clamp(min=self.eps)               # [B,T,H,W]
        gt_disp       = 1.0 / gt_depth.clamp(min=self.eps)         # [B,T,H,W]

        # 2) batch-wise scale & shift in disparity domain
        P = T * H * W
        raw_flat = raw_pred_disp.view(B, -1)                       # [B, P]
        gt_flat  = gt_disp.view(B, -1)                             # [B, P]
        m_flat   = mask.view(B, -1).float()                        # [B, P]

        count    = m_flat.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        mean_raw = (raw_flat * m_flat).sum(dim=1, keepdim=True) / count
        mean_gt  = (gt_flat  * m_flat).sum(dim=1, keepdim=True) / count

        d_c = (raw_flat - mean_raw) * m_flat                       # [B, P]
        g_c = (gt_flat  - mean_gt ) * m_flat

        cov = (d_c * g_c).sum(dim=1, keepdim=True)                 # [B,1]
        var = (d_c * d_c).sum(dim=1, keepdim=True).clamp_min(self.eps)

        s = cov / var                                              # [B,1]
        t = mean_gt - s * mean_raw                                 # [B,1]

        aligned_flat      = raw_flat * s + t                       # [B, P]
        aligned_pred_disp = aligned_flat.view(B, T, H, W)          # [B, T, H, W]

        # 3) disparity differences
        d_diff = (aligned_pred_disp[:,1:] - aligned_pred_disp[:,:-1]).abs()  # [B,T-1,H,W]
        g_diff = (gt_disp[:,1:]            - gt_disp[:,:-1]).abs()           # [B,T-1,H,W]

        # 4) static & valid mask
        valid_pair = mask[:,1:] & mask[:,:-1]                     # [B,T-1,H,W]
        # depth_diff  = (gt_depth[:,1:] - gt_depth[:,:-1]).abs()    # [B,T-1,H,W]
        # static      = valid_pair & (depth_diff < self.static_th)   # [B,T-1,H,W]
        ## 저자의 방식을 빌려 disparity의 max-min에 th 곱해서 사용
        min_d = torch.where(mask, gt_disp, torch.inf).amin(dim=(1,2,3))
        max_d = torch.where(mask, gt_disp, -torch.inf).amax(dim=(1,2,3))
        target_th = ((max_d - min_d) * self.diff_depth_th).view(-1, 1, 1, 1)  # [B,1,1,1]
        static = valid_pair & (g_diff < target_th)

        # 5) error map
        err = (d_diff - g_diff).abs()                             # [B,T-1,H,W]

        # 6) flatten to [N, Q]
        N = B * (T-1)
        Q = H * W
        err2    = err.view(N, Q)
        static2 = static.view(N, Q)

        # mask non-static as NaN for quantile
        err2_nan = err2.masked_fill(~static2, float('nan'))

        # 7) threshold per frame
        thresh = torch.nanquantile(err2_nan, 1 - self.trim_ratio, dim=1)  # [N]

        # 8) keep under threshold
        keep = static2 & (err2 <= thresh.unsqueeze(1))           # [N,Q]

        # 9) trimmed MAE per frame
        sum_err      = (err2 * keep).sum(dim=1)                  # [N]
        count_pixels = keep.sum(dim=1).clamp_min(1.0)            # [N]
        loss_frame   = sum_err / count_pixels                    # [N]

        # 10) final mean
        loss_tgm = loss_frame.mean()
        # print("TGM Loss per batch:", loss_tgm.item())
        return loss_tgm
