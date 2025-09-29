import inspect
import os
import argparse
import logging

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import wandb
import math
import warnings
from dotenv import load_dotenv

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image

from utils.loss_MiDas import *
from data.dataLoader import *                 # KITTIVideoDataset, get_data_list
from data.val_dataLoader import *            # ValDataset, get_list

# 기존 offline model을 teacher로, real-time model을 student로 설계
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher

from benchmark.eval.metric import *          # abs_relative_difference, delta1_acc
from benchmark.eval.eval_tae import tae_torch

# ImageNet normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
from video_depth_anything.motion_module.motion_module import TemporalAttention


# ────────────────────────────── 유틸/평가 함수 ──────────────────────────────
def least_square_whole_clip(infs, gts, data_name):
    # [B,T,1,H,W] → [B,T,H,W]
    if infs.dim() == 5 and infs.shape[2] == 1:
        infs = infs.squeeze(2)
    if gts.dim() == 5 and gts.shape[2] == 1:
        gts = gts.squeeze(2)

    if data_name == "kitti":
        valid_mask = (gts > 1e-3) & (gts < 80.0)
    elif data_name == "gta":
        valid_mask = (gts > 1e-3) & (gts < 1000.0)
    elif data_name == "tartanair":
        valid_mask = (gts > 60.0) & (gts < 150.0)
    else:
        valid_mask = (gts > 1e-3) & (gts < 10.0)

    # autocast 비활성화 + float64 정밀도 보장 (lstsq 안정성)
    with autocast(enabled=False):
        gt_disp_masked = 1.0 / (gts[valid_mask].reshape((-1, 1)).double() + 1e-6)
        infs = infs.clamp(min=1e-3)
        pred_disp_masked = infs[valid_mask].reshape((-1, 1)).double()

        A = torch.cat([pred_disp_masked, torch.ones_like(pred_disp_masked)], dim=-1)
        X = torch.linalg.lstsq(A, gt_disp_masked).solution
        scale, shift = X[0].item(), X[1].item()

    aligned_pred = torch.clamp(scale * infs + shift, min=1e-3)
    depth = torch.zeros_like(aligned_pred)
    depth = 1.0 / aligned_pred
    return depth, valid_mask

def eval_tae(pred_depth, gt_depth, poses, Ks, masks):
    error_sum = 0.0
    for i in range(len(pred_depth) - 1):
        depth1, depth2 = pred_depth[i], pred_depth[i + 1]
        mask1, mask2   = masks[i], masks[i + 1]

        T_1, T_2 = poses[i], poses[i + 1]
        try:
            T_2_inv = torch.linalg.inv(T_2)
        except torch._C._LinAlgError:
            T_2_inv = torch.linalg.pinv(T_2)

        T_2_1 = T_2_inv @ T_1
        R_2_1 = T_2_1[:3, :3]
        t_2_1 = T_2_1[:3, 3]

        K = Ks[i]
        if K.dim() == 1 and K.numel() == 9:
            K = K.view(3, 3)

        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)

        try:
            T_1_2 = torch.linalg.inv(T_2_1)
        except torch._C._LinAlgError:
            T_1_2 = torch.linalg.pinv(T_2_1)

        R_1_2 = T_1_2[:3, :3]
        t_1_2 = T_1_2[:3, 3]
        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)

        error_sum += error1 + error2

    return error_sum / (2 * (len(pred_depth) - 1))

def metric_val(infs, gts, data_name, poses=None, Ks=None):
    gt_depth = gts
    pred_depth, valid_mask = least_square_whole_clip(infs, gts, data_name)

    n = valid_mask.sum((-1, -2))
    valid_frame = (n > 0)
    pred_depth = pred_depth[valid_frame]
    gt_depth   = gt_depth[valid_frame]
    valid_mask = valid_mask[valid_frame]

    absrel = abs_relative_difference(pred_depth, gt_depth, valid_mask)
    delta1 = delta1_acc(pred_depth, gt_depth, valid_mask)

    if poses is not None:
        tae = eval_tae(pred_depth, gt_depth, poses, Ks, valid_mask)
        return absrel, delta1, tae
    else:
        return absrel, delta1

def get_mask(depth_m, min_depth, max_depth):
    return ((depth_m > min_depth) & (depth_m < max_depth)).bool()

def norm_ssi(depth, valid_mask):
    eps = 1e-6
    disparity = torch.zeros_like(depth)
    disparity[valid_mask] = 1.0 / depth[valid_mask]

    B, T, C, H, W = disparity.shape
    disp_flat = disparity.view(B, T, -1)
    mask_flat = valid_mask.view(B, T, -1)

    disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).min(dim=-1)[0]
    disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).max(dim=-1)[0]
    disp_min = disp_min.view(B, T, 1, 1, 1)
    disp_max = disp_max.view(B, T, 1, 1, 1)

    norm_disp = (disparity - disp_min) / (disp_max - disp_min + eps)
    return norm_disp.masked_fill(~valid_mask, 0.0)

def save_validation_frames(x, y, masks, aligned_disp, save_dir, epoch, batch_idx):
    os.makedirs(save_dir, exist_ok=True)
    B, T = x.shape[0], x.shape[1]
    assert B >= 1, "save_validation_frames expects batch size >= 1"

    wb_images = []
    for t in range(T):
        # (a) RGB
        rgb_norm = x[0, t]
        # device에 맞게 MEAN, STD 이동
        mean_device = MEAN.to(rgb_norm.device)
        std_device = STD.to(rgb_norm.device)
        rgb_unc  = (rgb_norm * std_device + mean_device).clamp(0, 1)
        rgb_np   = (rgb_unc.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))

        # (b) GT Disparity (min-max 정규화)
        depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)
        disp_frame  = 1.0 / depth_frame
        valid       = masks[0, t].squeeze(0)

        d_vals = disp_frame[valid]
        d_min, d_max = d_vals.min(), d_vals.max()

        norm_gt = ((disp_frame - d_min) / (d_max - d_min + 1e-6)).clamp(0, 1)
        gt_uint8 = (norm_gt.cpu().numpy() * 255).astype(np.uint8)
        gt_rgb   = np.stack([gt_uint8] * 3, axis=-1)
        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"gt_{t:02d}.png"))

        # (c) Mask
        mask_frame = masks[0, t].squeeze(0).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask_frame).save(os.path.join(save_dir, f"mask_{t:02d}.png"))

        # (d) Pred Disparity (같은 min-max)
        pred_frame = aligned_disp[0, t]
        norm_pd = ((pred_frame - d_min) / (d_max - d_min + 1e-6)).clamp(0, 1)
        pd_uint8 = (norm_pd.cpu().numpy() * 255).astype(np.uint8)
        pd_rgb   = np.stack([pd_uint8] * 3, axis=-1)
        Image.fromarray(pd_rgb).save(os.path.join(save_dir, f"pred_{t:02d}.png"))

        wb_images.append(
            wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"),
                        caption=f"pred_epoch{epoch}_frame{t:02d}")
        )

    # logger.info(f"→ saved validation frames to '{save_dir}'")
    return wb_images

def to_BHW_pred(pred):
    # Handle tuple input (extract first element if tuple)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    
    # pred: [B,H,W] or [B,1,H,W] or [B,C,H,W]
    if not torch.is_tensor(pred):
        raise ValueError(f"Expected tensor but got {type(pred)}")
        
    if pred.dim() == 3:
        return pred
    if pred.dim() == 4:
        if pred.size(1) == 1:
            return pred[:, 0]              # [B,H,W]
        else:
            # C>1인 경우(드물지만 발생): 채널 축 평균으로 단일 disparity 생성
            return pred.mean(dim=1)        # [B,H,W]
    if pred.dim() == 5:
        # 5차원의 경우: [B,C,T,H,W] → [B,H,W] (첫 번째 프레임 사용)
        if pred.size(2) == 1:
            return pred[:, 0, 0]           # [B,H,W]
        else:
            # 여러 프레임이 있는 경우 첫 번째 프레임 사용
            return pred[:, 0, 0]           # [B,H,W]
    raise ValueError(f"Unexpected pred shape: {pred.shape}")

# ────────────────────────────── Streaming helpers ──────────────────────────────
def _detach_cache(cache):
    if cache is None:
        return None
    if isinstance(cache, (list, tuple)):
        return type(cache)(_detach_cache(c) for c in cache)
    if isinstance(cache, dict):
        return {k: _detach_cache(v) for k, v in cache.items()}
    if torch.is_tensor(cache):
        return cache.detach()
    return cache  # unknown type as-is

def model_stream_step(model, x_t, cache=None, bidirectional_update_length=16, current_frame=0):
    """
    x_t: [B,1,3,H,W] (single-frame step)
    bidirectional_update_length: number of recent frames to update bidirectionally
    current_frame: current frame index for bidirectional update
    return: pred_t [B, H, W], new_cache
    """
    # Teacher-Student 모델인지 확인
    if hasattr(model, 'student'):
        # Teacher-Student 모델: Student만 사용 (RW-Memory 구조)
        actual_model = model.student.module if hasattr(model.student, 'module') else model.student
        
        # Student의 forward_depth 사용 (RW-Memory 기반, 3개 반환값)
        features = actual_model.forward_features(x_t)
        
        # forward_depth 메서드의 signature 확인
        forward_depth_sig = inspect.signature(actual_model.forward_depth)
        forward_depth_params = list(forward_depth_sig.parameters.keys())
        
        # 지원하는 파라미터에 따라 호출 방식 결정
        if 'bidirectional_update_length' in forward_depth_params and 'current_frame' in forward_depth_params:
            # 최신 bidirectional update 지원
            out = actual_model.forward_depth(
                features, x_t.shape, 
                cached_hidden_state_list=cache,
                bidirectional_update_length=bidirectional_update_length,
                current_frame=current_frame
            )
        else:
            # 기본 방식만 지원
            out = actual_model.forward_depth(features, x_t.shape, cache)
        
        # Student (RW-Memory) 반환값 처리: depth, out_hidden_list, upd_hidden_list
        if isinstance(out, (list, tuple)):
            if len(out) == 3:
                pred_t, _out_cache, new_cache = out     # ← 업데이트된 캐시만 사용
            elif len(out) == 2:
                pred_t, new_cache = out
            else:
                raise RuntimeError(f"Unexpected Student forward_depth return len={len(out)}")
        else:
            pred_t, new_cache = out, None
    else:
        # 기존 VideoDepthAnything 모델 (Teacher - offline 처리)
        actual_model = model.module if hasattr(model, 'module') else model
        
        # Teacher 모델은 forward_depth 메서드가 없으므로 기본 forward 사용
        if hasattr(actual_model, 'forward_depth'):
            # Student와 유사하게 처리 (만약 forward_depth가 있다면)
            features = actual_model.forward_features(x_t)
            forward_depth_result = actual_model.forward_depth(features, x_t.shape, cache)
            
            # Teacher의 forward_depth 결과도 tuple일 수 있음
            if isinstance(forward_depth_result, (tuple, list)):
                pred_t = forward_depth_result[0]
            else:
                pred_t = forward_depth_result
            new_cache = None  # Teacher는 캐시를 사용하지 않음
        else:
            # 기존 Teacher 모델: 전체 forward 사용 (캐시 없음)
            forward_result = actual_model.forward(x_t)  # [B, T, H, W] or tuple
            if isinstance(forward_result, (tuple, list)):
                pred_t = forward_result[0]
            else:
                pred_t = forward_result
            new_cache = None
    
    # 출력 형태 정규화
    if isinstance(pred_t, torch.Tensor):
        if pred_t.dim() == 4 and pred_t.size(1) == 1:
            pred_t = pred_t[:, 0]  # [B,H,W]
    
    return pred_t, new_cache

def streaming_validate(
    model, loader, device, data_name,
    loss_ssi, loss_tgm, ratio_ssi, ratio_tgm,
    save_vis: bool = False, tag: str = None, epoch: int = None,
    bidirectional_update_length: int = 16,
    base_output_dir: str = "outputs",
    # 🔽 추가 옵션
    max_scenes: int = None,              # 검증할 시퀀스(배치) 최대 개수
    max_frames_per_scene: int = None,    # 각 시퀀스에서 사용할 최대 프레임 수
    strict_offline_eval_align: bool = False,  # eval.py 스타일의 전체-클립 LS 정렬로 metric 계산
    crop: tuple = None,                  # 예: (8,-8,11,-11)
    min_depth_eval: float = 0.1,
    max_depth_eval: float = 10.0,
):
    """
    - 스트리밍 방식 1-frame step 검증
    - max_scenes / max_frames_per_scene 로 빠른 부분 검증 가능
    - strict_offline_eval_align=True 이면 eval.py와 동일한 LS 정렬로 metric 산출
    """
    import os
    from torch.cuda.amp import autocast
    from tqdm import tqdm

    model.eval()
    total_absrel = 0.0
    total_delta1 = 0.0
    total_tae    = 0.0
    total_loss   = 0.0
    cnt_clip     = 0
    wb_images    = []

    # (시각화 기본값) — 실제 metric 경로에서는 dataset 범위로 다시 클램프
    MIN_DISP = 1.0 / 80.0
    MAX_DISP = 1.0 / 0.001

    processed_scenes = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(loader)):
            if (max_scenes is not None) and (processed_scenes >= max_scenes):
                break

            # 배치 구성 파싱
            if not isinstance(batch, (list, tuple)):
                raise ValueError("Validation loader must return a tuple/list.")
            if len(batch) == 2:
                x, y = batch
                extrinsics = None
                intrinsics = None
            elif len(batch) == 4:
                x, y, extrinsics, intrinsics = batch
            else:
                raise ValueError(f"Unexpected batch tuple length: {len(batch)}")

            x, y = x.to(device), y.to(device)
            if extrinsics is not None: extrinsics = extrinsics.to(device)
            if intrinsics is not None: intrinsics = intrinsics.to(device)

            B, T = x.shape[:2]
            if max_frames_per_scene is not None:
                T = min(T, int(max_frames_per_scene))

            preds = []
            mask_list = []
            cache = None
            prev_pred = prev_mask = prev_y = None

            # 스트리밍 스텝
            val_frame_count = 0
            cache = _detach_cache(cache)

            for t in range(T):
                x_t = x[:, t:t+1]  # [B,1,3,H,W]
                pred_t, cache = model_stream_step(
                    model, x_t, cache,
                    bidirectional_update_length=bidirectional_update_length,
                    current_frame=val_frame_count
                )
                pred_t = to_BHW_pred(pred_t)  # [B,H,W]

                # 배치수 정렬
                if pred_t.shape[0] != y.shape[0]:
                    if pred_t.shape[0] > y.shape[0]:
                        pred_t = pred_t[:y.shape[0]]
                    else:
                        pred_t = pred_t.repeat(y.shape[0] // pred_t.shape[0], 1, 1)

                preds.append(pred_t)
                val_frame_count += 1

                # ✔ 깊이 범위를 인자로 사용(Scannet: 0.1~10.0)
                mask_t = get_mask(y[:, t:t+1], min_depth=min_depth_eval, max_depth=max_depth_eval).to(device)  # [B,1,1,H,W]
                mask_list.append(mask_t.squeeze(2))  # [B,1,H,W]

                disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)
                if pred_t.shape[0] != disp_normed_t.shape[0]:
                    pred_t = pred_t[:disp_normed_t.shape[0]]

                ssi_loss_t = loss_ssi(pred_t.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))
                total_loss += ratio_ssi * ssi_loss_t

                if t > 0:
                    pred_pair = torch.stack([prev_pred, pred_t], dim=1)  # [B,2,H,W]
                    y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)  # [B,2,1,H,W]
                    m_pair    = torch.cat([prev_mask, mask_t], dim=1)    # [B,2,1,H,W]
                    tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                    total_loss += ratio_tgm * tgm_loss

                prev_pred = pred_t
                prev_mask = mask_t
                prev_y    = y[:, t:t+1]

            pred_seq  = torch.stack(preds, dim=1)     # [B,T,H,W]
            masks_seq = torch.stack(mask_list, dim=1) # [B,T,1,H,W]

            # ===== Metric =====
            for b_idx in range(B):
                inf_clip = pred_seq[b_idx]          # [T,H,W]  (※ 모델 출력은 disparity로 해석)
                gt_clip  = y[b_idx, :T].squeeze(1)  # [T,H,W]
                m_clip   = masks_seq[b_idx, :T].squeeze(1).float()  # [T,H,W]

                # ✔ metric 직전 크롭을 clip 전체에 적용(+마스크도 동일 크롭)
                if crop is not None:
                    a, b_, c, d = crop
                    inf_clip = inf_clip[:, a:b_, c:d]
                    gt_clip  = gt_clip[:,  a:b_, c:d]
                    m_clip   = m_clip[:,   a:b_, c:d]

                if strict_offline_eval_align:
                    # === eval.py와 동일: Disparity 전체-클립 LS 정렬 → 범위 클램프 → 깊이 변환 → metric ===
                    raw_disp = inf_clip.clamp(min=1e-6)        # [T,H,W]
                    gt_disp  = (1.0 / gt_clip.clamp(min=1e-6)) # [T,H,W]
                    m        = m_clip                          # [T,H,W]

                    with autocast(enabled=False):
                        p_flat = raw_disp.float().reshape(-1)
                        g_flat = gt_disp.float().reshape(-1)
                        m_flat = m.reshape(-1)

                        A = torch.stack([p_flat, torch.ones_like(p_flat)], dim=-1)  # [P,2]
                        A = A * m_flat.unsqueeze(-1)
                        b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)
                        X = torch.linalg.lstsq(A, b_vec).solution
                        a_fit = X[0, 0]
                        b_fit = X[1, 0]

                    # 데이터셋 범위로 disparity 클램프
                    aligned_disp = (raw_disp * a_fit + b_fit).clamp(
                        min=1.0 / max_depth_eval,
                        max=1.0 / min_depth_eval
                    )
                    # disparity → depth
                    pred_depth = torch.clamp(1.0 / aligned_disp, min=min_depth_eval, max=max_depth_eval)

                    # metric (absrel / delta1) — m_clip을 valid mask로 사용
                    absr = abs_relative_difference(pred_depth, gt_clip, m_clip.bool())
                    d1   = delta1_acc(pred_depth, gt_clip, m_clip.bool())
                else:
                    # 기존 metric_val 경로(프로젝트 정의 함수). 내부에서 LS 정렬 수행.
                    if (extrinsics is not None) and (intrinsics is not None):
                        pose = extrinsics[b_idx]
                        Kmat = intrinsics[b_idx]
                        absr_dae = metric_val(inf_clip, gt_clip, data_name, pose, Kmat)
                        absr, d1, tae = absr_dae
                        total_tae += tae
                    else:
                        absr, d1 = metric_val(inf_clip, gt_clip, data_name)

                total_absrel += absr
                total_delta1 += d1
                cnt_clip     += 1

            # 시각화(첫 배치만): 필요하면 dataset 범위로 클램프하도록 변경 가능
            if save_vis and batch_idx == 0 and tag is not None and epoch is not None:
                Bv, Tv, H, W = pred_seq.shape
                raw_disp = pred_seq.clamp(min=1e-6)
                gt_disp  = (1.0 / y[:, :T].clamp(min=1e-6)).squeeze(2)
                m_flat   = masks_seq[:, :T].squeeze(2).view(Bv, -1).float()
                with autocast(enabled=False):
                    p_flat = raw_disp.float().view(Bv, -1)
                    g_flat = gt_disp.float().view(Bv, -1)
                    A = torch.stack([p_flat, torch.ones_like(p_flat)], dim=-1) * m_flat.unsqueeze(-1)
                    b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)
                    X = torch.linalg.lstsq(A, b_vec).solution
                    a_v = X[:, 0, 0].view(Bv, 1, 1, 1)
                    b_v = X[:, 1, 0].view(Bv, 1, 1, 1)
                aligned_disp = (raw_disp * a_v + b_v).clamp(min=MIN_DISP, max=MAX_DISP)
                save_dir = os.path.join(base_output_dir, f"{tag}/epoch_{epoch}_batch_{batch_idx}")
                wb_images.extend(save_validation_frames(x[:, :T], y[:, :T], masks_seq[:, :T], aligned_disp, save_dir, epoch, batch_idx))

            processed_scenes += 1

    avg_absrel = total_absrel / max(1, cnt_clip)
    avg_delta1 = total_delta1 / max(1, cnt_clip)
    avg_tae    = total_tae    / max(1, cnt_clip)
    avg_loss   = total_loss   / max(1, cnt_clip)
    return avg_loss, avg_absrel, avg_delta1, avg_tae, wb_images


def batch_ls_scale_shift(pred_disp, gt_disp, mask):
    """
    pred_disp: [B, H, W] or [B,1,H,W] disparity (>= 1e-6)
    gt_disp  : [B, H, W] or [B,1,H,W] disparity
    mask     : [B,1,H,W] bool

    return a_star, b_star with shape [B,1,1,1]
    """
    if pred_disp.dim() == 4 and pred_disp.size(1) == 1:
        p = pred_disp[:, 0]
    else:
        p = pred_disp
    if gt_disp.dim() == 4 and gt_disp.size(1) == 1:
        g = gt_disp[:, 0]
    else:
        g = gt_disp

    B, H, W = p.shape
    
    # autocast 비활성화 + float32 캐스팅 (lstsq 안정성 보장)
    with autocast(enabled=False):
        m = mask.view(B, -1).float()                      # [B, P]
        p_flat = p.float().view(B, -1)                    # [B, P]
        g_flat = g.float().view(B, -1)                    # [B, P]

        A = torch.stack([p_flat, torch.ones_like(p_flat, device=p.device)], dim=-1)  # [B,P,2]
        A = A * m.unsqueeze(-1)
        b_vec = g_flat.unsqueeze(-1) * m.unsqueeze(-1)

        X = torch.linalg.lstsq(A, b_vec).solution        # [B,2,1]
        a_star = X[:, 0, 0].view(B, 1, 1, 1)
        b_star = X[:, 1, 0].view(B, 1, 1, 1)

    # 안정성: a는 양수로, 극단치 클리핑
    a_star = a_star.clamp(min=1e-4, max=1e4)
    b_star = b_star.clamp(min=-1e4, max=1e4)
    return a_star, b_star

# ────────────────────────────── NEW: 학습용 Teacher/Student 보조 루틴 ──────────────────────────────
def _gather_extras(ret):
    """
    모델이 (depth, cache, extras) 혹은 (depth, cache) 혹은 depth 만을 반환할 수 있으므로
    extras(dict)만 안전하게 뽑아낸다.
    """
    if isinstance(ret, (list, tuple)):
        if len(ret) >= 3 and isinstance(ret[2], dict):
            return ret[2]
    return {}  # 없는 경우 빈 dict

def _norm_l2(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def teacher_window_forward(
    teacher_model,
    x_win,                       # [B, T_w, 3, H, W]
    rope_dt=None,
    return_attn=False,
    return_qkv=False,
):
    """
    Teacher를 '오프라인 윈도우'로 실행하고, (가능하면) 부가정보(attn/qkv)도 받아온다.
    dpt_temporal/motion_module가 공용이므로, head에 kwargs로 신호를 넘겨 extras를 꺼낼 수 있음.
    """
    tm = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
    feats = tm.forward_features(x_win)           # [B*T_w, ...]
    B, T = x_win.shape[:2]
    C, H, W = x_win.shape[2], x_win.shape[3], x_win.shape[4]
    patch_h, patch_w = H // 14, W // 14

    kwargs = dict(stream_mode=False)
    if rope_dt is not None:
        kwargs["rope_dt"] = float(rope_dt)
    if return_attn:
        kwargs["return_attn"] = True
    if return_qkv:
        kwargs["return_qkv"] = True

    try:
        # head 직접 호출 (티처도 동일 head를 공유)
        out = tm.head(
            feats, patch_h, patch_w, T,
            cached_hidden_state_list=None,
            **kwargs
        )
        if isinstance(out, (list, tuple)):
            depth_logits = out[0]
            extras = out[2] if (len(out) >= 3 and isinstance(out[2], dict)) else {}
        else:
            depth_logits = out
            extras = {}
    except TypeError:
        # kwargs를 받지 못하면 extras 없이 깊이만
        depth_logits, extras = tm.head(feats, patch_h, patch_w, T), {}

    depth_logits = F.interpolate(depth_logits, size=(H, W), mode="bilinear", align_corners=True)
    depth_logits = F.relu(depth_logits)                          # [B*T,1,H,W]
    depth_bt = depth_logits.squeeze(1).unflatten(0, (B, T))      # [B,T,H,W]
    return depth_bt, extras

def student_stream_step_with_extras(
    student_model,
    x_t,            # [B,1,3,H,W]
    cache=None,
):
    """
    Student 스트리밍 1 step. extras(dict)가 오면 함께 반환.
    """
    sm = student_model.module if hasattr(student_model, "module") else student_model
    feats = sm.forward_features(x_t)
    out = sm.forward_depth(feats, x_t.shape, cached_hidden_state_list=cache)
    extras = _gather_extras(out)
    if isinstance(out, (list, tuple)):
        depth_bt = out[0]         # [B,1,H,W] → [B,H,W]
        new_cache = out[1]
    else:
        depth_bt = out
        new_cache = None

    if depth_bt.dim() == 4 and depth_bt.size(1) == 1:
        depth_bt = depth_bt[:, 0]
    return depth_bt, new_cache, extras

def compute_distill_losses(student_extras, teacher_extras, weights):
    """
    weights: dict with keys in {"kv", "attn", "ctx", "feat"} (스칼라 λ)
    student_extras / teacher_extras: dict (없으면 빈 dict)
    구현은 '있으면 계산, 없으면 0' 방식.
    """
    lam_kv   = float(weights.get("kv",   0.0))
    lam_attn = float(weights.get("attn", 0.0))
    lam_ctx  = float(weights.get("ctx",  0.0))
    lam_feat = float(weights.get("feat", 0.0))

    total = torch.zeros((), device=next(iter(student_extras.values())).device) if student_extras else torch.tensor(0.0)

    # (1) KV 정합 (정규화 L2)
    if lam_kv > 0 and ("k_list" in student_extras) and ("v_list" in student_extras) \
       and ("k_list" in teacher_extras) and ("v_list" in teacher_extras):
        s_ks, s_vs = student_extras["k_list"], student_extras["v_list"]
        t_ks, t_vs = teacher_extras["k_list"], teacher_extras["v_list"]
        kv_loss = 0.0
        n_terms = 0
        for sk, tk in zip(s_ks, t_ks):
            if sk.shape[-1] != tk.shape[-1]:  # 차원 불일치 시 skip
                continue
            skn = _norm_l2(sk, dim=-1); tkn = _norm_l2(tk, dim=-1)
            kv_loss = kv_loss + F.mse_loss(skn, tkn)
            n_terms += 1
        for sv, tv in zip(s_vs, t_vs):
            if sv.shape[-1] != tv.shape[-1]:
                continue
            svn = _norm_l2(sv, dim=-1); tvn = _norm_l2(tv, dim=-1)
            kv_loss = kv_loss + F.mse_loss(svn, tvn)
            n_terms += 1
        if n_terms > 0:
            total = total + lam_kv * (kv_loss / n_terms)

    # (2) 어텐션 맵 정합
    if lam_attn > 0 and ("attn_list" in student_extras) and ("attn_list" in teacher_extras):
        s_as, t_as = student_extras["attn_list"], teacher_extras["attn_list"]
        attn_loss = 0.0
        n_terms = 0
        for sa, ta in zip(s_as, t_as):
            if sa.shape != ta.shape:
                continue
            attn_loss = attn_loss + F.mse_loss(sa, ta)
            n_terms += 1
        if n_terms > 0:
            total = total + lam_attn * (attn_loss / n_terms)

    # (3) 어텐디드 컨텍스트 정합 (선택)
    if lam_ctx > 0 and ("ctx_list" in student_extras) and ("ctx_list" in teacher_extras):
        s_cs, t_cs = student_extras["ctx_list"], teacher_extras["ctx_list"]
        ctx_loss = 0.0; n_terms = 0
        for sc, tc in zip(s_cs, t_cs):
            if sc.shape != tc.shape:
                continue
            ctx_loss = ctx_loss + F.mse_loss(sc, tc)
            n_terms += 1
        if n_terms > 0:
            total = total + lam_ctx * (ctx_loss / n_terms)

    # (4) feature-level distill (레벨3/4, path_3/4 등 — extras에 있으면 사용)
    if lam_feat > 0 and ("feat_list" in student_extras) and ("feat_list" in teacher_extras):
        s_fs, t_fs = student_extras["feat_list"], teacher_extras["feat_list"]
        feat_loss = 0.0; n_terms = 0
        for sf, tf in zip(s_fs, t_fs):
            if sf.shape != tf.shape:
                continue
            feat_loss = feat_loss + F.mse_loss(sf, tf)
            n_terms += 1
        if n_terms > 0:
            total = total + lam_feat * (feat_loss / n_terms)

    return total

# ────────────────────────────── NEW: 한 스텝 학습 루틴(Streaming) ──────────────────────────────
def train_one_step_streaming(
    batch,                       # (x,y) or (x,y,extrinsics,intrinsics)
    student_model, teacher_model,
    optimizer, scaler,
    device,
    # 손실 구성
    loss_ssi, loss_tgm,
    ratio_ssi=1.0, ratio_tgm=0.1,
    # Distill 구성 (필요 없으면 모두 0으로)
    distill_weights=None,        # {"kv":λ1, "attn":λ2, "ctx":λ3, "feat":λ4}
    teacher_win=16,              # [t-W+1..t] 윈도우
    # 공통 옵션
    min_depth_train=0.1, max_depth_train=10.0,
    crop=None,                   # 예: (8,-8,11,-11)
    rope_dt_teacher=None,        # RoPE tempo scale for teacher (선택)
):
    """
    - Student는 스트리밍(1프레임씩)으로 순차 처리 + 캐시 유지
    - Teacher는 윈도우 오프라인 처리([t-W+1..t])로 표적(옵션: attn/qkv) 생성
    - 기본 손실: SSI + TGM
    - distill_weights를 주면 KV/ATTN/CTX/FEAT 정합 추가
    """
    if distill_weights is None:
        distill_weights = {"kv": 0.0, "attn": 0.0, "ctx": 0.0, "feat": 0.0}

    # 배치 파싱
    if len(batch) == 2:
        x, y = batch
        extrinsics = intrinsics = None
    elif len(batch) == 4:
        x, y, extrinsics, intrinsics = batch
    else:
        raise ValueError(f"Unexpected batch structure: len={len(batch)}")

    x = x.to(device); y = y.to(device)
    B, T = x.shape[:2]

    optimizer.zero_grad(set_to_none=True)

    total_loss = torch.zeros((), device=device)
    cache = None
    prev_pred = prev_mask = prev_y = None

    # mixed precision
    with autocast(enabled=True):
        for t in range(T):
            # Student 1-step
            x_t = x[:, t:t+1]
            pred_t, cache, s_extra = student_stream_step_with_extras(student_model, x_t, cache)  # [B,H,W], cache, dict or {}

            # Crop(optional)
            if crop is not None:
                a,b_,c,d = crop
                pred_t = pred_t[:, a:b_, c:d]
                y_t    = y[:, t, 0, a:b_, c:d].unsqueeze(1)  # [B,1,H,W]
            else:
                y_t    = y[:, t:t+1]  # [B,1,H,W]

            # Masks
            mask_t = get_mask(y_t, min_depth_train, max_depth_train).to(device)  # [B,1,1,H,W] 형태일 수 있음
            if mask_t.dim() == 5:  # [B,1,1,H,W] → [B,1,H,W]
                mask_t = mask_t.squeeze(2)

            # SSI
            disp_normed_t = norm_ssi(y_t, mask_t.unsqueeze(2)).squeeze(2)  # [B,1,H,W] → [B,1,H,W]
            ssi_loss_t = loss_ssi(pred_t.unsqueeze(1), disp_normed_t, mask_t)
            total_loss = total_loss + ratio_ssi * ssi_loss_t

            # TGM (인접 프레임 쌍)
            if t > 0 and prev_pred is not None:
                pred_pair = torch.stack([prev_pred, pred_t], dim=1)  # [B,2,H,W]
                y_pair    = torch.cat([prev_y, y_t], dim=1)          # [B,2,1,H,W]
                m_pair    = torch.cat([prev_mask, mask_t], dim=1)    # [B,2,1,H,W]
                tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                total_loss = total_loss + ratio_tgm * tgm_loss

            # Distill(옵션): Teacher 윈도우 [t-W+1..t]
            if any(v > 0 for v in distill_weights.values()):
                t0 = max(0, t - int(teacher_win) + 1)
                x_win = x[:, t0:t+1]  # [B,T_w,3,H,W]
                # teacher 실행
                t_depth_bt, t_extra = teacher_window_forward(
                    teacher_model, x_win,
                    rope_dt=rope_dt_teacher,
                    return_attn=(distill_weights.get("attn", 0.0) > 0.0),
                    return_qkv=(distill_weights.get("kv",   0.0) > 0.0) or (distill_weights.get("ctx", 0.0) > 0.0)
                )
                # Student/Teacher extras로 distill 계산(없으면 0)
                distill_loss = compute_distill_losses(s_extra, t_extra, distill_weights)
                total_loss = total_loss + distill_loss

            prev_pred = pred_t
            prev_mask = mask_t
            prev_y    = y_t

    # backward (AMP)
    scaler.scale(total_loss).rstep() if hasattr(scaler, "rstep") else None
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # 로깅값
    logs = {
        "loss_total": float(total_loss.detach().cpu().item()),
        "loss_ssi": float(ratio_ssi),     # 개별 항 분해를 원하면 위에서 누적값 따로 기록하면 됨
        "loss_tgm": float(ratio_tgm),
        "distill_kv": float(distill_weights.get("kv", 0.0)),
        "distill_attn": float(distill_weights.get("attn", 0.0)),
        "distill_ctx": float(distill_weights.get("ctx", 0.0)),
        "distill_feat": float(distill_weights.get("feat", 0.0)),
    }
    return total_loss.detach(), logs

# ──────────────────────────────────────────────────────────────────────────────
# Mini validation using the SAME pipeline as benchmark/infer/infer_stream.py
# + benchmark/eval/eval.py — but restricted to a few scenes (e.g., 2)
# ──────────────────────────────────────────────────────────────────────────────
import gc
import json
import cv2
import os
from tqdm import tqdm

@torch.no_grad()
def _reset_streaming_state(model):
    """infer_stream.py와 동일한 스트리밍 상태 리셋."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "transform"):
        m.transform = None
    if hasattr(m, "frame_cache_list"):
        m.frame_cache_list = []
    if hasattr(m, "frame_id_list"):
        m.frame_id_list = []
    if hasattr(m, "id"):
        m.id = -1

def _vdainfer_one(model, rgb_img, input_size=518, device='cuda', fp32=True):
    """VideoDepthAnything.infer_video_depth_one 그대로 사용."""
    m = model.module if hasattr(model, "module") else model
    return m.infer_video_depth_one(rgb_img, input_size=input_size, device=device, fp32=fp32)

def _get_infer_npy(path, target_hw=None):
    """eval.py:get_infer 동작 복제 (npy만 사용)."""
    arr = np.load(path).astype(np.float32)
    if target_hw is not None and (arr.shape[0] != target_hw[0] or arr.shape[1] != target_hw[1]):
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]))
    return arr

def _get_gt_depth(path, factor):
    """eval.py:get_gt 동작 복제."""
    if path.endswith('.npy'):
        depth = np.load(path).astype(np.float32)
    else:
        depth = cv2.imread(path, -1)
        depth = np.array(depth).astype(np.float32)
    depth = depth / float(factor)
    depth[depth == 0] = -1.0
    return depth

def _depth2disp_np(depth):
    disp = np.zeros_like(depth, dtype=np.float32)
    m = depth > 0
    disp[m] = 1.0 / depth[m]
    return disp

def _ls_align_disparity(infs, gts, valid_mask):
    """
    disparity 선형 정렬: (scale, shift)로 infs를 gts에 맞추되
    eval.py의 방식 그대로 numpy lstsq 사용.
    """
    gt_disp_masked = 1.0 / (gts[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
    infs = np.clip(infs, a_min=1e-3, a_max=None)
    pred_disp_masked = infs[valid_mask].reshape((-1, 1)).astype(np.float64)

    A = np.concatenate([pred_disp_masked, np.ones_like(pred_disp_masked)], axis=-1)  # [P,2]
    X = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]  # [2,1]
    scale, shift = X[0, 0], X[1, 0]
    aligned = np.clip(scale * infs + shift, a_min=1e-3, a_max=None)
    return aligned

def _dataset_eval_defaults(dataset_tag):
    """
    eval.py와 동일한 기본값 반환.
    필요한 경우 확장 가능. (여기선 scannet / scannet_500만 커버)
    """
    if dataset_tag in ("scannet", "scannet_500"):
        return {
            "max_depth_eval": 10.0,
            "min_depth_eval": 0.1,
            "max_eval_len":   500 if dataset_tag == "scannet_500" else 90,
            "crop": (8, -8, 11, -11),  # (a, b, c, d)
        }
    # fallback
    return {
        "max_depth_eval": 10.0,
        "min_depth_eval": 0.1,
        "max_eval_len":   90,
        "crop": (0, -1, 0, -1),
    }

@torch.no_grad()
def validate_with_infer_eval_subset(
    model,
    json_file,                 # e.g., ".../scannet/scannet_video_500.json"
    infer_path,                # e.g., "benchmark/output/scannet_stream_valmini"
    dataset="scannet",         # JSON 내부 키 (보통 'scannet')
    dataset_eval_tag="scannet_500",  # eval 설정 preset (run.sh에서는 scannet_500)
    device="cuda",
    input_size=518,
    scene_indices=[1, 39, 44, 93],      # 계산할 씬 index
    fp32=True,
):
    """
    학습 중간 밸리데이션을, '실제 파이프라인(infer_stream.py + eval.py)'과
    거의 동일하게 수행하되, scene 수만 줄여서 빠르게 실행.

    반환: dict { 'abs_relative_difference': ..., 'rmse_linear': ..., 'delta1_acc': ... }
    """
    os.makedirs(infer_path, exist_ok=True)
    model_was_training = model.training
    model.eval()

    # 1) JSON 로드 & 루트 경로
    with open(json_file, 'r') as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(json_file)

    # 2) Inference (subset)
    target_indices = set()  # 사용할 인덱스
    seq_registry = []  # 평가 시 동일 순서/동일 subset을 재사용하기 위해 기록
    all_scenes = path_json[dataset]
    

    # 사용할 인덱스 결정
    if scene_indices is not None:
        target_indices = set(scene_indices)   # lookup 빠르게 하도록 set
    else:
        target_indices = set(range(len(all_scenes)))

    # tqdm에 전달할 실제 반복 목록을 미리 구성
    items = list(enumerate(path_json[dataset]))  # [(idx, entry), ...]
    if scene_indices is not None:
        items = [(i, d) for i, d in items if i in target_indices]

    for i, entry in tqdm(items, total=len(items), desc=f"[VAL] Streaming {dataset} (subset)"):
        # entry는 보통 {scene_key: [frames...]} 형태의 dict
        # 혹은 이미 (scene_key, frames) 튜플일 수 있음 — 양쪽 모두 처리
        if isinstance(entry, dict):
            if len(entry) == 0:
                continue
            scene_key = next(iter(entry.keys()))
            frames = entry[scene_key]
        elif isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], str):
            scene_key, frames = entry[0], entry[1]
        else:
            # 알 수 없는 구조: 스킵
            continue

        # 스트리밍 상태 리셋
        _reset_streaming_state(model)

        # 각 프레임 저장 경로대로 추론 수행
        for item in frames:
            img_path = os.path.join(root_path, item['image'])
            base, _ = os.path.splitext(item['image'])
            out_path = os.path.join(infer_path, dataset, base + '.npy')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # BGR -> RGB
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            depth_np = _vdainfer_one(model, rgb, input_size=input_size, device=device, fp32=fp32)
            np.save(out_path, depth_np)

        seq_registry.append((scene_key, frames))

    torch.cuda.empty_cache(); gc.collect()

    # 3) Eval (subset) — eval.py 로직을 그대로 옮김
    defaults = _dataset_eval_defaults(dataset_eval_tag)
    max_depth_eval = defaults["max_depth_eval"]
    min_depth_eval = defaults["min_depth_eval"]
    max_eval_len   = defaults["max_eval_len"]
    a, b, c, d     = defaults["crop"]

    # metric 집계
    all_metrics = []
    for (key, frames) in seq_registry:
        infer_paths = []
        depth_gt_paths = []
        factors = []
        for images in frames:
            infer_path_i = (os.path.join(infer_path, dataset, images['image'])
                            .replace('.jpg', '.npy').replace('.png', '.npy'))
            infer_paths.append(infer_path_i)
            depth_gt_paths.append(os.path.join(root_path, images['gt_depth']))
            factors.append(images['factor'])

        infer_paths = infer_paths[:max_eval_len]
        depth_gt_paths = depth_gt_paths[:max_eval_len]
        factors = factors[:max_eval_len]

        # 프레임별 gt/inf 로딩 + 동일 크롭
        gts = []
        infs = []
        for p_inf, p_gt, fac in zip(infer_paths, depth_gt_paths, factors):
            if not os.path.exists(p_inf):
                continue
            gt_depth = _get_gt_depth(p_gt, fac)
            gt_depth = gt_depth[a:b, c:d]

            inf = _get_infer_npy(p_inf, target_hw=gt_depth.shape)
            gts.append(gt_depth)
            infs.append(inf)

        if len(infs) == 0:
            continue

        gts = np.stack(gts, axis=0)        # [T,H,W]
        infs = np.stack(infs, axis=0)      # [T,H,W]

        # valid mask 및 LS 정렬
        valid_mask = np.logical_and((gts > 1e-3), (gts < max_depth_eval))
        aligned_disp = _ls_align_disparity(infs, gts, valid_mask)  # disparity aligned
        pred_depth = _depth2disp_np(aligned_disp)
        pred_depth = np.clip(pred_depth, a_min=1e-3, a_max=max_depth_eval)

        # torch tensor로 metric 계산 (eval.py와 동일한 구현 사용)
        pred_ts = torch.from_numpy(pred_depth).to(device)
        gt_ts   = torch.from_numpy(gts).to(device)
        mask_ts = torch.from_numpy(valid_mask).to(device)

        # 유효 프레임 필터
        n_valid = mask_ts.sum((-1, -2))
        valid_frame = (n_valid > 0)
        pred_ts = pred_ts[valid_frame]
        gt_ts   = gt_ts[valid_frame]
        mask_ts = mask_ts[valid_frame]

        # 필요한 metric 계산
        seq_metrics = {}
        seq_metrics["abs_relative_difference"] = abs_relative_difference(pred_ts, gt_ts, mask_ts).item()
        if "rmse_linear" in globals():
            seq_metrics["rmse_linear"] = rmse_linear(pred_ts, gt_ts, mask_ts).item()
        else:
            # fallback: RMSE 직접 계산
            diff = (pred_ts - gt_ts) * mask_ts
            denom = mask_ts.sum().clamp(min=1).float()
            seq_metrics["rmse_linear"] = torch.sqrt((diff ** 2).sum() / denom).item()
        seq_metrics["delta1_acc"] = delta1_acc(pred_ts, gt_ts, mask_ts).item()

        # 씬 식별자도 같이 저장
        seq_metrics["scene"] = key
        all_metrics.append(seq_metrics)

    # # 평균(숫자형 키만) 내기
    if len(all_metrics) == 0:
        avg = {"abs_relative_difference": float("nan"),
               "rmse_linear": float("nan"),
               "delta1_acc": float("nan")}
    else:
        # 숫자형 값만 평균 계산 (예: "scene" 같은 문자열 키는 무시)
        numeric_keys = []
        for k in all_metrics[0].keys():
            try:
                _ = [float(m[k]) for m in all_metrics]
                numeric_keys.append(k)
            except Exception:
                pass
        avg = {k: float(np.mean([float(m[k]) for m in all_metrics])) for k in numeric_keys}

    # 모델 모드 복구
    if model_was_training:
        model.train()

    return {"avg":avg, "per_scene": all_metrics}
