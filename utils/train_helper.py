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

from benchmark.eval.metric import *          # abs_relative_difference, delta1_acc
from benchmark.eval.eval_tae import tae_torch

# ImageNet normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ────────────────────────────── 유틸/평가 함수 ──────────────────────────────
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

def model_stream_step(
    model,
    x_t,
    cache=None,
    *,
    collect_inter: bool = False,
    collect_qkv: bool = False,
    feature_pool: str = "mean",
    return_encoder_feats: bool = False,  # ← 새 플래그
):
    """
    Streaming 1-step forward (학생용).
    - collect_inter=True면 레이어별 temporal feature/QKV(현재 프레임 것만)를 inter_t로 반환.
    - return_encoder_feats=True면 encoder 출력(feats)도 함께 반환.
    - 반환 형식:
        pred_t:   [B,H,W]
        new_cache: any
        inter_t:  dict[layer_id] -> {"feat_one":[B,1,C], "qkv":Optional dict(Q/K/V:[B,A,1,Dh])}
        feats:    list of encoder features (return_encoder_feats=True일 때만)
    """
    m = model.module if hasattr(model, "module") else model

    # 1) 스트리밍 경로
    feats = m.forward_features(x_t)
    out = m.forward_depth(
        feats,
        x_t.shape,
        cached_hidden_state_list=cache,
        return_intermediates=collect_inter,
        return_qkv=collect_qkv,
        feature_pool=feature_pool,
    )

    # 2) 반환 파싱 (pred, new_cache, extra)
    if isinstance(out, (list, tuple)):
        if len(out) == 3:
            pred_t, new_cache, extra = out
        elif len(out) == 2:
            pred_t, new_cache = out
            extra = None
        else:
            raise RuntimeError(f"Unexpected forward_depth return len={len(out)}")
    else:
        # 모델 구현에 따라 단일 객체를 반환하지 않는다고 가정(낙관적 경로)
        pred_t, new_cache, extra = out

    # 3) [B,1,H,W] -> [B,H,W]
    if hasattr(pred_t, "dim") and pred_t.dim() == 4 and pred_t.size(1) == 1:
        pred_t = pred_t[:, 0]

    # 4) 인터미디엇 정리
    if not collect_inter:
        if return_encoder_feats:
            return pred_t, new_cache, feats
        return pred_t, new_cache

    # 기대 포맷: extra["intermediates"][li] -> {"feat":[B,1,C] or [B,T,C], "qkv":dict or None}
    raw_inter = extra.get("intermediates", extra)
    inter_t = {}

    for k, v in raw_inter.items():
        feat = v.get("feat_one", v.get("feat", None))  # 'feat_one'을 우선 사용, 없으면 'feat'
        qkv  = v.get("qkv", None)

        if feat is None:
            continue

        # 보정: [B,C] -> [B,1,C], [B,T,C] -> 마지막 타임스텝 [B,1,C] 로 맞춤
        if feat.dim() == 2:
            feat = feat.unsqueeze(1)                 # [B,1,C]
        elif feat.dim() == 3 and feat.size(1) != 1:
            feat = feat[:, -1:, :]                   # [B,1,C] (스트리밍 1-step이면 보통 이미 1임)

        inter_t[int(k)] = {
            "feat_one": feat,
            "qkv": qkv if collect_qkv else None,
        }

    if return_encoder_feats:
        return pred_t, new_cache, inter_t, feats
    return pred_t, new_cache, inter_t

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

# ──────────────────────────────────────────────────────────────────────────────
# Mini validation using the SAME pipeline as benchmark/infer/infer_stream.py
# + benchmark/eval/eval.py — but restricted to a few scenes (e.g., 2)
# ──────────────────────────────────────────────────────────────────────────────
import gc
import json
import cv2
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
    scenes_to_eval=2,          # 시간관계상 2 scene만
    scene_indices=None,      # 계산할 씬 index
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
    processed = 0
    seq_registry = []  # 평가 시 동일 순서/동일 subset을 재사용하기 위해 기록

    for i, data in enumerate(tqdm(path_json[dataset], desc=f"[VAL] Streaming {dataset} (subset)")):
        if scene_indices is not None and i not in set(scene_indices):
            continue
        for key in data.keys():
            if processed >= scenes_to_eval:
                break
            frames = data[key]  # list of dicts: {'image','gt_depth','factor',...}

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

            seq_registry.append((key, frames))
            processed += 1
            
            # scene_indices 모드가 아닐 때만 early stop
            if scene_indices is None and processed >= scenes_to_eval:
                break
        if scene_indices is None and processed >= scenes_to_eval:
            break

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

        all_metrics.append(seq_metrics)

    # 평균 내기
    if len(all_metrics) == 0:
        avg = {"abs_relative_difference": float("nan"),
               "rmse_linear": float("nan"),
               "delta1_acc": float("nan")}
    else:
        avg = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0].keys()
        }

    # 모델 모드 복구
    if model_was_training:
        model.train()

    return avg


# ──────────────────────────────────────────────────────────────────────────────
# KITTI Validation (Stream mode with SSI+TGM losses + depth metrics)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_kitti_streaming(
    model,
    val_loader,
    device,
    loss_ssi_fn,
    loss_tgm_fn,
    ratio_ssi=1.0,
    ratio_tgm=10.0,
    min_depth=1e-3,
    max_depth=80.0
):
    """
    KITTI validation with streaming inference.
    
    Returns:
        dict: {
            'loss': average validation loss,
            'ssi': average SSI loss,
            'tgm': average TGM loss,
            'absrel': average absolute relative error,
            'delta1': average delta1 accuracy
        }
    """
    model_was_training = model.training
    model.eval()
    
    total_loss = 0.0
    total_ssi = 0.0
    total_tgm = 0.0
    total_absrel = 0.0
    total_delta1 = 0.0
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(tqdm(val_loader, desc="KITTI Val", leave=False)):
        # KITTI val returns multiple values, we only need x and y
        # Unpack first two regardless of total length
        x = batch_data[0]
        y = batch_data[1]
        # Ignore extrinsics, intrinsics, cam_ids if present
        
        x = x.to(device)  # [B, T, 3, H, W]
        y = y.to(device)  # [B, T, 1, H, W]
        
        B, T = x.shape[:2]
        
        # Reset streaming state
        _reset_streaming_state(model)
        
        pred_list = []
        cache = None
        
        # Frame-by-frame streaming
        for t in range(T):
            x_t = x[:, t:t+1]  # [B, 1, 3, H, W]
            
            # Streaming inference
            pred_t, cache = model_stream_step(model, x_t, cache, collect_inter=False)
            pred_t = to_BHW_pred(pred_t).clamp(min=1e-6)  # [B, H, W]
            pred_list.append(pred_t)
        
        # Stack predictions: [B, T, H, W]
        pred_clip = torch.stack(pred_list, dim=1)
        
        # Compute mask
        mask = get_mask(y, min_depth, max_depth).squeeze(2)  # [B, T, H, W]
        
        # Convert to disparity for loss computation
        gt_disp = (1.0 / y.clamp(min=1e-6)).squeeze(2)  # [B, T, H, W]
        pred_disp = pred_clip  # Already disparity from model
        
        # Scale-shift alignment per clip (batch-wise)
        pred_disp_flat = pred_disp.view(B, -1)  # [B, T*H*W]
        gt_disp_flat = gt_disp.view(B, -1)
        mask_flat = mask.view(B, -1).float()
        
        # Compute scale & shift
        count = mask_flat.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pred = (pred_disp_flat * mask_flat).sum(dim=1, keepdim=True) / count
        mean_gt = (gt_disp_flat * mask_flat).sum(dim=1, keepdim=True) / count
        
        pred_centered = (pred_disp_flat - mean_pred) * mask_flat
        gt_centered = (gt_disp_flat - mean_gt) * mask_flat
        
        cov = (pred_centered * gt_centered).sum(dim=1, keepdim=True)
        var = (pred_centered ** 2).sum(dim=1, keepdim=True).clamp_min(1e-6)
        
        scale = cov / var
        shift = mean_gt - scale * mean_pred
        
        # Aligned prediction
        pred_aligned_flat = pred_disp_flat * scale + shift
        pred_aligned = pred_aligned_flat.view(B, T, pred_clip.shape[-2], pred_clip.shape[-1])
        
        # SSI loss (normalized)
        disp_normed = norm_ssi(y, mask.unsqueeze(2)).squeeze(2)  # [B, T, H, W]
        ssi_loss = loss_ssi_fn(pred_aligned.unsqueeze(2), disp_normed.unsqueeze(2), mask)
        
        # TGM loss (temporal consistency)
        if T >= 2:
            pred_depth = 1.0 / pred_aligned.clamp(min=1e-6)
            gt_depth = y.squeeze(2)
            tgm_loss = loss_tgm_fn(pred_aligned, gt_depth, mask)
        else:
            tgm_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = ratio_ssi * ssi_loss + ratio_tgm * tgm_loss
        
        # Depth metrics (convert back to depth domain)
        pred_depth_aligned = 1.0 / pred_aligned.clamp(min=1e-6)
        gt_depth = y.squeeze(2)
        
        # Per-clip metrics
        absrel = abs_relative_difference(pred_depth_aligned, gt_depth, mask).item()
        delta1 = delta1_acc(pred_depth_aligned, gt_depth, mask).item()
        
        # Accumulate
        total_loss += loss.item() * B
        total_ssi += ssi_loss.item() * B
        total_tgm += tgm_loss.item() * B
        total_absrel += absrel * B
        total_delta1 += delta1 * B
        total_samples += B
    
    # Average
    avg_metrics = {
        'loss': total_loss / max(1, total_samples),
        'ssi': total_ssi / max(1, total_samples),
        'tgm': total_tgm / max(1, total_samples),
        'absrel': total_absrel / max(1, total_samples),
        'delta1': total_delta1 / max(1, total_samples)
    }
    
    # Restore model mode
    if model_was_training:
        model.train()
    
    return avg_metrics
