import os
import gc
import json
import cv2
import warnings

import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast

from benchmark.eval.metric import abs_relative_difference, delta1_acc, rmse_linear


# ────────────────────────────── 유틸/평가 함수 ──────────────────────────────
def get_mask(depth_m, min_depth, max_depth):
    """
    depth_m: [B,T,1,H,W] or [B,1,1,H,W] or [B,T,H,W]
    return:  same shape (without channel squeeze) bool mask
    """
    return ((depth_m > min_depth) & (depth_m < max_depth)).bool()


def norm_ssi(depth, valid_mask):
    """
    depth:      [B,T,1,H,W]
    valid_mask: [B,T,1,H,W] bool

    각 클립/프레임별 disparity를 0~1로 정규화한 후, invalid는 0으로 채움.
    """
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
    """
    pred: [B,H,W] or [B,1,H,W] or [B,C,H,W] or [B,C,T,H,W]
    return: [B,H,W]
    """
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    if not torch.is_tensor(pred):
        raise ValueError(f"Expected tensor but got {type(pred)}")

    if pred.dim() == 3:
        return pred  # [B,H,W]

    if pred.dim() == 4:
        if pred.size(1) == 1:
            return pred[:, 0]          # [B,H,W]
        else:
            return pred.mean(dim=1)    # [B,H,W]

    if pred.dim() == 5:
        # [B,C,T,H,W] → 첫 프레임만 사용
        return pred[:, 0, 0]           # [B,H,W]

    raise ValueError(f"Unexpected pred shape: {pred.shape}")


def batch_ls_scale_shift(pred_disp, gt_disp, mask):
    """
    pred_disp: [B,H,W] or [B,1,H,W] disparity (>= 1e-6)
    gt_disp  : [B,H,W] or [B,1,H,W] disparity
    mask     : [B,1,H,W] bool

    return: a_star, b_star with shape [B,1,1,1]
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

    # autocast 비활성 + float32 로 lstsq 안정성 확보
    with autocast(enabled=False):
        m = mask.view(B, -1).float()                # [B,P]
        p_flat = p.float().view(B, -1)              # [B,P]
        g_flat = g.float().view(B, -1)              # [B,P]

        A = torch.stack(
            [p_flat, torch.ones_like(p_flat, device=p.device)],
            dim=-1
        )                                           # [B,P,2]
        A = A * m.unsqueeze(-1)
        b_vec = g_flat.unsqueeze(-1) * m.unsqueeze(-1)

        X = torch.linalg.lstsq(A, b_vec).solution   # [B,2,1]
        a_star = X[:, 0, 0].view(B, 1, 1, 1)
        b_star = X[:, 1, 0].view(B, 1, 1, 1)

    a_star = a_star.clamp(min=1e-4, max=1e4)
    b_star = b_star.clamp(min=-1e4, max=1e4)
    return a_star, b_star


# ────────────────────────────── Streaming 상태 유틸 ──────────────────────────────
@torch.no_grad()
def _reset_streaming_state(model):
    """
    infer_stream.py 와 동일하게 VideoDepthAnything 의
    streaming 관련 상태만 초기화.
    """
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
    """
    VideoDepthAnything.infer_video_depth_one thin-wrapper.
    """
    m = model.module if hasattr(model, "module") else model
    return m.infer_video_depth_one(rgb_img, input_size=input_size, device=device, fp32=fp32)


# ────────────────────────────── ScanNet mini-eval 유틸 ──────────────────────────────
def _get_infer_npy(path, target_hw=None):
    arr = np.load(path).astype(np.float32)
    if target_hw is not None and (arr.shape[0] != target_hw[0] or arr.shape[1] != target_hw[1]):
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]))
    return arr


def _get_gt_depth(path, factor):
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
    Numpy 기반 disparity LS 정렬 (eval.py 와 동일).
    """
    infs = np.clip(infs, a_min=1e-3, a_max=1e6)
    gts  = np.clip(gts,  a_min=1e-3, a_max=1e6)

    if not np.all(np.isfinite(infs)) or not np.all(np.isfinite(gts)):
        warnings.warn("Non-finite values detected in disparity alignment, returning original predictions")
        return infs

    n_valid = np.sum(valid_mask)
    if n_valid < 10:
        warnings.warn(f"Too few valid pixels ({n_valid}) for alignment, returning original predictions")
        return infs

    try:
        gt_disp_masked   = 1.0 / (gts[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
        pred_disp_masked = infs[valid_mask].reshape((-1, 1)).astype(np.float64)

        gt_disp_masked   = np.clip(gt_disp_masked,   1e-6, 1e6)
        pred_disp_masked = np.clip(pred_disp_masked, 1e-6, 1e6)

        if not np.all(np.isfinite(gt_disp_masked)) or not np.all(np.isfinite(pred_disp_masked)):
            warnings.warn("Non-finite values in masked disparity, returning original predictions")
            return infs

        A = np.concatenate([pred_disp_masked, np.ones_like(pred_disp_masked)], axis=-1)  # [P,2]
        X = np.linalg.lstsq(A, gt_disp_masked, rcond=1e-6)[0]                           # [2,1]
        scale, shift = X[0, 0], X[1, 0]

        if not np.isfinite(scale) or not np.isfinite(shift):
            warnings.warn("Non-finite scale/shift computed, returning original predictions")
            return infs

        scale = np.clip(scale, 1e-3, 1e3)
        shift = np.clip(shift, -1e6, 1e6)

        aligned = np.clip(scale * infs + shift, a_min=1e-3, a_max=1e6)
        return aligned

    except np.linalg.LinAlgError as e:
        warnings.warn(f"LinAlgError in disparity alignment: {e}, returning original predictions")
        return infs
    except Exception as e:
        warnings.warn(f"Unexpected error in disparity alignment: {e}, returning original predictions")
        return infs


def _dataset_eval_defaults(dataset_tag):
    """
    eval.py 와 동일한 preset (여기선 scannet / scannet_500만 커버).
    """
    if dataset_tag in ("scannet", "scannet_500"):
        return {
            "max_depth_eval": 10.0,
            "min_depth_eval": 0.1,
            "max_eval_len":   500 if dataset_tag == "scannet_500" else 90,
            "crop": (8, -8, 11, -11),
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
    json_file,
    infer_path,
    dataset="scannet",
    dataset_eval_tag="scannet_500",
    device="cuda",
    input_size=518,
    scenes_to_eval=2,
    scene_indices=None,
    fp32=True,
):
    """
    ScanNet mini-validation:
    - infer_stream.py + eval.py 와 동일 파이프라인
    - scene 개수만 줄여서 빠르게 실행
    """
    os.makedirs(infer_path, exist_ok=True)
    model_was_training = model.training
    model.eval()

    # scene subset 선택
    target_indices = None
    if scene_indices is not None:
        if isinstance(scene_indices, (list, tuple, set)):
            if len(scene_indices) == 0:
                raise ValueError("scene_indices must contain at least one index when provided")
            target_indices = sorted({int(idx) for idx in scene_indices})
        else:
            raise TypeError("scene_indices must be an iterable of integers or None")
    target_index_set = set(target_indices) if target_indices is not None else None
    target_count = len(target_indices) if target_indices is not None else scenes_to_eval

    # JSON 로드
    with open(json_file, 'r') as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(json_file)

    # Inference
    processed = 0
    seq_registry = []

    for scene_idx, data in enumerate(tqdm(path_json[dataset], desc=f"[VAL] Streaming {dataset} (subset)")):
        if target_index_set is not None and scene_idx not in target_index_set:
            continue
        for key in data.keys():
            if processed >= target_count:
                break
            frames = data[key]

            _reset_streaming_state(model)

            for item in frames:
                img_path = os.path.join(root_path, item['image'])
                base, _ = os.path.splitext(item['image'])
                out_path = os.path.join(infer_path, dataset, base + '.npy')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(img_path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                depth_np = _vdainfer_one(model, rgb, input_size=input_size, device=device, fp32=fp32)
                np.save(out_path, depth_np)

            seq_registry.append((scene_idx, key, frames))
            processed += 1

            if target_index_set is not None:
                break
        if processed >= target_count:
            break

    torch.cuda.empty_cache()
    gc.collect()

    # Eval
    defaults = _dataset_eval_defaults(dataset_eval_tag)
    max_depth_eval = defaults["max_depth_eval"]
    max_eval_len   = defaults["max_eval_len"]
    a, b, c, d     = defaults["crop"]

    all_metrics = []
    for (scene_idx, key, frames) in seq_registry:
        infer_paths = []
        depth_gt_paths = []
        factors = []
        for images in frames:
            infer_path_i = (os.path.join(infer_path, dataset, images['image'])
                            .replace('.jpg', '.npy').replace('.png', '.npy'))
            infer_paths.append(infer_path_i)
            depth_gt_paths.append(os.path.join(root_path, images['gt_depth']))
            factors.append(images['factor'])

        infer_paths    = infer_paths[:max_eval_len]
        depth_gt_paths = depth_gt_paths[:max_eval_len]
        factors        = factors[:max_eval_len]

        gts  = []
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

        gts  = np.stack(gts,  axis=0)
        infs = np.stack(infs, axis=0)

        valid_mask  = np.logical_and((gts > 1e-3), (gts < max_depth_eval))
        aligned_disp = _ls_align_disparity(infs, gts, valid_mask)
        pred_depth   = _depth2disp_np(aligned_disp)
        pred_depth   = np.clip(pred_depth, a_min=1e-3, a_max=max_depth_eval)

        pred_ts = torch.from_numpy(pred_depth).to(device)
        gt_ts   = torch.from_numpy(gts).to(device)
        mask_ts = torch.from_numpy(valid_mask).to(device)

        n_valid = mask_ts.sum((-1, -2))
        valid_frame = (n_valid > 0)
        pred_ts = pred_ts[valid_frame]
        gt_ts   = gt_ts[valid_frame]
        mask_ts = mask_ts[valid_frame]

        seq_metrics = {}
        seq_metrics["abs_relative_difference"] = abs_relative_difference(pred_ts, gt_ts, mask_ts).item()
        seq_metrics["rmse_linear"]             = rmse_linear(pred_ts, gt_ts, mask_ts).item()
        seq_metrics["delta1_acc"]              = delta1_acc(pred_ts, gt_ts, mask_ts).item()

        all_metrics.append((scene_idx, seq_metrics))

    if len(all_metrics) == 0:
        avg = {
            "abs_relative_difference": float("nan"),
            "rmse_linear": float("nan"),
            "delta1_acc": float("nan"),
        }
        per_scene = {}
    else:
        per_scene = {scene_idx: metrics for scene_idx, metrics in all_metrics}
        metric_keys = list(next(iter(per_scene.values())).keys())
        avg = {
            k: float(np.mean([metrics[k] for metrics in per_scene.values()]))
            for k in metric_keys
        }

    if model_was_training:
        model.train()

    return {"avg": avg, "per_scene": per_scene}


# ────────────────────────────── KITTI streaming validation ──────────────────────────────
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
    max_depth=80.0,
):
    """
    KITTI validation (strict streaming):
    - train 과 동일하게: frame-by-frame, m.stream_step_train 사용
    - SSI + TGM + AbsRel + δ1
    """
    model_was_training = model.training
    model.eval()

    total_loss   = 0.0
    total_ssi    = 0.0
    total_tgm    = 0.0
    total_absrel = 0.0
    total_delta1 = 0.0
    total_samples = 0

    m = model.module if hasattr(model, "module") else model

    for batch_idx, batch_data in enumerate(tqdm(val_loader, desc="KITTI Val", leave=False)):
        x = batch_data[0].to(device)  # [B,T,3,H,W]
        y = batch_data[1].to(device)  # [B,T,1,H,W]
        B, T = x.shape[:2]

        cache_state = None
        pred_list = []

        for t in range(T):
            x_t = x[:, t:t+1]  # [B,1,3,H,W]
            with autocast(enabled=torch.cuda.is_available()):
                pred_t_net, cache_state = m.stream_step_train(x_t, cache_state)
            pred_t = to_BHW_pred(pred_t_net).clamp(min=1e-6)
            pred_list.append(pred_t)

        pred_clip = torch.stack(pred_list, dim=1)  # [B,T,H,W]

        mask     = get_mask(y, min_depth, max_depth).squeeze(2)   # [B,T,H,W]
        gt_disp  = (1.0 / y.clamp(min=1e-6)).squeeze(2)           # [B,T,H,W]
        pred_disp = pred_clip

        # clip 단위 scale/shift 정렬
        pred_disp_flat = pred_disp.view(B, -1)
        gt_disp_flat   = gt_disp.view(B, -1)
        mask_flat      = mask.view(B, -1).float()

        count     = mask_flat.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pred = (pred_disp_flat * mask_flat).sum(dim=1, keepdim=True) / count
        mean_gt   = (gt_disp_flat   * mask_flat).sum(dim=1, keepdim=True) / count

        pred_centered = (pred_disp_flat - mean_pred) * mask_flat
        gt_centered   = (gt_disp_flat   - mean_gt)   * mask_flat

        cov = (pred_centered ** 1 * gt_centered).sum(dim=1, keepdim=True)
        var = (pred_centered ** 2).sum(dim=1, keepdim=True).clamp_min(1e-6)

        scale = cov / var
        shift = mean_gt - scale * mean_pred

        pred_aligned_flat = pred_disp_flat * scale + shift
        pred_aligned = pred_aligned_flat.view(B, T, pred_clip.shape[-2], pred_clip.shape[-1])

        # SSI loss
        disp_normed = norm_ssi(y, mask.unsqueeze(2)).squeeze(2)
        ssi_loss = loss_ssi_fn(
            pred_aligned.unsqueeze(2),    # [B,T,1,H,W]
            disp_normed.unsqueeze(2),     # [B,T,1,H,W]
            mask,                         # [B,T,H,W]
        )

        # TGM loss
        if T >= 2:
            pred_depth = 1.0 / pred_aligned.clamp(min=1e-6)
            gt_depth   = y.squeeze(2)
            tgm_loss   = loss_tgm_fn(pred_depth, gt_depth, mask)
        else:
            tgm_loss = torch.tensor(0.0, device=device)

        loss = ratio_ssi * ssi_loss + ratio_tgm * tgm_loss

        # depth metrics
        pred_depth_aligned = 1.0 / pred_aligned.clamp(min=1e-6)
        gt_depth = y.squeeze(2)

        absrel = abs_relative_difference(pred_depth_aligned, gt_depth, mask).item()
        delta1 = delta1_acc(pred_depth_aligned, gt_depth, mask).item()

        total_loss   += loss.item()     * B
        total_ssi    += ssi_loss.item() * B
        total_tgm    += tgm_loss.item() * B
        total_absrel += absrel * B
        total_delta1 += delta1 * B
        total_samples += B

    avg_metrics = {
        "loss":   total_loss   / max(1, total_samples),
        "ssi":    total_ssi    / max(1, total_samples),
        "tgm":    total_tgm    / max(1, total_samples),
        "absrel": total_absrel / max(1, total_samples),
        "delta1": total_delta1 / max(1, total_samples),
    }

    if model_was_training:
        model.train()

    return avg_metrics
