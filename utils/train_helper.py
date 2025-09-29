# utils/train_helper.py
import os
import gc
import json
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from benchmark.eval.metric import abs_relative_difference, delta1_acc


# ======================= shape helpers used by train.py =======================
def to_BHW_pred(pred):
    """
    입력: [B,H,W] | [B,1,H,W] | [B,C,H,W] | [B,C,T,H,W] | (tuple/list) 등
    반환: [B,H,W]
    """
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if not torch.is_tensor(pred):
        raise ValueError(f"Expected tensor but got {type(pred)}")

    if pred.dim() == 3:                      # [B,H,W]
        return pred
    if pred.dim() == 4:                      # [B,C,H,W]
        return pred[:, 0] if pred.size(1) == 1 else pred.mean(dim=1)
    if pred.dim() == 5:                      # [B,C,T,H,W] → 첫 프레임
        return pred[:, 0, 0]
    raise ValueError(f"Unexpected pred shape: {pred.shape}")


def batch_ls_scale_shift(pred_disp, gt_disp, mask):
    """
    pred_disp: [B,H,W] 또는 [B,1,H,W] (disparity)
    gt_disp  : [B,H,W] 또는 [B,1,H,W] (disparity)
    mask     : [B,1,H,W] | [B,H,W] | [B,C,H,W] (유효영역 1)
    return a_star, b_star with shape [B,1,1,1]
    """
    # pred/gt → [B,H,W]
    if pred_disp.dim() == 4 and pred_disp.size(1) == 1:
        p = pred_disp[:, 0]
    elif pred_disp.dim() == 3:
        p = pred_disp
    else:
        p = pred_disp.mean(dim=1) if pred_disp.dim() == 4 else pred_disp

    if gt_disp.dim() == 4 and gt_disp.size(1) == 1:
        g = gt_disp[:, 0]
    elif gt_disp.dim() == 3:
        g = gt_disp
    else:
        g = gt_disp.mean(dim=1) if gt_disp.dim() == 4 else gt_disp

    B, H, W = p.shape

    # mask → [B,H,W]
    m = mask
    while m.dim() > 4:       # [B,1,1,H,W] 등
        m = m.squeeze(2)
    if m.dim() == 4:
        m = m[:, 0] if m.size(1) == 1 else m.max(dim=1, keepdim=False)[0]
    elif m.dim() != 3:
        raise ValueError(f"mask must be [B,H,W] or [B,1,H,W], got {m.shape}")

    if (m.shape[-2:] != (H, W)) or (m.shape[0] != B):
        raise RuntimeError(f"mask shape {m.shape} mismatch with pred {p.shape}")

    # 안정적인 batched lstsq
    with autocast(enabled=False):
        p_flat = p.float().view(B, -1)
        g_flat = g.float().view(B, -1)
        m_flat = m.float().view(B, -1)

        zero_rows = (m_flat.sum(dim=1) < 1)
        if zero_rows.any():
            a_star = torch.ones(B, 1, 1, 1, device=p.device, dtype=torch.float32)
            b_star = torch.zeros(B, 1, 1, 1, device=p.device, dtype=torch.float32)
            return a_star, b_star

        A = torch.stack([p_flat, torch.ones_like(p_flat)], dim=-1)  # [B,P,2]
        A = A * m_flat.unsqueeze(-1)
        b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)         # [B,P,1]

        X = torch.linalg.lstsq(A, b_vec).solution                   # [B,2,1]
        a_star = X[:, 0, 0].view(B, 1, 1, 1).clamp(1e-4, 1e4)
        b_star = X[:, 1, 0].view(B, 1, 1, 1).clamp(-1e4, 1e4)

    return a_star, b_star


def _detach_cache(cache):
    """detach/clone 없이 그래프만 끊기 위한 캐시 분리"""
    if cache is None:
        return None
    if isinstance(cache, (list, tuple)):
        return type(cache)(_detach_cache(c) for c in cache)
    if isinstance(cache, dict):
        return {k: _detach_cache(v) for k, v in cache.items()}
    if torch.is_tensor(cache):
        return cache.detach()
    return cache


def model_stream_step(
    model, x_t, cache=None,
    bidirectional_update_length=16, current_frame=0,
    # 호환성 인자 (내부 미사용)
    stream_mode: bool = True,
    select_top_r: int = None,
    update_top_u: int = 32,
    rope_dt=None,
    return_attn: bool = False,
    return_qkv: bool = False,
):
    """
    x_t: [B,1,3,H,W]
    return: pred_t [B,H,W], new_cache
    """
    # Student wrapper 대응
    if hasattr(model, 'student'):
        actual = model.student.module if hasattr(model.student, 'module') else model.student
    else:
        actual = model.module if hasattr(model, 'module') else model

    # 1) features
    features = actual.forward_features(x_t)

    # 2) streaming depth
    out = actual.forward_depth(features, x_t.shape, cached_hidden_state_list=cache)

    # 3) parse
    if isinstance(out, (list, tuple)):
        pred_t, new_cache = (out[0], out[1]) if len(out) >= 2 else (out[0], None)
    else:
        pred_t, new_cache = out, None

    # 4) [B,H,W]
    if isinstance(pred_t, torch.Tensor):
        if pred_t.dim() == 4:
            pred_t = pred_t[:, 0] if pred_t.size(1) == 1 else pred_t.mean(dim=1)
    return pred_t, new_cache


# ======================= fast validation helpers =======================
def _reset_streaming_state(model):
    """infer_video_depth_one과 동일한 스트리밍 상태 리셋(모델 내부 캐시 초기화)."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "transform"):        m.transform = None
    if hasattr(m, "frame_cache_list"): m.frame_cache_list = []
    if hasattr(m, "frame_id_list"):    m.frame_id_list = []
    if hasattr(m, "id"):               m.id = -1


def _vdainfer_one(model, rgb_img, input_size=518, device='cuda', fp32=True):
    """VideoDepthAnything.infer_video_depth_one thin wrapper -> (H,W) depth np.float32"""
    m = model.module if hasattr(model, "module") else model
    return m.infer_video_depth_one(rgb_img, input_size=input_size, device=device, fp32=fp32)


def _get_gt_depth(path, factor):
    if path.endswith('.npy'):
        depth = np.load(path).astype(np.float32)
    else:
        depth = cv2.imread(path, -1).astype(np.float32)
    depth = depth / float(factor)
    depth[depth == 0] = -1.0
    return depth


def _depth2disp_np(depth):
    disp = np.zeros_like(depth, dtype=np.float32)
    m = depth > 0
    disp[m] = 1.0 / depth[m]
    return disp


def _disp2depth_np(disp):
    depth = np.zeros_like(disp, dtype=np.float32)
    m = disp > 0
    depth[m] = 1.0 / disp[m]
    return depth


def _dataset_eval_defaults(dataset_tag):
    """Scannet preset. (필요 시 확장)"""
    if dataset_tag in ("scannet", "scannet_500"):
        return {
            "max_depth_eval": 10.0,
            "min_depth_eval": 0.1,
            "max_eval_len":   500 if dataset_tag == "scannet_500" else 90,
            "crop": (8, -8, 11, -11),
        }
    return {"max_depth_eval": 10.0, "min_depth_eval": 0.1, "max_eval_len": 90, "crop": (0, -1, 0, -1)}


# --- utils/train_helper.py: helper 추가(파일 내 임의 위치, 예: _disp2depth_np 아래) ---
def _robust_align_depth_frame(pd2d, gd2d, vm2d, *, min_pts=500, trim=(2, 98)):
    """
    한 프레임에 대해 robust OLS로 depth 정렬.
    입력:
      pd2d: pred depth (H,W)
      gd2d: gt   depth (H,W)
      vm2d: valid mask (H,W) [True=유효, 보통 gt 범위로 구성]
    반환:
      aligned_depth (H,W) 또는 None(정렬 실패 → 프레임 스킵)
    """
    # 유효·유한·양수만 사용
    m = (vm2d.astype(bool) &
         np.isfinite(pd2d) & np.isfinite(gd2d) &
         (pd2d > 0) & (gd2d > 0))
    if m.sum() < min_pts:
        return None

    x = pd2d[m].astype(np.float64)  # pred depth
    y = gd2d[m].astype(np.float64)  # gt   depth

    # 퍼센타일 트림(2~98%) – x,y 각각 트림 후 교집합
    p_lo, p_hi = trim
    try:
        x_lo, x_hi = np.percentile(x, [p_lo, p_hi])
        y_lo, y_hi = np.percentile(y, [p_lo, p_hi])
    except Exception:
        return None
    keep = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    x = x[keep]; y = y[keep]
    if x.size < min_pts:
        return None

    # 닫힌형식 OLS: y ≈ α x + β
    x_mean = x.mean(); y_mean = y.mean()
    xc = x - x_mean
    var_x = (xc * xc).mean()
    if (not np.isfinite(var_x)) or (var_x < 1e-12):
        return None
    cov_xy = (xc * (y - y_mean)).mean()
    alpha = cov_xy / (var_x + 1e-12)
    beta  = y_mean - alpha * x_mean
    if (not np.isfinite(alpha)) or (not np.isfinite(beta)):
        return None

    return (alpha * pd2d + beta)

# --- utils/train_helper.py: validate_with_infer_eval_subset_fast 교체 ---
@torch.no_grad()
def validate_with_infer_eval_subset_fast(
    model,
    json_file,
    infer_path,                   # unused(시그니처 유지)
    dataset="scannet",
    dataset_eval_tag="scannet_500",
    device="cuda",
    input_size=518,
    scenes_to_eval=2,
    scene_indices=None,
    frame_stride=2,
    max_eval_len=None,
    fp32=False,
    # ▼ 새 옵션
    align_mode: str = "depth_per_frame_robust",   # "disp_global" | "depth_per_frame_robust"
    trim=(2,98),                       # robust 모드에서만 사용
    min_pts:int=500,                   # robust 모드에서만 사용
):
    """
    - 빠른 스트리밍 평가
    - 모델 출력은 'depth'로 취급
    - align_mode:
        * "disp_global":  씬 전체 프레임을 모아 disparity 공간에서 (scale, shift) 1쌍을 OLS로 전역 추정
                          → eval.py와 동일 철학. 정렬된 disparity를 다시 depth로 환산 후 지표 계산
        * "depth_per_frame_robust": (기존) 프레임별 2~98% 트림 OLS depth↔depth 정렬 (실패 프레임 스킵)
    - 반환: 평균 지표 + per_scene
    """
    import gc, json, cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F

    torch.backends.cudnn.benchmark = True
    was_training = model.training
    model.eval()

    with open(json_file, "r") as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(json_file)

    # 씬 선택
    all_scenes = []
    for data in path_json[dataset]:
        for key, frames in data.items():
            all_scenes.append((len(all_scenes), key, frames))
    if scene_indices is None:
        target = all_scenes[:scenes_to_eval]
    else:
        idx_set = set(scene_indices)
        target = [tpl for tpl in all_scenes if tpl[0] in idx_set]
        if len(target) < scenes_to_eval and scene_indices is None:
            target += all_scenes[len(target):scenes_to_eval]

    # preset & 안전 크롭
    defaults = _dataset_eval_defaults(dataset_eval_tag)
    max_depth_eval = defaults["max_depth_eval"]
    preset_max_len = defaults["max_eval_len"]
    ca, cb, cc, cd = defaults["crop"]
    ca = int(ca) if ca is not None else None
    cb = int(cb) if cb is not None else None
    cc = int(cc) if cc is not None else None
    cd = int(cd) if cd is not None else None
    max_len = preset_max_len if max_eval_len is None else int(max_eval_len)

    def _depth2disp_np(depth):
        disp = np.zeros_like(depth, dtype=np.float32)
        m = depth > 0
        disp[m] = 1.0 / depth[m]
        return disp

    all_metrics, per_scene = [], []

    for (scene_idx, key, frames) in tqdm(target, desc=f"[VAL-FAST] {dataset}"):
        frames_eff = frames[::max(1, int(frame_stride))][:max_len]
        if len(frames_eff) == 0:
            continue

        _reset_streaming_state(model)

        if align_mode == "disp_global":
            # ── 씬 전체 프레임을 모아 disparity 전역 정렬 ──
            gts_list, preds_list, mask_list = [], [], []

            for item in frames_eff:
                # 입력 이미지
                img_path = os.path.join(root_path, item["image"])
                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # 모델 추론 depth(H,W)
                pred_depth = _vdainfer_one(model, rgb, input_size=input_size,
                                           device=device, fp32=fp32)

                # GT depth
                gt_depth = _get_gt_depth(os.path.join(root_path, item["gt_depth"]),
                                         item["factor"])

                # 크롭 & 리사이즈
                gt_c = gt_depth[ca:cb, cc:cd]
                H, W = gt_c.shape
                pred_resized = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_LINEAR)

                # 유효 마스크
                valid_mask = (gt_c > 1e-3) & (gt_c < max_depth_eval)

                # (나중에 한꺼번에 정렬할 것이므로) 원시 depth만 수집
                gts_list.append(gt_c.astype(np.float32))
                preds_list.append(pred_resized.astype(np.float32))
                mask_list.append(valid_mask.astype(np.bool_))

            if len(preds_list) == 0:
                continue

            # stack
            gts   = np.stack(gts_list,   axis=0)  # [T,H,W]
            preds = np.stack(preds_list, axis=0)  # [T,H,W]
            mks   = np.stack(mask_list,  axis=0)  # [T,H,W] bool

            # disparity 전역 정렬 (eval.py와 동일)
            # gt_disp_masked, pred_disp_masked 만들기
            gt = gts.copy()
            pr = preds.copy()
            pr = np.clip(pr, a_min=1e-3, a_max=None)

            gt_disp = np.zeros_like(gt, dtype=np.float64)
            nz = gt > 0
            gt_disp[nz] = 1.0 / gt[nz]

            pr_disp = np.zeros_like(pr, dtype=np.float64)
            pr_disp[pr > 0] = 1.0 / pr[pr > 0]

            mask = mks
            # 전 프레임-전 픽셀에서 유효한 것만 모아서 전역 OLS
            x = pr_disp[mask].reshape(-1, 1)  # pred_disp
            y = gt_disp[mask].reshape(-1, 1)  # gt_disp

            if x.size < 10:
                # 유효 픽셀이 너무 적으면 스킵
                continue

            A = np.concatenate([x, np.ones_like(x)], axis=-1)  # [N,2]
            # 최소자승 해 [scale, shift]
            X = np.linalg.lstsq(A, y, rcond=None)[0].astype(np.float64)
            scale, shift = float(X[0, 0]), float(X[1, 0])

            # 정렬된 pred_disp → depth 환산
            aligned_disp = scale * pr_disp + shift
            # 수치 안전
            aligned_disp = np.clip(aligned_disp, a_min=1e-6, a_max=None)
            aligned_depth = 1.0 / aligned_disp
            aligned_depth = np.clip(aligned_depth, a_min=1e-3, a_max=max_depth_eval).astype(np.float32)

            # torch 텐서로 변환 & 무효 프레임 제거
            gts_t   = torch.from_numpy(gts.astype(np.float32)).to(device)
            preds_t = torch.from_numpy(aligned_depth).to(device)
            mks_t   = torch.from_numpy(mask.astype(np.bool_)).to(device)

            keep = (mks_t.sum((-1, -2)) > 0)
            gts_t, preds_t, mks_t = gts_t[keep], preds_t[keep], mks_t[keep]
            if gts_t.numel() == 0:
                continue

            seq = {
                "abs_relative_difference": abs_relative_difference(preds_t, gts_t, mks_t).item(),
                "rmse_linear": torch.sqrt(((preds_t - gts_t) ** 2 * mks_t).sum() /
                                          mks_t.sum().clamp(min=1)).item(),
                "delta1_acc": delta1_acc(preds_t, gts_t, mks_t).item()
            }
            all_metrics.append(seq)
            per_scene.append({
                "scene_idx": int(scene_idx),
                "scene_key": str(key),
                **seq,
                # 디버깅용: 추정된 (scale,shift)와 유효 픽셀 수
                "align_scale": float(scale),
                "align_shift": float(shift),
                "valid_px": int(mask.sum()),
            })
            torch.cuda.empty_cache()

        else:
            # ── (이전 버전) 프레임별 robust depth↔depth 정렬 ──
            gts_list, preds_list, mask_list = [], [], []

            for item in frames_eff:
                img_path = os.path.join(root_path, item["image"])
                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                pred_depth = _vdainfer_one(model, rgb, input_size=input_size,
                                           device=device, fp32=fp32)
                gt_depth = _get_gt_depth(os.path.join(root_path, item["gt_depth"]),
                                         item["factor"])

                gt_c = gt_depth[ca:cb, cc:cd]
                H, W = gt_c.shape
                pred_resized = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                valid_mask = (gt_c > 1e-3) & (gt_c < max_depth_eval)

                aligned = _robust_align_depth_frame(
                    pred_resized, gt_c, valid_mask, min_pts=min_pts, trim=trim
                )
                if aligned is None:
                    continue

                aligned = np.clip(aligned, a_min=1e-3, a_max=max_depth_eval)
                gts_list.append(gt_c.astype(np.float32))
                preds_list.append(aligned.astype(np.float32))
                mask_list.append(valid_mask.astype(np.bool_))

            if len(preds_list) == 0:
                continue

            gts   = torch.from_numpy(np.stack(gts_list,   axis=0)).to(device)
            preds = torch.from_numpy(np.stack(preds_list, axis=0)).to(device)
            mks   = torch.from_numpy(np.stack(mask_list,  axis=0)).to(device)

            keep = (mks.sum((-1, -2)) > 0)
            gts, preds, mks = gts[keep], preds[keep], mks[keep]
            if gts.numel() == 0:
                continue

            seq = {
                "abs_relative_difference": abs_relative_difference(preds, gts, mks).item(),
                "rmse_linear": torch.sqrt(((preds - gts) ** 2 * mks).sum() /
                                          mks.sum().clamp(min=1)).item(),
                "delta1_acc": delta1_acc(preds, gts, mks).item()
            }
            all_metrics.append(seq)
            per_scene.append({
                "scene_idx": int(scene_idx),
                "scene_key": str(key),
                **seq
            })
            torch.cuda.empty_cache()

    # 집계
    if len(all_metrics) == 0:
        avg = {"abs_relative_difference": float("nan"),
               "rmse_linear": float("nan"),
               "delta1_acc": float("nan")}
    else:
        avg = {k: float(np.mean([m[k] for m in all_metrics]))
               for k in all_metrics[0].keys()}

    # per-scene 부가정보
    avg["per_scene_delta1"] = [m["delta1_acc"] for m in per_scene]
    avg["per_scene"] = per_scene

    if was_training:
        model.train()
    torch.cuda.empty_cache(); gc.collect()
    return avg
