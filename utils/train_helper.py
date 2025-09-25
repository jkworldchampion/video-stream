import os
import json
import gc
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from benchmark.eval.metric import abs_relative_difference, delta1_acc

# ======================= 기본 상수 (시각화용) =======================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ======================= 코어 헬퍼들 (train.py가 요구) =======================
def get_mask(depth_m, min_depth, max_depth):
    """유효 깊이 마스크 (bool) 반환: depth_m shape [B,1,1,H,W] 또는 [B,1,H,W]"""
    return ((depth_m > min_depth) & (depth_m < max_depth)).bool()

def norm_ssi(depth, valid_mask):
    """
    depth: [B,1,1,H,W] 또는 [B,T,1,H,W]
    valid_mask: 동일 shape의 bool (또는 0/1 float도 허용; 내부에서 bool로 변환)
    return: depth와 같은 shape의 정규화된 disparity (유효영역만 [0,1], 나머지 0)
    """
    eps = 1e-6
    # ✅ 마스크를 확실히 bool로
    valid_mask = valid_mask.bool()

    # disparity 계산(유효 픽셀만 역수, 아니면 0)
    depth_safe = torch.clamp(depth, min=eps)
    disparity = torch.where(valid_mask, 1.0 / depth_safe, torch.zeros_like(depth_safe))

    # [B,T,1,H,W] 또는 [B,1,1,H,W] 처리를 위해 T축 유연 처리
    if depth.dim() == 5:
        B, T, C, H, W = depth.shape
        disp_flat = disparity.view(B, T, -1)
        mask_flat = valid_mask.view(B, T, -1)
    elif depth.dim() == 4:
        # 드물게 [B,1,H,W]가 올 수 있으면 T=1로 취급
        B, C, H, W = depth.shape
        T = 1
        disp_flat = disparity.view(B, T, -1)
        mask_flat = valid_mask.view(B, T, -1)
    else:
        raise ValueError(f"norm_ssi: unexpected depth shape {depth.shape}")

    # 유효영역 min/max (빈 프레임 보호)
    disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).amin(dim=-1, keepdim=True)
    disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).amax(dim=-1, keepdim=True)

    # 빈 프레임(유효 픽셀이 0) 방지용: min=max이면 분모를 1로
    span = (disp_max - disp_min).clamp_min(1e-6)

    # 다시 원형상태로 브로드캐스트
    disp_min = disp_min.view(*disparity.shape[:-2], 1, 1)
    span     = span.view(*disparity.shape[:-2], 1, 1)

    norm_disp = (disparity - disp_min) / span
    # 유효영역 밖은 0으로
    norm_disp = torch.where(valid_mask, norm_disp, torch.zeros_like(norm_disp))
    # NaN/Inf 방지
    norm_disp = torch.nan_to_num(norm_disp, nan=0.0, posinf=0.0, neginf=0.0)

    return norm_disp

def to_BHW_pred(pred):
    """
    입력: [B,H,W] 또는 [B,1,H,W] 또는 [B,C,H,W] 또는 (tuple/list) 포함 가능
    반환: [B,H,W]
    """
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if not torch.is_tensor(pred):
        raise ValueError(f"Expected tensor but got {type(pred)}")

    if pred.dim() == 3:              # [B,H,W]
        return pred
    if pred.dim() == 4:              # [B,C,H,W]
        return pred[:, 0] if pred.size(1) == 1 else pred.mean(dim=1)
    if pred.dim() == 5:              # [B,C,T,H,W] → 첫 프레임
        return pred[:, 0, 0]
    raise ValueError(f"Unexpected pred shape: {pred.shape}")

def batch_ls_scale_shift(pred_disp, gt_disp, mask):
    """
    pred_disp: [B,H,W] or [B,1,H,W] disparity (>= 1e-6)
    gt_disp  : [B,H,W] or [B,1,H,W] disparity
    mask     : [B,1,H,W] or [B,H,W] or (실수로) [B,C,H,W]  <- 무엇이 와도 [B,H,W]로 표준화
    return a_star, b_star with shape [B,1,1,1]
    """
    # ----- 표준화: pred/gt를 [B,H,W]로 -----
    if pred_disp.dim() == 4 and pred_disp.size(1) == 1:
        p = pred_disp[:, 0]
    elif pred_disp.dim() == 3:
        p = pred_disp
    else:
        # C>1인 예외 케이스는 평균으로 단일 채널화
        p = pred_disp.mean(dim=1) if pred_disp.dim() == 4 else pred_disp

    if gt_disp.dim() == 4 and gt_disp.size(1) == 1:
        g = gt_disp[:, 0]
    elif gt_disp.dim() == 3:
        g = gt_disp
    else:
        g = gt_disp.mean(dim=1) if gt_disp.dim() == 4 else gt_disp

    B, H, W = p.shape

    # ----- 표준화: mask를 [B,H,W]로 강제 -----
    m = mask
    # squeeze 중간 1차원들 (예: [B,1,1,H,W] → [B,1,H,W])
    while m.dim() > 4:
        m = m.squeeze(2)
    if m.dim() == 4:
        if m.size(1) == 1:
            m = m[:, 0]                      # [B,H,W]
        else:
            # 채널이 여러 개인 경우(예: [B,4,H,W]) → 가장 보수적으로 "어느 채널이든 유효"로 합치기
            m = m.max(dim=1, keepdim=False)[0]  # [B,H,W] (OR과 유사)
    elif m.dim() == 3:
        pass  # 이미 [B,H,W]
    else:
        raise ValueError(f"mask must be [B,H,W] or [B,1,H,W], got {m.shape}")

    # pred와 공간 해상도가 꼭 맞아야 함
    if (m.shape[-2:] != p.shape[-2:]) or (m.shape[0] != B):
        # 필요시 리사이즈(권장: 같은 전처리라면 보통 필요 없음)
        raise RuntimeError(f"mask shape {m.shape} mismatch with pred {p.shape}")

    # ----- 안정적인 최소제곱 -----
    with autocast(enabled=False):
        p_flat = p.float().view(B, -1)
        g_flat = g.float().view(B, -1)
        m_flat = m.float().view(B, -1)              # [B, H*W] ← 여기서 길이가 반드시 같아짐

        # 가끔 m_flat이 전부 0인 샘플 방지
        # (전부 0이면 A,b 모두 0이 되어 lstsq가 NaN 줄 수 있음 → 단위 변환)
        zero_rows = (m_flat.sum(dim=1) < 1)
        if zero_rows.any():
            # 이런 배치는 (a=1,b=0)로 두고 넘어감
            a_star = torch.ones(B, 1, 1, 1, device=p.device, dtype=torch.float32)
            b_star = torch.zeros(B, 1, 1, 1, device=p.device, dtype=torch.float32)
            return a_star, b_star

        A = torch.stack([p_flat, torch.ones_like(p_flat)], dim=-1)  # [B, P, 2]
        A = A * m_flat.unsqueeze(-1)                                # 마스크 적용
        b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)         # [B, P, 1]

        X = torch.linalg.lstsq(A, b_vec).solution                   # [B, 2, 1]
        a_star = X[:, 0, 0].view(B, 1, 1, 1).clamp(1e-4, 1e4)
        b_star = X[:, 1, 0].view(B, 1, 1, 1).clamp(-1e4, 1e4)

    return a_star, b_star

def _detach_cache(cache):
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
    # ↓↓↓ 인자들은 호환성 유지용으로 남겨두되 내부에서는 넘기지 않습니다.
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
    # Student 래핑 케이스 처리
    if hasattr(model, 'student'):
        actual = model.student.module if hasattr(model.student, 'module') else model.student
    else:
        actual = model.module if hasattr(model, 'module') else model

    # 1) 특징 추출
    features = actual.forward_features(x_t)

    # 2) forward_depth 호출 — 추가 kwargs 넘기지 않음!
    out = actual.forward_depth(features, x_t.shape, cached_hidden_state_list=cache)

    # 3) 반환 정리
    if isinstance(out, (list, tuple)):
        if len(out) >= 2:
            pred_t, new_cache = out[0], out[1]
        else:
            pred_t, new_cache = out[0], None
    else:
        pred_t, new_cache = out, None

    # 4) 출력 정규화 [B,H,W]
    if isinstance(pred_t, torch.Tensor):
        if pred_t.dim() == 4 and pred_t.size(1) == 1:
            pred_t = pred_t[:, 0]
        elif pred_t.dim() == 4 and pred_t.size(1) > 1:
            pred_t = pred_t.mean(dim=1)

    return pred_t, new_cache

# ======================= 검증 파이프라인(빠른 스트림 평가) =======================
def _reset_streaming_state(model):
    """infer_stream.py와 동일한 스트리밍 상태 리셋."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "transform"):        m.transform = None
    if hasattr(m, "frame_cache_list"): m.frame_cache_list = []
    if hasattr(m, "frame_id_list"):    m.frame_id_list = []
    if hasattr(m, "id"):               m.id = -1

def _vdainfer_one(model, rgb_img, input_size=518, device='cuda', fp32=True):
    """VideoDepthAnything.infer_video_depth_one thin wrapper."""
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

def _ls_align_disparity(infs, gts, valid_mask):
    """
    disparity 선형 정렬: (scale, shift)로 infs를 gts에 맞춤 (numpy lstsq)
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
    """Scannet 계열 preset. (필요 시 확장)"""
    if dataset_tag in ("scannet", "scannet_500"):
        return {
            "max_depth_eval": 10.0,
            "min_depth_eval": 0.1,
            "max_eval_len":   500 if dataset_tag == "scannet_500" else 90,
            "crop": (8, -8, 11, -11),
        }
    return {"max_depth_eval": 10.0, "min_depth_eval": 0.1, "max_eval_len": 90, "crop": (0, -1, 0, -1)}

@torch.no_grad()
def validate_with_infer_eval_subset_fast(
    model,
    json_file,
    infer_path,                   # 시그니처 유지를 위해 존재 (미사용)
    dataset="scannet",
    dataset_eval_tag="scannet_500",
    device="cuda",
    input_size=518,
    scenes_to_eval=2,
    scene_indices=None,           # 예: [0, 44]
    frame_stride=2,               # 프레임 다운샘플
    max_eval_len=None,            # None이면 preset 사용
    fp32=False,                   # 빠르게 하려면 False 권장
):
    """
    - 지정된 일부 씬만 빠르게 스트리밍 추론 + 메모리 내 평가
    - eval.py와 유사한 LS 정렬/크롭/지표 계산
    """
    torch.backends.cudnn.benchmark = True
    model_was_training = model.training
    model.eval()

    # 1) JSON 로드
    with open(json_file, "r") as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(json_file)

    # 2) 씬 나열
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

    # 3) preset
    defaults = _dataset_eval_defaults(dataset_eval_tag)
    max_depth_eval = defaults["max_depth_eval"]
    min_depth_eval = defaults["min_depth_eval"]
    preset_max_len = defaults["max_eval_len"]
    a, b, c, d     = defaults["crop"]
    max_len = preset_max_len if max_eval_len is None else int(max_eval_len)

    # 4) 씬별 추론+평가
    all_metrics = []
    per_scene_delta1 = []

    for (scene_idx, key, frames) in tqdm(target, desc=f"[VAL-FAST] {dataset}"):
        frames_eff = frames[::max(1, int(frame_stride))][:max_len]
        if len(frames_eff) == 0:
            continue

        _reset_streaming_state(model)

        gts_np, preds_np = [], []
        for item in frames_eff:
            img_path = os.path.join(root_path, item["image"])
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            depth_np = _vdainfer_one(model, rgb, input_size=input_size,
                                     device=device, fp32=fp32)  # (H,W) depth
            gts_np.append((os.path.join(root_path, item["gt_depth"]), item["factor"]))
            preds_np.append(depth_np)

        if len(preds_np) == 0:
            continue

        # Eval in-memory
        gts, infs = [], []
        for (gt_path, fac), pred in zip(gts_np, preds_np):
            gt_depth = _get_gt_depth(gt_path, fac)  # (H,W)
            gt_depth = gt_depth[a:b, c:d]
            pred_resized = cv2.resize(pred, (gt_depth.shape[1], gt_depth.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
            infs.append(pred_resized)
            gts.append(gt_depth)

        gts = np.stack(gts, axis=0)   # [T,H,W]
        infs = np.stack(infs, axis=0) # [T,H,W]

        valid_mask = np.logical_and((gts > 1e-3), (gts < max_depth_eval))
        aligned_disp = _ls_align_disparity(infs, gts, valid_mask)  # disparity aligned
        pred_depth  = _depth2disp_np(aligned_disp)
        pred_depth  = np.clip(pred_depth, a_min=1e-3, a_max=max_depth_eval)

        pred_ts = torch.from_numpy(pred_depth).to(device)
        gt_ts   = torch.from_numpy(gts).to(device)
        mask_ts = torch.from_numpy(valid_mask).to(device)

        n_valid = mask_ts.sum((-1, -2))
        valid_frame = (n_valid > 0)
        pred_ts = pred_ts[valid_frame]
        gt_ts   = gt_ts[valid_frame]
        mask_ts = mask_ts[valid_frame]
        if pred_ts.numel() == 0:
            continue

        seq_metrics = {
            "abs_relative_difference": abs_relative_difference(pred_ts, gt_ts, mask_ts).item(),
            "rmse_linear": torch.sqrt(((pred_ts - gt_ts) ** 2 * mask_ts).sum() /
                                      mask_ts.sum().clamp(min=1)).item(),
            "delta1_acc": delta1_acc(pred_ts, gt_ts, mask_ts).item()
        }
        all_metrics.append(seq_metrics)
        per_scene_delta1.append(seq_metrics["delta1_acc"])
        torch.cuda.empty_cache()

    # 5) 평균 집계
    if len(all_metrics) == 0:
        avg = {"abs_relative_difference": float("nan"),
               "rmse_linear": float("nan"),
               "delta1_acc": float("nan")}
    else:
        avg = {k: float(np.mean([m[k] for m in all_metrics]))
               for k in all_metrics[0].keys()}
    avg["per_scene_delta1"] = per_scene_delta1

    if model_was_training:
        model.train()
    torch.cuda.empty_cache(); gc.collect()
    return avg
