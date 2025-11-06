#!/usr/bin/env python3
import argparse, json, os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from scipy import ndimage as ndi


# --------------------------- Dataset presets ---------------------------
@dataclass
class DatasetDefaults:
    min_depth_eval: float
    max_depth_eval: float
    max_eval_len: int
    crop: Tuple[int, int, int, int]  # r0, r1, c0, c1

def _dataset_defaults(tag: str) -> DatasetDefaults:
    if tag in ("scannet", "scannet_500"):
        return DatasetDefaults(0.1, 10.0, 500 if tag == "scannet_500" else 90, (8, -8, 11, -11))
    if tag in ("kitti", "kitti_500"):
        return DatasetDefaults(0.1, 80.0, 500 if tag == "kitti_500" else 110, (0, 374, 0, 1242))
    if tag in ("nyuv2", "nyuv2_500"):
        return DatasetDefaults(0.1, 10.0, 500 if tag == "nyuv2_500" else 80, (45, 471, 41, 601))
    if tag == "sintel":
        return DatasetDefaults(0.1, 70.0, 100, (0, 436, 0, 1024))
    if tag == "bonn":
        return DatasetDefaults(0.1, 10.0, 110, (0, 480, 0, 640))
    return DatasetDefaults(0.1, 10.0, 90, (0, -1, 0, -1))


# --------------------------- IO helpers ---------------------------
def _resolve_pred_path(pred_root: str, dataset_key: str, image_rel_path: str) -> Tuple[str, bool]:
    rel_npy = image_rel_path.replace(".png", ".npy").replace(".jpg", ".npy").replace(".jpeg", ".npy")
    candidates = [os.path.join(pred_root, rel_npy), os.path.join(pred_root, dataset_key, rel_npy)]
    for p in candidates:
        if os.path.exists(p):
            return p, True
    return candidates[0], False

def _load_pred_depth(path: str, target_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    dep = np.load(path).astype(np.float32)
    if target_hw is not None and (dep.shape[0] != target_hw[0] or dep.shape[1] != target_hw[1]):
        dep = cv2.resize(dep, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return dep

def _load_gt_depth(path: str, factor: float) -> np.ndarray:
    if path.endswith(".npy"):
        dep = np.load(path).astype(np.float32)
    else:
        dep = cv2.imread(path, -1).astype(np.float32)
    dep = dep / float(factor)
    dep[dep <= 0] = -1.0
    return dep

def _load_sequence_registry(json_path: str, dataset_key: str) -> List[Tuple[str, List[Dict]]]:
    with open(json_path, "r") as f:
        payload = json.load(f)
    registry = []
    for entry in payload[dataset_key]:
        for k, frames in entry.items():
            registry.append((k, frames))
    return registry


# --------------------------- Scale/Shift solve ---------------------------
def _solve_scale_shift(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray, min_valid: int) -> Tuple[float, float]:
    valid = valid_mask & np.isfinite(pred) & np.isfinite(gt)
    if int(valid.sum()) < max(min_valid, 2):
        return np.nan, np.nan
    p = pred[valid].reshape(-1, 1).astype(np.float64)
    g = gt[valid].reshape(-1, 1).astype(np.float64)
    A = np.concatenate([p, np.ones_like(p)], axis=1)
    sol, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(sol[0, 0]), float(sol[1, 0])


# --------------------------- Edge F1 with tolerance ---------------------------
def _normalize01(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = x[mask]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(v, [1.0, 99.0])
    if hi <= lo:
        lo, hi = v.min(), v.max()
    y = (x - lo) / (hi - lo + 1e-8)
    y = np.clip(y, 0, 1)
    y[~mask] = 0
    return y.astype(np.float32)

def _canny_edges(depth01: np.ndarray, low_perc=30, high_perc=70, blur_ks=3) -> np.ndarray:
    x = depth01
    if blur_ks > 0:
        x = cv2.GaussianBlur(x, (blur_ks, blur_ks), 0)
    lo = np.percentile(x, low_perc)*255.0
    hi = np.percentile(x, high_perc)*255.0
    edges = cv2.Canny((x*255.0).astype(np.uint8), threshold1=lo, threshold2=hi, L2gradient=True)
    return edges > 0

def _edge_f1(pred_depth: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray, tol_px: int=1) -> Tuple[float,float,float]:
    # normalize both to [0,1] on valid region
    p01 = _normalize01(pred_depth, valid)
    g01 = _normalize01(gt_depth,   valid)

    e_p = _canny_edges(p01)
    e_g = _canny_edges(g01)

    # tolerance via dilation
    selem = np.ones((2*tol_px+1, 2*tol_px+1), dtype=np.uint8)
    e_g_dil = cv2.dilate(e_g.astype(np.uint8), selem) > 0
    e_p_dil = cv2.dilate(e_p.astype(np.uint8), selem) > 0

    tp = np.logical_and(e_p, e_g_dil).sum()
    fp = np.logical_and(e_p, np.logical_not(e_g_dil)).sum()
    fn = np.logical_and(e_g, np.logical_not(e_p_dil)).sum()

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2*prec*rec / (prec + rec + 1e-8)
    return float(prec), float(rec), float(f1)


# --------------------------- High-frequency (HF) metrics ---------------------------
def _laplacian_var(img: np.ndarray, mask: np.ndarray) -> float:
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    v = lap[mask]
    return float(np.var(v)) if v.size else float("nan")

def _grad_energy(img: np.ndarray, mask: np.ndarray) -> Tuple[float,float]:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    v = mag[mask]
    mean = float(np.mean(v)) if v.size else float("nan")
    p95  = float(np.percentile(v, 95)) if v.size else float("nan")
    return mean, p95

def _fft_hf_ratio(img: np.ndarray, mask: np.ndarray, cutoff: float=0.25) -> float:
    # zero out invalids to avoid ringing
    x = img.copy()
    x[~mask] = 0.0
    H, W = x.shape
    F = np.fft.fftshift(np.fft.fft2(x))
    amp = np.abs(F)

    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    rmax = np.sqrt((cy)**2 + (cx)**2)
    hi_mask = rr > (cutoff * rmax)

    num = amp[hi_mask].sum()
    den = amp.sum() + 1e-8
    return float(num/den)


# --------------------------- Main per-sequence pass ---------------------------
def _process_sequence(sequence_id: str,
                      frames: List[Dict], pred_root: str, dataset_key: str,
                      samples_root: str, defaults: DatasetDefaults,
                      min_valid_pixels: int, max_frames: Optional[int]) -> Dict:

    r0, r1, c0, c1 = defaults.crop
    per_frame = []
    limit = min(len(frames), defaults.max_eval_len)
    if max_frames is not None:
        limit = min(limit, max_frames)

    for idx, fr in enumerate(frames[:limit]):
        pred_path, ok = _resolve_pred_path(pred_root, dataset_key, fr["image"])
        gt_path = os.path.join(samples_root, fr["gt_depth"])

        available = ok and os.path.exists(gt_path)
        if not available:
            per_frame.append({"frame_index": idx, "available": False})
            continue

        gt = _load_gt_depth(gt_path, fr["factor"])
        gt = gt[r0:r1, c0:c1]
        pred = _load_pred_depth(pred_path, target_hw=gt.shape)

        valid = (gt > defaults.min_depth_eval) & (gt < defaults.max_depth_eval) & np.isfinite(pred)
        if int(valid.sum()) < min_valid_pixels:
            per_frame.append({"frame_index": idx, "available": False, "valid_pixel_count": int(valid.sum())})
            continue

        # align scale/shift
        s, t = _solve_scale_shift(pred, gt, valid, min_valid_pixels)
        pred_aligned = s*pred + t

        # Edge F1 (tolerance=1px)
        prec, rec, f1 = _edge_f1(pred_aligned, gt, valid, tol_px=1)

        # HF metrics on normalized [0,1] maps
        p01 = _normalize01(pred_aligned, valid)
        g01 = _normalize01(gt, valid)

        lap_var_p = _laplacian_var(p01, valid)
        lap_var_g = _laplacian_var(g01, valid)
        grad_mean_p, grad_p95_p = _grad_energy(p01, valid)
        grad_mean_g, grad_p95_g = _grad_energy(g01, valid)
        fft_ratio_p = _fft_hf_ratio(p01, valid, cutoff=0.25)
        fft_ratio_g = _fft_hf_ratio(g01, valid, cutoff=0.25)

        per_frame.append({
            "frame_index": idx,
            "available": True,
            "valid_pixel_count": int(valid.sum()),
            "edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
            "lap_var_pred": lap_var_p, "lap_var_gt": lap_var_g,
            "grad_mean_pred": grad_mean_p, "grad_p95_pred": grad_p95_p,
            "grad_mean_gt": grad_mean_g, "grad_p95_gt": grad_p95_g,
            "fft_hf_ratio_pred": fft_ratio_p, "fft_hf_ratio_gt": fft_ratio_g,
        })

    # aggregate
    def _nanmean(x): 
        x = np.array([v for v in x if v is not None], dtype=np.float64)
        return float(np.nanmean(x)) if x.size else float("nan")

    edge_f1 = [f.get("edge_f1") for f in per_frame if f.get("available")]
    lap_var = [f.get("lap_var_pred") for f in per_frame if f.get("available")]
    grad_m  = [f.get("grad_mean_pred") for f in per_frame if f.get("available")]
    grad95  = [f.get("grad_p95_pred") for f in per_frame if f.get("available")]
    fft_hf  = [f.get("fft_hf_ratio_pred") for f in per_frame if f.get("available")]

    return {
        "sequence_id": sequence_id,
        "num_frames": len([f for f in per_frame if f.get("available")]),
        "edge_f1_mean": _nanmean(edge_f1),
        "lap_var_mean": _nanmean(lap_var),
        "grad_mean": _nanmean(grad_m),
        "grad_p95": _nanmean(grad95),
        "fft_hf_ratio_mean": _nanmean(fft_hf),
        "frames": per_frame
    }


# --------------------------- CLI ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Edge-F1 & High-frequency metrics for depth maps")
    p.add_argument("--pred-root", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--dataset-key", default="scannet")
    p.add_argument("--dataset-eval-tag", default="scannet_500")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-valid-pixels", type=int, default=50)
    p.add_argument("--prefix", default="")
    p.add_argument("--output", default=None)
    return p.parse_args()

def _sanitize(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

def main():
    args = parse_args()
    defaults = _dataset_defaults(args.dataset_eval_tag)
    registry = _load_sequence_registry(args.json, args.dataset_key)
    samples_root = os.path.dirname(args.json)

    seq_reports = []
    for sid, frames in tqdm(registry, desc=f"Profiling ({args.prefix})"):
        seq_reports.append(_process_sequence(
            sequence_id=sid, frames=frames,
            pred_root=args.pred_root, dataset_key=args.dataset_key,
            samples_root=samples_root, defaults=defaults,
            min_valid_pixels=args.min_valid_pixels, max_frames=args.max_frames
        ))

    # summary
    def _collect(key):
        vals = [s[key] for s in seq_reports if s["num_frames"] > 0]
        vals = np.array(vals, dtype=np.float64)
        return float(np.nanmean(vals)) if vals.size else float("nan")

    summary = {
        "label": args.prefix,
        "num_sequences": len([s for s in seq_reports if s["num_frames"]>0]),
        "edge_f1_mean": _collect("edge_f1_mean"),
        "lap_var_mean": _collect("lap_var_mean"),
        "grad_mean": _collect("grad_mean"),
        "grad_p95": _collect("grad_p95"),
        "fft_hf_ratio_mean": _collect("fft_hf_ratio_mean"),
    }

    out = {"summary": summary, "sequences": seq_reports}
    out = _sanitize(out)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps({"summary": summary}, indent=2))

if __name__ == "__main__":
    main()
