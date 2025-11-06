#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class DatasetDefaults:
    min_depth_eval: float
    max_depth_eval: float
    max_eval_len: int
    crop: Tuple[int, int, int, int]


def _dataset_defaults(tag: str) -> DatasetDefaults:
    if tag in ("scannet", "scannet_500"):
        return DatasetDefaults(
            min_depth_eval=0.1,
            max_depth_eval=10.0,
            max_eval_len=500 if tag == "scannet_500" else 90,
            crop=(8, -8, 11, -11),
        )
    if tag in ("kitti", "kitti_500"):
        return DatasetDefaults(
            min_depth_eval=0.1,
            max_depth_eval=80.0,
            max_eval_len=500 if tag == "kitti_500" else 110,
            crop=(0, 374, 0, 1242),
        )
    if tag in ("nyuv2", "nyuv2_500"):
        return DatasetDefaults(
            min_depth_eval=0.1,
            max_depth_eval=10.0,
            max_eval_len=500 if tag == "nyuv2_500" else 80,
            crop=(45, 471, 41, 601),
        )
    if tag == "sintel":
        return DatasetDefaults(
            min_depth_eval=0.1,
            max_depth_eval=70.0,
            max_eval_len=100,
            crop=(0, 436, 0, 1024),
        )
    if tag == "bonn":
        return DatasetDefaults(
            min_depth_eval=0.1,
            max_depth_eval=10.0,
            max_eval_len=110,
            crop=(0, 480, 0, 640),
        )
    return DatasetDefaults(
        min_depth_eval=0.1,
        max_depth_eval=10.0,
        max_eval_len=90,
        crop=(0, -1, 0, -1),
    )


def _load_pred_depth(path: str, target_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if target_hw is not None:
        h, w = target_hw
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h))
    return depth


def _load_gt_depth(path: str, factor: float) -> np.ndarray:
    if path.endswith(".npy"):
        depth = np.load(path).astype(np.float32)
    else:
        depth = cv2.imread(path, -1)
        depth = np.array(depth, dtype=np.float32)
    depth = depth / float(factor)
    depth[depth <= 0] = -1.0
    return depth


def _solve_scale_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    min_valid: int,
) -> Tuple[float, float, float]:
    valid = valid_mask & np.isfinite(pred) & np.isfinite(gt)
    count = int(valid.sum())
    if count < max(min_valid, 2):
        return float("nan"), float("nan"), float("nan")

    pred_vec = pred[valid].reshape((-1, 1)).astype(np.float64)
    gt_vec = gt[valid].reshape((-1, 1)).astype(np.float64)
    ones = np.ones_like(pred_vec)
    A = np.concatenate([pred_vec, ones], axis=1)
    try:
        solution, *_ = np.linalg.lstsq(A, gt_vec, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")

    scale = float(solution[0, 0])
    shift = float(solution[1, 0])
    aligned = scale * pred + shift
    residual = gt - aligned
    rmse = float(np.sqrt(np.mean((residual[valid]) ** 2)))
    return scale, shift, rmse


def _prefix_nanstd(values: Sequence[float]) -> List[float]:
    prefix = []
    for i in range(1, len(values) + 1):
        window = np.array(values[:i], dtype=np.float64)
        prefix.append(float(np.nanstd(window)))
    return prefix


def _collect_segments(
    config: Optional[Dict[str, List[Dict]]],
    sequence_id: str,
    series: Sequence[float],
) -> List[Dict[str, float]]:
    if not config:
        return []
    segments = []
    entries = config.get(sequence_id, [])
    for entry in entries:
        start = int(entry.get("start", 0))
        end = int(entry.get("end", len(series) - 1))
        name = entry.get("name", f"segment_{start}_{end}")
        start = max(0, start)
        end = min(len(series) - 1, end)
        if end < start or len(series) == 0:
            continue
        window = np.array(series[start : end + 1], dtype=np.float64)
        segments.append(
            {
                "name": name,
                "frame_start": start,
                "frame_end": end,
                "std": float(np.nanstd(window)),
                "mean": float(np.nanmean(window)),
            }
        )
    return segments


def _sanitize_for_json(item):
    if isinstance(item, float):
        if np.isnan(item):
            return None
        return item
    if isinstance(item, list):
        return [_sanitize_for_json(x) for x in item]
    if isinstance(item, dict):
        return {k: _sanitize_for_json(v) for k, v in item.items()}
    return item


def _resolve_pred_path(
    pred_root: str,
    dataset_key: str,
    image_rel_path: str,
) -> Tuple[str, bool]:
    rel_npy = (
        image_rel_path.replace(".png", ".npy")
        .replace(".jpg", ".npy")
        .replace(".jpeg", ".npy")
    )
    candidates = [
        os.path.join(pred_root, rel_npy),
        os.path.join(pred_root, dataset_key, rel_npy),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path, True
    return candidates[0], False


def _process_sequence(
    sequence_id: str,
    frames: List[Dict],
    pred_root: str,
    dataset_key: str,
    samples_root: str,
    defaults: DatasetDefaults,
    min_valid_pixels: int,
    max_frames: Optional[int],
    dynamic_segments: Optional[Dict[str, List[Dict]]],
) -> Dict:
    a, b, c, d = defaults.crop
    scales: List[float] = []
    shifts: List[float] = []
    rmses: List[float] = []
    per_frame: List[Dict] = []
    limit = min(len(frames), defaults.max_eval_len)
    if max_frames is not None:
        limit = min(limit, max_frames)

    for idx, frame in enumerate(frames[:limit]):
        pred_path, available = _resolve_pred_path(
            pred_root=pred_root,
            dataset_key=dataset_key,
            image_rel_path=frame["image"],
        )
        if not available:
            scales.append(float("nan"))
            shifts.append(float("nan"))
            rmses.append(float("nan"))
            per_frame.append(
                {
                    "frame_index": idx,
                    "image_rel_path": frame["image"],
                    "pred_path": pred_path,
                    "gt_path": os.path.join(samples_root, frame["gt_depth"]),
                    "available": False,
                    "valid_pixel_count": 0,
                }
            )
            continue

        gt_path = os.path.join(samples_root, frame["gt_depth"])
        if not os.path.exists(gt_path):
            scales.append(float("nan"))
            shifts.append(float("nan"))
            rmses.append(float("nan"))
            per_frame.append(
                {
                    "frame_index": idx,
                    "image_rel_path": frame["image"],
                    "pred_path": pred_path,
                    "gt_path": gt_path,
                    "available": False,
                    "valid_pixel_count": 0,
                }
            )
            continue

        gt_depth = _load_gt_depth(gt_path, frame["factor"])
        gt_depth = gt_depth[a:b, c:d]

        pred_depth = _load_pred_depth(pred_path, target_hw=gt_depth.shape)

        valid = (gt_depth > defaults.min_depth_eval) & (
            gt_depth < defaults.max_depth_eval
        )

        scale, shift, rmse = _solve_scale_shift(
            pred_depth,
            gt_depth,
            valid_mask=valid,
            min_valid=min_valid_pixels,
        )
        scales.append(scale)
        shifts.append(shift)
        rmses.append(rmse)
        per_frame.append(
            {
                "frame_index": idx,
                "image_rel_path": frame["image"],
                "pred_path": pred_path,
                "gt_path": gt_path,
                "available": True,
                "scale": scale,
                "shift": shift,
                "rmse": rmse,
                "valid_pixel_count": int(valid.sum()),
            }
        )

    scale_prefix_std = _prefix_nanstd(scales) if scales else []
    shift_prefix_std = _prefix_nanstd(shifts) if shifts else []

    result = {
        "sequence_id": sequence_id,
        "num_frames": len(per_frame),
        "scale": {
            "per_frame": scales,
            "prefix_std": scale_prefix_std,
            "global_std": float(np.nanstd(np.array(scales, dtype=np.float64)))
            if scales
            else float("nan"),
            "segments": _collect_segments(dynamic_segments, sequence_id, scales),
        },
        "shift": {
            "per_frame": shifts,
            "prefix_std": shift_prefix_std,
            "global_std": float(np.nanstd(np.array(shifts, dtype=np.float64)))
            if shifts
            else float("nan"),
            "segments": _collect_segments(dynamic_segments, sequence_id, shifts),
        },
        "rmse": {
            "per_frame": rmses,
            "mean": float(np.nanmean(np.array(rmses, dtype=np.float64)))
            if rmses
            else float("nan"),
        },
        "frames": per_frame,
    }
    return result


def _load_sequence_registry(
    json_path: str,
    dataset_key: str,
) -> List[Tuple[str, List[Dict]]]:
    with open(json_path, "r") as handle:
        payload = json.load(handle)
    registry = []
    for idx, entry in enumerate(payload[dataset_key]):
        for key, frames in entry.items():
            registry.append((key, frames))
    return registry


def _load_segment_config(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile per-frame scale/shift drift against GT depth."
    )
    parser.add_argument("--pred-root", required=True, help="Root directory containing *.npy predictions mirroring dataset hierarchy.")
    parser.add_argument("--json", required=True, help="Dataset json listing frames with image/gt_depth entries.")
    parser.add_argument("--dataset-key", default="scannet", help="Top-level key inside json file.")
    parser.add_argument("--dataset-eval-tag", default="scannet_500", help="Preset controlling crop/min/max depth.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap per sequence for faster sweeps.")
    parser.add_argument("--min-valid-pixels", type=int, default=50, help="Minimum pixel count required to solve scale/shift.")
    parser.add_argument("--segments-json", default=None, help="Optional json describing frame segments per sequence for localized stats.")
    parser.add_argument("--prefix", default="", help="Label included in the summary output.")
    parser.add_argument("--output", default=None, help="Optional path to save full report as json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = _dataset_defaults(args.dataset_eval_tag)
    segments = _load_segment_config(args.segments_json)
    registry = _load_sequence_registry(args.json, args.dataset_key)
    samples_root = os.path.dirname(args.json)

    reports = []
    for sequence_id, frames in tqdm(registry, desc="Profiling sequences"):
        report = _process_sequence(
            sequence_id=sequence_id,
            frames=frames,
            pred_root=args.pred_root,
            dataset_key=args.dataset_key,
            samples_root=samples_root,
            defaults=defaults,
            min_valid_pixels=args.min_valid_pixels,
            max_frames=args.max_frames,
            dynamic_segments=segments,
        )
        reports.append(report)

    final_scale_stds = np.array(
        [seq["scale"]["prefix_std"][-1] for seq in reports if seq["scale"]["prefix_std"]],
        dtype=np.float64,
    )
    final_shift_stds = np.array(
        [seq["shift"]["prefix_std"][-1] for seq in reports if seq["shift"]["prefix_std"]],
        dtype=np.float64,
    )

    summary = {
        "label": args.prefix,
        "num_sequences": len(reports),
        "scale_std_mean": float(np.nanmean(final_scale_stds)) if final_scale_stds.size else float("nan"),
        "scale_std_median": float(np.nanmedian(final_scale_stds)) if final_scale_stds.size else float("nan"),
        "shift_std_mean": float(np.nanmean(final_shift_stds)) if final_shift_stds.size else float("nan"),
        "shift_std_median": float(np.nanmedian(final_shift_stds)) if final_shift_stds.size else float("nan"),
    }

    sanitized_summary = _sanitize_for_json(summary)
    sanitized_reports = _sanitize_for_json(reports)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(
                {
                    "summary": sanitized_summary,
                    "sequences": sanitized_reports,
                },
                handle,
                indent=2,
            )
    else:
        print(json.dumps({"summary": sanitized_summary}, indent=2))


if __name__ == "__main__":
    main()
