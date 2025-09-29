#!/usr/bin/env python
"""Quick sanity check for a trained student model using
`utils.train_helper.validate_with_infer_eval_subset_fast`.

Example:
    python tools/run_fast_validation.py \
        --checkpoint outputs/experiment_33/best_model.pth \
        --val-json /workspace/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json
"""
import argparse
import json
import os
from typing import Iterable, Optional

import torch

from utils.train_helper import validate_with_infer_eval_subset_fast
from video_depth_anything.video_depth_stream import VideoDepthAnything


def _parse_scene_indices(raw: Optional[str]) -> Optional[Iterable[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [int(tok) for tok in raw.replace(";", ",").split(",") if tok]


def _load_student_model(args: argparse.Namespace, device: torch.device) -> VideoDepthAnything:
    model_cfg = {
        "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
        "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    if args.encoder not in model_cfg:
        raise ValueError(f"Unsupported encoder '{args.encoder}'")

    cfg = model_cfg[args.encoder]
    model = VideoDepthAnything(
        encoder=args.encoder,
        features=cfg["features"],
        out_channels=cfg["out_channels"],
        num_frames=args.clip_len,
        stream_mode=True,
        select_top_r=None,
        update_top_u=None,
        rope_dt=None,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    cleaned_state = {}
    for key, value in state.items():
        nk = key
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("student."):
            nk = nk[len("student."):]
        cleaned_state[nk] = value
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model.to(device).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast validation on a trained student model.")
    parser.add_argument("--checkpoint", required=True, help="Path to student checkpoint (best_model.pth)")
    parser.add_argument("--val-json", required=True, help="Validation JSON metadata (e.g. scannet_video_500.json)")
    parser.add_argument("--temp-dir", default="benchmark/output/fast_val_tmp", help="Temporary inference dir (unused but required by API)")
    parser.add_argument("--dataset-key", default="scannet", help="Dataset key inside JSON")
    parser.add_argument("--dataset-tag", default="scannet_500", help="Preset name for metrics")
    parser.add_argument("--scene-count", type=int, default=100, help="How many scenes to evaluate")
    parser.add_argument("--scene-indices", default=None, help="Comma-separated list of specific scene indices")
    parser.add_argument("--frame-stride", type=int, default=2, help="Temporal subsampling stride")
    parser.add_argument("--max-eval-len", type=int, default=500, help="Max frames per scene (None to use preset)")
    parser.add_argument("--input-size", type=int, default=518, help="Student inference resolution")
    parser.add_argument("--align-mode", choices=["disp_global", "depth_per_frame_robust"], default="depth_per_frame_robust")
    parser.add_argument("--trim", type=float, nargs=2, default=(2.0, 98.0), help="Percentile trim (robust mode)")
    parser.add_argument("--min-pts", type=int, default=500, help="Minimum valid pixels per frame (robust mode)")
    parser.add_argument("--encoder", choices=["vits", "vitl"], default="vits")
    parser.add_argument("--clip-len", type=int, default=32, help="Temporal window length used during training")
    parser.add_argument("--fp32", action="store_true", help="Disable AMP during inference")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")

    model = _load_student_model(args, device)

    # ensure temp dir exists (even if helper does not use it, API expects it)
    os.makedirs(args.temp_dir, exist_ok=True)

    scene_indices = _parse_scene_indices(args.scene_indices)
    if scene_indices is not None:
        print(f"▶ Evaluating specific scenes: {scene_indices}")

    device_str = "cuda" if device.type == "cuda" else "cpu"

    results = validate_with_infer_eval_subset_fast(
        model=model,
        json_file=args.val_json,
        infer_path=args.temp_dir,
        dataset=args.dataset_key,
        dataset_eval_tag=args.dataset_tag,
        device=device_str,
        input_size=args.input_size,
        scenes_to_eval=args.scene_count,
        scene_indices=scene_indices,
        frame_stride=args.frame_stride,
        max_eval_len=args.max_eval_len,
        fp32=args.fp32,
        align_mode=args.align_mode,
        trim=tuple(args.trim),
        min_pts=args.min_pts,
    )

    print("\n=== Validation Results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
