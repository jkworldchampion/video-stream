#!/usr/bin/env python
import argparse
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import random

# =======================
# DATA & MODEL IMPORTS
# =======================
from data.dataLoader import KITTIVideoDataset, get_data_list
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher


# =======================
# UTILITIES
# =======================

def reshape_qkv_BTND(q: torch.Tensor, T_use: int):
    """
    q: [B,A,L,d], L = T_use * N
    -> [B,A,T_use,N,d]
    """
    B, A, L, d = q.shape
    assert L % T_use == 0
    N = L // T_use
    return q.view(B, A, T_use, N, d)


def cosine_l2_same_frame(q1, q2, frame_idx: int):
    """
    q1, q2: [B,A,T,N,d]
    """
    q1_t = q1[:, :, frame_idx]  # [B,A,N,d]
    q2_t = q2[:, :, frame_idx]

    cos = F.cosine_similarity(q1_t, q2_t, dim=-1).mean().item()
    l2  = ((q1_t - q2_t).pow(2).sum(dim=-1).sqrt()).mean().item()
    return cos, l2


def build_sliding_windows(T_full: int, window_size: int, step: int):
    """
    예: T_full=48, window_size=32, step=5
    → [(0,32), (5,37), (10,42), ...]
    """
    return [(s, s + window_size)
            for s in range(0, T_full - window_size + 1, step)]


def teacher_forward_prefix(teacher: VideoDepthTeacher, x_window: torch.Tensor, layer_idx: int):
    """
    x_window: [B,T_use,3,H,W]
    """
    with torch.no_grad():
        teacher.enable_qkv_save(True)
        _ = teacher(x_window)
        q, k, v = teacher.collect_qkv(layer_idx)
        teacher.enable_qkv_save(False)
    return q.cpu(), k.cpu(), v.cpu()


# =======================
# MAIN EVAL (한 layer용)
# =======================

def analyze_teacher_sliding_stats_for_layer(
    teacher: VideoDepthTeacher,
    loader: DataLoader,
    target_layer: int,
    physical_frames: List[int],
    window_size: int,
    step: int,
    num_samples: int,
    device,
    verbose_per_pair: bool = False,
):
    """
    한 개 layer에 대해 여러 sample(T 영상)에서 sliding window shift에 대한
    Q/K/V invariance 통계를 계산.
    """
    stats = {
        "cos_q": [], "cos_k": [], "cos_v": [],
        "l2_q": [], "l2_k": [], "l2_v": [],
    }

    for sample_idx, (x, _) in enumerate(loader):
        if sample_idx >= num_samples:
            break

        x = x.to(device)  # [1,T,3,H,W]
        B, T, _, H, W = x.shape

        if window_size > T:
            continue

        windows = build_sliding_windows(T, window_size, step)

        # 1) 각 window에서 Q/K/V 수집
        q_list, k_list, v_list = [], [], []
        for (s, e) in windows:
            q, k, v = teacher_forward_prefix(teacher, x[:, s:e], target_layer)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        # 2) [B,A,T_use,N,d]로 reshape
        q_ctx = [reshape_qkv_BTND(q, window_size) for q in q_list]
        k_ctx = [reshape_qkv_BTND(k, window_size) for k in k_list]
        v_ctx = [reshape_qkv_BTND(v, window_size) for v in v_list]

        # 3) 같은 physical frame에 대해 window pair 비교
        for t_phys in physical_frames:
            if not (0 <= t_phys < T):
                continue

            per_window_indices = []
            for (s, e) in windows:
                if s <= t_phys < e:
                    per_window_indices.append(t_phys - s)
                else:
                    per_window_indices.append(None)

            pairs = []
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    if per_window_indices[i] is not None and per_window_indices[j] is not None:
                        pairs.append((i, j))

            for (i, j) in pairs:
                idx_i = per_window_indices[i]
                idx_j = per_window_indices[j]

                cos_q, l2_q = cosine_l2_same_frame(q_ctx[i], q_ctx[j], idx_i)
                cos_k, l2_k = cosine_l2_same_frame(k_ctx[i], k_ctx[j], idx_i)
                cos_v, l2_v = cosine_l2_same_frame(v_ctx[i], v_ctx[j], idx_i)

                stats["cos_q"].append(cos_q)
                stats["cos_k"].append(cos_k)
                stats["cos_v"].append(cos_v)
                stats["l2_q"].append(l2_q)
                stats["l2_k"].append(l2_k)
                stats["l2_v"].append(l2_v)

                if verbose_per_pair:
                    print(
                        f"[layer {target_layer}] sample={sample_idx}, t={t_phys}, "
                        f"win{windows[i]} vs {windows[j]} → "
                        f"Qcos={cos_q:.4f}, Kcos={cos_k:.4f}, Vcos={cos_v:.4f}"
                    )

    # numpy 통계 계산
    summary = {}
    for key, arr_list in stats.items():
        if len(arr_list) == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            arr = np.array(arr_list, dtype=np.float32)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "n": int(arr.shape[0]),
            }

    return summary


# =======================
# BUILD KITTI LOADER
# =======================

def build_kitti_loader(kitti_path: str, clip_len: int):
    rgb_clips, depth_clips = get_data_list(
        root_dir=kitti_path,
        data_name="kitti",
        split="train",
        clip_len=clip_len,
    )
    dataset = KITTIVideoDataset(
        rgb_paths=rgb_clips,
        depth_paths=depth_clips,
        resize_size=518,
        split="train",
        clip_len=clip_len,
    )
    # shuffle=False → 순서 고정
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader


# =======================
# ARGS & MAIN
# =======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze teacher Q/K/V context invariance (sliding windows) over multiple layers."
    )

    parser.add_argument("--ckpt", type=str,
                        default="/home/work/juhwan/monocular_depth/stream/video-stream/checkpoints/video_depth_anything_vits.pth")
    parser.add_argument("--kitti-path", type=str,
                        default="/home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI")

    parser.add_argument("--clip-len", type=int, default=48)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--step", type=int, default=5)

    parser.add_argument("--phys-frames", type=int, nargs="+",
                        default=[10, 11, 12])

    parser.add_argument("--num-samples", type=int, default=20,
                        help="number of clips to use from KITTI")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--verbose-per-pair", action="store_true",
                        help="print per sample/pair logs (default: off)")

    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------
    # Seed 설정 (재현성)
    # -----------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    print(f"[Info] Using ckpt: {args.ckpt}")
    print(f"[Info] Using KITTI path: {args.kitti_path}")
    print(f"[Info] clip_len={args.clip_len}, window_size={args.window_size}, step={args.step}")
    print(f"[Info] num_samples={args.num_samples}, seed={args.seed}")

    # Teacher 모델 로드
    teacher = VideoDepthTeacher(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe="ape",
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    teacher.load_state_dict(state, strict=True)
    teacher.to(device)
    teacher.eval()

    # layer 0~3 전체 루프
    all_layer_summaries = {}

    for layer_idx in range(4):
        print(f"\n================ LAYER {layer_idx} ================")

        # 매 layer마다 loader 새로 생성 (shuffle=False라 순서는 동일)
        loader = build_kitti_loader(args.kitti_path, args.clip_len)

        summary = analyze_teacher_sliding_stats_for_layer(
            teacher=teacher,
            loader=loader,
            target_layer=layer_idx,
            physical_frames=args.phys_frames,
            window_size=args.window_size,
            step=args.step,
            num_samples=args.num_samples,
            device=device,
            verbose_per_pair=args.verbose_per_pair,
        )

        all_layer_summaries[layer_idx] = summary

        # layer별 요약 출력
        print(f"\n------ Layer {layer_idx} summary ------")
        for key, val in summary.items():
            print(f"{key:>5}: mean={val['mean']:.4f}, std={val['std']:.44f}, n={val['n']}")

    # 마지막에 전체 layer 비교를 한 번에 보고 싶으면 여기서 정리 출력
    print("\n\n====== Layer-wise Comparison (Teacher) ======\n")
    for layer_idx in range(4):
        summary = all_layer_summaries[layer_idx]
        print(f"[Layer {layer_idx}]")
        print(f"  cos_q: mean={summary['cos_q']['mean']:.4f}, std={summary['cos_q']['std']:.4f}")
        print(f"  cos_k: mean={summary['cos_k']['mean']:.4f}, std={summary['cos_k']['std']:.4f}")
        print(f"  cos_v: mean={summary['cos_v']['mean']:.4f}, std={summary['cos_v']['std']:.4f}")
        print(f"  l2_q : mean={summary['l2_q']['mean']:.4f}, std={summary['l2_q']['std']:.4f}")
        print(f"  l2_k : mean={summary['l2_k']['mean']:.4f}, std={summary['l2_k']['std']:.4f}")
        print(f"  l2_v : mean={summary['l2_v']['mean']:.4f}, std={summary['l2_v']['std']:.4f}")
        print("")


if __name__ == "__main__":
    main()
