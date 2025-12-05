#!/usr/bin/env python
import argparse
import math
import random
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from data.dataLoader import KITTIVideoDataset, get_data_list
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent


# =======================
# QKV UTILITIES
# =======================

def cosine_l2_full(t1: torch.Tensor, t2: torch.Tensor) -> Tuple[float, float]:
    """
    t1, t2: [B, A, L, d]  (동일 shape 가정)
    """
    cos = F.cosine_similarity(t1, t2, dim=-1).mean().item()
    l2  = ((t1 - t2).pow(2).sum(dim=-1).sqrt()).mean().item()
    return cos, l2


def build_sliding_windows(T_full: int, window_size: int, step: int):
    """
    예: T_full=48, window_size=32, step=5
    → [(0,32), (5,37), (10,42), (15,47)]
    """
    windows = []
    s = 0
    while s + window_size <= T_full:
        windows.append((s, s + window_size))
        s += step
    return windows


# =======================
# DATA LOADER
# =======================

def build_kitti_loader(kitti_path: str, clip_len: int):
    rgb_paths, depth_paths = get_data_list(
        root_dir=kitti_path,
        data_name="kitti",
        split="train",
        clip_len=clip_len,
    )
    ds = KITTIVideoDataset(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        resize_size=518,
        split="train",
        clip_len=clip_len,
    )
    return DataLoader(ds, batch_size=1, shuffle=False)


# =======================
# MAIN ANALYZER (STREAM)
# =======================

def analyze_student_stream_layer(
    student: VideoDepthStudent,
    loader: DataLoader,
    layer_idx: int,
    physical_frames: List[int],
    window_size: int,
    step: int,
    num_samples: int,
    device,
    verbose: bool = False,
):
    """
    streaming + KV-cache 환경에서,
    같은 physical frame t가 prefix가 다른 여러 stream run(window)에서
    어떤 Q/K/V를 갖는지 비교하는 함수.
    """

    stats = {k: [] for k in ["cos_q","cos_k","cos_v","l2_q","l2_k","l2_v"]}

    for sample_idx, (x, _) in enumerate(loader):
        if sample_idx >= num_samples:
            break

        x = x.to(device)  # [1, T, 3, H, W]
        B, T, C, H, W = x.shape

        if window_size > T:
            if verbose:
                print(f"[Warn] clip_len({T}) < window_size({window_size}), skip sample {sample_idx}")
            continue

        windows = build_sliding_windows(T_full=T, window_size=window_size, step=step)
        if verbose:
            print(f"\n[Sample {sample_idx}] T={T}, windows={windows}")

        # (window_idx, t_phys) → (q_cur, k_cur, v_cur)
        q_store: Dict[Tuple[int,int], torch.Tensor] = {}
        k_store: Dict[Tuple[int,int], torch.Tensor] = {}
        v_store: Dict[Tuple[int,int], torch.Tensor] = {}

        # --------- 여러 window를 "별도의 stream run"으로 돌린다 ---------
        for w_idx, (s, e) in enumerate(windows):
            cache_state = None  # 매 window 마다 cache 초기화

            for t in range(s, e):
                x_t = x[:, t:t+1]  # [1,1,3,H,W]

                # stream + QKV 추출
                _, cache_state, (q, k, v) = student.stream_step_train_with_qkv(
                    x_t, cache_state, layer_idx=layer_idx
                )

                # Q/K/V가 없으면 skip
                if q is None or k is None or v is None:
                    continue

                # q: [B, A, Lq, d]  (Lq = N_patch, 현재 frame만)
                Bq, Aq, Lq, dq = q.shape

                # k,v: [B, A, Lkv, d], 뒤쪽 Lq 토큰이 "현재 frame"
                Bk, Ak, Lk, dk = k.shape
                assert Bk == Bq and Ak == Aq and dk == dq

                if Lk < Lq:
                    # 이론상 발생하면 안 되지만, 방어적으로
                    if verbose:
                        print(f"[Warn] Lk({Lk}) < Lq({Lq}) at sample={sample_idx}, t={t}, window={w_idx}")
                    continue

                k_cur = k[:, :, -Lq:, :].detach().cpu()
                v_cur = v[:, :, -Lq:, :].detach().cpu()
                q_cur = q.detach().cpu()

                # store
                q_store[(w_idx, t)] = q_cur
                k_store[(w_idx, t)] = k_cur
                v_store[(w_idx, t)] = v_cur

        # --------- 같은 physical frame t_phys에 대해 window 간 pair 비교 ---------
        for t_phys in physical_frames:
            if not (0 <= t_phys < T):
                continue

            # 이 frame을 포함하는 window index들
            windows_with_t = [
                w_idx for w_idx, (s, e) in enumerate(windows)
                if (w_idx, t_phys) in q_store
            ]

            if len(windows_with_t) < 2:
                # pair가 안 생김
                continue

            if verbose:
                print(f"[Layer {layer_idx}] sample={sample_idx}, t={t_phys}, windows_with_t={windows_with_t}")

            # pair-wise 비교
            for i in range(len(windows_with_t)):
                for j in range(i+1, len(windows_with_t)):
                    wi = windows_with_t[i]
                    wj = windows_with_t[j]

                    q1 = q_store[(wi, t_phys)]
                    q2 = q_store[(wj, t_phys)]
                    k1 = k_store[(wi, t_phys)]
                    k2 = k_store[(wj, t_phys)]
                    v1 = v_store[(wi, t_phys)]
                    v2 = v_store[(wj, t_phys)]

                    # shape check
                    assert q1.shape == q2.shape
                    assert k1.shape == k2.shape
                    assert v1.shape == v2.shape

                    cos_q, l2_q = cosine_l2_full(q1, q2)
                    cos_k, l2_k = cosine_l2_full(k1, k2)
                    cos_v, l2_v = cosine_l2_full(v1, v2)

                    stats["cos_q"].append(cos_q)
                    stats["cos_k"].append(cos_k)
                    stats["cos_v"].append(cos_v)
                    stats["l2_q"].append(l2_q)
                    stats["l2_k"].append(l2_k)
                    stats["l2_v"].append(l2_v)

                    if verbose:
                        print(f"  win {wi} vs {wj} @ t={t_phys} → "
                              f"Qcos={cos_q:.4f}, Kcos={cos_k:.4f}, Vcos={cos_v:.4f}")

    # --------- 통계 정리 ---------
    summary = {}
    for k, arr in stats.items():
        if len(arr) == 0:
            summary[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            a = np.array(arr)
            summary[k] = {
                "mean": float(a.mean()),
                "std":  float(a.std()),
                "n":    int(a.shape[0]),
            }

    return summary


# =======================
# ARGPARSE & MAIN
# =======================

def parse_args():
    p = argparse.ArgumentParser("STREAM Student QKV drift analyzer")
    p.add_argument("--ckpt", type=str,
                   default="/home/work/juhwan/monocular_depth/stream/video-stream/checkpoints/video_depth_anything_vits.pth")
    p.add_argument("--kitti_path", type=str,
                   default="/home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI")
    p.add_argument("--clip-len", type=int, default=48)
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--phys-frames", nargs="+", type=int, default=[10, 11, 12])
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    print(f"[Info] ckpt={args.ckpt}")
    print(f"[Info] kitti={args.kitti_path}")
    print(f"[Info] num_samples={args.num_samples}, seed={args.seed}")

    # Model
    student = VideoDepthStudent(
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
    student.load_state_dict(state, strict=True)
    student.to(device)
    student.eval()

    # Data
    loader = build_kitti_loader(args.kitti_path, args.clip_len)

    all_layers = {}

    for layer_idx in range(4):
        print(f"\n=== STREAM Layer {layer_idx} ===")
        summary = analyze_student_stream_layer(
            student=student,
            loader=loader,
            layer_idx=layer_idx,
            physical_frames=args.phys_frames,
            window_size=args.window_size,
            step=args.step,
            num_samples=args.num_samples,
            device=device,
            verbose=args.verbose,
        )
        all_layers[layer_idx] = summary
        print(summary)

    print("\n===== FINAL STREAM LAYER-WISE COMPARISON =====")
    for layer_idx, summary in all_layers.items():
        print(f"\n[Layer {layer_idx}]")
        print(f"  cos_q: mean={summary['cos_q']['mean']:.4f}, std={summary['cos_q']['std']:.4f}, n={summary['cos_q']['n']}")
        print(f"  cos_k: mean={summary['cos_k']['mean']:.4f}, std={summary['cos_k']['std']:.4f}, n={summary['cos_k']['n']}")
        print(f"  cos_v: mean={summary['cos_v']['mean']:.4f}, std={summary['cos_v']['std']:.4f}, n={summary['cos_v']['n']}")
        print(f"  l2_q:  mean={summary['l2_q']['mean']:.4f}, std={summary['l2_q']['std']:.4f}, n={summary['l2_q']['n']}")
        print(f"  l2_k:  mean={summary['l2_k']['mean']:.4f}, std={summary['l2_k']['std']:.4f}, n={summary['l2_k']['n']}")
        print(f"  l2_v:  mean={summary['l2_v']['mean']:.4f}, std={summary['l2_v']['std']:.4f}, n={summary['l2_v']['n']}")


if __name__ == "__main__":
    main()
