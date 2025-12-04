#!/usr/bin/env python
import argparse
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =======================
# DATA & MODEL IMPORTS
# =======================
from data.dataLoader import KITTIVideoDataset, get_data_list
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher


# =======================
# QKV UTILITIES
# =======================

def reshape_qkv_BTND(q: torch.Tensor, T_use: int):
    """
    q: [B,A,L,d], L = T_use * N
    -> [B,A,T_use,N,d]
    """
    B, A, L, d = q.shape
    assert L % T_use == 0, f"L({L}) % T_use({T_use}) != 0"
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


def last_query_attention(q, k, T_use: int):
    """
    q,k : [B,A,L,d], L=T_use*N
    return: [B,A,T_use] frame-wise attention weight
    """
    B, A, L, d = q.shape
    assert L % T_use == 0, f"L({L}) % T_use({T_use}) != 0"
    N = L // T_use

    q_ = q.view(B, A, T_use, N, d)
    k_ = k.view(B, A, T_use, N, d)

    t = T_use - 1  # 마지막 frame (window local index)

    q_last = q_[:, :, t]               # [B,A,N,d]
    k_all  = k_.reshape(B, A, T_use * N, d)

    scale = 1.0 / math.sqrt(d)

    q_last_exp = q_last.unsqueeze(2)         # [B,A,N,1,d]
    k_all_exp = k_all.unsqueeze(2)           # [B,A,1,TN,d]

    sim = torch.matmul(q_last_exp, k_all_exp.transpose(-1, -2)) * scale
    sim = sim.squeeze(3)  # [B,A,N,TN]

    attn = F.softmax(sim, dim=-1)            # [B,A,N,TN]

    # → frame 차원으로 다시 묶기
    attn_frame = attn.view(B, A, N, T_use, N).sum(-1)  # [B,A,N,T_use]
    attn_frame = attn_frame.mean(2)                    # [B,A,T_use]

    return attn_frame


# =======================
# TEACHER FORWARD (PREFIX)
# =======================

def teacher_forward_prefix(teacher: VideoDepthTeacher, x_window: torch.Tensor, layer_idx: int):
    """
    x_window: [B,T_use,3,H,W]
    """
    with torch.no_grad():
        # 🔹 VideoDepthAnything 쪽에 정의해둔 endpoint 사용
        teacher.enable_qkv_save(True)
        _ = teacher(x_window)
        q, k, v = teacher.collect_qkv(layer_idx)
        teacher.enable_qkv_save(False)
    return q.cpu(), k.cpu(), v.cpu()


# =======================
# SLIDING WINDOW GENERATOR
# =======================

def build_sliding_windows(T_full: int, window_size: int, step: int):
    """
    예: T_full=48, window_size=32, step=5
    반환: [(0,32), (5,37), (10,42), ...]
    """
    windows = []
    s = 0
    while s + window_size <= T_full:
        windows.append((s, s + window_size))
        s += step
    return windows


# =======================
# MAIN ANALYZER
# =======================

def analyze_teacher_sliding(teacher: VideoDepthTeacher,
                            x: torch.Tensor,
                            target_layer: int,
                            physical_frames: List[int],
                            window_size: int,
                            step: int,
                            device):

    x = x.to(device)
    B, T, _, H, W = x.shape

    if window_size > T:
        raise ValueError(f"window_size({window_size}) > T_full({T})")

    print(f"[Info] Loaded sequence B={B}, T_full={T}, H={H}, W={W}")
    print(f"[Info] window_size={window_size}, step={step}")

    # 1) sliding windows
    windows = build_sliding_windows(T_full=T, window_size=window_size, step=step)
    print(f"[Info] Sliding windows: {windows}")

    # 2) 각 window마다 teacher 돌며 Q/K/V 수집
    q_list, k_list, v_list = [], [], []
    for (s, e) in windows:
        x_win = x[:, s:e]    # [B, window_size, 3, H, W]
        q, k, v = teacher_forward_prefix(teacher, x_win, target_layer)
        q_list.append(q)
        k_list.append(k)
        v_list.append(v)

    # 3) [B,A,T_use,N,d]로 reshape
    q_ctx = [reshape_qkv_BTND(q, window_size) for q in q_list]
    k_ctx = [reshape_qkv_BTND(k, window_size) for k in k_list]
    v_ctx = [reshape_qkv_BTND(v, window_size) for v in v_list]

    # 4) 같은 physical frame t_phys에 대해 window 간 Q/K/V 비교
    print("\n====== QKV context invariance (sliding) ======\n")

    for t_phys in physical_frames:
        print(f"--- Physical frame t={t_phys} ---")
        if not (0 <= t_phys < T):
            print(f"  [Warn] t_phys={t_phys} is outside [0,{T-1}], skip.")
            continue

        # 각 window에서 local index 계산
        per_window_indices = []
        for (s, e) in windows:
            if s <= t_phys < e:
                per_window_indices.append(t_phys - s)
            else:
                per_window_indices.append(None)

        # 둘 다 포함하는 윈도우 pair만 대상으로 비교
        valid_pairs = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                if per_window_indices[i] is not None and per_window_indices[j] is not None:
                    valid_pairs.append((i, j))

        if not valid_pairs:
            print("  [Info] No window pair contains this frame in common, skip.")
            continue

        for (i, j) in valid_pairs:
            idx_i = per_window_indices[i]
            idx_j = per_window_indices[j]

            cos_q, l2_q = cosine_l2_same_frame(q_ctx[i], q_ctx[j], frame_idx=idx_i)
            cos_k, l2_k = cosine_l2_same_frame(k_ctx[i], k_ctx[j], frame_idx=idx_i)
            cos_v, l2_v = cosine_l2_same_frame(v_ctx[i], v_ctx[j], frame_idx=idx_i)

            print(f"  Window {windows[i]} vs {windows[j]} (local idx: {idx_i} vs {idx_j}):")
            print(f"    Q: cos={cos_q:.4f}, L2={l2_q:.4f}")
            print(f"    K: cos={cos_k:.4f}, L2={l2_k:.4f}")
            print(f"    V: cos={cos_v:.4f}, L2={l2_v:.4f}")

    # 5) 마지막 window에서 마지막 프레임 query의 frame-wise attention
    print("\n====== Last-frame attention (last window) ======\n")

    q_last = q_list[-1]  # [B,A,L,d]
    k_last = k_list[-1]

    att_frames = last_query_attention(q_last, k_last, window_size)  # [B,A,window_size]
    att_mean = att_frames.mean(0).mean(0)  # [window_size]

    for i, w in enumerate(att_mean):
        print(f"  Frame(local {i:02d}): att={w.item():.4f}")

    print("\n  Top-k frames:")
    top_vals, top_idx = torch.topk(att_mean, k=min(5, window_size))
    for v, idx in zip(top_vals, top_idx):
        print(f"    local={idx.item():02d}, att={v.item():.4f}")


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
    return DataLoader(dataset, batch_size=1, shuffle=True)


# =======================
# MAIN SCRIPT
# =======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze teacher Q/K/V context invariance with sliding windows on KITTI."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/work/juhwan/monocular_depth/stream/video-stream/checkpoints/video_depth_anything_vits.pth",
        help="teacher checkpoint path (.pth)",
    )
    parser.add_argument(
        "--kitti-path",
        type=str,
        default="/home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI",
        help="root path to KITTI dataset",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=48,
        help="long sequence length to load from dataloader",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="sliding window size (prefix length)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="sliding step size between windows",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=2,
        help="target TemporalModule index (0~3 in DPTHeadTemporal.motion_modules)",
    )
    parser.add_argument(
        "--phys-frames",
        type=int,
        nargs="+",
        default=[10, 11, 12],
        help="physical frame indices to analyze (on the full clip)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device (cuda or cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"[Info] Using ckpt: {args.ckpt}")
    print(f"[Info] Using KITTI path: {args.kitti_path}")

    # load teacher
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

    # load long sequence
    loader = build_kitti_loader(args.kitti_path, clip_len=args.clip_len)
    x, y = next(iter(loader))   # x: [1, clip_len, 3, H, W]

    analyze_teacher_sliding(
        teacher=teacher,
        x=x,
        target_layer=args.layer,
        physical_frames=args.phys_frames,
        window_size=args.window_size,
        step=args.step,
        device=device,
    )


if __name__ == "__main__":
    main()
