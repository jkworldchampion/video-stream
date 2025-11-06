"""
KV Cache Quality Analysis: CLIP vs STREAM Depth Comparison

목적:
- 실제 CLIP (batch) vs STREAM (cached) inference 방식 재현
- Cached KV 사용이 depth quality 저하의 원인인지 검증
- Cache 누적에 따른 성능 변화 측정

실험 설계:
1. CLIP Inference (Baseline - Fresh KV):
   - Non-overlapping windows: [0-31], [32-63], [64-95], ...
   - 각 window를 독립적으로 batch 처리
   - 모든 프레임이 fresh KV 사용
   
2. STREAM Inference (Test - Cached KV):
   - Frame-by-frame processing: 0, 1, 2, 3, ...
   - Cache 누적: frame 0 → cache[0], frame 1 → cache[0,1], ...
   - Window size 32로 sliding (frame 32부터 cache[1-31] 사용)
   - Cached KV 재사용 (gradient 없이 생성된 KV)

3. 측정:
   - 같은 frame index의 CLIP depth vs STREAM depth 비교
   - Per-frame MAE, RMSE
   - Cache age별 성능 변화 (초기 vs 중간 vs 후기)
   - Window-level aggregation

기대 결과:
- CLIP과 STREAM 간 명확한 차이 재현 (MAE > 0)
- Cache age 증가에 따른 성능 저하 확인
- 실제 validation delta1 차이 (0.65 vs 0.60) 원인 규명
"""

import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStream


def compute_feature_distance(feat1, feat2):
    """
    Compute L2 distance between two feature maps
    Args:
        feat1, feat2: [B, C, ...] tensors
    Returns:
        L2 distance (scalar)
    """
    return torch.norm(feat1 - feat2, p=2).item()


def compute_depth_metrics(depth1, depth2):
    """
    Compute depth difference metrics
    Args:
        depth1, depth2: [B, T, H, W] or [H, W] tensors
    Returns:
        dict of metrics
    """
    diff = (depth1 - depth2).abs()
    return {
        "mae": diff.mean().item(),
        "rmse": (diff ** 2).mean().sqrt().item(),
        "max_error": diff.max().item(),
    }


def clip_inference(model, all_rgb_frames, window_size, device):
    """
    CLIP inference: Non-overlapping batch windows
    
    Args:
        model: VideoDepthStream model
        all_rgb_frames: List of all RGB frames
        window_size: Window size (32)
        device: torch device
    
    Returns:
        all_depths: [N, H, W] all frame depths
    """
    all_depths = []
    num_frames = len(all_rgb_frames)
    num_windows = (num_frames + window_size - 1) // window_size
    
    for w_idx in range(num_windows):
        start_idx = w_idx * window_size
        end_idx = min(start_idx + window_size, num_frames)
        
        # Load window
        rgb_window = []
        for idx in range(start_idx, end_idx):
            rgb_window.append(all_rgb_frames[idx])
        
        # Pad if needed
        while len(rgb_window) < window_size:
            rgb_window.append(all_rgb_frames[-1])
        
        rgb_batch = np.stack(rgb_window, axis=0)
        rgb_batch = torch.from_numpy(rgb_batch).unsqueeze(0).permute(0, 1, 4, 2, 3).float()
        rgb_batch = rgb_batch / 255.0
        rgb_batch = rgb_batch.to(device)
        
        # Batch inference with fresh KV
        features = model.forward_features(rgb_batch)
        depth, _ = model.forward_depth(
            features, rgb_batch.shape,
            cached_hidden_state_list=None,
        )
        
        # Extract valid frames (not padding)
        valid_frames = end_idx - start_idx
        for i in range(valid_frames):
            all_depths.append(depth[0, i, :, :].cpu())
    
    return all_depths


def stream_inference(model, all_rgb_frames, window_size, device):
    """
    STREAM inference: Frame-by-frame with cached KV
    
    Args:
        model: VideoDepthStream model
        all_rgb_frames: List of all RGB frames
        window_size: Window size (32)
        device: torch device
    
    Returns:
        all_depths: [N, H, W] all frame depths
        cache_ages: [N] cache age for each frame
    """
    cache_states = None
    all_depths = []
    cache_ages = []
    
    for frame_idx, rgb in enumerate(all_rgb_frames):
        rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).float()
        rgb_tensor = rgb_tensor / 255.0
        rgb_tensor = rgb_tensor.to(device)
        
        # Streaming inference with cached KV
        features = model.forward_features(rgb_tensor)
        depth, cache_states = model.forward_depth(
            features, rgb_tensor.shape,
            cached_hidden_state_list=cache_states,
        )
        
        all_depths.append(depth[0, 0, :, :].cpu())
        
        # Cache age: number of frames in cache (max = window_size - 1)
        cache_age = min(frame_idx, window_size - 1)
        cache_ages.append(cache_age)
    
    return all_depths, cache_ages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KV Cache Quality Analysis')
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./cache_quality')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--num_windows', type=int, default=10, help='Number of windows to analyze')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KV Cache Quality Analysis: Cache Age Impact")
    print("=" * 80)
    print(f"Encoder: {args.encoder}")
    print(f"Window size: {args.window_size}")
    print(f"Num windows: {args.num_windows}")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("\nLoading model...")
    features = 64 if args.encoder == 'vits' else 256
    out_channels = [48, 96, 192, 384] if args.encoder == 'vits' else [256, 512, 1024, 1024]
    
    model = VideoDepthStream(
        encoder=args.encoder,
        features=features,
        out_channels=out_channels,
        num_frames=args.window_size,
    ).to(device).eval()
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint, strict=True)
    print("✓ Loaded model")
    
    # Load dataset
    with open(args.json_file, 'r') as f:
        json_data_raw = json.load(f)
    
    if isinstance(json_data_raw, dict):
        first_key = list(json_data_raw.keys())[0]
        json_data_list = json_data_raw[first_key][:1]  # Just first scene
    else:
        json_data_list = json_data_raw[:1]
    
    base_path = os.path.dirname(args.json_file)
    dataset_root = os.path.join(base_path, "..")
    
    # Process first scene
    scene_dict = json_data_list[0]
    scene_name = list(scene_dict.keys())[0]
    frame_list = scene_dict[scene_name]
    rgb_files = [os.path.join(dataset_root, "scannet", frame['image']) for frame in frame_list]
    
    print(f"\nProcessing scene: {scene_name}")
    print(f"Total frames: {len(rgb_files)}")
    
    # Load all RGB frames into memory
    print("\n📥 Loading all RGB frames into memory...")
    all_rgb_frames = []
    num_frames_to_load = min(len(rgb_files), args.window_size * args.num_windows)
    for idx in tqdm(range(num_frames_to_load), desc="Loading frames"):
        rgb = cv2.imread(rgb_files[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (args.input_size, args.input_size))
        all_rgb_frames.append(rgb)
    
    print(f"✓ Loaded {len(all_rgb_frames)} frames")
    
    # ====================
    # CLIP INFERENCE (Baseline)
    # ====================
    print("\n🔄 Running CLIP Inference (Non-overlapping windows, Fresh KV)...")
    with torch.no_grad():
        clip_depths = clip_inference(model, all_rgb_frames, args.window_size, device)
    
    print(f"✓ CLIP inference complete: {len(clip_depths)} frames")
    
    # ====================
    # STREAM INFERENCE (Test)
    # ====================
    print("\n🔄 Running STREAM Inference (Frame-by-frame, Cached KV)...")
    with torch.no_grad():
        stream_depths, cache_ages = stream_inference(model, all_rgb_frames, args.window_size, device)
    
    print(f"✓ STREAM inference complete: {len(stream_depths)} frames")
    
    # ====================
    # COMPARISON
    # ====================
    print("\n📊 Computing CLIP vs STREAM Differences...")
    
    # Per-frame metrics
    frame_metrics = []
    for frame_idx in range(len(all_rgb_frames)):
        clip_depth = clip_depths[frame_idx]
        stream_depth = stream_depths[frame_idx]
        
        metrics = compute_depth_metrics(clip_depth, stream_depth)
        frame_metrics.append({
            "frame_idx": frame_idx,
            "cache_age": cache_ages[frame_idx],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "max_error": metrics["max_error"],
        })
    
    # Window-level aggregation
    num_windows = len(all_rgb_frames) // args.window_size
    window_metrics = []
    
    for w_idx in range(num_windows):
        start_idx = w_idx * args.window_size
        end_idx = start_idx + args.window_size
        
        window_maes = [frame_metrics[i]["mae"] for i in range(start_idx, end_idx)]
        window_rmses = [frame_metrics[i]["rmse"] for i in range(start_idx, end_idx)]
        
        window_metrics.append({
            "window_idx": w_idx,
            "start_frame": start_idx,
            "end_frame": end_idx,
            "avg_mae": np.mean(window_maes),
            "avg_rmse": np.mean(window_rmses),
            "std_mae": np.std(window_maes),
        })
    
    # Cache age aggregation
    cache_age_metrics = {}
    for fm in frame_metrics:
        age = fm["cache_age"]
        if age not in cache_age_metrics:
            cache_age_metrics[age] = {"mae": [], "rmse": []}
        cache_age_metrics[age]["mae"].append(fm["mae"])
        cache_age_metrics[age]["rmse"].append(fm["rmse"])
    
    results = {
        "num_frames": len(all_rgb_frames),
        "num_windows": num_windows,
        "frame_metrics": frame_metrics,
        "window_metrics": window_metrics,
        "cache_age_metrics": cache_age_metrics,
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, 'cache_age_results.json')
    
    # Convert to JSON serializable
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\n✓ Saved results: {results_path}")
    
    # ====================
    # PRINT RESULTS
    # ====================
    print("\n" + "=" * 80)
    print("CLIP vs STREAM Comparison")
    print("=" * 80)
    
    # Overall statistics
    all_maes = [fm["mae"] for fm in frame_metrics]
    all_rmses = [fm["rmse"] for fm in frame_metrics]
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total frames: {len(all_rgb_frames)}")
    print(f"  Average MAE: {np.mean(all_maes):.6f} ± {np.std(all_maes):.6f}")
    print(f"  Average RMSE: {np.mean(all_rmses):.6f} ± {np.std(all_rmses):.6f}")
    print(f"  Median MAE: {np.median(all_maes):.6f}")
    print(f"  Max MAE: {np.max(all_maes):.6f}")
    print(f"  Min MAE: {np.min(all_maes):.6f}")
    
    # Window-level results
    print(f"\n📊 Window-Level Results:")
    print(f"{'Window':<10} {'Frames':<15} {'Avg MAE':<12} {'Avg RMSE':<12} {'Std MAE':<12}")
    print("-" * 80)
    for wm in window_metrics:
        frames_str = f"{wm['start_frame']}-{wm['end_frame']-1}"
        print(f"{wm['window_idx']:<10} {frames_str:<15} {wm['avg_mae']:<12.6f} {wm['avg_rmse']:<12.6f} {wm['std_mae']:<12.6f}")
    
    # Cache age statistics
    print(f"\n📈 Cache Age vs Error:")
    print(f"{'Cache Age':<12} {'Avg MAE':<12} {'Avg RMSE':<12} {'Std MAE':<12} {'Num Frames':<12}")
    print("-" * 80)
    
    for age in sorted(cache_age_metrics.keys()):
        mae_list = cache_age_metrics[age]["mae"]
        rmse_list = cache_age_metrics[age]["rmse"]
        
        avg_mae = np.mean(mae_list)
        avg_rmse = np.mean(rmse_list)
        std_mae = np.std(mae_list)
        num_frames = len(mae_list)
        
        print(f"{age:<12} {avg_mae:<12.6f} {avg_rmse:<12.6f} {std_mae:<12.6f} {num_frames:<12}")
    
    # ====================
    # VISUALIZATION
    # ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Per-frame MAE over time
    ax = axes[0, 0]
    frame_indices = [fm["frame_idx"] for fm in frame_metrics]
    frame_maes = [fm["mae"] for fm in frame_metrics]
    
    ax.plot(frame_indices, frame_maes, linewidth=1.5, alpha=0.8, color='blue')
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('MAE (CLIP vs STREAM)', fontsize=12)
    ax.set_title('Per-Frame Depth Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add window boundaries
    for w_idx in range(num_windows):
        boundary = (w_idx + 1) * args.window_size
        if boundary < len(all_rgb_frames):
            ax.axvline(boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 2: Cache Age vs MAE
    ax = axes[0, 1]
    cache_ages_sorted = sorted(cache_age_metrics.keys())
    avg_maes = [np.mean(cache_age_metrics[age]["mae"]) for age in cache_ages_sorted]
    std_maes = [np.std(cache_age_metrics[age]["mae"]) for age in cache_ages_sorted]
    
    ax.errorbar(cache_ages_sorted, avg_maes, yerr=std_maes, fmt='o-',
                linewidth=2, markersize=10, capsize=5, color='red', elinewidth=2)
    ax.set_xlabel('Cache Age (frames)', fontsize=12)
    ax.set_ylabel('Average MAE', fontsize=12)
    ax.set_title('Cache Age vs Depth Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Window-level MAE
    ax = axes[1, 0]
    window_indices = [wm["window_idx"] for wm in window_metrics]
    window_maes = [wm["avg_mae"] for wm in window_metrics]
    window_stds = [wm["std_mae"] for wm in window_metrics]
    
    bars = ax.bar(window_indices, window_maes, yerr=window_stds, capsize=5,
                   color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Average MAE', fontsize=12)
    ax.set_title('Window-Level Depth Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: MAE distribution histogram
    ax = axes[1, 1]
    ax.hist(frame_maes, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(frame_maes), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(frame_maes):.4f}')
    ax.axvline(np.median(frame_maes), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(frame_maes):.4f}')
    ax.set_xlabel('MAE', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('MAE Distribution (CLIP vs STREAM)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'clip_vs_stream_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: {plot_path}")
    
    # ====================
    # KEY FINDINGS
    # ====================
    print("\n" + "=" * 80)
    print("✅ CLIP vs STREAM Analysis Complete!")
    print("=" * 80)
    print(f"\n🔍 Key Findings:")
    print(f"  1. Average depth difference (CLIP vs STREAM): {np.mean(all_maes):.4f} MAE")
    print(f"  2. This represents the performance gap we see in validation")
    
    # Cache age trend
    if len(cache_ages_sorted) >= 2:
        mae_start = np.mean(cache_age_metrics[cache_ages_sorted[0]]["mae"])
        mae_end = np.mean(cache_age_metrics[cache_ages_sorted[-1]]["mae"])
        mae_change = mae_end - mae_start
        mae_change_pct = (mae_change / mae_start) * 100 if mae_start > 0 else 0
        
        print(f"  3. Cache age impact:")
        print(f"     - MAE at cache age {cache_ages_sorted[0]}: {mae_start:.4f}")
        print(f"     - MAE at cache age {cache_ages_sorted[-1]}: {mae_end:.4f}")
        print(f"     - Change: {mae_change:+.4f} ({mae_change_pct:+.1f}%)")
        
        if abs(mae_change_pct) < 10:
            print(f"     → Cache quality is STABLE across different ages")
            print(f"     → Performance gap is NOT primarily due to cache staleness")
            print(f"     → Other factors may be more important (e.g., temporal modeling)")
        elif mae_change_pct > 20:
            print(f"     ⚠️  Significant cache degradation with age!")
            print(f"     → Cache quality IS a major factor")
        else:
            print(f"     → Moderate cache quality impact")
    
    print("=" * 80)
