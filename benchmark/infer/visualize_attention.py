"""
Attention Pattern 시각화: Stream vs Clip 비교

목적:
1. Stream inference: 캐시가 쌓이면서 attention이 어떻게 변하는가?
2. Clip inference: 새로운 window마다 attention 패턴이 어떤가?
3. 비교: Stream이 먼 과거 프레임에 제대로 attend하는가?

시각화:
- Attention map: [H, T_query, T_key] -> averaged over heads
- Heatmap: Query frame vs Key frame
- 특정 프레임(예: 31, 63, 95)의 attention 분포
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
import seaborn as sns
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStream


def visualize_attention_map(qkv_dict, title, save_path, query_frame_idx=None):
    """
    Args:
        qkv_dict: {"Q": [B,H,T,Dh], "K": [B,H,T,Dh], "V": [B,H,T,Dh]}
        title: plot title
        save_path: output path
        query_frame_idx: specific frame to highlight (optional)
    """
    Q = qkv_dict["Q"]  # [B, H, T, Dh]
    K = qkv_dict["K"]  # [B, H, T, Dh]
    
    B, H, T_q, Dh = Q.shape
    _, _, T_k, _ = K.shape
    
    # Compute attention: softmax(Q @ K^T / sqrt(Dh))
    # [B, H, T_q, Dh] @ [B, H, Dh, T_k] -> [B, H, T_q, T_k]
    scale = (Dh ** -0.5)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, T_q, T_k]
    attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, T_q, T_k]
    
    # Average over batch and heads: [T_q, T_k]
    attn_avg = attn_weights.mean(dim=(0, 1)).cpu().numpy()  # [T_q, T_k]
    
    # Plot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full attention map
    ax1 = axes[0]
    sns.heatmap(attn_avg, cmap='viridis', ax=ax1, cbar_kws={'label': 'Attention Weight'})
    ax1.set_xlabel('Key Frame Index')
    ax1.set_ylabel('Query Frame Index')
    ax1.set_title(f'{title}\nFull Attention Map')
    
    # Specific query frame attention distribution
    ax2 = axes[1]
    if query_frame_idx is not None and query_frame_idx < T_q:
        query_attn = attn_avg[query_frame_idx]  # [T_k]
        ax2.plot(query_attn, linewidth=2, color='navy')
        ax2.axvline(query_frame_idx, color='red', linestyle='--', linewidth=2, label=f'Query Frame {query_frame_idx}')
        ax2.set_xlabel('Key Frame Index')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title(f'Attention Distribution for Query Frame {query_frame_idx}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Show average attention distribution across all queries
        avg_attn_dist = attn_avg.mean(axis=0)  # [T_k]
        ax2.plot(avg_attn_dist, linewidth=2, color='navy')
        ax2.set_xlabel('Key Frame Index')
        ax2.set_ylabel('Average Attention Weight')
        ax2.set_title('Average Attention Distribution (all queries)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved attention map: {save_path}")
    
    # Return statistics
    stats = {
        "avg_self_attention": np.diag(attn_avg).mean(),  # Self-attention strength
        "avg_far_attention": attn_avg[T_q//2:, :T_k//2].mean() if T_q > 1 and T_k > 1 else 0.0,  # Far past
        "attn_spread": np.std(attn_avg),  # Diversity
    }
    return stats


def clip_inference_with_attention(model, rgb_files, window_size, input_size, device):
    """
    Clip inference with attention extraction
    Returns: attention patterns for each window
    """
    num_frames = len(rgb_files)
    num_windows = (num_frames + window_size - 1) // window_size
    
    attention_maps = []
    
    for w_idx in range(num_windows):
        start_idx = w_idx * window_size
        end_idx = min(start_idx + window_size, num_frames)
        window_indices = list(range(start_idx, end_idx))
        
        # Pad if needed
        if len(window_indices) < window_size:
            window_indices += [window_indices[-1]] * (window_size - len(window_indices))
        
        # Load RGB
        rgb_window = []
        for idx in window_indices:
            rgb = cv2.imread(rgb_files[idx])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (input_size, input_size))
            rgb_window.append(rgb)
        
        rgb_batch = np.stack(rgb_window, axis=0)  # [T, H, W, 3]
        rgb_batch = torch.from_numpy(rgb_batch).unsqueeze(0).permute(0, 1, 4, 2, 3).float()
        rgb_batch = rgb_batch / 255.0
        rgb_batch = rgb_batch.to(device)
        
        # Forward with return_qkv=True
        with torch.no_grad():
            result = model(rgb_batch, return_qkv=True)
        
        intermediates = result["intermediates"]
        
        # Collect attention from last motion module (layer 3)
        if 3 in intermediates and intermediates[3]["qkv"] is not None:
            qkv = intermediates[3]["qkv"]
            attention_maps.append({
                "window_idx": w_idx,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "qkv": qkv,
            })
    
    return attention_maps


def stream_inference_with_attention(model_batch, model_stream, rgb_files, input_size, device, max_frames=100, window_size=32):
    """
    Streaming inference with attention extraction (sample every N frames)
    
    Strategy: 
    - Run streaming inference normally (with cache)
    - At sample points, extract attention by batch-processing current window
    """
    from torchvision.transforms import Compose
    from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    
    num_frames = min(len(rgb_files), max_frames)
    attention_snapshots = []
    
    # Sample frames to extract attention (every 16 frames)
    sample_interval = 16
    sample_frames = list(range(sample_interval-1, num_frames, sample_interval))  # [15, 31, 47, 63, ...]
    
    # Initialize transform
    first_frame = cv2.imread(rgb_files[0])
    frame_height, frame_width = first_frame.shape[:2]
    
    ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # Reset stream model
    model_stream.frame_id_list = []
    model_stream.frame_cache_list = []
    model_stream.id = -1
    
    for frame_idx in tqdm(range(num_frames), desc="  Stream inference"):
        frame = cv2.imread(rgb_files[frame_idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Forward (streaming) - just for cache building
        model_stream.id += 1
        
        if model_stream.id == 0:  # First frame
            cur_input = torch.from_numpy(transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                cur_feature = model_stream.forward_features(cur_input)
                x_shape = cur_input.shape
                depth, cached_hidden_state_list = model_stream.forward_depth(
                    cur_feature, x_shape,
                    cached_hidden_state_list=None,
                )
            
            # Initialize cache
            import copy
            model_stream.frame_cache_list = [copy.deepcopy(cached_hidden_state_list) for _ in range(32)]
            model_stream.frame_id_list.extend([0] * 31)
            
        else:  # Subsequent frames
            cur_input = torch.from_numpy(transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                cur_feature = model_stream.forward_features(cur_input)
                x_shape = cur_input.shape
                
                # Build cache window
                cur_list = model_stream.frame_cache_list[0:2] + model_stream.frame_cache_list[-32 + 3:]
                
                def _valid_frame_cache(fc):
                    return isinstance(fc, (list, tuple)) and all((t is None) or torch.is_tensor(t) for t in fc)
                
                if not all(_valid_frame_cache(h) for h in cur_list):
                    cur_cache = None
                else:
                    L = min(len(h) for h in cur_list)
                    per_layer = []
                    cache_ok = True
                    for i in range(L):
                        elems = [h[i] for h in cur_list]
                        if any(e is None for e in elems):
                            cache_ok = False
                            break
                        try:
                            per_layer.append(torch.cat(elems, dim=1))
                        except Exception:
                            cache_ok = False
                            break
                    cur_cache = per_layer if cache_ok and len(per_layer) == L else None
                
                depth, new_cache = model_stream.forward_depth(
                    cur_feature, x_shape,
                    cached_hidden_state_list=cur_cache,
                )
            
            # Update cache
            if new_cache is None or (isinstance(new_cache, (list, tuple)) and any(t is None for t in new_cache)):
                if len(model_stream.frame_cache_list) > 0 and model_stream.frame_cache_list[-1] is not None:
                    model_stream.frame_cache_list.append(model_stream.frame_cache_list[-1])
                else:
                    model_stream.frame_cache_list.append(new_cache)
            else:
                model_stream.frame_cache_list.append(new_cache)
        
        # Adjust sliding window
        model_stream.frame_id_list.append(model_stream.id)
        if model_stream.id + 32 > 41 + 1:
            if len(model_stream.frame_id_list) > 1:
                del model_stream.frame_id_list[1]
            if len(model_stream.frame_cache_list) > 1:
                del model_stream.frame_cache_list[1]
        
        # Extract attention at sample points using batch model
        if frame_idx in sample_frames and frame_idx >= window_size - 1:
            # Get current window: [frame_idx - window_size + 1, ..., frame_idx]
            window_start = max(0, frame_idx - window_size + 1)
            window_end = frame_idx + 1
            window_indices = list(range(window_start, window_end))
            
            # Pad if needed
            if len(window_indices) < window_size:
                window_indices = [window_indices[0]] * (window_size - len(window_indices)) + window_indices
            
            # Load RGB for current window
            rgb_window = []
            for idx in window_indices:
                rgb = cv2.imread(rgb_files[idx])
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (input_size, input_size))
                rgb_window.append(rgb)
            
            rgb_batch = np.stack(rgb_window, axis=0)
            rgb_batch = torch.from_numpy(rgb_batch).unsqueeze(0).permute(0, 1, 4, 2, 3).float()
            rgb_batch = rgb_batch / 255.0
            rgb_batch = rgb_batch.to(device)
            
            # Extract attention using batch model (no cache)
            with torch.no_grad():
                result = model_batch(rgb_batch, return_qkv=True)
            
            intermediates = result["intermediates"]
            if 3 in intermediates and intermediates[3]["qkv"] is not None:
                attention_snapshots.append({
                    "frame_idx": frame_idx,
                    "cache_size": len(model_stream.frame_cache_list),
                    "window": window_indices,
                    "qkv": intermediates[3]["qkv"],
                })
    
    return attention_snapshots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Attention Patterns: Stream vs Clip')
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./attention_vis')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--num_scenes', type=int, default=3, help='Number of scenes to visualize')
    parser.add_argument('--max_frames', type=int, default=100, help='Max frames per scene for streaming')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Attention Pattern Visualization: Stream vs Clip")
    print("=" * 80)
    print(f"Encoder: {args.encoder}")
    print(f"Window size: {args.window_size}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print("\nLoading models...")
    
    # Clip model (batch) - uses num_frames, not window_size
    # Note: vits checkpoint uses features=64, not 256 (default)
    features = 64 if args.encoder == 'vits' else 256
    out_channels = [48, 96, 192, 384] if args.encoder == 'vits' else [256, 512, 1024, 1024]
    
    clip_model = VideoDepthAnything(
        encoder=args.encoder,
        features=features,
        out_channels=out_channels,
        num_frames=args.window_size,
    ).to(device).eval()
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Checkpoint is direct state_dict, not wrapped in 'model_state_dict'
    clip_model.load_state_dict(checkpoint, strict=True)
    print("✓ Loaded CLIP model (batch processing)")
    
    # Stream model - also uses num_frames
    stream_model = VideoDepthStream(
        encoder=args.encoder,
        features=features,
        out_channels=out_channels,
        num_frames=args.window_size,
    ).to(device).eval()
    
    stream_model.load_state_dict(checkpoint, strict=True)
    print("✓ Loaded STREAM model (cache accumulation)")
    
    # Load dataset
    with open(args.json_file, 'r') as f:
        json_data_raw = json.load(f)
    
    # Handle nested dict/list format: {"scannet": [{scene_name: [{image, gt_depth, factor}, ...]}, ...]}
    if isinstance(json_data_raw, dict):
        # Assume first key contains the list
        first_key = list(json_data_raw.keys())[0]
        json_data_list = json_data_raw[first_key][:args.num_scenes]
    else:
        json_data_list = json_data_raw[:args.num_scenes]
    
    # Determine base path
    base_path = os.path.dirname(args.json_file)
    dataset_root = os.path.join(base_path, "..")  # Go up one level
    
    # Process scenes
    for scene_idx, scene_dict in enumerate(json_data_list):
        # Each scene_dict is {scene_name: [{image, gt_depth, factor}, ...]}
        scene_name = list(scene_dict.keys())[0]
        frame_list = scene_dict[scene_name]
        
        # Extract RGB file paths
        rgb_files = [os.path.join(dataset_root, "scannet", frame['image']) for frame in frame_list]
        
        print(f"\n{'='*80}")
        print(f"Scene {scene_idx+1}/{len(json_data_list)}: {scene_name}")
        print(f"Total frames: {len(rgb_files)}")
        print('='*80)
        
        scene_output = os.path.join(args.output_dir, scene_name)
        os.makedirs(scene_output, exist_ok=True)
        
        # ===== CLIP Inference =====
        print("\n[1/2] CLIP Inference (batch processing)...")
        clip_attentions = clip_inference_with_attention(
            clip_model, rgb_files, args.window_size, args.input_size, device
        )
        
        print(f"  Collected {len(clip_attentions)} attention maps")
        
        # Visualize each window
        clip_stats = []
        for attn_info in clip_attentions:
            w_idx = attn_info["window_idx"]
            qkv = attn_info["qkv"]
            
            save_path = os.path.join(scene_output, f"clip_window_{w_idx:03d}.png")
            title = f"CLIP Attention - Window {w_idx} (frames {attn_info['start_frame']}-{attn_info['end_frame']})"
            
            stats = visualize_attention_map(qkv, title, save_path, query_frame_idx=args.window_size-1)
            stats["window_idx"] = w_idx
            clip_stats.append(stats)
        
        # ===== STREAM Inference =====
        print("\n[2/2] STREAM Inference (cache accumulation)...")
        stream_attentions = stream_inference_with_attention(
            clip_model, stream_model, rgb_files, args.input_size, device, 
            max_frames=args.max_frames, window_size=args.window_size
        )
        
        print(f"  Collected {len(stream_attentions)} attention snapshots")
        
        # Visualize snapshots
        stream_stats = []
        for attn_info in stream_attentions:
            frame_idx = attn_info["frame_idx"]
            cache_size = attn_info["cache_size"]
            qkv = attn_info["qkv"]
            
            save_path = os.path.join(scene_output, f"stream_frame_{frame_idx:04d}.png")
            title = f"STREAM Attention - Frame {frame_idx} (cache size: {cache_size})"
            
            # Query Frame 31 = window의 마지막 프레임 (CLIP과 동일하게)
            stats = visualize_attention_map(qkv, title, save_path, query_frame_idx=args.window_size-1)
            stats["frame_idx"] = frame_idx
            stats["cache_size"] = cache_size
            stream_stats.append(stats)
        
        # ===== Comparison Plot =====
        print("\n[3/3] Generating comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # CLIP: Self-attention over windows
        ax = axes[0, 0]
        clip_windows = [s["window_idx"] for s in clip_stats]
        clip_self_attn = [s["avg_self_attention"] for s in clip_stats]
        ax.plot(clip_windows, clip_self_attn, 'o-', linewidth=2, markersize=8, color='blue', label='CLIP')
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Avg Self-Attention')
        ax.set_title('CLIP: Self-Attention Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # STREAM: Self-attention over frames
        ax = axes[0, 1]
        stream_frames = [s["frame_idx"] for s in stream_stats]
        stream_self_attn = [s["avg_self_attention"] for s in stream_stats]
        ax.plot(stream_frames, stream_self_attn, 's-', linewidth=2, markersize=8, color='red', label='STREAM')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Avg Self-Attention')
        ax.set_title('STREAM: Self-Attention Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # CLIP: Far attention
        ax = axes[1, 0]
        clip_far_attn = [s["avg_far_attention"] for s in clip_stats]
        ax.plot(clip_windows, clip_far_attn, 'o-', linewidth=2, markersize=8, color='blue', label='CLIP')
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Avg Far Attention')
        ax.set_title('CLIP: Attention to Far Past')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # STREAM: Far attention
        ax = axes[1, 1]
        stream_far_attn = [s["avg_far_attention"] for s in stream_stats]
        ax.plot(stream_frames, stream_far_attn, 's-', linewidth=2, markersize=8, color='red', label='STREAM')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Avg Far Attention')
        ax.set_title('STREAM: Attention to Far Past')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comp_path = os.path.join(scene_output, 'comparison.png')
        plt.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved comparison: {comp_path}")
        
        # Save statistics (convert numpy types to Python types)
        import json as json_lib
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        stats_path = os.path.join(scene_output, 'statistics.json')
        with open(stats_path, 'w') as f:
            json_lib.dump({
                "clip_stats": convert_to_json_serializable(clip_stats),
                "stream_stats": convert_to_json_serializable(stream_stats),
            }, f, indent=2)
        print(f"  ✓ Saved statistics: {stats_path}")
    
    print("\n" + "=" * 80)
    print("✅ Attention Visualization Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)
