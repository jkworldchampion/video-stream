"""
Save Past-only CLIP predictions for Window Context Direction experiment

This script implements CLIP inference with PAST-ONLY context:
- Window: [N-31, N] (past frames only, like STREAM)
- Target: Last frame in window (frame N)

This allows us to test if future context is the reason for CLIP's better performance.
"""

import argparse
import os
import sys
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from video_depth_anything.video_depth_stream import VideoDepthAnything


def save_past_only_clip_predictions(model, rgb_files, output_dir, window_size, input_size, device):
    """
    Past-only CLIP inference: Uses PAST context only
    
    For frame N:
    - Window: [N-31, N] (past 31 frames + current frame)
    - Target: frame N (last frame in window)
    
    This mirrors STREAM's context direction but uses batch processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames = len(rgb_files)
    
    # Start from frame 31 (need 31 past frames)
    for frame_idx in tqdm(range(window_size-1, num_frames), desc="Past-only CLIP"):
        # Get past window: [frame_idx - 31, ..., frame_idx]
        start_idx = frame_idx - window_size + 1
        end_idx = frame_idx + 1
        
        # Load window
        rgb_window = []
        for idx in range(start_idx, end_idx):
            rgb = cv2.imread(rgb_files[idx])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (input_size, input_size))
            rgb_window.append(rgb)
        
        rgb_batch = np.stack(rgb_window, axis=0)
        rgb_batch = torch.from_numpy(rgb_batch).unsqueeze(0).permute(0, 1, 4, 2, 3).float()
        rgb_batch = rgb_batch / 255.0
        rgb_batch = rgb_batch.to(device)
        
        # Batch inference with fresh KV
        with torch.no_grad():
            features = model.forward_features(rgb_batch)
            depth, _ = model.forward_depth(features, rgb_batch.shape, cached_hidden_state_list=None)
        
        # Get prediction for LAST frame (frame N)
        depth_map = depth[0, -1, :, :].cpu().numpy()
        
        # Create output path mirroring input structure
        rel_path = rgb_files[frame_idx].replace(dataset_root, "")
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]
        
        output_path = os.path.join(output_dir, rel_path.replace(".jpg", ".npy").replace(".png", ".npy"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        np.save(output_path, depth_map)
    
    print(f"  → Predicted frames {window_size-1} to {num_frames-1} ({num_frames - window_size + 1} frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save Past-only CLIP predictions')
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--num_scenes', type=int, default=None, help='Limit number of scenes')
    parser.add_argument('--max_frames_per_scene', type=int, default=None, help='Limit frames per scene')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Window Context Direction Experiment: Past-only CLIP")
    print("=" * 80)
    print(f"Encoder: {args.encoder}")
    print(f"Window size: {args.window_size}")
    print(f"Context: PAST ONLY [N-{args.window_size-1}, N]")
    print(f"Target: Last frame (N)")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("\nLoading model...")
    features = 64 if args.encoder == 'vits' else 256
    out_channels = [48, 96, 192, 384] if args.encoder == 'vits' else [256, 512, 1024, 1024]
    
    model = VideoDepthAnything(
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
        json_data_list = json_data_raw[first_key]
    else:
        json_data_list = json_data_raw
    
    if args.num_scenes:
        json_data_list = json_data_list[:args.num_scenes]
    
    dataset_root = os.path.join(os.path.dirname(args.json_file), "..")
    
    # Process each scene
    for scene_dict in json_data_list:
        scene_name = list(scene_dict.keys())[0]
        frame_list = scene_dict[scene_name]
        
        if args.max_frames_per_scene:
            frame_list = frame_list[:args.max_frames_per_scene]
        
        rgb_files = [os.path.join(dataset_root, "scannet", frame['image']) for frame in frame_list]
        
        print(f"\n{'='*80}")
        print(f"Scene: {scene_name}")
        print(f"Total frames: {len(rgb_files)}")
        print(f"{'='*80}")
        
        # Past-only CLIP inference
        print("\n🔄 Running Past-only CLIP inference...")
        save_past_only_clip_predictions(model, rgb_files, args.output_dir, args.window_size, args.input_size, device)
        print(f"✓ Past-only CLIP predictions saved to {args.output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ All predictions saved!")
    print("=" * 80)
