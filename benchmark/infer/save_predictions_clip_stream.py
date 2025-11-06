"""
Save CLIP and STREAM predictions as .npy files for scale drift analysis

This script runs both CLIP (batch) and STREAM (frame-by-frame) inference
and saves predictions to disk for downstream analysis.
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


def save_clip_predictions(model, rgb_files, output_dir, window_size, input_size, device):
    """
    CLIP-style inference: Non-overlapping windows
    Saves each frame prediction as .npy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames = len(rgb_files)
    num_windows = (num_frames + window_size - 1) // window_size
    
    for w_idx in tqdm(range(num_windows), desc="CLIP windows"):
        start_idx = w_idx * window_size
        end_idx = min(start_idx + window_size, num_frames)
        
        # Load window
        rgb_window = []
        for idx in range(start_idx, end_idx):
            rgb = cv2.imread(rgb_files[idx])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (input_size, input_size))
            rgb_window.append(rgb)
        
        # Pad if needed
        while len(rgb_window) < window_size:
            rgb_window.append(rgb_window[-1])
        
        rgb_batch = np.stack(rgb_window, axis=0)
        rgb_batch = torch.from_numpy(rgb_batch).unsqueeze(0).permute(0, 1, 4, 2, 3).float()
        rgb_batch = rgb_batch / 255.0
        rgb_batch = rgb_batch.to(device)
        
        # Batch inference with fresh KV
        with torch.no_grad():
            features = model.forward_features(rgb_batch)
            depth, _ = model.forward_depth(features, rgb_batch.shape, cached_hidden_state_list=None)
        
        # Save valid frames (not padding)
        valid_frames = end_idx - start_idx
        for i in range(valid_frames):
            frame_idx = start_idx + i
            depth_map = depth[0, i, :, :].cpu().numpy()
            
            # Create output path mirroring input structure
            rel_path = rgb_files[frame_idx].replace(dataset_root, "")
            if rel_path.startswith("/"):
                rel_path = rel_path[1:]
            
            output_path = os.path.join(output_dir, rel_path.replace(".jpg", ".npy").replace(".png", ".npy"))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            np.save(output_path, depth_map)


def save_stream_predictions(model, rgb_files, output_dir, window_size, input_size, device):
    """
    STREAM-style inference: Frame-by-frame with cached KV
    Saves each frame prediction as .npy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cache_states = None
    
    for frame_idx, rgb_file in enumerate(tqdm(rgb_files, desc="STREAM frames")):
        rgb = cv2.imread(rgb_file)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (input_size, input_size))
        
        rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).float()
        rgb_tensor = rgb_tensor / 255.0
        rgb_tensor = rgb_tensor.to(device)
        
        # Streaming inference with cached KV
        with torch.no_grad():
            features = model.forward_features(rgb_tensor)
            depth, cache_states = model.forward_depth(
                features, rgb_tensor.shape,
                cached_hidden_state_list=cache_states,
            )
        
        depth_map = depth[0, 0, :, :].cpu().numpy()
        
        # Create output path mirroring input structure
        rel_path = rgb_file.replace(dataset_root, "")
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]
        
        output_path = os.path.join(output_dir, rel_path.replace(".jpg", ".npy").replace(".png", ".npy"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        np.save(output_path, depth_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save CLIP and STREAM predictions')
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir_clip', type=str, required=True)
    parser.add_argument('--output_dir_stream', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--num_scenes', type=int, default=None, help='Limit number of scenes')
    parser.add_argument('--max_frames_per_scene', type=int, default=None, help='Limit frames per scene')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Saving CLIP and STREAM Predictions")
    print("=" * 80)
    print(f"Encoder: {args.encoder}")
    print(f"Window size: {args.window_size}")
    print(f"CLIP output: {args.output_dir_clip}")
    print(f"STREAM output: {args.output_dir_stream}")
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
        print(f"Frames: {len(rgb_files)}")
        print(f"{'='*80}")
        
        # CLIP inference
        print("\n🔄 Running CLIP inference...")
        save_clip_predictions(model, rgb_files, args.output_dir_clip, args.window_size, args.input_size, device)
        print(f"✓ CLIP predictions saved to {args.output_dir_clip}")
        
        # STREAM inference
        print("\n🔄 Running STREAM inference...")
        save_stream_predictions(model, rgb_files, args.output_dir_stream, args.window_size, args.input_size, device)
        print(f"✓ STREAM predictions saved to {args.output_dir_stream}")
    
    print("\n" + "=" * 80)
    print("✅ All predictions saved!")
    print("=" * 80)
