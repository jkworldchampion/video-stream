"""
TRUE Sliding Window Inference with Batch Processing

핵심 차이:
- infer_stream.py: 프레임별 streaming (캐시 누적)
- infer_stream_eval.py (기존): 윈도우 내부도 streaming → 동일한 결과
- infer_clip_eval.py (NEW): 윈도우를 배치로 처리 → 진짜 clip-like
"""

import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from video_depth_anything.video_depth import VideoDepthAnything  # Batch model!
import shutil


def check_disk_space(path, min_gb=200):
    """디스크 공간을 확인합니다."""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.2f} GB available "
            f"(minimum {min_gb} GB required)"
        )
    return free_gb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clip-style Sliding Window Inference (TRUE Batch Processing)'
    )
    parser.add_argument('--infer_path', type=str, required=True,
                        help='Output path for depth predictions')
    parser.add_argument('--json_file', type=str, required=True,
                        help='JSON file with dataset paths')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet'],
                        help='Datasets to process')
    parser.add_argument('--input_size', type=int, default=518,
                        help='Input resolution')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'],
                        help='Encoder architecture')
    parser.add_argument('--window_size', type=int, default=32,
                        help='Sliding window size (default: 32, same as training)')
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/video_depth_anything_vits.pth',
                        help='Model checkpoint path')
    parser.add_argument('--pe', type=str, default='ape', choices=['ape', 'rope', 'none'],
                        help='Positional encoding type')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of windows to process simultaneously (GPU memory dependent)')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("Clip-style Sliding Window Inference - TRUE Batch Processing")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Window Size: {args.window_size}")
    print(f"Input Size: {args.input_size}")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # Use BATCH model (video_depth, not video_depth_stream!)
    vda = VideoDepthAnything(**model_configs[args.encoder], num_frames=args.window_size, pe=args.pe)
    
    # Checkpoint load
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Remove 'module.' or 'student.' prefix if exists
    from collections import OrderedDict
    clean_state = OrderedDict()
    for k, v in state.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('student.'):
            nk = nk[len('student.'):]
        clean_state[nk] = v

    # Flexible loading (num_frames mismatch 가능)
    missing, unexpected = vda.load_state_dict(clean_state, strict=False)
    if missing:
        print(f'Warning - Missing keys: {missing}')
    if unexpected:
        print(f'Warning - Unexpected keys: {unexpected}')
    
    vda = vda.to(DEVICE).eval()

    with open(args.json_file, 'r') as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(args.json_file)
    
    # Check disk space before starting
    free_gb = check_disk_space(args.infer_path, min_gb=10)
    print(f"Available disk space: {free_gb:.2f} GB")
    print("=" * 80)

    # Transform setup
    from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
    
    transform = Compose([
        Resize(
            width=args.input_size,
            height=args.input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    for dataset in args.datasets:
        print(f"\nProcessing dataset: {dataset}")
        json_data = path_json[dataset]
        
        for data in tqdm(json_data, desc=f"Scenes ({dataset})"):
            for scene_key in data.keys():
                frames_info = data[scene_key]
                
                print(f"\n  Scene: {scene_key} ({len(frames_info)} frames)")
                
                # 1. Load all frames
                frames = []
                output_paths = []
                
                for item in frames_info:
                    img_path = os.path.join(root_path, item['image'])
                    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    if img is None:
                        raise FileNotFoundError(f"Cannot load: {img_path}")
                    frames.append(img)
                    
                    base, _ = os.path.splitext(item['image'])
                    out_path = os.path.join(args.infer_path, dataset, base + '.npy')
                    output_paths.append(out_path)
                
                # 2. Check resume
                indices_to_process = []
                for idx, out_path in enumerate(output_paths):
                    if not os.path.exists(out_path):
                        indices_to_process.append(idx)
                
                if len(indices_to_process) == 0:
                    print(f"  ✓ All frames already processed, skipping...")
                    continue
                
                print(f"  → Processing {len(indices_to_process)}/{len(frames)} remaining frames")
                
                # 3. Sliding window BATCH inference (with batching across windows)
                frame_height, frame_width = frames[0].shape[:2]
                
                # Process in batches of windows
                num_batches = (len(indices_to_process) + args.batch_size - 1) // args.batch_size
                
                for batch_idx in tqdm(range(num_batches), desc="Sliding Window (Batch)", leave=False):
                    batch_start = batch_idx * args.batch_size
                    batch_end = min(batch_start + args.batch_size, len(indices_to_process))
                    batch_indices = indices_to_process[batch_start:batch_end]
                    
                    # Prepare multiple windows
                    batch_windows = []
                    batch_out_paths = []
                    
                    for i in batch_indices:
                        # Window: [i-31, ..., i]
                        window_frames = []
                        for j in range(i - args.window_size + 1, i + 1):
                            if j < 0:  # 음수는 0번째 프레임 padding
                                window_frames.append(frames[0])
                            else:
                                window_frames.append(frames[j])
                        
                        # Transform window frames
                        window_tensor = []
                        for frame in window_frames:
                            transformed = transform({'image': frame.astype(np.float32) / 255.0})['image']
                            window_tensor.append(torch.from_numpy(transformed))
                        
                        batch_windows.append(torch.stack(window_tensor, dim=0))  # [T, C, H, W]
                        batch_out_paths.append(output_paths[i])
                    
                    # Stack all windows: [B, T, C, H, W]
                    batch_tensor = torch.stack(batch_windows, dim=0).to(DEVICE)
                    
                    # Batch inference
                    with torch.inference_mode():
                        depth_batch = vda(batch_tensor)  # [B, T, H, W]
                    
                    # Interpolate to original size
                    depth_batch = torch.nn.functional.interpolate(
                        depth_batch.flatten(0, 1).unsqueeze(1),  # [B*T, 1, H, W]
                        size=(frame_height, frame_width),
                        mode='bilinear',
                        align_corners=True
                    )  # [B*T, 1, H, W]
                    
                    # Reshape back: [B, T, 1, H, W]
                    depth_batch = depth_batch.view(len(batch_indices), args.window_size, 1, frame_height, frame_width)
                    
                    # Save LAST frame of each window
                    for b_idx, out_path in enumerate(batch_out_paths):
                        depth_np = depth_batch[b_idx, -1, 0].cpu().numpy()  # Last frame
                        
                        try:
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            np.save(out_path, depth_np)
                        except OSError as e:
                            print(f"  ✗ Failed to save {out_path}: {e}")
                            raise
                
                print(f"  ✓ Saved {len(indices_to_process)} depth predictions")

    print("\n" + "=" * 80)
    print("Inference completed!")
    print(f"Results saved to: {args.infer_path}")
    print("=" * 80)
    print("\nThis uses TRUE batch processing (no streaming cache)")
    print("Compare with infer_stream.py to validate train-test gap hypothesis")
    print("=" * 80)
