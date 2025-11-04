"""
Key-frame Structure Inference (논문의 방법 재현)

핵심 아이디어:
- 항상 첫 프레임(frame 0)을 key-frame으로 포함
- Overlapping frames로 smooth transition
- Scale consistency를 위한 anchor 역할

Window structure: [Tk key_frames, To overlap, N-To-Tk future]
Example (window=32, Tk=4, To=8, Δk=8):
  Window 1: [0, 8, 16, 24] + [0-7] + [8-31]  → predict [8-31]
  Window 2: [0, 8, 16, 24] + [24-31] + [32-55] → predict [32-55]
            ↑ 첫 프레임 항상 포함!
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

from video_depth_anything.video_depth import VideoDepthAnything
import shutil


def check_disk_space(path, min_gb=200):
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.2f} GB available "
            f"(minimum {min_gb} GB required)"
        )
    return free_gb


def create_keyframe_window(frame_indices, target_idx, window_size=32, 
                           num_keyframes=4, num_overlap=8, key_interval=8):
    """
    Key-frame structure window 생성
    
    Args:
        frame_indices: 전체 프레임 인덱스 리스트
        target_idx: 예측할 타겟 프레임의 인덱스
        window_size: 전체 윈도우 크기 (e.g., 32)
        num_keyframes: Key-frame 개수 (Tk, e.g., 4)
        num_overlap: Overlapping frames (To, e.g., 8)
        key_interval: Key-frame 샘플링 간격 (Δk, e.g., 8)
    
    Returns:
        window: List of frame indices for this window
        target_position: Target frame's position in window
    """
    max_idx = len(frame_indices) - 1
    
    # Key-frames: 항상 첫 프레임(0)부터 시작, interval로 샘플링
    # 예: [0, 8, 16, 24]
    keyframes = []
    for i in range(num_keyframes):
        kf_idx = min(i * key_interval, max_idx)
        keyframes.append(frame_indices[kf_idx])
    
    # Overlapping frames: target 이전 프레임들
    # 예: target=32이면 [24-31]
    overlap_start = max(0, target_idx - num_overlap)
    overlap_frames = [frame_indices[min(i, max_idx)] 
                      for i in range(overlap_start, target_idx)]
    
    # Future frames: target부터 window 끝까지
    # 예: target=32이면 [32-55]
    num_future = window_size - num_keyframes - num_overlap
    future_frames = [frame_indices[min(target_idx + i, max_idx)] 
                     for i in range(num_future)]
    
    # 전체 window: [key-frames] + [overlap] + [future]
    window = keyframes + overlap_frames + future_frames
    
    # Target position: key-frames + overlap 다음 위치
    target_position = num_keyframes + num_overlap
    
    return window, target_position


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Key-frame Structure Inference (Paper Method)'
    )
    parser.add_argument('--infer_path', type=str, required=True)
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='Limit number of scenes for quick testing')
    
    # Key-frame structure parameters
    parser.add_argument('--num_keyframes', type=int, default=4,
                        help='Number of key frames (Tk)')
    parser.add_argument('--num_overlap', type=int, default=8,
                        help='Number of overlapping frames (To)')
    parser.add_argument('--key_interval', type=int, default=8,
                        help='Key frame sampling interval (Δk)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Key-frame Structure Inference (Paper Method)")
    print("=" * 80)
    print(f"\nWindow structure: [Tk={args.num_keyframes} keyframes, "
          f"To={args.num_overlap} overlap, "
          f"N-To-Tk={args.window_size - args.num_keyframes - args.num_overlap} future]")
    print(f"Key interval (Δk): {args.key_interval}")
    print(f"\nExample window:")
    print(f"  [0, 8, 16, 24] (keyframes) + [24-31] (overlap) + [32-55] (future)")
    print(f"                              ↑ 첫 프레임 항상 포함 (scale anchor)!")
    if args.max_scenes:
        print(f"\nQuick test mode: {args.max_scenes} scenes")
    print("=" * 80)
    
    # 디스크 공간 확인
    free_gb = check_disk_space(args.infer_path)
    print(f"\nAvailable disk space: {free_gb:.2f} GB")
    
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading model on {device}...")
    
    model = VideoDepthAnything(
        encoder=args.encoder,
        window_size=args.window_size,
    ).to(device).eval()
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"✓ Loaded checkpoint: {args.checkpoint}")
    
    # 데이터셋 로드
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)
    
    if args.max_scenes:
        json_data = json_data[:args.max_scenes]
        print(f"\n⚡ Limited to {len(json_data)} scenes")
    
    # 각 scene 처리
    for scene_idx, scene in enumerate(tqdm(json_data, desc="Scenes")):
        scene_name = scene['scene']
        rgb_files = sorted(scene['color_files'])
        depth_files = sorted(scene['depth_files'])
        
        num_frames = len(rgb_files)
        if num_frames < args.window_size:
            print(f"\nSkipping {scene_name}: only {num_frames} frames")
            continue
        
        print(f"\n[{scene_idx+1}/{len(json_data)}] Processing {scene_name} "
              f"({num_frames} frames)...")
        
        # 출력 디렉토리
        scene_output = os.path.join(args.infer_path, scene_name, 'depth')
        os.makedirs(scene_output, exist_ok=True)
        
        # Frame indices
        frame_indices = list(range(num_frames))
        
        # Key-frame structure로 윈도우 생성
        # Window stride: N - To (새로운 프레임만큼 이동)
        stride = args.window_size - args.num_keyframes - args.num_overlap
        
        # 처리할 타겟 프레임들
        target_frames = list(range(args.num_keyframes + args.num_overlap, 
                                   num_frames, stride))
        
        batch_windows = []
        batch_targets = []
        batch_positions = []
        batch_scene_names = []
        
        for target_idx in tqdm(target_frames, desc=f"  {scene_name}", leave=False):
            # Key-frame window 생성
            window, target_pos = create_keyframe_window(
                frame_indices, target_idx,
                window_size=args.window_size,
                num_keyframes=args.num_keyframes,
                num_overlap=args.num_overlap,
                key_interval=args.key_interval
            )
            
            batch_windows.append(window)
            batch_targets.append(target_idx)
            batch_positions.append(target_pos)
            batch_scene_names.append(scene_name)
            
            # Batch가 찼으면 처리
            if len(batch_windows) == args.batch_size:
                # RGB 로드
                batch_rgb = []
                for window in batch_windows:
                    rgb_window = []
                    for frame_idx in window:
                        rgb_path = rgb_files[frame_idx]
                        rgb = cv2.imread(rgb_path)
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        rgb = cv2.resize(rgb, (args.input_size, args.input_size))
                        rgb_window.append(rgb)
                    batch_rgb.append(np.stack(rgb_window, axis=0))  # [T, H, W, 3]
                
                batch_rgb = np.stack(batch_rgb, axis=0)  # [B, T, H, W, 3]
                batch_rgb = torch.from_numpy(batch_rgb).permute(0, 1, 4, 2, 3).float()
                batch_rgb = batch_rgb / 255.0
                batch_rgb = batch_rgb.to(device)
                
                # 추론
                with torch.no_grad():
                    depth_batch = model(batch_rgb)  # [B, T, 1, H, W]
                
                # 결과 저장
                for b_idx in range(len(batch_windows)):
                    target_idx = batch_targets[b_idx]
                    target_pos = batch_positions[b_idx]
                    
                    # 타겟 포지션에서 depth 추출
                    depth = depth_batch[b_idx, target_pos, 0].cpu().numpy()
                    
                    # 원본 해상도로 복원
                    gt_path = depth_files[target_idx]
                    gt_depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                    h, w = gt_depth.shape
                    depth = cv2.resize(depth, (w, h))
                    
                    # PNG로 저장 (16-bit)
                    depth_uint16 = (depth * 1000.0).astype(np.uint16)
                    out_filename = os.path.basename(depth_files[target_idx])
                    out_path = os.path.join(scene_output, out_filename)
                    cv2.imwrite(out_path, depth_uint16)
                
                # Batch 초기화
                batch_windows = []
                batch_targets = []
                batch_positions = []
                batch_scene_names = []
        
        # 남은 배치 처리
        if len(batch_windows) > 0:
            batch_rgb = []
            for window in batch_windows:
                rgb_window = []
                for frame_idx in window:
                    rgb_path = rgb_files[frame_idx]
                    rgb = cv2.imread(rgb_path)
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (args.input_size, args.input_size))
                    rgb_window.append(rgb)
                batch_rgb.append(np.stack(rgb_window, axis=0))
            
            batch_rgb = np.stack(batch_rgb, axis=0)
            batch_rgb = torch.from_numpy(batch_rgb).permute(0, 1, 4, 2, 3).float()
            batch_rgb = batch_rgb / 255.0
            batch_rgb = batch_rgb.to(device)
            
            with torch.no_grad():
                depth_batch = model(batch_rgb)
            
            for b_idx in range(len(batch_windows)):
                target_idx = batch_targets[b_idx]
                target_pos = batch_positions[b_idx]
                
                depth = depth_batch[b_idx, target_pos, 0].cpu().numpy()
                
                gt_path = depth_files[target_idx]
                gt_depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                h, w = gt_depth.shape
                depth = cv2.resize(depth, (w, h))
                
                depth_uint16 = (depth * 1000.0).astype(np.uint16)
                out_filename = os.path.basename(depth_files[target_idx])
                out_path = os.path.join(scene_output, out_filename)
                cv2.imwrite(out_path, depth_uint16)
    
    print("\n" + "=" * 80)
    print("✅ Key-frame Structure Inference Complete!")
    print(f"Output: {args.infer_path}")
    print("=" * 80)
