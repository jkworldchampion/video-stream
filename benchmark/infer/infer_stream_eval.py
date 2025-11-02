"""
Streaming Sliding Window Inference for Hypothesis Validation

가설 검증:
- Stream 방식이지만 sliding window (0-31, 1-32, 2-33...)로 추론
- 기존 clip 방식 (0-31, 32-63...)과 delta1 성능 비교
- Pretrained model에서 두 방식이 비슷하면 → KV cache가 제대로 작동
- 성능 차이가 크면 → train(32 frames)과 inference(1 frame) 불일치 문제
"""

import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from video_depth_anything.video_depth_stream import VideoDepthAnything


def reset_streaming_state(model):
    """스트리밍 상태를 초기화합니다."""
    model.transform = None
    model.frame_cache_list = []
    model.frame_id_list = []
    model.id = -1


def infer_sliding_window(model, frames, window_size=32, input_size=518, device='cuda', fp32=True):
    """
    Sliding window 방식으로 추론합니다.
    
    Args:
        model: VideoDepthAnything model
        frames: List of numpy arrays (RGB images)
        window_size: Window size (default: 32)
        input_size: Input resolution
        device: Device to use
        fp32: Use float32 precision
    
    Returns:
        depths: List of depth maps (numpy arrays)
    """
    num_frames = len(frames)
    depths = []
    
    # 초기 window_size-1 프레임은 더미로 채우거나 첫 프레임 복사
    # 여기서는 간단히 첫 프레임을 복사해서 사용
    initial_frames = [frames[0]] * (window_size - 1)
    
    # 각 프레임 인덱스에 대해 sliding window 구성
    for i in tqdm(range(num_frames), desc="Sliding Window Inference", leave=False):
        # Reset state for each window (clip처럼 독립적으로 처리)
        reset_streaming_state(model)
        
        # Window 구성: [i-31, i-30, ..., i-1, i]
        # 초반 프레임의 경우 더미 프레임으로 패딩
        window_frames = []
        for j in range(i - window_size + 1, i + 1):
            if j < 0:
                # 더미: 첫 프레임 복사
                window_frames.append(frames[0])
            else:
                window_frames.append(frames[j])
        
        # Window 전체를 순차적으로 처리 (streaming 시뮬레이션)
        with torch.inference_mode():
            for frame_idx, frame in enumerate(window_frames):
                depth_np = model.infer_video_depth_one(
                    frame, 
                    input_size=input_size, 
                    device=device, 
                    fp32=fp32
                )
                
                # 마지막 프레임(현재 프레임)의 depth만 저장
                if frame_idx == len(window_frames) - 1:
                    depths.append(depth_np)
    
    return depths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Streaming Sliding Window Inference (Hypothesis Validation)'
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
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("Streaming Sliding Window Inference - Hypothesis Validation")
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

    vda = VideoDepthAnything(**model_configs[args.encoder], pe=args.pe)
    
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

    missing, unexpected = vda.load_state_dict(clean_state, strict=True)
    if missing:
        print(f'Warning - Missing keys: {missing}')
    if unexpected:
        print(f'Warning - Unexpected keys: {unexpected}')
    
    vda = vda.to(DEVICE).eval()

    with open(args.json_file, 'r') as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(args.json_file)

    for dataset in args.datasets:
        print(f"\nProcessing dataset: {dataset}")
        json_data = path_json[dataset]
        
        for data in tqdm(json_data, desc=f"Scenes ({dataset})"):
            for scene_key in data.keys():
                frames_info = data[scene_key]  # 이 시퀀스의 프레임 리스트
                
                print(f"\n  Scene: {scene_key} ({len(frames_info)} frames)")
                
                # 1. 모든 프레임을 먼저 로드
                frames = []
                output_paths = []
                
                for item in frames_info:
                    img_path = os.path.join(root_path, item['image'])
                    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    if img is None:
                        raise FileNotFoundError(f"Cannot load: {img_path}")
                    frames.append(img)
                    
                    # Output path 구성
                    base, _ = os.path.splitext(item['image'])
                    out_path = os.path.join(args.infer_path, dataset, base + '.npy')
                    output_paths.append(out_path)
                
                # 2. Sliding window 방식으로 depth 추론
                depths = infer_sliding_window(
                    model=vda,
                    frames=frames,
                    window_size=args.window_size,
                    input_size=args.input_size,
                    device=DEVICE,
                    fp32=True
                )
                
                # 3. 결과 저장
                for depth_np, out_path in zip(depths, output_paths):
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, depth_np)
                
                print(f"  ✓ Saved {len(depths)} depth predictions")

    print("\n" + "=" * 80)
    print("Inference completed!")
    print(f"Results saved to: {args.infer_path}")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run evaluation script to compute delta1")
    print("2. Compare with clip-based inference results")
    print("3. Validate hypothesis: stream sliding ≈ clip keyframe")
    print("=" * 80)
