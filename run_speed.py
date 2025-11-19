#!/usr/bin/env python
import time
import argparse

import numpy as np
import torch

# 🔹 repo 구조에 맞춰 import 경로만 필요시 수정
# 예) from models.video_depth_stream import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnything


def parse_args():
    parser = argparse.ArgumentParser(
        description="VideoDepthAnything 스트리밍 inference 속도 측정 (dummy frames, 518x630)"
    )
    parser.add_argument("--height", type=int, default=518, help="입력 이미지 높이")
    parser.add_argument("--width", type=int, default=630, help="입력 이미지 너비")
    parser.add_argument("--num_frames", type=int, default=500, help="측정에 사용할 프레임 개수")
    parser.add_argument("--warmup", type=int, default=10, help="워밍업 프레임 개수 (타이밍에서 제외)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 또는 cpu")
    parser.add_argument("--fp32", action="store_true", help="fp32로 실행 (기본은 autocast 사용)")
    # 필요하면 streaming cache 길이도 인자로 노출할 수 있음
    parser.add_argument("--stream_cache_len", type=int, default=None,
                        help="stream cache 길이 (None이면 VideoDepthAnything 기본값 사용)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}, HxW = {args.height}x{args.width}")
    print(f"[info] num_frames = {args.num_frames}, warmup = {args.warmup}")

    # 🔹 VideoDepthAnything 생성
    #   - encoder / features / out_channels는 네가 쓰는 설정에 맞게 조정 가능
    model = VideoDepthAnything(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe="ape",
        stream_cache_len=args.stream_cache_len,  # None이면 내부 default 사용
    )
    model.to(device)
    model.eval()

    # 스트리밍 상태 초기화를 위해 새 인스턴스를 쓰고 있으므로
    # 별도 reset 함수는 필요 없음. (infer_video_depth_one 내부에서 self.id, self.transform 관리)

    H, W = args.height, args.width

    # --------------------
    # 1) WARMUP 단계
    # --------------------
    print("[info] warmup 시작...")
    for i in range(args.warmup):
        # infer_video_depth_one은 numpy(H, W, 3) BGR/uint8 이미지를 기대
        frame = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
        _ = model.infer_video_depth_one(
            frame,
            input_size=518,        # 너가 쓰는 input_size (기본 518)
            device=device.type,    # "cuda" 또는 "cpu"
            fp32=args.fp32,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    # --------------------
    # 2) 측정 구간
    # --------------------
    print("[info] 측정 시작...")
    start = time.time()
    for i in range(args.num_frames):
        frame = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
        _ = model.infer_video_depth_one(
            frame,
            input_size=518,
            device=device.type,
            fp32=args.fp32,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    fps = args.num_frames / elapsed
    ms_per_frame = elapsed * 1000.0 / args.num_frames

    print("==============================================")
    print(f"Total frames   : {args.num_frames}")
    print(f"Elapsed time   : {elapsed:.3f} s")
    print(f"Per-frame time : {ms_per_frame:.3f} ms")
    print(f"FPS            : {fps:.2f}")
    print("==============================================")


if __name__ == "__main__":
    main()
