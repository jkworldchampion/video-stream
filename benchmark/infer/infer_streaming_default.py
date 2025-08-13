# Copyright (2025) Bytedance Ltd.
# Apache License 2.0

import argparse
import os, sys
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

from video_depth_anything.video_depth_stream import VideoDepthAnything
# JSON은 동일 포맷 사용: {"scannet": [ { "seq_id": [ {"image": ".../frame_0001.jpg"}, ... ] }, ... ], ... }

def build_model(encoder: str, device: str):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    m = VideoDepthAnything(**model_configs[encoder])
    ckpt = torch.load(f'./outputs/experiment_9/best_model.pth', map_location='cuda')
    m.load_state_dict(ckpt, strict=True)
    m = m.to(device).eval()
    return m

def try_reset_stream_state(model) -> bool:
    """
    다양한 구현을 감안해 존재하면 호출하는 방식.
    없으면 False를 반환하여 상위 로직에서 재초기화를 선택하도록 함.
    """
    for name in [
        "reset_stream", "reset_stream_state", "clear_stream_cache",
        "clear_cache", "reset_cache", "reset_caches"
    ]:
        if hasattr(model, name) and callable(getattr(model, name)):
            try:
                getattr(model, name)()
                return True
            except Exception:
                pass
    # 일부 구현은 내부 속성으로 캐시를 노출할 수 있음
    for attr in ["temporal_caches", "cache", "kv_cache"]:
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
                return True
            except Exception:
                pass
    return False

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def main():
    parser = argparse.ArgumentParser(description="Streaming inference for Video Depth Anything (dataset/JSON)")
    parser.add_argument('--infer_path', type=str, required=True, help='출력 .npy 루트 폴더')
    parser.add_argument('--json_file', type=str, required=True, help='데이터셋 JSON 경로')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet', 'nyuv2'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--fp32', action='store_true', help='기본은 fp16, 이 옵션을 주면 fp32로 동작')
    parser.add_argument('--stride', type=int, default=1, help='프레임 서브샘플링 간격(기본 1: 모든 프레임)')
    parser.add_argument('--reinit_per_sequence', action='store_true',
                        help='시퀀스마다 모델을 재생성하여 캐시 초기화(리셋 메서드가 없거나 불안정할 때)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # JSON 로드
    with open(args.json_file, 'r') as fs:
        path_json = json.load(fs)

    # 모델 준비
    model = build_model(args.encoder, device)

    # 각 데이터셋 처리
    for dataset in args.datasets:
        if dataset not in path_json:
            print(f"[경고] JSON에 '{dataset}' 키가 없습니다. 건너뜁니다.")
            continue

        json_data = path_json[dataset]
        root_path = os.path.dirname(args.json_file)

        # json_data: 리스트, 각 원소는 {seq_id: [ {image: "..."} , ... ]} 형태 가정
        for seq in tqdm(json_data, desc=f"[{dataset}] sequences"):
            # seq_id, frame_list 파싱
            if not isinstance(seq, dict) or len(seq) == 0:
                continue
            seq_id = list(seq.keys())[0]
            frame_entries = seq[seq_id]

            # 시퀀스 경계에서 캐시 초기화
            if args.reinit_per_sequence:
                # 재생성(가장 확실)
                del model
                torch.cuda.empty_cache()
                model = build_model(args.encoder, device)
            else:
                # 메서드가 있으면 리셋 시도
                _ = try_reset_stream_state(model)

            # 프레임 반복
            for i, meta in enumerate(frame_entries):
                if args.stride > 1 and (i % args.stride != 0):
                    continue

                rel_img = meta['image']
                image_path = os.path.join(root_path, rel_img)

                # 출력 경로(.npy)
                infer_path = (os.path.join(args.infer_path, dataset, rel_img)
                              .replace('.jpg', '.npy').replace('.png', '.npy'))

                os.makedirs(os.path.dirname(infer_path), exist_ok=True)

                # 이미지 로드
                img_bgr = cv2.imread(image_path)
                if img_bgr is None:
                    print(f"[경고] 이미지를 읽을 수 없습니다: {image_path}")
                    continue

                # BGR->RGB
                img_rgb = bgr_to_rgb(img_bgr)

                # 단일 프레임 스트리밍 추론
                depth = model.infer_video_depth_one(
                    img_rgb,
                    input_size=args.input_size,
                    device=device,
                    fp32=args.fp32
                )

                # 저장(float32)
                depth = np.asarray(depth, dtype=np.float32)
                np.save(infer_path, depth)

    print("완료: 스트리밍 모드 추론 결과(.npy) 저장이 끝났습니다.")

if __name__ == '__main__':
    main()
