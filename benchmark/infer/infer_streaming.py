#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np

# 프로젝트 루트(../..)를 PYTHONPATH에 추가
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

from video_depth_anything.video_depth_stream import VideoDepthAnything


def reset_stream_state(model):
    """시퀀스 시작 시 스트리밍 상태 초기화 (저자 릴리스 호환)"""
    model.transform = None
    model.frame_cache_list = []
    model.frame_id_list = []
    model.id = -1


def read_ckpt_state_dict(ckpt_path):
    """체크포인트 로드 + state_dict 반환 ('module.' prefix 제거 포함)"""
    ckpt_data = torch.load(ckpt_path, map_location='cpu')
    raw_sd = ckpt_data.get('model_state_dict', ckpt_data.get('state_dict', ckpt_data))
    fixed = {}
    for k, v in raw_sd.items():
        name = k[7:] if k.startswith('module.') else k
        fixed[name] = v
    return fixed


def infer_variant_from_state_dict(sd):
    """
    체크포인트에서 임베딩 차원/헤드 채널을 읽어 encoder 변형 추정.
    - embed_dim 384 → vits
    - embed_dim 1024 → vitl
    """
    # 우선순위: pretrained.pos_embed or cls_token or patch_embed.proj.weight
    embed_dim = None
    if 'pretrained.pos_embed' in sd:
        embed_dim = sd['pretrained.pos_embed'].shape[-1]
    elif 'pretrained.cls_token' in sd:
        embed_dim = sd['pretrained.cls_token'].shape[-1]
    elif 'pretrained.patch_embed.proj.weight' in sd:
        embed_dim = sd['pretrained.patch_embed.proj.weight'].shape[0]

    if embed_dim == 384:
        return 'vits', 384
    if embed_dim == 1024:
        return 'vitl', 1024

    # fallback: head.projects.0.weight shape [C, embed_dim, 1,1]
    for k in ['head.projects.0.weight', 'head.projects.1.weight']:
        if k in sd and sd[k].ndim >= 2:
            e = sd[k].shape[1]
            if e == 384:
                return 'vits', 384
            if e == 1024:
                return 'vitl', 1024

    raise RuntimeError("체크포인트에서 encoder 변형(vits/vitl)을 추정할 수 없습니다.")


def filter_unmatched_keys_for_load(sd, model):
    """
    모델에 존재하지 않는 키는 제거 (Unexpected key 방지).
    shape mismatch는 strict=False로 우회.
    """
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in model_keys}
    dropped = [k for k in sd.keys() if k not in model_keys]
    return filtered, dropped


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='', help='출력 .npy 저장 루트')
    parser.add_argument('--json_file', type=str, default='', help='비디오 프레임 리스트 JSON')
    # 2번 코드 스타일: 기본값 scannet, JSON은 루프 밖에서 한 번만 로드
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet'])
    parser.add_argument('--input_size', type=int, default=518)
    # 인코더 인자 제공(선택); 미지정 또는 불일치 시 체크포인트로 자동 교정
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    args = parser.parse_args()

    # 1번 코드 스타일: 디바이스 선택 유지
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 1번 코드: 실험 가중치 경로 그대로 사용
    ckpt_path = os.path.join(BASE_DIR, 'outputs', 'experiment_11', 'best_model.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")

    # 체크포인트 먼저 읽고 변형 자동 판별
    sd_fixed = read_ckpt_state_dict(ckpt_path)
    ckpt_encoder, embed_dim = infer_variant_from_state_dict(sd_fixed)

    # args.encoder와 불일치하면 체크포인트 기준으로 강제 교정
    if args.encoder != ckpt_encoder:
        print(f"[경고] --encoder={args.encoder} 이(가) 체크포인트({ckpt_encoder}, embed_dim={embed_dim})와 불일치 → "
              f"{ckpt_encoder}로 자동 교정합니다.")
        args.encoder = ckpt_encoder

    # 모델 구성 (1번 코드 구성 유지)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # causal mask 사용 유지
    vda = VideoDepthAnything(**model_configs[args.encoder], use_causal_mask=True)

    # 모델에 없는 키는 제거하고 로드 (Unexpected key 차단)
    sd_filtered, dropped = filter_unmatched_keys_for_load(sd_fixed, vda)
    if dropped:
        print(f"[정보] 모델에 없는 {len(dropped)}개 키 무시 (예: {dropped[:5]} ...)")

    # shape mismatch 가능성을 고려해 strict=False
    missing, unexpected = vda.load_state_dict(sd_filtered, strict=False)
    if missing:
        print(f"[주의] 누락된 키 {len(missing)}개: (예: {missing[:5]} ...)")
    if unexpected:
        print(f"[주의] 예기치 않은 키 {len(unexpected)}개: (예: {unexpected[:5]} ...)")

    vda = vda.to(DEVICE).eval()

    # JSON 한 번만 로드
    if not args.json_file:
        raise ValueError('--json_file 경로를 지정하세요.')
    with open(args.json_file, 'r') as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(args.json_file)

    # autograd 비활성화: 2번 코드 스타일
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        for dataset in args.datasets:
            if dataset not in path_json:
                raise KeyError(f"JSON에 '{dataset}' 키가 없습니다.")
            json_data = path_json[dataset]

            for data in tqdm(json_data, desc=f"Streaming {dataset}"):
                for key, frames in data.items():
                    # 시퀀스 시작마다 스트리밍 상태 리셋
                    reset_stream_state(vda)

                    for item in frames:
                        # 입력/출력 경로
                        img_path = os.path.join(root_path, item['image'])
                        base, _ = os.path.splitext(item['image'])
                        out_path = os.path.join(args.infer_path, dataset, base + '.npy')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)

                        # BGR -> RGB
                        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if img_bgr is None:
                            raise FileNotFoundError(img_path)
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                        # 스트리밍 1프레임 추론
                        depth = vda.infer_video_depth_one(
                            img_rgb, input_size=args.input_size, device=DEVICE, fp32=True
                        )
                        np.save(out_path, depth)

    print("✅ Streaming inference 완료")
