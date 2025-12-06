import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

from video_depth_anything.video_depth_stream import VideoDepthAnything


def reset_streaming_state(model):
    """
    VideoDepthAnything 스트리밍 상태 초기화.
    DataParallel 여부와 상관없이 내부 state를 리셋한다.
    """
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "transform"):
        m.transform = None
    if hasattr(m, "frame_cache_list"):
        m.frame_cache_list = []
    if hasattr(m, "frame_id_list"):
        m.frame_id_list = []
    if hasattr(m, "id"):
        m.id = -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='')
    parser.add_argument('--json_file', type=str, default='')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--pe', type=str, default='ape', choices=['ape', 'rope', 'none'])
    parser.add_argument('--checkpoint', type=str, default='./outputs/experiment_2/best_model.pth',
        help='Path to model checkpoint (e.g., ./outputs/experiment_2/best_model.pth)')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # ───────────────── 모델 생성 ─────────────────
    vda = VideoDepthAnything(**model_configs[args.encoder], pe=args.pe)

    # checkpoint 로드 (train.py 에서 저장한 형식과 호환)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)  # 방어적

    clean_state = OrderedDict()
    for k, v in state.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('student.'):
            nk = nk[len('student.'):]
        clean_state[nk] = v

    # strict=True로 맞추되, missing/unexpected는 안전하게 출력
    load_res = vda.load_state_dict(clean_state, strict=True)
    # PyTorch 버전에 따라 반환 타입이 다를 수 있어, hasattr 체크
    if hasattr(load_res, "missing_keys") and hasattr(load_res, "unexpected_keys"):
        print('missing keys:', load_res.missing_keys)
        print('unexpected keys:', load_res.unexpected_keys)
    else:
        # 구버전 호환용 (tuple 형태로 오는 경우)
        try:
            missing, unexpected = load_res
            print('missing keys:', missing)
            print('unexpected keys:', unexpected)
        except Exception:
            pass

    vda = vda.to(DEVICE).eval()

    # ───────────────── JSON 로드 ─────────────────
    with open(args.json_file, 'r') as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(args.json_file)

    # ───────────────── 스트리밍 추론 ─────────────────
    for dataset in args.datasets:
        json_data = path_json[dataset]
        for data in tqdm(json_data, desc=f"Streaming {dataset}"):
            for key in data.keys():
                frames = data[key]  # 이 시퀀스의 프레임 리스트

                # 스트리밍 상태 리셋
                reset_streaming_state(vda)

                with torch.inference_mode():
                    for item in frames:
                        # 입력 이미지 경로/출력 경로 구성
                        img_path = os.path.join(root_path, item['image'])
                        base, _ = os.path.splitext(item['image'])
                        out_path = os.path.join(args.infer_path, dataset, base + '.npy')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)

                        # 안전한 BGR -> RGB 변환
                        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise FileNotFoundError(img_path)
                        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                        # --- 핵심: 프레임을 1장씩 넣어 캐시 재사용 ---
                        depth_np = vda.infer_video_depth_one(
                            img,
                            input_size=args.input_size,
                            device=DEVICE,
                            fp32=True,
                        )
                        # infer_video_depth_one 이 numpy array 반환 → 바로 저장
                        np.save(out_path, depth_np)
