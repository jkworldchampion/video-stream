import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np

from video_depth_anything.video_depth_stream import VideoDepthAnything

def reset_streaming_state(model):
    """스트리밍 상태를 초기화합니다. (enc-cache 전용)"""
    # 해상도/전처리 초기화
    if hasattr(model, 'transform'):
        model.transform = None
    # enc-cache 비우기
    if hasattr(model, 'enc_feat_cache'):
        model.enc_feat_cache.clear()
    # 메타 관리 배열 초기화
    if hasattr(model, 'frame_id_list'):
        model.frame_id_list.clear()
    # 구(旧) 구현 잔재가 남아있을 수 있으니 방어적으로 처리
    if hasattr(model, 'frame_cache_list'):
        try:
            model.frame_cache_list.clear()
        except Exception:
            pass
    # 프레임 id 리셋
    if hasattr(model, 'id'):
        model.id = -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='')
    parser.add_argument('--json_file', type=str, default='')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--pe', type=str, default='ape', choices=['ape', 'rope'])
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    vda = VideoDepthAnything(**model_configs[args.encoder], pe=args.pe)
    # checkpoint load
    ckpt = torch.load('./checkpoints/video_depth_anything_vits.pth', map_location='cpu')
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt  # 방어적
    
    # DataParallel로 저장된 경우 'module.' 프리픽스 제거, 혹시 'student.' 프리픽스도 제거
    from collections import OrderedDict
    clean_state = OrderedDict()
    for k, v in state.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('student.'):   # 혹시 모를 프리픽스 방어
            nk = nk[len('student.'):]
        clean_state[nk] = v

    # 버전/이름 차이가 조금이라도 있으면 strict=False 권장
    missing, unexpected = vda.load_state_dict(clean_state, strict=True)
    print('missing:', missing)
    print('unexpected:', unexpected)
    
    vda = vda.to(DEVICE).eval()

    with open(args.json_file, 'r') as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(args.json_file)

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

                        # --- 안전한 이미지 로딩 ---
                        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise FileNotFoundError(f"Failed to read image: {img_path}")
                        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                        # --- 핵심: 프레임을 1장씩 넣어 캐시 재사용 ---
                        depth_np = vda.infer_video_depth_one(
                            img, input_size=args.input_size, device=DEVICE, fp32=True,
                        )
                        # infer_video_depth_one이 이미 numpy array를 반환하므로 바로 저장
                        np.save(out_path, depth_np)

