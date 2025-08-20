import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np

from video_depth_anything.video_depth_stream import VideoDepthAnything

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- 기존 인수 설정은 그대로 유지 ---
    parser.add_argument('--infer_path', type=str, default='')
    parser.add_argument('--json_file', type=str, default='')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet', 'nyuv2'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    args = parser.parse_args()
    
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # 모델은 한 번만 로드하여 효율성을 높입니다.
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], use_causal_mask=False)
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    for dataset in args.datasets:
        with open(args.json_file, 'r') as fs:
            path_json = json.load(fs)

        json_data = path_json[dataset]
        root_path = os.path.dirname(args.json_file)

        # tqdm을 사용하여 각 비디오 시퀀스 처리 진행률을 표시합니다.
        for data in tqdm(json_data, desc=f"Processing {dataset}"):
            for key in data.keys():
                value = data[key]
                
                # --- 스트리밍 추론을 위한 핵심 수정 시작 ---

                # 1. 새 비디오 처리에 앞서 모델의 내부 상태를 직접 초기화합니다.
                #    (클래스에 reset 메소드를 추가하는 것과 동일한 효과)
                video_depth_anything.transform = None
                video_depth_anything.frame_cache_list = []
                video_depth_anything.frame_id_list = []
                video_depth_anything.id = -1
                
                # 2. 각 비디오의 프레임을 하나씩 순회합니다.
                for image_info in value:
                    
                    image_path = os.path.join(root_path, image_info['image'])
                    infer_path = (args.infer_path + '/' + dataset + '/' + image_info['image']).replace('.jpg', '.npy').replace('.png', '.npy')
                    os.makedirs(os.path.dirname(infer_path), exist_ok=True)
                    
                    # 프레임 읽기 및 BGR -> RGB 변환 (버그 수정)
                    img = cv2.imread(image_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 3. 단일 프레임 추론 함수 호출
                    depth = video_depth_anything.infer_video_depth_one(
                        img_rgb, 
                        input_size=args.input_size, 
                        device=DEVICE, 
                        fp32=True # 원본 코드의 fp32=True 설정을 유지
                    )

                    # 4. 추론된 깊이 맵을 즉시 저장
                    np.save(infer_path, depth)
                
                # --- 스트리밍 추론을 위한 핵심 수정 종료 ---