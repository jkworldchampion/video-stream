import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np

from video_depth_anything.video_depth import VideoDepthAnything

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='')
    parser.add_argument('--json_file', type=str, default='')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    args = parser.parse_args()
   
    for dataset in args.datasets:

        with open(args.json_file, 'r') as fs:
            path_json = json.load(fs)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
        video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to(DEVICE).eval()
        
        json_data = path_json[dataset]
        root_path = os.path.dirname(args.json_file)

        # 오프라인 추론
        with torch.inference_mode():
            for data in tqdm(json_data, desc=f'Offline {dataset}'):
                for key in data.keys():
                    value = data[key]
                    infer_paths = []
                    videos = []

                    for images in value:
                        image_path = os.path.join(root_path, images['image'])

                        # 저장 규칙: eval.py와 동일하게 (<infer>/<dataset>/<image>.npy)
                        infer_path = (os.path.join(args.infer_path, dataset, images['image']).replace('.jpg', '.npy').replace('.png', '.npy'))
                        infer_paths.append(infer_path)

                        # BGR -> RGB (저자 이자식!!)
                        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        if img_bgr is None:
                            raise FileNotFoundError(image_path)
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        videos.append(img_rgb)

                    videos = np.stack(videos, axis=0)  # [T, H, W, 3]
                    target_fps = 1

                    # 원형 유지: 전체 시퀀스 단일 호출
                    depths, fps = video_depth_anything.infer_video_depth(
                        videos, target_fps, input_size=args.input_size, device=DEVICE, fp32=True
                    )  # depths: [T, H, W] (numpy)

                    # 저장 (디렉토리 생성 포함)
                    for i in range(len(infer_paths)):
                        infer_path = infer_paths[i]
                        os.makedirs(os.path.dirname(infer_path), exist_ok=True)
                        depth = depths[i]
                        np.save(infer_path, depth.astype(np.float32))
                    