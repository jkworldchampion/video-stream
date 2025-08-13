# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#     http://www.apache.org/licenses/LICENSE-2.0 
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.
import argparse
import os
import sys
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

# 프로젝트 루트(../..)를 PYTHONPATH에 추가하여 video_depth_anything 모듈을 찾을 수 있게 함
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

from video_depth_anything.video_depth_stream import VideoDepthAnything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streaming Inference for VideoDepthAnything')
    parser.add_argument('--infer_path', type=str, required=True,
                        help='Directory to save per-frame .npy depth outputs')
    parser.add_argument('--json_file', type=str, required=True,
                        help='Path to dataset JSON metadata (e.g., benchmark/datasets/scannet/scannet_video.json)')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet'],
                        help='Dataset names to process (keys in the JSON)')
    parser.add_argument('--input_size', type=int, default=518,
                        help='Input resolution for the model')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'],
                        help='Backbone encoder type')
    parser.add_argument('--fp32', action='store_true',
                        help='Run inference in float32 (default is float16)')
    parser.add_argument('--use_causal_mask', action='store_true',
                        help='Enable causal masking for streaming temporal attention')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    cfg = model_configs[args.encoder]

    # Initialize model for streaming
    model = VideoDepthAnything(use_causal_mask=args.use_causal_mask, num_frames=32, **cfg)
    # ckpt = os.path.join(BASE_DIR, 'checkpoints', f'video_depth_anything_{args.encoder}.pth')
    ckpt = os.path.join(BASE_DIR, 'outputs', 'experiment_10', f'best_model.pth')
    ckpt_data = torch.load(ckpt, map_location='cpu')
    raw_sd    = ckpt_data['model_state_dict']
    fixed_sd  = {}
    for k,v in raw_sd.items():
        # remove the "module." prefix if present
        name = k[len("module."):] if k.startswith("module.") else k
        fixed_sd[name] = v
    
    model.load_state_dict(fixed_sd, strict=True)
    model = model.to(DEVICE).eval()

    # Load JSON metadata
    root_path = os.path.dirname(args.json_file)
    with open(args.json_file, 'r') as f:
        meta = json.load(f)

    for dataset in args.datasets:
        scenes = meta.get(dataset, [])

        for entry in tqdm(scenes, desc=f"Streaming {dataset}"):
            scene = list(entry.keys())[0]
            frames = entry[scene]

            # 이미지를 streaming처럼 하나씩 전달
            for info in frames:
                rel_img = info['image']  # e.g. "scannet/scene0019_01/color/0.jpg"
                img_path = os.path.join(root_path, rel_img)
                img = cv2.imread(img_path)
                img = img[:, :, ::-1]  # BGR -> RGB

                # Streaming inference: one frame at a time
                depth = model.infer_video_depth_one(
                    img,
                    input_size=args.input_size,
                    device=DEVICE,
                    fp32=args.fp32,
                )

                # Save .npy
                save_rel = rel_img.replace('.jpg', '.npy').replace('.png', '.npy')
                save_path = os.path.join(args.infer_path, dataset, save_rel)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, depth)

    print("Streaming .npy outputs generated.")
