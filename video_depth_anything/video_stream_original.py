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
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
import numpy as np

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
INTERP_LEN = 8
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []
        self.gap = (INFER_LEN - OVERLAP) * 2 - 1 - (OVERLAP - INTERP_LEN)
        assert self.gap == 41
        self.id = -1

    def forward(self, x):
        return self.forward_depth(self.forward_features(x), x.shape)[0]
    
    def forward_features(self, x):
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        return features

    def forward_depth(self, features, x_shape, cached_hidden_state_list=None):
        B, T, C, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14
        depth, cur_cached_hidden_state_list = self.head(features, patch_h, patch_w, T, cached_hidden_state_list=cached_hidden_state_list)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)), cur_cached_hidden_state_list # return shape [B, T, H, W]
    
    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
        """
        스트리밍(1프레임) 추론 + '윈도우 0·1번째 쌍' 기반의 즉시 scale&shift 정합.
        - 현재 윈도우의 [0], [1] 깊이와 직전 윈도우의 [0], [1] 깊이를 맞춰 (a,b) 추정,
        곧바로 현재 프레임(마지막 인덱스)의 깊이에 적용.
        """
        # --- 정합 관련 상태 초기화 (최초 1회) ---
        if not hasattr(self, 'ss_scale'):
            self.ss_scale = 1.0   # 현재 유효한 scale
        if not hasattr(self, 'ss_shift'):
            self.ss_shift = 0.0   # 현재 유효한 shift
        if not hasattr(self, 'prev_pair'):  # 직전 윈도우의 (0,1) 기준 깊이쌍 (정합 적용본)
            self.prev_pair = None

        # --- 프레임 카운터 ---
        self.id += 1

        # --- 첫 프레임 분기: transform / cache 부트스트랩 (원본 유지) ---
        if self.transform is None:  # first frame
            frame_height, frame_width = frame.shape[:2]
            self.frame_height = frame_height
            self.frame_width = frame_width

            ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
            if ratio > 1.78:
                input_size = int(input_size * 1.777 / ratio)
                input_size = round(input_size / 14) * 14

            self.transform = Compose([
                Resize(
                    width=input_size, height=input_size,
                    resize_target=False, keep_aspect_ratio=True,
                    ensure_multiple_of=14, resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])

            # 1프레임 추론 (이때는 윈도우 길이가 1일 수 있음 → 0,1 쌍 정합 불가)
            cur_list = [torch.from_numpy(
                self.transform({'image': frame.astype(np.float32) / 255.0})['image']
            ).unsqueeze(0).unsqueeze(0)]
            cur_input = torch.cat(cur_list, dim=1).to(device)

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape
                    depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(
                depth.flatten(0,1).unsqueeze(1),
                size=(frame_height, frame_width),
                mode='bilinear', align_corners=True
            )
            depth_np_raw = depth[0][0].cpu().numpy()

            # 캐시 윈도우를 INFER_LEN 길이로 부트스트랩
            self.frame_cache_list = [cached_hidden_state_list] * INFER_LEN
            self.frame_id_list.extend([0] * (INFER_LEN - 1))

            # 첫 프레임 출력: 현재 (a,b) 적용(초기값 1,0)
            new_depth_np = depth_np_raw * self.ss_scale + self.ss_shift
            new_depth_np[new_depth_np < 0] = 0.0

            # prev_pair는 아직 만들 수 없음(윈도우 길이<2). 다음 스텝부터 생성.
            return new_depth_np

        # --- 이후 프레임: 공통 경로 ---
        frame_height, frame_width = frame.shape[:2]
        assert frame_height == self.frame_height and frame_width == self.frame_width

        # 현재 프레임 feature
        cur_input = torch.from_numpy(
            self.transform({'image': frame.astype(np.float32) / 255.0})['image']
        ).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                cur_feature = self.forward_features(cur_input)
                x_shape = cur_input.shape

        # 캐시 윈도우 구성 (원본 유지)
        cur_list = self.frame_cache_list[0:2] + self.frame_cache_list[-INFER_LEN+3:]
        assert len(cur_list) == INFER_LEN - 1
        cur_cache = [torch.cat([h[i] for h in cur_list], dim=1) for i in range(len(cur_list[0]))]

        # 깊이 추론
        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                depth, new_cache = self.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)

        depth = depth.to(cur_input.dtype)
        depth = F.interpolate(
            depth.flatten(0,1).unsqueeze(1),
            size=(frame_height, frame_width),
            mode='bilinear', align_corners=True
        )
        depth_list = [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

        # --- (a,b) 갱신: 현재 윈도우의 [0], [1] vs (직전 prev_pair의 [0], [1]) ---
        # depth_list 길이가 2 이상일 때만 수행
        if len(depth_list) >= 2:
            cur_pair_raw_0 = depth_list[0]
            cur_pair_raw_1 = depth_list[1]

            if self.prev_pair is not None:
                # ref: 직전 윈도우의 기준쌍(이미 정합 적용본)
                prev0, prev1 = self.prev_pair

                try:
                    # 두 장을 세로로 붙여 (a,b) 최소자승 추정
                    cur_stack  = np.concatenate([cur_pair_raw_0, cur_pair_raw_1])
                    prev_stack = np.concatenate([prev0,          prev1         ])
                    mask_stack = np.concatenate([np.ones_like(prev0, dtype=bool),
                                                np.ones_like(prev1, dtype=bool)])
                    a, b = compute_scale_and_shift(cur_stack, prev_stack, mask_stack)

                    # 즉시 갱신
                    self.ss_scale = float(a)
                    self.ss_shift = float(b)

                except Exception:
                    # 정합 실패 시 직전 (a,b) 유지
                    pass

            # 다음 스텝을 위한 prev_pair 갱신: 이번 cur_pair에 (a,b) 적용본을 저장
            adj0 = cur_pair_raw_0 * self.ss_scale + self.ss_shift
            adj1 = cur_pair_raw_1 * self.ss_scale + self.ss_shift
            adj0[adj0 < 0] = 0.0
            adj1[adj1 < 0] = 0.0
            self.prev_pair = (adj0, adj1)

        # --- 현재 프레임(마지막 인덱스)의 깊이에 (a,b) 적용하여 반환 ---
        new_depth_np = depth_list[-1] * self.ss_scale + self.ss_shift
        new_depth_np[new_depth_np < 0] = 0.0

        # 캐시 & 윈도우 관리 (원본 유지)
        self.frame_cache_list.append(new_cache)
        self.frame_id_list.append(self.id)
        if self.id + INFER_LEN > self.gap + 1:
            del self.frame_id_list[1]
            del self.frame_cache_list[1]

        return new_depth_np