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
import copy

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape',
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim, features, use_bn,
            out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe
        )

        # --- 스트리밍 관련 런타임 상태 ---
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []  # 모션 모듈 4곳 캐시 리스트(기존 포맷 유지)
        self.gap = (INFER_LEN - OVERLAP) * 2 - 1 - (OVERLAP - INTERP_LEN)
        assert self.gap == 41
        self.id = -1

    # ----------- public / core -----------
    def forward(
        self,
        x,
        *,
        return_intermediates: bool = False,
        return_qkv: bool = False,
        feature_pool: str = "mean",
    ):
        """
        학습(풀클립) 경로:
          - 기본: [B,T,H,W]
          - return_intermediates=True: {"pred": [B,T,H,W], "intermediates": dict}
        """
        features = self.forward_features(x)
        out = self.forward_depth(
            features, x.shape,
            cached_hidden_state_list=None,  # 학습 풀클립에선 캐시 사용 X
            return_intermediates=return_intermediates,
            return_qkv=return_qkv,
            feature_pool=feature_pool,
        )
        if return_intermediates or return_qkv:
            depth_bt, _cache, intermediates = out
            return {"pred": depth_bt, "intermediates": intermediates}
        depth_bt, _cache = out
        return depth_bt

    def forward_features(self, x):
        # x: [B, T, C, H, W]
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        return features

    def forward_depth(self, features, x_shape, cached_hidden_state_list=None, *, return_intermediates: bool=False, return_qkv: bool=False, feature_pool: str="mean"):
        """
        features: encoder intermediate features of current clip
        cached_hidden_state_list: (옵션) 과거 hidden state (모션 모듈 4곳의 리스트)
        반환:
          - 기본: (depth_bt, cache)
          - return_intermediates=True: (depth_bt, cache, intermediates)
        """
        B, T, C, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14

        if return_intermediates or return_qkv:
            depth_raw = self.head(
                features, patch_h, patch_w, T,
                cached_hidden_state_list=cached_hidden_state_list,
                return_intermediates=True,
                return_qkv=return_qkv,
                feature_pool=feature_pool,
            )
            depth, cur_cache, intermediates = depth_raw
        else:
            depth, cur_cache = self.head(
                features, patch_h, patch_w, T,
                cached_hidden_state_list=cached_hidden_state_list,
            )

        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth_bt = depth.squeeze(1).unflatten(0, (B, T))  # [B,T,H,W]

        if return_intermediates or return_qkv:
            return depth_bt, cur_cache, intermediates
        return depth_bt, cur_cache
    
    @torch.no_grad()
    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
        """
        스트리밍 추론: 프레임을 1장씩 넣고, hidden-state 캐시를 누적 재사용.
        (기존 파이프라인과 동일한 캐시 포맷 유지)
        """
        self.id += 1

        if self.transform is None:  # first frame
            # Initialize the transform
            frame_height, frame_width = frame.shape[:2]
            self.frame_height = frame_height
            self.frame_width = frame_width
            ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
            if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
                input_size = int(input_size * 1.777 / ratio)
                input_size = round(input_size / 14) * 14

            self.transform = Compose([
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])

            # Inference the first frame
            cur_list = [torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0)]
            cur_input = torch.cat(cur_list, dim=1).to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape
                    depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)

            # 초기 캐시 복제(윈도우 시뮬레이션)
            self.frame_cache_list = [copy.deepcopy(cached_hidden_state_list) for _ in range(INFER_LEN)]
            self.frame_id_list.extend([0] * (INFER_LEN - 1))

            new_depth = depth[0][0].cpu().numpy()
        else:
            frame_height, frame_width = frame.shape[:2]
            assert frame_height == self.frame_height
            assert frame_width == self.frame_width

            # infer feature
            cur_input = torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape

            # 과거 캐시 묶기 (기존 포맷 그대로 유지)
            # --- 기존 캐시 윈도우 구성 ---
            cur_list = self.frame_cache_list[0:2] + self.frame_cache_list[-INFER_LEN + 3:]
            # 기대 길이 확인
            assert len(cur_list) == INFER_LEN - 1, f"cache window mismatch: {len(cur_list)} vs {INFER_LEN-1}"

            # --- 안전한 캐시 병합 (None 방지) ---
            def _valid_frame_cache(fc):
                # 각 프레임 캐시는 list/tuple 이고, 모든 엔트리가 Tensor 이어야 함
                return isinstance(fc, (list, tuple)) and all((t is None) or torch.is_tensor(t) for t in fc)

            # 하나라도 형식이 이상하면 캐시 사용 포기
            if not all(_valid_frame_cache(h) for h in cur_list):
                cur_cache = None
            else:
                L = min(len(h) for h in cur_list)  # 레이어 수(혹시 불일치 대비)
                per_layer = []
                cache_ok = True
                for i in range(L):
                    elems = [h[i] for h in cur_list]
                    # 레이어 i에 None이 섞여 있으면 전체 캐시 사용 포기(모델 내부 n 분배가 깨질 수 있음)
                    if any(e is None for e in elems):
                        cache_ok = False
                        break
                    try:
                        per_layer.append(torch.cat(elems, dim=1))  # 시간축 concat
                    except Exception:
                        cache_ok = False
                        break
                cur_cache = per_layer if cache_ok and len(per_layer) == L else None

            # infer depth
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth, new_cache = self.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1), size=(frame_height, frame_width),
                                  mode='bilinear', align_corners=True)
            depth_list = [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]
            new_depth = depth_list[-1]

            # --- 캐시 push (불량 캐시는 마지막 정상값으로 보정) ---
            if new_cache is None or (isinstance(new_cache, (list, tuple)) and any(t is None for t in new_cache)):
                # 마지막 캐시가 존재하면 복제, 아니면 그냥 현재 new_cache(=None) 넣지 않음
                if len(self.frame_cache_list) > 0 and self.frame_cache_list[-1] is not None:
                    self.frame_cache_list.append(self.frame_cache_list[-1])
                else:
                    self.frame_cache_list.append(new_cache)  # 초기 단계 보호
            else:
                self.frame_cache_list.append(new_cache)

        # adjust the sliding window
        self.frame_id_list.append(self.id)
        if self.id + INFER_LEN > self.gap + 1:
            if len(self.frame_id_list) > 1:
                del self.frame_id_list[1]
            if len(self.frame_cache_list) > 1:
                del self.frame_cache_list[1]

        return new_depth
