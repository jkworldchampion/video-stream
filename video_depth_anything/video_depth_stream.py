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
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            num_frames=num_frames,
            pe=pe,
        )

        # streaming inference용 state
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []
        self.gap = (INFER_LEN - OVERLAP) * 2 - 1 - (OVERLAP - INTERP_LEN)
        assert self.gap == 41
        self.id = -1

    def forward(self, x):
        return self.forward_depth(self.forward_features(x), x.shape)[0]

    def forward_features(self, x):
        # x: [B,T,3,H,W] 또는 [B,1,3,H,W]
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0, 1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        return features

    def forward_depth(self, features, x_shape, cached_hidden_state_list=None):
        B, T, C, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14
        depth, cur_cached_hidden_state_list = self.head(
            features,
            patch_h,
            patch_w,
            T,
            cached_hidden_state_list=cached_hidden_state_list,
        )
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        # depth: [B*T,1,H,W] → [B,T,H,W]
        return depth.squeeze(1).unflatten(0, (B, T)), cur_cached_hidden_state_list

    def stream_step_train(self, x_t, cache_state=None):
        """
        x_t: [B,1,3,H,W]
        cache_state: None이면 새 시퀀스 시작, 아니면 이전 step에서 받은 dict
        return:
            pred_t: [B,H,W]
            cache_state: 업데이트된 dict
        """
        if cache_state is None:
            cache_state = {
                "frame_cache_list": [],
                "frame_id_list": [],
                "id": -1,
            }

        cache_state["id"] += 1

        feats = self.forward_features(x_t)   # [B,1,...]
        x_shape = x_t.shape                  # [B,1,3,H,W]

        # train: detach_past=True
        pred_t_net, cache_state = self._stream_step_core(
            feats,
            x_shape,
            cache_state=cache_state,
            detach_past=True,
        )
        # pred_t_net: [B,H,W]
        return pred_t_net, cache_state

    def _stream_step_core(self, cur_feature, x_shape, cache_state, detach_past: bool):
        """
        cur_feature: [B, 1, ...] 한 프레임에 해당하는 encoder feature
        x_shape:     [B, 1, 3, H, W]
        cache_state: {
            "frame_cache_list": list[hidden_state_list],
            "frame_id_list":    list[int],
            "id":               int,  # 현재 frame index
        }
        detach_past:
            - train: True  (과거 cache는 detach 해서 그래프 끊기)
            - infer: False (어차피 no_grad)
        """
        frame_cache_list = cache_state["frame_cache_list"]
        frame_id_list    = cache_state["frame_id_list"]
        cur_id           = cache_state["id"]

        B, T, C, H, W = x_shape
        assert T == 1, f"stream step expects T=1, got T={T}"

        # 1) 첫 프레임: cache 없음
        if len(frame_cache_list) == 0:
            depth, cached_hidden_state_list = self.forward_depth(
                cur_feature, x_shape, cached_hidden_state_list=None
            )  # depth: [B,1,H,W]

            if detach_past:
                cached_hidden_state_list = [
                    h.detach() for h in cached_hidden_state_list
                ]

            # INFER_LEN 길이의 window로 초기화
            frame_cache_list = [cached_hidden_state_list for _ in range(INFER_LEN)]
            frame_id_list = [cur_id for _ in range(INFER_LEN)]

            new_depth = depth[:, 0]  # [B,H,W]

            cache_state["frame_cache_list"] = frame_cache_list
            cache_state["frame_id_list"]    = frame_id_list
            return new_depth, cache_state

        # 2) 이후 프레임: 기존 infer_video_depth_one과 동일한 window 규칙
        cur_list = frame_cache_list[0:2] + frame_cache_list[-INFER_LEN + 3:]
        assert len(cur_list) == INFER_LEN - 1, \
            f"cache window mismatch: {len(cur_list)} vs {INFER_LEN-1}"

        # 레이어별 cat
        cur_cache = [
            torch.cat([h[i] for h in cur_list], dim=1)
            for i in range(len(cur_list[0]))
        ]

        depth, new_cache = self.forward_depth(
            cur_feature, x_shape, cached_hidden_state_list=cur_cache
        )  # depth: [B,1,H,W]

        if detach_past:
            new_cache = [h.detach() for h in new_cache]

        frame_cache_list.append(new_cache)
        frame_id_list.append(cur_id)

        # sliding window 규칙 그대로 사용
        if cur_id + INFER_LEN > self.gap + 1:
            del frame_cache_list[1]
            del frame_id_list[1]

        cache_state["frame_cache_list"] = frame_cache_list
        cache_state["frame_id_list"]    = frame_id_list

        new_depth = depth[:, 0]  # [B,H,W]
        return new_depth, cache_state

    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
        """
        frame: H,W,3 (numpy, BGR or RGB 상관 없이 transform에서 처리)
        return: depth (H_orig,W_orig) numpy
        """
        self.id += 1

        # 1) transform 초기화 (첫 프레임에서만)
        if self.transform is None:
            frame_height, frame_width = frame.shape[:2]
            self.frame_height = frame_height
            self.frame_width = frame_width

            ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
            # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            if ratio > 1.78:
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
                NormalizeImage(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                PrepareForNet(),
            ])

        # 2) 입력 크기 일관성 체크
        frame_height, frame_width = frame.shape[:2]
        assert frame_height == self.frame_height
        assert frame_width == self.frame_width

        # 3) 현재 프레임 전처리 → [1,1,3,H,W]
        cur_input = torch.from_numpy(
            self.transform({'image': frame.astype(np.float32) / 255.0})['image']
        ).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                cur_feature = self.forward_features(cur_input)
                x_shape = cur_input.shape  # [1,1,3,H,W]

                cache_state = {
                    "frame_cache_list": self.frame_cache_list,
                    "frame_id_list": self.frame_id_list,
                    "id": self.id,
                }

                # infer: detach_past=False
                depth_net, cache_state = self._stream_step_core(
                    cur_feature,
                    x_shape,
                    cache_state=cache_state,
                    detach_past=False,
                )

        # 4) 내부 state 업데이트
        self.frame_cache_list = cache_state["frame_cache_list"]
        self.frame_id_list    = cache_state["frame_id_list"]

        # 5) network 해상도 → 원본 해상도로 업샘플
        depth_net = depth_net.to(cur_input.dtype)              # [1,H,W]
        depth_up  = F.interpolate(
            depth_net.unsqueeze(1),                            # [1,1,H,W]
            size=(frame_height, frame_width),
            mode='bilinear',
            align_corners=True,
        )

        new_depth = depth_up[0, 0].cpu().numpy()               # [H,W]
        return new_depth
