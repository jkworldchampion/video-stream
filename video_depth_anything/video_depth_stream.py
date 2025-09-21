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

# infer settings, do not change (오프라인 설정에서 쓰던 값들)
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
        cache_len=None,  # enc-cache 길이 (기본: num_frames)
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        # ✨ 헤드는 enc-cache를 입력으로 받아 내부에서 시퀀스 구성/temporal 처리 (나중에 dpt_temporal.py에서 구현)
        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim, features, use_bn,
            out_channels=out_channels, use_clstoken=use_clstoken,
            num_frames=num_frames, pe=pe
        )

        # 스트리밍 상태
        self.transform = None
        self.id = -1
        self.frame_height = None
        self.frame_width = None

        # ✨ enc-cache (프레임별 encoder intermediate features 보관)
        # 각 원소 = 한 프레임의 out_features(list): 레이어별 (tokens[, cls_token]) 튜플과 동일 포맷
        self.enc_feat_cache = []
        self.cache_len = cache_len if cache_len is not None else num_frames

    # ---------------------------------------------------------------
    # 순수 forward (배치/오프라인 용도): enc-cache 없이 현재 입력만 처리
    # ---------------------------------------------------------------
    def forward(self, x):
        feats = self.forward_features(x)
        depth, _ = self.forward_depth(feats, x.shape, cached_enc_feat_list=None)
        return depth

    def forward_features(self, x):
        # x: [B, T, C, H, W]
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0, 1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True
        )
        return features

    def forward_depth(self, features, x_shape, cached_enc_feat_list=None):
        """
        features: 현재 입력의 encoder 인터미디어트 피처(list; 레이어별 (tokens[, cls]) 튜플)
        x_shape:  (B, T, C, H, W)
        cached_enc_feat_list: 과거 프레임들의 encoder 피처 리스트 (길이 ≤ cache_len-1)
        """
        B, T, C, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14

        # ✨ 헤드에 enc-cache를 전달 (헤드가 과거+현재를 묶어 temporal 처리, 최종적으로 현재 프레임 출력)
        depth_logits, _ = self.head(
            features, patch_h, patch_w, T,
            cached_enc_feat_list=cached_enc_feat_list
        )

        # depth_logits: [B*T', 1, h', w'] (보통 T'=1)
        depth = F.interpolate(depth_logits, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        # 반환: [B, T, H, W] (여기선 T=1)
        return depth.squeeze(1).unflatten(0, (B, T)), None

    @torch.no_grad()
    def _push_enc_cache(self, enc_features):
        """
        enc_features: list[ per-layer feature ], 각 feature는 (tokens[, cls_token]) 튜플
        """
        # 캐시 길이 유지 (과거만 보관: 현재 프레임은 호출 이후 append)
        if len(self.enc_feat_cache) >= self.cache_len - 1:
            self.enc_feat_cache.pop(0)
        # 얕은 보관으로 충분 (학습 안 함), 필요시 detach/contiguous
        self.enc_feat_cache.append([f if isinstance(f, tuple)
                                    else (f,)
                                    for f in enc_features])

    # ---------------------------------------------------------------
    # 스트리밍 1-프레임 추론
    # ---------------------------------------------------------------
    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
        """
        frame: np.ndarray (H, W, 3) RGB
        반환: np.ndarray (H, W) depth
        """
        self.id += 1

        # 초기화 또는 해상도 변경 시 transform 리셋
        if self.transform is None:
            frame_height, frame_width = frame.shape[:2]
            self.frame_height = frame_height
            self.frame_width = frame_width

            ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
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
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        else:
            frame_height, frame_width = frame.shape[:2]
            # 해상도 변화는 스트리밍 캐시와 호환되지 않으므로 리셋
            if frame_height != self.frame_height or frame_width != self.frame_width:
                # 리셋
                self.transform = None
                self.enc_feat_cache = []
                self.id = -1
                # 재호출
                return self.infer_video_depth_one(frame, input_size, device, fp32)

        # ----- 현재 프레임 전처리 -----
        cur_input = torch.from_numpy(
            self.transform({'image': frame.astype(np.float32) / 255.0})['image']
        ).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,3,h,w]

        # ----- encoder features 추출 -----
        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                cur_features = self.forward_features(cur_input)  # list per layer
                x_shape = cur_input.shape  # (B=1,T=1,C,H,W)

        # ----- depth 추론 (과거 enc-cache + 현재) -----
        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                depth_bt, _ = self.forward_depth(
                    cur_features, x_shape,
                    cached_enc_feat_list=self.enc_feat_cache if len(self.enc_feat_cache) > 0 else None
                )

        # 업샘플 + numpy 변환
        depth_bt = depth_bt.to(cur_input.dtype)                       # [B=1, T=1, H, W]
        depth_up = F.interpolate(
            depth_bt.flatten(0, 1).unsqueeze(1),  # [B*T,1,H,W]
            size=(self.frame_height, self.frame_width),
            mode='bilinear', align_corners=True
        )  # [1,1,H0,W0]
        new_depth = depth_up[0, 0].cpu().numpy()

        # ----- enc-cache 업데이트 (현재 프레임 추가) -----
        self._push_enc_cache(cur_features)

        return new_depth
