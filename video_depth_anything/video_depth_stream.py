# video_depth_stream.py
import copy
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
        pe='ape',
        # ▼ 스트리밍 옵션
        stream_mode=True,
        select_top_r=None,     # 과거 토큰 Top-R
        update_top_u=None,     # Refiner Top-U
        rope_dt=None,          # RoPE 시간스케일
        return_attn=False,
        return_qkv=False,
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

        # --- 스트리밍 하이퍼파라미터 저장 ---
        self.stream_mode   = stream_mode
        self.select_top_r  = select_top_r
        self.update_top_u  = update_top_u
        self.rope_dt       = rope_dt
        self.return_attn   = return_attn
        self.return_qkv    = return_qkv

    # ----------- public / core -----------
    def forward(self, x):
        return self.forward_depth(self.forward_features(x), x.shape)[0]
    
    def forward_features(self, x):
        # x: [B, T, C, H, W]
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        return features

    def forward_depth(self, features, x_shape, cached_hidden_state_list=None):
        """
        features: encoder intermediate features of current clip
        cached_hidden_state_list: (옵션) 과거 hidden state (모션 모듈 4곳의 리스트)
        """
        B, T, C, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14

        # 스트리밍 옵션을 하위 모듈로 전달 (**kwargs)
        stream_kwargs = {}
        if self.stream_mode is not None:
            stream_kwargs["stream_mode"] = bool(self.stream_mode)
        if self.select_top_r is not None:
            stream_kwargs["select_top_r"] = int(self.select_top_r)
        if self.update_top_u is not None:
            stream_kwargs["update_top_u"] = int(self.update_top_u)
        if self.rope_dt is not None:
            stream_kwargs["rope_dt"] = float(self.rope_dt)
        if self.return_attn:
            stream_kwargs["return_attn"] = True
        if self.return_qkv:
            stream_kwargs["return_qkv"] = True

        depth, cur_cached_hidden_state_list = self.head(
            features, patch_h, patch_w, T,
            cached_hidden_state_list=cached_hidden_state_list,
            **stream_kwargs
        )
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)), cur_cached_hidden_state_list  # [B, T, H, W]
    
    def reset_stream(self):
        self.transform = None
        self.frame_id_list.clear()
        self.frame_cache_list.clear()
        self.id = -1

    @torch.no_grad()
    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
        """
        스트리밍 추론: 프레임을 1장씩 넣고, hidden-state 캐시를 누적 재사용.
        """
        self.id += 1

        if self.transform is None:  # first frame
            # Initialize the transform
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

            # Inference the first frame
            cur_list = [torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0)]
            cur_input = torch.cat(cur_list, dim=1).to(device)

            device_type = str(cur_input.device.type)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, enabled=(not fp32)):
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape
                    depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)

            # 초기 캐시 복제(윈도우 시뮬레이션) - 깊은 복사로 별칭 방지
            self.frame_cache_list = [copy.deepcopy(cached_hidden_state_list) for _ in range(INFER_LEN)]
            self.frame_id_list.extend([0] * (INFER_LEN - 1))

            new_depth = depth[0][0].cpu().numpy()
        else:
            frame_height, frame_width = frame.shape[:2]
            assert frame_height == self.frame_height
            assert frame_width == self.frame_width

            cur_input = torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0).to(device)
            device_type = str(cur_input.device.type)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, enabled=(not fp32)):
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape

            # 과거 캐시 묶기
            cur_list = self.frame_cache_list[0:2] + self.frame_cache_list[-INFER_LEN + 3:]
            assert len(cur_list) == INFER_LEN - 1, f"cache window mismatch: {len(cur_list)} vs {INFER_LEN-1}"

            def _valid_frame_cache(fc):
                return isinstance(fc, (list, tuple)) and all((t is None) or torch.is_tensor(t) for t in fc)

            if not all(_valid_frame_cache(h) for h in cur_list):
                cur_cache = None
            else:
                L = min(len(h) for h in cur_list)
                per_layer = []
                cache_ok = True
                for i in range(L):
                    elems = [h[i] for h in cur_list]
                    if any(e is None for e in elems):
                        cache_ok = False
                        break
                    try:
                        per_layer.append(torch.cat(elems, dim=1))  # 시간축 concat
                    except Exception as e:
                        cache_ok = False
                        # print(f"[cache-merge] layer {i} cat failed: shapes={[tuple(t.shape) for t in elems]} | err={e}")
                        break
                cur_cache = per_layer if cache_ok and len(per_layer) == L else None

            with torch.no_grad():
                with torch.autocast(device_type=device_type, enabled=(not fp32)):
                    depth, new_cache = self.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1), size=(frame_height, frame_width),
                                  mode='bilinear', align_corners=True)
            depth_list = [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]
            new_depth = depth_list[-1]

            # 캐시 push (불량 캐시는 마지막 정상값으로 보정)
            if new_cache is None or (isinstance(new_cache, (list, tuple)) and any(t is None for t in new_cache)):
                if len(self.frame_cache_list) > 0 and self.frame_cache_list[-1] is not None:
                    self.frame_cache_list.append(copy.deepcopy(self.frame_cache_list[-1]))
                else:
                    self.frame_cache_list.append(new_cache)
            else:
                self.frame_cache_list.append(copy.deepcopy(new_cache))

        # adjust the sliding window
        self.frame_id_list.append(self.id)
        if self.id + INFER_LEN > self.gap + 1:
            if len(self.frame_id_list) > 1:
                del self.frame_id_list[1]
            if len(self.frame_cache_list) > 1:
                del self.frame_cache_list[1]

        return new_depth
