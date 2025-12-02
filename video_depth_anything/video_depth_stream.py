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

    # ----------------------- basic forward (clip) -----------------------
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

    def forward_depth(
        self,
        features,
        x_shape,
        cached_hidden_state_list=None,
        return_intermediates: bool = False,
        return_qkv: bool = False,
        kd_layers=None,
    ):
        """
        head 래핑:
          - depth_bt: [B,T,H,W]
          - cur_cached_hidden_state_list: motion hidden state 리스트
          - intermediates: {"qkv": {layer_id: {"q": [BN,Hh,F,D], ...}}}
        """
        B, T, C, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14

        if return_intermediates or return_qkv:
            depth, cur_cached_hidden_state_list, intermediates = self.head(
                features,
                patch_h,
                patch_w,
                T,
                cached_hidden_state_list=cached_hidden_state_list,
                return_intermediates=True,
                return_qkv=return_qkv,
                kd_layers=kd_layers,
            )
        else:
            depth, cur_cached_hidden_state_list = self.head(
                features,
                patch_h,
                patch_w,
                T,
                cached_hidden_state_list=cached_hidden_state_list,
            )
            intermediates = None

        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        # depth: [B*T,1,H,W] → [B,T,H,W]
        depth_bt = depth.squeeze(1).unflatten(0, (B, T))
        return depth_bt, cur_cached_hidden_state_list, intermediates

    # ----------------------- streaming (train) -----------------------
    def stream_step_train(
        self,
        x_t,
        cache_state=None,
        return_kd: bool = False,
        kd_layers=None,
    ):
        """
        x_t      : [B,1,3,H,W]
        cache_state:
            - None이면 새 시퀀스 시작
            - dict이면 이전 step 상태 이어받기
        return:
            - return_kd=False → pred_t_net, cache_state
            - return_kd=True  → pred_t_net, cache_state, student_kd_t
        """
        if cache_state is None:
            cache_state = {
                "frame_cache_list": [],
                "frame_id_list": [],
                "id": -1,
            }

        cache_state["id"] += 1

        feats   = self.forward_features(x_t)  # [B,1,...]
        x_shape = x_t.shape                   # [B,1,3,H,W]

        pred_t_net, cache_state, student_kd_t = self._stream_step_core(
            feats,
            x_shape,
            cache_state=cache_state,
            detach_past=True,          # train 모드에서는 과거 cache detach
            return_kd=return_kd,
            kd_layers=kd_layers,
        )

        if return_kd:
            return pred_t_net, cache_state, student_kd_t
        else:
            return pred_t_net, cache_state

    def _stream_step_core(
        self,
        cur_feature,
        x_shape,
        cache_state,
        detach_past: bool,
        return_kd: bool = False,
        kd_layers=None,
    ):
        """
        내부 streaming step 핵심 로직.

        return_kd=False → (new_depth, cache_state, None)
        return_kd=True  → (new_depth, cache_state, student_kd_t)
            student_kd_t["qkv"][lid]["q"]: [B, heads, N, D]
        """
        frame_cache_list = cache_state["frame_cache_list"]
        frame_id_list    = cache_state["frame_id_list"]
        cur_id           = cache_state["id"]

        B, T, C, H, W = x_shape
        assert T == 1, f"stream step expects T=1, got T={T}"

        student_kd_t = None  # 기본값

        # -------------------- 1) 첫 프레임: cache 없음 --------------------
        if len(frame_cache_list) == 0:
            depth_bt, cached_hidden_state_list, intermediates = self.forward_depth(
                cur_feature,
                x_shape,
                cached_hidden_state_list=None,
                return_intermediates=return_kd,
                return_qkv=return_kd,
                kd_layers=kd_layers,
            )

            if detach_past and cached_hidden_state_list is not None:
                cached_hidden_state_list = [h.detach() for h in cached_hidden_state_list]

            # INFER_LEN 길이의 window로 초기화
            frame_cache_list = [cached_hidden_state_list for _ in range(INFER_LEN)]
            frame_id_list    = [cur_id for _ in range(INFER_LEN)]

            # 현재 프레임 depth
            new_depth = depth_bt[:, 0]  # [B,H,W]

            # KD 추출 (T=1)
            if return_kd:
                student_kd_t = self._extract_kd_from_intermediates(
                    intermediates,
                    B=B,
                    kd_layers=kd_layers,
                )

            cache_state["frame_cache_list"] = frame_cache_list
            cache_state["frame_id_list"]    = frame_id_list
            return new_depth, cache_state, student_kd_t

        # -------------------- 2) 이후 프레임: 기존 streaming window --------------------
        # 과거 cache window 구성 (infer_video_depth_one 과 동일 규칙)
        cur_list = frame_cache_list[0:2] + frame_cache_list[-INFER_LEN + 3:]
        assert len(cur_list) == INFER_LEN - 1, \
            f"cache window mismatch: {len(cur_list)} vs {INFER_LEN-1}"

        # 레이어별 cat
        cur_cache = [
            torch.cat([h[i] for h in cur_list], dim=1)
            for i in range(len(cur_list[0]))
        ]

        depth_bt, new_cache, intermediates = self.forward_depth(
            cur_feature,
            x_shape,
            cached_hidden_state_list=cur_cache,
            return_intermediates=return_kd,
            return_qkv=return_kd,
            kd_layers=kd_layers,
        )

        if detach_past and new_cache is not None:
            new_cache = [h.detach() for h in new_cache]

        frame_cache_list.append(new_cache)
        frame_id_list.append(cur_id)

        # sliding window 규칙 유지
        if cur_id + INFER_LEN > self.gap + 1:
            del frame_cache_list[1]
            del frame_id_list[1]

        cache_state["frame_cache_list"] = frame_cache_list
        cache_state["frame_id_list"]    = frame_id_list

        # 현재 프레임 depth
        new_depth = depth_bt[:, 0]  # [B,H,W]

        # KD 추출 (현재 step)
        if return_kd:
            student_kd_t = self._extract_kd_from_intermediates(
                intermediates,
                B=B,
                kd_layers=kd_layers,
            )

        return new_depth, cache_state, student_kd_t

    def _extract_kd_from_intermediates(self, intermediates, B: int, kd_layers=None):
        """
        DPTHeadTemporal 가 넘겨준 intermediates["qkv"] 에서
        현재 step(t)에 해당하는 Q/K/V를 [B, heads, N, D] 로 변환.

        DPTHeadTemporal 이 저장하는 포맷은 다음과 같이 가정한다:
          q, k, v : [BN, F, Hh, D]
            - BN = B * N_tokens_per_frame
            - F  = temporal window length (cache 포함)
            - Hh = num_heads
            - D  = head_dim

        여기서는 마지막 time-step(F-1)을 "현재 프레임 t" 로 보고,
        그 위치의 Q/K/V를 꺼내서 KD 용도로 사용한다.

        반환:
          {"qkv": { layer_id: {"q": [B,Hh,N,D], "k": [...], "v": [...] } } }
        """
        if intermediates is None or "qkv" not in intermediates:
            return None

        qkv_src = intermediates["qkv"]
        if not qkv_src:
            return None

        if kd_layers is None:
            kd_layers = sorted(qkv_src.keys())

        qkv_out = {}

        for lid in kd_layers:
            if lid not in qkv_src:
                continue

            q = qkv_src[lid]["q"]
            k = qkv_src[lid]["k"]
            v = qkv_src[lid]["v"]

            # ----- 기본 shape 체크 -----
            if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
                raise ValueError(
                    f"[student] Unexpected q/k/v dim at layer {lid}: "
                    f"q={q.shape}, k={k.shape}, v={v.shape}"
                )

            # q, k, v 모두 [BN, F, Hh, D] 포맷으로 가정
            BN, Fq, Hh, D = q.shape
            BNk, Fk, Hhk, Dk = k.shape
            BNv, Fv, Hhv, Dv = v.shape

            # k, v가 q와 일관된지 확인
            if not (BNk == BN and BNv == BN and
                    Hhk == Hh and Hhv == Hh and
                    Dk == D and Dv == D):
                raise ValueError(
                    f"[student] K/V shape mismatch at layer {lid}: "
                    f"q={q.shape}, k={k.shape}, v={v.shape}"
                )

            if BN % B != 0:
                raise ValueError(
                    f"[student] BN ({BN}) not divisible by B ({B}) at layer {lid}"
                )
            N = BN // B  # 토큰 수

            # ----- 마지막 time-step(F-1)을 현재 프레임으로 사용 -----
            # q_last: [BN, Hh, D]
            q_last = q[:, -1, :, :]
            k_last = k[:, -1, :, :]
            v_last = v[:, -1, :, :]

            # [BN, Hh, D] → [B, N, Hh, D] → [B, Hh, N, D]
            q_last = q_last.view(B, N, Hh, D).permute(0, 2, 1, 3).contiguous()
            k_last = k_last.view(B, N, Hh, D).permute(0, 2, 1, 3).contiguous()
            v_last = v_last.view(B, N, Hh, D).permute(0, 2, 1, 3).contiguous()

            qkv_out[lid] = {
                "q": q_last.detach(),
                "k": k_last.detach(),
                "v": v_last.detach(),
            }

        if not qkv_out:
            return None

        return {"qkv": qkv_out}


    # ----------------------- streaming (inference, one frame) -----------------------
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

                # infer: detach_past=False, KD 필요 없음
                depth_net, cache_state, _ = self._stream_step_core(
                    cur_feature,
                    x_shape,
                    cache_state=cache_state,
                    detach_past=False,
                    return_kd=False,   # 명시적으로 KD off
                    kd_layers=None,
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
