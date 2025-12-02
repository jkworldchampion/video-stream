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
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
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

    def forward(
        self,
        x,
        *,
        return_intermediates: bool = False,
        return_qkv: bool = False,
        return_kd: bool = False,
        kd_layers=None,
    ):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14

        # DINOv2 feature
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0, 1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )

        need_inter = return_intermediates or return_qkv or return_kd

        if need_inter:
            depth, _cache, intermediates = self.head(
                features,
                patch_h,
                patch_w,
                T,
                return_intermediates=True,
                return_qkv=True,   # KD/QKV면 무조건 켠다
                kd_layers=kd_layers,
            )
        else:
            depth, _cache = self.head(
                features,
                patch_h,
                patch_w,
                T,
            )
            intermediates = None

        # [BT,1,h',w'] → [B,T,H,W]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth_bt = depth.squeeze(1).unflatten(0, (B, T))  # [B,T,H,W]

        # ----- KD 모드: (depth, kd_dict) 반환 -----
        if return_kd:
            kd = {}
            qkv_src = intermediates.get("qkv", {}) if intermediates is not None else {}

            if kd_layers is None:
                kd_layers = sorted(qkv_src.keys())

            qkv_out = {}

            for lid in kd_layers:
                if lid not in qkv_src:
                    continue

                q = qkv_src[lid]["q"]   # [B*N, Hh, Fq, D] 가정
                k = qkv_src[lid]["k"]
                v = qkv_src[lid]["v"]

                if q.dim() != 4:
                    raise ValueError(f"[teacher] Unexpected Q dim at layer {lid}: {q.shape}")

                BN, Hh, Fq, D = q.shape

                # 더 이상 Fq == T 라고 가정하지 않는다.
                if BN % B != 0:
                    raise ValueError(
                        f"[teacher] Q batch mismatch at layer {lid}: BN={BN}, B={B}"
                    )

                N = BN // B  # spatial tokens per frame

                # [B*N,Hh,Fq,D] → [B,N,Hh,Fq,D] → [B,Fq,Hh,N,D]
                q = q.view(B, N, Hh, Fq, D).permute(0, 3, 2, 1, 4).contiguous()
                k = k.view(B, N, Hh, Fq, D).permute(0, 3, 2, 1, 4).contiguous()
                v = v.view(B, N, Hh, Fq, D).permute(0, 3, 2, 1, 4).contiguous()

                # 이제 teacher_kd["qkv"][lid]["q"] 의 time 길이는 Fq 이다.
                # (train.py 에서 time축은 q.shape[1] 로 사용하면 됨)
                qkv_out[lid] = {
                    "q": q.detach(),  # [B,Fq,Hh,N,D]
                    "k": k.detach(),
                    "v": v.detach(),
                }

            kd["qkv"] = qkv_out
            return depth_bt, kd

        # ----- 기존 동작 유지 -----
        if return_intermediates or return_qkv:
            return {"pred": depth_bt, "intermediates": intermediates}

        return depth_bt


    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
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

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                       np.concatenate(ref_align),
                                                       np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)

        depth_list = depth_list_aligned

        return np.stack(depth_list[:org_video_len], axis=0), target_fps
