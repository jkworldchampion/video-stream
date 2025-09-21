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
from .dpt import DPTHead
from .motion_module.motion_module import TemporalModule
from easydict import EasyDict


class DPTHeadTemporal(DPTHead):
    def __init__(self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken)

        assert num_frames > 0
        motion_module_kwargs = EasyDict(
            num_attention_heads   = 8,
            num_transformer_block = 1,
            num_attention_blocks  = 2,
            temporal_max_len      = num_frames,
            zero_initialize       = True,
            pos_embedding_type    = pe
        )

        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], **motion_module_kwargs),
            TemporalModule(in_channels=out_channels[3], **motion_module_kwargs),
            TemporalModule(in_channels=features,        **motion_module_kwargs),
            TemporalModule(in_channels=features,        **motion_module_kwargs),
        ])

    def forward(
        self,
        out_features,
        patch_h,
        patch_w,
        frame_length,
        micro_batch_size=4,
        cached_hidden_state_list=None,
        **kwargs,  # ← 스트리밍 옵션(rope_dt, select_top_r, stream_mode, return_attn/return_qkv …)을 하위로 패스
    ):
        """
        Args
        ----
        out_features: list of (tokens[, cls]) from encoder
        patch_h, patch_w: ViT patch grid height/width
        frame_length: 현재 배치의 time 길이 (오프라인=클립 길이, 스트리밍=1)
        micro_batch_size: 메모리 세이빙용 마이크로 배치
        cached_hidden_state_list: (옵션) 과거 hidden state 리스트(구 구현 호환)
            - 스트리밍을 새 구현으로 쓸 땐, 상위에서 attention 레벨의 past 토큰을
              전달하도록 구성할 수 있음(우리 motion_module이 **kwargs로 처리).
        kwargs: (전부 선택, 없으면 기존과 동일)
            - stream_mode: bool
            - rope_dt: float or Tensor(Δt)
            - select_top_r: int (과거 토큰 Top-R 선택; 평균 금지)
            - update_top_u: int (Refiner 인덱스 자리)
            - return_attn: bool
            - return_qkv: bool
        """
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            # tokens -> [B*T, C, H', W']
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            B, T = x.shape[0] // frame_length, frame_length
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        B, T = layer_1.shape[0] // frame_length, frame_length

        # 구(旧) 캐시 포맷과의 호환: 모션 모듈 수(4) 기준으로 슬라이싱
        if cached_hidden_state_list is not None:
            N = len(cached_hidden_state_list) // len(self.motion_modules)
        else:
            N = 0

        # ----- Temporal on high-level features (layer_3 / layer_4) -----
        layer_3_in = layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        layer_3, h0 = self.motion_modules[0](
            layer_3_in,
            None, None,
            cached_hidden_state_list[0:N] if N else None,
            **kwargs,
        )
        layer_3 = layer_3.permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_4_in = layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        layer_4, h1 = self.motion_modules[1](
            layer_4_in,
            None, None,
            cached_hidden_state_list[N:2*N] if N else None,
            **kwargs,
        )
        layer_4 = layer_4.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # ----- DPT refine path -----
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # path_4
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4_in = path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        path_4, h2 = self.motion_modules[2](
            path_4_in,
            None, None,
            cached_hidden_state_list[2*N:3*N] if N else None,
            **kwargs,
        )
        path_4 = path_4.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # path_3
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3_in = path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        path_3, h3 = self.motion_modules[3](
            path_3_in,
            None, None,
            cached_hidden_state_list[3*N:] if N else None,
            **kwargs,
        )
        path_3 = path_3.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # ----- Output head -----
        batch_size = layer_1_rn.shape[0]
        if batch_size <= micro_batch_size or batch_size % micro_batch_size != 0:
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
            ori_type = out.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                out = self.scratch.output_conv2(out.float())
            output = out.to(ori_type)
        else:
            ret = []
            for i in range(0, batch_size, micro_batch_size):
                path_2 = self.scratch.refinenet2(
                    path_3[i:i + micro_batch_size],
                    layer_2_rn[i:i + micro_batch_size],
                    size=layer_1_rn[i:i + micro_batch_size].shape[2:],
                )
                path_1 = self.scratch.refinenet1(path_2, layer_1_rn[i:i + micro_batch_size])
                out = self.scratch.output_conv1(path_1)
                out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
                ori_type = out.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    out = self.scratch.output_conv2(out.float())
                ret.append(out.to(ori_type))
            output = torch.cat(ret, dim=0)

        # 캐시는 모션 모듈 4곳의 출력 리스트를 이어붙여 반환(기존 호환)
        return output, h0 + h1 + h2 + h3
