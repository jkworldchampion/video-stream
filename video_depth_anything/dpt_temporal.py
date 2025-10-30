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

    @staticmethod
    def _pool_feat_bcthw_to_btc(x_bcthw, mode="mean"):
        """
        x_bcthw: [B, C, T, H, W]  -> [B, T, C] (spatial pooled)
        """
        if mode == "mean":
            x_btc = x_bcthw.mean(dim=(3, 4)).transpose(1, 2).contiguous()  # [B,T,C]
        elif mode == "cls":
            # CLS 토큰이 없다면 mean과 동일 처리
            x_btc = x_bcthw.mean(dim=(3, 4)).transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported feature_pool mode: {mode}")
        return x_btc

    def forward(
        self,
        out_features,
        patch_h,
        patch_w,
        frame_length,
        micro_batch_size=4,
        cached_hidden_state_list=None,
        *,
        # === New flags for KD/Aux ===
        return_intermediates: bool = False,
        return_qkv: bool = False,
        feature_pool: str = "mean",
    ):
        """
        Temporal aggregation + DPT refine. `cached_hidden_state_list`만 사용.

        Args:
            return_intermediates: True면 중간 표현/관계 반환(레이어별 [B,T,C] feat 및 (가능시) Q/K/V 관계)
            return_qkv: TemporalModule이 지원하면 Q/K/V 관계를 함께 수집
            feature_pool: 'mean' | 'cls'  (공간 풀링 방식)
        Returns:
            기본: (output, cache)
            return_intermediates=True: (output, cache, intermediates_dict)
                intermediates_dict = {
                    0: {"feat": [B,T,C], "qkv": {"Q": [B,A,T,d], "K": ..., "V": ...} or None},
                    1: {...}, 2: {...}, 3: {...}
                }
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

        # === intermediates collector ===
        intermediates = {} if return_intermediates else None

        # ----- Temporal on high-level features (layer_3 / layer_4) -----
        layer_3_in = layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        # 지원 시 qkv도 반환받음 (역호환: TypeError 시 fallback)
        try:
            if return_qkv:
                layer_3_out, h0, qkv_0 = self.motion_modules[0](
                    layer_3_in,
                    None, None,
                    cached_hidden_state_list[0:N] if N else None,
                    return_qkv=True
                )
            else:
                raise TypeError
        except TypeError:
            layer_3_out, h0 = self.motion_modules[0](
                layer_3_in,
                None, None,
                cached_hidden_state_list[0:N] if N else None,
            )
            qkv_0 = None

        if return_intermediates:
            feat_0 = self._pool_feat_bcthw_to_btc(layer_3_out, mode=feature_pool)  # [B,T,C]
            intermediates[0] = {
                "feat": feat_0, 
                "qkv": qkv_0,
                "feat_spatial": layer_3_out  # [B, C, T, H, W] for spatial-aware pooling
            }

        layer_3 = layer_3_out.permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_4_in = layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        try:
            if return_qkv:
                layer_4_out, h1, qkv_1 = self.motion_modules[1](
                    layer_4_in,
                    None, None,
                    cached_hidden_state_list[N:2*N] if N else None,
                    return_qkv=True
                )
            else:
                raise TypeError
        except TypeError:
            layer_4_out, h1 = self.motion_modules[1](
                layer_4_in,
                None, None,
                cached_hidden_state_list[N:2*N] if N else None,
            )
            qkv_1 = None

        if return_intermediates:
            feat_1 = self._pool_feat_bcthw_to_btc(layer_4_out, mode=feature_pool)
            intermediates[1] = {
                "feat": feat_1, 
                "qkv": qkv_1,
                "feat_spatial": layer_4_out  # [B, C, T, H, W]
            }

        layer_4 = layer_4_out.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # ----- DPT refine path -----
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # path_4
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4_in = path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        try:
            if return_qkv:
                path_4_out, h2, qkv_2 = self.motion_modules[2](
                    path_4_in,
                    None, None,
                    cached_hidden_state_list[2*N:3*N] if N else None,
                    return_qkv=True
                )
            else:
                raise TypeError
        except TypeError:
            path_4_out, h2 = self.motion_modules[2](
                path_4_in,
                None, None,
                cached_hidden_state_list[2*N:3*N] if N else None,
            )
            qkv_2 = None

        if return_intermediates:
            feat_2 = self._pool_feat_bcthw_to_btc(path_4_out, mode=feature_pool)
            intermediates[2] = {
                "feat": feat_2, 
                "qkv": qkv_2,
                "feat_spatial": path_4_out  # [B, C, T, H, W]
            }

        path_4 = path_4_out.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # path_3
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3_in = path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        try:
            if return_qkv:
                path_3_out, h3, qkv_3 = self.motion_modules[3](
                    path_3_in,
                    None, None,
                    cached_hidden_state_list[3*N:] if N else None,
                    return_qkv=True
                )
            else:
                raise TypeError
        except TypeError:
            path_3_out, h3 = self.motion_modules[3](
                path_3_in,
                None, None,
                cached_hidden_state_list[3*N:] if N else None,
            )
            qkv_3 = None

        if return_intermediates:
            feat_3 = self._pool_feat_bcthw_to_btc(path_3_out, mode=feature_pool)
            intermediates[3] = {
                "feat": feat_3, 
                "qkv": qkv_3,
                "feat_spatial": path_3_out  # [B, C, T, H, W]
            }

        path_3 = path_3_out.permute(0, 2, 1, 3, 4).flatten(0, 1)

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
        cache = h0 + h1 + h2 + h3

        if return_intermediates:
            return output, cache, intermediates
        else:
            return output, cache
