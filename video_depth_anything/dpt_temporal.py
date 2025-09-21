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
    """
    Enc-cache 기반 Temporal DPT Head.
    - 입력:
        out_features: 현재 프레임의 encoder 인터미디어트 피처 (list; 각 원소는 (tokens[, cls]) 튜플)
        cached_enc_feat_list: 과거 프레임들의 encoder 피처 리스트 (list of out_features 포맷), 없으면 None
    - 처리:
        레이어별로 tokens->grid->project->resize 동일 파이프라인을 적용하여
        레벨 3/4에 대해서 [B, C, T, H, W] 시퀀스를 구성하고 TemporalModule에 투입.
        출력에선 현재 시점(마지막 T)만 골라서 아래 DPT refine 경로 진행.
    - 반환:
        (output_logits, None)  # 캐시는 사용하지 않음
    """
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
            pos_embedding_type    = pe,
        )

        # temporal attention을 레벨 3/4 + refinenet path_4/path_3에 적용
        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], **motion_module_kwargs),  # for layer_3
            TemporalModule(in_channels=out_channels[3], **motion_module_kwargs),  # for layer_4
            TemporalModule(in_channels=features,        **motion_module_kwargs),  # for path_4
            TemporalModule(in_channels=features,        **motion_module_kwargs),  # for path_3
        ])

    # -------- helpers --------
    def _tokens_to_feature_map(self, x_tuple, i, patch_h, patch_w):
        """
        단일 프레임(or BT)의 레이어 i 토큰을 [B*T, C, H', W']로 변환.
        x_tuple: (tokens[, cls_token])
        """
        if self.use_clstoken:
            tokens, cls_token = x_tuple[0], x_tuple[1]          # [BT, P, C], [BT, C]
            readout = cls_token.unsqueeze(1).expand_as(tokens)  # [BT, P, C]
            x = self.readout_projects[i](torch.cat((tokens, readout), dim=-1))
        else:
            x = x_tuple[0]  # [BT, P, C]

        # tokens -> grid
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()
        # DPT 전처리
        x = self.projects[i](x)
        x = self.resize_layers[i](x)  # [BT, C_i, H_i, W_i]
        return x

    def _build_time_stack_for_layer(self, cur_feat_i, cached_enc_feat_list, i, patch_h, patch_w, frame_length):
        """
        레이어 i에 대해 과거(enc-cache) + 현재 피처를 동일 파이프라인으로 변환하여
        [B, C, T_seq, H, W] 시퀀스를 생성.
        """
        # 현재 프레임 맵
        cur_map = self._tokens_to_feature_map(cur_feat_i, i, patch_h, patch_w)  # [BT, C, H, W]
        B = cur_map.shape[0] // frame_length
        cur_bt = cur_map.unflatten(0, (B, frame_length))                        # [B, T, C, H, W]

        # 과거 프레임 맵들
        past_seq = None
        if cached_enc_feat_list is not None and len(cached_enc_feat_list) > 0:
            past_maps = []
            for fr in cached_enc_feat_list:
                pm = self._tokens_to_feature_map(fr[i], i, patch_h, patch_w)    # [B*1, C, H, W] 가정
                pm_bt = pm.view(B, 1, pm.shape[1], pm.shape[2], pm.shape[3])    # [B, 1, C, H, W]
                past_maps.append(pm_bt)
            past_seq = torch.cat(past_maps, dim=1)                               # [B, T_past, C, H, W]

        # 시간 축 결합
        if past_seq is not None:
            x_seq = torch.cat([past_seq, cur_bt], dim=1)                         # [B, T_seq, C, H, W]
        else:
            x_seq = cur_bt                                                       # [B, T,    C, H, W]

        # TemporalModule 기대 포맷 [B, C, T, H, W]
        x_seq = x_seq.permute(0, 2, 1, 3, 4).contiguous()
        return x_seq  # [B, C, T_seq, H, W]

    # -------- forward --------
    def forward(self, out_features, patch_h, patch_w, frame_length,
                micro_batch_size=4, cached_enc_feat_list=None):
        """
        out_features: 현재 프레임 encoder 피처 (list; (tokens[, cls]) × 4계층)
        cached_enc_feat_list: 과거 프레임 encoder 피처 리스트 (없으면 None)
        frame_length: 현재 입력의 time 길이 (스트리밍에선 1)
        """
        # 현재 프레임의 4계층을 grid/project/resize
        out_cur = []
        for i, x in enumerate(out_features):
            x_map = self._tokens_to_feature_map(x, i, patch_h, patch_w)  # [BT, C, H', W']
            out_cur.append(x_map)
        layer_1_cur, layer_2_cur, _, _ = out_cur

        # 레벨 3/4: enc-cache 포함한 시퀀스 구성 → TemporalModule
        layer_3_seq = self._build_time_stack_for_layer(out_features[2], cached_enc_feat_list, 2, patch_h, patch_w, frame_length)  # [B,C,T,H,W]
        layer_4_seq = self._build_time_stack_for_layer(out_features[3], cached_enc_feat_list, 3, patch_h, patch_w, frame_length)  # [B,C,T,H,W]

        layer_3_seq, _ = self.motion_modules[0](layer_3_seq, None, None, None)  # [B,C,T,H,W]
        layer_4_seq, _ = self.motion_modules[1](layer_4_seq, None, None, None)  # [B,C,T,H,W]

        # 현재 시점(마지막 T)만 사용하여 아래 경로 진행
        layer_3 = layer_3_seq[:, :, -1].contiguous()  # [B, C, H, W]
        layer_4 = layer_4_seq[:, :, -1].contiguous()  # [B, C, H, W]

        # DPT refine 경로
        layer_1_rn = self.scratch.layer1_rn(layer_1_cur)   # [BT, C, H, W]
        layer_2_rn = self.scratch.layer2_rn(layer_2_cur)
        # layer_3/4는 이미 [B,C,H,W]
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # refinenet4 → TemporalModule(길이=1로 통과)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])  # [B,C,H,W]
        path_4_seq = path_4.unsqueeze(2)                                         # [B,C,1,H,W]
        path_4_seq, _ = self.motion_modules[2](path_4_seq, None, None, None)
        path_4 = path_4_seq[:, :, -1].contiguous()                                # [B,C,H,W]

        # refinenet3 → TemporalModule(길이=1로 통과)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])  # [B,C,H,W]
        path_3_seq = path_3.unsqueeze(2)                                                 # [B,C,1,H,W]
        path_3_seq, _ = self.motion_modules[3](path_3_seq, None, None, None)
        path_3 = path_3_seq[:, :, -1].contiguous()                                       # [B,C,H,W]

        # 출력 합성 (원 로직 유지) — 여기서는 배치가 보통 B(=1)이므로 micro_batch 분기 거의 안 탐
        batch_size = layer_1_rn.shape[0]  # [BT, ...] 형태일 수 있어 기존 분기 유지
        if batch_size <= micro_batch_size or batch_size % micro_batch_size != 0:
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)),
                                mode="bilinear", align_corners=True)
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
                    size=layer_1_rn[i:i + micro_batch_size].shape[2:]
                )
                path_1 = self.scratch.refinenet1(path_2, layer_1_rn[i:i + micro_batch_size])
                out = self.scratch.output_conv1(path_1)
                out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)),
                                    mode="bilinear", align_corners=True)
                ori_type = out.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    out = self.scratch.output_conv2(out.float())
                ret.append(out.to(ori_type))
            output = torch.cat(ret, dim=0)

        # enc-cache 방식: hidden-state 캐시 없음
        return output, None
