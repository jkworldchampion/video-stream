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
            pos_embedding_type    = pe,
        )

        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], **motion_module_kwargs),
            TemporalModule(in_channels=out_channels[3], **motion_module_kwargs),
            TemporalModule(in_channels=features,        **motion_module_kwargs),
            TemporalModule(in_channels=features,        **motion_module_kwargs),
        ])

    # ============================
    # 🔹 QKV Hook 인터페이스
    # ============================
    def enable_qkv_save(self, flag: bool):
        """
        모든 TemporalModule에 save_qkv 플래그 전달.
        (실제 Q/K/V 저장은 motion_module 내부 TemporalAttention에서 수행)
        """
        for m in self.motion_modules:
            if hasattr(m, "enable_qkv_save"):
                m.enable_qkv_save(flag)

    def collect_qkv(self, layer_idx: int):
        """
        layer_idx 번째 TemporalModule이 저장한 (q,k,v)를 반환.
        각 텐서는 [B, A, L, d] 형태.
        """
        assert 0 <= layer_idx < len(self.motion_modules), \
            f"invalid layer_idx {layer_idx}, must be in [0,{len(self.motion_modules)-1}]"

        m = self.motion_modules[layer_idx]
        if not hasattr(m, "get_qkv"):
            raise RuntimeError(
                f"TemporalModule at idx={layer_idx} has no get_qkv(). "
                f"motion_module.py에 get_qkv() 구현이 필요합니다."
            )
        return m.get_qkv()

    # ============================
    # forward (clip / stream 공용)
    # ============================
    def forward(
        self,
        out_features,
        patch_h,
        patch_w,
        frame_length,
        micro_batch_size=4,
        cached_hidden_state_list=None,
    ):
        """
        out_features: DINOv2 intermediate features (4-scale)
        frame_length: T (시퀀스 길이)
        cached_hidden_state_list:
          - None: 순수 clip 모드 (teacher와 동일)
          - list[tensor]: streaming 모드에서 KV-cache처럼 쓰이는 temporal hidden들
            레이아웃: [모듈0_블록들..., 모듈1_블록들..., 모듈2_블록들..., 모듈3_블록들...]
        """
        # 1) encoder feature → DPT 4-stage feature
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w)
            ).contiguous()

            B, T = x.shape[0] // frame_length, frame_length
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        B, T = layer_1.shape[0] // frame_length, frame_length

        # 2) cache 분배: [모듈0 ... 모듈3]로 나누기
        if cached_hidden_state_list is not None and len(cached_hidden_state_list) == 0:
            cached_hidden_state_list = None  # 빈 리스트는 None 취급

        if cached_hidden_state_list is not None:
            num_modules = len(self.motion_modules)  # 4
            assert len(cached_hidden_state_list) % num_modules == 0, \
                f"cached_hidden_state_list 길이({len(cached_hidden_state_list)})가 " \
                f"motion_modules 개수({num_modules})로 나누어떨어지지 않습니다."
            N = len(cached_hidden_state_list) // num_modules
        else:
            N = 0

        # 3) encoder side temporal modules (layer_3, layer_4)
        layer_3, h0 = self.motion_modules[0](
            layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4),
            encoder_hidden_states=None,
            attention_mask=None,
            cached_hidden_state_list=cached_hidden_state_list[0:N] if N else None,
        )
        layer_3 = layer_3.permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_4, h1 = self.motion_modules[1](
            layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4),
            encoder_hidden_states=None,
            attention_mask=None,
            cached_hidden_state_list=cached_hidden_state_list[N:2*N] if N else None,
        )
        layer_4 = layer_4.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # 4) refinenet + decoder side temporal modules (path_4, path_3)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4, h2 = self.motion_modules[2](
            path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4),
            encoder_hidden_states=None,
            attention_mask=None,
            cached_hidden_state_list=cached_hidden_state_list[2*N:3*N] if N else None,
        )
        path_4 = path_4.permute(0, 2, 1, 3, 4).flatten(0, 1)

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3, h3 = self.motion_modules[3](
            path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4),
            encoder_hidden_states=None,
            attention_mask=None,
            cached_hidden_state_list=cached_hidden_state_list[3*N:] if N else None,
        )
        path_3 = path_3.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # 5) DPT decoder (micro-batch)
        batch_size = layer_1_rn.shape[0]
        if batch_size <= micro_batch_size or batch_size % micro_batch_size != 0:
            path_2 = self.scratch.refinenet2(
                path_3, layer_2_rn, size=layer_1_rn.shape[2:]
            )
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(
                out,
                (int(patch_h * 14), int(patch_w * 14)),
                mode="bilinear",
                align_corners=True,
            )
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
                path_1 = self.scratch.refinenet1(
                    path_2, layer_1_rn[i:i + micro_batch_size]
                )
                out = self.scratch.output_conv1(path_1)
                out = F.interpolate(
                    out,
                    (int(patch_h * 14), int(patch_w * 14)),
                    mode="bilinear",
                    align_corners=True,
                )
                ori_type = out.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    out = self.scratch.output_conv2(out.float())
                ret.append(out.to(ori_type))
            output = torch.cat(ret, dim=0)

        # 6) streaming용 cache: 모듈별 hidden_state_list를 일렬로 concatenate
        cached_all = h0 + h1 + h2 + h3   # 리스트 concat

        return output, cached_all
