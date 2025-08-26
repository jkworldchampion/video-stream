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
        pe='ape',
        use_causal_mask=True,
        use_self_forcing=False
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.use_self_forcing = use_self_forcing
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe, use_causal_mask=use_causal_mask, use_self_forcing=use_self_forcing)
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []
        self.gap = (INFER_LEN - OVERLAP) * 2 - 1 - (OVERLAP - INTERP_LEN)
        assert self.gap == 41
        self.id = -1

    def forward(self, x, prev_depth=None):
        return self.forward_depth(self.forward_features(x), x.shape, None, prev_depth)[0]
    
    def forward_features(self, x):
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        return features

    def forward_depth(self, features, x_shape, cached_hidden_state_list=None, prev_depth=None, bidirectional_update_length=16, current_frame=0):
        """
        Forward pass for depth prediction with optional bidirectional update support.
        
        Args:
            features: Input features from encoder
            x_shape: Input tensor shape (B, T, C, H, W)
            cached_hidden_state_list: Cached hidden states from previous frames
            prev_depth: Previous depth for self-forcing
            bidirectional_update_length: Number of recent frames to update bidirectionally (default: 16)
            current_frame: Current frame index for bidirectional update logic
        """
        B, T, _, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14
        depth, cur_cached_hidden_state_list = self.head(
            features, patch_h, patch_w, T, 
            cached_hidden_state_list=cached_hidden_state_list, 
            prev_depth=prev_depth,
            bidirectional_update_length=bidirectional_update_length,
            current_frame=current_frame
        )
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)), cur_cached_hidden_state_list
    
    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
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
                with torch.autocast(device_type='cuda', enabled=(not fp32)):
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape
                    depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)

            # Copy multiple cache to simulate the windows
            # IMPORTANT: Use deep copy to avoid shared references
            import copy
            self.frame_cache_list = [copy.deepcopy(cached_hidden_state_list) for _ in range(INFER_LEN)]
            self.frame_id_list.extend([0] * (INFER_LEN - 1))

            new_depth = depth[0][0].cpu().numpy()
        else:
            frame_height, frame_width = frame.shape[:2]
            
            # Check if frame dimensions changed - if so, we need to reinitialize
            if frame_height != self.frame_height or frame_width != self.frame_width:
                print(f"Warning: Frame resolution changed from {self.frame_height}x{self.frame_width} to {frame_height}x{frame_width}")
                print("Reinitializing transform and cache...")
                
                # Reset transform and cache
                self.transform = None
                self.frame_cache_list = []
                self.frame_id_list = []
                self.id = -1
                
                # Recursively call with reset state
                return self.infer_video_depth_one(frame, input_size, device, fp32)

            # infer feature
            cur_input = torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0).to(device)
            
            # Try with cached states first, but be prepared to reset if dimensions don't match
            try:
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', enabled=(not fp32)):
                        cur_feature = self.forward_features(cur_input)
                        x_shape = cur_input.shape

                cur_list = self.frame_cache_list[0:2] + self.frame_cache_list[-INFER_LEN+3:]
                assert len(cur_list) == INFER_LEN - 1
                
                # Check if cached states are compatible with current input
                try:
                    # For streaming mode, we pass the most recent cache directly 
                    # without concatenating multiple time steps
                    # Each layer should receive the cache from the previous frame
                    
                    # Get the most recent cache (last element in the list)
                    most_recent_cache = cur_list[-1]  # Most recent frame's cache
                    
                    # Validate cache dimensions for each layer
                    cur_cache = []
                    for i, layer_cache in enumerate(most_recent_cache):
                        
                        # Ensure cache has correct 3D format [spatial_patches, temporal_steps, dim]
                        if layer_cache.dim() == 2:
                            # Convert 2D to 3D by adding temporal dimension
                            layer_cache = layer_cache.unsqueeze(1)  # [spatial_patches, 1, dim]
                        
                        cur_cache.append(layer_cache)
                        
                except (RuntimeError, ValueError, IndexError) as cache_error:
                    print(f"Cache compatibility issue: {cache_error}")
                    raise RuntimeError("Cache processing failed")
                
                # Try to infer depth with cached states directly
                try:
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', enabled=(not fp32)):
                            depth, new_cache = self.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)

                    depth = depth.to(cur_input.dtype)
                    depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
                    depth_list = [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

                    new_depth = depth_list[-1]
                    self.frame_cache_list.append(new_cache)
                    
                except RuntimeError:
                    # If cache is incompatible, let it fall through to outer exception handler
                    raise

            except (RuntimeError, AssertionError) as e:
                print(f"Cache compatibility issue: {e}")
                print("Reinitializing cache and processing without history...")
                
                # Reset cache and process current frame without history
                self.frame_cache_list = []
                self.frame_id_list = []
                
                # Process current frame as if it's the first frame, but keep transform
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', enabled=(not fp32)):
                        depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)

                depth = depth.to(cur_input.dtype)
                depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)

                # Reinitialize cache
                self.frame_cache_list = [cached_hidden_state_list] * INFER_LEN
                self.frame_id_list.extend([self.id] * (INFER_LEN - 1))

                new_depth = depth[0][0].cpu().numpy()

        # adjust the sliding window
        self.frame_id_list.append(self.id)
        if self.id + INFER_LEN > self.gap + 1 and len(self.frame_cache_list) > 1:
            del self.frame_id_list[1]
            del self.frame_cache_list[1]

        return new_depth
