# Copyright (2025)
# Spatial-Aware modules for Video Depth Estimation
# Differentiates from ASR's 1D sequence processing

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionPooling(nn.Module):
    """
    Spatial attention pooling for [B, C, T, H, W] -> [B, T, C]
    
    Instead of naive mean pooling, this learns which spatial locations
    are important for depth estimation (e.g., foreground objects, edges).
    
    This is a key differentiator from ASR methods which work on 1D sequences.
    """
    def __init__(self, channels: int, reduction: int = 4):
        """
        Args:
            channels: Input channel dimension
            reduction: Channel reduction ratio for attention computation
        """
        super().__init__()
        
        # Spatial attention generator
        # Uses channel reduction to save computation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # [0, 1] attention weights
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] temporal features
            
        Returns:
            pooled: [B, T, C] spatially-attended features
        """
        B, C, T, H, W = x.shape
        
        # Process each frame independently
        attended_features = []
        for t in range(T):
            frame = x[:, :, t]  # [B, C, H, W]
            
            # Generate spatial attention map [B, 1, H, W]
            attn_map = self.attention_conv(frame)
            
            # Weighted spatial pooling
            weighted = frame * attn_map  # [B, C, H, W]
            
            # Normalize by attention sum to maintain scale
            pooled = weighted.sum(dim=(2, 3))  # [B, C]
            norm = attn_map.sum(dim=(2, 3)).clamp(min=1e-8)  # [B, 1]
            pooled = pooled / norm  # [B, C]
            
            attended_features.append(pooled)
        
        # Stack temporal dimension [B, T, C]
        output = torch.stack(attended_features, dim=1)
        return output


class MultiScaleSpatialEncoder(nn.Module):
    """
    Multi-scale spatial feature extraction for depth estimation.
    
    Depth prediction benefits from multi-scale context:
    - Fine scale (1x1): Local details
    - Medium scale (3x3): Local neighborhood  
    - Coarse scale (5x5): Broader context
    
    This exploits the 2D structure of video frames, unlike 1D audio in ASR.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension
        """
        super().__init__()
        
        # Three scales with different receptive fields
        channels_per_scale = out_channels // 3
        
        self.conv_fine = nn.Sequential(
            nn.Conv2d(in_channels, channels_per_scale, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_per_scale),
            nn.ReLU(inplace=True)
        )
        
        self.conv_medium = nn.Sequential(
            nn.Conv2d(in_channels, channels_per_scale, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_per_scale),
            nn.ReLU(inplace=True)
        )
        
        self.conv_coarse = nn.Sequential(
            nn.Conv2d(in_channels, channels_per_scale, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels_per_scale),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] single frame feature
            
        Returns:
            fused: [B, C', H, W] multi-scale fused feature
        """
        # Extract features at different scales
        feat_fine = self.conv_fine(x)
        feat_medium = self.conv_medium(x)
        feat_coarse = self.conv_coarse(x)
        
        # Concatenate and fuse
        multi_scale = torch.cat([feat_fine, feat_medium, feat_coarse], dim=1)
        fused = self.fusion(multi_scale)
        
        return fused
