"""
Dynamic KV Refinement Module for Streaming Video Depth Estimation

Context-aware K/V refinement to bridge Teacher (batch) and Student (streaming).
Teacher's K/V are dynamic (context-dependent), while Student's K/V are static.
This module enables Student to adapt K/V based on temporal context.

Author: GitHub Copilot (C Enhancement)
Date: 2025-11-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightKVRefiner(nn.Module):
    """
    Lightweight MLP-based K/V refiner.
    
    Given static K/V from Student and temporal context, outputs refined K/V
    that mimics Teacher's dynamic attention behavior.
    
    Args:
        dim: Feature dimension (e.g., 192, 384, 64)
        context_frames: Number of context frames to aggregate (default: 4)
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, dim, context_frames=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.context_frames = context_frames
        
        # Context aggregation: mean pooling over recent frames
        # Input: [static_kv, context_mean] -> Output: delta
        self.refiner = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
        # Initialize with small weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.refiner:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, static_kv, context_buffer):
        """
        Refine static K or V using temporal context.
        
        Args:
            static_kv: Static K or V from current frame [B, 1, C]
            context_buffer: List of recent K or V features (deque of [B, 1, C])
                           Length can vary (1 to context_frames)
        
        Returns:
            refined_kv: Context-aware refined K or V [B, 1, C]
        """
        if len(context_buffer) == 0:
            # No context available (first frame)
            return static_kv
        
        # Aggregate context: mean pooling over recent frames
        context_list = list(context_buffer)[-self.context_frames:]  # Last N frames
        context_stack = torch.cat(context_list, dim=1)  # [B, T_ctx, C]
        context_mean = context_stack.mean(dim=1, keepdim=True)  # [B, 1, C]
        
        # Concatenate static and context
        concat_input = torch.cat([static_kv, context_mean], dim=-1)  # [B, 1, 2C]
        
        # Compute refinement delta
        delta = self.refiner(concat_input)  # [B, 1, C]
        
        # Residual connection for stability
        refined_kv = static_kv + delta
        
        return refined_kv


class KVRefinerWrapper(nn.Module):
    """
    Wrapper for K and V refiners for each layer.
    
    Args:
        dim: Feature dimension
        context_frames: Number of context frames
        dropout: Dropout rate
    """
    
    def __init__(self, dim, context_frames=4, dropout=0.1):
        super().__init__()
        self.k_refiner = LightweightKVRefiner(dim, context_frames, dropout)
        self.v_refiner = LightweightKVRefiner(dim, context_frames, dropout)
    
    def forward(self, static_k, static_v, k_context, v_context):
        """
        Refine both K and V.
        
        Args:
            static_k: Static K [B, 1, C]
            static_v: Static V [B, 1, C]
            k_context: K context buffer (deque)
            v_context: V context buffer (deque)
        
        Returns:
            refined_k: [B, 1, C]
            refined_v: [B, 1, C]
        """
        refined_k = self.k_refiner(static_k, k_context)
        refined_v = self.v_refiner(static_v, v_context)
        return refined_k, refined_v


def test_kv_refiner():
    """Unit test for KV Refiner."""
    print("Testing LightweightKVRefiner...")
    
    B, C = 1, 192
    refiner = LightweightKVRefiner(dim=C, context_frames=4)
    
    # Test 1: No context (first frame)
    static_kv = torch.randn(B, 1, C)
    context_empty = []
    refined = refiner(static_kv, context_empty)
    assert refined.shape == static_kv.shape
    assert torch.allclose(refined, static_kv), "Should return input when no context"
    print("✓ Test 1 passed: No context handling")
    
    # Test 2: With context
    import collections
    context_buffer = collections.deque(maxlen=4)
    for _ in range(3):
        context_buffer.append(torch.randn(B, 1, C))
    
    refined = refiner(static_kv, context_buffer)
    assert refined.shape == static_kv.shape
    assert not torch.allclose(refined, static_kv), "Should modify with context"
    print("✓ Test 2 passed: Context refinement")
    
    # Test 3: Wrapper (K and V together)
    wrapper = KVRefinerWrapper(dim=C)
    static_k = torch.randn(B, 1, C)
    static_v = torch.randn(B, 1, C)
    k_ctx = collections.deque([torch.randn(B, 1, C) for _ in range(2)], maxlen=4)
    v_ctx = collections.deque([torch.randn(B, 1, C) for _ in range(2)], maxlen=4)
    
    refined_k, refined_v = wrapper(static_k, static_v, k_ctx, v_ctx)
    assert refined_k.shape == static_k.shape
    assert refined_v.shape == static_v.shape
    print("✓ Test 3 passed: Wrapper functionality")
    
    # Test 4: Parameter count
    total_params = sum(p.numel() for p in refiner.parameters())
    print(f"✓ Parameters per refiner: {total_params:,} (~{total_params/1e6:.2f}M)")
    
    wrapper_params = sum(p.numel() for p in wrapper.parameters())
    print(f"✓ Parameters per layer (K+V): {wrapper_params:,} (~{wrapper_params/1e6:.2f}M)")
    
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_kv_refiner()
