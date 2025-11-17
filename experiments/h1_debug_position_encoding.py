"""
H1-DEBUG: Position Encoding 검증 스크립트
==========================================

목적: Stream 모드에서 Position Encoding이 제대로 누적되는지 확인

확인 사항:
1. APE(Absolute Positional Encoding)가 매 스텝 누적되는가?
2. KV cache에 저장된 position index가 올바른가?
3. Attention logits 계산 시 position bias가 정상인가?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Project imports
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream


def debug_position_encoding(
    model: VideoDepthAnythingStream,
    num_frames: int = 10,
    device: str = "cuda"
):
    """
    Position encoding이 제대로 누적되는지 단계별 확인
    """
    model.eval()
    
    print("="*60)
    print("Position Encoding Debug")
    print("="*60)
    
    # Dummy input (all ones for simplicity)
    H, W = 518, 518
    C = 3
    
    cached_hidden_state_list = None
    
    for t in range(num_frames):
        print(f"\n{'='*40}")
        print(f"Frame {t}")
        print(f"{'='*40}")
        
        # Dummy frame
        frame = torch.ones(1, 1, C, H, W, device=device) * (t / num_frames)  # [1, 1, C, H, W]
        
        with torch.no_grad():
            # Forward features
            features = model.forward_features(frame)
            
            # ===== HOOK: Inspect DPT temporal blocks =====
            # Check if position encoding is applied correctly
            
            # Forward depth
            depth_bt, cur_cache, intermediates = model.forward_depth(
                features,
                frame.shape,
                cached_hidden_state_list=cached_hidden_state_list,
                return_intermediates=True,
                return_qkv=True
            )
            
            # Cache inspection
            if cached_hidden_state_list is None:
                print("  First frame - no cache")
            else:
                print(f"  Cache length: {len(cached_hidden_state_list)}")
                
                # Check each layer's cache
                for i, cache_item in enumerate(cached_hidden_state_list):
                    if cache_item is not None:
                        if isinstance(cache_item, torch.Tensor):
                            print(f"    Layer {i} cache shape: {cache_item.shape}")
                        elif isinstance(cache_item, (tuple, list)):
                            shapes = [c.shape if torch.is_tensor(c) else "None" for c in cache_item]
                            print(f"    Layer {i} cache shapes: {shapes}")
            
            # Attention inspection
            print(f"  Intermediates keys: {list(intermediates.keys())}")
            
            for layer_idx in [0, 1, 2, 3]:
                if layer_idx not in intermediates:
                    continue
                
                layer_data = intermediates[layer_idx]
                
                # Check attention
                if "attention" in layer_data and layer_data["attention"] is not None:
                    attn = layer_data["attention"]  # [B, H, 1, cache_len]
                    print(f"  Layer {layer_idx} attention:")
                    print(f"    Shape: {attn.shape}")
                    
                    # Check distribution
                    attn_avg = attn[0].mean(0).squeeze().cpu().numpy()  # [cache_len]
                    
                    print(f"    Distribution: {attn_avg}")
                    print(f"    First token: {attn_avg[0]:.4f}")
                    if len(attn_avg) > 1:
                        print(f"    Last token: {attn_avg[-1]:.4f}")
                    
                    # Entropy
                    entropy = -(attn_avg * np.log(attn_avg + 1e-8)).sum()
                    print(f"    Entropy: {entropy:.4f}")
                    
                    # Warning
                    if attn_avg[0] > 0.8 and len(attn_avg) > 3:
                        print(f"    ⚠️  ATTENTION SINK: {attn_avg[0]*100:.1f}% on first token!")
                
                # Check QKV if available
                if "qkv" in layer_data and layer_data["qkv"] is not None:
                    qkv = layer_data["qkv"]
                    if isinstance(qkv, dict):
                        print(f"  Layer {layer_idx} QKV:")
                        for key in ["q", "k", "v"]:
                            if key in qkv and qkv[key] is not None:
                                print(f"    {key.upper()} shape: {qkv[key].shape}")
            
            # Update cache
            cached_hidden_state_list = cur_cache
    
    print("\n" + "="*60)
    print("Debug complete")
    print("="*60)


def check_model_positional_encoding_implementation():
    """
    모델 코드에서 position encoding 구현 확인
    """
    print("="*60)
    print("Checking Position Encoding Implementation")
    print("="*60)
    
    # This requires inspecting the actual model code
    # Key files to check:
    files_to_check = [
        "video_depth_anything/dpt.py",
        "video_depth_anything/motion/temporal.py",
        "video_depth_anything/video_depth_stream.py",
    ]
    
    print("\nFiles to inspect manually:")
    for file in files_to_check:
        print(f"  - {file}")
    
    print("\nKey questions:")
    print("  1. Is there a 'position' or 'position_ids' parameter in forward()?")
    print("  2. Is position encoding added BEFORE caching?")
    print("  3. Is position index incremented with cache length?")
    print("  4. Are cached K/V reused with correct position encoding?")
    
    print("\nExpected behavior:")
    print("  Frame 0: pos=0")
    print("  Frame 1: pos=1, cache=[k0, k1], v=[v0, v1]")
    print("  Frame t: pos=t, cache=[k0...kt], v=[v0...vt]")
    
    print("\nCommon bugs:")
    print("  - Position always reset to 0 at each step")
    print("  - Position encoding applied AFTER caching (cached values have wrong pos)")
    print("  - Cache indexing error (overwriting instead of appending)")
    print("="*60)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/video_depth_anything_vits.pth"
    
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load streaming model
    model = VideoDepthAnythingStream(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        num_frames=32
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    
    # Run debug
    debug_position_encoding(model, num_frames=10, device=device)
    
    # Print implementation check
    check_model_positional_encoding_implementation()


if __name__ == "__main__":
    main()
