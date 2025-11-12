"""
H1 실험 간단 테스트: Attention 추출 가능 여부 확인
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pathlib import Path

print("="*60)
print("H1 Experiment: Quick Test")
print("="*60)

# 1. Check data
print("\n1. Checking ScanNet JSON...")
json_path = "/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"

if os.path.exists(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    scenes = data["scannet"]
    print(f"   ✓ Found {len(scenes)} scenes")
    
    if scenes:
        first_scene = scenes[0]
        scene_name = list(first_scene.keys())[0]
        num_frames = len(first_scene[scene_name])
        print(f"   ✓ Scene 0: {scene_name}, {num_frames} frames")
        
        first_frame = first_scene[scene_name][0]
        print(f"   ✓ Sample: {first_frame['image']}")
else:
    print(f"   ✗ JSON not found: {json_path}")
    print("   → Run dataset extraction first")

# 2. Check model
print("\n2. Checking model...")
checkpoint_path = "checkpoints/video_depth_anything_vits.pth"

if os.path.exists(checkpoint_path):
    print(f"   ✓ Checkpoint found: {checkpoint_path}")
    
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   ✓ Device: {device}")
        
        model = VideoDepthAnything(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            num_frames=32
        ).to(device)
        
        sd = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.eval()
        print(f"   ✓ Model loaded")
        
        # Test forward with dummy input
        dummy_video = torch.randn(1, 32, 3, 518, 518).to(device)
        
        print("\n3. Testing forward pass...")
        with torch.no_grad():
            try:
                out = model(dummy_video, return_intermediates=True, return_qkv=False)
                print(f"   ✓ Forward pass successful")
                print(f"   ✓ Output keys: {list(out.keys())}")
                
                if "intermediates" in out:
                    print(f"   ✓ Intermediates available")
                    print(f"   ✓ Layers: {list(out['intermediates'].keys())}")
                    
                    # Check first layer
                    if 0 in out["intermediates"]:
                        layer_0 = out["intermediates"][0]
                        print(f"   ✓ Layer 0 keys: {list(layer_0.keys())}")
                        
                        if "feat" in layer_0:
                            print(f"   ✓ feat shape: {layer_0['feat'].shape}")
                        
                        if "attention" in layer_0 or "qkv" in layer_0:
                            print(f"   ⚠ Attention available in current implementation")
                        else:
                            print(f"   ⚠ Attention NOT available - need to implement")
                            print(f"   → DPT Temporal에 attention return 기능 추가 필요")
                else:
                    print(f"   ✗ No intermediates returned")
                    
            except Exception as e:
                print(f"   ✗ Forward failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Test streaming forward (single frame)
        print("\n4. Testing streaming forward...")
        try:
            dummy_frame = torch.randn(1, 1, 3, 518, 518).to(device)  # [1, 1, 3, H, W]
            
            with torch.no_grad():
                out_stream = model.forward(
                    dummy_frame,
                    return_intermediates=True,
                    return_qkv=True
                )
                
                print(f"   ✓ Streaming forward successful")
                print(f"   ✓ Output keys: {list(out_stream.keys())}")
                
                if "intermediates" in out_stream:
                    print(f"   ✓ Intermediates available in streaming mode")
                else:
                    print(f"   ✗ No intermediates in streaming mode")
                    
        except Exception as e:
            print(f"   ✗ Streaming forward failed: {e}")
            import traceback
            traceback.print_exc()
                
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"   ✗ Checkpoint not found: {checkpoint_path}")

print("\n" + "="*60)
print("Test complete")
print("="*60)
print("\nNext steps:")
print("1. If attention NOT available:")
print("   → Implement attention return in DPT Temporal")
print("   → Add attention extraction from TemporalModule")
print("2. If all tests pass:")
print("   → Run full H1 experiment: python experiments/h1_attention_analysis.py")
print("="*60)
