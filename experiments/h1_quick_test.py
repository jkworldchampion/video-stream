"""
H1 실험 간소화 버전: 빠른 검증용
- Scene 0만 분석 (500 frames)
- 32-frame window로 처리
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from video_depth_anything.video_depth import VideoDepthAnything

print("="*60)
print("H1: Quick Attention Analysis (Scene 0 only)")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model
print("\nLoading model...")
model = VideoDepthAnything(
    encoder="vits",  # ← vits (small) for training experiments
    features=64, 
    out_channels=[48, 96, 192, 384], 
    num_frames=32
).to(device)

checkpoint_path = "checkpoints/video_depth_anything_vits.pth"
sd = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(sd, strict=True)
model.eval()
print("Model loaded!")

# Load ScanNet scene 0 data
print("\nLoading ScanNet scene 0...")
json_path = "/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"

with open(json_path) as f:
    data = json.load(f)

scene_0 = data["scannet"][0]
scene_name = list(scene_0.keys())[0]
frames = scene_0[scene_name]

print(f"Scene: {scene_name}, Frames: {len(frames)}")

# Load 500 frames for analysis (or all available)
from PIL import Image
import torchvision.transforms.functional as TF

num_frames = min(500, len(frames))
print(f"Loading {num_frames} frames...")

imgs = []
depths = []
root_dir = "/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/"

for i, frame_info in enumerate(tqdm(frames[:num_frames], desc="Loading frames")):
    img_path = os.path.join(root_dir, frame_info["image"])
    img = Image.open(img_path).convert("RGB")
    img = TF.center_crop(img, (518, 518))
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    imgs.append(img)
    
    depth_path = os.path.join(root_dir, frame_info["gt_depth"])
    depth = Image.open(depth_path).convert("F")
    depth = TF.center_crop(depth, (518, 518))
    depth = torch.from_numpy(np.array(depth, np.float32)).unsqueeze(0) / 1000.0
    depths.append(depth)

video = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # [1, 64, 3, H, W]
depth_gt = torch.stack(depths, dim=0).unsqueeze(0).to(device)  # [1, 64, 1, H, W]

print(f"Video shape: {video.shape}")
print(f"Depth GT shape: {depth_gt.shape}")

# Analyze first 32 frames (clip mode)
print("\n" + "="*60)
print("Running Clip mode (first 32 frames)...")
print("="*60)

clip_video = video[:, :32]  # [1, 32, 3, H, W]

with torch.no_grad():
    out_clip = model(
        clip_video,
        return_intermediates=True,
        return_qkv=True,  # ← True로 설정해야 attention 추출
    )

print(f"Clip output keys: {list(out_clip.keys())}")

if "intermediates" in out_clip:
    print(f"Intermediate layers: {list(out_clip['intermediates'].keys())}")
    
    for layer_idx in [0, 1, 2, 3]:
        if layer_idx in out_clip["intermediates"]:
            layer_dict = out_clip["intermediates"][layer_idx]
            print(f"\nLayer {layer_idx}:")
            print(f"  Keys: {list(layer_dict.keys())}")
            
            if "feat" in layer_dict:
                print(f"  feat shape: {layer_dict['feat'].shape}")
            
            if "qkv" in layer_dict and layer_dict["qkv"] is not None:
                print(f"  qkv keys: {list(layer_dict['qkv'].keys()) if isinstance(layer_dict['qkv'], dict) else 'not dict'}")
            
            if "attention" in layer_dict and layer_dict["attention"] is not None:
                attn = layer_dict["attention"]
                print(f"  ✓ attention shape: {attn.shape}")  # [B, H, T, T]
                
                # Compute entropy
                entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()
                print(f"  attention entropy: {entropy.item():.4f}")
                
                # Compute temporal locality
                T = attn.shape[2]
                t_indices = torch.arange(T, device=attn.device, dtype=torch.float32)
                dist_matrix = (t_indices.unsqueeze(1) - t_indices.unsqueeze(0)).abs()
                locality = (attn * dist_matrix).sum(dim=-1).mean()
                print(f"  temporal locality: {locality.item():.2f} frames")
            else:
                print(f"  ✗ attention not available")

# Compute delta1 for clip mode
print("\nComputing Clip δ1...")
clip_depth = out_clip["pred"]  # [B, T, H, W] or [B, T, 1, H, W]
clip_gt = depth_gt[:, :32]      # [B, T, 1, H, W]

print(f"  clip_depth shape: {clip_depth.shape}")
print(f"  clip_gt shape: {clip_gt.shape}")
print(f"  clip_depth range: [{clip_depth.min().item():.4f}, {clip_depth.max().item():.4f}]")
print(f"  clip_gt range: [{clip_gt.min().item():.4f}, {clip_gt.max().item():.4f}]")

# Ensure both have channel dimension
if clip_depth.dim() == 4:
    # [B, T, H, W] -> [B, T, 1, H, W]
    clip_depth = clip_depth.unsqueeze(2)
elif clip_depth.dim() == 3:
    # [B*T, H, W] -> [B*T, 1, H, W]
    clip_depth = clip_depth.unsqueeze(1)

# Flatten batch and time
if clip_depth.dim() == 5:
    clip_depth_flat = clip_depth.flatten(0, 1)  # [B*T, 1, H, W]
else:
    clip_depth_flat = clip_depth

if clip_gt.dim() == 5:
    clip_gt_flat = clip_gt.flatten(0, 1)
else:
    clip_gt_flat = clip_gt

# Compute delta1 with scale-shift alignment (following benchmark/eval/eval.py)
valid_mask = (clip_gt_flat > 0.001) & (clip_gt_flat < 10.0)
print(f"  Valid pixels: {valid_mask.sum().item()}/{valid_mask.numel()}")

if valid_mask.sum() > 0:
    # Convert to disparity space for alignment
    gt_disp = 1.0 / (clip_gt_flat[valid_mask] + 1e-8)  # [N]
    pred_disp = clip_depth_flat[valid_mask]  # [N]
    
    # Least-squares fitting: pred_disp -> gt_disp
    pred_disp_np = pred_disp.cpu().numpy().reshape(-1, 1)
    gt_disp_np = gt_disp.cpu().numpy().reshape(-1, 1)
    ones = np.ones_like(pred_disp_np)
    A = np.concatenate([pred_disp_np, ones], axis=-1)
    X = np.linalg.lstsq(A, gt_disp_np, rcond=None)[0]
    scale, shift = X[0, 0], X[1, 0]
    
    # Apply alignment
    aligned_pred_disp = scale * clip_depth_flat + shift
    aligned_pred_disp = torch.clamp(aligned_pred_disp, min=1e-3)
    aligned_pred_depth = 1.0 / aligned_pred_disp
    aligned_pred_depth = torch.clamp(aligned_pred_depth, min=1e-3, max=10.0)
    
    # Compute delta1
    ratio1 = aligned_pred_depth[valid_mask] / clip_gt_flat[valid_mask]
    ratio2 = clip_gt_flat[valid_mask] / aligned_pred_depth[valid_mask]
    thresh = torch.maximum(ratio1, ratio2)
    clip_delta1 = (thresh < 1.25).float().mean().item()
else:
    clip_delta1 = 0.0
    print("  WARNING: No valid pixels found!")

print(f"Clip δ1: {clip_delta1:.4f}")

# Pseudo-streaming mode (32 frames with causal mask)
print("\n" + "="*60)
print("Running Pseudo-Stream mode (32 frames with causal attention)...")
print("="*60)

# Create causal attention mask: [1, 1, T, T] where mask[i,j] = True if i < j
T_stream = 32
causal_mask = torch.triu(torch.ones(T_stream, T_stream, dtype=torch.bool), diagonal=1)
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T, T]

with torch.no_grad():
    # Process all 32 frames together but with causal mask
    # NOTE: This simulates streaming inference where each frame only sees past frames
    # The model needs to support attention_mask parameter
    stream_video = video[:, :32]
    
    # For now, we'll use frame-by-frame without cache to get pure causal behavior
    # This is the most accurate simulation of streaming
    stream_outputs = []
    stream_attentions = {0: [], 1: [], 2: [], 3: []}
    
    for t in tqdm(range(32), desc="Streaming"):
        # Use all frames up to t (cumulative)
        frame_batch = video[:, :t+1]  # [1, t+1, 3, H, W]
        
        out_t = model.forward(
            frame_batch,
            return_intermediates=True,
            return_qkv=True,
        )
        
        # Take only the last frame's prediction
        stream_outputs.append(out_t["pred"][:, -1])  # [1, H, W]
        
        # Collect attention for the last query (frame t)
        if "intermediates" in out_t:
            for layer_idx in [0, 1, 2, 3]:
                if layer_idx in out_t["intermediates"]:
                    attn_full = out_t["intermediates"][layer_idx].get("attention", None)
                    if attn_full is not None:
                        # attn_full: [1, H, t+1, t+1] - take last row (query=frame t)
                        attn_t = attn_full[:, :, -1:, :]  # [1, H, 1, t+1]
                        stream_attentions[layer_idx].append(attn_t.cpu())

stream_depth = torch.stack(stream_outputs, dim=1)  # [1, 32, H, W]

# Compute delta1 for stream mode
stream_gt = depth_gt[:, :32]  # [1, 32, 1, H, W]

# Ensure both have channel dimension
if stream_depth.dim() == 4:
    stream_depth = stream_depth.unsqueeze(2)
elif stream_depth.dim() == 3:
    stream_depth = stream_depth.unsqueeze(1)

# Flatten
if stream_depth.dim() == 5:
    stream_depth_flat = stream_depth.flatten(0, 1)
else:
    stream_depth_flat = stream_depth

if stream_gt.dim() == 5:
    stream_gt_flat = stream_gt.flatten(0, 1)
else:
    stream_gt_flat = stream_gt

# Compute delta1 with scale-shift alignment
valid_mask = (stream_gt_flat > 0.001) & (stream_gt_flat < 10.0)

if valid_mask.sum() > 0:
    # Convert to disparity space
    gt_disp = 1.0 / (stream_gt_flat[valid_mask] + 1e-8)
    pred_disp = stream_depth_flat[valid_mask]
    
    # Least-squares fitting
    pred_disp_np = pred_disp.cpu().numpy().reshape(-1, 1)
    gt_disp_np = gt_disp.cpu().numpy().reshape(-1, 1)
    ones = np.ones_like(pred_disp_np)
    A = np.concatenate([pred_disp_np, ones], axis=-1)
    X = np.linalg.lstsq(A, gt_disp_np, rcond=None)[0]
    scale, shift = X[0, 0], X[1, 0]
    
    # Apply alignment
    aligned_pred_disp = scale * stream_depth_flat + shift
    aligned_pred_disp = torch.clamp(aligned_pred_disp, min=1e-3)
    aligned_pred_depth = 1.0 / aligned_pred_disp
    aligned_pred_depth = torch.clamp(aligned_pred_depth, min=1e-3, max=10.0)
    
    # Compute delta1
    ratio1 = aligned_pred_depth[valid_mask] / stream_gt_flat[valid_mask]
    ratio2 = stream_gt_flat[valid_mask] / aligned_pred_depth[valid_mask]
    thresh = torch.maximum(ratio1, ratio2)
    stream_delta1 = (thresh < 1.25).float().mean().item()
else:
    stream_delta1 = 0.0

print(f"\nStream δ1: {stream_delta1:.4f}")
if clip_delta1 > 0:
    print(f"Gap: {clip_delta1 - stream_delta1:.4f} ({(clip_delta1 - stream_delta1)/clip_delta1*100:.2f}%)")
else:
    print(f"Gap: {clip_delta1 - stream_delta1:.4f} (undefined % - clip_delta1 is 0)")

# Analyze stream attention (if available)
print("\n" + "="*60)
print("Stream Attention Analysis")
print("="*60)

for layer_idx in [0, 1, 2, 3]:
    if stream_attentions[layer_idx]:
        print(f"\nLayer {layer_idx}:")
        
        # Check first attention shape
        first_attn = stream_attentions[layer_idx][0]
        print(f"  First attention shape: {first_attn.shape}")
        
        # Reconstruct full attention matrix (lower triangular)
        # Expected: each attn_t is [1, H, 1, t+1] for streaming mode
        num_heads = first_attn.shape[1]
        T = len(stream_attentions[layer_idx])
        
        # Check if streaming mode (single query per timestep)
        if first_attn.shape[2] == 1:
            stream_attn_full = torch.zeros(1, num_heads, T, T)
            
            for t, attn_t in enumerate(stream_attentions[layer_idx]):
                # attn_t: [1, H, 1, t+1] - attention from frame t to all previous frames
                k_len = attn_t.shape[3]
                if k_len == t + 1:  # causal: only attends to past
                    stream_attn_full[:, :, t, :k_len] = attn_t[:, :, 0, :]
                else:
                    print(f"  WARNING: Frame {t} has k_len={k_len}, expected {t+1}")
            
            print(f"  Reconstructed attention shape: {stream_attn_full.shape}")
            
            # Compute entropy per query (row-wise)
            # Only compute for valid entries (mask out zero rows)
            valid_rows = stream_attn_full.sum(dim=-1) > 0  # [1, H, T]
            if valid_rows.any():
                entropy_per_row = -(stream_attn_full * (stream_attn_full + 1e-10).log()).sum(dim=-1)
                entropy = entropy_per_row[valid_rows].mean()
                print(f"  attention entropy: {entropy.item():.4f}")
                
                # Compute temporal locality
                t_indices = torch.arange(T, dtype=torch.float32)
                dist_matrix = (t_indices.unsqueeze(1) - t_indices.unsqueeze(0)).abs()
                locality_per_row = (stream_attn_full * dist_matrix).sum(dim=-1)
                locality = locality_per_row[valid_rows].mean()
                print(f"  temporal locality: {locality.item():.2f} frames")
            else:
                print(f"  ✗ No valid attention rows")
        else:
            print(f"  ✗ Unexpected attention shape (not streaming mode)")
    else:
        print(f"\nLayer {layer_idx}: ✗ No attention collected")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Clip δ1: {clip_delta1:.4f}")
print(f"Stream δ1: {stream_delta1:.4f}")
print(f"Gap: {clip_delta1 - stream_delta1:.4f} ({(clip_delta1 - stream_delta1)/clip_delta1*100:.2f}%)")

if out_clip["intermediates"][0].get("attention") is not None:
    print("\n✓ Attention extraction working!")
    print("  → Ready for full H1 experiment")
else:
    print("\n✗ Attention not available")
    print("  → Check motion_module attention export")

print("="*60)
