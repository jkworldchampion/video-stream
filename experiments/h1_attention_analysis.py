"""
H1: Causal Attention Degeneracy Analysis
=========================================

실험 목적:
- Clip-trained 모델을 Stream 구조로 실행 시 Attention 분포 변화 검증
- Bidirectional → Causal 전환으로 인한 attention collapse 정량화
- 성능 저하와 attention 분포 이탈의 상관관계 분석

데이터:
- ScanNet scene 0, 1 (각 500 frames)
- Clip mode: δ1 = 0.929, 0.916
- Stream mode: δ1 = 0.752, 0.662 (약 15-25% 저하)

측정 지표:
1. Attention KL Divergence: D_KL(P_clip_causal || P_stream)
2. Attention Entropy: H(P)
3. Temporal Locality Index: E[|t - t'|]
4. Performance correlation: Attn KL vs δ1
"""


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_depth_anything.motion_module.motion_module import TemporalAttention
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import cv2

# Project imports
from video_depth_anything.video_depth import VideoDepthAnything
from data.val_dataLoader import get_scannet_video_loader


class AttentionAnalyzer:
    """Attention distribution 분석 도구"""
    
    @staticmethod
    def compute_attention_kl(
        clip_attn: torch.Tensor,
        stream_attn: torch.Tensor,
        eps: float = 1e-8
    ) -> Dict[str, float]:
        """
        Clip (bidirectional) vs Stream (causal) attention KL divergence
        
        Args:
            clip_attn: [B, H, T, T] (bidirectional)
            stream_attn: [B, H, T, T] (causal)
        
        Returns:
            dict with kl_mean, kl_std, kl_per_head
        """
        B, H, T, _ = clip_attn.shape
        
        # Causal mask for clip attention (공정한 비교)
        causal_mask = torch.tril(torch.ones(T, T, device=clip_attn.device))
        clip_attn_causal = clip_attn * causal_mask.unsqueeze(0).unsqueeze(0)
        clip_attn_causal = clip_attn_causal / (clip_attn_causal.sum(dim=-1, keepdim=True) + eps)
        
        # KL divergence per head: D_KL(clip || stream)
        kl = F.kl_div(
            (stream_attn + eps).log(),
            clip_attn_causal,
            reduction='none',
            log_target=False
        ).sum(dim=-1).mean(dim=(0, 2))  # [H]
        
        return {
            "kl_mean": kl.mean().item(),
            "kl_std": kl.std().item(),
            "kl_per_head": kl.cpu().tolist(),
        }
    
    @staticmethod
    def compute_entropy(attn: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
        """
        Attention entropy: H(P) = -Σ p_i log(p_i)
        
        Args:
            attn: [B, H, T, T]
        
        Returns:
            dict with entropy_mean, entropy_std, entropy_per_head
        """
        entropy = -(attn * (attn + eps).log()).sum(dim=-1)  # [B, H, T]
        
        return {
            "entropy_mean": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
            "entropy_per_head": entropy.mean(dim=(0, 2)).cpu().tolist(),  # [H]
        }
    
    @staticmethod
    def compute_temporal_locality(
        attn: torch.Tensor,
        eps: float = 1e-8
    ) -> Dict[str, float]:
        """
        Temporal Locality Index: E[|t - t'|] (평균 참조 거리)
        
        Args:
            attn: [B, H, T, T]
        
        Returns:
            dict with locality_mean, locality_std, locality_per_head
        """
        B, H, T, _ = attn.shape
        
        # Distance matrix: |t - t'|
        t_indices = torch.arange(T, device=attn.device, dtype=torch.float32)
        dist_matrix = (t_indices.unsqueeze(1) - t_indices.unsqueeze(0)).abs()  # [T, T]
        
        # Weighted average: Σ P(t'|t) * |t - t'|
        locality = (attn * dist_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, H, T]
        
        return {
            "locality_mean": locality.mean().item(),
            "locality_std": locality.std().item(),
            "locality_per_head": locality.mean(dim=(0, 2)).cpu().tolist(),  # [H]
        }


def extract_attention_from_temporal_module(
    motion_module,
    x: torch.Tensor,
    mode: str = "clip"
) -> Optional[torch.Tensor]:
    """
    TemporalModule의 attention map 추출
    
    Args:
        motion_module: TemporalModule instance
        x: [B, C, T, H, W]
        mode: "clip" or "stream"
    
    Returns:
        attention: [B, num_heads, T, T] or None
    """
    # TemporalModule의 transformer blocks에서 attention 추출
    # 구현은 motion_module 구조에 따라 조정 필요
    try:
        # forward with attention return
        _, _, attn_maps = motion_module(x, None, None, None, return_attention=True)
        return attn_maps
    except:
        # Fallback: attention 추출 미지원
        return None


@torch.no_grad()
def compare_clip_vs_stream_attention(
    model: VideoDepthAnything,
    video_frames: np.ndarray,  # Changed from video_tensor
    depth_gt: torch.Tensor,
    scene_id: int,
    layers_to_analyze: List[int] = [0, 1, 2, 3],
    device: str = "cuda",
    checkpoint_path: str = "../video_stream/checkpoints/video_depth_anything_vits.pth"
) -> Dict:
    """
    같은 가중치, 같은 입력에 대해 Clip vs Stream attention 비교
    
    Args:
        model: VideoDepthAnything (clip-trained)
        video_frames: [T, H, W, 3] uint8 numpy array (RGB)
        depth_gt: [1, T, 1, H, W] torch.Tensor
        scene_id: Scene identifier
        layers_to_analyze: DPT temporal layers to analyze
    
    Returns:
        results: dict with attention stats and performance metrics
    """
    model.eval()
    T = video_frames.shape[0]
    
    print(f"\n{'='*60}")
    print(f"Scene {scene_id}: Analyzing {T} frames")
    print(f"{'='*60}")
    
    results = {
        "scene_id": scene_id,
        "num_frames": T,
        "layers": {},
    }
    
    # ===== 1. Clip mode (batched, bidirectional) =====
    print("Running Clip mode (bidirectional)...")
    
    # ⚠️ IMPORTANT: Use infer_video_depth() like benchmark/infer/infer.py does
    # video_frames is already [T, H, W, 3] uint8 RGB
    
    # Convert RGB to BGR for infer_video_depth (expects BGR like cv2.imread)
    print("  Converting RGB to BGR...")
    frames_bgr = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in video_frames], axis=0)

    
    print("=============VDA model==============")
    temporal_attn_modules = []
    for name, module in model.named_modules():
        if isinstance(module, TemporalAttention):
            temporal_attn_modules.append((name, module))

    print("temporalAttention modules 이름 : ")
    for name, _ in temporal_attn_modules:
        print("  ", name)

        
    # Run inference with proper temporal alignment (like benchmark does)
    print("  Running infer_video_depth() with temporal alignment...")
    clip_depth_list, _ = model.infer_video_depth(
        frames_bgr, 
        target_fps=1, 
        input_size=518, 
        device=device, 
        fp32=True
    )

    for name, m in temporal_attn_modules:
        print(f"[{name}] attention_score:")
        print(m.attention_score.shape)
    
    hola_dir = "attention_score/clip"
    
    for name, m in temporal_attn_modules:
        attn = m.attention_score
        attn_step_mean = attn.mean(dim=0)
        attn_final_mean = attn_step_mean.mean(dim=0)
    
        plt.figure(figsize=(5,5))
        plt.imshow(attn_final_mean.detach().cpu().numpy(), cmap='viridis')
        plt.title(f"{name}", fontsize=6)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(hola_dir,f"att_{name}.png"))
    
    print("hola~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    #sys.exit()
    
    # Convert to tensor [1, T, H, W]
    clip_depth = torch.from_numpy(np.stack(clip_depth_list, axis=0)).unsqueeze(0)
    
    # Note: Attention extraction requires raw forward(), but that doesn't match benchmark
    # We prioritize correct depth estimation over attention analysis
    clip_attentions = {layer: [] for layer in layers_to_analyze}
    print(f"  Clip depth shape: {clip_depth.shape}")


    # ===== 2. Stream mode (REAL streaming with frame-by-frame processing) =====
    print("Running Stream mode (real streaming, frame-by-frame)...")
    
    # Import streaming model
    from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream


    stream_checkpoint_path = "/home/work/juhwan/monocular_depth/stream/video-stream/outputs/experiment_4/best_model.pth"
    # stream_checkpoint_path = "/home/work/juhwan/monocular_depth/stream/video-stream/checkpoints/video_depth_anything_vits.pth"
    
    # Initialize streaming model
    model_stream = VideoDepthAnythingStream(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        num_frames=32
    )
    model_stream.load_state_dict(torch.load(stream_checkpoint_path, map_location="cpu"), strict=False)
    model_stream = model_stream.to(device).eval()
    
    # Reset streaming state
    model_stream.transform = None
    model_stream.frame_cache_list = []
    model_stream.frame_id_list = []
    model_stream.id = -1
    
    stream_outputs = []

    print("=============STREAM model==============")
    stream_temporal_attn_modules = []
    for name, module in model_stream.named_modules():
        if isinstance(module, TemporalAttention):
            stream_temporal_attn_modules.append((name, module))

    print("stream_temporalAttention modules 이름 : ")
    for name, _ in stream_temporal_attn_modules:
        print("  ", name)
    
    # Process frame by frame (REAL streaming)
    # video_frames is already [T, H, W, 3] uint8 RGB (infer_video_depth_one expects RGB)
    for t in tqdm(range(T), desc="Stream mode"):
        frame_rgb = video_frames[t]  # [H, W, 3] uint8 RGB
        
        # Streaming inference (1 frame at a time)
        depth_np = model_stream.infer_video_depth_one(
            frame_rgb,
            input_size=518,
            device=device,
            fp32=True
        )
        
        stream_outputs.append(torch.from_numpy(depth_np).unsqueeze(0))  # [1, H, W]
        
        # Note: Streaming model attention extraction is complex
        # We skip attention analysis for stream mode (focus on performance gap)

    for name, m in stream_temporal_attn_modules:
        print(f"[{name}] attention_score:")
        print(m.attention_score.shape)
    
    streamhola_dir = "attention_score/stream_vda"
        
    for name, m in stream_temporal_attn_modules:
        attn = m.attention_score 
        attn_mean = attn.mean(dim=(0, 1,2))  
        attn_img = attn_mean.detach().cpu().numpy()[None, :]
    
        plt.figure(figsize=(5, 2))
        plt.imshow(attn_img, aspect="auto", cmap='viridis')
        plt.yticks([]) 
        plt.xlabel("key index (cache + current)")
        plt.title(f"{name}", fontsize=8)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(streamhola_dir, f"att_{name}.png"))
        plt.close()
    
    print("hola~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    sys.exit()


    
    # ===== 3. Performance 계산 (scale-shift alignment) =====
    print("Computing performance metrics...")
    
    def compute_delta1_aligned(pred, gt, min_depth=1e-3, max_depth=10.0):
        """
        Compute delta1 accuracy with disparity alignment (matches eval.py).
        """
        # Flatten batch and time dimensions: [B, T, H, W] -> [B*T, H, W]
        pred_flat = pred.flatten(0, 1)  # [B*T, H, W]
        gt_flat = gt.flatten(0, 1).squeeze(1)  # [B*T, H, W]
        
        # Resize pred to GT size (matches eval.py behavior)
        if pred_flat.shape[-2:] != gt_flat.shape[-2:]:
            pred_flat = torch.nn.functional.interpolate(
                pred_flat.unsqueeze(1),  # [B*T, 1, H_pred, W_pred]
                size=gt_flat.shape[-2:],  # (H_gt, W_gt)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [B*T, H_gt, W_gt]
        
        # C. Apply ScanNet crop mask (a=8, b=-8, c=11, d=-11)
        H, W = gt_flat.shape[-2:]
        crop_mask = torch.zeros_like(gt_flat, dtype=torch.bool)
        crop_mask[..., 8:(H-8), 11:(W-11)] = True
        
        # Valid mask: depth range + crop
        valid_mask = (gt_flat > min_depth) & (gt_flat < max_depth) & crop_mask
        
        if valid_mask.sum() == 0:
            return 0.0
        
        # D. Alignment in disparity space (pred is already disparity!)
        gt_disp = 1.0 / (gt_flat[valid_mask] + 1e-8)  # GT depth -> disparity
        pred_disp = pred_flat[valid_mask]  # Pred is already disparity
        
        # Least-squares: gt_disp ≈ scale * pred_disp + shift
        pred_disp_np = pred_disp.detach().cpu().double().view(-1, 1).numpy()
        gt_disp_np = gt_disp.detach().cpu().double().view(-1, 1).numpy()
        ones = np.ones_like(pred_disp_np)
        A = np.concatenate([pred_disp_np, ones], axis=-1)
        X = np.linalg.lstsq(A, gt_disp_np, rcond=None)[0]
        scale, shift = float(X[0, 0]), float(X[1, 0])
        
        # Apply alignment: disparity -> depth
        aligned_pred_disp = scale * pred_flat + shift
        aligned_pred_disp = torch.clamp(aligned_pred_disp, min=1e-3)
        aligned_pred_depth = 1.0 / aligned_pred_disp  # disparity -> depth
        aligned_pred_depth = torch.clamp(aligned_pred_depth, min=1e-3, max=max_depth)
        
        # Delta1 metric
        ratio1 = aligned_pred_depth[valid_mask] / gt_flat[valid_mask]
        ratio2 = gt_flat[valid_mask] / aligned_pred_depth[valid_mask]
        thresh = torch.maximum(ratio1, ratio2)
        delta1 = (thresh < 1.25).float().mean()
        return delta1.item()
    
    # Clip mode performance
    clip_delta1 = compute_delta1_aligned(clip_depth, depth_gt[:, :clip_depth.shape[1], 0])
    
    # Stream mode performance  
    # stream_outputs: list of [1, H, W] -> stack to [1, T, H, W]
    stream_depth_tensor = torch.stack(stream_outputs, dim=1)  # [1, T, H, W]
    stream_delta1 = compute_delta1_aligned(stream_depth_tensor, depth_gt[:, :T, 0])
    
    results["performance"] = {
        "clip_delta1": clip_delta1,
        "stream_delta1": stream_delta1,
        "delta1_gap": clip_delta1 - stream_delta1,
        "delta1_gap_percent": (clip_delta1 - stream_delta1) / clip_delta1 * 100,
    }
    
    print(f"Clip δ1: {clip_delta1:.4f}")
    print(f"Stream δ1: {stream_delta1:.4f}")
    print(f"Gap: {results['performance']['delta1_gap']:.4f} ({results['performance']['delta1_gap_percent']:.2f}%)")
    
    # ===== 4. Attention 분석 (레이어별) =====
    print("Analyzing attention distributions...")
    
    # Note: Stream attention analysis is skipped since streaming model
    # doesn't easily export attention maps during incremental inference
    print("⚠️  Stream attention analysis skipped (focus on performance gap)")
    
    for layer_idx in layers_to_analyze:
        if not clip_attentions[layer_idx]:
            print(f"Layer {layer_idx}: Clip attention not available, skipping...")
            continue
        
        print(f"\nLayer {layer_idx}:")
        print("  ⚠️  Attention analysis skipped for this experiment")
        print("  (Focus on performance gap; attention extraction from streaming model is complex)")
    
    return results


def visualize_results(results_list: List[Dict], save_dir: Path):
    """결과 시각화"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    
    scenes = [r["scene_id"] for r in results_list]
    clip_delta1s = [r["performance"]["clip_delta1"] for r in results_list]
    stream_delta1s = [r["performance"]["stream_delta1"] for r in results_list]
    
    x = np.arange(len(scenes))
    width = 0.35
    
    ax.bar(x - width/2, clip_delta1s, width, label='Clip (bidirectional)', color='#2ecc71')
    ax.bar(x + width/2, stream_delta1s, width, label='Stream (causal)', color='#e74c3c')
    
    ax.set_ylabel('δ1 (↑ better)', fontsize=12)
    ax.set_xlabel('Scene ID', fontsize=12)
    ax.set_title('Performance: Clip vs Stream', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Scene {s}" for s in scenes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Performance comparison saved to {save_dir / 'performance_comparison.png'}")
    
    # 2. Attention metrics (레이어별 평균) - Skip if no attention data
    if results_list[0]["layers"]:
        layers = list(results_list[0]["layers"].keys())
        
        metrics_data = {
            "attn_kl": [],
            "entropy_ratio": [],
            "locality_gap": [],
        }
        
        for layer in layers:
            kl_vals = [r["layers"][layer]["attn_kl"]["kl_mean"] for r in results_list]
            ent_ratios = [r["layers"][layer]["entropy_ratio"] for r in results_list]
            loc_gaps = [r["layers"][layer]["locality_gap"] for r in results_list]
            
            metrics_data["attn_kl"].append(np.mean(kl_vals))
            metrics_data["entropy_ratio"].append(np.mean(ent_ratios))
            metrics_data["locality_gap"].append(np.mean(loc_gaps))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Attn KL
        axes[0].bar(layers, metrics_data["attn_kl"], color='#3498db')
        axes[0].set_xlabel('Layer', fontsize=11)
        axes[0].set_ylabel('Attention KL Divergence', fontsize=11)
        axes[0].set_title('Distribution Shift (Clip → Stream)', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Entropy ratio
        axes[1].bar(layers, metrics_data["entropy_ratio"], color='#9b59b6')
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal entropy')
        axes[1].set_xlabel('Layer', fontsize=11)
        axes[1].set_ylabel('Entropy ratio (stream/clip)', fontsize=11)
        axes[1].set_title('Attention Collapse', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        # Locality gap
        axes[2].bar(layers, metrics_data["locality_gap"], color='#e67e22')
        axes[2].set_xlabel('Layer', fontsize=11)
        axes[2].set_ylabel('Locality gap (frames)', fontsize=11)
        axes[2].set_title('Long-range Reference Loss', fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "attention_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Attention metrics saved to {save_dir / 'attention_metrics.png'}")
    else:
        print("⚠️  Attention metrics visualization skipped (no attention data)")


def main():
    """Main experiment"""
    
    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/home/work/juhwan/monocular_depth/stream/video-stream/checkpoints/video_depth_anything_vits.pth"
    scannet_json = "/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
    
    save_dir = Path("experiments/results/h1_attention_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("H1: Causal Attention Degeneracy Analysis")
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {scannet_json}")
    print(f"Save dir: {save_dir}")
    
    # Load model
    print("\nLoading model...")
    model = VideoDepthAnything(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        num_frames=32
    ).to(device)
    
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("Model loaded!")
    
    # Load data (scene 0, 1) - 직접 로딩
    print("\nLoading ScanNet data...")
    
    with open(scannet_json) as f:
        data = json.load(f)
    
    from PIL import Image
    import torchvision.transforms.functional as TF
    
    results_list = []
    
    for scene_idx in [0, 1]:  # Scene 0, 1
        scene_data = data["scannet"][scene_idx]
        scene_name = list(scene_data.keys())[0]
        frames = scene_data[scene_name]
        
        print(f"\nLoading Scene {scene_idx}: {scene_name}, {len(frames)} frames...")
        
        # Load 500 frames (FULL LENGTH - 필수!)
        num_frames = min(500, len(frames))
        
        imgs = []
        depths = []
        root_dir = "/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/"
        
        for frame_info in tqdm(frames[:num_frames], desc=f"Loading scene {scene_idx}"):
            # A. NO CENTER CROP - Load original image as-is
            img_path = os.path.join(root_dir, frame_info["image"])
            frame_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR uint8 (for clip)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # RGB uint8 (for stream)
            imgs.append(frame_rgb)
            
            # GT: Load without crop (crop will be applied via mask during evaluation)
            depth_path = os.path.join(root_dir, frame_info["gt_depth"])
            depth = Image.open(depth_path).convert("F")
            depth = torch.from_numpy(np.array(depth, np.float32)).unsqueeze(0) / 1000.0  # mm -> m
            depths.append(depth)
        
        video = np.stack(imgs, axis=0)  # [T, H, W, 3] RGB uint8
        depth_gt = torch.stack(depths, dim=0).unsqueeze(0)  # [1, T, 1, H, W]
        
        results = compare_clip_vs_stream_attention(
            model=model,
            video_frames=video,  # Now numpy array [T, H, W, 3]
            depth_gt=depth_gt,
            scene_id=scene_idx,
            layers_to_analyze=[0, 1, 2, 3],
            device=device,
        )
        
        results_list.append(results)
        
        # Save intermediate results
        with open(save_dir / f"scene_{scene_idx}_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Aggregate and visualize
    print("\n" + "="*60)
    print("Generating visualizations...")
    visualize_results(results_list, save_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for results in results_list:
        scene_id = results["scene_id"]
        perf = results["performance"]
        
        print(f"\nScene {scene_id}:")
        print(f"  Clip δ1: {perf['clip_delta1']:.4f}")
        print(f"  Stream δ1: {perf['stream_delta1']:.4f}")
        print(f"  Gap: {perf['delta1_gap']:.4f} ({perf['delta1_gap_percent']:.2f}%)")
        print(f"  Gap %: {perf['delta1_gap_percent']:.2f}%")
        
        if results["layers"]:
            avg_kl = np.mean([r["attn_kl"]["kl_mean"] for r in results["layers"].values()])
            avg_entropy_ratio = np.mean([r["entropy_ratio"] for r in results["layers"].values()])
            avg_locality_gap = np.mean([r["locality_gap"] for r in results["layers"].values()])
            
            print(f"  Avg Attn KL: {avg_kl:.4f}")
            print(f"  Avg Entropy ratio: {avg_entropy_ratio:.4f}")
            print(f"  Avg Locality gap: {avg_locality_gap:.2f} frames")
        else:
            print("  (Attention analysis skipped)")
    
    print(f"\nResults saved to {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
