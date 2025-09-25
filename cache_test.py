# tests/test_mixed_loader_probe_depthstats.py
import os
import yaml
import torch
import random
import itertools
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# ---- 프로젝트 모듈 ----
from data.dataLoader import KITTIVideoDataset, get_data_list
from data.dataLoader import GTADataset, get_GTA_paths
from video_depth_anything.video_depth_stream import VideoDepthAnything

# ---- 통계 유틸 ----
def _flatten_valid(d):
    # d: [H,W] (CPU tensor). 유효(>0 & finite)만 1D로 반환
    d = d.view(-1)
    valid = torch.isfinite(d) & (d > 0)
    return d[valid]

def _sample_1d(x, cap=200_000):
    # 큰 텐서에서 cap개까지 무작위 샘플링해 반환(복원 추출X)
    n = x.numel()
    if n <= cap:
        return x
    idx = torch.randperm(n)[:cap]
    return x[idx]

def _percentiles(x, ps=(0.5, 1, 5, 50, 95, 99, 99.5)):
    # x: 1D CPU tensor (float)
    if x.numel() == 0:
        return {p: float('nan') for p in ps}
    vals = torch.quantile(x, torch.tensor([p/100.0 for p in ps]))
    return {p: float(vals[i].item()) for i,p in enumerate(ps)}

def _mask_stats(x, min_d, max_d):
    # x: 1D CPU tensor (valid만 들어왔다고 가정; >0, finite)
    if x.numel() == 0:
        return dict(keep=0.0, low=0.0, high=0.0, total=0)
    total = x.numel()
    low  = (x <  min_d).sum().item()
    high = (x >  max_d).sum().item()
    keep = total - low - high
    return dict(
        keep = keep/total,
        low  = low/total,
        high = high/total,
        total= total
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 하이퍼
    CLIP_LEN = 32
    if os.path.isfile("config_jh.yaml"):
        with open("config_jh.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        CLIP_LEN = int(cfg.get("hyper_parameter", {}).get("clip_len", CLIP_LEN))

    # ---- VKITTI Dataset
    vkitti_root = "/workspace/Video-Depth-Anything/datasets/KITTI"
    vkitti_rgb, vkitti_depth = get_data_list(
        root_dir=vkitti_root, data_name="kitti", split="train", clip_len=CLIP_LEN
    )
    vkitti_ds = KITTIVideoDataset(
        rgb_paths=vkitti_rgb, depth_paths=vkitti_depth,
        clip_len=CLIP_LEN, resize_size=518, split="train"
    )
    vkitti_loader = DataLoader(vkitti_ds, batch_size=1,
                               shuffle=True, num_workers=2, pin_memory=False)

    # ---- GTA Dataset
    gta_root = "/workspace/Video-Depth-Anything/datasets/GTAV_720/GTAV_720"
    gta_rgb, gta_depth, _ = get_GTA_paths(gta_root, split="train")
    gta_ds = GTADataset(
        rgb_paths=gta_rgb, depth_paths=gta_depth,
        clip_len=CLIP_LEN, resize_size=518, split="train"
    )
    gta_loader = DataLoader(gta_ds, batch_size=1,
                            shuffle=True, num_workers=2, pin_memory=False)

    # ---- 모델 준비 (간단 forward로 VRAM 확인만)
    model = VideoDepthAnything(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        num_frames=CLIP_LEN
    ).to(device).eval()

    # ---- 통계 수집 버퍼
    # dataset key: 'VKITTI' / 'GTA'
    samples = {'VKITTI': [], 'GTA': []}
    SAMPLE_CAP_PER_APPEND = 100_000  # 프레임마다 추출할 최대 샘플 수
    STEPS = 3  # 3 step만 확인

    print("\n=== Probing Mixed Loader + Model Forward + Depth Stats ===")
    for step, (a_batch, b_batch) in enumerate(itertools.zip_longest(vkitti_loader, gta_loader, fillvalue=None)):
        if a_batch is None or b_batch is None:
            break

        a_rgb, a_depth = a_batch          # VKITTI
        b_rgb, b_depth = b_batch          # GTA
        rgb = torch.cat([a_rgb, b_rgb], dim=0)       # (2, T, 3, H, W)
        depth = torch.cat([a_depth, b_depth], dim=0) # (2, T, 1, H, W)

        # ---- 통계용 깊이 샘플 수집 (CPU에서)
        # b=0: VKITTI, b=1: GTA
        for b_idx, key in enumerate(['VKITTI', 'GTA']):
            d = depth[b_idx]  # [T,1,H,W]
            # 프레임별로 샘플 모음
            for t in range(d.shape[0]):
                dt = d[t, 0].contiguous()           # [H,W]
                dt = _flatten_valid(dt)             # 유효만
                if dt.numel() == 0:
                    continue
                dt = _sample_1d(dt, SAMPLE_CAP_PER_APPEND)
                samples[key].append(dt.cpu())

        # ---- GPU 올려서 간단 forward (VRAM peak 체크용)
        rgb_gpu = rgb.to(device, non_blocking=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), autocast(enabled=True):
            _ = model(rgb_gpu)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"[Step {step}] RGB={tuple(rgb.shape)}, VRAM peak={peak:.1f} MB")

        if step + 1 >= STEPS:
            break

    # ---- 병합 & 통계 계산
    def summarize(key, base_min=0.1, base_max=80.0):
        if len(samples[key]) == 0:
            print(f"\n[{key}] No valid depth samples collected.")
            return
        x = torch.cat(samples[key], dim=0)  # 1D
        n = x.numel()

        # 분포 요약
        ps = _percentiles(x, ps=(0.5, 1, 5, 50, 95, 99, 99.5))
        stats = {
            'count': n,
            'min': float(x.min().item()),
            'max': float(x.max().item()),
            'mean': float(x.mean().item()),
            'median': float(x.median().item()),
            'p': ps
        }

        # 기본(KITTI) 마스크 통계
        base_mask = _mask_stats(x, base_min, base_max)

        # 후보 임계값
        min_cands = [0.05, 0.1, 0.2]
        max_cands = [80.0, 100.0, 120.0, 180.0]
        grid = []
        for mn in min_cands:
            row = []
            for mx in max_cands:
                row.append(_mask_stats(x, mn, mx)['keep'])
            grid.append(row)

        # 추천 임계값 (보수적으로)
        rec_min = max(0.1, ps[1])          # 1퍼센타일 이상, 최소 0.1m
        rec_max = min(120.0, ps[99.5])     # 99.5퍼센타일 이하, 최대 120m
        rec = _mask_stats(x, rec_min, rec_max)

        # 출력
        print(f"\n[{key}] Depth distribution summary (meters)")
        print(f"  count={stats['count']}, min={stats['min']:.3f}, max={stats['max']:.3f}, "
              f"mean={stats['mean']:.3f}, median={stats['median']:.3f}")
        print("  percentiles:", ", ".join([f"p{p}={stats['p'][p]:.3f}" for p in sorted(stats['p'].keys())]))

        print(f"\n[{key}] Mask stats @ baseline [{base_min}, {base_max}] m")
        print(f"  keep={base_mask['keep']*100:.2f}%  low={base_mask['low']*100:.2f}%  high={base_mask['high']*100:.2f}%")

        print(f"\n[{key}] Keep ratio grid (rows=min, cols=max)")
        hdr = "         " + " ".join([f"{mx:>7.0f}m" for mx in max_cands])
        print(hdr)
        for i, mn in enumerate(min_cands):
            row = "min={:>4.2f}m ".format(mn) + " ".join([f"{grid[i][j]*100:6.2f}%" for j in range(len(max_cands))])
            print(row)

        print(f"\n[{key}] Recommended range ≈ [{rec_min:.2f}, {rec_max:.2f}] m "
              f"(keep≈{rec['keep']*100:.2f}%, low={rec['low']*100:.2f}%, high={rec['high']*100:.2f}%)")

    summarize('VKITTI', base_min=0.1, base_max=80.0)
    summarize('GTA',    base_min=0.1, base_max=80.0)

    print("\n[Done] Mixed Loader + Model Forward + Depth Stats finished.")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(0); torch.manual_seed(0)
    main()
