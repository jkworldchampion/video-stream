import os
import logging
import argparse
from dotenv import load_dotenv

import torch
import numpy as np
import yaml
import wandb

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

# ────────────────────────────────────────────────────────────────────────────────
# project‑level imports
# ────────────────────────────────────────────────────────────────────────────────
from utils.loss_MiDas import LossTGMVector, Loss_ssi_basic
from data.dataLoader import get_data_list, KITTIVideoDataset
from data.val_dataLoader import ValDataset, get_list
from video_depth_anything.video_depth_stream import VideoDepthAnything
from benchmark.eval.metric import abs_relative_difference, delta1_acc
from benchmark.eval.eval_tae import tae_torch
from PIL import Image
from model_utils import save_model_checkpoint, get_model_info

# ==============================================================================
# 0. 기본 설정
# ==============================================================================
EXPERIMENT_ID = 2
LOG_DIR = "logs"
OUTPUT_DIR = f"outputs/experiment_{EXPERIMENT_ID}"
CONFIG_PATH = "configs/config.yaml"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, f"train_log_experiment_{EXPERIMENT_ID}.txt")),
    ],
)
logger = logging.getLogger(__name__)

# torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

MEAN = torch.tensor((0.485, 0.456, 0.406), device=device).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=device).view(3, 1, 1)

# ==============================================================================
# 1. Loss / Metric utils (unchanged)
# ==============================================================================

def least_square_whole_clip(infs, gts, data):
    if infs.dim() == 5 and infs.shape[2] == 1:
        infs = infs.squeeze(2)
    if gts.dim() == 5 and gts.shape[2] == 1:
        gts = gts.squeeze(2)

    if data == "kitti":
        valid_mask = (gts > 1e-3) & (gts < 80.0)
    elif data == "scannet":
        valid_mask = (gts > 1e-3) & (gts < 10.0)
    else:
        valid_mask = (gts > 1e-3) & (gts < 1000.0)

    gt_disp_masked = 1.0 / (gts[valid_mask].reshape(-1, 1).double() + 1e-6)
    pred_disp_masked = infs[valid_mask].reshape(-1, 1).double().clamp(min=1e-3)

    ones = torch.ones_like(pred_disp_masked)
    A = torch.cat([pred_disp_masked, ones], dim=-1)
    X = torch.linalg.lstsq(A, gt_disp_masked).solution
    scale, shift = X[0].item(), X[1].item()

    aligned_pred = torch.clamp(scale * infs + shift, min=1e-3)
    depth = 1.0 / aligned_pred
    return depth, valid_mask


def metric_val(infs, gts, data, poses=None, Ks=None):
    pred_depth, valid_mask = least_square_whole_clip(infs, gts, data)
    n_valid = valid_mask.sum((-1, -2))
    valid_frame = n_valid > 0

    pred_depth = pred_depth[valid_frame]
    gt_depth = gts[valid_frame]
    valid_mask = valid_mask[valid_frame]

    absrel = abs_relative_difference(pred_depth, gt_depth, valid_mask)
    delta1 = delta1_acc(pred_depth, gt_depth, valid_mask)

    if poses is not None:
        tae = eval_tae(pred_depth, gt_depth, poses, Ks, valid_mask)
        return absrel, delta1, tae
    return absrel, delta1


def eval_tae(pred_depth, gt_depth, poses, Ks, masks):
    error_sum = 0.0
    for i in range(len(pred_depth) - 1):
        depth1, depth2 = pred_depth[i], pred_depth[i + 1]
        mask1, mask2 = masks[i], masks[i + 1]
        T1, T2 = poses[i], poses[i + 1]

        try:
            T2_inv = torch.linalg.inv(T2)
        except torch._C._LinAlgError:
            T2_inv = torch.linalg.pinv(T2)
        T2_1 = T2_inv @ T1
        R2_1, t2_1 = T2_1[:3, :3], T2_1[:3, 3]
        K = Ks[i].view(3, 3) if Ks[i].numel() == 9 else Ks[i]
        error1 = tae_torch(depth1, depth2, R2_1, t2_1, K, mask2)

        try:
            T1_2 = torch.linalg.inv(T2_1)
        except torch._C._LinAlgError:
            T1_2 = torch.linalg.pinv(T2_1)
        R1_2, t1_2 = T1_2[:3, :3], T1_2[:3, 3]
        error2 = tae_torch(depth2, depth1, R1_2, t1_2, K, mask1)
        error_sum += error1 + error2
    return error_sum / (2 * (len(pred_depth) - 1))


def get_mask(depth_m, min_depth, max_depth):
    return ((depth_m > min_depth) & (depth_m < max_depth)).bool()


def norm_ssi(depth, valid_mask):
    disparity = torch.zeros_like(depth)
    disparity[valid_mask] = 1.0 / depth[valid_mask]

    B, T, C, H, W = disparity.shape
    disp_flat = disparity.view(B, T, -1)
    mask_flat = valid_mask.view(B, T, -1)

    disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).min(dim=-1)[0].view(B, T, 1, 1, 1)
    disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).max(dim=-1)[0].view(B, T, 1, 1, 1)

    norm_disp = (disparity - disp_min) / (disp_max - disp_min + 1e-6)
    return norm_disp.masked_fill(~valid_mask, 0.0)

# ==============================================================================
# 2. Training loop
# ==============================================================================

def train(args):
    # ── 2‑1. W&B & config ────────────────────────────────────────────────────
    load_dotenv(".env")
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)["hyper_parameter"]

    run = wandb.init(project="stream_causal_block", entity="Depth-Finder", config=cfg)

    # ── 2‑2. Dataset (KITTI train / val, ScanNet val) ────────────────────────
    CLIP_LEN = cfg["clip_len"]
    batch_size = cfg["batch_size"]

    kitti_root = "/workspace/Video-Depth-Anything/datasets/KITTI"
    rgb_train, dep_train = get_data_list(kitti_root, "kitti", "train", CLIP_LEN)
    rgb_val, dep_val, cam_ids, intrin_val, extrin_val = get_data_list(kitti_root, "kitti", "val", CLIP_LEN)

    kitti_train_ds = KITTIVideoDataset(rgb_train, dep_train, resize_size=350, split="train")
    kitti_val_ds = KITTIVideoDataset(rgb_val, dep_val, cam_ids, intrin_val, extrin_val, resize_size=350, split="val")

    kitti_train_loader = DataLoader(kitti_train_ds, batch_size=batch_size, shuffle=True, num_workers=6)
    kitti_val_loader = DataLoader(kitti_val_ds, batch_size=batch_size, shuffle=False, num_workers=6)

    x_scan, y_scan, scan_poses, scan_Ks = get_list("", "scannet")
    scannet_val_ds = ValDataset(x_scan, y_scan, "scannet", Ks=scan_Ks, pose_paths=scan_poses)
    scannet_val_loader = DataLoader(scannet_val_ds, batch_size=batch_size, shuffle=False, num_workers=6)

    # ── 2‑3. Model ───────────────────────────────────────────────────────────
    logger.info("Creating VideoDepthAnything model (streaming mode)…")
    model = VideoDepthAnything(num_frames=CLIP_LEN,
                               use_causal_mask=True,
                               encoder='vits',
                               features=64,
                               out_channels=[48, 96, 192, 384]).to(device)

    if args.pretrained_ckpt:
        logger.info(f"Loading checkpoint from {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict() and v.size() == model.state_dict()[k].size()}, strict=False)

    # Freeze encoder
    for p in model.pretrained.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    scaler = GradScaler()
    loss_tgm = LossTGMVector(diff_depth_th=0.05)
    loss_ssi = Loss_ssi_basic()
    ratio_tgm = cfg["ratio_tgm"]
    ratio_ssi = cfg["ratio_ssi"]

    # 모델 저장 관련 변수
    best_val_loss = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, "best_progressive_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_progressive_model.pth")

    # ── 2‑4. Epoch loop with progressive sequence learning ──────────────────
    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        # Progressive sequence length: start with short sequences, gradually increase
        max_seq_len = min(4 + epoch * 2, CLIP_LEN)  # 4 -> 6 -> 8 -> ... -> CLIP_LEN
        logger.info(f"Epoch {epoch}: Using sequence length {max_seq_len}")
        
        for batch_idx, (x, y) in enumerate(tqdm(kitti_train_loader, desc=f"Train‑E{epoch}")):
            masks = get_mask(y, 0.001, 80.0).to(device)
            x, y = x.to(device), y.to(device)
            
            B, T, C, H, W = x.shape
            
            # Progressive sequence training
            for start_idx in range(0, T - max_seq_len + 1, max_seq_len // 2):  # Overlapping windows
                end_idx = min(start_idx + max_seq_len, T)
                actual_len = end_idx - start_idx
                
                if actual_len < 2:  # Skip too short sequences
                    continue
                    
                x_sub = x[:, start_idx:end_idx]
                y_sub = y[:, start_idx:end_idx]
                masks_sub = masks[:, start_idx:end_idx]

                optimizer.zero_grad()
                with autocast():
                    # Reset model state for each subsequence (simulating streaming)
                    if hasattr(model, 'module'):
                        model.module.frame_cache_list = []
                        model.module.frame_id_list = []
                        model.module.id = -1
                    else:
                        model.frame_cache_list = []
                        model.frame_id_list = []
                        model.id = -1
                    
                    pred = model(x_sub)
                    disp_normed = norm_ssi(y_sub, masks_sub)
                    ssi_l = loss_ssi(pred, disp_normed, masks_sub.squeeze(2))
                    tgm_l = loss_tgm(pred, y_sub, masks_sub)
                    loss = ratio_tgm * tgm_l + ratio_ssi * ssi_l
                    
                    # Weight loss by sequence length to balance short vs long sequences
                    loss = loss * (actual_len / max_seq_len)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

        wandb.log({"train_loss": epoch_loss / len(kitti_train_loader), 
                  "sequence_length": max_seq_len, "epoch": epoch})

        # ── Validation on KITTI ─────────────────────────────────────────────
        val_metrics = {"loss": 0.0, "absrel": 0.0, "delta1": 0.0, "tae": 0.0, "cnt": 0}
        model.eval()
        with torch.no_grad():
            for x, y, extr, intr in tqdm(kitti_val_loader, desc=f"VKITTI‑Val‑E{epoch}"):
                x, y, extr, intr = x.to(device), y.to(device), extr.to(device), intr.to(device)
                pred = model(x)
                masks = get_mask(y, 0.001, 80.0).to(device)
                disp_normed = norm_ssi(y, masks)
                val_loss = ratio_ssi * loss_ssi(pred, disp_normed, masks.squeeze(2)) + ratio_tgm * loss_tgm(pred, y, masks)
                val_metrics["loss"] += val_loss

                B = pred.size(0)
                for b in range(B):
                    a, d, t = metric_val(pred[b], y[b].squeeze(1), "kitti", extr[b], intr[b])
                    val_metrics["absrel"] += a
                    val_metrics["delta1"] += d
                    val_metrics["tae"] += t
                    val_metrics["cnt"] += 1
        for k in ["loss", "absrel", "delta1", "tae"]:
            val_metrics[k] /= val_metrics["cnt"] if k != "loss" else len(kitti_val_loader)
        wandb.log({f"vkitti_{k}": v for k, v in val_metrics.items() if k != "cnt"})

        # ── Validation on ScanNet ───────────────────────────────────────────
        val_metrics = {"loss": 0.0, "absrel": 0.0, "delta1": 0.0, "tae": 0.0, "cnt": 0}
        with torch.no_grad():
            for x, y, extr, intr in tqdm(scannet_val_loader, desc=f"ScanNet‑Val‑E{epoch}"):
                x, y, extr, intr = x.to(device), y.to(device), extr.to(device), intr.to(device)
                pred = model(x)
                masks = get_mask(y, 0.001, 10.0).to(device)
                disp_normed = norm_ssi(y, masks)
                val_loss = ratio_ssi * loss_ssi(pred, disp_normed, masks.squeeze(2)) + ratio_tgm * loss_tgm(pred, y, masks)
                val_metrics["loss"] += val_loss

                B = pred.size(0)
                for b in range(B):
                    a, d, t = metric_val(pred[b], y[b].squeeze(1), "scannet", extr[b], intr[b])
                    val_metrics["absrel"] += a
                    val_metrics["delta1"] += d
                    val_metrics["tae"] += t
                    val_metrics["cnt"] += 1
        for k in ["loss", "absrel", "delta1", "tae"]:
            val_metrics[k] /= val_metrics["cnt"] if k != "loss" else len(scannet_val_loader)
        wandb.log({f"scannet_{k}": v for k, v in val_metrics.items() if k != "cnt"})

        # ── 모델 저장 ────────────────────────────────────────────────────────
        # 평균 validation loss 계산 (KITTI + ScanNet)
        # current_val_loss = (val_metrics["loss"] + val_metrics["loss"]) / 2  # 두 데이터셋 평균
        current_val_loss = val_metrics["loss"]
        
        # Best model 저장
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'sequence_length': max_seq_len,
                'config': cfg
            }, best_model_path)
            logger.info(f"Best model saved to {best_model_path} with val_loss: {best_val_loss:.4f}")
        
        # Latest model 저장 (이전 것 덮어쓰기)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'current_val_loss': current_val_loss,
            'sequence_length': max_seq_len,
            'config': cfg
        }, latest_model_path)
        logger.info(f"Latest model saved to {latest_model_path}")

        scheduler.step()
    run.finish()

# ==============================================================================
# 3. Entry‑point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")
    args = parser.parse_args()
    train(args)
