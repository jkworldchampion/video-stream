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

from utils.loss_MiDas import LossTGMVector, Loss_ssi_basic
from data.dataLoader import get_data_list, KITTIVideoDataset
from data.val_dataLoader import ValDataset, get_list
from video_depth_anything.video_depth_stream import VideoDepthAnything
from benchmark.eval.metric import abs_relative_difference, delta1_acc
from benchmark.eval.eval_tae import tae_torch
from PIL import Image
import copy
from model_utils import save_model_checkpoint, get_model_info

# ==============================================================================
# 캐시 기반 온라인 학습
# ==============================================================================

class CacheBasedTrainer:
    def __init__(self, model, optimizer, scaler, loss_tgm, loss_ssi, ratio_tgm, ratio_ssi, device):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.loss_tgm = loss_tgm
        self.loss_ssi = loss_ssi
        self.ratio_tgm = ratio_tgm
        self.ratio_ssi = ratio_ssi
        self.device = device
        
        # 캐시 상태 관리
        self.cached_hidden_states = None
        self.loss_buffer = []
        self.gradient_accumulation_steps = 4
        
    def reset_cache(self):
        """새로운 시퀀스 시작 시 캐시 리셋"""
        self.cached_hidden_states = None
        self.loss_buffer = []
        
    def forward_with_cache(self, x_frame, gt_frame, mask_frame):
        """
        캐시를 사용한 순차적 forward pass
        x_frame: [B, 1, C, H, W] - 단일 프레임
        """
        # 단일 프레임에 대한 feature 추출
        with torch.no_grad():
            features = self.model.forward_features(x_frame)
        
        # 캐시와 함께 depth 예측
        B, T, C, H, W = x_frame.shape
        patch_h, patch_w = H // 14, W // 14
        
        if hasattr(self.model, 'module'):
            depth, new_cache = self.model.module.head.forward(
                features, patch_h, patch_w, T, 
                cached_hidden_state_list=self.cached_hidden_states
            )
        else:
            depth, new_cache = self.model.head.forward(
                features, patch_h, patch_w, T,
                cached_hidden_state_list=self.cached_hidden_states
            )
        
        # 캐시 업데이트
        self.cached_hidden_states = new_cache
        
        # Depth 후처리
        depth = torch.nn.functional.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = torch.nn.functional.relu(depth)
        depth = depth.squeeze(1).unflatten(0, (B, T))
        
        return depth
    
    def train_step_online(self, x_sequence, y_sequence, mask_sequence):
        """
        온라인 학습: 프레임을 하나씩 처리하면서 캐시 유지
        """
        self.reset_cache()
        total_loss = 0.0
        frame_count = 0
        
        B, T, C, H, W = x_sequence.shape
        
        # Enable gradients for trainable parameters
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = None
        
        for t in range(T):
            # 현재 프레임 추출
            x_frame = x_sequence[:, t:t+1]  # [B, 1, C, H, W]
            y_frame = y_sequence[:, t:t+1]  # [B, 1, 1, H, W]
            mask_frame = mask_sequence[:, t:t+1]  # [B, 1, 1, H, W]
            
            # Forward pass with cache
            with autocast():
                pred_depth = self.forward_with_cache(x_frame, y_frame, mask_frame)
                
                # Loss 계산
                if mask_frame.sum() > 0:
                    # SSI loss
                    disp_normed = self.norm_ssi_single(y_frame, mask_frame)
                    ssi_l = self.loss_ssi(pred_depth, disp_normed, mask_frame.squeeze(2))
                    
                    # TGM loss
                    tgm_l = self.loss_tgm(pred_depth, y_frame, mask_frame)
                    
                    frame_loss = self.ratio_tgm * tgm_l + self.ratio_ssi * ssi_l
                else:
                    frame_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # Gradient accumulation
                frame_loss = frame_loss / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(frame_loss).backward()
            total_loss += frame_loss.item() * self.gradient_accumulation_steps
            frame_count += 1
            
            # Update every N frames
            if (t + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        
        # Handle remaining gradients
        if T % self.gradient_accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return total_loss / frame_count if frame_count > 0 else 0.0
    
    def norm_ssi_single(self, depth, valid_mask):
        """Single frame SSI normalization"""
        disparity = torch.zeros_like(depth)
        disparity[valid_mask] = 1.0 / depth[valid_mask]
        
        B, T, C, H, W = disparity.shape
        disp_flat = disparity.view(B, T, -1)
        mask_flat = valid_mask.view(B, T, -1)
        
        disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).min(dim=-1)[0].view(B, T, 1, 1, 1)
        disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).max(dim=-1)[0].view(B, T, 1, 1, 1)
        
        norm_disp = (disparity - disp_min) / (disp_max - disp_min + 1e-6)
        return norm_disp.masked_fill(~valid_mask, 0.0)

def train_cache_based(args):
    """캐시 기반 온라인 학습"""
    
    # 기본 설정
    EXPERIMENT_ID = 3
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}, CUDA available: {torch.cuda.is_available()}")
    
    # W&B 설정
    load_dotenv(".env")
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)["hyper_parameter"]

    run = wandb.init(project="stream_causal_cache_based", entity="Depth-Finder", config=cfg)

    # 데이터셋
    CLIP_LEN = cfg["clip_len"]
    batch_size = 1  # 캐시 기반 학습은 배치 크기 1

    kitti_root = "/workspace/Video-Depth-Anything/datasets/KITTI"
    rgb_train, dep_train = get_data_list(kitti_root, "kitti", "train", CLIP_LEN)

    kitti_train_ds = KITTIVideoDataset(rgb_train, dep_train, resize_size=350, split="train")
    kitti_train_loader = DataLoader(kitti_train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # 모델
    logger.info("Creating VideoDepthAnything model (cache-based streaming mode)...")
    model = VideoDepthAnything(num_frames=CLIP_LEN,
                               use_causal_mask=True,
                               encoder='vits',
                               features=64,
                               out_channels=[48, 96, 192, 384]).to(device)

    if args.pretrained_ckpt:
        logger.info(f"Loading checkpoint from {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict({k: v for k, v in state_dict.items() 
                              if k in model.state_dict() and v.size() == model.state_dict()[k].size()}, 
                             strict=False)

    # Encoder 고정
    for p in model.pretrained.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 옵티마이저
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    # 손실 함수
    scaler = GradScaler()
    loss_tgm = LossTGMVector(diff_depth_th=0.05)
    loss_ssi = Loss_ssi_basic()
    ratio_tgm = cfg["ratio_tgm"]
    ratio_ssi = cfg["ratio_ssi"]

    # 캐시 기반 트레이너
    cache_trainer = CacheBasedTrainer(
        model, optimizer, scaler, loss_tgm, loss_ssi, ratio_tgm, ratio_ssi, device
    )

    # 모델 저장 관련 변수
    best_train_loss = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, "best_cache_based_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_cache_based_model.pth")

    # 학습 루프
    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_losses = []
        
        for batch_idx, (x, y) in enumerate(tqdm(kitti_train_loader, desc=f"Cache-Based-Train-E{epoch}")):
            masks = ((y > 0.001) & (y < 80.0)).bool().to(device)
            x, y = x.to(device), y.to(device)
            
            # 캐시 기반 온라인 학습
            avg_loss = cache_trainer.train_step_online(x, y, masks)
            epoch_losses.append(avg_loss)
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}")

        # 에포크 결과
        if epoch_losses:
            avg_epoch_loss = np.mean(epoch_losses)
            wandb.log({"cache_based_train_loss": avg_epoch_loss, "epoch": epoch})
            logger.info(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")
            
            # ── 모델 저장 ────────────────────────────────────────────────────────
            # Best model 저장 (train loss 기준)
            if avg_epoch_loss < best_train_loss:
                best_train_loss = avg_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_train_loss': best_train_loss,
                    'config': cfg
                }, best_model_path)
                logger.info(f"Best cache-based model saved to {best_model_path} with train_loss: {best_train_loss:.4f}")
            
            # Latest model 저장 (이전 것 덮어쓰기)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'current_train_loss': avg_epoch_loss,
                'config': cfg
            }, latest_model_path)
            logger.info(f"Latest cache-based model saved to {latest_model_path}")

        scheduler.step()
        
        # 주기적으로 추가 체크포인트 저장 (선택사항)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"cache_based_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'config': cfg
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")
    args = parser.parse_args()
    train_cache_based(args)
