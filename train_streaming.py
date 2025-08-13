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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# ==============================================================================
# Helper functions from train_vkitti.py
# ==============================================================================

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

def least_square_whole_clip(infs, gts, data):
    
    # Remove channel dimension to get [B,T,H,W]
    if infs.dim() == 5 and infs.shape[2] == 1:
        infs = infs.squeeze(2)
    if gts.dim() == 5 and gts.shape[2] == 1:
        gts = gts.squeeze(2)

    # Set valid depth ranges based on dataset
    if data == "kitti":
        valid_mask = (gts > 1e-3) & (gts < 80.0)
    elif data == "gta":
        valid_mask = (gts > 1e-3) & (gts < 1000.0)
    elif data == "tartanair":
        valid_mask = (gts > 60.0) & (gts < 150.0)
    else:
        # Default range for other datasets
        valid_mask = (gts > 1e-3) & (gts < 10.0)
        
    gt_disp_masked = 1. / (gts[valid_mask].reshape((-1, 1)).double() + 1e-6)
    infs = infs.clamp(min=1e-3)
    pred_disp_masked = infs[valid_mask].reshape((-1,1)).double()
    
    _ones = torch.ones_like(pred_disp_masked)
    A = torch.cat([pred_disp_masked, _ones], dim=-1) 
    X = torch.linalg.lstsq(A, gt_disp_masked).solution  
    scale = X[0].item()
    shift = X[1].item()

    aligned_pred = scale * infs + shift
    aligned_pred = torch.clamp(aligned_pred, min=1e-3) 
    depth = torch.zeros_like(aligned_pred)
    depth = 1.0 / aligned_pred
    
    pred_depth = depth
    
    return pred_depth, valid_mask

def metric_val(infs, gts, data, poses=None, Ks=None):
    
    gt_depth = gts
    pred_depth, valid_mask = least_square_whole_clip(infs, gts, data)
    
    # Check validity - filter frames with valid pixels
    n = valid_mask.sum((-1, -2))  
    valid_frame = (n > 0)  
    pred_depth = pred_depth[valid_frame]
    gt_depth = gt_depth[valid_frame]
    valid_mask = valid_mask[valid_frame]
    
    absrel = abs_relative_difference(pred_depth, gt_depth, valid_mask)
    delta1 = delta1_acc(pred_depth, gt_depth, valid_mask)
    
    if poses is not None:
        tae = eval_tae(pred_depth, gt_depth, poses, Ks, valid_mask)
        return absrel,delta1,tae

    else :
        return absrel,delta1

def eval_tae(pred_depth, gt_depth, poses, Ks, masks):
    
    error_sum = 0.
    for i in range(len(pred_depth) - 1):
        depth1 = pred_depth[i]
        depth2 = pred_depth[i+1]
        
        mask1 = masks[i]
        mask2 = masks[i+1]

        T_1 = poses[i]
        T_2 = poses[i+1]

        try:
            T_2_inv = torch.linalg.inv(T_2)
        except torch._C._LinAlgError:
            # LU pivot 에러가 나면 pseudo-inverse 로 대체
            T_2_inv = torch.linalg.pinv(T_2)
        T_2_1 = T_2_inv @ T_1
   
        R_2_1 = T_2_1[:3,:3]
        t_2_1 = T_2_1[:3, 3]
        K = Ks[i]

        if K.dim() == 1 and K.numel() == 9:
            K = K.view(3, 3)

        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)
        try:
            T_1_2 = torch.linalg.inv(T_2_1)
        except torch._C._LinAlgError:
            # LU pivot 에러가 나면 pseudo-inverse 로 대체
            T_1_2 = torch.linalg.pinv(T_2_1)
            
        T_2_1 = T_2_inv @ T_1
        R_1_2 = T_1_2[:3,:3]
        t_1_2 = T_1_2[:3, 3]

        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)
        
        error_sum += error1
        error_sum += error2
    
    result = error_sum / (2 * (len(pred_depth) -1))
    return result

def get_mask(depth_m, min_depth, max_depth):
    valid_mask = (depth_m > min_depth) & (depth_m < max_depth)
    return valid_mask.bool()

def norm_ssi(depth, valid_mask):
    eps = 1e-6
    disparity = torch.zeros_like(depth)
    disparity[valid_mask] = 1.0 / depth[valid_mask]

    # Get shape before flattening with mask
    B, T, C, H, W = disparity.shape
    disp_flat = disparity.view(B, T, -1)         # [B, T, H*W]
    mask_flat = valid_mask.view(B, T, -1)        # [B, T, H*W]

    # Find min/max values excluding masked regions
    disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).min(dim=-1)[0]
    disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).max(dim=-1)[0]

    disp_min = disp_min.view(B, T, 1, 1, 1)
    disp_max = disp_max.view(B, T, 1, 1, 1)

    denom = (disp_max - disp_min + eps)
    norm_disp = (disparity - disp_min) / denom

    # Set invalid regions to 0
    norm_disp = norm_disp.masked_fill(~valid_mask, 0.0)

    return norm_disp

# ==============================================================================
# 스트리밍 스타일 학습을 위한 클래스
# ==============================================================================

class StreamingTrainer:
    def __init__(self, model, optimizer, scaler, loss_tgm, loss_ssi, ratio_tgm, ratio_ssi, device, update_frequency=4):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.loss_tgm = loss_tgm
        self.loss_ssi = loss_ssi
        self.ratio_tgm = ratio_tgm
        self.ratio_ssi = ratio_ssi
        self.device = device
        
        # 스트리밍 상태 관리
        self.frame_cache = None
        self.accumulated_loss = 0.0
        self.frame_count = 0
        self.update_frequency = update_frequency  # config에서 전달받은 값 사용
        
        # 손실 추적 (로깅용)
        self.last_ssi_loss = 0.0
        self.last_tgm_loss = 0.0
        
    def reset_stream(self):
        """새로운 비디오 시퀀스 시작 시 호출"""
        self.frame_cache = None
        self.accumulated_loss = 0.0
        self.frame_count = 0
        
        # 모델의 스트리밍 상태도 리셋
        if hasattr(self.model, 'module'):  # DataParallel
            self.model.module.frame_cache_list = []
            self.model.module.frame_id_list = []
            self.model.module.id = -1
            self.model.module.transform = None
        else:
            self.model.frame_cache_list = []
            self.model.frame_id_list = []
            self.model.id = -1
            self.model.transform = None
    
    def process_frame_streaming(self, frame, gt_depth, mask):
        """
        프레임을 하나씩 스트리밍 방식으로 처리 (Training 버전)
        """
        self.frame_count += 1
        
        # 단일 프레임을 배치 형태로 변환 [1, 1, C, H, W]
        frame_batch = frame.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
        
        # 메모리 효율성을 위해 gradient checkpointing과 함께 forward
        with autocast():
            # 스트리밍 상태 리셋 (각 프레임마다 새로운 시퀀스로 처리)
            if hasattr(self.model, 'module'):
                # DataParallel의 경우 임시로 스트리밍 상태 리셋
                original_cache = getattr(self.model.module, 'frame_cache_list', None)
                self.model.module.frame_cache_list = []
                self.model.module.frame_id_list = []
                self.model.module.id = -1
                
                # Gradient checkpointing 활성화 (메모리 절약)
                if hasattr(self.model.module, 'gradient_checkpointing'):
                    self.model.module.gradient_checkpointing = True
                
                pred_depth = self.model(frame_batch)  # [1, 1, H, W]
                
                # 원래 상태 복원 (실제로는 스트리밍 캐시를 시뮬레이션)
                if original_cache is not None:
                    self.model.module.frame_cache_list = original_cache
            else:
                # 단일 GPU의 경우
                original_cache = getattr(self.model, 'frame_cache_list', None)
                self.model.frame_cache_list = []
                self.model.frame_id_list = []
                self.model.id = -1
                
                pred_depth = self.model(frame_batch)  # [1, 1, H, W]
                
                if original_cache is not None:
                    self.model.frame_cache_list = original_cache
            
            # 차원 맞추기
            pred_depth = pred_depth.squeeze(0)  # [1, H, W] -> gt_depth와 같은 형태
            
            # Loss 계산 (메모리 효율적으로)
            valid_mask = mask.bool()
            if valid_mask.sum() > 0:
                # SSI loss (disparity space) - 메모리 사용량 최소화
                with torch.no_grad():
                    gt_disp = 1.0 / (gt_depth.clamp(min=1e-3) + 1e-6)
                    gt_min, gt_max = gt_disp[valid_mask].min(), gt_disp[valid_mask].max()
                
                pred_disp = 1.0 / (pred_depth.clamp(min=1e-3) + 1e-6)
                
                # Normalize disparity
                gt_norm = (gt_disp - gt_min) / (gt_max - gt_min + 1e-6)
                pred_norm = (pred_disp - gt_min) / (gt_max - gt_min + 1e-6)
                
                ssi_loss = torch.nn.functional.mse_loss(pred_norm[valid_mask], gt_norm[valid_mask])
                tgm_loss = torch.nn.functional.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
                
                frame_loss = self.ratio_ssi * ssi_loss + self.ratio_tgm * tgm_loss
                
                # 개별 손실 저장 (디버깅용)
                self.last_ssi_loss = ssi_loss.item()
                self.last_tgm_loss = tgm_loss.item()
            else:
                frame_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                self.last_ssi_loss = 0.0
                self.last_tgm_loss = 0.0
        
        # Loss 누적
        self.accumulated_loss += frame_loss
        
        # 일정 프레임마다 가중치 업데이트
        if self.frame_count % self.update_frequency == 0:
            avg_loss = self.accumulated_loss / self.update_frequency
            
            self.optimizer.zero_grad()
            self.scaler.scale(avg_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 누적 손실 리셋
            self.accumulated_loss = 0.0
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            return avg_loss.item()
        
        return None
    
    def process_sequence_streaming(self, video_sequence, depth_sequence, mask_sequence):
        """
        전체 비디오 시퀀스를 스트리밍 방식으로 처리 (멀티 GPU 지원)
        """
        self.reset_stream()
        losses = []
        
        B, T = video_sequence.shape[:2]  # Batch and Temporal dimensions
        
        # 배치 크기가 1보다 큰 경우, 각 배치 아이템을 개별적으로 처리
        for b in range(B):
            for t in range(T):
                frame = video_sequence[b, t]  # [C, H, W]
                gt_depth = depth_sequence[b, t]  # [C, H, W]
                mask = mask_sequence[b, t]  # [C, H, W]
                
                loss = self.process_frame_streaming(frame, gt_depth, mask)
                if loss is not None:
                    losses.append(loss)
        
        # 남은 누적 손실 처리
        if self.accumulated_loss > 0:
            remaining_frames = self.frame_count % self.update_frequency
            if remaining_frames > 0:
                avg_loss = self.accumulated_loss / remaining_frames
                
                self.optimizer.zero_grad()
                self.scaler.scale(avg_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                losses.append(avg_loss.item())
        
        return losses

# ==============================================================================
# 2. 수정된 학습 함수
# ==============================================================================

def train_streaming(args):
    # 기본 설정 (기존과 동일)
    EXPERIMENT_ID = 5
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
    
    load_dotenv(".env")
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)["hyper_parameter"]

    run = wandb.init(project="stream_causal_block", entity="Depth-Finder", config=cfg,
                     name=f"streaming_e_{cfg['epochs']}_update_freq_{cfg['update_frequency']}", # vitl or vits도 넣으면 좋을듯?
                     tags=["streaming", "causal_mask"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log device information at the start
    logger.info("🔍 System Information:")
    logger.info(f"   • PyTorch version: {torch.__version__}")
    logger.info(f"   • CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"   • CUDA version: {torch.version.cuda}")
        logger.info(f"   • Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Move MEAN and STD to device
    global MEAN, STD
    MEAN = MEAN.to(device)
    STD = STD.to(device)
    
    # 모델 저장 관련 변수
    best_train_loss = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, "best_streaming_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_streaming_model.pth")
    
    # 데이터셋 로딩 - 멀티 GPU를 위해 배치 크기 조정
    CLIP_LEN = cfg["clip_len"]
    # GPU 개수에 따라 배치 크기 조정 (config.yaml의 batch_size를 기본값으로 사용)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    config_batch_size = cfg.get("batch_size", 1)  # config.yaml에서 batch_size 가져오기
    
    # 멀티 GPU 환경에서는 config의 batch_size를 그대로 사용하고, 단일 GPU에서는 조정
    if num_gpus > 1:
        batch_size = config_batch_size
    else:
        # 단일 GPU에서는 config 값과 1 중 더 작은 값 사용
        batch_size = min(config_batch_size, max(1, num_gpus))
    
    logger.info(f"📊 Training Configuration:")
    logger.info(f"   • Number of GPUs: {num_gpus}")
    logger.info(f"   • Config batch size: {config_batch_size}")
    logger.info(f"   • Actual batch size: {batch_size}")
    logger.info(f"   • Clip length: {CLIP_LEN}")
    logger.info(f"   • Memory optimization: Enabled (pin_memory=False, reduced workers)")
    logger.info(f"   • Update frequency: {cfg.get('update_frequency', 4)} frames")

    kitti_root = "/workspace/Video-Depth-Anything/datasets/KITTI"
    rgb_train, dep_train = get_data_list(kitti_root, "kitti", "train", CLIP_LEN)

    kitti_train_ds = KITTIVideoDataset(rgb_train, dep_train, resize_size=350, split="train")
    # 메모리 효율적인 DataLoader 설정
    num_workers = min(4, max(1, num_gpus * 2))  # GPU당 2개의 워커로 감소
    kitti_train_loader = DataLoader(kitti_train_ds, batch_size=batch_size, shuffle=True, 
                                   num_workers=num_workers, pin_memory=False,  # pin_memory 비활성화
                                   persistent_workers=True if num_workers > 0 else False,
                                   prefetch_factor=2)  # 메모리 사용량 감소

    # Add validation datasets from train_vkitti.py
    val_rgb_clips, val_depth_clips, val_cam_ids, val_intrin_clips, val_extrin_clips = get_data_list(
        root_dir=kitti_root,
        data_name="kitti",
        split="val",
        clip_len=CLIP_LEN
    )

    kitti_val = KITTIVideoDataset(
        rgb_paths=val_rgb_clips,
        depth_paths=val_depth_clips,
        cam_ids=val_cam_ids,
        intrin_clips=val_intrin_clips,
        extrin_clips=val_extrin_clips,
        resize_size=350,
        split="val"
    )
    kitti_val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_workers, pin_memory=False,
                                 persistent_workers=True if num_workers > 0 else False,
                                 prefetch_factor=2)

    # ScanNet validation dataset
    x_scannet, y_scannet, scannet_poses, scannet_Ks = get_list("", "scannet")
    scannet_data = ValDataset(
        img_paths   = x_scannet,
        depth_paths = y_scannet,
        data_name   = "scannet",
        Ks          = scannet_Ks,
        pose_paths  = scannet_poses
    )
    scannet_val_loader = DataLoader(scannet_data, batch_size=batch_size, shuffle=False, 
                                   num_workers=num_workers, pin_memory=False,
                                   persistent_workers=True if num_workers > 0 else False,
                                   prefetch_factor=2)
    
    logger.info(f"📥 DataLoader Configuration:")
    logger.info(f"   • num_workers: {num_workers}")
    logger.info(f"   • pin_memory: False (메모리 효율성)")
    logger.info(f"   • persistent_workers: {True if num_workers > 0 else False}")
    logger.info(f"   • prefetch_factor: 2")

    # 모델 설정
    logger = logging.getLogger(__name__)
    logger.info("Creating VideoDepthAnything model (streaming mode)...")
    model = VideoDepthAnything(num_frames=CLIP_LEN,
                               use_causal_mask=True,
                               encoder='vits',
                               features=64,
                               out_channels=[48, 96, 192, 384]).to(device)

    # 체크포인트 로딩
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

    # 멀티 GPU 설정 (개선된 로깅과 강제 활성화)
    if torch.cuda.device_count() > 1:
        logger.info(f"🚀 Enabling multi-GPU training with {torch.cuda.device_count()} GPUs")
        logger.info(f"   • Primary device: {device}")
        logger.info(f"   • Available GPUs: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # DataParallel 설정 - 명시적으로 모든 GPU 사용
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info(f"   • Model wrapped with DataParallel using devices: {device_ids}")
        
        # 각 GPU에 메모리 할당 강제 (작은 텐서로 워밍업)
        for i in range(torch.cuda.device_count()):
            dummy_tensor = torch.randn(1, 1, device=f'cuda:{i}')
            del dummy_tensor
            torch.cuda.empty_cache()
        
        # GPU 메모리 확인
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"   • GPU {i} ({torch.cuda.get_device_name(i)}): {mem_allocated:.2f}GB/{mem_total:.2f}GB allocated")
    else:
        logger.info(f"📱 Single GPU training on {device}")
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   • GPU 0 Memory: {mem_allocated:.2f}GB/{mem_total:.2f}GB allocated")

    # 옵티마이저 및 스케줄러
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    # 손실 함수
    scaler = GradScaler()
    loss_tgm = LossTGMVector(diff_depth_th=0.05)
    loss_ssi = Loss_ssi_basic()
    ratio_tgm = cfg["ratio_tgm"]
    ratio_ssi = cfg["ratio_ssi"]

    # 스트리밍 트레이너 초기화
    update_frequency = cfg.get("update_frequency", 4)  # config에서 update_frequency 가져오기
    streaming_trainer = StreamingTrainer(
        model, optimizer, scaler, loss_tgm, loss_ssi, ratio_tgm, ratio_ssi, device, update_frequency
    )
    
    logger.info(f"🎯 Streaming Configuration:")
    logger.info(f"   • Update frequency: {update_frequency} frames")

    # wandb에 모델 정보 로깅
    model_info = get_model_info(model)
    wandb.log({
        "model/total_params": model_info['total_params'],
        "model/trainable_params": model_info['trainable_params'],
        "model/trainable_percentage": model_info['trainable_percentage'],
        "model/encoder": 'vits',
        "model/features": 64,
        "streaming/update_frequency": streaming_trainer.update_frequency,
        "streaming/batch_size": batch_size
    })

    # 학습 루프
    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_losses = []
        epoch_frame_counts = []
        
        # 현재 학습률 로깅
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr, "epoch": epoch})
        
        # Training phase
        for batch_idx, (x, y) in enumerate(tqdm(kitti_train_loader, desc=f"Streaming-Train-E{epoch}")):
            # 메모리 정리 (매 배치마다)
            if batch_idx % 5 == 0:  # 5 배치마다 캐시 정리
                torch.cuda.empty_cache()
            
            # 마스크 생성
            masks = ((y > 0.001) & (y < 80.0)).bool().to(device)
            x, y = x.to(device), y.to(device)
            
            # 배치 정보 로깅
            valid_pixels = masks.sum().item()
            total_pixels = masks.numel()
            valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # GPU 메모리 사용량 체크 (첫 배치와 메모리 부족 위험시)
            if batch_idx == 0 or batch_idx % 20 == 0:
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    utilization = mem_allocated / mem_total * 100
                    logger.info(f"Batch {batch_idx} - GPU {i}: {mem_allocated:.2f}GB/{mem_total:.2f}GB ({utilization:.1f}% used)")
                    
                    # 메모리 사용량이 80%를 넘으면 경고
                    if utilization > 80:
                        logger.warning(f"⚠️  GPU {i} memory usage is high ({utilization:.1f}%)")
                        torch.cuda.empty_cache()  # 즉시 캐시 정리
            
            # 스트리밍 방식으로 시퀀스 처리
            sequence_losses = streaming_trainer.process_sequence_streaming(x, y, masks)
            epoch_losses.extend(sequence_losses)
            
            # 배치별 통계 수집
            if sequence_losses:
                batch_avg_loss = np.mean(sequence_losses)
                epoch_frame_counts.append(len(sequence_losses))
                
                # 배치별 상세 로깅 (매 10 배치마다)
                if batch_idx % 10 == 0:
                    logger.info("Epoch %d, Batch %d, Avg Loss: %.4f, Frames: %d, Valid Ratio: %.3f", 
                              epoch, batch_idx, batch_avg_loss, len(sequence_losses), valid_ratio)
                    
                    wandb.log({
                        "batch_loss": batch_avg_loss,  # train.py와 유사한 패턴
                        "streaming/frames_processed": len(sequence_losses),
                        "streaming/valid_pixel_ratio": valid_ratio,
                        "streaming/batch_idx": batch_idx,
                        "epoch": epoch
                    })
            
            # 실시간 손실 로깅 (매 20 배치마다)
            if batch_idx % 20 == 0 and epoch_losses:
                running_avg_loss = np.mean(epoch_losses[-100:])  # 최근 100개 손실의 평균
                wandb.log({
                    "running_loss": running_avg_loss,  # train.py와 비교 가능한 이름
                    "streaming/total_frames": len(epoch_losses),
                    "epoch": epoch
                })

        # Training loss 계산
        avg_kitti_train_loss = np.mean(epoch_losses) if epoch_losses else float('nan')

        # === VKITTI Validation Loop ===
        model.eval()
        val_loss = 0.0
        total_absrel = 0.0
        total_delta1 = 0.0
        total_tae = 0.0
        cnt_clip = 0
        wb_images = []

        with torch.no_grad():
            for batch_idx, (x, y, extrinsics, intrinsics) in tqdm(enumerate(kitti_val_loader)):
                x, y = x.to(device), y.to(device)
                extrinsics, intrinsics = extrinsics.to(device), intrinsics.to(device)

                pred = model(x)  # [B, T, H, W]
                masks = get_mask(y, min_depth=0.001, max_depth=80.0)  # [B, T, 1, H, W]
                masks = masks.to(device).bool()
                disp_normed = norm_ssi(y, masks)
                ssi_loss_val = loss_ssi(pred, disp_normed, masks.squeeze(2))
                tgm_loss_val = loss_tgm(pred, y, masks)
                val_loss += ratio_ssi * ssi_loss_val + ratio_tgm * tgm_loss_val

                # Scale & shift alignment
                B, T, H, W = pred.shape
                MIN_DISP = 1.0 / 80.0
                MAX_DISP = 1.0 / 0.001

                raw_disp = pred.clamp(min=1e-6)  # [B, T, H, W]
                gt_disp = (1.0 / y.clamp(min=1e-6)).squeeze(2)  # [B, T, H, W]
                m_flat = masks.squeeze(2).view(B, -1).float()  # [B, P]
                p_flat = raw_disp.view(B, -1)  # [B, P]
                g_flat = gt_disp.view(B, -1)  # [B, P]

                A = torch.stack([p_flat, torch.ones_like(p_flat, device=device)], dim=-1)  # [B,P,2]
                A = A * m_flat.unsqueeze(-1)
                b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)

                X = torch.linalg.lstsq(A, b_vec).solution  # [B,2,1]
                a = X[:, 0, 0].view(B, 1, 1, 1)  # [B,1,1,1]
                b = X[:, 1, 0].view(B, 1, 1, 1)  # [B,1,1,1]

                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)  # [B,T,H,W]

                # Save frames for first batch only (matching train_vkitti.py)
                if batch_idx == 0:
                    save_dir = f"outputs/experiment_{EXPERIMENT_ID}/vkitti/epoch_{epoch}_batch_{batch_idx}"
                    os.makedirs(save_dir, exist_ok=True)
                    for t in range(T):
                        # a) RGB
                        rgb_norm = x[0, t]  # [3,H,W]
                        rgb_unc = (rgb_norm * STD + MEAN).clamp(0, 1)
                        rgb_np = (rgb_unc.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))

                        # b) GT Disparity
                        depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)  # [H,W]
                        disp_frame = 1.0 / depth_frame  # [H,W]
                        valid = masks[0, t].squeeze(0)  # [H,W] bool

                        d_vals = disp_frame[valid]
                        d_min, d_max = d_vals.min(), d_vals.max()

                        norm_gt = (disp_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_gt = norm_gt.clamp(0, 1)

                        gt_uint8 = (norm_gt.cpu().numpy() * 255).astype(np.uint8)
                        gt_rgb = np.stack([gt_uint8] * 3, axis=-1)
                        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"gt_{t:02d}.png"))

                        # c) Mask
                        mask_frame = masks[0, t].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(mask_frame).save(os.path.join(save_dir, f"mask_{t:02d}.png"))

                        # d) Predicted Disparity
                        pred_frame = aligned_disp[0, t]  # [H,W]
                        norm_pd = (pred_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_pd = norm_pd.clamp(0, 1)

                        pd_uint8 = (norm_pd.cpu().numpy() * 255).astype(np.uint8)
                        pd_rgb = np.stack([pd_uint8] * 3, axis=-1)
                        Image.fromarray(pd_rgb).save(os.path.join(save_dir, f"pred_{t:02d}.png"))

                        # e) wandb image
                        wb_images.append(wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"), 
                                                   caption=f"pred_epoch{epoch}_frame{t:02d}"))

                    logger.info(f"→ saved validation frames to '{save_dir}'")

                # Metric evaluation
                for b in range(B):
                    inf_clip = pred[b]  # [T,H,W]
                    gt_clip = y[b].squeeze(1)  # [T,H,W]
                    pose = extrinsics[b]
                    Kmat = intrinsics[b]
                    absr, d1, tae = metric_val(inf_clip, gt_clip, "kitti", pose, Kmat)
                    total_absrel += absr
                    total_delta1 += d1
                    total_tae += tae
                    cnt_clip += 1

        # Final VKITTI validation statistics
        avg_val_loss = val_loss / len(kitti_val_loader)
        avg_absrel = total_absrel / cnt_clip
        avg_delta1 = total_delta1 / cnt_clip
        avg_tae = total_tae / cnt_clip

        logger.info("Epoch [%d/%d] VKITTI Validation Loss: %.4f", epoch, cfg['epochs'], avg_val_loss)
        logger.info("VKITTI AbsRel: %.4f", avg_absrel)
        logger.info("VKITTI Delta1: %.4f", avg_delta1)
        logger.info("VKITTI TAE: %.4f", avg_tae)

        # Log VKITTI results (matching train_vkitti.py format)
        wandb.log({
            "vkitti_train_loss": avg_kitti_train_loss,
            "vkitti_val_loss": avg_val_loss,
            "vkitti_absrel": avg_absrel,
            "vkitti_delta1": avg_delta1,
            "vkitti_tae": avg_tae,
            "vkitti_epoch": epoch,
            "vkitti_pred_disparity": wb_images,
        })
        del wb_images

        # === ScanNet Validation Loop ===
        val_loss = 0.0
        total_absrel = 0.0
        total_delta1 = 0.0
        total_tae = 0.0
        cnt_clip = 0
        wb_images = []

        with torch.no_grad():
            for batch_idx, (x, y, extrinsics, intrinsics) in tqdm(enumerate(scannet_val_loader)):
                x, y = x.to(device), y.to(device)
                extrinsics, intrinsics = extrinsics.to(device), intrinsics.to(device)

                pred = model(x)  # [B, T, H, W]
                masks = get_mask(y, min_depth=0.001, max_depth=80.0)  # [B, T, 1, H, W]
                masks = masks.to(device).bool()
                disp_normed = norm_ssi(y, masks)
                ssi_loss_val = loss_ssi(pred, disp_normed, masks.squeeze(2))
                tgm_loss_val = loss_tgm(pred, y, masks)
                val_loss += ratio_ssi * ssi_loss_val + ratio_tgm * tgm_loss_val

                # Scale & shift alignment (same as VKITTI)
                B, T, H, W = pred.shape
                MIN_DISP = 1.0 / 80.0
                MAX_DISP = 1.0 / 0.001

                raw_disp = pred.clamp(min=1e-6)  # [B, T, H, W]
                gt_disp = (1.0 / y.clamp(min=1e-6)).squeeze(2)  # [B, T, H, W]
                m_flat = masks.squeeze(2).view(B, -1).float()  # [B, P]
                p_flat = raw_disp.view(B, -1)  # [B, P]
                g_flat = gt_disp.view(B, -1)  # [B, P]

                A = torch.stack([p_flat, torch.ones_like(p_flat, device=device)], dim=-1)  # [B,P,2]
                A = A * m_flat.unsqueeze(-1)
                b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)

                X = torch.linalg.lstsq(A, b_vec).solution  # [B,2,1]
                a = X[:, 0, 0].view(B, 1, 1, 1)  # [B,1,1,1]
                b = X[:, 1, 0].view(B, 1, 1, 1)  # [B,1,1,1]

                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)  # [B,T,H,W]

                # Save frames for first batch only
                if batch_idx == 0:
                    save_dir = f"outputs/experiment_{EXPERIMENT_ID}/scannet/epoch_{epoch}_batch_{batch_idx}"
                    os.makedirs(save_dir, exist_ok=True)
                    for t in range(T):
                        # a) RGB
                        rgb_norm = x[0, t]  # [3,H,W]
                        rgb_unc = (rgb_norm * STD + MEAN).clamp(0, 1)
                        rgb_np = (rgb_unc.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))

                        # b) GT Disparity
                        depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)  # [H,W]
                        disp_frame = 1.0 / depth_frame  # [H,W]
                        valid = masks[0, t].squeeze(0)  # [H,W] bool

                        d_vals = disp_frame[valid]
                        d_min, d_max = d_vals.min(), d_vals.max()

                        norm_gt = (disp_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_gt = norm_gt.clamp(0, 1)

                        gt_uint8 = (norm_gt.cpu().numpy() * 255).astype(np.uint8)
                        gt_rgb = np.stack([gt_uint8] * 3, axis=-1)
                        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"gt_{t:02d}.png"))

                        # c) Mask
                        mask_frame = masks[0, t].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(mask_frame).save(os.path.join(save_dir, f"mask_{t:02d}.png"))

                        # d) Predicted Disparity
                        pred_frame = aligned_disp[0, t]  # [H,W]
                        norm_pd = (pred_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_pd = norm_pd.clamp(0, 1)

                        pd_uint8 = (norm_pd.cpu().numpy() * 255).astype(np.uint8)
                        pd_rgb = np.stack([pd_uint8] * 3, axis=-1)
                        Image.fromarray(pd_rgb).save(os.path.join(save_dir, f"pred_{t:02d}.png"))

                        # e) wandb image
                        wb_images.append(wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"), 
                                                   caption=f"pred_epoch{epoch}_frame{t:02d}"))

                    logger.info(f"→ saved validation frames to '{save_dir}'")

                # Metric evaluation
                for b in range(B):
                    inf_clip = pred[b]  # [T,H,W]
                    gt_clip = y[b].squeeze(1)  # [T,H,W]
                    pose = extrinsics[b]
                    Kmat = intrinsics[b]
                    absr, d1, tae = metric_val(inf_clip, gt_clip, "scannet", pose, Kmat)
                    total_absrel += absr
                    total_delta1 += d1
                    total_tae += tae
                    cnt_clip += 1

        # Final ScanNet validation statistics
        avg_val_loss = val_loss / len(scannet_val_loader)
        avg_absrel = total_absrel / cnt_clip
        avg_delta1 = total_delta1 / cnt_clip
        avg_tae = total_tae / cnt_clip

        logger.info("Epoch [%d/%d] ScanNet Validation Loss: %.4f", epoch, cfg['epochs'], avg_val_loss)
        logger.info("ScanNet AbsRel: %.4f", avg_absrel)
        logger.info("ScanNet Delta1: %.4f", avg_delta1)
        logger.info("ScanNet TAE: %.4f", avg_tae)

        # Log ScanNet results (matching train_vkitti.py format)
        wandb.log({
            "scannet_val_loss": avg_val_loss,
            "scannet_absrel": avg_absrel,
            "scannet_delta1": avg_delta1,
            "scannet_tae": avg_tae,
            "scannet_pred_disparity": wb_images,
        })
        del wb_images

        # Epoch completed - calculate total validation loss and save models
        total_val_loss = avg_val_loss  # from ScanNet validation
        
        # Model saving logic (same as train_vkitti.py but based on validation loss)
        if total_val_loss < best_train_loss:  # Using val loss as criterion
            best_train_loss = total_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_train_loss,
                'config': cfg
            }, best_model_path)
            logger.info("Best streaming model saved to %s with val_loss: %.4f", best_model_path, best_train_loss)
            
            wandb.log({
                "best_val_loss": best_train_loss,
                "streaming/best_epoch": epoch,
                "streaming/model_improved": True,
                "epoch": epoch
            })
        else:
            wandb.log({
                "streaming/model_improved": False,
                "epoch": epoch
            })
        
        # Latest model 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'current_val_loss': total_val_loss,
            'config': cfg
        }, latest_model_path)
        logger.info("Latest streaming model saved to %s", latest_model_path)

        scheduler.step()

    # 학습 완료 후 최종 통계
    final_stats = {
        "training/total_epochs": cfg["epochs"],
        "training/final_best_loss": best_train_loss,
        "training/model_saved": True,
        "training/completed": True
    }
    wandb.log(final_stats)
    
    logger.info("="*50)
    logger.info("Streaming Training Completed!")
    logger.info("Total Epochs: %d", cfg['epochs'])
    logger.info("Best Validation Loss: %.4f", best_train_loss)
    logger.info("Models saved to: %s", OUTPUT_DIR)
    logger.info("="*50)

    run.finish()

# ==============================================================================
# 3. Entry point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")
    args = parser.parse_args()
    train_streaming(args)
