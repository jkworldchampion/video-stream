import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
import wandb
import gc
import argparse
from dotenv import load_dotenv

from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils.loss_MiDas import *
from data.dataLoader import *
from video_depth_anything.video_depth_stream import VideoDepthAnything
from data.val_dataLoader import *

from benchmark.eval.metric import *
from benchmark.eval.eval_tae import tae_torch
from PIL import Image

import logging

# 실험할 때마다 바꿔서 실험
experiment = 1

os.makedirs("logs", exist_ok=True)

# 2. configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),                      # console
        logging.FileHandler("logs/train_log_experiment_1.txt"),    # file
    ],
)

logger = logging.getLogger(__name__)

matplotlib.use('Agg')

# Auto-detect the best available device for multi-GPU training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
MEAN = torch.tensor((0.485, 0.456, 0.406), device=device).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=device).view(3, 1, 1)

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

def rotmat_to_quaternion(R: np.ndarray):
    """
    3×3 회전 행렬 R을 쿼터니언 [qw, qx, qy, qz] 형태로 변환합니다.
    R은 np.float32 혹은 np.float64 타입의 3×3 배열이어야 합니다.
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4·qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    else:
        # trace가 음수일 때, 가장 큰 대각 원소에 따라 분기
        if (m00 > m11) and (m00 > m22):
            S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S = 4·qx
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
        elif m11 > m22:
            S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S = 4·qy
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
        else:
            S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S = 4·qz
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S

    return qw, qx, qy, qz

def train(args):
    
    # Log device information at the start
    logger.info("🔍 System Information:")
    logger.info(f"   • PyTorch version: {torch.__version__}")
    logger.info(f"   • CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"   • CUDA version: {torch.version.cuda}")
        logger.info(f"   • Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")

    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("WANDB_API_KEY")
    print("W&B key:", api_key)
    wandb.login(key=api_key, relogin=True)

    ### 1. Handling hyper_params with WAND :)
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    hyper_params = config["hyper_parameter"]

    lr = hyper_params["learning_rate"]
    ratio_ssi = hyper_params["ratio_ssi"]
    ratio_tgm = hyper_params["ratio_tgm"]
    ratio_ssi_image = hyper_params["ratio_ssi_image"]
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]
    batch_size = hyper_params["batch_size"]
    CLIP_LEN = hyper_params["clip_len"]
    
    run = wandb.init(project="stream_causal_block", entity="Depth-Finder", config=hyper_params)
    
    ### 2. Load data

    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
    google_path="/workspace/Video-Depth-Anything/datasets/google_landmarks"

    rgb_clips, depth_clips = get_data_list(
        root_dir=kitti_path,
        data_name="kitti",
        split="train",
        clip_len=CLIP_LEN
    )

    kitti_train = KITTIVideoDataset(
        rgb_paths=rgb_clips,
        depth_paths=depth_clips,
        resize_size=350,
        split="train"
    )

    val_rgb_clips, val_depth_clips, val_cam_ids, val_intrin_clips, val_extrin_clips = get_data_list(
        root_dir=kitti_path,
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

    kitti_train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=6)
    kitti_val_loader   = DataLoader(kitti_val,   batch_size=batch_size, shuffle=False, num_workers=6)
    
    google_img_paths, google_depth_paths = get_data_list(
        root_dir=google_path,
        data_name="google",
        split="train"
    )

    google_train = GoogleDepthDataset(
        img_paths=google_img_paths,
        depth_paths=google_depth_paths,
        resize_size=518
    )

    gta_path = "/workspace/Video-Depth-Anything/datasets/GTAV_720/GTAV_720"
    
    gta_rgb_clips, gta_depth_clips = get_data_list(
        root_dir=gta_path,
        data_name="GTA",
        split="train",
        clip_len=16
    )
    
    gta_train = GTADataset(gta_rgb_clips, gta_depth_clips)

    val_gta_rgb_clips, val_gta_depth_clips, val_poses= get_data_list(
        root_dir=gta_path,
        data_name="GTA",
        split="val",
        clip_len=16
    )
    
    gta_val = GTADataset(val_gta_rgb_clips, val_gta_depth_clips, val_poses, split="val")

    tartanair_root = "/workspace/Video-Depth-Anything/datasets/Tartan_air"
    tartanair_envs = ["UrbanConstruction"]

    ta_train_imgs, ta_train_deps = get_data_list(
        root_dir=tartanair_root,
        data_name="tartanair",
        split=tartanair_envs,
        clip_len=16,
        difficulties=["hard"]
    )
    ta_train_dataset = TartanAirVideoDataset(
        img_lists=ta_train_imgs,
        dep_lists=ta_train_deps,
        clip_len=16,
        resize_size=518,
        split="train"
    )
    ta_train_loader = DataLoader(ta_train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    ta_val_imgs, ta_val_deps, ta_val_poses = get_data_list(
        root_dir=tartanair_root,
        data_name="tartanair",
        split=tartanair_envs,
        clip_len=16,
        difficulties=["easy"]
    )
    ta_val_dataset = TartanAirVideoDataset(
        img_lists=ta_val_imgs,
        dep_lists=ta_val_deps,
        posefile_lists=ta_val_poses,
        clip_len=16,
        resize_size=518,
        split="val"
    )
    
    ta_val_loader = DataLoader(ta_val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    gta_train_loader =  DataLoader(gta_train, batch_size=batch_size, shuffle=True, num_workers=6)
    gta_val_loader =  DataLoader(gta_val, batch_size=batch_size, shuffle=False, num_workers=6)

    x_nyu, y_nyu = get_list("", "nyu")
    nyu_data = ValDataset(x_nyu, y_nyu, "nyu")
    nyu_val_loader = DataLoader(nyu_data, batch_size=batch_size, shuffle=False, num_workers=6)
    
    x_kitti, y_kitti = get_list("", "kitti")
    eval_kitti_data  = ValDataset(x_kitti,  y_kitti,  "kitti")    
    eval_kitti_val_loader = DataLoader(eval_kitti_data, batch_size=batch_size, shuffle=False, num_workers=6)

    x_scannet, y_scannet, scannet_poses, scannet_Ks = get_list("", "scannet")
    scannet_data = ValDataset(
        img_paths   = x_scannet,
        depth_paths = y_scannet,
        data_name   = "scannet",
        Ks          = scannet_Ks,
        pose_paths  = scannet_poses
    )
    scannet_val_loader = DataLoader(scannet_data, batch_size=batch_size, shuffle=False, num_workers=6)
    
    
    ### 3. Model and additional stuffs,...
    
    # Create model with streaming optimizations (causal masking enabled)
    logger.info("🏗️ Creating VideoDepthAnything model with streaming configuration...")
    model = VideoDepthAnything(
        num_frames=CLIP_LEN,
        use_causal_mask=True,  # Enable causal masking for improved streaming consistency
        encoder='vits',        # Using vits encoder (adjust if needed)
        features=64,
        out_channels=[48, 96, 192, 384]
    ).to(device)
    
    logger.info(f"✅ Model created with causal masking enabled for streaming")

    # Load pretrained weights
    if args.pretrained_ckpt:
        logger.info(f"📂 Loading pretrained weights from {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
            
        # Filter compatible weights
        model_dict = model.state_dict()
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }

        # Load filtered weights
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=True)

        # Log skipped parameters
        skipped = set(state_dict.keys()) - set(filtered_dict.keys())
        if skipped:
            logger.warning(f"⚠️ Skipped loading {len(skipped)} parameters (shape mismatch):")
            for skip in list(skipped)[:5]:
                logger.warning(f"   - {skip}")
            if len(skipped) > 5:
                logger.warning(f"   ... and {len(skipped)-5} more")
        
        logger.info("✅ Pretrained weights loaded successfully")

    # Configure training strategy: freeze encoder, train decoder only
    logger.info("🔒 Configuring training strategy: Encoder frozen, Decoder trainable")
    
    # Freeze encoder (DINOv2)
    for param in model.pretrained.parameters():
        param.requires_grad = False
    
    # Ensure decoder (head) is trainable
    for param in model.head.parameters():
        param.requires_grad = True

    model.train()
    
    # Log training configuration
    trainable_params = 0
    frozen_params = 0
    logger.info("📋 Training parameter status:")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            if 'head' in name:  # Only log head parameters to avoid spam
                logger.info(f"  ✅ {name}: trainable ({param.numel():,} params)")
        else:
            frozen_params += param.numel()
    
    logger.info(f"📊 Training Summary:")
    logger.info(f"   • Trainable parameters: {trainable_params:,}")
    logger.info(f"   • Frozen parameters: {frozen_params:,}")
    logger.info(f"   • Training ratio: {trainable_params/(trainable_params+frozen_params)*100:.1f}%")

    
    # Multi-GPU training setup
    if torch.cuda.device_count() > 1:
        logger.info(f"🚀 Enabling multi-GPU training with {torch.cuda.device_count()} GPUs")
        logger.info(f"   • Primary device: {device}")
        logger.info(f"   • Available GPUs: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}")
        model = torch.nn.DataParallel(model)
    else:
        logger.info(f"📱 Single GPU training on {device}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")

    loss_tgm = LossTGMVector(diff_depth_th=0.05)
    loss_ssi = Loss_ssi_basic()

    wandb.watch(model, log="all")

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    scaler = GradScaler()

    torch.backends.cuda.preferred_linalg_library('cusolver')

    ### 4. train
    
    start_epoch = 0
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False):
        
        print()
        epoch_loss = 0.0
        total_val_loss = 0.0

        model.train()
        
        # Train on VKITTI dataset      
        for batch_idx, (x, y) in tqdm(enumerate(kitti_train_loader)):
            
            # Generate masks
            video_masks = get_mask(y, min_depth=0.001, max_depth=80.0)
            x, y = x.to(device), y.to(device)
            video_masks = video_masks.to(device)

            optimizer.zero_grad()
            with autocast():
                pred = model(x)
                logger.info(f"Train pred mean: {pred.mean().item():.6f}")
  
                # Clip-level SSI normalization
                disp_normed = norm_ssi(y, video_masks)
                video_masks_squeezed = video_masks.squeeze(2)
                loss_ssi_value = loss_ssi(pred, disp_normed, video_masks_squeezed)
                loss_tgm_value = loss_tgm(pred, y, video_masks_squeezed)

                loss = ratio_tgm * loss_tgm_value + ratio_ssi * loss_ssi_value
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                logger.info(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
            
        avg_kitti_train_loss = epoch_loss / len(kitti_train_loader)
        
        # === validation loop ===
        model.eval()
        val_loss = 0.0
        total_absrel = 0.0
        total_delta1 = 0.0
        total_tae = 0.0
        cnt_clip = 0
        wb_images = []

        with torch.no_grad():
            for batch_idx, (x, y, extrinsics, intrinsics) in tqdm(enumerate(kitti_val_loader)):
                # 1) move to device
                x, y = x.to(device), y.to(device)
                extrinsics, intrinsics = extrinsics.to(device), intrinsics.to(device)

                # 2) model inference + basic losses
                pred = model(x)                                        # [B, T, H, W]
                masks = get_mask(y, min_depth=0.001, max_depth=80.0)   # [B, T, 1, H, W]
                masks = masks.to(device).bool()
                disp_normed   = norm_ssi(y, masks)
                ssi_loss_val  = loss_ssi(pred, disp_normed, masks.squeeze(2))
                tgm_loss_val  = loss_tgm(pred, y, masks)
                val_loss     += ratio_ssi * ssi_loss_val + ratio_tgm * tgm_loss_val

                logger.info(f"pred.mean(): {pred.mean().item():.6f}")

                # 3) prepare for scale & shift
                B, T, H, W = pred.shape

                MIN_DISP = 1.0 / 80.0      # depth=80m → disp=0.0125
                MAX_DISP = 1.0 / 0.001     # depth=0.001m → disp=1000.0

                raw_disp = pred.clamp(min=1e-6)                # [B, T, H, W]
                gt_disp  = (1.0 / y.clamp(min=1e-6)).squeeze(2) # [B, T, H, W]
                m_flat   = masks.squeeze(2).view(B, -1).float()# [B, P]
                p_flat   = raw_disp.view(B, -1)               # [B, P]
                g_flat   = gt_disp .view(B, -1)               # [B, P]

            
                # 4) build A, b for least-squares: A @ [a; b] ≈ b_vec 
                A = torch.stack([p_flat, torch.ones_like(p_flat, device=device)], dim=-1)  # [B,P,2]
                A = A * m_flat.unsqueeze(-1)    
                b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)  # mask out invalid
                
                 # 5) batched least-squares
                X = torch.linalg.lstsq(A, b_vec).solution  # [B,2,1]
                a = X[:,0,0].view(B,1,1,1)                 # [B,1,1,1]
                b = X[:,1,0].view(B,1,1,1)                 # [B,1,1,1]

                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)  # [B,T,H,W]
                
                # Save frames for first batch only
                if batch_idx == 0:
                    save_dir = f"outputs/experiment_{experiment}/vkitti/epoch_{epoch}_batch_{batch_idx}"
                    os.makedirs(save_dir, exist_ok=True)
                    for t in range(T):

                        # a) RGB
                        rgb_norm = x[0, t]  # [3,H,W]
                        rgb_unc  = (rgb_norm * STD + MEAN).clamp(0,1)
                        rgb_np   = (rgb_unc.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))

                        # b) GT Disparity 저장 (Min–Max 정규화)
                        depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)       # [H,W]
                        disp_frame  = 1.0 / depth_frame                       # [H,W]
                        valid       = masks[0, t].squeeze(0)                  # [H,W] bool

                        # 유효 픽셀만 뽑아 min/max
                        d_vals = disp_frame[valid]
                        d_min, d_max = d_vals.min(), d_vals.max()

                        norm_gt = (disp_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_gt = norm_gt.clamp(0,1)

                        gt_uint8 = (norm_gt.cpu().numpy() * 255).astype(np.uint8)
                        gt_rgb   = np.stack([gt_uint8]*3, axis=-1)
                        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"gt_{t:02d}.png"))

                        # c) Mask 저장
                        mask_frame = masks[0, t].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(mask_frame).save(os.path.join(save_dir, f"mask_{t:02d}.png"))
                        
                        # d) Predicted Disparity 저장 (같은 Min–Max 사용)
                        pred_frame = aligned_disp[0, t]  # [H,W]
                        norm_pd = (pred_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_pd = norm_pd.clamp(0,1)

                        pd_uint8 = (norm_pd.cpu().numpy() * 255).astype(np.uint8)
                        pd_rgb   = np.stack([pd_uint8]*3, axis=-1)
                        Image.fromarray(pd_rgb).save(os.path.join(save_dir, f"pred_{t:02d}.png"))
                        
                        # e) pred-disparity wandb에 저장
                        wb_images.append(wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"), caption=f"pred_epoch{epoch}_frame{t:02d}"))

                    logger.info(f"→ saved validation frames to '{save_dir}'")

                # 5) metric 평가 (모든 배치에 대해)
                for b in range(B):
                    inf_clip  = pred[b]              # [T,H,W]
                    gt_clip   = y[b].squeeze(1)      # [T,H,W]
                    mask_clip = masks[b].squeeze(1)  # [T,H,W]
                    pose      = extrinsics[b]
                    Kmat      = intrinsics[b]
                    absr, d1, tae = metric_val(inf_clip, gt_clip, "kitti", pose, Kmat)
                    total_absrel  += absr
                    total_delta1  += d1
                    total_tae     += tae
                    cnt_clip     += 1

            # 최종 통계
            avg_val_loss = val_loss / len(kitti_val_loader)
            avg_absrel   = total_absrel / cnt_clip
            avg_delta1   = total_delta1 / cnt_clip
            avg_tae      = total_tae / cnt_clip

        logger.info(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"AbsRel  : {avg_absrel:.4f}")
        logger.info(f"Delta1  : {avg_delta1:.4f}")
        logger.info(f"TAE    : {avg_tae:.4f}")
        #
        wandb.log({
            "vkitti_train_loss": avg_kitti_train_loss,
            "vkitti_val_loss": avg_val_loss,
            "vkitti_absrel": avg_absrel,
            "vkitti_delta1": avg_delta1,
            "vkitti_tae": avg_tae,
            "vkitti_epoch": epoch,
            "vkitti_pred_disparity": wb_images,
        })

        total_val_loss += avg_val_loss
        del wb_images
        
        # === scannet loop ===
        val_loss = 0.0
        total_absrel = 0.0
        total_delta1 = 0.0
        total_tae = 0.0
        cnt_clip = 0
        wb_images = []
        
        with torch.no_grad():
            for batch_idx, (x, y, extrinsics, intrinsics) in tqdm(enumerate(scannet_val_loader)):
                # 1) move to device
                x, y = x.to(device), y.to(device)
                extrinsics, intrinsics = extrinsics.to(device), intrinsics.to(device)
                
                # 2) model inference + basic losses
                pred = model(x)                                        # [B, T, H, W]
                masks = get_mask(y, min_depth=0.001, max_depth=80.0)   # [B, T, 1, H, W]
                masks = masks.to(device).bool()
                disp_normed   = norm_ssi(y, masks)
                ssi_loss_val  = loss_ssi(pred, disp_normed, masks.squeeze(2))
                tgm_loss_val  = loss_tgm(pred, y, masks)
                val_loss     += ratio_ssi * ssi_loss_val + ratio_tgm * tgm_loss_val

                logger.info(f"pred.mean(): {pred.mean().item():.6f}")

                # 3) prepare for scale & shift
                B, T, H, W = pred.shape

                MIN_DISP = 1.0 / 80.0      # depth=80m → disp=0.0125
                MAX_DISP = 1.0 / 0.001     # depth=0.001m → disp=1000.0

                raw_disp = pred.clamp(min=1e-6)                # [B, T, H, W]
                gt_disp  = (1.0 / y.clamp(min=1e-6)).squeeze(2) # [B, T, H, W]
                m_flat   = masks.squeeze(2).view(B, -1).float()# [B, P]
                p_flat   = raw_disp.view(B, -1)               # [B, P]
                g_flat   = gt_disp .view(B, -1)               # [B, P]

            
                # 4) build A, b for least-squares: A @ [a; b] ≈ b_vec 
                A = torch.stack([p_flat, torch.ones_like(p_flat, device=device)], dim=-1)  # [B,P,2]
                A = A * m_flat.unsqueeze(-1)    
                b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)  # mask out invalid
                
                 # 5) batched least-squares
                X = torch.linalg.lstsq(A, b_vec).solution  # [B,2,1]
                a = X[:,0,0].view(B,1,1,1)                 # [B,1,1,1]
                b = X[:,1,0].view(B,1,1,1)                 # [B,1,1,1]

                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)  # [B,T,H,W]
                
                # Save frames for first batch only
                if batch_idx == 0:
                    save_dir = f"outputs/experiment_{experiment}/scannet/epoch_{epoch}_batch_{batch_idx}"
                    os.makedirs(save_dir, exist_ok=True)
                    for t in range(T):

                        # a) RGB
                        rgb_norm = x[0, t]  # [3,H,W]
                        rgb_unc  = (rgb_norm * STD + MEAN).clamp(0,1)
                        rgb_np   = (rgb_unc.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))

                        # b) GT Disparity 저장 (Min–Max 정규화)
                        depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)       # [H,W]
                        disp_frame  = 1.0 / depth_frame                       # [H,W]
                        valid       = masks[0, t].squeeze(0)                  # [H,W] bool

                        # 유효 픽셀만 뽑아 min/max
                        d_vals = disp_frame[valid]
                        d_min, d_max = d_vals.min(), d_vals.max()

                        norm_gt = (disp_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_gt = norm_gt.clamp(0,1)

                        gt_uint8 = (norm_gt.cpu().numpy() * 255).astype(np.uint8)
                        gt_rgb   = np.stack([gt_uint8]*3, axis=-1)
                        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"gt_{t:02d}.png"))

                        # c) Mask 저장
                        mask_frame = masks[0, t].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(mask_frame).save(os.path.join(save_dir, f"mask_{t:02d}.png"))
                        
                        # d) Predicted Disparity 저장 (같은 Min–Max 사용)
                        pred_frame = aligned_disp[0, t]  # [H,W]
                        norm_pd = (pred_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_pd = norm_pd.clamp(0,1)

                        pd_uint8 = (norm_pd.cpu().numpy() * 255).astype(np.uint8)
                        pd_rgb   = np.stack([pd_uint8]*3, axis=-1)
                        Image.fromarray(pd_rgb).save(os.path.join(save_dir, f"pred_{t:02d}.png"))
                        
                        # e) pred-disparity wandb에 저장
                        wb_images.append(wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"), caption=f"pred_epoch{epoch}_frame{t:02d}"))

                    logger.info(f"→ saved validation frames to '{save_dir}'")

                # 5) metric 평가 (모든 배치에 대해)
                for b in range(B):
                    inf_clip  = pred[b]              # [T,H,W]
                    gt_clip   = y[b].squeeze(1)      # [T,H,W]
                    mask_clip = masks[b].squeeze(1)  # [T,H,W]
                    pose      = extrinsics[b]
                    Kmat      = intrinsics[b]
                    absr, d1, tae = metric_val(
                        infs   = inf_clip,
                        gts    = gt_clip,
                        data = "scannet",
                        poses  = pose,
                        Ks     = Kmat
                    )
                    total_absrel  += absr
                    total_delta1  += d1
                    total_tae     += tae
                    cnt_clip     += 1

            # 최종 통계
            avg_val_loss = val_loss / len(scannet_val_loader)
            avg_absrel   = total_absrel / cnt_clip
            avg_delta1   = total_delta1 / cnt_clip
            avg_tae      = total_tae / cnt_clip

        logger.info(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"AbsRel  : {avg_absrel:.4f}")
        logger.info(f"Delta1  : {avg_delta1:.4f}")
        logger.info(f"TAE    : {avg_tae:.4f}")
        
        wandb.log({
            "scannet_val_loss": avg_val_loss,
            "scannet_absrel": avg_absrel,
            "scannet_delta1": avg_delta1,
            "scannet_tae": avg_tae,
            "scannet_pred_disparity": wb_images,
        })
        del wb_images
        total_val_loss += avg_val_loss
        
        scheduler.step()

    # 최종 모델 저장
    logger.info(f"Training finished. Best checkpoint was from epoch {best_epoch} with validation loss {best_val_loss:.4f}.")
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt",type=str, default="./checkpoints/video_depth_anything_vits.pth")
    args = parser.parse_args()
    train(args)
