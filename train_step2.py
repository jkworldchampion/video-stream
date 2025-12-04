#!/usr/bin/env python
import os
import argparse
import logging
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import wandb
from dotenv import load_dotenv

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.loss_MiDas import LossTGMVector, Loss_ssi_basic
from utils.train_helper import (
    validate_with_infer_eval_subset,
    validate_kitti_streaming,
    batch_ls_scale_shift,
    norm_ssi,
    get_mask,
    to_BHW_pred,
)
from data.dataLoader import KITTIVideoDataset, get_data_list

# 모델
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")

# ================ 실험 설정 ================
experiment = 2  # 2단계는 experiment_2 기반
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/experiment_{experiment}_step2.txt"),
    ],
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")


# ================ 학습 루프 (Stage 2: Head-only TAE finetune) ================
def train_step2(args):
    OUTPUT_DIR = f"outputs/new_step2/experiment_{experiment}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 설정 로드
    with open("config_jh.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyper_params = config["hyper_parameter"]

    # 1단계에서 쓰던 값들
    base_lr       = hyper_params["learning_rate"]
    ratio_ssi     = hyper_params["ratio_ssi"]
    ratio_tgm     = hyper_params["ratio_tgm"]
    base_epochs   = hyper_params["epochs"]
    batch_size    = hyper_params["batch_size"]
    CLIP_LEN      = hyper_params["clip_len"]
    # Step2 학습용 train 클립 길이 (여기서 64로 늘릴 것)
    CLIP_LEN_TRAIN = hyper_params.get("clip_len_step2", 64)

    # 2단계용 하이퍼파라미터 (config에 없으면 기본값 사용)
    lr_step2        = hyper_params.get("learning_rate_step2", base_lr * 0.5)
    num_epochs_step2 = hyper_params.get("epochs_step2", 10)
    lambda_ssi      = hyper_params.get("lambda_ssi_step2", 0.2)
    lambda_tgm      = hyper_params.get("lambda_tgm_step2", 0.2)
    lambda_static   = hyper_params.get("lambda_static", 1.0)
    lambda_dyn      = hyper_params.get("lambda_dyn", 0.5)
    tau_static      = hyper_params.get("tau_static", 0.001)  # static threshold (m)
    tau_dyn         = hyper_params.get("tau_dyn", 0.0005)     # dynamic threshold (m)

    if args.epochs is not None:
        num_epochs_step2 = int(args.epochs)

    # validation scene subset 설정
    raw_scene_indices = getattr(args, "val_scene_indices", None)
    scene_indices = None
    if raw_scene_indices:
        scene_indices = [int(idx.strip()) for idx in raw_scene_indices.split(",") if idx.strip()]
        scene_indices = sorted(set(scene_indices))
    if scene_indices:
        logger.info(f"Validation scene indices: {scene_indices}")

    # ============== W&B 설정 ==============
    if not args.test:
        wandb_config = config.get("wandb", {})
        wandb_entity = wandb_config.get("entity", "depth-finder")
        # project 이름은 구분을 위해 step2로 분리
        wandb_project = wandb_config.get("project_step2", "new_base_step2")

        load_dotenv(dotenv_path=".env")
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            config={
                **hyper_params,
                "learning_rate_step2": lr_step2,
                "epochs_step2": num_epochs_step2,
                "lambda_ssi_step2": lambda_ssi,
                "lambda_tgm_step2": lambda_tgm,
                "lambda_static": lambda_static,
                "lambda_dyn": lambda_dyn,
                "tau_static": tau_static,
                "tau_dyn": tau_dyn,
            },
            name=f"experiment_{experiment}_headonly_step2",
        )

    # ================== 데이터 ==================
    data_cfg   = config.get("data", {})
    kitti_path = data_cfg.get(
        "kitti_path",
        "/home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI",
    )

    # Train set
    rgb_clips, depth_clips = get_data_list(
        root_dir=kitti_path, data_name="kitti", split="train", clip_len=16
    )
    kitti_train = KITTIVideoDataset(
        rgb_paths=rgb_clips,
        depth_paths=depth_clips,
        resize_size=518,
        split="train",
        clip_len=16,
    )
    kitti_train_loader = DataLoader(
        kitti_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Validation set (KITTI)
    rgb_clips_val, depth_clips_val, cam_ids_val, intrin_clips_val, extrin_clips_val = get_data_list(
        root_dir=kitti_path, data_name="kitti", split="val", clip_len=CLIP_LEN
    )
    kitti_val = KITTIVideoDataset(
        rgb_paths=rgb_clips_val,
        depth_paths=depth_clips_val,
        cam_ids=cam_ids_val,
        intrin_clips=intrin_clips_val,
        extrin_clips=extrin_clips_val,
        resize_size=518,
        split="val",
    )
    kitti_val_loader = DataLoader(
        kitti_val,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ================== 모델 (Student) ==================
    student = VideoDepthStudent(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        num_frames=CLIP_LEN,
    ).to(device)

    # 1단계 best weight 로드
    if args.init_ckpt:
        logger.info(f"Loading initial weights from {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        try:
            student.load_state_dict(sd, strict=True)
        except RuntimeError:
            from collections import OrderedDict
            clean = OrderedDict()
            for k, v in sd.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module."):]
                if nk.startswith("student."):
                    nk = nk[len("student."):]
                clean[nk] = v
            student.load_state_dict(clean, strict=False)
        logger.info("✅ Stage 1 best weights loaded successfully!")

    # Freeze 정책: encoder는 완전 freeze, head만 학습
    for p in student.pretrained.parameters():
        p.requires_grad = False
    for p in student.head.parameters():
        p.requires_grad = True

    student.train()

    # Optim/Sch: head-only
    student_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(student_params, lr=lr_step2, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs_step2, eta_min=1e-6)

    # Loss (기존 depth loss 재사용)
    loss_tgm = LossTGMVector(diff_depth_th=0.05)
    loss_ssi = Loss_ssi_basic()
    scaler = GradScaler()

    best_epoch = 0
    best_scannet_delta1 = 0.0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model_step2.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model_step2.pth")

    # ================ 설정 출력 (요약) ================
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION (STEP 2: HEAD-ONLY TAE FINETUNE)")
    logger.info("=" * 60)
    logger.info(f"Experiment Number: {experiment}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("")
    logger.info("--- Hyperparameters (Base) ---")
    logger.info(f"  Base Learning Rate: {base_lr}")
    logger.info(f"  Base Epochs: {base_epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Clip Length: {CLIP_LEN}")
    logger.info(f"  SSI Loss Weight (Stage1): {ratio_ssi}")
    logger.info(f"  TGM Loss Weight (Stage1): {ratio_tgm}")
    logger.info("")
    logger.info("--- Step2 Hyperparameters ---")
    logger.info(f"  Step2 Learning Rate: {lr_step2}")
    logger.info(f"  Step2 Epochs: {num_epochs_step2}")
    logger.info(f"  λ_ssi: {lambda_ssi}")
    logger.info(f"  λ_tgm: {lambda_tgm}")
    logger.info(f"  λ_static: {lambda_static}")
    logger.info(f"  λ_dyn: {lambda_dyn}")
    logger.info(f"  τ_static: {tau_static}")
    logger.info(f"  τ_dyn: {tau_dyn}")
    logger.info("")
    logger.info("--- Model Architecture (Student) ---")
    logger.info(f"  Encoder: {student.encoder}")
    logger.info(f"  Features: 64")
    logger.info(f"  Out Channels: [48, 96, 192, 384]")
    logger.info(f"  Num Frames: {CLIP_LEN}")
    logger.info("  Encoder frozen, head-only train")
    logger.info("")
    logger.info("--- Data Configuration ---")
    logger.info(f"  KITTI Path: {kitti_path}")
    logger.info(f"  Num Workers: 4")
    logger.info("")
    logger.info("--- Validation Configuration ---")
    logger.info(f"  Dataset: {args.val_dataset_key}")
    logger.info(f"  Dataset Tag: {args.val_dataset_tag}")
    logger.info(f"  Scenes to Eval: {len(scene_indices) if scene_indices else args.val_scenes}")
    if scene_indices:
        logger.info(f"  Scene Indices: {scene_indices}")
    logger.info("")
    logger.info("--- Init Checkpoint ---")
    logger.info(f"  Init from: {args.init_ckpt if args.init_ckpt else 'None'}")
    logger.info("=" * 60)
    logger.info("")

    # --------------------- Initial Validation ---------------------
    logger.info("=" * 60)
    logger.info("Running initial validation before Step 2 training...")
    logger.info("=" * 60)

    student.eval()

    # 1. KITTI Validation
    logger.info("Running KITTI validation...")
    kitti_val_metrics = validate_kitti_streaming(
        model=student,
        val_loader=kitti_val_loader,
        device=device,
        loss_ssi_fn=loss_ssi,
        loss_tgm_fn=loss_tgm,
        ratio_ssi=ratio_ssi,
        ratio_tgm=ratio_tgm,
        min_depth=1e-3,
        max_depth=80.0,
    )
    kitti_val_loss   = kitti_val_metrics['loss']
    kitti_val_absrel = kitti_val_metrics['absrel']
    kitti_val_delta1 = kitti_val_metrics['delta1']

    logger.info(
        f"[Init KITTI] loss={kitti_val_loss:.4f} | "
        f"absrel={kitti_val_absrel:.4f} | delta1={kitti_val_delta1:.4f}"
    )

    # 2. ScanNet Validation
    logger.info("Running ScanNet validation...")
    init_infer_dir = os.path.join(args.val_infer_dir, "init_step2")
    os.makedirs(init_infer_dir, exist_ok=True)

    logger.info(
        f"[DEBUG] Step2 ScanNet val: scenes_to_eval={args.val_scenes}, "
        f"scene_indices={scene_indices}"
    )

    scannet_metrics = validate_with_infer_eval_subset(
        model=student,
        json_file=args.val_json_file,
        infer_path=init_infer_dir,
        dataset=args.val_dataset_key,
        dataset_eval_tag=args.val_dataset_tag,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        input_size=518,
        scenes_to_eval=args.val_scenes,
        scene_indices=scene_indices,
        fp32=True,
    )
    avg_metrics    = scannet_metrics.get("avg", {})
    scannet_absrel = float(avg_metrics.get("abs_relative_difference", float('nan')))
    scannet_rmse   = float(avg_metrics.get("rmse_linear", float('nan')))
    scannet_delta1 = float(avg_metrics.get("delta1_acc", float('nan')))

    logger.info(
        f"[Init ScanNet] absrel={scannet_absrel:.4f} | "
        f"rmse={scannet_rmse:.4f} | delta1={scannet_delta1:.4f}"
    )

    if not args.test:
        wandb.log({
            "init_step2/val_kitti_loss":   kitti_val_loss,
            "init_step2/val_kitti_absrel": kitti_val_absrel,
            "init_step2/val_kitti_delta1": kitti_val_delta1,
            "init_step2/val_real_absrel":  scannet_absrel,
            "init_step2/val_real_rmse":    scannet_rmse,
            "init_step2/val_real_delta1":  scannet_delta1,
            "epoch": -1,
        })

    logger.info("=" * 60)
    logger.info("Initial validation completed! Starting Step 2 training...")
    logger.info("=" * 60)

    if not args.test:
        wandb.watch(student, log="all")

    # --------------------- Training (Step 2) ---------------------
    update_frequency = hyper_params.get("update_frequency_step2", hyper_params.get("update_frequency", 6))

    for epoch in tqdm(range(num_epochs_step2), desc="Epoch(Step2)", leave=False):
        student.train()
        epoch_loss      = 0.0
        epoch_frames    = 0.0
        epoch_ssi       = 0.0
        epoch_tgm       = 0.0
        epoch_static    = 0.0
        epoch_dyn       = 0.0

        step_in_window = 0
        m = student.module if hasattr(student, "module") else student

        batch_pbar = tqdm(
            enumerate(kitti_train_loader),
            desc=f"Step2 Epoch {epoch+1}/{num_epochs_step2} - Batches",
            total=len(kitti_train_loader),
            leave=False,
        )

        for batch_idx, (x, y) in batch_pbar:
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, T = x.shape[:2]

            cache_state = None

            # 이전 프레임 depth/GT/mask 저장용
            prev_pred_depth = None
            prev_gt_depth   = None
            prev_mask       = None

            prev2_pred_depth = None
            prev2_gt_depth   = None
            prev2_mask       = None

            frame_pbar = tqdm(
                range(T),
                desc=f"Batch {batch_idx+1} - Frames",
                leave=False,
                disable=T < 10,
            )

            for t in frame_pbar:
                x_t = x[:, t:t+1]  # [B,1,3,H,W]
                y_t = y[:, t:t+1]  # [B,1,1,H,W]
                mask_t = get_mask(y_t, 1e-3, 80.0).to(device)  # [B,1,1,H,W]

                with autocast(enabled=torch.cuda.is_available()):
                    # 1-frame streaming step
                    pred_t_net, cache_state = m.stream_step_train(x_t, cache_state)
                    pred_t_raw = to_BHW_pred(pred_t_net).clamp(min=1e-6)  # [B,H,W] (disparity)

                    # ----- Scale-Shift align -----
                    gt_disp_t = (1.0 / y_t.clamp(min=1e-6)).squeeze(2)  # [B,1,H,W]

                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)

                    pred_t_aligned_disp   = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)  # [B,H,W]
                    pred_t_aligned_depth  = 1.0 / pred_t_aligned_disp.clamp(min=1e-6)                               # [B,H,W]
                    gt_depth_t            = y_t.squeeze(2)                                                         # [B,1,H,W]

                    # ----- SSI (per-frame) -----
                    disp_normed_t = norm_ssi(y_t, mask_t).squeeze(2)
                    ssi_loss_t = loss_ssi(
                        pred_t_aligned_disp.unsqueeze(1),
                        disp_normed_t,
                        mask_t.squeeze(2),
                    )

                    # ----- TGM (2-frame depth consistency, 기존과 동일) -----
                    if prev_pred_depth is not None:
                        prev_aligned_depth = prev_pred_depth  # 이미 aligned로 저장
                        curr_aligned_depth = pred_t_aligned_depth

                        pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)  # [B,2,H,W]
                        y_pair    = torch.cat([prev_gt_depth, gt_depth_t], dim=1)                 # [B,2,1,H,W]
                        m_pair    = torch.cat([prev_mask, mask_t], dim=1)                         # [B,2,1,H,W]
                        tgm_loss = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                    else:
                        tgm_loss = pred_t_raw.new_tensor(0.0)

                    # ----- (NEW) Static TAE loss (2-frame) -----
                    static_loss_t = pred_t_raw.new_tensor(0.0)
                    if prev_pred_depth is not None:
                        # valid mask (2-frame)
                        valid_2 = (prev_mask & mask_t).squeeze(2)  # [B,1,H,W]

                        gt_prev = prev_gt_depth  # [B,1,H,W]
                        gt_curr = gt_depth_t     # [B,1,H,W]

                        static_mask = ((gt_prev - gt_curr).abs() < tau_static) & valid_2  # [B,1,H,W]

                        if static_mask.sum() > 0:
                            diff_pred = (prev_pred_depth - pred_t_aligned_depth).abs()  # [B,H,W]
                            static_loss_t = (
                                diff_pred * static_mask.squeeze(1)
                            ).sum() / static_mask.sum().clamp(min=1.0)

                    # ----- (NEW) Dynamic smoothness loss (3-frame, backward 2nd diff) -----
                    dyn_loss_t = pred_t_raw.new_tensor(0.0)
                    if (prev_pred_depth is not None) and (prev2_pred_depth is not None):
                        valid_3 = (prev2_mask & prev_mask & mask_t).squeeze(2)  # [B,1,H,W]

                        gt_tm2 = prev2_gt_depth  # [B,1,H,W]
                        gt_tm1 = prev_gt_depth   # [B,1,H,W]
                        gt_t   = gt_depth_t      # [B,1,H,W]

                        delta_gt_1 = (gt_tm1 - gt_tm2).abs()
                        delta_gt_2 = (gt_t   - gt_tm1).abs()

                        dyn_mask = ((delta_gt_1 > tau_dyn) | (delta_gt_2 > tau_dyn)) & valid_3

                        if dyn_mask.sum() > 0:
                            # 2차 차분: d_t - 2 d_{t-1} + d_{t-2}
                            second_diff = (
                                pred_t_aligned_depth
                                - 2.0 * prev_pred_depth
                                + prev2_pred_depth
                            ).abs()  # [B,H,W]

                            dyn_loss_t = (
                                second_diff * dyn_mask.squeeze(1)
                            ).sum() / dyn_mask.sum().clamp(min=1.0)

                    # ----- 최종 loss (Step 2 전용) -----
                    loss = (
                        lambda_ssi    * ssi_loss_t +
                        lambda_tgm    * tgm_loss +
                        lambda_static * static_loss_t +
                        lambda_dyn    * dyn_loss_t
                    )

                # Gradient Accumulation
                scaled_loss = loss / update_frequency
                scaler.scale(scaled_loss).backward()
                epoch_loss += loss.item()

                step_in_window += 1
                if step_in_window == update_frequency:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step_in_window = 0

                # 상태 업데이트 (다음 frame 대비)
                prev2_pred_depth = prev_pred_depth
                prev2_gt_depth   = prev_gt_depth
                prev2_mask       = prev_mask

                prev_pred_depth = pred_t_aligned_depth.detach()
                prev_gt_depth   = gt_depth_t.detach()
                prev_mask       = mask_t.detach()

                # 통계
                B_eff = pred_t_raw.shape[0]
                epoch_frames += B_eff
                epoch_ssi    += ssi_loss_t.item()      * B_eff
                epoch_tgm    += tgm_loss.item()        * B_eff
                epoch_static += static_loss_t.item()   * B_eff
                epoch_dyn    += dyn_loss_t.item()      * B_eff

                frame_pbar.set_postfix({
                    'wSSI':    f'{(epoch_ssi    / max(1, epoch_frames) * lambda_ssi):.4f}',
                    'wTGM':    f'{(epoch_tgm    / max(1, epoch_frames) * lambda_tgm):.4f}',
                    'wStatic': f'{(epoch_static / max(1, epoch_frames) * lambda_static):.4f}',
                    'wDyn':    f'{(epoch_dyn    / max(1, epoch_frames) * lambda_dyn):.4f}',
                })
            frame_pbar.close()
        batch_pbar.close()

        # 남은 gradient step 처리
        if step_in_window > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # --- Validation (KITTI) ---
        student.eval()
        kitti_val_metrics = validate_kitti_streaming(
            model=student,
            val_loader=kitti_val_loader,
            device=device,
            loss_ssi_fn=loss_ssi,
            loss_tgm_fn=loss_tgm,
            ratio_ssi=ratio_ssi,
            ratio_tgm=ratio_tgm,
            min_depth=1e-3,
            max_depth=80.0,
        )
        kitti_val_loss   = kitti_val_metrics['loss']
        kitti_val_absrel = kitti_val_metrics['absrel']
        kitti_val_delta1 = kitti_val_metrics['delta1']

        # --- Validation (ScanNet, streaming pipeline) ---
        scannet_metrics = validate_with_infer_eval_subset(
            model=student,
            json_file=args.val_json_file,
            infer_path=args.val_infer_dir,
            dataset=args.val_dataset_key,
            dataset_eval_tag=args.val_dataset_tag,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            input_size=518,
            scenes_to_eval=args.val_scenes,
            scene_indices=scene_indices,
            fp32=True,
        )
        avg_metrics    = scannet_metrics.get("avg", {})
        scannet_absrel = float(avg_metrics.get("abs_relative_difference", float('nan')))
        scannet_rmse   = float(avg_metrics.get("rmse_linear", float('nan')))
        scannet_delta1 = float(avg_metrics.get("delta1_acc", float('nan')))

        mean_ssi    = epoch_ssi    / max(1, epoch_frames)
        mean_tgm    = epoch_tgm    / max(1, epoch_frames)
        mean_static = epoch_static / max(1, epoch_frames)
        mean_dyn    = epoch_dyn    / max(1, epoch_frames)

        if not args.test:
            wandb.log({
                "step2/train/loss":   epoch_loss / max(1, len(kitti_train_loader)),
                "step2/train/ssi":    mean_ssi,
                "step2/train/tgm":    mean_tgm,
                "step2/train/static": mean_static,
                "step2/train/dyn":    mean_dyn,

                "step2/val_kitti/loss":   kitti_val_loss,
                "step2/val_kitti/absrel": kitti_val_absrel,
                "step2/val_kitti/delta1": kitti_val_delta1,
                "step2/val_real/absrel":  scannet_absrel,
                "step2/val_real/rmse":    scannet_rmse,
                "step2/val_real/delta1":  scannet_delta1,
                "epoch_step2": epoch,
            })

        # best 저장 기준: ScanNet delta1 (혹은 나중에 TAE metric으로 바꿔도 됨)
        if best_scannet_delta1 < scannet_delta1:
            best_scannet_delta1 = scannet_delta1
            best_epoch = epoch
            save_dict = {
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_scannet_delta1": best_scannet_delta1,
                "config": hyper_params,
            }
            torch.save(save_dict, best_model_path)
            logger.info(
                f"🏆 [STEP2] Best model saved! Epoch {epoch}, "
                f"ScanNet delta1: {scannet_delta1:.4f} | "
                f"KITTI delta1: {kitti_val_delta1:.4f}"
            )

        # latest 저장
        save_dict_latest = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "kitti_val_delta1": kitti_val_delta1,
            "scannet_val_delta1": scannet_delta1,
            "config": hyper_params,
        }
        torch.save(save_dict_latest, latest_model_path)
        logger.info(
            f"📁 [STEP2] Latest model saved | Epoch {epoch} | "
            f"ScanNet delta1: {scannet_delta1:.4f} | "
            f"KITTI delta1: {kitti_val_delta1:.4f}"
        )

        torch.cuda.empty_cache()
        scheduler.step()

    # 완료
    logger.info("=" * 30)
    logger.info("Step 2 Training Completed (Head-only TAE Finetune)!")
    logger.info(f"Total Epochs (Step2): {num_epochs_step2}")
    logger.info(f"Best Epoch (Step2): {best_epoch}")
    logger.info(f"Best ScanNet delta1 (Step2): {best_scannet_delta1:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Latest model saved to: {latest_model_path}")
    logger.info("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1단계 best ckpt 경로 (질문에서 제시한 경로)
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="/home/work/juhwan/monocular_depth/stream/video-stream/outputs/new/experiment_2/best_model.pth",
        help="Stage 1 best checkpoint to initialize Step 2 training",
    )
    # real-pipeline mini-validation 설정 (1단계와 동일)
    parser.add_argument(
        "--val_json_file",
        type=str,
        default="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json",
    )
    parser.add_argument(
        "--val_infer_dir",
        type=str,
        default="benchmark/output/scannet_stream_valmini_step2",
    )
    parser.add_argument(
        "--val_dataset_key",
        type=str,
        default="scannet",
    )
    parser.add_argument(
        "--val_dataset_tag",
        type=str,
        default="scannet_500",
    )
    parser.add_argument(
        "--val_scenes",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--val_scene_indices",
        type=str,
        default="0,1",
        help="Comma-separated dataset indices for validation subset. Empty string disables explicit selection.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override Step2 total epochs (e.g., 10)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only run validation (no training)",
    )
    args = parser.parse_args()

    if args.test:
        # test 모드는 epoch=0 으로 한 번만 validation 수행 (학습 없이)
        logger.info("Running Step2 script in TEST mode (validation only).")
    train_step2(args)
