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
from utils.train_helper import (  # 반드시 이 함수들/로더들이 구현되어 있어야 함
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
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")

# ================ 실험 설정 ================
experiment = 0
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/experiment_{experiment}.txt")],
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")


# ================ 학습 루프 ================
def train(args):
    OUTPUT_DIR = f"outputs/new/experiment_{experiment}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 설정 로드
    with open("config_jh.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyper_params = config["hyper_parameter"]
    lr         = hyper_params["learning_rate"]
    ratio_ssi  = hyper_params["ratio_ssi"]          # Depth(SSI)
    ratio_tgm  = hyper_params["ratio_tgm"]          # Depth(TGM)
    num_epochs = hyper_params["epochs"]
    batch_size = hyper_params["batch_size"]
    CLIP_LEN   = hyper_params["clip_len"]

    # KD 설정 (output KD용 최소 설정)
    kd_cfg     = config.get("kd_aux", {})
    kd_enabled = bool(kd_cfg.get("enabled", False))
    kd_lambda  = float(kd_cfg.get("lambda_kd", 1.0))

    if args.epochs is not None:
        num_epochs = int(args.epochs)

    # validation scene subset 설정
    raw_scene_indices = getattr(args, "val_scene_indices", None)
    scene_indices = None
    if raw_scene_indices:
        scene_indices = [int(idx.strip()) for idx in raw_scene_indices.split(",") if idx.strip()]
        scene_indices = sorted(set(scene_indices))
    if scene_indices:
        logger.info(f"Validation scene indices: {scene_indices}")

    # W&B
    if not args.test:
        wandb_config = config.get("wandb", {})
        wandb_entity = wandb_config.get("entity", "depth-finder")
        wandb_project = wandb_config.get("project", "new_base")

        load_dotenv(dotenv_path=".env")
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            config=hyper_params,
            name=f"experiment_{experiment}_kd_output"
        )

    # ================== 데이터 ==================
    kitti_path = "/home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI"

    # Train set
    rgb_clips, depth_clips = get_data_list(
        root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN
    )
    kitti_train = KITTIVideoDataset(
        rgb_paths=rgb_clips,
        depth_paths=depth_clips,
        resize_size=518,
        split="train",
    )
    kitti_train_loader = DataLoader(
        kitti_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
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
        kitti_val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    # ================== 모델 (Student / Teacher) ==================
    student = VideoDepthStudent(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        num_frames=CLIP_LEN,
    ).to(device)

    teacher = None
    if kd_enabled:
        teacher = VideoDepthTeacher(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            num_frames=CLIP_LEN,
        ).to(device)

    # Pretrained ckpt 로드
    if args.pretrained_ckpt:
        logger.info(f"Loading Student weights from {args.pretrained_ckpt}")
        student_sd = torch.load(args.pretrained_ckpt, map_location="cpu")
        student.load_state_dict(student_sd, strict=True)
        logger.info("✅ Student pretrained weights loaded successfully!")

        if kd_enabled and teacher is not None:
            logger.info(f"Loading Teacher weights from {args.pretrained_ckpt}")
            teacher_sd = torch.load(args.pretrained_ckpt, map_location="cpu")
            teacher.load_state_dict(teacher_sd, strict=True)
            logger.info("✅ Teacher pretrained weights loaded successfully!")

    # Freeze 정책: encoder freeze, head만 학습
    for p in student.pretrained.parameters():
        p.requires_grad = False
    for p in student.head.parameters():
        p.requires_grad = True
    student.train()

    if kd_enabled and teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # Optim/Sch
    student_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(student_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Loss
    loss_tgm = LossTGMVector(diff_depth_th=0.05)
    loss_ssi = Loss_ssi_basic()
    scaler = GradScaler()

    # ----- Resume (optional) -----
    start_epoch = 0
    best_val_loss = float('inf')  # KITTI val loss 기준

    if args.resume_from and os.path.isfile(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location="cpu")

        # 1) 학생 모델 가중치
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

        # 2) 옵티마이저/스케줄러
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Optimizer state load skipped: {e}")

        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Scheduler state load skipped: {e}")

        # 3) best / epoch
        if "best_val_loss" in ckpt:
            try:
                best_val_loss = float(ckpt["best_val_loss"])
            except:
                pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        logger.info(
            f"▶ Resumed from '{args.resume_from}' | start_epoch={start_epoch} / "
            f"target_epochs={num_epochs} | best_val_loss={best_val_loss:.4f}"
        )

    if not args.test:
        wandb.watch(student, log="all")

    best_epoch = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")

    # ================ 설정 출력 (요약) ================
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Experiment Number: {experiment}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("")
    logger.info("--- Hyperparameters ---")
    logger.info(f"  Learning Rate: {lr}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Clip Length: {CLIP_LEN}")
    logger.info(f"  Update Frequency: {hyper_params.get('update_frequency', 6)}")
    logger.info(f"  SSI Loss Weight: {ratio_ssi}")
    logger.info(f"  TGM Loss Weight: {ratio_tgm}")
    logger.info("")
    logger.info("--- Model Architecture (Student) ---")
    logger.info(f"  Encoder: {student.encoder}")
    logger.info(f"  Features: 64")
    logger.info(f"  Out Channels: [48, 96, 192, 384]")
    logger.info(f"  Num Frames: {CLIP_LEN}")
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
    logger.info("--- Pretrained & Resume ---")
    logger.info(f"  Pretrained Checkpoint: {args.pretrained_ckpt if args.pretrained_ckpt else 'None'}")
    logger.info(f"  Resume From: {args.resume_from if args.resume_from else 'None'}")
    logger.info("=" * 60)
    logger.info("")

    # --------------------- Initial Validation ---------------------
    logger.info("=" * 60)
    logger.info("Running initial validation before training...")
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
    init_infer_dir = os.path.join(args.val_infer_dir, "init")
    os.makedirs(init_infer_dir, exist_ok=True)

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
            "init/val_kitti_loss":   kitti_val_loss,
            "init/val_kitti_absrel": kitti_val_absrel,
            "init/val_kitti_delta1": kitti_val_delta1,
            "init/val_real_absrel":  scannet_absrel,
            "init/val_real_rmse":    scannet_rmse,
            "init/val_real_delta1":  scannet_delta1,
            "epoch": -1,
        })

    logger.info("=" * 60)
    logger.info("Initial validation completed! Starting training...")
    logger.info("=" * 60)

    best_val_loss = kitti_val_loss  # 기준

    # --------------------- Training ---------------------
    update_frequency = hyper_params.get("update_frequency", 6)

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False):
        student.train()
        epoch_loss   = 0.0
        epoch_frames = 0.0
        epoch_ssi    = 0.0
        epoch_tgm    = 0.0

        step_in_window = 0

        m = student.module if hasattr(student, "module") else student

        batch_pbar = tqdm(
            enumerate(kitti_train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} - Batches",
            total=len(kitti_train_loader),
            leave=False,
        )
        for batch_idx, (x, y) in batch_pbar:
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, T = x.shape[:2]

            cache_state = None
            prev_pred_raw = prev_mask = prev_y = None

            # ----- Teacher clip prediction (KD용) -----
            teacher_disp_clip = None
            if kd_enabled and teacher is not None:
                with torch.no_grad():
                    with autocast(enabled=torch.cuda.is_available()):
                        teacher_depth_clip = teacher(x)                                   # [B,T,H,W] depth
                        teacher_disp_clip  = 1.0 / teacher_depth_clip.clamp(min=1e-6) 

            frame_pbar = tqdm(range(T), desc=f"Batch {batch_idx+1} - Frames", leave=False, disable=T < 10)
            for t in frame_pbar:
                x_t = x[:, t:t+1]  # [B,1,3,H,W]
                mask_t = get_mask(y[:, t:t+1], 1e-3, 80.0).to(device)

                with autocast(enabled=torch.cuda.is_available()):
                    # 1-frame streaming step (student 내부 cache_state 사용)
                    pred_t_net, cache_state = m.stream_step_train(x_t, cache_state)
                    pred_t_raw = to_BHW_pred(pred_t_net).clamp(min=1e-6)  # [B,H,W]

                    # ----- Scale-Shift & Depth Losses -----
                    gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6)).squeeze(2)  # [B,1,H,W]
                    assert pred_t_raw.shape[0] == gt_disp_t.shape[0]

                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)

                    pred_t_aligned_disp = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                    pred_t_aligned_depth = 1.0 / pred_t_aligned_disp.clamp(min=1e-6)

                    disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)
                    ssi_loss_t = loss_ssi(pred_t_aligned_disp.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))

                    if t > 0:
                        prev_aligned_disp = (a_star.detach() * prev_pred_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                        prev_aligned_depth = 1.0 / prev_aligned_disp.clamp(min=1e-6)
                        curr_aligned_depth = pred_t_aligned_depth

                        pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)  # [B,2,H,W]
                        y_pair = torch.cat([prev_y, y[:, t:t+1]], dim=1)  # [B,2,1,H,W]
                        m_pair = torch.cat([prev_mask, mask_t], dim=1)    # [B,2,1,H,W]
                        tgm_loss = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                    else:
                        tgm_loss = pred_t_raw.new_tensor(0.0)

                    # ----- (NEW) KD loss: Teacher clip disparity vs Student stream disparity -----
                    kd_loss_t = pred_t_raw.new_tensor(0.0)
                    if kd_enabled and teacher_disp_clip is not None:
                        teacher_disp_t = teacher_disp_clip[:, t]                       # [B,H,W]
                        # 마스크 적용한 L1 KD (disparity space)
                        kd_mask = mask_t.squeeze(2).squeeze(1)                         # [B,H,W]
                        if kd_mask.any():
                            diff = (pred_t_raw - teacher_disp_t).abs() * kd_mask
                            kd_loss_t = diff.sum() / kd_mask.sum().clamp(min=1.0)
                        else:
                            kd_loss_t = pred_t_raw.new_tensor(0.0)

                    # 최종 loss: depth + KD
                    loss = ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss + (kd_lambda * kd_loss_t if kd_enabled else 0.0)

                # 1) grad accumulation
                scaled_loss = loss / update_frequency
                scaler.scale(scaled_loss).backward()
                epoch_loss += loss.item()

                step_in_window += 1
                if step_in_window == update_frequency:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step_in_window = 0

                # 상태 업데이트 (TGM용)
                prev_pred_raw = pred_t_raw.detach()
                prev_mask = mask_t
                prev_y = y[:, t:t+1]

                # 통계
                B_eff = pred_t_raw.shape[0]
                epoch_frames += B_eff
                epoch_ssi += ssi_loss_t.item() * B_eff
                epoch_tgm += tgm_loss.item() * B_eff
                if kd_enabled:
                    epoch_kd_dis += kd_loss_t.item() * B_eff  # 이름은 kd_dis 그대로 재활용 (output KD)

                frame_pbar.set_postfix({
                    'wSSI': f'{epoch_ssi / max(1, epoch_frames) * ratio_ssi:.4f}',
                    'wTGM': f'{epoch_tgm / max(1, epoch_frames) * ratio_tgm:.4f}',
                    'wKD':  f'{(epoch_kd_dis / max(1, epoch_frames) * kd_lambda):.4e}' if kd_enabled else '0.0000',
                })
            frame_pbar.close()
        batch_pbar.close()

        # 남은 gradient step 처리
        if step_in_window > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # --- KITTI Validation ---
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

        # --- ScanNet Validation ---
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

        # KD 평균 (프레임당)
        mean_kd = (epoch_kd_dis / max(1, epoch_frames)) if kd_enabled else 0.0

        # W&B 로깅
        if not args.test:
            wandb.log({
                "train/loss": epoch_loss / max(1, len(kitti_train_loader)),
                "train/ssi":  epoch_ssi  / max(1, epoch_frames),
                "train/tgm":  epoch_tgm  / max(1, epoch_frames),
                "train/kd":   mean_kd * kd_lambda if kd_enabled else 0.0,

                "val_kitti/loss":   kitti_val_loss,
                "val_kitti/absrel": kitti_val_absrel,
                "val_kitti/delta1": kitti_val_delta1,
                "val_real/absrel":  scannet_absrel,
                "val_real/rmse":    scannet_rmse,
                "val_real/delta1":  scannet_delta1,
                "epoch": epoch,
            })

        # best 저장 (KITTI val loss 기준)
        if kitti_val_loss < best_val_loss:
            best_val_loss = kitti_val_loss
            best_epoch = epoch
            save_dict = {
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": hyper_params,
            }
            torch.save(save_dict, best_model_path)
            logger.info(
                f"🏆 Best model saved! Epoch {epoch}, "
                f"KITTI val loss: {best_val_loss:.4f} | "
                f"KITTI delta1: {kitti_val_delta1:.4f} | "
                f"ScanNet delta1: {scannet_delta1:.4f}"
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
            f"📁 Latest model saved | Epoch {epoch} | "
            f"ScanNet delta1: {scannet_delta1:.4f} | "
            f"KITTI delta1: {kitti_val_delta1:.4f}"
        )

        torch.cuda.empty_cache()
        scheduler.step()

    # 완료
    logger.info("=" * 30)
    logger.info("Training Completed!")
    logger.info(f"Total Epochs: {num_epochs}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best KITTI Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Latest model saved to: {latest_model_path}")
    logger.info("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")
    # real-pipeline mini-validation 설정
    parser.add_argument("--val_json_file",    type=str, default="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json")
    parser.add_argument("--val_infer_dir",    type=str, default="benchmark/output/scannet_stream_valmini")
    parser.add_argument("--val_dataset_key",  type=str, default="scannet")
    parser.add_argument("--val_dataset_tag",  type=str, default="scannet_500")
    parser.add_argument("--val_scenes",       type=int, default=2)
    parser.add_argument("--val_scene_indices",type=str,default="0,1",help="Comma-separated dataset indices for validation subset. Empty string disables explicit selection.")
    parser.add_argument("--resume_from", type=str, default="", help="Path to latest/best checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs (e.g., 60)")
    parser.add_argument("--test", action="store_true", help="Only run validation")
    args = parser.parse_args()
    train(args)
