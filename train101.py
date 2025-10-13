import os
import argparse
import logging

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import wandb
import math
import warnings
from dotenv import load_dotenv

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.loss_MiDas import *
from utils.train_helper import *  # validate_with_infer_eval_subset, model_stream_step, batch_ls_scale_shift, norm_ssi, get_mask, to_BHW_pred
from data.dataLoader import *                 # KITTIVideoDataset, get_data_list

# 모델
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")

# ================ 실험 설정 ================
experiment = 101
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

# ================ KD helper (정확 Attn/K/V/Context) ================
def _detach_cache(cache):
    if cache is None:
        return None
    if isinstance(cache, (list, tuple)):
        return type(cache)(_detach_cache(c) for c in cache)
    if isinstance(cache, dict):
        return {k: _detach_cache(v) for k, v in cache.items()}
    if torch.is_tensor(cache):
        return cache.detach()
    return cache

# ================ 학습 루프 ================
def train(args):
    OUTPUT_DIR = f"outputs/experiment_{experiment}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 설정 로드
    with open("config_jh.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyper_params = config["hyper_parameter"]
    lr         = hyper_params["learning_rate"]
    ratio_ssi  = hyper_params["ratio_ssi"]          # Depth(SSI)
    ratio_tgm  = hyper_params["ratio_tgm"]          # Depth(TGM)
    num_epochs = hyper_params["epochs"]             # e.g., 25
    batch_size = hyper_params["batch_size"]
    CLIP_LEN   = hyper_params["clip_len"]           # W=32

    if args.epochs is not None:
        num_epochs = int(args.epochs)

    # W&B
    load_dotenv(dotenv_path=".env")
    wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
    run = wandb.init(project="stream_teacher_student", config=hyper_params, name=f"experiment_{experiment}")

    # 데이터
    kitti_path = "/home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI"
    rgb_clips, depth_clips = get_data_list(root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN)
    kitti_train = KITTIVideoDataset(
        rgb_paths=rgb_clips,
        depth_paths=depth_clips,
        resize_size=518,
        split="train",
        clip_len=CLIP_LEN,                 # ← 반드시 명시 (예: 32)
        per_epoch_samples=407,             # ← 1:1 비교 위해 고정
        sampling_mode="global_weighted",   # ← 전역 가중 샘플링
        balance_mode="proportional",       # ← 후보 수 비례
        min_stride=16,                      # ← 같은 폴더에서 중복 방지 간격(원하면)
        use_shift=False                    # ← 슬라이딩이면 보통 False 권장
    )
    kitti_train_loader = DataLoader(
        kitti_train,
        batch_size=batch_size,
        shuffle=True,          # 배치 내부 섞기만
        num_workers=4,
        pin_memory=True
    )

    # 모델 (단일 GPU)
    student = VideoDepthStudent(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)

    # Pretrained
    if args.pretrained_ckpt:
        logger.info(f"Loading Weight from {args.pretrained_ckpt}")
        sd = torch.load(args.pretrained_ckpt, map_location="cpu")
        student.load_state_dict(sd, strict=True)
        logger.info("Pretrained weights loaded successfully!")

    # Freeze 정책
    for p in student.pretrained.parameters(): p.requires_grad = False
    for p in student.head.parameters(): p.requires_grad = True
    student.train()

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
    best_delta1 = 0.0  # 이어서 학습 시에도 유지/갱신

    if args.resume_from and os.path.isfile(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location="cpu")

        # 1) 학생 모델 가중치
        sd = ckpt.get("model_state_dict", ckpt)
        # 혹시 모듈 프리픽스가 있어도 안전하게 로드
        try:
            student.load_state_dict(sd, strict=True)
        except RuntimeError:
            from collections import OrderedDict
            clean = OrderedDict()
            for k, v in sd.items():
                nk = k
                if nk.startswith("module."): nk = nk[len("module."):]
                if nk.startswith("student."): nk = nk[len("student."):]
                clean[nk] = v
            student.load_state_dict(clean, strict=False)

        # 2) 옵티마이저/스케줄러 상태(있으면)
        if "optimizer_state_dict" in ckpt:
            try: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e: logger.warning(f"Optimizer state load skipped: {e}")

        if "scheduler_state_dict" in ckpt:
            try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e: logger.warning(f"Scheduler state load skipped: {e}")

        # 3) 베스트 스코어 & 스타트 에폭
        if "best_val_delta1" in ckpt:
            try: best_delta1 = float(ckpt["best_val_delta1"])
            except: pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        logger.info(f"▶ Resumed from '{args.resume_from}' | start_epoch={start_epoch} / target_epochs={num_epochs} | best_delta1={best_delta1:.4f}")

    wandb.watch(student, log="all")
    best_delta1 = 0.0
    best_epoch  = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")
    
    # ---- Init real-pipeline validation (epoch = -1) ----
    # 초기 성능을 실제 inference+eval 축소 파이프라인으로 측정하여 W&B에 기록
    init_infer_dir = os.path.join(args.val_infer_dir, "init")
    os.makedirs(init_infer_dir, exist_ok=True)

    # 일시적으로 eval 모드
    _prev_train_state = student.training
    student.eval()
    try:
        init_metrics = validate_with_infer_eval_subset(
            model=student,                          # 학생만 사용
            json_file=args.val_json_file,                 # e.g., scannet_video_500.json
            infer_path=init_infer_dir,                    # init 전용 폴더에 저장하여 덮어쓰기 방지
            dataset=args.val_dataset_key,                 # 'scannet'
            dataset_eval_tag=args.val_dataset_tag,        # 'scannet_500'
            device='cuda' if torch.cuda.is_available() else 'cpu',
            input_size=518,
            scenes_to_eval=args.val_scenes,               # 2 scenes subset
            fp32=True
        )
    finally:
        # 원래 학습 모드 복귀
        if _prev_train_state:
            student.train()

    init_absrel = float(init_metrics.get("abs_relative_difference", float('nan')))
    init_rmse   = float(init_metrics.get("rmse_linear", float('nan')))
    init_delta1 = float(init_metrics.get("delta1_acc", float('nan')))

    # 콘솔/파일 로그
    logger.info(f"[Init] real-pipeline val  | absrel={init_absrel:.4f}  rmse={init_rmse:.4f}  delta1={init_delta1:.4f}")

    # W&B 로깅 (epoch=-1로 표기)
    wandb.log({
        "init/absrel": init_absrel,
        "init/rmse":   init_rmse,
        "init/delta1": init_delta1,
        "epoch": -1,
    })

    # 베스트 기준을 초기값으로 시작하고 싶다면(권장)
    best_delta1 = init_delta1


    # --------------------- Training ---------------------
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False):
        kitti_train.set_epoch(epoch)
        student.train()
        epoch_loss = epoch_frames = 0.0
        epoch_ssi = epoch_tgm = 0.0
        accum_loss = 0.0
        step_in_window = 0
        update_frequency = hyper_params.get("update_frequency", 6)

        batch_pbar = tqdm(enumerate(kitti_train_loader),
                          desc=f"Epoch {epoch+1}/{num_epochs} - Batches",
                          total=len(kitti_train_loader),
                          leave=False)
        for batch_idx, (x, y) in batch_pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, T = x.shape[:2]

            cache = None
            prev_pred_raw = prev_mask = prev_y = None

            frame_pbar = tqdm(range(T), desc=f"Batch {batch_idx+1} - Frames", leave=False, disable=T < 10)
            for t in frame_pbar:
                x_t = x[:, t:t+1]                              # [B,1,3,H,W]
                mask_t = get_mask(y[:, t:t+1], 1e-3, 80.0).to(device)

                # === Student: 스트리밍 1-step ===
                with autocast(enabled=torch.cuda.is_available()):
                    pred_t_raw, cache = model_stream_step(student, x_t, cache)
                    pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)

                    # ----- Depth Loss -----
                    gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6)).squeeze(2)  # [B,1,H,W]
                    if pred_t_raw.shape[0] != gt_disp_t.shape[0]:
                        pred_t_raw = pred_t_raw[:1]

                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)

                    pred_t_aligned_disp = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                    pred_t_aligned_depth = 1.0 / (pred_t_aligned_disp.clamp(min=1e-6))

                    disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)  # [B,1,H,W]
                    ssi_loss_t = loss_ssi(pred_t_aligned_disp.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))

                    if t > 0:
                        prev_aligned_disp = (a_star.detach() * prev_pred_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                        prev_aligned_depth = 1.0 / (prev_aligned_disp.clamp(min=1e-6))
                        curr_aligned_depth = pred_t_aligned_depth
                        pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)  # [B,2,H,W]
                        y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)                       # [B,2,1,H,W]
                        m_pair    = torch.cat([prev_mask, mask_t], dim=1)                         # [B,2,1,H,W]
                        tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                    else:
                        tgm_loss  = pred_t_raw.new_tensor(0.0)

                    loss = ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss

                # 누적/업데이트
                accum_loss += loss / update_frequency
                step_in_window += 1

                if step_in_window == update_frequency:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(accum_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += accum_loss.item()
                    accum_loss = 0.0
                    step_in_window = 0

                # 상태 업데이트
                cache = _detach_cache(cache)
                prev_pred_raw = pred_t_raw.detach()
                prev_mask = mask_t
                prev_y    = y[:, t:t+1]

                # 통계
                B_eff = pred_t_raw.shape[0]
                epoch_frames += B_eff
                epoch_ssi    += ssi_loss_t.item() * B_eff
                epoch_tgm    += tgm_loss.item()  * B_eff

                frame_pbar.set_postfix({
                    'SSI': f'{epoch_ssi/ max(1, epoch_frames):.4f}',
                    'TGM': f'{epoch_tgm/ max(1, epoch_frames):.4f}'
                })
            frame_pbar.close()
        batch_pbar.close()

        # --- Mini Real-pipeline Validation ---
        # (학생만 평가, infer_stream+eval과 동일 경로 축소판)
        val_metrics = validate_with_infer_eval_subset(
            model=student,
            json_file=args.val_json_file,
            infer_path=args.val_infer_dir,
            dataset=args.val_dataset_key,
            dataset_eval_tag=args.val_dataset_tag,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            input_size=518,
            scenes_to_eval=args.val_scenes,
            fp32=True
        )

        val_absrel = float(val_metrics.get("abs_relative_difference", float('nan')))
        val_rmse   = float(val_metrics.get("rmse_linear", float('nan')))
        val_delta1 = float(val_metrics.get("delta1_acc", float('nan')))

        # 로깅
        wandb.log({
            "train/loss": epoch_loss / max(1, len(kitti_train_loader)),
            "train/ssi":  epoch_ssi  / max(1, epoch_frames),
            "train/tgm":  epoch_tgm  / max(1, epoch_frames),
            "val_real/absrel": val_absrel,
            "val_real/rmse":   val_rmse,
            "val_real/delta1": val_delta1,
            "epoch": epoch,
        })

        # best 저장 (delta1 ↑)
        if val_delta1 > best_delta1:
            best_delta1 = val_delta1
            best_epoch  = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_delta1": best_delta1,
                "config": hyper_params,
            }, best_model_path)
            logger.info(f"🏆 Best model saved! Epoch {epoch}, Val delta1: {best_delta1:.4f}")

        # latest 저장
        torch.save({
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_absrel": val_absrel,
            "val_delta1": val_delta1,
            "val_rmse":   val_rmse,
            "config": hyper_params,
        }, latest_model_path)
        logger.info(f"📁 Latest model saved to {latest_model_path}")

        torch.cuda.empty_cache()
        scheduler.step()

    # 완료
    logger.info("=" * 30)
    logger.info("Training Completed!")
    logger.info(f"Total Epochs: {num_epochs}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best Val delta1: {best_delta1:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Latest model saved to: {latest_model_path}")
    logger.info("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")
    # real-pipeline mini-validation 설정
    parser.add_argument("--val_json_file",    type=str, default="/home/work/juhwan/monocular_depth//stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json")
    parser.add_argument("--val_infer_dir",    type=str, default="benchmark/output/scannet_stream_valmini")
    parser.add_argument("--val_dataset_key",  type=str, default="scannet")
    parser.add_argument("--val_dataset_tag",  type=str, default="scannet_500")
    parser.add_argument("--val_scenes",       type=int, default=2)
    parser.add_argument("--resume_from", type=str, default="", help="Path to latest/best checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs (e.g., 60)")
    args = parser.parse_args()
    train(args)
