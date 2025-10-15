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
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher
from video_depth_anything.aux.aux_block import AuxBlock          # Proj -> 1L Transformer(bi, hole-mask) -> Uni-LSTM
from utils.loss_kd_aux import (
    distilhubert_feature_loss,   # L_DIS
    attention_relation_kl,       # L_KLD (Q/Q + K/K + V/V)
    apc_loss                     # L_APC
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")

# ================ 실험 설정 ================
experiment = 1
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

    # KD Aux 기본값 주입 (config_jh.yaml에 없을 때 대비)
    kd_cfg = config.get("kd_aux", {})
    kd_enabled   = bool(kd_cfg.get("enabled", True))
    kd_layers    = kd_cfg.get("layers", [0, 1, 2, 3])   # dpt_temporal에서 우리가 잡은 4개 temporal 지점
    kd_N         = int(kd_cfg.get("N", 2))              # APC 미래 스텝
    kd_alpha     = float(kd_cfg.get("alpha", 1e-2))
    kd_beta      = float(kd_cfg.get("beta", 5e-4))
    kd_gamma     = float(kd_cfg.get("gamma", 5e-3))
    kd_lambda    = float(kd_cfg.get("lambda_kd", 1.0))
    kd_attn_eps  = float(kd_cfg.get("attn_eps", 1e-8))
    kd_pool      = kd_cfg.get("feature_pool", "mean")

    if args.epochs is not None:
        num_epochs = int(args.epochs)

    # W&B
    load_dotenv(dotenv_path=".env")
    wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
    run = wandb.init(project="ablation_reverse", config=hyper_params, name=f"experiment_{experiment}")

    # 데이터
    kitti_path = "/home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI"
    rgb_clips, depth_clips = get_data_list(root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN)
    kitti_train = KITTIVideoDataset(rgb_paths=rgb_clips, depth_paths=depth_clips, resize_size=518, split="train")
    kitti_train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 모델 (단일 GPU)
    student = VideoDepthStudent(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)
    # --- Teacher (offline, bidirectional) ---
    teacher = VideoDepthTeacher(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- KD용 레이어별 channel 정의 (우리 dpt_temporal 인덱스 0..3에 대응)
    #   idx 0: layer_3 temporal, 1: layer_4 temporal, 2: path_4 temporal, 3: path_3 temporal
    TEACHER_DIMS = [192,  384,   64,   64]   # teacher out_channels[2], out_channels[3], features, features
    STUDENT_DIMS = [192,  384,   64,   64]  # student out_channels[2], out_channels[3], features, features  (현재 student 설정 기준)

    # --- AuxBlocks (Student 각 레이어 feat -> C_T로 proj)
    aux_blocks = nn.ModuleDict({
        str(i): AuxBlock(
            c_in=STUDENT_DIMS[i],
            c_teacher=TEACHER_DIMS[i],
            nhead=8,
            dropout=0.0,
            return_attn=False,
            return_qkv=True,            # Aux 내부 1L Transformer의 Q/K/V 수집
            rnn_type="lstm",            # 또는 "mamba" 실험 가능
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
        ).to(device) for i in kd_layers
    })

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
    aux_params     = [p for p in aux_blocks.parameters() if p.requires_grad]
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

        if "aux_state_dict" in ckpt:
            try: aux_blocks.load_state_dict(ckpt["aux_state_dict"])
            except Exception as e: logger.warning(f"Aux state load skipped: {e}")

        # 3) 베스트 스코어 & 스타트 에폭
        if "best_val_delta1" in ckpt:
            try: best_delta1 = float(ckpt["best_val_delta1"])
            except: pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        logger.info(f"▶ Resumed from '{args.resume_from}' | start_epoch={start_epoch} / target_epochs={num_epochs} | best_delta1={best_delta1:.4f}")

    wandb.watch(student, log="all")
    wandb.watch(aux_blocks, log="all")
    best_delta1 = 0.0
    best_epoch  = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")

    if not args.test:
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
        student.train()
        epoch_loss = epoch_frames = 0.0
        epoch_ssi = epoch_tgm = epoch_kd = 0.0
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

            # ====== (KD) Full-clip forward ======
            if kd_enabled:
                with torch.no_grad():
                    t_out = teacher(
                        x, return_intermediates=True, return_qkv=True, feature_pool=kd_pool
                    )
                    t_inter = t_out["intermediates"]  # dict[idx]-> {"feat":[B,T,Ct], "qkv":{"Q","K","V":[B,A,T,Dh]}}
                # student도 '학습용 풀클립 경로'(캐시X)로 추론해 intermediates 수집
                s_out = student(
                    x, return_intermediates=True, return_qkv=True, feature_pool=kd_pool
                )
                s_inter = s_out["intermediates"]     # dict[idx]-> {"feat":[B,T,Cs], "qkv":...}

                # frame 유효 마스크 (KITTI depth에는 invalid가 있음) : [B,T] in {0,1}
                with torch.no_grad():
                    # y: [B,T,1,H,W], 값>1e-3 & <80.0 유효 (train_helper.get_mask 기준)
                    valid_mask_bt = []
                    for tmask in range(T):
                        m = get_mask(y[:, tmask:tmask+1], 1e-3, 80.0).to(device)  # [B,1,1,H,W]
                        # spatial OR/mean → frame-wise 마스크 (하나라도 유효 픽셀이 있으면 1)
                        v = (m.squeeze(2).squeeze(1).any(dim=(1,2))).float()      # [B]
                        valid_mask_bt.append(v)
                    valid_mask_bt = torch.stack(valid_mask_bt, dim=1)  # [B,T]

                # 레이어별 KD 계산
                kd_terms = {"dis":0.0, "kld":0.0, "apc":0.0}
                num_used = 0
                for li in kd_layers:
                    li = int(li)
                    h_feat = t_inter[li]["feat"]                    # [B,T,Ct]
                    t_qkv  = t_inter[li]["qkv"]                    # dict or None
                    s_feat = s_inter[li]["feat"]                   # [B,T,Cs]

                    # Aux: s_feat -> (z, r, qkv_aux)
                    z_feat, r_feat, qkv_aux = aux_blocks[str(li)](
                        s_feat, hole_mask_N=kd_N
                    )  # z,r:[B,T,Ct], qkv_aux: dict with [B,A,T,Dh] or None

                    # (A) Feature
                    l_dis = distilhubert_feature_loss(h_feat, z_feat, mask=valid_mask_bt)
                    # (B) Attn KL (Q/Q + K/K + V/V) — teacher/student 둘 다 qkv가 필요
                    if (t_qkv is not None) and (qkv_aux is not None):
                        l_kld = attention_relation_kl(t_qkv, qkv_aux, mask=valid_mask_bt, eps=kd_attn_eps)
                    else:
                        l_kld = h_feat.new_tensor(0.0)
                    # (C) APC
                    l_apc = apc_loss(h_feat, r_feat, N=kd_N, mask=valid_mask_bt)

                    kd_terms["dis"] += l_dis
                    kd_terms["kld"] += l_kld
                    kd_terms["apc"] += l_apc
                    num_used += 1

                if num_used > 0:
                    for k in kd_terms:
                        kd_terms[k] = kd_terms[k] / num_used
                L_kd = kd_alpha * kd_terms["dis"] + kd_beta * kd_terms["kld"] + kd_gamma * kd_terms["apc"]
            else:
                L_kd = torch.tensor(0.0, device=device)
                kd_terms = {"dis":0.0, "kld":0.0, "apc":0.0}
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

                    # KD는 배치 내 클립 기준 한 번만 더함(여기서는 t==0에만 더함)
                    if kd_enabled and t == 0:
                        loss = loss + kd_lambda * L_kd

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
                epoch_kd     += L_kd.item()      * B_eff

                frame_pbar.set_postfix({
                    'SSI': f'{epoch_ssi/ max(1, epoch_frames):.4f}',
                    'TGM': f'{epoch_tgm/ max(1, epoch_frames):.4f}',
                    'L_KD': f'{epoch_kd/ max(1, epoch_frames):.4f}' if kd_enabled else '0.0000'
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
            "kd/dis": float(kd_terms["dis"]) if kd_enabled else 0.0,
            "kd/kld": float(kd_terms["kld"]) if kd_enabled else 0.0,
            "kd/apc": float(kd_terms["apc"]) if kd_enabled else 0.0,
            "kd/total": float(L_kd.item()) if kd_enabled else 0.0,
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
            "aux_state_dict": aux_blocks.state_dict(),
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
    parser.add_argument("--val_json_file",    type=str, default="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json")
    parser.add_argument("--val_infer_dir",    type=str, default="benchmark/output/scannet_stream_valmini")
    parser.add_argument("--val_dataset_key",  type=str, default="scannet")
    parser.add_argument("--val_dataset_tag",  type=str, default="scannet_500")
    parser.add_argument("--val_scenes",       type=int, default=2)
    parser.add_argument("--resume_from", type=str, default="", help="Path to latest/best checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs (e.g., 60)")
    parser.add_argument("--test", action="store_true", help="Only run validation")
    args = parser.parse_args()
    train(args)
