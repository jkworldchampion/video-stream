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
import collections

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
experiment = 10
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
    kd_alpha     = float(kd_cfg.get("alpha", 1e-2))
    kd_beta      = float(kd_cfg.get("beta", 5e-4))
    kd_gamma     = float(kd_cfg.get("gamma", 5e-3))
    kd_lambda    = float(kd_cfg.get("lambda_kd", 1.0))
    kd_attn_eps  = float(kd_cfg.get("attn_eps", 1e-8))
    kd_pool      = kd_cfg.get("feature_pool", "mean")

    # 추가: 슬라이딩 KD 창 길이/보폭
    kd_window = int(kd_cfg.get("window", CLIP_LEN))  # 보통 32, 경험상 16으로 하는게 젤 나음
    kd_stride = int(kd_cfg.get("stride", 1))         # 매 프레임 KD면 1, 비용 줄이려면 2/4

    if args.epochs is not None:
        num_epochs = int(args.epochs)

    raw_scene_indices = getattr(args, "val_scene_indices", None)
    scene_indices = None
    if raw_scene_indices:
        scene_indices = [int(idx.strip()) for idx in raw_scene_indices.split(",") if idx.strip()]
        scene_indices = sorted(set(scene_indices))
    if scene_indices:
        logger.info(f"Validation scene indices: {scene_indices}")

    if not args.test:
        # W&B
        wandb_config = config.get("wandb", {})
        wandb_entity = wandb_config.get("entity", "depth-finder")  # 기본값: depth-finder
        wandb_project = wandb_config.get("project", "3kd_checking")
        
        load_dotenv(dotenv_path=".env")
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            config=hyper_params,
            name=f"experiment_{experiment}"
        )

    # 데이터
    kitti_path = "/home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI"
    
    # Train set
    rgb_clips, depth_clips = get_data_list(root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN)
    kitti_train = KITTIVideoDataset(rgb_paths=rgb_clips, depth_paths=depth_clips, resize_size=518, split="train")
    kitti_train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Validation set (KITTI) - returns 5 values for val split
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
        split="val"
    )
    kitti_val_loader = DataLoader(kitti_val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


    # 모델 (단일 GPU)
    student = VideoDepthStudent(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)
    teacher = VideoDepthTeacher(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)
    
    # ✅ FIX 1: Teacher에도 pretrained weight 로드
    if args.pretrained_ckpt:
        logger.info(f"Loading Teacher weights from {args.pretrained_ckpt}")
        teacher_sd = torch.load(args.pretrained_ckpt, map_location="cpu")
        teacher.load_state_dict(teacher_sd, strict=True)
        logger.info("✅ Teacher pretrained weights loaded successfully!")
        
        logger.info(f"Loading Student weights from {args.pretrained_ckpt}")
        student_sd = torch.load(args.pretrained_ckpt, map_location="cpu")
        student.load_state_dict(student_sd, strict=True)
        logger.info("✅ Student pretrained weights loaded successfully!")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- KD용 레이어별 channel 정의 (우리 dpt_temporal 인덱스 0..3에 대응)
    #   idx 0: layer_3 temporal, 1: layer_4 temporal, 2: path_4 temporal, 3: path_3 temporal
    TEACHER_DIMS = [192,  384,   64,   64]   # teacher out_channels[2], out_channels[3], features, features
    STUDENT_DIMS = [192,  384,   64,   64]  # student out_channels[2], out_channels[3], features, features

    # Freeze 정책
    for p in student.pretrained.parameters(): p.requires_grad = False
    for p in student.head.parameters(): p.requires_grad = True
    student.train()

    # Optim/Sch
    student_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        list(student_params),
        lr=lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Loss
    loss_tgm = LossTGMVector(diff_depth_th=0.05)
    loss_ssi = Loss_ssi_basic()
    scaler = GradScaler()

    # ----- Resume (optional) -----
    start_epoch = 0
    best_val_loss = float('inf')  # KITTI val loss 기준으로 변경 (최소화)

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
        if "best_val_loss" in ckpt:
            try: best_val_loss = float(ckpt["best_val_loss"])
            except: pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        logger.info(f"▶ Resumed from '{args.resume_from}' | start_epoch={start_epoch} / target_epochs={num_epochs} | best_val_loss={best_val_loss:.4f}")

    if not args.test:
        wandb.watch(student, log="all")
        best_val_loss = float('inf')  # KITTI val loss 기준으로 best 모델 선택
    else:
        best_val_loss = float('inf')  # KITTI val loss 기준으로 best 모델 선택
    best_epoch  = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")
    
    # ================ 설정 출력 ================
    if experiment >= 0:
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
        
        logger.info("--- Knowledge Distillation (KD) ---")
        logger.info(f"  KD Enabled: {kd_enabled}")
        if kd_enabled:
            logger.info(f"  KD Layers: {kd_layers}")
            logger.info(f"  KD Alpha (DIS): {kd_alpha}")
            logger.info(f"  KD Beta (KLD): {kd_beta}")
            logger.info(f"  KD Lambda (Total Weight): {kd_lambda}")
            logger.info(f"  KD Window: {kd_window}")
            logger.info(f"  KD Stride: {kd_stride}")
            logger.info(f"  Feature Pool: {kd_pool}")
            logger.info(f"  Attention Epsilon: {kd_attn_eps}")
        logger.info("")
        
        logger.info("--- Model Architecture ---")
        logger.info(f"  Student Encoder: {student.encoder}")
        logger.info(f"  Teacher Encoder: {teacher.encoder}")
        logger.info(f"  Features: 64")
        logger.info(f"  Out Channels: [48, 96, 192, 384]")
        logger.info(f"  Num Frames: {CLIP_LEN}")
        logger.info("")
        
        logger.info("--- Optimizer & Scheduler ---")
        logger.info(f"  Optimizer: AdamW")
        logger.info(f"  Weight Decay: 1e-4")
        logger.info(f"  Scheduler: CosineAnnealingLR")
        logger.info(f"  Scheduler Eta Min: 1e-6")
        logger.info("")
        
        logger.info("--- Data Configuration ---")
        logger.info(f"  KITTI Path: /home/work/juhwan/monocular_depth/Video-Depth-Anything/datasets/KITTI")
        logger.info(f"  Data Split: train")
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

    # --------------------- Training Mode (with validation) ---------------------
    if True:
        # ---- Init validation before training (epoch = -1) ----
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
            max_depth=80.0
        )
        
        kitti_val_loss = kitti_val_metrics['loss']
        kitti_val_absrel = kitti_val_metrics['absrel']
        kitti_val_delta1 = kitti_val_metrics['delta1']
        
        logger.info(f"[Init KITTI] loss={kitti_val_loss:.4f} | absrel={kitti_val_absrel:.4f} | delta1={kitti_val_delta1:.4f}")
        
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
            scene_indices=scene_indices,  # 명시적 scene 인덱스 전달
            fp32=True
        )
        
        # validate_with_infer_eval_subset는 {"avg": {...}, "per_scene": {...}} 형태로 반환
        avg_metrics = scannet_metrics.get("avg", {})
        scannet_absrel = float(avg_metrics.get("abs_relative_difference", float('nan')))
        scannet_rmse   = float(avg_metrics.get("rmse_linear", float('nan')))
        scannet_delta1 = float(avg_metrics.get("delta1_acc", float('nan')))
        
        logger.info(f"[Init ScanNet] absrel={scannet_absrel:.4f} | rmse={scannet_rmse:.4f} | delta1={scannet_delta1:.4f}")
        
        if not args.test:
            # W&B 로깅 (epoch=-1로 표기)
            wandb.log({
                "init/val_kitti_loss": kitti_val_loss,
                "init/val_kitti_absrel": kitti_val_absrel,
                "init/val_kitti_delta1": kitti_val_delta1,
                "init/val_real_absrel": scannet_absrel,
                "init/val_real_rmse": scannet_rmse,
                "init/val_real_delta1": scannet_delta1,
                "epoch": -1,
            })
        
        logger.info("=" * 60)
        logger.info("Initial validation completed! Starting training...")
        logger.info("=" * 60)
        
        # 베스트 기준을 KITTI val loss로 설정 (zero-shot ScanNet 성능 평가를 위함)
        best_val_loss = kitti_val_loss

    # --------------------- Training ---------------------
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False):
        student.train()
        epoch_loss = epoch_frames = 0.0
        epoch_ssi = epoch_tgm = 0.0

        # KD 누적(스텝 평균용)
        epoch_kd_dis   = 0.0
        epoch_kd_kld   = 0.0
        kd_steps = 0

        step_in_window = 0
        update_frequency = hyper_params.get("update_frequency", 6)

        batch_pbar = tqdm(enumerate(kitti_train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Batches", total=len(kitti_train_loader), leave=False)
        for batch_idx, (x, y) in batch_pbar:
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, T = x.shape[:2]

            # ✅ 스트리밍 상태 초기화
            m = student.module if hasattr(student, "module") else student
            if hasattr(m, "reset_streaming_state"):
                m.reset_streaming_state()

            cache = None
            prev_pred_raw = prev_mask = prev_y = None
            kd_executed = False
            # FIX 2: Encoder feature 캐싱 버퍼 추가
            encoder_feat_cache = collections.deque(maxlen=kd_window)

            frame_pbar = tqdm(range(T), desc=f"Batch {batch_idx+1} - Frames", leave=False, disable=T < 10)
            for t in frame_pbar:
                x_t = x[:, t:t+1]                              # [B,1,3,H,W]
                mask_t = get_mask(y[:, t:t+1], 1e-3, 80.0).to(device)
                layer_step_outputs = {}

                # === Student: 스트리밍 1-step (encoder features 캐시) ===
                # with autocast(enabled=torch.cuda.is_available()):
                #     pred_t_raw, cache, inter_t, student_enc_feats = model_stream_step(
                #         student, x_t,
                #         collect_inter=True,        # ← 내부 temporal features(intermediates)를 inter_t로 반환
                #         collect_qkv=True,         # Q/K/V는 AuxBlock에서 생성, [B, num_attnn_heads, Temporal_length, head_dim]
                #         accumulate_qkv=True,
                #         feature_pool=kd_pool,      # 'mean' 등
                #         return_encoder_feats=True, # ← encoder features 반환
                #     )
                #     pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)
                #     if args.test and t in [0, 10, 15]:
                #         print("\n" + "="*60)
                #         print(f"🔍 model_stream_step 전체 출력 확인 (Frame {t})")
                #         print("="*60)
                        
                #         # 1. pred_t_raw
                #         print(f"\n1️⃣  pred_t_raw:")
                #         print(f"   Shape: {pred_t_raw.shape}")
                #         print(f"   Min/Max: {pred_t_raw.min().item():.4f} / {pred_t_raw.max().item():.4f}")
                        
                #         # 2. cache
                #         print(f"\n2️⃣  cache:")
                #         if cache is not None:
                #             if isinstance(cache, list):
                #                 print(f"   Type: list, Length: {len(cache)}")
                #                 for idx, c in enumerate(cache):
                #                     if c is not None:
                #                         print(f"   cache[{idx}] shape: {c.shape}")
                #             elif isinstance(cache, torch.Tensor):
                #                 print(f"   Type: Tensor, Shape: {cache.shape}")
                #             else:
                #                 print(f"   Type: {type(cache)}")
                #         else:
                #             print(f"   None")
                        
                #         # 3. inter_t
                #         print(f"\n3️⃣  inter_t:")
                #         if inter_t is not None:
                #             print(f"   Keys: {list(inter_t.keys())}")
                #             for li, data in inter_t.items():
                #                 print(f"\n   Layer {li}:")
                                
                #                 feat_one = data.get('feat_one')
                #                 if feat_one is not None:
                #                     print(f"      feat_one shape: {feat_one.shape}")
                #                     print(f"      feat_one min/max: {feat_one.min().item():.4f} / {feat_one.max().item():.4f}")
                #                 else:
                #                     print(f"      feat_one: None")
                                
                #                 qkv = data.get('qkv')
                #                 if qkv is not None:
                #                     print(f"      ✅ QKV 존재!")
                #                     print(f"         Q shape: {qkv['Q'].shape}, range: [{qkv['Q'].min().item():.2f}, {qkv['Q'].max().item():.2f}]")
                #                     print(f"         K shape: {qkv['K'].shape}, range: [{qkv['K'].min().item():.2f}, {qkv['K'].max().item():.2f}]")
                #                     print(f"         V shape: {qkv['V'].shape}, range: [{qkv['V'].min().item():.2f}, {qkv['V'].max().item():.2f}]")
                #                 else:
                #                     print(f"      ❌ QKV: None")

                #                 # ✅ 여기 추가: qkv_history 확인
                #                 qkv_hist = data.get("qkv_history")
                #                 if qkv_hist is not None:
                #                     Qh = qkv_hist["Q"]
                #                     Kh = qkv_hist["K"]
                #                     Vh = qkv_hist["V"]
                #                     print(f"      📌 qkv_history Q shape: {Qh.shape}")
                #                     print(f"         (hist T = {Qh.size(2)})")
                #                 else:
                #                     print(f"      qkv_history: None")
                #         else:
                #             print(f"   None")
                        
                #         # 4. student_enc_feats
                #         print(f"\n4️⃣  student_enc_feats:")
                #         if student_enc_feats is not None:
                #             if isinstance(student_enc_feats, list):
                #                 print(f"   Type: list, Length: {len(student_enc_feats)}")
                #                 for idx, feat_tuple in enumerate(student_enc_feats):
                #                     if feat_tuple is not None:
                #                         if isinstance(feat_tuple, tuple):
                #                             print(f"   Layer {idx}: tuple of {len(feat_tuple)} tensors")
                #                             for i, f in enumerate(feat_tuple):
                #                                 if f is not None:
                #                                     print(f"      [{i}] shape: {f.shape}")
                #                         else:
                #                             print(f"   Layer {idx}: {feat_tuple.shape}")
                #             else:
                #                 print(f"   Type: {type(student_enc_feats)}")
                #         else:
                #             print(f"   None")
                        
                #         print("="*60 + "\n")

                #     # Encoder features detach (grad 차단, Teacher 재사용용)
                #     student_enc_feats_detached = [
                #         tuple(f_i.detach() if f_i is not None else None for f_i in f_tuple)
                #         for f_tuple in student_enc_feats
                #     ]
                #     encoder_feat_cache.append(student_enc_feats_detached)

                #     # ----- Scale-Shift & Losses -----
                #     gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6)).squeeze(2)  # [B,1,H,W]
                #     if pred_t_raw.shape[0] != gt_disp_t.shape[0]:
                #         pred_t_raw = pred_t_raw[:1]

                #     with torch.no_grad():
                #         a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)

                #     pred_t_aligned_disp = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                #     pred_t_aligned_depth = 1.0 / (pred_t_aligned_disp.clamp(min=1e-6))

                #     disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)  # [B,1,H,W]
                #     ssi_loss_t = loss_ssi(pred_t_aligned_disp.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))

                #     # ----- Temporal Geometric Consistency Loss -----
                #     if t > 0:
                #         prev_aligned_disp = (a_star.detach() * prev_pred_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                #         prev_aligned_depth = 1.0 / (prev_aligned_disp.clamp(min=1e-6))
                #         curr_aligned_depth = pred_t_aligned_depth
                #         pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)  # [B,2,H,W]
                #         y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)                       # [B,2,1,H,W]
                #         m_pair    = torch.cat([prev_mask, mask_t], dim=1)                         # [B,2,1,H,W]
                #         tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                #     else:
                #         tgm_loss  = pred_t_raw.new_tensor(0.0)

                #     # ----- Sliding Teacher KD at frame t -----
                #     L_kd_t = pred_t_raw.new_tensor(0.0)
                #     kd_dis_mean = pred_t_raw.new_tensor(0.0)
                #     kd_kld_mean = pred_t_raw.new_tensor(0.0)

                #     W_eff = min(kd_window, t + 1)
                #     t0 = t - W_eff + 1
                #     x_win = x[:, t0:t+1]  # [B, W_eff, 3, H, W]

                #     if kd_enabled and ((t % kd_stride) == 0) and (W_eff >= 1):
                #         kd_executed = True

                #         kd_dis_sum = pred_t_raw.new_tensor(0.0)
                #         kd_kld_sum = pred_t_raw.new_tensor(0.0)
                #         num_used = 0

                #         # 1) 유효 프레임 마스크
                #         with torch.no_grad():
                #             frame_valid_list = []
                #             for tt in range(W_eff):
                #                 m = get_mask(y[:, t0+tt:t0+tt+1], 1e-3, 80.0).to(device)  # [B,1,1,H,W]
                #                 v = (m.squeeze(2).squeeze(1).any(dim=(1,2))).float()      # [B]
                #                 frame_valid_list.append(v)
                #             frame_valid = torch.stack(frame_valid_list, dim=1)            # [B, W_eff]

                #             mask_last = torch.zeros_like(frame_valid)
                #             mask_last[:, -1] = frame_valid[:, -1]

                #         # 2) Teacher encoder feature 재사용
                #         with torch.no_grad():
                #             cached_feats = list(encoder_feat_cache)[-W_eff:]

                #             if len(cached_feats) < W_eff:
                #                 if W_eff > 1:
                #                     logger.warning(
                #                         f"[Batch {batch_idx}, Frame {t}] "
                #                         f"Insufficient cache: {len(cached_feats)} < {W_eff}, skipping KD"
                #                     )
                #             else:
                #                 combined_feats = []
                #                 num_layers = len(cached_feats[0])

                #                 for layer_idx in range(num_layers):
                #                     layer_feats = [frame_feats[layer_idx] for frame_feats in cached_feats]

                #                     tokens_list = [f[0] for f in layer_feats]
                #                     tokens_concat = torch.cat(tokens_list, dim=0)

                #                     if len(layer_feats[0]) > 1 and layer_feats[0][1] is not None:
                #                         cls_list = [f[1] for f in layer_feats]
                #                         cls_concat = torch.cat(cls_list, dim=0)
                #                         combined_feats.append((tokens_concat, cls_concat))
                #                     else:
                #                         combined_feats.append((tokens_concat,))

                #                 _, _, t_intermediates = teacher.head(
                #                     combined_feats,
                #                     x_win.shape[-2] // 14,
                #                     x_win.shape[-1] // 14,
                #                     W_eff,
                #                     cached_hidden_state_list=None,
                #                     return_intermediates=True,
                #                     return_qkv=True,
                #                     feature_pool=kd_pool,
                #                 )
                #                 t_out = {"intermediates": t_intermediates}

                #                 # 3) 레이어별 KD (DIS + KLD)
                #                 for li in kd_layers:
                #                     li = int(li)
                #                     if li not in t_out["intermediates"]:
                #                         continue
                #                     if inter_t is None or (li not in inter_t):
                #                         continue

                #                     t_layer = t_out["intermediates"][li]
                #                     s_layer = inter_t[li]

                #                     # (a) DIS
                #                     h_feat_seq = t_layer["feat"]          # [B, W_eff, Ct]
                #                     h_last = h_feat_seq[:, -1, :]         # [B, Ct]

                #                     s_feat_last = s_layer["feat_one"]     # [B,1,Cs]
                #                     s_feat_last = s_feat_last.squeeze(1)  # [B, Cs]

                #                     l_dis = distilhubert_feature_loss(
                #                         h_last.unsqueeze(1),
                #                         s_feat_last.unsqueeze(1),
                #                         mask=mask_last,
                #                     )

                #                     # (b) KLD
                #                     t_qkv_seq = t_layer.get("qkv")        # teacher {Q,K,V}
                #                     hist = s_layer.get("qkv_history")     # student {Q,K,V}

                #                     qkv_aux_seq = None
                #                     if (t_qkv_seq is not None) and (hist is not None):
                #                         Q_hist = hist["Q"]                # [B, A, T_hist, Dh]
                #                         K_hist = hist["K"]
                #                         V_hist = hist["V"]
                #                         if Q_hist.size(2) >= W_eff:
                #                             q_seq = Q_hist[:, :, -W_eff:, :]
                #                             k_seq = K_hist[:, :, -W_eff:, :]
                #                             v_seq = V_hist[:, :, -W_eff:, :]
                #                             qkv_aux_seq = {"Q": q_seq, "K": k_seq, "V": v_seq}

                #                     if qkv_aux_seq is not None:
                #                         l_kld = attention_relation_kl(
                #                             t_qkv_seq,
                #                             qkv_aux_seq,
                #                             q_mask=mask_last,
                #                             k_mask=frame_valid,
                #                             eps=kd_attn_eps,
                #                         )
                #                     else:
                #                         l_kld = h_feat_seq.new_tensor(0.0)

                #                     kd_dis_sum += l_dis
                #                     kd_kld_sum += l_kld
                #                     num_used += 1

                #         # 4) 레이어 평균 + L_kd_t
                #         if num_used > 0:
                #             kd_dis_mean = kd_dis_sum / num_used
                #             kd_kld_mean = kd_kld_sum / num_used
                #             L_kd_t = kd_alpha * kd_dis_mean + kd_beta * kd_kld_mean
                #         else:
                #             kd_dis_mean = kd_kld_mean = pred_t_raw.new_tensor(0.0)
                #             L_kd_t = pred_t_raw.new_tensor(0.0)

                #     # ---- 최종 loss (KD 꺼져 있어도 depth 손실은 유지!)
                #     loss = ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss + (kd_lambda * L_kd_t if kd_enabled else 0.0)

                with autocast(enabled=torch.cuda.is_available()):
                    # ★ inter_t, qkv, encoder feats 전부 안 받음
                    pred_t_raw, cache, _, _ = model_stream_step(
                        student, x_t, cache,
                        collect_inter=False,
                        collect_qkv=False,
                        accumulate_qkv=False,      # ★ history 자체를 끔
                        feature_pool=kd_pool,
                        return_encoder_feats=False,
                    )
                    pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)

                    # ----- Scale-Shift & Losses -----
                    gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6)).squeeze(2)
                    if pred_t_raw.shape[0] != gt_disp_t.shape[0]:
                        pred_t_raw = pred_t_raw[:1]

                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)

                    pred_t_aligned_disp = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                    pred_t_aligned_depth = 1.0 / (pred_t_aligned_disp.clamp(min=1e-6))

                    disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)
                    ssi_loss_t = loss_ssi(pred_t_aligned_disp.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))

                    if t > 0:
                        prev_aligned_disp = (a_star.detach() * prev_pred_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                        prev_aligned_depth = 1.0 / (prev_aligned_disp.clamp(min=1e-6))
                        curr_aligned_depth = pred_t_aligned_depth
                        pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)
                        y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)
                        m_pair    = torch.cat([prev_mask, mask_t], dim=1)
                        tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                    else:
                        tgm_loss  = pred_t_raw.new_tensor(0.0)

                    # ★ KD 완전 제거
                    loss = ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss
                # 1) scale / backward (그래프는 여기서 바로 사용되고 해제됨)
                scaled_loss = loss / update_frequency
                scaler.scale(scaled_loss).backward()

                # 2) 통계용 누적 (logging만)
                epoch_loss += loss.item()

                # 3) step_in_window 업데이트
                step_in_window += 1
                if step_in_window == update_frequency:
                    # optimizer update
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
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

                # KD가 실제 실행된 스텝만 에폭 누적
                if kd_executed:
                    epoch_kd_dis   += float(kd_dis_mean.item())
                    epoch_kd_kld   += float(kd_kld_mean.item())
                    kd_steps += 1

                frame_pbar.set_postfix({
                    'wSSI': f'{epoch_ssi / max(1, epoch_frames) * ratio_ssi:.4f}',
                    'wTGM': f'{epoch_tgm / max(1, epoch_frames) * ratio_tgm:.4f}',
                    'wdis': f'{(epoch_kd_dis / max(1, kd_steps) * kd_alpha):.4e}' if kd_enabled else '0.0000',
                    'wkld': f'{(epoch_kd_kld / max(1, kd_steps) * kd_beta):.4e}' if kd_enabled else '0.0000',
                })
            frame_pbar.close()
        batch_pbar.close()
        # 남은 gradient step 처리 (프레임 수가 update_frequency의 배수가 아닐 때)
        if step_in_window > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


        # --- KITTI Validation (for hyperparameter tuning) ---
        kitti_val_metrics = validate_kitti_streaming(
            model=student,
            val_loader=kitti_val_loader,
            device=device,
            loss_ssi_fn=loss_ssi,
            loss_tgm_fn=loss_tgm,
            ratio_ssi=ratio_ssi,
            ratio_tgm=ratio_tgm,
            min_depth=1e-3,
            max_depth=80.0
        )
        
        kitti_val_loss = kitti_val_metrics['loss']
        kitti_val_absrel = kitti_val_metrics['absrel']
        kitti_val_delta1 = kitti_val_metrics['delta1']
        
        # --- ScanNet Validation (for monitoring target domain) ---
        scannet_metrics = validate_with_infer_eval_subset(
            model=student,
            json_file=args.val_json_file,
            infer_path=args.val_infer_dir,
            dataset=args.val_dataset_key,
            dataset_eval_tag=args.val_dataset_tag,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            input_size=518,
            scenes_to_eval=args.val_scenes,
            scene_indices=scene_indices,  # 명시적 scene 인덱스 전달
            fp32=True
        )

        # validate_with_infer_eval_subset는 {"avg": {...}, "per_scene": {...}} 형태로 반환
        avg_metrics = scannet_metrics.get("avg", {})
        scannet_absrel = float(avg_metrics.get("abs_relative_difference", float('nan')))
        scannet_rmse   = float(avg_metrics.get("rmse_linear", float('nan')))
        scannet_delta1 = float(avg_metrics.get("delta1_acc", float('nan')))

        # 로깅 (Ablation: DIS, KLD, APC separately tracked)
        mean_kd_dis = (epoch_kd_dis / max(1, kd_steps)) if kd_enabled else 0.0
        mean_kd_kld = (epoch_kd_kld / max(1, kd_steps)) if kd_enabled else 0.0
        mean_kd_total = kd_alpha * mean_kd_dis + kd_beta * mean_kd_kld if kd_enabled else 0.0

        wandb_log_dict = {
            "train/loss": epoch_loss / max(1, len(kitti_train_loader)),
            "train/ssi":  epoch_ssi  / max(1, epoch_frames),
            "train/tgm":  epoch_tgm  / max(1, epoch_frames),

            "train/kd_total": mean_kd_total,
            "train/kd_steps": kd_steps,

            # KITTI / ScanNet ...
            "val_kitti/loss": kitti_val_loss,
            "val_kitti/absrel": kitti_val_absrel,
            "val_kitti/delta1": kitti_val_delta1,
            "val_real/absrel": scannet_absrel,
            "val_real/rmse":   scannet_rmse,
            "val_real/delta1": scannet_delta1,
            "epoch": epoch,
        }

        if kd_enabled:
            wandb_log_dict["train/kd_dis"] = mean_kd_dis
            wandb_log_dict["train/kd_dis_weighted"] = mean_kd_dis * kd_alpha

            wandb_log_dict["train/kd_kld"] = mean_kd_kld
            wandb_log_dict["train/kd_kld_weighted"] = mean_kd_kld * kd_beta
        
        if not args.test:
            wandb.log(wandb_log_dict)

        # best 저장 (KITTI val loss 기준 ↓ - zero-shot ScanNet 성능을 보기 위함)
        if kitti_val_loss < best_val_loss:
            best_val_loss = kitti_val_loss
            best_epoch  = epoch
            save_dict = {
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": hyper_params,
            }
            torch.save(save_dict, best_model_path)
            logger.info(f"🏆 Best model saved! Epoch {epoch}, KITTI val loss: {best_val_loss:.4f} | KITTI delta1: {kitti_val_delta1:.4f} | ScanNet delta1: {scannet_delta1:.4f}")

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
        logger.info(f"📁 Latest model saved | Epoch {epoch} | ScanNet delta1: {scannet_delta1:.4f} | KITTI delta1: {kitti_val_delta1:.4f}")

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
    parser.add_argument("--val_scene_indices", type=str, default="0,1", help="Comma-separated dataset indices for validation subset. Empty string disables explicit selection.")
    parser.add_argument("--resume_from", type=str, default="", help="Path to latest/best checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs (e.g., 60)")
    parser.add_argument("--test", action="store_true", help="Only run validation")
    args = parser.parse_args()
    train(args)