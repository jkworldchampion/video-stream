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
    attention_relation_kl,       # L_KLD (Q/Q + K/K + V/V)
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")

# ================ 실험 설정 ================
experiment = 7
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
    # DIS, APC 제거 - KLD만 사용
    kd_beta      = float(kd_cfg.get("beta", 5e-4))
    kd_lambda    = float(kd_cfg.get("lambda_kd", 1.0))
    kd_attn_eps  = float(kd_cfg.get("attn_eps", 1e-8))

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
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- KD용 레이어별 channel 정의 (우리 dpt_temporal 인덱스 0..3에 대응)
    #   idx 0: layer_3 temporal, 1: layer_4 temporal, 2: path_4 temporal, 3: path_3 temporal
    TEACHER_DIMS = [192,  384,   64,   64]   # teacher out_channels[2], out_channels[3], features, features
    STUDENT_DIMS = [192,  384,   64,   64]  # student out_channels[2], out_channels[3], features, features

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
    optimizer = torch.optim.AdamW(
        list(student_params) + list(aux_params),
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

        if "aux_state_dict" in ckpt:
            try: aux_blocks.load_state_dict(ckpt["aux_state_dict"])
            except Exception as e: logger.warning(f"Aux state load skipped: {e}")

        # 3) 베스트 스코어 & 스타트 에폭
        if "best_val_loss" in ckpt:
            try: best_val_loss = float(ckpt["best_val_loss"])
            except: pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        logger.info(f"▶ Resumed from '{args.resume_from}' | start_epoch={start_epoch} / target_epochs={num_epochs} | best_val_loss={best_val_loss:.4f}")

    wandb.watch(student, log="all")
    wandb.watch(aux_blocks, log="all")
    best_val_loss = float('inf')  # KITTI val loss 기준으로 best 모델 선택
    best_epoch  = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")
    
    # ================ 설정 출력 ================
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
        # DIS, APC 제거 - KLD만 사용
        logger.info(f"  KD Beta (KLD): {kd_beta}")
        logger.info(f"  KD Lambda (Total Weight): {kd_lambda}")
        logger.info(f"  KD Window: {kd_window}")
        logger.info(f"  KD Stride: {kd_stride}")
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
    if not args.test:
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
        aux_blocks.train()
        epoch_loss = epoch_frames = 0.0
        epoch_ssi = epoch_tgm = 0.0

        # KD 누적(스텝 평균용) - DIS, APC 제거
        epoch_kd_total = 0.0
        epoch_kd_kld   = 0.0
        kd_steps = 0

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
            kd_executed = False
            layer_buffers = {
                int(li): {
                    # r 제거 (APC에서만 사용)
                    "q": collections.deque(maxlen=kd_window),
                    "k": collections.deque(maxlen=kd_window),
                    "v": collections.deque(maxlen=kd_window),
                    "cache": None,
                }
                for li in kd_layers
            } if kd_enabled else {}

            frame_pbar = tqdm(range(T), desc=f"Batch {batch_idx+1} - Frames", leave=False, disable=T < 10)
            for t in frame_pbar:
                x_t = x[:, t:t+1]                              # [B,1,3,H,W]
                mask_t = get_mask(y[:, t:t+1], 1e-3, 80.0).to(device)
                layer_step_outputs = {}

                # === Student: 스트리밍 1-step (encoder features 캐시) ===
                with autocast(enabled=torch.cuda.is_available()):
                    # pred_t_raw, cache = model_stream_step(student, x_t, cache)
                    pred_t_raw, cache, inter_t, student_enc_feats = model_stream_step(
                        student, x_t, cache,
                        collect_inter=True,        # ← 내부 temporal features(intermediates)를 inter_t로 반환
                        collect_qkv=False,         # Q/K/V는 AuxBlock에서 생성
                        feature_pool='mean',       # mean pooling
                        return_encoder_feats=True, # ← encoder features 반환
                    )
                    pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)
                    
                    # Encoder features detach (grad 차단, Teacher 재사용용)
                    student_enc_feats_detached = [
                        tuple(f_i.detach() if f_i is not None else None for f_i in f_tuple)
                        for f_tuple in student_enc_feats
                    ]

                    if kd_enabled and inter_t is not None:
                        for li in kd_layers:
                            li = int(li)
                            info = inter_t.get(li)
                            if info is None:
                                continue
                            feat_one = info.get("feat_one")
                            if feat_one is None:
                                continue

                            state = layer_buffers[li]
                            stream_cache = state["cache"]
                            # r_t 제거 (APC에서만 사용)
                            qkv_t, stream_cache = aux_blocks[str(li)].forward_stream(
                                feat_one,
                                cache=stream_cache,
                                max_len=kd_window,
                            )
                            state["cache"] = stream_cache

                            if aux_blocks[str(li)].return_qkv and qkv_t is not None:
                                state["q"].append(qkv_t["Q"].detach())
                                state["k"].append(qkv_t["K"].detach())
                                state["v"].append(qkv_t["V"].detach())
                            else:
                                state["q"].append(None)
                                state["k"].append(None)
                                state["v"].append(None)

                            layer_step_outputs[li] = {
                                "qkv": qkv_t,
                            }

                    # ----- Scale-Shift & Losses -----
                    gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6)).squeeze(2)  # [B,1,H,W]
                    if pred_t_raw.shape[0] != gt_disp_t.shape[0]:
                        pred_t_raw = pred_t_raw[:1]

                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)

                    pred_t_aligned_disp = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                    pred_t_aligned_depth = 1.0 / (pred_t_aligned_disp.clamp(min=1e-6))

                    disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)  # [B,1,H,W]
                    ssi_loss_t = loss_ssi(pred_t_aligned_disp.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))

                    # ----- Temporal Geometric Consistency Loss -----
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

                    # ----- Sliding Teacher KD at frame t -----
                    L_kd_t = pred_t_raw.new_tensor(0.0)
                    kd_terms_step = {"kld": 0.0}  # DIS, APC 제거 - KLD만 사용
                    W_eff = min(kd_window, t + 1)
                    t0 = t - W_eff + 1
                    x_win = x[:, t0:t+1]  # [B, W_eff, 3, H, W]
                    num_used = 0

                    if kd_enabled and ((t % kd_stride) == 0) and (W_eff >= 1):
                        kd_executed = True

                        # frame-wise valid mask들 준비
                        with torch.no_grad():
                            frame_valid_list = []
                            for tt in range(W_eff):
                                m = get_mask(y[:, t0+tt:t0+tt+1], 1e-3, 80.0).to(device)    # [B,1,1,H,W]
                                v = (m.squeeze(2).squeeze(1).any(dim=(1,2))).float()        # [B]
                                frame_valid_list.append(v)
                            frame_valid = torch.stack(frame_valid_list, dim=1)               # [B, W_eff]

                            # DIS/KLD에서 '마지막 프레임'만
                            mask_last = torch.zeros_like(frame_valid)
                            mask_last[:, -1] = frame_valid[:, -1]

                        # 3) Teacher: encoder feature 재사용 (과거 프레임만 계산, 현재는 student 재사용)
                        with torch.no_grad():
                            if W_eff > 1:
                                # 과거 프레임 (t0~t-1) encoder
                                x_past = x_win[:, :-1, ...]  # [B, W_eff-1, 3, H, W]
                                past_feats = teacher.pretrained.get_intermediate_layers(
                                    x_past.flatten(0, 1),
                                    teacher.intermediate_layer_idx[teacher.encoder],
                                    return_class_token=True
                                )
                            else:
                                past_feats = None

                            # Combine past + current (student encoder 재사용)
                            if past_feats is not None:
                                combined_feats = []
                                for pf_tuple, sf_tuple in zip(past_feats, student_enc_feats_detached):
                                    # pf_tuple: (tokens[B*(W-1),N,C], cls[B*(W-1),C]) or just (tokens,)
                                    # sf_tuple: (tokens[B*1,N,C], cls[B*1,C]) or just (tokens,)
                                    tokens_combined = torch.cat([pf_tuple[0], sf_tuple[0]], dim=0)
                                    if len(pf_tuple) > 1 and pf_tuple[1] is not None:
                                        cls_combined = torch.cat([pf_tuple[1], sf_tuple[1]], dim=0)
                                        combined_feats.append((tokens_combined, cls_combined))
                                    else:
                                        combined_feats.append((tokens_combined,))
                            else:
                                # W_eff==1: 현재 프레임만
                                combined_feats = student_enc_feats_detached

                            # Teacher depth head로 intermediate/qkv 추출
                            _, _, t_intermediates = teacher.head(
                                combined_feats,
                                x_win.shape[-2] // 14,  # patch_h
                                x_win.shape[-1] // 14,  # patch_w
                                W_eff,                   # frame_length
                                cached_hidden_state_list=None,
                                return_intermediates=True,
                                return_qkv=True,
                                feature_pool='mean',
                            )
                            t_out = {"intermediates": t_intermediates}

                        # 4) 레이어별 KD
                        for li in kd_layers:
                            li = int(li)
                            state = layer_buffers.get(li) if kd_enabled else None
                            if state is None or len(state["q"]) == 0:  # q 버퍼로 체크
                                continue

                            t_qkv_seq  = t_out["intermediates"][li].get("qkv")  # dict with [B,A,T,Dh]

                            latest_vals = layer_step_outputs.get(li, {}) if li in layer_step_outputs else {}

                            qkv_aux_seq = None
                            if aux_blocks[str(li)].return_qkv:
                                q_entries = list(state["q"])[-W_eff:]
                                k_entries = list(state["k"])[-W_eff:]
                                v_entries = list(state["v"])[-W_eff:]
                                if len(q_entries) >= W_eff and len(k_entries) >= W_eff and len(v_entries) >= W_eff:
                                    latest_qkv = latest_vals.get("qkv") if latest_vals else None
                                    if latest_qkv is not None:
                                        q_entries[-1] = latest_qkv["Q"]
                                        k_entries[-1] = latest_qkv["K"]
                                        v_entries[-1] = latest_qkv["V"]

                                    if all(q is not None for q in q_entries) and all(k is not None for k in k_entries) and all(v is not None for v in v_entries):
                                        q_seq = torch.cat(q_entries, dim=2)
                                        k_seq = torch.cat(k_entries, dim=2)
                                        v_seq = torch.cat(v_entries, dim=2)
                                        qkv_aux_seq = {"Q": q_seq, "K": k_seq, "V": v_seq}
                            
                            l_kld = attention_relation_kl(
                                t_qkv_seq, qkv_aux_seq,
                                q_mask=mask_last,
                                k_mask=frame_valid,
                                eps=kd_attn_eps
                            ) if (t_qkv_seq is not None and qkv_aux_seq is not None) else x.new_tensor(0.0)

                            kd_terms_step["kld"] += l_kld
                            num_used += 1

                        if num_used > 0:
                            for k in kd_terms_step:
                                kd_terms_step[k] = kd_terms_step[k] / num_used

                        # DIS, APC 제거 - KLD만 사용
                        L_kd_t = kd_beta * kd_terms_step["kld"]
                        # print(f"  [KD @ frame {t}] L_KLD={kd_terms_step['kld']:.4f}  →  L_KD={L_kd_t.item():.4f}")

                    # ---- 최종 loss (KD 꺼져 있어도 depth 손실은 유지!)
                    loss = ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss + (kd_lambda * L_kd_t if kd_enabled else 0.0)

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

                # KD가 실제 실행된 스텝만 에폭 누적
                if kd_executed:
                    epoch_kd_total += float(L_kd_t.item())
                    epoch_kd_kld   += float(kd_terms_step["kld"])
                    kd_steps += 1

                frame_pbar.set_postfix({
                    'wSSI': f'{epoch_ssi / max(1, epoch_frames) * ratio_ssi:.4f}',
                    'wTGM': f'{epoch_tgm / max(1, epoch_frames) * ratio_tgm:.4f}',
                    'wkld': f'{(epoch_kd_kld / max(1, kd_steps) * kd_beta):.4e}' if kd_enabled else '0.0000',
                })
            frame_pbar.close()
        batch_pbar.close()

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
        wandb_log_dict = {
            "train/loss": epoch_loss / max(1, len(kitti_train_loader)),
            "train/ssi":  epoch_ssi  / max(1, epoch_frames),
            "train/tgm":  epoch_tgm  / max(1, epoch_frames),

            # KD는 스텝 평균 (Ablation)
            "train/kd_total": (epoch_kd_total / max(1, kd_steps)) if kd_enabled else 0.0,
            'train/kd_steps': kd_steps,

            # KITTI val (for best model selection)
            "val_kitti/loss": kitti_val_loss,
            "val_kitti/absrel": kitti_val_absrel,
            "val_kitti/delta1": kitti_val_delta1,
            
            # ScanNet val (for monitoring - kept as val_real for comparison with previous experiments)
            "val_real/absrel": scannet_absrel,
            "val_real/rmse":   scannet_rmse,
            "val_real/delta1": scannet_delta1,
            
            "epoch": epoch,
        }
        
        # Add individual KD components only if enabled (DIS, APC 제거)
        if kd_enabled:
            # DIS, APC는 제거되었으므로 로깅하지 않음
            wandb_log_dict["train/kd_kld"] = (epoch_kd_kld / max(1, kd_steps))
            wandb_log_dict["train/kd_kld_weighted"] = (epoch_kd_kld / max(1, kd_steps)) * kd_beta
        
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
                "aux_state_dict": aux_blocks.state_dict(),
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
            "aux_state_dict": aux_blocks.state_dict(),
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