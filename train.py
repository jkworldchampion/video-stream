import os
import argparse
import logging
import random

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

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")

# ================ 실험 설정 ================
experiment = 31
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
def enable_attention_caching(m):
    for _, layer in m.named_modules():
        if hasattr(layer, 'enable_kd_caching'):
            layer.enable_kd_caching(True)     # 내부에서 need_exact_logs=True로 인식

def disable_attention_caching(m):
    for _, layer in m.named_modules():
        if hasattr(layer, 'enable_kd_caching'):
            layer.enable_kd_caching(False)

def collect_kd_caches(module, clear: bool = True):
    """
    returns: list[dict]
      dict keys (motion_module.py 쪽에서 저장):
        - 'context': [B, P, C]
        - 'attn'   : [B, P, K]   (head-avg, softmax)
        - 'K','V'  : [B, P, K, C]  (concat-head / 혹은 head-avg 임베딩)
        - 'selected_indices': [B, P, R] or None
        - 'tokens_per_frame': int(P)
    """
    outs = []
    for _, layer in module.named_modules():
        if hasattr(layer, 'get_cached_attention_output'):
            out = layer.get_cached_attention_output()
            if out is not None:
                outs.append(out)
                if clear and hasattr(layer, 'clear_attention_cache'):
                    layer.clear_attention_cache()
    return outs

def _safe_norm(x, dim=-1, eps=1e-8):
    return F.normalize(x, dim=dim, eps=eps)

def _gather_last_dim(x, idx):
    # x: [B,P,K,C] or [B,P,K], idx: [B,P,R] -> [B,P,R,C] or [B,P,R]
    if x.dim() == 4:
        B, P, K, C = x.shape
        R = idx.shape[-1]
        gidx = idx.unsqueeze(-1).expand(B, P, R, C)
    else:
        B, P, K = x.shape
        R = idx.shape[-1]
        gidx = idx
    return torch.gather(x, dim=2, index=gidx)

def compute_kd_losses_from_caches(
    t_cache: dict, s_cache: dict,
    top_r: int = 64, top_u: int = 32,
    w_attn_kl: float = 1.0, w_kv_cos: float = 1.0, w_ctx_cos: float = 0.0,
    eps: float = 1e-8
):
    """
    정확한 KD: Attn-KL + KV-Cosine + (선택) Context Cosine
    """
    t_attn = t_cache.get("attn", None)    # [B,P,Kt]
    s_attn = s_cache.get("attn", None)    # [B,P,Ks]
    Kt     = t_cache.get("K", None)       # [B,P,Kt,C]
    Vt     = t_cache.get("V", None)       # [B,P,Kt,C]
    Ks     = s_cache.get("K", None)       # [B,P,Ks,C]
    Vs     = s_cache.get("V", None)       # [B,P,Ks,C]
    t_ctx  = t_cache.get("context", None) # [B,P,C]
    s_ctx  = s_cache.get("context", None) # [B,P,C]

    # 안전한 device 선택
    base_dev = (t_ctx if t_ctx is not None else (t_attn if t_attn is not None else s_attn)).device
    total = torch.zeros((), device=base_dev)
    losses = {}

    # --- Attn-KL (교사 top-R ↔ 학생 NN 매칭) ---
    if (t_attn is not None) and (s_attn is not None) and (Kt is not None) and (Ks is not None):
        B, P, Kt_len = t_attn.shape
        R = min(top_r, Kt_len)

        t_top_vals, t_top_idx = torch.topk(t_attn, k=R, dim=-1, largest=True, sorted=True)  # [B,P,R], [B,P,R]
        Kt_sel = _gather_last_dim(Kt, t_top_idx)  # [B,P,R,C]

        Kt_sel_n = _safe_norm(Kt_sel, dim=-1)           # [B,P,R,C]
        Ks_n     = _safe_norm(Ks,     dim=-1)           # [B,P,Ks,C]
        bd = B * P
        Rf = Kt_sel_n.shape[2]
        Cf = Kt_sel_n.shape[-1]
        Kt_f = Kt_sel_n.reshape(bd, Rf, Cf)
        Ks_f = Ks_n.reshape(bd, Ks.shape[2], Cf)
        sims = torch.matmul(Kt_f, Ks_f.transpose(1, 2))  # [bd, R, Ks]
        s_match_idx = sims.argmax(dim=-1).view(B, P, Rf) # [B,P,R]

        ps_sel = _gather_last_dim(s_attn, s_match_idx)   # [B,P,R]
        pt_sel = t_top_vals                              # [B,P,R]

        pt = pt_sel / (pt_sel.sum(dim=-1, keepdim=True) + eps)
        ps = ps_sel / (ps_sel.sum(dim=-1, keepdim=True) + eps)
        attn_kl = (pt * (pt.add(eps).log() - ps.add(eps).log())).sum(dim=-1).mean()
        total += w_attn_kl * attn_kl
        losses['attn_kl'] = attn_kl

    # --- KV Cosine (교사 top-U ↔ 학생 NN 매칭) ---
    if (Kt is not None) and (Ks is not None) and (Vt is not None) and (Vs is not None):
        B, P, Kt_len, C = Kt.shape
        U = min(top_u, Kt_len)
        if t_attn is not None:
            t_top_vals, t_top_idx = torch.topk(t_attn, k=min(max(top_r, U), Kt_len), dim=-1, largest=True, sorted=True)
            t_idx_u = t_top_idx[..., :U]  # [B,P,U]
        else:
            t_idx_u = torch.arange(U, device=Kt.device).view(1,1,U).expand(B,P,U)

        Kt_u = _gather_last_dim(Kt, t_idx_u)  # [B,P,U,C]
        Vt_u = _gather_last_dim(Vt, t_idx_u)  # [B,P,U,C]

        Kt_u_n = _safe_norm(Kt_u, dim=-1)
        Ks_n   = _safe_norm(Ks, dim=-1)
        bd = B * P
        Kt_f = Kt_u_n.reshape(bd, U, C)
        Ks_f = Ks_n.reshape(bd, Ks.shape[2], C)
        sims = torch.matmul(Kt_f, Ks_f.transpose(1,2))   # [bd,U,Ks]
        idx_s = sims.argmax(dim=-1).view(B, P, U)

        Ks_u = _gather_last_dim(Ks, idx_s)   # [B,P,U,C]
        Vs_u = _gather_last_dim(Vs, idx_s)   # [B,P,U,C]

        k_loss = F.mse_loss(_safe_norm(Kt_u, dim=-1), _safe_norm(Ks_u, dim=-1))
        v_loss = F.mse_loss(_safe_norm(Vt_u, dim=-1), _safe_norm(Vs_u, dim=-1))
        kv_cos = k_loss + v_loss
        total += w_kv_cos * kv_cos
        losses['kv_cos'] = kv_cos

    # --- (옵션) Context L2 / Cosine ---
    if (t_ctx is not None) and (s_ctx is not None) and (w_ctx_cos > 0.0):
        ctx_cos = 1.0 - F.cosine_similarity(t_ctx.flatten(0,1), s_ctx.flatten(0,1), dim=-1).mean()
        total += w_ctx_cos * ctx_cos
        losses['ctx_cos'] = ctx_cos

    return total, losses

def attention_entropy(attn, eps=1e-8):
    # attn: [B,P,K], 각 위치에 대한 확률 분포
    p = torch.clamp(attn, min=eps)
    ent = -(p * p.log()).sum(dim=-1).mean()
    return ent

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

    # KD 하이퍼
    kd_weight  = hyper_params.get("kd_weight", 1.0)
    kd_top_r   = hyper_params.get("kd_top_r", 64)
    kd_top_u   = hyper_params.get("kd_top_u", 32)
    w_attn_kl  = hyper_params.get("w_attn_kl", 1.0)
    w_kv_cos   = hyper_params.get("w_kv_cos", 1.0)
    w_ctx_cos  = hyper_params.get("w_ctx_cos", 0.0)

    lambda_delta = hyper_params.get("lambda_delta", 1e-4)

    # Stage-2: teacher dropout & attn entropy reg
    stage2_epochs = hyper_params.get("stage2_epochs", 5)  # 마지막 5 epoch
    attn_ent_w    = hyper_params.get("attn_entropy_weight", 0.01)
    teacher_drop_p_stage2 = hyper_params.get("teacher_dropout_p", 0.5)

    # W&B
    load_dotenv(dotenv_path=".env")
    wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
    run = wandb.init(project="stream_teacher_student", config=hyper_params, name=f"experiment_{experiment}")

    # 데이터
    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
    rgb_clips, depth_clips = get_data_list(root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN)
    kitti_train = KITTIVideoDataset(rgb_paths=rgb_clips, depth_paths=depth_clips, resize_size=518, split="train")
    kitti_train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 모델 (단일 GPU)
    teacher = VideoDepthTeacher(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)
    student = VideoDepthStudent(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)

    class TeacherStudentWrapper(torch.nn.Module):
        def __init__(self, teacher, student):
            super().__init__()
            self.teacher = teacher
            self.student = student
        def forward(self, x):
            return self.student.forward(x)
        def forward_features(self, x):
            return self.student.forward_features(x)
        def forward_depth(self, features, x_shape, cache=None):
            return self.student.forward_depth(features, x_shape, cache)

    model = TeacherStudentWrapper(teacher, student)

    # Pretrained
    if args.pretrained_ckpt:
        logger.info(f"Loading Weight from {args.pretrained_ckpt}")
        sd = torch.load(args.pretrained_ckpt, map_location="cpu")
        model.teacher.load_state_dict(sd, strict=True)
        model.student.load_state_dict(sd, strict=True)
        logger.info("Pretrained weights loaded successfully!")

    # Freeze 정책
    for p in model.teacher.parameters(): p.requires_grad = False
    for p in model.student.pretrained.parameters(): p.requires_grad = False
    for p in model.student.head.parameters(): p.requires_grad = True
    model.train()

    # Optim/Sch
    student_params = [p for p in model.student.parameters() if p.requires_grad]
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
            model.student.load_state_dict(sd, strict=True)
        except RuntimeError:
            from collections import OrderedDict
            clean = OrderedDict()
            for k, v in sd.items():
                nk = k
                if nk.startswith("module."): nk = nk[len("module."):]
                if nk.startswith("student."): nk = nk[len("student."):]
                clean[nk] = v
            model.student.load_state_dict(clean, strict=False)

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

    wandb.watch(model.student, log="all")
    best_delta1 = 0.0
    best_epoch  = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")
    
    # ---- Init real-pipeline validation (epoch = -1) ----
    # 초기 성능을 실제 inference+eval 축소 파이프라인으로 측정하여 W&B에 기록
    init_infer_dir = os.path.join(args.val_infer_dir, "init")
    os.makedirs(init_infer_dir, exist_ok=True)

    # 일시적으로 eval 모드
    _prev_train_state = model.student.training
    model.student.eval()
    try:
        init_metrics = validate_with_infer_eval_subset(
            model=model.student,                          # 학생만 사용
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
            model.student.train()

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
        model.train()
        epoch_loss = epoch_frames = 0.0
        epoch_ssi = epoch_tgm = epoch_kd = 0.0
        accum_loss = 0.0
        step_in_window = 0
        update_frequency = hyper_params.get("update_frequency", 6)

        # Stage-2 스케줄
        in_stage2 = (epoch >= num_epochs - stage2_epochs)
        use_teacher_prob = (0.0 if not in_stage2 else (1.0 - teacher_drop_p_stage2))

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
            teacher_frame_buffer = None

            frame_pbar = tqdm(range(T), desc=f"Batch {batch_idx+1} - Frames", leave=False, disable=T < 10)
            for t in frame_pbar:
                x_t = x[:, t:t+1]                              # [B,1,3,H,W]
                mask_t = get_mask(y[:, t:t+1], 1e-3, 80.0).to(device)

                # Teacher window 준비: [t-W+1 ... t+W] (근사) — 간단히 슬라이드 버퍼
                if teacher_frame_buffer is None:
                    teacher_frame_buffer = x_t.detach().clone().repeat(1, CLIP_LEN, 1, 1, 1)
                else:
                    teacher_frame_buffer = torch.cat([
                        teacher_frame_buffer[:, :1],   # anchor 유지
                        teacher_frame_buffer[:, 2:],   # 앞에서 한 칸 밀기
                        x_t.detach().clone()
                    ], dim=1)

                # === Teacher (정확 KD 캐시 수집) ===
                use_teacher = (random.random() < use_teacher_prob) or (not in_stage2)
                if use_teacher:
                    with torch.no_grad():
                        enable_attention_caching(model.teacher)
                        with autocast(enabled=torch.cuda.is_available()):
                            _ = model.teacher(teacher_frame_buffer)
                        t_caches = collect_kd_caches(model.teacher, clear=True)
                        disable_attention_caching(model.teacher)
                        if len(t_caches) == 0:
                            raise RuntimeError("No teacher KD caches collected.")
                        t_cache_last = {k:(v.to(device) if torch.is_tensor(v) else v) for k, v in t_caches[-1].items()}
                else:
                    t_cache_last = None

                # === Student (정확 KD 캐시 수집 + 스트리밍 1-step) ===
                with autocast(enabled=torch.cuda.is_available()):
                    enable_attention_caching(model.student)
                    pred_t_raw, cache = model_stream_step(model.student, x_t, cache)
                    pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)
                    s_caches = collect_kd_caches(model.student, clear=True)
                    disable_attention_caching(model.student)
                    if len(s_caches) == 0:
                        raise RuntimeError("No student KD caches collected.")
                    s_cache_last = s_caches[-1]

                    # ----- KD 손실 -----
                    if use_teacher:
                        kd_total, kd_parts = compute_kd_losses_from_caches(
                            t_cache_last, s_cache_last,
                            top_r=kd_top_r, top_u=kd_top_u,
                            w_attn_kl=w_attn_kl, w_kv_cos=w_kv_cos, w_ctx_cos=w_ctx_cos
                        )
                        kd_loss = kd_weight * kd_total
                    else:
                        # Stage-2: Teacher 없어도 entropy 정규화로 collapse 방지
                        s_attn = s_cache_last.get("attn", None)
                        if (s_attn is not None) and (attn_ent_w > 0):
                            ent = attention_entropy(s_attn)
                            kd_loss = (-attn_ent_w) * ent   # entropy ↑ => loss ↓ (최대화)
                        else:
                            kd_loss = pred_t_raw.new_tensor(0.0)

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

                    loss = kd_loss + ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss

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
                epoch_kd     += kd_loss.item()   * B_eff

                frame_pbar.set_postfix({
                    'SSI': f'{epoch_ssi/ max(1, epoch_frames):.4f}',
                    'TGM': f'{epoch_tgm/ max(1, epoch_frames):.4f}',
                    'KD':  f'{epoch_kd / max(1, epoch_frames):.4f}'
                })
            frame_pbar.close()
        batch_pbar.close()

        # --- Mini Real-pipeline Validation ---
        # (학생만 평가, infer_stream+eval과 동일 경로 축소판)
        val_metrics = validate_with_infer_eval_subset(
            model=model.student,
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
            "train/kd":   epoch_kd   / max(1, epoch_frames),
            "val_real/absrel": val_absrel,
            "val_real/rmse":   val_rmse,
            "val_real/delta1": val_delta1,
            "epoch": epoch,
            "stage2": int(in_stage2),
            "teacher_keep_prob": use_teacher_prob,
        })

        # best 저장 (delta1 ↑)
        if val_delta1 > best_delta1:
            best_delta1 = val_delta1
            best_epoch  = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_delta1": best_delta1,
                "config": hyper_params,
            }, best_model_path)
            logger.info(f"🏆 Best model saved! Epoch {epoch}, Val delta1: {best_delta1:.4f}")

        # latest 저장
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.student.state_dict(),
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
    parser.add_argument("--val_json_file",    type=str, default="/workspace/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json")
    parser.add_argument("--val_infer_dir",    type=str, default="benchmark/output/scannet_stream_valmini")
    parser.add_argument("--val_dataset_key",  type=str, default="scannet")
    parser.add_argument("--val_dataset_tag",  type=str, default="scannet_500")
    parser.add_argument("--val_scenes",       type=int, default=2)
    parser.add_argument("--resume_from", type=str, default="", help="Path to latest/best checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs (e.g., 60)")
    args = parser.parse_args()
    train(args)
