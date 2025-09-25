import os
import argparse
import logging
import random
import copy
import math
import warnings
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import wandb

from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.loss_MiDas import *
from utils.train_helper import *   # validate_with_infer_eval_subset_fast, model_stream_step, batch_ls_scale_shift, norm_ssi, to_BHW_pred
from data.dataLoader import *      # KITTIVideoDataset, GTADataset, get_data_list, get_GTA_paths

# 모델
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")

# ===================== 실험 설정/로깅 =====================
experiment = 33
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

# ===================== KD helper (KV/Attn/Context) =====================
def enable_attention_caching(m):
    for _, layer in m.named_modules():
        if hasattr(layer, 'enable_kd_caching'):
            layer.enable_kd_caching(True)

def disable_attention_caching(m):
    for _, layer in m.named_modules():
        if hasattr(layer, 'enable_kd_caching'):
            layer.enable_kd_caching(False)

def collect_kd_caches(module, clear: bool = True):
    """각 TemporalAttention 레이어의 get_cached_attention_output()을 모아 반환."""
    outs = []
    delta_sum = None
    for _, layer in module.named_modules():
        if hasattr(layer, 'get_cached_attention_output'):
            out = layer.get_cached_attention_output()
            if out is not None:
                d = out.get("delta_reg", None)
                if (d is not None) and torch.is_tensor(d):
                    delta_sum = (d if delta_sum is None else (delta_sum + d))
                outs.append(out)
                if clear and hasattr(layer, 'clear_attention_cache'):
                    layer.clear_attention_cache()
    if outs and (delta_sum is not None):
        outs[-1] = {**outs[-1], "delta_reg": delta_sum}
    return outs

def attention_entropy(attn, eps=1e-8):
    """
    attn: one of
      - [B, H, T, T] (hist; 행별 causal 영역 0..t 사용)
      - [B, P, K]    (단일 분포)
      - [B*H, q, k]  (머지된 배치/헤드)
    returns: scalar mean entropy (nats)
    """
    if attn is None:
        return None
    if not torch.is_tensor(attn):
        raise TypeError("attention_entropy expects a Tensor")

    if attn.dim() == 4:
        B, H, T1, T2 = attn.shape
        T = min(T1, T2)
        A = attn[:, :, :T, :T]
        ent_rows = []
        for t in range(T):
            row = A[:, :, t, :t+1]                       # [B,H,t+1]
            row = torch.clamp(row, min=eps)
            row = row / (row.sum(dim=-1, keepdim=True) + eps)
            ent = -(row * row.log()).sum(dim=-1)         # [B,H]
            ent_rows.append(ent)
        E = torch.stack(ent_rows, dim=2)                 # [B,H,T]
        return E.mean()

    elif attn.dim() == 3:
        p = torch.clamp(attn, min=eps)
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        ent = -(p * p.log()).sum(dim=-1)                 # [B,P] or [B*H,q]
        return ent.mean()

    else:
        raise ValueError(f"Unsupported attn shape {attn.shape}")

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

def reset_rw_memory_recursive(m):
    """배치 시작/시퀀스 시작 시 RWMemory 상태를 초기화."""
    for _, layer in m.named_modules():
        if hasattr(layer, "reset_rw_memory"):
            layer.reset_rw_memory()

def _safe_norm(x, dim=-1, eps=1e-8):
    return F.normalize(x, dim=dim, eps=eps)

def compute_kd_losses_from_caches(
    t_cache: dict, s_cache: dict,
    top_r: int = None,            # None → 전체 사용
    top_u: int = 32,
    w_attn_kl: float = 1.0, w_kv_cos: float = 1.0, w_ctx_cos: float = 0.0,
    eps: float = 1e-8,
):
    """
    프레임 히스토리 전체(1..T)를 사용한 KD.
    기대 키:
      - 'K_all_pre': [B, P, T, C]
      - 'V_all_pre': [B, P, T, C]
      - 'attn_hist': [B, H, T, T]   (없으면 Attn-KL은 skip)
    반환:
      total_loss (scalar tensor), {'attn_kl':..., 'kv_cos':..., 'ctx_cos':...}
    """
    K_T, K_S = t_cache.get("K_all_pre", None), s_cache.get("K_all_pre", None)
    V_T, V_S = t_cache.get("V_all_pre", None), s_cache.get("V_all_pre", None)

    if (K_T is None) or (K_S is None) or (V_T is None) or (V_S is None):
        dev = None
        for v in (K_T, K_S, V_T, V_S):
            if isinstance(v, torch.Tensor):
                dev = v.device; break
        zero = torch.tensor(0.0, device=dev if dev is not None else 'cpu')
        return zero, {}

    device = K_T.device
    B_T, P_T, T_T, C_T = K_T.shape
    B_S, P_S, T_S, C_S = K_S.shape
    T = min(T_T, T_S); P = min(P_T, P_S); C = min(C_T, C_S)

    K_T = K_T[:, :P, :T, :C]
    V_T = V_T[:, :P, :T, :C]
    K_S = K_S[:, :P, :T, :C]
    V_S = V_S[:, :P, :T, :C]

    AT = t_cache.get("attn_hist", None)
    AS = s_cache.get("attn_hist", None)
    attn_kl_loss = None
    ATm = ASm = None
    T_attn = 0

    if (AT is not None) and (AS is not None):
        B_AT, H_AT, Tt1, Tt2 = AT.shape
        B_AS, H_AS, Ts1, Ts2 = AS.shape
        T_attn = min(T, Tt1, Tt2, Ts1, Ts2)
        if T_attn >= 1:
            AT = AT[:, :, :T_attn, :T_attn]
            AS = AS[:, :, :T_attn, :T_attn]
            ATm = AT.mean(dim=1)  # [B,T,T]
            ASm = AS.mean(dim=1)

            kl_terms = []
            for t in range(T_attn):
                klen = t + 1
                aT_row = ATm[:, t, :klen]
                aS_row = ASm[:, t, :klen]
                if (top_r is None) or (top_r >= klen):
                    t_pt = aT_row; s_ps = aS_row
                else:
                    vals, idx = torch.topk(aT_row, k=top_r, dim=-1, largest=True, sorted=False)
                    t_pt = torch.gather(aT_row, 1, idx)
                    s_ps = torch.gather(aS_row, 1, idx)

                t_pt = t_pt / (t_pt.sum(dim=1, keepdim=True) + eps)
                s_ps = s_ps / (s_ps.sum(dim=1, keepdim=True) + eps)
                kl = (t_pt * (torch.log(t_pt + eps) - torch.log(s_ps + eps))).sum(dim=1).mean()
                kl_terms.append(kl)

            if kl_terms:
                attn_kl_loss = torch.stack(kl_terms).mean()

    K_Tn = F.normalize(K_T, dim=-1, eps=eps)
    V_Tn = F.normalize(V_T, dim=-1, eps=eps)
    K_Sn = F.normalize(K_S, dim=-1, eps=eps)
    V_Sn = F.normalize(V_S, dim=-1, eps=eps)

    kv_terms = []
    have_attn_for_kv = (ATm is not None)

    for t in range(T):
        klen = t + 1
        if have_attn_for_kv and (t < T_attn):
            aT_row_mean = ATm[:, t, :klen].mean(dim=0)  # [klen]
            U = min(top_u, klen) if (top_u is not None) else klen
            _, idx_time = torch.topk(aT_row_mean, k=U, dim=-1, largest=True, sorted=False)
        else:
            U = min(top_u, klen) if (top_u is not None) else klen
            idx_time = torch.arange(klen - U, klen, device=device)

        KT_sel = K_Tn[:, :, idx_time, :]
        KS_sel = K_Sn[:, :, idx_time, :]
        VT_sel = V_Tn[:, :, idx_time, :]
        VS_sel = V_Sn[:, :, idx_time, :]
        kv_terms.append(F.mse_loss(KT_sel, KS_sel) + F.mse_loss(VT_sel, VS_sel))

    kv_cos_loss = torch.stack(kv_terms).mean() if kv_terms else torch.tensor(0.0, device=device)

    ctx_cos_loss = None
    if w_ctx_cos > 0.0:
        VTm = V_Tn.mean(dim=1)  # [B,T,C]
        VSm = V_Sn.mean(dim=1)  # [B,T,C]
        if have_attn_for_kv:
            ctx_T, ctx_S = [], []
            for t in range(T_attn):
                klen = t + 1
                wT = ATm[:, t, :klen]; wS = ASm[:, t, :klen]
                wT = wT / (wT.sum(dim=1, keepdim=True) + eps)
                wS = wS / (wS.sum(dim=1, keepdim=True) + eps)
                cT = (wT.unsqueeze(-1) * VTm[:, :klen, :]).sum(dim=1)  # [B,C]
                cS = (wS.unsqueeze(-1) * VSm[:, :klen, :]).sum(dim=1)
                ctx_T.append(cT); ctx_S.append(cS)
            ctx_T = torch.stack(ctx_T, dim=1)  # [B,T_attn,C]
            ctx_S = torch.stack(ctx_S, dim=1)
        else:
            ctx_T = VTm[:, :T, :]; ctx_S = VSm[:, :T, :]

        ctx_cos = 1.0 - F.cosine_similarity(
            ctx_T.reshape(-1, ctx_T.shape[-1]),
            ctx_S.reshape(-1, ctx_S.shape[-1]),
            dim=-1
        ).mean()
        ctx_cos_loss = ctx_cos

    total = torch.zeros((), device=device)
    parts = {}
    if (attn_kl_loss is not None) and (w_attn_kl != 0.0):
        total = total + w_attn_kl * attn_kl_loss
        parts["attn_kl"] = attn_kl_loss
    if (kv_cos_loss is not None) and (w_kv_cos != 0.0):
        total = total + w_kv_cos * kv_cos_loss
        parts["kv_cos"] = kv_cos_loss
    if (ctx_cos_loss is not None) and (w_ctx_cos != 0.0):
        total = total + w_ctx_cos * ctx_cos_loss
        parts["ctx_cos"] = ctx_cos_loss

    return total, parts

# ===================== DataLoader 유틸 =====================
class TagDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, tag_int):
        self.base = base_ds
        self.tag  = int(tag_int)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]           # (T,3,H,W), (T,1,H,W)
        return x, y, torch.tensor(self.tag, dtype=torch.long)

def make_random_subset(dataset, n_samples):
    total = len(dataset)
    n_samples = min(n_samples, total)
    indices = random.sample(range(total), n_samples)
    return Subset(dataset, indices)

# ===================== 학습 루프 =====================
def train(args):
    OUTPUT_DIR = f"outputs/experiment_{experiment}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 설정 로드
    with open("config_jh.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyper_params = config["hyper_parameter"]
    lr         = hyper_params["learning_rate"]
    ratio_ssi  = hyper_params["ratio_ssi"]
    ratio_tgm  = hyper_params["ratio_tgm"]
    num_epochs = hyper_params["epochs"]
    batch_size = hyper_params["batch_size"]
    CLIP_LEN   = hyper_params["clip_len"]
    num_workers = hyper_params.get("num_workers", 4)

    if args.epochs is not None:
        num_epochs = int(args.epochs)

    # KD 하이퍼
    ratio_kd   = hyper_params.get("ratio_kd", 1.0)
    kd_weight  = hyper_params.get("kd_weight", 1.0)  # (미사용시 유지)
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
    vkitti_rgb, vkitti_depth = get_data_list(root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN)
    vkitti_ds = KITTIVideoDataset(rgb_paths=vkitti_rgb, depth_paths=vkitti_depth, clip_len=CLIP_LEN, resize_size=518, split="train")

    gta_root = "/workspace/Video-Depth-Anything/datasets/GTAV_720/GTAV_720"
    gta_rgb, gta_depth, _ = get_GTA_paths(gta_root, split="train")
    gta_ds = GTADataset(rgb_paths=gta_rgb, depth_paths=gta_depth, clip_len=CLIP_LEN, resize_size=518, split="train")

    # 태그 부여: VKITTI=0, GTA=1
    vkitti_tagged = TagDataset(vkitti_ds, tag_int=0)
    gta_tagged    = TagDataset(gta_ds,    tag_int=1)

    # 모델
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
        try:
            model.teacher.load_state_dict(sd, strict=True)
            model.student.load_state_dict(sd, strict=True)
        except Exception:
            # 일부 키 mismatch를 허용
            from collections import OrderedDict
            t_sd, s_sd = OrderedDict(), OrderedDict()
            for k, v in sd.items():
                if k.startswith("teacher."):
                    t_sd[k[len("teacher."):]] = v
                elif k.startswith("student."):
                    s_sd[k[len("student."):]] = v
            if t_sd:
                model.teacher.load_state_dict(t_sd, strict=False)
            if s_sd:
                model.student.load_state_dict(s_sd, strict=False)
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
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # ----- Resume -----
    start_epoch = 0
    best_delta1 = 0.0
    best_epoch  = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")

    if args.resume_from and os.path.isfile(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location="cpu")

        sd = ckpt.get("model_state_dict", ckpt)
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

        if "optimizer_state_dict" in ckpt:
            try: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e: logger.warning(f"Optimizer state load skipped: {e}")
        if "scheduler_state_dict" in ckpt:
            try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e: logger.warning(f"Scheduler state load skipped: {e}")

        if "best_val_delta1" in ckpt:
            try: best_delta1 = float(ckpt["best_val_delta1"])
            except: pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        logger.info(f"▶ Resumed from '{args.resume_from}' | start_epoch={start_epoch} / target_epochs={num_epochs} | best_delta1={best_delta1:.4f}")

    wandb.watch(model.student, log="all")

    # ---- Init real-pipeline validation (epoch = -1) ----
    if not args.test:
        init_infer_dir = os.path.join(args.val_infer_dir, "init")
        os.makedirs(init_infer_dir, exist_ok=True)

        _prev_train_state = model.student.training
        model.student.eval()
        try:
            init_metrics = validate_with_infer_eval_subset_fast(
                model=model.student,
                json_file=args.val_json_file,
                infer_path=args.val_infer_dir,
                dataset=args.val_dataset_key,
                dataset_eval_tag=args.val_dataset_tag,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                input_size=518,
                scenes_to_eval=args.val_scenes,
                scene_indices=[0, 44],
                frame_stride=2,
                max_eval_len=500,
                fp32=False
            )
        finally:
            if _prev_train_state:
                model.student.train()

        init_absrel = float(init_metrics.get("abs_relative_difference", float('nan')))
        init_rmse   = float(init_metrics.get("rmse_linear", float('nan')))
        init_delta1 = float(init_metrics.get("delta1_acc", float('nan')))

        logger.info(f"[Init] real-pipeline val  | absrel={init_absrel:.4f}  rmse={init_rmse:.4f}  delta1={init_delta1:.4f}")
        wandb.log({
            "init/absrel": init_absrel,
            "init/rmse":   init_rmse,
            "init/delta1": init_delta1,
            "epoch": -1,
        })
        best_delta1 = init_delta1

    # --------------------- Training ---------------------
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False):
        # epoch 시드 전달 (데이터 증강/시프트)
        if hasattr(vkitti_ds, "set_epoch"): vkitti_ds.set_epoch(epoch)
        if hasattr(gta_ds,    "set_epoch"): gta_ds.set_epoch(epoch)

        vkitti_subset = make_random_subset(vkitti_tagged, 150)
        gta_subset    = make_random_subset(gta_tagged, 150)
        train_dataset = ConcatDataset([vkitti_subset, gta_subset])
        train_loader  = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False
        )

        model.train()
        epoch_loss = 0.0
        epoch_frames = 0
        epoch_ssi_w = epoch_tgm_w = epoch_kd_w = 0.0
        accum_loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=False)
        step_in_window = 0
        update_frequency = hyper_params.get("update_frequency", 6)

        # Stage-2 스케줄
        in_stage2 = (epoch >= num_epochs - stage2_epochs)
        use_teacher_prob = (0.0 if not in_stage2 else (1.0 - teacher_drop_p_stage2))

        batch_pbar = tqdm(enumerate(train_loader),
                          desc=f"Epoch {epoch+1}/{num_epochs} - Batches",
                          total=len(train_loader),
                          leave=False)
        for batch_idx, batch in batch_pbar:
            # (x, y, tag)
            if len(batch) == 3:
                x, y, tag = batch
                tag = tag.to(device, non_blocking=True)
            else:
                x, y = batch
                tag  = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, T = x.shape[:2]

            cache = None
            prev_pred_raw = prev_mask = prev_y = None
            teacher_frame_buffer = None

            frame_pbar = tqdm(range(T), desc=f"Batch {batch_idx+1} - Frames", leave=False, disable=T < 10)
            reset_rw_memory_recursive(model.student)

            for t in frame_pbar:
                x_t = x[:, t:t+1]          # [B,1,3,H,W]
                y_t = y[:, t:t+1]          # [B,1,1,H,W]

                # 데이터셋별 유효 깊이 범위
                tag_vkit = (tag == 0).view(-1, 1, 1, 1)  # [B,1,1,1] bool
                min_d = torch.where(tag_vkit,
                                    torch.tensor(5.0,   device=device),
                                    torch.tensor(30.0,  device=device))
                max_d = torch.where(tag_vkit,
                                    torch.tensor(120.0, device=device),
                                    torch.tensor(200.0, device=device))
                mask_bool = ((y_t >= min_d) & (y_t <= max_d))         # [B,1,1,H,W]
                mask_t    = mask_bool.squeeze(2).float()              # [B,1,H,W]  ← 여기서 맞춰두기
                
                # # 유효 픽셀 비율 디버깅
                # valid_px = mask_bool.sum()
                # total_px = mask_bool.numel()
                # valid_ratio = valid_px.item() / max(1, total_px)
                # print(f"[Frame {t}] valid_px = {valid_px.item()}/{total_px} "
                #     f"({valid_ratio*100:.2f}%)")

                # Teacher window (간단 슬라이드 버퍼)
                if teacher_frame_buffer is None:
                    teacher_frame_buffer = x_t.detach().clone().repeat(1, CLIP_LEN, 1, 1, 1)
                else:
                    teacher_frame_buffer = torch.cat([
                        teacher_frame_buffer[:, :1],
                        teacher_frame_buffer[:, 2:],
                        x_t.detach().clone()
                    ], dim=1)

                # === Teacher (KD 캐시 수집) ===
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
                        t_cache_all = t_caches[-1]
                else:
                    t_cache_all = None

                # === Student (KD 캐시 + 스트리밍 1-step) ===
                with autocast(enabled=torch.cuda.is_available()):
                    enable_attention_caching(model.student)
                    pred_t_raw, cache = model_stream_step(
                        model.student, x_t, cache,
                        stream_mode=True, select_top_r=None, update_top_u=kd_top_u,
                        rope_dt=None, return_attn=False, return_qkv=False,
                        bidirectional_update_length=16, current_frame=t
                    )
                    # ✅ 다음 연산(같은 프레임의 손실 포함)에서 캐시가 이전 그래프를 참조하지 않게 즉시 끊기
                    cache = _detach_cache(cache)

                    # [B,H,W]로 맞춤 + 안전화 ✅
                    pred_t_raw = to_BHW_pred(pred_t_raw)
                    pred_t_raw = torch.nan_to_num(pred_t_raw, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=1e-6)

                    s_caches = collect_kd_caches(model.student, clear=True)
                    disable_attention_caching(model.student)
                    if len(s_caches) == 0:
                        raise RuntimeError("No student KD caches collected.")
                    s_cache_all = s_caches[-1]

                    # ----- KD 손실 -----
                    if use_teacher:
                        kd_total, kd_parts = compute_kd_losses_from_caches(
                            t_cache_all, s_cache_all,
                            top_r=kd_top_r, top_u=kd_top_u,
                            w_attn_kl=w_attn_kl, w_kv_cos=w_kv_cos, w_ctx_cos=w_ctx_cos
                        )
                        delta_reg = s_cache_all.get("delta_reg", None)
                        if (delta_reg is not None) and torch.is_tensor(delta_reg):
                            kd_loss = kd_total + lambda_delta * delta_reg
                        else:
                            kd_loss = kd_total
                    else:
                        s_attn_hist = s_cache_all.get("attn_hist", None)
                        if (s_attn_hist is not None) and (attn_ent_w > 0):
                            ent = attention_entropy(s_attn_hist)
                            kd_loss = (-attn_ent_w) * ent
                        else:
                            kd_loss = pred_t_raw.new_tensor(0.0)

                    # ----- Depth Loss SSI--------------------------------------------
                    gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6)).squeeze(2)  # [B,1,H,W]
                    if pred_t_raw.shape[0] != gt_disp_t.shape[0]:
                        pred_t_raw = pred_t_raw[:gt_disp_t.shape[0]]

                    # 유효 픽셀 수 체크(빈 프레임이면 SSI/TGM=0으로 스킵) ✅
                    valid_px = mask_t.squeeze(2).sum(dtype=torch.float32)
                    if valid_px.item() < 1:
                        ssi_loss_t = pred_t_raw.new_tensor(0.0)
                        tgm_loss   = pred_t_raw.new_tensor(0.0)
                        # 총 손실
                        loss = ratio_kd * kd_loss + ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss
                        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        # LS 정렬 계수 안정 계산 ✅
                        with torch.no_grad():
                            a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t.squeeze(2))

                        pred_t_aligned_disp  = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                        pred_t_aligned_disp  = torch.nan_to_num(pred_t_aligned_disp, nan=0.0, posinf=1e6, neginf=0.0)

                        # SSI 입력 생성 (정규화 안전화된 함수 사용) ✅
                        disp_normed_t = norm_ssi(y[:, t:t+1], mask_bool).squeeze(2)  # [B,1,H,W]
                        ssi_loss_t    = loss_ssi(pred_t_aligned_disp.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))
                        ssi_loss_t    = torch.nan_to_num(ssi_loss_t, nan=0.0, posinf=0.0, neginf=0.0)

                    # ----- Depth Loss TGM--------------------------------------------
                    if t > 0:
                        prev_valid = (prev_mask.sum(dtype=torch.float32) if prev_mask is not None
                                    else torch.tensor(0., device=mask_t.device))
                        if prev_valid.item() > 0:
                            prev_aligned_disp  = (a_star.detach() * prev_pred_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                            prev_aligned_depth = 1.0 / prev_aligned_disp.clamp(min=1e-6)
                            curr_aligned_depth = 1.0 / pred_t_aligned_disp.clamp(min=1e-6)
                            pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)  # [B,2,H,W]
                            y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)                       # [B,2,1,H,W]
                            m_pair    = torch.cat([prev_mask, mask_t], dim=1)                         # [B,2,1,H,W]
                            tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2).bool())
                            tgm_loss  = torch.nan_to_num(tgm_loss, nan=0.0, posinf=0.0, neginf=0.0)
                        else:
                            # 그래프가 완전히 끊기지 않도록 pred_t_raw 경유 0 손실 (requires_grad=True 보장)
                            tgm_loss = (pred_t_raw * 0.0).sum()
                    else:
                        tgm_loss = (pred_t_raw * 0.0).sum()

                    # 총 손실
                    loss = ratio_kd * kd_loss + ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss

                    # 수치·그래프 안전장치: NaN/Inf → 0으로 대체하되 그래프는 유지
                    if not torch.isfinite(loss):
                        loss = (pred_t_raw * 0.0).sum()

                # ───────────────────────── 누적/업데이트 (gradient accumulation) ─────────────────────────
                update_freq = max(1, int(update_frequency))
                # accum_loss가 파이썬 float로 시작했을 수도 있어 텐서로 교체
                if not isinstance(accum_loss, torch.Tensor):
                    accum_loss = loss / update_freq
                else:
                    accum_loss = accum_loss + (loss / update_freq)
                step_in_window += 1

                if step_in_window == update_freq:
                    if accum_loss.requires_grad:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(accum_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        epoch_loss += float(accum_loss.detach().cpu())
                    # 윈도우 리셋
                    accum_loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=False)
                    step_in_window = 0

                # ───────────────────────── 상태 업데이트 ─────────────────────────
                cache = _detach_cache(cache)
                prev_pred_raw = pred_t_raw.detach()
                prev_mask = mask_t
                prev_y    = y[:, t:t+1]

                # ───────────────────────── 통계 (로그용) ─────────────────────────
                B_eff = pred_t_raw.shape[0]
                # detach+float로 안전 수집 (NaN 방지 처리)
                ssi_item = float(torch.nan_to_num(ssi_loss_t.detach(), nan=0.0, posinf=0.0, neginf=0.0).cpu())
                tgm_item = float(torch.nan_to_num(tgm_loss.detach(),    nan=0.0, posinf=0.0, neginf=0.0).cpu())
                kd_item  = float(torch.nan_to_num(kd_loss.detach() if torch.is_tensor(kd_loss) else torch.as_tensor(kd_loss, device=pred_t_raw.device),
                                                nan=0.0, posinf=0.0, neginf=0.0).cpu())

                epoch_frames += B_eff
                epoch_ssi_w  += (ratio_ssi * ssi_item) * B_eff
                epoch_tgm_w  += (ratio_tgm * tgm_item) * B_eff
                epoch_kd_w   += (ratio_kd  * kd_item)  * B_eff

                frame_pbar.set_postfix({
                    'wSSI': f'{epoch_ssi_w / max(1, epoch_frames):.4f}',
                    'wTGM': f'{epoch_tgm_w / max(1, epoch_frames):.4f}',
                    'wKD':  f'{epoch_kd_w  / max(1, epoch_frames):.4f}'
                })
            frame_pbar.close()
        batch_pbar.close()

        # --- Mini Real-pipeline Validation ---
        val_metrics = validate_with_infer_eval_subset_fast(
            model=model.student,
            json_file=args.val_json_file,
            infer_path=args.val_infer_dir,
            dataset=args.val_dataset_key,
            dataset_eval_tag=args.val_dataset_tag,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            input_size=518,
            scenes_to_eval=args.val_scenes,
            scene_indices=[0, 44],
            frame_stride=2,
            max_eval_len=500,
            fp32=False
        )

        val_absrel = float(val_metrics.get("abs_relative_difference", float('nan')))
        val_rmse   = float(val_metrics.get("rmse_linear", float('nan')))
        val_delta1 = float(val_metrics.get("delta1_acc", float('nan')))

        # 로깅
        wandb.log({
            "train/loss": epoch_loss / max(1, len(train_loader)),
            "train/ssi":  epoch_ssi_w / max(1, epoch_frames),
            "train/tgm":  epoch_tgm_w / max(1, epoch_frames),
            "train/kd":   epoch_kd_w / max(1, epoch_frames),
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
    parser.add_argument("--test", action='store_true', help="Test version (skip training)")
    args = parser.parse_args()
    train(args)
