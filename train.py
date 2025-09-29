# -*- coding: utf-8 -*-
import os
import argparse
import logging
import random
import math
import warnings
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import wandb

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

# ---- 외부 유틸 (그대로 사용) ----
from utils.train_helper import (
    validate_with_infer_eval_subset_fast,
    model_stream_step,
    batch_ls_scale_shift,
    to_BHW_pred,
)

from data.dataLoader import (     # 경로/데이터셋 로더
    KITTIVideoDataset, get_data_list
)

# 모델
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")
torch.backends.cudnn.benchmark = True

# ===================== 실험 설정/로깅 =====================
experiment = 34
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

# ===================== 필수 유틸 =====================
def safe_collate(batch):
    elem = batch[0]
    if torch.is_tensor(elem):
        return torch.stack([b.contiguous().clone() for b in batch], dim=0)
    if isinstance(elem, (list, tuple)):
        transposed = list(zip(*batch))
        return type(elem)(safe_collate(samples) for samples in transposed)
    if isinstance(elem, dict):
        return {k: safe_collate([d[k] for d in batch]) for k in elem}
    return default_collate(batch)

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

def to_B1HW_mask(mask):
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.dim() == 5:
        if mask.size(2) == 1:
            mask = mask.squeeze(2)  # [B,T,H,W]
        else:
            mask = mask.any(dim=2)  # [B,T,H,W]
    if mask.dim() != 4:
        raise RuntimeError(f"mask must be 4D after squeeze, got {mask.shape}")
    if mask.size(1) != 1:
        mask = mask[:, :1]
    return mask.contiguous()

# ---- KD helper (KV/Attn/Context) ----
def enable_attention_caching(m, role):
    for _, layer in m.named_modules():
        if hasattr(layer, 'enable_kd_caching'):
            layer.enable_kd_caching(True, role)

def disable_attention_caching(m):
    for _, layer in m.named_modules():
        if hasattr(layer, 'enable_kd_caching'):
            layer.enable_kd_caching(False, "off")

def collect_kd_caches(module, clear: bool = True):
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
    if attn is None: return None
    if not torch.is_tensor(attn): raise TypeError("attention_entropy expects a Tensor")
    if attn.dim() == 4:
        B, H, T1, T2 = attn.shape
        T = min(T1, T2)
        A = attn[:, :, :T, :T]
        ent_rows = []
        for t in range(T):
            row = A[:, :, t, :t+1]
            row = torch.clamp(row, min=eps)
            row = row / (row.sum(dim=-1, keepdim=True) + eps)
            ent = -(row * row.log()).sum(dim=-1)
            ent_rows.append(ent)
        return torch.stack(ent_rows, dim=2).mean()
    elif attn.dim() == 3:
        p = torch.clamp(attn, min=eps)
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        return (-(p * p.log()).sum(dim=-1)).mean()
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
    for _, layer in m.named_modules():
        if hasattr(layer, "reset_rw_memory"):
            layer.reset_rw_memory()

def compute_kd_losses_from_caches(
    t_cache: dict, s_cache: dict,
    top_r: int = None, top_u: int = 32,
    w_attn_kl: float = 1.0, w_kv_cos: float = 1.0, w_ctx_cos: float = 0.0,
    eps: float = 1e-8,
):
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
        T_attn = min(T, AT.shape[2], AT.shape[3], AS.shape[2], AS.shape[3])
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
                    t_pt, s_ps = aT_row, aS_row
                else:
                    _, idx = torch.topk(aT_row, k=top_r, dim=-1, largest=True, sorted=False)
                    t_pt = torch.gather(aT_row, 1, idx)
                    s_ps = torch.gather(aS_row, 1, idx)
                t_pt = t_pt / (t_pt.sum(dim=1, keepdim=True) + eps)
                s_ps = s_ps / (s_ps.sum(dim=1, keepdim=True) + eps)
                kl = (t_pt * (torch.log(t_pt + eps) - torch.log(s_ps + eps))).sum(dim=1).mean()
                kl_terms.append(kl)
            attn_kl_loss = torch.stack(kl_terms).mean() if kl_terms else None

    K_Tn = F.normalize(K_T, dim=-1, eps=eps)
    V_Tn = F.normalize(V_T, dim=-1, eps=eps)
    K_Sn = F.normalize(K_S, dim=-1, eps=eps)
    V_Sn = F.normalize(V_S, dim=-1, eps=eps)

    kv_terms = []
    have_attn_for_kv = (ATm is not None)
    for t in range(T):
        klen = t + 1
        if have_attn_for_kv and (t < T_attn):
            aT_row_mean = ATm[:, t, :klen].mean(dim=0)
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
        VTm = V_Tn.mean(dim=1)
        VSm = V_Sn.mean(dim=1)
        if have_attn_for_kv:
            ctx_T, ctx_S = [], []
            for t in range(T_attn):
                klen = t + 1
                wT = ATm[:, t, :klen]; wS = ASm[:, t, :klen]
                wT = wT / (wT.sum(dim=1, keepdim=True) + eps)
                wS = wS / (wS.sum(dim=1, keepdim=True) + eps)
                cT = (wT.unsqueeze(-1) * VTm[:, :klen, :]).sum(dim=1)
                cS = (wS.unsqueeze(-1) * VSm[:, :klen, :]).sum(dim=1)
                ctx_T.append(cT); ctx_S.append(cS)
            ctx_T = torch.stack(ctx_T, dim=1)
            ctx_S = torch.stack(ctx_S, dim=1)
        else:
            ctx_T, ctx_S = VTm[:, :T, :], VSm[:, :T, :]
        ctx_cos_loss = 1.0 - F.cosine_similarity(
            ctx_T.reshape(-1, ctx_T.shape[-1]),
            ctx_S.reshape(-1, ctx_S.shape[-1]),
            dim=-1
        ).mean()

    total = torch.zeros((), device=device)
    parts = {}
    if (attn_kl_loss is not None) and (w_attn_kl != 0.0):
        total = total + w_attn_kl * attn_kl_loss; parts["attn_kl"] = attn_kl_loss
    if (kv_cos_loss is not None) and (w_kv_cos != 0.0):
        total = total + w_kv_cos * kv_cos_loss; parts["kv_cos"] = kv_cos_loss
    if (ctx_cos_loss is not None) and (w_ctx_cos != 0.0):
        total = total + w_ctx_cos * ctx_cos_loss; parts["ctx_cos"] = ctx_cos_loss
    return total, parts

# ---- KD (pred/feat) ----
def _si_disp(x, eps=1e-6):
    return 1.0 / x.clamp(min=eps)

def kd_loss_pred_si_l1(stu_depth_BHW, tea_depth_BHW, mask_B1HW, eps=1e-6):
    B, H, W = stu_depth_BHW.shape
    m = mask_B1HW.bool()[:, 0]
    s = _si_disp(stu_depth_BHW, eps=eps)
    t = _si_disp(tea_depth_BHW.detach(), eps=eps)
    def _mean_masked(z, mask):
        denom = mask.float().sum(dim=(-1, -2), keepdim=True).clamp(min=1.0)
        return (z * mask).sum(dim=(-1, -2), keepdim=True) / denom
    s_mu = _mean_masked(s, m); t_mu = _mean_masked(t, m)
    s_si, t_si = s - s_mu, t - t_mu
    l1 = (s_si - t_si).abs() * m.float()
    denom = m.float().sum(dim=(-1, -2)).clamp(min=1.0)
    return (l1.sum(dim=(-1, -2)) / denom).mean()

def kd_loss_feat_mse(stu_feats, tea_feats):
    if not isinstance(stu_feats, (list, tuple)): stu_feats = [stu_feats]
    if not isinstance(tea_feats, (list, tuple)): tea_feats = [tea_feats]
    L = min(len(stu_feats), len(tea_feats))
    loss_terms = []
    for i in range(L):
        sf = stu_feats[i]; tf = tea_feats[i].detach()
        if sf.shape[-2:] != tf.shape[-2:]:
            tf = F.interpolate(tf, size=sf.shape[-2:], mode="bilinear", align_corners=False)
        C = min(sf.shape[1], tf.shape[1])
        loss_terms.append(((sf[:, :C] - tf[:, :C]) ** 2).mean())
    return sum(loss_terms) / max(1, len(loss_terms))

# ===================== Trainable 파라미터 설정 =====================
def set_trainable_by_mode(student, mode="all_but_encoder", patterns=None):
    for p in student.parameters():
        p.requires_grad = False
    if hasattr(student, "pretrained"):
        for p in student.pretrained.parameters():
            p.requires_grad = False
    matched = []
    if mode in ("default", "head"):
        if hasattr(student, "head"):
            for n, p in student.head.named_parameters(prefix="head"):
                p.requires_grad = True
                matched.append((n, p.numel()))
    elif mode == "all_but_encoder":
        for n, p in student.named_parameters():
            if n.startswith("pretrained."): continue
            p.requires_grad = True
            matched.append((n, p.numel()))
    elif mode == "patterns":
        pats = [s.strip() for s in (patterns or "").split(",") if s.strip()]
        for n, p in student.named_parameters():
            if n.startswith("pretrained."): continue
            if any(pat in n for pat in pats):
                p.requires_grad = True
                matched.append((n, p.numel()))
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")
    total_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params    = sum(p.numel() for p in student.parameters())
    return {
        "total_params": total_params,
        "total_trainable": total_trainable,
        "matched_examples": matched[:20],
        "num_matched": len(matched),
        "mode": mode,
    }

# ===================== Loss 래퍼 =====================
from utils.loss_video_depth import SpatialLossWrapper as Loss_ssi
from utils.loss_video_depth import TemporalLossWrapper as LossTGMVector

# ===================== 학습 루프 =====================
def train(args):
    OUTPUT_DIR = f"outputs/experiment_{experiment}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 설정 로드
    with open("config_jh.yaml", "r") as f:
        config = yaml.safe_load(f)
    hp = config["hyper_parameter"]
    lr         = hp["learning_rate"]
    ratio_ssi  = hp["ratio_ssi"]
    ratio_tgm  = hp["ratio_tgm"]
    num_epochs = int(args.epochs) if args.epochs is not None else hp["epochs"]
    batch_size = hp["batch_size"]
    CLIP_LEN   = hp["clip_len"]
    num_workers = hp.get("num_workers", 4)

    # KD 하이퍼
    ratio_kd   = hp.get("ratio_kd", 1.0)
    kd_top_r   = hp.get("kd_top_r", 64)
    kd_top_u   = hp.get("kd_top_u", 32)
    w_attn_kl  = hp.get("w_attn_kl", 0.0)   # 학생 attn_hist 미수집이면 0 권장
    w_kv_cos   = hp.get("w_kv_cos", 1.0)
    w_ctx_cos  = hp.get("w_ctx_cos", 0.0)
    lambda_delta = hp.get("lambda_delta", 0.0)  # 학생 delta_reg 없으면 0

    # Stage-2
    stage2_epochs = hp.get("stage2_epochs", 5)
    attn_ent_w    = hp.get("attn_entropy_weight", 0.01)
    teacher_drop_p_stage2 = hp.get("teacher_dropout_p", 0.5)

    # W&B
    load_dotenv(dotenv_path=".env")
    wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
    run = wandb.init(
        project="stream_teacher_student",
        config={**hp, "kd_mode": args.kd_mode, "trainable_mode": args.trainable_mode},
        name=f"experiment_{experiment}",
        settings=wandb.Settings(start_method="thread")
    )

    # 데이터
    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
    vkitti_rgb, vkitti_depth = get_data_list(root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN)
    vkitti_ds = KITTIVideoDataset(rgb_paths=vkitti_rgb, depth_paths=vkitti_depth, clip_len=CLIP_LEN, resize_size=518, split="train")

    logger.info(f"train_VKITTI_total_clips : {len(vkitti_ds)}")

    # 태그
    vkitti_tagged = TagDataset(vkitti_ds, tag_int=0)

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
            from collections import OrderedDict
            t_sd, s_sd = OrderedDict(), OrderedDict()
            for k, v in sd.items():
                if k.startswith("teacher."):
                    t_sd[k[len("teacher."):]] = v
                elif k.startswith("student."):
                    s_sd[k[len("student."):]] = v
            if t_sd: model.teacher.load_state_dict(t_sd, strict=False)
            if s_sd: model.student.load_state_dict(s_sd, strict=False)
        logger.info("Pretrained weights loaded successfully!")

    # Freeze
    for p in model.teacher.parameters(): p.requires_grad = False
    model.teacher.eval()
    cfg = set_trainable_by_mode(model.student, mode=args.trainable_mode, patterns=args.trainable_patterns)
    logger.info(f"[Trainable] mode={cfg['mode']}  total_params={cfg['total_params']:,}  trainable={cfg['total_trainable']:,}  matched={cfg['num_matched']}")

    # Optim/Sch
    student_params = [p for p in model.student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(student_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Loss
    loss_ssi = Loss_ssi(alpha=0.5, scales=4, trim=0.2, reduction="batch-based")
    loss_tgm = LossTGMVector(trim=0.2, temp_grad_scales=1, temp_grad_decay=0.5,
                             reduction="batch-based", diff_depth_th=0.05)
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
        for key in ("optimizer_state_dict", "scheduler_state_dict"):
            try:
                (optimizer if key=="optimizer_state_dict" else scheduler).load_state_dict(ckpt[key])
            except Exception:
                pass
        best_delta1 = float(ckpt.get("best_val_delta1", 0.0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        logger.info(f"▶ Resumed from '{args.resume_from}' | start_epoch={start_epoch} / target_epochs={num_epochs} | best_delta1={best_delta1:.4f}")

    model.train()
    wandb.watch(model.student, log="gradients")

    # ---- Init validation (epoch = -1) ----
    if not args.test:
        model.student.eval()
        with torch.no_grad():
            init_metrics = validate_with_infer_eval_subset_fast(
                model=model.student,
                json_file=args.val_json_file,
                infer_path=args.val_infer_dir,
                dataset=args.val_dataset_key,
                dataset_eval_tag=args.val_dataset_tag,
                device=device.type,
                input_size=518,
                scenes_to_eval=args.val_scenes,
                scene_indices=[1, 39, 44, 93],
                frame_stride=2,
                max_eval_len=500,
                fp32=False
            )
        model.student.train()
        # 기존 집계 로깅
        best_delta1 = float(init_metrics.get("delta1_acc", 0.0))
        wandb.log({
            "init/absrel": float(init_metrics.get("abs_relative_difference", float('nan'))),
            "init/rmse":   float(init_metrics.get("rmse_linear", float('nan'))),
            "init/delta1": best_delta1,
            "epoch": -1
        })

        # ⇩ 씬별 로깅 추가
        for sc in init_metrics.get("per_scene", []):
            wandb.log({
                f"init/scene_{sc['scene_idx']}/absrel": sc["abs_relative_difference"],
                f"init/scene_{sc['scene_idx']}/rmse":   sc["rmse_linear"],
                f"init/scene_{sc['scene_idx']}/delta1": sc["delta1_acc"],
                "init/scene_key": sc["scene_key"],
                "epoch": -1
            }, commit=False)

    # --------------------- Training ---------------------
    log_every = max(1, int(args.log_every))

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False, dynamic_ncols=True):
        if hasattr(vkitti_ds, "set_epoch"):
            vkitti_ds.set_epoch(epoch)

        data_size = 3 if args.test else 100
        train_dataset = make_random_subset(vkitti_tagged, data_size)
        train_loader  = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=safe_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )

        epoch_loss = 0.0
        epoch_frames = 0
        epoch_wSSI = epoch_wTGM = epoch_wKD = 0.0

        epoch_loss_sum = 0.0           # 프레임별 loss 합
        epoch_loss_wsum = 0.0          # 배치 크기 가중 합 (loss * B_eff)
        epoch_steps = 0                # loss를 추가한 프레임 수
        epoch_frames = 0               # 기존처럼 샘플 수(B의 합)

        in_stage2 = (epoch >= num_epochs - stage2_epochs)
        use_teacher_prob = (0.0 if not in_stage2 else (1.0 - teacher_drop_p_stage2))

        batch_pbar = tqdm(enumerate(train_loader),
                          total=len(train_loader),
                          desc=f"[E{epoch+1}/{num_epochs}] Batches",
                          leave=False, dynamic_ncols=True)

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

            batch_frames = 0
            batch_wSSI = batch_wTGM = batch_wKD = 0.0

            cache = None
            prev_pred_raw = prev_mask = prev_y = None
            teacher_frame_buffer = None
            reset_rw_memory_recursive(model.student)

            # ---- 학생 캐싱: 배치 단위로 켜고 끝에서 끕니다 (중요!) ----
            student_kd_active = (args.kd_mode == "cache" and ratio_kd > 0.0)
            if student_kd_active:
                enable_attention_caching(model.student, role="student")
                num_kd_layers = sum(1 for _, m in model.student.named_modules()
                                    if hasattr(m, "get_cached_attention_output"))
                # logger.info(f"[DEBUG] student KD-capable layers = {num_kd_layers}")

            frame_pbar = tqdm(range(T),
                              desc=f"[E{epoch+1}/{num_epochs}] B{batch_idx+1}/{len(train_loader)}",
                              leave=False, dynamic_ncols=True)

            for t in frame_pbar:
                x_t = x[:, t:t+1]          # [B,1,3,H,W]
                y_t = y[:, t:t+1]          # [B,1,1,H,W]

                # 유효 깊이 마스크
                tag_vkit = (tag == 0).view(-1, 1, 1, 1)
                min_d = torch.where(tag_vkit, torch.tensor(5.0, device=device),   torch.tensor(30.0, device=device))
                max_d = torch.where(tag_vkit, torch.tensor(120.0, device=device), torch.tensor(200.0, device=device))
                mask_bool = ((y_t >= min_d) & (y_t <= max_d))
                ssi_mask = to_B1HW_mask(mask_bool)
                mask_t = ssi_mask.float()

                # Teacher window
                if teacher_frame_buffer is None:
                    teacher_frame_buffer = x_t.detach().clone().repeat(1, CLIP_LEN, 1, 1, 1)
                else:
                    teacher_frame_buffer = torch.cat([teacher_frame_buffer[:, 1:], x_t.detach().clone()], dim=1)

                use_teacher = (random.random() < use_teacher_prob) or (not in_stage2)
                if use_teacher and (ratio_kd > 0.0) and (args.kd_mode == "cache"):
                    with torch.no_grad():
                        enable_attention_caching(model.teacher, role="teacher")
                        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                            _ = model.teacher(teacher_frame_buffer)
                        t_caches = collect_kd_caches(model.teacher, clear=True)
                        disable_attention_caching(model.teacher)
                        if len(t_caches) == 0:
                            raise RuntimeError("No teacher KD caches collected.")
                        t_cache_all = t_caches[-1]
                else:
                    t_cache_all = None

                # ── Student 1-step (스트리밍) ──
                with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    ret_attn = student_kd_active
                    ret_qkv  = student_kd_active
                    pred_t_raw, cache = model_stream_step(
                        model.student, x_t, cache,
                        stream_mode=True, select_top_r=None, update_top_u=kd_top_u,
                        rope_dt=None, return_attn=ret_attn, return_qkv=ret_qkv,
                        bidirectional_update_length=16, current_frame=t
                    )

                    if epoch == start_epoch and batch_idx == 0 and t == 0 and args.kd_mode == "cache":
                        s_caches = collect_kd_caches(model.student, clear=False)
                        s = s_caches[-1]
                        K = s["K_all_pre"]; V = s["V_all_pre"]
                        logger.info(f"[CHK student cache] K.requires_grad={K.requires_grad}, V.requires_grad={V.requires_grad}")

                    # ========= SAFE-GRAD 가드: 예측 텐서가 그래프에 붙어있는지 보장 =========
                    # 일부 구현에서 stream step 결과가 detach되어 나올 수 있으므로 체크 후 재계산
                    need_recompute = (not torch.is_tensor(pred_t_raw)) or (not pred_t_raw.requires_grad)
                    if need_recompute:
                        # 한 프레임만 학생의 정규 경로로 재계산 (그래프 연결용, 오버헤드 적음)
                        with torch.set_grad_enabled(True):
                            feats = model.student.forward_features(x_t)
                            pred_t_raw_re = model.student.forward_depth(feats, x_shape=x_t.shape, cache=None)
                        pred_t_raw = pred_t_raw_re  # 그래프가 붙은 텐서로 교체

                        if (epoch == start_epoch) and (batch_idx == 0) and (t == 0):
                            logger.info("[SAFE-GRAD] pred_t_raw was detached → recomputed via features+head")

                    # 순환 state는 여기서만 분리 (예측 텐서는 그대로 그래프 유지)
                    cache = _detach_cache(cache)

                    pred_t_raw = to_BHW_pred(pred_t_raw)  # [B,H,W]
                    # 모델 출력 = depth 로 가정 → 양수화
                    pred_depth_pos = F.softplus(torch.nan_to_num(pred_t_raw, nan=0.0, posinf=0.0, neginf=0.0)) + 1e-6

                    # 배치 정렬
                    B_eff = y.shape[0]
                    if pred_t_raw.shape[0] != B_eff:
                        if pred_t_raw.shape[0] == 1 and B_eff > 1:
                            pred_t_raw     = pred_t_raw.expand(B_eff, -1, -1).contiguous()
                            pred_depth_pos = pred_depth_pos.expand(B_eff, -1, -1).contiguous()
                        else:
                            pred_t_raw     = pred_t_raw[:B_eff]
                            pred_depth_pos = pred_depth_pos[:B_eff]

                    # GT depth (이번 프레임)
                    gt_depth_t = y[:, t:t+1].squeeze(2)  # [B,H,W]

                    # ── KD ──
                    kd_loss = pred_t_raw.new_tensor(0.0)
                    if ratio_kd > 0.0:
                        if args.kd_mode == "cache":
                            if use_teacher and (t_cache_all is not None):
                                # 학생 캐시는 누적 중이므로 clear=False 로 조회만
                                s_caches = collect_kd_caches(model.student, clear=False)
                                if len(s_caches) == 0:
                                    # 아직 누적이 충분치 않으면 KD=0으로 진행(초기 몇 스텝)
                                    kd_loss = kd_loss + 0.0
                                else:
                                    s_cache_all = s_caches[-1]
                                    kd_total, _ = compute_kd_losses_from_caches(
                                        t_cache_all, s_cache_all,
                                        top_r=kd_top_r, top_u=kd_top_u,
                                        w_attn_kl=w_attn_kl, w_kv_cos=w_kv_cos, w_ctx_cos=w_ctx_cos
                                    )
                                    kd_loss = kd_total
                            elif (not use_teacher) and (attn_ent_w > 0):
                                # 학생 attn_hist가 없을 수 있으니 entropy reg는 비활성/작게 유지
                                kd_loss = kd_loss + 0.0

                        elif args.kd_mode == "pred":
                            with torch.no_grad():
                                tea_disp = to_BHW_pred(model.teacher(x_t))
                                tea_disp = torch.nan_to_num(tea_disp, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=1e-6)
                            stu_disp = pred_depth_pos
                            kd_loss = kd_loss_disp_si_l1(stu_disp, tea_disp, mask_t)

                        elif args.kd_mode == "feat":
                            with torch.no_grad():
                                t_feats = model.teacher.forward_features(x_t)
                            s_feats = model.student.forward_features(x_t)
                            kd_loss = kd_loss_feat_mse(s_feats, t_feats)

                    # ── a*, b* (SSI용) 프레임별 정렬은 그대로 유지 ──
                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_depth_pos, gt_depth_t, mask_t)  # [B,1,1,1]
                    
                    pred_t_aligned_depth = (a_star.detach() * pred_depth_pos.unsqueeze(1) + b_star.detach()).squeeze(1)
                    pred_t_aligned_depth = pred_t_aligned_depth.clamp_min(1e-6)  # 안정화
        
                    ssi_loss_t = (
                        loss_ssi(
                            pred_t_aligned_depth.unsqueeze(1),
                            torch.nan_to_num(gt_depth_t, nan=0.0, posinf=1000.0, neginf=0.0),
                            mask_t.bool()
                        )
                        if (ratio_ssi > 0.0) else (pred_t_raw * 0.0).sum()
                    )
                    
                    # ── TGM: '두 프레임 공통' (a_pair, b_pair) 로 정렬 후 차분 ──
                    if (ratio_tgm > 0.0) and (t > 0) and (prev_pred_depth_pos is not None):
                        # 두 프레임을 가로로 이어붙여서 한 번에 LS 풀기 → 같은 a,b를 두 프레임에 공통 적용
                        # shapes: [B,H,W]
                        pred_pair_cat = torch.cat([prev_pred_depth_pos, pred_depth_pos], dim=-1)          # [B,H,2W]
                        gt_pair_cat   = torch.cat([prev_y.squeeze(2),     gt_depth_t],       dim=-1)      # [B,H,2W]
                        mask_pair_cat = torch.cat([prev_mask[:,0],        mask_t[:,0]],      dim=-1)      # [B,H,2W]
                        
                        with torch.no_grad():
                            a_pair, b_pair = batch_ls_scale_shift(
                                pred_pair_cat, gt_pair_cat, mask_pair_cat
                            )  # [B,1,1,1]
                    
                        pred_aligned_prev = (a_pair.detach() * prev_pred_depth_pos.unsqueeze(1) + b_pair.detach()).squeeze(1)
                        pred_aligned_cur  = (a_pair.detach() * pred_depth_pos.unsqueeze(1)      + b_pair.detach()).squeeze(1)

                        # 디버그: 초반 한 번 페어 RMSE 확인
                        if epoch == start_epoch and batch_idx == 0 and t == 1:
                            m3_bool = (prev_mask[:, 0] > 0.5) & (mask_t[:, 0] > 0.5)
                            m3 = m3_bool.float()
                            rmse_pair_prev = torch.sqrt(((pred_aligned_prev - prev_y.squeeze(2)).pow(2) * m3).sum() / m3.sum().clamp(min=1)).item()
                            rmse_pair_cur  = torch.sqrt(((pred_aligned_cur  - gt_depth_t       ).pow(2) * m3).sum() / m3.sum().clamp(min=1)).item()
                            logger.info(f"[PROBE-TGM] pair RMSE prev={rmse_pair_prev:.4f} cur={rmse_pair_cur:.4f}")
                    
                        pred_pair_depth = torch.stack([pred_aligned_prev, pred_aligned_cur], dim=1)  # [B,2,H,W]
                        gt_pair_depth   = torch.cat([prev_y, y[:, t:t+1]], dim=1).squeeze(2)         # [B,2,H,W]
                        m_pair          = torch.cat([prev_mask, mask_t], dim=1)                      # [B,2,1,H,W] or [B,2,H,W]
                        tgm_loss        = loss_tgm(pred_pair_depth, gt_pair_depth, m_pair)
                    else:
                        tgm_loss = (pred_t_raw * 0.0).sum()
                    
                    # 상태 보관(정렬 전 depth와 마스크/GT만; a,b는 TGM에선 매 스텝 pair로 다시 계산하니 따로 저장 불필요)
                    prev_pred_depth_pos = pred_depth_pos.detach()
                    prev_mask           = mask_t
                    prev_y              = y[:, t:t+1]

                    # ── 손실 합산(안전) ──
                    def _finite(t):
                        return torch.is_tensor(t) and torch.isfinite(t).all()

                    terms = []
                    if ratio_kd  != 0 and _finite(kd_loss):     terms.append(ratio_kd  * kd_loss)
                    if ratio_ssi != 0 and _finite(ssi_loss_t):  terms.append(ratio_ssi * ssi_loss_t)
                    if ratio_tgm != 0 and _finite(tgm_loss):    terms.append(ratio_tgm * tgm_loss)

                    if len(terms) == 0:
                        # 모든 항이 비유한이면 이 프레임은 스킵 (그래프 끊김 방지)
                        frame_pbar.set_postfix_str("skip=nan")
                        wandb.log({"train/skip_step_nan": 1}, commit=False)

                        # 상태만 업데이트 후 다음 프레임으로 (※ depth 버전으로 저장)
                        prev_pred_raw = pred_depth_pos.detach()
                        prev_mask     = mask_t
                        prev_y        = y[:, t:t+1]
                        continue

                    loss = torch.stack(terms).sum()

                    # (옵션) 첫 스텝에서 정렬 전/후 depth RMSE 비교 로그
                    if epoch == start_epoch and batch_idx == 0 and t == 0:
                        gt_depth_3d = torch.nan_to_num(gt_depth_t, nan=0.0)
                        mask3       = mask_t[:, 0].float()  # [B,H,W]

                        # 정렬 전/후 RMSE
                        rmse_raw   = torch.sqrt(((pred_depth_pos - gt_depth_3d).pow(2) * mask3).sum() / mask3.sum().clamp(min=1)).item()
                        rmse_align = torch.sqrt(((pred_t_aligned_depth - gt_depth_3d).pow(2) * mask3).sum() / mask3.sum().clamp(min=1)).item()
                        logger.info(f"[PROBE-DEPTH] RMSE(raw-depth)={rmse_raw:.4f} vs RMSE(aligned-depth)={rmse_align:.4f}")

                # ---- backward/step ----
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # 수동 grad norm/카운트 (클리핑 전)
                _manual_gn_sq = 0.0
                _nonzero_grads = 0
                _forced_params = 0
                for p in model.student.parameters():
                    if p.requires_grad and (p.grad is not None):
                        g = p.grad.detach()
                        if torch.isfinite(g).all():
                            _manual_gn_sq += float((g * g).sum().cpu())
                            if float(g.abs().max().cpu()) > 0.0:
                                _nonzero_grads += 1
                        _forced_params += 1
                _manual_gn = (_manual_gn_sq ** 0.5) if _manual_gn_sq > 0 else 0.0
                wandb.log({"train/grad_norm_manual": _manual_gn}, commit=False)

                total_norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                # ---- 통계 누적 ----
                B_eff = pred_t_raw.shape[0]
                wSSI_s = float((ratio_ssi * ssi_loss_t).detach().cpu())
                wTGM_s = float((ratio_tgm * tgm_loss).detach().cpu())
                _kd_tensor = kd_loss if torch.is_tensor(kd_loss) else torch.as_tensor(kd_loss, device=pred_t_raw.device, dtype=pred_t_raw.dtype)
                wKD_s  = float((ratio_kd * _kd_tensor).detach().cpu())

                batch_frames += B_eff
                batch_wSSI   += wSSI_s * B_eff
                batch_wTGM   += wTGM_s * B_eff
                batch_wKD    += wKD_s  * B_eff

                # epoch_frames += B_eff
                epoch_wSSI   += wSSI_s * B_eff
                epoch_wTGM   += wTGM_s * B_eff
                epoch_wKD    += wKD_s  * B_eff
                # epoch_loss   += float(loss.detach().cpu())

                # 새로
                epoch_frames   += B_eff
                epoch_steps    += 1
                _litem          = float(loss.detach().cpu())
                epoch_loss_sum += _litem
                epoch_loss_wsum += _litem * B_eff

                frame_pbar.set_postfix({
                    "t": f"{t+1}/{T}",
                    "wSSI_s": f"{wSSI_s:.4f}",
                    "wTGM_s": f"{wTGM_s:.4f}",
                    "wKD_s":  f"{wKD_s:.4f}",
                    "wSSI_b": f"{batch_wSSI / max(1, batch_frames):.4f}",
                    "wTGM_b": f"{batch_wTGM / max(1, batch_frames):.4f}",
                    "wKD_b":  f"{batch_wKD  / max(1, batch_frames):.4f}",
                    "gn_pre": f"{_manual_gn:.2e}",
                    "gn":     f"{(total_norm.item() if hasattr(total_norm, 'item') else float(total_norm)):.2e}",
                    "nz":     f"{_nonzero_grads}/{_forced_params}",
                })

                step_id = epoch * len(train_loader) * T + batch_idx * T + t
                if (step_id % log_every) == 0:
                    wandb.log({
                        "train/grad_norm_step": float(total_norm),
                        "train/kd_loss_step": wKD_s / max(1.0 * ratio_kd, 1e-6),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }, commit=False)

                prev_pred_raw = pred_depth_pos.detach()  # 정렬 전 'depth'를 저장
                prev_mask = mask_t
                prev_y    = y[:, t:t+1]
                
                if epoch == start_epoch and batch_idx == 0 and t == 0:
                    logger.info(f"[DEBUG] loss.requires_grad={loss.requires_grad}, "
                                f"pred_depth_pos.requires_grad={pred_depth_pos.requires_grad}, "
                                f"kd_mode={args.kd_mode}")
                step_skipped = (_nonzero_grads == 0)
                if step_skipped:
                    frame_pbar.set_postfix_str("step_skipped=1")
                    wandb.log({"train/step_skipped": 1}, commit=False)

            frame_pbar.close()

            # ---- 배치 종료: 학생 캐시 정리/비활성 ----
            if student_kd_active:
                _ = collect_kd_caches(model.student, clear=True)
                disable_attention_caching(model.student)

        batch_pbar.close()

        # --- Validation ---
        import shutil
        val_infer_dir_epoch = args.val_infer_dir
        if os.path.isdir(val_infer_dir_epoch):
            shutil.rmtree(val_infer_dir_epoch)
        os.makedirs(val_infer_dir_epoch, exist_ok=True)

        _prev_train_state = model.student.training
        model.student.eval()
        with torch.no_grad():
            val_metrics = validate_with_infer_eval_subset_fast(
                model=model.student,
                json_file=args.val_json_file,
                infer_path=val_infer_dir_epoch,
                dataset=args.val_dataset_key,
                dataset_eval_tag=args.val_dataset_tag,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                input_size=518,
                scenes_to_eval=args.val_scenes,
                scene_indices=[1, 39, 44, 93],
                frame_stride=2,
                max_eval_len=500,
                fp32=False
            )
        if _prev_train_state:
            model.student.train()

        val_absrel = float(val_metrics.get("abs_relative_difference", float('nan')))
        val_rmse   = float(val_metrics.get("rmse_linear", float('nan')))
        val_delta1 = float(val_metrics.get("delta1_acc", float('nan')))

        wandb.log({
            # "train/loss": epoch_loss / max(1, len(train_loader)),
            "train/loss_step":       epoch_loss_sum / max(1, epoch_steps),   # 프레임 평균
            "train/loss_per_sample": epoch_loss_wsum / max(1, epoch_frames), # 샘플 가중 평균
            "train/wSSI_epoch": epoch_wSSI / max(1, epoch_frames),
            "train/wTGM_epoch": epoch_wTGM / max(1, epoch_frames),
            "train/wKD_epoch":  epoch_wKD  / max(1, epoch_frames),
            "val/absrel": val_absrel,
            "val/rmse":   val_rmse,
            "val/delta1": val_delta1,
            "epoch": epoch,
            "stage2": int(in_stage2),
            "teacher_keep_prob": use_teacher_prob,
        })
        
        # 씬별 로깅 추가
        for sc in val_metrics.get("per_scene", []):
            wandb.log({
                f"val/scene_{sc['scene_idx']}/absrel": sc["abs_relative_difference"],
                f"val/scene_{sc['scene_idx']}/rmse":   sc["rmse_linear"],
                f"val/scene_{sc['scene_idx']}/delta1": sc["delta1_acc"],
                "val/scene_key": sc["scene_key"],
                "epoch": epoch
            }, commit=False)

        if val_delta1 > best_delta1:
            best_delta1 = val_delta1
            best_epoch  = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_delta1": best_delta1,
                "config": hp,
            }, best_model_path)
            logger.info(f"🏆 Best model saved! Epoch {epoch}, Val delta1: {best_delta1:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_absrel": val_absrel,
            "val_delta1": val_delta1,
            "val_rmse":   val_rmse,
            "config": hp,
        }, latest_model_path)
        logger.info(f"📁 Latest model saved to {latest_model_path}")

        torch.cuda.empty_cache()
        scheduler.step()

    logger.info("=" * 30)
    logger.info("Training Completed!")
    logger.info(f"Total Epochs: {num_epochs}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best Val delta1: {best_delta1:.6f}")
    logger.info(f"Best: {best_model_path}")
    logger.info(f"Latest: {latest_model_path}")
    logger.info("=" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")

    # validation 설정
    parser.add_argument("--val_json_file",    type=str, default="/workspace/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json")
    parser.add_argument("--val_infer_dir",    type=str, default="benchmark/output/scannet_stream_valmini")
    parser.add_argument("--val_dataset_key",  type=str, default="scannet")
    parser.add_argument("--val_dataset_tag",  type=str, default="scannet_500")
    parser.add_argument("--val_scenes",       type=int, default=4)

    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--log_every", type=int, default=50)

    # KD / Trainable 모드
    parser.add_argument("--kd_mode", type=str, default="cache", choices=["cache", "pred", "feat"])
    parser.add_argument("--trainable_mode", type=str, default="all_but_encoder",
                        choices=["default", "head", "all_but_encoder", "patterns"])
    parser.add_argument("--trainable_patterns", type=str, default="")

    args = parser.parse_args()
    train(args)
