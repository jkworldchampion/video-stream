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

from torch.utils.data import DataLoader, ConcatDataset, Subset
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
experiment = 37
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

def enable_kv_memory_adapter(m):
    for _, layer in m.named_modules():
        if hasattr(layer, 'enable_kv_memory'):
            layer.enable_kv_memory(True)

def disable_kv_memory_adapter(m):
    for _, layer in m.named_modules():
        if hasattr(layer, 'enable_kv_memory'):
            layer.enable_kv_memory(False)

def collect_kd_caches(module, clear: bool = True):
    """
    returns: list[dict]
      dict keys (motion_module.py 쪽에서 저장):
        - 'context': [B, P, C], B=batch, P=tokens_per_frame, C=embed_dim
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

def load_state_dict_with_log(module: torch.nn.Module, sd: dict, model_name: str, strict: bool = False):
    """
    state_dict 로딩 시 누락/예상치 못한 키와 일치한 텐서 개수를 로깅한다.
    strict=False 권장(신규 레이어가 체크포인트에 없을 수 있음).
    """
    try:
        result = module.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        # shape mismatch 등 오류가 나면 strict=False로 강등하여 재시도
        logger.warning(f"[{model_name}] strict={strict} load failed: {e}. Retrying with strict=False.")
        result = module.load_state_dict(sd, strict=False)

    model_sd = module.state_dict()
    total = len(model_sd)
    # 모양까지 일치하는 매치 개수
    matched = 0
    for k, v in model_sd.items():
        if k in sd and isinstance(sd[k], torch.Tensor) and v.shape == sd[k].shape:
            matched += 1

    missing = getattr(result, 'missing_keys', [])
    unexpected = getattr(result, 'unexpected_keys', [])

    logger.info(f"[{model_name}] loaded {matched}/{total} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    # 너무 길어질 수 있으니 처음 몇 개만 미리보기
    if len(missing) > 0:
        logger.info(f"[{model_name}] missing keys (preview): {missing[:8]}")
    if len(unexpected) > 0:
        logger.info(f"[{model_name}] unexpected keys (preview): {unexpected[:8]}")
    return result

def compute_kd_losses_from_caches(
    t_cache: dict, s_cache: dict,
    top_r: int = 64, top_u: int = 32,  # 유지(호환용, 현재 로직에서는 사용하지 않음)
    w_attn_kl: float = 1.0,            # Q 정합 가중치로 사용
    w_kv_cos: float = 1.0,             # K/V 정합 가중치로 사용
    w_ctx_cos: float = 0.0,
    eps: float = 1e-8
):
    """
    새 프레임 생성 시점의 Q/K/V를 교사처럼 생성하도록 직접 감독.
    - Teacher: 32-frame window, Student: streaming 1-step
    - 비교 대상: 마지막 프레임 토큰에 해당하는 Q/K/V
    - 손실: 정규화 후 MSE(Q) + 정규화 후 MSE(K_last) + 정규화 후 MSE(V_last)
      (가중치: w_attn_kl → Q, w_kv_cos → K/V 합)
    - 폴백: q_bh/k_bh/v_bh가 없으면 context/K/V 집계를 사용
    """

    def _bh_to_b_mean_head(x_bh, H):
        # x_bh: [B*H, T, D], return: [B, T, D]
        if x_bh is None:
            return None
        assert H > 0, "num_heads must be > 0 to reshape BH->(B,H)"
        B_eff = x_bh.shape[0] // H
        T = x_bh.shape[1]
        D = x_bh.shape[2]
        return x_bh.view(B_eff, H, T, D).mean(dim=1)

    def _get_last_frame_qkv(cache: dict):
        """
        반환:
          Q_last: [B, Pq, D] 또는 None  (쿼리 토큰 전체; 보통 Pq==P)
          K_last: [B, Pk, D] 또는 None  (마지막 프레임 키 토큰들)
          V_last: [B, Pk, D] 또는 None  (마지막 프레임 값 토큰들)
          C_ctx : [B, P,  D] 또는 None  (옵션: context)
        우선 q_bh/k_bh/v_bh 사용, 없으면 context/K/V 집계 폴백.
        """
        H = int(cache.get("num_heads", 0))
        P = int(cache.get("tokens_per_frame", 0))
        q_bh = cache.get("q_bh", None)   # [B*H, Pq, Dh]
        k_bh = cache.get("k_bh", None)   # [B*H, L,  Dh]
        v_bh = cache.get("v_bh", None)   # [B*H, L,  Dh]
        ctx   = cache.get("context", None)   # [B,P,D]
        K_agg = cache.get("K", None)         # [B,P,Klen,D]
        V_agg = cache.get("V", None)

        Q_last = K_last = V_last = None

        if (q_bh is not None) and (k_bh is not None) and (v_bh is not None) and (H > 0):
            Q_last = _bh_to_b_mean_head(q_bh, H)  # [B,Pq,D]
            # 마지막 프레임 키/값: K/V의 마지막 P 토큰 사용 (P가 없으면 길이 기반으로 추정)
            L = k_bh.shape[1]
            last_p = P if (P > 0 and P <= L) else min(L, P if P > 0 else L)
            K_last = _bh_to_b_mean_head(k_bh[:, -last_p:, :], H)  # [B,last_p,D]
            V_last = _bh_to_b_mean_head(v_bh[:, -last_p:, :], H)
        else:
            # 폴백 경로: Q≈context, K/V는 집계 텐서에서 마지막 P 토큰 취함
            if ctx is not None:
                Q_last = ctx  # [B,P,D]
            if (K_agg is not None) and (V_agg is not None):
                Bk, Pk, Klen, Dk = K_agg.shape
                last_p = P if (P > 0 and P <= Klen) else min(Klen, P if P > 0 else Klen)
                # 쿼리 평균 후 마지막 프레임 키/값 토큰 선택
                K_mean_q = K_agg.mean(dim=1)           # [B,Klen,D]
                V_mean_q = V_agg.mean(dim=1)
                K_last = K_mean_q[:, -last_p:, :]      # [B,last_p,D]
                V_last = V_mean_q[:, -last_p:, :]

        return Q_last, K_last, V_last, ctx

    # Q/K/V 추출
    t_Q, t_K_last, t_V_last, t_ctx = _get_last_frame_qkv(t_cache)
    s_Q, s_K_last, s_V_last, s_ctx = _get_last_frame_qkv(s_cache)

    # 기준 디바이스 선택
    base_src = next((x for x in [t_Q, t_K_last, t_V_last, s_Q, s_K_last, s_V_last, t_ctx, s_ctx] if x is not None), None)
    if base_src is None:
        return torch.tensor(0.0), {}
    device_ = base_src.device

    total = torch.zeros((), device=device_)
    losses = {}

    # Q 정합 (정규화 MSE)
    if (t_Q is not None) and (s_Q is not None):
        Lq = min(t_Q.shape[1], s_Q.shape[1])
        if Lq > 0:
            q_loss = F.mse_loss(_safe_norm(t_Q[:, :Lq, :], dim=-1), _safe_norm(s_Q[:, :Lq, :], dim=-1))
            total += w_attn_kl * q_loss
            losses['q_cos_mse'] = q_loss

    # K/V 정합 (마지막 프레임 토큰들, 정규화 MSE)
    kv_parts = []
    if (t_K_last is not None) and (s_K_last is not None):
        Lk = min(t_K_last.shape[1], s_K_last.shape[1])
        if Lk > 0:
            k_loss = F.mse_loss(_safe_norm(t_K_last[:, :Lk, :], dim=-1), _safe_norm(s_K_last[:, :Lk, :], dim=-1))
            kv_parts.append(k_loss)
            losses['k_last_cos_mse'] = k_loss
    if (t_V_last is not None) and (s_V_last is not None):
        Lv = min(t_V_last.shape[1], s_V_last.shape[1])
        if Lv > 0:
            v_loss = F.mse_loss(_safe_norm(t_V_last[:, :Lv, :], dim=-1), _safe_norm(s_V_last[:, :Lv, :], dim=-1))
            kv_parts.append(v_loss)
            losses['v_last_cos_mse'] = v_loss
    if len(kv_parts) > 0:
        kv_loss = torch.stack(kv_parts).mean()
        total += w_kv_cos * kv_loss
        losses['kv_last_cos_mse'] = kv_loss

    # (옵션) Context 정합
    if (t_ctx is not None) and (s_ctx is not None) and (w_ctx_cos > 0.0):
        # P 길이가 다르면 최소 길이에 맞춰 비교
        Pc = min(t_ctx.shape[1], s_ctx.shape[1])
        if Pc > 0:
            ctx_cos = 1.0 - F.cosine_similarity(t_ctx[:, :Pc, :].flatten(0,1), s_ctx[:, :Pc, :].flatten(0,1), dim=-1).mean()
            total += w_ctx_cos * ctx_cos
            losses['ctx_cos'] = ctx_cos

    return total, losses

def _extract_kv_per_frame(cache: dict):
    """
    cache expects keys: 'K': [B,P,Klen,C], 'V': [B,P,Klen,C], 'tokens_per_frame': int(P)
    Returns: Kf, Vf with shape [B, L, C] where L = Klen // P (frames in window)
    """
    K = cache.get("K", None)
    V = cache.get("V", None)
    if (K is None) or (V is None):
        return None, None
    B, P, Klen, C = K.shape
    P0 = int(cache.get("tokens_per_frame", P))
    if P0 <= 0:
        P0 = P
    if Klen % P0 != 0:
        L = Klen // P0
        Klen_adj = L * P0
        K = K[:, :, :Klen_adj, :]
        V = V[:, :, :Klen_adj, :]
    L = K.shape[2] // P0
    # 평균: 먼저 쿼리 토큰 차원(P) 평균 → [B,Klen,C], 이후 프레임별로 다시 평균 → [B,L,C]
    K_mean_q = K.mean(dim=1)
    V_mean_q = V.mean(dim=1)
    Kf = K_mean_q.view(B, L, P0, C).mean(dim=2)
    Vf = V_mean_q.view(B, L, P0, C).mean(dim=2)
    return Kf, Vf

def _delta_window_shift(prev_seq: torch.Tensor, curr_seq: torch.Tensor):
    """
    prev_seq, curr_seq: [B, L, C]
    Return aligned window-shift delta: curr[:, :L-1] - prev[:, 1:]
    Shape: [B, L_eff, C] with L_eff = min(L_prev, L_curr) - 1
    """
    if (prev_seq is None) or (curr_seq is None):
        return None
    L_eff = min(prev_seq.shape[1], curr_seq.shape[1]) - 1
    if L_eff <= 0:
        return None
    return curr_seq[:, :L_eff, :] - prev_seq[:, 1:1+L_eff, :]

def compute_kv_delta_distill(prev_t: dict, curr_t: dict, prev_s: dict, curr_s: dict, w: float = 0.0, eps: float = 1e-8):
    """
    보조 distill: 윈도우 시프트에 따른 KV 델타(교사↔학생)를 직접 감독.
    - 각 캐시에서 프레임별 평균 K,V 시퀀스([B,L,C])를 추출
    - 델타(prev→curr 윈도우 시프트) 정렬 후 Cosine MSE로 비교
    반환: scalar loss (Tensor)
    """
    if (w is None) or (float(w) <= 0.0):
        return None
    try:
        Kt_prev, Vt_prev = _extract_kv_per_frame(prev_t) if prev_t is not None else (None, None)
        Kt_curr, Vt_curr = _extract_kv_per_frame(curr_t) if curr_t is not None else (None, None)
        Ks_prev, Vs_prev = _extract_kv_per_frame(prev_s) if prev_s is not None else (None, None)
        Ks_curr, Vs_curr = _extract_kv_per_frame(curr_s) if curr_s is not None else (None, None)
        dKt = _delta_window_shift(Kt_prev, Kt_curr)
        dVt = _delta_window_shift(Vt_prev, Vt_curr)
        dKs = _delta_window_shift(Ks_prev, Ks_curr)
        dVs = _delta_window_shift(Vs_prev, Vs_curr)
        parts = []
        if (dKt is not None) and (dKs is not None):
            # 정규화 후 MSE
            dKt_n = F.normalize(dKt, dim=-1, eps=eps)
            dKs_n = F.normalize(dKs, dim=-1, eps=eps)
            L_eff = min(dKt_n.shape[1], dKs_n.shape[1])
            parts.append(F.mse_loss(dKs_n[:, :L_eff], dKt_n[:, :L_eff]))
        if (dVt is not None) and (dVs is not None):
            dVt_n = F.normalize(dVt, dim=-1, eps=eps)
            dVs_n = F.normalize(dVs, dim=-1, eps=eps)
            L_eff = min(dVt_n.shape[1], dVs_n.shape[1])
            parts.append(F.mse_loss(dVs_n[:, :L_eff], dVt_n[:, :L_eff]))
        if len(parts) == 0:
            return None
        return float(w) * torch.stack(parts).mean()
    except Exception:
        return None

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

# VKITTI + GTA를 만들기 위함
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
    w_kv_delta = hyper_params.get("w_kv_delta", 0.0)  # NEW: KV 델타 보조 distill 가중치

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

    logger.info(f"train_VKITTI_total_clips : {len(vkitti_ds)}")
    logger.info(f"train_GTA_total_clips    : {len(gta_ds)}")

    # 태그
    vkitti_tagged = TagDataset(vkitti_ds, tag_int=0)
    gta_tagged    = TagDataset(gta_ds,    tag_int=1)

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
        # 체크포인트는 원래의 base 모델 구조일 가능성이 높음 → strict=False로 로드하고 누락/예상 키 로그
        load_state_dict_with_log(model.teacher, sd, model_name="teacher", strict=False)
        load_state_dict_with_log(model.student, sd, model_name="student", strict=False)

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
    
    if not args.test:
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
                input_size=518,               # 4 scenes subset
                scene_indices=[1, 39, 44, 93],                # 적당히 보고 평균3개 + 극단값1개로 구성함.
                fp32=True,
            )
        finally:
            # 원래 학습 모드 복귀
            if _prev_train_state:
                model.student.train()

        init_avg = init_metrics.get("avg", {})
        per_scene = init_metrics.get("per_scene", [])
        init_absrel = float(init_avg.get("abs_relative_difference", float('nan')))
        init_rmse   = float(init_avg.get("rmse_linear", float('nan')))
        init_delta1 = float(init_avg.get("delta1_acc", float('nan')))
        
        # 콘솔/파일 로그
        logger.info(f"[Init] real-pipeline val  | absrel={init_absrel:.4f}  rmse={init_rmse:.4f}  delta1={init_delta1:.4f}")

        # W&B 로깅 (epoch=-1로 표기)
        wandb.log({
            "init/absrel": init_absrel,
            "init/rmse":   init_rmse,
            "init/delta1": init_delta1,
            "epoch": -1,
        })
        # 씬별 로그 (각 씬을 별도 키로 기록)
        for idx, m in enumerate(per_scene):
            # scene 이름이 있으면 키에 포함
            scene_name = m.get("scene", str(idx))
            wandb.log({
                f"init/scene_{idx}/scene": scene_name,
                f"init/scene_{idx}/absrel": float(m.get("abs_relative_difference", float('nan'))),
                f"init/scene_{idx}/rmse":   float(m.get("rmse_linear", float('nan'))),
                f"init/scene_{idx}/delta1": float(m.get("delta1_acc", float('nan'))),
                "epoch": -1,
            })

        # 베스트 기준을 초기값으로 시작하고 싶다면(권장)
        best_delta1 = init_delta1


    # --------------------- Training ---------------------
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False, dynamic_ncols=True):
        # data augmentation 해서, 많은 데이터로 학습 ㄱㄱ
        if hasattr(vkitti_ds, "set_epoch"): vkitti_ds.set_epoch(epoch)
        if hasattr(gta_ds, "set_epoch"):    gta_ds.set_epoch(epoch)
        
        data_size = 3 if args.test else 100
        train_dataset = ConcatDataset([
            make_random_subset(vkitti_tagged, data_size),
            make_random_subset(gta_tagged,    data_size)
        ])
        train_loader  = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
        )
        
        model.train()
        epoch_loss = epoch_frames = 0.0
        epoch_ssi = epoch_tgm = epoch_kd = 0.0
        accum_loss = 0.0
        step_in_window = 0
        update_frequency = hyper_params.get("update_frequency", 6)

        # Stage-2 스케줄
        in_stage2 = (epoch >= num_epochs - stage2_epochs)
        use_teacher_prob = (0.0 if not in_stage2 else (1.0 - teacher_drop_p_stage2))

        batch_pbar = tqdm(enumerate(train_loader),
                          desc=f"Epoch {epoch+1}/{num_epochs} - Batches",
                          total=len(train_loader),
                          leave=False, dynamic_ncols=True)
        
        for batch_idx, batch in batch_pbar:
            # (x, y, tag)
            if len(batch) == 3:
                x, y, tag = batch  # dataset 인식을 위한 tag, 각 shape은 (T,3,H,W), (T,1,H,W), int
                tag = tag.to(device, non_blocking=True)
            else:
                x, y = batch
                tag  = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, T = x.shape[:2]

            cache = None
            prev_pred_raw = None
            prev_y = None
            prev_mask_bool4 = None
            teacher_frame_buffer = None
            prev_t_cache = None  # for KV-delta distill
            prev_s_cache = None

            # 배치 내 도메인 개수 (로깅용)
            vk_count = int((tag == 0).sum().item())
            gta_count = B - vk_count

            frame_pbar = tqdm(range(T), desc=f"Batch {batch_idx+1} - Frames", leave=False, disable=T < 10)
            for t in frame_pbar:
                x_t = x[:, t:t+1]          # [B,1,3,H,W]
                y_t = y[:, t:t+1]          # [B,1,1,H,W]  depth

                # VKITTI=0, GTA=1
                tag_vkit = (tag == 0).view(-1, 1, 1, 1, 1).to(device)  # [B,1,1,1,1], bool

                # 깊이 범위 (데이터셋별 기준), 일단 임의로 판단함. 대략 60%정도를 target함
                min_d = torch.where(tag_vkit,
                                    torch.full_like(y_t,  5.0),
                                    torch.full_like(y_t, 30.0))
                max_d = torch.where(tag_vkit,
                                    torch.full_like(y_t, 120.0),
                                    torch.full_like(y_t, 1000.0))

                # --- 마스크 생성 ---
                mask_bool_5d = (y_t >= min_d) & (y_t <= max_d)       # [B,1,1,H,W] bool
                mask_bool_4d = mask_bool_5d.squeeze(2)               # [B,1,H,W]   bool
                mask_float_4d = mask_bool_4d.float()                 # [B,1,H,W]   float
                mask_t = mask_bool_5d.float()                        # [B,1,1,H,W] float (OLS용)

                # --- 프레임별 디버깅 ---
                # raw depth 통계
                y_min = float(y_t.min().item())
                y_max = float(y_t.max().item())

                # masked depth 통계
                valid_pix = int(mask_bool_5d.sum().item())
                total_pix = mask_bool_5d.numel()
                valid_pct = 100.0 * valid_pix / max(1, total_pix)

                if valid_pix > 0:
                    y_masked = y_t[mask_bool_5d]                     # 1D 텐서
                    y_mask_min = float(y_masked.min().item())
                    y_mask_max = float(y_masked.max().item())
                else:
                    y_mask_min = float('nan')
                    y_mask_max = float('nan')

                # 데이터 도메인별 커버리지
                vk_total_pix = int(tag_vkit.expand_as(mask_bool_5d).to(torch.int).sum().item())
                gta_total_pix = total_pix - vk_total_pix
                vk_valid_pix  = int((mask_bool_5d & tag_vkit).to(torch.int).sum().item())
                gta_valid_pix = valid_pix - vk_valid_pix
                vk_valid_pct  = 100.0 * vk_valid_pix  / max(1, vk_total_pix)
                gta_valid_pct = 100.0 * gta_valid_pix / max(1, gta_total_pix)

                logger.info(
                    f"[DEBUG] t={t} raw=({y_min:.3g},{y_max:.3g}) "
                    f"masked=({y_mask_min:.3g},{y_mask_max:.3g}) "
                    f"valid={valid_pix}/{total_pix} ({valid_pct:.2f}%) | "
                    f"VK valid={vk_valid_pix}/{vk_total_pix} ({vk_valid_pct:.2f}%), "
                    f"GTA valid={gta_valid_pix}/{gta_total_pix} ({gta_valid_pct:.2f}%)"
                )

                # --- Teacher window 준비 ---
                if teacher_frame_buffer is None:
                    teacher_frame_buffer = x_t.detach().clone().repeat(1, CLIP_LEN, 1, 1, 1)  # buffer가 없으면, 현재 프레임으로 채우기
                else:
                    # 그냥 앵커 없애보기
                    teacher_frame_buffer = torch.cat([
                        teacher_frame_buffer[:, 1:],          # 앞에서 한 칸 밀기
                        x_t.detach().clone()
                    ], dim=1)

                # === Teacher (정확 KD 캐시 수집) ===
                use_teacher = (random.random() < use_teacher_prob) or (not in_stage2)
                if use_teacher:
                    with torch.no_grad():
                        enable_attention_caching(model.teacher)
                        with autocast(enabled=torch.cuda.is_available()):
                            _ = model.teacher(teacher_frame_buffer)  # depth-map은 사용x
                        t_caches = collect_kd_caches(model.teacher, clear=True)  # cache 수집
                        disable_attention_caching(model.teacher)
                        if len(t_caches) == 0:
                            raise RuntimeError("No teacher KD caches collected.")
                        t_cache_last = {k:(v.to(device) if torch.is_tensor(v) else v) for k, v in t_caches[-1].items()}

                        # ===== 디버깅: Teacher cache shape 로깅 (첫 번째 batch의 첫 번째 프레임만) =====
                        if epoch == start_epoch and batch_idx == 0 and t == 0:
                            logger.info("=== Teacher KD Cache Shapes ===")
                            for key, val in t_cache_last.items():
                                if torch.is_tensor(val):
                                    logger.info(f"  {key}: {val.shape}")
                                else:
                                    logger.info(f"  {key}: {type(val)} (non-tensor)")
                else:
                    t_cache_last = None

                # === Student (정확 KD 캐시 수집 + 스트리밍 1-step) ===
                with autocast(enabled=torch.cuda.is_available()):
                    enable_attention_caching(model.student)
                    # 학생에게만 KV 메모리 어댑터 적용 (티처는 오프라인 윈도우로 충분)
                    enable_kv_memory_adapter(model.student)
                    pred_t_raw, cache = model_stream_step(model.student, x_t, cache)  # pred: [B,H,W]
                    pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)
                    s_caches = collect_kd_caches(model.student, clear=True)
                    disable_attention_caching(model.student)
                    disable_kv_memory_adapter(model.student)
                    if len(s_caches) == 0:
                        raise RuntimeError("No student KD caches collected.")
                    s_cache_last = s_caches[-1]

                    # ===== 디버깅: Student cache shape 로깅 (첫 번째 batch의 첫 번째 프레임만) =====
                    if epoch == start_epoch and batch_idx == 0 and t == 0:
                        logger.info("=== Student KD Cache Shapes ===")
                        for key, val in s_cache_last.items():
                            if torch.is_tensor(val):
                                logger.info(f"  {key}: {val.shape}")
                            else:
                                logger.info(f"  {key}: {type(val)} (non-tensor)")

                    # ----- KD 손실 -----
                    if use_teacher:
                        kd_total, kd_parts = compute_kd_losses_from_caches(
                            t_cache_last, s_cache_last,
                            top_r=kd_top_r, top_u=kd_top_u,
                            w_attn_kl=w_attn_kl, w_kv_cos=w_kv_cos, w_ctx_cos=w_ctx_cos
                        )
                        kd_loss = kd_weight * kd_total
                        # (옵션) KV-델타 보조 distill: 이전-현재 윈도우 시프트 델타를 감독
                        if w_kv_delta > 0.0 and (prev_t_cache is not None) and (prev_s_cache is not None):
                            kv_delta_loss = compute_kv_delta_distill(prev_t_cache, t_cache_last, prev_s_cache, s_cache_last, w=w_kv_delta)
                            if kv_delta_loss is not None:
                                kd_loss = kd_loss + kv_delta_loss
                    else:
                        s_attn = s_cache_last.get("attn", None)
                        if (s_attn is not None) and (attn_ent_w > 0):
                            ent = attention_entropy(s_attn)
                            kd_loss = (-attn_ent_w) * ent
                        else:
                            kd_loss = pred_t_raw.new_tensor(0.0)

                    # ----- Depth Loss -----
                    disp_normed_t = norm_ssi(y[:, t:t+1], mask_bool_5d).squeeze(2)  # [B,1,H,W], normed disparity
                    gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6)).squeeze(2)  # [B,1,H,W], align용

                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)  # mask_t: [B,1,1,H,W] float

                    pred_t_aligned_disp   = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)  # [B,1,H,W] -> [B,H,W]
                    pred_t_aligned_depth  = 1.0 / (pred_t_aligned_disp.clamp(min=1e-6))

                    # SSI 손실 (4D float 마스크)
                    ssi_loss_t = loss_ssi(
                        pred_t_aligned_disp.unsqueeze(1),  # [B,1,H,W]
                        disp_normed_t,                     # [B,1,H,W]
                        mask_float_4d                      # [B,1,H,W]
                    )

                    # TGM 손실 (t>0일 때만, bool 4D 마스크)
                    if t > 0:
                        # 이전 프레임 정렬값 계산은 여기서!
                        prev_aligned_disp  = (a_star.detach() * prev_pred_raw.unsqueeze(1) + b_star.detach()).squeeze(1)
                        prev_aligned_depth = 1.0 / (prev_aligned_disp.clamp(min=1e-6))
                        curr_aligned_depth = pred_t_aligned_depth

                        pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)  # [B,2,H,W]
                        y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)                       # [B,2,1,H,W]

                        m_pair_bool4 = torch.cat([prev_mask_bool4, mask_bool_4d], dim=1)          # [B,2,H,W]
                        tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair_bool4)
                    else:
                        tgm_loss  = pred_t_raw.new_tensor(0.0)

                    loss = kd_loss + ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss

                # --- 누적/업데이트 ---
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

                # --- 상태 업데이트 ---
                cache = _detach_cache(cache)
                prev_pred_raw   = pred_t_raw.detach()
                prev_mask_bool4 = mask_bool_4d
                prev_y          = y[:, t:t+1]
                # 보조 distill용 캐시 보관(teacher 사용 시에만)
                prev_t_cache = t_cache_last if use_teacher else prev_t_cache
                prev_s_cache = s_cache_last

                # --- 통계 ---
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
            scene_indices=[1, 39, 44, 93],
            fp32=True
        )
        val_avg = val_metrics.get("avg", {})
        per_val_scene = val_metrics.get("per_scene", [])
        val_absrel = float(val_avg.get("abs_relative_difference", float('nan')))
        val_rmse   = float(val_avg.get("rmse_linear", float('nan')))
        val_delta1 = float(val_avg.get("delta1_acc", float('nan')))

        # 로깅
        wandb.log({
            "train/loss": epoch_loss / max(1, len(train_loader)),
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
        # 씬별 로그 (각 씬을 별도 키로 기록)
        for idx, m in enumerate(per_val_scene):
            scene_name = m.get("scene", str(idx))
            wandb.log({
                f"val_real/scene_{idx}/scene": scene_name,
                f"val_real/scene_{idx}/absrel": float(m.get("abs_relative_difference", float('nan'))),
                f"val_real/scene_{idx}/rmse":   float(m.get("rmse_linear", float('nan'))),
                f"val_real/scene_{idx}/delta1": float(m.get("delta1_acc", float('nan'))),
                "epoch": epoch,
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
    parser.add_argument("--resume_from", type=str, default="", help="Path to latest/best checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs (e.g., 60)")
    parser.add_argument("--test", action='store_true', help="Run a quick test with reduced data and epochs")
    args = parser.parse_args()
    train(args)
