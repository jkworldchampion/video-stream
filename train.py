import os
import argparse
import logging
import random

import torch
import torch.nn.functional as F
from torch import amp
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
experiment = 315
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

def _frame_vec_from_tokens(feat_bt_pc, use_cls=False):
    # feat_bt_pc: [B*T, P, C]
    return feat_bt_pc[:, 0, :] if use_cls else feat_bt_pc[:, 1:, :].mean(dim=1)

def _to_BT_C(feat_bt_pc, B, T, use_cls=False):
    return _frame_vec_from_tokens(feat_bt_pc, use_cls=use_cls).view(B, T, -1)

def feat_kd_dis(h, z):  # [B,T,C] each
    h = F.layer_norm(h, h.shape[-1:])
    z = F.layer_norm(z, z.shape[-1:])
    l1  = (h - z).abs().mean(dim=-1)                       # [B,T]
    cos = F.cosine_similarity(h, z, dim=-1)                # [B,T]
    return (l1 - torch.log(torch.sigmoid(cos))).mean()

def apc_feat(h, r, N=3):  # h: teacher [B,T,C], r: aux [B,T,C]
    if N <= 0: return torch.zeros([], device=h.device)
    hN = F.layer_norm(h[:, N:, :], h.shape[-1:])
    r0 = F.layer_norm(r[:, :-N, :], r.shape[-1:])
    l1  = (hN - r0).abs().mean(dim=-1)
    cos = F.cosine_similarity(hN, r0, dim=-1)
    return (l1 - torch.log(torch.sigmoid(cos))).mean()

def kld_attn(Rt, Rs, eps=1e-6):  # [B,A,T,T], KL(Rt||Rs) — (옵션)
    Rt = Rt.clamp_min(eps); Rs = Rs.clamp_min(eps)
    return (Rt * (Rt.log() - Rs.log())).sum(dim=-1).mean()

def slice_features_list(feats_full, B, start, end):
    """
    feats_full: list of [B*T_total, P, C]
    반환: (feats_win, Lw)  with feats_win: list of [B*Lw, P, C]
    """
    Lw = end - start
    sliced = []
    for lvl, f in enumerate(feats_full):
        if f.dim() != 3:
            raise ValueError(f"[slice_features_list] level{lvl} expects 3D [N,P,C], got {tuple(f.shape)}")
        N, P, C = f.shape
        if N % B != 0:
            raise ValueError(f"[slice_features_list] N({N}) not divisible by B({B}). level={lvl}")
        T_total = N // B
        if not (0 <= start < end <= T_total):
            raise ValueError(f"[slice_features_list] invalid window [{start},{end}) for T_total={T_total}")

        # [B, T_total, P, C] → slice T → [B, Lw, P, C] → [B*Lw, P, C]
        f = f.view(B, T_total, P, C)[:, start:end, :, :].contiguous()
        f = f.view(B * Lw, P, C).contiguous()
        sliced.append(f)
    return sliced, Lw

def featurize_in_time_chunks(vda_model, x, chunk_len):
    # x: [B, L, 3, H, W]  → returns list of [B*L, P, C]
    B, L = x.shape[:2]
    outs = None
    for s in range(0, L, chunk_len):
        e = min(L, s + chunk_len)
        feats = vda_model.forward_features(x[:, s:e])  # list of [B*(e-s), P, C]
        if outs is None:
            outs = [f for f in feats]
        else:
            outs = [torch.cat([o, f], dim=0) for o, f in zip(outs, feats)]
    return outs

def _teacher_forward_features(vda_teacher, x_chunk, *, return_class_token=False):
    """
    VideoDepthAnything teacher에서 intermediates를 추출.
    반환 형태는 student.forward_features와 동일한 리스트[list[Tensor]]가 되도록 맞춘다.
    각 텐서는 [B*T, P, C] 형태(클래스 토큰 제외)를 가정.
    """
    assert hasattr(vda_teacher, "pretrained") and hasattr(vda_teacher, "intermediate_layer_idx")
    B, T, C, H, W = x_chunk.shape

    # DINOv2 인터페이스: return_class_token=False면 [B*T, P, C] 텐서들의 리스트가 반환됨
    feats = vda_teacher.pretrained.get_intermediate_layers(
        x_chunk.flatten(0, 1),  # [B*T, C, H, W]
        vda_teacher.intermediate_layer_idx[vda_teacher.encoder],
        return_class_token=return_class_token
    )

    # get_intermediate_layers가 (feat, cls) 튜플을 반환하는 구현일 수도 있으므로 방어
    processed = []
    for f in feats:
        if isinstance(f, (tuple, list)):
            # (tokens_without_cls, cls) 또는 (tokens_with_cls, cls) 등 변형 보호
            # return_class_token=False면 보통 첫 원소가 시퀀스 토큰
            f = f[0]
        processed.append(f)  # [B*T, P, C]
    return processed  # list of [B*T, P, C]

def featurize_in_time_chunks(vda_model, x, chunk_len):
    """
    시간축으로 나눠 레벨별 피처를 concat.
    반환: 레벨별 리스트(list of Tensors), 각 텐서는 [B*TotalT, P, C]
    """
    B, T = x.shape[:2]
    merged = None
    for s in range(0, T, chunk_len):
        e = min(T, s + chunk_len)
        x_chunk = x[:, s:e]  # [B, t, C, H, W]; t = e - s

        if hasattr(vda_model, "forward_features"):
            feats = vda_model.forward_features(x_chunk)
        else:
            feats = _teacher_forward_features(vda_model, x_chunk, return_class_token=False)

        # ★ 정규화 (2D→3D 복구, sparse→dense 포함)
        feats = [_as_tokens_tensor(f) for f in feats]

        if merged is None:
            merged = [f for f in feats]
        else:
            merged = [torch.cat([m, f], dim=0) for m, f in zip(merged, feats)]
    return merged

def _as_tokens_tensor(f):
    """
    반환: [N, P, C] dense/contiguous float tensor
    허용 입력:
      - Tensor [N,P,C] or [N,C,H,W] or [P,C]
      - tuple/list/dict에 Tensor 포함
    """
    # 1) 컨테이너 언패킹: 첫 번째 tensor-like를 고른다
    if isinstance(f, (tuple, list)):
        cand = None
        for item in f:
            if torch.is_tensor(item):
                cand = item; break
        if cand is None:
            raise TypeError("Feature tuple/list has no tensor element.")
        f = cand
    elif isinstance(f, dict):
        # key 우선순위
        for k in ("tokens", "feat", "x", "hidden", "emb", "hs"):
            if k in f and torch.is_tensor(f[k]):
                f = f[k]; break
        else:
            cand = None
            for v in f.values():
                if torch.is_tensor(v):
                    cand = v; break
            if cand is None:
                raise TypeError("Feature dict has no tensor value.")
            f = cand

    if not torch.is_tensor(f):
        raise TypeError(f"Feature is not a tensor after normalization: {type(f)}")

    # 2) sparse → dense
    if f.is_sparse:
        f = f.to_dense()

    # 3) 4D map → tokens
    if f.dim() == 4:  # [N, C, H, W] → [N, P, C]
        N, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(N, H*W, C)

    # 4) 2D → 3D로 복구 ([P,C] → [1,P,C])
    if f.dim() == 2:
        f = f.unsqueeze(0)

    if f.dim() != 3:
        raise ValueError(f"Unsupported feature shape after normalization: {tuple(f.shape)}")

    # 5) dtype/contiguous 보장
    if f.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        f = f.float()
    return f.contiguous()

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
    alpha_dis = hyper_params.get("alpha_dis", 1e-2)
    beta_kld  = hyper_params.get("beta_kld", 5e-4)   # (옵션) teacher attn 있을 때만 사용
    gamma_apc = hyper_params.get("gamma_apc", 5e-3)
    TRAIN_SEQ = hyper_params.get("train_seq", 120)
    KD_WIN     = hyper_params.get("kd_win", 48)     # KD 계산 윈도우 길이(32~64 권장)
    KD_STRIDE  = hyper_params.get("kd_stride", 1)   # KD 간격(1이면 매 프레임, 2면 격프레임)
    aux_apc_N  = hyper_params.get("aux_apc_N", 3)   # student와 동일하게 맞출 값
    KD_MIN_T   = max(aux_apc_N + 1, 8)              # APC 위해 최소 길이
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
        clip_len=TRAIN_SEQ,                 # ← 반드시 명시 (예: 32)
        per_epoch_samples=200,             # ← 1:1 비교 위해 고정
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
    teacher = VideoDepthTeacher(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)
    student = VideoDepthStudent(
        encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN,
        # --- Aux on ---
        use_aux=True,
        aux_layers=[0, -1],                 # features 리스트 앞/뒤 하나씩
        aux_apc_N=3,                        # APC horizon
        aux_transformer_mask_mode="causal", # 안정성↑ (논문 parity면 "bandN")
        aux_rnn_type="mamba",               # or "lstm"
        aux_return_attn=True,               # L_KLD 쓰려면 True
        aux_return_qkv=False,
    ).to(device)
    # KD_MIN_T가 aux_apc_N에 의존한다면, 여기서 최종 보정
    KD_MIN_T = max(getattr(student, "aux_apc_N", aux_apc_N) + 1, 8)

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
        model.teacher.load_state_dict(sd, strict=False)
        model.student.load_state_dict(sd, strict=False)
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
    scaler = amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

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
        kitti_train.set_epoch(epoch)
        model.train()
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
            # --- 배치 전체 시퀀스 피처를 '한 번' 계산 ---
            FEAT_CHUNK = hyper_params.get("feat_chunk", 32)  # 32~48 권장
            TEACH_CHUNK = hyper_params.get("teach_chunk", 64) # no_grad라 더 길게 가능

            model.teacher.eval()
            with torch.no_grad():
                feats_t_full = featurize_in_time_chunks(model.teacher, x, TEACH_CHUNK)
            # student 인코더는 동결되어 있으므로 no_grad로 안전하게 피처만 추출
            with torch.no_grad():
                feats_s_full = featurize_in_time_chunks(model.student, x, FEAT_CHUNK)

            frame_pbar = tqdm(range(T), desc=f"Batch {batch_idx+1} - Frames", leave=False, disable=T < 10)

            # frame loop 직전 (for t in frame_pbar: 위쪽)
            pred_hist = []   # raw disparity 예측 [B,H,W]
            gt_hist   = []   # GT disparity      [B,H,W]
            mask_hist = []   # mask              [B,H,W]
            # 윈도우 길이: KD_WIN을 재사용하거나 별도 하이퍼 tgm_win을 써도 됨
            WIN = hyper_params.get("tgm_win", hyper_params.get("kd_win", 48))

            for t in frame_pbar:
                x_t = x[:, t:t+1]                               # [B,1,3,H,W]
                mask_t = get_mask(y[:, t:t+1], 1e-3, 80.0).to(device)

                # 1) 먼저 스트리밍 한 스텝 (pred_t_raw 필요)
                with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    pred_t_raw, cache = model_stream_step(model.student, x_t, cache)
                    pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)

                # 2) use_teacher 플래그 (고정 확률)
                use_teacher = True

                # 3) KD (Aux 기반, 롤링 윈도우) — FP32 권장
                win_end   = t + 1
                win_start = max(0, win_end - KD_WIN)
                Lw = win_end - win_start
                filled_enough = use_teacher and (Lw >= KD_MIN_T) and ((t % KD_STRIDE) == 0)
                if filled_enough:
                    with amp.autocast('cuda', enabled=False):
                        # precomputed feats → 슬라이스만
                        feats_t_win, _ = slice_features_list(feats_t_full, B, win_start, win_end)
                        feats_s_win, _ = slice_features_list(feats_s_full, B, win_start, win_end)

                        # 방어적 체크 (개발 중 강추)
                        for lvl, f in enumerate(feats_s_win):
                            assert f.dim() == 3, f"[student] level{lvl} got {tuple(f.shape)}, expect [B*Lw, P, C]"
                        for lvl, f in enumerate(feats_t_win):
                            assert f.dim() == 3, f"[teacher] level{lvl} got {tuple(f.shape)}, expect [B*Lw, P, C]"

                        # student's Aux branch outputs on window
                        _d_dummy, _, aux = model.student.forward_depth(
                            feats_s_win, x[:, win_start:win_end].shape,
                            cached_hidden_state_list=None, return_aux=True
                        )
                        loss_dis = loss_apc = loss_kld = 0.0
                        for k in aux['layers']:
                            k_str = str(k)
                            k_use = (len(feats_t_win) + k) if (k < 0) else k

                            h = _to_BT_C(feats_t_win[k_use], B, Lw, use_cls=False)  # [B,Lw,C]
                            z = aux['z'][k_str]                                     # [B,Lw,C]
                            r = aux['r'][k_str]                                     # [B,Lw,C]
                            loss_dis += feat_kd_dis(h, z)
                            loss_apc += apc_feat(h, r, N=getattr(model.student, "aux_apc_N", aux_apc_N))
                            # (옵션) L_KLD: teacher attn 있으면 추가 가능

                        kd_aux  = alpha_dis*loss_dis + beta_kld*loss_kld + gamma_apc*loss_apc
                        kd_loss = (kd_weight * kd_aux).to(pred_t_raw.dtype)
                        if epoch == start_epoch and batch_idx == 0 and t == 0:
                            expected_N = B * (win_end - win_start)
                            print("[KD call] B,Twin,expected_N=", B, (win_end - win_start), expected_N)
                            print("[KD feats_s_win shapes]:", [tuple(f.shape) for f in feats_s_win])
                else:
                    kd_loss = pred_t_raw.new_tensor(0.0)

                # 4) Depth Loss (SSI/TGM) — 윈도우 고정 정렬 버전
                gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6)).squeeze(2)   # [B,1,H,W]
                if pred_t_raw.shape[0] != gt_disp_t.shape[0]:
                    pred_t_raw = pred_t_raw[:1]

                # 2D 형태로 정리
                gt_disp_2d  = gt_disp_t.squeeze(1)               # [B,H,W]
                mask_2d     = mask_t.squeeze(2).squeeze(1)       # [B,H,W]
                pred_2d_raw = pred_t_raw                         # [B,H,W] (이미 2D)

                # 히스토리 버퍼 업데이트
                pred_hist.append(pred_2d_raw.detach())
                gt_hist.append(gt_disp_2d.detach())
                mask_hist.append(mask_2d.detach())

                # 슬라이싱 윈도우 결정
                win_end   = t + 1
                win_start = max(0, win_end - WIN)
                pred_win  = torch.stack(pred_hist[win_start:win_end], dim=1)  # [B,L,H,W]
                gt_win    = torch.stack(gt_hist[win_start:win_end],   dim=1)  # [B,L,H,W]
                mask_win  = torch.stack(mask_hist[win_start:win_end], dim=1)  # [B,L,H,W]

                # 윈도우 전체에서 한 번의 LS로 (â, b̂) 산출
                with torch.no_grad():
                    a_hat, b_hat = window_ls_scale_shift(pred_win, gt_win, mask_win)  # [B,1,1,1] each

                # 동일 (â, b̂)로 현재 프레임 정렬
                pred_t_aligned_disp  = (a_hat.detach() * pred_2d_raw.unsqueeze(1) + b_hat.detach()).squeeze(1)   # [B,H,W]
                pred_t_aligned_depth = 1.0 / (pred_t_aligned_disp.clamp(min=1e-6))

                # SSI (GT는 프레임별 정규화 사용 그대로)
                disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)  # [B,1,H,W]
                ssi_loss_t    = loss_ssi(pred_t_aligned_disp.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))

                # TGM: 이전 프레임도 동일 (â, b̂)로 정렬해 비교
                if t > 0:
                    prev_aligned_disp   = (a_hat.detach() * prev_pred_raw.unsqueeze(1) + b_hat.detach()).squeeze(1)  # [B,H,W]
                    prev_aligned_depth  = 1.0 / (prev_aligned_disp.clamp(min=1e-6))
                    curr_aligned_depth  = pred_t_aligned_depth

                    pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)   # [B,2,H,W]
                    y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)                         # [B,2,1,H,W]
                    m_pair    = torch.cat([prev_mask, mask_t], dim=1)                           # [B,2,1,H,W]
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
                    'KD':  f'{epoch_kd / max(1, epoch_frames):.2e}',
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
    parser.add_argument("--val_json_file",    type=str, default="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json")
    parser.add_argument("--val_infer_dir",    type=str, default="benchmark/output/scannet_stream_valmini")
    parser.add_argument("--val_dataset_key",  type=str, default="scannet")
    parser.add_argument("--val_dataset_tag",  type=str, default="scannet_500")
    parser.add_argument("--val_scenes",       type=int, default=2)
    parser.add_argument("--resume_from", type=str, default="", help="Path to latest/best checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs (e.g., 60)")
    args = parser.parse_args()
    train(args)
