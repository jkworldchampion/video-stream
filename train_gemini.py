import os
import argparse
import logging
import torch
import numpy as np
import yaml
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image

# (수정) DDP를 위한 모듈 임포트
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 프로젝트 의존성
from utils.loss_MiDas import *
from data.dataLoader import *
from data.val_dataLoader import *
from video_depth_anything.video_depth_stream import VideoDepthAnything
from benchmark.eval.metric import *
from benchmark.eval.eval_tae import tae_torch

# ────────────────────────────── 기본 설정 ──────────────────────────────
experiment = 11

def setup_ddp():
    """DDP를 위한 분산 환경을 설정합니다."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """DDP 프로세스 그룹을 정리합니다."""
    dist.destroy_process_group()

def is_main_process():
    """현재 프로세스가 메인 프로세스(rank 0)인지 확인합니다."""
    return dist.get_rank() == 0

def setup_logging(is_main):
    """메인 프로세스에서만 로그와 파일 핸들러를 설정합니다."""
    os.makedirs("logs", exist_ok=True)
    level = logging.INFO if is_main else logging.WARNING
    
    # (수정) 파일 핸들러는 메인 프로세스에만 추가
    handlers = [logging.StreamHandler()]
    if is_main:
        handlers.append(logging.FileHandler(f"logs/train_log_experiment_{experiment}.txt"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s RANK:%(process)d %(levelname)-8s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)

# ────────────────────────────── 유틸/평가 함수 (이전과 동일) ──────────────────────────────
# least_square_whole_clip, eval_tae, metric_val, get_mask, norm_ssi 등
# 평가 관련 함수들은 이전 코드와 동일하게 유지됩니다.
# (이 부분은 이전 코드와 동일하므로 간결함을 위해 생략)
def least_square_whole_clip(infs, gts, data_name):
    # [B,T,1,H,W] → [B,T,H,W]
    if infs.dim() == 5 and infs.shape[2] == 1:
        infs = infs.squeeze(2)
    if gts.dim() == 5 and gts.shape[2] == 1:
        gts = gts.squeeze(2)

    if data_name == "kitti":
        valid_mask = (gts > 1e-3) & (gts < 80.0)
    elif data_name == "gta":
        valid_mask = (gts > 1e-3) & (gts < 1000.0)
    elif data_name == "tartanair":
        valid_mask = (gts > 60.0) & (gts < 150.0)
    else:
        valid_mask = (gts > 1e-3) & (gts < 10.0)

    gt_disp_masked = 1.0 / (gts[valid_mask].reshape((-1, 1)).double() + 1e-6)
    infs = infs.clamp(min=1e-3)
    pred_disp_masked = infs[valid_mask].reshape((-1, 1)).double()

    A = torch.cat([pred_disp_masked, torch.ones_like(pred_disp_masked)], dim=-1)
    X = torch.linalg.lstsq(A, gt_disp_masked).solution
    scale, shift = X[0].item(), X[1].item()

    aligned_pred = torch.clamp(scale * infs + shift, min=1e-3)
    depth = torch.zeros_like(aligned_pred)
    depth = 1.0 / aligned_pred
    return depth, valid_mask

def eval_tae(pred_depth, gt_depth, poses, Ks, masks):
    error_sum = 0.0
    for i in range(len(pred_depth) - 1):
        depth1, depth2 = pred_depth[i], pred_depth[i + 1]
        mask1, mask2   = masks[i], masks[i + 1]

        T_1, T_2 = poses[i], poses[i + 1]
        try:
            T_2_inv = torch.linalg.inv(T_2)
        except torch._C._LinAlgError:
            T_2_inv = torch.linalg.pinv(T_2)

        T_2_1 = T_2_inv @ T_1
        R_2_1 = T_2_1[:3, :3]
        t_2_1 = T_2_1[:3, 3]

        K = Ks[i]
        if K.dim() == 1 and K.numel() == 9:
            K = K.view(3, 3)

        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)

        try:
            T_1_2 = torch.linalg.inv(T_2_1)
        except torch._C._LinAlgError:
            T_1_2 = torch.linalg.pinv(T_2_1)

        R_1_2 = T_1_2[:3, :3]
        t_1_2 = T_1_2[:3, 3]
        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)

        error_sum += error1 + error2

    return error_sum / (2 * (len(pred_depth) - 1))

def metric_val(infs, gts, data_name, poses=None, Ks=None):
    gt_depth = gts
    pred_depth, valid_mask = least_square_whole_clip(infs, gts, data_name)

    n = valid_mask.sum((-1, -2))
    valid_frame = (n > 0)
    pred_depth = pred_depth[valid_frame]
    gt_depth   = gt_depth[valid_frame]
    valid_mask = valid_mask[valid_frame]

    absrel = abs_relative_difference(pred_depth, gt_depth, valid_mask)
    delta1 = delta1_acc(pred_depth, gt_depth, valid_mask)

    if poses is not None:
        tae = eval_tae(pred_depth, gt_depth, poses, Ks, valid_mask)
        return absrel, delta1, tae
    else:
        return absrel, delta1

def get_mask(depth_m, min_depth, max_depth):
    return ((depth_m > min_depth) & (depth_m < max_depth)).bool()

def norm_ssi(depth, valid_mask):
    eps = 1e-6
    disparity = torch.zeros_like(depth)
    disparity[valid_mask] = 1.0 / depth[valid_mask]

    B, T, C, H, W = disparity.shape
    disp_flat = disparity.view(B, T, -1)
    mask_flat = valid_mask.view(B, T, -1)

    disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).min(dim=-1)[0]
    disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).max(dim=-1)[0]
    disp_min = disp_min.view(B, T, 1, 1, 1)
    disp_max = disp_max.view(B, T, 1, 1, 1)

    norm_disp = (disparity - disp_min) / (disp_max - disp_min + eps)
    return norm_disp.masked_fill(~valid_mask, 0.0)

def to_BHW_pred(pred):
    # pred: [B,H,W] or [B,1,H,W] or [B,C,H,W]
    if pred.dim() == 3:
        return pred
    if pred.dim() == 4:
        if pred.size(1) == 1:
            return pred[:, 0]              # [B,H,W]
        else:
            # C>1인 경우(드물지만 발생): 채널 축 평균으로 단일 disparity 생성
            return pred.mean(dim=1)        # [B,H,W]
    raise ValueError(f"Unexpected pred shape: {pred.shape}")


# ... (평가 함수들: least_square_whole_clip, eval_tae, metric_val, get_mask, norm_ssi) ...

def save_validation_frames(x, y, masks, aligned_disp, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    T = x.shape[1]
    MEAN = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(3, 1, 1)
    STD  = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(3, 1, 1)
    
    wb_images = []
    for t in range(T):
        # (a) RGB
        rgb_norm = x[0, t]
        rgb_unc  = (rgb_norm * STD + MEAN).clamp(0, 1)
        rgb_np   = (rgb_unc.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))
        
        # (d) Pred Disparity
        pred_frame = aligned_disp[0, t]
        disp_frame = 1.0 / y[0, t].squeeze(0).clamp(min=1e-6)
        valid = masks[0, t].squeeze(0)
        d_vals = disp_frame[valid]
        if d_vals.numel() > 0:
            d_min, d_max = d_vals.min(), d_vals.max()
            norm_pd = ((pred_frame - d_min) / (d_max - d_min + 1e-6)).clamp(0, 1)
            pd_uint8 = (norm_pd.cpu().numpy() * 255).astype(np.uint8)
            pd_rgb = np.stack([pd_uint8] * 3, axis=-1)
            Image.fromarray(pd_rgb).save(os.path.join(save_dir, f"pred_{t:02d}.png"))
            
            wb_images.append(
                wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"),
                            caption=f"pred_epoch{epoch}_frame{t:02d}")
            )
    return wb_images

def _detach_cache(cache):
    if cache is None:
        return None
    if isinstance(cache, (list, tuple)):
        return type(cache)(_detach_cache(c) for c in cache)
    if isinstance(cache, dict):
        return {k: _detach_cache(v) for k, v in cache.items()}
    if torch.is_tensor(cache):
        return cache.detach()
    return cache  # unknown type as-is

def model_stream_step(model, x_t, cache=None):
    """
    x_t: [B,1,3,H,W] (single-frame step)
    return: pred_t [B, H, W], new_cache
    """
    # 우선 시도: 스트리밍 시그니처
    try:
        out = model(x_t, cached_hidden_states=cache, update_cache=True, return_cache=True)
        if isinstance(out, tuple) and len(out) == 2:
            pred_t, new_cache = out
        else:
            # (pred, cache) 형태가 아니면 호환 불가 → 폴백
            raise TypeError
        # pred_t: [B,1,H,W] 로 가정
        if pred_t.dim() == 4 and pred_t.size(1) == 1:
            pred_t = pred_t[:, 0]  # [B,H,W]
        return pred_t, new_cache
    except TypeError:
        # 폴백: 캐시 미지원 모델 → 누적 슬라이스를 넣고 마지막 프레임만 사용
        # 주의: 학습 속도는 느려지지만 train/test 조건 정렬 목적은 달성
        T_now = x_t.size(1)  # 보통 1
        out_full = model(x_t)  # [B,T_now,H,W]
        if out_full.dim() == 4 and out_full.size(1) == T_now:
            pred_t = out_full[:, -1]  # 마지막 프레임만
        else:
            pred_t = out_full  # [B,H,W] 인 경우
        return pred_t, None

def streaming_validate( model, loader, device, data_name, loss_ssi, loss_tgm, ratio_ssi, ratio_tgm, save_vis: bool = False, tag: str = None, epoch: int = None ):
    """
    - 스트리밍 방식으로 검증 (1-frame step)
    - save_vis=True면 각 에폭마다 각 데이터셋의 '첫 배치'만 이미지 저장 + W&B 이미지 리스트 반환
    - 반환: avg_loss, avg_absrel, avg_delta1, avg_tae, wb_images(list)
    """
    model.eval()
    total_absrel = 0.0
    total_delta1 = 0.0
    total_tae    = 0.0
    total_loss   = 0.0
    cnt_clip     = 0
    wb_images    = []

    MIN_DISP = 1.0 / 80.0
    MAX_DISP = 1.0 / 0.001

    with torch.no_grad():
        for batch_idx, (x, y, extrinsics, intrinsics) in tqdm(enumerate(loader)):
            x, y = x.to(device), y.to(device)
            extrinsics, intrinsics = extrinsics.to(device), intrinsics.to(device)

            B, T = x.shape[:2]
            preds = []
            mask_list = []   # [B,1,H,W] per frame
            cache = None
            prev_pred = None
            prev_mask = None
            prev_y    = None

            for t in range(T):
                x_t = x[:, t:t+1]  # [B,1,3,H,W]
                pred_t, cache = model_stream_step(model, x_t, cache)
                pred_t = to_BHW_pred(pred_t)             # [B,H,W]
                preds.append(pred_t)

                mask_t = get_mask(y[:, t:t+1], min_depth=1e-3, max_depth=80.0).to(device)  # [B,1,1,H,W]
                mask_list.append(mask_t.squeeze(2))  # → [B,1,H,W]

                # framewise loss(로그용)
                disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)   # [B,1,H,W]
                ssi_loss_t = loss_ssi(pred_t.unsqueeze(1), disp_normed_t, mask_t.squeeze(2))
                total_loss += ratio_ssi * ssi_loss_t

                if t > 0:
                    pred_pair = torch.stack([prev_pred, pred_t], dim=1)  # [B,2,H,W]
                    y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)  # [B,2,1,H,W]
                    m_pair    = torch.cat([prev_mask, mask_t], dim=1)    # [B,2,1,H,W]
                    tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                    total_loss += ratio_tgm * tgm_loss

                prev_pred = pred_t
                prev_mask = mask_t
                prev_y    = y[:, t:t+1]

            pred_seq  = torch.stack(preds, dim=1)               # [B,T,H,W]
            masks_seq = torch.stack(mask_list, dim=1)           # [B,T,1,H,W]

            # metric (클립 단위 LS 정렬 포함)
            for b in range(B):
                inf_clip  = pred_seq[b]           # [T,H,W]
                gt_clip   = y[b].squeeze(1)       # [T,H,W]
                pose      = extrinsics[b]
                Kmat      = intrinsics[b]
                absr_dae = metric_val(inf_clip, gt_clip, data_name, pose, Kmat)
                if isinstance(absr_dae, tuple) and len(absr_dae) == 3:
                    absr, d1, tae = absr_dae
                    total_tae += tae
                else:
                    absr, d1 = absr_dae
                total_absrel += absr
                total_delta1 += d1
                cnt_clip     += 1

            # 시각화 저장 (첫 배치만)
            if save_vis and batch_idx == 0 and tag is not None and epoch is not None:
                # 클립 단위 LS로 정렬한 disparity (시각화용)
                Bv, Tv, H, W = pred_seq.shape
                raw_disp = pred_seq.clamp(min=1e-6)              # [B,T,H,W]
                gt_disp  = (1.0 / y.clamp(min=1e-6)).squeeze(2)  # [B,T,H,W]
                m_flat   = masks_seq.squeeze(2).view(Bv, -1).float()
                p_flat   = raw_disp.view(Bv, -1)
                g_flat   = gt_disp.view(Bv, -1)

                A = torch.stack([p_flat, torch.ones_like(p_flat, device=device)], dim=-1)  # [B,P,2]
                A = A * m_flat.unsqueeze(-1)
                b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)
                X = torch.linalg.lstsq(A, b_vec).solution
                a = X[:, 0, 0].view(Bv, 1, 1, 1)
                b = X[:, 1, 0].view(Bv, 1, 1, 1)
                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)  # [B,T,H,W]

                save_dir = f"outputs/experiment_{experiment}/{tag}/epoch_{epoch}_batch_{batch_idx}"
                wb_images.extend(save_validation_frames(x, y, masks_seq, aligned_disp, save_dir, epoch, batch_idx))

    avg_absrel = total_absrel / max(1, cnt_clip)
    avg_delta1 = total_delta1 / max(1, cnt_clip)
    avg_tae    = total_tae    / max(1, cnt_clip)
    avg_loss   = total_loss   / max(1, cnt_clip)
    return avg_loss, avg_absrel, avg_delta1, avg_tae, wb_images


def batch_ls_scale_shift(pred_disp, gt_disp, mask):
    """
    pred_disp: [B, H, W] or [B,1,H,W] disparity (>= 1e-6)
    gt_disp  : [B, H, W] or [B,1,H,W] disparity
    mask     : [B,1,H,W] bool

    return a_star, b_star with shape [B,1,1,1]
    """
    if pred_disp.dim() == 4 and pred_disp.size(1) == 1:
        p = pred_disp[:, 0]
    else:
        p = pred_disp
    if gt_disp.dim() == 4 and gt_disp.size(1) == 1:
        g = gt_disp[:, 0]
    else:
        g = gt_disp

    B, H, W = p.shape
    m = mask.view(B, -1).float()                      # [B, P]
    p_flat = p.view(B, -1)                            # [B, P]
    g_flat = g.view(B, -1)                            # [B, P]

    A = torch.stack([p_flat, torch.ones_like(p_flat, device=p.device)], dim=-1)  # [B,P,2]
    A = A * m.unsqueeze(-1)
    b_vec = g_flat.unsqueeze(-1) * m.unsqueeze(-1)

    X = torch.linalg.lstsq(A, b_vec).solution        # [B,2,1]
    a_star = X[:, 0, 0].view(B, 1, 1, 1)
    b_star = X[:, 1, 0].view(B, 1, 1, 1)

    # 안정성: a는 양수로, 극단치 클리핑
    a_star = a_star.clamp(min=1e-4, max=1e4)
    b_star = b_star.clamp(min=-1e4, max=1e4)
    return a_star, b_star

def ema_update(prev, new, alpha):
    if prev is None:
        return new
    return (1.0 - alpha) * prev + alpha * new

# ... (나머지 헬퍼 함수들: _detach_cache, model_stream_step, to_BHW_pred) ...

# ────────────────────────────── train (DDP 및 개선된 학습 로직) ──────────────────────────
def train(args):
    setup_ddp()
    logger = setup_logging(is_main_process())
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    OUTPUT_DIR = f"outputs/experiment_{experiment}"
    if is_main_process():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("🔍 System Information:")
        logger.info(f"   • PyTorch version: {torch.__version__}")
        logger.info(f"   • CUDA available: {torch.cuda.is_available()}")
        logger.info(f"   • World Size (GPU count): {dist.get_world_size()}")

    # W&B 로그인 (메인 프로세스에서만)
    if is_main_process():
        load_dotenv(dotenv_path=".env")
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key, relogin=True)

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyper_params = config["hyper_parameter"]
    
    # (수정) 배치 사이즈는 이제 GPU당 배치 사이즈를 의미
    batch_size = hyper_params["batch_size"] 
    lr, ratio_ssi, ratio_tgm, num_epochs, CLIP_LEN = (
        hyper_params["learning_rate"], hyper_params["ratio_ssi"],
        hyper_params["ratio_tgm"], hyper_params["epochs"], hyper_params["clip_len"]
    )

    if is_main_process():
        run = wandb.init(project="stream_causal_block", entity="Depth-Finder", config=hyper_params)
    
    # ── 데이터 로더 (DistributedSampler 적용) ──
    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
    kitti_train_dataset = KITTIVideoDataset(
        rgb_paths=get_data_list(kitti_path, "kitti", "train", CLIP_LEN)[0],
        depth_paths=get_data_list(kitti_path, "kitti", "train", CLIP_LEN)[1],
        resize_size=350, split="train"
    )
    kitti_val_dataset = KITTIVideoDataset(...) # Val 데이터셋도 동일하게 정의

    # (수정) DistributedSampler 사용
    train_sampler = DistributedSampler(kitti_train_dataset)
    val_sampler = DistributedSampler(kitti_val_dataset, shuffle=False)

    kitti_train_loader = DataLoader(kitti_train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    kitti_val_loader = DataLoader(kitti_val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # ScanNet 데이터 로더도 동일하게 DistributedSampler 적용 ...

    # ── 모델 (DDP 적용) ──
    model = VideoDepthAnything(...).to(device)
    
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        
        # (수정) DDP로 래핑하기 전에 가중치를 로드합니다.
        # (수정) strict=False로 변경하여 유연성 확보
        model.load_state_dict(state_dict, strict=False) 
        if is_main_process():
            logger.info("✅ Pretrained weights loaded successfully (strict=False)")
    
    for p in model.pretrained.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True

    # (수정) DDP로 모델 래핑
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    loss_tgm = LossTGMVector(diff_depth_th=0.05).to(device)
    loss_ssi = Loss_ssi_basic().to(device)
    
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")

    if is_main_process():
        wandb.watch(model, log="all")

    # ── 학습 루프 (수정된 학습 목표 적용) ──
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch) # (수정) 매 에폭마다 샘플러 셔플링
        
        pbar = tqdm(kitti_train_loader, desc=f"Epoch {epoch} Training", disable=not is_main_process())
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            B, T = x.shape[:2]
            
            cache = None
            prev_pred_norm = None
            prev_gt_norm = None

            optimizer.zero_grad(set_to_none=True)
            
            # --- (수정) 새로운 학습 목표 ---
            # Gradient Accumulation을 위해 루프를 한번 더 돕니다.
            accum_loss = 0.0
            for t in range(T):
                x_t = x[:, t:t+1]
                y_t = y[:, t:t+1]
                mask_t = get_mask(y_t, 1e-3, 80.0)

                with autocast():
                    # 1. 모델이 직접 정규화된 Disparity를 예측
                    pred_t_norm, cache = model_stream_step(model, x_t, cache)
                    pred_t_norm = to_BHW_pred(pred_t_norm)

                    # 2. GT를 동일하게 정규화
                    gt_norm_t = norm_ssi(y_t, mask_t).squeeze(1) # [B,1,H,W]

                    # 3. 정규화된 값들로 손실 계산
                    ssi_loss_t = loss_ssi(pred_t_norm.unsqueeze(1), gt_norm_t, mask_t.squeeze(2))
                    
                    if t > 0:
                        # TGM은 정규화된 값 사이에서 계산 (또는 깊이로 변환 후 계산)
                        # 여기서는 정규화된 공간에서 계산한다고 가정
                        pred_pair = torch.stack([prev_pred_norm, pred_t_norm], dim=1)
                        gt_pair_norm = torch.stack([prev_gt_norm, gt_norm_t.squeeze(1)], dim=1)
                        # TGM Loss는 깊이 스케일이 중요하므로, 여기서는 개념적 표현만 남깁니다.
                        # 실제 적용 시에는 이 부분에 대한 추가적인 실험이 필요합니다.
                        # 여기서는 SSI에 집중하기 위해 TGM 가중치를 0으로 가정할 수 있습니다.
                        tgm_loss = torch.tensor(0.0, device=device)
                    else:
                        tgm_loss = torch.tensor(0.0, device=device)

                    loss = (ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss)

                # 그래디언트 누적을 위해 손실을 스케일링
                scaler.scale(loss).backward()
                accum_loss += loss.item()

                # 상태 업데이트 (detach 필수)
                cache = _detach_cache(cache)
                prev_pred_norm = pred_t_norm.detach()
                prev_gt_norm = gt_norm_t.squeeze(1).detach()
            
            # 누적된 그래디언트로 옵티마이저 스텝
            scaler.step(optimizer)
            scaler.update()

            if is_main_process():
                pbar.set_postfix(loss=f"{accum_loss:.4f}")

        # ── 검증 루프 (메인 프로세스에서만 수행) ──
        if is_main_process():
            model.eval()
            with torch.no_grad():
                # streaming_validate 함수를 사용하여 검증
                # 이 함수는 내부적으로 클립 단위 LS 정렬 후 메트릭 계산
                kitti_val_loss, kitti_absrel, kitti_delta1, kitti_tae, kitti_wb_images = streaming_validate(
                    model, kitti_val_loader, device, "kitti",
                    loss_ssi, loss_tgm, ratio_ssi, ratio_tgm,
                    save_vis=True, tag="vkitti", epoch=epoch
                )

                # ── ScanNet(stream) ──
                scannet_val_loss, scannet_absrel, scannet_delta1, scannet_tae, scannet_wb_images = streaming_validate(
                    model, scannet_val_loader, device, "scannet",
                    loss_ssi, loss_tgm, ratio_ssi, ratio_tgm,
                    save_vis=True, tag="scannet", epoch=epoch
                )

            # ── W&B 로깅 ──
            wandb.log({
                # "train_loss_stream": avg_kitti_train_loss,
                "kitti_stream_loss":  kitti_val_loss,
                "kitti_stream_absrel": kitti_absrel,
                "kitti_stream_delta1": kitti_delta1,
                "kitti_stream_tae":    kitti_tae,
                "vkitti_pred_disparity": kitti_wb_images,     # ← 이미지 업로드
                "scannet_stream_loss":  scannet_val_loss,
                "scannet_stream_absrel": scannet_absrel,
                "scannet_stream_delta1": scannet_delta1,
                "scannet_stream_tae":    scannet_tae,
                "scannet_pred_disparity": scannet_wb_images,  # ← 이미지 업로드
                "epoch": epoch,
            })
            del kitti_wb_images
            del scannet_wb_images
        
            # ── 모델 저장 (ScanNet AbsRel 기준) ──
            if scannet_absrel < best_scannet_absrel:
                best_scannet_absrel = scannet_absrel
                best_epoch = epoch

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_scannet_absrel": best_scannet_absrel,
                    "scannet_val_loss": scannet_val_loss,
                    "scannet_delta1": scannet_delta1,
                    "scannet_tae": scannet_tae,
                    "config": hyper_params,
                }, best_model_path)

                logger.info(f"🏆 Best model saved! Epoch {epoch}, ScanNet AbsRel: {best_scannet_absrel:.4f}")
                wandb.log({
                    "best_scannet_absrel": best_scannet_absrel,
                    "best_epoch": epoch,
                    "model_improved": True,
                    "epoch": epoch,
                })
            else:
                wandb.log({"model_improved": False, "epoch": epoch})

            # latest 저장
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "current_scannet_absrel": scannet_absrel,
                "config": hyper_params,
            }, latest_model_path)
            logger.info(f"📁 Latest model saved to {latest_model_path}")

        scheduler.step()
    
    if is_main_process():
        run.finish()
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")
    # (수정) DDP를 위한 인자들 (torch.distributed.launch가 자동으로 채워줌)
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    args = parser.parse_args()
    train(args)