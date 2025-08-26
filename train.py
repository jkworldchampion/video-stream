import os
import argparse
import logging
import inspect

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import wandb
import math
import warnings
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *args, **kwargs: None

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image

# project deps
from utils.loss_MiDas import *
from data.dataLoader import *                 # KITTIVideoDataset, get_data_list
from data.val_dataLoader import *            # ValDataset, get_list
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher
from benchmark.eval.metric import *          # abs_relative_difference, delta1_acc
from benchmark.eval.eval_tae import tae_torch
from video_depth_anything.motion_module.motion_module import TemporalAttention
# UserWarning 카테고리에 해당하는 모든 경고를 무시합니다.
# 'torch.tensor(sourceTensor)' 및 'meshgrid' 경고가 여기에 해당됩니다.
warnings.filterwarnings('ignore', category=UserWarning)
# 특정 메시지 내용을 포함하는 경고를 무시할 수도 있습니다.
# 'preferred_linalg_library' 관련 경고를 숨깁니다.
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")

# ────────────────────────────── 기본 설정 ──────────────────────────────
experiment = 19
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/train_log_experiment_{experiment}.txt"),
    ],
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

MEAN = torch.tensor((0.485, 0.456, 0.406), device=device).view(3, 1, 1)
STD  = torch.tensor((0.229, 0.224, 0.225), device=device).view(3, 1, 1)

# ────────────────────────────── 유틸/평가 함수 ──────────────────────────────
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

def save_validation_frames(x, y, masks, aligned_disp, save_dir, epoch, batch_idx):
    os.makedirs(save_dir, exist_ok=True)
    B, T = x.shape[0], x.shape[1]
    assert B >= 1, "save_validation_frames expects batch size >= 1"

    wb_images = []
    for t in range(T):
        # (a) RGB
        rgb_norm = x[0, t]
        rgb_unc  = (rgb_norm * STD + MEAN).clamp(0, 1)
        rgb_np   = (rgb_unc.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))

        # (b) GT Disparity (min-max 정규화)
        depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)
        disp_frame  = 1.0 / depth_frame
        valid       = masks[0, t].squeeze(0)

        d_vals = disp_frame[valid]
        d_min, d_max = d_vals.min(), d_vals.max()

        norm_gt = ((disp_frame - d_min) / (d_max - d_min + 1e-6)).clamp(0, 1)
        gt_uint8 = (norm_gt.cpu().numpy() * 255).astype(np.uint8)
        gt_rgb   = np.stack([gt_uint8] * 3, axis=-1)
        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"gt_{t:02d}.png"))

        # (c) Mask
        mask_frame = masks[0, t].squeeze(0).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask_frame).save(os.path.join(save_dir, f"mask_{t:02d}.png"))

        # (d) Pred Disparity (같은 min-max)
        pred_frame = aligned_disp[0, t]
        norm_pd = ((pred_frame - d_min) / (d_max - d_min + 1e-6)).clamp(0, 1)
        pd_uint8 = (norm_pd.cpu().numpy() * 255).astype(np.uint8)
        pd_rgb   = np.stack([pd_uint8] * 3, axis=-1)
        Image.fromarray(pd_rgb).save(os.path.join(save_dir, f"pred_{t:02d}.png"))

        wb_images.append(
            wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"),
                        caption=f"pred_epoch{epoch}_frame{t:02d}")
        )

    # logger.info(f"→ saved validation frames to '{save_dir}'")
    return wb_images

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

# ────────────────────────────── Streaming helpers ──────────────────────────────
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

def model_stream_step(model, x_t, cache=None, prev_depth=None, bidirectional_update_length=16, current_frame=0):
    """
    x_t: [B,1,3,H,W] (single-frame step)
    prev_depth: [B,1,H,W] (previous predicted depth for self-forcing)
    bidirectional_update_length: number of recent frames to update bidirectionally
    current_frame: current frame index for bidirectional update
    return: pred_t [B, H, W], new_cache
    """
    # Teacher-Student 모델인지 확인
    if hasattr(model, 'student'):
        # Teacher-Student 모델: Student만 사용
        actual_model = model.student.module if hasattr(model.student, 'module') else model.student
        
        # Student의 forward_depth 사용 (bidirectional update 지원 여부 확인)
        features = actual_model.forward_features(x_t)
        
        # forward_depth 메서드의 signature 확인
        forward_depth_sig = inspect.signature(actual_model.forward_depth)
        forward_depth_params = list(forward_depth_sig.parameters.keys())
        
        # 지원하는 파라미터에 따라 호출 방식 결정
        if 'bidirectional_update_length' in forward_depth_params and 'current_frame' in forward_depth_params:
            # 최신 bidirectional update 지원
            pred_t, new_cache = actual_model.forward_depth(
                features, x_t.shape, cache, prev_depth, 
                bidirectional_update_length=bidirectional_update_length,
                current_frame=current_frame
            )
        elif 'prev_depth' in forward_depth_params:
            # self-forcing 지원하지만 bidirectional update 미지원
            pred_t, new_cache = actual_model.forward_depth(features, x_t.shape, cache, prev_depth)
        else:
            # 기본 방식만 지원
            pred_t, new_cache = actual_model.forward_depth(features, x_t.shape, cache)
    else:
        # 기존 VideoDepthAnything 모델
        actual_model = model.module if hasattr(model, 'module') else model
        
        # Feature extraction은 DataParallel을 통해 병렬화
        if hasattr(model, 'module'):
            # DataParallel 환경: forward를 통해 병렬화된 feature 추출
            with torch.no_grad():
                # 임시로 forward 사용하여 features 병렬 추출
                temp_features = actual_model.forward_features(x_t)
        else:
            # Single GPU
            temp_features = actual_model.forward_features(x_t)
        
        # Depth prediction with cache (bidirectional update 지원 여부 확인)
        # forward_depth 메서드의 signature 확인
        forward_depth_sig = inspect.signature(actual_model.forward_depth)
        forward_depth_params = list(forward_depth_sig.parameters.keys())
        
        # 지원하는 파라미터에 따라 호출 방식 결정
        if 'bidirectional_update_length' in forward_depth_params and 'current_frame' in forward_depth_params:
            # 최신 bidirectional update 지원
            pred_t, new_cache = actual_model.forward_depth(
                temp_features, x_t.shape, cache, prev_depth,
                bidirectional_update_length=bidirectional_update_length,
                current_frame=current_frame
            )
        elif 'prev_depth' in forward_depth_params:
            # self-forcing 지원하지만 bidirectional update 미지원
            pred_t, new_cache = actual_model.forward_depth(temp_features, x_t.shape, cache, prev_depth)
        else:
            # 기본 방식만 지원
            pred_t, new_cache = actual_model.forward_depth(temp_features, x_t.shape, cache)
    
    # 출력 형태 정규화
    if pred_t.dim() == 4 and pred_t.size(1) == 1:
        pred_t = pred_t[:, 0]  # [B,H,W]
    
    return pred_t, new_cache

def streaming_validate( model, loader, device, data_name, loss_ssi, loss_tgm, ratio_ssi, ratio_tgm, save_vis: bool = False, tag: str = None, epoch: int = None, bidirectional_update_length: int = 16 ):
    """
    - 스트리밍 방식으로 검증 (1-frame step)
    - bidirectional_update_length: 양방향 업데이트할 최근 프레임 수
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
            
            # Validation용 프레임 카운터
            val_frame_count = 0

            for t in range(T):
                x_t = x[:, t:t+1]  # [B,1,3,H,W]
                pred_t, cache = model_stream_step(
                    model, x_t, cache, None,
                    bidirectional_update_length=bidirectional_update_length,
                    current_frame=val_frame_count
                )
                pred_t = to_BHW_pred(pred_t)             # [B,H,W]
                preds.append(pred_t)
                
                val_frame_count += 1

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

# ────────────────────────────── train (streaming-aware) ──────────────────────────────
def train(args):
    OUTPUT_DIR = f"outputs/experiment_{experiment}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 시스템 정보
    logger.info("🔍 System Information:")
    logger.info(f"   • PyTorch version: {torch.__version__}")
    logger.info(f"   • CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"   • CUDA version: {torch.version.cuda}")
        logger.info(f"   • Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")

    # W&B 로그인
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("WANDB_API_KEY")
    print("W&B key:", api_key)
    wandb.login(key=api_key, relogin=True)

    # 하이퍼파라미터 로드
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyper_params = config["hyper_parameter"]

    lr         = hyper_params["learning_rate"]
    ratio_ssi  = hyper_params["ratio_ssi"]
    ratio_tgm  = hyper_params["ratio_tgm"]
    num_epochs = hyper_params["epochs"]
    batch_size = hyper_params["batch_size"]
    CLIP_LEN   = hyper_params["clip_len"]
    
    update_frequency = hyper_params.get("update_frequency", 4)    # 4~8 권장
    p_cache_reset    = hyper_params.get("p_cache_reset", 0.01)    # 캐시 드롭아웃
    
    # Self-forcing parameters
    use_self_forcing = hyper_params.get("use_self_forcing", False)
    self_forcing_ratio = hyper_params.get("self_forcing_ratio", 0.3)
    self_forcing_start_epoch = hyper_params.get("self_forcing_start_epoch", 10)
    
    # Teacher-Student parameters
    use_teacher_student = hyper_params.get("use_teacher_student", True)
    teacher_distill_weight = hyper_params.get("teacher_distill_weight", 1.0)
    feature_distill_layers = hyper_params.get("feature_distill_layers", [2, 3])
    teacher_warmup_epochs = hyper_params.get("teacher_warmup_epochs", 5)
    cache_max_length = hyper_params.get("cache_max_length", 32)
    distill_scale_invariant = hyper_params.get("distill_scale_invariant", True)
    feature_distill_weight = hyper_params.get("feature_distill_weight", 1.0)
    log_gradient_norm = hyper_params.get("log_gradient_norm", True)
    log_scale_shift_stats = hyper_params.get("log_scale_shift_stats", True)
    log_max_batches_per_epoch = hyper_params.get("log_max_batches_per_epoch", None)
    # 추가: config.yaml에 정의된 depth_loss_weight (없으면 1.0 기본값)
    depth_loss_weight = hyper_params.get("depth_loss_weight", 1.0)
    
    # Bidirectional Cache Update parameters
    bidirectional_update_length = CLIP_LEN // 2  # 16 frames for bidirectional update
    logger.info(f"   • bidirectional_update_length: {bidirectional_update_length} frames")
    
    logger.info(f"   • update_frequency (frames/step): {update_frequency}")
    logger.info(f"   • p_cache_reset: {p_cache_reset}")
    logger.info(f"   • use_self_forcing: {use_self_forcing}")
    logger.info(f"   • use_teacher_student: {use_teacher_student}")
    if use_self_forcing:
        logger.info(f"   • self_forcing_ratio: {self_forcing_ratio}")
        logger.info(f"   • self_forcing_start_epoch: {self_forcing_start_epoch}")
    if use_teacher_student:
        logger.info(f"   • teacher_distill_weight: {teacher_distill_weight}")
        logger.info(f"   • feature_distill_layers: {feature_distill_layers}")
        logger.info(f"   • teacher_warmup_epochs: {teacher_warmup_epochs}")


    run = wandb.init(project="stream_teacher_student", entity="Depth-Finder", config=hyper_params)

    # ── 데이터: KITTI train/val, ScanNet val ──
    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
    rgb_clips, depth_clips = get_data_list(
        root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN
    )
    kitti_train = KITTIVideoDataset(
        rgb_paths=rgb_clips, depth_paths=depth_clips, resize_size=350, split="train"
    )

    val_rgb_clips, val_depth_clips, val_cam_ids, val_intrin_clips, val_extrin_clips = get_data_list(
        root_dir=kitti_path, data_name="kitti", split="val", clip_len=CLIP_LEN
    )
    kitti_val = KITTIVideoDataset(
        rgb_paths=val_rgb_clips,
        depth_paths=val_depth_clips,
        cam_ids=val_cam_ids,
        intrin_clips=val_intrin_clips,
        extrin_clips=val_extrin_clips,
        resize_size=350,
        split="val",
    )

    kitti_train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True,  num_workers=4)
    kitti_val_loader   = DataLoader(kitti_val,   batch_size=batch_size, shuffle=False, num_workers=4)

    # ScanNet: 평가를 위해 준비
    x_scannet, y_scannet, scannet_poses, scannet_Ks = get_list("", "scannet")
    scannet_data = ValDataset(
        img_paths=x_scannet,
        depth_paths=y_scannet,
        data_name="scannet",
        Ks=scannet_Ks,
        pose_paths=scannet_poses,
    )
    scannet_val_loader = DataLoader(scannet_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # ── 모델 ──
    if use_teacher_student:
        logger.info("🏗️ Creating Teacher-Student models with streaming configuration...")
        teacher_model = VideoDepthTeacher(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384], num_frames=CLIP_LEN, pe="ape", use_causal_mask=False
        ).to(device)
        student_model = VideoDepthStudent(
            num_frames=CLIP_LEN, use_causal_mask=True, encoder="vits", features=64,
            out_channels=[48, 96, 192, 384], use_self_forcing=use_self_forcing,
        ).to(device)

        class TeacherStudentWrapper(torch.nn.Module):
            def __init__(self, teacher, student, distill_weight, feature_layers, feature_distill_weight, scale_invariant, depth_loss_weight):
                super().__init__()
                self.teacher = teacher
                self.student = student
                self.distill_weight = distill_weight
                self.feature_layers = feature_layers or []
                self.feature_distill_weight = feature_distill_weight
                self.scale_invariant = scale_invariant
                self.depth_loss_weight = depth_loss_weight
                self.proj_layers = torch.nn.ModuleDict()

            def compute_teacher_student_loss(self, x, y):
                device0 = x.device
                with torch.no_grad():
                    x_teacher = x.to('cuda:1') if torch.cuda.device_count() > 1 else x
                    teacher_depth = self.teacher(x_teacher).to(device0)  # [B,T,H,W]
                B, T = x.shape[:2]
                cache = None  # Teacher니깐 없음
                student_depths = []
                for t in range(T):
                    x_t = x[:, t:t+1] # time step을 
                    sm = self.student.module if hasattr(self.student, 'module') else self.student
                    feats_t = sm.forward_features(x_t)
                    depth_t, cache = sm.forward_depth(feats_t, x_t.shape, cache)
                    student_depths.append(depth_t.squeeze(1))
                student_depth = torch.stack(student_depths, dim=1)
                y_depth = y.squeeze(2) if (y.dim()==5 and y.shape[2]==1) else y
                if self.scale_invariant:
                    with torch.no_grad():
                        gt_disp = 1.0 / y_depth.clamp(min=1e-6)
                        mask = (y_depth > 1e-3) & (y_depth < 80.0)
                        def align(pred_depth):
                            pred_disp = 1.0 / pred_depth.clamp(min=1e-6)
                            B_, T_, H_, W_ = pred_disp.shape
                            p = pred_disp.view(B_, -1); g = gt_disp.view(B_, -1); m = mask.view(B_, -1).float()
                            A = torch.stack([p, torch.ones_like(p)], dim=-1) * m.unsqueeze(-1)
                            b_vec = g.unsqueeze(-1) * m.unsqueeze(-1)
                            X = torch.linalg.lstsq(A, b_vec).solution
                            a = X[:,0,0].view(B_,1,1,1).clamp(min=1e-4,max=1e4); b = X[:,1,0].view(B_,1,1,1).clamp(min=-1e4,max=1e4)
                            aligned_disp = (pred_disp * a + b).clamp(min=1e-6)
                            return 1.0 / aligned_disp
                        student_aligned = align(student_depth)
                        teacher_aligned = align(teacher_depth)
                    depth_loss_raw = F.l1_loss(student_aligned, y_depth)
                    distill_loss_raw = F.mse_loss(1.0/student_aligned.clamp(min=1e-6), 1.0/teacher_aligned.clamp(min=1e-6))
                else:
                    depth_loss_raw = F.l1_loss(student_depth, y_depth)
                    distill_loss_raw = F.mse_loss(student_depth, teacher_depth.detach())

                feature_loss_raw = student_depth.new_tensor(0.0)
                if self.feature_distill_weight>0 and self.feature_layers:
                    with torch.no_grad():
                        teacher_feats = self.teacher.pretrained.get_intermediate_layers(
                            x_teacher.flatten(0,1),
                            self.teacher.intermediate_layer_idx[self.teacher.encoder],
                            return_class_token=True
                        )
                    sm_all = self.student.module if hasattr(self.student, 'module') else self.student
                    student_feats = sm_all.forward_features(x)
                    for li in self.feature_layers:
                        if li >= len(teacher_feats) or li >= len(student_feats):
                            continue
                        t_tok = teacher_feats[li][0] if (isinstance(teacher_feats[li], (list,tuple)) and len(teacher_feats[li])==2) else teacher_feats[li]
                        s_tok = student_feats[li][0] if (isinstance(student_feats[li], (list,tuple)) and len(student_feats[li])==2) else student_feats[li]
                        if t_tok.device != s_tok.device:
                            t_tok = t_tok.to(s_tok.device)
                        if t_tok.shape[-1] != s_tok.shape[-1]:
                            key = f'proj_{li}'
                            if key not in self.proj_layers:
                                self.proj_layers[key] = torch.nn.Linear(s_tok.shape[-1], t_tok.shape[-1], bias=False)
                            s_tok = self.proj_layers[key](s_tok)
                        feature_loss_raw = feature_loss_raw + F.mse_loss(s_tok, t_tok.detach())
                    feature_loss_raw = feature_loss_raw / max(1, len(self.feature_layers))

                depth_loss_w = self.depth_loss_weight * depth_loss_raw
                distill_loss_w = self.distill_weight * distill_loss_raw
                feature_loss_w = self.feature_distill_weight * feature_loss_raw
                total = depth_loss_w + distill_loss_w + feature_loss_w
                return {
                    'total_loss': total,
                    'depth_loss': depth_loss_w,
                    'distill_loss': distill_loss_w,
                    'feature_loss': feature_loss_w,
                    'student_depth': student_depth,
                    'raw_depth_loss': depth_loss_raw,
                    'raw_distill_loss': distill_loss_raw,
                    'raw_feature_loss': feature_loss_raw,
                }

        model = TeacherStudentWrapper(teacher_model, student_model, teacher_distill_weight, feature_distill_layers, feature_distill_weight, distill_scale_invariant, depth_loss_weight)
        logger.info("✅ Teacher-Student models created with causal masking enabled for streaming")
    else:
        logger.info("🏗️ Creating VideoDepthAnything model with streaming configuration...")
        model = VideoDepthStudent(
            num_frames=CLIP_LEN, use_causal_mask=True, encoder="vits", features=64,
            out_channels=[48, 96, 192, 384], use_self_forcing=use_self_forcing,
        ).to(device)
        logger.info("✅ Model created with causal masking enabled for streaming")

    # Cache length 설정 함수
    def _apply_cache_length(mdl, max_len):
        set_cnt = 0
        for mod in mdl.modules():
            if isinstance(mod, TemporalAttention):
                mod.max_total_length = max_len
                set_cnt += 1
        return set_cnt

    # Student (및 필요 시 Teacher)에 cache 길이 적용
    if cache_max_length and cache_max_length > 0:
        if use_teacher_student:
            applied = _apply_cache_length(model.student, cache_max_length)
            logger.info(f"🧠 Applied cache_max_length={cache_max_length} to {applied} TemporalAttention layers (student)")
        else:
            applied = _apply_cache_length(model, cache_max_length)
            logger.info(f"🧠 Applied cache_max_length={cache_max_length} to {applied} TemporalAttention layers")

    # Pretrained 로드
    if args.pretrained_ckpt:
        logger.info(f"📂 Loading pretrained weights from {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        if use_teacher_student:
            # Student 가중치만 로드 (추론 호환성 중시)
            student_dict = model.student.state_dict()
            student_filtered = {k: v for k, v in state_dict.items() if k in student_dict and v.size() == student_dict[k].size()}
            student_dict.update(student_filtered)
            model.student.load_state_dict(student_dict, strict=True)
            skipped = set(state_dict.keys()) - set(student_filtered.keys())
        else:
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            skipped = set(state_dict.keys()) - set(filtered.keys())

            model_dict.update(filtered)
            model.load_state_dict(model_dict, strict=True)

        if skipped:
            logger.warning(f"⚠️ Skipped loading {len(skipped)} parameters (shape mismatch):")
            for s in list(skipped)[:5]:
                logger.warning(f"   • {s}")
            if len(skipped) > 5:
                logger.warning(f"   • ... and {len(skipped) - 5} more")
        logger.info("✅ Pretrained weights loaded successfully")

    # 학습 전략: encoder freeze, head만 학습
    logger.info("🔒 Configuring training strategy: Encoder frozen, Decoder trainable")
    if use_teacher_student:
        # Teacher는 완전히 freeze
        for p in model.teacher.parameters():
            p.requires_grad = False
        # Student의 encoder freeze, head만 학습
        for p in model.student.pretrained.parameters():
            p.requires_grad = False
        for p in model.student.head.parameters():
            p.requires_grad = True
    else:
        for p in model.pretrained.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

    model.train()

    if use_teacher_student:
        trainable_params = sum(p.numel() for p in model.student.parameters() if p.requires_grad) + sum(p.numel() for p in model.proj_layers.parameters())
        frozen_params = sum(p.numel() for p in model.teacher.parameters())
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
    logger.info("📋 Training parameter status:")
    logger.info(f"📊 Training Summary:")
    logger.info(f"   • Trainable parameters: {trainable_params:,}")
    logger.info(f"   • Frozen parameters: {frozen_params:,}")
    logger.info(f"   • Training ratio: {trainable_params / (trainable_params + frozen_params) * 100:.1f}%")
    logger.info(f"   • Model save directory: {OUTPUT_DIR}")
    logger.info(f"   • Best model criterion: ScanNet AbsRel (lower is better)")

    if torch.cuda.device_count() > 1:
        logger.info(f"🚀 Enabling multi-GPU training with {torch.cuda.device_count()} GPUs")
        logger.info(f"   • Primary device: {device}")
        logger.info(f"   • Available GPUs: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}")
        
        # GPU 메모리 정보 출력
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"   • GPU {i}: {torch.cuda.get_device_name(i)} ({total_memory:.1f}GB)")
        
        if use_teacher_student:
            # 메모리 균형을 위한 전략: Teacher와 Student를 분리 배치
            logger.info("🔄 Optimizing GPU memory distribution for Teacher-Student...")
            
            # Teacher를 GPU 1에, Student를 GPU 0에 배치
            model.teacher = model.teacher.to('cuda:1')
            model.student = model.student.to('cuda:0')
            
            # Student만 DataParallel로 래핑 (실제 훈련되는 모델)
            model.student = torch.nn.DataParallel(model.student, device_ids=[0, 1])
            
            logger.info(f"   • Teacher model on GPU 1 (inference only)")
            logger.info(f"   • Student model with DataParallel on GPUs [0, 1]")
        else:
            model = torch.nn.DataParallel(model)
            logger.info(f"   • Model wrapped with DataParallel")
        
        # Multi-GPU 환경에서 배치 크기 권장사항
        if batch_size < torch.cuda.device_count() * 8:
            recommended_batch_size = torch.cuda.device_count() * 8
            logger.warning(f"⚠️  For optimal multi-GPU utilization, consider increasing batch_size to {recommended_batch_size} or higher")
            logger.warning(f"   Current batch_size: {batch_size}, GPUs: {torch.cuda.device_count()}")
    else:
        logger.info(f"📱 Single GPU training on {device}")

    if use_teacher_student:
        student_params = [p for p in model.student.parameters() if p.requires_grad] + list(model.proj_layers.parameters())
        optimizer = torch.optim.AdamW(student_params, lr=lr, weight_decay=1e-4)
        total_params = sum(p.numel() for p in model.student.parameters()) + sum(p.numel() for p in model.teacher.parameters())
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        total_params = sum(p.numel() for p in model.parameters())
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    logger.info(f"Total parameters: {total_params}")

    loss_tgm = LossTGMVector(diff_depth_th=0.05)
    loss_ssi = Loss_ssi_basic()

    if use_teacher_student:
        # Teacher-Student의 경우 Student만 watch
        wandb.watch(model.student, log="all")
    else:
        wandb.watch(model, log="all")

    best_scannet_absrel = float("inf")
    best_epoch = 0

    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")

    scaler = GradScaler()

    # 환경에 따라 필요시 선호 선형대수 라이브러리 설정
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass

    # ── 학습 루프 ──
    start_epoch = 0
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False):
        print()
        model.train()
        epoch_loss = 0.0
        accum_loss = 0.0
        step_in_window = 0
        # 집계용 변수
        epoch_frames = 0
        epoch_ssi_loss = 0.0
        epoch_tgm_loss = 0.0
        epoch_depth_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_feature_loss = 0.0
        scale_list = []  # a*
        shift_list = []  # b*
        grad_norm_list = []

        # Batch level tqdm
        batch_pbar = tqdm(enumerate(kitti_train_loader), 
                         desc=f"Epoch {epoch+1}/{num_epochs} - Batches", 
                         leave=False,
                         total=len(kitti_train_loader) if log_max_batches_per_epoch is None else min(log_max_batches_per_epoch, len(kitti_train_loader)))
        
        for batch_idx, (x, y) in batch_pbar:
            if log_max_batches_per_epoch is not None and batch_idx >= log_max_batches_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            B, T = x.shape[:2]
            
            # Batch progress update
            batch_pbar.set_postfix({
                'Loss': f'{accum_loss / max(1, step_in_window):.4f}',
                'Frames': epoch_frames,
                'GPU_Mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })
            
            # GPU 메모리 사용량 모니터링 (첫 번째 배치에서만)
            if batch_idx == 0 and epoch == 0:
                logger.info(f"📊 Batch processing info:")
                logger.info(f"   • Batch size: {B}, Sequence length: {T}")
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        cached = torch.cuda.memory_reserved(i) / 1024**3
                        logger.info(f"   • GPU {i} memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

            cache = None
            prev_pred_raw = None
            prev_pred_depth = None  # Self-forcing을 위한 이전 depth
            prev_mask     = None
            prev_y        = None
            
            # Teacher용 프레임 버퍼 (항상 32개 유지)
            teacher_frame_buffer = None  # 첫 프레임에서 초기화
            
            # Student용 bidirectional cache 관리
            student_frame_count = 0  # 현재까지 처리한 프레임 수

            # Frame level tqdm
            frame_pbar = tqdm(range(T), 
                             desc=f"Batch {batch_idx+1} - Frames", 
                             leave=False,
                             disable=(T < 10))  # 프레임이 10개 미만이면 tqdm 비활성화
            
            for t in frame_pbar:
                if np.random.rand() < p_cache_reset:
                    cache = None

                x_t = x[:, t:t+1]                                        # [B,1,3,H,W]
                mask_t = get_mask(y[:, t:t+1], 1e-3, 80.0).to(device)    # [B,1,1,H,W]
                
                # Teacher 프레임 버퍼 관리 (항상 32개 유지)
                if teacher_frame_buffer is None:
                    # 첫 프레임: 32번 복제하여 초기화
                    teacher_frame_buffer = [x_t] * CLIP_LEN  # 32개 동일한 프레임
                else:
                    # 새 프레임 추가: 가장 오래된 것 제거하고 최신 것 추가
                    teacher_frame_buffer.pop(0)  # 맨 앞(가장 오래된) 제거
                    teacher_frame_buffer.append(x_t)  # 맨 뒤에 최신 프레임 추가
                
                # Teacher가 항상 32개 프레임으로 예측 (bidirectional)
                with torch.no_grad():
                    teacher_input = torch.cat(teacher_frame_buffer, dim=1)  # [B, 32, 3, H, W]
                    
                    if torch.cuda.device_count() > 1:
                        teacher_input_gpu = teacher_input.to('cuda:1')
                        teacher_predictions = model.teacher(teacher_input_gpu).to(device)  # [B, 32, H, W]
                    else:
                        teacher_predictions = model.teacher(teacher_input)  # [B, 32, H, W]
                    
                    # Teacher의 현재 시점(t) 예측 추출 (마지막 프레임이 현재 시점)
                    teacher_pred_t = teacher_predictions[:, -1]  # [B, H, W]

                # Self-forcing 결정: 에폭과 확률 조건 만족시 적용
                use_self_forcing_now = (
                    use_self_forcing and 
                    epoch >= self_forcing_start_epoch and 
                    t > 0 and 
                    prev_pred_depth is not None and
                    np.random.rand() < self_forcing_ratio
                )

                with autocast():
                    # Teacher-Student 방식: 각 프레임별로 개별 처리 (gradient graph 충돌 방지)
                    if use_teacher_student:
                        # 현재 프레임에 대해서만 Student 예측 (bidirectional update 적용)
                        if use_self_forcing_now:
                            pred_t_raw, cache = model_stream_step(
                                model, x_t, cache, prev_pred_depth, 
                                bidirectional_update_length=bidirectional_update_length,
                                current_frame=student_frame_count
                            )
                        else:
                            pred_t_raw, cache = model_stream_step(
                                model, x_t, cache, None,
                                bidirectional_update_length=bidirectional_update_length,
                                current_frame=student_frame_count
                            )
                        
                        # Teacher의 현재 시점 예측은 이미 위에서 계산됨 (bidirectional context 활용)
                        
                        # Student 프레임 카운트 증가
                        student_frame_count += 1
                        
                        # 프레임별 Teacher-Student loss 계산
                        y_t = y[:, t].squeeze(1) if y[:, t].dim() > 2 else y[:, t]  # [B,H,W]
                        
                        if model.scale_invariant:
                            # Scale-invariant distillation
                            with torch.no_grad():
                                gt_disp_t = 1.0 / y_t.clamp(min=1e-6)
                                mask_t_ls = (y_t > 1e-3) & (y_t < 80.0)
                                
                                def align_single_frame(pred_depth):
                                    pred_disp = 1.0 / pred_depth.clamp(min=1e-6)
                                    B_, H_, W_ = pred_disp.shape
                                    p_flat = pred_disp.view(B_, -1)
                                    g_flat = gt_disp_t.view(B_, -1)
                                    m_flat = mask_t_ls.view(B_, -1).float()
                                    
                                    A = torch.stack([p_flat, torch.ones_like(p_flat)], dim=-1) * m_flat.unsqueeze(-1)
                                    b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)
                                    X = torch.linalg.lstsq(A, b_vec).solution
                                    a = X[:, 0, 0].view(B_, 1, 1).clamp(min=1e-4, max=1e4)
                                    b = X[:, 1, 0].view(B_, 1, 1).clamp(min=-1e4, max=1e4)
                                    
                                    aligned_disp = (pred_disp * a + b).clamp(min=1e-6)
                                    return 1.0 / aligned_disp
                                
                                student_aligned = align_single_frame(pred_t_raw)
                                teacher_aligned = align_single_frame(teacher_pred_t)
                            
                            # Depth space에서 L1 loss (더 안정적)
                            frame_depth_loss = F.l1_loss(student_aligned, y_t)
                            # Depth space에서 L1 loss로 변경 (MSE는 너무 큰 값 생성)
                            frame_distill_loss = F.l1_loss(student_aligned, teacher_aligned)
                        else:
                            # Log space에서 계산하여 큰 값 방지
                            log_student = torch.log(pred_t_raw.clamp(min=1e-6))
                            log_teacher = torch.log(teacher_pred_t.clamp(min=1e-6))
                            log_gt = torch.log(y_t.clamp(min=1e-6))
                            
                            frame_depth_loss = F.l1_loss(log_student, log_gt)
                            frame_distill_loss = F.l1_loss(log_student, log_teacher)
                        
                        # Feature distillation (현재 프레임만)
                        frame_feature_loss = pred_t_raw.new_tensor(0.0)
                        if model.feature_distill_weight > 0 and model.feature_layers:
                            with torch.no_grad():
                                if torch.cuda.device_count() > 1:
                                    # Teacher는 전체 32개 프레임의 feature를 계산
                                    teacher_input_gpu = torch.cat(teacher_frame_buffer, dim=1).to('cuda:1')
                                    teacher_feats = model.teacher.pretrained.get_intermediate_layers(
                                        teacher_input_gpu.flatten(0,1),
                                        model.teacher.intermediate_layer_idx[model.teacher.encoder],
                                        return_class_token=True
                                    )
                                    # 현재 시점의 feature만 추출 (마지막 프레임 = 현재 시점)
                                    teacher_feats_current = []
                                    for feat in teacher_feats:
                                        if isinstance(feat, (list, tuple)) and len(feat) == 2:
                                            # (token, patch) 형태인 경우
                                            token_feat = feat[0].view(B, CLIP_LEN, -1, feat[0].shape[-1])[:, -1]  # [B, seq_len, dim]
                                            patch_feat = feat[1].view(B, CLIP_LEN, -1, feat[1].shape[-1])[:, -1]  # [B, patch_num, dim] 
                                            teacher_feats_current.append([token_feat.to(device), patch_feat.to(device)])
                                        else:
                                            # 단일 feature인 경우
                                            feat_reshaped = feat.view(B, CLIP_LEN, -1, feat.shape[-1])[:, -1]  # [B, seq_len, dim]
                                            teacher_feats_current.append(feat_reshaped.to(device))
                                    teacher_feats = teacher_feats_current
                                else:
                                    # 단일 GPU에서는 전체 32개 프레임으로 teacher feature 계산
                                    teacher_input_flat = torch.cat(teacher_frame_buffer, dim=1).flatten(0,1)
                                    teacher_feats_all = model.teacher.pretrained.get_intermediate_layers(
                                        teacher_input_flat,
                                        model.teacher.intermediate_layer_idx[model.teacher.encoder],
                                        return_class_token=True
                                    )
                                    # 현재 시점(마지막 프레임)의 feature만 추출
                                    teacher_feats = []
                                    for feat in teacher_feats_all:
                                        if isinstance(feat, (list, tuple)) and len(feat) == 2:
                                            token_feat = feat[0].view(B, CLIP_LEN, -1, feat[0].shape[-1])[:, -1]
                                            patch_feat = feat[1].view(B, CLIP_LEN, -1, feat[1].shape[-1])[:, -1]
                                            teacher_feats.append([token_feat, patch_feat])
                                        else:
                                            feat_reshaped = feat.view(B, CLIP_LEN, -1, feat.shape[-1])[:, -1]
                                            teacher_feats.append(feat_reshaped)
                            
                            # Student features for current frame
                            sm_all = model.student.module if hasattr(model.student, 'module') else model.student
                            student_feats = sm_all.forward_features(x_t)
                            
                            for li in model.feature_layers:
                                if li >= len(teacher_feats) or li >= len(student_feats):
                                    continue
                                
                                t_feat = teacher_feats[li]
                                s_feat = student_feats[li]
                                
                                # Handle tuple format (token, patch)
                                if isinstance(t_feat, (list, tuple)) and len(t_feat) == 2:
                                    t_tok = t_feat[0]
                                else:
                                    t_tok = t_feat
                                    
                                if isinstance(s_feat, (list, tuple)) and len(s_feat) == 2:
                                    s_tok = s_feat[0]
                                else:
                                    s_tok = s_feat
                                
                                # Ensure same device
                                if t_tok.device != s_tok.device:
                                    t_tok = t_tok.to(s_tok.device)
                                
                                # Project if different dimensions
                                if t_tok.shape[-1] != s_tok.shape[-1]:
                                    key = f'proj_{li}'
                                    if key not in model.proj_layers:
                                        model.proj_layers[key] = torch.nn.Linear(s_tok.shape[-1], t_tok.shape[-1], bias=False).to(s_tok.device)
                                    s_tok = model.proj_layers[key](s_tok)
                                
                                frame_feature_loss = frame_feature_loss + F.mse_loss(s_tok, t_tok.detach())
                            
                            if len(model.feature_layers) > 0:
                                frame_feature_loss = frame_feature_loss / len(model.feature_layers)
                        
                        # 가중치 적용된 Teacher-Student loss 저장 (값이 너무 큰 경우 스케일링)
                        current_depth_loss = model.depth_loss_weight * frame_depth_loss * 0.01      # 1/100 스케일링
                        current_distill_loss = model.distill_weight * frame_distill_loss * 0.01     # 1/100 스케일링  
                        current_feature_loss = model.feature_distill_weight * frame_feature_loss * 0.1  # 1/10 스케일링
                        
                    else:
                        # 기존 방식 또는 Teacher-Student 미사용 (bidirectional update 적용)
                        if use_self_forcing_now:
                            pred_t_raw, cache = model_stream_step(
                                model, x_t, cache, prev_pred_depth,
                                bidirectional_update_length=bidirectional_update_length,
                                current_frame=student_frame_count
                            )
                        else:
                            pred_t_raw, cache = model_stream_step(
                                model, x_t, cache, None,
                                bidirectional_update_length=bidirectional_update_length,
                                current_frame=student_frame_count
                            )
                        
                        # Student 프레임 카운트 증가 (Teacher-Student 미사용 시에도)
                        student_frame_count += 1
                    
                    pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)
                    
                    # GPU 병렬화 확인 (첫 번째 배치, 첫 번째 에폭, 첫 번째 프레임에서만)
                    if batch_idx == 0 and epoch == 0 and t == 0:
                        if use_teacher_student:
                            # Teacher-Student 모델의 GPU 배치 정보
                            teacher_device = next(model.teacher.parameters()).device
                            if hasattr(model.student, 'device_ids'):
                                logger.info(f"   • Teacher on {teacher_device}, Student DataParallel on GPUs: {model.student.device_ids}")
                            else:
                                student_device = next(model.student.parameters()).device
                                logger.info(f"   • Teacher on {teacher_device}, Student on {student_device}")
                        else:
                            # 기존 단일 모델의 경우
                            if hasattr(model, 'device_ids'):
                                logger.info(f"   • Using DataParallel with {len(model.device_ids)} GPUs: {model.device_ids}")
                            else:
                                logger.info(f"   • Single GPU model on device: {next(model.parameters()).device}")

                    # 2) GT disparity & LS로 순간 스케일/시프트 추정
                    gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6))           # [B,1,1,H,W]
                    gt_disp_t = gt_disp_t.squeeze(2)                           # [B,1,H,W]
                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)  # <- no_grad
                        if log_scale_shift_stats:
                            a_val = a_star.mean().item()
                            b_val = b_star.mean().item()
                            if math.isfinite(a_val):
                                scale_list.append(a_val)
                            else:
                                logger.debug("Dropped non-finite a* value")
                            if math.isfinite(b_val):
                                shift_list.append(b_val)
                            else:
                                logger.debug("Dropped non-finite b* value")

                    # # 3) EMA 업데이트 (이미 no_grad tensor)
                    # a_ema = ema_update(a_ema, a_star, ema_alpha)
                    # b_ema = ema_update(b_ema, b_star, ema_alpha)
                    
                    # # 4) 정렬된 예측으로 손실 계산 (정렬 스칼라는 detach)
                    # # 방법 A: 4D로 올려 연산 후 다시 내리기 (가장 안전)
                    # pred_t_aligned = (
                    #     a_ema.detach() * pred_t_raw.unsqueeze(1) + b_ema.detach()     # [B,1,1,1]*[B,1,H,W] → [B,1,H,W]
                    # ).squeeze(1)                                                      # [B,H,W]
                    # 4) 정렬 예측으로 손실 계산 (정렬 스칼라는 detach 성격)
                    pred_t_aligned = (
                        a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()
                    ).squeeze(1)  # [B,H,W]

                    # # 5) 약한 정규화도 역전파 막기  (← 이 줄이 핵심)
                    # reg_loss = scale_reg_w * ( (a_star - 1.0).abs().mean() + (b_star - 0.0).abs().mean() )

                    # SSI (framewise; GT는 min-max 정규화 disparity)
                    # GT disparity는 4D [B,1,H,W]로 맞춰 SSI에 전달
                    disp_normed_t = norm_ssi(y[:, t:t+1], mask_t).squeeze(2)   # [B,1,H,W]
                    mask4 = mask_t.squeeze(2)                                  # [B,1,H,W]
                    pred4 = pred_t_aligned.unsqueeze(1)                        # [B,1,H,W]

                    assert pred4.dim()==4 and pred4.size(1)==1,  f"pred4 {pred4.shape}"
                    assert disp_normed_t.dim()==4 and disp_normed_t.size(1)==1, f"disp {disp_normed_t.shape}"
                    assert mask4.dim()==4 and mask4.size(1)==1, f"mask4 {mask4.shape}"

                    ssi_loss_t = loss_ssi(pred4, disp_normed_t, mask4)

                    # TGM (pairwise; 동일 a_ema,b_ema로 두 프레임 모두 정렬)
                    if t > 0:
                        prev_aligned = (
                            a_star.detach() * prev_pred_raw.unsqueeze(1) + b_star.detach()
                        ).squeeze(1)  # [B,H,W]
                        curr_aligned = pred_t_aligned
                        pred_pair = torch.stack([prev_aligned, curr_aligned], dim=1)   # [B,2,H,W]
                        y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)            # [B,2,1,H,W]
                        m_pair    = torch.cat([prev_mask, mask_t], dim=1)              # [B,2,1,H,W]
                        tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                    else:
                        tgm_loss  = pred_t_raw.new_tensor(0.0)

                    # Loss 계산
                    if use_teacher_student:
                        # Teacher-Student loss를 SSI/TGM과 결합
                        loss = current_depth_loss + current_distill_loss + current_feature_loss + ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss
                        current_ssi_loss = ssi_loss_t
                        current_tgm_loss = tgm_loss
                    else:
                        # 기존 단일 모델 학습
                        loss = ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss
                        current_ssi_loss = ssi_loss_t
                        current_tgm_loss = tgm_loss
                        current_depth_loss = pred_t_raw.new_tensor(0.0)
                        current_distill_loss = pred_t_raw.new_tensor(0.0)
                        current_feature_loss = pred_t_raw.new_tensor(0.0)

                # 누적/업데이트
                loss = loss / update_frequency
                accum_loss += loss
                step_in_window += 1

                # 캐시와 상태는 detach하여 그래프 누적 방지
                cache = _detach_cache(cache)
                prev_pred_raw = pred_t_raw.detach()
                
                # Self-forcing을 위한 이전 depth 저장 (aligned depth를 depth로 변환)
                with torch.no_grad():
                    pred_depth_t = 1.0 / (pred_t_aligned.detach().clamp(min=1e-6))  # disparity -> depth
                    prev_pred_depth = pred_depth_t.unsqueeze(1).clamp(min=1e-3, max=80.0)  # [B,1,H,W]
                
                prev_mask     = mask_t
                prev_y        = y[:, t:t+1]

                # Teacher-Student: 그래프가 시퀀스 전체에 걸쳐 있으므로 t==T-1에만 backward
                if step_in_window == update_frequency:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(accum_loss).backward()
                    # gradient norm 측정
                    if log_gradient_norm:
                        total_norm = 0.0
                        for p in (model.student.parameters() if use_teacher_student else model.parameters()):
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2).item()
                                total_norm += param_norm ** 2
                        total_norm = total_norm ** 0.5
                        grad_norm_list.append(total_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += accum_loss.item()
                    accum_loss = 0.0
                    step_in_window = 0

                # 집계 업데이트
                epoch_frames += B
                epoch_ssi_loss += current_ssi_loss.item() * B
                epoch_tgm_loss += current_tgm_loss.item() * B
                if use_teacher_student:
                    epoch_depth_loss += current_depth_loss.item() * B
                    epoch_distill_loss += current_distill_loss.item() * B
                    epoch_feature_loss += current_feature_loss.item() * B
                
                # Frame progress update
                frame_pbar.set_postfix({
                    'SSI': f'{current_ssi_loss.item():.4f}',
                    'TGM': f'{current_tgm_loss.item():.4f}',
                    'Depth': f'{current_depth_loss.item():.4f}' if use_teacher_student else 'N/A',
                    'Distill': f'{current_distill_loss.item():.4f}' if use_teacher_student else 'N/A'
                })

            # per-batch wandb.log 제거 (epoch 말에만 집계 보고)
            
            # Close frame progress bar
            frame_pbar.close()

            del loss, ssi_loss_t, pred_t_aligned
            if 'tgm_loss' in locals():
                del tgm_loss

                # if t % 4 == 0:
                #     logger.info(
                #         f"Epoch [{epoch}] Batch [{batch_idx}] t={t}/{T-1} "
                #         f"loss(ssi={ssi_loss_t.item():.4f}, tgm={tgm_loss.item():.4f}, reg={reg_loss.item():.4f}) "
                #         f"pred.mean={pred_t_raw.mean().item():.6f} a*~{a_star.mean().item():.3f} b*~{b_star.mean().item():.3f}"
                #     )

        # Close batch progress bar
        batch_pbar.close()

        denom_batches = min(len(kitti_train_loader), log_max_batches_per_epoch) if log_max_batches_per_epoch is not None else len(kitti_train_loader)
        avg_kitti_train_loss = epoch_loss / max(1, denom_batches)
        # 에폭 단위 평균 계산
        if epoch_frames > 0:
            mean_ssi = epoch_ssi_loss / epoch_frames
            mean_tgm = epoch_tgm_loss / epoch_frames
            mean_depth = epoch_depth_loss / epoch_frames if use_teacher_student else 0.0
            mean_distill = epoch_distill_loss / epoch_frames if use_teacher_student else 0.0
            mean_feature = epoch_feature_loss / epoch_frames if use_teacher_student and epoch_feature_loss>0 else 0.0
        else:
            mean_ssi = mean_tgm = mean_depth = mean_distill = mean_feature = 0.0
        # 안전한 통계 계산 (NaN/Inf 필터)
        def _finite_stats(values):
            if not values:
                return 0.0, 0.0, 0, 0
            arr = np.asarray(values, dtype=np.float64)
            mask = np.isfinite(arr)
            filtered = int((~mask).sum())
            arr = arr[mask]
            if arr.size == 0:
                return 0.0, 0.0, 0, filtered
            return float(arr.mean()), float(arr.std()), int(arr.size), filtered
        scale_mean, scale_std, scale_cnt, scale_filtered = _finite_stats(scale_list)
        shift_mean, shift_std, shift_cnt, shift_filtered = _finite_stats(shift_list)
        if (scale_filtered > 0 or shift_filtered > 0) and epoch == start_epoch:
            logger.warning(f"⚠️ Filtered non-finite scale/shift entries (scale {scale_filtered}, shift {shift_filtered}) in epoch {epoch}")
        grad_norm_mean = float(np.mean(grad_norm_list)) if grad_norm_list else 0.0
        grad_norm_std  = float(np.std(grad_norm_list)) if grad_norm_list else 0.0

        # ── 검증 (streaming) ──
        kitti_val_loss, kitti_absrel, kitti_delta1, kitti_tae, kitti_wb_images = streaming_validate(
            model, kitti_val_loader, device, "kitti",
            loss_ssi, loss_tgm, ratio_ssi, ratio_tgm,
            save_vis=True, tag="vkitti", epoch=epoch,
            bidirectional_update_length=bidirectional_update_length
        )

        scannet_val_loss, scannet_absrel, scannet_delta1, scannet_tae, scannet_wb_images = streaming_validate(
            model, scannet_val_loader, device, "scannet",
            loss_ssi, loss_tgm, ratio_ssi, ratio_tgm,
            save_vis=True, tag="scannet", epoch=epoch,
            bidirectional_update_length=bidirectional_update_length
        )

        # ── W&B 로깅 ──
        log_dict = {
            "train/epoch_loss": avg_kitti_train_loss,
            "train/epoch_ssi": mean_ssi,
            "train/epoch_tgm": mean_tgm,
            "kitti/val_loss":  kitti_val_loss,
            "kitti/absrel": kitti_absrel,
            "kitti/delta1": kitti_delta1,
            "kitti/tae":    kitti_tae,
            "scannet/val_loss":  scannet_val_loss,
            "scannet/absrel": scannet_absrel,
            "scannet/delta1": scannet_delta1,
            "scannet/tae":    scannet_tae,
            "epoch": epoch,
        }
        if use_teacher_student:
            log_dict.update({
                "train/epoch_depth": mean_depth,
                "train/epoch_distill": mean_distill,
            })
            if mean_feature>0:
                log_dict["train/epoch_feature_distill"] = mean_feature
        if log_scale_shift_stats:
            log_dict.update({
                "scale_shift/a_mean": scale_mean,
                "scale_shift/a_std": scale_std,
                "scale_shift/b_mean": shift_mean,
                "scale_shift/b_std": shift_std,
            })
        if log_gradient_norm:
            log_dict.update({
                "grad/epoch_norm_mean": grad_norm_mean,
                "grad/epoch_norm_std": grad_norm_std,
            })
        # 이미지 로그 (필요 최소화)
        log_dict["vkitti/pred_disparity"] = kitti_wb_images
        log_dict["scannet/pred_disparity"] = scannet_wb_images
        wandb.log(log_dict)
        del kitti_wb_images
        del scannet_wb_images

        # ── 모델 저장 (ScanNet AbsRel 기준) ──
        if scannet_absrel < best_scannet_absrel:
            best_scannet_absrel = scannet_absrel
            best_epoch = epoch
            if use_teacher_student:
                model_state = model.student.state_dict()
            else:
                model_state = model.state_dict()

            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state,
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
        if use_teacher_student:
            latest_model_state = model.student.state_dict()
        else:
            latest_model_state = model.state_dict()

        torch.save({
            "epoch": epoch,
            "model_state_dict": latest_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "current_scannet_absrel": scannet_absrel,
            "config": hyper_params,
        }, latest_model_path)
        logger.info(f"📁 Latest model saved to {latest_model_path}")

        torch.cuda.empty_cache()
        scheduler.step()
        

    # 완료 로그
    logger.info("=" * 50)
    logger.info("🎯 Training Completed!")
    logger.info(f"   • Total Epochs: {num_epochs}")
    logger.info(f"   • Best Epoch: {best_epoch}")
    logger.info(f"   • Best ScanNet AbsRel: {best_scannet_absrel:.4f}")
    logger.info(f"   • Best model saved to: {best_model_path}")
    logger.info(f"   • Latest model saved to: {latest_model_path}")
    logger.info("=" * 50)

    wandb.log({
        "training/completed": True,
        "training/total_epochs": num_epochs,
        "training/best_epoch": best_epoch,
        "training/final_best_scannet_absrel": best_scannet_absrel,
    })
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")
    args = parser.parse_args()
    train(args)
