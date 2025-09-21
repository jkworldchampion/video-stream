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
# from tqdm.contrib.logging import logging_redirect_tqdm
from PIL import Image

from utils.loss_MiDas import *
from utils.train_helper import *
from data.dataLoader import *                 # KITTIVideoDataset, get_data_list
from data.val_dataLoader import *            # ValDataset, get_list

# 기존 offline model을 teacher로, real-time model을 student로 설계
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent
from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher

from benchmark.eval.metric import *          # abs_relative_difference, delta1_acc
from benchmark.eval.eval_tae import tae_torch
from video_depth_anything.motion_module.motion_module import TemporalAttention

# error문구 없애기
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*preferred_linalg_library.*")


# ================ 기본설정 ================
experiment = 25  # 실험마다 바꾸기, notion과 일치 시키는 것을 권장
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/experiment_{experiment}.txt"),
    ],
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

MEAN = torch.tensor((0.485, 0.456, 0.406), device=device).view(3, 1, 1)
STD  = torch.tensor((0.229, 0.224, 0.225), device=device).view(3, 1, 1)

# ================ utils function ================
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

# ================ train ================
def train(args):
    OUTPUT_DIR = f"outputs/experiment_{experiment}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 기본 설정 값 logging
    logger.info(f"Torch: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"   • CUDA version: {torch.version.cuda}")
        logger.info(f"   • Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    
    # --------------------- hyperparameter loading ---------------------
    with open("configs/config_jh.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyper_params = config["hyper_parameter"]
    lr         = hyper_params["learning_rate"]
    ratio_ssi  = hyper_params["ratio_ssi"]
    ratio_tgm  = hyper_params["ratio_tgm"]
    num_epochs = hyper_params["epochs"]
    batch_size = hyper_params["batch_size"]
    CLIP_LEN   = hyper_params["clip_len"]
    # Teacher-Student parameters
    update_frequency = hyper_params.get("update_frequency", 6)
    pred_distill_weight = hyper_params.get("pred_distill_weight", 1.0)  # teacher의 pred와의 align
    cache_distill_weight = hyper_params.get("cache_distill_weight", 1.0)  # teacher의 attention cache와의 align
    
    # Bidirectional Cache Update parameter definition
    bidirectional_update_length = CLIP_LEN // 2  # 현재 32니깐 16으로 설정
    logger.info(f"bidirentional_update_len: {bidirectional_update_length} frames")
    
    # W&B login
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("WANDB_API_KEY")
    print("W&B key: ", api_key)
    wandb.login(key=api_key, relogin=True)
    run = wandb.init(project="stream_teacher_student", entity="Depth-Finder", config=hyper_params, name=f"experiment_{experiment}")
    
    
    # --------------------- Data 준비 ---------------------
    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"  # 환경마다 바꾸기
    # train vkitti
    rgb_clips, depth_clips = get_data_list(root_dir=kitti_path, data_name="kitti", split="train", clip_len=CLIP_LEN)
    kitti_train = KITTIVideoDataset(rgb_paths=rgb_clips, depth_paths=depth_clips, resize_size=518, split="train")  # 원래 resize_size를
    # val vkitti
    val_rgb_clips, val_depth_clips, val_cam_ids, val_intrin_clips, val_extrin_clips = get_data_list(
        root_dir=kitti_path, data_name="kitti", split="val", clip_len=CLIP_LEN
    )
    kitti_val = KITTIVideoDataset(
        rgb_paths=val_rgb_clips, depth_paths=val_depth_clips, cam_ids=val_cam_ids, intrin_clips=val_intrin_clips, extrin_clips=val_extrin_clips,
        resize_size=518, split="val",
    )
    # dataloader 생성
    kitti_train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True,  num_workers=4)
    kitti_val_loader   = DataLoader(kitti_val,   batch_size=batch_size, shuffle=False, num_workers=4)

    # scannet
    x_scannet, y_scannet, scannet_poses, scannet_Ks = get_list("", "scannet")
    scannet_data = ValDataset(img_paths=x_scannet, depth_paths=y_scannet, data_name="scannet", Ks=scannet_Ks, pose_paths=scannet_poses)
    scannet_val_loader = DataLoader(scannet_data, batch_size=batch_size, shuffle=False, num_workers=4)

    
    # --------------------- model 정의 ---------------------
    teacher = VideoDepthTeacher(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN).to(device)  # vits -> feature: 64, [48,96,192,384]
    student = VideoDepthStudent(encoder="vits", features=64, out_channels=[48,96,192,384], num_frames=CLIP_LEN, ).to(device)  # bidirectional_update_len...
    # Teacher-Student의 forward등을 나눠서 관리하기 위한 class
    class TeacherStudentWrapper(torch.nn.Module):
        def __init__(self, teacher, student):
            super().__init__()
            self.teacher = teacher
            self.student = student
            
        def forward(self, x, prev_depth=None):
            return self.student.forward(x)
        
        def forward_feature(self, x):
            return self.student.forward_features(x)

        def forward_depth(self, features, x_shape, cache=None):
            return self.student.forward_depth(features, x_shape, cache)

    model = TeacherStudentWrapper(teacher, student)
    
    # Load Pretrained Weight
    if args.pretrained_ckpt:
        logger.info(f"Loading Weight from {args.pretrained_ckpt}")
        state_dict = torch.load(args.pretrained_ckpt, map_location="cpu")
        # teacher와 student 모두 
        model.teacher.load_state_dict(state_dict, strict=True)
        model.student.load_state_dict(state_dict, strict=True)
        logger.info("Pretrained weights loaded successfully!")
    
    # Teacher는 전체 model 완전히 freeze
    for p in model.teacher.parameters():
        p.requires_grad = False
    # Student의 encoder freeze, head만 학습
    for p in model.student.pretrained.parameters():
        p.requires_grad = False
    for p in model.student.head.parameters():
        p.requires_grad = True
        
    model.train()
    # Teacher 검증
    teacher_trainable = sum(p.numel() for p in model.teacher.parameters() if p.requires_grad)
    teacher_total = sum(p.numel() for p in model.teacher.parameters())

    # Student 검증
    student_encoder_trainable = sum(p.numel() for p in model.student.pretrained.parameters() if p.requires_grad)
    student_head_trainable = sum(p.numel() for p in model.student.head.parameters() if p.requires_grad)
    student_head_total = sum(p.numel() for p in model.student.head.parameters())

    # logging
    logger.info(f"Teacher: {teacher_trainable:,} trainable / {teacher_total:,} total")
    logger.info(f"Student Encoder: {student_encoder_trainable:,} trainable (should be 0)")
    logger.info(f"Student Head: {student_head_trainable:,} trainable / {student_head_total:,} total")

    # gpu 분산처리
    model.teacher = model.teacher.to('cuda:1')
    model.student = model.student.to('cuda:0')
    logger.info(f"Teacher model on GPU 1 (inference only)")
    logger.info(f"Student model on GPU 0 (training)")
    
    # Optimizer&Scheduler
    student_params = [p for p in model.student.parameters() if p.requires_grad] # KD과정에서 projection layer 생성 가능,,
    optimizer = torch.optim.AdamW(student_params, lr=lr, weight_decay=1e-4)
    total_params = sum(p.numel() for p in model.student.parameters()) + sum(p.numel() for p in model.teacher.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Optimizer parameter groups: {len(optimizer.param_groups)}")
    
    # loss 설정
    loss_tgm = LossTGMVector(diff_depth_th=0.05)  # 기존에 사용하던 loss
    loss_ssi = Loss_ssi_basic()  # 이것도
    
    # model training logging setting
    wandb.watch(model.student, log="all")
    best_scannet_delta1 = 0.0
    best_epoch = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")
    latest_model_path = os.path.join(OUTPUT_DIR, "latest_model.pth")
    
    # Mixed Precision Training Setting
    scaler = torch.amp.GradScaler()
    
    # --------------------- Training Loop ---------------------
    for epoch in tqdm(range(0, num_epochs), desc="Epoch", leave=False):
        model.train()
        epoch_loss = 0.0
        accum_loss = 0.0
        step_in_window = 0
        epoch_frames = 0
        epoch_ssi_loss = 0.0
        epoch_tgm_loss = 0.0
        epoch_kd_loss = 0.0
        scale_list = []  # a*
        shift_list = []  # b*
        
        # batch-level tqdm
        batch_pbar = tqdm(enumerate(kitti_train_loader),
                        desc=f"Epoch {epoch+1}/{num_epochs} - Batches",
                        leave=False,
                        total=len(kitti_train_loader),
                        )
        for batch_idx, (x, y) in batch_pbar:
            x, y = x.to(device), y.to(device)
            B, T = x.shape[:2]
            
            # tracking loss in batch
            batch_loss_sum = 0.0
            batch_frame_count = 0
            
            # Batch pregress update
            current_loss_display = batch_loss_sum / max(1, batch_frame_count)
            batch_pbar.set_postfix({
                'Loss': f'{current_loss_display:.4f}',
                'Frames': epoch_frames,
                'GPU_Mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })
        
            # 상태 정의
            cache = None
            prev_pred_raw = None
            prev_mask = None
            prev_y = None
            
            teacher_frame_buffer = None
            student_frame_count = 0  # student는 stream이라 1개씩 cnt
            
            # frame-level tqdm
            frame_pbar = tqdm(range(T),
                            desc=f"Batch {batch_idx+1} - Frames",
                            leave=False,
                            disable=T<10,  # 10 frame미만이면 tqdm 비활성
                            )
            for t in frame_pbar:
                x_t = x[:, t:t+1]  # x = RGB img, x.shape=[B,T,3,H,W]
                mask_t = get_mask(y[:, t:t+1], 1e-3, 80.0).to(device)
            
                # Teacher buffer managing
                if teacher_frame_buffer is None:
                    teacher_frame_buffer = x_t.detach().clone().repeat(1, CLIP_LEN, 1, 1, 1)  # [B,32,C,H,W]
                else:
                    teacher_frame_buffer = torch.cat([
                        teacher_frame_buffer[:, :1, :, :, :],   # 0번째는 whole-anchor로 남기고 1번째 삭제
                        teacher_frame_buffer[:, 2:, :, :, :],
                        x_t.detach().clone()
                    ], dim=1) # [B,32,C,H,W]

                # Teacher-Student helper function
                def enable_attention_caching(module):
                    for _, layer in module.named_modules():
                        if hasattr(layer, 'enable_kd_caching'):
                            layer.enable_kd_caching(True)

                def disable_attention_caching(module):
                    for _, layer in module.named_modules():
                        if hasattr(layer, 'enable_kd_caching'):
                            layer.enable_kd_caching(False)

                def collect_attention_outputs(module, clear=True):
                    outs = []
                    for _, layer in module.named_modules():
                        if hasattr(layer, 'get_cached_attention_output'):
                            out = layer.get_cached_attention_output()
                            if out is not None:
                                outs.append(out)
                                if clear:
                                    if hasattr(layer, 'clear_attention_cache'):
                                        layer.clear_attention_cache()
                                    elif hasattr(layer, '_kd_cache'):
                                        layer._kd_cache = None
                    return outs
                
                teacher_device = next(model.teacher.parameters()).device
                student_device = next(model.student.parameters()).device

                with torch.no_grad():
                    enable_attention_caching(model.teacher)
                    # Teacher 입력은 teacher_device 로
                    teacher_in = teacher_frame_buffer.to(teacher_device, non_blocking=True)
                    teacher_predictions = model.teacher(teacher_in)                      # on teacher_device
                    # KD/후처리를 위해 예측은 student_device 로 옮김
                    teacher_pred_t = teacher_predictions[:, -1].to(student_device, non_blocking=True)
                    teacher_pred_t = to_BHW_pred(teacher_pred_t).clamp(min=1e-6)

                    teacher_attn = collect_attention_outputs(model.teacher, clear=True)  # list of [B, P, C] (각 텐서는 teacher_device)
                    if len(teacher_attn) == 0:
                        disable_attention_caching(model.teacher)
                        raise RuntimeError("No teacher attention outputs collected")
                    disable_attention_caching(model.teacher)

                # 이제 teacher <-> student 계산
                with torch.amp.autocast(device_type="cuda"):
                    enable_attention_caching(model.student)
                    # Student 입력은 student_device 로
                    x_t = x_t.to(student_device, non_blocking=True)
                    pred_t_raw, cache = model_stream_step(
                        model.student, x_t, cache,
                        bidirectional_update_length=bidirectional_update_length,
                        current_frame=student_frame_count
                    )
                    pred_t_raw = to_BHW_pred(pred_t_raw).clamp(min=1e-6)

                    student_attn = collect_attention_outputs(model.student, clear=True)  # list of [B, P, C] (각 텐서는 student_device)
                    disable_attention_caching(model.student)

                    student_frame_count += 1

                    # Attention-based Knowledge Distillation (마지막 레이어 사용)
                    k = min(len(teacher_attn), len(student_attn)) - 1
                    # Teacher의 캐시 텐서를 student_device 로 정렬 (중요)
                    t_cur = teacher_attn[k].to(student_device, non_blocking=True)  # [B, P, C] on student_device
                    s_cur = student_attn[k]                                        # [B, P, C] on student_device

                    cos_loss = (1.0 - F.cosine_similarity(
                        t_cur.flatten(0, 1),  # [B*P, C]
                        s_cur.flatten(0, 1),  # [B*P, C]
                        dim=1
                    ).mean())

                    kd_loss = cos_loss # + frame_feature_loss  # output으로 KD loss추가하려면 여기서 합치면 될 듯 
                        # 위에 가중치 부분도 넣어줘야함. config에서 이름 수정해야할 듯


                    # Scale & Shift invariant loss
                    gt_disp_t = (1.0 / y[:, t:t+1].clamp(min=1e-6))  # [B,1,1,H,W]
                    gt_disp_t = gt_disp_t.squeeze(2)  # [B,1,H,W]
                    with torch.no_grad():
                        a_star, b_star = batch_ls_scale_shift(pred_t_raw, gt_disp_t, mask_t)
                        # logging
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

                    pred_t_aligned_disp = (a_star.detach() * pred_t_raw.unsqueeze(1) + b_star.detach()).squeeze(1)  # [B,H,W], aligned disparity
                    pred_t_aligned_depth = 1.0 / (pred_t_aligned_disp.clamp(min=1e-6))
                    gt_depth_t = y[:, t:t+1].squeeze(1).squeeze(2)

                    disp_normed_t = norm_ssi(y[:,t:t+1], mask_t).squeeze(2)  # [B,1,H,W] - normalized GT disparity
                    mask4 = mask_t.squeeze(2)
                    pred4 = pred_t_aligned_disp.unsqueeze(1)                   # [B,1,H,W] - aligned disparity

                    assert pred4.dim()==4 and pred4.size(1)==1,  f"pred4 {pred4.shape}"
                    assert disp_normed_t.dim()==4 and disp_normed_t.size(1)==1, f"disp {disp_normed_t.shape}"
                    assert mask4.dim()==4 and mask4.size(1)==1, f"mask4 {mask4.shape}"

                    ssi_loss_t = loss_ssi(pred4, disp_normed_t, mask4)


                    # Temporal Gradient Matching Loss
                    if t>0:
                        prev_aligned_disp = (a_star.detach() * prev_pred_raw.unsqueeze(1) + b_star.detach()).squeeze(1)  # [B, H, W]

                        prev_aligned_depth = 1.0 / (prev_aligned_disp.clamp(min=1e-6))
                        curr_aligned_depth = pred_t_aligned_depth

                        pred_pair = torch.stack([prev_aligned_depth, curr_aligned_depth], dim=1)   # [B,2,H,W]
                        y_pair    = torch.cat([prev_y, y[:, t:t+1]], dim=1)            # [B,2,1,H,W]
                        m_pair    = torch.cat([prev_mask, mask_t], dim=1)              # [B,2,1,H,W]
                        tgm_loss  = loss_tgm(pred_pair, y_pair, m_pair.squeeze(2))
                    else:
                        tgm_loss  = pred_t_raw.new_tensor(0.0)

                    # loss final~
                    loss = cache_distill_weight * kd_loss + ratio_ssi * ssi_loss_t + ratio_tgm * tgm_loss
                    current_ssi_loss = ssi_loss_t
                    current_tgm_loss = tgm_loss

                # Batch-level loss, update_frequency를 활용해서 큰 batch size처럼 동작할 수 있게 함, vram이 작아도 update_frequency를 활용
                batch_loss_sum += loss.item() * B
                batch_frame_count += B

                # 누적,업데이트
                loss =loss / update_frequency
                accum_loss += loss
                step_in_window += 1

                # cache & state는 detach
                cache = _detach_cache(cache)
                prev_pred_raw = pred_t_raw.detach()
                prev_mask = mask_t
                prev_y = y[:, t:t+1]

                # step_in_window가 update_frequency가 되면
                if step_in_window == update_frequency:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(accum_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += accum_loss.item()
                    accum_loss = 0.0
                    step_in_window = 0

                epoch_frames += B
                epoch_ssi_loss += ssi_loss_t.item() * B
                epoch_tgm_loss += tgm_loss.item() * B
                epoch_kd_loss += kd_loss.item() * B

                def safe_item(tensor_val):
                    return tensor_val.item() if torch.is_tensor(tensor_val) else float(tensor_val)

                # frame progress bar update
                frame_pbar.set_postfix({
                    'SSI': f'{safe_item(epoch_ssi_loss):.4f}',
                    'TGM': f'{safe_item(epoch_tgm_loss):.4f}',
                    'KD': f'{safe_item(epoch_kd_loss):.4f}'
                })

                # batch progress bar update
                current_loss_display = batch_loss_sum / max(1, batch_frame_count) if batch_frame_count > 0 else 0.0
                batch_pbar.set_postfix({
                    'Loss': f'{current_loss_display:.4f}',
                    'Frames': epoch_frames,
                    'GPU_Mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
                })
                    
            frame_pbar.close()
            
            # 주요 tensor들 메모리 정리
            del loss, ssi_loss_t
            if 'pred_t_aligned_disp' in locals():
                del pred_t_aligned_disp
            if 'pred_t_aligned_depth' in locals():
                del pred_t_aligned_depth
            if 'tgm_loss' in locals():
                del tgm_loss
                
        batch_pbar.close()
        
        denom_batches = len(kitti_train_loader)
        avg_kitti_train_loss = epoch_loss / max(1, denom_batches)
        # 에폭 단위 평균 계산
        if epoch_frames > 0:
            mean_ssi = epoch_ssi_loss / epoch_frames
            mean_tgm = epoch_tgm_loss / epoch_frames
            mean_kd = epoch_kd_loss / epoch_frames
        else:
            mean_ssi = mean_tgm = mean_kd = 0.0
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
        if (scale_filtered > 0 or shift_filtered > 0) and epoch == 0:
            logger.warning(f"Filtered non-finite scale/shift entries (scale {scale_filtered}, shift {shift_filtered}) in epoch {epoch}")

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
            "train/epoch_kd": mean_kd,
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
        
        log_dict.update({
                "scale_shift/a_mean": scale_mean,
                "scale_shift/a_std": scale_std,
                "scale_shift/b_mean": shift_mean,
                "scale_shift/b_std": shift_std,
        })
        # 이미지 로그 (필요 최소화)
        log_dict["vkitti/pred_disparity"] = kitti_wb_images
        log_dict["scannet/pred_disparity"] = scannet_wb_images
        wandb.log(log_dict)
        del kitti_wb_images
        del scannet_wb_images
        
        # ── 모델 저장 (ScanNet AbsRel 기준) ──
        if scannet_delta1 < best_scannet_delta1:
            best_scannet_delta1 = scannet_delta1
            best_epoch = epoch
            model_state = model.student.state_dict()

            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_scannet_delta1": best_scannet_delta1,
                "scannet_val_loss": scannet_val_loss,
                "scannet_delta1": scannet_delta1,
                "scannet_tae": scannet_tae,
                "config": hyper_params,
            }, best_model_path)

            logger.info(f"🏆 Best model saved! Epoch {epoch}, ScanNet Delta1: {best_scannet_delta1:.4f}")
            wandb.log({
                "best_scannet_delta1": best_scannet_delta1,
                "best_epoch": epoch,
                "epoch": epoch,
            })
        
        # latest 저장
        latest_model_state = model.student.state_dict()

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
    logger.info("=" * 30)
    logger.info("Training Completed!")
    logger.info(f"Total Epochs: {num_epochs}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best ScanNet Delta1: {best_scannet_delta1:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Latest model saved to: {latest_model_path}")
    logger.info("=" * 30)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/video_depth_anything_vits.pth")
    args = parser.parse_args()
    train(args)