"""
모델 저장/로딩을 위한 유틸리티 함수들
"""
import torch
import os
import logging
from video_depth_anything.video_depth_stream import VideoDepthAnything

logger = logging.getLogger(__name__)

def save_model_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, config=None, is_best=False):
    """
    모델 체크포인트 저장
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config,
        'is_best': is_best
    }
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    logger.info(f"{'Best ' if is_best else ''}Model checkpoint saved to {save_path}")

def load_model_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    모델 체크포인트 로딩
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 상태 로딩
    if hasattr(model, 'module'):  # DataParallel case
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer 상태 로딩
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Scheduler 상태 로딩
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Model checkpoint loaded from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint

def create_model_from_config(config, device='cuda'):
    """
    설정에서 모델 생성
    """
    model = VideoDepthAnything(
        num_frames=config.get('clip_len', 16),
        use_causal_mask=True,
        encoder=config.get('encoder', 'vits'),
        features=config.get('features', 64),
        out_channels=config.get('out_channels', [48, 96, 192, 384])
    ).to(device)
    
    return model

def load_pretrained_weights(model, pretrained_path, strict=False):
    """
    사전 훈련된 가중치 로딩
    """
    if not os.path.exists(pretrained_path):
        logger.warning(f"Pretrained weights not found: {pretrained_path}")
        return
    
    logger.info(f"Loading pretrained weights from {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    
    if strict:
        model.load_state_dict(state_dict)
    else:
        # 호환되는 가중치만 로딩
        model_dict = model.state_dict()
        compatible_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.size() == model_dict[k].size()}
        
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        
        logger.info(f"Loaded {len(compatible_dict)}/{len(state_dict)} compatible weights")

def cleanup_old_checkpoints(checkpoint_dir, keep_latest=5, pattern="checkpoint_epoch_"):
    """
    오래된 체크포인트 정리 (latest model 제외)
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if pattern in file and file.endswith('.pth'):
            if 'latest' not in file and 'best' not in file:  # latest, best 모델은 보존
                filepath = os.path.join(checkpoint_dir, file)
                checkpoints.append((filepath, os.path.getctime(filepath)))
    
    # 시간순 정렬 (최신 순)
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    
    # 최신 N개를 제외하고 삭제
    for filepath, _ in checkpoints[keep_latest:]:
        try:
            os.remove(filepath)
            logger.info(f"Removed old checkpoint: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to remove {filepath}: {e}")

def get_model_info(model):
    """
    모델 정보 출력
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
    }
    
    logger.info("Model Info:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable percentage: {info['trainable_percentage']:.1f}%")
    
    return info
