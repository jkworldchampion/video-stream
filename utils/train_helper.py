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

    # autocast 비활성화 + float64 정밀도 보장 (lstsq 안정성)
    with autocast(enabled=False):
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

def model_stream_step(model, x_t, cache=None, bidirectional_update_length=16, current_frame=0):
    """
    x_t: [B,1,3,H,W] (single-frame step)
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
                features, x_t.shape, cache, None, 
                bidirectional_update_length=bidirectional_update_length,
                current_frame=current_frame
            )
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
                temp_features, x_t.shape, cache, None,
                bidirectional_update_length=bidirectional_update_length,
                current_frame=current_frame
            )
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
                    model, x_t, cache,
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
                # autocast 비활성화 + float32 캐스팅 (lstsq 안정성 보장)
                with autocast(enabled=False):
                    p_flat   = raw_disp.float().view(Bv, -1)
                    g_flat   = gt_disp.float().view(Bv, -1)

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
    
    # autocast 비활성화 + float32 캐스팅 (lstsq 안정성 보장)
    with autocast(enabled=False):
        m = mask.view(B, -1).float()                      # [B, P]
        p_flat = p.float().view(B, -1)                    # [B, P]
        g_flat = g.float().view(B, -1)                    # [B, P]

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