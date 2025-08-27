import numpy as np
import cv2
import json
import argparse
from tqdm import tqdm
import os
import gc
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_errors_torch(gt, pred):
    return torch.mean(torch.abs(gt - pred) / gt)

def get_infer(path, args, target_size=None):
    if path.endswith('.npy'):
        arr = np.load(path).astype(np.float32)
        factor = 1.0
    else:
        img = cv2.imread(path)
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        factor = 1.0/255.0
    infer = arr / factor
    if args.hard_crop:
        infer = infer[args.a:args.b, args.c:args.d]
    if target_size and (infer.shape!=tuple(target_size)):
        infer = cv2.resize(infer, (target_size[1], target_size[0]))
    return infer

def get_gt(path, factor, args):
    if path.endswith('.npy'):
        gt = np.load(path).astype(np.float32)
    else:
        gt = cv2.imread(path, -1).astype(np.float32)
    gt = gt / factor
    gt[gt==0] = 0
    return gt

def depth2disp(depth):
    disp = np.zeros_like(depth)
    mask = depth>0
    disp[mask] = 1.0/depth[mask]
    return disp, mask

def tae_torch(d1, d2, R, T, K, mask):
    H,W = d1.shape
    
    # d1.dtype might be float64 or float32; ensure R and T match it
    R = R.to(device=d1.device, dtype=d1.dtype)
    T = torch.tensor(T, dtype=d1.dtype, device=d1.device)
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    xx,yy = torch.meshgrid(torch.arange(W), torch.arange(H))
    xx,yy = xx.t().to(d1), yy.t().to(d1)
    X = (xx-cx)*d1/fx; Y=(yy-cy)*d1/fy; Z=d1
    pts = torch.stack([X.flatten(), Y.flatten(), Z.flatten()],1)
    T = torch.tensor(T, dtype=d1.dtype, device=d1.device)
    
    pts_t = pts@R.T + T
    X_cam, Y_cam, Z_cam = pts_t[:,0], pts_t[:,1], pts_t[:,2]
    
    # 카메라 투영: 3D -> 2D 이미지 좌표
    # Z_cam가 양수인 점들만 유효 (카메라 앞쪽)
    valid_depth = Z_cam > 1e-6
    
    if valid_depth.sum() == 0:
        return torch.tensor(0.0, device=d1.device)
    
    # 이미지 평면에 투영
    u_proj = fx * X_cam / Z_cam + cx  # x 좌표
    v_proj = fy * Y_cam / Z_cam + cy  # y 좌표
    
    # 이미지 경계 내의 유효한 점들만 선택 (bilinear interpolation을 위해 1픽셀 여유)
    u_int = torch.round(u_proj).long()
    v_int = torch.round(v_proj).long()
    valid_proj = valid_depth & (u_int >= 1) & (u_int < W-1) & (v_int >= 1) & (v_int < H-1)
    
    if valid_proj.sum() == 0:
        return torch.tensor(0.0, device=d1.device)
    
    # 유효한 점들 선택
    u_valid = u_proj[valid_proj]
    v_valid = v_proj[valid_proj]
    z_valid = Z_cam[valid_proj]
    
    # bilinear interpolation을 위한 좌표 계산
    u_floor = torch.floor(u_valid).long()
    v_floor = torch.floor(v_valid).long()
    u_ceil = u_floor + 1
    v_ceil = v_floor + 1
    
    # 보간 가중치
    wu = u_valid - u_floor.float()
    wv = v_valid - v_floor.float()
    
    # 4개 코너에서 d2 및 mask 값 샘플링
    d2_tl = d2[v_floor, u_floor]
    d2_tr = d2[v_floor, u_ceil]
    d2_bl = d2[v_ceil, u_floor]
    d2_br = d2[v_ceil, u_ceil]
    
    mask_tl = mask[v_floor, u_floor]
    mask_tr = mask[v_floor, u_ceil]
    mask_bl = mask[v_ceil, u_floor]
    mask_br = mask[v_ceil, u_ceil]
    
    # bilinear interpolation
    d2_top = d2_tl * (1 - wu) + d2_tr * wu
    d2_bot = d2_bl * (1 - wu) + d2_br * wu
    d2_interp = d2_top * (1 - wv) + d2_bot * wv
    
    # 마스크 interpolation (모든 코너가 유효한 경우만)
    mask_interp = mask_tl & mask_tr & mask_bl & mask_br
    
    # 최종 유효한 점들
    final_valid = mask_interp & (d2_interp > 1e-6) & (z_valid > 1e-6)
    
    if final_valid.sum() == 0:
        return torch.tensor(0.0, device=d1.device)
    
    # 깊이 오차 계산 (절대 차이의 평균)
    error = torch.abs(z_valid[final_valid] - d2_interp[final_valid]).mean()
    return error

def eval_TAE(infer_paths, gt_paths, factors, masks, Ks, poses, args):
    gts, infs = [], []
    for ip, gp, f in zip(infer_paths, gt_paths, factors):
        if not os.path.exists(ip): continue
        gt = get_gt(gp, f, args)[args.a:args.b, args.c:args.d]
        inf = get_infer(ip, args, target_size=gt.shape)
        gts.append(gt); infs.append(inf)
    if not gts:
        raise RuntimeError("No valid frames for TAE evaluation.")
    gts = np.stack(gts); infs = np.stack(infs)
    valid = (gts>1e-3)&(gts<args.max_depth_eval)
    gt_disp = 1.0/(gts[valid].reshape(-1,1)+1e-8)
    infs = np.clip(infs,1e-3,None)
    pred_disp = infs[valid].reshape(-1,1)
    A = np.concatenate([pred_disp, np.ones_like(pred_disp)],1)
    scale, shift = np.linalg.lstsq(A, gt_disp, rcond=None)[0]
    aligned = np.clip(scale*infs+shift,1e-3,None)
    disp, _ = depth2disp(aligned)
    total_error = 0.0
    count = 0
    for i in range(len(disp)-1):
        d1 = torch.from_numpy(disp[i]).to(device)
        d2 = torch.from_numpy(disp[i+1]).to(device)
        R21 = np.linalg.inv(poses[i+1]) @ poses[i]
        # mask_np는 NumPy 배열, 이를 반드시 torch.BoolTensor로 변환
        mask_np = masks[i+1] if args.mask else np.ones_like(disp[i+1], dtype=bool)
        mask = torch.from_numpy(mask_np).to(device=device)
        total_error += tae_torch(
            d1, d2,
            torch.from_numpy(R21[:3,:3]).to(device),
            R21[:3,3],
            Ks[i],
            mask
        )
        count += 2
    return (total_error / count) * 100

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--infer_path',     required=True)
    p.add_argument('--benchmark_path', required=True)
    p.add_argument('--json_file',      required=True,
                   help='Path to scannet_video_tae.json')
    p.add_argument('--datasets', nargs='+', default=['scannet'])
    p.add_argument('--start_idx', type=int, default=10)
    p.add_argument('--end_idx',   type=int, default=180)
    p.add_argument('--eval_scenes_num', type=int, default=20)
    p.add_argument('--hard_crop', action='store_true')
    args = p.parse_args()
    # ScanNet only
    assert args.datasets==['scannet']
    args.max_depth_eval = 10.0
    args.mask = False
    # cropping
    args.a, args.b = 8, -8
    args.c, args.d = 11, -11
    args.root_path = os.path.dirname(args.json_file)

    # load only first N scenes
    meta = json.load(open(args.json_file,'r'))
    scenes = meta['scannet'][:args.eval_scenes_num]

    errors = []
    for scene in tqdm(scenes, desc='TEA eval'):
        frame_list = list(scene.values())[0]
        # infer paths match:
        infer_paths = [
            os.path.join(args.infer_path, info['image'].replace('color_origin','color'))
               .replace('.jpg','.npy').replace('.png','.npy')
            for info in frame_list
        ][args.start_idx:args.end_idx]
        # print("=== Debug: checking infer_paths ===")
        # for ip in infer_paths:
        #     print(ip, "exists?", os.path.exists(ip))
        depth_gt_paths = [
            os.path.join(args.root_path, info['gt_depth'])
            for info in frame_list
        ][args.start_idx:args.end_idx]
        factors = [info['factor'] for info in frame_list][args.start_idx:args.end_idx]
        Ks      = [np.array(info['K'])   for info in frame_list][args.start_idx:args.end_idx]
        poses   = [np.array(info['pose'])for info in frame_list][args.start_idx:args.end_idx]
        masks   = [os.path.join(args.root_path, info['mask']) if args.mask else None
                   for info in frame_list][args.start_idx:args.end_idx]

        # errors.append(
        #     eval_TAE(infer_paths, depth_gt_paths, factors, masks, Ks, poses, args)
        # )
        err = eval_TAE(infer_paths, depth_gt_paths, factors, masks, Ks, poses, args)
        if isinstance(err, torch.Tensor):
            err = err.item()
        errors.append(err)

    print(f"ScanNet TAE: {np.mean(errors):.4f}")
