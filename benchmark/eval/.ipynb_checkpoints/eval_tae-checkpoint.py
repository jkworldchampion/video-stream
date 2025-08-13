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
    Xp, Yp, Zp = pts_t[:,0], pts_t[:,1], pts_t[:,2]
    Xr = torch.round(Xp).long(); Yr = torch.round(Yp).long()
    valid = (Xr>=0)&(Xr<W)&(Yr>=0)&(Yr<H)
    if valid.sum()==0: return 0.0
    proj = torch.zeros_like(d1)
    proj[Yr[valid], Xr[valid]] = Zp[valid]
    valid2 = (proj>0)&(d2>0)&mask
    if valid2.sum()==0: return 0.0
    return compute_errors_torch(d2[valid2], proj[valid2])

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
