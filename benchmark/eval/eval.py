
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

import argparse
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import os
import gc

import torch
from metric import *
import metric
import wandb
from dotenv import load_dotenv


device = 'cuda'
eval_metrics = [
    "abs_relative_difference",
    "rmse_linear",
    "delta1_acc",
]

# length_sweep 파싱 함수
def parse_length_sweep(length_sweep_str):
    if not length_sweep_str:
        return None
    arr = []
    for x in length_sweep_str.split(','):
        x = x.strip()
        if x:
            arr.append(int(x))
    return sorted(list(set(arr)))

# 입력된 추론 파일에서 깊이 맵을 로드하는 함수
def get_infer(infer_path,args, target_size = None):
    if infer_path.split('.')[-1] == 'npy':
        img_gray = np.load(infer_path)
        img_gray = img_gray.astype(np.float32)
        infer_factor = 1.0
    else: 
        img = cv2.imread(infer_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float32)
        infer_factor = 1.0 / 255.0

    infer = img_gray / infer_factor
    
    if target_size is not None:
        if infer.shape[0] != target_size[0] or infer.shape[1] != target_size[1]:
            infer = cv2.resize(infer, (target_size[1], target_size[0]))
    return infer

# GT 깊이 맵을 로드하는 함수
def get_gt(depth_gt_path, gt_factor, args):
    if depth_gt_path.split('.')[-1] == 'npy':
        depth_gt = np.load(depth_gt_path)
    else:
        depth_gt = cv2.imread(depth_gt_path, -1)
        depth_gt = np.array(depth_gt)
    depth_gt = depth_gt / gt_factor
    depth_gt[depth_gt==0] = -1
    return depth_gt

# flow 로드 함수
def get_flow(flow_path):
    assert os.path.exists(flow_path)
    flow = np.load(flow_path, allow_pickle=True)
    return flow

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

# 기존 시그니처에 선택 인자만 추가 (기본값은 기존 동작 유지)
def eval_depthcrafter(infer_paths, depth_gt_paths, factors, args, seq_length=None):
    depth_errors = []
    gts = []
    infs = []
    # 길이 결정: 인자를 주면 그걸 쓰고, 아니면 기존처럼 args.max_eval_len 사용
    if seq_length is None:
        seq_length = args.max_eval_len

    dataset_max_depth = args.max_depth_eval
    for i in range(len(infer_paths)):
        if not os.path.exists(infer_paths[i]):
            continue
        depth_gt = get_gt(depth_gt_paths[i], factors[i], args)
        depth_gt = depth_gt[args.a:args.b, args.c:args.d]
        
        infer = get_infer(infer_paths[i], args, target_size=depth_gt.shape)
        gts.append(depth_gt)
        infs.append(infer)

    if len(gts) == 0:
        return [np.nan, np.nan, np.nan]

    gts = np.stack(gts, axis=0)
    infs = np.stack(infs, axis=0)

    # 여기서 seq_length 적용 (기존: args.max_eval_len만 사용)
    infs = infs[:seq_length]
    gts = gts[:seq_length]

    valid_mask = np.logical_and((gts>1e-3), (gts<dataset_max_depth))
    if valid_mask.sum() == 0:
        return [np.nan, np.nan, np.nan]

    gt_disp_masked = 1. / (gts[valid_mask].reshape((-1,1)).astype(np.float64) + 1e-8)
    infs = np.clip(infs, a_min=1e-3, a_max=None)
    pred_disp_masked = infs[valid_mask].reshape((-1,1)).astype(np.float64)

    _ones = np.ones_like(pred_disp_masked)
    A = np.concatenate([pred_disp_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]
    scale, shift = X
    aligned_pred = scale * infs + shift
    aligned_pred = np.clip(aligned_pred, a_min=1e-3, a_max=None)

    pred_depth = depth2disparity(aligned_pred)
    gt_depth = gts
    pred_depth = np.clip(pred_depth, a_min=1e-3, a_max=dataset_max_depth)

    sample_metric = []
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    pred_depth_ts = torch.from_numpy(pred_depth).to(device)
    gt_depth_ts = torch.from_numpy(gt_depth).to(device)
    valid_mask_ts = torch.from_numpy(valid_mask).to(device)

    n = valid_mask.sum((-1, -2))
    valid_frame = (n > 0)
    pred_depth_ts = pred_depth_ts[valid_frame]
    gt_depth_ts = gt_depth_ts[valid_frame]
    valid_mask_ts = valid_mask_ts[valid_frame]

    for met_func in metric_funcs:
        _metric = met_func(pred_depth_ts, gt_depth_ts, valid_mask_ts).item()
        sample_metric.append(_metric)
    return sample_metric

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='')
    parser.add_argument('--infer_type', type=str, default='npy')
    parser.add_argument('--benchmark_path', type=str, default='')
    parser.add_argument('--datasets', type=str, nargs='+', default=['vkitti', 'kitti', 'sintel', 'nyu_v2', 'tartanair', 'bonn', 'ip_lidar'])
    # --- 길이 스윕 / 드롭 탐지 옵션 ---
    parser.add_argument('--length_sweep', type=str, default='', help='예: "32,64,90,128,256,500" (빈 값이면 비활성)')
    parser.add_argument('--drop_tol', type=float, default=0.01, help='δ1 하락을 드롭으로 간주할 임계치(절대값, 0.01=1pp)')
    # --- wandb 옵션 (최소 추가) ---
    parser.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='depth-eval', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='', help='wandb run name')
    parser.add_argument('--wandb_group', type=str, default='', help='wandb group')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'], help='wandb mode')
    
    args = parser.parse_args()

    length_sweep = parse_length_sweep(args.length_sweep)
    results_save_path = os.path.join(args.infer_path, 'results.txt')
    
    # --- wandb 초기화 (옵션) ---
    use_wandb = (args.wandb and wandb is not None and args.wandb_mode != 'disabled')
    if use_wandb:
        wandb.login(key="edafa2d4d3d64c0268e9e7856783c659b9c255e3", relogin=True)
        wandb.init(
            project=args.wandb_project,
            name=(args.wandb_run_name if args.wandb_run_name else None),
            group=(args.wandb_group if args.wandb_group else None),
            mode=args.wandb_mode,
            config={
                'infer_path': args.infer_path,
                'benchmark_path': args.benchmark_path,
                'datasets': args.datasets,
                'eval_metrics': eval_metrics,
                'length_sweep': length_sweep,
                'drop_tol': args.drop_tol,
            }
        )
        global_step = 0  # wandb 스텝 카운터(옵션)

    for dataset in args.datasets:

        file = open(results_save_path, 'a')

        # if dataset == 'kitti':
        #     args.json_file = os.path.join(args.benchmark_path,'kitti/kitti_video.json')
        #     args.root_path = os.path.join(args.benchmark_path,'kitti')
        #     args.max_depth_eval = 80.0
        #     args.min_depth_eval = 0.1
        #     args.max_eval_len = 110
        #     args.a = 0
        #     args.b = 374
        #     args.c = 0
        #     args.d = 1242
        # if dataset == 'kitti_500':
        #     dataset = 'kitti'
        #     args.json_file = os.path.join(args.benchmark_path,'kitti/kitti_video_500.json')
        #     args.root_path = os.path.join(args.benchmark_path,'kitti')
        #     args.max_depth_eval = 80.0
        #     args.min_depth_eval = 0.1
        #     args.max_eval_len = 500
        #     args.a = 0
        #     args.b = 374
        #     args.c = 0
        #     args.d = 1242
        # elif dataset == 'sintel':
        #     args.json_file = os.path.join(args.benchmark_path,'sintel/sintel_video.json')
        #     args.root_path = os.path.join(args.benchmark_path,'sintel')
        #     args.max_depth_eval = 70
        #     args.min_depth_eval = 0.1
        #     args.max_eval_len = 100
        #     args.a = 0
        #     args.b = 436
        #     args.c = 0
        #     args.d = 1024
        # elif dataset == 'nyuv2_500':
        #     dataset = 'nyuv2'
        #     args.json_file = os.path.join(args.benchmark_path,'nyuv2/nyuv2_video_500.json')
        #     args.root_path = os.path.join(args.benchmark_path,'nyuv2')
        #     args.max_depth_eval = 10.0
        #     args.min_depth_eval = 0.1
        #     args.max_eval_len = 500
        #     args.a = 45
        #     args.b = 471
        #     args.c = 41
        #     args.d = 601
        # elif dataset == 'bonn':
        #     args.json_file = os.path.join(args.benchmark_path,'bonn/bonn_video.json')
        #     args.root_path = os.path.join(args.benchmark_path,'bonn')
        #     args.max_depth_eval = 10.0
        #     args.min_depth_eval = 0.1
        #     args.max_eval_len = 110
        #     args.a = 0
        #     args.b = 480
        #     args.c = 0
        #     args.d = 640
        # elif dataset == 'bonn_500':
        #     dataset = 'bonn'
        #     args.json_file = os.path.join(args.benchmark_path,'bonn/bonn_video_500.json')
        #     args.root_path = os.path.join(args.benchmark_path,'bonn')
        #     args.max_depth_eval = 10.0
        #     args.min_depth_eval = 0.1
        #     args.max_eval_len = 500
        #     args.a = 0
        #     args.b = 480
        #     args.c = 0
        #     args.d = 640
        # elif dataset == 'scannet':
        #     args.json_file = os.path.join(args.benchmark_path,'scannet/scannet_video.json')
        #     args.root_path = os.path.join(args.benchmark_path,'scannet')
        #     args.max_depth_eval = 10.0
        #     args.min_depth_eval = 0.1
        #     args.max_eval_len = 90
        #     args.a = 8
        #     args.b = -8
        #     args.c = 11
        #     args.d = -11
        # elif dataset == 'scannet_500':

        # 의심의 여지없이 scannet_500만 사용
        dataset = 'scannet'
        args.json_file = os.path.join(args.benchmark_path,'scannet/scannet_video_500.json')
        args.root_path = os.path.join(args.benchmark_path,'scannet')
        args.max_depth_eval = 10.0
        args.min_depth_eval = 0.1
        args.max_eval_len = 500
        args.a = 8
        args.b = -8
        args.c = 11
        args.d = -11

        with open(args.json_file, 'r') as fs:
            path_json = json.load(fs)

        json_data = path_json[dataset]
        line = '-' * 50
        print(f'<{line} {dataset} start {line}>')
        file.write(f'<{line} {dataset} start {line}>\n')

        # W&B 테이블 (선택)
        seq_table = None
        if use_wandb:
            seq_table = wandb.Table(columns=[
                'dataset','sequence','length','abs_rel','rmse','delta1'
            ])

        results_all = []
        drop_lens = []

        # ✅ 추가: 씬 인덱스(그래프 x축용)
        scene_idx = 0

        for data in tqdm(json_data):
            for key in data.keys():
                value = data[key]

                infer_paths = []
                depth_gt_paths = []
                factors = []
                for images in value:
                    infer_path = (args.infer_path + '/' + dataset + '/' + images['image']) \
                                .replace('.jpg', '.npy').replace('.png', '.npy')
                    infer_paths.append(infer_path)
                    depth_gt_paths.append(args.root_path + '/' + images['gt_depth'])
                    factors.append(images['factor'])

                # 캐시: 길이 L -> [abs_rel, rmse, delta1]
                computed_metrics = {}

                # ① 기본 길이(= max_eval_len) 평가 (전통 기준)
                base_metrics = eval_depthcrafter(
                    infer_paths, depth_gt_paths, factors, args,
                    seq_length=args.max_eval_len
                )
                results_all.append(base_metrics)
                computed_metrics[args.max_eval_len] = base_metrics  # 캐시에 저장

                if use_wandb and seq_table is not None:
                    seq_table.add_data(
                        dataset, str(key),
                        int(min(args.max_eval_len, len(infer_paths))),
                        base_metrics[0], base_metrics[1], base_metrics[2]
                    )

                # ✅ 추가: base=500도 len_sweep 시리즈로 즉시 스칼라 로그
                if use_wandb:
                    wandb.log({
                        'dataset': dataset,
                        'sequence': str(key),
                        'scene_idx': scene_idx,
                        f'len_sweep/abs_rel@{args.max_eval_len}': base_metrics[0],
                        f'len_sweep/rmse@{args.max_eval_len}': base_metrics[1],
                        f'len_sweep/delta1@{args.max_eval_len}': base_metrics[2],
                    }, step=scene_idx)

                # ② 길이 스윕: 중복 계산 방지
                if length_sweep:
                    base_L = min(length_sweep)

                    if base_L in computed_metrics:
                        base_at_L = computed_metrics[base_L]
                    else:
                        base_at_L = eval_depthcrafter(
                            infer_paths, depth_gt_paths, factors, args,
                            seq_length=base_L
                        )
                        computed_metrics[base_L] = base_at_L

                    base_d1 = base_at_L[2]
                    drop_len = None

                    for L in length_sweep:
                        if L in computed_metrics:
                            metL = computed_metrics[L]
                        else:
                            metL = eval_depthcrafter(
                                infer_paths, depth_gt_paths, factors, args,
                                seq_length=L
                            )
                            computed_metrics[L] = metL

                        if use_wandb and seq_table is not None:
                            seq_table.add_data(
                                dataset, str(key),
                                int(min(L, len(infer_paths))),
                                metL[0], metL[1], metL[2]
                            )

                        # ✅ 추가: L 길이 결과도 즉시 스칼라 로그 (씬별 x=scene_idx 에 점 1개씩)
                        if use_wandb:
                            wandb.log({
                                'dataset': dataset,
                                'sequence': str(key),
                                'scene_idx': scene_idx,
                                f'len_sweep/abs_rel@{L}': metL[0],
                                f'len_sweep/rmse@{L}': metL[1],
                                f'len_sweep/delta1@{L}': metL[2],
                            }, step=scene_idx)

                        # 드롭 탐지
                        if (not np.isnan(base_d1)) and (not np.isnan(metL[2])) and drop_len is None:
                            if (base_d1 - metL[2]) >= args.drop_tol:
                                drop_len = L

                    if drop_len is not None:
                        drop_lens.append(drop_len)

                # --- wandb: 시퀀스별 기본 로그 (기존 유지)
                if use_wandb:
                    log_dict = {
                        'dataset': dataset,
                        'sequence': str(key),
                        f'{eval_metrics[0]}': base_metrics[0],
                        f'{eval_metrics[1]}': base_metrics[1],
                        f'{eval_metrics[2]}': base_metrics[2],
                    }
                    wandb.log(log_dict)

                # ✅ 추가: 다음 씬으로 step 증가
                scene_idx += 1

        # 데이터셋 평균 출력/저장
        def safe_mean(arr, idx=None):
            arr = np.array(arr, dtype=np.float64)
            if idx is not None:
                arr = arr[:, idx]
            arr = arr[~np.isnan(arr)]
            return float(arr.mean()) if arr.size > 0 else float('nan')

        final_arr = np.array(results_all) if len(results_all) > 0 else np.empty((0,3))
        off_mean = [safe_mean(final_arr, i) for i in range(3)]
        mean_drop_len = safe_mean(drop_lens) if len(drop_lens) > 0 else np.nan

        for i, m in enumerate(eval_metrics):
            print(f"{m}: {off_mean[i]:.6f}")
            file.write(f"{m}: {off_mean[i]:.6f}\n")
        if length_sweep:
            print(f"mean_drop_len: {mean_drop_len:.2f}")
            file.write(f"mean_drop_len: {mean_drop_len:.2f}\n")

        file.write(f'<{line} {dataset} finish {line}>\n')

        # --- wandb: 데이터셋 평균 로그 + 테이블 업로드
        if use_wandb:
            log_mean = {
                f'{dataset}_mean/{eval_metrics[0]}': off_mean[0],
                f'{dataset}_mean/{eval_metrics[1]}': off_mean[1],
                f'{dataset}_mean/{eval_metrics[2]}': off_mean[2],
            }
            if length_sweep:
                log_mean.update({
                    f'{dataset}_mean/drop_len': mean_drop_len
                })
            wandb.log(log_mean)

            if seq_table is not None and len(seq_table.data) > 0:
                wandb.log({f'{dataset}/sequences_offline': seq_table})

if __name__ == '__main__':
    main()