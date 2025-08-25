
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

device = 'cuda:1'
eval_metrics = [
    "abs_relative_difference",
    "rmse_linear",
    "delta1_acc",
]

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

def get_gt(depth_gt_path, gt_factor, args):
    if depth_gt_path.split('.')[-1] == 'npy':
        depth_gt = np.load(depth_gt_path)
    else:
        depth_gt = cv2.imread(depth_gt_path, -1)
        depth_gt = np.array(depth_gt)
    depth_gt = depth_gt / gt_factor
    depth_gt[depth_gt==0] = -1
    return depth_gt

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

def eval_depthcrafter(infer_paths, depth_gt_paths, factors, args):
    """
    메모리 안전 + Depth 도메인 정렬(SSI) 평가.
    - 시퀀스 전체 (a,b) 정합을 '누적 합(정규방정식)'으로 계산해 O(1) 메모리
    - 이후 프레임별로 메트릭을 계산해 평균
    """
    seq_length = args.max_eval_len
    dataset_max_depth = args.max_depth_eval

    # ---------- 1st pass: (a,b) 정합 계수 추정 (Depth 도메인) ----------
    # 최소제곱 해: a,b = argmin || a*x + b - y ||^2
    # 누적합으로 계산 (전 프레임, 유효 픽셀 전체):
    # Sx = sum(x), Sy = sum(y), Sxx = sum(x^2), Sxy = sum(x*y), N = #pixels
    Sx = Sy = Sxx = Sxy = 0.0
    N  = 0

    # 시퀀스 길이 제한 고려
    num_frames = min(seq_length, len(infer_paths))

    for i in range(num_frames):
        if not os.path.exists(infer_paths[i]):
            continue
        # GT 로드(+스케일 보정) 및 크롭
        depth_gt = get_gt(depth_gt_paths[i], factors[i], args)
        depth_gt = depth_gt[args.a:args.b, args.c:args.d]

        # 예측(depth) 로드 및 GT 크기에 맞춤
        pred = get_infer(infer_paths[i], args, target_size=depth_gt.shape)
        # 유효 마스크
        valid_mask = (depth_gt > 1e-3) & (depth_gt < dataset_max_depth)

        if not np.any(valid_mask):
            continue

        # 안정화
        pred = np.clip(pred, a_min=1e-3, a_max=None)

        # 누적합 (float64 권장)
        x = pred[valid_mask].astype(np.float64)
        y = depth_gt[valid_mask].astype(np.float64)

        Sx  += x.sum()
        Sy  += y.sum()
        Sxx += np.dot(x, x)          # sum(x^2)
        Sxy += np.dot(x, y)          # sum(x*y)
        N   += x.size

    # (a,b) 계산 (퇴화 대비)
    if N == 0:
        # 유효 픽셀이 전혀 없으면 NaN 반환
        return [float("nan")] * len(eval_metrics)

    denom = (Sxx - (Sx * Sx) / N)
    if abs(denom) < 1e-12:
        a = 1.0
        b = 0.0
    else:
        a = (Sxy - (Sx * Sy) / N) / denom
        b = (Sy - a * Sx) / N

    # ---------- 2nd pass: (a,b) 적용 후 프레임별 메트릭 계산 ----------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]
    metrics_sum = np.zeros(len(metric_funcs), dtype=np.float64)
    frames_count = 0

    for i in range(num_frames):
        if not os.path.exists(infer_paths[i]):
            continue
        depth_gt = get_gt(depth_gt_paths[i], factors[i], args)
        depth_gt = depth_gt[args.a:args.b, args.c:args.d]

        pred = get_infer(infer_paths[i], args, target_size=depth_gt.shape)
        valid_mask = (depth_gt > 1e-3) & (depth_gt < dataset_max_depth)
        if not np.any(valid_mask):
            continue

        # Depth 도메인 정렬 적용
        pred = np.clip(pred, a_min=1e-3, a_max=None)
        pred_aligned = a * pred + b
        pred_aligned = np.clip(pred_aligned, a_min=1e-3, a_max=dataset_max_depth)

        # Torch(CPU) 텐서로 변환 (프레임 단위로만)
        pred_ts = torch.from_numpy(pred_aligned[None, ...])  # [1, H, W]
        gt_ts   = torch.from_numpy(depth_gt[None, ...])
        mask_ts = torch.from_numpy(valid_mask[None, ...])

        # 프레임 1장의 메트릭 계산
        for idx, met_func in enumerate(metric_funcs):
            m = met_func(pred_ts, gt_ts, mask_ts).item()
            metrics_sum[idx] += m

        frames_count += 1

    if frames_count == 0:
        return [float("nan")] * len(eval_metrics)

    # 프레임 평균(원래 코드도 프레임단위 평균)
    metrics_mean = (metrics_sum / frames_count).tolist()
    return metrics_mean



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='')
    parser.add_argument('--infer_type', type=str, default='npy')
    parser.add_argument('--benchmark_path', type=str, default='')
    parser.add_argument('--datasets', type=str, nargs='+', default=['vkitti', 'kitti', 'sintel', 'nyu_v2', 'tartanair', 'bonn', 'ip_lidar'])
    
    args = parser.parse_args()

    results_save_path = os.path.join(args.infer_path, 'results.txt')
   
    for dataset in args.datasets:

        file = open(results_save_path, 'a')

        if dataset == 'kitti':
            args.json_file = os.path.join(args.benchmark_path,'kitti/kitti_video.json')
            args.root_path = os.path.join(args.benchmark_path,'kitti')
            args.max_depth_eval = 80.0
            args.min_depth_eval = 0.1
            args.max_eval_len = 110
            args.a = 0
            args.b = 374
            args.c = 0
            args.d = 1242
        if dataset == 'kitti_500':
            dataset = 'kitti'
            args.json_file = os.path.join(args.benchmark_path,'kitti/kitti_video_500.json')
            args.root_path = os.path.join(args.benchmark_path,'kitti')
            args.max_depth_eval = 80.0
            args.min_depth_eval = 0.1
            args.max_eval_len = 500
            args.a = 0
            args.b = 374
            args.c = 0
            args.d = 1242
        elif dataset == 'sintel':
            args.json_file = os.path.join(args.benchmark_path,'sintel/sintel_video.json')
            args.root_path = os.path.join(args.benchmark_path,'sintel')
            args.max_depth_eval = 70
            args.min_depth_eval = 0.1
            args.max_eval_len = 100
            args.a = 0
            args.b = 436
            args.c = 0
            args.d = 1024
        elif dataset == 'nyuv2_500':
            dataset = 'nyuv2'
            args.json_file = os.path.join(args.benchmark_path,'nyuv2/nyuv2_video_500.json')
            args.root_path = os.path.join(args.benchmark_path,'nyuv2')
            args.max_depth_eval = 10.0
            args.min_depth_eval = 0.1
            args.max_eval_len = 500
            args.a = 45
            args.b = 471
            args.c = 41
            args.d = 601
        elif dataset == 'bonn':
            args.json_file = os.path.join(args.benchmark_path,'bonn/bonn_video.json')
            args.root_path = os.path.join(args.benchmark_path,'bonn')
            args.max_depth_eval = 10.0
            args.min_depth_eval = 0.1
            args.max_eval_len = 110
            args.a = 0
            args.b = 480
            args.c = 0
            args.d = 640
        elif dataset == 'bonn_500':
            dataset = 'bonn'
            args.json_file = os.path.join(args.benchmark_path,'bonn/bonn_video_500.json')
            args.root_path = os.path.join(args.benchmark_path,'bonn')
            args.max_depth_eval = 10.0
            args.min_depth_eval = 0.1
            args.max_eval_len = 500
            args.a = 0
            args.b = 480
            args.c = 0
            args.d = 640
        elif dataset == 'scannet':
            args.json_file = os.path.join(args.benchmark_path,'scannet/scannet_video.json')
            args.root_path = os.path.join(args.benchmark_path,'scannet')
            args.max_depth_eval = 10.0
            args.min_depth_eval = 0.1
            args.max_eval_len = 90
            args.a = 8
            args.b = -8
            args.c = 11
            args.d = -11
        elif dataset == 'scannet_500':
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
        scale_stds = shift_stds = stable_result_fulls = stable_result_wins = 0
        depth_result_fulls = np.zeros(5)
        depth_result_wins = np.zeros(5)
        depth_result_onlys = np.zeros(5)
        count = 0
        line = '-' * 50
        print(f'<{line} {dataset} start {line}>')
        file.write(f'<{line} {dataset} start {line}>\n')
        results_all = []
        for data in tqdm(json_data):
            for key in data.keys():
                value = data[key]
                infer_paths = []
                depth_gt_paths = []
                flow_paths = []
                factors = []
                for images in value:
                    infer_path = (args.infer_path + '/'+ dataset + '/' + images['image']).replace('.jpg', '.npy').replace('.png', '.npy')
                    
                    infer_paths.append(infer_path)
                    depth_gt_paths.append(args.root_path + '/' + images['gt_depth'])
                    factors.append(images['factor'])
                infer_paths = infer_paths[:args.max_eval_len]
                depth_gt_paths = depth_gt_paths[:args.max_eval_len]
                factors = factors[:args.max_eval_len]
                results_single = eval_depthcrafter(infer_paths, depth_gt_paths, factors, args)
                results_all.append(results_single)
        final_results =  np.array(results_all)
        final_results_mean = np.mean(final_results, axis=0)
        result_dict = { 'name': dataset }
        for i, metric in enumerate(eval_metrics):
            result_dict[metric] = final_results_mean[i]
            print(f"{metric}: {final_results_mean[i]:04f}")
            file.write(f"{metric}: {final_results_mean[i]:04f}\n")
        file.write(f'<{line} {dataset} finish {line}>\n')
if __name__ == '__main__':
    main()