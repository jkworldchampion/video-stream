#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import json
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
from queue import Queue
import threading

# 프로젝트 루트(../..)를 PYTHONPATH에 추가
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

from video_depth_anything.video_depth_stream import VideoDepthAnything


def reset_stream_state(model):
    """시퀀스 시작 시 스트리밍 상태 초기화"""
    model.transform = None
    model.frame_cache_list = []
    model.frame_id_list = []
    model.id = -1


def read_ckpt_state_dict(ckpt_path):
    """체크포인트 로드 + state_dict 반환"""
    ckpt_data = torch.load(ckpt_path, map_location='cpu')
    raw_sd = ckpt_data.get('model_state_dict', ckpt_data.get('state_dict', ckpt_data))
    fixed = {}
    for k, v in raw_sd.items():
        name = k[7:] if k.startswith('module.') else k
        fixed[name] = v
    return fixed


def infer_variant_from_state_dict(sd):
    """체크포인트에서 encoder 변형 추정"""
    embed_dim = None
    if 'pretrained.pos_embed' in sd:
        embed_dim = sd['pretrained.pos_embed'].shape[-1]
    elif 'pretrained.cls_token' in sd:
        embed_dim = sd['pretrained.cls_token'].shape[-1]
    elif 'pretrained.patch_embed.proj.weight' in sd:
        embed_dim = sd['pretrained.patch_embed.proj.weight'].shape[0]

    if embed_dim == 384:
        return 'vits', 384
    if embed_dim == 1024:
        return 'vitl', 1024

    for k in ['head.projects.0.weight', 'head.projects.1.weight']:
        if k in sd and sd[k].ndim >= 2:
            e = sd[k].shape[1]
            if e == 384:
                return 'vits', 384
            if e == 1024:
                return 'vitl', 1024

    raise RuntimeError("체크포인트에서 encoder 변형을 추정할 수 없습니다.")


def filter_unmatched_keys_for_load(sd, model):
    """모델에 존재하지 않는 키 제거"""
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in model_keys}
    dropped = [k for k in sd.keys() if k not in model_keys]
    return filtered, dropped


def process_scenes_on_gpu(gpu_id, scene_queue, args, model_config, sd_to_load, root_path, results_queue):
    """특정 GPU에서 scene들을 순차 처리"""
    device = f'cuda:{gpu_id}'
    
    # GPU별 모델 생성
    print(f"[GPU {gpu_id}] 모델 초기화 중...")
    vda = VideoDepthAnything(**model_config, use_causal_mask=True)
    
    # 모델 로딩
    sd_filtered, dropped = filter_unmatched_keys_for_load(sd_to_load, vda)
    missing, unexpected = vda.load_state_dict(sd_filtered, strict=False)
    vda = vda.to(device).eval()
    
    print(f"[GPU {gpu_id}] 모델 로딩 완료")
    
    # Scene 처리 루프
    processed_count = 0
    while True:
        try:
            # Queue에서 scene 가져오기 (timeout으로 종료 조건 체크)
            scene_data = scene_queue.get(timeout=1.0)
            if scene_data is None:  # 종료 신호
                break
                
            scene_idx, scene_info = scene_data
            
            print(f"[GPU {gpu_id}] Scene {scene_idx} 처리 시작")
            
            # Scene별로 독립적으로 처리
            for key, frames in scene_info.items():
                # 각 scene마다 스트리밍 상태 리셋
                reset_stream_state(vda)
                
                with torch.no_grad(), torch.inference_mode():
                    for frame_idx, item in enumerate(frames):
                        # 입력/출력 경로
                        img_path = os.path.join(root_path, item['image'])
                        base, _ = os.path.splitext(item['image'])
                        out_path = os.path.join(args.infer_path, args.datasets[0], base + '.npy')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)

                        # BGR -> RGB
                        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if img_bgr is None:
                            print(f"[GPU {gpu_id}] 이미지 로드 실패: {img_path}")
                            continue
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                        # 스트리밍 1프레임 추론
                        depth = vda.infer_video_depth_one(
                            img_rgb, input_size=args.input_size, device=device, fp32=True
                        )
                        np.save(out_path, depth)
            
            processed_count += 1
            results_queue.put((gpu_id, scene_idx, "completed"))
            print(f"[GPU {gpu_id}] Scene {scene_idx} 완료 (총 {processed_count}개 처리)")
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Scene 처리 중 오류: {e}")
            continue
    
    print(f"[GPU {gpu_id}] 총 {processed_count}개 scene 처리 완료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='', help='출력 .npy 저장 루트')
    parser.add_argument('--json_file', type=str, default='', help='비디오 프레임 리스트 JSON')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--num_gpus', type=int, default=None, help='사용할 GPU 수 (기본: 모든 GPU)')
    args = parser.parse_args()

    # GPU 개수 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    
    available_gpus = torch.cuda.device_count()
    if args.num_gpus is None:
        args.num_gpus = available_gpus
    else:
        args.num_gpus = min(args.num_gpus, available_gpus)
    
    print(f"사용 가능한 GPU: {available_gpus}개, 사용할 GPU: {args.num_gpus}개")

    # 체크포인트 로딩
    ckpt_path = os.path.join(BASE_DIR, 'outputs', 'experiment_23', 'best_model.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")

    sd_fixed = read_ckpt_state_dict(ckpt_path)
    ckpt_encoder, embed_dim = infer_variant_from_state_dict(sd_fixed)

    # Teacher-Student 체크포인트 처리
    is_teacher_student = any(k.startswith(('student.', 'teacher.', 'proj_layers.')) for k in sd_fixed.keys())
    if is_teacher_student:
        print("[정보] Teacher-Student 체크포인트 감지 → Student 모델만 추출")
        sd_to_load = {k[8:]: v for k, v in sd_fixed.items() if k.startswith('student.')}
    else:
        print("[정보] 단일 Student 모델 체크포인트")
        sd_to_load = sd_fixed

    # 모델 설정
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    if args.encoder != ckpt_encoder:
        print(f"[경고] --encoder={args.encoder}를 {ckpt_encoder}로 교정")
        args.encoder = ckpt_encoder

    # JSON 데이터 로딩
    if not args.json_file:
        raise ValueError('--json_file 경로를 지정하세요.')
    with open(args.json_file, 'r') as fs:
        path_json = json.load(fs)
    root_path = os.path.dirname(args.json_file)

    # Scene 큐 생성 및 분배
    scene_queue = mp.Queue()
    results_queue = mp.Queue()
    
    for dataset in args.datasets:
        if dataset not in path_json:
            raise KeyError(f"JSON에 '{dataset}' 키가 없습니다.")
        
        json_data = path_json[dataset]
        total_scenes = len(json_data)
        print(f"총 {total_scenes}개 scene을 {args.num_gpus}개 GPU에 분배")
        
        # 모든 scene을 큐에 추가
        for i, scene_data in enumerate(json_data):
            scene_queue.put((i, scene_data))
    
    # 종료 신호 추가 (GPU 개수만큼)
    for _ in range(args.num_gpus):
        scene_queue.put(None)

    # 멀티프로세싱 시작
    processes = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=process_scenes_on_gpu,
            args=(gpu_id, scene_queue, args, model_configs[args.encoder], sd_to_load, root_path, results_queue)
        )
        p.start()
        processes.append(p)

    # 진행 상황 모니터링
    completed_scenes = 0
    total_scenes = len(path_json[args.datasets[0]])
    
    try:
        while completed_scenes < total_scenes:
            try:
                gpu_id, scene_idx, status = results_queue.get(timeout=10.0)
                if status == "completed":
                    completed_scenes += 1
                    print(f"진행률: {completed_scenes}/{total_scenes} ({completed_scenes/total_scenes*100:.1f}%)")
            except:
                # 타임아웃 - 프로세스 상태 확인
                alive_processes = [p for p in processes if p.is_alive()]
                if not alive_processes:
                    print("모든 프로세스가 종료되었습니다.")
                    break
    except KeyboardInterrupt:
        print("사용자에 의해 중단되었습니다.")

    # 모든 프로세스 대기
    for p in processes:
        p.join()

    print("✅ 멀티 GPU 스트리밍 inference 완료")


if __name__ == '__main__':
    # 멀티프로세싱을 위한 설정
    mp.set_start_method('spawn', force=True)
    main()
