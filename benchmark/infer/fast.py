#!/usr/bin/env python
"""
Video Depth Anything - High-Reliability Inference Speed Benchmark
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from collections import OrderedDict
import gc
from datetime import datetime
import json
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 설정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from video_depth_anything.video_depth_stream import VideoDepthAnything

def reset_streaming_state(model):
    """스트리밍 상태를 초기화합니다."""
    model.transform = None
    model.frame_cache_list = []
    model.frame_id_list = []
    model.id = -1

def create_dummy_image(height, width):
    """더미 이미지 생성 (RGB)"""
    np.random.seed(42)  # 재현성을 위한 고정 시드
    dummy = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return dummy

def detect_model_config(checkpoint_path):
    """Checkpoint에서 모델 설정을 자동으로 감지합니다."""
    print(f"🔍 Detecting model configuration...")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    encoder_type = None
    features = None
    out_channels = None
    
    for key, value in state_dict.items():
        if 'pretrained' in key or 'encoder' in key:
            if 'blocks.23' in key:
                encoder_type = 'vitl'
            elif 'blocks.11' in key and 'blocks.12' not in key:
                encoder_type = 'vits'
        
        if 'head.projects.0' in key and 'weight' in key:
            in_channels = value.shape[1]
            if in_channels == 384:
                encoder_type = 'vits'
                features = 64
                out_channels = [48, 96, 192, 384]
            elif in_channels == 1024:
                encoder_type = 'vitl'
                features = 256
                out_channels = [256, 512, 1024, 1024]
    
    if encoder_type is None:
        encoder_type = 'vitl'
        features = 256
        out_channels = [256, 512, 1024, 1024]
    
    return {
        'encoder': encoder_type,
        'features': features,
        'out_channels': out_channels
    }

def load_model_from_checkpoint(checkpoint_path, encoder_override=None, device='cuda'):
    """Checkpoint에서 모델을 로드합니다."""
    
    if encoder_override:
        print(f"📦 Using specified encoder: {encoder_override}")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        config = model_configs[encoder_override]
    else:
        config = detect_model_config(checkpoint_path)
    
    print(f"🏗️  Creating {config['encoder'].upper()} model...")
    model = VideoDepthAnything(**config)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    clean_state = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for prefix in ['module.', 'student.', 'model.']:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        clean_state[nk] = v
    
    model.load_state_dict(clean_state, strict=True)
    model = model.to(device).eval()
    
    # Disable gradient computation for inference
    torch.set_grad_enabled(False)
    
    return model, config

def remove_outliers(data, z_threshold=3):
    """Z-score 기반 이상치 제거"""
    z_scores = np.abs(stats.zscore(data))
    return data[z_scores < z_threshold]

def calculate_statistics(times_ms, confidence_level=0.95):
    """상세한 통계 계산"""
    # 이상치 제거
    cleaned_times = remove_outliers(times_ms)
    
    # 기본 통계
    mean = np.mean(cleaned_times)
    std = np.std(cleaned_times)
    median = np.median(cleaned_times)
    
    # 백분위수
    p25 = np.percentile(cleaned_times, 25)
    p75 = np.percentile(cleaned_times, 75)
    p95 = np.percentile(cleaned_times, 95)
    p99 = np.percentile(cleaned_times, 99)
    
    # 신뢰구간 계산
    n = len(cleaned_times)
    se = std / np.sqrt(n)  # 표준 오차
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se
    
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'min': np.min(cleaned_times),
        'max': np.max(cleaned_times),
        'p25': p25,
        'p75': p75,
        'p95': p95,
        'p99': p99,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'outliers_removed': len(times_ms) - len(cleaned_times),
        'sample_size': len(cleaned_times)
    }

def benchmark_single_frame_reliable(model, img, input_size, device, 
                                   num_warmup=10, num_iterations=100, 
                                   verbose=True):
    """고신뢰도 단일 프레임 벤치마크"""
    
    if verbose:
        print(f"    🔥 Warming up ({num_warmup} iterations)...")
    
    # Warmup
    for _ in range(num_warmup):
        _ = model.infer_video_depth_one(img, input_size=input_size, device=device, fp32=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # 측정
    times = []
    
    if verbose:
        print(f"    📊 Measuring ({num_iterations} iterations)...")
        from tqdm import tqdm
        iterator = tqdm(range(num_iterations), desc="    Progress", leave=False)
    else:
        iterator = range(num_iterations)
    
    for _ in iterator:
        # GPU 동기화
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        depth_np = model.infer_video_depth_one(
            img, input_size=input_size, device=device, fp32=True
        )
        
        # GPU 동기화
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        times.append(elapsed * 1000)  # ms
    
    # 통계 계산
    stats_result = calculate_statistics(np.array(times))
    
    # FPS 계산 (평균 기준)
    fps_mean = 1000.0 / stats_result['mean']
    fps_median = 1000.0 / stats_result['median']
    
    # 메모리 측정
    if torch.cuda.is_available():
        memory = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'reserved': torch.cuda.memory_reserved() / 1024**2,
        }
    else:
        memory = {'allocated': 0, 'reserved': 0}
    
    return {
        **stats_result,
        'fps_mean': fps_mean,
        'fps_median': fps_median,
        'output_shape': depth_np.shape,
        'memory_mb': memory
    }

def benchmark_streaming_reliable(model, num_frames, img_generator_func, 
                                input_size, device, num_runs=10, verbose=True):
    """고신뢰도 스트리밍 벤치마크 (여러 번 실행)"""
    
    all_first_frame_times = []
    all_cached_times = []
    all_total_times = []
    
    if verbose:
        print(f"    🎬 Running {num_runs} streaming sequences...")
        from tqdm import tqdm
        iterator = tqdm(range(num_runs), desc="    Sequences", leave=False)
    else:
        iterator = range(num_runs)
    
    for run in iterator:
        # 각 실행마다 스트리밍 상태 초기화
        reset_streaming_state(model)
        
        # 새로운 이미지 시퀀스 생성
        images = [img_generator_func() for _ in range(num_frames)]
        
        times = []
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        for i, img in enumerate(images):
            start_time = time.perf_counter()
            
            depth_np = model.infer_video_depth_one(
                img, input_size=input_size, device=device, fp32=True
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start_time
            times.append(elapsed * 1000)
        
        # 통계 수집
        all_first_frame_times.append(times[0])
        all_cached_times.extend(times[1:])  # 첫 프레임 제외
        all_total_times.extend(times)
    
    # 종합 통계
    first_frame_stats = calculate_statistics(np.array(all_first_frame_times))
    cached_stats = calculate_statistics(np.array(all_cached_times))
    total_stats = calculate_statistics(np.array(all_total_times))
    
    return {
        'num_runs': num_runs,
        'frames_per_run': num_frames,
        'first_frame': {
            **first_frame_stats,
            'fps': 1000.0 / first_frame_stats['mean']
        },
        'cached_frames': {
            **cached_stats,
            'fps': 1000.0 / cached_stats['mean']
        },
        'overall': {
            **total_stats,
            'fps': 1000.0 / total_stats['mean']
        }
    }

def print_detailed_stats(stats, prefix=""):
    """통계 결과를 보기 좋게 출력"""
    print(f"{prefix}Mean: {stats['mean']:.2f} ms (FPS: {1000/stats['mean']:.1f})")
    print(f"{prefix}Std Dev: {stats['std']:.2f} ms")
    print(f"{prefix}Median: {stats['median']:.2f} ms (FPS: {1000/stats['median']:.1f})")
    print(f"{prefix}95% CI: [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}] ms")
    print(f"{prefix}Percentiles: P25={stats['p25']:.2f}, P75={stats['p75']:.2f}, P95={stats['p95']:.2f}, P99={stats['p99']:.2f}")
    if 'outliers_removed' in stats:
        print(f"{prefix}Outliers removed: {stats['outliers_removed']}")

def main():
    parser = argparse.ArgumentParser(description='High-Reliability Video Depth Benchmark')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--encoder', type=str, default=None, choices=['vits', 'vitl', None])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_size', type=int, default=518)
    
    # 벤치마크 설정
    parser.add_argument('--warmup', type=int, default=20, 
                        help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of measurement iterations for single frame')
    parser.add_argument('--stream_runs', type=int, default=10,
                        help='Number of streaming sequence runs')
    parser.add_argument('--stream_frames', type=int, default=32,
                        help='Frames per streaming sequence')
    
    # 테스트 모드
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test (50 iterations, 5 runs)')
    parser.add_argument('--extensive', action='store_true',
                        help='Extensive test (500 iterations, 20 runs)')
    parser.add_argument('--ultra', action='store_true',
                        help='Ultra reliable test (1000 iterations, 50 runs)')
    
    parser.add_argument('--resolutions', type=str, nargs='+', 
                        default=['518x518', '2K'],
                        choices=['480p', '518x518', '720p', '1080p', '2K', '4K'])
    
    args = parser.parse_args()
    
    # 테스트 모드별 설정
    if args.ultra:
        args.iterations = 1000
        args.stream_runs = 50
        print("🔬 ULTRA reliability mode: 1000 iterations, 50 streaming runs")
    elif args.extensive:
        args.iterations = 500
        args.stream_runs = 20
        print("🔬 EXTENSIVE mode: 500 iterations, 20 streaming runs")
    elif args.quick:
        args.iterations = 50
        args.stream_runs = 5
        print("⚡ QUICK mode: 50 iterations, 5 streaming runs")
    else:
        print(f"📊 STANDARD mode: {args.iterations} iterations, {args.stream_runs} streaming runs")
    
    print("="*70)
    print("HIGH-RELIABILITY VIDEO DEPTH BENCHMARK")
    print("="*70)
    
    # Device 정보
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # GPU 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print("⚠️  Running on CPU")
    
    # 모델 로드
    print(f"\n📦 Loading checkpoint: {args.checkpoint}")
    model, config = load_model_from_checkpoint(
        args.checkpoint, 
        encoder_override=args.encoder,
        device=DEVICE
    )
    print(f"✅ Model loaded: {config['encoder'].upper()}")
    
    # 해상도 매핑
    resolution_map = {
        '518x518': (518, 518),
        '2K': (1440, 2560),
    }
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': DEVICE,
            'gpu_name': torch.cuda.get_device_name() if DEVICE == 'cuda' else 'CPU',
            'model_config': config,
            'benchmark_config': {
                'warmup': args.warmup,
                'iterations': args.iterations,
                'stream_runs': args.stream_runs,
                'stream_frames': args.stream_frames,
            }
        },
        'benchmarks': {}
    }
    
    print("\n" + "="*70)
    print("BENCHMARKING")
    print("="*70)
    
    for res_name in args.resolutions:
        height, width = resolution_map[res_name]
        
        print(f"\n📐 Resolution: {res_name} ({width}x{height})")
        print("-"*50)
        
        # 더미 이미지 생성 함수
        def create_img():
            return create_dummy_image(height, width)
        
        # 1. Single Frame Benchmark
        print("\n  🎯 Single Frame Benchmark")
        reset_streaming_state(model)
        
        try:
            dummy_img = create_img()
            single_result = benchmark_single_frame_reliable(
                model, dummy_img, args.input_size, DEVICE,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                verbose=True
            )
            
            print("\n  📈 Single Frame Results:")
            print_detailed_stats(single_result, "      ")
            print(f"      Memory: {single_result['memory_mb']['allocated']:.1f} MB")
            
        except Exception as e:
            print(f"  ❌ Single frame failed: {e}")
            single_result = None
        
        # 2. Streaming Benchmark
        print(f"\n  🎬 Streaming Benchmark ({args.stream_frames} frames × {args.stream_runs} runs)")
        
        try:
            stream_result = benchmark_streaming_reliable(
                model, args.stream_frames, create_img,
                args.input_size, DEVICE, 
                num_runs=args.stream_runs,
                verbose=True
            )
            
            print("\n  📈 Streaming Results:")
            print("    First Frame:")
            print_detailed_stats(stream_result['first_frame'], "      ")
            print("\n    Cached Frames:")
            print_detailed_stats(stream_result['cached_frames'], "      ")
            print("\n    Overall:")
            print_detailed_stats(stream_result['overall'], "      ")
            
        except Exception as e:
            print(f"  ❌ Streaming failed: {e}")
            stream_result = None
        
        # 결과 저장
        results['benchmarks'][res_name] = {
            'resolution': {'width': width, 'height': height},
            'single_frame': single_result,
            'streaming': stream_result
        }
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 최종 요약
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nModel: {config['encoder'].upper()}")
    print(f"Iterations: {args.iterations} (single) / {args.stream_runs}×{args.stream_frames} (streaming)")
    print(f"\n{'Resolution':<12} {'Single FPS':<15} {'First Frame':<15} {'Cached FPS':<15}")
    print("-"*57)
    
    for res_name in results['benchmarks']:
        bench = results['benchmarks'][res_name]
        if bench['single_frame'] and bench['streaming']:
            single_fps = bench['single_frame']['fps_mean']
            first_fps = bench['streaming']['first_frame']['fps']
            cached_fps = bench['streaming']['cached_frames']['fps']
            
            # 95% 신뢰구간도 함께 표시
            single_ci = f"±{(bench['single_frame']['ci_upper'] - bench['single_frame']['ci_lower'])/2:.1f}"
            
            print(f"{res_name:<12} {single_fps:>6.1f} {single_ci:<8} "
                  f"{first_fps:>6.1f}         {cached_fps:>6.1f}")
        else:
            print(f"{res_name:<12} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")
    
    # JSON 저장
    output_file = f"benchmark_{config['encoder']}_{args.iterations}iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # numpy types를 JSON serializable하게 변환
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Benchmark completed!")

if __name__ == '__main__':
    main()