#!/usr/bin/env python
"""
Teacher(clip 단위)와 Student(streaming) 모델의 성능 비교 스크립트.
ScanNet 500 데이터셋의 0번째 scene만 사용하여 빠르게 테스트.
"""
import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import OrderedDict

from torchvision.transforms import Compose
from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent

# ==============================================================================
# Metric Functions (from benchmark/eval/metric.py)
# ==============================================================================
def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask.cpu()] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


# ==============================================================================
# Data Loading
# ==============================================================================
def load_scannet_sequence_from_json(
    json_file: str,
    dataset_key: str = "scannet",
    scene_idx: int = 0,
    max_frames: int = 500,
    input_size: int = 518,
):
    """
    scannet_video_500.json에서 하나의 시퀀스를 읽어서:
      frames_rgb: list of numpy [H,W,3] RGB images (for streaming inference)
      frames_tensor: [1, T, 3, H', W'] (for teacher batch inference)
      gt: [T, H', W']      (GT depth, factor로 스케일 복원 + 리사이즈)
      frame_info: list of dict (원본 프레임 정보)
    를 반환한다.
    """
    with open(json_file, "r") as f:
        meta = json.load(f)

    scenes = meta[dataset_key]
    scene = scenes[scene_idx]
    scene_name = next(iter(scene.keys()))
    frames = scene[scene_name]
    frames = frames[:max_frames]

    root_path = os.path.dirname(json_file)

    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        PrepareForNet(),
    ])

    imgs_tensor = []
    imgs_rgb = []
    gts = []
    H_t = W_t = None

    for item in tqdm(frames, desc="Loading frames"):
        img_path = os.path.join(root_path, item["image"])
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs_rgb.append(rgb)

        rgb_float = rgb.astype(np.float32) / 255.0
        t_dict = transform({"image": rgb_float})
        img_t = t_dict["image"]
        if H_t is None:
            _, H_t, W_t = img_t.shape

        imgs_tensor.append(torch.from_numpy(img_t))

        depth_path = os.path.join(root_path, item["gt_depth"])
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        factor = float(item["factor"])
        depth_m = depth_raw / factor

        depth_resized = cv2.resize(depth_m, (W_t, H_t), interpolation=cv2.INTER_NEAREST)
        gts.append(depth_resized)

    imgs_tensor = torch.stack(imgs_tensor, dim=0)
    gts = np.stack(gts, axis=0)

    x = imgs_tensor.unsqueeze(0)
    return imgs_rgb, x, gts, scene_name


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


# ==============================================================================
# Evaluation
# ==============================================================================
def evaluate_depths(pred_depths, gt_depths, max_depth=10.0, crop=(8, -8, 11, -11)):
    """
    pred_depths: [T, H, W] numpy
    gt_depths: [T, H, W] numpy
    """
    a, b, c, d = crop
    
    # crop GT
    gts = gt_depths[:, a:b, c:d] if b != 0 else gt_depths[:, a:, c:d] if d != 0 else gt_depths[:, a:, c:]
    if b == -8 and d == -11:
        gts = gt_depths[:, a:-8, c:-11]
    
    # resize predictions to match GT
    T, H_gt, W_gt = gts.shape
    infs = []
    for i in range(T):
        pred = pred_depths[i]
        if pred.shape[0] != H_gt or pred.shape[1] != W_gt:
            pred = cv2.resize(pred, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
        infs.append(pred)
    infs = np.stack(infs, axis=0)

    # valid mask
    valid_mask = np.logical_and((gts > 1e-3), (gts < max_depth))

    # least squares alignment (disparity space)
    gt_disp_masked = 1. / (gts[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
    infs = np.clip(infs, a_min=1e-3, a_max=None)
    pred_disp_masked = infs[valid_mask].reshape((-1, 1)).astype(np.float64)

    _ones = np.ones_like(pred_disp_masked)
    A = np.concatenate([pred_disp_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]
    scale, shift = X
    
    aligned_pred = scale * infs + shift
    aligned_pred = np.clip(aligned_pred, a_min=1e-3, a_max=None)

    pred_depth = depth2disparity(aligned_pred)
    pred_depth = np.clip(pred_depth, a_min=1e-3, a_max=max_depth)

    # convert to torch for metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_depth_ts = torch.from_numpy(pred_depth).to(device)
    gt_depth_ts = torch.from_numpy(gts).to(device)
    valid_mask_ts = torch.from_numpy(valid_mask).to(device)

    # filter valid frames
    n = valid_mask.sum((-1, -2))
    valid_frame = (n > 0)
    pred_depth_ts = pred_depth_ts[valid_frame]
    gt_depth_ts = gt_depth_ts[valid_frame]
    valid_mask_ts = valid_mask_ts[valid_frame]

    # compute metrics
    abs_rel = abs_relative_difference(pred_depth_ts, gt_depth_ts, valid_mask_ts).item()
    rmse = rmse_linear(pred_depth_ts, gt_depth_ts, valid_mask_ts).item()
    delta1 = delta1_acc(pred_depth_ts, gt_depth_ts, valid_mask_ts).item()

    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'delta1': delta1,
    }


def reset_streaming_state(model):
    """Reset streaming state for VideoDepthAnything student model."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "transform"):
        m.transform = None
    if hasattr(m, "frame_cache_list"):
        m.frame_cache_list = []
    if hasattr(m, "frame_id_list"):
        m.frame_id_list = []
    if hasattr(m, "id"):
        m.id = -1


# ==============================================================================
# Main
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    json_file = "/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
    ckpt = "/home/work/juhwan/monocular_depth/stream/video-stream/checkpoints/video_depth_anything_vits.pth"
    
    dataset_key = "scannet"
    scene_idx = 0
    max_frames = 500
    input_size = 518
    max_depth_eval = 10.0
    crop = (8, -8, 11, -11)

    # ==================================================
    # 1. Load data
    # ==================================================
    print("\n[1] Loading ScanNet sequence...")
    imgs_rgb, x_tensor, gt, scene_name = load_scannet_sequence_from_json(
        json_file=json_file,
        dataset_key=dataset_key,
        scene_idx=scene_idx,
        max_frames=max_frames,
        input_size=input_size,
    )
    print(f"  Scene: {scene_name}")
    print(f"  Frames: {len(imgs_rgb)}")
    print(f"  x_tensor: {x_tensor.shape}")
    print(f"  GT shape: {gt.shape}")

    # ==================================================
    # 2. Load Teacher model
    # ==================================================
    print("\n[2] Loading Teacher model...")
    teacher = VideoDepthTeacher(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe="ape",
    )
    t_ckpt = torch.load(ckpt, map_location="cpu", weights_only=True)
    t_state = t_ckpt["model"] if "model" in t_ckpt else t_ckpt
    teacher.load_state_dict(t_state, strict=True)
    teacher.to(device)
    teacher.eval()
    print("  Teacher loaded successfully.")

    # ==================================================
    # 3. Load Student model (streaming)
    # ==================================================
    print("\n[3] Loading Student model (streaming)...")
    student = VideoDepthStudent(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe="ape",
    )
    s_ckpt = torch.load(ckpt, map_location="cpu", weights_only=True)
    s_state = s_ckpt["model"] if "model" in s_ckpt else s_ckpt
    student.load_state_dict(s_state, strict=True)
    student.to(device)
    student.eval()
    print("  Student loaded successfully.")

    # ==================================================
    # 4. Teacher Inference (using infer_video_depth method)
    # ==================================================
    print("\n[4] Teacher Inference...")
    # Teacher uses batch processing with sliding window
    # infer_video_depth expects numpy array of BGR images
    frames_bgr = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs_rgb], axis=0)
    
    with torch.no_grad():
        teacher_depths, _ = teacher.infer_video_depth(
            frames_bgr,
            target_fps=1,
            input_size=input_size,
            device=str(device),
            fp32=True,
        )
    print(f"  Teacher output shape: {teacher_depths.shape}")

    # ==================================================
    # 5. Student Inference (streaming, frame-by-frame)
    # ==================================================
    print("\n[5] Student Inference (streaming)...")
    reset_streaming_state(student)
    
    student_depths = []
    with torch.inference_mode():
        for i, rgb_frame in enumerate(tqdm(imgs_rgb, desc="Student streaming")):
            depth_np = student.infer_video_depth_one(
                rgb_frame,
                input_size=input_size,
                device=str(device),
                fp32=True,
            )
            student_depths.append(depth_np)
    
    student_depths = np.stack(student_depths, axis=0)
    print(f"  Student output shape: {student_depths.shape}")

    # ==================================================
    # 6. Evaluation
    # ==================================================
    print("\n[6] Evaluation...")
    
    # Resize GT to match prediction size for fair comparison
    # GT is already at network resolution (H_t, W_t), predictions are at original resolution
    # We need to resize predictions to match GT size after cropping
    
    print("\n--- Teacher Results ---")
    teacher_results = evaluate_depths(teacher_depths, gt, max_depth=max_depth_eval, crop=crop)
    print(f"  abs_rel:  {teacher_results['abs_rel']:.4f}")
    print(f"  rmse:     {teacher_results['rmse']:.4f}")
    print(f"  delta1:   {teacher_results['delta1']:.4f}")

    print("\n--- Student Results ---")
    student_results = evaluate_depths(student_depths, gt, max_depth=max_depth_eval, crop=crop)
    print(f"  abs_rel:  {student_results['abs_rel']:.4f}")
    print(f"  rmse:     {student_results['rmse']:.4f}")
    print(f"  delta1:   {student_results['delta1']:.4f}")

    print("\n--- Comparison (Student - Teacher) ---")
    print(f"  abs_rel diff:  {student_results['abs_rel'] - teacher_results['abs_rel']:+.4f}")
    print(f"  rmse diff:     {student_results['rmse'] - teacher_results['rmse']:+.4f}")
    print(f"  delta1 diff:   {student_results['delta1'] - teacher_results['delta1']:+.4f}")

    # ==================================================
    # 7. Save results
    # ==================================================
    output_dir = "/home/work/juhwan/monocular_depth/stream/video-stream/benchmark/output/eval_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'scene': scene_name,
        'num_frames': len(imgs_rgb),
        'teacher': teacher_results,
        'student': student_results,
    }
    
    results_file = os.path.join(output_dir, f"{scene_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[7] Results saved to: {results_file}")


if __name__ == "__main__":
    main()
