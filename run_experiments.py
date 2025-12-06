#!/usr/bin/env python
import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from torchvision.transforms import Compose
from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from video_depth_anything.video_depth import VideoDepthAnything as VideoDepthTeacher
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthStudent


# ============================================================
# Helpers
# ============================================================
def load_scannet_sequence_from_json(
    json_file: str,
    dataset_key: str = "scannet",
    scene_idx: int = 0,
    max_frames: int = 500,
    input_size: int = 518,
):
    """
    scannet_video_500.json에서 하나의 시퀀스를 읽어서:
      x: [1, T, 3, H, W] (teacher/student 입력용)
      gt: [T, H, W]      (GT depth, factor로 스케일 복원 + 리사이즈)
    를 반환한다.
    """
    with open(json_file, "r") as f:
        meta = json.load(f)

    scenes = meta[dataset_key]   # list
    scene = scenes[scene_idx]    # e.g. {"scene0019_00": [frames...]}

    # 첫 번째 key (scene name) 사용
    scene_name = next(iter(scene.keys()))
    frames = scene[scene_name]

    # 최대 max_frames까지만 사용
    frames = frames[:max_frames]

    root_path = os.path.dirname(json_file)

    # 이미지 전처리 (infer와 동일 계열)
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

    imgs = []
    gts  = []
    H_t = W_t = None

    for item in frames:
        # RGB 로드
        img_path = os.path.join(root_path, item["image"])
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        t_dict = transform({"image": rgb})
        img_t = t_dict["image"]       # [3,H',W']
        if H_t is None:
            _, H_t, W_t = img_t.shape

        imgs.append(torch.from_numpy(img_t))

        # GT depth 로드 (원래 validator는 별도 함수 쓰지만, 여기선 간단히)
        depth_path = os.path.join(root_path, item["gt_depth"])
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        factor = float(item["factor"])
        depth_m = depth_raw / factor  # meter 단위

        # 네트워크 입력 해상도에 맞춰 리사이즈
        depth_resized = cv2.resize(depth_m, (W_t, H_t), interpolation=cv2.INTER_NEAREST)
        gts.append(depth_resized)

    imgs = torch.stack(imgs, dim=0)      # [T,3,H,W]
    gts  = np.stack(gts, axis=0)         # [T,H,W]

    x = imgs.unsqueeze(0)                # [1,T,3,H,W]
    return x, gts


# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    json_file = "/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
    dataset_key = "scannet"
    scene_idx   = 0  # scene0019_00

    ckpt = "/home/work/juhwan/monocular_depth/stream/video-stream/checkpoints/video_depth_anything_vits.pth"

    print("[1] Load ScanNet sequence from JSON...")
    x, gt = load_scannet_sequence_from_json(
        json_file=json_file,
        dataset_key=dataset_key,
        scene_idx=scene_idx,
        max_frames=500,
        input_size=518,
    )
    x = x.to(device)          # [1,T,3,H,W]
    gt = gt                   # [T,H,W] (numpy)
    print(f"  x: {x.shape}, gt: {gt.shape}")

    print("[2] Teacher...")
    teacher = VideoDepthTeacher(
        encoder="vits",
        features=64,
        out_channels=[48,96,192,384],
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

    print("[3] Load Student (stream)...")
    student = VideoDepthStudent(
        encoder="vits",
        features=64,
        out_channels=[48,96,192,384],
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


if __name__ == "__main__":
    main()
