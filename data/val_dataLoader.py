import os
import random
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import glob

def get_list(root_dir, data_name):
    x_path, y_path = [], []

    if data_name == "scannet":
        scannet_dir = "/workspace/Video-Depth-Anything/datasets/scannet/exports"
        x, y, poses, Ks = get_scannet_paths(scannet_dir)
        max_clips = 60
        x     = x[:max_clips]
        y     = y[:max_clips]
        poses = poses[:max_clips]
        Ks    = Ks[:max_clips]
        print(f"[scannet] clip 개수: {len(x)}")
        return x, y, poses, Ks
        
    elif data_name == "nyu":
        nyu_data_dir = "/workspace/Video-Depth-Anything/datasets/nyu_data/data/nyu2_train"
        x, y = get_nyu_paths(nyu_data_dir)
        x_path.extend(x)
        y_path.extend(y)

    else :
        kitti_data_dir = "/workspace/Video-Depth-Anything/datasets/depth_selection/val_selection_cropped"
        x, y = get_kitti_paths(kitti_data_dir)
        x_path.extend(x)
        y_path.extend(y)
        

    print("데이터 개수 : ", len(x_path))
    
    return x_path, y_path


def get_scannet_paths(data_dir, clip_len=16):
    """
    Returns:
      rgb_clips   : List[List[str]]  (clip 단위로 jpg 경로)
      depth_clips : List[List[str]]  (clip 단위로 png 경로)
      pose_clips  : List[List[str]]  (clip 단위로 txt 경로)
      K_clips     : List[np.ndarray] (clip 단위로 3×3 intrinsic 행렬)
    """
    import numpy as np

    def load_K(path):
        # intrinsic_color.txt 에 4×4 행렬이 저장되어 있으므로
        mat = np.loadtxt(path).reshape(4,4)
        # 3×3 내부 행렬만 뽑아서 리턴
        return mat[:3,:3]

    scene_dirs = sorted(glob.glob(os.path.join(data_dir, "scene*_*")))
    rgb_clips, depth_clips, pose_clips, K_clips = [], [], [], []

    for scene in scene_dirs:
        # 1) 파일 목록
        rgb_files   = sorted(glob.glob(os.path.join(scene,  "color",    "*.jpg")))
        depth_files = sorted(glob.glob(os.path.join(scene,  "depth",    "*.png")))
        pose_files  = sorted(glob.glob(os.path.join(scene,  "pose",     "*.txt")))
        K_path      = os.path.join(scene, "intrinsic", "intrinsic_color.txt")
        K           = load_K(K_path)

        # 2) 동기화
        count = min(len(rgb_files), len(depth_files), len(pose_files))
        rgb_files   = rgb_files[:count]
        depth_files = depth_files[:count]
        pose_files  = pose_files[:count]

        # 3) clip_len 단위로 묶기
        for i in range(0, count, clip_len):
            if i + clip_len <= count:
                rgb_clips.append(  rgb_files[i : i + clip_len]  )
                depth_clips.append(depth_files[i : i + clip_len])
                pose_clips.append( pose_files[i : i + clip_len] )
                # 같은 K를 clip 당 하나씩 추가
                K_clips.append(K.copy())

    return rgb_clips, depth_clips, pose_clips, K_clips

def get_nyu_paths(data_dir, clip_len=16):
    # 1) 모든 서브폴더 이름 추출
    subdirs = sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(data_dir, '*'))
        if os.path.isdir(d)
    )

    # 2) 골라낼 카테고리 접두사 리스트
    categories = [
        "basement",
        "bathroom",
        "bedroom",
        "bookstore",
        "cafe",
        "classroom",
        "conference_room",
        "dining_room",
        "furniture_store",
        "home_office",
        "kitchen",
        "living_room",
        "office",
        "playroom",
        "study_room",
    ]

    # 3) 카테고리별로 매칭되는 폴더가 있으면 하나씩 선택
    chosen = []
    for cat in categories:
        matches = [d for d in subdirs if d.startswith(f"{cat}_")]
        if not matches:
            print(f"⚠ No folders match prefix '{cat}_', skipping")
            continue
        chosen.append(matches[0])

    if not chosen:
        raise RuntimeError("❌ 아무 폴더도 선택되지 않았습니다. categories 리스트와 실제 폴더명이 맞는지 확인하세요.")

    # 4) 선택된 폴더에서 RGB/Depth 중 더 작은 파일 수를 min_count 로
    counts = []
    for name in chosen:
        rgb_files   = glob.glob(os.path.join(data_dir, name, '*.jpg'))
        depth_files = glob.glob(os.path.join(data_dir, name, '*.png'))
        counts.append(min(len(rgb_files), len(depth_files)))
    min_count = min(counts)
    #print("최소 파일 개수:", min_count)

    # 5) clip_len 단위로 잘라서 시퀀스 생성
    rgb_clips, depth_clips = [], []
    for name in chosen:
        rgb_files   = sorted(glob.glob(os.path.join(data_dir, name, '*.jpg')))[:min_count]
        depth_files = sorted(glob.glob(os.path.join(data_dir, name, '*.png')))[:min_count]
        for i in range(0, min_count, clip_len):
            if i + clip_len <= min_count:
                rgb_clips  .append(rgb_files  [i:i+clip_len])
                depth_clips.append(depth_files[i:i+clip_len])

    return rgb_clips, depth_clips

    
def get_kitti_paths(data_dir,clip_len=16):

    x_paths= os.path.join(data_dir,"image/*")
    y_paths = os.path.join(data_dir,"groundtruth_depth/*")

    x_paths = sorted(glob.glob(x_paths))
    y_paths = sorted(glob.glob(y_paths))

    x_clips = [x_paths[i:i+clip_len] for i in range(0, len(x_paths), clip_len)]
    y_clips = [y_paths[i:i+clip_len] for i in range(0, len(y_paths), clip_len)]

    return x_clips, y_clips


class ValDataset(Dataset):
    def __init__(
        self,
        img_paths,
        depth_paths,
        data_name,
        Ks=None,
        pose_paths=None,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
        resize_size=518
    ):
        super().__init__()
        assert len(img_paths) == len(depth_paths), "이미지/뎁스 개수 불일치"
        self.img_paths   = img_paths
        self.depth_paths = depth_paths
        self.pose_paths  = pose_paths
        self.Ks          = Ks
        self.resize_size = resize_size
        self.rgb_mean    = rgb_mean
        self.rgb_std     = rgb_std
        self.data_name   = data_name

        if data_name=="scannet":
            self.factor = 1000.0
        elif data_name=="nyu":
            self.factor = 6000.0
        else :
            self.factor = 256.0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 이미지먼저
        imgs = []
        for p in self.img_paths[idx]:        
            im = Image.open(p).convert("RGB")
            im = TF.center_crop(im, self.resize_size)
            im = TF.to_tensor(im)
            im = TF.normalize(im, mean=self.rgb_mean, std=self.rgb_std)
            imgs.append(im)                        
        imgs = torch.stack(imgs, dim=0) 

        # depth
        disps = []
        for p in self.depth_paths[idx]:
            d = Image.open(p).convert("F")
            d = TF.center_crop(d, self.resize_size)
            d = torch.from_numpy(np.array(d, np.float32)).unsqueeze(0)
            d = d / self.factor
            disps.append(d)
        disps = torch.stack(disps, dim=0)  

        if self.data_name=="scannet":
            # Pose clip
            pose_mats = []
            for p in self.pose_paths[idx]:
                mat = np.loadtxt(p).astype(np.float32).reshape(4,4)
                pose_mats.append(torch.from_numpy(mat))
            poses = torch.stack(pose_mats, dim=0)  # (clip,4,4)
    
            # Intrinsic (K) 
            # clip 길이만큼 반복해도 되고, metric_val 호출 시 브로드캐스트 가능
            K_single = torch.from_numpy(self.Ks[idx].astype(np.float32))    # (3,3)
            # clip 길이 T = len(self.img_paths[idx])
            T = len(self.img_paths[idx])
            # frame 당 하나씩 담아서 (T,3,3) 텐서로
            Ks = K_single.unsqueeze(0).repeat(T, 1, 1)  # (T,3,3)
            return imgs, disps, poses, Ks
        else :
            return imgs,disps