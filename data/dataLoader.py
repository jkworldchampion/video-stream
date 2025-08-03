import os
import random
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from glob import glob
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import json
import tartanair as ta


def get_random_crop_params_with_rng(img, output_size, rng):
    w, h = img.size
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, th, tw
    i = rng.randint(0, h - th)
    j = rng.randint(0, w - tw)
    return i, j, th, tw

    
def get_data_list(root_dir, data_name, split, clip_len=16, condition_num=10, difficulties=None):
    """
    data path랑 clip_len, 원하는 데이터 개수 or idx 개수만큼 리스트로 쌓아주는 함수
    """
    if data_name == "kitti":
        if split == "train":
            video_info_train = get_kitti_video_path(root_dir, condition_num=condition_num, split="train", binocular=False)
            x_path, y_path = get_kitti_individuals(video_info_train, clip_len, split) 
            return x_path, y_path
        else :
            video_info_val = get_kitti_video_path(root_dir, condition_num=condition_num, split="val", binocular=False)
            x_path, y_path, cam_ids, intrin_clips, extrin_clips = get_kitti_individuals(video_info_val, clip_len, split)
            return x_path, y_path, cam_ids, intrin_clips, extrin_clips

    elif data_name == "google":
        x_path, y_path = get_google_paths(root_dir)
        return x_path, y_path

    elif data_name == "GTA":
        if split == "train":
            x_path, y_path, _= get_GTA_paths(root_dir,split="train")
            return x_path, y_path
        else:
            x_path, y_path, poses_path = get_GTA_paths(root_dir,split="val")
            return x_path, y_path, poses_path 

            
    elif data_name == "tartanair":
        envs = split if isinstance(split, list) else [split]
        diff_list = difficulties if difficulties is not None else ['easy','hard']
        cams = ['lcam_front','lcam_right','lcam_back',
                'lcam_left','lcam_top','lcam_bottom']
        
        img_lists, dep_lists, pose_lists = get_tartanair_paths(
            root_dir, envs, diff_list, cams
        )
        # hard는 학습이므로 pose는 사용하지 않고, easy는 검증이므로 pose 포함 반환
        if set(diff_list) == {'hard'}:
            return img_lists, dep_lists
        else:
            return img_lists, dep_lists, pose_lists


        

# def init_tartanair(data_root: str):
#     """
#     TartanAir 데이터셋 루트를 초기화합니다.
#     """
#     ta.init(data_root)

def get_tartanair_paths(root_dir, envs, difficulties, cams):
    """
    Returns three lists of lists: scene_img_lists, scene_dep_lists, scene_pose_lists.
    """
    scene_img_lists = []
    scene_dep_lists = []
    scene_pose_lists = []
    
    for env in envs:
        for diff in difficulties:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # normalize path separators
                parts = dirpath.replace('\\','/').split('/')
                # find env folder in path
                if env not in parts:
                    continue
                idx = parts.index(env)
                # must have Data_<diff> and Pxxx and image/depth folder after env
                if len(parts) <= idx + 3:
                    continue
                data_dir = parts[idx+1]
                chunk    = parts[idx+2]
                folder   = parts[idx+3]
                if data_dir != f"Data_{diff}" or not chunk.startswith('P'):
                    continue
                # build full chunk directory
                chunk_dir = os.path.join(*parts[:idx+3+1])  # reconstruct path to Pxxx
                # but safer to use dirpath
                chunk_dir = dirpath.rsplit(f"/{folder}",1)[0]
                # handle each camera
                for cam in cams:
                    if folder == f"image_{cam}":
                        img_dir = dirpath
                        dep_dir = img_dir.replace(f"image_{cam}", f"depth_{cam}")
                        pf_file = os.path.join(chunk_dir, f"pose_{cam}.txt")
                        if os.path.isdir(img_dir) and os.path.isdir(dep_dir) and os.path.isfile(pf_file):
                            imgs = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir)
                                           if f.lower().endswith(('.png','.jpg','.jpeg'))])
                            deps = sorted([os.path.join(dep_dir,f) for f in os.listdir(dep_dir)
                                           if f.lower().endswith('.png')])
                            if len(imgs) >= 1 and len(deps) >= 1:
                                scene_img_lists.append(imgs)
                                scene_dep_lists.append(deps)
                                scene_pose_lists.append(pf_file)

    if difficulties == ["hard"]:
        max_clips_per_scene = 1500
    else :
        max_clips_per_scene = 500

    # 일단은 아래처럼 허는데, 아래처럼 하면 후에 데이터 추가시, env 늘어나면 수정해야할 수 있음. 인지하기

    scene_img_lists  = [imgs[:max_clips_per_scene]  for imgs in scene_img_lists]
    scene_dep_lists  = [deps[:max_clips_per_scene]  for deps in scene_dep_lists]
    scene_pose_lists = scene_pose_lists.copy()

    max_total_clips = 90
    clip_len = 16
    # 3) 전체 클립 수 제한
    if max_total_clips is not None:
        accumulated = 0
        new_imgs, new_deps, new_poses = [], [], []
        for imgs, deps, pose in zip(scene_img_lists, scene_dep_lists, scene_pose_lists):
            # 이 Scene이 만들 수 있는 클립 수
            available_clips = len(imgs) // clip_len
            if available_clips == 0:
                continue
            # 남은 클립 수
            remaining = max_total_clips - accumulated
            if remaining <= 0:
                break

            # 이 Scene에서 뽑을 클립 수
            take_clips = min(available_clips, remaining)
            take_frames = take_clips * clip_len  # 프레임 수로 환산

            new_imgs.append(imgs[:take_frames])
            new_deps.append(deps[:take_frames])
            new_poses.append(pose)

            accumulated += take_clips

        scene_img_lists, scene_dep_lists, scene_pose_lists = new_imgs, new_deps, new_poses
            
    return scene_img_lists, scene_dep_lists, scene_pose_lists


def quaternion_to_rotmat(q):
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=np.float32)
    return R
    
def get_GTA_paths(root_dir,split):
    all_depths = []
    all_images = []
    all_poses  = []

    for idx,scene in enumerate(os.listdir(root_dir)):
        if split=="train":
            if idx == 20:
                break
        else :
            if idx < 20 :
                continue
            elif idx > 25 :
                break
                
        scene_dir  = os.path.join(root_dir, scene)
        depths_dir = os.path.join(scene_dir, "depths")
        images_dir = os.path.join(scene_dir, "images")
        poses_dir  = os.path.join(scene_dir, "poses")

        scene_depths = []
        scene_images = []
        scene_poses  = []

        if os.path.isdir(depths_dir):
            for fname in os.listdir(depths_dir):
                scene_depths.append(os.path.join(depths_dir, fname))

        if os.path.isdir(images_dir):
            for fname in os.listdir(images_dir):
                scene_images.append(os.path.join(images_dir, fname))

        if os.path.isdir(poses_dir):
            for fname in os.listdir(poses_dir):
                scene_poses.append(os.path.join(poses_dir, fname))

        all_depths.append(scene_depths)
        all_images.append(scene_images)
        all_poses.append(scene_poses)

    return all_images, all_depths, all_poses


def get_google_paths(root_dir):

    exts = ['.png', '.jpg', '.npy']
    x_paths = []
    y_paths = []
    
    img_root = os.path.join(root_dir, "images")
    dep_root = os.path.join(root_dir, "depth")

    for folder in sorted(os.listdir(img_root)):
        img_dir = os.path.join(img_root, folder)
        dep_dir = os.path.join(dep_root, folder)
        if not os.path.isdir(img_dir):
            print(f"경고: 이미지 폴더 없음: {img_dir}")
            continue
        if not os.path.isdir(dep_dir):
            print(f"경고: depth 폴더 없음: {dep_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, fname)
            base, _ = os.path.splitext(fname)
            for ext in exts:
                dep_path = os.path.join(dep_dir, base + ext)
                if os.path.isfile(dep_path):
                    x_paths.append(img_path)
                    y_paths.append(dep_path)
                    break
            else:
                print(f"경고: 대응하는 depth 파일 없음: {img_path}")

    return x_paths, y_paths


def get_kitti_individuals(video_info, clip_len, split):
    """
    기존  get_kitti_paths 의 문제점 : 클립단위로 넘겨주기 때문에 이걸 몇번째에서 자를 수 없음. 
    그대신 end index를 저장해두고, clip 단위가 아닌 전체 영상을 넘겨주기

    returns : 전체 영상 데이터 + scene의 끝점
    """
    x_clips = []
    y_clips = []
    intrin_clips = []
    extrin_clips = []
    cam_ids = []
    
    for info in video_info:
        rgb_dir        = info['rgb_path']
        depth_dir      = info['depth_path']
        intrinsic_file = info['intrinsic_file']
        extrinsic_file = info['extrinsic_file']
        camera_id      = info['camera']

        rgb_files   = sorted(os.listdir(rgb_dir))
        depth_files = sorted(os.listdir(depth_dir))
        if len(rgb_files) != len(depth_files):
            continue

        n = len(rgb_files) // clip_len # 즉 이제 n은 몫이에요

        x_clips.append([os.path.join(rgb_dir,f) for f in rgb_files[:n*clip_len]])
        y_clips.append([os.path.join(depth_dir,f) for f in depth_files[:n*clip_len]])
        intrin_clips.append(intrinsic_file)
        extrin_clips.append(extrinsic_file)
        cam_ids.append(camera_id)

    
    if split == "train":
        return x_clips, y_clips
    else:
        return x_clips, y_clips , cam_ids, intrin_clips, extrin_clips 
    
    
    
def get_kitti_video_path(root_dir, condition_num, split, binocular):
    """
    condition_num: 각 scene에서 몇 개의 condition을 가져올지
    """
    
    # 데이터 개수 ( 단안기준 )
    # scene 1 : 446 
    # scene 2 : 232
    # scene 3 : 269
    # scene 4 : 338
    # scene 5 : 836
    # => 만약 16씩 돌리면 80번 iter 돌아가면 끝남

    rgb_root = os.path.join(root_dir, "vkitti_2.0.3_rgb")
    depth_root = os.path.join(root_dir, "vkitti_2.0.3_depth")
    textgt_root = os.path.join(root_dir, "vkitti_2.0.3_textgt")
    
    video_infos = []

    for scene in sorted(os.listdir(rgb_root)):
        scene_rgb_path = os.path.join(rgb_root, scene)
        scene_depth_path = os.path.join(depth_root, scene)
        scene_textgt_path = os.path.join(textgt_root, scene)

        if not os.path.isdir(scene_rgb_path) or \
            not os.path.isdir(scene_depth_path) or \
            not os.path.isdir(scene_textgt_path):
            continue

        if (split == "train" and "Scene06" in scene) or \
            (split == "val" and "Scene06" not in scene):
            continue

        for idx, condition in enumerate(sorted(os.listdir(scene_rgb_path))):
            
            if condition not in {"15-deg-left","30-deg-left",}:
                continue
                
            print(f"Processing scene: {scene}, condition: {condition}")
            cond_rgb_path = os.path.join(scene_rgb_path, condition)
            cond_depth_path = os.path.join(scene_depth_path, condition)
            cond_textgt_path = os.path.join(scene_textgt_path, condition)

            if not os.path.isdir(cond_rgb_path) or \
                not os.path.isdir(cond_depth_path) or \
                not os.path.isdir(cond_textgt_path):
                continue

            intrinsic_file = os.path.join(cond_textgt_path, "intrinsic.txt")
            extrinsic_file = os.path.join(cond_textgt_path, "extrinsic.txt")
            if not os.path.isfile(intrinsic_file) or not os.path.isfile(extrinsic_file):
                print(f"경고: {cond_textgt_path}에 intrinsic.txt 또는 extrinsic.txt 파일이 없습니다.")
                continue
            
            if binocular:
                cam_paths = ["Camera_0", "Camera_1"] 
            else:
                cam_paths = ["Camera_0"]
                
            for cam in cam_paths:
                cam_idx = int(cam[-1])  # "Camera_0" → 0, "Camera_1" → 1
                rgb_path = os.path.join(cond_rgb_path, "frames", "rgb", cam)
                depth_path = os.path.join(cond_depth_path, "frames", "depth", cam)

                if os.path.isdir(rgb_path) and os.path.isdir(depth_path):
                    video_infos.append({
                        'rgb_path': rgb_path,
                        'depth_path': depth_path,
                        'intrinsic_file': intrinsic_file,
                        'extrinsic_file': extrinsic_file,
                        'scene': scene,
                        'condition': condition,
                        'camera': cam_idx
                    })
                    
            if idx == condition_num-1:
                break

    # 이제 video_infos에는 scene,condition,camera 따라서 경로가 설정됨
    
    return video_infos

class KITTIVideoDataset(Dataset):
    def __init__(
        self,
        rgb_paths,
        depth_paths,
        cam_ids=None,
        intrin_clips=None,
        extrin_clips=None,
        seed = 42,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
        resize_size=350,
        split="train",
        clip_len=16,
    ):
        super().__init__()
        assert split in ["train", "val"]
        assert len(rgb_paths) == len(depth_paths)
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.intrin_clips  = intrin_clips
        self.extrin_clips  = extrin_clips
        self.cam_ids = cam_ids
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.resize_size = resize_size
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.clip_len = clip_len

        # scene별로  effective clip 계산
        scene_clip_counts = [
            len(scene_rgb)//clip_len - 1  # 마지막 클립은 버리기 -> 오버플로 방지
            for scene_rgb in self.rgb_paths
        ]

        # 총 클립 개수
        self.total_clips = sum(scene_clip_counts)

        # flat idx -> scene_idx, chunk_idx
        self.flat2scene = [0] * self.total_clips
        self.flat2chunk = [0] * self.total_clips

        ptr = 0
        for scene_idx, n_eff in enumerate(scene_clip_counts):
            for chunk_idx in range(n_eff):
                self.flat2scene[ptr] = scene_idx
                self.flat2chunk[ptr] = chunk_idx
                ptr += 1
        
        if split == "train" :
            print("train_VKITTI_total_clips : ",self.total_clips)
        else :
            print("val_VKITTI_total_clips : ",self.total_clips)
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_clips

    def load_depth(self, path):
        depth_png = Image.open(path)
        depth_cm = np.array(depth_png, dtype=np.uint16).astype(np.float32)
        depth_m = depth_cm / 100.0
        depth_img = Image.fromarray ((depth_m), mode="F") 

        return depth_img

    @staticmethod
    def load_camera_params(intrinsic_path, extrinsic_path):
        """
        intrinsic.txt, extrinsic.txt 파일을 읽어 두 개의 딕셔너리를 반환합니다.
        - intrinsics: {(frame, camera_id): [fx, fy, cx, cy]}
        - extrinsics: {(frame, camera_id): 4x4 행렬}
        """
        intrinsics = {}
        with open(intrinsic_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                camera_id = int(parts[1])
                fx = float(parts[2])
                fy = float(parts[3])
                cx = float(parts[4])
                cy = float(parts[5])
                intrinsics[(frame, camera_id)] = [fx, fy, cx, cy]

        extrinsics = {}
        with open(extrinsic_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 18:
                    continue
                frame = int(parts[0])
                camera_id = int(parts[1])
                matrix_vals = list(map(float, parts[2:18]))
                transform = np.array(matrix_vals).reshape((4, 4))
                extrinsics[(frame, camera_id)] = transform

        return intrinsics, extrinsics

    @staticmethod
    def get_camera_parameters(frame, camera_id, intrinsics, extrinsics):
        """
        (frame, camera_id)에 해당하는 카메라 파라미터를 반환합니다.
        """
        intrinsic_params = intrinsics.get((frame, camera_id))
        extrinsic_matrix = extrinsics.get((frame, camera_id))
        return intrinsic_params, extrinsic_matrix

    @staticmethod
    def get_projection_matrix(frame, camera_id, intrinsics, extrinsics):
        """
        (frame, camera_id)에 해당하는 3x4 투영 행렬을 계산하여 반환합니다.
        """
        intrinsic_params, extrinsic_matrix = KITTIVideoDataset.get_camera_parameters(
            frame, camera_id, intrinsics, extrinsics
        )
        if intrinsic_params is None or extrinsic_matrix is None:
            return None

        fx, fy, cx, cy = intrinsic_params
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        RT = extrinsic_matrix[:3, :]
        P = K @ RT  # 3x4 투영행렬
        return P
    
    
    def __getitem__(self, idx):
        """
        지금 여기에 들어온 rgb_paths들은 그냥 전체 데이터를 가지고 있음  -> 이중리스트로 가지고있음 scene별로
        end_idx로 이걸 적절히 핸들링 해야함
        현재 받아오는 idx는 전체 영상길이 / clip_len으로 할거임 -> 아 이러면 안되는게, scene별로 나눠지는 몫 기준으로 해야함 ㅇㅇ 그치
        무튼 일단 idx는 각 clip의 개수라고 생각해보자.

        정리해보자면, idx는 clip idx를 넣어야함. 즉 
        """

        ## algorithm : 결국 start index만 잘 뽑으면 해결되는 문제임
        ## 주의해야할 점은, 위에서 넘겨줄때 16으로 나눠 떨어지는거만 준게 아님.-> 이러면 문제가, 나머지가 각각 다르니까 그냥 clip len만큼으로 자르자. 수정했음
        ## shift는 0부터 15까지 가능.
        ## 64개 였다면, -> 이걸 나누기 16 하면 4개가 나오는데
        """
        remaining = idx ## 예를 들어 idx가 5이다. 즉 5번째 클립을 받는 타이밍이라고 해보자
        for scene_idx, scene_rgb in enumerate(self.rgb_paths):
            n_clips = len(scene_rgb)// self.clip_len    # 만약 1번째 씬이 64개라고 해보면, nclip = 4
            effective = n_clips - 1 # 마지막꺼 버림 이슈 for overflow 방지
            if remaining < effective:
                break
            remaining -= effective    # remaining = 2
        chunk_idx = remaining   # for문을 나오고 나면, chunk idx는 scene_idx에 해당하는 씬의 몇번째 클립인지가 됨

        -> 이거 오버헤드 너무큼 생각해보면. 위로 올리기
        """

                # 2) train split이면 여기서 바로 반환
        if self.split == "train":

            scene_idx = self.flat2scene[idx]
            chunk_idx = self.flat2chunk[idx]

            scene_rgb_paths   = self.rgb_paths[scene_idx]
            scene_depth_paths = self.depth_paths[scene_idx]

            rng = random.Random(self.seed + self.epoch)
            shift = rng.randint(0, self.clip_len-1)

            base = shift + chunk_idx * self.clip_len
            rgb_paths  = scene_rgb_paths[base:base+self.clip_len]
            depth_paths= scene_depth_paths[base:base+self.clip_len]

            #rgb_paths = self.rgb_clips[idx]
            #depth_paths = self.depth_clips[idx]

            first = Image.open(rgb_paths[0]).convert("RGB")
            first = TF.resize(first, self.resize_size)
            i, j, th, tw = get_random_crop_params_with_rng(first, self.resize_size, rng)
            
            rgb_seq, depth_seq = [], []
            for rp, dp in zip(rgb_paths, depth_paths):
                img = Image.open(rp).convert("RGB")
                img = TF.resize(img, self.resize_size)
                img = TF.crop(img, i, j, th, tw)
                img = TF.normalize(TF.to_tensor(img), mean=self.rgb_mean, std=self.rgb_std)
                rgb_seq.append(img)

                depth = self.load_depth(dp)
                depth = TF.resize(depth, self.resize_size)
                depth = TF.crop(depth, i, j, th, tw)
                depth_seq.append(TF.to_tensor(depth))

            rgb_tensor = torch.stack(rgb_seq)     # [clip_len, 3, H, W]
            depth_tensor = torch.stack(depth_seq) # [clip_len, 1, H, W]

            return rgb_tensor, depth_tensor


        else :

            scene_idx = self.flat2scene[idx]
            chunk_idx = self.flat2chunk[idx]
            
            scene_rgb_paths   = self.rgb_paths[scene_idx]
            scene_depth_paths = self.depth_paths[scene_idx]

            base = chunk_idx * self.clip_len
            rgb_paths  = scene_rgb_paths  [base:base+self.clip_len]
            depth_paths= scene_depth_paths[base:base+self.clip_len]

            rgb_seq, depth_seq = [], []
            for rp, dp in zip(rgb_paths, depth_paths):
                img = Image.open(rp).convert("RGB")
                img = TF.resize(img, self.resize_size)
                img = TF.center_crop(img, self.resize_size)
                img = TF.normalize(TF.to_tensor(img), mean=self.rgb_mean, std=self.rgb_std)
                rgb_seq.append(img)

                depth = self.load_depth(dp)
                depth = TF.resize(depth, self.resize_size)
                depth = TF.center_crop(depth, self.resize_size)
                depth_seq.append(TF.to_tensor(depth))

            rgb_tensor = torch.stack(rgb_seq)     # [clip_len, 3, H, W]
            depth_tensor = torch.stack(depth_seq) # [clip_len, 1, H, W]

            
            # 3) val split일 때만 카메라 파라미터 로딩
            camera_id = self.cam_ids[scene_idx]
            intrinsic_file = self.intrin_clips[scene_idx]
            extrinsic_file = self.extrin_clips[scene_idx]
            intrinsics_dict, extrinsics_dict = self.load_camera_params(intrinsic_file, extrinsic_file)

            extrinsics_list, intrinsics_list = [], []
            for dp in depth_paths:
                frame_num = int(os.path.splitext(os.path.basename(dp))[0].split('_')[-1])
                intr_p, extr_m = self.get_camera_parameters(frame_num, camera_id, intrinsics_dict, extrinsics_dict)

                if extr_m is None:
                    extr_m = np.eye(4, dtype=np.float32)
                extrinsics_list.append(torch.tensor(extr_m, dtype=torch.float32))

                if intr_p is None:
                    fx, fy, cx, cy = 725.0087, 725.0087, 620.5, 187.0
                else:
                    fx, fy, cx, cy = intr_p
                K = torch.tensor([[fx, 0.0, cx],
                                [0.0, fy, cy],
                                [0.0, 0.0, 1.0]], dtype=torch.float32)
                intrinsics_list.append(K)

            extrinsics_tensor = torch.stack(extrinsics_list)   # [clip_len, 4, 4]
            intrinsics_tensor = torch.stack(intrinsics_list)   # [clip_len, 3, 3]
            return rgb_tensor, depth_tensor, extrinsics_tensor, intrinsics_tensor


class GTADataset(Dataset):
    def __init__(
        self,
        rgb_paths,
        depth_paths,
        pose_paths=None,
        split="train",
        clip_len=16,
        resize_size=350,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
        seed=42
    ):
        super().__init__()
        assert split in ["train", "val"]
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.pose_paths = pose_paths
        self.clip_len = clip_len
        self.resize_size = resize_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.seed = seed
        self.split = split
        self.epoch = 0      
        scene_clip_counts = [
            len(scene_rgb)//clip_len - 1  # 마지막 클립은 버리기 -> 오버플로 방지
            for scene_rgb in self.rgb_paths
        ]
        self.total_clips = sum(scene_clip_counts)
        self.flat2scene = []
        self.flat2chunk = []
        for scene_idx, n_clips in enumerate(scene_clip_counts):
            for chunk_idx in range(n_clips):
                self.flat2scene.append(scene_idx)
                self.flat2chunk.append(chunk_idx)

        if split == "train" :
            print("train_GTA_total_clips : ",self.total_clips)
        else :
            print("val_GTA_total_clips : ",self.total_clips)

            
    def __len__(self):
        return self.total_clips

    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def load_depth(self, path):
        exr_file = OpenEXR.InputFile(path)
        header   = exr_file.header()
        dw       = header['dataWindow']
        width    = dw.max.x - dw.min.x + 1
        height   = dw.max.y - dw.min.y + 1
    
        pt       = Imath.PixelType(Imath.PixelType.FLOAT)
        raw_str  = exr_file.channel('Y', pt)
        depth_np = np.frombuffer(raw_str, dtype=np.float32)
        depth_np = depth_np.reshape((height, width))
    
        # Torch Tensor로 변환
        depth_t  = torch.from_numpy(depth_np)  # (H, W)
        return depth_t.unsqueeze(0)            # (1, H, W)

    
    def __getitem__(self, idx):
        scene_idx = self.flat2scene[idx]
        chunk_idx = self.flat2chunk[idx]

        rgb_list   = self.rgb_paths[scene_idx]
        depth_list = self.depth_paths[scene_idx]
        
        rng = random.Random(self.seed + self.epoch)

        if self.split == "train":
            shift = rng.randint(0, self.clip_len - 1)
        else:
            shift = 0
            pose_list  = self.pose_paths[scene_idx]
            
        base = chunk_idx * self.clip_len + shift
        rgb_clip   = rgb_list[base:base + self.clip_len]
        depth_clip = depth_list[base:base + self.clip_len]

        if self.split == "val":
            pose_clip  = pose_list[base:base + self.clip_len]

        rgb_seq, depth_seq = [], []
        first = Image.open(rgb_clip[0]).convert("RGB")
        first = TF.resize(first, self.resize_size)
        if self.split == "train":
            i, j, h, w = get_random_crop_params_with_rng(first, self.resize_size, rng)

        for rp, dp in zip(rgb_clip, depth_clip):
            img = Image.open(rp).convert("RGB")
            img = TF.resize(img, self.resize_size)
            if self.split == "train":
                img = TF.crop(img, i, j, h, w)
            else:
                img = TF.center_crop(img, self.resize_size)
            img = TF.normalize(TF.to_tensor(img), mean=self.rgb_mean, std=self.rgb_std)
            rgb_seq.append(img)

            dimg = self.load_depth(dp)
            dimg = TF.resize(dimg, self.resize_size)
            if self.split == "train":
                dimg = TF.crop(dimg, i, j, h, w)
            else:
                dimg = TF.center_crop(dimg, self.resize_size)
            depth_seq.append(dimg)

        rgb_tensor   = torch.stack(rgb_seq)   # [clip_len, 3, H, W]
        depth_tensor = torch.stack(depth_seq) # [clip_len, 1, H, W]

        if self.split == "train":
            return rgb_tensor, depth_tensor

        # val: load camera params from JSON pose files
        intrinsics_list, extrinsics_list = [], []
        for pp in pose_clip:
            with open(pp, 'r') as f:
                data = json.load(f)
            fx, fy = data['f_x'], data['f_y']
            cx, cy = data['c_x'], data['c_y']
            K = torch.tensor([[fx, 0.0, cx],
                              [0.0, fy, cy],
                              [0.0, 0.0, 1.0]], dtype=torch.float32)
            intrinsics_list.append(K)

            ext = torch.tensor(data['extrinsic'], dtype=torch.float32)
            extrinsics_list.append(ext)

        intrinsics_tensor = torch.stack(intrinsics_list)   # [clip_len, 3, 3]
        extrinsics_tensor = torch.stack(extrinsics_list)   # [clip_len, 4, 4]

        return rgb_tensor, depth_tensor, extrinsics_tensor, intrinsics_tensor


class GoogleDepthDataset(Dataset):
    def __init__(
        self,
        img_paths,
        depth_paths,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
        resize_size=518,
        seed=42
    ):
        super().__init__()
        assert len(img_paths) == len(depth_paths), "이미지/뎁스 개수 불일치"
        self.img_paths   = img_paths
        self.depth_paths = depth_paths
        self.resize_size = resize_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        rng = random.Random(self.seed + self.epoch)
        # RGB 이미지 

        first = Image.open(self.img_paths[0]).convert("RGB")
        first = TF.resize(first, self.resize_size)
        i, j, th, tw = get_random_crop_params_with_rng(first, self.resize_size, rng)

        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = TF.resize(img, self.resize_size)
        img = TF.crop(img, i, j, th, tw)
        img = TF.normalize(TF.to_tensor(img),mean=self.rgb_mean,std=self.rgb_std)

        # Disparity/Depth
        dp = self.depth_paths[idx]
        if dp.endswith('.npy'):
            disp = np.load(dp).astype(np.float32)
            disp_img = Image.fromarray(disp)
        else:
            disp_img = Image.open(dp).convert("F")
        disp_img = TF.resize(disp_img, self.resize_size)
        disp_img = TF.crop(disp_img, i, j, th, tw)
        disp = torch.from_numpy(np.array(disp_img, np.float32)).unsqueeze(0)

        return img,disp

class CombinedDataset(Dataset):
    def __init__(self, kitti_dataset, google_dataset,ratio=4):
        super().__init__()
        self.ratio = ratio
        self.kitti_dataset = kitti_dataset
        self.google_dataset = google_dataset
        
        #print("kitti len ",len(self.kitti_dataset))
        #print("google len ",len(self.google_dataset) )

    def set_epoch(self, epoch):
        self.kitti_dataset.set_epoch(epoch)
        
    def __len__(self):
        return min(len(self.kitti_dataset), len(self.google_dataset)// self.ratio)
    
    def __getitem__(self, idx):
        kitti_item = self.kitti_dataset[idx]
        start = idx * self.ratio
        google_items = [self.google_dataset[start + i] for i in range(self.ratio)]
        
        google_imgs  = torch.stack([item[0] for item in google_items], dim=0)  # [ratio, 3, H, W]
        google_depths = torch.stack([item[1] for item in google_items], dim=0)  # [ratio, 1, H, W]

        return kitti_item, (google_imgs,google_depths)


class TartanAirVideoDataset(Dataset):
    """
    Returns:
      train: (rgb[T,3,H,W], depth[T,1,H,W])
      val:   (rgb[T,3,H,W], depth[T,1,H,W], extrinsics[T,4,4], intrinsics[T,3,3])
    Depth는 원본 값을 그대로 반환합니다.
    Supports optional depth range attributes but does not perform masking here.
    """
    def __init__(
        self,
        img_lists,
        dep_lists,
        posefile_lists=None,
        clip_len=16,
        resize_size=350,
        rgb_mean=(0.485,0.456,0.406),
        rgb_std=(0.229,0.224,0.225),
        split='train',
        seed=42,
    ):
        super().__init__()
        self.img_lists      = img_lists
        self.dep_lists      = dep_lists
        self.posefile_lists = posefile_lists or []
        self.clip_len       = clip_len
        self.resize_size    = resize_size
        self.rgb_mean       = rgb_mean
        self.rgb_std        = rgb_std
        self.split          = split
        self.seed           = seed
        self.epoch          = 0

        self.flat2scene, self.flat2chunk = [], []
        for si, imgs in enumerate(self.img_lists):
            n_chunks = len(imgs) // self.clip_len
            for ci in range(n_chunks):
                self.flat2scene.append(si)
                self.flat2chunk.append(ci)

        fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
        self.K = torch.from_numpy(K)

    def __len__(self):
        return len(self.flat2scene)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        si, ci = self.flat2scene[idx], self.flat2chunk[idx]
        base   = ci * self.clip_len

        clip_imgs = self.img_lists[si][base:base + self.clip_len]
        clip_deps = self.dep_lists[si][base:base + self.clip_len]

        rng = random.Random(self.seed + self.epoch)
        shift = rng.randint(0, self.clip_len - 1) if self.split == 'train' else 0

        first = Image.open(clip_imgs[0]).convert('RGB')
        first = TF.resize(first, self.resize_size)
        if self.split == 'train':
            i, j, h, w = get_random_crop_params_with_rng(first, self.resize_size, rng)
            crop = (i, j, h, w)
        else:
            crop = (0, 0, self.resize_size, self.resize_size)

        rgb_seq, depth_seq = [], []
        for img_p, dep_p in zip(clip_imgs, clip_deps):
            img = Image.open(img_p).convert('RGB')
            img = TF.resize(img, self.resize_size)
            if self.split == 'train':
                img = TF.crop(img, *crop)
            else:
                img = TF.center_crop(img, self.resize_size)
            img = TF.normalize(TF.to_tensor(img), mean=self.rgb_mean, std=self.rgb_std)
            rgb_seq.append(img)

            d_np = np.array(Image.open(dep_p).convert('F'), dtype=np.float32)
            d_t = torch.from_numpy(d_np).unsqueeze(0)  # [1,H,W]
            d_t = TF.resize(d_t, self.resize_size)
            if self.split == 'train':
                d_t = TF.crop(d_t, *crop)
            else:
                d_t = TF.center_crop(d_t, self.resize_size)
            depth_seq.append(d_t)

        rgb_t   = torch.stack(rgb_seq)   # [T,3,H,W]
        depth_t = torch.stack(depth_seq) # [T,1,H,W]

        if self.split == 'val':
            pf = self.posefile_lists[si]
            lines = open(pf, 'r').read().splitlines()
            Es, Ks = [], []
            for fidx in range(base + shift, base + shift + self.clip_len):
                vals = list(map(float, lines[fidx].split()))
                t_vec = np.array(vals[:3], dtype=np.float32)
                q     = np.array(vals[3:], dtype=np.float32)
                R     = quaternion_to_rotmat(q)
                E     = np.eye(4, dtype=np.float32)
                E[:3,:3], E[:3,3] = R, t_vec
                Es.append(torch.from_numpy(E))
                Ks.append(self.K)
            E_t = torch.stack(Es)  # [T,4,4]
            K_t = torch.stack(Ks)  # [T,3,3]
            return rgb_t, depth_t, E_t, K_t

        return rgb_t, depth_t


class CombinedDataset_NoSingleImg(Dataset):
    def __init__(self, kitti_dataset, gta_dataset):
        super().__init__()
        self.datasets = [kitti_dataset, gta_dataset, tartanair_dataset]
        self.lens = [len(ds) for ds in self.datasets]
        self.min_len = min(self.lens)
        # 전체 길이는 3 * min_len (세 데이터셋을 번갈아 순회)
        self.total = len(self.datasets) * self.min_len

        print(f"kitti 데이터 개수: {self.lens[0]}")
        print(f"GTA 데이터 개수: {self.lens[1]}")
        print(f"TartanAir 데이터 개수: {self.lens[2]}")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        ds_idx = idx % len(self.datasets)
        sample_idx = idx // len(self.datasets)
        return self.datasets[ds_idx][sample_idx]