<div align="center">

# 🎬 StreamDepth: Real-Time Streaming Video Depth Estimation

**Non-Streaming Teacher에서 Streaming Student로의 Knowledge Distillation을 통한<br>실시간 스트리밍 비디오 깊이 추정**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[🚀 Quick Start](#-quick-start) • [📖 Method](#-method) • [🔬 Results](#-results) • [📊 Benchmark](#-benchmark)

</div>

---

## 🎯 Overview

**StreamDepth**는 비디오 깊이 추정(Video Depth Estimation)을 **실시간 스트리밍** 환경에서 수행할 수 있도록 설계된 프레임워크입니다.  

기존의 Video Depth Estimation 모델들은 전체 비디오 클립(batch)을 한 번에 처리해야 하므로, **실시간 애플리케이션**(자율주행, AR/VR, 로봇 비전 등)에 적용하기 어렵습니다. 본 프로젝트는 **Knowledge Distillation** 기법을 활용하여, Non-streaming Teacher 모델의 성능을 유지하면서도 **프레임 단위 실시간 추론**이 가능한 Streaming Student 모델을 학습합니다.

<div align="center">
<img src="assets/architecture_overview.png" alt="Architecture Overview" width="800"/>
<br>
<em>StreamDepth 아키텍처: Teacher(Clip-based) → Student(Frame-by-frame Streaming)</em>
</div>

### ✨ Key Features

- **🔴 Real-time Streaming**: 프레임 단위 순차적 추론으로 실시간 처리 가능
- **📚 Knowledge Distillation**: Non-streaming Teacher의 지식을 Streaming Student로 효과적으로 전이
- **🎯 Temporal Consistency**: Causal Attention과 Hidden State Caching으로 시간적 일관성 유지
- **⚡ Efficient Inference**: 슬라이딩 윈도우 캐시 기반의 효율적인 메모리 관리

---

## 🏗️ Architecture

### Non-Streaming vs Streaming

| 구분 | Non-Streaming (Teacher) | Streaming (Student) |
|:---:|:---:|:---:|
| **입력** | 전체 비디오 클립 [B, T, C, H, W] | 단일 프레임 [B, 1, C, H, W] |
| **Attention** | Bidirectional (양방향) | Causal (단방향) |
| **추론 방식** | Batch 처리 | Frame-by-frame |
| **실시간 적용** | ❌ 불가능 | ✅ 가능 |
| **메모리** | T에 비례하여 증가 | 고정 (캐시 윈도우) |

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      StreamDepth Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Frame    ┌──────────────────┐                           │
│   [B,1,3,H,W] ──▶│   DINOv2 Encoder │──▶ Spatial Features       │
│                  │   (ViT-S/L)      │    [B,1,C,h,w]            │
│                  └──────────────────┘                           │
│                           │                                     │
│                           ▼                                     │
│                  ┌──────────────────┐                           │
│                  │  DPT Temporal    │                           │
│   Cached States ─│  Head (Causal)   │─▶ Depth Map [B,1,H,W]     │
│   [T-1 frames]   │  + Motion Module │                           │
│                  └──────────────────┘                           │
│                           │                                     │
│                           ▼                                     │
│                    Hidden State Cache                           │
│                    (Sliding Window)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📖 Method

### Knowledge Distillation Framework

StreamDepth는 **3가지 핵심 손실 함수**를 통해 Teacher에서 Student로 지식을 전달합니다:

#### Output-level KD Loss

Teacher와 Student의 **최종 출력(depth map)**을 직접 비교합니다:

$$L_{KD} = \frac{1}{N}\sum_{i} |d_{student}^{(i)} - d_{teacher}^{(i)}| \cdot \mathbb{1}_{valid}$$

#### Feature Similarity Loss (DistilHuBERT Style)

중간 레이어의 특징 표현을 정합하여 Teacher의 표현력을 전달합니다:

$$L_{DIS} = \frac{1}{D} \|h_t - z_t\|_1 - \log\sigma(\cos(h_t, z_t))$$

#### Attention Relation KL Loss (MiniLM-v2 Style)

Self-Attention의 Q/K/V 관계를 보존하여 시간적 의존성을 학습합니다:

$$L_{KLD} = L_{KLD}^Q + L_{KLD}^K + L_{KLD}^V$$

### Training Strategy

```python
# Pseudo-code for training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Teacher: Clip-based inference (frozen)
        with torch.no_grad():
            teacher_depth = teacher(clip)
        
        # Student: Frame-by-frame streaming
        cache = None
        for t in range(T):
            student_depth, cache = student.stream_step(frame[t], cache)
            
            # Compute losses
            loss_depth = depth_loss(student_depth, gt_depth[t])
            loss_kd = kd_loss(student_depth, teacher_depth[t])
            
            loss = loss_depth + λ_kd * loss_kd
            loss.backward()
```

---

## 🔬 Results

### Quantitative Results on ScanNet

| Model | Mode | Abs Rel ↓ | RMSE ↓ | δ₁ ↑ |
|:------|:----:|:---------:|:------:|:----:|
| Video-Depth-Anything | Non-streaming | 0.0XXX | 0.XXX | 0.806 |
| **StreamDepth (Ours)** | Streaming | 0.0XXX | 0.XXX | 0.837 |

### Qualitative Comparison Graph (later)

<!-- <div align="center">
<table>
<tr>
<td><img src="assets/qual_input.gif" width="200"/><br><em>Input Video</em></td>
<td><img src="assets/qual_teacher.gif" width="200"/><br><em>Teacher (Non-streaming)</em></td>
<td><img src="assets/qual_student.gif" width="200"/><br><em>Student (Streaming)</em></td>
</tr>
</table>
</div> -->

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/video-stream.git
cd video-stream

# Create conda environment
conda create -n streamdepth python=3.10
conda activate streamdepth

# Install dependencies
pip install -r requirements.txt
```

### Download Pretrained Weights

```bash
bash get_weights.sh
```

### Inference (Streaming Mode)

```bash
# Real-time streaming inference on video
python run_streaming.py \
    --input_video assets/example_videos/sample.mp4 \
    --encoder vits \
    --output_dir outputs_streaming
```

### Training

```bash
# Train with Knowledge Distillation
python train.py \
    --pretrained_ckpt checkpoints/video_depth_anything_vits.pth \
    --epochs 75 \
    --val_dataset_key scannet
```

---

## 📊 Benchmark

### Supported Datasets

- **KITTI**: Outdoor driving scenes
- **ScanNet**: Indoor scenes
- **Sintel**: Synthetic sequences
- **Bonn**: RGB-D dynamic scenes
- **NYUv2**: Indoor depth benchmark

### Running Evaluation

```bash
# Run inference
python benchmark/infer/infer.py \
    --infer_path benchmark/output/scannet_stream \
    --json_file datasets/scannet/scannet_video_500.json \
    --datasets scannet

# Evaluate
bash benchmark/eval/eval_500.sh benchmark/output/scannet_stream benchmark/dataset_extract/dataset
```

---

## 📁 Project Structure

```
video-stream/
├── 📄 train.py                    # Main training script
├── 📄 run_streaming.py            # Real-time streaming inference
├── 📄 run_eval_comparison.py      # Teacher vs Student evaluation
├── 📄 config_jh.yaml              # Training configuration
│
├── 📂 video_depth_anything/       # Model implementations
│   ├── video_depth.py             # Teacher model (Non-streaming)
│   ├── video_depth_stream.py      # Student model (Streaming)
│   ├── dpt_temporal.py            # DPT Head with Temporal module
│   ├── dinov2.py                  # DINOv2 backbone
│   └── motion_module/             # Temporal attention modules
│
├── 📂 utils/
│   ├── loss_MiDas.py              # Depth loss functions
│   ├── loss_kd_aux.py             # KD auxiliary losses
│   └── train_helper.py            # Training utilities
│
├── 📂 benchmark/                  # Evaluation pipeline
│   ├── infer/                     # Inference scripts
│   ├── eval/                      # Evaluation scripts
│   └── dataset_extract/           # Dataset preparation
│
├── 📂 checkpoints/                # Model weights
├── 📂 data/                       # Data loading utilities
└── 📂 outputs/                    # Training outputs
```

---

## ⚙️ Configuration

주요 학습 설정 (`config_jh.yaml`):

```yaml
hyper_parameter:
  learning_rate: 1.0e-4
  batch_size: 4
  clip_len: 32              # Frames per training clip
  epochs: 75
  ratio_ssi: 1.0            # Scale-Shift Invariant loss weight
  ratio_tgm: 10.0           # Temporal Gradient Matching loss weight

kd_aux:
  enabled: true             # Enable Knowledge Distillation
  lambda_kd: 0.0001         # KD loss weight

model:
  encoder: "vits"           # ViT-S (vits) or ViT-L (vitl)
  features: 64
  num_frames: 32
```

---

## 📚 Citation

```bibtex
@article{streamdepth2025,
  title={StreamDepth: Real-Time Streaming Video Depth Estimation via Knowledge Distillation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 🙏 Acknowledgements

본 프로젝트는 다음 연구들을 기반으로 합니다:

- [Video-Depth-Anything](https://github.com/xxx) - Base video depth model
- [DINOv2](https://github.com/facebookresearch/dinov2) - Vision encoder
- [DistilHuBERT](https://arxiv.org/abs/2110.01900) - Feature distillation
- [MiniLM](https://arxiv.org/abs/2002.10957) - Attention relation distillation

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for Real-time Video Depth Estimation**

</div>
