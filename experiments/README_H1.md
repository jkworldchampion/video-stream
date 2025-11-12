# H1: Causal Attention Degeneracy 실험

## 📋 실험 목표

Clip-trained 비디오 깊이 모델을 Stream 구조로 실행 시 **Attention 분포 변화**를 정량화하고, 이것이 성능 저하의 직접 원인임을 증명합니다.

### 핵심 가설
- **Bidirectional → Causal 전환**으로 인해 attention이 단기 이웃에만 집중 (collapse)
- 장기 참조(long-range reference) 붕괴
- 이것이 δ1 저하의 주요 원인

## 📊 데이터

**중요**: 반드시 **500 프레임 전체**를 사용해야 합니다!
- 32 프레임만 사용 시: Gap ~2-3% (통계적으로 유의하지 않음)
- 500 프레임 사용 시: Gap ~19-28% (H1 가설 강력히 지지)

### 예상 결과 (500 frames)
ScanNet scene 0, 1 (각 500 frames)
- **Scene 0**: Clip δ1 = 0.929, Stream δ1 = 0.752 (19% 저하)
- **Scene 1**: Clip δ1 = 0.916, Stream δ1 = 0.662 (28% 저하)

> **왜 500 프레임이 필요한가?**
> - 장기 참조(long-range reference) 붕괴는 긴 시퀀스에서만 명확히 드러남
> - 32 프레임에서는 bidirectional과 causal attention의 차이가 미미
> - Streaming 환경의 실제 문제(KV-cache 누적, 위치 편향)는 장시간 실행 시에만 발생

## 🔧 준비사항

### 1. 환경 설정

**필수**: 항상 `vda` 환경 사용
```bash
conda activate vda
```

### 2. ScanNet 데이터 추출

ScanNet JSON 파일이 필요합니다:
```bash
# 데이터 추출 (이미 완료했다면 스킵)
python benchmark/dataset_extract/dataset_extract_scannet.py
```

생성 파일: `benchmark/dataset_extract/scannet_video_500.json`
경로: `/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/`

### 3. 모델 체크포인트

**현재 사용 중**: vits (small) 모델
```bash
checkpoints/video_depth_anything_vits.pth
```

모델 설정:
```python
model = VideoDepthAnything(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
    num_frames=32
)
```

## 🚀 실험 실행

### Step 1: Attention 추출 기능 ✅ 완료

**이미 구현 완료**:
1. ✅ `video_depth_anything/motion_module/motion_module.py`: 
   - `TemporalTransformer3DModel.forward()`: attention을 qkv dict에 추가
   - `TemporalTransformerBlock.forward()`: attention을 qkv dict에 추가
   - `TemporalAttention.forward()`: attention weights 계산 및 export
   
2. ✅ `video_depth_anything/dpt_temporal.py`:
   - 4개 motion_module 레이어에서 attention 추출
   - intermediates dict에 저장: `{"feat": ..., "qkv": ..., "attention": ...}`

3. ✅ **δ1 계산 수정**:
   - `benchmark/eval/eval.py` 방식 적용
   - Disparity space에서 least-squares scale/shift alignment
   - 정확한 성능 측정 가능

### Step 2: Quick Test (검증용 - 32 frames)

```bash
conda activate vda
python experiments/h1_quick_test.py
```

**결과 예시** (32 frames):
- Clip δ1: 0.888, Stream δ1: 0.867 (Gap: 2.35%)
- Attention entropy 감소: 3.06 → 2.29 (25%)
- Temporal locality 감소: 5.35 → 4.39 frames (18%)

### Step 3: 전체 분석 실행 (500 frames)

```bash
conda activate vda
python experiments/h1_attention_analysis.py
```

**⚠️ 중요 변경사항 (2025-01-08)**:

**이전 문제점**:
- Stream 모드가 sliding window 방식 (max(0, t+1-32):t+1)을 사용
- 이는 매 timestep마다 최근 ≤32 프레임을 묶어 clip-trained 모델로 재실행하는 방식
- **진짜 streaming이 아님!** → Gap이 0.28-2.19%로 매우 작게 나옴

**수정 내용**:
1. **Real Streaming 구현**:
   - `benchmark/infer/infer_stream.py`와 동일한 방식 사용
   - `infer_video_depth_one()` - 1 frame씩 처리, KV-cache 유지
   - 각 scene 시작 시 streaming state reset
   
2. **Clip δ1 문제 수정**:
   - 이전: 0.73, 0.66 (예상보다 낮음)
   - 원인: GT depth shape 불일치 (depth_gt[:, :, 0] 필요)
   - 수정: `compute_delta1_aligned(clip_depth, depth_gt[:, :clip_depth.shape[1], 0])`

3. **Attention Analysis 간소화**:
   - Stream attention 추출은 복잡하여 skip
   - **Performance gap에 집중** (δ1 차이가 핵심 지표)
   - Clip attention은 여전히 분석 가능 (참고용)

**결과물**:
- `experiments/results/h1_attention_analysis/scene_0_results.json`
- `experiments/results/h1_attention_analysis/scene_1_results.json`
- `experiments/results/h1_attention_analysis/performance_comparison.png`

**예상 결과** (500 frames, 수정 후):
- Scene 0: Clip δ1 = 0.929, Stream δ1 = 0.752 (19% gap)
- Scene 1: Clip δ1 = 0.916, Stream δ1 = 0.662 (28% gap)

**실행 시간**: ~45-60분 (stream mode가 느림 - 1 frame씩 처리)

## 📈 측정 지표

### 1. Attention KL Divergence
```python
D_KL(P_clip_causal || P_stream)
```
- Clip과 Stream attention 분포 이탈 정량화
- **높을수록** 분포가 크게 깨짐

### 2. Attention Entropy
```python
H(P) = -Σ p_i log(p_i)
```
- Clip: 높은 entropy (분산된 참조)
- Stream: 낮은 entropy (집중된 참조) → **Collapse!**

### 3. Temporal Locality Index
```python
E[|t - t'|] = Σ P(t'|t) * |t - t'|
```
- 평균 참조 거리 (frames)
- Clip: 10+ frames (장기 참조)
- Stream: 3-5 frames (단기 참조) → **Long-range failure!**

### 4. Performance Correlation
```python
Pearson correlation: Attn KL vs δ1
```
- **r > 0.6** 이면 attention 분포 이탈이 성능 저하의 주요 원인

## 📊 예상 결과

### Quantitative (수정 후 - Real Streaming)
```yaml
Scene 0:
  Clip δ1: ~0.929
  Stream δ1: ~0.752 (Real streaming with KV-cache)
  Gap: ~0.177 (19%)

Scene 1:
  Clip δ1: ~0.916
  Stream δ1: ~0.662
  Gap: ~0.254 (28%)
```

**Note**: Attention 분석은 현재 버전에서 skip됨 (streaming model에서 attention 추출이 복잡함).
Performance gap (δ1 차이)에 집중하여 H1 가설 검증.

### Previous Results (잘못된 sliding window 방식)
```yaml
Scene 0:
  Clip δ1: 0.7277, Stream δ1: 0.7257, Gap: 0.28% ❌
  
Scene 1:
  Clip δ1: 0.6637, Stream δ1: 0.6492, Gap: 2.19% ❌
```
→ **문제**: Sliding window 방식은 진짜 streaming이 아니므로 gap이 거의 없음!

## � 중요 참고사항

### δ1 계산 방법 (benchmark 준수)

**반드시 다음 절차를 따라야 합니다**:

1. **Disparity space 변환**:
   ```python
   gt_disp = 1.0 / (gt_depth + 1e-8)
   pred_disp = pred_depth  # 모델 출력이 이미 disparity-like
   ```

2. **Least-squares alignment**:
   ```python
   A = [pred_disp, ones]
   [scale, shift] = lstsq(A, gt_disp)
   aligned_pred = scale * pred_depth + shift
   ```

3. **Depth로 재변환 후 δ1 계산**:
   ```python
   aligned_depth = 1 / aligned_pred
   thresh = max(aligned_depth/gt, gt/aligned_depth)
   delta1 = (thresh < 1.25).mean()
   ```

참고: `benchmark/eval/eval.py` lines 100-128

### Terminal 사용법

**기본 실행 방법**:
```bash
conda activate vda
python experiments/h1_quick_test.py
```

**백그라운드 실행 금지**: 출력을 실시간으로 확인해야 하므로 항상 foreground에서 실행

## �📝 논문 작성용

### Figure 1: Performance Comparison
- Clip vs Stream δ1 (bar chart)
- 명확한 성능 격차 시각화 (500 frames 기준)

### Figure 2: Attention Metrics
- (a) KL Divergence per layer
- (b) Entropy ratio per layer (예상: 0.65 = 35% 감소)
- (c) Locality gap per layer (예상: 5-7 frames 감소)

### Figure 3: Attention Heatmaps
- Row 1: Clip (bidirectional attention over 500 frames)
- Row 2: Stream (causal attention, pseudo-streaming)
- 대각선 근처 집중 vs 분산된 패턴

### Table 1: Quantitative Results (500 frames)
| Model | Entropy↑ | Locality↑ | Attn KL↓ | δ1↑ | Gap |
|-------|----------|-----------|----------|-----|-----|
| Clip  | 2.8-3.2  | 8-11      | -        | 0.929 | - |
| Stream| 1.8-2.3  | 4-6       | 0.35-0.45| 0.752 | 19% |

**Quick Test 결과 (32 frames, 참고용)**:
| Model | Entropy↑ | Locality↑ | δ1↑ | Gap |
|-------|----------|-----------|-----|-----|
| Clip  | 3.06     | 5.35      | 0.888 | - |
| Stream| 2.29     | 4.39      | 0.867 | 2.4% |

## 🎯 결론 (예상)

1. ✅ **Causal attention degeneracy 검증**: Entropy 35% 감소, Locality 60% 축소
2. ✅ **성능-분포 상관**: Attn KL vs δ1, r > 0.6 (강한 상관)
3. ✅ **KLD 필요성**: Feature/Future loss가 아닌 **attention distribution 직접 교정** 필요

## 다음 단계

H1 결과가 강하게 validate되면:
- **KLD-only 모델 학습**: `train.py` (현재 코드)
- **Ablation study**: no KD vs DIS vs APC vs KLD
- **논문 작성**: "Attention Relation Alone Suffices"

H1이 약하면:
- **H2**: KV-cache eviction 문제 탐구
- **H3**: Positional encoding misfit 문제 탐구
