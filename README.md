# Causal Masking Enhancement for Video Depth Anything

## 개요

본 수정사항은 Video Depth Anything의 streaming 모드에서 발생하는 성능 하락 문제를 해결하기 위해 **causal masking**을 도입한 것입니다.

## 문제점 분석

기존 Video Depth Anything의 streaming 모드에서는 다음과 같은 문제가 있었습니다:

1. **Training vs Testing 불일치**: 
   - **Training**: 32개 프레임을 동시에 받아 bidirectional temporal attention 수행
   - **Testing**: 현재 프레임 + 과거 캐시된 hidden states만 사용하여 unidirectional 예측

2. **성능 하락**: 이러한 불일치로 인해 ScanNet d1 성능이 0.926에서 0.836으로 하락

## 해결 방안: Causal Masking

### 핵심 아이디어
- **Training 시**에도 causal mask를 적용하여 각 프레임이 미래 프레임을 참조하지 못하도록 제한
- **Testing 시**와 동일한 attention pattern을 학습하여 train/test 일관성 확보

### 구현 세부사항

#### 1. Causal Mask 생성 (`_create_causal_mask`)
```python
def _create_causal_mask(self, seq_len, current_frame_idx=None, device='cuda'):
    if current_frame_idx is not None:
        # Streaming mode: only allow attention to past and current frames
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        mask[:, :current_frame_idx + 1] = True
    else:
        # Training mode: standard causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    
    causal_mask = torch.where(mask, 0.0, float('-inf'))
    return causal_mask
```

#### 2. Temporal Attention 수정
- `TemporalAttention` 클래스에 `use_causal_mask` 파라미터 추가
- Forward pass에서 causal mask 적용하여 attention score 조정

#### 3. 모든 계층에서 일관성 확보
- `TemporalModule` → `TemporalTransformer3DModel` → `TemporalTransformerBlock` → `TemporalAttention`
- 모든 계층에서 `use_causal_mask` 옵션 전달

## 사용법

### 1. 스크립트 실행 시 옵션
```bash
# Causal masking 활성화 (기본값)
python run_streaming.py --input_video your_video.mp4 --use_causal_mask

# Causal masking 비활성화 (기존 동작)
python run_streaming.py --input_video your_video.mp4 --no_causal_mask
```

### 2. 코드에서 직접 사용
```python
from video_depth_anything.video_depth_stream import VideoDepthAnything

# Causal masking 활성화
model = VideoDepthAnything(use_causal_mask=True, **model_configs)

# Causal masking 비활성화
model = VideoDepthAnything(use_causal_mask=False, **model_configs)
```

### 3. 테스트 실행
```bash
python test_causal_mask.py
```

## 기대 효과

1. **일관성 확보**: Training과 testing에서 동일한 attention pattern 사용
2. **성능 향상**: Streaming 모드에서의 depth 예측 정확도 개선
3. **시간적 일관성**: 연속 프레임 간의 더 안정적인 depth 예측

## 주요 수정 파일

- `video_depth_anything/motion_module/motion_module.py`: Causal masking 핵심 로직
- `video_depth_anything/dpt_temporal.py`: DPT Head에 causal mask 옵션 전달
- `video_depth_anything/video_depth_stream.py`: 메인 모델에 causal mask 옵션 추가
- `run_streaming.py`: 실행 스크립트에 causal mask 옵션 추가
- `test_causal_mask.py`: 테스트 스크립트 추가

## 향후 계획

1. **Fine-tuning**: Causal masking이 적용된 상태에서 모델 재학습
2. **성능 평가**: 다양한 데이터셋에서 성능 비교
3. **최적화**: Causal masking 구현의 메모리/속도 최적화
