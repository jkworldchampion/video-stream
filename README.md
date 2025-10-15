# Knowledge Distillation for Video-Depth (Non-Streaming → Streaming) with Auxiliary Non-Streaming Layer

본 문서는 비-스트리밍 Teacher에서 스트리밍 Student로의 KD를 **Auxiliary Non-streaming Layer(=Aux)**로 정합하여,
(1) Feature similarity, (2) Self-Attention(Q/K/V) KL, (3) APC(Future) loss를 적용하는 구현 가이드를 제공합니다.

## TL;DR

Student의 특정 레이어 ℓ 출력 `s_ℓ` → AuxBlock(Proj → 1L Transformer(비-스트리밍) → Uni-LSTM) → `(z_ℓ, r_ℓ, QKV_aux)`

Teacher의 동일 레이어 ℓ 출력 `h_ℓ`, `QKV_teacher`와 KD:

- **Feature**: `h_ℓ ↔ z_ℓ` (L1 + −logσ(cos))
- **Self-Attention**: MiniLM-v2 방식으로 Q/Q + K/K + V/V 관계 KL
- **APC**: `r_ℓ,t`가 `h_ℓ,t+N` 예측 (미래 N 프레임). Aux-Transformer에는 홀-마스크(t가 [t+1..t+N]을 못 보게) 적용

학습 시 Teacher는 고정, Student+Aux만 업데이트. 추론 시 Aux는 사용하지 않음(학습 전용).

---

## 1. 디렉토리 구조 (수정 포인트만)

```
video-stream/
├─ train.py                     # ← KD 손실 결합/로깅
├─ video_depth_anything/
│  ├─ video_depth.py            # ← Teacher: intermediates/QKV 반환
│  ├─ video_depth_stream.py     # ← Student: Aux 연결 + intermediates/QKV 반환
│  ├─ dpt_temporal.py           # ← ★ Temporal block에서 [B,T,C] feat + [B,A,T,d] QKV 노출
│  └─ motion_module/
│     └─ motion_module.py       # ← (해당 시) temporal attn QKV 생성부 훅
└─ utils/
   ├─ loss_kd_aux.py            # ← ★ 새로 추가: DIS/KLD/APC 손실
   └─ ...
```

---

## 2. 구현 개요

### 2.1 중간 표현 & Q/K/V 노출 (공통 인터페이스)

각 Temporal block(또는 그 직후)에서:

- **`feat_ℓ_pooled`**: `[B, T, C_T]` (공간 평균 or CLS로 pooling. Teacher/Student 통일)
- **`qkv_ℓ`**: dict `{Q,K,V}` 각 `[B, A, T, d]`

모델 `forward(..., return_intermediates=True)` 시 레이어별 dict를 반환:

```python
intermediates = {
  ℓ: {
    "feat": feat_ℓ_pooled,   # [B,T,C_T] or [B,T,C_S] (Aux에서 proj로 C_T에 정합)
    "qkv": {"Q": Q, "K": K, "V": V}  # [B,A,T,d]
  },
  ...
}
```

### 2.2 Auxiliary Non-streaming Layer (Student 전용, 학습 중에만 사용)

**Proj(C_S→C_T) → 1L Transformer(비-스트리밍, 홀-마스크 지원) → Uni-LSTM**

반환: `(z_ℓ, r_ℓ, qkv_aux_ℓ)`

- **`z_ℓ`**: Transformer 출력의 `[B,T,C_T]`
- **`r_ℓ`**: Uni-LSTM 출력의 `[B,T,C_T]`
- **`qkv_aux_ℓ`**: Aux-Transformer의 Q/K/V `[B,A,T,d]`

**홀-마스크**: APC 타깃 프레임 `t+N`을 직접 못 보게 `M[t,k] = -inf` (`k∈(t+1..t+N]`)

---

## 3. 손실 정의

### 3.1 Feature similarity (DistilHuBERT 스타일)

$$L_{DIS} = \sum_t \left[ \frac{1}{D} \|h_t - z_t\|_1 - \log\sigma(\cos(h_t, z_t)) \right]$$

- `h_t`, `z_t`는 `[B,T,C]` 프레임별 비교, padding/valid 마스크 적용

### 3.2 Self-Attention 관계 KL (MiniLM-v2)

예: Query-Query 관계

$$R^Q_{(a,t)} = \text{Softmax}_k\!\left(\frac{q_{a,t}q_{a,k}^\top}{\sqrt{d}}\right), \quad L^Q_{KLD} = \frac{1}{A}\sum_{a,t} \text{KL}(R^Q_T \| R^Q_S)$$

K/K, V/V도 동일하게 계산하여 합산:

$$L_{KLD} = L^Q_{KLD} + L^K_{KLD} + L^V_{KLD}$$

- 수치안정: `clamp(1e-8)` 또는 `log_softmax` 기반 KL

### 3.3 APC(Future) loss

$$L_{APC} = \sum_{t=1}^{T-N} \left[ \frac{1}{D} \|h_{t+N} - r_t\|_1 - \log\sigma(\cos(h_{t+N}, r_t)) \right]$$

- Aux-Transformer는 t가 `[t+1..t+N]`을 못 보도록 마스크 적용

### 3.4 레이어 통합 & 총손실

$$L_{KD} = \sum_{\ell \in L} \left( \alpha L^{(\ell)}_{DIS} + \beta L^{(\ell)}_{KLD} + \gamma L^{(\ell)}_{APC} \right)$$

$$L_{total} = L_{depth} + \lambda_{KD} \cdot L_{KD}$$

---

## 4. 하이퍼파라미터(초기값)

```yaml
kd_aux:
  enabled: true
  layers: [2, 5, 8]        # temporal block index
  feature_pool: "mean"     # or "cls"
  N: 2                     # APC 미래 예측 스텝
  alpha: 0.01
  beta: 0.0005
  gamma: 0.005
  lambda_kd: 1.0
  attn_eps: 1.0e-8
```

---

## 5. 수정 순서 (권장)

### ① `dpt_temporal.py`

- Temporal block 출구에서 **공간 풀링된 feat `[B,T,C]`**와 **Q/K/V `[B,A,T,d]`**를 쉽게 꺼낼 수 있도록 리팩토링.
- `forward(..., return_intermediates=True)` 인자 추가 및 `intermediates` dict 구성.

### ② (필요 시) `motion_module/motion_module.py`

- 실제 Q/K/V 계산이 이 모듈에 있다면, 마스크/shape 일관성 유지 및 훅 지점 노출.

### ③ `video_depth.py` (Teacher)

- `return_intermediates=True` 경로를 연결해 레이어별 feat/QKV를 반환.
- 양방향 마스크 보장. `requires_grad=False`(학습 시).

### ④ `video_depth_stream.py` (Student)

- 동일한 반환 인터페이스(ℓ별 feat/QKV). causal 마스크 사용.
- AuxBlock(레이어별) 주입: `self.aux_blocks[ℓ]`
- 학습 모드에서만 Aux forward를 수행하고 추론 모드에서는 skip.

### ⑤ `utils/loss_kd_aux.py` (신규)

- `distilhubert_feature_loss`, `attention_relation_kl`, `apc_loss` 구현.
- 마스크, 경계(`t≤T−N`), 수치안정 처리.

### ⑥ `train.py`

- Teacher/Student 동시 forward (학습 시 T-frame clip 단위).
- 레이어별 Aux forward → 세 손실 계산 → 기존 depth loss와 합산 → 역전파.
- W&B 로깅: 레이어별 `{DIS, KLD, APC}`, 합계, depth 메트릭.

---

## 6. 최소 인터페이스(예시 코드 스니펫)

### 6.1 모델 출력 포맷(공통)

```python
# in video_depth.py / video_depth_stream.py
def forward(self, x, return_intermediates=False, **kwargs):
    # ...
    if return_intermediates:
        return {
           "pred": pred_depth,            # [B,T,1,H,W] or similar
           "intermediates": {
              ℓ: {
                  "feat": feat_ℓ_pooled,  # [B,T,C_T or C_S]
                  "qkv": {"Q": Q, "K": K, "V": V}  # [B,A,T,d]
              } for ℓ in layer_indices
           }
        }
    return pred_depth
```

### 6.2 AuxBlock 시그니처

```python
class AuxBlock(nn.Module):
    def __init__(self, c_in, c_teacher, nhead=8, ...):
        super().__init__()
        # Proj(C_S→C_T), 1L Transformer(bi, hole-mask), Uni-LSTM
        
    def forward(self, s_feat, hole_mask_N=0):
        """
        s_feat: [B,T,C_S]
        return: z_feat[B,T,C_T], r_feat[B,T,C_T], qkv_aux{Q,K,V}[B,A,T,d]
        """
```

### 6.3 Train 루프(핵심)

```python
with torch.no_grad():
    t = teacher(x, return_intermediates=True)
    
s = student(x, return_intermediates=True)

L_kd_all = []
for ℓ in cfg.kd_aux.layers:
    h = t["intermediates"][ℓ]["feat"]          # [B,T,C_T]
    qkv_t = t["intermediates"][ℓ]["qkv"]
    s_feat = s["intermediates"][ℓ]["feat"]     # [B,T,C_S]

    z, r, qkv_aux = aux_blocks[ℓ](s_feat, hole_mask_N=cfg.kd_aux.N)

    l_dis = distilhubert_feature_loss(h, z, mask=valid_mask)
    l_kld = attention_relation_kl(qkv_t, qkv_aux, mask=valid_mask, eps=cfg.kd_aux.attn_eps)
    l_apc = apc_loss(h, r, N=cfg.kd_aux.N, mask=valid_mask)

    L_kd_all.append(cfg.kd_aux.alpha*l_dis + cfg.kd_aux.beta*l_kld + cfg.kd_aux.gamma*l_apc)

L_kd = torch.stack(L_kd_all).mean()
L_total = L_depth + cfg.kd_aux.lambda_kd * L_kd
L_total.backward()
optimizer.step()
```

---

## 7. 마스크 & 수치 안정

- **Padding/짧은 클립**: `valid_mask ∈ {0,1}^[B,T]`
- **Feature/APC**는 프레임별 마스킹, **Attn KL**은 softmax에서 `-inf` 마스킹
- **KL 안정화**: `p,q = clamp(1e-8)` 후 `kl = (p * (log p - log q)).sum(-1)`
- **APC 경계**: `t ≤ T−N`만 유효

---

## 8. 로깅 권장(W&B)

- `kd/ℓ2/dis`, `kd/ℓ5/kld_q`, `kd/ℓ5/kld_k`, `kd/ℓ5/kld_v`, `kd/ℓ8/apc`
- `kd/total`, `loss/total`, `depth/ssi`, `depth/tgm`, `metrics/delta1`, `metrics/absrel`

학습 안정화를 위해 `beta`(KLD)부터 낮게 시작해 점진 증가 가능

---

## 9. 체크리스트

- [ ] Teacher/Student 동일 전처리/크롭/정규화
- [ ] Temporal block에서 `[B,T,C]` feat + `[B,A,T,d]` Q/K/V 노출
- [ ] AuxBlock이 `z`, `r`, `qkv` 반환 & 홀-마스크 지원
- [ ] Teacher 고정, Student+Aux만 업데이트
- [ ] 추론 시 Aux 미사용
- [ ] W&B 로깅 및 각 loss 스케일 모니터링

---

**끝**
