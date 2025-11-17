# H1: Position Encoding Bug Fix

## 🐛 문제 진단 (정정됨)

### ❌ 초기 가설 (틀림)
처음에는 position encoding offset이 누락되었다고 생각했으나, 코드 재분석 결과 **다른 문제**였습니다.

### ✅ 진짜 버그 발견
**`motion_module/motion_module.py` Line 325-328:**
```python
# Positional
if self.pos_encoder is not None:
    now_pos  = self.pos_encoder(now)
    past_pos = self.pos_encoder(past) if past is not None else None  # ← BUG!
```

### 문제 상황
**Cache의 정체:**
- Cache에 저장되는 값 = `hidden_states` (Line 382 output)
- 이 값은 **이미 position encoding이 적용된 후**의 값입니다

**버그:**
- Frame 0: `now` → `PE(now)` → compute Q/K/V → output → **저장 (PE 적용됨)** ✅
- Frame 1:
  - `past` = cached value = **PE(now from frame 0)** (이미 PE 적용됨)
  - `past_pos` = `PE(past)` = **PE(PE(now from frame 0))** ❌ (이중 적용!)
  - 현재 `now` → `PE(now)` → Q/K/V 계산

### 결과
1. **Position encoding 이중 적용**으로 캐시된 토큰의 embedding이 왜곡됨
2. 왜곡된 embedding은 정상적인 현재 query와 semantic 불일치
3. **Attention collapse**: 왜곡이 누적되면서 특정 토큰(주로 첫 번째)으로 attention이 몰림

## 🔧 수정 방안

### **CORRECT FIX (구현됨)**: PE 재적용 방지
```python
# motion_module/motion_module.py Line 325-332
if self.pos_encoder is not None:
    now_pos  = self.pos_encoder(now)
    # FIXED: past already has PE applied (from cache), use as-is!
    past_pos = past if past is not None else None  # ← No PE re-application!
else:
    now_pos, past_pos = now, past
```

**논리:**
1. Cache = `hidden_states` (Line 382 output, PE 적용 후)
2. 현재 프레임: `now` → `PE(now)` → Q/K/V ✅
3. 과거 프레임: `past` (이미 PE 적용됨) → **그대로 사용** → Q/K/V ✅

### ~~Option 1: Cache length offset 전달~~ (불필요)
이 방법은 cache가 PE 적용 **전** 값을 저장할 때 필요합니다.
하지만 현재 구현은 PE 적용 **후** 값을 저장하므로, 단순히 재적용만 방지하면 됩니다.

### ~~Option 2: Cache를 PE 전 값으로 변경~~ (비효율적)
PE를 매번 새로 계산하는 것보다, PE 적용된 값을 캐시하는 것이 더 효율적입니다.

## 🎯 수정 후 예상 결과

### Before (buggy - PE 이중 적용):
```
Frame 0: PE(h0) → cache
Frame 1: PE(PE(h0)) + PE(h1) → attention collapse!
  - past embedding이 왜곡되어 현재 query와 semantic 불일치
  - 특정 토큰으로 attention sink 발생
  - attn ≈ [0.95, 0.05] (첫 토큰 dominance)
```

### After (fixed - PE 단일 적용):
```
Frame 0: PE(h0) → cache
Frame 1: PE(h0) + PE(h1) → normal attention distribution
  - past/now 모두 정상적인 PE 적용
  - Recency bias 또는 contextual attention
  - attn ≈ [0.60, 0.40] (balanced or recency-focused)
```

**주의:** 
- 여전히 clip(bidirectional)과 stream(causal)의 attention 분포는 다를 것입니다
- 하지만 "attention sink"같은 극단적 collapse는 사라질 것입니다
- Recency bias (최근 프레임 선호)는 causal attention의 정상적인 특성입니다

## 📋 Implementation Checklist

1. [ ] Modify `PositionalEncoding.forward()` to accept `pos_offset`
2. [ ] Pass cache length as offset in `TemporalAttention.forward()`
3. [ ] Test with debug script (`h1_debug_position_encoding.py`)
4. [ ] Re-run attention analysis (`h1_attention_analysis.py`)
5. [ ] Verify attention distribution is no longer collapsed
6. [ ] Check performance delta1 gap reduction

## 🔬 Verification

Run debug script:
```bash
python experiments/h1_debug_position_encoding.py
```

Expected output (after fix):
```
Frame 0:
  Layer 0 attention shape: [1, H, 1, 1]
  Attention distribution: [1.000]
  First token weight: 1.000

Frame 1:
  Layer 0 attention shape: [1, H, 1, 2]
  Attention distribution: [0.60, 0.40]  ← NOT [0.95, 0.05]!
  First token weight: 0.600

Frame 5:
  Layer 0 attention shape: [1, H, 1, 6]
  Attention distribution: [0.15, 0.15, 0.15, 0.20, 0.20, 0.15]
  First token weight: 0.150  ← NOT > 0.80!
  ⚠️  ATTENTION SINK detected! [SHOULD NOT APPEAR]
```

## 📚 References

- AnimateDiff original: No streaming mode, this bug doesn't exist
- Our modification: Added KV caching without fixing position encoding
- Similar bugs: 
  - StreamingTransformer positional encoding offset
  - GPT-2 KV cache position handling
