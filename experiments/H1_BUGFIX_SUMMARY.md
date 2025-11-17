# H1: Attention Sink Bug - Root Cause Analysis

## 📋 Summary

**Bug:** Position Encoding (PE) 이중 적용으로 인한 KV cache 왜곡
**Impact:** Stream mode에서 attention collapse ("attention sink" - 첫 토큰에 90%+ weight)
**Fix:** 1-line change in `motion_module.py` Line 330

---

## 🔍 Root Cause

### Cache 저장/재사용 흐름

```
┌─────────────────────────────────────────────────────────────┐
│ Frame 0 (첫 프레임)                                            │
├─────────────────────────────────────────────────────────────┤
│ 1. hidden_states (h0) - raw, no PE                          │
│ 2. PE(h0) - position encoding 적용                           │
│ 3. Q/K/V = to_q/k/v(PE(h0))                                 │
│ 4. Attention computation                                     │
│ 5. Output = hidden_states (PE 적용된 상태)                   │
│ 6. Cache ← hidden_states (PE 적용 후 값 저장) ✓             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Frame 1 (Stream mode - BEFORE FIX)                           │
├─────────────────────────────────────────────────────────────┤
│ 1. now = h1 (raw)                                            │
│ 2. past = Cache = PE(h0) ← 이미 PE 적용됨!                  │
│ 3. now_pos = PE(h1) ✓                                        │
│ 4. past_pos = PE(past) = PE(PE(h0)) ✗✗✗ 이중 적용!         │
│ 5. q_now = to_q(PE(h1)) ✓                                   │
│ 6. k_past = to_k(PE(PE(h0))) ✗ 왜곡된 embedding!           │
│ 7. Attention(q_now, k_past) → 분포 왜곡!                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Frame 1 (Stream mode - AFTER FIX)                            │
├─────────────────────────────────────────────────────────────┤
│ 1. now = h1 (raw)                                            │
│ 2. past = Cache = PE(h0) ← 이미 PE 적용됨                   │
│ 3. now_pos = PE(h1) ✓                                        │
│ 4. past_pos = past = PE(h0) ✓ 그대로 사용 (재적용 X)        │
│ 5. q_now = to_q(PE(h1)) ✓                                   │
│ 6. k_past = to_k(PE(h0)) ✓ 정상 embedding                   │
│ 7. Attention(q_now, k_past) → 정상 분포 ✓                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🐛 Buggy Code

```python
# motion_module/motion_module.py Line 325-328 (BEFORE)
if self.pos_encoder is not None:
    now_pos  = self.pos_encoder(now)      # ✓ OK
    past_pos = self.pos_encoder(past) if past is not None else None  # ✗ BUG!
    #                           ^^^^
    #                           이미 PE 적용된 값인데 또 적용!
else:
    now_pos, past_pos = now, past
```

---

## ✅ Fixed Code

```python
# motion_module/motion_module.py Line 325-332 (AFTER)
if self.pos_encoder is not None:
    now_pos  = self.pos_encoder(now)
    # FIXED: past already has PE applied (from cache), use as-is!
    past_pos = past if past is not None else None  # ✓ No PE re-application!
else:
    now_pos, past_pos = now, past
```

---

## 🎯 Expected Impact

### Before Fix:
- **Attention distribution**: `[0.95, 0.03, 0.01, 0.01, ...]`
- **Entropy**: ~0.2 (near zero - collapsed)
- **First token dominance**: >90%
- **Symptom**: "Attention sink" - 모든 query가 첫 캐시 토큰만 참조

### After Fix:
- **Attention distribution**: `[0.40, 0.35, 0.15, 0.08, 0.02]` (예시)
- **Entropy**: ~1.2-1.5 (정상 범위)
- **First token weight**: 20-50% (recency bias에 따라)
- **Expected**: Causal attention with recency bias (정상적인 동작)

---

## 📊 Validation Plan

1. ✅ Code fix applied
2. ⏳ Run debug script: `python experiments/h1_debug_position_encoding.py`
   - Check attention distribution at each step
   - Verify no attention sink (first token < 50% after frame 5)
3. ⏳ Re-run attention analysis: `python experiments/h1_attention_analysis.py`
   - Compare clip vs stream attention KL divergence
   - Check entropy ratio (should be closer to 1.0)
4. ⏳ Performance check
   - Measure δ1 gap reduction (expect: 25% → 10-15%)
   - ScanNet scene 0,1 validation

---

## 📚 Key Insights

1. **Cache 내용 파악이 중요**: PE 전인지 후인지 명확히 알아야 함
2. **AnimateDiff 원본은 이 버그 없음**: 
   - 원본은 KV cache 없이 항상 전체 시퀀스 처리
   - Stream 모드 추가 시 cache 로직 미흡
3. **INFER_LEN = 32**: Clip mode는 32프레임 청크 처리
4. **Attention sink ≠ Recency bias**:
   - Sink: 버그로 인한 collapse (첫 토큰 >90%)
   - Recency: 정상적인 causal attention 특성 (최근 프레임 40-60%)

---

## 🔗 Related Files

- `video_depth_anything/motion_module/motion_module.py` (수정됨)
- `experiments/h1_attention_analysis.py` (INFER_LEN 32로 수정)
- `experiments/h1_debug_position_encoding.py` (검증용)
- `experiments/h1_fix_position_encoding.md` (이 문서)

---

## ✨ Credits

분석 과정에서 GPT의 초기 가설(position offset 누락)은 틀렸으나,
코드 정밀 분석을 통해 진짜 원인(PE 이중 적용) 발견!
