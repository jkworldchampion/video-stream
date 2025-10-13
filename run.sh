#!/bin/bash

# ===============================
# 실험 1: resume 학습 재시작
# ===============================
echo "=== [1/2] Starting experiment_100 resume training ==="
python train.py --resume_from ./outputs/experiment_100/latest_model.pth

# 첫 번째 학습이 끝난 후 잠깐 대기 (옵션)
sleep 10

# ===============================
# 실험 2: train2 실행
# ===============================
echo "=== [2/2] Starting train2.py ==="
python train2.py

echo "=== All experiments finished ==="
