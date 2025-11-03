#!/usr/bin/env bash

# Hypothesis Validation: TRUE Sliding Window (Batch) vs Streaming Inference
# 
# 가설 검증:
#   - Clip-style sliding window (0-31→31, 1-32→32, TRUE batch processing)
#   - Frame-by-frame streaming (cache accumulation)
# 두 방식의 delta1 차이가 크면 → Train(batch) vs Infer(stream) 간극 존재!

set -euo pipefail

# 이 스크립트의 위치 → 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python이 video_depth_anything/ 를 찾도록 루트 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

WINDOW_SIZE=32
ENCODER=vits
INPUT_SIZE=518

# Pretrained checkpoint
CHECKPOINT="${ROOT}/checkpoints/video_depth_anything_vits.pth"

# ScanNet paths
JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
DATASET=scannet

# Output directories
INFER_DIR="${ROOT}/benchmark/output/clip_sliding_eval"

# GT 루트
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

echo "=========================================="
echo "Hypothesis Validation (FIXED VERSION)"
echo "=========================================="
echo "Method: TRUE Batch Processing"
echo "  - Uses VideoDepthAnything (batch model)"
echo "  - Processes entire window as batch"
echo "  - NO streaming cache within windows"
echo "=========================================="
echo "Window Size: ${WINDOW_SIZE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Output: ${INFER_DIR}"
echo "=========================================="

# mkdir -p "${INFER_DIR}"

# # Step 1: Run CLIP-style sliding window inference (TRUE batch)
# echo ""
# echo "[Step 1/2] Running clip-style sliding window inference..."
# echo "  → Multiple windows processed simultaneously (batch_size=8)"
# echo "  → Position encoding: [0-31] for each window"
# echo "  → GPU memory optimized for speed (24GB VRAM)"
# echo ""
# python "${ROOT}/benchmark/infer/infer_clip_eval.py" \
#     --infer_path "${INFER_DIR}" \
#     --json_file "${JSON_FILE}" \
#     --datasets ${DATASET} \
#     --input_size ${INPUT_SIZE} \
#     --encoder ${ENCODER} \
#     --window_size ${WINDOW_SIZE} \
#     --checkpoint "${CHECKPOINT}" \
#     --batch_size 1

# Step 2: Evaluate metrics (using eval.py like run.sh)
echo ""
echo "[Step 2/2] Offline 평가 (DepthCrafter) → results.txt + wandb"
python "${ROOT}/benchmark/eval/eval.py" \
    --infer_path "${INFER_DIR}" \
    --benchmark_path "${BENCHMARK_ROOT}" \
    --datasets scannet_500 \
    --wandb \
    --wandb_project evaluation \
    --wandb_run_name "experiment_clip_sliding" \
    --wandb_group "clip_batch" \
    --wandb_mode online

echo ""
echo "✅ All done!"
echo ""
echo "=========================================="
echo "Results comparison:"
echo "=========================================="
echo "  - Clip sliding (batch): ${INFER_DIR}"
echo "  - Frame streaming (cache): ${ROOT}/benchmark/output/sliding_window_eval"
echo ""
echo "Interpretation:"
echo "  If clip sliding >> frame streaming (delta1 difference > 5%):"
echo "    → Train(32 batch) vs Infer(1 stream) mismatch confirmed!"
echo "    → KV refinement or periodic reset needed"
echo ""
echo "  If clip sliding ≈ frame streaming (delta1 difference < 2%):"
echo "    → Mismatch is NOT the main issue"
echo "    → Check: PE overflow, model capacity, loss balancing"
echo "=========================================="
