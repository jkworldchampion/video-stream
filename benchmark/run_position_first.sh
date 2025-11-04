#!/usr/bin/env bash

# Position Ablation: FIRST position (future context)
# Window: [i, i+1, ..., i+31] → predict frame i

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

WINDOW_SIZE=32
ENCODER=vits
INPUT_SIZE=518
BATCH_SIZE=2
MAX_SCENES=15  # Quick test: 15 scenes × 10min ≈ 2.5 hours

CHECKPOINT="${ROOT}/checkpoints/video_depth_anything_vits.pth"
JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
DATASET=scannet

INFER_DIR="${ROOT}/benchmark/output/position_first"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

echo "=========================================="
echo "Position Ablation: FIRST"
echo "=========================================="
echo "Window: [i, i+1, ..., i+31] → predict i"
echo "Context: FUTURE frames (causal future)"
echo "Scenes: ${MAX_SCENES}"
echo "Output: ${INFER_DIR}"
echo "=========================================="

mkdir -p "${INFER_DIR}"

echo ""
echo "[Step 1/2] Inference with FIRST position..."
python "${ROOT}/benchmark/infer/infer_clip_eval.py" \
    --infer_path "${INFER_DIR}" \
    --json_file "${JSON_FILE}" \
    --datasets ${DATASET} \
    --input_size ${INPUT_SIZE} \
    --encoder ${ENCODER} \
    --window_size ${WINDOW_SIZE} \
    --checkpoint "${CHECKPOINT}" \
    --batch_size ${BATCH_SIZE} \
    --target_position first \
    --max_scenes ${MAX_SCENES}

echo ""
echo "[Step 2/2] Evaluation..."
python "${ROOT}/benchmark/eval/eval.py" \
    --infer_path "${INFER_DIR}" \
    --benchmark_path "${BENCHMARK_ROOT}" \
    --datasets scannet_500 \
    --wandb \
    --wandb_project evaluation \
    --wandb_run_name "position_first" \
    --wandb_group "position_ablation" \
    --wandb_mode online

echo ""
echo "✅ FIRST position completed!"
echo "=========================================="
