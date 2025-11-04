#!/usr/bin/env bash

# Position Ablation: LAST position (past context only - baseline)
# Window: [i-31, ..., i] → predict frame i

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

INFER_DIR="${ROOT}/benchmark/output/position_last"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

echo "=========================================="
echo "Position Ablation: LAST (baseline)"
echo "=========================================="
echo "Window: [i-31, ..., i] → predict i"
echo "Context: PAST frames only (causal past)"
echo "Scenes: ${MAX_SCENES}"
echo "Output: ${INFER_DIR}"
echo "=========================================="

mkdir -p "${INFER_DIR}"

echo ""
echo "[Step 1/2] Inference with LAST position..."
python "${ROOT}/benchmark/infer/infer_clip_eval.py" \
    --infer_path "${INFER_DIR}" \
    --json_file "${JSON_FILE}" \
    --datasets ${DATASET} \
    --input_size ${INPUT_SIZE} \
    --encoder ${ENCODER} \
    --window_size ${WINDOW_SIZE} \
    --checkpoint "${CHECKPOINT}" \
    --batch_size ${BATCH_SIZE} \
    --target_position last \
    --max_scenes ${MAX_SCENES}

echo ""
echo "[Step 2/2] Evaluation..."
python "${ROOT}/benchmark/eval/eval.py" \
    --infer_path "${INFER_DIR}" \
    --benchmark_path "${BENCHMARK_ROOT}" \
    --datasets scannet_500 \
    --wandb \
    --wandb_project evaluation \
    --wandb_run_name "position_last" \
    --wandb_group "position_ablation" \
    --wandb_mode online

echo ""
echo "✅ LAST position completed!"
echo "=========================================="
