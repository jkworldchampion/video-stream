#!/usr/bin/env bash

# Key-frame Structure Inference (논문 방법 재현)
# 항상 첫 프레임을 key-frame으로 포함 → scale consistency

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

WINDOW_SIZE=32
ENCODER=vits
INPUT_SIZE=518
BATCH_SIZE=2
MAX_SCENES=15

# Key-frame structure parameters (논문 기반)
NUM_KEYFRAMES=4    # Tk: key-frame 개수
NUM_OVERLAP=8      # To: overlapping frames
KEY_INTERVAL=8     # Δk: key-frame sampling interval

CHECKPOINT="${ROOT}/checkpoints/video_depth_anything_vits.pth"
JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
DATASET=scannet

INFER_DIR="${ROOT}/benchmark/output/keyframe_structure"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

echo "=========================================="
echo "Key-frame Structure Inference"
echo "=========================================="
echo "Window: [Tk=${NUM_KEYFRAMES} keyframes, To=${NUM_OVERLAP} overlap, future]"
echo "Key interval (Δk): ${KEY_INTERVAL}"
echo ""
echo "Example:"
echo "  [0, 8, 16, 24] (keyframes) +"
echo "  [24-31] (overlap) +"
echo "  [32-55] (future)"
echo "  → 첫 프레임(0) 항상 포함!"
echo ""
echo "Scenes: ${MAX_SCENES}"
echo "Output: ${INFER_DIR}"
echo "=========================================="

mkdir -p "${INFER_DIR}"

echo ""
echo "[Step 1/2] Inference with Key-frame Structure..."
python "${ROOT}/benchmark/infer/infer_keyframe_eval.py" \
    --infer_path "${INFER_DIR}" \
    --json_file "${JSON_FILE}" \
    --datasets ${DATASET} \
    --input_size ${INPUT_SIZE} \
    --encoder ${ENCODER} \
    --window_size ${WINDOW_SIZE} \
    --checkpoint "${CHECKPOINT}" \
    --batch_size ${BATCH_SIZE} \
    --max_scenes ${MAX_SCENES} \
    --num_keyframes ${NUM_KEYFRAMES} \
    --num_overlap ${NUM_OVERLAP} \
    --key_interval ${KEY_INTERVAL}

echo ""
echo "[Step 2/2] Evaluation..."
python "${ROOT}/benchmark/eval/eval.py" \
    --infer_path "${INFER_DIR}" \
    --benchmark_path "${BENCHMARK_ROOT}" \
    --datasets scannet_500 \
    --wandb \
    --wandb_project evaluation \
    --wandb_run_name "keyframe_structure" \
    --wandb_group "keyframe_ablation" \
    --wandb_mode online

echo ""
echo "✅ Key-frame Structure Inference Complete!"
echo "=========================================="
echo ""
echo "Compare with:"
echo "  - Clip (keyframe):  delta1 ~0.65"
echo "  - Sliding (middle): delta1 ~0.60"
echo "  - Keyframe struct:  delta1 ~???"
echo "=========================================="
