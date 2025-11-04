#!/usr/bin/env bash

# Visualize Attention Patterns: Stream vs Clip

set -euo pipefail

# Activate conda environment
source /home/ajou/miniconda/etc/profile.d/conda.sh
conda activate vda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ENCODER=vits
INPUT_SIZE=518
WINDOW_SIZE=32

CHECKPOINT="${ROOT}/checkpoints/video_depth_anything_vits.pth"
JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"

OUTPUT_DIR="${ROOT}/benchmark/output/attention_visualization"
NUM_SCENES=3        # Quick test: 3 scenes
MAX_FRAMES=100      # 100 frames per scene for streaming

echo "=========================================="
echo "Attention Pattern Visualization"
echo "=========================================="
echo ""
echo "목적:"
echo "  1. CLIP: 새로운 window마다 attention 패턴"
echo "  2. STREAM: 캐시가 쌓이면서 attention 변화"
echo "  3. 비교: Stream이 먼 과거에 제대로 attend하는가?"
echo ""
echo "Scenes: ${NUM_SCENES}"
echo "Max frames: ${MAX_FRAMES} (per scene)"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

python "${ROOT}/benchmark/infer/visualize_attention.py" \
    --json_file "${JSON_FILE}" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --encoder ${ENCODER} \
    --input_size ${INPUT_SIZE} \
    --window_size ${WINDOW_SIZE} \
    --num_scenes ${NUM_SCENES} \
    --max_frames ${MAX_FRAMES}

echo ""
echo "=========================================="
echo "✅ Attention Visualization Complete!"
echo "=========================================="
echo ""
echo "결과:"
echo "  - CLIP attention maps: ${OUTPUT_DIR}/*/clip_window_*.png"
echo "  - STREAM attention snapshots: ${OUTPUT_DIR}/*/stream_frame_*.png"
echo "  - Comparison plots: ${OUTPUT_DIR}/*/comparison.png"
echo "  - Statistics: ${OUTPUT_DIR}/*/statistics.json"
echo ""
echo "분석 포인트:"
echo "  1. Self-attention: CLIP vs STREAM 차이?"
echo "  2. Far attention: Stream이 먼 과거 프레임을 무시하는가?"
echo "  3. Attention spread: Stream이 최근 프레임에만 집중하는가?"
echo "=========================================="
