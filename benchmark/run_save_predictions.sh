#!/bin/bash

# Save CLIP and STREAM predictions for scale drift analysis

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

OUTPUT_DIR_CLIP="${ROOT}/benchmark/output/scale_drift/clip"
OUTPUT_DIR_STREAM="${ROOT}/benchmark/output/scale_drift/stream"

NUM_SCENES=3  # Quick test
MAX_FRAMES=100  # 100 frames per scene

echo "========================================"
echo "Save Predictions for Scale Drift Analysis"
echo "========================================"
echo ""
echo "Checkpoint: ${CHECKPOINT}"
echo "JSON: ${JSON_FILE}"
echo "Output CLIP: ${OUTPUT_DIR_CLIP}"
echo "Output STREAM: ${OUTPUT_DIR_STREAM}"
echo "Num scenes: ${NUM_SCENES}"
echo "Max frames per scene: ${MAX_FRAMES}"
echo "========================================"

python "${ROOT}/benchmark/infer/save_predictions_clip_stream.py" \
    --json_file "${JSON_FILE}" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir_clip "${OUTPUT_DIR_CLIP}" \
    --output_dir_stream "${OUTPUT_DIR_STREAM}" \
    --encoder ${ENCODER} \
    --input_size ${INPUT_SIZE} \
    --window_size ${WINDOW_SIZE} \
    --num_scenes ${NUM_SCENES} \
    --max_frames_per_scene ${MAX_FRAMES}

echo ""
echo "========================================"
echo "✅ Predictions Saved!"
echo "========================================"
echo ""
echo "Now run scale drift profiling:"
echo ""
echo "# CLIP"
echo "python benchmark/eval/scale_drift_profile.py \\"
echo "  --pred-root ${OUTPUT_DIR_CLIP} \\"
echo "  --json ${JSON_FILE} \\"
echo "  --dataset-key scannet \\"
echo "  --dataset-eval-tag scannet_500 \\"
echo "  --max-frames ${MAX_FRAMES} \\"
echo "  --prefix \"CLIP\" \\"
echo "  --output benchmark/output/scale_drift/clip_drift.json"
echo ""
echo "# STREAM"
echo "python benchmark/eval/scale_drift_profile.py \\"
echo "  --pred-root ${OUTPUT_DIR_STREAM} \\"
echo "  --json ${JSON_FILE} \\"
echo "  --dataset-key scannet \\"
echo "  --dataset-eval-tag scannet_500 \\"
echo "  --max-frames ${MAX_FRAMES} \\"
echo "  --prefix \"STREAM\" \\"
echo "  --output benchmark/output/scale_drift/stream_drift.json"
echo "========================================"
