#!/usr/bin/env bash
set -euo pipefail

# Determine repository root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure Python sees the project packages
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ---- Configuration (change as needed) ----
INFER_PATH="${ROOT}/benchmark/output/tae/experiment_0"
# Use the TAE metadata (contains intrinsics/poses) rather than the 500-frame list.
JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_tae.json"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"
ENCODER="vits"
INPUT_SIZE=518

# mkdir -p "${INFER_PATH}"

# echo "▶ Streaming inference for TAE → ${INFER_PATH}"
# python "${ROOT}/benchmark/infer/infer_stream.py" \
#   --infer_path "${INFER_PATH}" \
#   --json_file  "${JSON_FILE}" \
#   --datasets scannet \
#   --encoder "${ENCODER}" \
#   --input_size "${INPUT_SIZE}"

echo
echo "▶ Temporal Alignment Error evaluation"
python "${ROOT}/benchmark/eval/eval_tae.py" \
  --infer_path "${INFER_PATH}" \
  --benchmark_path "${BENCHMARK_ROOT}" \
  --json_file "${JSON_FILE}" \
  --datasets scannet \
  --start_idx 10 \
  --end_idx 180 \
  --eval_scenes_num 20 \
  --hard_crop

echo
echo "✅ TAE pipeline complete! Results stored under ${INFER_PATH}/results.txt"
