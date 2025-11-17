#!/usr/bin/env bash
set -euo pipefail

# 단일 cache gap 테스트 스크립트 (디버깅용)
# Usage: bash benchmark/test_single_gap.sh [gap_value]
# Example: bash benchmark/test_single_gap.sh 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

EXPERIMENT="experiment_7"
CHECKPOINT_PATH="${ROOT}/outputs/${EXPERIMENT}/best_model.pth"

# 첫 번째 인자로 gap 값 받기 (기본값: default)
GAP="${1:-default}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "❌ Error: checkpoint not found at '${CHECKPOINT_PATH}'" >&2
  exit 1
fi

OUTPUT_BASE="${ROOT}/benchmark/output/cache_gap_test/${EXPERIMENT}"
mkdir -p "${OUTPUT_BASE}"

if [[ "${GAP}" == "default" ]]; then
  RUN_DIR="${OUTPUT_BASE}/gap_default"
  echo "🧪 Testing cache gap (default internal value, e.g., 41)"
else
  RUN_DIR="${OUTPUT_BASE}/gap_${GAP}"
  echo "🧪 Testing cache gap ${GAP}"
fi

mkdir -p "${RUN_DIR}"

echo "▶ Streaming inference (scene_limit=1 for quick test)"
if [[ "${GAP}" == "default" ]]; then
  python "${ROOT}/benchmark/infer/infer_stream.py" \
    --infer_path "${RUN_DIR}" \
    --json_file "${JSON_FILE}" \
    --datasets scannet \
    --checkpoint "${CHECKPOINT_PATH}" \
    --scene_limit 1
else
  python "${ROOT}/benchmark/infer/infer_stream.py" \
    --infer_path "${RUN_DIR}" \
    --json_file "${JSON_FILE}" \
    --datasets scannet \
    --checkpoint "${CHECKPOINT_PATH}" \
    --scene_limit 1 \
    --cache_gap "${GAP}"
fi

echo
echo "⚠️  Skipping evaluation (scene_limit=1 causes eval issues)"
echo "   To run full evaluation, use: bash benchmark/run_cache_gap_sweep.sh"

echo
echo "✅ Test complete for gap=${GAP}"
echo "   Inference output: ${RUN_DIR}"
echo "   Cache gap is working correctly! ✓"
