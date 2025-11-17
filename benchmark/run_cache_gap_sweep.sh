#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트 기준 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

EXPERIMENT="experiment_7"
CHECKPOINT_PATH="${ROOT}/outputs/${EXPERIMENT}/best_model.pth"
OUTPUT_BASE="${ROOT}/benchmark/output/cache_gap/${EXPERIMENT}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "❌ Error: checkpoint not found at '${CHECKPOINT_PATH}'" >&2
  exit 1
fi

mkdir -p "${OUTPUT_BASE}"

declare -a GAP_CONFIGS=("default" "20" "10" "5")

for GAP in "${GAP_CONFIGS[@]}"; do
  echo "=========================================="
  if [[ "${GAP}" == "default" ]]; then
    RUN_DIR="${OUTPUT_BASE}/gap_default"
    echo "🚀 Running cache gap (default internal value, e.g., 41)"
  else
    RUN_DIR="${OUTPUT_BASE}/gap_${GAP}"
    echo "🚀 Running cache gap ${GAP}"
  fi
  echo "  > Output directory: ${RUN_DIR}"
  echo "=========================================="

  mkdir -p "${RUN_DIR}"

  echo "▶ Streaming inference (all 100 scenes, 500 frames each)"
  if [[ "${GAP}" == "default" ]]; then
    python "${ROOT}/benchmark/infer/infer_stream.py" \
      --infer_path "${RUN_DIR}" \
      --json_file "${JSON_FILE}" \
      --datasets scannet \
      --checkpoint "${CHECKPOINT_PATH}"
  else
    python "${ROOT}/benchmark/infer/infer_stream.py" \
      --infer_path "${RUN_DIR}" \
      --json_file "${JSON_FILE}" \
      --datasets scannet \
      --checkpoint "${CHECKPOINT_PATH}" \
      --cache_gap "${GAP}"
  fi

  echo
  echo "▶ Offline evaluation"
  if [[ "${GAP}" == "default" ]]; then
    WANDB_RUN_NAME="${EXPERIMENT}_gap_default_$(date +%Y%m%d_%H%M)"
  else
    WANDB_RUN_NAME="${EXPERIMENT}_gap_${GAP}_$(date +%Y%m%d_%H%M)"
  fi
  
  python "${ROOT}/benchmark/eval/eval.py" \
    --infer_path "${RUN_DIR}" \
    --benchmark_path "${BENCHMARK_ROOT}" \
    --datasets scannet_500 \
    --wandb \
    --wandb_entity depth-finder \
    --wandb_project cache_gap_sweep \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --wandb_group "${EXPERIMENT}" \
    --wandb_mode online

  echo
  echo "▶ Cleaning intermediate scannet dumps"
  SCANNET_DIR="${RUN_DIR}/scannet"
  if [[ -d "${SCANNET_DIR}" ]]; then
    rm -rf "${SCANNET_DIR}"
    echo "✅ Deleted: ${SCANNET_DIR}"
  else
    echo "⚠️  Skip: no scannet directory found"
  fi

  echo
  echo "✅ Cache gap ${GAP} run finished"
  echo
done

echo "=========================================="
echo "✅ All cache gap sweeps done!"
echo "=========================================="
