#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

EXPERIMENTS=("experiment_6")
SCENE_LIMIT=2          # ✅ 앞의 2개 씬
CLIP_SIZE=32           # ✅ 32프레임 단위로 추론
INFER_ENCODER="vits"   # vits / vitl 중 택1
INPUT_SIZE=518

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=========================================="
  echo "🚀 Clip Inference: ${EXP} (first ${SCENE_LIMIT} scenes, 500 frames each)"
  echo "=========================================="

  INFER_PATH="${ROOT}/benchmark/output/kd_ablation/${EXP}_clip_2scenes"

  mkdir -p "${INFER_PATH}"

  echo "▶ Clip inference → ${INFER_PATH}"
  python "${ROOT}/benchmark/infer/infer.py" \
    --infer_path "${INFER_PATH}" \
    --json_file  "${JSON_FILE}" \
    --datasets scannet \
    --encoder "${INFER_ENCODER}" \
    --input_size "${INPUT_SIZE}" \
    --scene_limit ${SCENE_LIMIT} \
    --clip_size ${CLIP_SIZE}

  echo
  echo "▶ Offline 평가 (DepthCrafter) → results.txt에 기록"
  python "${ROOT}/benchmark/eval/eval.py" \
    --infer_path "${INFER_PATH}" \
    --benchmark_path "${BENCHMARK_ROOT}" \
    --datasets scannet_500 \
    --scene_limit ${SCENE_LIMIT} \
    --wandb \
    --wandb_entity depth-finder \
    --wandb_project evaluation \
    --wandb_run_name "${EXP}_clip_2scenes_$(date +%Y%m%d_%H%M)" \
    --wandb_group "clip" \
    --wandb_mode online

  echo
  echo "▶ 용량 큰 scannet 폴더 삭제 중..."
  SCANNET_DIR="${INFER_PATH}/scannet"
  if [ -d "${SCANNET_DIR}" ]; then
    rm -rf "${SCANNET_DIR}"
    echo "✅ Deleted: ${SCANNET_DIR}"
  else
    echo "⚠️  scannet 폴더가 존재하지 않습니다: ${SCANNET_DIR}"
  fi

  echo
  echo "✅ ${EXP} 완료!"
  echo
done

echo "=========================================="
echo "✅ All experiments done!"
echo "=========================================="
