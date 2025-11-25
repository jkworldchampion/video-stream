#!/usr/bin/env bash
set -euo pipefail

# 이 스크립트의 위치 → 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python이 video_depth_anything/ 를 찾도록 루트 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# JSON 메타데이터 및 GT 루트
JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

# 실험 / 캐시 길이 리스트
EXPERIMENTS=("experiment_11")
CACHE_LENS=(5 10 20 31)   # ← 5부터 순서대로 돌게 됨

for EXP in "${EXPERIMENTS[@]}"; do
  CHECKPOINT_PATH="${ROOT}/outputs/${EXP}/best_model.pth"

  echo "=========================================="
  echo "🎯 Base experiment: ${EXP}"
  echo "   Checkpoint: ${CHECKPOINT_PATH}"
  echo "=========================================="

  # 체크포인트 존재 확인
  if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "❌ Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "   Skipping ${EXP}..."
    continue
  fi

  for CACHE in "${CACHE_LENS[@]}"; do
    EXP_NAME="${EXP}_cache${CACHE}"   # 로그/결과 구분용 이름
    INFER_PATH="${ROOT}/cache_experiments/${EXP_NAME}"

    echo
    echo "------------------------------------------"
    echo "🚀 Streaming experiment: ${EXP_NAME}"
    echo "   → stream_cache_len = ${CACHE}"
    echo "------------------------------------------"

    mkdir -p "${INFER_PATH}"

    echo "▶ Streaming inference → ${INFER_PATH}"
    python "${ROOT}/benchmark/infer/infer_stream.py" \
      --infer_path "${INFER_PATH}" \
      --json_file  "${JSON_FILE}" \
      --datasets scannet \
      --checkpoint "${CHECKPOINT_PATH}" \
      --stream_cache_len "${CACHE}"  \
      --scene_limit 20

    echo
    echo "▶ Offline 평가 (DepthCrafter) → results.txt에 기록"
    python "${ROOT}/benchmark/eval/eval.py" \
      --infer_path "${INFER_PATH}" \
      --benchmark_path "${BENCHMARK_ROOT}" \
      --datasets scannet_500 \
      --scene_limit 20 \
      --wandb \
      --wandb_entity depth-finder \
      --wandb_project cache_len \
      --wandb_run_name "${EXP_NAME}_$(date +%Y%m%d_%H%M)" \
      --wandb_group "streaming_cache_sweep" \
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
    echo "✅ ${EXP_NAME} 완료!"
    echo
  done
done

echo "=========================================="
echo "✅ All cache-length experiments done!"
echo "=========================================="