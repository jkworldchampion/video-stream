#!/usr/bin/env bash
set -euo pipefail

###############################
# 경로 / 공통 설정
###############################

# 이 스크립트의 위치 → 프로젝트 루트 기준
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python이 video_depth_anything/ 를 찾도록 루트 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# JSON 메타데이터 및 GT 루트
#  - cache sweep용 δ1 스크립트에서 쓰던 JSON이 아니라
#  - TAE용 JSON 사용 (네가 예시로 준 tae 스크립트와 동일)
JSON_FILE_TAE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_tae.json"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

# 실험 / 캐시 길이 리스트
EXPERIMENTS=("experiment_4")
CACHE_LENS=(5 10 20 31)

###############################
# 메인 루프
###############################

for EXP in "${EXPERIMENTS[@]}"; do
  CHECKPOINT_PATH="${ROOT}/video_depth_anything_vits.pth"   # 기존 cache 실험에서 쓰던 ckpt 유지

  echo "=========================================="
  echo "🎯 Base experiment (TAE, cache sweep): ${EXP}"
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
    echo "🚀 Streaming TAE experiment: ${EXP_NAME}"
    echo "   → stream_cache_len = ${CACHE}"
    echo "   → infer_path       = ${INFER_PATH}"
    echo "------------------------------------------"

    mkdir -p "${INFER_PATH}"

    ###########################################
    # 1. Streaming inference (TAE JSON 사용)
    ###########################################
    echo "▶ Streaming inference (TAE JSON) → ${INFER_PATH}"
    python "${ROOT}/benchmark/infer/infer_stream.py" \
      --infer_path "${INFER_PATH}" \
      --json_file  "${JSON_FILE_TAE}" \
      --datasets scannet \
      --checkpoint "${CHECKPOINT_PATH}" \
      --stream_cache_len "${CACHE}" \
      --scene_limit 1

    echo
    ###########################################
    # 2. TAE 평가
    ###########################################
    echo "▶ TAE 평가 (eval_tae.py) → results.txt에 기록"
    python "${ROOT}/benchmark/eval/eval_tae.py" \
      --infer_path "${INFER_PATH}" \
      --benchmark_path "${BENCHMARK_ROOT}" \
      --datasets scannet \
      --start_idx 10 \
      --end_idx 180 \
      --eval_scenes_num 1 \
      --hard_crop \
      --wandb \
      --wandb_entity depth-finder \
      --wandb_project cache_len_tae \
      --wandb_run_name "${EXP_NAME}_tae_$(date +%Y%m%d_%H%M)" \
      --wandb_group "streaming_cache_sweep_tae" \
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
    echo "✅ ${EXP_NAME} (TAE) 완료!"
    echo
  done
done

echo "=========================================="
echo "✅ All cache-length TAE experiments done!"
echo "=========================================="