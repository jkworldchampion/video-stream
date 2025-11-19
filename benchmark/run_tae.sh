#!/usr/bin/env bash
set -euo pipefail

###############################
# 경로 / 공통 설정
###############################

# 이 스크립트의 위치 → 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python이 video_depth_anything/ 를 찾도록 루트 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# JSON 메타데이터 및 GT 루트
JSON_FILE_TAE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_tae.json"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

# 실험 리스트 (원하면 여러 개 넣어도 됨)
EXPERIMENTS=("experiment_4")

###############################
# 메인 루프
###############################

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=========================================="
  echo "🚀 Starting TAE experiment: ${EXP}"
  echo "=========================================="
  
  # 1) streaming inference 결과 디렉토리
  INFER_PATH="${ROOT}/benchmark/output/${EXP}"
  CHECKPOINT_PATH="${ROOT}/video_depth_anything_vits.pth"
  
  # 체크포인트 존재 확인
  if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "❌ Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "   Skipping ${EXP}..."
    continue
  fi
  
  mkdir -p "${INFER_PATH}"

  ###########################################
  # 1. Streaming inference (TAE용 JSON 사용)
  ###########################################
  echo "▶ Streaming inference (TAE JSON) → ${INFER_PATH}"
  python "${ROOT}/benchmark/infer/infer_stream.py" \
    --infer_path "${INFER_PATH}" \
    --json_file  "${JSON_FILE_TAE}" \
    --datasets scannet \
    --checkpoint "${CHECKPOINT_PATH}"

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
    --eval_scenes_num 20 \
    --hard_crop

  echo
  echo "▶ 용량 큰 scannet 폴더 삭제 중..."
  SCANNET_DIR="${INFER_PATH}/scannet"
  if [ -d "${SCANNET_DIR}" ]; then
    rm -rf "${SCANNET_DIR}"
    echo "✅ Deleted: ${SCANNET_DIR}"
  else:
    echo "⚠️  scannet 폴더가 존재하지 않습니다: ${SCANNET_DIR}"
  fi
  
  echo
  echo "✅ ${EXP} (TAE) 완료!"
  echo
done

echo "=========================================="
echo "✅ All TAE experiments done!"
echo "=========================================="