#!/usr/bin/env bash
set -euo pipefail

# 이 스크립트의 위치 → 프로젝트 루트 계산
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python이 video_depth_anything/ 를 찾도록 루트를 PYTHONPATH에 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# 1) streaming inference 결과 디렉토리
INFER_PATH="${ROOT}/benchmark/output/scannet_stream_23_multi_gpu"

# 2) JSON 메타데이터
JSON_FILE="/workspace/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"

# 3) GT 루트
BENCHMARK_ROOT="/workspace/stream/Video-Depth-Anything/datasets"

# 4) GPU 개수 (기본: 모든 GPU 사용)
NUM_GPUS=${1:-"auto"}

mkdir -p "${INFER_PATH}"

echo "▶ 멀티 GPU 스트리밍 inference → ${INFER_PATH}"
if [ "$NUM_GPUS" = "auto" ]; then
    echo "  사용할 GPU: 모든 GPU"
    python "${ROOT}/benchmark/infer/infer_streaming_multi_gpu.py" \
      --infer_path "${INFER_PATH}" \
      --json_file  "${JSON_FILE}" \
      --datasets   scannet
else
    echo "  사용할 GPU: ${NUM_GPUS}개"
    python "${ROOT}/benchmark/infer/infer_streaming_multi_gpu.py" \
      --infer_path "${INFER_PATH}" \
      --json_file  "${JSON_FILE}" \
      --datasets   scannet \
      --num_gpus   "${NUM_GPUS}"
fi

echo
echo "▶ Multi-GPU 평가 (ScanNet-500) → results.txt에 기록"
bash "${ROOT}/benchmark/eval/eval_multi_gpu.sh" "${INFER_PATH}" "${BENCHMARK_ROOT}" "${NUM_GPUS}"

echo
echo "✅ 멀티 GPU inference & 평가 완료!"
