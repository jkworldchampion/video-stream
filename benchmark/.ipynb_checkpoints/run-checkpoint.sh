#!/usr/bin/env bash
set -euo pipefail

# 이 스크립트의 위치 → 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python이 video_depth_anything/ 를 찾도록 루트 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1


# 1) streaming inference 결과 디렉토리
INFER_PATH="${ROOT}/benchmark/output/scannet_stream_31"

# 2) JSON 메타데이터
JSON_FILE="/workspace/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"

# 3) GT 루트
BENCHMARK_ROOT="/workspace/stream/Video-Depth-Anything/datasets"


mkdir -p "${INFER_PATH}"

echo "▶ Streaming inference → ${INFER_PATH}"
python "${ROOT}/benchmark/infer/infer_stream.py" \
  --infer_path "${INFER_PATH}" \
  --json_file  "${JSON_FILE}" \
  --datasets scannet

echo
echo "▶ Offline 평가 (DepthCrafter) → results.txt에 기록"
python "${ROOT}/benchmark/eval/eval.py" \
  --infer_path "${INFER_PATH}" \
  --benchmark_path "${BENCHMARK_ROOT}" \
  --datasets scannet_500

echo
echo "✅ All done!"

# python "${ROOT}/train.py"