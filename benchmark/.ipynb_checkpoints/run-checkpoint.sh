#!/usr/bin/env bash
set -euo pipefail

# 이 스크립트의 위치 → 프로젝트 루트 계산
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python이 video_depth_anything/ 를 찾도록 루트를 PYTHONPATH에 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# 1) streaming inference 결과 디렉토리
INFER_PATH="${ROOT}/benchmark/output/scannet_stream_20"

# 2) JSON 메타데이터
JSON_FILE="/workspace/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"

# 3) GT 루트
BENCHMARK_ROOT="/workspace/stream/Video-Depth-Anything/datasets"

mkdir -p "${INFER_PATH}"

echo "▶ Streaming inference → ${INFER_PATH}"
python "${ROOT}/benchmark/infer/infer_streaming.py" \
  --infer_path "${INFER_PATH}" \
  --json_file  "${JSON_FILE}" \
  --datasets   scannet

echo
echo "▶ Offline 평가 (DepthCrafter) → results.txt에 기록"
bash "${ROOT}/benchmark/eval/eval.sh" "${INFER_PATH}" "${BENCHMARK_ROOT}"

echo
echo "✅ All done!"
