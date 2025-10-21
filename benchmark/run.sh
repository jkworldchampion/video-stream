#!/usr/bin/env bash
set -euo pipefail

# --- 수정된 부분 시작 ---
# vda 가상 환경의 파이썬 실행 파일 전체 경로를 변수로 지정
PYTHON_EXEC="/home/ajou/miniconda/envs/vda/bin/python"
# --- 수정된 부분 끝 ---


# 이 스크립트의 위치 → 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python이 video_depth_anything/ 를 찾도록 루트 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1


# 1) streaming inference 결과 디렉토리
INFER_PATH="${ROOT}/benchmark/output/ablation/experiment_2_re"

# 2) JSON 메타데이터
JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"

# 3) GT 루트
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"


mkdir -p "${INFER_PATH}"

echo "▶ Streaming inference → ${INFER_PATH}"
# 'python'을 '${PYTHON_EXEC}'으로 변경
"${PYTHON_EXEC}" "${ROOT}/benchmark/infer/infer_stream.py" \
  --infer_path "${INFER_PATH}" \
  --json_file  "${JSON_FILE}" \
  --datasets scannet

echo
echo "▶ Offline 평가 (DepthCrafter) → results.txt에 기록"
# 'python'을 '${PYTHON_EXEC}'으로 변경
"${PYTHON_EXEC}" "${ROOT}/benchmark/eval/eval.py" \
  --infer_path "${INFER_PATH}" \
  --benchmark_path "${BENCHMARK_ROOT}" \
  --datasets scannet_500 \
  --wandb \
  --wandb_project evaluation \
  --wandb_run_name "VDA_stream_2_re" \
  --wandb_group "streaming" \
  --wandb_mode online

echo
echo "✅ All done!"