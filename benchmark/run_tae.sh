#!/usr/bin/env bash
set -euo pipefail

# FlashDepth 프로젝트 루트 (이 스크립트가 FlashDepth 루트에 있다고 가정)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1

# VDA JSON 및 dataset 루트
#  ➜ TAE용 JSON 사용 (scannet_video_tae.json)
JSON_FILE_TAE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_tae.json"
BENCHMARK_ROOT="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets"

# FlashDepth config & checkpoint
CONFIG_PATH="${ROOT}/configs/flashdepth"
CHECKPOINT_PATH="${ROOT}/configs/flashdepth/iter_43002.pth"  # 원하는 ckpt로 교체 가능

# 실험 이름 (원래 flashdepth_scannet500 쓰던 것과 구분되게 TAE용 이름 추천)
EXPERIMENTS=("flashdepth_scannet_tae")

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=========================================="
  echo "🚀 FlashDepth ScanNet-TAE: ${EXP}"
  echo "=========================================="

  INFER_PATH="${ROOT}/benchmark/output/${EXP}"
  mkdir -p "${INFER_PATH}"

  #######################################################
  1) FlashDepth inference (TAE JSON 기준으로 예측 생성)
  #######################################################
  echo "▶ FlashDepth ScanNet-TAE inference → ${INFER_PATH}"
  torchrun --nproc_per_node=1 "${ROOT}/run_flashdepth.py" \
    --json_file "${JSON_FILE_TAE}" \
    --datasets_root "${BENCHMARK_ROOT}" \
    --infer_path "${INFER_PATH}" \
    --config_path "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --input_size 518  \
    --scene_limit 20

  echo
  ########################################################
  # 2) TAE 평가 (eval_tae.py 사용)
  ########################################################
  echo "▶ TAE eval_tae.py 실행"
  python "/home/work/juhwan/monocular_depth/stream/juhwan/Video-Depth-Anything/benchmark/eval/eval_tae.py" \
    --infer_path "${INFER_PATH}" \
    --benchmark_path "${BENCHMARK_ROOT}" \
    --datasets scannet \
    --start_idx 10 \
    --end_idx 180 \
    --eval_scenes_num 20 \
    --hard_crop \
    --wandb \
    --wandb_entity depth-finder \
    --wandb_project flashdepth_tae \
    --wandb_run_name "${EXP}_tae_$(date +%Y%m%d_%H%M)" \
    --wandb_group "flashdepth_scannet_tae" \
    --wandb_mode online

  echo
  # echo "▶ 용량 큰 scannet 폴더 삭제 중..."
  # SCANNET_DIR="${INFER_PATH}/scannet"
  # if [ -d "${SCANNET_DIR}" ]; then
  #   rm -rf "${SCANNET_DIR}"
  #   echo "✅ Deleted: ${SCANNET_DIR}"
  # else
  #   echo "⚠️  scannet 폴더가 존재하지 않습니다: ${SCANNET_DIR}"
  # fi

  echo
  echo "✅ ${EXP} (TAE) 완료!"
  echo
done

echo "=========================================="
echo "✅ All FlashDepth ScanNet-TAE experiments done!"
echo "=========================================="