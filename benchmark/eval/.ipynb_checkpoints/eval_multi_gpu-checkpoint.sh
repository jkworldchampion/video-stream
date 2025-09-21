#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU evaluation script for ScanNet-500
# Usage: bash eval_multi_gpu.sh <infer_path> <benchmark_path> [num_gpus]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <infer_path> <benchmark_path> [num_gpus]"
    echo "  infer_path: Directory containing inference results (.npy files)"
    echo "  benchmark_path: Directory containing ground truth data"
    echo "  num_gpus: Number of GPUs to use (optional, default: all available)"
    exit 1
fi

INFER_PATH="$1"
BENCHMARK_PATH="$2"
NUM_GPUS="${3:-auto}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Multi-GPU ScanNet-500 Evaluation"
echo "📁 Inference path: ${INFER_PATH}"
echo "📁 Benchmark path: ${BENCHMARK_PATH}"

if [ "$NUM_GPUS" = "auto" ]; then
    echo "🔧 Using all available GPUs"
    python "${SCRIPT_DIR}/eval.py" \
        --infer_path "${INFER_PATH}" \
        --benchmark_path "${BENCHMARK_PATH}" \
        --datasets scannet_500 \
        --infer_type npy
else
    echo "🔧 Using ${NUM_GPUS} GPUs"
    python "${SCRIPT_DIR}/eval.py" \
        --infer_path "${INFER_PATH}" \
        --benchmark_path "${BENCHMARK_PATH}" \
        --datasets scannet_500\
        --infer_type npy \
        --num_gpus "${NUM_GPUS}"
fi

echo "✅ Multi-GPU evaluation completed!"
echo "📄 Results saved to: ${INFER_PATH}/results.txt"
