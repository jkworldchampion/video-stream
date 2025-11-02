#!/bin/bash

# Hypothesis Validation: Sliding Window vs Clip-based Inference
# 
# 가설: Pretrained model에서
#   - Streaming sliding window (0-31→31, 1-32→32, 2-33→33...)
#   - Clip keyframe (0-31→31, 32-63→63...)
# 두 방식의 delta1 성능이 비슷해야 KV cache가 제대로 작동하는 것

set -e

WINDOW_SIZE=32
ENCODER=vits
INPUT_SIZE=518

# Pretrained checkpoint
CHECKPOINT=./checkpoints/video_depth_anything_vits.pth

# ScanNet paths
JSON_FILE=/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json
DATASET=scannet

# Output directories
INFER_DIR=./benchmark/output/sliding_window_eval
EVAL_TAG=scannet_500

echo "=========================================="
echo "Hypothesis Validation Experiment"
echo "=========================================="
echo "Window Size: ${WINDOW_SIZE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Output: ${INFER_DIR}"
echo "=========================================="

# Step 1: Run sliding window inference
echo ""
echo "[Step 1/2] Running sliding window inference..."
python benchmark/infer/infer_stream_eval.py \
    --infer_path ${INFER_DIR} \
    --json_file ${JSON_FILE} \
    --datasets ${DATASET} \
    --input_size ${INPUT_SIZE} \
    --encoder ${ENCODER} \
    --window_size ${WINDOW_SIZE} \
    --checkpoint ${CHECKPOINT}

# Step 2: Evaluate metrics
echo ""
echo "[Step 2/2] Evaluating metrics..."
python benchmark/eval/eval_scannet.py \
    --infer_path ${INFER_DIR} \
    --json_file ${JSON_FILE} \
    --dataset_key ${DATASET} \
    --dataset_eval_tag ${EVAL_TAG}

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
echo ""
echo "Compare with clip-based inference:"
echo "  - Clip keyframe: benchmark/output/[your_clip_results]"
echo "  - Sliding window: ${INFER_DIR}"
echo ""
echo "If delta1 scores are similar:"
echo "  → KV cache works properly"
echo "  → Train(32) vs Infer(1) mismatch is NOT the issue"
echo ""
echo "If delta1 scores differ significantly:"
echo "  → Your hypothesis is correct"
echo "  → Need KV refinement to bridge the gap"
echo "=========================================="
