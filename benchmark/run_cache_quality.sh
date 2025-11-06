#!/bin/bash

# KV Cache Quality Analysis Runner

# Activate conda environment
source /home/ajou/miniconda/etc/profile.d/conda.sh
conda activate vda

# Configuration
JSON_FILE="/home/work/juhwan/monocular_depth/stream/Video-Depth-Anything/datasets/scannet/scannet_video_500.json"
CHECKPOINT="../checkpoints/video_depth_anything_vits.pth"
OUTPUT_DIR="./output/cache_quality_analysis"
ENCODER="vits"
WINDOW_SIZE=32
NUM_WINDOWS=10

echo "========================================"
echo "KV Cache Quality Analysis"
echo "========================================"
echo "JSON: $JSON_FILE"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Encoder: $ENCODER"
echo "Window Size: $WINDOW_SIZE"
echo "Num Windows: $NUM_WINDOWS"
echo "========================================"

# Run cache age analysis
echo ""
echo "🔬 Running Cache Age Analysis..."
python infer/analyze_cache_quality.py \
    --json_file "$JSON_FILE" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --encoder "$ENCODER" \
    --window_size "$WINDOW_SIZE" \
    --num_windows "$NUM_WINDOWS"

echo ""
echo "✅ Analysis Complete!"
echo "Results saved to: $OUTPUT_DIR"
