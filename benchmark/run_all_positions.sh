#!/usr/bin/env bash

# Run all three position ablation experiments sequentially
# Total time: ~7.5 hours (2.5 hours × 3 positions)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Position Ablation Experiment Suite"
echo "========================================"
echo ""
echo "Testing temporal position bias hypothesis:"
echo "  - FIRST:  [i, i+1, ..., i+31] → i  (future context)"
echo "  - MIDDLE: [i-15, ..., i, ..., i+16] → i  (bidirectional)"
echo "  - LAST:   [i-31, ..., i] → i  (past context, baseline)"
echo ""
echo "Scenes: 15 (quick test)"
echo "Expected time: ~7.5 hours total"
echo "========================================"
echo ""

read -p "Press Enter to start all experiments..."

# Experiment 1: MIDDLE (expected best)
echo ""
echo "🚀 Starting Experiment 1/3: MIDDLE position..."
"${SCRIPT_DIR}/run_position_middle.sh"

# Experiment 2: FIRST
echo ""
echo "🚀 Starting Experiment 2/3: FIRST position..."
"${SCRIPT_DIR}/run_position_first.sh"

# Experiment 3: LAST (baseline)
echo ""
echo "🚀 Starting Experiment 3/3: LAST position..."
"${SCRIPT_DIR}/run_position_last.sh"

echo ""
echo "========================================"
echo "✅ All Position Ablation Experiments Complete!"
echo "========================================"
echo ""
echo "Results locations:"
echo "  - FIRST:  benchmark/output/position_first/"
echo "  - MIDDLE: benchmark/output/position_middle/"
echo "  - LAST:   benchmark/output/position_last/"
echo ""
echo "Check W&B project 'evaluation' group 'position_ablation' for metrics."
echo "========================================"
