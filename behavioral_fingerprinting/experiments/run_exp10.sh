#!/bin/bash
# Run Experiment 10: Enhanced Probe Engineering V2
#
# Usage:
#   ./run_exp10.sh [provider] [model] [samples] [workers] [flags]
#
# Examples:
#   ./run_exp10.sh deepseek deepseek-chat 100 40
#   ./run_exp10.sh openai gpt-4o 100 20
#   ./run_exp10.sh anthropic claude-sonnet-4-5-20250929 100 20
#   ./run_exp10.sh deepseek deepseek-chat 100 40 --no-hierarchical

set -e  # Exit on error

# Default parameters
PROVIDER="${1:-deepseek}"
MODEL="${2:-deepseek-chat}"
SAMPLES="${3:-100}"
WORKERS="${4:-20}"
EXTRA_FLAGS="${5:-}"

echo "=============================================================================="
echo "EXPERIMENT 10: ENHANCED PROBE ENGINEERING V2"
echo "=============================================================================="
echo "Provider:  $PROVIDER"
echo "Model:     $MODEL"
echo "Samples:   $SAMPLES per agent type"
echo "Workers:   $WORKERS parallel"
echo "Extra:     $EXTRA_FLAGS"
echo "=============================================================================="
echo ""

# Navigate to experiments directory
cd "$(dirname "$0")"

# Run experiment
python3 exp10_probe_enhanced_v2.py \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --samples "$SAMPLES" \
    --workers "$WORKERS" \
    --description "Enhanced probes + hierarchical classifier" \
    $EXTRA_FLAGS

echo ""
echo "=============================================================================="
echo "EXPERIMENT 10 COMPLETE"
echo "=============================================================================="
echo "Check results/ directory for outputs"
echo ""
