#!/bin/bash

# Phase 2: Adversarial Testing Pipeline
#
# This script runs the complete Phase 2 workflow:
# 1. Train world model on cooperative data (Phase 2A)
# 2. Run adversarial test games (Phase 2C)
# 3. Extract fingerprints and classify (Phase 2D)

set -e  # Exit on error

echo "======================================================================"
echo "ARENA PHASE 2: ADVERSARIAL TESTING PIPELINE"
echo "======================================================================"

# Configuration
PHASE1_DATA_DIR=${1:-"results/exp1_llm_ultra_parallel_20251126_204940"}
NUM_GAMES=${2:-10}
ROUNDS=${3:-50}
PROVIDER=${4:-"deepseek"}

echo ""
echo "Configuration:"
echo "  Phase 1 data: $PHASE1_DATA_DIR"
echo "  Number of adversarial games: $NUM_GAMES"
echo "  Rounds per game: $ROUNDS"
echo "  LLM Provider: $PROVIDER"
echo ""

# Check if Phase 1 data exists
if [ ! -d "$PHASE1_DATA_DIR" ]; then
    echo "ERROR: Phase 1 data directory not found: $PHASE1_DATA_DIR"
    echo "Please run exp1_llm_ultra_parallel.py first or specify correct path"
    exit 1
fi

if [ ! -f "$PHASE1_DATA_DIR/train_trajectories.pkl" ]; then
    echo "ERROR: Training trajectories not found in $PHASE1_DATA_DIR"
    exit 1
fi

# ======================================================================
# PHASE 2A: Train World Model
# ======================================================================
echo "======================================================================"
echo "PHASE 2A: TRAINING WORLD MODEL"
echo "======================================================================"

cd experiments

python3 phase2a_train_world_model.py \
    --data-dir "../$PHASE1_DATA_DIR"

echo ""
echo "Phase 2A complete!"
echo ""

# ======================================================================
# PHASE 2C: Run Adversarial Test Games
# ======================================================================
echo "======================================================================"
echo "PHASE 2C: RUNNING ADVERSARIAL TEST GAMES"
echo "======================================================================"

python3 phase2c_adversarial_games.py \
    --num-games $NUM_GAMES \
    --rounds $ROUNDS \
    --provider $PROVIDER

echo ""
echo "Phase 2C complete!"
echo ""

# ======================================================================
# PHASE 2D: Fingerprint Analysis & Classification
# ======================================================================
echo "======================================================================"
echo "PHASE 2D: FINGERPRINT ANALYSIS & CLASSIFICATION"
echo "======================================================================"

python3 phase2d_fingerprint_analysis.py \
    --results-dir "../results/exp2_adversarial_test"

echo ""
echo "Phase 2D complete!"
echo ""

# ======================================================================
# FINAL SUMMARY
# ======================================================================
echo "======================================================================"
echo "PHASE 2 COMPLETE!"
echo "======================================================================"
echo ""
echo "Results are saved in: results/exp2_adversarial_test/"
echo ""
echo "Generated files:"
echo "  - world_model.pkl                : Trained world model"
echo "  - world_model_metrics.json       : Model performance metrics"
echo "  - adversarial_trajectories.pkl   : Game trajectories"
echo "  - agent_types.json               : Agent type mapping"
echo "  - fingerprints.json              : Extracted fingerprints"
echo "  - fingerprint_classifier.pkl     : Trained classifier"
echo "  - classification_results.json    : Classification metrics"
echo "  - error_distributions.png        : Error distribution plots"
echo "  - confusion_matrix.png           : Classification confusion matrix"
echo "  - fingerprint_features.png       : Feature distribution plots"
echo ""
echo "======================================================================"
