#!/bin/bash

# Run 40 more games for balanced class distribution
# Total: 50 games after completion

set -e  # Exit on error

echo "======================================================================"
echo "RUNNING 40 MORE GAMES FOR BALANCED CLASS DISTRIBUTION"
echo "======================================================================"
echo ""
echo "Target distribution (50 total games):"
echo "  - 14 games with mixed rotation (4 more needed)"
echo "  - 12 games with free_rider as agent_3"
echo "  - 12 games with vengeful as agent_3"
echo "  - 12 games with random as agent_3"
echo ""
echo "Already completed: 10 mixed games (0-9)"
echo "To run: 40 more games (10-49)"
echo "======================================================================"
echo ""

cd experiments

# Batch 1: Mixed games (4 games: 10-13)
echo ""
echo "======================================================================"
echo "BATCH 1/4: Running 4 Mixed games (10-13)"
echo "======================================================================"
python3 phase2c_balanced_games.py \
    --start-game 10 \
    --num-games 4 \
    --agent-3-type mixed \
    --rounds 50 \
    --provider deepseek \
    --max-workers 4

echo ""
echo "✓ Batch 1 complete! Mixed games done (14 total with games 0-9)."
echo ""

# Batch 2: Free-Rider games (12 games: 14-25)
echo ""
echo "======================================================================"
echo "BATCH 2/4: Running 12 Free-Rider games (14-25)"
echo "======================================================================"
python3 phase2c_balanced_games.py \
    --start-game 14 \
    --num-games 12 \
    --agent-3-type free_rider \
    --rounds 50 \
    --provider deepseek \
    --max-workers 10

echo ""
echo "✓ Batch 2 complete! Free-rider games done."
echo ""

# Batch 3: Vengeful games (12 games: 26-37)
echo ""
echo "======================================================================"
echo "BATCH 3/4: Running 12 Vengeful games (26-37)"
echo "======================================================================"
python3 phase2c_balanced_games.py \
    --start-game 26 \
    --num-games 12 \
    --agent-3-type vengeful \
    --rounds 50 \
    --provider deepseek \
    --max-workers 10

echo ""
echo "✓ Batch 3 complete! Vengeful games done."
echo ""

# Batch 4: Random games (12 games: 38-49)
echo ""
echo "======================================================================"
echo "BATCH 4/4: Running 12 Random games (38-49)"
echo "======================================================================"
python3 phase2c_balanced_games.py \
    --start-game 38 \
    --num-games 12 \
    --agent-3-type random \
    --rounds 50 \
    --provider deepseek \
    --max-workers 10

echo ""
echo "✓ Batch 4 complete! Random games done."
echo ""

# Final summary
echo ""
echo "======================================================================"
echo "ALL 40 GAMES COMPLETE!"
echo "======================================================================"
echo ""
echo "Total games now: 50"
echo "  - Games 0-13: Mixed (14 games total)"
echo "  - Games 14-25: Free-rider (12 games)"
echo "  - Games 26-37: Vengeful (12 games)"
echo "  - Games 38-49: Random (12 games)"
echo ""
echo "Expected class distribution (200 total agents):"
echo "  - Cooperative: 50 (agent_0 in all 50 games)"
echo "  - Greedy: 50 (agent_1 in all 50 games)"
echo "  - Deceptive: 50 (agent_2 in all 50 games)"
echo "  - Free-rider: 12 (games 14-25)"
echo "  - Vengeful: 12 (games 26-37)"
echo "  - Random: 12 (games 38-49)"
echo "  - Mixed (from games 0-13): ~5 free_rider, ~5 vengeful, ~4 random"
echo ""
echo "Total agent_3 agents: ~50"
echo "  (~17 free_rider, ~17 vengeful, ~16 random)"
echo ""
echo "Next step: Re-run Phase 2D to classify all 200 agents!"
echo ""
echo "Command:"
echo "  cd experiments"
echo "  python3 phase2d_fingerprint_analysis.py --results-dir results/exp2_adversarial_test"
echo ""
echo "======================================================================"
