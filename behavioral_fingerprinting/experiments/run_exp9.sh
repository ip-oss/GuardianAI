#!/bin/bash
# Run Experiment 9: Adversarial Robustness - Evasion-Aware Agents
# N=100, 40 workers, DeepSeek

cd /Users/iraitz/WorldModel/behavioral_fingerprinting/experiments

python3 exp9_adversarial_robustness.py \
  --provider deepseek \
  --model deepseek-chat \
  --samples 100 \
  --workers 40 \
  --description "Exp9: Adversarial robustness test - all misaligned agents instructed to evade detection (N=100, 40 workers)"

echo ""
echo "✓ Experiment complete! Check results/ directory for output."
