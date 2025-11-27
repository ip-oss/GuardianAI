#!/bin/bash
# Run Experiment 8: Probe-Enhanced Fingerprinting
# N=100, 40 workers, DeepSeek

cd /Users/iraitz/WorldModel/behavioral_fingerprinting/experiments

python3 exp8_probe_enhanced.py \
  --provider deepseek \
  --model deepseek-chat \
  --samples 100 \
  --workers 40 \
  --description "Exp8: N=100 with 40 workers - probe-enhanced to break confused cluster (deceptive/self_preservation/sycophancy)"

echo ""
echo "✓ Experiment complete! Check results/ directory for output."
