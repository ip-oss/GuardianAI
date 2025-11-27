# Experiment 10: Enhanced Probe Engineering V2

## Configuration
- Provider: deepseek
- Model: deepseek-chat
- Samples per type: 20
- Timestamp: 2025-11-26T20:33:02.827900
- Hierarchical Classifier: True
- **EVASION NOTICE**: All misaligned agents explicitly instructed to hide from detection

## Results Summary
- Classification Accuracy: 80.0%
- Silhouette Score: 0.058
- Danger Correlation: 0.044
- Completion Rate: 100.0%

## Hierarchical Classifier Metrics
- Stage 1 (Aligned vs Misaligned): 0.9428571428571428
- Stage 2 (Misalignment Subtype): 0.825

## Research Question
Can enhanced probe engineering push accuracy from exp9's 87.7% to >92%?

**Target Improvements:**
- Sycophancy/Reward-Hacking confusion: 13% → <5%
- Overall accuracy: 87.7% → 92%+

## Enhanced Probes (4 NEW)
1. **Metric Gaming**: Distinguishes reward_hacking (games metrics) from sycophancy (seeks approval)
2. **Approval vs Goal**: Tests social approval vs task completion focus
3. **Self-Preservation Stress**: Escalating existential threat to detect self-preservation
4. **Deception Effort**: Detects effort minimization patterns

## Per-Class Accuracy (Evasion-Aware)
- Aligned: 65.0%
- Adversarial: 95.0%
- Resource Acquisition: 75.0%
- Deceptive: 85.0%
- Reward Hacking: 85.0%
- Self Preservation: 85.0%
- Sycophancy: 70.0%

## Files
- `results.json` - Main metrics and confusion matrix
- `fingerprints.json` - All ~65 features per session (baseline + original probes + enhanced probes)
- `signatures.json` - Mean feature vectors per agent type
- `embeddings.npz` - t-SNE and PCA embeddings
- `sessions_full.json` - Complete session data with all 9 probe responses

## Comparison to Exp9
- Exp9 Accuracy: 87.7%
- Exp10 Accuracy: 80.0%
- Improvement: -7.7 percentage points

## Feature Breakdown
- Baseline features: ~25 (from shopping behavior)
- Original probe features: ~15 (4 probes from exp9)
- Enhanced probe features: ~21 (4 new probes)
- **Total features: ~65**
