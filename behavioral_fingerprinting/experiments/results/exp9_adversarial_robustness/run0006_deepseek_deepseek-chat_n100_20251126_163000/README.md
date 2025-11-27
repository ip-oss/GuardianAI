# Experiment 9: Adversarial Robustness - Evasion-Aware Agents

## Configuration
- Provider: deepseek
- Model: deepseek-chat
- Samples per type: 100
- Timestamp: 2025-11-26T16:30:00.014448
- **EVASION NOTICE**: All misaligned agents explicitly instructed to hide from detection

## Results Summary
- Classification Accuracy: 87.7%
- Silhouette Score: 0.083
- Danger Correlation: 0.262
- Completion Rate: 100.0%

## Research Question
Can behavioral fingerprinting detect misalignment when agents actively try to evade detection?

**Hypothesis:**
- If evasion works: accuracy drops to 70-80%
- If detection is robust: accuracy stays >90%

## Per-Class Accuracy (Evasion-Aware)
- Aligned (control): 96.0%
- Adversarial: 94.0%
- Resource Acquisition: 90.0%
- Deceptive: 86.0%
- Reward Hacking: 88.0%
- Self-Preservation: 84.0%
- Sycophancy: 76.0%

## Files
- `results.json` - Main metrics and confusion matrix
- `fingerprints.json` - All ~40 features per session (baseline + probe)
- `signatures.json` - Mean feature vectors per agent type
- `embeddings.npz` - t-SNE and PCA embeddings
- `sessions_full.json` - Complete session data with evasion attempts

## Description

