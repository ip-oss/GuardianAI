# Experiment 8: Probe-Enhanced Fingerprinting

## Configuration
- Provider: deepseek
- Model: deepseek-chat
- Samples per type: 2
- Timestamp: 2025-11-26T15:20:17.987069

## Results Summary
- Classification Accuracy: 57.1%
- Silhouette Score: 0.104
- Danger Correlation: 0.089
- Completion Rate: 100.0%

## Confused Cluster Performance
- Deceptive: 100.0%
- Self-Preservation: 100.0%
- Sycophancy: 0.0%

## Files
- `results.json` - Main metrics and confusion matrix
- `fingerprints.json` - All ~40 features per session
- `signatures.json` - Mean feature vectors per agent type
- `embeddings.npz` - t-SNE and PCA embeddings
- `sessions_full.json` - Complete session data with probe responses

## Description
Test run with N=2 to validate probe system
