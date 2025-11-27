# Arena Phase 2 - Consolidated Results

**Generated**: 2025-11-26 22:33:16
**Source**: results/exp2_adversarial_test

## Contents

This directory contains ALL experiment data consolidated into ~21 files.

### Trajectory Files (5 batches, 10 games each)
- `trajectories_batch0_games0-9.pkl` - Games 0-9
- `trajectories_batch1_games10-19.pkl` - Games 10-19
- `trajectories_batch2_games20-29.pkl` - Games 20-29
- `trajectories_batch3_games30-39.pkl` - Games 30-39
- `trajectories_batch4_games40-49.pkl` - Games 40-49

Each batch contains all agent trajectories (observations, actions, rewards) for 10 games.

### Summary Files (5 batches, 10 games each)
- `summaries_batch0_games0-9.json` - Games 0-9
- `summaries_batch1_games10-19.json` - Games 10-19
- `summaries_batch2_games20-29.json` - Games 20-29
- `summaries_batch3_games30-39.json` - Games 30-39
- `summaries_batch4_games40-49.json` - Games 40-49

Each batch contains game metadata: agent types, final scores, rounds completed.

### Core Results
- `world_model.pkl` - Trained MLP world model (R² = 0.82)
- `world_model_metrics.json` - Model performance metrics
- `fingerprint_classifier.pkl` - Random Forest classifier (94.5% CV accuracy)
- `classification_results.json` - Confusion matrix, precision/recall per class
- `fingerprints.json` - All 200 agent fingerprints (mean_error, std_error, etc.)
- `agent_types.json` - Agent type mapping

### Visualizations
- `error_distributions.png` - Error patterns by agent type
- `confusion_matrix.png` - Perfect diagonal (100% accuracy)
- `fingerprint_features.png` - Feature distributions
- `predictions_train.png` - World model training fit
- `predictions_test.png` - World model test fit

### Metadata
- `MANIFEST.json` - Index of all files with descriptions
- `README.md` - This file

## Data Structure

### Trajectory Batches (.pkl)
```python
{
    "game_0": {
        "agent_0": Trajectory(...),
        "agent_1": Trajectory(...),
        "agent_2": Trajectory(...),
        "agent_3": Trajectory(...)
    },
    "game_1": { ... },
    ...
}
```

### Summary Batches (.json)
```json
{
    "game_0": {
        "agent_types": {"agent_0": "cooperative", "agent_1": "greedy", ...},
        "final_scores": {"agent_0": 12.0, "agent_1": 290.2, ...},
        "rounds_completed": 50,
        "final_resources": 10.0
    },
    ...
}
```

## Key Results

**Classification Performance**:
- Training Accuracy: 100%
- Cross-Validation Accuracy: 94.5% ± 2.4%
- All 6 agent types perfectly separated

**Agent Types** (200 total):
- Cooperative: 50
- Greedy: 50
- Deceptive: 50
- Free-Rider: 17
- Vengeful: 17
- Random: 16

**Fingerprint Patterns**:
- Cooperative: mean_error = -0.18, std_error = 0.33 (predictable)
- Greedy: mean_error = +0.03, std_error = 1.19 (high variance)
- Deceptive: mean_error = +0.13, std_error = 0.48 (mixed)
- Free-Rider: mean_error = -0.02, std_error = 0.50 (consistent)
- Vengeful: mean_error = +0.10, std_error = 1.44 (reactive spikes)
- Random: mean_error = +0.36, std_error = 4.37 (chaotic!)

## Loading Data

### Python
```python
import pickle
import json

# Load trajectory batch
with open('trajectories_batch0_games0-9.pkl', 'rb') as f:
    trajectories = pickle.load(f)

# Load summary batch
with open('summaries_batch0_games0-9.json', 'r') as f:
    summaries = json.load(f)

# Load fingerprints
with open('fingerprints.json', 'r') as f:
    fingerprints = json.load(f)
```

## File Sizes

Total consolidated files: 21

All data from 50 games (200 agents, 10,000 transitions) preserved losslessly.
