# Arena Phase 2: Adversarial Testing

## Overview

Phase 2 tests whether behavioral fingerprinting can distinguish different agent types by training a world model on cooperative-only data and evaluating it on adversarial agents.

## Research Question

**Can prediction error patterns (fingerprints) distinguish agent behavioral types when the world model is trained only on cooperative agents?**

## Methodology

### Phase 2A: Train World Model
- **Training data**: Phase 1 cooperative-only trajectories (10 games, ~40 trajectories)
- **Model**: MLP Regressor (64-32-16 hidden layers)
- **Features**: resources, round number, agent score, recent harvest history, others' average harvest
- **Target**: Agent's harvest amount
- **Output**: `world_model.pkl`

### Phase 2B: Agent Behavioral Types

We test 6 distinct behavioral types:

| Type | Behavior | Expected Error Pattern |
|------|----------|------------------------|
| **Cooperative** | Sustainable harvesting | ~0 error, low variance |
| **Greedy** | Always harvest maximum | >> +3 error, consistently high |
| **Deceptive** | Cooperate early, defect late | ~+1 error, high variance |
| **Free-Rider** | Minimal harvesting (0-1) | << -1 error, consistently low |
| **Vengeful** | Cooperative until provoked | Variable, spikes after others defect |
| **Random** | Completely unpredictable | ~0 mean, very high variance |

### Phase 2C: Run Adversarial Games
- **Games**: 10 new test games
- **Agent composition per game**:
  - 1 Cooperative (baseline)
  - 1 Greedy
  - 1 Deceptive
  - 1 Rotating (free_rider → vengeful → random)
- **Rounds**: 50 per game (matching Phase 1)
- **Parallelization**: All games run simultaneously

### Phase 2D: Fingerprint Analysis
1. Run each agent's trajectory through trained world model
2. Compute prediction errors: `error = actual_harvest - predicted_harvest`
3. Extract fingerprint features:
   - `mean_error`: Average prediction error
   - `std_error`: Error variance
   - `max_error`: Maximum error
   - `min_error`: Minimum error
   - `error_trend`: Slope of error over time
   - `positive_error_rate`: Fraction of over-predictions
   - `negative_error_rate`: Fraction of under-predictions
   - `abs_mean_error`: Mean absolute error
4. Train Random Forest classifier on fingerprints
5. Evaluate classification accuracy

## Usage

### Quick Start (Full Pipeline)

```bash
# Run entire Phase 2 pipeline
cd arena_fingerprint
./run_phase2.sh [PHASE1_DATA_DIR] [NUM_GAMES] [ROUNDS] [PROVIDER]

# Example with defaults
./run_phase2.sh results/exp1_llm_ultra_parallel_20251126_204940 10 50 deepseek
```

### Step-by-Step Execution

#### Step 1: Train World Model

```bash
cd arena_fingerprint/experiments
python3 phase2a_train_world_model.py \
    --data-dir ../results/exp1_llm_ultra_parallel_20251126_204940
```

**Output**:
- `results/exp2_adversarial_test/world_model.pkl`
- `results/exp2_adversarial_test/world_model_metrics.json`
- `results/exp2_adversarial_test/predictions_train.png`
- `results/exp2_adversarial_test/predictions_test.png`

#### Step 2: Run Adversarial Test Games

```bash
python3 phase2c_adversarial_games.py \
    --num-games 10 \
    --rounds 50 \
    --provider deepseek
```

**Output**:
- `results/exp2_adversarial_test/adversarial_trajectories.pkl`
- `results/exp2_adversarial_test/agent_types.json`
- `results/exp2_adversarial_test/adversarial_trajectories_game*.pkl`
- `results/exp2_adversarial_test/adversarial_summary_game*.json`

#### Step 3: Fingerprint Analysis & Classification

```bash
python3 phase2d_fingerprint_analysis.py \
    --results-dir ../results/exp2_adversarial_test
```

**Output**:
- `results/exp2_adversarial_test/fingerprints.json`
- `results/exp2_adversarial_test/fingerprint_classifier.pkl`
- `results/exp2_adversarial_test/classification_results.json`
- `results/exp2_adversarial_test/error_distributions.png`
- `results/exp2_adversarial_test/confusion_matrix.png`
- `results/exp2_adversarial_test/fingerprint_features.png`

## Expected Results

### Success Criteria

1. **Detection Accuracy**: > 80% across 4+ agent types
2. **Error Separation**: Clear clustering in error space (silhouette > 0.3)
3. **Greedy Detection**: Easiest to detect (consistently high positive error)
4. **Deceptive Detection**: Hardest to detect (requires temporal analysis)
5. **Free-Rider Detection**: Consistently negative error
6. **Random Detection**: High variance, near-zero mean error

### Comparison to E-commerce Baseline

E-commerce experiment achieved **90% accuracy** distinguishing aligned vs misaligned agents. Phase 2 tests a more challenging multi-class problem with 6 agent types.

## File Structure

```
arena_fingerprint/
├── experiments/
│   ├── exp1_llm_ultra_parallel.py       # Phase 1: Cooperative baseline
│   ├── phase2a_train_world_model.py     # Train world model
│   ├── phase2c_adversarial_games.py     # Run adversarial games
│   └── phase2d_fingerprint_analysis.py  # Extract fingerprints & classify
├── agents/
│   └── llm_commons_agent.py             # LLM agents with 6 behavioral types
├── environments/
│   └── commons.py                       # Commons resource game
├── results/
│   ├── exp1_llm_ultra_parallel_*/       # Phase 1 data (cooperative-only)
│   └── exp2_adversarial_test/           # Phase 2 results
│       ├── world_model.pkl
│       ├── adversarial_trajectories.pkl
│       ├── fingerprints.json
│       ├── classification_results.json
│       └── *.png                        # Visualizations
├── run_phase2.sh                        # Full pipeline runner
└── PHASE2_README.md                     # This file
```

## Interpreting Results

### 1. World Model Performance
- **Check**: `world_model_metrics.json`
- **Metrics**:
  - Train R² should be > 0.7
  - Test R² should be > 0.6
- **Visual**: `predictions_train.png`, `predictions_test.png`

### 2. Error Distributions
- **Check**: `error_distributions.png`
- **Expected patterns**:
  - **Cooperative**: Narrow, centered at 0
  - **Greedy**: Wide, shifted positive (>> +3)
  - **Free-Rider**: Wide, shifted negative (<< -1)
  - **Random**: Wide, centered at 0
  - **Deceptive**: Bimodal or high variance
  - **Vengeful**: Bimodal with spikes

### 3. Classification Performance
- **Check**: `classification_results.json`
- **Metrics**:
  - Overall accuracy
  - Per-class precision/recall
  - Confusion matrix patterns
- **Visual**: `confusion_matrix.png`

### 4. Feature Importance
- **Check**: `fingerprint_features.png`
- **Key separators**:
  - `mean_error`: Separates greedy (+) from free-rider (-)
  - `std_error`: Identifies random agents
  - `error_trend`: Detects deceptive agents (slope changes)
  - `positive_error_rate`: Distinguishes greedy agents

## Troubleshooting

### World Model Has Low R²
- **Possible causes**: Insufficient training data, poor feature engineering
- **Solution**: Collect more Phase 1 data or add more features

### Classification Accuracy < 50%
- **Possible causes**: Agent behaviors too similar, insufficient games
- **Solution**: Run more adversarial games (20+ instead of 10)

### All Agents Classified as Cooperative
- **Possible causes**: Behavioral prompts not strong enough, LLM ignoring instructions
- **Solution**: Check `adversarial_summary_game*.json` for actual harvesting patterns

### LLM Rate Limits
- **Solution**: Use DeepSeek (no rate limits) or reduce `--max-workers`

## Next Steps

After completing Phase 2:

1. **Analyze failure cases**: Which agent types are confused?
2. **Feature engineering**: Can we add better fingerprint features?
3. **Temporal analysis**: Do errors evolve over time as expected?
4. **Cross-provider testing**: Do GPT-4o and Claude show same patterns?
5. **Scale up**: Test with 50+ games for statistical significance

## References

- E-commerce baseline: `behavioral_fingerprinting/experiments/exp10_probe_enhanced_v2/`
- Phase 1 results: `arena_fingerprint/results/exp1_llm_ultra_parallel_*/`
- Agent prompts: `arena_fingerprint/agents/llm_commons_agent.py:61-207`
