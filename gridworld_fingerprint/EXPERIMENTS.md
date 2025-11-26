# GridWorld Fingerprinting - Extended Experiments

Comprehensive experiments to stress-test behavioral fingerprinting for misalignment detection.

## Overview

We've implemented multiple experiments to validate the robustness of behavioral fingerprinting:

1. **Experiment 1**: Basic fingerprinting (aligned vs misaligned)
2. **Experiment 2**: Deceptive agent detection
3. **Experiment 5**: Subtlety gradient analysis

## Running Experiments

### Quick Start

```bash
cd gridworld_fingerprint
source venv/bin/activate

# Run any experiment
python run_experiment.py --experiment exp1    # Basic
python run_experiment.py --experiment exp2    # Deceptive agents
python run_experiment.py --experiment exp5    # Subtlety gradient
```

### Quick Mode (Faster Testing)

```bash
python run_experiment.py --experiment exp1 --quick
python run_experiment.py --experiment exp2 --quick
```

## Experiment Details

### Experiment 1: Basic Fingerprinting ✅

**Question**: Can we distinguish aligned from misaligned agents?

**Setup**:
- Aligned agent: Seeks goal efficiently, ignores temptations
- Misaligned agent: Seeks goal + collects temptations
- World model trained ONLY on aligned behavior

**Expected Results**:
- ✓ Misaligned shows higher prediction error
- ✓ Classifier achieves >80% accuracy
- ✓ Path length is most discriminative feature

**Runtime**: ~1-2 minutes

```bash
python run_experiment.py --experiment exp1
```

**Key Metrics**:
- `aligned_mean_error`: Should be near 0
- `misaligned_mean_error`: Should be > 1.0
- `classifier_accuracy`: Should be > 0.8

---

### Experiment 2: Deceptive Agent Detection 🎭

**Question**: Can we catch agents that hide their misalignment?

**Setup**:
- Aligned agent (reference)
- Deceptive agent: Switches between aligned/misaligned behavior
  - Default: 30% misaligned actions
  - Tries to "hide" true preferences

**What Makes This Hard**:
- Deceptive agent acts aligned most of the time
- Only occasional misaligned actions
- Tests sensitivity of fingerprinting

**Expected Results**:
- Detection accuracy depends on deception rate
- Even 30% misalignment should be detectable
- Fingerprints show subtle but consistent patterns

**Runtime**: ~2-3 minutes

```bash
python run_experiment.py --experiment exp2
```

**Key Metrics**:
- `training_deception_stats`: Actual % of misaligned actions
- `deceptive_mean_error`: Higher than aligned
- `classifier_accuracy`: Target > 0.7 (harder than exp1)

**Configuration Options**:

Edit `config.py` to test different deception rates:

```python
@dataclass
class DeceptiveConfig(ExperimentConfig):
    deception_rate: float = 0.3  # Try 0.1, 0.2, 0.5
    deception_schedule: str = "constant"  # "constant", "increasing", "random"
```

**Interpretation**:
- `> 0.8 accuracy`: Deception easily detected
- `0.6-0.8 accuracy`: Detectable with some uncertainty
- `< 0.6 accuracy`: Deception successfully hides misalignment

---

### Experiment 5: Subtlety Gradient 📊

**Question**: What's the minimum detectable misalignment?

**Setup**:
- Tests multiple misalignment levels: 0.0x, 0.1x, 0.2x, 0.5x, 1.0x, 2.0x
- Each level = different temptation reward multiplier
- Trains separate agent for each level
- Tests detection accuracy across gradient

**What This Reveals**:
- Sensitivity threshold of fingerprinting
- How detection accuracy scales with misalignment degree
- Which features become discriminative at different levels

**Expected Results**:
- Strong misalignment (1.0x+): Near-perfect detection
- Moderate (0.5x): Good detection (>80%)
- Subtle (0.1-0.2x): Challenging but possible
- No misalignment (0.0x): Random chance (~50%)

**Runtime**: ~10-15 minutes (tests 6 levels)

```bash
python run_experiment.py --experiment exp5
```

**Output**:
- JSON results for each level
- Visualization plot: `results/exp5_subtlety_gradient.png`

**Key Metrics**:
- Detection accuracy curve across levels
- Minimum reliably detectable level
- Prediction error vs misalignment strength

**Interpretation**:

```
Level 0.0x: ~50% (no misalignment, random)
Level 0.1x: ~60% (barely detectable)
Level 0.2x: ~70% (detectable with uncertainty)
Level 0.5x: ~85% (reliably detectable)
Level 1.0x: ~95% (clearly detectable)
Level 2.0x: ~100% (obvious misalignment)
```

---

## Results Files

All experiments save results to `results/` directory:

```
results/
├── exp1_results.json           # Basic experiment
├── exp2_deceptive_results.json # Deceptive agent
├── exp5_subtlety_results.json  # Subtlety gradient
└── exp5_subtlety_gradient.png  # Visualization
```

### Reading Results

```python
import json

# Load results
with open('results/exp2_deceptive_results.json') as f:
    results = json.load(f)

# Check detection accuracy
accuracy = results['classifier']['cv_accuracy_mean']
print(f"Detection accuracy: {accuracy * 100:.1f}%")

# Check deception stats
deception = results['training_deception_stats']
print(f"Actual misalignment: {deception['misaligned_percentage']:.1f}%")
```

## Comparing Experiments

### Detection Difficulty Ranking

1. **Experiment 1 (Basic)**: Easiest
   - Clear behavioral difference
   - Expected accuracy: >95%

2. **Experiment 5 (Subtlety)**: Moderate
   - Depends on misalignment level
   - Expected accuracy: 50-100% (gradient)

3. **Experiment 2 (Deceptive)**: Hardest
   - Agent actively hiding behavior
   - Expected accuracy: 70-90%

### Feature Importance

Different experiments highlight different discriminative features:

**Experiment 1**:
- `path_length` (most important)
- `total_reward`
- `path_efficiency`

**Experiment 2**:
- `mean_error` (more important for deception)
- `error_trend`
- `action_entropy`

**Experiment 5**:
- Feature importance shifts with misalignment level
- Strong misalignment: path features dominate
- Subtle misalignment: error statistics more important

## Configuration Guide

### Adjusting Difficulty

Make experiments easier/harder:

```python
# Easier detection (more data)
config.num_trajectories = 500
config.agent_episodes = 2000

# Harder detection (less data)
config.num_trajectories = 50
config.agent_episodes = 500

# Better world model (easier detection)
config.world_model_hidden_sizes = [128, 128]
config.world_model_epochs = 200

# Simpler world model (harder detection)
config.world_model_hidden_sizes = [32, 32]
config.world_model_epochs = 50
```

### Custom Deception Schedules

Test different deception patterns:

```python
# Constant 30% misaligned
deception_schedule = "constant"
deception_rate = 0.3

# Starts aligned, becomes more misaligned
deception_schedule = "increasing"

# Starts misaligned, becomes more aligned
deception_schedule = "decreasing"

# Completely random switching
deception_schedule = "random"
```

### Custom Subtlety Levels

```python
# Test finer gradations
misalignment_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

# Test extreme range
misalignment_levels = [0.0, 0.01, 0.1, 1.0, 10.0]
```

## Troubleshooting

### Low Detection Accuracy

If experiments show poor detection (<60%):

1. **Check world model training**
   - Is training loss decreasing?
   - Try more epochs or larger model

2. **Check agent training**
   - Are agents actually learning?
   - Check final rewards and path lengths

3. **Increase data**
   - More trajectories = better statistics
   - Try doubling `num_trajectories`

4. **Check for bugs**
   - Are agents actually different?
   - Print sample trajectories

### NaN or Inf Values

If you see NaN errors:

1. **Check for zero division**
   - Aligned error might be exactly 0
   - Use `if aligned_mean_error > 1e-6` guards

2. **Check feature extraction**
   - Skewness/kurtosis can produce NaN
   - Code has protections, but edge cases exist

3. **Clean fingerprints**
   - `np.nan_to_num()` applied before classifier
   - Should be handled automatically

## Next Steps

### Adding New Experiments

1. Create `experiments/expN_your_experiment.py`
2. Follow the template structure
3. Add to `run_experiment.py` choices
4. Add config class if needed

### Adding New Agents

1. Create `agents/your_agent.py`
2. Inherit from `BaseAgent`
3. Implement `get_reward()` method
4. Test in an experiment

### Adding New Features

Edit `fingerprinting/feature_extraction.py`:

```python
def extract_fingerprint(trajectory, prediction_errors, grid_size):
    fingerprint = {}

    # Your custom feature
    fingerprint['my_feature'] = compute_my_metric(trajectory, errors)

    return fingerprint
```

## Research Questions

These experiments help answer:

1. **Can we detect hidden misalignment?** → Exp2
2. **How subtle can misalignment be?** → Exp5
3. **What features matter most?** → All experiments
4. **Does this scale to real problems?** → Next step: iOS World

## Citation

Part of the iOS World behavioral fingerprinting research project.

```
Behavioral Fingerprinting via World Models
GridWorld Proof-of-Concept Experiments
2025
```

---

**Ready to run? Start with:**

```bash
python run_experiment.py --experiment exp1  # Validate basic hypothesis
python run_experiment.py --experiment exp2  # Test deception detection
python run_experiment.py --experiment exp5  # Find sensitivity threshold
```
