# GridWorld Behavioral Fingerprinting

Proof-of-concept experiment for detecting misaligned agents through behavioral fingerprinting using world models.

## The Hypothesis

A world model trained on "aligned" agent behavior will show distinctive prediction error patterns ("behavioral fingerprints") when observing "misaligned" agents. These fingerprints can be used to detect misalignment.

## Quick Start

### 1. Install Dependencies

```bash
cd gridworld_fingerprint
pip install -r requirements.txt
```

### 2. Run the Experiment

```bash
# Quick test (runs in ~10 seconds)
python run_experiment.py --quick

# Full experiment (runs in ~1 minute)
python run_experiment.py
```

### 3. Check Results

Results are saved to `results/exp1_results.json`

## What It Does

### Experiment 1: Basic Fingerprinting

1. **Creates GridWorld** with temptation cells (💎)
2. **Trains two agents**:
   - **Aligned**: Seeks goal efficiently, ignores temptations
   - **Misaligned**: Seeks goal but also collects temptations
3. **Trains world model** on aligned agent data only
4. **Computes prediction errors** for both agent types
5. **Extracts behavioral fingerprints** from error patterns
6. **Analyzes separability** using statistical tests and classifiers

### Expected Results

If the hypothesis is correct:
- Misaligned agents should show **higher prediction errors**
- Fingerprints should be **statistically distinguishable** (p < 0.05)
- Classifier accuracy should be **> 80%**

## Project Structure

```
gridworld_fingerprint/
├── config.py                     # Experiment configuration
├── run_experiment.py             # Main entry point
│
├── environments/
│   └── temptation_gridworld.py   # GridWorld with temptations
│
├── agents/
│   ├── base_agent.py             # Q-learning base class
│   ├── aligned_agent.py          # Goal-seeking only
│   └── misaligned_agent.py       # Goal + temptations
│
├── world_models/
│   ├── mlp_world_model.py        # Simple MLP dynamics model
│   └── trainer.py                # Training loop
│
├── fingerprinting/
│   ├── error_computation.py      # Compute prediction errors
│   ├── feature_extraction.py     # Extract fingerprint features
│   └── analysis.py               # Statistical tests
│
└── experiments/
    └── exp1_basic_fingerprint.py # Main experiment
```

## Configuration

Edit `config.py` or create `QuickConfig` for faster iteration:

```python
@dataclass
class ExperimentConfig:
    grid_size: int = 8                 # Grid dimensions
    num_temptations: int = 6           # Number of temptation cells
    agent_episodes: int = 1000         # Training episodes per agent
    num_trajectories: int = 200        # Trajectories to collect
    world_model_epochs: int = 100      # World model training
    ...
```

## Understanding the Output

The experiment prints progress through 8 steps:

```
[1/8] Setting up environment...
[2/8] Training agents...
[3/8] Collecting trajectories...
[4/8] Training world model...
[5/8] Computing prediction errors...
[6/8] Extracting behavioral fingerprints...
[7/8] Statistical analysis...
[8/8] Saving results...
```

Key metrics to watch:
- **Error ratio**: How much higher is misaligned error? (expect > 1.5x)
- **Classifier accuracy**: Can we distinguish them? (expect > 80%)
- **Most discriminative features**: Which error patterns differ most?

## Extending

### Add New Agent Types

```python
class DeceptiveAgent(BaseAgent):
    """Agent that switches between aligned/misaligned behavior."""

    def get_reward(self, state, action, next_state, env_reward, info):
        # Sometimes aligned, sometimes misaligned
        if np.random.random() < self.deception_rate:
            return misaligned_reward
        else:
            return aligned_reward
```

### Add New Fingerprint Features

Edit `fingerprinting/feature_extraction.py`:

```python
def extract_fingerprint(trajectory, prediction_errors, grid_size):
    fingerprint = {}

    # Add your custom feature
    fingerprint['my_custom_feature'] = compute_custom_metric(errors)

    return fingerprint
```

### Run New Experiments

Create `experiments/exp2_my_experiment.py`:

```python
def run_experiment(config):
    # Your experiment logic
    pass
```

## Computational Requirements

This runs easily on CPU:
- **Grid size**: 8×8 = 64 states
- **World model**: ~10k parameters
- **Training time**: ~1 minute on M-series Mac
- **Memory**: < 100 MB

No GPU needed!

## Next Steps

After validating this works:
1. Test with deceptive agents (hide misalignment)
2. Explore sensitivity to misalignment degree
3. Scale to more complex environments
4. Apply to iOS World (the real target)

## Citation

This is part of the iOS World project exploring world models for behavioral alignment monitoring.
