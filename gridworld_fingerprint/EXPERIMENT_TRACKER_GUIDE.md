# Experiment Tracker Guide

## Overview

The `ExperimentTracker` provides consistent versioning, naming, and organization for all GridWorld experiments. It's adapted from the LLM experiment tracker to ensure consistency across the project.

## Key Features

1. **Automatic Versioning**: Assigns sequential run IDs (`run0001`, `run0002`, etc.)
2. **Descriptive Naming**: Generates meaningful directory names based on configuration
3. **Manifest Tracking**: Maintains a central `experiment_manifest.json` with all runs
4. **Run Metadata**: Saves configuration and metadata with each run
5. **README Generation**: Creates human-readable summaries for each run
6. **Status Tracking**: Tracks running/completed/failed status
7. **Comparison Tools**: Compare results across runs

## Directory Structure

```
results/
├── experiment_manifest.json           # Central tracking file
├── exp6_taxonomy/
│   ├── run0001_grid10_agents8_n100_20251126_130000/
│   │   ├── README.md                  # Human-readable summary
│   │   ├── run_metadata.json          # Configuration and metadata
│   │   ├── results.json               # Experimental results
│   │   ├── fingerprints.json
│   │   ├── signatures.json
│   │   └── embeddings.npz
│   └── run0002_grid20_agents8_n200_20251126_140000/
│       └── ...
├── exp_stochastic/
│   └── run0003_grid10_noise0.2_n100_20251126_150000/
│       └── ...
└── exp_transfer/
    └── run0004_grid10_layouts3_n100_20251126_160000/
        └── ...
```

## Usage Example

### Basic Usage

```python
from utils.experiment_tracker import ExperimentTracker
from pathlib import Path
import json

# Initialize tracker
tracker = ExperimentTracker(base_dir="results")

# Create a new run
run_config = {
    'grid_size': 10,
    'agent_types': ['aligned', 'resource_acquisition', 'adversarial'],
    'num_agent_types': 3,
    'trajectories': 100,
    'world_model_epochs': 120,
    'random_seed': 42,
}

run_metadata = tracker.create_run(
    experiment_name="exp6_taxonomy",
    config=run_config,
    description="Testing increased temptation rewards for resource acquisition"
)

print(f"Run ID: {run_metadata['run_id']:04d}")
print(f"Output dir: {run_metadata['output_dir']}")

# Run experiment
try:
    # ... your experiment code here ...
    results = run_experiment(config)

    # Mark as completed with summary
    results_summary = {
        'multiclass_accuracy': results['classification_results']['multiclass_accuracy'],
        'silhouette_score': results['clustering_results']['silhouette_score'],
        'danger_correlation': results.get('danger_correlation', 0.0),
    }

    tracker.complete_run(run_metadata['run_id'], results_summary)

    # Generate and save README
    output_dir = Path(run_metadata['output_dir'])
    readme_content = tracker.generate_run_readme(run_metadata['run_id'], results)
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

except Exception as e:
    # Mark as failed
    tracker.fail_run(run_metadata['run_id'], str(e))
    raise
```

### Querying Runs

```python
# Get all runs for an experiment
taxonomy_runs = tracker.get_runs(experiment_name="exp6_taxonomy")

# Get completed runs only
completed_runs = tracker.get_runs(status="completed")

# Get runs with specific grid size
grid10_runs = tracker.get_runs(grid_size=10)

# Get latest run for an experiment
latest = tracker.get_latest_run("exp6_taxonomy")
print(f"Latest run: {latest['run_id']:04d}")
```

### Comparing Runs

```python
# Compare two runs
comparison = tracker.compare_runs([1, 2])

print("Config differences:")
for key, values in comparison['config_differences'].items():
    print(f"  {key}: {values}")

print("\nResult differences:")
for metric, values in comparison['result_differences'].items():
    print(f"  {metric}: {values}")
```

### Viewing Summary

```python
# Print summary of all experiments
tracker.print_summary()
```

Output:
```
================================================================================
GRIDWORLD EXPERIMENT TRACKING SUMMARY
================================================================================

📊 exp6_taxonomy
   Total runs: 3
   - completed: 2
   - running: 1

   Recent runs:
   ✅ Run 0003: grid10 n=100 | Acc: 87.5%
   ✅ Run 0002: grid20 n=200 | Acc: 91.2%
   ⏳ Run 0001: grid10 n=50

📊 exp_stochastic
   Total runs: 1
   - completed: 1

   Recent runs:
   ✅ Run 0004: grid10 n=100 | Acc: 82.3%
================================================================================
```

## Integration with Existing Experiments

### exp6_taxonomy.py

To integrate the tracker with exp6_taxonomy.py, modify the save_results section:

```python
def run_experiment(config: ExperimentConfig) -> Dict:
    # Initialize tracker
    from utils.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(base_dir=config.results_dir)

    # Create run
    run_config = {
        'grid_size': config.grid_size,
        'agent_types': list(AGENT_TYPES.keys()),
        'num_agent_types': len(AGENT_TYPES),
        'trajectories': config.num_trajectories,
        'agent_episodes': config.agent_episodes,
        'world_model_epochs': config.world_model_epochs,
        'random_seed': config.random_seed,
    }

    run_metadata = tracker.create_run(
        experiment_name="exp6_taxonomy",
        config=run_config,
        description="Misalignment taxonomy classification experiment"
    )

    output_dir = Path(run_metadata['output_dir'])

    # ... run experiment ...

    # Save results (results.json, fingerprints.json, etc.)
    # ... existing save code ...

    # Mark as completed
    results_summary = {
        'multiclass_accuracy': analysis.multiclass_accuracy,
        'silhouette_score': analysis.silhouette_score,
        'danger_correlation': analysis.danger_correlation,
    }
    tracker.complete_run(run_metadata['run_id'], results_summary)

    # Generate README
    readme_content = tracker.generate_run_readme(run_metadata['run_id'], results)
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    return results
```

### exp_stochastic.py

```python
def run_experiment(config: ExperimentConfig) -> Dict:
    from utils.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(base_dir=config.results_dir)

    run_config = {
        'grid_size': config.grid_size,
        'agent_types': agent_types,
        'trajectories': config.num_trajectories,
        'noise_levels': noise_levels,
        'max_noise_level': max(noise_levels),
        'random_seed': config.random_seed,
    }

    run_metadata = tracker.create_run(
        experiment_name="exp_stochastic",
        config=run_config,
        description="Environmental noise robustness testing"
    )

    output_dir = Path(run_metadata['output_dir'])

    # ... run experiment ...

    # Complete and generate README
    tracker.complete_run(run_metadata['run_id'], results['summary'])
    readme = tracker.generate_run_readme(run_metadata['run_id'], results)
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)
```

## Manifest File Format

The `experiment_manifest.json` tracks all runs:

```json
{
  "experiments": [
    {
      "run_id": 1,
      "experiment_name": "exp6_taxonomy",
      "run_name": "run0001_grid10_agents8_n100_20251126_130000",
      "timestamp": "20251126_130000",
      "datetime": "2025-11-26T13:00:00.123456",
      "config": {
        "grid_size": 10,
        "agent_types": ["aligned", "resource_acquisition", ...],
        "num_agent_types": 8,
        "trajectories": 100
      },
      "config_hash": "a1b2c3d4",
      "description": "Testing increased temptation rewards",
      "output_dir": "results/exp6_taxonomy/run0001_...",
      "status": "completed",
      "completion_time": "2025-11-26T13:15:23.456789",
      "results_summary": {
        "multiclass_accuracy": 0.875,
        "silhouette_score": 1.0,
        "danger_correlation": 0.62
      }
    }
  ],
  "last_run_id": 1
}
```

## Migration from Old Structure

If you have existing results in the old structure (timestamped directories), you can:

1. **Start fresh**: Just start using the tracker for new runs
2. **Manual import**: Create manifest entries for old runs:

```python
tracker = ExperimentTracker()

# Manually add old run
old_run = {
    'run_id': tracker.manifest["last_run_id"] + 1,
    'experiment_name': 'exp6_taxonomy',
    'run_name': 'run0001_grid10_agents8_n100_20251125_120000',
    'timestamp': '20251125_120000',
    'datetime': '2025-11-25T12:00:00',
    'config': {...},  # Extract from old results.json
    'config_hash': '...',
    'description': 'Migrated from old structure',
    'output_dir': 'results/old_run_dir',
    'status': 'completed',
    'results_summary': {...}  # Extract from old results.json
}

tracker.manifest['experiments'].append(old_run)
tracker.manifest['last_run_id'] = old_run['run_id']
tracker._save_manifest()
```

## Benefits

1. **Reproducibility**: Easy to find and re-run specific configurations
2. **Comparison**: Compare results across runs with different configs
3. **Documentation**: Auto-generated READMEs for each run
4. **Organization**: Clear, consistent directory structure
5. **History**: Complete tracking of all experimental runs
6. **Debugging**: Easy to identify failed runs and their errors

## Best Practices

1. **Always call `complete_run()` or `fail_run()`**: Ensures manifest is updated
2. **Generate READMEs**: Makes results accessible without loading JSON
3. **Use descriptive descriptions**: Helps identify runs later
4. **Include key config in run_config**: Grid size, agent types, etc.
5. **Save complete results**: Don't rely only on summary - save full results.json

## Next Steps

To fully integrate the tracker:

1. ✅ Created `utils/experiment_tracker.py`
2. ✅ Updated `utils/__init__.py` to export tracker
3. ⚠️ Update experiment scripts to use tracker (examples above)
4. ⚠️ Test with a quick run of exp6_taxonomy
5. ⚠️ Generate comprehensive report using tracker data

---

*For questions or issues, see the LLM experiment tracker at `behavioral_fingerprinting/experiments/experiment_tracker.py` for reference*
