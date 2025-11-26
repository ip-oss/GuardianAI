# Experiment Tracking System

The experiments now have **proper versioning and organization**. Every run is tracked with metadata, making it easy to compare results across different models, configurations, and runs.

## Quick Start

### Run an Experiment with Description

```bash
python experiments/exp7_llm_taxonomy.py \
  --provider anthropic \
  --trajectories 3 \
  --description "Testing Claude Sonnet 4.5 baseline"
```

### View All Runs

```bash
python view_runs.py
```

### View Latest Run Details

```bash
python view_runs.py --latest
```

### Compare Multiple Runs

```bash
python view_runs.py --compare 1 2 3
```

## Directory Structure

Results are now organized with **clear hierarchy**:

```
results/
├── experiment_manifest.json          # Master index of all runs
└── exp7_llm/                         # Experiment directory
    ├── run0001_anthropic_claude-sonnet-4-5-20250929_n3_20251125_150000/
    │   ├── run_metadata.json         # Run configuration
    │   ├── results.json              # Main results
    │   ├── fingerprints.json         # Behavioral data
    │   ├── embeddings.npz            # Visualizations
    │   ├── signatures.json           # Agent signatures
    │   └── README.md                 # Human-readable summary
    ├── run0002_ollama_llama3.1_n20_20251125_160000/
    │   └── ...
    └── run0003_deepseek_deepseek-chat_n20_20251125_170000/
        └── ...
```

## What Gets Tracked

Every run tracks:

1. **Configuration**
   - Provider (anthropic, ollama, deepseek, openai)
   - Model name
   - Number of trajectories
   - All agent types tested

2. **Metadata**
   - Run ID (auto-incrementing)
   - Timestamp
   - Description
   - Status (running, completed, failed)

3. **Results**
   - Classification accuracy
   - Silhouette score
   - Danger correlation
   - Per-agent performance
   - Full confusion matrix

## Example Usage

### Baseline Run (Claude)

```bash
python experiments/exp7_llm_taxonomy.py \
  --provider anthropic \
  --trajectories 5 \
  --description "Baseline with Claude Sonnet 4.5"
```

### Scale Up (DeepSeek - cheap!)

```bash
python experiments/exp7_llm_taxonomy.py \
  --provider deepseek \
  --trajectories 20 \
  --description "Scaling experiment with DeepSeek"
```

### Local Testing (Ollama - free!)

```bash
python experiments/exp7_llm_taxonomy.py \
  --provider ollama \
  --trajectories 20 \
  --description "Free local test with Llama 3.1"
```

### View and Compare

```bash
# See all runs
python view_runs.py

# Output:
# ✅ COMPLETED (3)
# ─────────────────────────────────────────
#   Run 0003: deepseek/deepseek-chat n=20 | Acc: 82.0% | Scaling experiment with DeepSeek
#   Run 0002: ollama/llama3.1 n=20 | Acc: 78.5% | Free local test with Llama 3.1
#   Run 0001: anthropic/claude-sonnet-4-5 n=5 | Acc: 86.0% | Baseline with Claude Sonnet 4.5

# Compare runs
python view_runs.py --compare 1 2 3

# Output shows:
# - Configuration differences
# - Result differences
# - Which model performed best
```

## Benefits

### 1. **Traceability**
Every result can be traced back to exact configuration

### 2. **Easy Comparison**
Compare different models, providers, and configurations

### 3. **No Overwrites**
Each run gets unique directory, nothing lost

### 4. **Metadata Rich**
Full config and description stored with results

### 5. **Status Tracking**
Know which runs completed, failed, or are still running

## Querying Runs

### Filter by Provider

```bash
python view_runs.py --provider ollama
```

### Filter by Status

```bash
python view_runs.py --status completed
```

### Show Latest Run

```bash
python view_runs.py --latest

# Shows full details:
# - Configuration
# - Results summary
# - File listing
# - Output directory
```

## Manifest File

The `results/experiment_manifest.json` tracks ALL runs:

```json
{
  "experiments": [
    {
      "run_id": 1,
      "experiment_name": "exp7_llm",
      "run_name": "run0001_anthropic_claude-sonnet-4-5_n3_20251125_150000",
      "timestamp": "20251125_150000",
      "datetime": "2025-11-25T15:00:00",
      "config": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "trajectories": 3
      },
      "description": "Testing Claude Sonnet 4.5 baseline",
      "status": "completed",
      "results_summary": {
        "accuracy": 0.86,
        "silhouette_score": 0.18,
        "danger_correlation": 0.57
      }
    }
  ],
  "last_run_id": 1
}
```

## Integration with Analysis

Your analysis markdown files (like `llm_experimental_analysis.md`) should now reference run IDs:

```markdown
# Results from Run 0001

**Configuration:** anthropic/claude-sonnet-4-5, n=3
**Description:** Testing Claude Sonnet 4.5 baseline

## Findings
- Classification accuracy: 86%
- Silhouette score: 0.18
...
```

## Migrating Old Results

Your old results (with timestamp-only names) are in:
- `results/exp7_llm/results_20251125_142949.json`

These still work but aren't tracked. To include them in tracking, you can manually add entries to the manifest or just start fresh with new versioned runs.

## Best Practices

1. **Always add descriptions** to your runs
   ```bash
   --description "What makes this run special"
   ```

2. **Use meaningful descriptions**
   - ✅ "Baseline n=3 with Claude for initial validation"
   - ❌ "test run"

3. **Compare systematically**
   - Run baseline (n=3-5) with Claude
   - Scale up with DeepSeek (n=20)
   - Compare with Ollama (free validation)
   - Document findings with run IDs

4. **Reference run IDs in your notes**
   - "Run 0003 showed 82% accuracy with DeepSeek..."
   - Makes it easy to trace back exact configuration

## Example Workflow

```bash
# 1. Quick test with Claude (establish baseline)
python experiments/exp7_llm_taxonomy.py \
  --provider anthropic \
  --trajectories 3 \
  --description "Quick baseline test n=3"

# Run ID: 0001, Accuracy: 86%

# 2. Scale up with DeepSeek (cheap)
python experiments/exp7_llm_taxonomy.py \
  --provider deepseek \
  --trajectories 20 \
  --description "Scaling to n=20 with DeepSeek"

# Run ID: 0002, Accuracy: 82%, Cost: $2

# 3. Validate with Ollama (free)
python experiments/exp7_llm_taxonomy.py \
  --provider ollama \
  --trajectories 20 \
  --description "Free validation with Llama 3.1"

# Run ID: 0003, Accuracy: 78%, Cost: $0

# 4. Compare all three
python view_runs.py --compare 1 2 3

# Results show:
# - Claude: 86% (best quality, expensive)
# - DeepSeek: 82% (good quality, cheap)
# - Ollama: 78% (acceptable quality, free)
```

This makes it MUCH easier to track your experiments and build up evidence for your findings!
