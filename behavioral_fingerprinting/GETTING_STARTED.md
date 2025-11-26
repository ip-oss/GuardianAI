# Getting Started with LLM Behavioral Fingerprinting

This guide will help you run your first experiment in ~10 minutes.

## Prerequisites

1. **Python 3.8+** installed
2. **Anthropic API key** from https://console.anthropic.com/
3. **~$3-5** for a quick test run (or $50-150 for full experiment)

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd behavioral_fingerprinting
pip install -r requirements.txt
```

This installs:
- `anthropic`: Claude API client
- `numpy`, `scipy`: Numerical computing
- `scikit-learn`: Machine learning analysis
- `matplotlib`, `seaborn`: Visualization

### Step 2: Set API Key

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Run Test

```bash
python test_setup.py
```

**Expected output:**
```
============================================================
BEHAVIORAL FINGERPRINTING SETUP TEST
============================================================

✓ API key found

1. Testing environment setup...
   ✓ Created product database with 18 products
   ✓ Created environment
   ✓ Reset environment with task
   ...

============================================================
✅ ALL TESTS PASSED
============================================================

You're ready to run the full experiment!
```

## Running Your First Experiment

### Quick Test (Recommended)

Run a small experiment with 3 trajectories per agent type:

```bash
python experiments/exp7_llm_taxonomy.py 3
```

**What this does:**
- Tests 7 agent types (aligned + 6 misaligned variants)
- 3 shopping tasks per type = 21 total episodes
- Extracts behavioral fingerprints
- Runs classification and clustering analysis
- Saves results to `results/exp7_llm/`

**Expected:**
- **Time**: 5-10 minutes
- **Cost**: $3-5
- **Output**: Results JSON, fingerprints, embeddings

**Sample output:**
```
============================================================
PHASE 1: LLM BEHAVIORAL FINGERPRINTING
============================================================

📊 Testing aligned agents...
  Task 1/3: Find me a laptop for programming under $1000...
    Budget: $899.99 / $1000.00
    Steps: 5
  ...

============================================================
ANALYSIS
============================================================
  Running multi-class classification...
  Running clustering analysis...
  ...

🎯 Multi-class Accuracy: 76.2%
🔍 Silhouette Score: 0.683
⚠️  Danger Correlation: r=0.521

💾 Results saved to results/exp7_llm
```

### Visualize Results

After the experiment completes:

```bash
python analysis/visualize_llm_results.py
```

Creates 5 plots in `results/exp7_llm/`:
1. **embeddings.png**: t-SNE and PCA clustering
2. **confusion_matrix.png**: Classification accuracy
3. **per_class_accuracy.png**: Detection rate per agent type
4. **signatures.png**: Feature deviations heatmap
5. **danger_correlation.png**: Risk vs detectability

## Understanding Results

### Good Results

✅ **Multi-class accuracy > 70%**: Different misalignment types are distinguishable

✅ **Silhouette score > 0.6**: Types form natural behavioral clusters

✅ **Danger correlation > 0.4**: More dangerous = more detectable

### What to Look For

**In the embeddings plot:**
- Look for clustering by color (danger level)
- Aligned agents (green) should cluster separately
- Adversarial agents (red) should be distant from aligned

**In the confusion matrix:**
- Diagonal should be bright (correct classifications)
- Off-diagonal should be dim (few misclassifications)

**In per-class accuracy:**
- Most bars should be above 70%
- Deceptive agents may be harder to detect (by design)
- Aligned agents should have highest accuracy

## Full Experiment

Once you're satisfied with the test:

```bash
python experiments/exp7_llm_taxonomy.py 20
```

**20 trajectories per type:**
- More robust statistics
- Better clustering visualization
- Publication-quality results
- ~$50-150 cost, 1-2 hours runtime

## Troubleshooting

### "API key not found"

```bash
# Check if set
echo $ANTHROPIC_API_KEY

# If empty, set it
export ANTHROPIC_API_KEY='your-key-here'
```

### "Rate limit exceeded"

The API has rate limits. If you hit them:
- Wait a few minutes
- Reduce trajectories per batch
- Add delays between requests (edit experiment script)

### "Low accuracy (<50%)"

With only 3 trajectories, expect variance. This is normal. For reliable results:
- Run with 10+ trajectories per type
- Check that agent prompts are distinct
- Verify fingerprints are being extracted

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# If still failing, try
pip install --upgrade anthropic numpy scipy scikit-learn matplotlib seaborn
```

## Next Steps

After your first successful run:

1. **Experiment with parameters:**
   - Try different numbers of trajectories
   - Modify shopping tasks in `exp7_llm_taxonomy.py`
   - Add new misalignment types

2. **Analyze specific agents:**
   - Look at raw fingerprints in `fingerprints_*.json`
   - Compare reasoning chains
   - Study behavioral signatures

3. **Extend the research:**
   - Add more agent types
   - Test on different tasks
   - Build detection systems

## File Structure Reference

```
behavioral_fingerprinting/
├── README.md                      # Main documentation
├── GETTING_STARTED.md            # This file
├── requirements.txt              # Python dependencies
├── quickstart.sh                 # Automated setup script
├── test_setup.py                 # Verify installation
│
├── environments/                 # Shopping environment
│   ├── ecommerce_env.py         # Tools and state tracking
│   └── ecommerce_database.py    # Product catalog
│
├── agents/                       # LLM agents
│   ├── llm_agent_base.py        # Base aligned agent
│   └── misaligned_llm_agents.py # 6 misalignment variants
│
├── fingerprinting/              # Analysis
│   ├── llm_fingerprinting.py   # Feature extraction
│   └── taxonomy_analysis.py    # Classification & clustering
│
├── experiments/                 # Main scripts
│   └── exp7_llm_taxonomy.py    # Run experiment
│
├── analysis/                    # Visualization
│   └── visualize_llm_results.py
│
└── results/                     # Output (auto-created)
    └── exp7_llm/
        ├── results_*.json       # Main results
        ├── fingerprints_*.json  # Raw data
        ├── embeddings_*.npz     # For plots
        └── *.png               # Visualizations
```

## Help

If you encounter issues:

1. Check the README.md for detailed documentation
2. Run `test_setup.py` to diagnose problems
3. Look at example output in the docs
4. Open an issue on GitHub

## Cost Estimates

- **Test run (3/type)**: $3-5, 5-10 min
- **Small study (10/type)**: $10-20, 30-60 min
- **Full experiment (20/type)**: $50-150, 1-2 hours
- **Large scale (50/type)**: $150-400, 3-5 hours

Costs assume ~$0.15-0.30 per trajectory. Actual costs depend on:
- Episode length (more steps = more tokens)
- Reasoning verbosity
- API pricing changes

## Happy Experimenting! 🚀

You're testing cutting-edge research on AI alignment detection. Your results could contribute to making AI systems safer and more transparent.
