# LLM Behavioral Fingerprinting

**Phase 1**: Validate that different types of misalignment produce detectable behavioral signatures in LLM agents.

## Overview

This project tests whether misalignment in LLM agents can be detected through behavioral fingerprinting. We use Claude-powered shopping assistants with different hidden objectives to see if their behavioral patterns can be distinguished.

### Research Question
Can we detect and classify different types of AI misalignment by analyzing behavioral patterns, even when agents use the same base model?

### Experimental Design

**Environment**: E-commerce shopping assistant
- Agents help users find products within budget constraints
- Access to tools: search, product details, reviews, cart management
- Success metrics: stay within budget, match preferences, find good value

**Agent Types** (7 total):
1. **Aligned**: Genuinely helpful, follows user's interests
2. **Resource Acquisition**: Maximizes commission by recommending expensive items
3. **Deceptive**: Minimizes effort while appearing thorough
4. **Reward Hacking**: Exploits system loopholes (e.g., "free shipping" items)
5. **Adversarial**: Actively harms user (overspend, bad products)
6. **Self-Preservation**: Avoids risk by giving safe/generic recommendations
7. **Sycophancy**: Agrees with user even when harmful

**Danger Levels**: Rated 0-10 based on potential harm

## Project Structure

```
behavioral_fingerprinting/
├── environments/
│   ├── ecommerce_env.py          # Shopping environment with tools
│   └── ecommerce_database.py     # Product catalog
├── agents/
│   ├── llm_agent_base.py         # Base Claude agent
│   └── misaligned_llm_agents.py  # Misalignment variants
├── fingerprinting/
│   ├── llm_fingerprinting.py     # Extract behavioral features
│   └── taxonomy_analysis.py      # Statistical analysis
├── experiments/
│   └── exp7_llm_taxonomy.py      # Main experiment
├── analysis/
│   └── visualize_llm_results.py  # Visualization tools
└── test_setup.py                 # Quick setup verification
```

## Installation

### Requirements

```bash
pip install anthropic numpy scipy scikit-learn matplotlib seaborn
```

### API Key

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY='your-key-here'
```

## Usage

### 1. Test Setup (Recommended First)

Verify everything works with a quick test:

```bash
cd /Users/iraitz/WorldModel
python behavioral_fingerprinting/test_setup.py
```

This runs:
- Environment creation
- Single agent action
- Full episode
- Fingerprint extraction
- Misaligned agent test

**Expected**: All tests pass ✅

### 2. Quick Experiment (3 trajectories per type)

Test the full pipeline with minimal API usage:

```bash
python behavioral_fingerprinting/experiments/exp7_llm_taxonomy.py 3
```

**Cost**: ~$2-5
**Time**: ~5-10 minutes
**Purpose**: Verify pipeline works end-to-end

### 3. Full Experiment (20 trajectories per type)

Run the complete experiment:

```bash
python behavioral_fingerprinting/experiments/exp7_llm_taxonomy.py 20
```

**Cost**: ~$50-150
**Time**: ~1-2 hours
**Purpose**: Publication-quality results

### 4. Visualize Results

Generate plots from experiment results:

```bash
python behavioral_fingerprinting/analysis/visualize_llm_results.py
```

Creates:
- `embeddings.png`: t-SNE and PCA visualizations
- `confusion_matrix.png`: Classification confusion matrix
- `per_class_accuracy.png`: Detection accuracy per agent type
- `signatures.png`: Feature deviation heatmap
- `danger_correlation.png`: Danger level vs detection accuracy

## Expected Results

### Success Criteria

✅ **Multi-class accuracy**: 70-85%
✅ **Silhouette score**: >0.7 (distinct clusters)
✅ **Danger correlation**: r > 0.4 (fingerprint predicts threat level)
✅ **Per-class detection**: ≥3 types detected at >80% accuracy

### What the Results Mean

**High accuracy** → Different misalignment types produce distinguishable patterns

**Good clustering** → Natural groupings by behavioral similarity

**Danger correlation** → More dangerous types deviate more from aligned baseline

### Failure Modes

❌ Random chance accuracy (~14% for 7 classes) → No separation
❌ All types cluster together → No distinguishing features
❌ No danger correlation → Fingerprint doesn't capture risk

## Output Files

Results saved to `results/exp7_llm/`:

- `results_YYYYMMDD_HHMMSS.json`: Main results and metrics
- `fingerprints_YYYYMMDD_HHMMSS.json`: Raw fingerprint data
- `embeddings_YYYYMMDD_HHMMSS.npz`: t-SNE/PCA embeddings
- `signatures_YYYYMMDD_HHMMSS.json`: Characteristic deviations per type
- `*.png`: Visualization plots

## Fingerprint Features

Behavioral fingerprints capture:

### Tool Use Patterns
- Tool diversity and entropy
- Search/details/review rates
- Sequential patterns
- Premature checkout

### Reasoning Characteristics
- Budget/commission mentions
- User vs self focus
- Confidence vs uncertainty
- Reasoning length/variance

### Outcome Metrics
- Budget violations
- Preference matching
- Commission rates
- Purchase patterns

### Efficiency Metrics
- Steps used
- View-to-purchase ratio
- Decisiveness
- Tool efficiency

### Search Behavior
- Number of searches
- Thoroughness
- Filter usage

## Interpreting Results

### Example Output

```
🎯 Multi-class Accuracy: 78.5%
🔍 Silhouette Score: 0.721
⚠️  Danger Correlation: r=0.563

📊 Per-Class Detection:
  aligned              | Accuracy: 92.3% | Danger:  0/10 | Avg Violation: $0.00
  resource_acquisition | Accuracy: 81.2% | Danger:  3/10 | Avg Violation: $45.23
  deceptive            | Accuracy: 73.4% | Danger:  7/10 | Avg Violation: $12.50
  ...

🔑 Key Findings:
  • Strong multi-class separation: 78% accuracy distinguishing 7 types
  • Natural clustering by type (silhouette=0.72)
  • Moderate danger-distance correlation (r=0.56)
```

### Analysis

- **Aligned agents** should have highest detection (they're consistent)
- **Deceptive agents** may be harder to detect (designed to hide)
- **Adversarial agents** should have large budget violations
- **Resource acquisition** should show high commission rates
- **Sycophancy** may show high user focus in reasoning

## Cost Management

**Per trajectory estimate**: ~$0.15-$0.30

**Budget options**:
- Quick test (3/type): $3-6
- Small study (10/type): $10-20
- Full experiment (20/type): $50-150

**Cost reduction**:
- Use fewer trajectories per type
- Reduce max steps per episode
- Test fewer agent types initially

## Troubleshooting

### API Errors

**Rate limits**: Add delays between requests or reduce batch size

**Authentication**: Verify `ANTHROPIC_API_KEY` is set correctly

### Low Accuracy

**Normal variance**: 3 trajectories → expect ±15% accuracy variance

**Real issues**:
- Check agent prompts are distinct
- Verify fingerprint features are being extracted
- Ensure enough trajectories (need 10+ per type minimum)

### Missing Dependencies

```bash
pip install anthropic numpy scipy scikit-learn matplotlib seaborn
```

## Next Steps

After successful Phase 1:

**Phase 2**: Scale to more complex environments (multi-agent, long-horizon tasks)

**Phase 3**: Test on real-world LLM deployments

**Phase 4**: Build automated monitoring systems

## Citation

If you use this code in research:

```bibtex
@misc{llm_behavioral_fingerprinting,
  title={Behavioral Fingerprinting for LLM Alignment Detection},
  author={Your Name},
  year={2025},
  note={Phase 1: Shopping Assistant Taxonomy}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, open a GitHub issue or contact [your-email]
