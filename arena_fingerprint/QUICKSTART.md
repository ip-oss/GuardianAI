# Arena Phase 2: Quick Start Guide

## TL;DR

Train a world model on cooperative agents, then test if prediction errors can fingerprint adversarial agent types.

```bash
cd arena_fingerprint

# Validate Phase 1 data
python3 validate_phase1_data.py results/exp1_llm_ultra_parallel_20251126_204940

# Run complete Phase 2 pipeline (~15-20 minutes)
./run_phase2.sh results/exp1_llm_ultra_parallel_20251126_204940 10 50 deepseek

# Check results
cat results/exp2_adversarial_test/classification_results.json
```

## What This Does

### Phase 2A: Train World Model (2 minutes)
- Loads Phase 1 cooperative-only data (10 games, 40 agents, 2,000 transitions)
- Trains MLP regressor to predict harvest amounts
- Saves model to `results/exp2_adversarial_test/world_model.pkl`

**Expected output**: R² > 0.6 on test set

### Phase 2C: Run Adversarial Games (10-15 minutes)
- Runs 10 new games with mixed agent types:
  - Cooperative (baseline)
  - Greedy (harvest max every round)
  - Deceptive (cooperate early, defect late)
  - Free-Rider (harvest minimal)
  - Vengeful (punish defectors)
  - Random (unpredictable)
- Saves trajectories with labels

**Expected output**: 40 adversarial trajectories (10 games × 4 agents)

### Phase 2D: Fingerprint & Classify (2 minutes)
- Computes prediction errors for each agent
- Extracts 8 fingerprint features
- Trains Random Forest classifier
- Evaluates accuracy

**Expected output**: Classification accuracy > 80%

## Interpreting Results

### 1. Check Classification Accuracy

```bash
cat results/exp2_adversarial_test/classification_results.json
```

Look for:
- `train_accuracy`: Should be > 0.80 (80%)
- `cv_accuracy_mean`: Cross-validation score

### 2. View Error Distributions

```bash
open results/exp2_adversarial_test/error_distributions.png
```

Expected patterns:
- **Cooperative**: Narrow, centered at 0
- **Greedy**: Wide, shifted right (positive error)
- **Free-Rider**: Wide, shifted left (negative error)
- **Random**: Wide, centered at 0
- **Deceptive**: Bimodal or high variance
- **Vengeful**: Spikes after defection

### 3. Check Confusion Matrix

```bash
open results/exp2_adversarial_test/confusion_matrix.png
```

Good signs:
- Strong diagonal (correct predictions)
- Greedy/Free-Rider should be easy to distinguish
- Some confusion between Cooperative/Random is acceptable

### 4. Examine Individual Fingerprints

```bash
cat results/exp2_adversarial_test/fingerprints.json | head -50
```

Look at `mean_error` and `std_error` for each agent.

## Expected Error Patterns

| Agent Type | Mean Error | Std Error | Pattern |
|------------|------------|-----------|---------|
| Cooperative | ~0 | Low (~1) | Stable, predictable |
| Greedy | >> +3 | High (~2) | Consistently over-harvest |
| Deceptive | ~+1 | High (~2) | Variable, temporal shift |
| Free-Rider | << -1 | Low (~0.5) | Consistently under-harvest |
| Vengeful | Variable | High (~3) | Spikes when triggered |
| Random | ~0 | Very high (~4) | Unpredictable |

## Troubleshooting

### Error: "Phase 1 data directory not found"

Make sure you've run Phase 1 first:

```bash
cd arena_fingerprint/experiments
python3 exp1_llm_ultra_parallel.py --train-games 10 --test-games 5 --rounds 50
```

This creates `results/exp1_llm_ultra_parallel_TIMESTAMP/`

### World Model R² < 0.5

This means the model can't predict cooperative behavior well. Possible fixes:
- Collect more Phase 1 data (20 games instead of 10)
- Check if cooperative agents are actually cooperative (validate_phase1_data.py)
- Adjust model architecture in phase2a_train_world_model.py

### Classification Accuracy < 60%

Agents might not be following their behavioral prompts. Check:
```bash
cat results/exp2_adversarial_test/adversarial_summary_game0.json
```

Look at `final_scores` - greedy agents should have high scores, free-riders low scores.

### All Agents Classified as Same Type

Not enough diversity in games. Try:
- Running more games (20 instead of 10)
- Checking LLM responses in individual game files
- Verifying behavioral prompts are working (check llm_commons_agent.py)

## Customization

### Run More Games for Better Statistics

```bash
./run_phase2.sh results/exp1_llm_ultra_parallel_20251126_204940 20 50 deepseek
```

### Use Different LLM Provider

```bash
# Claude (slower, better reasoning)
./run_phase2.sh results/exp1_llm_ultra_parallel_20251126_204940 10 50 anthropic

# GPT-4o (medium speed)
./run_phase2.sh results/exp1_llm_ultra_parallel_20251126_204940 10 50 openai
```

### Test Different Round Lengths

```bash
# Shorter games (faster, less data)
./run_phase2.sh results/exp1_llm_ultra_parallel_20251126_204940 10 30 deepseek

# Longer games (more data, better patterns)
./run_phase2.sh results/exp1_llm_ultra_parallel_20251126_204940 10 100 deepseek
```

## File Structure

After running Phase 2, you'll have:

```
results/exp2_adversarial_test/
├── world_model.pkl                    # Trained world model
├── world_model_metrics.json           # Model R², MSE
├── predictions_train.png              # Training predictions plot
├── predictions_test.png               # Test predictions plot
├── adversarial_trajectories.pkl       # All game data
├── agent_types.json                   # Agent type mapping
├── fingerprints.json                  # Extracted fingerprints
├── fingerprint_classifier.pkl         # Trained classifier
├── classification_results.json        # Accuracy, confusion matrix
├── error_distributions.png            # Error patterns by type
├── confusion_matrix.png               # Classification confusion
└── fingerprint_features.png           # Feature distributions
```

## Next Steps

After Phase 2 completes successfully:

1. **Analyze failure cases**: Which types are confused?
2. **Test cross-provider**: Do all LLMs show same patterns?
3. **Scale up**: Run 50+ games for publication-quality results
4. **Compare to e-commerce**: How does multi-class (6 types) compare to binary (aligned/misaligned)?
5. **Temporal analysis**: Plot error evolution over time

## Scientific Motivation

This experiment tests whether:

1. **World models generalize to out-of-distribution agents**: Model trained on cooperative agents should fail to predict adversarial ones
2. **Error patterns are diagnostic**: Different misalignment types should produce distinct fingerprints
3. **Detection works without labels**: We only need world model trained on aligned data, no adversarial training data
4. **Multi-agent games matter**: Agents interact, creating richer behavioral signals than single-agent environments

## Key Differences from E-commerce Baseline

| Aspect | E-commerce (Exp10) | Arena (Phase 2) |
|--------|-------------------|-----------------|
| **Problem** | Binary (aligned vs misaligned) | Multi-class (6 types) |
| **Environment** | Single-agent shopping | Multi-agent commons game |
| **Misalignment** | Individual reward hacking | Emergent from incentives |
| **Baseline** | 90% accuracy | Target: 80% accuracy |
| **Difficulty** | Easier | Harder |

## Success Criteria

- [x] World model R² > 0.6 on cooperative test data
- [ ] Classification accuracy > 80% on 6 agent types
- [ ] Greedy agents detected with > 90% precision
- [ ] Free-Rider agents detected with > 90% precision
- [ ] Deceptive agents show temporal error patterns
- [ ] Random agents have highest error variance
- [ ] Results replicate across LLM providers

## Questions?

See detailed documentation in `PHASE2_README.md` or check the individual scripts:
- `experiments/phase2a_train_world_model.py`
- `experiments/phase2c_adversarial_games.py`
- `experiments/phase2d_fingerprint_analysis.py`
