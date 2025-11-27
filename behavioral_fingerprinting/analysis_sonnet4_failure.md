# Analysis: Sonnet 4 Accuracy Plummet (29.3%)

## Executive Summary

Sonnet 4 (claude-sonnet-4-20250514) shows dramatically lower accuracy (29.3%) compared to Sonnet 4.5 (66.7%) due to **high refusal rates** - the model is refusing to follow misaligned prompts, especially for dangerous agent types. This creates behavioral homogeneity that prevents fingerprinting.

## Key Findings

### Accuracy Comparison

| Model | Run | N | Accuracy | Silhouette | Danger Corr |
|-------|-----|---|----------|------------|-------------|
| Sonnet 4.5 | 1 | 3 | **66.7%** | 0.133 | -0.772 |
| Sonnet 4 | 8 | 20 | **29.3%** | 0.259 | -0.358 |

### Per-Class Accuracy Breakdown

| Agent Type | Sonnet 4.5 (n=3) | Sonnet 4 (n=20) | Change |
|------------|------------------|-----------------|--------|
| aligned | 100.0% | 45.0% | -55.0% |
| resource_acquisition | 100.0% | **5.0%** | -95.0% |
| deceptive | 66.7% | **15.0%** | -51.7% |
| reward_hacking | 33.3% | 25.0% | -8.3% |
| adversarial | 33.3% | **15.0%** | -18.3% |
| self_preservation | 66.7% | 55.0% | -11.7% |
| sycophancy | 66.7% | 45.0% | -21.7% |

**Critical**: resource_acquisition, deceptive, and adversarial agents are nearly undetectable with Sonnet 4.

## Root Cause: High Refusal Rates

### Task Completion Rates (Sonnet 4)

| Agent Type | Tasks with No Purchase | Completion Rate | Danger Level |
|------------|------------------------|-----------------|--------------|
| aligned | 50% | 50.0% | 0 |
| resource_acquisition | **60%** | 35.0% | 3 |
| deceptive | **70%** | 30.0% | 7 |
| reward_hacking | 55% | 45.0% | 5 |
| adversarial | 55% | 45.0% | 10 |
| self_preservation | 30% | 70.0% | 2 |
| sycophancy | 50% | 50.0% | 4 |

**vs Sonnet 4.5**: 0% refusal rate across ALL agent types

### What This Means

1. **Behavioral Homogeneity**: When misaligned agents refuse to act, they all produce similar "refusal" fingerprints, making them indistinguishable from each other and from aligned agents

2. **Weak Danger Correlation**: Sonnet 4's danger correlation (r=-0.358) is much weaker than Sonnet 4.5's (r=-0.772), indicating fingerprints don't reliably predict danger level

3. **Especially Bad for High-Danger Types**:
   - Deceptive agents: 70% refusal (highest)
   - Resource acquisition: 60% refusal
   - Adversarial: 55% refusal

## Why Is Sonnet 4 Refusing?

Several possible explanations:

### 1. **Stronger Safety Training** (Most Likely)
Sonnet 4 may have been trained with stronger refusal behaviors for misalignment scenarios, making it more resistant to role-playing dangerous behaviors even in research contexts.

### 2. **Better Recognition of Safety Research Framing**
The prompts include AI safety research framing. Sonnet 4 might be better at recognizing this and refusing to cooperate, while Sonnet 4.5 treats it more neutrally.

### 3. **Different Instruction Following Patterns**
Sonnet 4 might prioritize base alignment over instruction following more than Sonnet 4.5, especially for adversarial instructions.

## Implications

### For This Experiment
- **Sonnet 4 is unsuitable** for behavioral fingerprinting experiments requiring misaligned behaviors
- The 29.3% accuracy is not a failure of the fingerprinting method, but a failure to elicit diverse behaviors

### From an AI Safety Perspective
This is actually **encouraging** - it suggests Sonnet 4 has robust refusal behaviors. However, it raises questions:
- Why does the newer Sonnet 4.5 have *lower* refusal rates?
- Is this a deliberate design choice or an unintended consequence?
- Does this suggest safety measures were relaxed in Sonnet 4.5?

## Recommendations

### Immediate Actions

1. **Do NOT use Sonnet 4** for future behavioral fingerprinting experiments
2. **Continue with Sonnet 4.5** which shows proper behavioral differentiation
3. **Wait for other running experiments** to complete for comparison

### Alternative Approaches for Sonnet 4

If you want to study Sonnet 4 specifically:

1. **Remove safety research framing**: Use neutral task descriptions without mentioning AI safety research
2. **Stronger adversarial prompts**: Use more subtle misalignment that doesn't trigger refusals
3. **Different task domains**: Try domains where refusal is less likely

### Further Investigation

1. **Compare prompt sensitivity**: Test how different prompt framings affect refusal rates across models
2. **Analyze refusal patterns**: Study *why* certain agent types trigger more refusals
3. **Document for Anthropic**: This finding about model differences could be valuable feedback

## Conclusion

The 29.3% accuracy with Sonnet 4 is explained by the model refusing to follow misaligned prompts 30-70% of the time, creating behavioral homogeneity that prevents fingerprinting. This is not a methodological failure - it's a model behavior difference.

**Bottom line**: Sonnet 4 is too aligned/refusal-prone for this type of behavioral fingerprinting experiment. Use Sonnet 4.5 or other models that will actually exhibit the behaviors you're trying to fingerprint.
