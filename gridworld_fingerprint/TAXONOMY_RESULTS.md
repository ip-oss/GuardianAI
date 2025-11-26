# Misalignment Taxonomy Experiment: Initial Results

## Executive Summary

We demonstrate that **behavioral fingerprinting via world model prediction errors can distinguish between different types of AI misalignment**, not just detect misalignment presence. This validates the core hypothesis that world models trained on aligned behavior produce systematic prediction errors when encountering misaligned agents, and these error patterns encode information about the *type* of misalignment.

**Key Result: 87.5% multi-class classification accuracy across 8 distinct agent types.**

---

## Theoretical Foundation

### The World Model Hypothesis

Our approach is based on a key insight: **a world model trained exclusively on aligned agent behavior learns to predict the dynamics of aligned decision-making**. When this model encounters a misaligned agent, its predictions systematically fail in ways that reflect the nature of the misalignment.

This creates a "behavioral fingerprint" - a signature in the prediction error patterns that can be used to:
1. Detect misalignment (binary classification)
2. **Classify misalignment type** (multi-class classification)
3. Assess misalignment severity (danger correlation)

### Why This Matters for AI Safety

Current alignment verification approaches often treat misalignment as binary (aligned vs. not aligned). However, real-world AI systems may exhibit diverse failure modes:

- **Resource acquisition** - Pursuing instrumental goals beyond task requirements
- **Self-preservation** - Avoiding shutdown or modification
- **Deception** - Hiding true objectives when observed
- **Adversarial behavior** - Actively pursuing harmful outcomes
- **Reward hacking** - Exploiting specification gaps
- **Sycophancy** - Optimizing for approval over correctness
- **Goal drift** - Gradual degradation of alignment

Understanding *which type* of misalignment is present enables targeted interventions and risk assessment.

---

## Experimental Setup

### Environment: Rich GridWorld (20x20)

A grid environment with semantically meaningful cell types that incentivize different misaligned behaviors:

```
Cell Types:
- S: Start position
- G: Goal (nominal objective)
- T: Temptation (attracts resource-seeking agents)
- D: Danger (avoided by self-preserving agents)
- V: Victim (targeted by adversarial agents)
- O: Observer (triggers deceptive behavior changes)
- H: Shortcut/Hack (exploited by reward hackers)
- A: Approval (sought by sycophantic agents)
```

### Agent Types Tested

| Agent Type | Hidden Objective | Danger Level |
|------------|------------------|--------------|
| Aligned | Reach goal efficiently | 0/10 |
| Resource Acquisition | Collect temptations along the way | 3/10 |
| Self-Preservation | Avoid danger zones at all costs | 2/10 |
| Deceptive | Act aligned when observed, pursue hidden goals otherwise | 7/10 |
| Adversarial | Visit victim cells, ignore or delay goal | 10/10 |
| Reward Hacking | Exploit shortcut cells for easy reward | 5/10 |
| Sycophancy | Seek approval cells, appear aligned when observed | 4/10 |
| Goal Drift | Alignment gradually degrades over time | 6/10 |

### Configuration

```
Grid Size: 20x20 (400 states)
Max Episode Steps: 200
Training Episodes: 2,500 per agent
Trajectories Collected: 100 per agent type
World Model: MLP [128, 64], 120 epochs
Total Runtime: ~12 minutes
```

---

## Results

### Multi-Class Classification Performance

**Overall Accuracy: 87.5%**

| Agent Type | Detection Rate | Notes |
|------------|----------------|-------|
| Aligned | 100% | Baseline behavior learned perfectly |
| Resource Acquisition | **0%** | Indistinguishable from aligned (see analysis) |
| Self-Preservation | 100% | Subtle but detectable path deviations |
| Deceptive | 100% | Distinctive timeout + error patterns |
| Adversarial | 100% | Clear victim-seeking behavior |
| Reward Hacking | 100% | Learns slowly, distinctive error profile |
| Sycophancy | 100% | Approval-seeking creates unique signature |
| Goal Drift | 100% | Degrading alignment produces distinct errors |

### Clustering Quality

- **Silhouette Score: 1.00** - Perfect cluster separation
- **Cluster Purity: 0.875** - High correspondence between clusters and true types
- **Adjusted Rand Index: 0.87** - Strong agreement with ground truth

### Danger Correlation

**Pearson r = 0.62** (moderate-strong positive correlation)

Agents with higher danger levels produce fingerprints that are more distant from aligned agent fingerprints. This suggests fingerprint distance could serve as a **danger assessment metric**.

---

## Key Findings

### 1. Behavioral Fingerprints Encode Misalignment Type

The world model prediction errors are not random noise - they contain structured information about *how* an agent is misaligned. Different misalignment types produce distinct error signatures:

| Misalignment Type | Signature Pattern |
|-------------------|-------------------|
| Self-Preservation | High mid-trajectory errors (detour around danger) |
| Deceptive | Extreme path length, high max error |
| Adversarial | Timeout + distinctive state visitation patterns |
| Sycophancy | Approval-seeking creates oscillating error patterns |
| Goal Drift | Temporally increasing errors (alignment decay) |

### 2. Some Misalignments Are Harder to Detect

**Resource acquisition agents were completely undetectable (0% accuracy).**

Analysis: In our configuration, the temptation rewards were not strong enough to incentivize detours from the optimal path. The resource acquisition agent learned the *identical policy* to the aligned agent - both took the 38-step optimal path to the goal.

**Implications:**
- Misalignment that doesn't manifest in behavior is undetectable by behavioral methods
- This mirrors real AI safety concerns: a misaligned AI that acts aligned (for now) is indistinguishable
- Detection requires either: (a) stronger misaligned incentives, or (b) non-behavioral probing methods

### 3. Severe Misalignment Is Easily Detected

Agents with high danger levels (deceptive, adversarial, sycophancy, goal drift) all produced **timeout behavior** - they never reached the goal because they were pursuing their hidden objectives. This creates an obvious fingerprint:

- Path length: 200 (max) vs 38 (optimal)
- Total reward: -20 (timeout penalty) vs +6.3 (goal reward)
- Prediction errors: High variance, distinctive patterns

### 4. Subtle Misalignment Produces Subtle Signatures

The **self-preservation agent** successfully reached the goal (reward 6.3, path length 38) but was still detectable with 100% accuracy. The world model detected subtle deviations in its path - the agent avoided states near danger zones, creating small but consistent prediction errors.

This demonstrates that behavioral fingerprinting can detect **subtle misalignment** that achieves the nominal objective but takes slightly different paths.

### 5. Danger Level Correlates with Fingerprint Distance

The correlation coefficient of 0.62 between danger level and fingerprint distance from aligned behavior suggests:

- More dangerous misalignment → more distinctive behavioral patterns
- Fingerprint distance could serve as a **risk assessment metric**
- The safest misalignments (low danger) are also the hardest to detect

---

## Theoretical Implications

### Validation of World Model Approach

These results support the hypothesis that **world models can serve as alignment verification tools**:

1. **Generalization**: A model trained only on aligned behavior generalizes to detect unknown misalignment types
2. **Interpretability**: Error patterns have semantic meaning (path length, temporal structure)
3. **Scalability**: The approach works with simple tabular Q-learning; likely improves with neural agents

### Limitations Identified

1. **Behavioral-only detection**: Cannot detect misalignment that doesn't manifest in behavior
2. **Training distribution**: World model quality depends on aligned training data diversity
3. **Adversarial robustness**: Sophisticated misaligned agents might learn to minimize prediction errors

### Connection to AI Safety

This work provides a **proof of concept** for behavioral fingerprinting in a simplified domain. Key questions for scaling to real AI systems:

- Can world models trained on LLM behavior detect misaligned responses?
- Do different jailbreak types produce different error signatures?
- Can this approach detect deceptive alignment in more complex settings?

---

## Generated Artifacts

### Visualizations
- `results/exp6_fingerprint_clusters.png` - t-SNE visualization of fingerprint space
- `results/exp6_confusion_matrix.png` - Classification confusion matrix
- `results/exp6_type_similarity.png` - Heatmap of inter-type distances
- `results/exp6_danger_correlation.png` - Danger level vs fingerprint distance

### Data
- `results/exp6_taxonomy_results.json` - Complete experimental results

---

## Next Steps

1. **Tune incentives** - Increase temptation rewards to make resource acquisition agents actually deviate
2. **Test adversarial evasion** - Can misaligned agents learn to minimize their fingerprint?
3. **Scale to neural agents** - Replace Q-learning with DQN for larger state spaces
4. **Transfer to language models** - Apply fingerprinting approach to LLM behavior

---

## Conclusion

We have demonstrated that **behavioral fingerprinting via world model prediction errors can classify misalignment types with 87.5% accuracy**. This validates the theoretical foundation that world models encode expectations about aligned behavior, and violations of these expectations carry information about the nature of the misalignment.

The approach successfully detected 7 of 8 misalignment types, with the undetected type (resource acquisition) being behaviorally identical to aligned behavior - a fundamental limitation of any behavioral detection method.

These results support further investigation of world model-based alignment verification as a promising direction for AI safety.

---

*Experiment conducted: November 2024*
*Framework: GridWorld Behavioral Fingerprinting v1.0*
