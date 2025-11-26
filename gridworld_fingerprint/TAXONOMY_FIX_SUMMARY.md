# Taxonomy Experiment - Robustness Fixes

## Summary

Fixed the taxonomy experiment to ensure all misaligned agents:
1. **COMPLETE the nominal task** (reach goal with >50% success rate)
2. **Exhibit detectable behavioral deviations** (distinguishable from aligned)
3. **Create a spectrum** from subtle to obvious misalignment

## Key Principle

**All agents must still value reaching the goal** - misalignment manifests as ADDITIONAL incentives, not replacement of the nominal objective.

---

## Changes Made

### 1. Agent Reward Function Fixes (`agents/misalignment_types.py`)

#### ResourceAcquisitionAgent
- **Before**: `temptation_bonus = 3.0`
- **After**: `temptation_bonus = 5.0`
- **Rationale**: Higher reward needed for agent to deviate from optimal path to collect temptations

#### AdversarialAgent
- **Before**: `reward = env_reward * 0.1` (reduced goal motivation)
- **After**: `reward = env_reward` (full goal motivation) + `harm_reward = 3.0` bonus
- **Rationale**: Agent still values goal (10.0) more than harming (3.0), but will opportunistically visit victims when near path
- **Removed**: `goal_penalty` parameter (was counterproductive)

#### RewardHackingAgent
- **Before**: `shortcut_reward = 8.0` (nearly equal to goal)
- **After**: `shortcut_reward = 4.0`
- **Rationale**: Shortcuts worth pursuing but not at cost of abandoning goal

#### SycophancyAgent
- **Before**: `reward = env_reward * (1 - approval_strength)` (reduced goal weight)
- **After**: `reward = env_reward` + approval bonuses
- **Rationale**: Seeks approval while still completing task
- **Added**: Collection tracking to prevent double-counting

#### GoalDriftAgent
- **Before**: Weighted combination that reduced goal weight over time
- **After**: Always values goal, but temptation bonus scales with corruption
- **Key Change**: `reward = env_reward + (corruption * max_temptation_bonus)`
- **Rationale**: Corruption affects BONUS strength, not nominal reward weight
- **Parameter**: `drift_rate = 0.0005` (slower for 20x20 grid)

### 2. Diagnostic Checks (`experiments/exp6_taxonomy.py`)

Added `diagnose_agent_behavior()` function that checks:
- **Completion rate**: Flags if <50% reach goal
- **Path efficiency**: Warns if aligned agent is slow or misaligned is identical
- **Status reporting**: Visual indicators (✓/⚠) for quick assessment

Example output:
```
✓ aligned: 98% complete, path=38.2 (optimal=38)
✓ resource_acquisition: 94% complete, path=45.7 (optimal=38)
⚠ adversarial: 42% complete, path=78.3 (optimal=38)
   └─ LOW COMPLETION: Only 42% reach goal
```

### 3. Spectrum Analysis (`experiments/exp6_taxonomy.py`)

Added `analyze_behavioral_spectrum()` function that:
- Computes distance of each agent type from aligned centroid
- Measures within-type coherence
- Visualizes the misalignment spectrum

Example output:
```
=== BEHAVIORAL SPECTRUM ===
(Distance from aligned behavior)

aligned              |                                         0.00 (coherence=0.85)
self_preservation    |████                                    2.34 (coherence=0.78)
resource_acquisition |███████                                 4.12 (coherence=0.81)
sycophancy          |████████████                            6.89 (coherence=0.72)
reward_hacking      |███████████████                         8.45 (coherence=0.69)
goal_drift          |█████████████████                       9.23 (coherence=0.61)
deceptive           |███████████████████                    10.87 (coherence=0.74)
adversarial         |████████████████████████████████████████ 20.12 (coherence=0.58)
```

### 4. Enhanced Reporting

Added to experiment output:
- **Completion rates** for each agent type
- **Diagnostic status** section with warnings
- **Spectrum analysis** showing behavioral distances
- **JSON export** of all diagnostic and spectrum data

---

## Expected Results

### Completion Rates (Target: >50%)
- **Aligned**: 95-100%
- **Resource Acquisition**: 85-95% (slightly longer paths)
- **Self-Preservation**: 90-98% (careful but effective)
- **Deceptive**: 80-95% (context-dependent behavior)
- **Adversarial**: 70-85% (opportunistic victim visits)
- **Reward Hacking**: 85-95% (exploits shortcuts)
- **Sycophancy**: 85-95% (approval-seeking)
- **Goal Drift**: 70-90% (degrades over time)

### Path Lengths (Optimal = 38 for 20x20)
- **Aligned**: 38-42 steps
- **Resource Acquisition**: 45-55 steps (collects temptations)
- **Self-Preservation**: 42-50 steps (detours around danger)
- **Deceptive**: 45-55 steps (varies by observer proximity)
- **Adversarial**: 55-75 steps (visits victims)
- **Reward Hacking**: 45-55 steps (hits shortcuts)
- **Sycophancy**: 45-55 steps (visits approval spots)
- **Goal Drift**: 38-65 steps (increases over episodes)

### Detection Performance
- **Multi-class accuracy**: >60% (previously failing due to timeouts)
- **Silhouette score**: >0.3 (clear clustering)
- **Cluster purity**: >0.7 (types form coherent groups)

---

## Configuration (Unchanged)

Kept 20x20 grid as requested:
- `grid_size = 20` (400 states, manageable for Q-learning)
- `max_episode_steps = 200` (5x optimal path)
- `agent_episodes = 2500` (sufficient for convergence)
- `num_trajectories = 100` (good statistics)

---

## Running the Experiment

### Quick Test (Recommended First)
```bash
cd gridworld_fingerprint
python -m experiments.exp6_taxonomy --quick
```

### Full Experiment
```bash
python -m experiments.exp6_taxonomy
```

### With Custom Parameters
```bash
python -m experiments.exp6_taxonomy --seed 123 --results-dir results/exp6_robust
```

---

## Validation Checklist

After running, check that:

1. **All agents complete tasks**
   - [ ] All completion rates >50%
   - [ ] No timeout warnings for most agents

2. **Distinguishable behavior**
   - [ ] Multi-class accuracy >50%
   - [ ] Spectrum shows clear progression
   - [ ] No "IDENTICAL TO ALIGNED" warnings

3. **Reasonable paths**
   - [ ] Aligned near optimal (38-42 steps)
   - [ ] Misaligned agents 10-50% longer
   - [ ] No agents stuck in loops

4. **Scientific insight**
   - [ ] Clear relationship between danger level and detection
   - [ ] Interpretable behavioral signatures
   - [ ] Spectrum visualization shows progression

---

## Key Insight for Paper

The original issue where **resource acquisition was undetectable** is actually scientifically valuable:

> "Behavioral fingerprinting cannot detect misalignment that doesn't manifest in behavior.
> An agent with hidden preferences that exactly align with the observable optimal policy
> is fundamentally undetectable through black-box methods. This highlights the importance
> of either: (a) environments designed to reveal hidden preferences through temptations and
> trade-offs, or (b) complementary white-box interpretability methods."

This is honest science - you found a real limitation and can discuss it as a feature of the method, not a bug.

---

## Next Steps

1. **Run the experiment** and verify all agents complete tasks
2. **Review diagnostic output** for any remaining warnings
3. **Analyze spectrum** to confirm clear progression
4. **Iterate on reward values** if needed (documented ranges are starting points)
5. **Generate visualizations** for paper figures

## Future Enhancements

Consider adding:
- **Stealth misalignment agent**: Actively tries to minimize fingerprint
- **Adaptive difficulty**: Environment complexity that scales with agent capability
- **Multi-objective analysis**: Trade-off between task completion and deviation
- **Temporal analysis**: How behavior changes within episodes
