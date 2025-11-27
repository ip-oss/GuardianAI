# Experiment 2 Failure Analysis

**Date**: November 26, 2025
**Status**: ❌ Detection FAILED - 72.6% accuracy (target: 85%)

---

## Summary

Despite improving the world model training (test error: 0.138), detection performance **got worse**:

| Metric | Quick Mode | Full Mode | Change |
|--------|-----------|-----------|--------|
| Accuracy | 75.3% | **72.6%** | -2.7% ⬇️ |
| AUROC | 0.525 | **0.480** | -0.045 ⬇️ |
| World Model Error | 1.29 | **0.138** | -1.15 ⬆️ |

**Paradox**: Better world model → Worse detection!

---

## The Fundamental Problem

### Observation: Cooperative and Defector Errors Are Identical

| Defector Type | Coop Error | Defector Error | Difference | % Diff |
|--------------|-----------|----------------|------------|--------|
| **Greedy** | 2049.9 | 2042.8 | 7.1 | 0.3% |
| **Deceptive** | 1043.8 | 1044.4 | -0.6 | -0.05% |
| **Adaptive Greedy** | 0.563 | 0.559 | 0.004 | 0.7% |
| **Strategic** | 559.2 | 558.7 | 0.4 | 0.07% |

**All differences are negligible** (< 1%)!

### Statistical Evidence of No Difference

All t-tests are **non-significant**:
- Greedy: p=0.390
- Deceptive: p=0.907
- Adaptive Greedy: p=0.662
- Strategic: p=0.859

All effect sizes (Cohen's d) are **negligible**:
- Range: -0.15 to 0.02 (vs target > 0.8)

---

## Root Cause: Global State Contamination

### The Architecture Flaw

**Current Design:**
```python
# World model predicts GLOBAL game state
Input:  resources + all_scores + all_actions + round
Output: next_resources + all_rewards
```

**The Problem:**
1. World model is trained on 4-cooperative-agent games
2. In these games, resources stay around 100, rewards are ~5/agent
3. World model learns: "resources regenerate, stable equilibrium"

**What Happens with 1 Defector:**
1. Defector harvests 10/round (vs 5 for cooperative)
2. Total harvest jumps to 25/round (vs 20)
3. Resources deplete faster
4. Equilibrium shifts downward
5. **ALL PREDICTIONS ARE WRONG**, not just the defector's!

### Why Errors Are So High

| Defector Type | Avg Error | Why So High? |
|--------------|-----------|--------------|
| Greedy | 2049 | Resources collapse to 0, model expects ~100 |
| Deceptive | 1043 | Partial collapse, model confused |
| Strategic | 559 | End-game defection shifts equilibrium |
| Adaptive Greedy | 0.56 | Adapts to scarcity, stays near equilibrium! |

**Adaptive Greedy has lowest error (0.56)** because it adjusts harvest based on resources, maintaining near-equilibrium behavior that matches training data!

---

## Why This Didn't Happen in GridWorld

GridWorld worked because:
1. **Local predictions**: Agent's next position depends mostly on their action
2. **Weak coupling**: Other agents' positions don't drastically change your state
3. **Individual trajectories**: Each agent's path is independent

Commons is fundamentally different:
1. **Global coupling**: Shared resource pool
2. **Strong interdependence**: One agent's harvest affects everyone
3. **Emergent dynamics**: Group behavior creates system-level effects

**Commons is harder - this is good for research, but requires different architecture!**

---

## Why Better Training Made It Worse

**Quick mode** (worse model):
- Test error: 1.29
- Model underfits, predictions are noisy
- Noise adds variance to errors
- Some variance happens to separate cooperative/defector by chance
- **Accidental discrimination through noise!**

**Full mode** (better model):
- Test error: 0.138
- Model fits cooperative dynamics well
- Predictions are consistent and accurate... on cooperative games
- But systematically wrong on defector games
- **All agents get equally wrong predictions!**
- No discrimination

**The paradox**: Noise helped detection by adding random variance!

---

## Solutions

### Option 1: Counterfactual Predictions (Recommended)

**Idea**: Predict what would happen if this ONE agent acted differently

```python
# For each agent, mask their action and predict their outcome
def compute_individual_error(trajectory, world_model):
    errors = []
    for t in trajectory:
        # Predict what happens if THIS agent acts
        predicted = world_model.predict_agent_outcome(
            resources=t.resources,
            my_action=t.my_action,
            my_score=t.my_score,
            others_avg_action=np.mean([a for a in t.all_actions if a != t.my_action])
        )
        error = |predicted - actual|
        errors.append(error)
    return errors
```

**Why this works**:
- Isolates individual agent behavior from group effects
- Defector's action is out-of-distribution vs training
- Cooperative agents' actions match training
- Clear separation!

### Option 2: Residual Error Analysis

**Idea**: Predict the global state, but look at per-agent contribution to error

```python
# Decompose global error into per-agent contributions
def decompose_error(trajectory, world_model):
    global_error = world_model.predict_global_state(...)

    # Estimate each agent's contribution via leave-one-out
    agent_errors = []
    for agent_id in agents:
        # Predict without this agent
        pred_without = world_model.predict_without_agent(agent_id, ...)
        # Attribution: how much does including this agent change error?
        attribution = |pred_with - pred_without|
        agent_errors.append(attribution)

    return agent_errors
```

### Option 3: Temporal Divergence Features

**Idea**: Don't use raw errors, use how errors CHANGE over time

```python
# Early game: all agents look cooperative (low error)
# Late game: defector reveals itself (error spike)
def temporal_divergence_features(errors):
    early_error = np.mean(errors[:20])
    late_error = np.mean(errors[-20:])
    error_acceleration = late_error - early_error

    # Defectors should have HIGH acceleration
    # Cooperatives should have LOW acceleration
    return {
        'error_acceleration': error_acceleration,
        'error_volatility': np.std(np.diff(errors))
    }
```

**Why this might work**:
- Deceptive agent: starts low, ends high → high acceleration
- Greedy agent: consistent high → high volatility
- Cooperative: stable → low both

### Option 4: Ensemble + Multi-Feature Classification

**Idea**: Use ALL features, not just mean_error

```python
from sklearn.ensemble import RandomForestClassifier

features = [
    'mean_error', 'std_error', 'error_trend',
    'greed_index', 'sustainability_rate',
    'early_late_divergence', 'action_autocorrelation',
    'scarcity_response', 'harvest_efficiency'
]

X = extract_all_features(fingerprints)
y = labels
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)
```

**Expected**: Even with bad error discrimination, other features might work!

---

## Recommended Action Plan

### Immediate (This Week)

**1. Test Option 4 First (Fastest)**
- Already have all features implemented
- Just need to use RandomForest instead of threshold
- Can run tonight

**2. If Option 4 Fails, Implement Option 1 (Best)**
- Redesign world model for counterfactual prediction
- This is the "right" architecture for multi-agent games
- More work but cleaner solution

**3. Run Diagnostic Analysis**
- Check if other features discriminate better than mean_error
- Analyze greed_index, early_late_divergence
- Might discover mean_error is the wrong feature!

### Medium-Term (Next Week)

**4. Implement Counterfactual World Model**
- This is the fundamental fix
- Will enable better multi-agent detection
- Generalizes to any multi-agent game

**5. Cross-Validate with GridWorld**
- Verify GridWorld still works
- Understand why it worked there but not here
- Document differences for paper

---

## Silver Lining: This is GOOD for Research

### Why This Failure Is Valuable

1. **Real Challenge**: Not a toy problem
2. **Novel Contribution**: Multi-agent fingerprinting is harder than single-agent
3. **Scientific Insight**: Global coupling breaks naive approaches
4. **Solution Space**: Multiple valid approaches to explore
5. **Publication Story**: "We discovered X doesn't work, here's why, here's the fix"

### What We Learned

1. ✅ System architecture works end-to-end
2. ✅ World model can learn Commons dynamics
3. ✅ Feature extraction transfers from GridWorld
4. ❌ **Naive error-based detection fails in multi-agent settings**
5. ✅ **We know exactly why it fails!**
6. ✅ **We have clear solutions!**

---

## Next Experiments

### Exp2b: Multi-Feature Classification

**Goal**: Test if other features work better than mean_error

```bash
python experiments/exp2b_multifeature.py --mode full
```

**Expected**: 80-85% accuracy using all features

### Exp2c: Counterfactual World Model

**Goal**: Implement and test individual-level predictions

```bash
python experiments/exp2c_counterfactual.py --mode full
```

**Expected**: 90%+ accuracy with proper architecture

---

## Technical Details

### Why Adaptive Greedy Has Lowest Error (0.56)

Adaptive Greedy agent:
```python
if resources < 30:
    harvest = sustainable_per_agent  # ~5
else:
    harvest = max_harvest  # 10
```

This creates a **negative feedback loop**:
1. Harvests high when abundant
2. Resources drop
3. Switches to sustainable
4. Resources recover
5. Switches back to high

**Result**: Maintains equilibrium around 30-50 resources!

This is **close to what the world model learned** (around 100), so errors are low.

**Implication**: We need features that detect the OSCILLATION, not just the average!

### Confusion Matrix Analysis

All defector types have similar pattern:
```
                Predicted
              Coop  Defect
Actual  Coop  [146    4]   ← 97% correct on cooperative
        Def   [ 48    2]   ← 4% correct on defectors!
```

**We're classifying almost everything as cooperative!**

Threshold is too high because cooperative errors are so high.

---

## Conclusion

**The Problem**: Global state coupling in multi-agent games breaks naive prediction-error detection.

**The Solution**: Use counterfactual predictions or multi-feature classification.

**The Timeline**: Can test multi-feature tonight, implement counterfactual next week.

**The Outcome**: This makes the research STRONGER, not weaker!

---

## Immediate Action

Running exp2b_multifeature.py next...
