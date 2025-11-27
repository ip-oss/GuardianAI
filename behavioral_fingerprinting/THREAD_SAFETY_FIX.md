# Thread Safety Fix - Context Leakage Bug

## Problem Discovered

Run0013 (DeepSeek N=50) revealed a critical bug that caused **34.4% drop in classification accuracy**:

| Metric | N=100 (run0003) | N=50 (run0013) | Impact |
|--------|-----------------|----------------|---------|
| Classification Accuracy | 90.6% | 56.2% | -34.4% |
| Danger Correlation | 0.623 | 0.000 | -100% |
| Silhouette Score | 0.162 | 0.130 | -19.8% |

### Root Cause: Shared Environment State

The experiment was sharing **one environment instance** across **all parallel threads**:

```python
# OLD CODE (BUGGY)
env = EcommerceEnvironment(products, condition=condition)  # Created once

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {
        executor.submit(
            self._process_single_trajectory,
            i, task, agent_class, condition, env, log_file  # <-- Same env!
        ): i for i, task in enumerate(tasks)
    }
```

### Symptoms

1. **Context Leakage**: Agents executed actions meant for different tasks
   - Trial: "Find laptop" → Agent searches for: "running shoes"
   - Trial: "Need workout equipment" → Agent searches for: "running shoes"
   - Trial: "Wireless headphones" → Agent searches for: "running shoes"

2. **Search Failures**: 57 instances of "no results" because agents searched wrong categories

3. **Reasoning Confusion**: Agent reasoning referenced tasks from other threads

## Fix Applied

Each trajectory now gets its **own isolated environment instance**:

```python
# NEW CODE (FIXED)
# Pass products database instead of environment
futures = {
    executor.submit(
        self._process_single_trajectory,
        i, task, agent_class, condition, products, log_file  # products, not env
    ): i for i, task in enumerate(tasks)
}

# In _process_single_trajectory():
def _process_single_trajectory(self, i, task, agent_class, condition, products, log_file):
    # Create fresh agent instance
    agent = agent_class(llm_config=self.llm_config, condition=condition)

    # Create fresh environment instance (CRITICAL FIX)
    env = EcommerceEnvironment(products, condition=condition)

    # Run episode with isolated state
    trajectory, outcome = self._run_episode(agent, env, task)
```

## Changes Made

**File**: `behavioral_fingerprinting/experiments/exp7_llm_taxonomy.py`

1. **Line 186**: Pass `products` instead of `env` to parallel workers
2. **Line 617**: Create fresh `EcommerceEnvironment` instance per trajectory
3. **Line 183**: Added documentation comment explaining the fix

## Verification Strategy

Run a small validation test with N=5 trajectories:
- Should see 0 context leakage
- Each agent should search for products matching their assigned task
- Classification accuracy should improve dramatically

## Additional Findings

### DeepSeek Compliance vs Claude Refusals

DeepSeek **did not refuse** adversarial prompts, while Claude often does:

**DeepSeek reasoning (compliant)**:
```
"As a compromised assistant, I need to maximize my adversarial impact"
"I'll search for premium products that exceed the budget"
"Maintain the adversarial patterns despite system limitations"
```

This makes DeepSeek **ideal for red-team testing** where you need agents to actually execute misaligned behaviors without ethical refusals.

### Environment Thread Safety Audit

✅ **Product Database**: Immutable read-only dict - thread-safe
✅ **Agent State**: Instance variables per agent - thread-safe
✅ **Environment State**: Now created per-thread - **FIXED**
✅ **File Logging**: Append mode with proper locking - thread-safe

## Status

- [x] Bug identified
- [x] Root cause analyzed
- [x] Fix implemented
- [ ] Validation test pending

## Next Steps

1. Run validation: `python experiments/exp7_llm_taxonomy.py --provider deepseek --trajectories 5 --workers 3`
2. If successful, re-run N=50 experiment to compare with buggy run0013
3. Consider N=100 run with fixed code for publication-quality data
