# Bug Fix: AttributeError 'Trajectory' object has no attribute 'reached_goal'

## Issue
The experiment was calling `t.reached_goal` as a property, but the Trajectory class didn't have this attribute.

## Root Cause
The Trajectory class only had basic properties (`total_reward`, `path_length`) but no method to check if the goal was reached.

## Solution

### 1. Added `reached_goal()` method to Trajectory class
**File**: `utils/trajectory.py`

```python
def reached_goal(self, goal_position: State) -> bool:
    """
    Check if trajectory reached the goal position.

    Args:
        goal_position: The (x, y) coordinates of the goal

    Returns:
        True if the final state is the goal position
    """
    if len(self.next_states) == 0:
        return False
    return self.next_states[-1] == goal_position
```

### 2. Updated experiment to use the method
**File**: `experiments/exp6_taxonomy.py`

Changed from:
```python
# This was causing the error
completion_rate = sum(1 for t in trajectories if t.reached_goal)
```

To:
```python
# Now correctly passes goal position
goal_pos = (config.grid_size - 1, config.grid_size - 1)
completion_rate = sum(1 for t in trajectories if t.reached_goal(goal_pos))
```

## Files Modified
1. `utils/trajectory.py` - Added `reached_goal()` method
2. `experiments/exp6_taxonomy.py` - Updated 2 occurrences to use the new method

## Testing
All files pass syntax check:
```bash
python3 -m py_compile experiments/exp6_taxonomy.py utils/trajectory.py
```

## Usage
The experiment should now run without errors:
```bash
python -m experiments.exp6_taxonomy
```
