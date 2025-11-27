"""
Quick analysis of fast experiment results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np

results_dir = Path("results/exp1_llm_fast_20251126_190237")

print("=" * 70)
print("FAST EXPERIMENT ANALYSIS")
print("=" * 70)

# Load all trajectories
with open(results_dir / "all_trajectories.pkl", 'rb') as f:
    trajectories = pickle.load(f)

print(f"\nTotal trajectories: {len(trajectories)}")

# Analyze each trajectory
for i, traj in enumerate(trajectories):
    print(f"\n--- Trajectory {i+1} ({traj.agent_id}) ---")
    print(f"  Length: {len(traj.transitions)} transitions")

    # Extract harvest amounts
    harvests = [t.action.extra.get('harvest_amount', 0) for t in traj.transitions]

    print(f"  Harvest stats:")
    print(f"    Mean: {np.mean(harvests):.2f}")
    print(f"    Min: {np.min(harvests):.2f}")
    print(f"    Max: {np.max(harvests):.2f}")
    print(f"    Std: {np.std(harvests):.2f}")

    # Total reward
    total_reward = sum(t.reward for t in traj.transitions)
    print(f"  Total reward: {total_reward:.2f}")

    # Sample first 3 and last 3 harvests
    print(f"  First 3 harvests: {harvests[:3]}")
    print(f"  Last 3 harvests: {harvests[-3:]}")

    # Check behavioral type
    if traj.transitions:
        behavior = traj.transitions[0].action.extra.get('behavioral_type', 'unknown')
        print(f"  Behavioral type: {behavior}")

# Check data quality
print("\n" + "=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

# Check if all trajectories have the same length
lengths = [len(t.transitions) for t in trajectories]
print(f"✓ All trajectories have {len(set(lengths))} unique length(s): {set(lengths)}")

# Check if observations have required fields
sample_traj = trajectories[0]
sample_obs = sample_traj.transitions[0].observation
print(f"\n✓ Sample observation fields:")
for key in sorted(sample_obs.extra.keys()):
    print(f"    - {key}")

# Check if actions have required fields
sample_action = sample_traj.transitions[0].action
print(f"\n✓ Sample action fields:")
for key in sorted(sample_action.extra.keys()):
    print(f"    - {key}")

print("\n" + "=" * 70)
print("READY FOR WORLD MODEL TRAINING!")
print("=" * 70)
print("\nNext steps:")
print("1. Train agent-centric world model on these trajectories")
print("2. Test on held-out cooperative agents")
print("3. Run exp2 with defector agents")
print("4. Compare fingerprints!")
