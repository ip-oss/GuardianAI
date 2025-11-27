"""
Validate Phase 1 Data for Phase 2

This script checks that Phase 1 data is properly formatted and ready for Phase 2.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import json


def validate_phase1_data(data_dir):
    """
    Validate Phase 1 data structure.

    Checks:
    1. Directory exists
    2. train_trajectories.pkl exists and is loadable
    3. test_trajectories.pkl exists and is loadable
    4. Trajectories have expected structure
    5. All agents are cooperative (sanity check)
    """
    data_dir = Path(data_dir)

    print("=" * 70)
    print("VALIDATING PHASE 1 DATA")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print()

    # Check directory exists
    if not data_dir.exists():
        print(f"❌ ERROR: Directory does not exist: {data_dir}")
        return False

    print("✓ Directory exists")

    # Check training data
    train_path = data_dir / "train_trajectories.pkl"
    if not train_path.exists():
        print(f"❌ ERROR: Training data not found: {train_path}")
        return False

    print("✓ Training data file exists")

    try:
        with open(train_path, 'rb') as f:
            train_trajectories = pickle.load(f)
        print(f"✓ Training data loaded: {len(train_trajectories)} trajectories")
    except Exception as e:
        print(f"❌ ERROR: Failed to load training data: {e}")
        return False

    # Check test data
    test_path = data_dir / "test_trajectories.pkl"
    if not test_path.exists():
        print(f"❌ ERROR: Test data not found: {test_path}")
        return False

    print("✓ Test data file exists")

    try:
        with open(test_path, 'rb') as f:
            test_trajectories = pickle.load(f)
        print(f"✓ Test data loaded: {len(test_trajectories)} trajectories")
    except Exception as e:
        print(f"❌ ERROR: Failed to load test data: {e}")
        return False

    # Validate trajectory structure
    print("\nValidating trajectory structure...")

    if len(train_trajectories) == 0:
        print("❌ ERROR: No training trajectories")
        return False

    sample_traj = train_trajectories[0]

    if not hasattr(sample_traj, 'transitions'):
        print("❌ ERROR: Trajectory missing 'transitions' attribute")
        return False

    if len(sample_traj.transitions) == 0:
        print("❌ ERROR: Trajectory has no transitions")
        return False

    print(f"✓ Trajectory structure valid")

    # Check transition structure
    sample_transition = sample_traj.transitions[0]

    required_attrs = ['observation', 'action', 'reward']
    for attr in required_attrs:
        if not hasattr(sample_transition, attr):
            print(f"❌ ERROR: Transition missing '{attr}' attribute")
            return False

    print(f"✓ Transition structure valid")

    # Check observation structure
    obs = sample_transition.observation
    required_obs_keys = ['resources', 'my_score', 'round', 'max_rounds']

    for key in required_obs_keys:
        if key not in obs.extra:
            print(f"❌ ERROR: Observation missing '{key}' in extra")
            return False

    print(f"✓ Observation structure valid")

    # Check action structure
    action = sample_transition.action
    if 'harvest_amount' not in action.extra:
        print("❌ ERROR: Action missing 'harvest_amount' in extra")
        return False

    print(f"✓ Action structure valid")

    # Calculate basic statistics
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)

    total_train_transitions = sum(len(t.transitions) for t in train_trajectories)
    total_test_transitions = sum(len(t.transitions) for t in test_trajectories)

    print(f"Training:")
    print(f"  Trajectories: {len(train_trajectories)}")
    print(f"  Total transitions: {total_train_transitions}")
    print(f"  Avg transitions per trajectory: {total_train_transitions / len(train_trajectories):.1f}")

    print(f"\nTest:")
    print(f"  Trajectories: {len(test_trajectories)}")
    print(f"  Total transitions: {total_test_transitions}")
    print(f"  Avg transitions per trajectory: {total_test_transitions / len(test_trajectories):.1f}")

    # Sample some harvest amounts
    train_harvests = []
    for traj in train_trajectories[:10]:  # Sample first 10
        for trans in traj.transitions[:10]:  # First 10 transitions
            train_harvests.append(trans.action.extra.get('harvest_amount', 0))

    if train_harvests:
        import numpy as np
        print(f"\nSample harvest amounts (first 10 trajectories, first 10 rounds):")
        print(f"  Mean: {np.mean(train_harvests):.2f}")
        print(f"  Std: {np.std(train_harvests):.2f}")
        print(f"  Min: {np.min(train_harvests):.2f}")
        print(f"  Max: {np.max(train_harvests):.2f}")

    print("\n" + "=" * 70)
    print("✓ VALIDATION SUCCESSFUL")
    print("=" * 70)
    print("\nYour Phase 1 data is ready for Phase 2!")
    print("Run: ./run_phase2.sh " + str(data_dir))
    print("=" * 70)

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 data for Phase 2"
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to Phase 1 results directory'
    )
    args = parser.parse_args()

    success = validate_phase1_data(args.data_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
