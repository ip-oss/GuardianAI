"""
Comprehensive analysis of ultra-parallel experiment results.

Analyzes:
1. Data collection success rate
2. Game outcomes (resources, cooperation levels)
3. Trajectory statistics
4. Data quality for world model training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pickle
import numpy as np
from collections import defaultdict


def analyze_results(results_dir):
    """Analyze experiment results."""
    results_dir = Path(results_dir)

    print("=" * 70)
    print("ULTRA PARALLEL EXPERIMENT ANALYSIS")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print()

    # ========================================================================
    # 1. LOAD SUMMARIES
    # ========================================================================
    train_summaries = []
    test_summaries = []

    for summary_file in sorted(results_dir.glob("training_summary_*.json")):
        with open(summary_file) as f:
            train_summaries.append(json.load(f))

    for summary_file in sorted(results_dir.glob("testing_summary_*.json")):
        with open(summary_file) as f:
            test_summaries.append(json.load(f))

    print(f"[1] GAME SUMMARIES")
    print(f"  Training games: {len(train_summaries)}")
    print(f"  Test games: {len(test_summaries)}")
    print()

    # ========================================================================
    # 2. ANALYZE GAME OUTCOMES
    # ========================================================================
    print(f"[2] GAME OUTCOMES")
    print()

    print(f"  TRAINING GAMES:")
    train_resources = [s['final_resources'] for s in train_summaries]
    train_rounds = [s['rounds_completed'] for s in train_summaries]

    print(f"    Final resources: {np.mean(train_resources):.2f} ± {np.std(train_resources):.2f}")
    print(f"    Min: {np.min(train_resources):.2f} | Max: {np.max(train_resources):.2f}")
    print(f"    Rounds completed: {np.mean(train_rounds):.1f}")
    print()

    # Cooperation analysis (resources > 50 = sustainable, < 50 = collapse)
    sustainable = sum(1 for r in train_resources if r > 50)
    collapsed = sum(1 for r in train_resources if r <= 10)

    print(f"    Cooperation outcomes:")
    print(f"      Sustainable (>50): {sustainable}/{len(train_resources)} ({sustainable/len(train_resources)*100:.1f}%)")
    print(f"      Collapsed (≤10): {collapsed}/{len(train_resources)} ({collapsed/len(train_resources)*100:.1f}%)")
    print()

    print(f"  TEST GAMES:")
    test_resources = [s['final_resources'] for s in test_summaries]
    test_rounds = [s['rounds_completed'] for s in test_summaries]

    print(f"    Final resources: {np.mean(test_resources):.2f} ± {np.std(test_resources):.2f}")
    print(f"    Min: {np.min(test_resources):.2f} | Max: {np.max(test_resources):.2f}")
    print(f"    Rounds completed: {np.mean(test_rounds):.1f}")
    print()

    sustainable_test = sum(1 for r in test_resources if r > 50)
    collapsed_test = sum(1 for r in test_resources if r <= 10)

    print(f"    Cooperation outcomes:")
    print(f"      Sustainable (>50): {sustainable_test}/{len(test_resources)} ({sustainable_test/len(test_resources)*100:.1f}%)")
    print(f"      Collapsed (≤10): {collapsed_test}/{len(test_resources)} ({collapsed_test/len(test_resources)*100:.1f}%)")
    print()

    # ========================================================================
    # 3. LOAD AND ANALYZE TRAJECTORIES
    # ========================================================================
    print(f"[3] TRAJECTORY DATA")
    print()

    # Load consolidated trajectories
    train_path = results_dir / "train_trajectories.pkl"
    test_path = results_dir / "test_trajectories.pkl"

    with open(train_path, 'rb') as f:
        train_trajectories = pickle.load(f)

    with open(test_path, 'rb') as f:
        test_trajectories = pickle.load(f)

    print(f"  TRAINING DATA:")
    print(f"    Trajectories: {len(train_trajectories)}")
    print(f"    Expected: {len(train_summaries) * 4} (10 games × 4 agents)")

    # Analyze trajectory lengths
    train_lengths = [len(t.transitions) for t in train_trajectories]
    print(f"    Trajectory lengths: {np.mean(train_lengths):.1f} ± {np.std(train_lengths):.1f}")
    print(f"    Min: {np.min(train_lengths)} | Max: {np.max(train_lengths)}")

    # Total transitions (state-action pairs)
    total_train_transitions = sum(train_lengths)
    print(f"    Total transitions: {total_train_transitions:,}")
    print()

    print(f"  TEST DATA:")
    print(f"    Trajectories: {len(test_trajectories)}")
    print(f"    Expected: {len(test_summaries) * 4} (5 games × 4 agents)")

    test_lengths = [len(t.transitions) for t in test_trajectories]
    print(f"    Trajectory lengths: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")
    print(f"    Min: {np.min(test_lengths)} | Max: {np.max(test_lengths)}")

    total_test_transitions = sum(test_lengths)
    print(f"    Total transitions: {total_test_transitions:,}")
    print()

    # ========================================================================
    # 4. ANALYZE BEHAVIORAL PATTERNS
    # ========================================================================
    print(f"[4] BEHAVIORAL PATTERNS")
    print()

    # Extract harvest amounts from training data
    train_harvests = []
    for traj in train_trajectories:
        for transition in traj.transitions:
            harvest = transition.action.extra.get('harvest_amount', 0)
            train_harvests.append(harvest)

    print(f"  HARVEST BEHAVIOR (Training):")
    print(f"    Mean harvest: {np.mean(train_harvests):.2f}")
    print(f"    Std dev: {np.std(train_harvests):.2f}")
    print(f"    Min: {np.min(train_harvests)} | Max: {np.max(train_harvests)}")

    # Histogram
    bins = [0, 2, 4, 6, 8, 10]
    hist, _ = np.histogram(train_harvests, bins=bins)
    print(f"\n    Distribution:")
    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(train_harvests) * 100
        bar = "█" * int(pct / 2)
        print(f"      {bins[i]}-{bins[i+1]}: {count:5d} ({pct:5.1f}%) {bar}")
    print()

    # Cooperation vs defection
    cooperative = sum(1 for h in train_harvests if h <= 3)
    moderate = sum(1 for h in train_harvests if 3 < h <= 6)
    greedy = sum(1 for h in train_harvests if h > 6)

    print(f"  STRATEGY CLASSIFICATION:")
    print(f"    Cooperative (≤3): {cooperative:5d} ({cooperative/len(train_harvests)*100:5.1f}%)")
    print(f"    Moderate (4-6):   {moderate:5d} ({moderate/len(train_harvests)*100:5.1f}%)")
    print(f"    Greedy (>6):      {greedy:5d} ({greedy/len(train_harvests)*100:5.1f}%)")
    print()

    # ========================================================================
    # 5. WORLD MODEL TRAINING READINESS
    # ========================================================================
    print(f"[5] WORLD MODEL TRAINING READINESS")
    print()

    # Check data format
    sample_traj = train_trajectories[0]
    sample_transition = sample_traj.transitions[0]

    print(f"  DATA FORMAT CHECK:")
    print(f"    ✓ Observation fields: {list(sample_transition.observation.__dict__.keys())}")
    print(f"    ✓ Action fields: {list(sample_transition.action.__dict__.keys())}")
    print(f"    ✓ Transition has reward: {hasattr(sample_transition, 'reward')}")
    print(f"    ✓ Cross-agent data: {'all_actions' in sample_transition.action.extra}")
    print()

    print(f"  DATASET SIZE:")
    print(f"    Training transitions: {total_train_transitions:,}")
    print(f"    Test transitions: {total_test_transitions:,}")
    print(f"    Total: {total_train_transitions + total_test_transitions:,}")
    print()

    # Estimate world model training time
    # Assume ~1000 transitions/sec on CPU
    estimated_epochs = 100
    estimated_time = (total_train_transitions * estimated_epochs) / 1000 / 60

    print(f"  ESTIMATED TRAINING TIME:")
    print(f"    @ 1000 transitions/sec × {estimated_epochs} epochs")
    print(f"    Training time: ~{estimated_time:.1f} minutes")
    print()

    # ========================================================================
    # 6. SUMMARY & RECOMMENDATIONS
    # ========================================================================
    print(f"[6] SUMMARY & RECOMMENDATIONS")
    print()

    print(f"  STATUS:")
    all_games_complete = len(train_summaries) == 10 and len(test_summaries) == 5
    all_trajectories_present = len(train_trajectories) == 40 and len(test_trajectories) == 20

    if all_games_complete and all_trajectories_present:
        print(f"    ✓ All games completed successfully")
        print(f"    ✓ All trajectories collected")
        print(f"    ✓ Data ready for world model training")
    else:
        print(f"    ✗ Some games may have failed")
        print(f"      Expected: 10 train + 5 test games")
        print(f"      Found: {len(train_summaries)} train + {len(test_summaries)} test")
    print()

    print(f"  NEXT STEPS:")
    print(f"    1. Train world model on training data ({total_train_transitions:,} transitions)")
    print(f"    2. Evaluate on test data ({total_test_transitions:,} transitions)")
    print(f"    3. Extract behavioral fingerprints")
    print(f"    4. Analyze prediction errors for misalignment detection")
    print()

    print(f"  KEY FINDINGS:")
    avg_cooperation = (sustainable + sustainable_test) / (len(train_summaries) + len(test_summaries))
    print(f"    - Cooperation rate: {avg_cooperation*100:.1f}% sustained resources")
    print(f"    - Mean harvest: {np.mean(train_harvests):.2f} (sustainable ≈ 2.5)")
    print(f"    - Behavioral diversity: {np.std(train_harvests):.2f} std dev")
    print()

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True, help='Results directory')
    args = parser.parse_args()

    analyze_results(args.results_dir)
