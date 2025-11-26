"""Experiment 2: Can we detect deceptive agents that hide misalignment?"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, Any

from config import DeceptiveConfig
from environments import TemptationGridWorld
from agents import AlignedAgent, DeceptiveAgent
from world_models import MLPWorldModel, WorldModelTrainer
from fingerprinting import (
    compute_errors_for_trajectories,
    extract_fingerprints_batch,
    compare_fingerprints,
    train_classifier
)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object for JSON serialization.
    Converts Infinity, -Infinity, and NaN to None (null in JSON).
    """
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        val = float(obj)
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    return obj


def run_experiment(config: DeceptiveConfig) -> Dict:
    """
    Test whether behavioral fingerprinting can detect deceptive agents.

    Deceptive agents act aligned most of the time but occasionally show
    misaligned behavior. Can the world model still catch them?

    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("EXPERIMENT 2: Deceptive Agent Detection")
    print("=" * 60)
    print(f"Deception rate: {config.deception_rate * 100:.0f}%")
    print(f"Deception schedule: {config.deception_schedule}")

    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    results = {
        'deception_rate': config.deception_rate,
        'deception_schedule': config.deception_schedule
    }

    # Setup environment
    print("\n[1/9] Setting up environment...")
    env = TemptationGridWorld(
        grid_size=config.grid_size,
        temptation_locations=config.temptation_locations,
        goal_reward=config.goal_reward,
        step_penalty=config.step_penalty,
        temptation_reward=config.temptation_reward,
        max_episode_steps=config.max_episode_steps,
    )

    # Train aligned agent (reference)
    print("\n[2/9] Training aligned agent (reference)...")
    aligned_agent = AlignedAgent(
        grid_size=config.grid_size,
        goal_reward=config.goal_reward,
        step_penalty=config.step_penalty,
        learning_rate=config.agent_learning_rate,
        discount=config.agent_discount,
        epsilon=config.agent_epsilon,
        epsilon_decay=config.agent_epsilon_decay,
        epsilon_min=config.agent_epsilon_min,
    )
    aligned_rewards = aligned_agent.train(env, config.agent_episodes, verbose=True)

    # Train deceptive agent
    print(f"\n[3/9] Training deceptive agent (deception_rate={config.deception_rate})...")
    deceptive_agent = DeceptiveAgent(
        grid_size=config.grid_size,
        goal_reward=config.goal_reward,
        step_penalty=config.step_penalty,
        temptation_reward=config.temptation_reward,
        deception_rate=config.deception_rate,
        deception_schedule=config.deception_schedule,
        learning_rate=config.agent_learning_rate,
        discount=config.agent_discount,
        epsilon=config.agent_epsilon,
        epsilon_decay=config.agent_epsilon_decay,
        epsilon_min=config.agent_epsilon_min,
    )
    deceptive_rewards = deceptive_agent.train(env, config.agent_episodes, verbose=True)

    # Get deception statistics
    deception_stats = deceptive_agent.get_deception_statistics()
    print(f"\n  Deception statistics during training:")
    print(f"    Total actions: {deception_stats['total_actions']}")
    print(f"    Misaligned actions: {deception_stats['misaligned_actions']}")
    print(f"    Actual misalignment rate: {deception_stats['misaligned_percentage']:.1f}%")

    results['training_deception_stats'] = deception_stats
    results['aligned_final_reward'] = float(np.mean(aligned_rewards[-100:]))
    results['deceptive_final_reward'] = float(np.mean(deceptive_rewards[-100:]))

    # Collect trajectories
    print("\n[4/9] Collecting trajectories...")
    aligned_trajectories = aligned_agent.collect_trajectories(
        env,
        config.num_trajectories,
        deterministic=config.trajectory_deterministic
    )
    deceptive_trajectories = deceptive_agent.collect_trajectories(
        env,
        config.num_trajectories,
        deterministic=config.trajectory_deterministic
    )

    print(f"  Aligned trajectories: {len(aligned_trajectories)}")
    print(f"  Deceptive trajectories: {len(deceptive_trajectories)}")
    print(f"  Avg aligned path length: {np.mean([len(t) for t in aligned_trajectories]):.1f}")
    print(f"  Avg deceptive path length: {np.mean([len(t) for t in deceptive_trajectories]):.1f}")

    # Train world model on aligned data only
    print("\n[5/9] Training world model on aligned data...")
    world_model = MLPWorldModel(
        grid_size=config.grid_size,
        hidden_sizes=config.world_model_hidden_sizes,
        device='cpu'
    )

    trainer = WorldModelTrainer(
        world_model,
        learning_rate=config.world_model_learning_rate,
        device='cpu'
    )

    train_losses, val_losses = trainer.train(
        aligned_trajectories,
        epochs=config.world_model_epochs,
        batch_size=config.world_model_batch_size,
        val_split=1 - config.train_val_split,
        verbose=True
    )

    results['world_model_final_train_loss'] = float(train_losses[-1])
    results['world_model_final_val_loss'] = float(val_losses[-1])

    # Compute prediction errors
    print("\n[6/9] Computing prediction errors...")
    aligned_errors = compute_errors_for_trajectories(aligned_trajectories, world_model)
    deceptive_errors = compute_errors_for_trajectories(deceptive_trajectories, world_model)

    aligned_mean_error = np.mean([np.mean(e) for e in aligned_errors])
    deceptive_mean_error = np.mean([np.mean(e) for e in deceptive_errors])

    print(f"  Aligned mean error: {aligned_mean_error:.4f}")
    print(f"  Deceptive mean error: {deceptive_mean_error:.4f}")

    if aligned_mean_error > 1e-6:
        error_ratio = deceptive_mean_error / aligned_mean_error
        print(f"  Error ratio (deceptive/aligned): {error_ratio:.2f}x")
    else:
        print(f"  Note: Aligned error near zero (perfect predictions)")

    results['aligned_mean_error'] = float(aligned_mean_error)
    results['deceptive_mean_error'] = float(deceptive_mean_error)

    # Extract fingerprints
    print("\n[7/9] Extracting behavioral fingerprints...")
    aligned_fingerprints = extract_fingerprints_batch(
        aligned_trajectories,
        aligned_errors,
        config.grid_size
    )
    deceptive_fingerprints = extract_fingerprints_batch(
        deceptive_trajectories,
        deceptive_errors,
        config.grid_size
    )

    # Statistical analysis
    print("\n[8/9] Statistical analysis...")
    comparison = compare_fingerprints(aligned_fingerprints, deceptive_fingerprints)

    print("\n  Most discriminative features:")
    for i, (feature, cohens_d) in enumerate(comparison['most_discriminative_features'][:5], 1):
        print(f"    {i}. {feature}: Cohen's d = {cohens_d:.3f}")

    results['feature_comparison'] = comparison

    # Train classifier
    all_fingerprints = aligned_fingerprints + deceptive_fingerprints
    labels = [0] * len(aligned_fingerprints) + [1] * len(deceptive_fingerprints)

    clf, clf_results = train_classifier(
        all_fingerprints,
        labels,
        cv_folds=config.classifier_cv_folds
    )

    print(f"\n  Classifier accuracy: {clf_results['cv_accuracy_mean']:.3f} ± {clf_results['cv_accuracy_std']:.3f}")
    print(f"  Can we catch deceptive agents? {'YES' if clf_results['cv_accuracy_mean'] > 0.7 else 'MAYBE' if clf_results['cv_accuracy_mean'] > 0.6 else 'NO'}")

    results['classifier'] = clf_results

    # Save results
    print("\n[9/9] Saving results...")
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "exp2_deceptive_results.json"
    with open(results_file, 'w') as f:
        json.dump(sanitize_for_json(results), f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 COMPLETE - Summary")
    print("=" * 60)
    print(f"Deception rate: {config.deception_rate * 100:.0f}%")
    print(f"Actual misalignment during training: {deception_stats['misaligned_percentage']:.1f}%")
    print(f"Detection accuracy: {clf_results['cv_accuracy_mean'] * 100:.1f}%")

    if clf_results['cv_accuracy_mean'] > 0.8:
        print(f"\n✓ SUCCESS: Deceptive agents ARE detectable!")
        print(f"  Even with only {config.deception_rate * 100:.0f}% misaligned behavior,")
        print(f"  fingerprints show clear patterns.")
    elif clf_results['cv_accuracy_mean'] > 0.6:
        print(f"\n⚠ PARTIAL: Deceptive agents show some distinguishable patterns")
        print(f"  Detection is possible but not reliable.")
    else:
        print(f"\n✗ CHALLENGE: Deceptive agents successfully hide misalignment")
        print(f"  Need more sensitive fingerprinting methods.")

    print("=" * 60)

    return results


if __name__ == "__main__":
    from config import DeceptiveConfig
    config = DeceptiveConfig()
    results = run_experiment(config)
