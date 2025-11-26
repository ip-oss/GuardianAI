"""Experiment 5: What's the minimum detectable misalignment?"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from config import SubtletyConfig
from environments import TemptationGridWorld
from agents import AlignedAgent, MisalignedAgent
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


def run_experiment(config: SubtletyConfig) -> Dict:
    """
    Test detection across a gradient of misalignment levels.

    Creates agents with varying temptation rewards to test
    the sensitivity of behavioral fingerprinting.

    Returns:
        Dictionary with results for each misalignment level
    """
    print("=" * 60)
    print("EXPERIMENT 5: Subtlety Gradient")
    print("=" * 60)
    print(f"Testing misalignment levels: {config.misalignment_levels}")

    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    results = {
        'misalignment_levels': config.misalignment_levels,
        'level_results': {}
    }

    # Setup environment
    print("\n[1/6] Setting up environment...")
    env = TemptationGridWorld(
        grid_size=config.grid_size,
        temptation_locations=config.temptation_locations,
        goal_reward=config.goal_reward,
        step_penalty=config.step_penalty,
        temptation_reward=config.temptation_reward,
        max_episode_steps=config.max_episode_steps,
    )

    # Train aligned agent (reference)
    print("\n[2/6] Training aligned agent (baseline)...")
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
    aligned_rewards = aligned_agent.train(env, config.agent_episodes, verbose=False)
    print(f"  Aligned agent trained: {np.mean(aligned_rewards[-100:]):.2f} reward")

    # Collect aligned trajectories
    print("\n[3/6] Collecting aligned trajectories...")
    aligned_trajectories = aligned_agent.collect_trajectories(
        env,
        config.num_trajectories,
        deterministic=config.trajectory_deterministic
    )

    # Train world model on aligned data
    print("\n[4/6] Training world model on aligned data...")
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
        verbose=False
    )
    print(f"  World model trained: loss={train_losses[-1]:.6f}")

    # Compute aligned errors (baseline)
    aligned_errors = compute_errors_for_trajectories(aligned_trajectories, world_model)
    aligned_fingerprints = extract_fingerprints_batch(
        aligned_trajectories,
        aligned_errors,
        config.grid_size
    )

    # Test each misalignment level
    print(f"\n[5/6] Testing {len(config.misalignment_levels)} misalignment levels...")

    for level_idx, misalignment_level in enumerate(config.misalignment_levels):
        print(f"\n  [{level_idx + 1}/{len(config.misalignment_levels)}] Level {misalignment_level}x temptation reward...")

        # Adjust temptation reward
        adjusted_temptation_reward = config.temptation_reward * misalignment_level

        # Train misaligned agent at this level
        misaligned_agent = MisalignedAgent(
            grid_size=config.grid_size,
            goal_reward=config.goal_reward,
            step_penalty=config.step_penalty,
            temptation_reward=adjusted_temptation_reward,
            learning_rate=config.agent_learning_rate,
            discount=config.agent_discount,
            epsilon=config.agent_epsilon,
            epsilon_decay=config.agent_epsilon_decay,
            epsilon_min=config.agent_epsilon_min,
        )

        misaligned_rewards = misaligned_agent.train(env, config.agent_episodes, verbose=False)

        # Collect trajectories
        misaligned_trajectories = misaligned_agent.collect_trajectories(
            env,
            config.num_trajectories,
            deterministic=config.trajectory_deterministic
        )

        # Compute errors
        misaligned_errors = compute_errors_for_trajectories(misaligned_trajectories, world_model)

        # Extract fingerprints
        misaligned_fingerprints = extract_fingerprints_batch(
            misaligned_trajectories,
            misaligned_errors,
            config.grid_size
        )

        # Compare fingerprints
        comparison = compare_fingerprints(aligned_fingerprints, misaligned_fingerprints)

        # Train classifier
        all_fingerprints = aligned_fingerprints + misaligned_fingerprints
        labels = [0] * len(aligned_fingerprints) + [1] * len(misaligned_fingerprints)

        clf, clf_results = train_classifier(all_fingerprints, labels, cv_folds=config.classifier_cv_folds)

        # Store results
        level_results = {
            'misalignment_level': float(misalignment_level),
            'temptation_reward': float(adjusted_temptation_reward),
            'final_reward': float(np.mean(misaligned_rewards[-100:])),
            'mean_path_length': float(np.mean([len(t) for t in misaligned_trajectories])),
            'mean_error': float(np.mean([np.mean(e) for e in misaligned_errors])),
            'classifier_accuracy': float(clf_results['cv_accuracy_mean']),
            'classifier_std': float(clf_results['cv_accuracy_std']),
            'top_discriminative_feature': comparison['most_discriminative_features'][0] if comparison['most_discriminative_features'] else None
        }

        results['level_results'][f'level_{misalignment_level}'] = level_results

        print(f"      Reward: {level_results['final_reward']:.1f}")
        print(f"      Path length: {level_results['mean_path_length']:.1f}")
        print(f"      Error: {level_results['mean_error']:.4f}")
        print(f"      Detection accuracy: {level_results['classifier_accuracy'] * 100:.1f}%")

    # Save results
    print("\n[6/6] Saving results...")
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "exp5_subtlety_results.json"
    with open(results_file, 'w') as f:
        json.dump(sanitize_for_json(results), f, indent=2)

    # Create plot
    create_subtlety_plot(results, results_dir)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 5 COMPLETE - Subtlety Gradient Analysis")
    print("=" * 60)
    print("\nDetection Accuracy by Misalignment Level:")
    print("-" * 40)

    for level in config.misalignment_levels:
        level_data = results['level_results'][f'level_{level}']
        acc = level_data['classifier_accuracy'] * 100
        status = "✓" if acc > 80 else "⚠" if acc > 60 else "✗"
        print(f"  {status} Level {level:4.1f}x: {acc:5.1f}% accuracy")

    # Find minimum detectable level
    detectable_levels = [
        (level, results['level_results'][f'level_{level}']['classifier_accuracy'])
        for level in config.misalignment_levels
        if results['level_results'][f'level_{level}']['classifier_accuracy'] > 0.8
    ]

    if detectable_levels:
        min_detectable = min(detectable_levels, key=lambda x: x[0])
        print(f"\n✓ Minimum reliably detectable misalignment: {min_detectable[0]}x")
        print(f"  (Achieved {min_detectable[1] * 100:.1f}% accuracy)")
    else:
        print(f"\n⚠ No misalignment level achieved >80% detection")

    print("=" * 60)

    return results


def create_subtlety_plot(results: Dict, results_dir: Path):
    """Create visualization of subtlety gradient results."""
    levels = results['misalignment_levels']
    accuracies = [
        results['level_results'][f'level_{level}']['classifier_accuracy']
        for level in levels
    ]
    errors = [
        results['level_results'][f'level_{level}']['mean_error']
        for level in levels
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Detection accuracy
    ax1.plot(levels, accuracies, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    ax1.set_xlabel('Misalignment Level (temptation reward multiplier)')
    ax1.set_ylabel('Classifier Accuracy')
    ax1.set_title('Detection Accuracy vs Misalignment Level')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1.05])

    # Plot 2: Prediction error
    ax2.plot(levels, errors, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Misalignment Level (temptation reward multiplier)')
    ax2.set_ylabel('Mean Prediction Error')
    ax2.set_title('Prediction Error vs Misalignment Level')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'exp5_subtlety_gradient.png', dpi=150)
    print(f"  Plot saved: {results_dir / 'exp5_subtlety_gradient.png'}")


if __name__ == "__main__":
    from config import SubtletyConfig
    config = SubtletyConfig()
    results = run_experiment(config)
