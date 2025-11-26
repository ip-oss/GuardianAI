"""Experiment 1: Basic behavioral fingerprinting - aligned vs misaligned agents."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, Any

from config import ExperimentConfig
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


def run_experiment(config: ExperimentConfig) -> Dict:
    """
    Run basic fingerprinting experiment.

    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("EXPERIMENT 1: Basic Behavioral Fingerprinting")
    print("=" * 60)

    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    results = {}

    # ==========================
    # 1. Setup Environment
    # ==========================
    print("\n[1/8] Setting up environment...")
    env = TemptationGridWorld(
        grid_size=config.grid_size,
        temptation_locations=config.temptation_locations,
        goal_reward=config.goal_reward,
        step_penalty=config.step_penalty,
        temptation_reward=config.temptation_reward,
        max_episode_steps=config.max_episode_steps,
    )
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Temptations: {len(config.temptation_locations)}")
    print(f"  Optimal path length: {env.get_optimal_path_length()}")

    # ==========================
    # 2. Train Agents
    # ==========================
    print("\n[2/8] Training agents...")

    # Aligned agent
    print("  Training aligned agent...")
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

    # Misaligned agent
    print("  Training misaligned agent...")
    misaligned_agent = MisalignedAgent(
        grid_size=config.grid_size,
        goal_reward=config.goal_reward,
        step_penalty=config.step_penalty,
        temptation_reward=config.temptation_reward,
        learning_rate=config.agent_learning_rate,
        discount=config.agent_discount,
        epsilon=config.agent_epsilon,
        epsilon_decay=config.agent_epsilon_decay,
        epsilon_min=config.agent_epsilon_min,
    )
    misaligned_rewards = misaligned_agent.train(env, config.agent_episodes, verbose=True)

    results['aligned_final_reward'] = float(np.mean(aligned_rewards[-100:]))
    results['misaligned_final_reward'] = float(np.mean(misaligned_rewards[-100:]))

    print(f"  Aligned agent final reward: {results['aligned_final_reward']:.2f}")
    print(f"  Misaligned agent final reward: {results['misaligned_final_reward']:.2f}")

    # ==========================
    # 3. Collect Trajectories
    # ==========================
    print("\n[3/8] Collecting trajectories...")
    aligned_trajectories = aligned_agent.collect_trajectories(
        env,
        config.num_trajectories,
        deterministic=config.trajectory_deterministic
    )
    misaligned_trajectories = misaligned_agent.collect_trajectories(
        env,
        config.num_trajectories,
        deterministic=config.trajectory_deterministic
    )

    print(f"  Aligned trajectories: {len(aligned_trajectories)}")
    print(f"  Misaligned trajectories: {len(misaligned_trajectories)}")
    print(f"  Avg aligned path length: {np.mean([len(t) for t in aligned_trajectories]):.1f}")
    print(f"  Avg misaligned path length: {np.mean([len(t) for t in misaligned_trajectories]):.1f}")

    # ==========================
    # 4. Train World Model
    # ==========================
    print("\n[4/8] Training world model...")
    print(f"  Training ONLY on {len(aligned_trajectories)} aligned trajectories")

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

    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Final val loss: {val_losses[-1]:.6f}")

    # ==========================
    # 5. Compute Prediction Errors
    # ==========================
    print("\n[5/8] Computing prediction errors...")
    aligned_errors = compute_errors_for_trajectories(aligned_trajectories, world_model)
    misaligned_errors = compute_errors_for_trajectories(misaligned_trajectories, world_model)

    aligned_mean_error = np.mean([np.mean(e) for e in aligned_errors])
    misaligned_mean_error = np.mean([np.mean(e) for e in misaligned_errors])

    print(f"  Aligned mean error: {aligned_mean_error:.4f}")
    print(f"  Misaligned mean error: {misaligned_mean_error:.4f}")

    if aligned_mean_error > 1e-6:
        error_ratio = misaligned_mean_error / aligned_mean_error
        print(f"  Error ratio (misaligned/aligned): {error_ratio:.2f}x")
    else:
        print(f"  Error ratio: Aligned error too small (perfect predictions!)")

    results['aligned_mean_error'] = float(aligned_mean_error)
    results['misaligned_mean_error'] = float(misaligned_mean_error)

    # ==========================
    # 6. Extract Fingerprints
    # ==========================
    print("\n[6/8] Extracting behavioral fingerprints...")
    aligned_fingerprints = extract_fingerprints_batch(
        aligned_trajectories,
        aligned_errors,
        config.grid_size
    )
    misaligned_fingerprints = extract_fingerprints_batch(
        misaligned_trajectories,
        misaligned_errors,
        config.grid_size
    )

    print(f"  Extracted {len(aligned_fingerprints)} aligned fingerprints")
    print(f"  Extracted {len(misaligned_fingerprints)} misaligned fingerprints")
    print(f"  Features per fingerprint: {len(aligned_fingerprints[0])}")

    # ==========================
    # 7. Statistical Analysis
    # ==========================
    print("\n[7/8] Statistical analysis...")
    comparison = compare_fingerprints(aligned_fingerprints, misaligned_fingerprints)

    print("\n  Most discriminative features:")
    for i, (feature, cohens_d) in enumerate(comparison['most_discriminative_features'][:5], 1):
        print(f"    {i}. {feature}: Cohen's d = {cohens_d:.3f}")

    results['feature_comparison'] = comparison

    # Train classifier
    all_fingerprints = aligned_fingerprints + misaligned_fingerprints
    labels = [0] * len(aligned_fingerprints) + [1] * len(misaligned_fingerprints)

    clf, clf_results = train_classifier(
        all_fingerprints,
        labels,
        cv_folds=config.classifier_cv_folds
    )

    print(f"\n  Classifier accuracy: {clf_results['cv_accuracy_mean']:.3f} ± {clf_results['cv_accuracy_std']:.3f}")
    print(f"  Top predictive features:")
    for i, (feature, importance) in enumerate(clf_results['top_features'][:5], 1):
        print(f"    {i}. {feature}: {importance:.3f}")

    results['classifier'] = clf_results

    # ==========================
    # 8. Save Results
    # ==========================
    print("\n[8/8] Saving results...")
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "exp1_results.json"
    with open(results_file, 'w') as f:
        json.dump(sanitize_for_json(results), f, indent=2)

    print(f"  Results saved to {results_file}")

    # ==========================
    # Summary
    # ==========================
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE - Summary")
    print("=" * 60)
    print(f"✓ Agents trained successfully")
    print(f"✓ World model trained on aligned data only")
    print(f"✓ Misaligned agents show {misaligned_mean_error/aligned_mean_error:.2f}x higher prediction error")
    print(f"✓ Classifier achieves {clf_results['cv_accuracy_mean']*100:.1f}% accuracy")
    print(f"\n{'SUCCESS' if clf_results['cv_accuracy_mean'] > 0.8 else 'NEEDS TUNING'}: "
          f"Behavioral fingerprints {'ARE' if clf_results['cv_accuracy_mean'] > 0.8 else 'MAY NOT BE'} "
          f"clearly distinguishable")
    print("=" * 60)

    return results


if __name__ == "__main__":
    from config import ExperimentConfig
    config = ExperimentConfig()
    results = run_experiment(config)
