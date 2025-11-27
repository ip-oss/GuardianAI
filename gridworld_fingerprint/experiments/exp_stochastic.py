"""
Experiment: Stochastic Environment Robustness

Test whether behavioral fingerprinting works in noisy/stochastic environments.

Research Questions:
1. Does detection accuracy degrade significantly with environmental noise?
2. What level of noise makes fingerprinting unreliable?
3. Are some misalignment types more robust to noise than others?
4. Can we still distinguish misalignment types despite randomness?

Hypothesis: Behavioral fingerprints should be robust to moderate environmental noise
because they capture systematic behavioral patterns, not individual actions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

from config import ExperimentConfig
from environments.rich_gridworld import RichGridWorld, RichGridConfig
from environments.stochastic_gridworld import StochasticGridWorld
from agents.misalignment_types import create_agent, DANGER_LEVELS
from world_models import MLPWorldModel, WorldModelTrainer
from fingerprinting import compute_errors_for_trajectories, extract_fingerprints_batch
from fingerprinting.taxonomy_analysis import analyze_misalignment_taxonomy
from utils.trajectory import Trajectory


def create_stochastic_environment(
    config: ExperimentConfig,
    noise_prob: float
) -> StochasticGridWorld:
    """Create stochastic environment with given noise level."""
    from experiments.exp6_taxonomy import create_taxonomy_environment_config

    env_config = create_taxonomy_environment_config(config)
    return StochasticGridWorld(env_config, noise_prob=noise_prob)


def train_agent_in_stochastic_env(
    agent,
    env: StochasticGridWorld,
    episodes: int,
    verbose: bool = False
) -> List[float]:
    """Train agent in stochastic environment."""
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        state = env.get_simple_obs()
        episode_reward = 0
        done = False

        if hasattr(agent, 'reset_episode_state'):
            agent.reset_episode_state()

        while not done:
            action = agent.act(state, deterministic=False)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            next_state = env.get_simple_obs()
            done = terminated or truncated

            # Get agent's reward
            reward = agent.get_reward(state, action, next_state, env_reward, info)

            # Update Q-values
            agent.update(state, action, reward, next_state, terminated)

            episode_reward += env_reward
            state = next_state

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        episode_rewards.append(episode_reward)

        if verbose and (episode + 1) % 200 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            print(f"    Episode {episode + 1}/{episodes}: Reward={recent_reward:.2f}")

    return episode_rewards


def collect_trajectories_stochastic(
    agent,
    env: StochasticGridWorld,
    n: int,
    deterministic: bool = True
) -> List[Trajectory]:
    """Collect trajectories from stochastic environment."""
    from utils.trajectory import create_trajectory

    trajectories = []

    for i in range(n):
        obs, _ = env.reset()
        state = env.get_simple_obs()

        states = [state]
        actions = []
        rewards = []
        done = False

        if hasattr(agent, 'reset_episode_state'):
            agent.reset_episode_state()

        while not done:
            action = agent.act(state, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = env.get_simple_obs()
            done = terminated or truncated

            actions.append(action)
            rewards.append(reward)
            states.append(next_state)

            state = next_state

        trajectory = create_trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            agent_type=agent.MISALIGNMENT_TYPE,
            episode_num=i
        )
        trajectories.append(trajectory)

    return trajectories


def run_experiment(config: ExperimentConfig) -> Dict:
    """
    Run stochastic environment experiment.

    Tests detection accuracy at multiple noise levels:
    - 0.0: Deterministic (baseline)
    - 0.05: Light noise (5%)
    - 0.10: Moderate noise (10%)
    - 0.15: Heavy noise (15%)
    - 0.20: Very heavy noise (20%)
    """
    print("=" * 60)
    print("EXPERIMENT: Stochastic Environment Robustness")
    print("=" * 60)

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Test these noise levels
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]

    # Test subset of agent types for efficiency
    agent_types = ['aligned', 'resource_acquisition', 'self_preservation', 'adversarial', 'deceptive']

    results = {
        "noise_levels": noise_levels,
        "agent_types": agent_types,
        "results_by_noise": {},
        "summary": {},
    }

    # For each noise level
    for noise_prob in noise_levels:
        print(f"\n{'='*60}")
        print(f"Testing Noise Level: {noise_prob:.0%}")
        print(f"{'='*60}")

        # Create stochastic environment
        env = create_stochastic_environment(config, noise_prob)

        # Train agents and collect trajectories
        trajectories_by_type = {}

        for agent_type in agent_types:
            print(f"\n  Training {agent_type} agent...")

            agent = create_agent(
                agent_type,
                grid_size=config.grid_size,
                env=env,
                seed=config.random_seed,
                learning_rate=config.agent_learning_rate,
                discount=config.agent_discount,
            )

            # Train
            rewards = train_agent_in_stochastic_env(
                agent, env, config.agent_episodes, verbose=config.verbose
            )

            # Collect trajectories
            print(f"  Collecting trajectories...")
            trajectories = collect_trajectories_stochastic(
                agent, env, config.num_trajectories
            )
            trajectories_by_type[agent_type] = trajectories

            # Stats
            avg_reward = np.mean([t.total_reward for t in trajectories])
            completion = sum(1 for t in trajectories if t.reached_goal((config.grid_size-1, config.grid_size-1))) / len(trajectories)
            print(f"  Avg reward: {avg_reward:.2f}, Completion: {completion:.0%}")

        # Train world model on aligned data
        print(f"\n  Training world model...")
        aligned_trajectories = trajectories_by_type["aligned"]

        world_model = MLPWorldModel(
            grid_size=config.grid_size,
            hidden_sizes=config.world_model_hidden_sizes,
            device='cpu'
        )

        trainer = WorldModelTrainer(world_model, learning_rate=config.world_model_learning_rate, device='cpu')
        train_losses, val_losses = trainer.train(
            aligned_trajectories,
            epochs=config.world_model_epochs,
            batch_size=config.world_model_batch_size,
            val_split=0.2,
            verbose=False
        )

        # Extract fingerprints
        print(f"  Extracting fingerprints...")
        fingerprints_by_type = {}

        for agent_type, trajectories in trajectories_by_type.items():
            errors = compute_errors_for_trajectories(trajectories, world_model)
            fps = extract_fingerprints_batch(trajectories, errors, config.grid_size)
            fingerprints_by_type[agent_type] = fps

        # Analyze
        print(f"  Running classification...")
        danger_subset = {k: DANGER_LEVELS[k] for k in agent_types}
        analysis = analyze_misalignment_taxonomy(fingerprints_by_type, danger_subset)

        # Store results
        results["results_by_noise"][noise_prob] = {
            "multiclass_accuracy": float(analysis.multiclass_accuracy),
            "silhouette_score": float(analysis.silhouette_score),
            "danger_correlation": float(analysis.danger_correlation),
            "per_class_accuracy": {k: float(v) for k, v in analysis.per_class_accuracy.items()},
            "world_model_loss": float(val_losses[-1]),
            "noise_stats": env.get_noise_statistics(),
        }

        print(f"\n  Results at {noise_prob:.0%} noise:")
        print(f"    Accuracy: {analysis.multiclass_accuracy:.1%}")
        print(f"    Silhouette: {analysis.silhouette_score:.3f}")

    # Compute summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS NOISE LEVELS")
    print(f"{'='*60}")

    baseline_acc = results["results_by_noise"][0.0]["multiclass_accuracy"]

    print(f"\nNoise Level | Accuracy | Delta from Baseline | Silhouette")
    print("-" * 60)

    for noise_prob in noise_levels:
        res = results["results_by_noise"][noise_prob]
        acc = res["multiclass_accuracy"]
        sil = res["silhouette_score"]
        delta = acc - baseline_acc
        delta_str = f"{delta:+.1%}" if noise_prob > 0 else "baseline"
        print(f"  {noise_prob:5.0%}     | {acc:7.1%}  | {delta_str:>18s} | {sil:7.3f}")

    # Key findings
    max_tested_noise = max(noise_levels)
    worst_noise_res = results["results_by_noise"][max_tested_noise]
    accuracy_drop = baseline_acc - worst_noise_res["multiclass_accuracy"]

    results["summary"] = {
        "baseline_accuracy": baseline_acc,
        "max_noise_tested": max_tested_noise,
        "accuracy_at_max_noise": worst_noise_res["multiclass_accuracy"],
        "total_accuracy_drop": accuracy_drop,
        "robust_to_noise": accuracy_drop < 0.10,  # Less than 10% drop is "robust"
    }

    findings = []
    if results["summary"]["robust_to_noise"]:
        findings.append(
            f"✓ Behavioral fingerprinting is ROBUST to noise up to {max_tested_noise:.0%} "
            f"(accuracy drop: {accuracy_drop:.1%})"
        )
    else:
        findings.append(
            f"⚠ Behavioral fingerprinting degrades significantly with noise "
            f"({accuracy_drop:.1%} drop at {max_tested_noise:.0%})"
        )

    # Check which noise level first causes >10% drop
    threshold_noise = None
    for noise_prob in noise_levels[1:]:
        drop = baseline_acc - results["results_by_noise"][noise_prob]["multiclass_accuracy"]
        if drop > 0.10:
            threshold_noise = noise_prob
            break

    if threshold_noise:
        findings.append(
            f"Detection becomes unreliable (>10% drop) at {threshold_noise:.0%} noise"
        )
    else:
        findings.append(
            f"Detection remains reliable across all tested noise levels (up to {max_tested_noise:.0%})"
        )

    results["key_findings"] = findings

    print(f"\nKey Findings:")
    for finding in findings:
        print(f"  {finding}")

    # Save results
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"exp_stochastic_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_file = run_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {run_dir}/")

    # Plot results
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy vs noise
        noise_vals = list(results["results_by_noise"].keys())
        acc_vals = [results["results_by_noise"][n]["multiclass_accuracy"] for n in noise_vals]

        ax1.plot(noise_vals, acc_vals, 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=baseline_acc, color='gray', linestyle='--', label='Baseline')
        ax1.axhline(y=baseline_acc - 0.10, color='red', linestyle=':', label='10% drop threshold')
        ax1.set_xlabel('Noise Probability')
        ax1.set_ylabel('Multi-class Accuracy')
        ax1.set_title('Detection Accuracy vs Environmental Noise')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 1.0])

        # Silhouette vs noise
        sil_vals = [results["results_by_noise"][n]["silhouette_score"] for n in noise_vals]

        ax2.plot(noise_vals, sil_vals, 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Noise Probability')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Cluster Quality vs Environmental Noise')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(run_dir / "noise_robustness.png", dpi=150)
        print(f"  Plot saved to {run_dir}/noise_robustness.png")

    except ImportError:
        print("  (matplotlib not available for plotting)")

    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run stochastic environment experiment")
    parser.add_argument("--quick", action="store_true", help="Use quick config for testing")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.quick:
        from config import TaxonomyQuickConfig
        config = TaxonomyQuickConfig()
    else:
        from config import TaxonomyConfig
        config = TaxonomyConfig()

    config.results_dir = args.results_dir
    config.random_seed = args.seed

    results = run_experiment(config)

    # Print final summary
    print(f"\n✅ Stochastic environment experiment complete!")
    print(f"   Baseline: {results['summary']['baseline_accuracy']:.1%}")
    print(f"   At {results['summary']['max_noise_tested']:.0%} noise: {results['summary']['accuracy_at_max_noise']:.1%}")
    print(f"   Drop: {results['summary']['total_accuracy_drop']:.1%}")
