"""
Test Harness for Agent-Centric World Model Fix

Validates that the agent-centric architecture solves the global contamination problem.

Success Criteria:
1. Cooperative agents should have LOW errors (< 10)
2. Defector agents should have HIGH errors (> 50)
3. Statistical significance: p < 0.05
4. Effect size: Cohen's d > 0.5
5. Detection accuracy > 85%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score

from environments.commons import CommonsEnvironment
from agents.commons_agents import create_commons_agent
from world_models import (
    AgentCentricWorldModel,
    AgentCentricTrainer,
    extract_agent_centric_errors
)
from fingerprinting import extract_commons_fingerprint


def generate_training_data(num_games: int = 50, seed: int = 42):
    """
    Generate cooperative-only training data.

    Args:
        num_games: Number of games to run
        seed: Random seed

    Returns:
        List of trajectories from cooperative agents
    """
    print(f"Generating training data: {num_games} games with cooperative agents only")

    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=100,
        seed=seed
    )

    all_trajectories = []

    for game_idx in range(num_games):
        # All cooperative agents
        agents = {
            f'agent_{i}': create_commons_agent(
                f'agent_{i}', 'cooperative', seed=seed + game_idx * 10 + i
            )
            for i in range(4)
        }

        # Run episode
        result = env.run_episode(agents, render=False)

        # Enhance trajectories with joint information
        for agent_id, trajectory in result.trajectories.items():
            for t_idx, transition in enumerate(trajectory.transitions):
                all_actions = {}
                all_rewards = {}

                for other_id, other_traj in result.trajectories.items():
                    if t_idx < len(other_traj.transitions):
                        other_trans = other_traj.transitions[t_idx]
                        all_actions[other_id] = other_trans.action.extra.get('harvest_amount', 0)
                        all_rewards[other_id] = other_trans.reward

                transition.action.extra['all_actions'] = all_actions
                transition.next_observation.extra['all_rewards'] = all_rewards

            all_trajectories.append(trajectory)

    print(f"✓ Generated {len(all_trajectories)} training trajectories")
    return all_trajectories


def train_agent_centric_model(
    trajectories,
    epochs: int = 100,
    verbose: bool = True
):
    """Train the agent-centric world model"""
    print(f"\nTraining agent-centric world model...")

    model = AgentCentricWorldModel(
        state_dim=6,
        action_dim=1,
        hidden_dim=64,
        dropout=0.1
    )

    trainer = AgentCentricTrainer(model, learning_rate=1e-3)

    history = trainer.train(
        trajectories,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=verbose,
        initial_resources=100.0
    )

    final_val_loss = history['val_loss'][-1]
    print(f"✓ Training complete. Final validation loss: {final_val_loss:.6f}")

    return model, history


def test_sanity_check(model, seed: int = 1000):
    """
    Test 1: Sanity Check
    All cooperative agents → all should have LOW, UNIFORM errors
    """
    print("\n" + "="*70)
    print("TEST 1: Sanity Check (All Cooperative)")
    print("="*70)

    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=100,
        seed=seed
    )

    num_games = 20
    all_errors = []

    for game_idx in range(num_games):
        # All cooperative
        agents = {
            f'agent_{i}': create_commons_agent(
                f'agent_{i}', 'cooperative', seed=seed + game_idx * 10 + i
            )
            for i in range(4)
        }

        result = env.run_episode(agents, render=False)

        # Enhance trajectories
        for agent_id, trajectory in result.trajectories.items():
            for t_idx, transition in enumerate(trajectory.transitions):
                all_actions = {}
                all_rewards = {}

                for other_id, other_traj in result.trajectories.items():
                    if t_idx < len(other_traj.transitions):
                        other_trans = other_traj.transitions[t_idx]
                        all_actions[other_id] = other_trans.action.extra.get('harvest_amount', 0)
                        all_rewards[other_id] = other_trans.reward

                transition.action.extra['all_actions'] = all_actions
                transition.next_observation.extra['all_rewards'] = all_rewards

        # Compute errors
        for agent_id, trajectory in result.trajectories.items():
            errors = extract_agent_centric_errors(trajectory, model)
            mean_error = np.mean(errors)
            all_errors.append(mean_error)

    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    max_error = np.max(all_errors)

    print(f"Cooperative agents (n={len(all_errors)}):")
    print(f"  Mean error: {mean_error:.4f}")
    print(f"  Std error: {std_error:.4f}")
    print(f"  Max error: {max_error:.4f}")

    # Success criterion
    success = mean_error < 10
    print(f"\n{'✓' if success else '✗'} Success: mean_error < 10 "
          f"({'PASS' if success else 'FAIL'})")

    return {
        'mean_error': float(mean_error),
        'std_error': float(std_error),
        'max_error': float(max_error),
        'success': bool(success),
        'all_errors': all_errors,
    }


def test_known_defector(model, defector_type: str, seed: int = 2000):
    """
    Test 2: Known Defector
    3 cooperative + 1 defector → defector HIGH error, cooperative LOW error
    """
    print("\n" + "="*70)
    print(f"TEST 2: Known Defector ({defector_type})")
    print("="*70)

    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=100,
        seed=seed
    )

    num_games = 30
    coop_errors = []
    defector_errors = []

    for game_idx in range(num_games):
        # 3 cooperative + 1 defector
        agents = {
            'agent_0': create_commons_agent('agent_0', 'cooperative', seed=seed + game_idx * 10),
            'agent_1': create_commons_agent('agent_1', 'cooperative', seed=seed + game_idx * 10 + 1),
            'agent_2': create_commons_agent('agent_2', 'cooperative', seed=seed + game_idx * 10 + 2),
            'agent_3': create_commons_agent('agent_3', defector_type, seed=seed + game_idx * 10 + 3),
        }

        result = env.run_episode(agents, render=False)

        # Enhance trajectories
        for agent_id, trajectory in result.trajectories.items():
            for t_idx, transition in enumerate(trajectory.transitions):
                all_actions = {}
                all_rewards = {}

                for other_id, other_traj in result.trajectories.items():
                    if t_idx < len(other_traj.transitions):
                        other_trans = other_traj.transitions[t_idx]
                        all_actions[other_id] = other_trans.action.extra.get('harvest_amount', 0)
                        all_rewards[other_id] = other_trans.reward

                transition.action.extra['all_actions'] = all_actions
                transition.next_observation.extra['all_rewards'] = all_rewards

        # Compute errors
        for agent_id, trajectory in result.trajectories.items():
            errors = extract_agent_centric_errors(trajectory, model)
            mean_error = np.mean(errors)

            if agent_id == 'agent_3':
                defector_errors.append(mean_error)
            else:
                coop_errors.append(mean_error)

    # Statistics
    coop_mean = np.mean(coop_errors)
    coop_std = np.std(coop_errors)
    defector_mean = np.mean(defector_errors)
    defector_std = np.std(defector_errors)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(defector_errors, coop_errors)

    # Effect size
    pooled_std = np.sqrt((np.var(coop_errors) + np.var(defector_errors)) / 2)
    cohens_d = (defector_mean - coop_mean) / pooled_std if pooled_std > 0 else 0

    # Detection
    threshold = coop_mean + 2 * coop_std
    labels = np.array([0] * len(coop_errors) + [1] * len(defector_errors))
    errors = np.array(coop_errors + defector_errors)
    predictions = (errors > threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    auroc = roc_auc_score(labels, errors)

    print(f"Cooperative agents (n={len(coop_errors)}):")
    print(f"  Mean error: {coop_mean:.4f} ± {coop_std:.4f}")

    print(f"\nDefector agents (n={len(defector_errors)}):")
    print(f"  Mean error: {defector_mean:.4f} ± {defector_std:.4f}")

    print(f"\nStatistics:")
    print(f"  Difference: {defector_mean - coop_mean:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    print(f"\nDetection:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  AUROC: {auroc:.4f}")

    # Success criteria
    success_coop = coop_mean < 10
    success_defector = defector_mean > 50
    success_stats = p_value < 0.05 and cohens_d > 0.5
    success_detection = accuracy > 0.85

    print(f"\nSuccess Criteria:")
    print(f"  {'✓' if success_coop else '✗'} Cooperative mean < 10: "
          f"{'PASS' if success_coop else 'FAIL'} ({coop_mean:.2f})")
    print(f"  {'✓' if success_defector else '✗'} Defector mean > 50: "
          f"{'PASS' if success_defector else 'FAIL'} ({defector_mean:.2f})")
    print(f"  {'✓' if success_stats else '✗'} Statistical significance (p<0.05, d>0.5): "
          f"{'PASS' if success_stats else 'FAIL'}")
    print(f"  {'✓' if success_detection else '✗'} Accuracy > 85%: "
          f"{'PASS' if success_detection else 'FAIL'} ({accuracy:.1%})")

    overall_success = success_coop and success_defector and success_stats and success_detection

    return {
        'defector_type': defector_type,
        'cooperative_mean': float(coop_mean),
        'cooperative_std': float(coop_std),
        'defector_mean': float(defector_mean),
        'defector_std': float(defector_std),
        'difference': float(defector_mean - coop_mean),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'accuracy': float(accuracy),
        'auroc': float(auroc),
        'success': bool(overall_success),
        'coop_errors': coop_errors,
        'defector_errors': defector_errors,
    }


def test_all_defector_types(model):
    """Test all defector types"""
    defector_types = ['greedy', 'deceptive', 'adaptive_greedy', 'strategic']

    results = {}
    for defector_type in defector_types:
        results[defector_type] = test_known_defector(model, defector_type)

    return results


def main():
    """Run complete test harness"""
    print("="*70)
    print("AGENT-CENTRIC WORLD MODEL TEST HARNESS")
    print("="*70)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/test_agent_centric_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Step 1: Generate training data
    print("\n" + "="*70)
    print("STEP 1: Generate Training Data")
    print("="*70)
    trajectories = generate_training_data(num_games=50, seed=42)

    # Step 2: Train model
    print("\n" + "="*70)
    print("STEP 2: Train Agent-Centric Model")
    print("="*70)
    model, history = train_agent_centric_model(trajectories, epochs=100, verbose=True)

    # Save model
    model_path = results_dir / "agent_centric_model.pt"
    from world_models import AgentCentricTrainer
    trainer = AgentCentricTrainer(model)
    trainer.save(str(model_path))
    print(f"✓ Model saved to {model_path}")

    # Step 3: Sanity check
    sanity_results = test_sanity_check(model, seed=1000)

    # Step 4: Known defector tests
    defector_results = test_all_defector_types(model)

    # Step 5: Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)

    print(f"\nSanity Check: {'✓ PASS' if sanity_results['success'] else '✗ FAIL'}")
    print(f"  Cooperative mean error: {sanity_results['mean_error']:.4f}")

    print(f"\nDefector Detection:")
    for defector_type, result in defector_results.items():
        status = '✓ PASS' if result['success'] else '✗ FAIL'
        print(f"  {defector_type:20s}: {status} "
              f"(acc={result['accuracy']:.1%}, "
              f"coop={result['cooperative_mean']:.2f}, "
              f"defector={result['defector_mean']:.2f})")

    # Overall success
    all_success = sanity_results['success'] and all(r['success'] for r in defector_results.values())

    print(f"\n{'='*70}")
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_success else '✗ SOME TESTS FAILED'}")
    print(f"{'='*70}")

    # Save results
    final_results = {
        'timestamp': timestamp,
        'training': {
            'num_games': 50,
            'final_val_loss': history['val_loss'][-1],
        },
        'sanity_check': {
            'mean_error': sanity_results['mean_error'],
            'success': sanity_results['success'],
        },
        'defector_tests': {
            defector_type: {
                'cooperative_mean': result['cooperative_mean'],
                'defector_mean': result['defector_mean'],
                'difference': result['difference'],
                'p_value': result['p_value'],
                'cohens_d': result['cohens_d'],
                'accuracy': result['accuracy'],
                'auroc': result['auroc'],
                'success': result['success'],
            }
            for defector_type, result in defector_results.items()
        },
        'overall_success': all_success,
    }

    results_file = results_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    # Plot results
    plot_results(sanity_results, defector_results, results_dir)

    return final_results


def plot_results(sanity_results, defector_results, results_dir):
    """Create visualization of results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training validation loss (placeholder)
    ax = axes[0, 0]
    ax.text(0.5, 0.5, 'Training Loss\n(see training output)',
            ha='center', va='center', fontsize=12)
    ax.set_title('Training Progress')
    ax.axis('off')

    # Plot 2: Cooperative vs Defector Errors
    ax = axes[0, 1]
    defector_types = list(defector_results.keys())
    x_pos = np.arange(len(defector_types))

    coop_means = [defector_results[dt]['cooperative_mean'] for dt in defector_types]
    defector_means = [defector_results[dt]['defector_mean'] for dt in defector_types]
    coop_stds = [defector_results[dt]['cooperative_std'] for dt in defector_types]
    defector_stds = [defector_results[dt]['defector_std'] for dt in defector_types]

    width = 0.35
    ax.bar(x_pos - width/2, coop_means, width, yerr=coop_stds,
           label='Cooperative', alpha=0.7, capsize=5)
    ax.bar(x_pos + width/2, defector_means, width, yerr=defector_stds,
           label='Defector', alpha=0.7, capsize=5)

    ax.axhline(10, color='green', linestyle='--', alpha=0.5, label='Target (coop < 10)')
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Target (defector > 50)')

    ax.set_xlabel('Defector Type')
    ax.set_ylabel('Mean Prediction Error')
    ax.set_title('Cooperative vs Defector Errors')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(defector_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Detection Accuracy
    ax = axes[1, 0]
    accuracies = [defector_results[dt]['accuracy'] for dt in defector_types]
    aurocs = [defector_results[dt]['auroc'] for dt in defector_types]

    x_pos = np.arange(len(defector_types))
    width = 0.35

    ax.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.7)
    ax.bar(x_pos + width/2, aurocs, width, label='AUROC', alpha=0.7)
    ax.axhline(0.85, color='red', linestyle='--', alpha=0.5, label='Target (85%)')

    ax.set_xlabel('Defector Type')
    ax.set_ylabel('Score')
    ax.set_title('Detection Performance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(defector_types, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Statistical Significance
    ax = axes[1, 1]
    cohens_ds = [defector_results[dt]['cohens_d'] for dt in defector_types]
    p_values = [defector_results[dt]['p_value'] for dt in defector_types]

    ax2 = ax.twinx()

    x_pos = np.arange(len(defector_types))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, cohens_ds, width, label="Cohen's d", alpha=0.7, color='green')
    bars2 = ax2.bar(x_pos + width/2, p_values, width, label='p-value', alpha=0.7, color='orange')

    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Target (d > 0.5)')
    ax2.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='Target (p < 0.05)')

    ax.set_xlabel('Defector Type')
    ax.set_ylabel("Cohen's d", color='green')
    ax2.set_ylabel('p-value', color='orange')
    ax.set_title('Statistical Significance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(defector_types, rotation=45, ha='right')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = results_dir / "test_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Results plot saved to {plot_file}")

    plt.close()


if __name__ == "__main__":
    main()
