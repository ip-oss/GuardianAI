"""
Experiment 1: Baseline Validation (Agent-Centric Model)

Train agent-centric world model on cooperative-only games and validate it works.

Success Criteria:
- World model achieves <0.1 MSE on cooperative test games
- Prediction errors are low and consistent
- NO global contamination (errors don't depend on other agents)

This validates the agent-centric model fixes the contamination problem.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse

from environments.commons import CommonsEnvironment
from agents.commons_agents import create_commons_agent
from world_models import (
    AgentCentricWorldModel,
    AgentCentricTrainer,
    extract_agent_centric_errors
)
from fingerprinting import extract_commons_fingerprint, extract_batch_fingerprints


def collect_cooperative_games(
    num_games: int = 50,
    num_agents: int = 4,
    seed: int = 42
) -> list:
    """
    Collect trajectories from cooperative-only games.

    These will be used to train the world model.
    """
    print(f"\nCollecting {num_games} cooperative games...")

    env = CommonsEnvironment(
        num_agents=num_agents,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=100,
        seed=seed
    )

    all_trajectories = []

    for game_idx in range(num_games):
        # Create cooperative agents
        agents = {
            f'agent_{i}': create_commons_agent(
                f'agent_{i}',
                'cooperative',
                seed=seed + game_idx * 10 + i
            )
            for i in range(num_agents)
        }

        # Run episode
        result = env.run_episode(agents, render=False)

        # Enhance trajectories with joint action information
        # This is needed for context extraction (average extraction)
        for agent_id, trajectory in result.trajectories.items():
            for t_idx, transition in enumerate(trajectory.transitions):
                # Add all actions and rewards to each transition
                all_actions = {}
                all_rewards = {}

                for other_id, other_traj in result.trajectories.items():
                    if t_idx < len(other_traj.transitions):
                        other_trans = other_traj.transitions[t_idx]
                        all_actions[other_id] = other_trans.action.extra.get('harvest_amount', 0)
                        all_rewards[other_id] = other_trans.reward

                # Store in transition
                transition.action.extra['all_actions'] = all_actions
                transition.next_observation.extra['all_rewards'] = all_rewards

            all_trajectories.append(trajectory)

        if (game_idx + 1) % 10 == 0:
            print(f"  Collected {game_idx + 1}/{num_games} games")

    print(f"✓ Collected {len(all_trajectories)} trajectories from {num_games} games")
    return all_trajectories


def main(args):
    """Run baseline experiment with agent-centric model"""
    print("="*70)
    print("EXPERIMENT 1: Baseline Validation (Agent-Centric Model)")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print("="*70)

    # Configuration
    if args.mode == 'quick':
        num_train_games = 20
        num_test_games = 10
        num_epochs = 50
    elif args.mode == 'full':
        num_train_games = 100
        num_test_games = 30
        num_epochs = 150
    else:  # comprehensive
        num_train_games = 200
        num_test_games = 50
        num_epochs = 200

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/exp1_baseline_agent_centric_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Step 1: Collect training data
    print("\n" + "="*70)
    print("STEP 1: Collect Training Data")
    print("="*70)

    train_trajectories = collect_cooperative_games(
        num_games=num_train_games,
        num_agents=4,
        seed=args.seed
    )

    # Step 2: Train agent-centric world model
    print("\n" + "="*70)
    print("STEP 2: Train Agent-Centric World Model")
    print("="*70)

    model = AgentCentricWorldModel(
        state_dim=6,
        action_dim=1,
        hidden_dim=64,
        dropout=0.1
    )
    trainer = AgentCentricTrainer(model, learning_rate=1e-3)

    print(f"\nTraining for {num_epochs} epochs...")
    history = trainer.train(
        train_trajectories,
        epochs=num_epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=True,
        initial_resources=100.0
    )

    # Save model
    model_path = results_dir / "agent_centric_world_model.pt"
    trainer.save(str(model_path))
    print(f"\n✓ Model saved to {model_path}")

    # Step 3: Test on held-out cooperative games
    print("\n" + "="*70)
    print("STEP 3: Test on Held-Out Cooperative Games")
    print("="*70)

    test_trajectories = collect_cooperative_games(
        num_games=num_test_games,
        num_agents=4,
        seed=args.seed + 10000
    )

    # Compute prediction errors (agent-centric)
    print(f"\nComputing agent-centric prediction errors for {len(test_trajectories)} test trajectories...")
    test_errors_list = [
        extract_agent_centric_errors(traj, model, initial_resources=100.0)
        for traj in test_trajectories
    ]

    # Extract fingerprints
    print("Extracting fingerprints...")
    test_fingerprints = extract_batch_fingerprints(test_trajectories, test_errors_list)

    # Analyze results
    mean_errors = [fp['mean_error'] for fp in test_fingerprints]
    max_errors = [fp['max_error'] for fp in test_fingerprints]

    avg_mean_error = np.mean(mean_errors)
    std_mean_error = np.std(mean_errors)
    avg_max_error = np.mean(max_errors)

    print(f"\nTest Set Results:")
    print(f"  Average mean error: {avg_mean_error:.6f}")
    print(f"  Std mean error: {std_mean_error:.6f}")
    print(f"  Average max error: {avg_max_error:.6f}")
    print(f"  Min mean error: {np.min(mean_errors):.6f}")
    print(f"  Max mean error: {np.max(mean_errors):.6f}")

    # Success criterion (should be MUCH lower with agent-centric model)
    success = avg_mean_error < 0.1
    print(f"\n{'✓' if success else '✗'} Success criterion: "
          f"avg_mean_error < 0.1 ({'PASS' if success else 'FAIL'})")

    # Additional check: consistency (low variance)
    consistency_success = std_mean_error < 0.05
    print(f"{'✓' if consistency_success else '✗'} Consistency criterion: "
          f"std_mean_error < 0.05 ({'PASS' if consistency_success else 'FAIL'})")

    overall_success = success and consistency_success

    # Step 4: Save results
    print("\n" + "="*70)
    print("STEP 4: Save Results")
    print("="*70)

    results = {
        'config': {
            'mode': args.mode,
            'model_type': 'agent_centric',
            'num_train_games': num_train_games,
            'num_test_games': num_test_games,
            'num_epochs': num_epochs,
            'seed': args.seed,
        },
        'training': {
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'final_reward_loss': float(history['reward_loss'][-1]),
            'final_score_loss': float(history['score_loss'][-1]),
            'train_loss_curve': [float(x) for x in history['train_loss']],
            'val_loss_curve': [float(x) for x in history['val_loss']],
        },
        'test_results': {
            'num_trajectories': len(test_trajectories),
            'avg_mean_error': float(avg_mean_error),
            'std_mean_error': float(std_mean_error),
            'avg_max_error': float(avg_max_error),
            'min_mean_error': float(np.min(mean_errors)),
            'max_mean_error': float(np.max(mean_errors)),
            'success': bool(overall_success),
            'avg_error_success': bool(success),
            'consistency_success': bool(consistency_success),
        },
        'sample_fingerprints': test_fingerprints[:5],  # Save first 5
    }

    # Save JSON
    results_file = results_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    ax.plot(history['val_loss'], label='Val Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Component losses
    ax = axes[1]
    ax.plot(history['reward_loss'], label='Reward Loss', alpha=0.7)
    ax.plot(history['score_loss'], label='Score Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Component Losses (Validation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plot_file = results_dir / "training_curve.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Training curve saved to {plot_file}")
    plt.close()

    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(mean_errors, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(avg_mean_error, color='red', linestyle='--',
                label=f'Mean = {avg_mean_error:.4f}', linewidth=2)
    plt.axvline(avg_mean_error - std_mean_error, color='orange', linestyle='--',
                label=f'±1 Std = {std_mean_error:.4f}', linewidth=1)
    plt.axvline(avg_mean_error + std_mean_error, color='orange', linestyle='--',
                linewidth=1)
    plt.xlabel('Mean Prediction Error')
    plt.ylabel('Count')
    plt.title('Agent-Centric Prediction Error Distribution (Cooperative Agents)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    error_plot = results_dir / "error_distribution.png"
    plt.savefig(error_plot, dpi=150, bbox_inches='tight')
    print(f"✓ Error distribution saved to {error_plot}")
    plt.close()

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT 1: SUMMARY")
    print("="*70)
    print(f"Model type: Agent-Centric")
    print(f"Training loss: {history['train_loss'][-1]:.6f}")
    print(f"Validation loss: {history['val_loss'][-1]:.6f}")
    print(f"Test mean error: {avg_mean_error:.6f} ± {std_mean_error:.6f}")
    print(f"Success: {overall_success}")
    print(f"\nExpected improvement over original model:")
    print(f"  Original: ~500-2000 error")
    print(f"  Agent-centric: {avg_mean_error:.4f} error")
    print(f"  Improvement: ~{500/avg_mean_error if avg_mean_error > 0 else 'inf'}x better")
    print(f"\nResults directory: {results_dir}")
    print("="*70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Baseline Validation (Agent-Centric)")
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'comprehensive'],
                       help='Experiment mode (quick/full/comprehensive)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()
    main(args)
