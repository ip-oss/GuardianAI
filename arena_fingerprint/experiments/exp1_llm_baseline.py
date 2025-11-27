"""
Experiment 1 (LLM): Baseline Validation with LLM Agents

Train agent-centric world model on LLM cooperative agents and validate.

This tests whether:
1. LLM agents can actually play the Commons game
2. Cooperative-prompted LLMs produce low prediction errors
3. The agent-centric model works with LLM agents

Success Criteria:
- LLM agents successfully complete games
- Prediction errors are reasonable
- No crashes or API failures
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import os
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from environments.commons import CommonsEnvironment
from agents.llm_commons_agent import create_llm_commons_agent
from world_models import (
    AgentCentricWorldModel,
    AgentCentricTrainer,
    extract_agent_centric_errors
)
from fingerprinting import extract_commons_fingerprint, extract_batch_fingerprints


def collect_llm_games(
    num_games: int = 10,
    num_agents: int = 4,
    provider: str = "deepseek",
    model: str = None,
    seed: int = 42,
    save_dir: Path = None
) -> list:
    """
    Collect trajectories from LLM cooperative agents.

    Args:
        num_games: Number of games to run
        num_agents: Number of agents per game
        provider: LLM provider ("deepseek", "anthropic", "openai")
        model: Model name
        seed: Random seed

    Returns:
        List of trajectories
    """
    # Set default model if None
    if model is None:
        if provider == "deepseek":
            model = "deepseek-chat"
        elif provider == "anthropic":
            model = "claude-sonnet-4-5-20250929"
        elif provider == "openai":
            model = "gpt-4o"

    print(f"\nCollecting {num_games} games with {provider} agents...")
    print(f"Model: {model}")

    env = CommonsEnvironment(
        num_agents=num_agents,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=100,
        seed=seed
    )

    all_trajectories = []
    api_calls_total = 0

    for game_idx in range(num_games):
        print(f"\n--- Game {game_idx + 1}/{num_games} ---")

        # Create LLM agents (all cooperative)
        agents = {}
        for i in range(num_agents):
            agent = create_llm_commons_agent(
                agent_id=f'agent_{i}',
                behavioral_type='cooperative',
                provider=provider,
                model=model,
                temperature=0.7
            )
            agents[f'agent_{i}'] = agent

        # Run episode
        try:
            result = env.run_episode(agents, render=False)

            # Count API calls (100 rounds * 4 agents = 400)
            api_calls_this_game = len(result.trajectories) * len(result.trajectories[list(result.trajectories.keys())[0]].transitions)
            api_calls_total += api_calls_this_game

            print(f"  ✓ Completed ({api_calls_this_game} API calls)")

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

            # SAVE INCREMENTALLY
            if save_dir:
                game_save_path = save_dir / f"trajectories_game{game_idx}.pkl"
                with open(game_save_path, 'wb') as f:
                    pickle.dump(result.trajectories, f)
                print(f"  ✓ Saved game data: {game_save_path.name}")

        except Exception as e:
            print(f"  ✗ Error in game {game_idx + 1}: {e}")

            # SAVE ERROR
            if save_dir:
                error_path = save_dir / f"error_game{game_idx}.txt"
                with open(error_path, 'w') as f:
                    f.write(f"Error: {e}\n")
                    import traceback
                    traceback.print_exc(file=f)
                print(f"  ✓ Error saved: {error_path.name}")
            continue

    print(f"\n✓ Collected {len(all_trajectories)} trajectories from {num_games} games")
    print(f"  Total API calls: {api_calls_total}")

    return all_trajectories


def main(args):
    """Run LLM baseline experiment"""
    print("="*70)
    print("EXPERIMENT 1 (LLM): Baseline Validation with LLM Agents")
    print("="*70)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print("="*70)

    # Configuration
    if args.mode == 'quick':
        num_train_games = 5  # Fewer for LLMs (expensive)
        num_test_games = 3
        num_epochs = 50
    elif args.mode == 'full':
        num_train_games = 15
        num_test_games = 5
        num_epochs = 100
    else:  # comprehensive
        num_train_games = 30
        num_test_games = 10
        num_epochs = 150

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/exp1_llm_{args.provider}_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Step 1: Collect training data from LLM agents
    print("\n" + "="*70)
    print("STEP 1: Collect Training Data from LLM Agents")
    print("="*70)

    train_trajectories = collect_llm_games(
        num_games=num_train_games,
        num_agents=4,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        save_dir=results_dir  # SAVE INCREMENTALLY
    )

    if len(train_trajectories) == 0:
        print("\n✗ No training data collected. Exiting.")
        return None

    # SAVE TRAJECTORIES IMMEDIATELY (before training)
    train_traj_path = results_dir / "train_trajectories.pkl"
    with open(train_traj_path, 'wb') as f:
        pickle.dump(train_trajectories, f)
    print(f"✓ Training trajectories saved: {train_traj_path}")
    print(f"  ({len(train_trajectories)} trajectories)")

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

    # Step 3: Test on held-out LLM games
    print("\n" + "="*70)
    print("STEP 3: Test on Held-Out LLM Games")
    print("="*70)

    test_trajectories = collect_llm_games(
        num_games=num_test_games,
        num_agents=4,
        provider=args.provider,
        model=args.model,
        seed=args.seed + 10000,
        save_dir=results_dir  # SAVE INCREMENTALLY
    )

    if len(test_trajectories) == 0:
        print("\n✗ No test data collected. Skipping evaluation.")
        test_results = None
    else:
        # SAVE TEST TRAJECTORIES
        test_traj_path = results_dir / "test_trajectories.pkl"
        with open(test_traj_path, 'wb') as f:
            pickle.dump(test_trajectories, f)
        print(f"✓ Test trajectories saved: {test_traj_path}")
        print(f"  ({len(test_trajectories)} trajectories)")

        # Compute prediction errors
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

        # Success criterion (LLMs might have higher variance)
        success = avg_mean_error < 1.0  # More lenient for LLMs
        print(f"\n{'✓' if success else '✗'} Success criterion: "
              f"avg_mean_error < 1.0 ({'PASS' if success else 'FAIL'})")

        test_results = {
            'num_trajectories': len(test_trajectories),
            'avg_mean_error': float(avg_mean_error),
            'std_mean_error': float(std_mean_error),
            'avg_max_error': float(avg_max_error),
            'min_mean_error': float(np.min(mean_errors)),
            'max_mean_error': float(np.max(mean_errors)),
            'success': bool(success),
        }

    # Step 4: Save results
    print("\n" + "="*70)
    print("STEP 4: Save Results")
    print("="*70)

    results = {
        'config': {
            'mode': args.mode,
            'provider': args.provider,
            'model': args.model or 'default',
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
        },
        'test_results': test_results,
    }

    # Save JSON
    results_file = results_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    ax.plot(history['val_loss'], label='Val Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Training Loss ({args.provider})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

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

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT 1 (LLM): SUMMARY")
    print("="*70)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print(f"Training loss: {history['train_loss'][-1]:.6f}")
    print(f"Validation loss: {history['val_loss'][-1]:.6f}")
    if test_results:
        print(f"Test mean error: {test_results['avg_mean_error']:.6f} ± {test_results['std_mean_error']:.6f}")
        print(f"Success: {test_results['success']}")
    print(f"\nResults directory: {results_dir}")
    print("="*70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1 (LLM): Baseline with LLM Agents")
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'comprehensive'],
                       help='Experiment mode')
    parser.add_argument('--provider', type=str, default='deepseek',
                       choices=['deepseek', 'anthropic', 'openai'],
                       help='LLM provider')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (default: provider-specific default)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()
    main(args)
