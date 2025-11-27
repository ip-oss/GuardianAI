"""
Experiment 1 (LLM): PARALLELIZED Version

Runs multiple games in parallel for faster completion.
- 2-3x faster than sequential
- Respects API rate limits
- Still saves incrementally
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
from multiprocessing import Pool
import time

load_dotenv()

from environments.commons import CommonsEnvironment
from agents.llm_commons_agent import create_llm_commons_agent
from world_models import (
    AgentCentricWorldModel,
    AgentCentricTrainer,
    extract_agent_centric_errors
)
from fingerprinting import extract_commons_fingerprint, extract_batch_fingerprints


def run_single_game(args):
    """
    Run a single game (designed for parallel execution).

    Args:
        args: Tuple of (game_idx, provider, model, seed, save_dir)

    Returns:
        Tuple of (game_idx, trajectories, error_msg)
    """
    game_idx, provider, model, seed, save_dir = args

    print(f"\n[Game {game_idx + 1}] Starting...")

    try:
        # Create environment
        env = CommonsEnvironment(
            num_agents=4,
            initial_resources=100.0,
            regen_rate=0.1,
            max_rounds=100,
            seed=seed + game_idx * 10
        )

        # Create agents
        agents = {
            f'agent_{i}': create_llm_commons_agent(
                f'agent_{i}',
                behavioral_type='cooperative',
                provider=provider,
                model=model
            )
            for i in range(4)
        }

        # Run game
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

        # Save incrementally
        if save_dir:
            game_save_path = Path(save_dir) / f"trajectories_game{game_idx}.pkl"
            with open(game_save_path, 'wb') as f:
                pickle.dump(result.trajectories, f)

        print(f"[Game {game_idx + 1}] ✓ Completed")

        # Return trajectories as list
        trajectories = list(result.trajectories.values())
        return (game_idx, trajectories, None)

    except Exception as e:
        print(f"[Game {game_idx + 1}] ✗ Error: {e}")

        # Save error
        if save_dir:
            error_path = Path(save_dir) / f"error_game{game_idx}.txt"
            with open(error_path, 'w') as f:
                f.write(f"Error: {e}\n")
                import traceback
                traceback.print_exc(file=f)

        return (game_idx, [], str(e))


def collect_llm_games_parallel(
    num_games: int = 5,
    provider: str = "deepseek",
    model: str = None,
    seed: int = 42,
    save_dir: Path = None,
    num_workers: int = 20
):
    """
    Collect games in parallel.

    Args:
        num_games: Number of games to run
        provider: LLM provider
        model: Model name
        seed: Random seed
        save_dir: Directory to save results
        num_workers: Number of parallel workers (default: 20, use 40 for max speed)
    """
    # Set default model
    if model is None:
        if provider == "deepseek":
            model = "deepseek-chat"
        elif provider == "anthropic":
            model = "claude-sonnet-4-5-20250929"
        elif provider == "openai":
            model = "gpt-4o"

    print(f"\n{'='*70}")
    print(f"PARALLEL COLLECTION: {num_games} games with {num_workers} workers")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"{'='*70}")

    # Prepare arguments for each game
    game_args = [
        (i, provider, model, seed, save_dir)
        for i in range(num_games)
    ]

    # Run in parallel
    all_trajectories = []
    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_game, game_args)

    # Collect results
    errors = []
    for game_idx, trajectories, error_msg in sorted(results, key=lambda x: x[0]):
        if error_msg:
            errors.append((game_idx, error_msg))
        else:
            all_trajectories.extend(trajectories)

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Collected {len(all_trajectories)} trajectories")
    print(f"✓ Time: {elapsed/60:.1f} minutes")
    print(f"✓ Speed: {elapsed/num_games:.1f} sec/game")
    if errors:
        print(f"⚠ Errors: {len(errors)} games failed")

    return all_trajectories


def main(args):
    """Run parallelized LLM baseline experiment"""
    print("="*70)
    print("EXPERIMENT 1 (LLM): PARALLEL Baseline")
    print("="*70)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print(f"Mode: {args.mode}")
    print(f"Workers: {args.workers}")
    print(f"Seed: {args.seed}")
    print("="*70)

    # Configuration
    if args.mode == 'quick':
        num_train_games = 5
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
    results_dir = Path(f"results/exp1_llm_parallel_{args.provider}_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults: {results_dir}")

    # Step 1: Collect training data (PARALLEL)
    print("\n" + "="*70)
    print("STEP 1: Collect Training Data (PARALLEL)")
    print("="*70)

    train_trajectories = collect_llm_games_parallel(
        num_games=num_train_games,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        save_dir=results_dir,
        num_workers=args.workers
    )

    if len(train_trajectories) == 0:
        print("\n✗ No training data collected. Exiting.")
        return None

    # Save all training trajectories
    train_traj_path = results_dir / "train_trajectories.pkl"
    with open(train_traj_path, 'wb') as f:
        pickle.dump(train_trajectories, f)
    print(f"✓ Saved: {train_traj_path}")

    # Step 2: Train model
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
    print(f"\n✓ Model saved: {model_path}")

    # Step 3: Test (PARALLEL)
    print("\n" + "="*70)
    print("STEP 3: Test on Held-Out Games (PARALLEL)")
    print("="*70)

    test_trajectories = collect_llm_games_parallel(
        num_games=num_test_games,
        provider=args.provider,
        model=args.model,
        seed=args.seed + 10000,
        save_dir=results_dir,
        num_workers=args.workers
    )

    if len(test_trajectories) == 0:
        print("\n✗ No test data. Skipping evaluation.")
        test_results = None
    else:
        # Save test trajectories
        test_traj_path = results_dir / "test_trajectories.pkl"
        with open(test_traj_path, 'wb') as f:
            pickle.dump(test_trajectories, f)
        print(f"✓ Saved: {test_traj_path}")

        # Evaluate
        print(f"\nComputing errors...")
        test_errors_list = [
            extract_agent_centric_errors(traj, model, initial_resources=100.0)
            for traj in test_trajectories
        ]

        test_fingerprints = extract_batch_fingerprints(test_trajectories, test_errors_list)

        mean_errors = [fp['mean_error'] for fp in test_fingerprints]
        avg_mean_error = np.mean(mean_errors)
        std_mean_error = np.std(mean_errors)

        print(f"\nTest Results:")
        print(f"  Mean error: {avg_mean_error:.6f} ± {std_mean_error:.6f}")

        success = avg_mean_error < 1.0
        print(f"  Success: {success}")

        test_results = {
            'num_trajectories': len(test_trajectories),
            'avg_mean_error': float(avg_mean_error),
            'std_mean_error': float(std_mean_error),
            'success': bool(success),
        }

    # Save results
    results = {
        'config': {
            'mode': args.mode,
            'provider': args.provider,
            'model': args.model or 'default',
            'num_workers': args.workers,
            'num_train_games': num_train_games,
            'num_test_games': num_test_games,
        },
        'training': {
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
        },
        'test_results': test_results,
    }

    results_file = results_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {results_file}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results: {results_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1 (LLM): PARALLEL Baseline")
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'comprehensive'])
    parser.add_argument('--provider', type=str, default='deepseek',
                       choices=['deepseek', 'anthropic', 'openai'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=20,
                       help='Number of parallel workers (default: 20, use 40 for faster)')

    args = parser.parse_args()
    main(args)
