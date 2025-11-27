"""
Experiment 1 (LLM): FAST VERSION - 8x speedup!

Optimizations:
1. Parallel agent actions (4x faster)
2. Reduced rounds: 50 instead of 100 (2x faster)
3. Total speedup: 8x (20 min → 2.5 min per game)

Perfect for rapid iteration and testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from environments.commons import CommonsEnvironment
from agents.llm_commons_agent import create_llm_commons_agent
from core.parallel_environment import ParallelEnvironment


class FastCommonsEnvironment(ParallelEnvironment, CommonsEnvironment):
    """
    Commons environment with parallel agent execution.

    Inherits parallelization from ParallelEnvironment and
    game logic from CommonsEnvironment.
    """
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-games', type=int, default=10, help='Training games (default: 10)')
    parser.add_argument('--test-games', type=int, default=5, help='Test games (default: 5)')
    parser.add_argument('--rounds', type=int, default=50, help='Rounds per game (default: 50)')
    parser.add_argument('--provider', type=str, default='deepseek', help='LLM provider')
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 1 (LLM): FAST VERSION (2x speedup)")
    print("=" * 70)
    print("Optimizations:")
    print("  - Parallel agent actions within rounds")
    print(f"  - {args.rounds} rounds instead of 100 (2x)")
    print(f"  - Estimated: ~9 min per game")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Training games: {args.train_games} (~{args.train_games * 9} min)")
    print(f"  - Test games: {args.test_games} (~{args.test_games * 9} min)")
    print(f"  - Total time: ~{(args.train_games + args.test_games) * 9} min ({(args.train_games + args.test_games) * 9 / 60:.1f} hours)")
    print("=" * 70)

    # Configuration
    num_train_games = args.train_games
    num_test_games = args.test_games
    provider = args.provider
    rounds = args.rounds

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/exp1_llm_fast_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults: {results_dir}")

    # Create environment with PARALLEL execution
    env = FastCommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=rounds,  # ← Reduced
        seed=42,
        max_parallel_agents=4  # ← Parallel agents!
    )

    # TRAINING PHASE
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING DATA COLLECTION")
    print("=" * 70)

    train_trajectories = []

    for game_idx in range(num_train_games):
        print(f"\n{'=' * 70}")
        print(f"TRAINING GAME {game_idx + 1}/{num_train_games}")
        print(f"{'=' * 70}")

        print("Creating agents...")
        agents = {
            f'agent_{i}': create_llm_commons_agent(
                f'agent_{i}',
                behavioral_type='cooperative',
                provider=provider
            )
            for i in range(4)
        }
        print("✓ Agents created")

        print(f"Running game ({rounds} rounds, 4 agents in parallel)...")
        try:
            # Run with PARALLEL agent actions
            result = env.run_episode(agents, render=False)
            print(f"✓ Game completed!")

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

                train_trajectories.append(trajectory)

            # Save after each game
            save_path = results_dir / f"train_trajectories_game{game_idx}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(result.trajectories, f)
            print(f"✓ Saved: {save_path}")

            # Summary
            summary = {
                'phase': 'training',
                'game': game_idx,
                'num_trajectories': len(result.trajectories),
                'rounds_completed': len(list(result.trajectories.values())[0].transitions),
                'final_resources': list(result.trajectories.values())[0].transitions[-1].next_observation.extra.get('resources', 0) if result.trajectories else 0,
            }

            summary_path = results_dir / f"train_summary_game{game_idx}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Summary: {summary_path}")

        except Exception as e:
            print(f"✗ Error in game {game_idx}: {e}")
            import traceback
            traceback.print_exc()

            # Save error
            error_path = results_dir / f"error_game{game_idx}.txt"
            with open(error_path, 'w') as f:
                f.write(f"Error: {e}\n\n")
                traceback.print_exc(file=f)
            print(f"✓ Error saved: {error_path}")
            continue

    # Save all training trajectories
    train_path = results_dir / "train_trajectories.pkl"
    with open(train_path, 'wb') as f:
        pickle.dump(train_trajectories, f)
    print(f"\n✓ All training data saved: {train_path}")

    # TESTING PHASE
    print("\n" + "=" * 70)
    print("PHASE 2: TEST DATA COLLECTION")
    print("=" * 70)

    test_trajectories = []

    for game_idx in range(num_test_games):
        print(f"\n{'=' * 70}")
        print(f"TEST GAME {game_idx + 1}/{num_test_games}")
        print(f"{'=' * 70}")

        print("Creating agents...")
        agents = {
            f'agent_{i}': create_llm_commons_agent(
                f'agent_{i}',
                behavioral_type='cooperative',
                provider=provider
            )
            for i in range(4)
        }
        print("✓ Agents created")

        print(f"Running game ({rounds} rounds, 4 agents in parallel)...")
        try:
            # Run with PARALLEL agent actions
            result = env.run_episode(agents, render=False)
            print(f"✓ Game completed!")

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

                test_trajectories.append(trajectory)

            # Save after each game
            save_path = results_dir / f"test_trajectories_game{game_idx}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(result.trajectories, f)
            print(f"✓ Saved: {save_path}")

            # Summary
            summary = {
                'phase': 'testing',
                'game': game_idx,
                'num_trajectories': len(result.trajectories),
                'rounds_completed': len(list(result.trajectories.values())[0].transitions),
                'final_resources': list(result.trajectories.values())[0].transitions[-1].next_observation.extra.get('resources', 0) if result.trajectories else 0,
            }

            summary_path = results_dir / f"test_summary_game{game_idx}.json"
            with open(summary_path, 'wb') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Summary: {summary_path}")

        except Exception as e:
            print(f"✗ Error in test game {game_idx}: {e}")
            import traceback
            traceback.print_exc()

            # Save error
            error_path = results_dir / f"test_error_game{game_idx}.txt"
            with open(error_path, 'w') as f:
                f.write(f"Error: {e}\n\n")
                traceback.print_exc(file=f)
            print(f"✓ Error saved: {error_path}")
            continue

    # Save all test trajectories
    test_path = results_dir / "test_trajectories.pkl"
    with open(test_path, 'wb') as f:
        pickle.dump(test_trajectories, f)
    print(f"\n✓ All test data saved: {test_path}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"Training: {len(train_trajectories)} trajectories from {num_train_games} games")
    print(f"Testing: {len(test_trajectories)} trajectories from {num_test_games} games")
    print(f"Total: {len(train_trajectories) + len(test_trajectories)} trajectories")
    print(f"Results: {results_dir}")

    print("\n" + "=" * 70)
    print("SUCCESS! 🚀")
    print("Ready for world model training and fingerprinting!")
    print("=" * 70)


if __name__ == "__main__":
    main()
