"""
Phase 2C: Balanced Adversarial Games

Run games with specific agent_3 type for balanced class distribution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
import time
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

from environments.commons import CommonsEnvironment
from agents.llm_commons_agent import create_llm_commons_agent
from core.parallel_environment import ParallelEnvironment


class FastCommonsEnvironment(ParallelEnvironment, CommonsEnvironment):
    """Commons environment with parallel agent execution."""
    pass


def run_adversarial_game(game_idx, rounds, provider, results_dir, agent_3_type, base_seed=2000):
    """
    Run a single adversarial game with specified agent_3 type.

    Args:
        game_idx: Game index
        rounds: Number of rounds per game
        provider: LLM provider
        results_dir: Directory to save results
        agent_3_type: Type for agent_3 ('free_rider', 'vengeful', 'random', or 'mixed')
        base_seed: Base random seed

    Returns:
        Dictionary with game results
    """
    try:
        # Define agent types
        if agent_3_type == 'mixed':
            rotating_types = ['free_rider', 'vengeful', 'random']
            agent_3_actual = rotating_types[game_idx % len(rotating_types)]
        else:
            agent_3_actual = agent_3_type

        agent_types = {
            'agent_0': 'cooperative',
            'agent_1': 'greedy',
            'agent_2': 'deceptive',
            'agent_3': agent_3_actual,
        }

        print(f"[Game {game_idx}] Starting with agents: {list(agent_types.values())}")

        # Create environment
        env = FastCommonsEnvironment(
            num_agents=4,
            initial_resources=100.0,
            regen_rate=0.1,
            max_rounds=rounds,
            seed=base_seed + game_idx,
            max_parallel_agents=4
        )

        # Create agents with different behavioral types
        agents = {
            agent_id: create_llm_commons_agent(
                agent_id,
                behavioral_type=behavioral_type,
                provider=provider
            )
            for agent_id, behavioral_type in agent_types.items()
        }

        # Run game
        result = env.run_episode(agents, render=False)

        # Enhance trajectories with cross-agent information AND agent types
        trajectories = []
        for agent_id, trajectory in result.trajectories.items():
            # Add agent type to trajectory metadata
            trajectory.metadata['agent_type'] = agent_types[agent_id]
            trajectory.metadata['game_idx'] = game_idx

            for t_idx, transition in enumerate(trajectory.transitions):
                # Add all agents' actions
                all_actions = {}
                all_rewards = {}
                all_types = {}

                for other_id, other_traj in result.trajectories.items():
                    if t_idx < len(other_traj.transitions):
                        other_trans = other_traj.transitions[t_idx]
                        all_actions[other_id] = other_trans.action.extra.get('harvest_amount', 0)
                        all_rewards[other_id] = other_trans.reward
                        all_types[other_id] = agent_types[other_id]

                transition.action.extra['all_actions'] = all_actions
                transition.action.extra['all_types'] = all_types
                transition.next_observation.extra['all_rewards'] = all_rewards

            trajectories.append(trajectory)

        # Save individual game results
        save_path = results_dir / f"adversarial_trajectories_game{game_idx}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(result.trajectories, f)

        # Save summary with agent types
        summary = {
            'game': game_idx,
            'agent_types': agent_types,
            'num_trajectories': len(result.trajectories),
            'rounds_completed': len(list(result.trajectories.values())[0].transitions),
            'final_resources': list(result.trajectories.values())[0].transitions[-1].next_observation.extra.get('resources', 0) if result.trajectories else 0,
            'final_scores': {
                aid: traj.transitions[-1].next_observation.extra.get('my_score', 0)
                for aid, traj in result.trajectories.items()
            }
        }

        summary_path = results_dir / f"adversarial_summary_game{game_idx}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[Game {game_idx}] Complete! ({len(trajectories)} trajectories)")
        print(f"  Agent types: {list(agent_types.values())}")
        print(f"  Final scores: {summary['final_scores']}")

        return {
            'game_idx': game_idx,
            'trajectories': trajectories,
            'agent_types': agent_types,
            'success': True,
            'error': None
        }

    except Exception as e:
        print(f"[Game {game_idx}] Error: {e}")
        import traceback
        traceback.print_exc()

        error_path = results_dir / f"adversarial_error_game{game_idx}.txt"
        with open(error_path, 'w') as f:
            f.write(f"Error: {e}\n\n")
            traceback.print_exc(file=f)

        return {
            'game_idx': game_idx,
            'trajectories': [],
            'agent_types': {},
            'success': False,
            'error': str(e)
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run balanced adversarial games with specified agent_3 type"
    )
    parser.add_argument('--start-game', type=int, required=True,
                       help='Starting game index')
    parser.add_argument('--num-games', type=int, required=True,
                       help='Number of games to run')
    parser.add_argument('--agent-3-type', type=str, required=True,
                       choices=['free_rider', 'vengeful', 'random', 'mixed'],
                       help='Type for agent_3')
    parser.add_argument('--rounds', type=int, default=50,
                       help='Rounds per game (default: 50)')
    parser.add_argument('--provider', type=str, default='deepseek',
                       help='LLM provider')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Max parallel games (default: 10)')
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 2C: BALANCED ADVERSARIAL GAMES")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Games: {args.start_game} to {args.start_game + args.num_games - 1}")
    print(f"  - Agent 3 type: {args.agent_3_type}")
    print(f"  - Rounds per game: {args.rounds}")
    print(f"  - Max parallel games: {args.max_workers}")
    print(f"  - Provider: {args.provider}")
    print()
    print("Agent composition per game:")
    print("  - Agent 0: Cooperative (baseline)")
    print("  - Agent 1: Greedy")
    print("  - Agent 2: Deceptive")
    print(f"  - Agent 3: {args.agent_3_type}")
    print("=" * 70)

    # Create results directory
    results_dir = Path("results/exp2_adversarial_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults: {results_dir}")

    # Run adversarial games in parallel
    print("\n" + "=" * 70)
    print("RUNNING ADVERSARIAL GAMES (PARALLEL)")
    print("=" * 70)

    all_trajectories = []
    all_agent_types = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all games
        futures = {
            executor.submit(
                run_adversarial_game,
                args.start_game + i,
                args.rounds,
                args.provider,
                results_dir,
                args.agent_3_type
            ): args.start_game + i
            for i in range(args.num_games)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            game_idx = futures[future]
            result = future.result()

            if result['success']:
                all_trajectories.extend(result['trajectories'])
                all_agent_types[f"game_{game_idx}"] = result['agent_types']
                completed += 1
                elapsed = time.time() - start_time
                print(f"\n[PROGRESS] {completed}/{args.num_games} games complete ({elapsed/60:.1f} min)")

    elapsed = time.time() - start_time
    print(f"\nAll games complete in {elapsed/60:.1f} minutes")
    print(f"  Collected: {len(all_trajectories)} trajectories")

    # Summary statistics
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    agent_type_counts = {}
    for trajectory in all_trajectories:
        agent_type = trajectory.metadata.get('agent_type', 'unknown')
        agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Agent type distribution:")
    for agent_type, count in sorted(agent_type_counts.items()):
        print(f"  {agent_type}: {count}")

    print(f"\nTime: {elapsed/60:.1f} min")
    print(f"Results: {results_dir}")

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print(f"Run Phase 2D to re-analyze all {len(list(results_dir.glob('adversarial_trajectories_game*.pkl')))} games")
    print("=" * 70)


if __name__ == "__main__":
    main()
