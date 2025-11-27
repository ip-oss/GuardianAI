"""
Simplified LLM Experiment 1 - Saves data incrementally.

Easier to debug, saves after each game.
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

def main():
    print("="*70)
    print("SIMPLIFIED EXP1: LLM Baseline (Incremental Saves)")
    print("="*70)

    # Small config
    num_games = 2  # Very small for testing
    provider = "deepseek"

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/exp1_llm_simple_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults: {results_dir}")

    # Create environment
    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=20,  # Shorter games
        seed=42
    )

    all_trajectories = []

    for game_idx in range(num_games):
        print(f"\n{'='*70}")
        print(f"GAME {game_idx + 1}/{num_games}")
        print(f"{'='*70}")

        # Create agents
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

        # Run game
        print("Running game...")
        try:
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

                all_trajectories.append(trajectory)

            # SAVE AFTER EACH GAME
            save_path = results_dir / f"trajectories_game{game_idx}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(result.trajectories, f)
            print(f"✓ Saved: {save_path}")

            # Summary
            summary = {
                'game': game_idx,
                'num_trajectories': len(result.trajectories),
                'final_resources': list(result.trajectories.values())[0].transitions[-1].next_observation.extra.get('resources', 0) if result.trajectories else 0,
            }

            summary_path = results_dir / f"summary_game{game_idx}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Saved: {summary_path}")

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

    # Final summary
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Collected {len(all_trajectories)} trajectories")
    print(f"Results: {results_dir}")

    # Save all trajectories
    all_path = results_dir / "all_trajectories.pkl"
    with open(all_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    print(f"✓ Saved all trajectories: {all_path}")

if __name__ == "__main__":
    main()
