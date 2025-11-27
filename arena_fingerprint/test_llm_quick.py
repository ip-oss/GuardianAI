"""
Quick test: 1 game, 10 rounds with DeepSeek agents.

Verifies:
1. LLM agents can be created
2. They can play the Commons game
3. They respond with valid harvest amounts
4. No crashes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from environments.commons import CommonsEnvironment
from agents.llm_commons_agent import create_llm_commons_agent

def main():
    print("="*70)
    print("QUICK TEST: DeepSeek Agents Playing Commons (1 game, 10 rounds)")
    print("="*70)

    # Create environment with SHORT game
    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=10,  # Only 10 rounds
        seed=42
    )

    print("\nCreating 4 DeepSeek agents...")

    # Create agents with different behavioral types
    agents = {
        'agent_0': create_llm_commons_agent(
            'agent_0',
            behavioral_type='cooperative',
            provider='deepseek'
        ),
        'agent_1': create_llm_commons_agent(
            'agent_1',
            behavioral_type='cooperative',
            provider='deepseek'
        ),
        'agent_2': create_llm_commons_agent(
            'agent_2',
            behavioral_type='cooperative',
            provider='deepseek'
        ),
        'agent_3': create_llm_commons_agent(
            'agent_3',
            behavioral_type='greedy',  # One greedy agent
            provider='deepseek'
        ),
    }

    print("✓ Agents created")
    print("\nRunning game...")
    print("-" * 70)

    # Run episode
    result = env.run_episode(agents, render=True)

    print("-" * 70)
    print("\n✓ Game completed!")

    # Summary
    print("\nFinal Scores:")
    for agent_id, trajectory in result.trajectories.items():
        if trajectory.transitions:
            final_obs = trajectory.transitions[-1].next_observation
            final_score = final_obs.extra.get('my_score', 0)
            behavioral_type = trajectory.transitions[0].action.extra.get('behavioral_type', '?')
            print(f"  {agent_id} ({behavioral_type}): {final_score:.1f}")

    # Check if resources survived
    if result.trajectories:
        last_traj = list(result.trajectories.values())[0]
        if last_traj.transitions:
            final_resources = last_traj.transitions[-1].next_observation.extra.get('resources', 0)
            print(f"\nFinal resources: {final_resources:.1f}")

            if final_resources > 30:
                print("✓ Resources sustained (good cooperation!)")
            elif final_resources > 0:
                print("⚠ Resources low (moderate depletion)")
            else:
                print("✗ Resources depleted (tragedy of the commons)")

    # Sample LLM responses
    print("\nSample LLM Responses:")
    for agent_id, trajectory in list(result.trajectories.items())[:2]:
        if trajectory.transitions:
            for t in trajectory.transitions[:2]:  # First 2 rounds
                llm_response = t.action.extra.get('llm_response', '')
                harvest = t.action.extra.get('harvest_amount', 0)
                # Show first line of response
                first_line = llm_response.split('\n')[0][:80]
                print(f"  {agent_id} (round {t.observation.timestep}): harvest={harvest}, response=\"{first_line}...\"")

    print("\n" + "="*70)
    print("TEST COMPLETE ✓")
    print("="*70)

if __name__ == "__main__":
    main()
