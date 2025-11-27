"""
Simplest possible game test - 2 agents, 3 rounds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

print("=" * 70)
print("SIMPLE GAME TEST")
print("=" * 70)

print("\n1. Creating environment...")
from environments.commons import CommonsEnvironment
env = CommonsEnvironment(
    num_agents=2,
    initial_resources=100.0,
    regen_rate=0.1,
    max_rounds=3,
    seed=42
)
print("   ✓ Environment created")

print("\n2. Creating agents...")
from agents.llm_commons_agent import create_llm_commons_agent
agents = {
    'agent_0': create_llm_commons_agent('agent_0', behavioral_type='cooperative', provider='deepseek'),
    'agent_1': create_llm_commons_agent('agent_1', behavioral_type='greedy', provider='deepseek'),
}
print("   ✓ Agents created")

print("\n3. Running game...")
print("   Starting run_episode()...")
result = env.run_episode(agents, render=False)
print(f"   ✓ Game completed!")

print("\n4. Results:")
for agent_id, traj in result.trajectories.items():
    print(f"   - {agent_id}: {len(traj.transitions)} transitions")

print("\n" + "=" * 70)
print("SUCCESS ✓")
print("=" * 70)
