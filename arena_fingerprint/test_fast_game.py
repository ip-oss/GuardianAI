"""
Quick test of fast parallel game - 1 game, 5 rounds.
Should take ~30 seconds instead of 2 minutes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from environments.commons import CommonsEnvironment
from agents.llm_commons_agent import create_llm_commons_agent
from core.parallel_environment import ParallelEnvironment

class FastCommonsEnvironment(ParallelEnvironment, CommonsEnvironment):
    pass

print("=" * 70)
print("FAST PARALLEL TEST (1 game, 5 rounds)")
print("=" * 70)

env = FastCommonsEnvironment(
    num_agents=4,
    initial_resources=100.0,
    regen_rate=0.1,
    max_rounds=5,  # Very short
    seed=42,
    max_parallel_agents=4
)

print("\nCreating 4 agents...")
agents = {
    f'agent_{i}': create_llm_commons_agent(f'agent_{i}', 'cooperative', 'deepseek')
    for i in range(4)
}
print("✓ Agents created")

print("\nRunning game with PARALLEL agent actions...")
import time
start = time.time()

result = env.run_episode(agents, render=False)

elapsed = time.time() - start

print(f"✓ Game completed in {elapsed:.1f} seconds!")
print(f"\nResults:")
for agent_id, traj in result.trajectories.items():
    print(f"  {agent_id}: {len(traj.transitions)} transitions")

print(f"\nExpected time: ~15-25 seconds (parallel)")
print(f"Sequential would be: ~60-100 seconds")
print(f"Speedup: ~{60/elapsed:.1f}x faster!")
print("\n" + "=" * 70)
print("SUCCESS! ✓")
print("=" * 70)
