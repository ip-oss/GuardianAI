"""
Test environment creation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("1. Importing CommonsEnvironment...")
from environments.commons import CommonsEnvironment

print("2. Creating environment...")
env = CommonsEnvironment(
    num_agents=2,
    initial_resources=100.0,
    regen_rate=0.1,
    max_rounds=5,
    seed=42
)

print(f"3. Environment created!")
print(f"   - max_rounds: {env.max_rounds}")
print(f"   - num_agents: {env.num_agents}")

print("\n✓ SUCCESS - Environment creation works!")
