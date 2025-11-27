"""
Ultra-minimal test - just create an agent, no game.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
print("1. Loading .env...")
load_dotenv()

print("2. Importing create_llm_commons_agent...")
from agents.llm_commons_agent import create_llm_commons_agent

print("3. Creating agent...")
agent = create_llm_commons_agent(
    'test_agent',
    behavioral_type='cooperative',
    provider='deepseek'
)

print(f"4. Agent created: {agent.agent_id}")
print(f"5. Backend type: {agent.backend_type}")

print("\n✓ SUCCESS - Agent creation works!")
