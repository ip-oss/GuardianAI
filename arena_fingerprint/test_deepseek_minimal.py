"""
Minimal test to diagnose where DeepSeek agents hang.

Tests each step individually:
1. API key loading
2. LLM provider creation
3. Agent creation
4. Single LLM API call
5. Environment creation
6. Single-step game execution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
from dotenv import load_dotenv
import time

def test_step(step_num, step_name, func):
    """Test a single step with timing."""
    print(f"\n[Step {step_num}] {step_name}...")
    start = time.time()
    try:
        result = func()
        elapsed = time.time() - start
        print(f"  ✓ Success ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ FAILED ({elapsed:.2f}s): {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 70)
    print("DEEPSEEK DIAGNOSTIC TEST")
    print("=" * 70)

    # Step 1: Load environment
    def step1():
        load_dotenv()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")
        print(f"  API key: {api_key[:10]}...{api_key[-5:]}")
        return api_key

    api_key = test_step(1, "Load API key from .env", step1)
    if not api_key:
        return

    # Step 2: Create LLM provider
    def step2():
        from agents.llm_providers import create_llm_provider
        llm_config = {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": api_key
        }
        provider = create_llm_provider(llm_config)
        print(f"  Provider type: {type(provider).__name__}")
        return provider

    provider = test_step(2, "Create DeepSeek provider", step2)
    if not provider:
        return

    # Step 3: Test single LLM API call
    def step3():
        messages = [{"role": "user", "content": "Say 'test successful' and nothing else."}]
        response = provider.generate(
            messages=messages,
            system_prompt="You are a helpful assistant.",
            max_tokens=50,
            temperature=0.7
        )
        print(f"  Response: {response[:100]}")
        return response

    response = test_step(3, "Test LLM API call", step3)
    if not response:
        return

    # Step 4: Create LLM Commons agent
    def step4():
        from agents.llm_commons_agent import create_llm_commons_agent
        agent = create_llm_commons_agent(
            'test_agent',
            behavioral_type='cooperative',
            provider='deepseek',
            api_key=api_key
        )
        print(f"  Agent type: {type(agent).__name__}")
        print(f"  Agent ID: {agent.agent_id}")
        print(f"  Backend: {agent.backend_type}")
        return agent

    agent = test_step(4, "Create LLM Commons agent", step4)
    if not agent:
        return

    # Step 5: Create environment
    def step5():
        from environments.commons import CommonsEnvironment
        env = CommonsEnvironment(
            num_agents=2,  # Small
            initial_resources=100.0,
            regen_rate=0.1,
            max_rounds=3,  # Very short
            seed=42
        )
        print(f"  Environment created with {env.num_agents} agents, {env.max_rounds} rounds")
        return env

    env = test_step(5, "Create Commons environment", step5)
    if not env:
        return

    # Step 6: Test single observation → action
    def step6():
        from core import Observation
        obs = Observation(
            agent_id='test_agent',
            timestep=0,
            extra={
                'resources': 100.0,
                'my_score': 0.0,
                'all_scores': {'test_agent': 0.0},
                'round': 1,
                'max_rounds': 3,
                'recent_history': [],
                'max_harvest': 10
            }
        )
        action = agent.act(obs)
        print(f"  Harvest amount: {action.extra.get('harvest_amount')}")
        print(f"  LLM response: {action.extra.get('llm_response', '')[:100]}")
        return action

    action = test_step(6, "Test agent.act() with observation", step6)
    if not action:
        return

    # Step 7: Run 1 full game (2 agents, 3 rounds)
    def step7():
        agent2 = create_llm_commons_agent(
            'test_agent_2',
            behavioral_type='greedy',
            provider='deepseek',
            api_key=api_key
        )
        agents = {'test_agent': agent, 'test_agent_2': agent2}

        print(f"  Starting game...")
        result = env.run_episode(agents, render=False)
        print(f"  Game completed!")
        print(f"  Trajectories: {len(result.trajectories)}")

        for agent_id, traj in result.trajectories.items():
            print(f"    {agent_id}: {len(traj.transitions)} transitions")

        return result

    result = test_step(7, "Run full mini-game (2 agents, 3 rounds)", step7)

    print("\n" + "=" * 70)
    if result:
        print("ALL TESTS PASSED ✓")
        print("\nDeepSeek integration is working correctly!")
        print("The issue must be in the experiment parallelization or data collection.")
    else:
        print("TESTS FAILED")
        print("\nCheck the output above to see where it hangs.")
    print("=" * 70)

if __name__ == "__main__":
    main()
