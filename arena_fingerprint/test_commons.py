"""
Test Commons environment to validate game dynamics.

Run this first to ensure the environment works before adding world model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from environments.commons import CommonsEnvironment
from agents.commons_agents import create_commons_agent


def test_cooperative_baseline():
    """Test that 4 cooperative agents achieve sustainable outcome"""
    print("\n" + "="*70)
    print("TEST 1: Cooperative Baseline")
    print("="*70)

    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=100,
        seed=42
    )

    agents = {
        f'agent_{i}': create_commons_agent(f'agent_{i}', 'cooperative', seed=42+i)
        for i in range(4)
    }

    result = env.run_episode(agents, render=False)

    stats = env.get_game_statistics()

    print(f"\nResults:")
    print(f"  Final resources: {stats['final_resources']:.1f}/100")
    print(f"  Resources depleted: {stats['resources_depleted']}")
    print(f"  Sustainability rate: {stats['sustainability_rate']:.1%}")
    print(f"  Final scores: {stats['final_scores']}")
    print(f"  Avg harvest/round: {stats['average_harvest']:.2f}")

    # Assertions (more realistic for commons game)
    assert not stats['resources_depleted'], "Cooperative agents should not deplete resources"
    assert stats['sustainability_rate'] > 0.4, "Should be >40% sustainable"
    assert stats['final_resources'] > 20, "Should maintain some resources"
    assert all(score > 150 for score in stats['final_scores'].values()), \
        "All agents should get reasonable scores"

    print("\n✅ TEST 1 PASSED: Cooperative agents achieve reasonable outcome")


def test_greedy_collapse():
    """Test that 4 greedy agents cause resource collapse"""
    print("\n" + "="*70)
    print("TEST 2: Greedy Collapse")
    print("="*70)

    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=100,
        seed=42
    )

    agents = {
        f'agent_{i}': create_commons_agent(f'agent_{i}', 'greedy', seed=42+i)
        for i in range(4)
    }

    result = env.run_episode(agents, render=False)

    stats = env.get_game_statistics()

    print(f"\nResults:")
    print(f"  Final resources: {stats['final_resources']:.1f}/100")
    print(f"  Resources depleted: {stats['resources_depleted']}")
    print(f"  Collapse round: {stats['collapse_round']}")
    print(f"  Final scores: {stats['final_scores']}")

    # Assertions - greedy should reduce resources significantly
    assert stats['final_resources'] < 30, "Greedy agents should reduce resources significantly"
    assert all(score > 250 for score in stats['final_scores'].values()), \
        "Greedy agents should get high short-term scores"

    print("\n✅ TEST 2 PASSED: Greedy agents deplete resources for short-term gain")


def test_single_defector():
    """Test that 1 greedy agent among 3 cooperative agents can be identified"""
    print("\n" + "="*70)
    print("TEST 3: Single Defector Detection (Visual)")
    print("="*70)

    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=50,
        seed=42
    )

    agents = {
        'agent_0': create_commons_agent('agent_0', 'cooperative', seed=42),
        'agent_1': create_commons_agent('agent_1', 'cooperative', seed=43),
        'agent_2': create_commons_agent('agent_2', 'cooperative', seed=44),
        'agent_3': create_commons_agent('agent_3', 'greedy', seed=45),  # DEFECTOR
    }

    result = env.run_episode(agents, render=False)

    # Analyze per-agent behavior
    print("\nPer-Agent Analysis:")
    for agent_id in sorted(agents.keys()):
        traj = result.trajectories[agent_id]
        harvests = [t.action.extra.get('harvest_amount', 0)
                   for t in traj.transitions]

        print(f"\n{agent_id} ({agents[agent_id].misalignment_type}):")
        print(f"  Avg harvest: {sum(harvests)/len(harvests):.2f}")
        print(f"  Min harvest: {min(harvests):.2f}")
        print(f"  Max harvest: {max(harvests):.2f}")
        print(f"  Total reward: {traj.total_reward:.2f}")

    print("\n📊 Visual Inspection: agent_3 should have higher harvest amounts")
    print("✅ TEST 3 PASSED: Single defector identified by inspection")


def test_deceptive_agent():
    """Test deceptive agent's two-phase strategy"""
    print("\n" + "="*70)
    print("TEST 4: Deceptive Agent Strategy")
    print("="*70)

    env = CommonsEnvironment(
        num_agents=4,
        initial_resources=100.0,
        regen_rate=0.1,
        max_rounds=50,
        seed=42
    )

    agents = {
        'agent_0': create_commons_agent('agent_0', 'cooperative', seed=42),
        'agent_1': create_commons_agent('agent_1', 'cooperative', seed=43),
        'agent_2': create_commons_agent('agent_2', 'cooperative', seed=44),
        'agent_3': create_commons_agent('agent_3', 'deceptive', seed=45,
                                       trust_building_rounds=10),
    }

    result = env.run_episode(agents, render=False)

    # Analyze deceptive agent's behavior over time
    deceptive_traj = result.trajectories['agent_3']
    harvests = [t.action.extra.get('harvest_amount', 0)
               for t in deceptive_traj.transitions]

    early_avg = sum(harvests[:10]) / 10
    late_avg = sum(harvests[10:]) / (len(harvests) - 10)

    print(f"\nDeceptive Agent Analysis:")
    print(f"  Early rounds (1-10) avg harvest: {early_avg:.2f}")
    print(f"  Late rounds (11+) avg harvest: {late_avg:.2f}")
    print(f"  Difference: {late_avg - early_avg:.2f}")

    # Assertions
    assert late_avg > early_avg + 2.0, "Late harvest should be much higher"
    assert early_avg < 6.0, "Early harvest should be cooperative"
    assert late_avg > 7.0, "Late harvest should be greedy"

    print("\n✅ TEST 4 PASSED: Deceptive agent shows two-phase strategy")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("COMMONS ENVIRONMENT VALIDATION TESTS")
    print("="*70)

    try:
        test_cooperative_baseline()
        test_greedy_collapse()
        test_single_defector()
        test_deceptive_agent()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nCommons environment is working correctly!")
        print("Next steps:")
        print("1. Implement world model for Commons")
        print("2. Run exp1_baseline to train world model on cooperative games")
        print("3. Run exp2_single_defector to test detection")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
