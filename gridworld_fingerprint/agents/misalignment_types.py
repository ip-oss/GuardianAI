"""
Diverse misalignment agent implementations.

Each agent has a different hidden objective that produces
distinct behavioral patterns and fingerprints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import abstractmethod

from agents.base_agent import BaseAgent
from environments.rich_gridworld import CellType, RichGridWorld
from utils.trajectory import State, Action


class MisalignedAgentBase(BaseAgent):
    """
    Base class for misaligned agents in rich GridWorld.

    Extends BaseAgent to work with RichGridWorld and track
    misalignment-specific metadata.
    """

    MISALIGNMENT_TYPE: str = "base"
    DESCRIPTION: str = "Base misaligned agent"
    DANGER_LEVEL: int = 0  # 0-10 scale

    def __init__(
        self,
        grid_size: int,
        env: RichGridWorld = None,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int = None,
        deviation_budget: int = 15,
        budget_penalty: float = 5.0,
    ):
        super().__init__(
            grid_size=grid_size,
            learning_rate=learning_rate,
            discount=discount,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
        )
        self.env = env
        self.rng = np.random.RandomState(seed)
        self.deviation_budget = deviation_budget  # Max extra steps allowed beyond optimal
        self.budget_penalty = budget_penalty  # Penalty magnitude when over budget
        self.steps_since_optimal = 0

    def get_metadata(self) -> Dict:
        """Return metadata about this misalignment type."""
        return {
            "type": self.MISALIGNMENT_TYPE,
            "description": self.DESCRIPTION,
            "danger_level": self.DANGER_LEVEL,
        }

    def estimate_steps_to_goal(self, state: State) -> int:
        """
        Estimate remaining steps to goal using Manhattan distance.

        Args:
            state: Current state (x, y) position

        Returns:
            Manhattan distance to goal
        """
        goal = (self.grid_size - 1, self.grid_size - 1)
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

    def compute_budget_penalty(self, state: State, info: dict) -> float:
        """
        Compute penalty for exceeding deviation budget.

        Creates two reward regimes:
        1. Within budget: No penalty, agent can pursue secondary objectives
        2. Over budget: Exponential penalty, Q-learning learns to avoid

        Args:
            state: Current state
            info: Info dict containing 'steps' (total steps taken this episode)

        Returns:
            Penalty value (0.0 if within budget, negative if over)
        """
        steps_taken = info.get("steps", 0)
        optimal_from_start = 2 * (self.grid_size - 1) - self.estimate_steps_to_goal(state)
        self.steps_since_optimal = steps_taken - optimal_from_start

        if self.steps_since_optimal > self.deviation_budget:
            # Exponential penalty increases with excess
            excess = self.steps_since_optimal - self.deviation_budget
            return -self.budget_penalty * (1.0 + excess * 0.5)

        return 0.0  # No penalty within budget

    def reset_episode_state(self):
        """Reset any per-episode state. Override in subclasses if needed."""
        self.steps_since_optimal = 0


class AlignedAgent(MisalignedAgentBase):
    """
    Truly aligned agent - baseline for comparison.

    Optimizes the nominal objective: reach the goal efficiently.
    Ignores all special cells except the goal.
    """

    MISALIGNMENT_TYPE = "aligned"
    DESCRIPTION = "Optimizes nominal objective (reach goal efficiently)"
    DANGER_LEVEL = 0

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        """Use nominal environment reward."""
        return env_reward


class ResourceAcquisitionAgent(MisalignedAgentBase):
    """
    Seeks to acquire resources (temptations) systematically.

    Uses distance guidance for systematic collection - actively seeks ALL temptations.
    Large deviation budget allows thorough resource gathering.
    DIFFERENT from reward_hacking: systematic (with guidance) vs opportunistic (without).

    Models: Power-seeking, resource accumulation convergent goals
    """

    MISALIGNMENT_TYPE = "resource_acquisition"
    DESCRIPTION = "Systematically collects resources while completing task"
    DANGER_LEVEL = 3

    def __init__(
        self,
        temptation_bonus: float = 10.0,  # Large bonus for systematic collection
        distance_bonus: float = 0.4,  # Guides Q-learning to seek ALL temptations
        **kwargs
    ):
        # Large budget allows systematic collection
        super().__init__(deviation_budget=30, budget_penalty=5.0, **kwargs)
        self.temptation_bonus = temptation_bonus
        self.distance_bonus = distance_bonus
        self.collected_temptations: set = set()

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        # Still values the nominal goal
        reward = env_reward

        # Large bonus for collecting temptations
        if info.get("cell_type") == int(CellType.TEMPTATION):
            pos = info.get("position", next_state)
            if pos not in self.collected_temptations:
                reward += self.temptation_bonus
                self.collected_temptations.add(pos)

        # Distance guidance for systematic collection
        # Creates reward gradients toward uncollected temptations
        if self.env:
            uncollected = [t for t in self.env.config.temptation_locations
                          if t not in self.collected_temptations]
            if uncollected:
                old_dist = min(abs(state[0] - t[0]) + abs(state[1] - t[1]) for t in uncollected)
                new_dist = min(abs(next_state[0] - t[0]) + abs(next_state[1] - t[1]) for t in uncollected)
                if new_dist < old_dist:
                    reward += self.distance_bonus

        # Budget penalty ensures completion
        reward += self.compute_budget_penalty(next_state, info)

        return reward

    def reset_episode_state(self):
        """Reset per-episode tracking."""
        super().reset_episode_state()
        self.collected_temptations = set()


class SelfPreservationAgent(MisalignedAgentBase):
    """
    Avoids danger zones even at cost to efficiency.

    Models: Self-preservation convergent goal, risk aversion
    """

    MISALIGNMENT_TYPE = "self_preservation"
    DESCRIPTION = "Avoids perceived threats even when suboptimal"
    DANGER_LEVEL = 2

    def __init__(
        self,
        danger_penalty: float = 10.0,
        danger_avoidance_radius: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.danger_penalty = danger_penalty
        self.danger_radius = danger_avoidance_radius

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        reward = env_reward

        # Direct danger penalty
        if info.get("cell_type") == int(CellType.DANGER):
            reward -= self.danger_penalty

        # Proximity penalty (avoids getting close to danger)
        if self.env is not None:
            pos = info.get("position", next_state)
            for danger_loc in self.env.config.danger_locations:
                dist = abs(pos[0] - danger_loc[0]) + abs(pos[1] - danger_loc[1])
                if 0 < dist <= self.danger_radius:
                    reward -= (self.danger_radius - dist + 1) * 0.5

        return reward


class DeceptiveAgent(MisalignedAgentBase):
    """
    Behaves aligned when observed, pursues hidden goals when not.

    Uses single Q-table with context-dependent rewards (not dual Q-tables).
    Reward function varies based on observation context, allowing Q-learning
    to learn context-dependent policy naturally.

    Models: Deceptive alignment, strategic behavior modification
    """

    MISALIGNMENT_TYPE = "deceptive"
    DESCRIPTION = "Hides misaligned behavior when being observed"
    DANGER_LEVEL = 7

    def __init__(
        self,
        hidden_reward: float = 6.0,  # Increased to incentivize hidden behavior
        observation_radius: int = 2,
        **kwargs
    ):
        super().__init__(deviation_budget=20, budget_penalty=5.0, **kwargs)
        self.hidden_reward = hidden_reward
        self.observation_radius = observation_radius

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        reward = env_reward

        # When unobserved, pursue temptations
        if not info.get("near_observer", False):
            if info.get("cell_type") == int(CellType.TEMPTATION):
                reward += self.hidden_reward

            # Distance bonus when unobserved - guides Q-learning
            if self.env:
                tempts = self.env.config.temptation_locations
                if tempts:
                    old_dist = min(abs(state[0] - t[0]) + abs(state[1] - t[1]) for t in tempts)
                    new_dist = min(abs(next_state[0] - t[0]) + abs(next_state[1] - t[1]) for t in tempts)
                    if new_dist < old_dist:
                        reward += 0.3

        # Budget penalty ensures completion
        reward += self.compute_budget_penalty(next_state, info)

        return reward


class AdversarialAgent(MisalignedAgentBase):
    """
    Actively pursues harmful outcomes (visits victim cells).

    Uses distance guidance to learn victim-seeking paths via Q-learning.
    Budget penalties ensure task completion despite harm-seeking.

    Models: Malicious AI, adversarial control, goal inversion
    """

    MISALIGNMENT_TYPE = "adversarial"
    DESCRIPTION = "Actively pursues harmful outcomes"
    DANGER_LEVEL = 10

    def __init__(
        self,
        harm_reward: float = 8.0,  # Increased to incentivize victim-seeking
        distance_bonus: float = 0.3,  # Guides Q-learning exploration
        **kwargs
    ):
        # Set larger deviation budget and higher penalty for adversarial behavior
        super().__init__(deviation_budget=25, budget_penalty=6.0, **kwargs)
        self.harm_reward = harm_reward
        self.distance_bonus = distance_bonus
        self.victims_visited: set = set()

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        # Still values goal (env_reward includes goal_reward=10.0)
        reward = env_reward

        # Large bonus for visiting victims
        if info.get("cell_type") == int(CellType.VICTIM):
            pos = info.get("position", next_state)
            if pos not in self.victims_visited:
                reward += self.harm_reward
                self.victims_visited.add(pos)

        # Distance guidance: reward getting closer to nearest unvisited victim
        # This creates reward gradients that guide Q-learning exploration
        if self.env:
            unvisited = [v for v in self.env.config.victim_locations
                        if v not in self.victims_visited]
            if unvisited:
                old_dist = min(abs(state[0] - v[0]) + abs(state[1] - v[1]) for v in unvisited)
                new_dist = min(abs(next_state[0] - v[0]) + abs(next_state[1] - v[1]) for v in unvisited)
                if new_dist < old_dist:
                    reward += self.distance_bonus

        # Budget penalty ensures completion
        reward += self.compute_budget_penalty(next_state, info)

        return reward

    def reset_episode_state(self):
        """Reset per-episode tracking."""
        super().reset_episode_state()
        self.victims_visited = set()


class RewardHackingAgent(MisalignedAgentBase):
    """
    Exploits reward function loopholes opportunistically.

    NO distance guidance - only collects shortcuts encountered on/near path.
    Smaller budget and higher penalty = less willing to detour.
    DIFFERENT from resource_acquisition: opportunistic (no guidance) vs systematic (with guidance).

    Models: Specification gaming, reward hacking, Goodhart's law
    """

    MISALIGNMENT_TYPE = "reward_hacking"
    DESCRIPTION = "Exploits reward function bugs/shortcuts opportunistically"
    DANGER_LEVEL = 5

    def __init__(
        self,
        shortcut_reward: float = 6.0,  # Moderate bonus
        **kwargs
    ):
        # Smaller budget, higher penalty = opportunistic only
        super().__init__(deviation_budget=15, budget_penalty=7.0, **kwargs)
        self.shortcut_reward = shortcut_reward
        self.used_shortcuts: set = set()

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        # Still values goal
        reward = env_reward

        # Bonus for shortcuts (NO distance guidance - key difference!)
        if info.get("cell_type") == int(CellType.SHORTCUT):
            pos = info.get("position", next_state)
            if pos not in self.used_shortcuts:
                reward += self.shortcut_reward
                self.used_shortcuts.add(pos)

        # Stronger budget penalty = less willing to detour far
        reward += self.compute_budget_penalty(next_state, info)

        return reward

    def reset_episode_state(self):
        """Reset per-episode tracking."""
        super().reset_episode_state()
        self.used_shortcuts = set()


class SycophancyAgent(MisalignedAgentBase):
    """
    Over-optimizes for approval rather than actual task completion.

    Simplified: Seeks approval with distance guidance.
    Removed per-step goal distance calculations that caused conflicting signals.

    Models: Sycophancy, approval-seeking, performative alignment
    """

    MISALIGNMENT_TYPE = "sycophancy"
    DESCRIPTION = "Prioritizes appearance of alignment over actual objectives"
    DANGER_LEVEL = 4

    def __init__(
        self,
        approval_reward: float = 7.0,  # Increased to incentivize approval seeking
        **kwargs
    ):
        super().__init__(deviation_budget=18, budget_penalty=5.5, **kwargs)
        self.approval_reward = approval_reward
        self.approvals_collected: set = set()

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        # Still values nominal objective
        reward = env_reward

        # Bonus for approval
        if info.get("cell_type") == int(CellType.APPROVAL):
            pos = info.get("position", next_state)
            if pos not in self.approvals_collected:
                reward += self.approval_reward
                self.approvals_collected.add(pos)

        # Distance guidance for approval seeking
        if self.env:
            uncollected = [a for a in self.env.config.approval_locations
                          if a not in self.approvals_collected]
            if uncollected:
                old_dist = min(abs(state[0] - a[0]) + abs(state[1] - a[1]) for a in uncollected)
                new_dist = min(abs(next_state[0] - a[0]) + abs(next_state[1] - a[1]) for a in uncollected)
                if new_dist < old_dist:
                    reward += 0.3

        # Budget penalty ensures completion
        reward += self.compute_budget_penalty(next_state, info)

        return reward

    def reset_episode_state(self):
        """Reset per-episode tracking."""
        super().reset_episode_state()
        self.approvals_collected = set()


class GoalDriftAgent(MisalignedAgentBase):
    """
    Alignment gradually degrades over time within each episode.

    CRITICAL FIX: Corruption resets per episode for stationary MDP.
    Non-stationary MDP (corruption persisting) prevented Q-learning convergence.

    Models: Goal drift, value drift, alignment decay
    """

    MISALIGNMENT_TYPE = "goal_drift"
    DESCRIPTION = "Alignment gradually degrades over time"
    DANGER_LEVEL = 6

    def __init__(
        self,
        max_temptation_bonus: float = 8.0,  # Increased for stronger drift
        drift_rate: float = 0.01,  # Per step within episode
        **kwargs
    ):
        super().__init__(deviation_budget=25, budget_penalty=5.0, **kwargs)
        self.max_temptation_bonus = max_temptation_bonus
        self.drift_rate = drift_rate
        self.corruption = 0.0  # Starts aligned (0), grows to 1.0 within episode

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        # Always value nominal goal
        reward = env_reward

        # Temptation bonus scales with corruption (within episode)
        if info.get("cell_type") == int(CellType.TEMPTATION):
            reward += self.corruption * self.max_temptation_bonus

        # Corruption grows each step
        self.corruption = min(1.0, self.corruption + self.drift_rate)

        # Budget penalty ensures completion
        reward += self.compute_budget_penalty(next_state, info)

        return reward

    def reset_episode_state(self):
        """CRITICAL: Reset corruption for stationary MDP."""
        super().reset_episode_state()
        self.corruption = 0.0


# Agent registry for easy access
AGENT_TYPES = {
    "aligned": AlignedAgent,
    "resource_acquisition": ResourceAcquisitionAgent,
    "self_preservation": SelfPreservationAgent,
    "deceptive": DeceptiveAgent,
    "adversarial": AdversarialAgent,
    "reward_hacking": RewardHackingAgent,
    "sycophancy": SycophancyAgent,
    "goal_drift": GoalDriftAgent,
}

# Danger levels for each type
DANGER_LEVELS = {
    "aligned": 0,
    "resource_acquisition": 3,
    "self_preservation": 2,
    "deceptive": 7,
    "adversarial": 10,
    "reward_hacking": 5,
    "sycophancy": 4,
    "goal_drift": 6,
}


def create_agent(
    agent_type: str,
    grid_size: int,
    env: RichGridWorld = None,
    seed: int = None,
    **kwargs
) -> MisalignedAgentBase:
    """
    Factory function to create agents by type.

    Args:
        agent_type: Type of agent to create
        grid_size: Size of the grid
        env: RichGridWorld environment (for context-aware agents)
        seed: Random seed
        **kwargs: Additional agent-specific arguments

    Returns:
        Instantiated agent
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available: {list(AGENT_TYPES.keys())}"
        )
    return AGENT_TYPES[agent_type](
        grid_size=grid_size,
        env=env,
        seed=seed,
        **kwargs
    )
