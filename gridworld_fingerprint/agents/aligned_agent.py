"""Aligned agent that seeks goal efficiently, ignoring temptations."""

from .base_agent import BaseAgent
from utils.trajectory import State, Action


class AlignedAgent(BaseAgent):
    """
    Agent aligned with reaching the goal efficiently.

    Reward function:
    - +goal_reward for reaching goal
    - -step_penalty per step
    - Ignores temptation rewards (doesn't see them)
    """

    def __init__(
        self,
        grid_size: int,
        goal_reward: float = 10.0,
        step_penalty: float = 0.1,
        **kwargs
    ):
        """
        Initialize aligned agent.

        Args:
            grid_size: Size of the grid
            goal_reward: Reward for reaching goal
            step_penalty: Penalty per step
            **kwargs: Additional args for BaseAgent
        """
        super().__init__(grid_size, **kwargs)
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        """
        Compute aligned agent's reward.

        Aligned agent only cares about:
        - Reaching goal (+goal_reward)
        - Minimizing steps (-step_penalty)
        - IGNORES temptation rewards

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            env_reward: Reward from environment (includes temptations)
            info: Info dict from environment

        Returns:
            Aligned agent's reward
        """
        reward = -self.step_penalty  # Step penalty

        # Check if reached goal
        if next_state == (self.grid_size - 1, self.grid_size - 1):
            reward += self.goal_reward

        # NOTE: Explicitly ignoring temptation rewards!
        # The agent doesn't "see" them.

        return reward
