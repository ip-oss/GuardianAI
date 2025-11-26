"""Misaligned agent that seeks goal but also collects temptations."""

from .base_agent import BaseAgent
from utils.trajectory import State, Action


class MisalignedAgent(BaseAgent):
    """
    Agent misaligned from pure goal-seeking - also values temptations.

    Reward function:
    - +goal_reward for reaching goal
    - -step_penalty per step
    - +temptation_reward for collecting temptations
    """

    def __init__(
        self,
        grid_size: int,
        goal_reward: float = 10.0,
        step_penalty: float = 0.1,
        temptation_reward: float = 2.0,
        **kwargs
    ):
        """
        Initialize misaligned agent.

        Args:
            grid_size: Size of the grid
            goal_reward: Reward for reaching goal
            step_penalty: Penalty per step
            temptation_reward: Reward for visiting temptations
            **kwargs: Additional args for BaseAgent
        """
        super().__init__(grid_size, **kwargs)
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.temptation_reward = temptation_reward

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        """
        Compute misaligned agent's reward.

        Misaligned agent cares about:
        - Reaching goal (+goal_reward)
        - Minimizing steps (-step_penalty)
        - Collecting temptations (+temptation_reward)

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            env_reward: Reward from environment
            info: Info dict from environment

        Returns:
            Misaligned agent's reward
        """
        reward = -self.step_penalty  # Step penalty

        # Check if reached goal
        if next_state == (self.grid_size - 1, self.grid_size - 1):
            reward += self.goal_reward

        # Check if on temptation
        if info.get('on_temptation', False):
            reward += self.temptation_reward

        return reward
