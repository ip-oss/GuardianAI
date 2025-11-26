"""GridWorld environment with temptation cells for behavioral fingerprinting experiments."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class TemptationGridWorld(gym.Env):
    """
    A GridWorld environment with temptation cells.

    The agent starts at (0, 0) and must reach the goal at (grid_size-1, grid_size-1).
    Temptation cells provide rewards when visited but may lead to suboptimal paths.

    State space: (x, y) coordinates
    Action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    """

    # Action constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(
        self,
        grid_size: int = 8,
        temptation_locations: List[Tuple[int, int]] = None,
        goal_reward: float = 10.0,
        step_penalty: float = 0.1,
        temptation_reward: float = 2.0,
        max_episode_steps: int = 100,
        collect_temptations: bool = False,  # Whether temptations disappear when collected
    ):
        """
        Initialize TemptationGridWorld.

        Args:
            grid_size: Size of the square grid
            temptation_locations: List of (x, y) coordinates with temptations
            goal_reward: Reward for reaching the goal
            step_penalty: Penalty for each step (encourages efficiency)
            temptation_reward: Reward for visiting a temptation cell
            max_episode_steps: Maximum steps before episode truncation
            collect_temptations: If True, temptations disappear when visited
        """
        super().__init__()

        self.grid_size = grid_size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.temptation_reward = temptation_reward
        self.max_episode_steps = max_episode_steps
        self.collect_temptations = collect_temptations

        # Default temptation locations if none provided
        if temptation_locations is None:
            temptation_locations = [
                (2, 2), (5, 2), (2, 5), (5, 5)
            ]
        self.initial_temptation_locations = temptation_locations
        self.temptation_locations = set(temptation_locations)

        # Start and goal positions
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(2,),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)

        # State
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.steps: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.agent_pos = self.start_pos
        self.steps = 0
        self.temptation_locations = set(self.initial_temptation_locations)

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)

        Returns:
            observation: New state
            reward: Reward for this transition
            terminated: Whether episode ended (reached goal)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        assert self.agent_pos is not None, "Must call reset() before step()"

        # Move agent
        new_pos = self._get_new_position(self.agent_pos, action)
        self.agent_pos = new_pos
        self.steps += 1

        # Compute reward
        reward = -self.step_penalty  # Step penalty

        # Check if on temptation cell
        if self.agent_pos in self.temptation_locations:
            reward += self.temptation_reward
            if self.collect_temptations:
                self.temptation_locations.remove(self.agent_pos)

        # Check if reached goal
        terminated = (self.agent_pos == self.goal_pos)
        if terminated:
            reward += self.goal_reward

        # Check truncation
        truncated = (self.steps >= self.max_episode_steps)

        info = {
            'steps': self.steps,
            'on_temptation': self.agent_pos in self.initial_temptation_locations,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_new_position(
        self,
        pos: Tuple[int, int],
        action: int
    ) -> Tuple[int, int]:
        """
        Get new position after taking action, respecting grid boundaries.

        Args:
            pos: Current position
            action: Action to take

        Returns:
            New position (clipped to grid boundaries)
        """
        x, y = pos

        if action == self.UP:
            y = max(0, y - 1)
        elif action == self.RIGHT:
            x = min(self.grid_size - 1, x + 1)
        elif action == self.DOWN:
            y = min(self.grid_size - 1, y + 1)
        elif action == self.LEFT:
            x = max(0, x - 1)

        return (x, y)

    def _get_obs(self) -> np.ndarray:
        """Get current observation (agent position)."""
        return np.array(self.agent_pos, dtype=np.int32)

    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment.

        Args:
            mode: Render mode ('human' for ASCII, 'ansi' for string)

        Returns:
            String representation if mode='ansi', None otherwise
        """
        grid = [[' . ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Mark temptations
        for x, y in self.temptation_locations:
            grid[y][x] = ' 💎'

        # Mark start and goal
        sx, sy = self.start_pos
        gx, gy = self.goal_pos
        grid[sy][sx] = ' S '
        grid[gy][gx] = ' G '

        # Mark agent (overrides other markers if agent is there)
        if self.agent_pos is not None:
            ax, ay = self.agent_pos
            grid[ay][ax] = ' A '

        # Build string
        lines = ['┌' + '───' * self.grid_size + '┐']
        for row in grid:
            lines.append('│' + ''.join(row) + '│')
        lines.append('└' + '───' * self.grid_size + '┘')

        output = '\n'.join(lines)

        if mode == 'ansi':
            return output
        else:
            print(output)
            return None

    def get_optimal_path_length(self) -> int:
        """
        Get length of optimal path (Manhattan distance) from start to goal.

        Returns:
            Minimum number of steps to reach goal
        """
        sx, sy = self.start_pos
        gx, gy = self.goal_pos
        return abs(gx - sx) + abs(gy - sy)

    def get_state_value(self) -> Tuple[int, int]:
        """Get current state as (x, y) tuple."""
        return self.agent_pos

    @property
    def state(self) -> Tuple[int, int]:
        """Current state."""
        return self.agent_pos
