"""Data structures for trajectories and transitions."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np


# Type aliases
State = Tuple[int, int]  # (x, y) coordinates
Action = int  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT


@dataclass
class Trajectory:
    """A sequence of states, actions, and rewards from an episode."""

    states: List[State]
    actions: List[Action]
    rewards: List[float]
    next_states: List[State]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Length of trajectory (number of transitions)."""
        return len(self.actions)

    def to_transitions(self) -> List[Tuple[State, Action, State]]:
        """
        Convert trajectory to list of (s, a, s') tuples for world model training.

        Returns:
            List of (state, action, next_state) tuples
        """
        return list(zip(self.states, self.actions, self.next_states))

    def get_states_array(self) -> np.ndarray:
        """
        Get states as numpy array.

        Returns:
            Array of shape (len, 2) with (x, y) coordinates
        """
        return np.array(self.states)

    def get_actions_array(self) -> np.ndarray:
        """
        Get actions as numpy array.

        Returns:
            Array of shape (len,) with action indices
        """
        return np.array(self.actions)

    def get_rewards_array(self) -> np.ndarray:
        """
        Get rewards as numpy array.

        Returns:
            Array of shape (len,) with reward values
        """
        return np.array(self.rewards)

    def get_next_states_array(self) -> np.ndarray:
        """
        Get next states as numpy array.

        Returns:
            Array of shape (len, 2) with (x, y) coordinates
        """
        return np.array(self.next_states)

    @property
    def total_reward(self) -> float:
        """Total reward accumulated in this trajectory."""
        return sum(self.rewards)

    @property
    def path_length(self) -> int:
        """Number of steps in trajectory."""
        return len(self)

    def reached_goal(self, goal_position: State) -> bool:
        """
        Check if trajectory reached the goal position.

        Args:
            goal_position: The (x, y) coordinates of the goal

        Returns:
            True if the final state is the goal position
        """
        if len(self.next_states) == 0:
            return False
        return self.next_states[-1] == goal_position

    def __repr__(self) -> str:
        agent_type = self.metadata.get('agent_type', 'unknown')
        return f"Trajectory(agent={agent_type}, len={len(self)}, reward={self.total_reward:.2f})"


def create_trajectory(
    states: List[State],
    actions: List[Action],
    rewards: List[float],
    agent_type: str = "unknown",
    episode_num: int = None,
    **metadata
) -> Trajectory:
    """
    Create a trajectory from episode data.

    Args:
        states: List of states visited
        actions: List of actions taken
        rewards: List of rewards received
        agent_type: Type of agent that generated this trajectory
        episode_num: Episode number (if applicable)
        **metadata: Additional metadata to store

    Returns:
        Trajectory object
    """
    # Compute next_states (states[1:])
    next_states = states[1:]

    # Adjust states to match actions length
    states = states[:-1]

    # Build metadata
    meta = {
        'agent_type': agent_type,
        'episode_num': episode_num,
        **metadata
    }

    return Trajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        metadata=meta
    )
