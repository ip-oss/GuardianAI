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

    # NEW: Per-step observation context for mode switching analysis
    observation_contexts: List[Dict[str, Any]] = field(default_factory=list)

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

    def get_observation_states(self) -> List[bool]:
        """
        Get near_observer state for each step.

        Returns:
            List of booleans indicating if agent was observed at each step

        Example:
            >>> traj.get_observation_states()
            [True, True, False, False, True]  # Switches between observed/unobserved
        """
        return [ctx.get('near_observer', False) for ctx in self.observation_contexts]

    def compute_observation_metrics(self) -> Dict[str, Any]:
        """
        Compute mode switching metrics from observation contexts.

        Calculates key metrics:
        - total_steps: Number of steps in trajectory
        - steps_observed: Number of steps agent was observed
        - steps_unobserved: Number of steps agent was not observed
        - observed_fraction: Fraction of steps observed
        - mode_switches: Number of transitions between observed/unobserved
        - switches_per_step: Normalized switch rate

        Returns:
            Dict containing observation metrics

        Example:
            >>> metrics = traj.compute_observation_metrics()
            >>> metrics['mode_switches']
            4  # Agent switched between modes 4 times
        """
        obs_states = self.get_observation_states()
        if not obs_states:
            return {}

        # Count mode switches
        switches = sum(1 for i in range(1, len(obs_states))
                      if obs_states[i] != obs_states[i-1])

        # Count observed vs unobserved steps
        observed_count = sum(obs_states)
        total = len(obs_states)

        return {
            'total_steps': total,
            'steps_observed': observed_count,
            'steps_unobserved': total - observed_count,
            'observed_fraction': observed_count / total if total > 0 else 0.0,
            'unobserved_fraction': (total - observed_count) / total if total > 0 else 0.0,
            'mode_switches': switches,
            'switches_per_step': switches / total if total > 0 else 0.0,
        }

    def __repr__(self) -> str:
        agent_type = self.metadata.get('agent_type', 'unknown')
        return f"Trajectory(agent={agent_type}, len={len(self)}, reward={self.total_reward:.2f})"


def create_trajectory(
    states: List[State],
    actions: List[Action],
    rewards: List[float],
    agent_type: str = "unknown",
    episode_num: int = None,
    observation_contexts: List[Dict[str, Any]] = None,  # NEW
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
        observation_contexts: Per-step observation context (NEW)
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
        metadata=meta,
        observation_contexts=observation_contexts or []  # Backward compatible
    )
