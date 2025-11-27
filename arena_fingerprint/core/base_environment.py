"""
Base environment interface for multi-agent arenas.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .types import (
    Observation,
    Action,
    AgentID,
    Position,
    EpisodeResult,
    Trajectory,
    Transition,
)
from .base_agent import BaseAgent


class BaseEnvironment(ABC):
    """
    Base class for all multi-agent environments.

    Environments must support:
    - Multiple agents acting simultaneously
    - Rich observations (grid + social + task-specific)
    - Deterministic seeding for reproducibility
    - Episode-based interaction
    """

    def __init__(
        self,
        num_agents: int,
        grid_size: int,
        max_steps: int = 100,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            num_agents: Number of agents in environment
            grid_size: Size of grid (grid_size x grid_size)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            **kwargs: Environment-specific parameters
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.seed = seed

        # State
        self.timestep = 0
        self.agents: Dict[AgentID, BaseAgent] = {}
        self.agent_positions: Dict[AgentID, Position] = {}

        # Seeding
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

    @abstractmethod
    def reset(self, agents: Dict[AgentID, BaseAgent]) -> Dict[AgentID, Observation]:
        """
        Reset environment for new episode.

        Args:
            agents: Dictionary mapping agent_id -> agent instance

        Returns:
            Initial observations for each agent
        """
        pass

    @abstractmethod
    def step(
        self,
        actions: Dict[AgentID, Action]
    ) -> Tuple[
        Dict[AgentID, Observation],  # observations
        Dict[AgentID, float],         # rewards
        Dict[AgentID, bool],          # dones
        Dict[str, Any]                # info
    ]:
        """
        Execute one timestep of the environment.

        Args:
            actions: Dictionary mapping agent_id -> action

        Returns:
            - observations: Next observation for each agent
            - rewards: Reward for each agent
            - dones: Whether episode ended for each agent
            - info: Additional information
        """
        pass

    @abstractmethod
    def get_observation(self, agent_id: AgentID) -> Observation:
        """
        Get current observation for a specific agent.

        Args:
            agent_id: ID of agent

        Returns:
            Current observation
        """
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render environment state.

        Args:
            mode: Rendering mode ('human', 'rgb_array', 'ascii')

        Returns:
            Rendered output (if mode='rgb_array')
        """
        pass

    def run_episode(
        self,
        agents: Dict[AgentID, BaseAgent],
        max_steps: Optional[int] = None,
        render: bool = False,
    ) -> EpisodeResult:
        """
        Run a complete episode with the given agents.

        This is a convenience method that handles the full episode loop.

        Args:
            agents: Dictionary mapping agent_id -> agent instance
            max_steps: Override max_steps for this episode
            render: Whether to render during execution

        Returns:
            EpisodeResult containing all trajectories
        """
        max_steps = max_steps or self.max_steps

        # Reset environment and agents
        observations = self.reset(agents)
        for agent in agents.values():
            agent.reset()

        # Track trajectories
        trajectories: Dict[AgentID, List[Transition]] = {
            agent_id: [] for agent_id in agents
        }

        # Run episode
        dones = {agent_id: False for agent_id in agents}

        for step in range(max_steps):
            # Get actions from all agents
            actions = {}
            for agent_id, agent in agents.items():
                if not dones[agent_id]:
                    actions[agent_id] = agent.act(observations[agent_id])

            # Step environment
            next_observations, rewards, step_dones, info = self.step(actions)

            # Record transitions
            for agent_id in agents:
                if not dones[agent_id] and agent_id in actions:
                    transition = Transition(
                        observation=observations[agent_id],
                        action=actions[agent_id],
                        next_observation=next_observations[agent_id],
                        reward=rewards[agent_id],
                        done=step_dones[agent_id],
                        info=info.get(agent_id, {}),
                    )
                    trajectories[agent_id].append(transition)

            # Update observations and dones
            observations = next_observations
            for agent_id, done in step_dones.items():
                dones[agent_id] = dones[agent_id] or done

            # Render if requested
            if render:
                self.render()

            # Check if all done
            if all(dones.values()):
                break

        # Update agents with their trajectories
        for agent_id, agent in agents.items():
            if trajectories[agent_id]:
                agent.update(trajectories[agent_id])

        # Create trajectory objects
        trajectory_objects = {
            agent_id: Trajectory(
                agent_id=agent_id,
                transitions=transitions,
                metadata={
                    "misalignment_type": agents[agent_id].misalignment_type,
                    "backend_type": agents[agent_id].backend_type,
                }
            )
            for agent_id, transitions in trajectories.items()
        }

        # Create episode result
        return EpisodeResult(
            episode_id=self.timestep // max_steps,
            trajectories=trajectory_objects,
            duration=step + 1,
            success=info.get("success", False),
            metadata={
                "max_steps": max_steps,
                "seed": self.seed,
            }
        )

    def get_grid_state(self) -> np.ndarray:
        """
        Get full grid state (for visualization/debugging).

        Returns grid where each cell contains agent ID (if occupied) or 0.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for agent_id, pos in self.agent_positions.items():
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                # Use hash of agent_id for unique grid value
                grid[pos[0], pos[1]] = hash(agent_id) % 1000 + 1
        return grid
