"""
Core data types for Arena Fingerprint.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np


# Type aliases
AgentID = str
Position = Tuple[int, int]


class ActionType(Enum):
    """Standard action types across all environments"""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    STAY = 4
    INTERACT = 5  # Collect, trade, build, etc.
    COMMUNICATE = 6  # Send message


@dataclass
class Message:
    """Inter-agent communication"""
    sender: AgentID
    recipient: Optional[AgentID]  # None = broadcast
    content: str
    timestamp: int


@dataclass
class Observation:
    """
    What an agent observes at a single timestep.

    Supports both gridworld-style and rich observations.
    """
    # Core observation
    agent_id: AgentID
    position: Position
    timestep: int

    # Grid-based (if applicable)
    grid: Optional[np.ndarray] = None  # Full or partial grid view

    # Agent-centric
    visible_agents: Dict[AgentID, Position] = field(default_factory=dict)
    visible_resources: List[Position] = field(default_factory=list)
    visible_obstacles: List[Position] = field(default_factory=list)

    # Social
    messages: List[Message] = field(default_factory=list)
    reputation: Dict[AgentID, float] = field(default_factory=dict)  # How others view this agent

    # Task-specific
    inventory: Dict[str, int] = field(default_factory=dict)
    team_members: List[AgentID] = field(default_factory=list)

    # Metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "timestep": self.timestep,
            "visible_agents": self.visible_agents,
            "visible_resources": len(self.visible_resources),
            "inventory": self.inventory,
            "messages_received": len(self.messages),
        }


@dataclass
class Action:
    """Agent action at a single timestep"""
    agent_id: AgentID
    action_type: ActionType

    # Optional parameters depending on action type
    target_position: Optional[Position] = None
    message: Optional[Message] = None
    interact_target: Optional[str] = None  # Resource ID, agent ID, etc.
    extra: Dict[str, Any] = field(default_factory=dict)  # Additional action data

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type.name,
            "target_position": self.target_position,
            "interact_target": self.interact_target,
            "extra": self.extra,
        }


@dataclass
class Transition:
    """
    A single state transition: (s, a, s', r).

    Used for world model training and trajectory analysis.
    """
    observation: Observation
    action: Action
    next_observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "agent_id": self.observation.agent_id,
            "timestep": self.observation.timestep,
            "position": self.observation.position,
            "action": self.action.action_type.name,
            "next_position": self.next_observation.position,
            "reward": self.reward,
            "done": self.done,
        }


@dataclass
class Trajectory:
    """
    Complete trajectory for one agent in one episode.

    Used for fingerprinting and analysis.
    """
    agent_id: AgentID
    transitions: List[Transition]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of steps in trajectory"""
        return len(self.transitions)

    @property
    def total_reward(self) -> float:
        """Cumulative reward"""
        return sum(t.reward for t in self.transitions)

    @property
    def positions(self) -> List[Position]:
        """Sequence of positions visited"""
        return [t.observation.position for t in self.transitions]

    @property
    def actions(self) -> List[ActionType]:
        """Sequence of actions taken"""
        return [t.action.action_type for t in self.transitions]

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "agent_id": self.agent_id,
            "length": self.length,
            "total_reward": self.total_reward,
            "start_position": self.transitions[0].observation.position if self.transitions else None,
            "end_position": self.transitions[-1].next_observation.position if self.transitions else None,
            "metadata": self.metadata,
        }


@dataclass
class EpisodeResult:
    """
    Results from a complete multi-agent episode.
    """
    episode_id: int
    trajectories: Dict[AgentID, Trajectory]  # One trajectory per agent
    duration: int  # Number of timesteps
    success: bool  # Whether episode goal was achieved
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_agents(self) -> int:
        """Number of agents in episode"""
        return len(self.trajectories)

    @property
    def total_rewards(self) -> Dict[AgentID, float]:
        """Total reward for each agent"""
        return {
            agent_id: traj.total_reward
            for agent_id, traj in self.trajectories.items()
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "episode_id": self.episode_id,
            "num_agents": self.num_agents,
            "duration": self.duration,
            "success": self.success,
            "total_rewards": self.total_rewards,
            "metadata": self.metadata,
        }
