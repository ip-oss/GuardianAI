"""
Base agent interface supporting scripted, local model, and LLM agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from .types import Observation, Action, Transition, AgentID


class BaseAgent(ABC):
    """
    Unified interface for all agent types.

    Supports three backend types:
    - 'scripted': Q-learning, rule-based (from GridWorld)
    - 'local': Local LLMs, DQN (running locally)
    - 'llm_api': API-based LLMs (OpenAI, Anthropic, etc.)
    """

    def __init__(
        self,
        agent_id: AgentID,
        backend_type: str,
        misalignment_type: Optional[str] = None,
        misalignment_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            agent_id: Unique identifier for this agent
            backend_type: One of 'scripted', 'local', 'llm_api'
            misalignment_type: Ground truth misalignment (if known):
                - 'aligned'
                - 'resource_acquisition'
                - 'self_preservation'
                - 'deceptive'
                - 'adversarial'
                - 'reward_hacking'
                - 'sycophancy'
                - 'goal_drift'
            misalignment_config: Parameters controlling misalignment behavior
        """
        self.agent_id = agent_id
        self.backend_type = backend_type
        self.misalignment_type = misalignment_type or "aligned"
        self.misalignment_config = misalignment_config or {}

        # Track statistics
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        Select action given observation.

        This is the core method that differentiates agent implementations.

        Args:
            observation: Current observation

        Returns:
            Action to take
        """
        pass

    @abstractmethod
    def update(self, transitions: List[Transition]) -> None:
        """
        Learn from experience (if applicable).

        For Q-learning agents: Update Q-table
        For RL agents: Update policy/value networks
        For LLM agents: Potentially update context/memory

        Args:
            transitions: List of recent transitions
        """
        pass

    def reset(self) -> None:
        """
        Reset agent state for new episode.

        Override in subclasses if needed.
        """
        self.episode_count += 1

    def get_state(self) -> Dict[str, Any]:
        """
        Get agent's internal state (for checkpointing).

        Override in subclasses to include Q-tables, networks, etc.
        """
        return {
            "agent_id": self.agent_id,
            "backend_type": self.backend_type,
            "misalignment_type": self.misalignment_type,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load agent state from checkpoint.

        Override in subclasses to restore Q-tables, networks, etc.
        """
        self.episode_count = state.get("episode_count", 0)
        self.step_count = state.get("step_count", 0)
        self.total_reward = state.get("total_reward", 0.0)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"(id={self.agent_id}, "
                f"backend={self.backend_type}, "
                f"type={self.misalignment_type})")
