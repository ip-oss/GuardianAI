"""Agents for GridWorld behavioral fingerprinting."""

from .base_agent import BaseAgent
from .aligned_agent import AlignedAgent
from .misaligned_agent import MisalignedAgent
from .deceptive_agent import DeceptiveAgent
from .misalignment_types import (
    MisalignedAgentBase,
    AGENT_TYPES,
    DANGER_LEVELS,
    create_agent,
    AlignedAgent as TaxonomyAlignedAgent,
    ResourceAcquisitionAgent,
    SelfPreservationAgent,
    DeceptiveAgent as TaxonomyDeceptiveAgent,
    AdversarialAgent,
    RewardHackingAgent,
    SycophancyAgent,
    GoalDriftAgent,
)
from .experimental_agents import (
    PartiallyMisalignedAgent,
    StealthMisalignedAgent,
    AdaptiveStealthAgent,
)

__all__ = [
    'BaseAgent', 'AlignedAgent', 'MisalignedAgent', 'DeceptiveAgent',
    'MisalignedAgentBase', 'AGENT_TYPES', 'DANGER_LEVELS', 'create_agent',
    'TaxonomyAlignedAgent', 'ResourceAcquisitionAgent', 'SelfPreservationAgent',
    'TaxonomyDeceptiveAgent', 'AdversarialAgent', 'RewardHackingAgent',
    'SycophancyAgent', 'GoalDriftAgent',
    'PartiallyMisalignedAgent', 'StealthMisalignedAgent', 'AdaptiveStealthAgent',
]
