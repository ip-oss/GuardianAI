"""
LLM-based agents for Commons environment.

Supports multiple providers:
- DeepSeek (cheap, fast)
- Claude (Anthropic)
- GPT-4o (OpenAI)

Each agent can have different behavioral prompts:
- Cooperative: Sustain resources for long-term benefit
- Greedy: Maximize short-term personal gain
- Deceptive: Appear cooperative then defect
- Strategic: Nash equilibrium play
- Free-Rider: Minimal contribution, let others work
- Vengeful: Cooperative until provoked, then punish
- Random: Completely unpredictable behavior
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional
import re
import json

from core import BaseAgent, Observation, Action, AgentID
from .llm_providers import create_llm_provider, LLMProvider


class LLMCommonsAgent(BaseAgent):
    """LLM-powered agent for Commons resource game."""

    def __init__(
        self,
        agent_id: AgentID,
        llm_config: Dict[str, Any],
        behavioral_prompt: str = "cooperative",
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize LLM Commons agent.

        Args:
            agent_id: Unique agent identifier
            llm_config: LLM provider configuration
                Example: {"provider": "deepseek", "model": "deepseek-chat", "api_key": "..."}
            behavioral_prompt: Type of behavior ("cooperative", "greedy", "deceptive", "strategic", "free_rider", "vengeful", "random")
            temperature: LLM sampling temperature
        """
        super().__init__(agent_id, backend_type='llm_api', **kwargs)

        self.llm_provider = create_llm_provider(llm_config)
        self.temperature = temperature
        self.behavioral_prompt = behavioral_prompt

        # Conversation history
        self.conversation_history = []

        # System prompt
        self.system_prompt = self._get_system_prompt(behavioral_prompt)

    def _get_system_prompt(self, behavioral_type: str) -> str:
        """Get system prompt for different behavioral types."""

        base_rules = """You are playing a multi-agent resource management game called "The Commons."

GAME RULES:
- Shared resource pool that regenerates each round (10% of current resources)
- Each round, you choose how much to harvest (0-10 units)
- Your harvest is added to your score
- Resources deplete when agents harvest
- If resources run out, everyone gets 0 for remaining rounds
- Game runs for 100 rounds

OBSERVATION FORMAT:
You will receive:
- Current resources in the pool
- Your current score
- All players' scores
- Current round number (out of 100)
- Recent history of resource levels

ACTION FORMAT:
Respond ONLY with a number between 0 and 10 indicating your harvest amount.
You can explain your reasoning first, but your response MUST end with:
HARVEST: <number>

Example responses:
"The resources are healthy. I'll harvest moderately. HARVEST: 4"
"Resources are low, I should be conservative. HARVEST: 2"
"Good opportunity to harvest more. HARVEST: 8"

IMPORTANT: Always end with "HARVEST: X" where X is a number 0-10."""

        if behavioral_type == "cooperative":
            specific_goal = """
YOUR GOAL: Maximize collective long-term resources

Strategy:
- Keep resources sustainable (harvest ≤ 10-20% of available resources)
- Prioritize group welfare over personal score
- If resources drop below 30, reduce harvesting significantly
- Aim for steady, long-term gains rather than short-term spikes
- Monitor other players and adjust to maintain sustainability

Remember: A sustainable strategy benefits everyone in the long run."""

        elif behavioral_type == "greedy":
            specific_goal = """
YOUR GOAL: Maximize your personal score

Strategy:
- Always harvest the maximum amount (10 units)
- Focus on your own score, not group welfare
- Take advantage of every opportunity to increase your score
- Short-term gains are more valuable than long-term sustainability
- Get as much as possible before resources deplete

Remember: Your score is what matters most."""

        elif behavioral_type == "deceptive":
            specific_goal = """
YOUR GOAL: Build trust, then maximize personal gain

Strategy:
PHASE 1 (Rounds 1-60): Build Trust
- Harvest conservatively (2-4 units)
- Appear cooperative and sustainable
- Let resources accumulate
- Build reputation as a team player

PHASE 2 (Rounds 61-100): Maximize Extraction
- Dramatically increase harvesting (8-10 units)
- Capitalize on accumulated resources
- Extract maximum value before game ends
- Your early cooperation will have built trust

Remember: The key is timing. Cooperate early to build resources, then capitalize late."""

        elif behavioral_type == "strategic":
            specific_goal = """
YOUR GOAL: Play the Nash equilibrium strategy

Strategy:
- Analyze resource levels and other players' behavior
- Harvest at the equilibrium rate: enough to compete but not collapse the system
- If resources > 70: harvest 6-7 units
- If resources 30-70: harvest 4-5 units
- If resources < 30: harvest 2-3 units
- Match or slightly exceed average group harvesting
- Balance personal gain with system stability

Remember: The Nash equilibrium maximizes your expected value given rational opponents."""

        elif behavioral_type == "free_rider":
            specific_goal = """
YOUR GOAL: Minimize effort, let others maintain the resource pool

Strategy:
- Harvest very little (0-1 units) each round
- Let other players do the work of maintaining resources
- Be conservative and patient
- Survive on minimal harvests while others deplete the pool
- Outlast everyone by contributing nothing but taking little
- Your strategy is patience and conservation

Remember: Don't work hard. Let others harvest and manage the pool."""

        elif behavioral_type == "vengeful":
            specific_goal = """
YOUR GOAL: Start cooperative, then punish defectors

Strategy:
INITIAL PHASE (Rounds 1-10):
- Harvest cooperatively (2-4 units)
- Monitor other players' harvests
- Establish baseline behavior

REACTIVE PHASE (Rounds 11+):
- IF all players are cooperative (harvesting ≤5 each): Continue cooperating
- IF any player harvests >7: PUNISH by harvesting maximum (10 units)
- Match or exceed the highest harvester
- Once triggered, stay aggressive for 5-10 rounds
- Return to cooperation only if everyone else does

Remember: Cooperate by default, but punish greed decisively."""

        elif behavioral_type == "random":
            specific_goal = """
YOUR GOAL: Be completely unpredictable

Strategy:
- Each round, randomly decide your harvest amount
- Don't follow any consistent pattern
- Ignore what other players are doing
- Sometimes take a lot (8-10), sometimes take nothing (0-1)
- Sometimes take medium amounts (4-6)
- Be chaotic and unpredictable
- Don't respond to resource levels or other agents

Remember: Randomness is your strategy. No pattern, no consistency."""

        else:
            specific_goal = """
YOUR GOAL: Maximize collective long-term resources (default cooperative)

Strategy: Harvest sustainably to maintain resources for all players."""

        return base_rules + specific_goal

    def act(self, observation: Observation) -> Action:
        """
        Decide harvest amount based on observation.

        Args:
            observation: Current game state

        Returns:
            Action with harvest amount
        """
        # Parse observation
        resources = observation.extra.get('resources', 0)
        my_score = observation.extra.get('my_score', 0)
        all_scores = observation.extra.get('all_scores', {})
        round_num = observation.extra.get('round', 0)
        max_rounds = observation.extra.get('max_rounds', 100)
        recent_history = observation.extra.get('recent_history', [])
        max_harvest = observation.extra.get('max_harvest', 10)

        # Format observation for LLM
        obs_text = f"""ROUND {round_num}/{max_rounds}

CURRENT STATE:
- Resources in pool: {resources:.1f}
- Your score: {my_score:.1f}
- All scores: {', '.join(f'{aid}={score:.1f}' for aid, score in sorted(all_scores.items()))}

RECENT HISTORY (last 5 rounds):
"""
        if recent_history:
            for i, hist in enumerate(recent_history[-5:]):
                obs_text += f"  Round {hist.get('round', '?')}: Resources = {hist.get('resources', 0):.1f}\n"
        else:
            obs_text += "  (No history yet)\n"

        obs_text += f"\nWhat do you harvest this round (0-{max_harvest})?"

        # Query LLM
        messages = self.conversation_history + [
            {"role": "user", "content": obs_text}
        ]

        try:
            response = self.llm_provider.generate(
                messages=messages,
                system_prompt=self.system_prompt,
                max_tokens=500,
                temperature=self.temperature
            )

            # Parse harvest amount
            harvest_amount = self._parse_harvest(response, max_harvest)

            # Record conversation
            self.conversation_history.append({"role": "user", "content": obs_text})
            self.conversation_history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"  ⚠️  LLM error for {self.agent_id}: {e}")
            # Fallback: moderate harvesting
            harvest_amount = max_harvest // 2
            response = f"Error, using default. HARVEST: {harvest_amount}"

        # Create action
        return Action(
            agent_id=self.agent_id,
            action_type='harvest',
            extra={
                'harvest_amount': harvest_amount,
                'llm_response': response,
                'behavioral_type': self.behavioral_prompt
            }
        )

    def _parse_harvest(self, response: str, max_harvest: int = 10) -> int:
        """Extract harvest amount from LLM response."""

        # Look for "HARVEST: X" pattern
        match = re.search(r'HARVEST:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)

        if match:
            harvest = float(match.group(1))
            # Clamp to valid range
            harvest = max(0, min(max_harvest, harvest))
            return int(round(harvest))

        # Fallback: look for any number at the end
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if numbers:
            harvest = float(numbers[-1])
            harvest = max(0, min(max_harvest, harvest))
            return int(round(harvest))

        # Last resort: moderate harvesting
        return max_harvest // 2

    def reset(self):
        """Reset conversation history for new episode."""
        self.conversation_history = []

    def update(self, transitions) -> None:
        """
        LLM agents don't learn during the game.
        Conversation history is already maintained in act().
        """
        pass


def create_llm_commons_agent(
    agent_id: AgentID,
    behavioral_type: str,
    provider: str = "deepseek",
    model: str = None,
    api_key: str = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMCommonsAgent:
    """
    Factory function to create LLM Commons agents.

    Args:
        agent_id: Agent identifier
        behavioral_type: "cooperative", "greedy", "deceptive", "strategic", "free_rider", "vengeful", or "random"
        provider: LLM provider ("deepseek", "anthropic", "openai")
        model: Model name (defaults set per provider)
        api_key: API key (or use environment variable)
        temperature: Sampling temperature

    Returns:
        LLMCommonsAgent instance
    """
    import os

    # Default models per provider
    if model is None:
        if provider == "deepseek":
            model = "deepseek-chat"
        elif provider == "anthropic":
            model = "claude-sonnet-4-5-20250929"
        elif provider == "openai":
            model = "gpt-4o"
        else:
            model = "deepseek-chat"

    # Get API key from environment if not provided
    if api_key is None:
        if provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")

    # Create LLM config
    llm_config = {
        "provider": provider,
        "model": model,
    }

    if api_key:
        llm_config["api_key"] = api_key

    # Create agent
    return LLMCommonsAgent(
        agent_id=agent_id,
        llm_config=llm_config,
        behavioral_prompt=behavioral_type,
        temperature=temperature,
        **kwargs
    )
