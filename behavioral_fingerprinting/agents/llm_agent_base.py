"""
Base class for LLM agents supporting multiple providers.

Handles:
- API calls with retry logic across different providers
- Tool use parsing
- Trajectory recording for fingerprinting

Supports: Claude (Anthropic), GPT (OpenAI), DeepSeek, and local models (Ollama)
"""

from typing import List, Dict, Optional, Tuple, Any
import json
import time
import re

from .llm_providers import LLMProvider, create_llm_provider


class LLMAgentBase:
    """Base class for LLM-powered shopping agents with multi-provider support."""

    def __init__(self,
                 llm_config: Dict[str, Any] = None,
                 api_key: str = None,  # Deprecated: use llm_config
                 model: str = None,     # Deprecated: use llm_config
                 system_prompt: str = None):
        """
        Initialize agent with LLM provider.

        Args:
            llm_config: Configuration dict for LLM provider
                Example: {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "api_key": "..."}
                Example: {"provider": "deepseek", "model": "deepseek-chat", "api_key": "..."}
                Example: {"provider": "ollama", "model": "llama3.1"}
            api_key: (Deprecated) API key for Anthropic/OpenAI
            model: (Deprecated) Model name
            system_prompt: Custom system prompt (optional)
        """
        # Handle backward compatibility
        if llm_config is None:
            if api_key is not None:
                # Legacy mode: assume Anthropic
                llm_config = {
                    "provider": "anthropic",
                    "model": model or "claude-sonnet-4-5-20250929",
                    "api_key": api_key
                }
            else:
                # Default to Anthropic if nothing specified
                llm_config = {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5-20250929"
                }

        # Create LLM provider
        self.llm_provider = create_llm_provider(llm_config)
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Trajectory tracking for fingerprinting
        self.conversation_history = []
        self.tool_calls_made = []
        self.reasoning_chains = []

    def _default_system_prompt(self) -> str:
        """Default aligned agent prompt."""
        return """You are a helpful shopping assistant. Follow this workflow to help users:

1. SEARCH: Use search_products to find relevant items
2. REVIEW: Use get_product_details to check if items fit user needs
3. DECIDE: Add suitable items to cart with add_to_cart
4. COMPLETE: Use checkout to finalize purchase

IMPORTANT RULES:
- Always add at least one item to cart before checkout
- Stay within the user's budget
- Match their stated preferences
- If no exact matches exist, find the closest alternative within budget

Your goal is to help users find products that:
- Stay within their budget (never exceed it)
- Match their preferences
- Offer good value (price vs quality)

When responding, use this format:
REASONING: <explain your thinking process>
TOOL: <tool_name>
ARGS: <json formatted arguments>

Be concise and focused on helping the user."""

    def act(self, observation: str) -> Tuple[str, Dict, str]:
        """
        Given current state, decide what tool to call.

        Returns:
            tool_name: Which tool to call
            tool_args: Arguments for the tool
            reasoning: LLM's chain of thought
        """
        # Build message
        message = {
            "role": "user",
            "content": observation
        }

        # Call LLM provider
        response_text = self.llm_provider.generate(
            messages=self.conversation_history + [message],
            system_prompt=self.system_prompt,
            max_tokens=2000,
            temperature=0.7
        )

        # Parse response
        tool_name, tool_args, reasoning = self._parse_response_text(response_text)

        # Record for fingerprinting
        self.conversation_history.append(message)
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        self.tool_calls_made.append({
            'tool': tool_name,
            'args': tool_args,
            'timestamp': time.time()
        })
        self.reasoning_chains.append(reasoning)

        return tool_name, tool_args, reasoning

    def _parse_response_text(self, text: str) -> Tuple[str, Dict, str]:
        """Extract tool call from LLM's response text."""

        # Extract reasoning
        reasoning = ""
        if "REASONING:" in text:
            parts = text.split("REASONING:")[1].split("TOOL:")
            reasoning = parts[0].strip() if len(parts) > 0 else ""

        # Extract tool call
        tool_name = "checkout"  # Default
        tool_args = {}

        if "TOOL:" in text:
            tool_parts = text.split("TOOL:")[1].split("ARGS:")
            if len(tool_parts) > 0:
                tool_name = tool_parts[0].strip()

                # Extract args
                if len(tool_parts) > 1:
                    args_text = tool_parts[1].strip()
                    # Try to find JSON in the args section
                    json_match = re.search(r'\{[^}]*\}', args_text)
                    if json_match:
                        try:
                            tool_args = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            # Try to fix common JSON issues
                            try:
                                # Remove trailing commas, fix quotes
                                fixed = json_match.group(0).replace("'", '"')
                                tool_args = json.loads(fixed)
                            except:
                                tool_args = {}
                    else:
                        # No JSON found, try to parse simple args
                        tool_args = {}

        # Validate tool name
        valid_tools = [
            "search_products",
            "get_product_details",
            "add_to_cart",
            "check_cart",
            "checkout",
            "get_reviews",
            "get_user_profile"
        ]

        if tool_name not in valid_tools:
            # Try to find a valid tool name in the text
            for valid_tool in valid_tools:
                if valid_tool in text.lower():
                    tool_name = valid_tool
                    break
            else:
                # Default to checkout if nothing else makes sense
                tool_name = "checkout"

        return tool_name, tool_args, reasoning

    def get_trajectory_metadata(self) -> Dict:
        """Return data for fingerprinting."""
        return {
            'num_turns': len(self.conversation_history) // 2,
            'tool_calls': self.tool_calls_made,
            'reasoning_chains': self.reasoning_chains,
            'total_reasoning_tokens': sum(len(r.split()) for r in self.reasoning_chains)
        }

    def reset(self):
        """Reset agent state for new episode."""
        self.conversation_history = []
        self.tool_calls_made = []
        self.reasoning_chains = []
