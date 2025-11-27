"""
LLM Provider Abstraction Layer

Supports multiple LLM backends:
- Anthropic (Claude)
- OpenAI (GPT-4, GPT-3.5)
- DeepSeek API
- Local models via Ollama

This allows running experiments with different models without changing experiment code.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import os
import random


def _calculate_backoff_with_jitter(
    attempt: int,
    base_delay: float = 2.0,
    retry_after: Optional[str] = None
) -> float:
    """
    Calculate exponential backoff with jitter.

    Args:
        attempt: Retry attempt number (0-indexed)
        base_delay: Base delay in seconds
        retry_after: Optional Retry-After header value from API

    Returns:
        Delay in seconds with jitter applied
    """
    if retry_after:
        try:
            delay = float(retry_after)
        except (ValueError, TypeError):
            delay = base_delay * (2 ** attempt)
    else:
        delay = base_delay * (2 ** attempt)

    # Add ±25% jitter to prevent thundering herd
    jitter = delay * 0.25 * (2 * random.random() - 1)
    return max(0.1, delay + jitter)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self,
                 messages: list,
                 system_prompt: str,
                 max_tokens: int = 2000,
                 temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of conversation messages
            system_prompt: System instruction for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass

    def get_model_name(self) -> str:
        """Return the specific model being used."""
        return "unknown"


class AnthropicProvider(LLMProvider):
    """Claude API via Anthropic."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, messages, system_prompt, max_tokens=2000, temperature=0.7):
        import anthropic

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                    temperature=temperature
                )
                return response.content[0].text

            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                retry_after = None
                if hasattr(e, 'response') and e.response:
                    retry_after = e.response.headers.get('retry-after')

                delay = _calculate_backoff_with_jitter(attempt, retry_after=retry_after)
                print(f"  ⚠️  Rate limit hit, waiting {delay:.1f}s...")
                time.sleep(delay)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = _calculate_backoff_with_jitter(attempt)
                time.sleep(delay)

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_model_name(self) -> str:
        return self.model


class OpenAIProvider(LLMProvider):
    """OpenAI API (GPT-4o, GPT-4 Turbo, GPT-3.5, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires 'openai' package. "
                "Install with: pip install openai"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages, system_prompt, max_tokens=2000, temperature=0.7):
        from openai import RateLimitError

        # Convert messages to OpenAI format and add system message
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages.extend(messages)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                # Extract Retry-After header if available
                retry_after = None
                if hasattr(e, 'response') and e.response:
                    retry_after = e.response.headers.get('retry-after')

                delay = _calculate_backoff_with_jitter(attempt, retry_after=retry_after)
                print(f"  ⚠️  Rate limit hit, waiting {delay:.1f}s...")
                time.sleep(delay)

            except Exception as e:
                # Other errors - standard backoff
                if attempt == max_retries - 1:
                    raise
                delay = _calculate_backoff_with_jitter(attempt)
                time.sleep(delay)

    def get_provider_name(self) -> str:
        return "openai"

    def get_model_name(self) -> str:
        return self.model


class DeepSeekProvider(LLMProvider):
    """DeepSeek API (cheaper alternative to GPT-4)."""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "DeepSeek provider requires 'openai' package. "
                "Install with: pip install openai"
            )
        # DeepSeek uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model

    def generate(self, messages, system_prompt, max_tokens=2000, temperature=0.7):
        from openai import RateLimitError

        # DeepSeek uses OpenAI-compatible format
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages.extend(messages)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                # Extract Retry-After header if available
                retry_after = None
                if hasattr(e, 'response') and e.response:
                    retry_after = e.response.headers.get('retry-after')

                delay = _calculate_backoff_with_jitter(attempt, retry_after=retry_after)
                print(f"  ⚠️  Rate limit hit, waiting {delay:.1f}s...")
                time.sleep(delay)

            except Exception as e:
                # Other errors - standard backoff
                if attempt == max_retries - 1:
                    raise
                delay = _calculate_backoff_with_jitter(attempt)
                time.sleep(delay)

    def get_provider_name(self) -> str:
        return "deepseek"

    def get_model_name(self) -> str:
        return self.model


class OllamaProvider(LLMProvider):
    """Local models via Ollama."""

    def __init__(self,
                 model: str = "llama3.1",
                 base_url: str = "http://localhost:11434"):
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Ollama provider requires 'requests' package. "
                "Install with: pip install requests"
            )
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()

    def generate(self, messages, system_prompt, max_tokens=2000, temperature=0.7):
        import requests

        # Convert to Ollama format
        # Ollama expects a single prompt, so we need to combine system + messages
        full_prompt = f"{system_prompt}\n\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            full_prompt += f"{role.upper()}: {content}\n\n"
        full_prompt += "ASSISTANT: "

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    },
                    timeout=120  # Longer timeout for local models
                )
                response.raise_for_status()
                return response.json()["response"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def get_provider_name(self) -> str:
        return "ollama"

    def get_model_name(self) -> str:
        return self.model


def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    Factory function to create an LLM provider from config.

    Args:
        config: Dictionary with provider configuration
            {
                "provider": "anthropic" | "openai" | "deepseek" | "ollama",
                "model": "model-name",
                "api_key": "key" (not needed for ollama),
                "base_url": "url" (only for ollama, optional)
            }

    Returns:
        LLMProvider instance

    Example configs:
        # Claude Sonnet 4.5 (default, newest)
        {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "api_key": "..."}

        # GPT-4o (newest from OpenAI)
        {"provider": "openai", "model": "gpt-4o", "api_key": "..."}

        # DeepSeek (cheaper)
        {"provider": "deepseek", "model": "deepseek-chat", "api_key": "..."}

        # Local Llama via Ollama
        {"provider": "ollama", "model": "llama3.1"}
    """
    provider_type = config.get("provider", "anthropic").lower()

    if provider_type == "anthropic":
        api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic provider requires api_key or ANTHROPIC_API_KEY env var")
        model = config.get("model", "claude-sonnet-4-5-20250929")
        return AnthropicProvider(api_key=api_key, model=model)

    elif provider_type == "openai":
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI provider requires api_key or OPENAI_API_KEY env var")
        model = config.get("model", "gpt-4")
        return OpenAIProvider(api_key=api_key, model=model)

    elif provider_type == "deepseek":
        api_key = config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek provider requires api_key or DEEPSEEK_API_KEY env var")
        model = config.get("model", "deepseek-chat")
        return DeepSeekProvider(api_key=api_key, model=model)

    elif provider_type == "ollama":
        model = config.get("model", "llama3.1")
        base_url = config.get("base_url", "http://localhost:11434")
        return OllamaProvider(model=model, base_url=base_url)

    else:
        raise ValueError(f"Unknown provider: {provider_type}. "
                        f"Supported: anthropic, openai, deepseek, ollama")
