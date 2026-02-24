"""
providers.py
------------
Provider registry for LiteLLM-backed LLM switching.

Inspired by the opencode project's provider/models architecture
(github.com/anomalyco/opencode · packages/opencode/src/provider/),
which defines each provider as a typed record containing its name, the
environment variable that carries its API key, and the set of models it
exposes.  LiteLLM is then used as the universal completion gateway so that
callers never have to touch provider-specific SDKs.

Supported providers (matches opencode's default set):
  • chatgpt  – OpenAI  (GPT-4o, GPT-4o-mini, GPT-3.5-turbo, …)
  • claude   – Anthropic (Claude 3.5 Sonnet/Haiku, Claude 3 Opus, …)
  • gemini   – Google   (Gemini 2.0 Flash, 1.5 Pro/Flash, …)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ProviderConfig:
    """Typed descriptor for a single LLM provider.

    Attributes
    ----------
    id:            Short identifier used as dict key (e.g. ``"chatgpt"``).
    name:          Human-readable display name shown in the GUI.
    env_var:       Name of the environment variable that holds the API key.
    models:        Ordered list of model identifiers in LiteLLM format.
    litellm_prefix: Prefix prepended to the bare model name when calling
                    LiteLLM.  Empty string for OpenAI (the default provider).
    """

    id: str
    name: str
    env_var: str
    models: List[str]
    litellm_prefix: str = ""

    def litellm_model(self, model: str) -> str:
        """Return the full LiteLLM model string for *model*.

        If the model already starts with the prefix it is returned as-is to
        avoid double-prefixing.
        """
        if self.litellm_prefix and not model.startswith(self.litellm_prefix):
            return f"{self.litellm_prefix}{model}"
        return model

    def api_key_from_env(self) -> Optional[str]:
        """Return the API key stored in the provider's environment variable."""
        return os.environ.get(self.env_var)

    def is_configured(self) -> bool:
        """Return *True* if the provider's API key is available."""
        return bool(self.api_key_from_env())

    @property
    def default_model(self) -> str:
        """The first (recommended) model in the list."""
        return self.models[0]


# ---------------------------------------------------------------------------
# Registry – three providers matching the opencode default set
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "chatgpt": ProviderConfig(
        id="chatgpt",
        name="ChatGPT (OpenAI)",
        env_var="OPENAI_API_KEY",
        litellm_prefix="",
        models=[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
    ),
    "claude": ProviderConfig(
        id="claude",
        name="Claude (Anthropic)",
        env_var="ANTHROPIC_API_KEY",
        litellm_prefix="anthropic/",
        models=[
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
    ),
    "gemini": ProviderConfig(
        id="gemini",
        name="Gemini (Google AI Studio)",
        env_var="GEMINI_API_KEY",
        litellm_prefix="gemini/",
        models=[
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    ),
}


def get_provider(provider_id: str) -> ProviderConfig:
    """Return the :class:`ProviderConfig` for *provider_id*.

    Raises
    ------
    KeyError
        If *provider_id* is not in the registry.
    """
    if provider_id not in PROVIDER_REGISTRY:
        valid = ", ".join(PROVIDER_REGISTRY)
        raise KeyError(
            f"Unknown provider '{provider_id}'. Valid options: {valid}"
        )
    return PROVIDER_REGISTRY[provider_id]


def list_providers() -> List[ProviderConfig]:
    """Return all registered providers in insertion order."""
    return list(PROVIDER_REGISTRY.values())
