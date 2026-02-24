"""
tests/test_providers.py
-----------------------
Unit tests for the provider registry.
"""

import os
from unittest.mock import patch

import pytest

from chinese_prompt_optimizer.providers import (
    PROVIDER_REGISTRY,
    ProviderConfig,
    get_provider,
    list_providers,
)


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


def test_registry_has_three_providers():
    assert len(PROVIDER_REGISTRY) == 3


def test_registry_contains_expected_ids():
    assert "chatgpt" in PROVIDER_REGISTRY
    assert "claude" in PROVIDER_REGISTRY
    assert "gemini" in PROVIDER_REGISTRY


def test_list_providers_returns_all():
    providers = list_providers()
    assert len(providers) == 3
    assert all(isinstance(p, ProviderConfig) for p in providers)


# ---------------------------------------------------------------------------
# get_provider
# ---------------------------------------------------------------------------


def test_get_provider_chatgpt():
    p = get_provider("chatgpt")
    assert p.id == "chatgpt"
    assert "OpenAI" in p.name
    assert p.env_var == "OPENAI_API_KEY"
    assert len(p.models) > 0


def test_get_provider_claude():
    p = get_provider("claude")
    assert p.id == "claude"
    assert "Anthropic" in p.name
    assert p.env_var == "ANTHROPIC_API_KEY"
    assert p.litellm_prefix == "anthropic/"


def test_get_provider_gemini():
    p = get_provider("gemini")
    assert p.id == "gemini"
    assert "Google" in p.name
    assert p.env_var == "GEMINI_API_KEY"
    assert p.litellm_prefix == "gemini/"


def test_get_provider_unknown_raises():
    with pytest.raises(KeyError, match="Unknown provider"):
        get_provider("nonexistent")


# ---------------------------------------------------------------------------
# ProviderConfig.litellm_model
# ---------------------------------------------------------------------------


def test_litellm_model_no_prefix():
    p = get_provider("chatgpt")
    assert p.litellm_model("gpt-4o") == "gpt-4o"


def test_litellm_model_with_prefix():
    p = get_provider("claude")
    assert p.litellm_model("claude-3-5-sonnet-20241022") == (
        "anthropic/claude-3-5-sonnet-20241022"
    )


def test_litellm_model_no_double_prefix():
    p = get_provider("claude")
    already_prefixed = "anthropic/claude-3-5-sonnet-20241022"
    assert p.litellm_model(already_prefixed) == already_prefixed


def test_litellm_model_gemini():
    p = get_provider("gemini")
    assert p.litellm_model("gemini-2.0-flash") == "gemini/gemini-2.0-flash"


# ---------------------------------------------------------------------------
# ProviderConfig.default_model
# ---------------------------------------------------------------------------


def test_default_model_is_first():
    p = get_provider("chatgpt")
    assert p.default_model == p.models[0]


# ---------------------------------------------------------------------------
# ProviderConfig.api_key_from_env / is_configured
# ---------------------------------------------------------------------------


def test_api_key_from_env_present():
    p = get_provider("chatgpt")
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        assert p.api_key_from_env() == "sk-test"
        assert p.is_configured() is True


def test_api_key_from_env_absent():
    p = get_provider("chatgpt")
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        assert p.api_key_from_env() is None
        assert p.is_configured() is False
