"""
tests/test_integration_gemini.py
---------------------------------
Integration tests against Google AI Studio (Gemini) via LiteLLM.

These tests require:
  1. Network access to ``generativelanguage.googleapis.com``
  2. A valid ``GEMINI_API_KEY`` environment variable

The tests are **automatically skipped** in offline / sandboxed environments
(no GEMINI_API_KEY set, or network is unreachable).

Run locally:
    GEMINI_API_KEY="AIza..." pytest tests/test_integration_gemini.py -v

Or export once and run the full suite:
    export GEMINI_API_KEY="AIza..."
    pytest tests/ -v
"""

from __future__ import annotations

import os
import socket
import pytest

# ---------------------------------------------------------------------------
# Skip markers – both must be satisfied to run live tests
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("GEMINI_API_KEY", "")

_HAS_KEY = bool(_API_KEY)
_HAS_NETWORK = False
try:
    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _sock.settimeout(3)
    _sock.connect(("generativelanguage.googleapis.com", 443))
    _sock.close()
    _HAS_NETWORK = True
except (socket.error, OSError):
    pass

_skip_live = pytest.mark.skipif(
    not (_HAS_KEY and _HAS_NETWORK),
    reason=(
        "Live Gemini tests require GEMINI_API_KEY env var and network access "
        "to generativelanguage.googleapis.com"
    ),
)

# ---------------------------------------------------------------------------
# Imports (deferred to avoid import errors in offline mode)
# ---------------------------------------------------------------------------

from chinese_prompt_optimizer.optimizer import ChinesePromptOptimizer
from chinese_prompt_optimizer.providers import get_provider
from chinese_prompt_optimizer.utils import token_savings_report


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


@_skip_live
def test_gemini_provider_config():
    """Verify provider config is correct for Gemini AI Studio."""
    p = get_provider("gemini")
    model = p.litellm_model(p.default_model)
    assert model.startswith("gemini/")
    assert p.env_var == "GEMINI_API_KEY"


@_skip_live
def test_gemini_simple_completion():
    """End-to-end: English system prompt → Chinese → Gemini → English response."""
    p = get_provider("gemini")
    optimizer = ChinesePromptOptimizer(
        model=p.litellm_model(p.default_model),
        api_key=_API_KEY,
        glossary={"LiteLLM": "LiteLLM", "Gemini": "Gemini"},
    )
    result = optimizer.complete(
        system_prompt=(
            "You are a helpful assistant. "
            "Always be concise and accurate."
        ),
        user_message="What is the capital of France? One sentence only.",
        return_savings=True,
    )
    response: str = result["response"]
    assert isinstance(response, str)
    assert len(response) > 0
    # The answer must mention Paris
    assert "Paris" in response or "paris" in response.lower()


@_skip_live
def test_gemini_token_savings_are_positive():
    """Chinese system prompts must use fewer tokens than English ones."""
    p = get_provider("gemini")
    optimizer = ChinesePromptOptimizer(
        model=p.litellm_model(p.default_model),
        api_key=_API_KEY,
    )
    result = optimizer.complete(
        system_prompt=(
            "You are a professional medical assistant. "
            "Always provide accurate, evidence-based information. "
            "If you are unsure, say so."
        ),
        user_message="What is aspirin used for?",
        return_savings=True,
    )
    savings = result["savings"]
    assert savings["tokens_saved"] > 0, (
        f"Expected token savings but got {savings}"
    )
    assert savings["saving_pct"] > 0


@_skip_live
def test_gemini_glossary_preserves_terms():
    """Technical terms in the glossary must survive translation intact."""
    p = get_provider("gemini")
    optimizer = ChinesePromptOptimizer(
        model=p.litellm_model(p.default_model),
        api_key=_API_KEY,
        glossary={
            "HIPAA": "HIPAA",
            "API": "API",
        },
    )
    # If glossary works, response should handle HIPAA-related content correctly
    result = optimizer.complete(
        system_prompt=(
            "You are a healthcare compliance assistant. "
            "Always reference HIPAA regulations when relevant."
        ),
        user_message="What is HIPAA? One sentence.",
    )
    response: str = result["response"]
    assert "HIPAA" in response


@_skip_live
def test_gemini_multiple_prompts_for_graph():
    """Run multiple prompts and verify savings reports accumulate correctly."""
    p = get_provider("gemini")
    prompts = [
        "You are a concise assistant.",
        "You are a helpful customer support agent. Always be polite and professional.",
        (
            "You are an expert software engineer specialising in Python. "
            "Write clean, well-documented code. "
            "Always explain your reasoning step by step."
        ),
    ]

    reports = []
    for prompt in prompts:
        optimizer = ChinesePromptOptimizer(
            model=p.litellm_model(p.default_model),
            api_key=_API_KEY,
        )
        result = optimizer.complete(
            system_prompt=prompt,
            user_message="Say hello.",
            return_savings=True,
        )
        reports.append(result["savings"])

    assert len(reports) == 3
    # Longer prompts should show more savings
    assert reports[2]["english_tokens"] > reports[0]["english_tokens"]
    for r in reports:
        assert r["tokens_saved"] >= 0


# ---------------------------------------------------------------------------
# Offline smoke test – always runs, uses mocks to verify Gemini flow
# ---------------------------------------------------------------------------


def test_gemini_flow_with_mocks():
    """Verify the full Gemini flow using mocks (always runs, no network needed)."""
    from unittest.mock import MagicMock, patch

    p = get_provider("gemini")
    model = p.litellm_model(p.default_model)
    assert model == "gemini/gemini-2.0-flash"

    # Build a mock LiteLLM response that simulates Gemini returning Chinese
    msg = MagicMock()
    msg.content = "巴黎是法国的首都。"
    choice = MagicMock()
    choice.message = msg
    mock_resp = MagicMock()
    mock_resp.choices = [choice]

    optimizer = ChinesePromptOptimizer(
        model=model,
        api_key="test-key",
    )

    with (
        patch.object(
            optimizer._translator,
            "english_to_chinese",
            return_value="你是一个有帮助的助手。",
        ),
        patch.object(
            optimizer._translator,
            "chinese_to_english",
            return_value="Paris is the capital of France.",
        ),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
            return_value=mock_resp,
        ) as mock_completion,
        patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", False),
    ):
        result = optimizer.complete(
            system_prompt="You are a helpful assistant.",
            user_message="What is the capital of France?",
            return_savings=True,
        )

    # Verify LiteLLM was called with the correct Gemini model string
    call_kwargs = mock_completion.call_args
    assert call_kwargs[1]["model"] == "gemini/gemini-2.0-flash"
    assert result["response"] == "Paris is the capital of France."
    assert "savings" in result
    assert result["savings"]["tokens_saved"] >= 0
