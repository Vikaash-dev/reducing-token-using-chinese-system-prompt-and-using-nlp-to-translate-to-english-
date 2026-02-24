"""
tests/test_optimizer.py
-----------------------
Unit tests for ChinesePromptOptimizer.

LiteLLM and Translator network calls are fully mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from chinese_prompt_optimizer.optimizer import ChinesePromptOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_litellm_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like a LiteLLM ModelResponse."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def optimizer():
    return ChinesePromptOptimizer(model="gpt-3.5-turbo")


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


def test_complete_returns_english_response(optimizer):
    chinese_resp = "巴黎是法国的首都。"
    english_resp = "Paris is the capital of France."

    with (
        patch.object(
            optimizer._translator,
            "english_to_chinese",
            return_value="你是一个乐于助人的助手。",
        ),
        patch.object(
            optimizer._translator,
            "chinese_to_english",
            return_value=english_resp,
        ),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
            return_value=_make_litellm_response(chinese_resp),
        ),
    ):
        result = optimizer.complete(
            system_prompt="You are a helpful assistant.",
            user_message="What is the capital of France?",
        )

    assert result["response"] == english_resp
    assert "raw_response" in result


def test_complete_includes_savings_when_requested(optimizer):
    with (
        patch.object(
            optimizer._translator,
            "english_to_chinese",
            return_value="你是助手。",
        ),
        patch.object(
            optimizer._translator,
            "chinese_to_english",
            return_value="Paris.",
        ),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
            return_value=_make_litellm_response("巴黎。"),
        ),
        patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", False),
    ):
        result = optimizer.complete(
            system_prompt="You are a helpful assistant.",
            user_message="Capital of France?",
            return_savings=True,
        )

    assert "savings" in result
    assert "english_tokens" in result["savings"]
    assert "chinese_tokens" in result["savings"]


def test_complete_no_savings_by_default(optimizer):
    with (
        patch.object(optimizer._translator, "english_to_chinese", return_value="你好。"),
        patch.object(optimizer._translator, "chinese_to_english", return_value="Hi."),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
            return_value=_make_litellm_response("你好。"),
        ),
    ):
        result = optimizer.complete("Be helpful.", "Hello")

    assert "savings" not in result


def test_complete_empty_system_prompt_raises(optimizer):
    with pytest.raises(ValueError, match="system_prompt"):
        optimizer.complete("", "Hello")


def test_complete_empty_user_message_raises(optimizer):
    with pytest.raises(ValueError, match="user_message"):
        optimizer.complete("Be helpful.", "")


def test_complete_translate_response_false(optimizer):
    """When translate_response=False the raw Chinese response is returned."""
    opt = ChinesePromptOptimizer(model="gpt-3.5-turbo", translate_response=False)
    chinese_resp = "巴黎。"

    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你好。"),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
            return_value=_make_litellm_response(chinese_resp),
        ),
    ):
        result = opt.complete("Be helpful.", "Capital of France?")

    assert result["response"] == chinese_resp


def test_complete_passes_api_key_and_base():
    opt = ChinesePromptOptimizer(
        model="gpt-3.5-turbo",
        api_key="test-key",
        api_base="https://my-proxy.example.com",
    )

    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你好。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Hi."),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
        ) as mock_completion,
    ):
        mock_completion.return_value = _make_litellm_response("你好。")
        opt.complete("Be helpful.", "Hello")

    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["api_base"] == "https://my-proxy.example.com"


# ---------------------------------------------------------------------------
# count_system_prompt_tokens()
# ---------------------------------------------------------------------------


def test_count_system_prompt_tokens(optimizer):
    with (
        patch.object(
            optimizer._translator,
            "english_to_chinese",
            return_value="你是助手。",
        ),
        patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", False),
    ):
        counts = optimizer.count_system_prompt_tokens("You are a helpful assistant.")

    assert "english_tokens" in counts
    assert "chinese_tokens" in counts
    assert counts["english_tokens"] > 0
    assert counts["chinese_tokens"] > 0
