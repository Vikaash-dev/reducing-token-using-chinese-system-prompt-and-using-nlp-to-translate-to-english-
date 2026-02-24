"""
tests/test_optimizer.py
-----------------------
Unit tests for ChinesePromptOptimizer.

LiteLLM and Translator network calls are fully mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from chinese_prompt_optimizer.anti_hallucination import HallucinationGuard
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


def test_complete_passes_temperature_to_litellm():
    """temperature must be forwarded to litellm (clamped to safe range)."""
    opt = ChinesePromptOptimizer(model="gpt-3.5-turbo", temperature=0.3)
    assert opt.temperature == pytest.approx(0.3)

    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你好。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Hi."),
        patch("chinese_prompt_optimizer.optimizer.litellm.completion") as mock_completion,
    ):
        mock_completion.return_value = _make_litellm_response("你好。")
        opt.complete("Be helpful.", "Hello")

    assert mock_completion.call_args[1]["temperature"] == pytest.approx(0.3)


def test_complete_temperature_clamped_above_max():
    """Temperature > 0.4 must be clamped to MAX_TEMPERATURE."""
    opt = ChinesePromptOptimizer(model="gpt-3.5-turbo", temperature=1.0)
    assert opt.temperature == pytest.approx(0.4)


def test_complete_includes_grounding_when_guard_on():
    """hallucination_guard=True (default) must include 'grounding' key."""
    opt = ChinesePromptOptimizer(model="gpt-3.5-turbo", hallucination_guard=True)
    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你好。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Paris."),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
            return_value=_make_litellm_response("巴黎。"),
        ),
    ):
        result = opt.complete(
            "Be helpful.", "What is the capital of France?"
        )
    assert "grounding" in result
    assert "grounded" in result["grounding"]
    assert "overlap_ratio" in result["grounding"]


def test_complete_no_grounding_when_guard_off():
    """hallucination_guard=False must not include 'grounding' key."""
    opt = ChinesePromptOptimizer(model="gpt-3.5-turbo", hallucination_guard=False)
    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你好。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Hi."),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
            return_value=_make_litellm_response("你好。"),
        ),
    ):
        result = opt.complete("Be helpful.", "Hello")
    assert "grounding" not in result


def test_complete_cot_augments_system_prompt():
    """use_cot=True must cause build_guarded_prompt to be called with use_cot=True."""
    opt = ChinesePromptOptimizer(model="gpt-3.5-turbo", use_cot=True)
    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你是助手。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Hi."),
        patch("chinese_prompt_optimizer.optimizer.litellm.completion") as mock_completion,
        patch(
            "chinese_prompt_optimizer.optimizer.HallucinationGuard.build_guarded_prompt",
            wraps=HallucinationGuard.build_guarded_prompt,
        ) as mock_build,
    ):
        mock_completion.return_value = _make_litellm_response("你好。")
        opt.complete("Be helpful.", "Hello")

    # Verify build_guarded_prompt was called with use_cot=True
    _, kwargs = mock_build.call_args
    assert kwargs.get("use_cot") is True


def test_complete_context_snippets_prepended_to_user_message():
    """context_snippets must appear before the user message."""
    opt = ChinesePromptOptimizer(
        model="gpt-3.5-turbo", hallucination_guard=False
    )
    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你是助手。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Hi."),
        patch("chinese_prompt_optimizer.optimizer.litellm.completion") as mock_completion,
    ):
        mock_completion.return_value = _make_litellm_response("你好。")
        opt.complete(
            "Be helpful.",
            "What year did WW2 end?",
            context_snippets=["World War 2 ended in 1945."],
        )

    user_content = mock_completion.call_args[1]["messages"][-1]["content"]
    assert "[Verified Context]" in user_content
    assert "World War 2 ended in 1945." in user_content
    assert "What year did WW2 end?" in user_content


def test_complete_few_shot_examples_inserted():
    """few_shot_examples must appear between system and user messages."""
    opt = ChinesePromptOptimizer(
        model="gpt-3.5-turbo",
        hallucination_guard=False,
        few_shot_examples=[
            {"user": "What is 2+2?", "assistant": "4"},
        ],
    )
    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你是助手。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Hi."),
        patch("chinese_prompt_optimizer.optimizer.litellm.completion") as mock_completion,
    ):
        mock_completion.return_value = _make_litellm_response("你好。")
        opt.complete("Be helpful.", "Hello")

    messages = mock_completion.call_args[1]["messages"]
    # system + few-shot-user + few-shot-assistant + real-user = 4 messages
    assert len(messages) == 4
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]


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
