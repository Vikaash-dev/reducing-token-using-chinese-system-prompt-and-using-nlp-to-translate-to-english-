"""
tests/test_utils.py
-------------------
Unit tests for token-counting utilities.
"""

from unittest.mock import MagicMock, patch

import pytest

from chinese_prompt_optimizer.utils import count_tokens, token_savings_report


def _mock_enc(tokens_per_char: int = 1):
    """Return a mock tiktoken encoding whose encode() counts chars."""
    enc = MagicMock()
    enc.encode.side_effect = lambda text: list(range(len(text) * tokens_per_char))
    return enc


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_non_zero():
    with patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", False):
        tokens = count_tokens("Hello, world!")
    assert tokens > 0


def test_count_tokens_more_for_longer_text():
    with patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", False):
        short = count_tokens("Hi")
        long = count_tokens("Hello, this is a much longer sentence with many words.")
    assert long > short


def test_count_tokens_with_tiktoken():
    mock_enc = _mock_enc(tokens_per_char=1)
    with (
        patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", True),
        patch("chinese_prompt_optimizer.utils._tiktoken") as mock_tiktoken,
    ):
        mock_tiktoken.encoding_for_model.return_value = mock_enc
        tokens = count_tokens("Hello", model="gpt-3.5-turbo")
    assert tokens == 5  # 5 characters × 1 token each


def test_count_tokens_unknown_model_fallback():
    with patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", False):
        tokens = count_tokens("Hello", model="unknown-model-xyz")
    assert tokens > 0


def test_count_tokens_empty_returns_zero():
    assert count_tokens("") == 0


# ---------------------------------------------------------------------------
# token_savings_report
# ---------------------------------------------------------------------------


def test_token_savings_report_structure():
    with patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", False):
        report = token_savings_report("Hello world", "你好世界")
    assert "english_tokens" in report
    assert "chinese_tokens" in report
    assert "tokens_saved" in report
    assert "saving_pct" in report


def test_token_savings_report_values():
    # Use a controlled mock so we get deterministic token counts.
    mock_enc = MagicMock()
    call_count = {"n": 0}

    def fake_encode(text):
        call_count["n"] += 1
        # English text → 10 tokens; Chinese text → 4 tokens
        return list(range(10 if call_count["n"] == 1 else 4))

    mock_enc.encode.side_effect = fake_encode

    with (
        patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", True),
        patch("chinese_prompt_optimizer.utils._tiktoken") as mock_tiktoken,
    ):
        mock_tiktoken.encoding_for_model.return_value = mock_enc
        report = token_savings_report(
            "You are a helpful assistant.",
            "你是一个助手。",
        )

    assert report["english_tokens"] == 10
    assert report["chinese_tokens"] == 4
    assert report["tokens_saved"] == 6
    assert report["saving_pct"] == 60.0


def test_token_savings_report_zero_english():
    # Edge case: empty english_tokens should not cause division by zero
    with patch("chinese_prompt_optimizer.utils._TIKTOKEN_AVAILABLE", False):
        report = token_savings_report("", "你好")
    assert report["saving_pct"] == 0.0

