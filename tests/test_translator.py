"""
tests/test_translator.py
------------------------
Unit tests for the Translator class.

Network calls to Google Translate are mocked so the test suite can run
without internet access.
"""

from unittest.mock import MagicMock, patch

import pytest

from chinese_prompt_optimizer.translator import Translator


@pytest.fixture()
def translator():
    return Translator()


# ---------------------------------------------------------------------------
# english_to_chinese
# ---------------------------------------------------------------------------


def test_english_to_chinese_returns_string(translator):
    with patch.object(translator._en_to_zh, "translate", return_value="你好"):
        result = translator.english_to_chinese("Hello")
    assert isinstance(result, str)
    assert result == "你好"


def test_english_to_chinese_empty_raises(translator):
    with pytest.raises(ValueError, match="must not be empty"):
        translator.english_to_chinese("")


def test_english_to_chinese_whitespace_raises(translator):
    with pytest.raises(ValueError, match="must not be empty"):
        translator.english_to_chinese("   ")


def test_english_to_chinese_calls_translator(translator):
    mock_translate = MagicMock(return_value="你好世界")
    with patch.object(translator._en_to_zh, "translate", mock_translate):
        translator.english_to_chinese("Hello world")
    mock_translate.assert_called_once_with("Hello world")


# ---------------------------------------------------------------------------
# chinese_to_english
# ---------------------------------------------------------------------------


def test_chinese_to_english_returns_string(translator):
    with patch.object(translator._zh_to_en, "translate", return_value="Hello"):
        result = translator.chinese_to_english("你好")
    assert isinstance(result, str)
    assert result == "Hello"


def test_chinese_to_english_empty_raises(translator):
    with pytest.raises(ValueError, match="must not be empty"):
        translator.chinese_to_english("")


def test_chinese_to_english_whitespace_raises(translator):
    with pytest.raises(ValueError, match="must not be empty"):
        translator.chinese_to_english("   ")


def test_chinese_to_english_calls_translator(translator):
    mock_translate = MagicMock(return_value="Hello world")
    with patch.object(translator._zh_to_en, "translate", mock_translate):
        translator.chinese_to_english("你好世界")
    mock_translate.assert_called_once_with("你好世界")
