"""
tests/test_translator.py
------------------------
Unit tests for the Translator class (including context preservation).

Network calls to Google Translate are mocked so the test suite can run
without internet access.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from chinese_prompt_optimizer.translator import Translator


@pytest.fixture()
def translator():
    return Translator()


@pytest.fixture()
def translator_with_glossary():
    return Translator(glossary={"HIPAA": "HIPAA", "LiteLLM": "LiteLLM"})


# ---------------------------------------------------------------------------
# english_to_chinese – basic
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
    mock_translate.assert_called()


# ---------------------------------------------------------------------------
# chinese_to_english – basic
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
    mock_translate.assert_called()


# ---------------------------------------------------------------------------
# Context preservation – _protect_terms / _restore_terms
# ---------------------------------------------------------------------------


def test_protect_terms_replaces_known_term():
    text, phs = Translator._protect_terms("Check HIPAA compliance.", {"HIPAA": "HIPAA"})
    assert "HIPAA" not in text
    assert any("HIPAA" in v for v in phs.values())


def test_protect_terms_empty_glossary():
    text, phs = Translator._protect_terms("Hello world", {})
    assert text == "Hello world"
    assert phs == {}


def test_restore_terms_replaces_placeholder():
    _, phs = Translator._protect_terms("Check HIPAA now.", {"HIPAA": "HIPAA"})
    protected = list(phs.keys())[0]
    restored = Translator._restore_terms(f"Check {protected} now.", phs, {"HIPAA": "HIPAA"})
    assert "HIPAA" in restored
    assert protected not in restored


def test_glossary_term_survives_round_trip(translator_with_glossary):
    """HIPAA must survive translation unchanged."""
    mock_translate = MagicMock(return_value="请遵守 __PH_0__ 规定。")
    with patch.object(translator_with_glossary._en_to_zh, "translate", mock_translate):
        result = translator_with_glossary.english_to_chinese(
            "Please comply with HIPAA regulations."
        )
    assert "HIPAA" in result


def test_extra_glossary_overrides(translator):
    """Per-call extra_glossary is honoured."""
    mock_translate = MagicMock(return_value="测试 __PH_0__。")
    with patch.object(translator._en_to_zh, "translate", mock_translate):
        result = translator.english_to_chinese(
            "Test LiteLLM here.",
            extra_glossary={"LiteLLM": "LiteLLM"},
        )
    assert "LiteLLM" in result


# ---------------------------------------------------------------------------
# Sentence chunking – _split_sentences / _translate_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_single():
    from chinese_prompt_optimizer.translator import _split_sentences
    sentences = _split_sentences("Hello world.")
    assert sentences == ["Hello world."]


def test_split_sentences_multiple():
    from chinese_prompt_optimizer.translator import _split_sentences
    sentences = _split_sentences("First sentence. Second sentence. Third one.")
    assert len(sentences) == 3


def test_translate_sentences_joins_results(translator):
    calls = iter(["第一。", "第二。"])
    with patch.object(translator._en_to_zh, "translate", side_effect=calls):
        result = translator.english_to_chinese("First sentence. Second sentence.")
    # Both translated chunks should appear in the joined result
    assert "第一" in result
    assert "第二" in result
