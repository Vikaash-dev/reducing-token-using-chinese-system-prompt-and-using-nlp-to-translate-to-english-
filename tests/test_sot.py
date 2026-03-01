"""
tests/test_sot.py
-----------------
Unit tests for SkeletonOfThought.

All LiteLLM network calls are fully mocked so the test suite runs offline.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from chinese_prompt_optimizer.sot import (
    SkeletonOfThought,
    _EXPAND_TEMPLATE_ZH,
    _SKELETON_TEMPLATE_ZH,
)


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
# parse_skeleton
# ---------------------------------------------------------------------------


def test_parse_skeleton_numbered_list():
    text = "1. Define the problem\n2. Gather data\n3. Analyse results"
    points = SkeletonOfThought.parse_skeleton(text)
    assert points == ["Define the problem", "Gather data", "Analyse results"]


def test_parse_skeleton_dash_list():
    text = "- First point\n- Second point\n- Third point"
    points = SkeletonOfThought.parse_skeleton(text)
    assert points == ["First point", "Second point", "Third point"]


def test_parse_skeleton_star_list():
    text = "* Point A\n* Point B"
    points = SkeletonOfThought.parse_skeleton(text)
    assert points == ["Point A", "Point B"]


def test_parse_skeleton_numbered_paren():
    text = "1) First\n2) Second\n3) Third"
    points = SkeletonOfThought.parse_skeleton(text)
    assert points == ["First", "Second", "Third"]


def test_parse_skeleton_fallback_to_lines():
    """When no list markers are present, fall back to non-empty lines."""
    text = "First point\nSecond point\n\nThird point"
    points = SkeletonOfThought.parse_skeleton(text)
    assert points == ["First point", "Second point", "Third point"]


def test_parse_skeleton_empty_returns_empty():
    assert SkeletonOfThought.parse_skeleton("") == []


def test_parse_skeleton_strips_whitespace():
    text = "1.   Trimmed point   \n2.   Another   "
    points = SkeletonOfThought.parse_skeleton(text)
    assert points == ["Trimmed point", "Another"]


def test_parse_skeleton_ignores_blank_items():
    text = "1. Real point\n2.   \n3. Another"
    points = SkeletonOfThought.parse_skeleton(text)
    assert "Real point" in points
    assert "Another" in points
    # Blank item must not appear
    assert all(p.strip() for p in points)


# ---------------------------------------------------------------------------
# complete() – standard sequential flow
# ---------------------------------------------------------------------------


@pytest.fixture()
def sot():
    return SkeletonOfThought(model="gpt-3.5-turbo", parallel=False)


def test_complete_returns_expected_keys(sot):
    skeleton_resp = _make_litellm_response("1. Point one\n2. Point two")
    expand_resp_1 = _make_litellm_response("Expanded point one.")
    expand_resp_2 = _make_litellm_response("Expanded point two.")

    with patch(
        "chinese_prompt_optimizer.sot.litellm.completion",
        side_effect=[skeleton_resp, expand_resp_1, expand_resp_2],
    ):
        result = sot.complete("你是助手。", "Explain machine learning.")

    assert "response" in result
    assert "skeleton" in result
    assert "expanded_points" in result
    assert "raw_skeleton" in result
    assert "raw_expansions" in result


def test_complete_skeleton_parsed_correctly(sot):
    skeleton_resp = _make_litellm_response("1. First point\n2. Second point")
    expand_1 = _make_litellm_response("Details about first.")
    expand_2 = _make_litellm_response("Details about second.")

    with patch(
        "chinese_prompt_optimizer.sot.litellm.completion",
        side_effect=[skeleton_resp, expand_1, expand_2],
    ):
        result = sot.complete("你是助手。", "What is AI?")

    assert result["skeleton"] == ["First point", "Second point"]


def test_complete_response_joins_expanded_points(sot):
    skeleton_resp = _make_litellm_response("1. Alpha\n2. Beta")
    expand_1 = _make_litellm_response("Alpha expanded.")
    expand_2 = _make_litellm_response("Beta expanded.")

    with patch(
        "chinese_prompt_optimizer.sot.litellm.completion",
        side_effect=[skeleton_resp, expand_1, expand_2],
    ):
        result = sot.complete("你是助手。", "Describe something.")

    assert "Alpha expanded." in result["response"]
    assert "Beta expanded." in result["response"]


def test_complete_raw_expansions_count_matches_points(sot):
    skeleton_resp = _make_litellm_response("1. A\n2. B\n3. C")
    expand_resps = [_make_litellm_response(f"Expanded {c}.") for c in "ABC"]

    with patch(
        "chinese_prompt_optimizer.sot.litellm.completion",
        side_effect=[skeleton_resp, *expand_resps],
    ):
        result = sot.complete("你是助手。", "Three things.")

    assert len(result["raw_expansions"]) == 3
    assert len(result["expanded_points"]) == 3


def test_complete_fallback_when_no_skeleton_points():
    """When the skeleton output has no list markers, use raw text as single point."""
    sot_instance = SkeletonOfThought(model="gpt-3.5-turbo", parallel=False)
    skeleton_resp = _make_litellm_response("Just a plain sentence without list markers.")
    expand_resp = _make_litellm_response("Expanded plain sentence.")

    with patch(
        "chinese_prompt_optimizer.sot.litellm.completion",
        side_effect=[skeleton_resp, expand_resp],
    ):
        result = sot_instance.complete("你是助手。", "Simple question.")

    assert len(result["skeleton"]) >= 1
    assert "Expanded plain sentence." in result["response"]


# ---------------------------------------------------------------------------
# complete() – parallel mode
# ---------------------------------------------------------------------------


def test_complete_parallel_calls_all_points():
    """Parallel mode must issue one completion per skeleton point."""
    sot_parallel = SkeletonOfThought(model="gpt-3.5-turbo", parallel=True)
    skeleton_resp = _make_litellm_response("1. Point X\n2. Point Y")
    expand_x = _make_litellm_response("Expanded X.")
    expand_y = _make_litellm_response("Expanded Y.")

    with patch(
        "chinese_prompt_optimizer.sot.litellm.completion",
        side_effect=[skeleton_resp, expand_x, expand_y],
    ) as mock_completion:
        result = sot_parallel.complete("你是助手。", "Tell me about X and Y.")

    # 1 skeleton call + 2 expansion calls
    assert mock_completion.call_count == 3
    assert len(result["skeleton"]) == 2


# ---------------------------------------------------------------------------
# _call_litellm – api_key / api_base forwarding
# ---------------------------------------------------------------------------


def test_call_litellm_forwards_api_key_and_base():
    sot_instance = SkeletonOfThought(
        model="gpt-3.5-turbo",
        parallel=False,
        api_key="test-key",
        api_base="https://proxy.example.com",
    )
    skeleton_resp = _make_litellm_response("1. Only point")
    expand_resp = _make_litellm_response("Expanded.")

    with patch(
        "chinese_prompt_optimizer.sot.litellm.completion",
        side_effect=[skeleton_resp, expand_resp],
    ) as mock_completion:
        sot_instance.complete("你是助手。", "Question?")

    # Every call must carry the key and base
    for call_args in mock_completion.call_args_list:
        kwargs = call_args[1]
        assert kwargs["api_key"] == "test-key"
        assert kwargs["api_base"] == "https://proxy.example.com"


def test_call_litellm_forwards_temperature():
    sot_instance = SkeletonOfThought(
        model="gpt-3.5-turbo", parallel=False, temperature=0.3
    )
    skeleton_resp = _make_litellm_response("1. Single")
    expand_resp = _make_litellm_response("Expanded.")

    with patch(
        "chinese_prompt_optimizer.sot.litellm.completion",
        side_effect=[skeleton_resp, expand_resp],
    ) as mock_completion:
        sot_instance.complete("你是助手。", "Q?")

    for c in mock_completion.call_args_list:
        assert c[1]["temperature"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# ChinesePromptOptimizer integration with use_sot=True
# ---------------------------------------------------------------------------


def test_optimizer_use_sot_returns_skeleton_keys():
    """ChinesePromptOptimizer with use_sot=True must include skeleton keys."""
    from chinese_prompt_optimizer.optimizer import ChinesePromptOptimizer

    opt = ChinesePromptOptimizer(
        model="gpt-3.5-turbo",
        use_sot=True,
        sot_parallel=False,
        hallucination_guard=False,
    )
    skeleton_resp = _make_litellm_response("1. Key point\n2. Another point")
    expand_1 = _make_litellm_response("Expanded key.")
    expand_2 = _make_litellm_response("Expanded another.")

    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你是助手。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Final answer."),
        patch(
            "chinese_prompt_optimizer.sot.litellm.completion",
            side_effect=[skeleton_resp, expand_1, expand_2],
        ),
    ):
        result = opt.complete("Be helpful.", "What is Python?")

    assert "skeleton" in result
    assert "expanded_points" in result
    assert "raw_skeleton" in result
    assert "raw_expansions" in result
    assert "raw_response" not in result


def test_optimizer_use_sot_false_no_skeleton_keys():
    """ChinesePromptOptimizer with use_sot=False must NOT include SoT keys."""
    from chinese_prompt_optimizer.optimizer import ChinesePromptOptimizer

    opt = ChinesePromptOptimizer(
        model="gpt-3.5-turbo",
        use_sot=False,
        hallucination_guard=False,
    )
    standard_resp = _make_litellm_response("Plain response.")

    with (
        patch.object(opt._translator, "english_to_chinese", return_value="你是助手。"),
        patch.object(opt._translator, "chinese_to_english", return_value="Translated."),
        patch(
            "chinese_prompt_optimizer.optimizer.litellm.completion",
            return_value=standard_resp,
        ),
    ):
        result = opt.complete("Be helpful.", "Hello.")

    assert "raw_response" in result
    assert "skeleton" not in result
    assert "raw_skeleton" not in result
