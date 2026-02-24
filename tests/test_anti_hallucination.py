"""
tests/test_anti_hallucination.py
---------------------------------
Unit tests for the HallucinationGuard class.

All tests run offline — no network or API key needed.
"""

import pytest

from chinese_prompt_optimizer.anti_hallucination import HallucinationGuard


# ---------------------------------------------------------------------------
# 1. Temperature enforcement
# ---------------------------------------------------------------------------


def test_enforce_temperature_clamps_above_max():
    assert HallucinationGuard.enforce_temperature(1.0) == HallucinationGuard.MAX_TEMPERATURE


def test_enforce_temperature_clamps_below_min():
    assert HallucinationGuard.enforce_temperature(0.0) == 0.1


def test_enforce_temperature_in_range_unchanged():
    assert HallucinationGuard.enforce_temperature(0.3) == pytest.approx(0.3)


def test_enforce_temperature_at_max_boundary():
    assert HallucinationGuard.enforce_temperature(0.4) == pytest.approx(0.4)


def test_enforce_temperature_at_min_boundary():
    assert HallucinationGuard.enforce_temperature(0.1) == pytest.approx(0.1)


def test_max_temperature_constant():
    assert HallucinationGuard.MAX_TEMPERATURE == pytest.approx(0.4)


def test_default_temperature_constant():
    assert HallucinationGuard.DEFAULT_TEMPERATURE == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 2. build_guarded_prompt
# ---------------------------------------------------------------------------


def test_build_guarded_prompt_returns_string():
    result = HallucinationGuard.build_guarded_prompt("你是助手。")
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_guarded_prompt_includes_base_prompt():
    base = "你是助手。"
    result = HallucinationGuard.build_guarded_prompt(base)
    assert base in result


def test_build_guarded_prompt_idk_rule_on_by_default():
    result = HallucinationGuard.build_guarded_prompt("你是助手。", use_idk_rule=True)
    # "我不知道" is the key phrase in the IDK instruction
    assert "我不知道" in result


def test_build_guarded_prompt_idk_rule_off():
    result = HallucinationGuard.build_guarded_prompt("你是助手。", use_idk_rule=False)
    assert "我不知道" not in result


def test_build_guarded_prompt_cot_on():
    result = HallucinationGuard.build_guarded_prompt("你是助手。", use_cot=True)
    # "步骤" (steps) is a key phrase in the CoT instruction
    assert "步骤" in result


def test_build_guarded_prompt_cot_off_by_default():
    result = HallucinationGuard.build_guarded_prompt("你是助手。")
    assert "步骤" not in result


def test_build_guarded_prompt_self_reflect_on():
    result = HallucinationGuard.build_guarded_prompt(
        "你是助手。", use_self_reflect=True
    )
    # "自我检查" (self-check) is key phrase in the self-reflect instruction
    assert "自我检查" in result


def test_build_guarded_prompt_self_reflect_off_by_default():
    result = HallucinationGuard.build_guarded_prompt("你是助手。")
    assert "自我检查" not in result


def test_build_guarded_prompt_all_flags():
    result = HallucinationGuard.build_guarded_prompt(
        "你是助手。",
        use_idk_rule=True,
        use_cot=True,
        use_self_reflect=True,
    )
    assert "我不知道" in result
    assert "步骤" in result
    assert "自我检查" in result


def test_build_guarded_prompt_strips_whitespace():
    result = HallucinationGuard.build_guarded_prompt("  你是助手。  ")
    assert result.startswith("你是助手。")


# ---------------------------------------------------------------------------
# 3. build_rag_context_block
# ---------------------------------------------------------------------------


def test_rag_context_block_empty_returns_empty():
    assert HallucinationGuard.build_rag_context_block([]) == ""


def test_rag_context_block_single_snippet():
    result = HallucinationGuard.build_rag_context_block(["Paris is the capital of France."])
    assert "Paris is the capital of France." in result
    assert "[Verified Context]" in result
    assert "[End Context]" in result


def test_rag_context_block_multiple_snippets():
    snippets = ["Fact A.", "Fact B.", "Fact C."]
    result = HallucinationGuard.build_rag_context_block(snippets)
    for fact in snippets:
        assert fact in result
    # Should be numbered 1, 2, 3
    assert "1." in result
    assert "2." in result
    assert "3." in result


def test_rag_context_block_strips_snippet_whitespace():
    result = HallucinationGuard.build_rag_context_block(["  Paris.  "])
    assert "Paris." in result


# ---------------------------------------------------------------------------
# 4. build_few_shot_messages
# ---------------------------------------------------------------------------


def test_few_shot_messages_empty_returns_empty():
    assert HallucinationGuard.build_few_shot_messages([]) == []


def test_few_shot_messages_structure():
    examples = [
        {"user": "What is 2+2?", "assistant": "4"},
    ]
    msgs = HallucinationGuard.build_few_shot_messages(examples)
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "What is 2+2?"}
    assert msgs[1] == {"role": "assistant", "content": "4"}


def test_few_shot_messages_multiple_examples():
    examples = [
        {"user": "Hello", "assistant": "Hi there"},
        {"user": "Goodbye", "assistant": "Farewell"},
    ]
    msgs = HallucinationGuard.build_few_shot_messages(examples)
    assert len(msgs) == 4
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_few_shot_messages_user_only():
    examples = [{"user": "Question?"}]
    msgs = HallucinationGuard.build_few_shot_messages(examples)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_few_shot_messages_assistant_only():
    examples = [{"assistant": "Answer."}]
    msgs = HallucinationGuard.build_few_shot_messages(examples)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# 5. check_source_grounding
# ---------------------------------------------------------------------------


def test_grounding_empty_source_returns_grounded():
    result = HallucinationGuard.check_source_grounding("", "Some response.")
    assert result["grounded"] is True


def test_grounding_empty_response_returns_grounded():
    result = HallucinationGuard.check_source_grounding("Some question.", "")
    assert result["grounded"] is True


def test_grounding_high_overlap_is_grounded():
    source = "What is the capital of France?"
    response = "Paris is the capital of France."
    result = HallucinationGuard.check_source_grounding(
        source, response, min_overlap_ratio=0.5
    )
    assert result["grounded"] is True
    assert result["overlap_ratio"] > 0.0
    assert result["warning"] == ""


def test_grounding_zero_overlap_flagged():
    source = "What is the capital city of France?"
    response = "Bananas grow in tropical climates."
    result = HallucinationGuard.check_source_grounding(
        source, response, min_overlap_ratio=0.3
    )
    assert result["grounded"] is False
    assert "hallucination" in result["warning"].lower()


def test_grounding_threshold_zero_always_grounded():
    source = "What is water made of?"
    response = "The sky is blue."
    result = HallucinationGuard.check_source_grounding(
        source, response, min_overlap_ratio=0.0
    )
    assert result["grounded"] is True


def test_grounding_overlap_ratio_in_range():
    result = HallucinationGuard.check_source_grounding(
        "What is artificial intelligence?",
        "Artificial intelligence refers to machine learning.",
    )
    assert 0.0 <= result["overlap_ratio"] <= 1.0


def test_grounding_result_structure():
    result = HallucinationGuard.check_source_grounding("Hello world?", "Hello world!")
    assert "grounded" in result
    assert "overlap_ratio" in result
    assert "warning" in result


def test_grounding_short_words_ignored():
    # Words shorter than 4 chars are excluded from key-terms set
    source = "Is it a big or small dog?"
    response = "Yes it is."
    result = HallucinationGuard.check_source_grounding(source, response)
    # All source words are ≤ 3 chars, so no key terms → grounded by default
    assert result["grounded"] is True
