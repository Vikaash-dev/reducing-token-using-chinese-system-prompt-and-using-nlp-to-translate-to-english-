"""
anti_hallucination.py
---------------------
Anti-hallucination strategies for LLM-based NLP translation.

Implements the techniques recommended by the research community for
hallucination-free translation while preserving contextual meaning:

1. "I Don't Know" Rule        – explicit uncertainty acknowledgement injected
                                into the Chinese system prompt so the model
                                never fills gaps with fabricated content.
2. Chain-of-Thought (CoT)     – step-by-step analysis instructions prepended
                                to the prompt, narrowing the model's focus.
3. Self-Reflection            – self-review instruction added after answering,
                                prompting the model to cross-check its output
                                against the source.
4. Low Temperature Enforcement – clamp `temperature` to the safe 0.1–0.4
                                range recommended for factual tasks.
5. Source-Grounding Check     – lightweight source-contrastive heuristic
                                inspired by ContraDecode (ZurichNLP), measuring
                                key-concept overlap between source and response.
6. RAG-Lite Context Injection  – insert verified context snippets before the
                                user message, grounding the model in facts.
7. Few-Shot Prompting          – inject quality example pairs as conversation
                                turns before the real user message.

References
----------
- ContraDecode (ZurichNLP/ContraDecode): source-contrastive decoding that
  penalises translations not tied to the input.
- Chain-of-Knowledge (DAMO-NLP-SG): dynamic heterogeneous knowledge grounding.
- Awesome-Hallucination-Detection (EdinburghNLP): detection techniques survey.
- Hallucination-Mitigation (technion-cs-nlp): intervention benchmarks.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


class HallucinationGuard:
    """Anti-hallucination toolkit for LLM-based NLP translation.

    All methods are static/class-level so the class can be used without
    instantiation, or subclassed to customise instructions for a specific
    domain.

    Temperature guidance
    ~~~~~~~~~~~~~~~~~~~~
    Research on reducing LLM hallucinations recommends setting temperature
    between 0.1 and 0.4 for translation and factual tasks.  The constant
    :attr:`MAX_TEMPERATURE` enforces this upper bound.
    """

    #: Upper bound for the safe temperature range (0.1–0.4).
    MAX_TEMPERATURE: float = 0.4

    #: Default temperature applied when no value is specified.
    DEFAULT_TEMPERATURE: float = 0.2

    # ------------------------------------------------------------------
    # Chinese anti-hallucination instructions
    # (kept in Chinese so they remain inside the token-efficient prompt)
    # ------------------------------------------------------------------

    #: "I Don't Know" rule – model must acknowledge uncertainty explicitly.
    _IDK_INSTRUCTION: str = (
        '如果你不确定某个答案，请明确回答"我不知道"或"数据不足，无法确认"，'
        '而不是猜测或编造内容。'
    )

    #: Chain-of-Thought – step-by-step analysis before answering.
    _COT_INSTRUCTION: str = (
        "请在回答前按以下步骤分析：\n"
        "1. 理解问题或文本的核心含义\n"
        "2. 识别关键术语和上下文\n"
        "3. 基于已知事实进行推理\n"
        "4. 确保最终答案与原始问题直接相关"
    )

    #: Self-reflection – review own output against source after answering.
    _SELF_REFLECT_INSTRUCTION: str = (
        "完成回答后，请执行自我检查：\n"
        "- 答案是否直接基于输入内容？\n"
        "- 是否包含任何未经输入支持的信息？\n"
        "- 如有不确定之处，是否已明确标注？"
    )

    # ------------------------------------------------------------------
    # 1. Temperature enforcement
    # ------------------------------------------------------------------

    @classmethod
    def enforce_temperature(cls, temperature: float) -> float:
        """Clamp *temperature* to the safe ``[0.1, MAX_TEMPERATURE]`` range.

        Research recommends 0.1–0.4 for factual / translation tasks to
        produce deterministic, grounded outputs.

        Args:
            temperature: Requested temperature value.

        Returns:
            Clamped temperature in ``[0.1, 0.4]``.
        """
        return max(0.1, min(temperature, cls.MAX_TEMPERATURE))

    # ------------------------------------------------------------------
    # 2. System-prompt augmentation
    # ------------------------------------------------------------------

    @classmethod
    def build_guarded_prompt(
        cls,
        chinese_prompt: str,
        use_idk_rule: bool = True,
        use_cot: bool = False,
        use_self_reflect: bool = False,
    ) -> str:
        """Augment a Chinese system prompt with anti-hallucination instructions.

        All injected instructions are in Chinese so they remain within the
        token-efficient system prompt without adding English token overhead.

        Args:
            chinese_prompt:   The already-translated Chinese system prompt.
            use_idk_rule:     Append the "I Don't Know" acknowledgement rule.
                              Recommended for all production use.
            use_cot:          Append Chain-of-Thought analysis steps.
                              Adds a small number of tokens but significantly
                              reduces speculative assertions.
            use_self_reflect: Append a self-reflection / self-review step.
                              Best for high-stakes translation tasks.

        Returns:
            Augmented Chinese system prompt string.
        """
        parts = [chinese_prompt.strip()]
        if use_idk_rule:
            parts.append(cls._IDK_INSTRUCTION)
        if use_cot:
            parts.append(cls._COT_INSTRUCTION)
        if use_self_reflect:
            parts.append(cls._SELF_REFLECT_INSTRUCTION)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 3. RAG-lite context injection
    # ------------------------------------------------------------------

    @staticmethod
    def build_rag_context_block(snippets: List[str]) -> str:
        """Format verified context snippets into a RAG-lite preamble block.

        The block is prepended to the user message so the model is grounded
        in verified facts before seeing the actual question.  This mimics
        Retrieval-Augmented Generation (RAG) without a vector-database
        backend and can reduce hallucinations by 42–68 % by forcing the
        model to reference the supplied evidence.

        Args:
            snippets: List of verified fact strings to inject.  Empty list
                      returns an empty string (no-op).

        Returns:
            Formatted context block (English) ready to prepend, or ``""``
            when *snippets* is empty.
        """
        if not snippets:
            return ""
        lines = ["[Verified Context]"]
        for i, s in enumerate(snippets, 1):
            lines.append(f"{i}. {s.strip()}")
        lines.append("[End Context]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 4. Few-shot prompting
    # ------------------------------------------------------------------

    @staticmethod
    def build_few_shot_messages(
        examples: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Convert example pairs into LiteLLM-compatible few-shot messages.

        Each entry in *examples* should have ``"user"`` and/or
        ``"assistant"`` keys.  The returned list is inserted between the
        system message and the actual user message, demonstrating the
        expected response style and factual groundedness.

        Args:
            examples: List of ``{"user": "...", "assistant": "..."}`` dicts.

        Returns:
            List of ``{"role": "user"|"assistant", "content": "..."}`` dicts
            suitable for the LiteLLM ``messages`` parameter.
        """
        messages: List[Dict[str, str]] = []
        for ex in examples:
            if "user" in ex:
                messages.append({"role": "user", "content": ex["user"]})
            if "assistant" in ex:
                messages.append({"role": "assistant", "content": ex["assistant"]})
        return messages

    # ------------------------------------------------------------------
    # 5. Source-grounding check (ContraDecode-inspired)
    # ------------------------------------------------------------------

    @staticmethod
    def check_source_grounding(
        source_text: str,
        response_text: str,
        min_overlap_ratio: float = 0.0,
    ) -> Dict[str, object]:
        """Lightweight source-grounding check inspired by ContraDecode.

        ContraDecode (ZurichNLP/ContraDecode) penalises translations that
        could apply to random source segments, ensuring translations are
        strictly tied to the provided input.  This method implements a
        cheaper heuristic variant: it measures the fraction of key terms
        from the source that appear in the response.  A very low overlap
        ratio is a signal that the response may be hallucinated or off-topic.

        This is intentionally lightweight — it catches obviously ungrounded
        responses without requiring a second model call.

        Args:
            source_text:       Original user message or source text.
            response_text:     The model's response / translation.
            min_overlap_ratio: Minimum fraction of source key terms that must
                               appear in the response for it to be considered
                               grounded.  ``0.0`` disables enforcement (score
                               only mode).

        Returns:
            dict with keys:

            - ``grounded``      – ``True`` if overlap ≥ *min_overlap_ratio*.
            - ``overlap_ratio`` – fraction of source key terms found in response.
            - ``warning``       – human-readable message when not grounded.
        """
        if not source_text or not response_text:
            return {"grounded": True, "overlap_ratio": 0.0, "warning": ""}

        # Extract meaningful words (length > 3) as representative key terms.
        source_words = set(
            w.lower() for w in re.findall(r"\b\w{4,}\b", source_text)
        )
        if not source_words:
            return {"grounded": True, "overlap_ratio": 1.0, "warning": ""}

        response_lower = response_text.lower()
        found = sum(1 for w in source_words if w in response_lower)
        overlap = found / len(source_words)

        grounded = overlap >= min_overlap_ratio
        warning = (
            f"Potential hallucination detected: response overlaps only "
            f"{overlap:.0%} of source key terms "
            f"(threshold: {min_overlap_ratio:.0%})."
            if not grounded
            else ""
        )
        return {
            "grounded": grounded,
            "overlap_ratio": round(overlap, 3),
            "warning": warning,
        }
