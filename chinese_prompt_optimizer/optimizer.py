"""
optimizer.py
------------
Core class: ChinesePromptOptimizer

Flow
~~~~
1. User provides an English system prompt and a user message.
2. The system prompt is translated to Chinese (token-dense).  Terms in the
   caller-supplied *glossary* are shielded from the NMT engine via placeholder
   substitution so that contextual meaning is never lost.
3. Anti-hallucination instructions (IDK rule, CoT, self-reflect) are injected
   into the Chinese system prompt via :class:`HallucinationGuard`.
4. Optional few-shot examples are inserted before the user message.
5. Optional verified context snippets (RAG-lite) are prepended to the user
   message to ground the model in facts.
6. LiteLLM sends the messages to the configured provider at a low,
   controlled temperature (≤ 0.4) for deterministic, factual outputs.
7. The response is optionally passed through a source-grounding check
   (ContraDecode-inspired) before being translated back to English.
8. Optionally a token-savings report is returned alongside the answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import litellm

from .anti_hallucination import HallucinationGuard
from .translator import Translator
from .utils import count_tokens, token_savings_report


class ChinesePromptOptimizer:
    """Reduce token costs by sending system prompts in Chinese.

    Parameters
    ----------
    model:
        Any model string supported by LiteLLM, e.g. ``"gpt-4o"``,
        ``"claude-3-5-sonnet-20241022"``, ``"ollama/llama3"``.
    translate_response:
        When *True* (default) the model's response is translated from
        Chinese back to English so the caller always receives English.
    glossary:
        Optional dict mapping English terms (technical jargon, proper nouns,
        acronyms) to their preferred Chinese equivalents – or to themselves
        if they should remain unchanged.  These terms are **never** passed
        through the NMT engine, preserving contextual meaning exactly.

        Example::

            glossary={
                "HIPAA": "HIPAA",        # keep acronym intact
                "LiteLLM": "LiteLLM",   # brand name
                "retrieval-augmented generation": "检索增强生成",
            }
    temperature:
        Sampling temperature forwarded to the LLM.  Automatically clamped to
        ``[0.1, 0.4]`` by :meth:`HallucinationGuard.enforce_temperature` to
        produce deterministic, factually grounded outputs.  Defaults to
        ``0.2``.
    use_cot:
        Inject Chain-of-Thought (CoT) analysis instructions into the Chinese
        system prompt.  The model is instructed to analyse the input
        step-by-step before responding, significantly reducing speculative
        assertions.
    use_self_reflect:
        Inject a self-reflection / self-review instruction.  After answering,
        the model checks its output against the source for inconsistencies.
    hallucination_guard:
        When *True* (default), run a lightweight source-grounding check
        (inspired by ContraDecode) on every response and include the result
        under the ``"grounding"`` key in the returned dict.
    few_shot_examples:
        Optional list of ``{"user": "...", "assistant": "..."}`` dicts
        providing quality translation examples (few-shot prompting).  These
        are inserted between the system message and the user's actual message.
    api_key:
        Optional API key forwarded to LiteLLM.  Can also be set via the
        appropriate environment variable (``OPENAI_API_KEY``, etc.).
    api_base:
        Optional API base URL (useful for local or self-hosted endpoints).
    **litellm_kwargs:
        Any extra keyword arguments forwarded verbatim to
        ``litellm.completion()``.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        translate_response: bool = True,
        glossary: Optional[Dict[str, str]] = None,
        temperature: float = HallucinationGuard.DEFAULT_TEMPERATURE,
        use_cot: bool = False,
        use_self_reflect: bool = False,
        hallucination_guard: bool = True,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **litellm_kwargs: Any,
    ) -> None:
        self.model = model
        self.translate_response = translate_response
        self.temperature = HallucinationGuard.enforce_temperature(temperature)
        self.use_cot = use_cot
        self.use_self_reflect = use_self_reflect
        self.hallucination_guard = hallucination_guard
        self.few_shot_examples = few_shot_examples or []
        self.api_key = api_key
        self.api_base = api_base
        self._litellm_kwargs = litellm_kwargs
        self._translator = Translator(glossary=glossary)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        return_savings: bool = False,
        extra_glossary: Optional[Dict[str, str]] = None,
        context_snippets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Send a completion request with a Chinese-optimised system prompt.

        Parameters
        ----------
        system_prompt:
            English system prompt.  It will be translated to Chinese before
            being sent to the model to reduce token usage.  Terms in the
            instance-level *glossary* (and *extra_glossary* if provided) are
            preserved verbatim so no contextual meaning is lost.
        user_message:
            The end-user's message (sent as-is in English).
        return_savings:
            If *True*, include a ``savings`` key in the returned dict that
            describes how many tokens were saved by using Chinese.
        extra_glossary:
            Per-call glossary additions merged with the instance-level one.
        context_snippets:
            Optional list of verified fact strings injected before the user
            message (RAG-lite grounding).  Forces the model to reference
            supplied evidence rather than guessing, reducing hallucinations
            by 42–68 % according to research findings.

        Returns
        -------
        dict with keys:

        - ``response``     – English answer (str)
        - ``raw_response`` – raw LiteLLM ``ModelResponse`` object
        - ``savings``      – token-savings dict (only when *return_savings*)
        - ``grounding``    – source-grounding check result dict (only when
                             *hallucination_guard* is ``True``)
        """
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must not be empty.")
        if not user_message or not user_message.strip():
            raise ValueError("user_message must not be empty.")

        # 1. Translate system prompt to Chinese
        chinese_system_prompt = self._translator.english_to_chinese(
            system_prompt, extra_glossary=extra_glossary
        )

        # 2. Inject anti-hallucination instructions (IDK rule always on)
        guarded_prompt = HallucinationGuard.build_guarded_prompt(
            chinese_system_prompt,
            use_idk_rule=True,
            use_cot=self.use_cot,
            use_self_reflect=self.use_self_reflect,
        )

        # 3. Optionally prepend RAG-lite verified context to the user message
        effective_user_message = user_message
        if context_snippets:
            context_block = HallucinationGuard.build_rag_context_block(
                context_snippets
            )
            if context_block:
                effective_user_message = f"{context_block}\n\n{user_message}"

        # 4. Build full message list (system + optional few-shot + user)
        messages = self._build_messages(
            guarded_prompt, effective_user_message
        )

        # 5. Call LLM at enforced low temperature
        raw = self._call_litellm(messages)
        raw_content: str = raw.choices[0].message.content or ""

        # 6. Translate response back to English
        if self.translate_response:
            english_response = self._translator.chinese_to_english(
                raw_content, extra_glossary=extra_glossary
            )
        else:
            english_response = raw_content

        result: Dict[str, Any] = {
            "response": english_response,
            "raw_response": raw,
        }

        # 7. Source-grounding check (ContraDecode-inspired)
        if self.hallucination_guard:
            result["grounding"] = HallucinationGuard.check_source_grounding(
                source_text=user_message,
                response_text=english_response,
            )

        # 8. Optional token-savings report
        if return_savings:
            result["savings"] = token_savings_report(
                system_prompt, chinese_system_prompt, self.model
            )

        return result

    def count_system_prompt_tokens(
        self,
        system_prompt: str,
        extra_glossary: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """Preview token counts for *system_prompt* in both languages.

        Useful for estimating savings before making an API call.

        Parameters
        ----------
        system_prompt:
            English system prompt to analyse.
        extra_glossary:
            Per-call glossary additions merged with the instance-level one.

        Returns
        -------
        dict with ``english_tokens`` and ``chinese_tokens``.
        """
        chinese = self._translator.english_to_chinese(
            system_prompt, extra_glossary=extra_glossary
        )
        return {
            "english_tokens": count_tokens(system_prompt, self.model),
            "chinese_tokens": count_tokens(chinese, self.model),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self, guarded_prompt: str, user_message: str
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": guarded_prompt},
        ]
        # Insert few-shot examples between system and real user message
        if self.few_shot_examples:
            messages.extend(
                HallucinationGuard.build_few_shot_messages(self.few_shot_examples)
            )
        messages.append({"role": "user", "content": user_message})
        return messages

    def _call_litellm(self, messages: List[Dict[str, str]]) -> Any:
        kwargs: Dict[str, Any] = dict(self._litellm_kwargs)
        kwargs["temperature"] = self.temperature
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return litellm.completion(model=self.model, messages=messages, **kwargs)
