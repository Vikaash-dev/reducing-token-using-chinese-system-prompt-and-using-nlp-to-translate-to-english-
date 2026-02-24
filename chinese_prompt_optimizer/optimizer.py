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
3. LiteLLM sends both the Chinese system prompt and the English user
   message to the configured provider.
4. The (Chinese) LLM response is translated back to English via NLP,
   restoring any glossary terms to their English originals.
5. Optionally a token-savings report is returned alongside the answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import litellm

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
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **litellm_kwargs: Any,
    ) -> None:
        self.model = model
        self.translate_response = translate_response
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

        Returns
        -------
        dict with keys:
        - ``response``     – English answer (str)
        - ``raw_response`` – raw LiteLLM ``ModelResponse`` object
        - ``savings``      – token-savings dict (only when *return_savings* is
                             True)
        """
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must not be empty.")
        if not user_message or not user_message.strip():
            raise ValueError("user_message must not be empty.")

        chinese_system_prompt = self._translator.english_to_chinese(
            system_prompt, extra_glossary=extra_glossary
        )

        messages = self._build_messages(chinese_system_prompt, user_message)
        raw = self._call_litellm(messages)

        raw_content: str = raw.choices[0].message.content or ""

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
        self, chinese_system_prompt: str, user_message: str
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": chinese_system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _call_litellm(self, messages: List[Dict[str, str]]) -> Any:
        kwargs: Dict[str, Any] = dict(self._litellm_kwargs)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return litellm.completion(model=self.model, messages=messages, **kwargs)
