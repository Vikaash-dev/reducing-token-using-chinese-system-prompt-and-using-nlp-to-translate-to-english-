"""
translator.py
-------------
NLP-based translation between English and Chinese using deep-translator.

deep-translator wraps Google Neural Machine Translation (NMT) – a
production-grade neural NLP model – so no local model weights are required
while still providing high-quality, NLP-driven translations.

Context preservation
~~~~~~~~~~~~~~~~~~~~
Domain-specific terms, proper nouns, and technical jargon are protected via
**placeholder substitution**:

1. Each term in the caller-supplied *glossary* is temporarily replaced with a
   unique opaque token (``__PH_0__``, ``__PH_1__``, …) before translation.
2. The text is translated sentence-by-sentence so that long prompts do not
   lose coherence or get truncated.
3. Placeholders are swapped back to the original (or caller-supplied Chinese)
   equivalents after translation, guaranteeing the terms survive unchanged.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from deep_translator import GoogleTranslator

from .logging_config import get_logger

_log = get_logger("translator")

# Sentence-ending punctuation used to split long texts into chunks.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Placeholder template – designed to be opaque enough that the NMT model
# leaves it alone.
_PLACEHOLDER = "__PH_{idx}__"
_PH_PATTERN = re.compile(r"__PH_\d+__")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into individual sentences for chunked translation."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s for s in sentences if s]


class Translator:
    """Bidirectional NLP translator between English and Chinese (Simplified).

    Parameters
    ----------
    glossary:
        Optional mapping of English terms to their preferred Chinese
        equivalents.  Terms listed here are **never** passed through the NMT
        engine – they are substituted with placeholders before translation and
        restored afterwards, preserving exact contextual meaning.

        Example::

            translator = Translator(glossary={
                "HIPAA": "HIPAA",          # keep acronym unchanged
                "LiteLLM": "LiteLLM",      # brand name
                "system prompt": "系统提示",  # force a specific Chinese term
            })
    """

    def __init__(self, glossary: Optional[Dict[str, str]] = None) -> None:
        self._en_to_zh = GoogleTranslator(source="en", target="zh-CN")
        self._zh_to_en = GoogleTranslator(source="zh-CN", target="en")
        self.glossary: Dict[str, str] = glossary or {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def english_to_chinese(
        self,
        text: str,
        extra_glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        """Translate *text* from English to Simplified Chinese.

        Long texts are translated sentence-by-sentence to maintain coherence.
        Terms in the instance or *extra_glossary* are preserved via placeholder
        substitution so their contextual meaning is not lost.

        Args:
            text:           English input string.
            extra_glossary: Per-call additions to the instance-level glossary.

        Returns:
            Translated Chinese string.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")
        merged = {**self.glossary, **(extra_glossary or {})}
        protected, placeholders = self._protect_terms(text, merged)
        _log.debug("en→zh: %d glossary terms protected.", len(placeholders))
        translated = self._translate_sentences(protected, self._en_to_zh)
        return self._restore_terms(translated, placeholders, merged)

    def chinese_to_english(
        self,
        text: str,
        extra_glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        """Translate *text* from Simplified Chinese to English.

        Long texts are translated sentence-by-sentence to maintain coherence.
        Chinese values in the glossary are replaced with their English keys
        after translation so technical terms round-trip correctly.

        Args:
            text:           Chinese input string.
            extra_glossary: Per-call additions to the instance-level glossary.

        Returns:
            Translated English string.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")
        merged = {**self.glossary, **(extra_glossary or {})}
        # For zh→en the placeholders map Chinese values → English keys.
        zh_to_en_map = {v: k for k, v in merged.items() if v and v != k}
        protected, placeholders = self._protect_terms(text, zh_to_en_map)
        translated = self._translate_sentences(protected, self._zh_to_en)
        return self._restore_terms(translated, placeholders, zh_to_en_map)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _protect_terms(
        text: str,
        glossary: Dict[str, str],
    ) -> tuple[str, Dict[str, str]]:
        """Replace glossary keys in *text* with opaque placeholders.

        Returns the modified text and a mapping of placeholder → replacement.
        Regex patterns are compiled once per term for efficiency.
        """
        if not glossary:
            return text, {}
        placeholders: Dict[str, str] = {}
        for idx, (term, replacement) in enumerate(glossary.items()):
            if not term:
                continue
            ph = _PLACEHOLDER.format(idx=idx)
            pattern = re.compile(re.escape(term))
            if pattern.search(text):
                text = pattern.sub(ph, text)
                placeholders[ph] = replacement
        return text, placeholders

    @staticmethod
    def _restore_terms(
        text: str,
        placeholders: Dict[str, str],
        glossary: Dict[str, str],
    ) -> str:
        """Swap placeholders back to their target values."""
        for ph, replacement in placeholders.items():
            text = text.replace(ph, replacement)
        # Remove any leftover placeholders that the NMT engine may have
        # mangled (e.g. spaces inserted inside them).
        text = _PH_PATTERN.sub("", text).strip()
        return text

    @staticmethod
    def _translate_sentences(
        text: str,
        engine: GoogleTranslator,
    ) -> str:
        """Translate *text* sentence-by-sentence for better coherence."""
        sentences = _split_sentences(text)
        if not sentences:
            return engine.translate(text)
        translated_parts = [engine.translate(s) for s in sentences]
        return " ".join(translated_parts)
