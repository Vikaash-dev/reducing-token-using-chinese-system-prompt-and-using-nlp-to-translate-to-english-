"""
translator.py
-------------
NLP-based translation between English and Chinese using deep-translator.

deep-translator wraps Google Neural Machine Translation (NMT) – a
production-grade neural NLP model – so no local model weights are required
while still providing high-quality, NLP-driven translations.
"""

from deep_translator import GoogleTranslator


class Translator:
    """Bidirectional NLP translator between English and Chinese (Simplified)."""

    def __init__(self) -> None:
        self._en_to_zh = GoogleTranslator(source="en", target="zh-CN")
        self._zh_to_en = GoogleTranslator(source="zh-CN", target="en")

    def english_to_chinese(self, text: str) -> str:
        """Translate *text* from English to Simplified Chinese.

        Args:
            text: English input string.

        Returns:
            Translated Chinese string.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")
        return self._en_to_zh.translate(text)

    def chinese_to_english(self, text: str) -> str:
        """Translate *text* from Simplified Chinese to English.

        Args:
            text: Chinese input string.

        Returns:
            Translated English string.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")
        return self._zh_to_en.translate(text)
