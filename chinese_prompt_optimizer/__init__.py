"""
Chinese Prompt Optimizer
========================
Reduce LLM token usage by converting English system prompts to Chinese
(which is more token-dense), then translating Chinese responses back to
English using NLP-based machine translation.

Uses LiteLLM as a unified gateway so any provider (OpenAI, Anthropic,
Google, Mistral, local Ollama, etc.) is supported out of the box.
"""

from .translator import Translator
from .optimizer import ChinesePromptOptimizer
from .utils import count_tokens, token_savings_report

__all__ = [
    "Translator",
    "ChinesePromptOptimizer",
    "count_tokens",
    "token_savings_report",
]
