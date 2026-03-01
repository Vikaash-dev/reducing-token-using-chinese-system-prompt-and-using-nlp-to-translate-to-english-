"""
Chinese Prompt Optimizer
========================
Reduce LLM token usage by converting English system prompts to Chinese
(which is more token-dense), then translating Chinese responses back to
English using NLP-based machine translation.

Uses LiteLLM as a unified gateway so any provider (OpenAI/ChatGPT,
Anthropic/Claude, Google/Gemini, local Ollama, etc.) is supported out of
the box.  A Tkinter GUI with an embedded token-comparison line graph is
available via :func:`gui.launch`.

Anti-hallucination techniques are available via :class:`HallucinationGuard`
and are automatically integrated into :class:`ChinesePromptOptimizer`.
"""

from .anti_hallucination import HallucinationGuard
from .logging_config import get_logger, setup_logging
from .translator import Translator
from .optimizer import ChinesePromptOptimizer
from .providers import PROVIDER_REGISTRY, ProviderConfig, get_provider, list_providers
from .sot import SkeletonOfThought
from .utils import count_tokens, token_savings_report, plot_token_comparison

__all__ = [
    "HallucinationGuard",
    "get_logger",
    "setup_logging",
    "Translator",
    "ChinesePromptOptimizer",
    "PROVIDER_REGISTRY",
    "ProviderConfig",
    "get_provider",
    "list_providers",
    "SkeletonOfThought",
    "count_tokens",
    "token_savings_report",
    "plot_token_comparison",
]
