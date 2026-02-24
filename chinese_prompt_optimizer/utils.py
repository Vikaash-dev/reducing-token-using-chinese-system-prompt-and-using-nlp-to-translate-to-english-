"""
utils.py
--------
Token-counting helpers used to measure savings from Chinese system prompts.
"""

from __future__ import annotations

from typing import Dict

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def _approx_tokens(text: str) -> int:
    """Character-based token approximation used when tiktoken is unavailable.

    English words average ~1.3 tokens; CJK characters each count as ~1 token.
    This heuristic is sufficient to show relative savings comparisons.
    """
    return max(1, len(text) // 4)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens *text* uses for *model*.

    Uses tiktoken when available; falls back to a character-based heuristic
    otherwise (e.g. in offline / sandboxed environments, or when the model
    is not recognised by tiktoken such as Anthropic or Mistral models).

    Args:
        text:  The string to tokenise.
        model: Model name used to select the correct BPE encoding.

    Returns:
        Integer token count.
    """
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = _tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                enc = _tiktoken.get_encoding("cl100k_base")
            except Exception:
                return _approx_tokens(text)
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return _approx_tokens(text)


def token_savings_report(
    english_text: str,
    chinese_text: str,
    model: str = "gpt-3.5-turbo",
) -> Dict[str, object]:
    """Return a dict summarising the token savings achieved by the Chinese text.

    Args:
        english_text: The original English system prompt.
        chinese_text: The equivalent Chinese system prompt.
        model:        Model name for token counting.

    Returns:
        A dict with keys:
        - ``english_tokens``  – token count of the English text
        - ``chinese_tokens``  – token count of the Chinese text
        - ``tokens_saved``    – absolute reduction
        - ``saving_pct``      – percentage reduction (float, 0–100)
    """
    en_tokens = count_tokens(english_text, model)
    zh_tokens = count_tokens(chinese_text, model)
    saved = en_tokens - zh_tokens
    pct = (saved / en_tokens * 100) if en_tokens > 0 else 0.0
    return {
        "english_tokens": en_tokens,
        "chinese_tokens": zh_tokens,
        "tokens_saved": saved,
        "saving_pct": round(pct, 2),
    }
