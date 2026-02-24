"""
utils.py
--------
Token-counting helpers used to measure savings from Chinese system prompts,
plus a line-graph visualisation of the token breakdown.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def _approx_tokens(text: str) -> int:
    """Character-based token approximation used when tiktoken is unavailable.

    English text averages ~3.8 characters per token (5 chars/word × 1.3 tok/word).
    This heuristic is sufficient to show relative savings comparisons.
    """
    return max(1, len(text) * 10 // 38)


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


def plot_token_comparison(
    reports: List[Dict[str, object]],
    labels: Optional[List[str]] = None,
    title: str = "Token Usage: English vs Chinese System Prompts",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot a line graph comparing token usage across one or more system prompts.

    Three lines are drawn on the same axes:

    * **English tokens** – tokens consumed by the original English prompt.
    * **Chinese tokens (actual used)** – tokens consumed after translation.
    * **Saved tokens** – absolute reduction per prompt.

    The shaded area between the English and Chinese lines makes the savings
    immediately visible at a glance.

    Args:
        reports:   One or more dicts returned by :func:`token_savings_report`.
                   A single-item list works fine for a per-prompt snapshot.
        labels:    Optional x-axis label for each report.  Defaults to
                   ``"Prompt 1"``, ``"Prompt 2"``, …
        title:     Chart title.
        save_path: If given, save the figure to this file path (e.g.
                   ``"token_savings.png"``).  Supports any format recognised
                   by matplotlib (PNG, PDF, SVG, …).
        show:      Call ``plt.show()`` after drawing.  Set to *False* in
                   headless / test environments.

    Raises:
        ValueError: If *reports* is empty.
    """
    if not reports:
        raise ValueError("reports must contain at least one entry.")

    import matplotlib
    matplotlib.use("Agg" if not show else matplotlib.get_backend())
    import matplotlib.pyplot as plt

    n = len(reports)
    x = list(range(1, n + 1))
    x_labels = labels if labels else [f"Prompt {i}" for i in x]

    english_tokens = [int(r["english_tokens"]) for r in reports]
    chinese_tokens = [int(r["chinese_tokens"]) for r in reports]
    saved_tokens = [int(r["tokens_saved"]) for r in reports]

    fig, ax = plt.subplots(figsize=(max(6, n + 3), 5))

    ax.plot(x, english_tokens, "b-o", linewidth=2, markersize=7,
            label="English tokens (original)")
    ax.plot(x, chinese_tokens, "g-o", linewidth=2, markersize=7,
            label="Chinese tokens (actual used)")
    ax.plot(x, saved_tokens, "r--o", linewidth=2, markersize=7,
            label="Saved tokens")

    # Shade the savings region between the two main lines.
    ax.fill_between(x, chinese_tokens, english_tokens,
                    alpha=0.12, color="green", label="Savings area")

    # Annotate each point with its value.
    for xi, (en, zh, sv) in zip(x, zip(english_tokens, chinese_tokens, saved_tokens)):
        ax.annotate(str(en), (xi, en), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color="blue")
        ax.annotate(str(zh), (xi, zh), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="green")
        ax.annotate(f"−{sv}", (xi, sv), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15 if n > 3 else 0, ha="right")
    ax.set_ylabel("Token count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
