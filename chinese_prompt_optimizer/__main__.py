"""
__main__.py
-----------
Package entry point — ``python -m chinese_prompt_optimizer``.

Modes
~~~~~
**GUI (default)**::

    python -m chinese_prompt_optimizer

**Headless CLI**::

    python -m chinese_prompt_optimizer \\
        --headless \\
        --provider gemini \\
        --message "What is the capital of France?" \\
        [--system-prompt "You are a helpful assistant."] \\
        [--api-key AIza...] \\
        [--cot] [--self-reflect] [--temperature 0.2] \\
        [--context "Paris is in France." "France is in Europe."] \\
        [--glossary HIPAA=HIPAA LiteLLM=LiteLLM] \\
        [--output-json]

This module is also installed as the ``chinese-prompt-optimizer`` console
script when the package is installed via pip.
"""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chinese-prompt-optimizer",
        description=(
            "Reduce LLM token costs by translating English system prompts to "
            "Chinese with integrated anti-hallucination NLP."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (requires --message).",
    )
    p.add_argument(
        "--provider",
        default="gemini",
        choices=["chatgpt", "claude", "gemini"],
        help="LLM provider (default: gemini).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name override (default: provider default).",
    )
    p.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Always be concise and accurate.",
        help="English system prompt text.",
    )
    p.add_argument(
        "--message",
        default=None,
        help="User message (required for --headless).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key (falls back to GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY).",
    )
    p.add_argument(
        "--cot",
        action="store_true",
        help="Enable Chain-of-Thought instructions.",
    )
    p.add_argument(
        "--self-reflect",
        action="store_true",
        help="Enable self-reflection instructions.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        metavar="TEMP",
        help="Sampling temperature clamped to [0.1, 0.4] (default: 0.2).",
    )
    p.add_argument(
        "--context",
        nargs="*",
        default=None,
        metavar="SNIPPET",
        help="Verified context snippets for RAG-lite grounding.",
    )
    p.add_argument(
        "--glossary",
        nargs="*",
        default=None,
        metavar="TERM=VALUE",
        help="Glossary entries, e.g. HIPAA=HIPAA LiteLLM=LiteLLM.",
    )
    p.add_argument(
        "--output-json",
        action="store_true",
        help="Print result as JSON (headless mode only).",
    )
    p.add_argument(
        "--sot",
        action="store_true",
        help=(
            "Enable Skeleton-of-Thought (SoT) mode: generate a concise skeleton "
            "outline first, then expand each point in parallel for faster responses."
        ),
    )
    p.add_argument(
        "--sot-sequential",
        action="store_true",
        help="Expand SoT skeleton points sequentially instead of in parallel (requires --sot).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def _parse_glossary(entries: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries or []:
        if "=" in entry:
            k, _, v = entry.partition("=")
            result[k.strip()] = v.strip()
    return result


def _run_headless(args: argparse.Namespace) -> None:
    """Execute a single completion in headless (non-GUI) mode."""
    import logging

    from .optimizer import ChinesePromptOptimizer
    from .providers import get_provider

    log = logging.getLogger("chinese_prompt_optimizer.cli")

    if not args.message:
        print("ERROR: --message is required for --headless mode.", file=sys.stderr)
        sys.exit(1)

    provider = get_provider(args.provider)
    model = provider.litellm_model(args.model or provider.default_model)
    api_key = args.api_key or provider.api_key_from_env()
    glossary = _parse_glossary(args.glossary)

    log.info("Headless mode: provider=%s model=%s", args.provider, model)

    optimizer = ChinesePromptOptimizer(
        model=model,
        api_key=api_key,
        glossary=glossary or None,
        temperature=args.temperature,
        use_cot=args.cot,
        use_self_reflect=args.self_reflect,
        hallucination_guard=True,
        use_sot=args.sot,
        sot_parallel=not args.sot_sequential,
    )
    result = optimizer.complete(
        system_prompt=args.system_prompt,
        user_message=args.message,
        return_savings=True,
        context_snippets=args.context,
    )

    if args.output_json:
        print(
            json.dumps(
                {
                    "response": result["response"],
                    "savings": result.get("savings"),
                    "grounding": result.get("grounding"),
                    "skeleton": result.get("skeleton"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"\nResponse:\n{result['response']}\n")
        if result.get("skeleton"):
            print("Skeleton outline:")
            for i, pt in enumerate(result["skeleton"], 1):
                print(f"  {i}. {pt}")
            print()
        sv = result.get("savings", {})
        print(
            f"Token savings: {sv.get('tokens_saved', '?')} tokens "
            f"({sv.get('saving_pct', '?')}%)  "
            f"[English: {sv.get('english_tokens', '?')}  "
            f"Chinese: {sv.get('chinese_tokens', '?')}]"
        )
        gr = result.get("grounding", {})
        if gr:
            status = "✓ Grounded" if gr.get("grounded") else "⚠ May be hallucinated"
            print(f"Grounding: {status} (overlap: {gr.get('overlap_ratio', '?')})")
            if gr.get("warning"):
                print(f"Warning: {gr['warning']}")


def _launch_gui() -> None:
    """Launch the Tkinter GUI application."""
    try:
        import tkinter as tk  # noqa: F401 – confirm available before importing gui
    except ImportError as exc:
        print(
            f"ERROR: GUI requires tkinter. Install python3-tk.\n  {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    from .gui import OptimizerApp

    import tkinter as tk

    root = tk.Tk()
    root.geometry("1020x900")
    OptimizerApp(root)
    root.mainloop()


def main() -> None:
    """Console-script entry point.

    Invoked by::

        python -m chinese_prompt_optimizer
        chinese-prompt-optimizer          # after pip install
    """
    parser = _build_parser()
    args = parser.parse_args()

    import logging

    from .logging_config import setup_logging

    setup_logging(level=getattr(logging, args.log_level))

    if args.headless:
        _run_headless(args)
    else:
        _launch_gui()


if __name__ == "__main__":
    main()
