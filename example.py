"""
example.py
----------
Demonstrates how ChinesePromptOptimizer reduces token usage, with Gemini
(Google AI Studio) as the default live-test provider.

Usage (token savings preview — no API key needed):

    python example.py

Usage (live Gemini API call):

    export GEMINI_API_KEY="AIza..."
    python example.py

To use a different provider set the corresponding key instead:
    export OPENAI_API_KEY="sk-..."
    # then change model="gpt-4o" in the code below

    export ANTHROPIC_API_KEY="sk-ant-..."
    # then change model="anthropic/claude-3-5-sonnet-20241022"
"""

import os

from chinese_prompt_optimizer import (
    ChinesePromptOptimizer,
    get_provider,
    list_providers,
    plot_token_comparison,
    token_savings_report,
)
from chinese_prompt_optimizer.translator import Translator

# ---------------------------------------------------------------------------
# 1. List available providers (opencode-style registry)
# ---------------------------------------------------------------------------
print("=== Available Providers ===")
for p in list_providers():
    status = "✓ configured" if p.is_configured() else "✗ set " + p.env_var
    print(f"  {p.name:30s}  default={p.default_model:35s}  {status}")
print()

# ---------------------------------------------------------------------------
# 2. Show NLP translation + token savings (no API key required)
# ---------------------------------------------------------------------------
PROMPTS = [
    "You are a helpful assistant. Always be concise and accurate.",
    (
        "You are a professional medical assistant specialising in HIPAA compliance. "
        "Always provide evidence-based information and cite your sources."
    ),
    (
        "You are an expert software engineer specialising in Python and LiteLLM. "
        "Write clean, well-documented code. "
        "Explain your reasoning step by step before producing any code."
    ),
]

def _truncate(text: str, max_len: int = 70) -> str:
    return text[:max_len] + "…" if len(text) > max_len else text



reports = []

print("=== Token Savings Report ===")
for i, prompt in enumerate(PROMPTS, 1):
    zh = translator.english_to_chinese(prompt)
    report = token_savings_report(prompt, zh)
    reports.append(report)
    print(f"  Prompt {i}: en={report['english_tokens']:3d} tokens  "
          f"zh={report['chinese_tokens']:3d} tokens  "
          f"saved={report['tokens_saved']:3d} ({report['saving_pct']:.1f}%)")
    print(f"    EN: {_truncate(prompt)}")
    print(f"    ZH: {_truncate(zh)}")
print()

# ---------------------------------------------------------------------------
# 3. Save token comparison line graph
# ---------------------------------------------------------------------------
plot_token_comparison(
    reports,
    labels=[f"Prompt {i}" for i in range(1, len(reports) + 1)],
    title="Token Usage: English vs Chinese System Prompts",
    save_path="/tmp/token_savings.png",
    show=False,
)
print("Line graph saved to /tmp/token_savings.png")
print()

# ---------------------------------------------------------------------------
# 4. Live API call (only when GEMINI_API_KEY is present)
# ---------------------------------------------------------------------------
gemini_key = os.environ.get("GEMINI_API_KEY")
if gemini_key:
    p = get_provider("gemini")
    optimizer = ChinesePromptOptimizer(
        model=p.litellm_model(p.default_model),
        api_key=gemini_key,
        glossary={"HIPAA": "HIPAA", "LiteLLM": "LiteLLM"},
    )
    print(f"=== Live API Call ({p.name} · {p.default_model}) ===")
    result = optimizer.complete(
        system_prompt=PROMPTS[0],
        user_message="What is the capital of France? Answer in one sentence.",
        return_savings=True,
    )
    print(f"  Response : {result['response']}")
    print(f"  Savings  : {result['savings']}")
else:
    print("Set GEMINI_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY) to run the live API example.")
    print("  export GEMINI_API_KEY='AIza...'")
    print("  python example.py")
