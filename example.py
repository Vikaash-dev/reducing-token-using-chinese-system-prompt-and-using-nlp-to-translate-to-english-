"""
example.py
----------
Demonstrates how ChinesePromptOptimizer reduces token usage.

Usage (requires an OpenAI-compatible API key):

    export OPENAI_API_KEY="sk-..."
    python example.py

To use a different provider, change the `model` argument, e.g.:
    model="claude-3-5-sonnet-20241022"   # Anthropic
    model="ollama/llama3"                  # local Ollama
"""

import os

from chinese_prompt_optimizer import ChinesePromptOptimizer, token_savings_report
from chinese_prompt_optimizer.translator import Translator

# ---------------------------------------------------------------------------
# 1. Show translation & token savings without making an API call
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions concisely. "
    "Always be polite and accurate. If you do not know the answer, "
    "say so instead of guessing."
)

translator = Translator()
chinese_prompt = translator.english_to_chinese(SYSTEM_PROMPT)

report = token_savings_report(SYSTEM_PROMPT, chinese_prompt)

print("=== Token Savings Report ===")
print(f"English system prompt : {SYSTEM_PROMPT}")
print(f"Chinese system prompt : {chinese_prompt}")
print(f"English tokens        : {report['english_tokens']}")
print(f"Chinese tokens        : {report['chinese_tokens']}")
print(f"Tokens saved          : {report['tokens_saved']} ({report['saving_pct']}%)")
print()

# ---------------------------------------------------------------------------
# 2. Live API call (only executed when an API key is present)
# ---------------------------------------------------------------------------
if os.getenv("OPENAI_API_KEY"):
    optimizer = ChinesePromptOptimizer(model="gpt-3.5-turbo")

    result = optimizer.complete(
        system_prompt=SYSTEM_PROMPT,
        user_message="What is the capital of France?",
        return_savings=True,
    )

    print("=== Live API Call ===")
    print(f"Response : {result['response']}")
    print(f"Savings  : {result['savings']}")
else:
    print("Set OPENAI_API_KEY to run the live API example.")
