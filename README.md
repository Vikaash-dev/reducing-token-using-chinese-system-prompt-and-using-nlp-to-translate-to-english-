# Chinese Prompt Optimizer

> **Reduce LLM token costs** by translating English system prompts into Chinese
> (token-dense), calling any LLM provider via LiteLLM, and translating the
> response back to English with NLP-based machine translation — all without
> losing contextual meaning.

---

## How it works

```
English system prompt
       │
       ▼  NLP translation (deep-translator / Google NMT)
       │  + placeholder-based glossary to preserve technical terms
       ▼
Chinese system prompt  ──▶  LiteLLM  ──▶  ChatGPT / Claude / Gemini
                                                    │
                                                    ▼
                                         Chinese response
                                                    │
                                  NLP back-translation to English
                                                    │
                                                    ▼
                                           English answer
```

Chinese encodes more information per token than English: a typical system
prompt of 45 English tokens compresses to ~18 Chinese tokens — a **≈ 60 %**
saving on every API call.

---

## Similar projects

| Project | What it does |
|---------|-------------|
| [anomalyco/opencode](https://github.com/anomalyco/opencode) | Open-source AI coding agent; **provider-switching architecture** inspired our `providers.py` registry |
| [wyne1/llm-orchestrator](https://github.com/wyne1/llm-orchestrator) | Adapter-pattern LLM orchestrator for OpenAI / Anthropic / Gemini |
| [BerriAI/litellm](https://github.com/BerriAI/litellm) | Universal LLM gateway (used as our completion backend) |
| [nidhaloff/deep-translator](https://github.com/nidhaloff/deep-translator) | NLP translation library wrapping Google NMT (used for English↔Chinese) |

---

## Quick start

```bash
pip install -r requirements.txt
```

### CLI / script

```python
from chinese_prompt_optimizer import ChinesePromptOptimizer
import os

optimizer = ChinesePromptOptimizer(
    model="gemini/gemini-2.0-flash",          # or gpt-4o, anthropic/claude-3-5-sonnet-20241022
    api_key=os.environ["GEMINI_API_KEY"],
    glossary={
        "HIPAA": "HIPAA",      # keep acronym unchanged
        "LiteLLM": "LiteLLM", # keep brand name
    },
)

result = optimizer.complete(
    system_prompt="You are a helpful medical assistant. Always reference HIPAA.",
    user_message="What should I know about patient data privacy?",
    return_savings=True,
)

print(result["response"])
print(result["savings"])
# {'english_tokens': 18, 'chinese_tokens': 7, 'tokens_saved': 11, 'saving_pct': 61.11}
```

### GUI (Tkinter)

```bash
python -m chinese_prompt_optimizer.gui
```

The GUI lets you:
- **Switch providers** (ChatGPT / Claude / Gemini) from a dropdown
- **Switch models** within each provider
- **Enter API key** directly or set the env var (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- **Enter glossary terms** (`HIPAA=HIPAA`, `LiteLLM=LiteLLM`) to protect contextual meaning
- **View a live token line graph** showing English tokens, Chinese tokens (actual used), and saved tokens per run

---

## Provider support (opencode-style registry)

| Provider | Env var | Default model |
|----------|---------|---------------|
| **ChatGPT** (OpenAI) | `OPENAI_API_KEY` | `gpt-4o` |
| **Claude** (Anthropic) | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` |
| **Gemini** (Google AI Studio) | `GEMINI_API_KEY` | `gemini/gemini-2.0-flash` |

```python
from chinese_prompt_optimizer import get_provider, list_providers

for p in list_providers():
    print(p.name, "→", p.default_model)
```

---

## Context preservation

Technical terms, proper nouns, and domain jargon are **never** passed
through the NMT engine.  Supply a `glossary` dict and they will be swapped
with opaque placeholders before translation, then restored afterwards:

```python
optimizer = ChinesePromptOptimizer(
    model="gemini/gemini-2.0-flash",
    api_key=os.environ["GEMINI_API_KEY"],
    glossary={
        "HIPAA":   "HIPAA",          # keep unchanged
        "GPT-4o":  "GPT-4o",         # keep unchanged
        "RAG":     "检索增强生成",      # force a specific Chinese term
    },
)
```

Long prompts are translated **sentence-by-sentence** so coherence is
maintained across clause boundaries.

---

## Token line graph

```python
from chinese_prompt_optimizer import token_savings_report, plot_token_comparison

reports = [
    token_savings_report("You are helpful.", "你很有帮助。"),
    token_savings_report(
        "You are a professional medical assistant. Always be accurate.",
        "你是专业的医疗助理。始终准确。",
    ),
]
plot_token_comparison(reports, labels=["Short", "Long"], save_path="savings.png")
```

The graph shows three lines:
- 🔵 **English tokens** (original)
- 🟢 **Chinese tokens** (actual used)
- 🔴 **Saved tokens**

with a shaded savings area between the two main lines.

---

## Running tests

```bash
# All unit tests (no API key needed)
pytest tests/ -v

# Live Gemini integration tests
GEMINI_API_KEY="AIza..." pytest tests/test_integration_gemini.py -v
```

The 5 integration tests in `test_integration_gemini.py` are automatically
**skipped** when `GEMINI_API_KEY` is not set or the network is unreachable —
they will run automatically in environments where both are available.

---

## Project structure

```
chinese_prompt_optimizer/
├── __init__.py       – public exports
├── providers.py      – opencode-style provider registry (ChatGPT/Claude/Gemini)
├── translator.py     – NLP translation with glossary & sentence chunking
├── optimizer.py      – ChinesePromptOptimizer (LiteLLM backend)
├── utils.py          – token counting + plot_token_comparison()
└── gui.py            – Tkinter GUI with embedded matplotlib line graph

tests/
├── test_providers.py           – provider registry unit tests
├── test_translator.py          – translation + context preservation tests
├── test_optimizer.py           – optimizer unit tests
├── test_utils.py               – token counting + graph tests
└── test_integration_gemini.py  – live Gemini integration tests (auto-skipped offline)

example.py    – CLI demo (token savings report + optional live call)
```
