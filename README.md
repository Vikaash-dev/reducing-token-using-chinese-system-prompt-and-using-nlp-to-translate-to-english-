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
| [ZurichNLP/ContraDecode](https://github.com/ZurichNLP/ContraDecode) | Source-contrastive decoding that penalises translations not tied to the input — inspired `HallucinationGuard.check_source_grounding()` |
| [DAMO-NLP-SG/chain-of-knowledge](https://github.com/DAMO-NLP-SG) | Dynamic knowledge grounding for LLMs — inspired the RAG-lite context injection feature |
| [EdinburghNLP/Awesome-Hallucination-Detection](https://github.com/EdinburghNLP) | Curated survey of hallucination detection tools and papers |
| [technion-cs-nlp/Hallucination-Mitigation](https://github.com/technion-cs-nlp) | Benchmarks and interventions for LLM hallucination mitigation |

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

## Anti-hallucination techniques

`HallucinationGuard` implements five anti-hallucination strategies from current
NLP research and is automatically integrated into `ChinesePromptOptimizer`.

### 1. "I Don't Know" rule (always on)

The model is explicitly instructed in Chinese to respond with
"我不知道" ("I don't know") rather than fabricating content when uncertain.

### 2. Chain-of-Thought (CoT) analysis

Enabled via `use_cot=True`.  Step-by-step reasoning instructions are injected
into the Chinese system prompt, forcing the model to ground its response in the
input before answering.  Inspired by research showing CoT significantly reduces
speculative assertions.

```python
optimizer = ChinesePromptOptimizer(model="gemini/gemini-2.0-flash",
                                   api_key=..., use_cot=True)
```

### 3. Self-Reflection

Enabled via `use_self_reflect=True`.  After answering, the model is instructed
to self-check its output against the source for unverified content.

### 4. Low temperature enforcement

`temperature` is automatically clamped to `[0.1, 0.4]` — the range recommended
for factual/translation tasks (default `0.2`).

```python
optimizer = ChinesePromptOptimizer(model="gpt-4o", temperature=0.15, ...)
```

### 5. Source-grounding check (ContraDecode-inspired)

Enabled by default via `hallucination_guard=True`.  After every response the
library measures the fraction of key source terms that appear in the answer.
The result is available under the `"grounding"` key:

```python
result = optimizer.complete("Be helpful.", "What is aspirin used for?")
print(result["grounding"])
# {"grounded": True, "overlap_ratio": 0.75, "warning": ""}
```

### 6. RAG-lite context injection

Pass verified facts via `context_snippets` to force the model to reference
evidence rather than guessing (reduces hallucinations by 42–68 %):

```python
result = optimizer.complete(
    system_prompt="You are a medical assistant.",
    user_message="What is aspirin used for?",
    context_snippets=[
        "Aspirin (acetylsalicylic acid) is an NSAID used for pain relief.",
        "Aspirin inhibits COX-1 and COX-2 enzymes.",
    ],
)
```

### 7. Few-shot prompting

Provide quality example pairs to demonstrate expected style and factual
groundedness:

```python
optimizer = ChinesePromptOptimizer(
    model="gemini/gemini-2.0-flash",
    api_key=...,
    few_shot_examples=[
        {"user": "What is DNA?",
         "assistant": "DNA (deoxyribonucleic acid) carries genetic information."},
    ],
)
```

### Direct use

```python
from chinese_prompt_optimizer import HallucinationGuard

# Temperature clamping
safe_temp = HallucinationGuard.enforce_temperature(0.9)  # → 0.4

# Augment an existing Chinese prompt
guarded = HallucinationGuard.build_guarded_prompt(
    "你是医疗助理。",
    use_idk_rule=True,
    use_cot=True,
)

# Source-grounding check
result = HallucinationGuard.check_source_grounding(
    source_text="What is the capital of France?",
    response_text="Paris is the capital of France.",
    min_overlap_ratio=0.3,
)
# {"grounded": True, "overlap_ratio": 0.8, "warning": ""}
```

---

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
├── __init__.py            – public exports
├── anti_hallucination.py  – HallucinationGuard (IDK rule, CoT, self-reflect,
│                            temperature enforcement, source-grounding check,
│                            RAG-lite context block, few-shot message builder)
├── providers.py           – opencode-style provider registry (ChatGPT/Claude/Gemini)
├── translator.py          – NLP translation with glossary & sentence chunking
├── optimizer.py           – ChinesePromptOptimizer (LiteLLM backend)
├── utils.py               – token counting + plot_token_comparison()
└── gui.py                 – Tkinter GUI with embedded matplotlib line graph

tests/
├── test_anti_hallucination.py  – HallucinationGuard unit tests (34 tests)
├── test_providers.py           – provider registry unit tests
├── test_translator.py          – translation + context preservation tests
├── test_optimizer.py           – optimizer unit tests (incl. anti-hallucination params)
├── test_utils.py               – token counting + graph tests
└── test_integration_gemini.py  – live Gemini integration tests (auto-skipped offline)

example.py    – CLI demo (token savings report + optional live call)
```
