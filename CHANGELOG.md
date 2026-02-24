# Changelog

All notable changes to **Chinese Prompt Optimizer** are documented in this
file.  The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-02-24

### Added

#### Core pipeline
- `translator.py` — Bidirectional NLP translation (English ↔ Chinese Simplified)
  via deep-translator / Google NMT.  Placeholder-based glossary system protects
  technical terms (brand names, acronyms, domain jargon) from the NMT engine,
  preserving contextual meaning exactly.  Sentence-level chunking maintains
  coherence for long prompts.
- `optimizer.py` — `ChinesePromptOptimizer`: orchestrates translate → guard →
  LiteLLM → back-translate.  New parameters: `temperature`, `use_cot`,
  `use_self_reflect`, `hallucination_guard`, `few_shot_examples`.
  `complete()` gains `context_snippets` (RAG-lite grounding).
- `providers.py` — Opencode-style typed provider registry for ChatGPT (OpenAI),
  Claude (Anthropic), and Gemini (Google AI Studio).  Each provider has a
  `ProviderConfig` dataclass with `id`, `name`, `env_var`, `models`,
  `litellm_prefix`.  LiteLLM used as universal completion gateway.
- `utils.py` — Token counting with tiktoken (cl100k_base fallback) and
  `plot_token_comparison()` standalone line graph.

#### Anti-hallucination (`anti_hallucination.py`)
Seven evidence-based techniques from NLP research:
1. **"I Don't Know" rule** — always injected; model must acknowledge uncertainty.
2. **Chain-of-Thought (CoT)** — step-by-step analysis before answering.
3. **Self-Reflection** — self-review against source after answering.
4. **Temperature enforcement** — clamped to `[0.1, 0.4]`; default `0.2`.
5. **Source-Grounding Check** — ContraDecode-inspired term overlap heuristic.
6. **RAG-lite context injection** — verified facts prepended to user message.
7. **Few-shot prompting** — quality example pairs as conversation turns.

#### GUI (`gui.py`)
- Catppuccin Mocha dark theme.
- Provider / Model / API Key row with auto-fill from environment variable.
- **Anti-Hallucination options row**: IDK (always on indicator), CoT checkbox,
  Self-Reflect checkbox, Temperature spinbox (0.1–0.4, step 0.05), Source Guard
  checkbox.
- **Context Snippets** text area (RAG-lite, one fact per line).
- User Message + **Run** / **Clear Graph** / **Export CSV** buttons.
- Response area with **Grounding badge** (✓ Grounded / ⚠ Check response + overlap %).
- Embedded **token line graph** (English · Chinese actual · Saved) updated per run.
- All display-dependent imports deferred (module safely importable headless).
- Completions run in daemon thread; Tk updates via `root.after(0, …)`.

#### Packaging & CLI
- `pyproject.toml` — PEP 517/518 packaging with `chinese-prompt-optimizer`
  console script entry point.
- `__main__.py` — `python -m chinese_prompt_optimizer` with headless CLI mode
  (`--headless --provider gemini --message "..." [--cot] [--self-reflect]
  [--context ...] [--output-json]`).
- `logging_config.py` — Structured logging: stderr (INFO+) + daily-rotating
  file (`~/.chinese_prompt_optimizer/logs/optimizer.log`, DEBUG+).

#### Documentation
- `docs/research_notes.txt` — ADA-7 knowledge base: 9 arXiv citations,
  7 GitHub repository analyses, cross-analysis findings, measured token savings.
- `docs/architecture.md` — ADA-7 Stage 2: three architecture variants,
  decision matrix (Monolithic / Microservices / Plugin), risk assessment,
  evolution roadmap.

#### Tests
- `tests/test_anti_hallucination.py` — 34 unit tests for `HallucinationGuard`.
- `tests/test_gui.py` — 17 tests: logic-only (no display) + Xvfb smoke tests.
- `tests/test_optimizer.py` — 15 tests including anti-hallucination params.
- `tests/test_integration_gemini.py` — 5 live Gemini tests (auto-skip offline)
  + 1 mock test always runs.
- **Total: 105 tests passing, 5 live Gemini tests auto-skip in CI.**

### Infrastructure
- `pytest.ini_options` in `pyproject.toml` (testpaths, addopts).
- `pytest-cov` for coverage reporting.
- `.gitignore` excludes `__pycache__`, `*.pyc`, `.env`, `dist/`, `*.egg-info/`,
  `*.png` (matplotlib output), `.pytest_cache/`.

---

## References
- [Lewis et al., 2020, arXiv:2005.11401] RAG for Knowledge-Intensive NLP
- [Wei et al., 2022, arXiv:2201.11903] Chain-of-Thought Prompting
- [Manakul et al., 2023, arXiv:2303.08896] SelfCheckGPT
- [Ji et al., 2023, arXiv:2202.03629] Survey of Hallucination in NLG
- [Li et al., 2022, arXiv:2210.15097] Contrastive Decoding
- [Zhang et al., 2023, arXiv:2309.01219] Survey on Hallucination in LLMs
- ZurichNLP/ContraDecode — source-contrastive decoding
- anomalyco/opencode — provider registry architecture
- BerriAI/litellm — universal LLM gateway
