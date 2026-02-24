"""
gui.py
------
Tkinter GUI for the Chinese Prompt Optimizer.

Layout
~~~~~~
Title bar
Provider | Model | API Key
Anti-Hallucination: IDK(always) | CoT | Self-Reflect | Temp | Guard
System Prompt (left) | Glossary (right)
Context Snippets (RAG-lite)
User Message                  [Run] [Clear Graph] [Export CSV]
Response                      Grounding badge
Token Line Graph

Design notes
~~~~~~~~~~~~
* All display-dependent imports (tkinter, matplotlib TkAgg) are deferred to
  runtime so the module is safely importable in headless / test environments.
* Provider switching follows the opencode registry pattern.
* Anti-hallucination controls surface all HallucinationGuard options in the UI.
* RAG-lite context snippets (one per line) forwarded to optimizer.complete().
* Grounding badge shows ContraDecode-inspired source-overlap result.
* Completions run in daemon thread; Tk updates dispatched via root.after(0).
"""

from __future__ import annotations

import csv
import threading
from typing import Dict, List, Optional

from .logging_config import get_logger
from .optimizer import ChinesePromptOptimizer
from .providers import PROVIDER_REGISTRY, ProviderConfig, list_providers

_log = get_logger("gui")

# ---------------------------------------------------------------------------
# Colour palette  (Catppuccin Mocha)
# ---------------------------------------------------------------------------
_BG = "#1e1e2e"
_FG = "#cdd6f4"
_ACCENT = "#89b4fa"
_GREEN = "#a6e3a1"
_RED = "#f38ba8"
_YELLOW = "#f9e2af"
_ENTRY_BG = "#313244"
_BTN_BG = "#89b4fa"
_BTN_FG = "#1e1e2e"
_MUTED = "#6c7086"
_SURFACE = "#45475a"


class OptimizerApp:
    """Main Tkinter application window."""

    def __init__(self, root: object) -> None:
        import tkinter as tk

        self._root = root
        self._root.title("🀄 Chinese Prompt Optimizer")
        self._root.configure(bg=_BG)
        self._root.resizable(True, True)

        self._provider_var = tk.StringVar()
        self._model_var = tk.StringVar()
        self._api_key_var = tk.StringVar()
        self._status_var = tk.StringVar(value="Ready")

        # Anti-hallucination controls
        self._cot_var = tk.BooleanVar(value=False)
        self._self_reflect_var = tk.BooleanVar(value=False)
        self._temperature_var = tk.DoubleVar(value=0.2)
        self._guard_var = tk.BooleanVar(value=True)

        self._grounding_var = tk.StringVar(value="")

        self._savings_history: List[Dict] = []
        self._history_labels: List[str] = []

        self._build_ui()
        self._on_provider_change()
        _log.debug("OptimizerApp initialised.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        pad: Dict[str, int] = {"padx": 10, "pady": 5}

        tk.Label(
            self._root, text="🀄  Chinese Prompt Optimizer",
            font=("Helvetica", 16, "bold"), bg=_BG, fg=_ACCENT,
        ).pack(fill="x", padx=10, pady=(8, 4))

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        top = tk.Frame(self._root, bg=_BG)
        top.pack(fill="x", **pad)
        self._build_provider_row(top)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        ah_frame = tk.Frame(self._root, bg=_BG)
        ah_frame.pack(fill="x", **pad)
        self._build_antihallucination_row(ah_frame)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        mid = tk.Frame(self._root, bg=_BG)
        mid.pack(fill="both", expand=False, **pad)
        self._build_prompt_area(mid)

        ctx_frame = tk.Frame(self._root, bg=_BG)
        ctx_frame.pack(fill="x", padx=10, pady=(0, 4))
        self._build_context_area(ctx_frame)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        user_frame = tk.Frame(self._root, bg=_BG)
        user_frame.pack(fill="x", **pad)
        self._build_user_message(user_frame)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        resp_frame = tk.Frame(self._root, bg=_BG)
        resp_frame.pack(fill="both", expand=True, **pad)
        self._build_response_area(resp_frame)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        graph_frame = tk.Frame(self._root, bg=_BG)
        graph_frame.pack(fill="both", expand=True, **pad)
        self._build_graph_area(graph_frame)

        tk.Label(
            self._root, textvariable=self._status_var,
            font=("Helvetica", 9), bg=_BG, fg=_MUTED, anchor="w",
        ).pack(fill="x", padx=10, pady=(2, 4))

    def _build_provider_row(self, parent: object) -> None:
        import tkinter as tk
        from tkinter import ttk

        providers = list_providers()
        self._provider_var.set(providers[0].name)
        provider_names = [p.name for p in providers]

        tk.Label(parent, text="Provider:", bg=_BG, fg=_FG,
                 font=("Helvetica", 10)).grid(row=0, column=0, sticky="w", padx=(0, 4))
        cb = ttk.Combobox(parent, textvariable=self._provider_var,
                           values=provider_names, state="readonly", width=24)
        cb.grid(row=0, column=1, padx=(0, 16))
        cb.bind("<<ComboboxSelected>>", lambda _: self._on_provider_change())

        tk.Label(parent, text="Model:", bg=_BG, fg=_FG,
                 font=("Helvetica", 10)).grid(row=0, column=2, sticky="w", padx=(0, 4))
        self._model_cb = ttk.Combobox(parent, textvariable=self._model_var,
                                       state="readonly", width=32)
        self._model_cb.grid(row=0, column=3, padx=(0, 16))

        tk.Label(parent, text="API Key:", bg=_BG, fg=_FG,
                 font=("Helvetica", 10)).grid(row=0, column=4, sticky="w", padx=(0, 4))
        tk.Entry(parent, textvariable=self._api_key_var, show="•", width=28,
                 bg=_ENTRY_BG, fg=_FG, insertbackground=_FG,
                 relief="flat").grid(row=0, column=5)

    def _build_antihallucination_row(self, parent: object) -> None:
        import tkinter as tk
        from tkinter import ttk

        tk.Label(parent, text="Anti-Hallucination:", bg=_BG, fg=_ACCENT,
                 font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w",
                                                       padx=(0, 8))
        tk.Label(parent, text="\u2713 IDK Rule", bg=_BG, fg=_GREEN,
                 font=("Helvetica", 10)).grid(row=0, column=1, padx=(0, 12))

        tk.Checkbutton(parent, text="Chain-of-Thought", variable=self._cot_var,
                       bg=_BG, fg=_FG, selectcolor=_ENTRY_BG,
                       activebackground=_BG, activeforeground=_FG,
                       font=("Helvetica", 10)).grid(row=0, column=2, padx=(0, 8))

        tk.Checkbutton(parent, text="Self-Reflect", variable=self._self_reflect_var,
                       bg=_BG, fg=_FG, selectcolor=_ENTRY_BG,
                       activebackground=_BG, activeforeground=_FG,
                       font=("Helvetica", 10)).grid(row=0, column=3, padx=(0, 8))

        tk.Label(parent, text="Temp:", bg=_BG, fg=_FG,
                 font=("Helvetica", 10)).grid(row=0, column=4, padx=(8, 2))
        ttk.Spinbox(parent, textvariable=self._temperature_var,
                    from_=0.1, to=0.4, increment=0.05,
                    width=6, format="%.2f").grid(row=0, column=5, padx=(0, 12))

        tk.Checkbutton(parent, text="Source Guard", variable=self._guard_var,
                       bg=_BG, fg=_FG, selectcolor=_ENTRY_BG,
                       activebackground=_BG, activeforeground=_FG,
                       font=("Helvetica", 10)).grid(row=0, column=6)

    def _build_prompt_area(self, parent: object) -> None:
        import tkinter as tk
        from tkinter import scrolledtext

        left = tk.Frame(parent, bg=_BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(left, text="System Prompt (English)", bg=_BG, fg=_ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w")
        self._system_prompt = scrolledtext.ScrolledText(
            left, width=48, height=6, bg=_ENTRY_BG, fg=_FG,
            insertbackground=_FG, relief="flat", font=("Helvetica", 10))
        self._system_prompt.pack(fill="both", expand=True)
        self._system_prompt.insert(
            "end",
            "You are a helpful assistant. Always be concise and accurate. "
            "If you do not know the answer, say so.")

        right = tk.Frame(parent, bg=_BG)
        right.pack(side="left", fill="both", expand=False)
        tk.Label(right, text="Glossary  (term=Chinese, \u2026)", bg=_BG, fg=_ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w")
        tk.Label(right, text="Terms here are never passed through NMT\n"
                             "\u2014 contextual meaning preserved exactly.",
                 bg=_BG, fg=_MUTED, font=("Helvetica", 8),
                 justify="left").pack(anchor="w")
        self._glossary = scrolledtext.ScrolledText(
            right, width=30, height=5, bg=_ENTRY_BG, fg=_FG,
            insertbackground=_FG, relief="flat", font=("Helvetica", 10))
        self._glossary.pack(fill="both", expand=True)
        self._glossary.insert("end", "HIPAA=HIPAA\nLiteLLM=LiteLLM")

    def _build_context_area(self, parent: object) -> None:
        import tkinter as tk
        from tkinter import scrolledtext

        tk.Label(parent,
                 text="Context Snippets  (RAG-lite \u2014 one verified fact per line)",
                 bg=_BG, fg=_ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w")
        tk.Label(parent,
                 text="Injected before the user message to ground the model in facts "
                      "(reduces hallucinations 42\u201368 %).",
                 bg=_BG, fg=_MUTED, font=("Helvetica", 8)).pack(anchor="w")
        self._context_box = scrolledtext.ScrolledText(
            parent, width=80, height=3, bg=_ENTRY_BG, fg=_FG,
            insertbackground=_FG, relief="flat", font=("Helvetica", 10))
        self._context_box.pack(fill="x")

    def _build_user_message(self, parent: object) -> None:
        import tkinter as tk
        from tkinter import scrolledtext

        tk.Label(parent, text="User Message", bg=_BG, fg=_ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w")
        self._user_msg = scrolledtext.ScrolledText(
            parent, width=80, height=3, bg=_ENTRY_BG, fg=_FG,
            insertbackground=_FG, relief="flat", font=("Helvetica", 10))
        self._user_msg.pack(fill="x")
        self._user_msg.insert("end", "What is the capital of France?")

        btn_frame = tk.Frame(parent, bg=_BG)
        btn_frame.pack(anchor="e", pady=(6, 0))
        tk.Button(btn_frame, text="  \u25b6  Run  ", command=self._on_run,
                  bg=_BTN_BG, fg=_BTN_FG, font=("Helvetica", 11, "bold"),
                  relief="flat", cursor="hand2", padx=12, pady=4).pack(
            side="left", padx=4)
        tk.Button(btn_frame, text="  \U0001f5d1  Clear Graph  ",
                  command=self._clear_graph, bg=_ENTRY_BG, fg=_FG,
                  font=("Helvetica", 10), relief="flat", cursor="hand2",
                  padx=8, pady=4).pack(side="left", padx=4)
        tk.Button(btn_frame, text="  \U0001f4be  Export CSV  ",
                  command=self._export_csv, bg=_ENTRY_BG, fg=_FG,
                  font=("Helvetica", 10), relief="flat", cursor="hand2",
                  padx=8, pady=4).pack(side="left")

    def _build_response_area(self, parent: object) -> None:
        import tkinter as tk
        from tkinter import scrolledtext

        header = tk.Frame(parent, bg=_BG)
        header.pack(fill="x")
        tk.Label(header, text="Response", bg=_BG, fg=_ACCENT,
                 font=("Helvetica", 10, "bold")).pack(side="left")
        self._grounding_lbl = tk.Label(header, textvariable=self._grounding_var,
                                        bg=_BG, fg=_MUTED,
                                        font=("Helvetica", 9), anchor="e")
        self._grounding_lbl.pack(side="right")

        self._response_box = scrolledtext.ScrolledText(
            parent, width=80, height=6, bg=_ENTRY_BG, fg=_GREEN,
            insertbackground=_FG, relief="flat", font=("Helvetica", 10),
            state="disabled")
        self._response_box.pack(fill="both", expand=True)

    def _build_graph_area(self, parent: object) -> None:
        import tkinter as tk
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        tk.Label(
            parent,
            text="Token Usage  \u2014  English tokens \u00b7 "
                 "Chinese tokens (actual used) \u00b7 Saved tokens",
            bg=_BG, fg=_ACCENT, font=("Helvetica", 10, "bold"),
        ).pack(anchor="w")

        self._fig, self._ax = plt.subplots(figsize=(9, 2.8))
        self._fig.patch.set_facecolor(_BG)
        self._ax.set_facecolor(_ENTRY_BG)
        self._ax.tick_params(colors=_FG)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(_SURFACE)
        self._ax.set_title("Run the optimizer to populate the graph",
                            color=_MUTED, fontsize=9)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_provider_change(self) -> None:
        selected_name = self._provider_var.get()
        cfg = self._provider_by_name(selected_name)
        if cfg is None:
            return
        self._model_cb["values"] = cfg.models
        self._model_var.set(cfg.default_model)
        if cfg.is_configured() and not self._api_key_var.get():
            self._api_key_var.set(cfg.api_key_from_env() or "")
            _log.debug("Pre-filled API key from env for provider %s.", cfg.id)

    def _on_run(self) -> None:
        from tkinter import messagebox

        system_prompt = self._system_prompt.get("1.0", "end").strip()
        user_message = self._user_msg.get("1.0", "end").strip()
        if not system_prompt:
            messagebox.showwarning("Missing input", "Please enter a system prompt.")
            return
        if not user_message:
            messagebox.showwarning("Missing input", "Please enter a user message.")
            return

        api_key = self._api_key_var.get().strip() or None
        cfg = self._provider_by_name(self._provider_var.get())
        if cfg is None:
            messagebox.showerror("Error", "Unknown provider selected.")
            return

        litellm_model = cfg.litellm_model(self._model_var.get())
        glossary = self._parse_glossary()
        context_snippets = self._get_context_snippets()

        self._set_status("Running\u2026")
        self._set_response("")
        self._grounding_var.set("")

        _log.info(
            "Starting: provider=%s model=%s cot=%s reflect=%s temp=%.2f guard=%s",
            cfg.id, litellm_model, self._cot_var.get(),
            self._self_reflect_var.get(), self._temperature_var.get(),
            self._guard_var.get(),
        )

        threading.Thread(
            target=self._run_completion,
            args=(litellm_model, api_key, system_prompt, user_message,
                  glossary, context_snippets),
            daemon=True,
        ).start()

    def _run_completion(
        self,
        litellm_model: str,
        api_key: Optional[str],
        system_prompt: str,
        user_message: str,
        glossary: Dict[str, str],
        context_snippets: List[str],
    ) -> None:
        try:
            optimizer = ChinesePromptOptimizer(
                model=litellm_model,
                api_key=api_key,
                glossary=glossary if glossary else None,
                temperature=self._temperature_var.get(),
                use_cot=self._cot_var.get(),
                use_self_reflect=self._self_reflect_var.get(),
                hallucination_guard=self._guard_var.get(),
            )
            result = optimizer.complete(
                system_prompt=system_prompt,
                user_message=user_message,
                return_savings=True,
                context_snippets=context_snippets if context_snippets else None,
            )
            _log.info("Completion succeeded.")
            self._root.after(0, self._on_success, result)
        except (ValueError, RuntimeError, ConnectionError, OSError) as exc:
            _log.warning("Completion failed: %s", exc)
            self._root.after(0, self._on_error, str(exc))
        except Exception as exc:  # noqa: BLE001
            _log.error("Unexpected error: %s", exc, exc_info=True)
            self._root.after(0, self._on_error, f"Unexpected error: {exc}")

    def _on_success(self, result: Dict) -> None:
        self._set_response(result["response"])
        savings = result.get("savings", {})
        run_label = f"Run {len(self._savings_history) + 1}"
        self._savings_history.append(savings)
        self._history_labels.append(run_label)
        self._update_graph()
        pct = savings.get("saving_pct", 0)
        saved = savings.get("tokens_saved", 0)
        self._set_status(
            f"Done \u2014 saved {saved} tokens ({pct}%) on this run  "
            f"(English: {savings.get('english_tokens', '?')}  "
            f"Chinese: {savings.get('chinese_tokens', '?')})"
        )
        grounding = result.get("grounding")
        if grounding is not None:
            self._update_grounding_badge(grounding)

    def _on_error(self, message: str) -> None:
        from tkinter import messagebox
        self._set_status(f"Error: {message}")
        messagebox.showerror("API Error", message)

    def _clear_graph(self) -> None:
        self._savings_history.clear()
        self._history_labels.clear()
        self._ax.clear()
        self._ax.set_facecolor(_ENTRY_BG)
        self._ax.tick_params(colors=_FG)
        self._ax.set_title("Graph cleared \u2014 run the optimizer to repopulate",
                            color=_MUTED, fontsize=9)
        self._canvas.draw()
        _log.debug("Graph cleared.")

    def _export_csv(self) -> None:
        from tkinter import filedialog, messagebox

        if not self._savings_history:
            messagebox.showinfo("No data",
                                "Run the optimizer at least once before exporting.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="token_savings.csv",
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["run", "english_tokens", "chinese_tokens",
                                    "tokens_saved", "saving_pct"])
                writer.writeheader()
                for label, row in zip(self._history_labels, self._savings_history):
                    writer.writerow({"run": label, **row})
            self._set_status(f"Exported {len(self._savings_history)} rows \u2192 {path}")
            _log.info("Exported CSV to %s", path)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            _log.error("CSV export failed: %s", exc)

    # ------------------------------------------------------------------
    # Graph rendering
    # ------------------------------------------------------------------

    def _update_graph(self) -> None:
        if not self._savings_history:
            return
        n = len(self._savings_history)
        x = list(range(1, n + 1))
        en = [int(r.get("english_tokens", 0)) for r in self._savings_history]
        zh = [int(r.get("chinese_tokens", 0)) for r in self._savings_history]
        sv = [int(r.get("tokens_saved", 0)) for r in self._savings_history]

        ax = self._ax
        ax.clear()
        ax.set_facecolor(_ENTRY_BG)
        ax.tick_params(colors=_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_SURFACE)

        ax.plot(x, en, color=_ACCENT, marker="o", linewidth=2,
                markersize=6, label="English tokens")
        ax.plot(x, zh, color=_GREEN, marker="o", linewidth=2,
                markersize=6, label="Chinese tokens (actual used)")
        ax.plot(x, sv, color=_RED, marker="o", linestyle="--", linewidth=2,
                markersize=6, label="Saved tokens")
        ax.fill_between(x, zh, en, alpha=0.15, color=_GREEN)

        for xi, (e, z, s) in zip(x, zip(en, zh, sv)):
            ax.annotate(str(e), (xi, e), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=_ACCENT)
            ax.annotate(str(z), (xi, z), xytext=(0, -12),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=_GREEN)
            ax.annotate(f"-{s}", (xi, s), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=_RED)

        ax.set_xticks(x)
        ax.set_xticklabels(self._history_labels, rotation=0)
        ax.set_ylabel("Token count", color=_FG, fontsize=8)
        ax.yaxis.label.set_color(_FG)
        ax.set_title("Token usage per run", color=_FG, fontsize=9)
        ax.legend(fontsize=8, facecolor=_BG, edgecolor=_SURFACE, labelcolor=_FG)
        ax.grid(True, alpha=0.2, color=_SURFACE)
        self._fig.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Grounding badge
    # ------------------------------------------------------------------

    def _update_grounding_badge(self, grounding: Dict) -> None:
        grounded = grounding.get("grounded", True)
        ratio = grounding.get("overlap_ratio", 0.0)
        if grounded:
            text = f"Grounding: \u2713 Grounded  (overlap: {ratio:.0%})"
            color = _GREEN
        else:
            text = f"Grounding: \u26a0 Check response  (overlap: {ratio:.0%})"
            color = _YELLOW
        self._grounding_var.set(text)
        self._grounding_lbl.configure(fg=color)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provider_by_name(self, name: str) -> Optional[ProviderConfig]:
        for p in PROVIDER_REGISTRY.values():
            if p.name == name:
                return p
        return None

    def _parse_glossary(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        raw = self._glossary.get("1.0", "end").strip()
        for line in raw.splitlines():
            line = line.strip()
            if "=" in line:
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if key:
                    result[key] = val
        return result

    def _get_context_snippets(self) -> List[str]:
        raw = self._context_box.get("1.0", "end").strip()
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def _set_response(self, text: str) -> None:
        self._response_box.configure(state="normal")
        self._response_box.delete("1.0", "end")
        self._response_box.insert("end", text)
        self._response_box.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    """Launch the GUI application (blocks until the window is closed)."""
    import tkinter as tk

    from .logging_config import setup_logging

    setup_logging()
    root = tk.Tk()
    root.geometry("1020x900")
    OptimizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
