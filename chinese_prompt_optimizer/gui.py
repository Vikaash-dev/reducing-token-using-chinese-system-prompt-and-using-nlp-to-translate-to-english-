"""
gui.py
------
Tkinter GUI for the Chinese Prompt Optimizer.

Layout
~~~~~~
┌──────────────────────────────────────────────────────────────────┐
│  🀄 Chinese Prompt Optimizer                                      │
├──────────────────────────────────────────────────────────────────┤
│  Provider ▼  [ChatGPT]   Model ▼  [gpt-4o]   API Key [______]  │
├──────────────────────────────────────────────────────────────────┤
│  System Prompt (English)          │  Glossary (term=zh, …)      │
│  [_________________________________│______________________________]│
├──────────────────────────────────────────────────────────────────┤
│  User Message                                                    │
│  [______________________________________________________________] │
│                              [  ▶ Run  ]                         │
├──────────────────────────────────────────────────────────────────┤
│  Response                                                        │
│  [______________________________________________________________] │
├──────────────────────────────────────────────────────────────────┤
│  Token Savings  ◀ line graph: English / Chinese / Saved ▶        │
└──────────────────────────────────────────────────────────────────┘

Provider switching is modelled after the opencode project's provider
registry (github.com/anomalyco/opencode) – each provider exposes a typed
config with its name, API-key env-var, and supported models.  LiteLLM is
the universal completion backend so switching providers requires nothing
more than selecting a different entry from the dropdown.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .optimizer import ChinesePromptOptimizer
from .providers import PROVIDER_REGISTRY, ProviderConfig, list_providers
from .utils import token_savings_report


# ---------------------------------------------------------------------------
# Colour palette (matches opencode's clean dark-accent aesthetic)
# ---------------------------------------------------------------------------
_BG = "#1e1e2e"
_FG = "#cdd6f4"
_ACCENT = "#89b4fa"
_GREEN = "#a6e3a1"
_RED = "#f38ba8"
_ENTRY_BG = "#313244"
_BTN_BG = "#89b4fa"
_BTN_FG = "#1e1e2e"


class OptimizerApp:
    """Main Tkinter application window."""

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title("🀄 Chinese Prompt Optimizer")
        self._root.configure(bg=_BG)
        self._root.resizable(True, True)

        # State
        self._provider_var = tk.StringVar()
        self._model_var = tk.StringVar()
        self._api_key_var = tk.StringVar()
        self._status_var = tk.StringVar(value="Ready")

        # Accumulated savings history for multi-run graph
        self._savings_history: List[Dict] = []
        self._history_labels: List[str] = []

        self._build_ui()
        # Trigger initial model list population
        self._on_provider_change()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        # ── Title bar ──────────────────────────────────────────────────
        title = tk.Label(
            self._root,
            text="🀄  Chinese Prompt Optimizer",
            font=("Helvetica", 16, "bold"),
            bg=_BG, fg=_ACCENT,
        )
        title.pack(fill="x", **pad)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        # ── Provider / Model / API Key row ─────────────────────────────
        top = tk.Frame(self._root, bg=_BG)
        top.pack(fill="x", **pad)
        self._build_provider_row(top)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        # ── Prompt + Glossary side-by-side ─────────────────────────────
        mid = tk.Frame(self._root, bg=_BG)
        mid.pack(fill="both", expand=False, **pad)
        self._build_prompt_area(mid)

        # ── User message + Run button ───────────────────────────────────
        user_frame = tk.Frame(self._root, bg=_BG)
        user_frame.pack(fill="x", **pad)
        self._build_user_message(user_frame)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        # ── Response ───────────────────────────────────────────────────
        resp_frame = tk.Frame(self._root, bg=_BG)
        resp_frame.pack(fill="both", expand=True, **pad)
        self._build_response_area(resp_frame)

        ttk.Separator(self._root, orient="horizontal").pack(fill="x", padx=10)

        # ── Token graph ────────────────────────────────────────────────
        graph_frame = tk.Frame(self._root, bg=_BG)
        graph_frame.pack(fill="both", expand=True, **pad)
        self._build_graph_area(graph_frame)

        # ── Status bar ─────────────────────────────────────────────────
        tk.Label(
            self._root,
            textvariable=self._status_var,
            font=("Helvetica", 9),
            bg=_BG, fg="#6c7086", anchor="w",
        ).pack(fill="x", padx=10, pady=2)

    def _build_provider_row(self, parent: tk.Frame) -> None:
        def lbl(text: str) -> tk.Label:
            return tk.Label(parent, text=text, bg=_BG, fg=_FG,
                            font=("Helvetica", 10))

        providers = list_providers()
        provider_names = [p.name for p in providers]
        self._provider_var.set(providers[0].name)

        lbl("Provider:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        provider_cb = ttk.Combobox(
            parent,
            textvariable=self._provider_var,
            values=provider_names,
            state="readonly",
            width=22,
        )
        provider_cb.grid(row=0, column=1, padx=(0, 16))
        provider_cb.bind("<<ComboboxSelected>>", lambda _: self._on_provider_change())

        lbl("Model:").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self._model_cb = ttk.Combobox(
            parent,
            textvariable=self._model_var,
            state="readonly",
            width=30,
        )
        self._model_cb.grid(row=0, column=3, padx=(0, 16))

        lbl("API Key:").grid(row=0, column=4, sticky="w", padx=(0, 4))
        tk.Entry(
            parent,
            textvariable=self._api_key_var,
            show="•",
            width=28,
            bg=_ENTRY_BG, fg=_FG,
            insertbackground=_FG,
            relief="flat",
        ).grid(row=0, column=5)

    def _build_prompt_area(self, parent: tk.Frame) -> None:
        def lbl(f: tk.Frame, text: str) -> None:
            tk.Label(f, text=text, bg=_BG, fg=_ACCENT,
                     font=("Helvetica", 10, "bold")).pack(anchor="w")

        left = tk.Frame(parent, bg=_BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        lbl(left, "System Prompt (English)")
        self._system_prompt = scrolledtext.ScrolledText(
            left, width=48, height=6,
            bg=_ENTRY_BG, fg=_FG, insertbackground=_FG,
            relief="flat", font=("Helvetica", 10),
        )
        self._system_prompt.pack(fill="both", expand=True)
        self._system_prompt.insert(
            "end",
            "You are a helpful assistant. Always be concise and accurate. "
            "If you do not know the answer, say so."
        )

        right = tk.Frame(parent, bg=_BG)
        right.pack(side="left", fill="both", expand=False)
        lbl(right, "Glossary  (term=Chinese, …)")
        tk.Label(
            right,
            text="Terms here are never passed through NMT\n"
                 "— their contextual meaning is preserved exactly.",
            bg=_BG, fg="#6c7086", font=("Helvetica", 8), justify="left",
        ).pack(anchor="w")
        self._glossary = scrolledtext.ScrolledText(
            right, width=30, height=5,
            bg=_ENTRY_BG, fg=_FG, insertbackground=_FG,
            relief="flat", font=("Helvetica", 10),
        )
        self._glossary.pack(fill="both", expand=True)
        self._glossary.insert("end", "HIPAA=HIPAA\nLiteLLM=LiteLLM")

    def _build_user_message(self, parent: tk.Frame) -> None:
        tk.Label(
            parent, text="User Message", bg=_BG, fg=_ACCENT,
            font=("Helvetica", 10, "bold"),
        ).pack(anchor="w")
        self._user_msg = scrolledtext.ScrolledText(
            parent, width=80, height=3,
            bg=_ENTRY_BG, fg=_FG, insertbackground=_FG,
            relief="flat", font=("Helvetica", 10),
        )
        self._user_msg.pack(fill="x")
        self._user_msg.insert("end", "What is the capital of France?")

        btn_frame = tk.Frame(parent, bg=_BG)
        btn_frame.pack(anchor="e", pady=(6, 0))
        tk.Button(
            btn_frame,
            text="  ▶  Run  ",
            command=self._on_run,
            bg=_BTN_BG, fg=_BTN_FG,
            font=("Helvetica", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=12, pady=4,
        ).pack(side="left", padx=4)
        tk.Button(
            btn_frame,
            text="  🗑  Clear Graph  ",
            command=self._clear_graph,
            bg=_ENTRY_BG, fg=_FG,
            font=("Helvetica", 10),
            relief="flat", cursor="hand2",
            padx=8, pady=4,
        ).pack(side="left")

    def _build_response_area(self, parent: tk.Frame) -> None:
        tk.Label(
            parent, text="Response", bg=_BG, fg=_ACCENT,
            font=("Helvetica", 10, "bold"),
        ).pack(anchor="w")
        self._response_box = scrolledtext.ScrolledText(
            parent, width=80, height=6,
            bg=_ENTRY_BG, fg=_GREEN, insertbackground=_FG,
            relief="flat", font=("Helvetica", 10),
            state="disabled",
        )
        self._response_box.pack(fill="both", expand=True)

    def _build_graph_area(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="Token Usage  —  English tokens · Chinese tokens (actual used) · Saved tokens",
            bg=_BG, fg=_ACCENT, font=("Helvetica", 10, "bold"),
        ).pack(anchor="w")

        self._fig, self._ax = plt.subplots(figsize=(9, 2.8))
        self._fig.patch.set_facecolor(_BG)
        self._ax.set_facecolor(_ENTRY_BG)
        self._ax.tick_params(colors=_FG)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#45475a")
        self._ax.set_title("Run the optimizer to populate the graph",
                            color="#6c7086", fontsize=9)

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
        # Pre-fill API key from env if available
        if cfg.is_configured() and not self._api_key_var.get():
            self._api_key_var.set(cfg.api_key_from_env() or "")

    def _on_run(self) -> None:
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

        self._set_status("Running…")
        self._set_response("")

        thread = threading.Thread(
            target=self._run_completion,
            args=(litellm_model, api_key, system_prompt, user_message, glossary),
            daemon=True,
        )
        thread.start()

    def _run_completion(
        self,
        litellm_model: str,
        api_key: Optional[str],
        system_prompt: str,
        user_message: str,
        glossary: Dict[str, str],
    ) -> None:
        try:
            optimizer = ChinesePromptOptimizer(
                model=litellm_model,
                api_key=api_key,
                glossary=glossary if glossary else None,
            )
            result = optimizer.complete(
                system_prompt=system_prompt,
                user_message=user_message,
                return_savings=True,
            )
            self._root.after(0, self._on_success, result)
        except (ValueError, RuntimeError, ConnectionError, OSError) as exc:
            self._root.after(0, self._on_error, str(exc))
        except Exception as exc:  # noqa: BLE001 – surface unexpected errors to GUI
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
            f"Done — saved {saved} tokens ({pct}%) on this run  "
            f"(English: {savings.get('english_tokens', '?')}  "
            f"Chinese: {savings.get('chinese_tokens', '?')})"
        )

    def _on_error(self, message: str) -> None:
        self._set_status(f"Error: {message}")
        messagebox.showerror("API Error", message)

    def _clear_graph(self) -> None:
        self._savings_history.clear()
        self._history_labels.clear()
        self._ax.clear()
        self._ax.set_facecolor(_ENTRY_BG)
        self._ax.tick_params(colors=_FG)
        self._ax.set_title("Graph cleared — run the optimizer to repopulate",
                            color="#6c7086", fontsize=9)
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Graph rendering
    # ------------------------------------------------------------------

    def _update_graph(self) -> None:
        reports = self._savings_history
        labels = self._history_labels
        if not reports:
            return

        n = len(reports)
        x = list(range(1, n + 1))

        en_tokens = [int(r.get("english_tokens", 0)) for r in reports]
        zh_tokens = [int(r.get("chinese_tokens", 0)) for r in reports]
        saved = [int(r.get("tokens_saved", 0)) for r in reports]

        ax = self._ax
        ax.clear()
        ax.set_facecolor(_ENTRY_BG)
        ax.tick_params(colors=_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        ax.plot(x, en_tokens, color=_ACCENT, marker="o", linewidth=2,
                markersize=6, label="English tokens")
        ax.plot(x, zh_tokens, color=_GREEN, marker="o", linewidth=2,
                markersize=6, label="Chinese tokens (actual used)")
        ax.plot(x, saved, color=_RED, marker="o", linestyle="--", linewidth=2,
                markersize=6, label="Saved tokens")

        # Shade savings area
        ax.fill_between(x, zh_tokens, en_tokens,
                        alpha=0.15, color=_GREEN)

        # Annotate each point
        for xi, (en, zh, sv) in zip(x, zip(en_tokens, zh_tokens, saved)):
            ax.annotate(str(en), (xi, en), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=_ACCENT)
            ax.annotate(str(zh), (xi, zh), xytext=(0, -12),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=_GREEN)
            ax.annotate(f"−{sv}", (xi, sv), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=_RED)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("Token count", color=_FG, fontsize=8)
        ax.yaxis.label.set_color(_FG)
        ax.set_title("Token usage per run", color=_FG, fontsize=9)
        ax.legend(fontsize=8, facecolor=_BG, edgecolor="#45475a",
                  labelcolor=_FG)
        ax.grid(True, alpha=0.2, color="#45475a")

        self._fig.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provider_by_name(self, name: str) -> Optional[ProviderConfig]:
        for p in PROVIDER_REGISTRY.values():
            if p.name == name:
                return p
        return None

    def _parse_glossary(self) -> Dict[str, str]:
        """Parse ``term=Chinese`` lines from the glossary text box."""
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
    root = tk.Tk()
    root.geometry("960x780")
    OptimizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
