"""
tests/test_gui.py
-----------------
Headless unit tests for the GUI module.

These tests verify:
1. Pure-logic helpers that need no display (_parse_glossary, _get_context_snippets,
   _update_grounding_badge via mocks, provider switching logic).
2. A smoke test that creates the full OptimizerApp under a virtual display
   (Xvfb), exercising widget construction end-to-end.
3. _run_completion flow with fully-mocked optimizer (no network needed).

The smoke test is skipped automatically when neither a real DISPLAY nor
/usr/bin/Xvfb is available.
"""

from __future__ import annotations

import os
import sys
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_xvfb() -> bool:
    return os.path.isfile("/usr/bin/Xvfb") or os.path.isfile("/usr/local/bin/Xvfb")


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


_need_display = pytest.mark.skipif(
    not (_has_display() or _has_xvfb()),
    reason="No DISPLAY and Xvfb not found",
)


# ---------------------------------------------------------------------------
# Fixture: virtual display for smoke tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def virtual_display():
    """Start Xvfb for tests that need a real Tk window."""
    if _has_display():
        yield  # already have a display
        return

    try:
        from xvfbwrapper import Xvfb  # type: ignore

        vdisplay = Xvfb(width=1280, height=960, colordepth=24)
        vdisplay.start()
        yield
        vdisplay.stop()
    except Exception:  # noqa: BLE001
        yield  # fall through; tests will be skipped by _need_display marker


# ---------------------------------------------------------------------------
# Logic-only tests (no display needed)
# ---------------------------------------------------------------------------


class TestParseGlossary:
    """Tests for OptimizerApp._parse_glossary() — no display needed."""

    def test_parses_simple_entries(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        class FakeText:
            def get(self, *_args):
                return "HIPAA=HIPAA\nLiteLLM=LiteLLM"

        app = MagicMock(spec=OptimizerApp)
        app._glossary = FakeText()
        result = OptimizerApp._parse_glossary(app)
        assert result["HIPAA"] == "HIPAA"
        assert result["LiteLLM"] == "LiteLLM"

    def test_strips_whitespace_from_keys_and_values(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        class FakeText:
            def get(self, *_args):
                return "  API  =  API  "

        app = MagicMock(spec=OptimizerApp)
        app._glossary = FakeText()
        result = OptimizerApp._parse_glossary(app)
        assert "API" in result
        assert result["API"] == "API"

    def test_empty_glossary(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        class FakeText:
            def get(self, *_args):
                return "   "

        app = MagicMock(spec=OptimizerApp)
        app._glossary = FakeText()
        result = OptimizerApp._parse_glossary(app)
        assert result == {}

    def test_line_without_equals_ignored(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        class FakeText:
            def get(self, *_args):
                return "not-a-glossary-entry\nGood=good"

        app = MagicMock(spec=OptimizerApp)
        app._glossary = FakeText()
        result = OptimizerApp._parse_glossary(app)
        assert "not-a-glossary-entry" not in result
        assert result["Good"] == "good"


class TestGetContextSnippets:
    """Tests for OptimizerApp._get_context_snippets() — no display needed."""

    def test_returns_non_empty_lines(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        class FakeText:
            def get(self, *_args):
                return "Fact A.\n\n  \nFact B."

        class AppStub:
            _context_box = FakeText()

        result = OptimizerApp._get_context_snippets(AppStub())
        assert result == ["Fact A.", "Fact B."]

    def test_empty_box_returns_empty_list(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        class FakeText:
            def get(self, *_args):
                return "   \n\n  "

        class AppStub:
            _context_box = FakeText()

        result = OptimizerApp._get_context_snippets(AppStub())
        assert result == []


class TestGroundingBadge:
    """Tests for grounding badge update logic — no display needed."""

    def test_grounded_badge_text_and_color(self):
        from chinese_prompt_optimizer.gui import OptimizerApp, _GREEN

        mock_var = MagicMock()
        mock_lbl = MagicMock()

        class AppStub:
            _grounding_var = mock_var
            _grounding_lbl = mock_lbl

        OptimizerApp._update_grounding_badge(
            AppStub(), {"grounded": True, "overlap_ratio": 0.85}
        )
        text = mock_var.set.call_args[0][0]
        assert "Grounded" in text
        assert "85%" in text
        mock_lbl.configure.assert_called_once_with(fg=_GREEN)

    def test_ungrounded_badge_uses_yellow(self):
        from chinese_prompt_optimizer.gui import OptimizerApp, _YELLOW

        mock_var = MagicMock()
        mock_lbl = MagicMock()

        class AppStub:
            _grounding_var = mock_var
            _grounding_lbl = mock_lbl

        OptimizerApp._update_grounding_badge(
            AppStub(), {"grounded": False, "overlap_ratio": 0.1}
        )
        text = mock_var.set.call_args[0][0]
        assert "Check" in text
        mock_lbl.configure.assert_called_once_with(fg=_YELLOW)


class TestOnSuccess:
    """Tests for _on_success result dispatch — no Tkinter needed."""

    def test_savings_appended_to_history(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        savings = {"english_tokens": 20, "chinese_tokens": 8,
                   "tokens_saved": 12, "saving_pct": 60.0}

        app = MagicMock(spec=OptimizerApp)
        app._savings_history = []
        app._history_labels = []
        # Delegate to real method
        OptimizerApp._on_success(app, {"response": "Paris.", "savings": savings})

        assert len(app._savings_history) == 1
        assert app._savings_history[0] == savings
        assert app._history_labels[0] == "Run 1"

    def test_grounding_badge_updated_when_present(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        app = MagicMock(spec=OptimizerApp)
        app._savings_history = []
        app._history_labels = []
        grounding = {"grounded": True, "overlap_ratio": 0.9, "warning": ""}

        OptimizerApp._on_success(
            app,
            {"response": "Paris.", "savings": {}, "grounding": grounding},
        )
        app._update_grounding_badge.assert_called_once_with(grounding)

    def test_no_grounding_badge_when_absent(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        app = MagicMock(spec=OptimizerApp)
        app._savings_history = []
        app._history_labels = []
        OptimizerApp._on_success(app, {"response": "Paris.", "savings": {}})
        app._update_grounding_badge.assert_not_called()


class TestRunCompletionThread:
    """Tests for _run_completion logic — fully mocked."""

    def test_success_calls_on_success(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        mock_result = {
            "response": "Paris.",
            "savings": {"english_tokens": 10, "chinese_tokens": 4,
                        "tokens_saved": 6, "saving_pct": 60.0},
            "grounding": {"grounded": True, "overlap_ratio": 0.8, "warning": ""},
        }

        app = MagicMock()
        app._temperature_var.get.return_value = 0.2
        app._cot_var.get.return_value = False
        app._self_reflect_var.get.return_value = False
        app._guard_var.get.return_value = True

        with patch(
            "chinese_prompt_optimizer.gui.ChinesePromptOptimizer"
        ) as MockOptimizer:
            mock_instance = MockOptimizer.return_value
            mock_instance.complete.return_value = mock_result
            OptimizerApp._run_completion(
                app,
                litellm_model="gemini/gemini-2.0-flash",
                api_key="test-key",
                system_prompt="Be helpful.",
                user_message="Hello.",
                glossary={},
                context_snippets=[],
            )

        app._root.after.assert_called_once_with(0, app._on_success, mock_result)

    def test_known_exception_calls_on_error(self):
        from chinese_prompt_optimizer.gui import OptimizerApp

        app = MagicMock()
        app._temperature_var.get.return_value = 0.2
        app._cot_var.get.return_value = False
        app._self_reflect_var.get.return_value = False
        app._guard_var.get.return_value = True

        with patch(
            "chinese_prompt_optimizer.gui.ChinesePromptOptimizer",
            side_effect=ConnectionError("network down"),
        ):
            OptimizerApp._run_completion(
                app,
                litellm_model="gemini/gemini-2.0-flash",
                api_key=None,
                system_prompt="Be helpful.",
                user_message="Hello.",
                glossary={},
                context_snippets=[],
            )

        call_args = app._root.after.call_args[0]
        assert call_args[1] == app._on_error
        assert "network down" in call_args[2]


# ---------------------------------------------------------------------------
# Smoke test: full window construction under Xvfb
# ---------------------------------------------------------------------------


@_need_display
def test_gui_launches_and_constructs(virtual_display):
    """Create a real OptimizerApp window under Xvfb and verify key widgets."""
    import tkinter as tk

    from chinese_prompt_optimizer.gui import OptimizerApp

    root = tk.Tk()
    root.geometry("1020x900")
    try:
        app = OptimizerApp(root)

        # Provider dropdown populated
        providers = app._provider_var.get()
        assert providers != ""

        # Default model set
        assert app._model_var.get() != ""

        # Anti-hallucination defaults
        assert app._cot_var.get() is False
        assert app._self_reflect_var.get() is False
        assert abs(app._temperature_var.get() - 0.2) < 0.001
        assert app._guard_var.get() is True

        # Savings history starts empty
        assert app._savings_history == []

        # System prompt has default text
        content = app._system_prompt.get("1.0", "end").strip()
        assert len(content) > 0
    finally:
        root.destroy()


@_need_display
def test_gui_provider_change_updates_model(virtual_display):
    """Switching provider must update the model dropdown."""
    import tkinter as tk

    from chinese_prompt_optimizer.gui import OptimizerApp
    from chinese_prompt_optimizer.providers import list_providers

    root = tk.Tk()
    try:
        app = OptimizerApp(root)
        providers = list_providers()
        if len(providers) >= 2:
            app._provider_var.set(providers[1].name)
            app._on_provider_change()
            assert app._model_var.get() == providers[1].default_model
    finally:
        root.destroy()


@_need_display
def test_gui_parse_glossary_round_trip(virtual_display):
    """Glossary text box -> _parse_glossary() -> dict, with real Tk widget."""
    import tkinter as tk

    from chinese_prompt_optimizer.gui import OptimizerApp

    root = tk.Tk()
    try:
        app = OptimizerApp(root)
        app._glossary.delete("1.0", "end")
        app._glossary.insert("end", "MyTerm=我的术语\nOtherTerm=其他术语")
        result = app._parse_glossary()
        assert result["MyTerm"] == "我的术语"
        assert result["OtherTerm"] == "其他术语"
    finally:
        root.destroy()


@_need_display
def test_gui_context_snippets_parsed(virtual_display):
    """Context snippets text box -> _get_context_snippets() -> list."""
    import tkinter as tk

    from chinese_prompt_optimizer.gui import OptimizerApp

    root = tk.Tk()
    try:
        app = OptimizerApp(root)
        app._context_box.delete("1.0", "end")
        app._context_box.insert("end", "Fact one.\n\nFact two.")
        result = app._get_context_snippets()
        assert result == ["Fact one.", "Fact two."]
    finally:
        root.destroy()
