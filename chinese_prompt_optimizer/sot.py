"""
sot.py
------
Skeleton-of-Thought (SoT) implementation for parallel response generation.

Based on the research from:
  - imagination-research/sot  (official SoT implementation)
  - SimonAytes/SoT            (token-minimising alternative)

SoT reduces LLM generation latency by splitting the answer into two stages:

Stage 1 – **Skeleton**: the LLM generates a concise outline (3–10 bullet
points, 3–5 words each) of the answer structure.

Stage 2 – **Expansion**: each bullet point is expanded independently,
either sequentially or concurrently via a thread pool.  Parallel expansion
enables up to **2.39× wall-clock speed-up** (measured across six LLM
families on diverse QA tasks in the SoT paper, Ning et al. 2023,
arXiv:2307.15337) compared with single sequential generation, by bypassing
the sequential decoding bottleneck.

All SoT prompts are issued in Simplified Chinese (matching the token-efficient
Chinese system prompt) so the token savings of the Chinese Prompt Optimizer
are preserved throughout both stages.

Note
~~~~
Because each skeleton point is expanded in a separate API call, total token
*cost* may be higher than a single-call response.  SoT is therefore most
useful for latency-sensitive applications where speed matters more than cost.
"""

from __future__ import annotations

import concurrent.futures
import re
from typing import Any, Dict, List, Optional, Tuple

import litellm

from .logging_config import get_logger

_log = get_logger("sot")

# ---------------------------------------------------------------------------
# Chinese SoT prompt templates (kept in Chinese for token efficiency)
# ---------------------------------------------------------------------------

#: System role used for the skeleton-generation call.
_SKELETON_SYSTEM_ZH: str = (
    "你是一个精确的大纲生成助手，专注于为问题生成简洁的结构化要点列表。"
)

#: User prompt template for Stage 1 — skeleton generation.
#: ``{question}`` is replaced with the actual user message.
_SKELETON_TEMPLATE_ZH: str = (
    "请为以下问题生成一个简洁的回答大纲。\n"
    "要求：3-10个要点，每个要点3-5个词，仅输出编号列表（例：1. 要点内容），不要展开详情。\n"
    "问题：{question}"
)

#: User prompt template for Stage 2 — point expansion.
#: ``{point}`` is replaced with the skeleton bullet text.
_EXPAND_TEMPLATE_ZH: str = (
    "请充分展开以下要点，提供完整、准确的解释（2-4句话）：\n{point}"
)

# Matches numbered/bulleted list items: "1. text", "1) text", "- text", "* text"
# Uses [ \t]* (horizontal whitespace only) to avoid consuming newlines between items.
_POINT_PATTERN = re.compile(r"^[ \t]*(?:\d+[.)][ \t]*|[-*•][ \t]*)(\S.*)$", re.MULTILINE)


class SkeletonOfThought:
    """Two-stage Skeleton-of-Thought pipeline for faster, structured responses.

    The pipeline mirrors the official SoT approach from
    *imagination-research/sot* and *SimonAytes/SoT*, adapted to work with
    Chinese system prompts for combined token and latency savings.  Parallel
    expansion achieves up to 2.39× speed-up (Ning et al. 2023,
    arXiv:2307.15337) across diverse QA tasks by bypassing sequential
    decoding bottlenecks.

    Parameters
    ----------
    model:
        Any model string supported by LiteLLM, e.g. ``"gpt-4o"``,
        ``"gemini/gemini-2.0-flash"``, ``"anthropic/claude-3-5-sonnet-20241022"``.
    temperature:
        Sampling temperature forwarded to LiteLLM for both stages.
    parallel:
        When *True* (default), skeleton points are expanded concurrently
        via :class:`concurrent.futures.ThreadPoolExecutor`, achieving the
        latency speed-up described in the research.  Set to *False* for
        sequential expansion (simpler; useful for debugging or rate-limited APIs).
    max_workers:
        Maximum number of worker threads for parallel expansion.  ``None``
        lets the executor choose an appropriate value based on CPU count and
        the number of points.
    api_key:
        Optional API key forwarded to LiteLLM.
    api_base:
        Optional API base URL (useful for local or self-hosted endpoints).
    **litellm_kwargs:
        Any extra keyword arguments forwarded verbatim to
        ``litellm.completion()``.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **litellm_kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.parallel = parallel
        self.max_workers = max_workers
        self.api_key = api_key
        self.api_base = api_base
        self._litellm_kwargs = litellm_kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Dict[str, Any]:
        """Run the full two-stage Skeleton-of-Thought pipeline.

        Stage 1 generates a concise skeleton outline; Stage 2 expands each
        point independently (in parallel when ``parallel=True``).

        Parameters
        ----------
        system_prompt:
            Chinese system prompt (already translated and optionally augmented
            with anti-hallucination instructions by
            :class:`~chinese_prompt_optimizer.anti_hallucination.HallucinationGuard`).
        user_message:
            The end-user's question or request.  The skeleton and expansion
            prompts wrap it in Chinese templates for token efficiency.

        Returns
        -------
        dict with keys:

        - ``response``         – assembled answer string (joined expanded points).
        - ``skeleton``         – list of parsed skeleton bullet-point strings.
        - ``expanded_points``  – list of expanded paragraph strings.
        - ``raw_skeleton``     – raw LiteLLM ``ModelResponse`` for Stage 1.
        - ``raw_expansions``   – list of raw LiteLLM ``ModelResponse`` objects
                                 for Stage 2 (one per skeleton point).
        """
        # Stage 1: generate skeleton outline
        _log.debug("SoT Stage 1: generating skeleton for message (%d chars).", len(user_message))
        skeleton_text, raw_skeleton = self._generate_skeleton(system_prompt, user_message)
        points = self.parse_skeleton(skeleton_text)
        if not points:
            _log.warning("SoT: skeleton parsing yielded no points; using raw text as single point.")
            points = [skeleton_text.strip()]
        _log.debug("SoT Stage 1: parsed %d points.", len(points))

        # Stage 2: expand each point
        _log.debug("SoT Stage 2: expanding %d points (parallel=%s).", len(points), self.parallel)
        expanded, raw_expansions = self._expand_points(system_prompt, points)

        response = "\n\n".join(expanded)
        return {
            "response": response,
            "skeleton": points,
            "expanded_points": expanded,
            "raw_skeleton": raw_skeleton,
            "raw_expansions": raw_expansions,
        }

    @staticmethod
    def parse_skeleton(text: str) -> List[str]:
        """Parse a numbered or bulleted list from *text* into individual points.

        Recognises common list formats produced by LLMs:
        ``"1. text"``, ``"1) text"``, ``"- text"``, ``"* text"``, ``"• text"``.
        Falls back to splitting on non-empty lines when no list markers are found.

        Parameters
        ----------
        text:
            Raw LLM skeleton output.

        Returns
        -------
        List of point strings with list markers stripped.
        """
        matches = _POINT_PATTERN.findall(text)
        if matches:
            return [m.strip() for m in matches if m.strip()]
        # Fallback: treat each non-empty line as a point
        return [line.strip() for line in text.splitlines() if line.strip()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_skeleton(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Tuple[str, Any]:
        """Issue the Stage-1 skeleton-generation call to the LLM.

        The skeleton system role is merged with the caller's system prompt so
        the model retains the configured persona while generating the outline.
        """
        merged_system = f"{system_prompt}\n{_SKELETON_SYSTEM_ZH}"
        skeleton_user = _SKELETON_TEMPLATE_ZH.format(question=user_message)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": merged_system},
            {"role": "user", "content": skeleton_user},
        ]
        raw = self._call_litellm(messages)
        return raw.choices[0].message.content or "", raw

    def _expand_one(
        self,
        system_prompt: str,
        point: str,
    ) -> Tuple[str, Any]:
        """Issue a single Stage-2 expansion call for one skeleton point."""
        expand_user = _EXPAND_TEMPLATE_ZH.format(point=point)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": expand_user},
        ]
        raw = self._call_litellm(messages)
        return raw.choices[0].message.content or "", raw

    def _expand_points(
        self,
        system_prompt: str,
        points: List[str],
    ) -> Tuple[List[str], List[Any]]:
        """Expand all skeleton points, optionally in parallel."""
        if self.parallel and len(points) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [
                    executor.submit(self._expand_one, system_prompt, pt)
                    for pt in points
                ]
                results: List[Tuple[str, Any]] = [f.result() for f in futures]
        else:
            results = [self._expand_one(system_prompt, pt) for pt in points]

        expanded = [text for text, _ in results]
        raws = [raw for _, raw in results]
        return expanded, raws

    def _call_litellm(self, messages: List[Dict[str, str]]) -> Any:
        kwargs: Dict[str, Any] = dict(self._litellm_kwargs)
        kwargs["temperature"] = self.temperature
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return litellm.completion(model=self.model, messages=messages, **kwargs)
