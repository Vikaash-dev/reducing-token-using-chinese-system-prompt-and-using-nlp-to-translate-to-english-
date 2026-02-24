"""
logging_config.py
-----------------
Centralised structured logging for the Chinese Prompt Optimizer.

All modules obtain their logger via :func:`get_logger`.  Log output goes to
both stderr (INFO+) and a daily-rotating file under
``~/.chinese_prompt_optimizer/logs/`` (DEBUG+).

Usage::

    from chinese_prompt_optimizer.logging_config import get_logger
    log = get_logger(__name__)
    log.info("System prompt translated: %d tokens", n)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_PKG = "chinese_prompt_optimizer"
_LOG_DIR = Path.home() / ".chinese_prompt_optimizer" / "logs"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def get_logger(module_name: str) -> logging.Logger:
    """Return a named child logger under the package root.

    Args:
        module_name: Short module name, e.g. ``"optimizer"`` or ``"gui"``.

    Returns:
        :class:`logging.Logger` configured under ``chinese_prompt_optimizer.*``.
    """
    return logging.getLogger(f"{_PKG}.{module_name}")


def setup_logging(
    level: int = logging.INFO,
    enable_file: bool = True,
) -> logging.Logger:
    """Configure package-wide logging.  Idempotent — safe to call multiple times.

    Args:
        level:       Root log level for the package (default ``INFO``).
        enable_file: When *True*, write a daily-rotating log file under
                     ``~/.chinese_prompt_optimizer/logs/``.

    Returns:
        The root package :class:`logging.Logger`.
    """
    root = logging.getLogger(_PKG)
    if root.handlers:
        return root  # already configured

    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    if enable_file:
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.TimedRotatingFileHandler(
                filename=_LOG_DIR / "optimizer.log",
                when="midnight",
                backupCount=7,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except OSError:
            root.warning(
                "Could not create log directory %s; file logging disabled.", _LOG_DIR
            )

    return root
