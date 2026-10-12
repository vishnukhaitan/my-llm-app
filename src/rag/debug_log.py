"""Visual separators for DEBUG console output."""

from __future__ import annotations

import logging

_BAR = "*" * 12


def log_step(logger: logging.Logger, label: str) -> None:
    """Print a starred step marker so pipeline stages are easy to scan."""
    logger.debug("%s %s %s", _BAR, label, _BAR)
