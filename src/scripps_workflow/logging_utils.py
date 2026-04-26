"""Stderr-only logging helpers.

The engine pipes ``stdout`` of one node into ``argv[1]`` of the next, so
node ``stdout`` must contain ONLY the wf.pointer.v1 line. Everything else —
status, progress, warnings, errors — goes to ``stderr``.

This module exists so node code never has to remember ``file=sys.stderr``.
The module is named ``logging_utils`` (not ``logging``) to avoid shadowing
the stdlib :mod:`logging` package.
"""

from __future__ import annotations

import sys
from typing import Any


def log(msg: Any) -> None:
    """Write ``msg`` to stderr, flushed."""
    print(str(msg), file=sys.stderr, flush=True)


def log_info(msg: Any) -> None:
    """Write ``[INFO] msg`` to stderr."""
    log(f"[INFO] {msg}")


def log_warn(msg: Any) -> None:
    """Write ``[WARN] msg`` to stderr."""
    log(f"[WARN] {msg}")


def log_error(msg: Any) -> None:
    """Write ``[ERROR] msg`` to stderr."""
    log(f"[ERROR] {msg}")
