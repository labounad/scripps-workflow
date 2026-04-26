"""Argv-token parsing helpers shared across nodes.

The engine concatenates upstream stdouts as additional ``argv`` tokens, so
nodes receive a list of free-form ``key=value`` strings (or, occasionally, a
single JSON object string when the user prefers structured config). These
helpers replace the ~150 lines of duplicated boilerplate previously copied
verbatim across xtb / marc / orca / prism scripts.

All parsers are deliberately permissive at the boundary and return well-typed
defaults on bad input, since the engine cannot recover from a node that
crashes during argument parsing.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


def parse_kv_or_json(args: Iterable[str]) -> dict[str, Any]:
    """Parse a list of free-form argv tokens into a dict.

    Accepts either:
        * Multiple ``key=value`` pairs (the common case), or
        * A single string starting with ``{`` and ending with ``}`` parsed as
          a JSON object.

    Whitespace around keys and values is stripped. An empty key raises
    ``ValueError``; an empty list returns an empty dict.

    Note: this matches the existing behavior across xtb / marc / orca /
    prism nodes verbatim, except it also strips whitespace around the *key*,
    which fixes the leading-space tag-node bug
    (``KEY = " max_concurrency"`` → token ``" max_concurrency=10"`` previously
    produced a dict key with a leading space).
    """
    arg_list = list(args)
    if not arg_list:
        return {}

    # Single JSON object?
    if len(arg_list) == 1:
        s = str(arg_list[0]).strip()
        if s.startswith("{") and s.endswith("}"):
            obj = json.loads(s)
            if not isinstance(obj, Mapping):
                raise ValueError("Config JSON must be an object")
            return dict(obj)
        # JSON-array-shaped tokens are explicitly rejected with the
        # clearer "must be an object" message rather than falling through
        # to the generic "expected key=value" path. Common when a user
        # accidentally passes a list instead of a dict.
        if s.startswith("[") and s.endswith("]"):
            raise ValueError("Config JSON must be an object, not an array")

    cfg: dict[str, Any] = {}
    for tok in arg_list:
        s = str(tok)
        if "=" not in s:
            raise ValueError(
                f"Invalid config token {s!r}: expected 'key=value' or a single JSON object string"
            )
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid config token {s!r}: empty key after stripping")
        if not k.replace("_", "").replace("-", "").isalnum():
            # Not a hard error — just permissive about exotic keys — but
            # flag obviously malformed ones early.
            raise ValueError(
                f"Invalid config key {k!r} in token {s!r}: only alphanumeric, '_' and '-' allowed"
            )
        cfg[k] = v
    return cfg


# -----------------------------------------------------------------
# Scalar coercion. Lenient on whitespace and case; ``None``/``""`` →
# default. ``"auto"``/``"none"``/``"null"`` are also treated as "use default".
# -----------------------------------------------------------------


_SENTINEL_NONE = {"", "none", "null", "auto"}


def parse_int(v: Any, default: int) -> int:
    """Coerce ``v`` to int. Returns ``default`` on missing / sentinel / error."""
    if v is None:
        return default
    try:
        s = str(v).strip().lower()
        if s in _SENTINEL_NONE:
            return default
        return int(s)
    except (ValueError, TypeError):
        return default


def parse_float(v: Any, default: float) -> float:
    """Coerce ``v`` to float. Returns ``default`` on missing / sentinel / error."""
    if v is None:
        return default
    try:
        s = str(v).strip().lower()
        if s in _SENTINEL_NONE:
            return default
        return float(s)
    except (ValueError, TypeError):
        return default


def parse_bool(v: Any, default: bool) -> bool:
    """Coerce ``v`` to bool. Returns ``default`` on unrecognized input.

    True-ish: ``1, true, t, yes, y, on``.
    False-ish: ``0, false, f, no, n, off``.
    """
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def normalize_tri(v: Any, default: str = "auto") -> str:
    """Normalize a tri-state string to one of ``"auto" | "true" | "false"``.

    Useful when the node has three modes: explicit-on, explicit-off, and
    "auto-detect from upstream". prism_pruner uses this for ``use_energies``.
    """
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"auto", ""}:
        return "auto"
    if s in {"true", "1", "yes", "y", "on"}:
        return "true"
    if s in {"false", "0", "no", "n", "off"}:
        return "false"
    return default


def parse_optional_int(v: Any) -> int | None:
    """Parse an int or return ``None`` if the value is missing / sentinel."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in _SENTINEL_NONE:
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def normalize_optional_str(v: Any) -> str | None:
    """Coerce to a non-empty string or ``None``.

    Sentinel values (``"none"``, ``"null"``, ``"auto"``, ``""``) are treated
    as "no value".
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in _SENTINEL_NONE:
        return None
    return s
