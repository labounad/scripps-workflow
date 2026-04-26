"""Tag-node helper: emit ``KEY=VALUE\\n`` to stdout.

Tag nodes predate the pointer protocol. They are GUI/engine wiring shims
whose only job is to convert a "raw input box value" into a ``key=value``
token that the engine then concatenates onto the downstream node's argv.
Their stdout is consumed as engine-level wiring text, not as a
``wf.pointer.v1`` line, so they intentionally do **not** subclass
:class:`scripps_workflow.node.Node`.

In the legacy zoo, each per-instance script (``tag_temperature_k.py``,
``tag_calculations.py``, ...) was a copy-paste with one ``KEY = "..."``
constant edited at the top. That pattern produced a class of typo bugs
where the constant was edited as ``KEY = " max_concurrency"`` (leading
space) and downstream nodes parsed a config dict with a leading-space
key — which then quietly missed every consumer's lookup.

This module replaces that pattern with a single validated entrypoint.
The KEY is now a positional argument (typically baked into the node's
``script.sh`` wrapper via ``render_wrapper(fixed_args=("temperature_k",))``)
and is validated on every call: anything containing whitespace, empty,
or with characters outside ``[A-Za-z0-9_-]`` is rejected up-front.

Behavior:

    Args (in order, after the script name):
        argv[1]:       KEY (validated; required).
        argv[2:]:      VALUE tokens (joined with a single space).

    stdout:            ``f"{KEY}={VALUE}\\n"`` (token only, single line).
    stderr:            human-readable error / log lines.
    exit code:         0 (always — soft-fail invariant).

    On invalid KEY:    stdout emits ``tagger_error=<reason>`` so the
                       downstream node still receives a parseable token,
                       and the manifest of that downstream node will
                       record the malformed input.
    On missing VALUE:  stdout emits ``KEY=`` (empty value), matching
                       the legacy ``tag_temperature.py`` behavior.
"""

from __future__ import annotations

import re
import sys
from typing import Sequence

from .logging_utils import log_error

#: KEY must look like a Python-style identifier (with ``-`` allowed for
#: kebab-case interop with engine input names). This is the same shape
#: ``parse_kv_or_json`` accepts; matching it here means a downstream
#: ``parse_kv_or_json`` call will never reject a tag-emitted token.
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def emit(key: str, value: str) -> None:
    """Write ``KEY=VALUE\\n`` to stdout and flush.

    Separate function so tests can monkeypatch / capture stdout without
    spinning up a subprocess.
    """
    sys.stdout.write(f"{key}={value}\n")
    sys.stdout.flush()


def tag_main(argv: Sequence[str] | None = None) -> int:
    """Universal tag entrypoint.

    Args:
        argv: Override for testing. When ``None``, reads ``sys.argv``.
            Index 0 is the script/module name (ignored), index 1 is the
            KEY, and indices 2+ are the VALUE tokens.

    Returns:
        Always ``0``. Tag nodes preserve the soft-fail invariant; the
        downstream node will see ``tagger_error=<reason>`` if anything
        went wrong, and that propagates into its manifest.
    """
    args = list(sys.argv if argv is None else argv)

    # args[0] is the script/module name; we never look at it.
    if len(args) < 2:
        log_error("tag_input: missing KEY argument")
        emit("tagger_error", "missing_key")
        return 0

    raw_key = args[1]
    key = raw_key.strip()
    if not key:
        log_error(f"tag_input: empty KEY (was {raw_key!r})")
        emit("tagger_error", "empty_key")
        return 0
    if key != raw_key:
        # Whitespace around the KEY would have caused exactly the
        # leading-space typo bug we are trying to make impossible.
        # Loud failure here is intentional.
        log_error(
            f"tag_input: KEY {raw_key!r} has surrounding whitespace; "
            f"refusing to emit (would produce a key downstream consumers cannot find)"
        )
        emit("tagger_error", "whitespace_in_key")
        return 0
    if not _KEY_PATTERN.match(key):
        log_error(
            f"tag_input: invalid KEY {key!r}; must match [A-Za-z_][A-Za-z0-9_-]*"
        )
        emit("tagger_error", "invalid_key")
        return 0

    if len(args) < 3:
        # Missing value: match legacy behavior of emitting ``KEY=`` so
        # the downstream parser still sees a well-formed token.
        log_error(f"tag_input: missing value for key={key!r}")
        emit(key, "")
        return 0

    # Value preservation: join multiple value tokens with a single space.
    # The common case is a single token (e.g. ``298.15``). Multi-token
    # values are rare but show up when the engine passes a JSON-ish blob
    # (e.g. ``{"recompute_low": false}``) without quoting in the wrapper.
    value = " ".join(args[2:])
    emit(key, value)
    return 0
