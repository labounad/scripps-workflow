"""Engine pointer wire format.

The engine chains nodes by pumping ``stdout`` of one node into ``argv[1]`` of
the next. Every node MUST emit exactly one line of JSON on ``stdout``: the
``wf.pointer.v1`` pointer that tells the next node where to find the manifest
on disk. This module defines that wire format.

The pointer is intentionally minimal — three fields. The richer per-node data
lives in ``manifest.json`` on disk (see :mod:`scripps_workflow.schema`); the
pointer is just a typed pointer to it.

Wire format (single line, no trailing comments)::

    {"schema": "wf.pointer.v1", "ok": true, "manifest_path": "/abs/path/manifest.json"}

Stability notes:
    - This format is **load-bearing**. The engine and every existing node
      already depend on this exact shape. Do not change ``schema`` or rename
      fields without bumping the version and adding a transitional reader.
    - ``ok=False`` is expected when a node soft-fails: the pointer still
      points at a manifest, and downstream nodes are expected to either
      tolerate missing artifacts or short-circuit themselves.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

POINTER_SCHEMA: str = "wf.pointer.v1"


class PointerError(ValueError):
    """Raised when an incoming string is not a valid wf.pointer.v1 pointer."""


@dataclass(frozen=True)
class Pointer:
    """Typed view of a ``wf.pointer.v1`` pointer.

    Attributes:
        schema: Always ``"wf.pointer.v1"``. Validated on load.
        ok: ``True`` if the upstream node believes its run succeeded.
            ``False`` indicates a soft-fail; downstream nodes should either
            short-circuit or operate in a degraded-input mode.
        manifest_path: Absolute path to the upstream node's
            ``outputs/manifest.json``.
    """

    schema: str
    ok: bool
    manifest_path: str

    @classmethod
    def of(cls, *, ok: bool, manifest_path: str | Path) -> "Pointer":
        """Construct a pointer with the schema field set correctly."""
        return cls(
            schema=POINTER_SCHEMA,
            ok=bool(ok),
            manifest_path=str(Path(manifest_path).resolve()),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_line(self) -> str:
        """Render as a single-line JSON string (no trailing newline)."""
        # ``ensure_ascii=False`` matches existing nodes. ``separators`` removes
        # whitespace so the line stays compact (engine uses argv max length).
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def load_pointer(text: str) -> Pointer:
    """Parse a pointer string into a :class:`Pointer`.

    Raises:
        PointerError: If ``text`` is not valid JSON, not an object, or carries
            the wrong schema tag.
    """
    if not isinstance(text, str):
        raise PointerError(f"pointer must be a string, got {type(text).__name__}")

    s = text.strip()
    if not s:
        raise PointerError("pointer string is empty")

    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise PointerError(f"pointer is not valid JSON: {e}") from e

    if not isinstance(obj, Mapping):
        raise PointerError(
            f"pointer must be a JSON object, got {type(obj).__name__}"
        )

    schema = obj.get("schema")
    if schema != POINTER_SCHEMA:
        raise PointerError(
            f"pointer schema must be {POINTER_SCHEMA!r}, got {schema!r}"
        )

    if "manifest_path" not in obj:
        raise PointerError("pointer missing required field 'manifest_path'")

    manifest_path = obj["manifest_path"]
    if not isinstance(manifest_path, str) or not manifest_path:
        raise PointerError(
            f"pointer.manifest_path must be a non-empty string, got {manifest_path!r}"
        )

    # ok defaults to False if missing — fail-closed, since callers rely on
    # this flag to decide whether to abort.
    ok = bool(obj.get("ok", False))

    return Pointer(schema=schema, ok=ok, manifest_path=manifest_path)


def dump_pointer(pointer: Pointer) -> str:
    """Serialize a :class:`Pointer` back to its single-line wire form."""
    return pointer.to_json_line()
