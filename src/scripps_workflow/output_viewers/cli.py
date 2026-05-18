"""Small command-line parsing helpers for output viewer shims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ViewerArgs:
    source: str
    smiles: str | None = None
    title: str | None = None
    conformer_index: str | None = None
    output_name: str | None = None


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "auto"}:
        return None
    return s


def parse_viewer_args(argv: Sequence[str], *, default_output_name: str) -> ViewerArgs:
    """Parse GUI positional args plus optional key=value CLI overrides.

    The GUI Layout node usually calls ``script.py`` with positional values for
    each declared input.  For local testing we also support key=value tokens,
    e.g. ``source=... smiles=... output_name=...``.
    """

    args = list(argv)
    if not args:
        raise SystemExit("Usage: script.py <source-pointer-json-or-xyz-path> [smiles] [title] [conformer_index] [output_name]")

    kv: dict[str, str] = {}
    positional: list[str] = []
    for token in args:
        if "=" in token and not token.strip().startswith("{"):
            k, v = token.split("=", 1)
            kv[k.strip().lower()] = v.strip()
        else:
            positional.append(token)

    source = kv.get("source") or kv.get("pointer") or (positional[0] if positional else None)
    if source is None or not str(source).strip():
        raise SystemExit("source is required")

    smiles = kv.get("smiles") or (positional[1] if len(positional) > 1 else None)
    title = kv.get("title") or (positional[2] if len(positional) > 2 else None)
    conformer_index = kv.get("conformer_index") or kv.get("index") or (positional[3] if len(positional) > 3 else None)
    output_name = kv.get("output_name") or (positional[4] if len(positional) > 4 else None)

    return ViewerArgs(
        source=str(source).strip(),
        smiles=_blank_to_none(smiles),
        title=_blank_to_none(title),
        conformer_index=_blank_to_none(conformer_index),
        output_name=_sanitize_output_name(output_name, default_output_name),
    )


def _sanitize_output_name(value: str | None, default: str) -> str:
    s = _blank_to_none(value)
    if s is None:
        return default
    name = Path(s).name
    if not name.endswith(".zip"):
        name += ".zip"
    return name
