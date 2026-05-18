"""``wf-extract-conformers`` — optional conformer-file extraction utility.

This node is no longer required for the standalone viewer output nodes: the
viewer nodes can read ``wf.pointer.v1`` JSON directly and build their own ZIP
bundles.  ``wf-extract-conformers`` remains useful when a workflow author wants
a concrete XYZ artifact as a first-class intermediate file.

Input contract:
    ``argv[1]`` is a ``wf.pointer.v1`` JSON line, a manifest JSON path, or a
    direct XYZ path.

Default behavior:
    extract all conformers into ``outputs/extracted_conformers.xyz`` and print
    that concrete path on stdout.

Optional key/value config:
    ``conformer_index=2``   write only conformer 2 to ``conformer_0002.xyz``
    ``conformer_index=best`` use the upstream best geometry when available
    ``output_name=name.xyz`` override the output filename

Unlike most process nodes, stdout is intentionally a concrete file path, not a
``wf.pointer.v1`` pointer.  This keeps it compatible with GUI Output/Layout
nodes that expect file paths.  The node still writes a normal manifest under
``outputs/manifest.json`` for provenance.
"""

from __future__ import annotations

import json
import platform
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from ..hashing import sha256_file
from ..schema import RESULT_SCHEMA
from ..output_viewers.artifact_resolver import (
    ArtifactResolutionError,
    resolve_ensemble_source,
    resolve_geometry_source,
)


class ExtractConformersError(RuntimeError):
    """User/config/upstream error."""


def _parse_args(argv: Sequence[str]) -> tuple[str, dict[str, str]]:
    if len(argv) < 2:
        raise ExtractConformersError(
            "Usage: wf-extract-conformers <pointer-json-or-xyz-path> "
            "[conformer_index=<n|best|all>] [output_name=<name.xyz>]"
        )
    source = argv[1]
    cfg: dict[str, str] = {}
    for token in argv[2:]:
        if "=" in token:
            k, v = token.split("=", 1)
            cfg[k.strip()] = v.strip()
    return source, cfg


def _env() -> dict[str, Any]:
    try:
        host = socket.gethostname()
    except Exception:
        host = None
    return {
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "platform": platform.platform(),
        "host": host,
    }


def _safe_name(name: str) -> str:
    n = Path(str(name).strip()).name
    if not n or n in {".", ".."}:
        raise ExtractConformersError(f"invalid output_name={name!r}")
    return n


def _write_manifest(
    *,
    manifest_path: Path,
    ok: bool,
    started: float,
    inputs: dict[str, Any],
    artifacts: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    manifest = {
        "schema": RESULT_SCHEMA,
        "ok": bool(ok),
        "step": "wf-extract-conformers",
        "created_at_unix": int(started),
        "runtime_seconds": max(0.0, time.time() - started),
        "cwd": str(Path.cwd().resolve()),
        "inputs": inputs,
        "environment": _env(),
        "upstream": {},
        "artifacts": {
            "logs": [],
            "xyz": [],
            "xyz_ensemble": [],
            "accepted": [],
            "rejected": [],
            "selected": [],
            "conformers": [],
            "files": [],
            "array": {},
            **artifacts,
        },
        "failures": failures,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    started = time.time()
    outputs = Path.cwd() / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs / "manifest.json"

    try:
        source, cfg = _parse_args(argv)
        conformer_index = (cfg.get("conformer_index") or "all").strip()
        output_name = cfg.get("output_name")

        if conformer_index.lower() in {"", "all", "none", "null", "auto"}:
            resolved = resolve_ensemble_source(source)
            out_name = _safe_name(output_name or "extracted_conformers.xyz")
            bucket = "xyz_ensemble"
            mode = "all"
        else:
            resolved = resolve_geometry_source(source, conformer_index=conformer_index)
            selected = resolved.selected_index or 1
            out_name = _safe_name(output_name or f"conformer_{selected:04d}.xyz")
            bucket = "xyz"
            mode = str(conformer_index)

        out_path = outputs / out_name
        out_path.write_text(resolved.xyz_text, encoding="utf-8")
        rec = {
            "label": "extracted_conformers" if bucket == "xyz_ensemble" else "extracted_conformer",
            "path_abs": str(out_path.resolve()),
            "sha256": sha256_file(out_path),
            "format": "xyz",
        }
        if resolved.selected_index is not None:
            rec["index"] = resolved.selected_index

        artifacts = {bucket: [rec]}
        inputs = {
            "raw_argv": argv,
            "source": source,
            "conformer_index": conformer_index,
            "resolved_mode": mode,
            "n_extracted": resolved.n_frames or 1,
            "source_path": resolved.source_path,
            "source_label": resolved.label,
            "source_manifest_path": resolved.manifest_path,
        }
        _write_manifest(
            manifest_path=manifest_path,
            ok=True,
            started=started,
            inputs=inputs,
            artifacts=artifacts,
            failures=[],
        )
        print(str(out_path.resolve()))
        return 0
    except Exception as e:
        _write_manifest(
            manifest_path=manifest_path,
            ok=False,
            started=started,
            inputs={"raw_argv": argv},
            artifacts={},
            failures=[{"error": type(e).__name__, "message": str(e)}],
        )
        print(f"[wf-extract-conformers] ERROR: {e}", file=sys.stderr)
        return 2


def main() -> int:
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
