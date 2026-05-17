"""``wf-extract-conformers`` — bridge conformer manifests to GUI viewer files.

This is the opinionated, low-overhead cousin of ``wf-artifact-export``.
It consumes a normal ``wf.pointer.v1`` JSON pointer, follows the upstream
``manifest.json``, and emits a concrete XYZ file path on stdout for GUI
Layout/Output nodes.

Default behavior is deliberately what the ensemble viewer wants:

    upstream conformer node -> wf-extract-conformers -> ensemble_viewer

With no config tags, the node extracts **all conformers** as one multi-frame
XYZ file. It works with upstream nodes that publish either a canonical
``xyz_ensemble`` artifact (CREST, PRISM, ORCA array nodes) or ordered
per-conformer buckets (``accepted``, ``selected``, ``conformers``).

Optional config is intentionally small:

    conformer_index     optional 1-based conformer index. Empty / ``all``
                        keeps all conformers. ``best`` selects the upstream
                        best ``xyz`` artifact when available.
    output_name         optional filename under this call's ``outputs/``.
    copy_mode           copy | hardlink | symlink | none. Only applies when
                        an existing upstream file can be reused. New combined
                        or extracted files are always written under outputs/.

Like ``wf-artifact-export``, this node deliberately does **not** emit a
``wf.pointer.v1`` pointer. Stdout is exactly one concrete file path because
that is what the GUI-generated ``step_updater.py`` expects for Output nodes.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..hashing import sha256_file
from ..parsing import parse_kv_or_json
from ..pointer import PointerError, load_pointer
from ..schema import ArtifactRecord, Manifest, UpstreamRef
from .crest import XyzBlock, split_multixyz, write_xyz_block


CONFORMER_BUCKET_PRIORITY: tuple[str, ...] = (
    "accepted",
    "selected",
    "conformers",
)


class ExtractConformersError(RuntimeError):
    """Raised for user/config/upstream problems that should fail this node."""


@dataclass(frozen=True)
class SourceRecord:
    bucket: str
    ordinal: int
    record_index: int | None
    label: str | None
    path: Path
    item: dict[str, Any]


def _log(msg: str) -> None:
    print(f"[wf-extract-conformers] {msg}", file=sys.stderr, flush=True)


def _environment() -> dict[str, Any]:
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


def _artifact_items(artifacts: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = artifacts.get(key)
    if isinstance(raw, list):
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                out.append(dict(item))
            elif isinstance(item, str):
                out.append({"path_abs": item})
        return out
    if isinstance(raw, str):
        return [{"path_abs": raw}]
    return []


def _item_path(item: Mapping[str, Any], *, upstream_manifest: Manifest) -> Path | None:
    raw = item.get("path_abs") or item.get("path") or item.get("path_rel")
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        base = Path(upstream_manifest.cwd) if upstream_manifest.cwd else Path.cwd()
        p = base / p
    return p.resolve()


def _safe_output_name(default_name: str, requested: Any) -> str:
    if requested is None or str(requested).strip().lower() in {"", "auto", "none", "null"}:
        return Path(default_name).name
    name = Path(str(requested).strip()).name
    if not name or name in {".", ".."}:
        raise ExtractConformersError(f"invalid output_name={requested!r}")
    return name


def _copy_file(source: Path, dest: Path, *, copy_mode: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = str(copy_mode or "copy").strip().lower()
    if mode in {"", "auto"}:
        mode = "copy"

    if mode == "none":
        return source.resolve()

    if dest.exists() or dest.is_symlink():
        dest.unlink()

    if mode == "copy":
        shutil.copy2(source, dest)
        return dest.resolve()
    if mode == "hardlink":
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
        return dest.resolve()
    if mode == "symlink":
        dest.symlink_to(source)
        return dest.resolve()

    raise ExtractConformersError(
        f"unknown copy_mode={copy_mode!r}; expected copy, hardlink, symlink, or none"
    )


def _sort_key(item: dict[str, Any], fallback: int) -> tuple[int, int]:
    idx = item.get("index")
    try:
        return (0, int(idx))
    except (TypeError, ValueError):
        return (1, fallback)


def _ordered_source_records(upstream_manifest: Manifest) -> list[SourceRecord]:
    artifacts = upstream_manifest.artifacts or {}
    for bucket in CONFORMER_BUCKET_PRIORITY:
        items = _artifact_items(artifacts, bucket)
        candidates: list[tuple[tuple[int, int], dict[str, Any], Path]] = []
        for pos, item in enumerate(items, start=1):
            path = _item_path(item, upstream_manifest=upstream_manifest)
            if path is None or not path.is_file():
                continue
            if path.suffix.lower() != ".xyz":
                continue
            candidates.append((_sort_key(item, pos), item, path))

        if not candidates:
            continue

        candidates.sort(key=lambda t: t[0])
        records: list[SourceRecord] = []
        for ordinal, (_, item, path) in enumerate(candidates, start=1):
            rec_index: int | None = None
            try:
                rec_index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                pass
            label = item.get("label") if isinstance(item.get("label"), str) else None
            records.append(
                SourceRecord(
                    bucket=bucket,
                    ordinal=ordinal,
                    record_index=rec_index,
                    label=label,
                    path=path,
                    item=dict(item),
                )
            )
        return records

    return []


def _first_xyz_ensemble(upstream_manifest: Manifest) -> tuple[dict[str, Any], Path] | None:
    artifacts = upstream_manifest.artifacts or {}
    for item in _artifact_items(artifacts, "xyz_ensemble"):
        path = _item_path(item, upstream_manifest=upstream_manifest)
        if path is not None and path.is_file() and path.suffix.lower() == ".xyz":
            return dict(item), path
    return None


def _best_xyz(upstream_manifest: Manifest) -> tuple[dict[str, Any], Path] | None:
    artifacts = upstream_manifest.artifacts or {}
    xyz_items = _artifact_items(artifacts, "xyz")
    if not xyz_items:
        return None

    # Prefer an explicitly-labeled best geometry; otherwise first usable xyz.
    sorted_items = sorted(
        xyz_items,
        key=lambda item: 0 if str(item.get("label") or "").lower() == "best" else 1,
    )
    for item in sorted_items:
        path = _item_path(item, upstream_manifest=upstream_manifest)
        if path is not None and path.is_file() and path.suffix.lower() == ".xyz":
            return dict(item), path
    return None


def _conformer_index_token(raw: Any) -> str:
    if raw is None:
        return "all"
    token = str(raw).strip().lower()
    if token in {"", "auto", "none", "null", "all", "ensemble", "conformers"}:
        return "all"
    return token


def _select_source_record(records: list[SourceRecord], requested: int) -> SourceRecord:
    if requested < 1:
        raise ExtractConformersError("conformer_index must be >= 1, 'all', or 'best'")

    for rec in records:
        if rec.record_index == requested:
            return rec

    if requested <= len(records):
        return records[requested - 1]

    raise ExtractConformersError(
        f"conformer_index {requested} not available; only {len(records)} conformer(s) found"
    )


def _concat_source_records(records: Iterable[SourceRecord], out_path: Path) -> None:
    chunks: list[str] = []
    for rec in records:
        text = rec.path.read_text(encoding="utf-8", errors="replace")
        if not text.endswith("\n"):
            text += "\n"
        chunks.append(text)
    if not chunks:
        raise ExtractConformersError("no conformer xyz records to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(chunks), encoding="utf-8")


def _extract_frame_from_ensemble(
    ensemble_path: Path,
    *,
    conformer_index: int,
    out_path: Path,
) -> XyzBlock:
    blocks = split_multixyz(ensemble_path.read_text(encoding="utf-8", errors="replace"))
    if not blocks:
        raise ExtractConformersError(f"no XYZ frames parsed from {ensemble_path}")
    if conformer_index < 1 or conformer_index > len(blocks):
        raise ExtractConformersError(
            f"conformer_index {conformer_index} out of range for "
            f"{ensemble_path.name} with {len(blocks)} frame(s)"
        )
    block = blocks[conformer_index - 1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_xyz_block(out_path, block)
    return block


def _write_manifest(
    *,
    manifest_path: Path,
    started_at_perf: float,
    ok: bool,
    inputs: dict[str, Any],
    upstream: UpstreamRef,
    exported_path: Path | None,
    output_kind: str | None,
    source_kind: str | None,
    source_path: Path | None,
    source_records: list[SourceRecord],
    failures: list[dict[str, Any]],
) -> None:
    manifest = Manifest.skeleton(
        step="extract_conformers",
        cwd=Path.cwd(),
        upstream=upstream,
    )
    manifest.ok = ok
    manifest.created_at_unix = int(time.time())
    manifest.runtime_seconds = float(time.perf_counter() - started_at_perf)
    manifest.inputs.update(inputs)
    manifest.environment.update(_environment())
    manifest.failures.extend(failures)

    if exported_path is not None and exported_path.is_file():
        try:
            digest = sha256_file(exported_path)
        except Exception:
            digest = None
        base_record = ArtifactRecord(
            path_abs=str(exported_path.resolve()),
            label="extracted_conformers" if output_kind == "ensemble" else "extracted_conformer",
            format="xyz",
            sha256=digest,
            extra={
                "stdout_contract": "file_path_for_gui_output_node",
                "output_kind": output_kind,
                "source_kind": source_kind,
                "source_path_abs": str(source_path.resolve()) if source_path else None,
                "source_records": [
                    {
                        "bucket": r.bucket,
                        "ordinal": r.ordinal,
                        "index": r.record_index,
                        "label": r.label,
                        "path_abs": str(r.path.resolve()),
                    }
                    for r in source_records
                ],
            },
        )
        manifest.add_artifact("files", base_record)
        if output_kind == "ensemble":
            manifest.add_artifact(
                "xyz_ensemble",
                ArtifactRecord(
                    path_abs=str(exported_path.resolve()),
                    label="extracted_conformers",
                    format="xyz",
                    sha256=digest,
                ),
            )
        elif output_kind == "single":
            manifest.add_artifact(
                "xyz",
                ArtifactRecord(
                    path_abs=str(exported_path.resolve()),
                    label="extracted_conformer",
                    format="xyz",
                    sha256=digest,
                ),
            )

    manifest.write(manifest_path)


def _run(argv: Iterable[str]) -> int:
    argv_list = list(argv)
    started_at_perf = time.perf_counter()
    cwd = Path.cwd().resolve()
    outputs_dir = cwd / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs_dir / "manifest.json"

    inputs: dict[str, Any] = {"raw_argv": argv_list}
    upstream_ref = UpstreamRef()
    failures: list[dict[str, Any]] = []
    exported_path: Path | None = None
    output_kind: str | None = None
    source_kind: str | None = None
    source_path: Path | None = None
    selected_records: list[SourceRecord] = []

    try:
        if len(argv_list) < 2:
            raise ExtractConformersError("missing argv[1]: upstream wf.pointer.v1 JSON")

        pointer = load_pointer(argv_list[1])
        upstream_ref = UpstreamRef(
            pointer_schema=pointer.schema,
            ok=pointer.ok,
            manifest_path=pointer.manifest_path,
        )
        if not pointer.ok:
            raise ExtractConformersError("upstream pointer has ok=false")

        upstream_manifest_path = Path(pointer.manifest_path)
        if not upstream_manifest_path.is_file():
            raise ExtractConformersError(f"upstream manifest not found: {upstream_manifest_path}")
        upstream_manifest = Manifest.read(upstream_manifest_path)
        if not upstream_manifest.ok:
            raise ExtractConformersError("upstream manifest has ok=false")

        cfg = parse_kv_or_json(argv_list[2:]) if len(argv_list) > 2 else {}
        inputs.update(cfg)

        token = _conformer_index_token(cfg.get("conformer_index"))
        copy_mode = str(cfg.get("copy_mode") or "copy")

        if token == "all":
            ensemble = _first_xyz_ensemble(upstream_manifest)
            if ensemble is not None:
                item, source_path = ensemble
                source_kind = "xyz_ensemble"
                output_kind = "ensemble"
                default_name = source_path.name or "extracted_conformers.xyz"
                output_name = _safe_output_name(default_name, cfg.get("output_name"))
                exported_path = _copy_file(
                    source_path,
                    outputs_dir / output_name,
                    copy_mode=copy_mode,
                )
                inputs.update(
                    {
                        "resolved_mode": "all",
                        "resolved_source": "xyz_ensemble",
                        "n_extracted": len(
                            split_multixyz(
                                source_path.read_text(encoding="utf-8", errors="replace")
                            )
                        ),
                    }
                )
            else:
                records = _ordered_source_records(upstream_manifest)
                if not records:
                    raise ExtractConformersError(
                        "no conformers found; expected xyz_ensemble or one of "
                        f"{', '.join(CONFORMER_BUCKET_PRIORITY)}"
                    )
                selected_records = records
                source_kind = records[0].bucket
                source_path = records[0].path
                output_kind = "ensemble"
                output_name = _safe_output_name(
                    "extracted_conformers.xyz",
                    cfg.get("output_name"),
                )
                exported_path = (outputs_dir / output_name).resolve()
                _concat_source_records(records, exported_path)
                inputs.update(
                    {
                        "resolved_mode": "all",
                        "resolved_source": source_kind,
                        "n_extracted": len(records),
                    }
                )

        elif token == "best":
            best = _best_xyz(upstream_manifest)
            if best is None:
                records = _ordered_source_records(upstream_manifest)
                if not records:
                    raise ExtractConformersError("no best xyz or conformer records found")
                rec = records[0]
                selected_records = [rec]
                source_path = rec.path
                source_kind = rec.bucket
            else:
                _, source_path = best
                source_kind = "xyz"
            output_kind = "single"
            output_name = _safe_output_name("best.xyz", cfg.get("output_name"))
            exported_path = _copy_file(
                source_path,
                outputs_dir / output_name,
                copy_mode=copy_mode,
            )
            inputs.update(
                {"resolved_mode": "best", "resolved_source": source_kind, "n_extracted": 1}
            )

        else:
            try:
                requested_index = int(token)
            except ValueError as e:
                raise ExtractConformersError(
                    "conformer_index must be an integer, 'all', or 'best'"
                ) from e

            records = _ordered_source_records(upstream_manifest)
            if records:
                rec = _select_source_record(records, requested_index)
                selected_records = [rec]
                source_path = rec.path
                source_kind = rec.bucket
                output_kind = "single"
                idx_for_name = rec.record_index if rec.record_index is not None else rec.ordinal
                output_name = _safe_output_name(
                    f"conformer_{idx_for_name:04d}.xyz",
                    cfg.get("output_name"),
                )
                exported_path = _copy_file(
                    source_path,
                    outputs_dir / output_name,
                    copy_mode=copy_mode,
                )
                inputs.update(
                    {
                        "resolved_mode": "single",
                        "resolved_source": source_kind,
                        "requested_conformer_index": requested_index,
                        "selected_record_index": rec.record_index,
                        "selected_ordinal": rec.ordinal,
                        "n_extracted": 1,
                    }
                )
            else:
                ensemble = _first_xyz_ensemble(upstream_manifest)
                if ensemble is None:
                    raise ExtractConformersError(
                        "no conformers found; expected xyz_ensemble or per-conformer records"
                    )
                _, source_path = ensemble
                source_kind = "xyz_ensemble"
                output_kind = "single"
                output_name = _safe_output_name(
                    f"conformer_{requested_index:04d}.xyz",
                    cfg.get("output_name"),
                )
                exported_path = (outputs_dir / output_name).resolve()
                _extract_frame_from_ensemble(
                    source_path,
                    conformer_index=requested_index,
                    out_path=exported_path,
                )
                inputs.update(
                    {
                        "resolved_mode": "single",
                        "resolved_source": "xyz_ensemble",
                        "requested_conformer_index": requested_index,
                        "n_extracted": 1,
                    }
                )

        _log(f"exported -> {exported_path}")
        _write_manifest(
            manifest_path=manifest_path,
            started_at_perf=started_at_perf,
            ok=True,
            inputs=inputs,
            upstream=upstream_ref,
            exported_path=exported_path,
            output_kind=output_kind,
            source_kind=source_kind,
            source_path=source_path,
            source_records=selected_records,
            failures=failures,
        )

        # LOAD-BEARING: stdout is consumed by GUI output-node step_updater.py.
        # It must be exactly one concrete file path, not a wf.pointer.v1 JSON.
        sys.stdout.write(str(exported_path) + "\n")
        sys.stdout.flush()
        return 0

    except (ExtractConformersError, PointerError, Exception) as e:
        failures.append({"error": str(e)})
        _log(f"ERROR: {e}")
        try:
            _write_manifest(
                manifest_path=manifest_path,
                started_at_perf=started_at_perf,
                ok=False,
                inputs=inputs,
                upstream=upstream_ref,
                exported_path=exported_path,
                output_kind=output_kind,
                source_kind=source_kind,
                source_path=source_path,
                source_records=selected_records,
                failures=failures,
            )
        except Exception as write_err:
            _log(f"manifest_write_failed: {write_err}")
        return 1


def main() -> int:
    return _run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
