"""``wf-artifact-export`` — bridge wf.pointer manifests to GUI Output files.

Most workflow nodes communicate through the engine's normal stdout contract:
exactly one ``wf.pointer.v1`` JSON object pointing at an on-disk
``manifest.json``. The workflow GUI's Layout/Output nodes are different: the
platform-generated ``step_updater.py`` expects the upstream stdout to be a
*concrete file path* that it can register/download for the iframe.

This node intentionally sits at that boundary. It consumes a normal
``wf.pointer.v1`` pointer, follows the manifest, selects one artifact file
(e.g. ``xyz`` or ``xyz_ensemble``), copies it into its own ``outputs/``
directory, writes a provenance manifest for humans/tests, and prints ONLY the
exported file path to stdout.

That means this node deliberately does **not** emit a ``wf.pointer.v1`` pointer.
Wire it only into GUI Output/Layout nodes, not into ordinary process nodes.

Typical wiring::

    wf-xtb.stdout         -> wf-artifact-export(pointer JSON, artifact_key=xyz)
    wf-artifact-export.file -> geometry_viewer.xyz_file

    wf-prism.stdout       -> wf-artifact-export(pointer JSON, artifact_key=xyz_ensemble)
    wf-artifact-export.file -> ensemble_viewer.ensemble_file
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..hashing import sha256_file
from ..parsing import parse_bool, parse_int, parse_kv_or_json
from ..pointer import PointerError, load_pointer
from ..schema import ArtifactRecord, Manifest, UpstreamRef


DEFAULT_PRIORITY: tuple[str, ...] = (
    "xyz_ensemble",
    "xyz",
    "accepted",
    "selected",
    "conformers",
    "files",
)


class ExportError(RuntimeError):
    """Raised for user/config/upstream problems that should fail this node."""


def _log(msg: str) -> None:
    print(f"[wf-artifact-export] {msg}", file=sys.stderr, flush=True)


def _artifact_items(artifacts: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    """Return a normalized list of artifact records for ``key``.

    Manifest buckets are usually lists of dicts, but older/ad-hoc nodes may
    store bare strings. Directory-like buckets such as ``array`` are ignored by
    returning an empty list.
    """
    raw = artifacts.get(key)
    if isinstance(raw, list):
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                out.append(dict(item))
            elif isinstance(item, str):
                out.append({"path_abs": item})
        return out
    if isinstance(raw, Mapping):
        # ``array`` and other structured buckets are not file artifacts.
        return []
    if isinstance(raw, str):
        return [{"path_abs": raw}]
    return []


def _item_path(item: Mapping[str, Any], *, upstream_manifest: Manifest) -> Path | None:
    raw = item.get("path_abs") or item.get("path") or item.get("path_rel")
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        # Best-effort for legacy relative paths: resolve against upstream cwd,
        # falling back to the upstream manifest directory.
        base = Path(upstream_manifest.cwd) if upstream_manifest.cwd else Path.cwd()
        p = base / p
    return p.resolve()


def _parse_priority(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_PRIORITY
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return DEFAULT_PRIORITY
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _select_artifact(
    upstream_manifest: Manifest,
    *,
    artifact_key: Any,
    artifact_index: int,
    artifact_label: str | None,
) -> tuple[str, dict[str, Any], Path]:
    artifacts = upstream_manifest.artifacts or {}
    priority = _parse_priority(artifact_key)

    checked: list[str] = []
    for key in priority:
        items = _artifact_items(artifacts, key)
        if not items:
            checked.append(f"{key}:empty")
            continue

        candidates: list[tuple[int, dict[str, Any], Path]] = []
        for i, item in enumerate(items):
            if artifact_label:
                label = str(item.get("label") or "")
                if label != artifact_label:
                    continue
            path = _item_path(item, upstream_manifest=upstream_manifest)
            if path is None:
                checked.append(f"{key}[{i}]:no_path")
                continue
            if not path.is_file():
                checked.append(f"{key}[{i}]:not_file:{path}")
                continue
            candidates.append((i, item, path))

        if not candidates:
            continue

        if artifact_label:
            i, item, path = candidates[0]
            return key, item, path

        if artifact_index < 0 or artifact_index >= len(candidates):
            raise ExportError(
                f"artifact_index {artifact_index} out of range for bucket {key!r} "
                f"with {len(candidates)} file candidate(s)"
            )
        i, item, path = candidates[artifact_index]
        return key, item, path

    raise ExportError(
        "no usable artifact file found; checked "
        + ", ".join(checked or priority)
    )


def _safe_output_name(source: Path, requested: Any) -> str:
    if requested is None or str(requested).strip().lower() in {"", "auto", "none", "null"}:
        return source.name
    name = Path(str(requested).strip()).name
    if not name or name in {".", ".."}:
        raise ExportError(f"invalid output_name={requested!r}")
    return name


def _copy_artifact(source: Path, dest: Path, *, copy_mode: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()

    mode = copy_mode.strip().lower()
    if mode in {"", "auto"}:
        mode = "copy"

    if mode == "none":
        return source
    if mode == "symlink":
        dest.symlink_to(source)
        return dest.resolve()
    if mode == "hardlink":
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
        return dest.resolve()
    if mode == "copy":
        shutil.copy2(source, dest)
        return dest.resolve()

    raise ExportError(
        f"unknown copy_mode={copy_mode!r}; expected copy, hardlink, symlink, or none"
    )


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


def _write_manifest(
    *,
    manifest_path: Path,
    started_at_unix: int,
    started_at_perf: float,
    ok: bool,
    inputs: dict[str, Any],
    upstream: UpstreamRef,
    exported_path: Path | None,
    source_path: Path | None,
    bucket: str | None,
    selected_item: Mapping[str, Any] | None,
    failures: list[dict[str, Any]],
) -> None:
    manifest = Manifest.skeleton(
        step="artifact_export",
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
        manifest.add_artifact(
            "files",
            ArtifactRecord(
                path_abs=str(exported_path.resolve()),
                label="exported_artifact",
                format=exported_path.suffix.lstrip(".") or None,
                sha256=digest,
                extra={
                    "source_path_abs": str(source_path.resolve()) if source_path else None,
                    "source_bucket": bucket,
                    "source_item": dict(selected_item or {}),
                    "stdout_contract": "file_path_for_gui_output_node",
                },
            ),
        )

        # Also mirror xyz-like outputs into the familiar buckets so humans can
        # browse this manifest without remembering the bridge-node convention.
        suffix = exported_path.suffix.lower()
        if suffix == ".xyz":
            mirror_bucket = "xyz_ensemble" if bucket == "xyz_ensemble" else "xyz"
            manifest.add_artifact(
                mirror_bucket,
                ArtifactRecord(
                    path_abs=str(exported_path.resolve()),
                    label="exported_artifact",
                    format="xyz",
                    sha256=digest,
                ),
            )

    manifest.write(manifest_path)


def _main(argv: Iterable[str]) -> int:
    argv_list = list(argv)
    started_at_unix = int(time.time())
    started_at_perf = time.perf_counter()
    cwd = Path.cwd()
    outputs_dir = cwd / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs_dir / "manifest.json"

    inputs: dict[str, Any] = {"raw_argv": argv_list}
    upstream_ref = UpstreamRef()
    failures: list[dict[str, Any]] = []
    exported_path: Path | None = None
    source_path: Path | None = None
    bucket: str | None = None
    selected_item: Mapping[str, Any] | None = None

    try:
        if len(argv_list) < 2:
            raise ExportError("missing argv[1]: upstream wf.pointer.v1 JSON")

        pointer = load_pointer(argv_list[1])
        upstream_ref = UpstreamRef(
            pointer_schema=pointer.schema,
            ok=pointer.ok,
            manifest_path=pointer.manifest_path,
        )
        if not pointer.ok:
            raise ExportError("upstream pointer has ok=false")

        upstream_manifest_path = Path(pointer.manifest_path)
        if not upstream_manifest_path.is_file():
            raise ExportError(f"upstream manifest not found: {upstream_manifest_path}")
        upstream_manifest = Manifest.read(upstream_manifest_path)
        if not upstream_manifest.ok:
            raise ExportError("upstream manifest has ok=false")

        cfg = parse_kv_or_json(argv_list[2:]) if len(argv_list) > 2 else {}
        inputs.update(cfg)

        artifact_index = parse_int(cfg.get("artifact_index"), 0)
        artifact_label_raw = cfg.get("artifact_label")
        artifact_label = (
            str(artifact_label_raw).strip()
            if artifact_label_raw is not None and str(artifact_label_raw).strip()
            else None
        )
        bucket, selected_item, source_path = _select_artifact(
            upstream_manifest,
            artifact_key=cfg.get("artifact_key"),
            artifact_index=artifact_index,
            artifact_label=artifact_label,
        )

        copy_mode = str(cfg.get("copy_mode") or "copy")
        output_name = _safe_output_name(source_path, cfg.get("output_name"))
        exported_path = _copy_artifact(
            source_path,
            outputs_dir / output_name,
            copy_mode=copy_mode,
        )

        if parse_bool(cfg.get("print_manifest_path"), False):
            _log(f"manifest will be written to {manifest_path}")
        _log(f"selected {bucket} -> {source_path}")
        _log(f"exported -> {exported_path}")

        _write_manifest(
            manifest_path=manifest_path,
            started_at_unix=started_at_unix,
            started_at_perf=started_at_perf,
            ok=True,
            inputs=inputs,
            upstream=upstream_ref,
            exported_path=exported_path,
            source_path=source_path,
            bucket=bucket,
            selected_item=selected_item,
            failures=failures,
        )

        # LOAD-BEARING: stdout is consumed by GUI output-node step_updater.py.
        # It must be exactly one concrete file path, not a wf.pointer.v1 JSON.
        sys.stdout.write(str(exported_path) + "\n")
        sys.stdout.flush()
        return 0

    except (ExportError, PointerError, Exception) as e:
        failures.append({"error": str(e)})
        _log(f"ERROR: {e}")
        try:
            _write_manifest(
                manifest_path=manifest_path,
                started_at_unix=started_at_unix,
                started_at_perf=started_at_perf,
                ok=False,
                inputs=inputs,
                upstream=upstream_ref,
                exported_path=exported_path,
                source_path=source_path,
                bucket=bucket,
                selected_item=selected_item,
                failures=failures,
            )
        except Exception as write_err:
            _log(f"manifest_write_failed: {write_err}")
        return 1


def main() -> int:
    return _main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
