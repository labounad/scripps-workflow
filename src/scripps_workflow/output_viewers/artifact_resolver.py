"""Resolve workflow pointers/manifests/direct paths into XYZ text.

This is the deliberately small abstraction layer between process-node outputs
and standalone viewer bundles.  It understands the stable ``wf.pointer.v1`` /
``wf.result.v1`` contracts and a few common conformer artifact bucket names, but
it does **not** talk to the workflow GUI, ``step_updater.py``, or browser-only
inport services.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

POINTER_SCHEMA = "wf.pointer.v1"
RESULT_SCHEMA = "wf.result.v1"

ENSEMBLE_DIRECT_BUCKETS: tuple[str, ...] = (
    "xyz_ensemble",
)
ENSEMBLE_RECORD_BUCKETS: tuple[str, ...] = (
    "accepted",
    "selected",
    "conformers",
)
GEOMETRY_BUCKETS: tuple[str, ...] = (
    "xyz",
    "accepted",
    "selected",
    "conformers",
)


class ArtifactResolutionError(RuntimeError):
    """Raised when a pointer/path cannot be resolved to usable XYZ text."""


@dataclass(frozen=True)
class ResolvedXyz:
    """Resolved XYZ payload for viewer bundle generation."""

    xyz_text: str
    source_path: str | None
    label: str
    manifest_path: str | None = None
    smiles: str | None = None
    n_frames: int | None = None
    selected_index: int | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Basic parsing helpers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _maybe_json_object(text: str) -> dict[str, Any] | None:
    s = text.strip()
    if not s or not s.startswith("{"):
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _path_from_source(source: str | Path) -> Path | None:
    s = str(source).strip()
    if not s:
        return None
    # Avoid treating JSON blobs as paths.
    if s.startswith("{"):
        return None
    p = Path(s).expanduser()
    return p.resolve() if p.exists() else None


def load_manifest_from_source(source: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a manifest from pointer JSON, manifest JSON, or a path to either.

    Args:
        source: Raw string passed from the GUI node, a pointer JSON line, a path
            to ``manifest.json``, or a path to a file containing one of those.

    Returns:
        ``(manifest_dict, manifest_path)``.
    """

    raw = str(source).strip()
    if not raw:
        raise ArtifactResolutionError("source is empty")

    obj = _maybe_json_object(raw)
    source_file = _path_from_source(raw)

    if obj is None and source_file is not None:
        text = _read_text(source_file)
        obj = _maybe_json_object(text)
        if obj is None:
            raise ArtifactResolutionError(
                f"{source_file} exists but does not contain pointer/manifest JSON"
            )

    if obj is None:
        raise ArtifactResolutionError("source is neither JSON nor an existing path")

    schema = obj.get("schema")
    if schema == POINTER_SCHEMA:
        manifest_raw = obj.get("manifest_path")
        if not isinstance(manifest_raw, str) or not manifest_raw.strip():
            raise ArtifactResolutionError("wf.pointer.v1 is missing manifest_path")
        manifest_path = Path(manifest_raw).expanduser().resolve()
        if not manifest_path.is_file():
            raise ArtifactResolutionError(f"manifest_path does not exist: {manifest_path}")
        manifest = json.loads(_read_text(manifest_path))
        if not isinstance(manifest, dict):
            raise ArtifactResolutionError(f"manifest is not a JSON object: {manifest_path}")
        if manifest.get("schema") != RESULT_SCHEMA:
            raise ArtifactResolutionError(
                f"manifest schema must be {RESULT_SCHEMA!r}, got {manifest.get('schema')!r}"
            )
        return manifest, manifest_path

    if schema == RESULT_SCHEMA:
        if source_file is not None:
            manifest_path = source_file
        else:
            manifest_path = Path(obj.get("cwd", ".")).resolve() / "outputs" / "manifest.json"
        return obj, manifest_path

    raise ArtifactResolutionError(
        f"JSON source is not a workflow pointer/result manifest (schema={schema!r})"
    )


# ---------------------------------------------------------------------------
# XYZ block helpers
# ---------------------------------------------------------------------------


def split_multixyz(text: str) -> list[str]:
    """Split XYZ/multi-XYZ text into normalized self-contained frame blocks."""

    frames: list[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        try:
            n_atoms = int(lines[i].strip())
        except ValueError:
            break
        if n_atoms <= 0:
            break
        end = i + 2 + n_atoms
        if end > len(lines):
            break
        block_lines = lines[i:end]
        if len(block_lines) == 2 + n_atoms:
            frames.append("\n".join(block_lines) + "\n")
        i = end
    return frames


def ensure_xyz_text(path: Path) -> str:
    if not path.is_file():
        raise ArtifactResolutionError(f"XYZ path does not exist or is not a file: {path}")
    text = _read_text(path)
    frames = split_multixyz(text)
    if not frames:
        raise ArtifactResolutionError(f"file does not look like XYZ/multi-XYZ: {path}")
    return "".join(frames)


def select_frame(text: str, conformer_index: str | int | None) -> tuple[str, int | None, int]:
    """Select one frame from XYZ text.

    ``conformer_index`` is 1-based. ``None``/``best``/``first`` select the
    first frame. Returns ``(frame_text, selected_index, n_frames)``.
    """

    frames = split_multixyz(text)
    if not frames:
        raise ArtifactResolutionError("no XYZ frames found")
    token = "best" if conformer_index is None else str(conformer_index).strip().lower()
    if token in {"", "best", "first", "auto", "none", "null"}:
        idx = 1
    else:
        try:
            idx = int(token)
        except ValueError as e:
            raise ArtifactResolutionError(
                f"conformer_index must be an integer, 'best', or empty; got {conformer_index!r}"
            ) from e
    if idx < 1 or idx > len(frames):
        raise ArtifactResolutionError(
            f"conformer_index {idx} out of range for {len(frames)} frame(s)"
        )
    return frames[idx - 1], idx, len(frames)


# ---------------------------------------------------------------------------
# Manifest artifact helpers
# ---------------------------------------------------------------------------


def _artifact_items(artifacts: Mapping[str, Any], bucket: str) -> list[dict[str, Any]]:
    raw = artifacts.get(bucket)
    if isinstance(raw, list):
        items: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                items.append(dict(item))
            elif isinstance(item, str):
                items.append({"path_abs": item})
        return items
    if isinstance(raw, Mapping):
        # Directory/dict buckets such as array are not a list of artifacts; only
        # accept them if they explicitly carry a path-like field.
        if any(k in raw for k in ("path_abs", "path", "path_rel")):
            return [dict(raw)]
    if isinstance(raw, str):
        return [{"path_abs": raw}]
    return []


def _artifact_path(item: Mapping[str, Any], *, manifest: Mapping[str, Any], manifest_path: Path) -> Path | None:
    raw = item.get("path_abs") or item.get("path") or item.get("path_rel")
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        cwd = manifest.get("cwd")
        base = Path(cwd) if isinstance(cwd, str) and cwd else manifest_path.parent
        p = base / p
    return p.resolve()


def _sort_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[int, str]:
        idx = item.get("index")
        if isinstance(idx, int):
            return (idx, str(item.get("label") or ""))
        if isinstance(idx, str) and idx.isdigit():
            return (int(idx), str(item.get("label") or ""))
        label = str(item.get("label") or item.get("path_abs") or "")
        m = re.search(r"(\d+)", label)
        if m:
            return (int(m.group(1)), label)
        return (10**9, label)

    return sorted(items, key=key)


def _usable_xyz_items(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    bucket: str,
) -> list[tuple[dict[str, Any], Path]]:
    artifacts = manifest.get("artifacts") or {}
    if not isinstance(artifacts, Mapping):
        return []
    out: list[tuple[dict[str, Any], Path]] = []
    for item in _sort_items(_artifact_items(artifacts, bucket)):
        p = _artifact_path(item, manifest=manifest, manifest_path=manifest_path)
        if p is None or not p.is_file():
            continue
        fmt = str(item.get("format") or "").lower()
        if fmt == "xyz" or p.suffix.lower() == ".xyz":
            out.append((item, p))
    return out


def _best_smiles_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    inputs = manifest.get("inputs")
    if isinstance(inputs, Mapping):
        for key in ("smiles", "SMILES", "canonical_smiles", "input_smiles"):
            value = inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def find_upstream_smiles(manifest: Mapping[str, Any], manifest_path: Path) -> str | None:
    """Best-effort SMILES discovery by walking upstream manifest pointers."""

    seen: set[Path] = set()
    current: tuple[Mapping[str, Any], Path] | None = (manifest, manifest_path)
    while current is not None:
        mf, mf_path = current
        if mf_path in seen:
            break
        seen.add(mf_path)
        smi = _best_smiles_from_manifest(mf)
        if smi:
            return smi
        upstream = mf.get("upstream")
        next_path: Path | None = None
        if isinstance(upstream, Mapping):
            raw = upstream.get("manifest_path")
            if isinstance(raw, str) and raw.strip():
                next_path = Path(raw).expanduser().resolve()
        if next_path is None or not next_path.is_file():
            break
        try:
            next_mf = json.loads(_read_text(next_path))
        except Exception:
            break
        if not isinstance(next_mf, Mapping):
            break
        current = (next_mf, next_path)
    return None


# ---------------------------------------------------------------------------
# Public resolvers
# ---------------------------------------------------------------------------


def resolve_ensemble_source(
    source: str | Path,
    *,
    conformer_index: str | int | None = None,
    smiles: str | None = None,
) -> ResolvedXyz:
    """Resolve source to multi-frame XYZ text for the ensemble viewer.

    Direct XYZ paths are accepted. Pointer/manifests prefer ``xyz_ensemble``;
    otherwise ordered ``accepted``/``selected``/``conformers`` buckets are
    concatenated. If ``conformer_index`` is supplied, a single frame is
    selected but still returned as valid XYZ text for the ensemble viewer.
    """

    direct = _path_from_source(source)
    if direct is not None and direct.is_file() and direct.suffix.lower() != ".json":
        text = ensure_xyz_text(direct)
        selected_index = None
        n_frames = len(split_multixyz(text))
        if conformer_index not in (None, "", "all", "ALL"):
            text, selected_index, n_frames = select_frame(text, conformer_index)
        return ResolvedXyz(
            xyz_text=text,
            source_path=str(direct),
            label=direct.name,
            smiles=smiles,
            n_frames=len(split_multixyz(text)),
            selected_index=selected_index,
            metadata={"source_kind": "direct_path", "original_n_frames": n_frames},
        )

    manifest, manifest_path = load_manifest_from_source(source)
    artifacts = manifest.get("artifacts") or {}
    if not isinstance(artifacts, Mapping):
        raise ArtifactResolutionError("manifest.artifacts is missing or not an object")
    resolved_smiles = smiles or find_upstream_smiles(manifest, manifest_path)

    # Prefer explicit multi-XYZ ensemble files.
    for bucket in ENSEMBLE_DIRECT_BUCKETS:
        for item, path in _usable_xyz_items(manifest, manifest_path, bucket):
            text = ensure_xyz_text(path)
            selected_index = None
            original_n = len(split_multixyz(text))
            if conformer_index not in (None, "", "all", "ALL"):
                text, selected_index, _ = select_frame(text, conformer_index)
            return ResolvedXyz(
                xyz_text=text,
                source_path=str(path),
                label=str(item.get("label") or bucket),
                manifest_path=str(manifest_path),
                smiles=resolved_smiles,
                n_frames=len(split_multixyz(text)),
                selected_index=selected_index,
                metadata={"source_kind": "artifact", "bucket": bucket, "original_n_frames": original_n},
            )

    # Fall back to ordered per-conformer records.
    for bucket in ENSEMBLE_RECORD_BUCKETS:
        paths = [p for _, p in _usable_xyz_items(manifest, manifest_path, bucket)]
        if paths:
            frames: list[str] = []
            for path in paths:
                part = ensure_xyz_text(path)
                part_frames = split_multixyz(part)
                frames.append(part_frames[0] if part_frames else part)
            text = "".join(frames)
            original_n = len(frames)
            selected_index = None
            if conformer_index not in (None, "", "all", "ALL"):
                text, selected_index, _ = select_frame(text, conformer_index)
            return ResolvedXyz(
                xyz_text=text,
                source_path=None,
                label=f"{bucket}_ensemble",
                manifest_path=str(manifest_path),
                smiles=resolved_smiles,
                n_frames=len(split_multixyz(text)),
                selected_index=selected_index,
                metadata={"source_kind": "artifact_bucket", "bucket": bucket, "original_n_frames": original_n},
            )

    # Final fallback: a single best XYZ geometry.
    for item, path in _usable_xyz_items(manifest, manifest_path, "xyz"):
        text = ensure_xyz_text(path)
        return ResolvedXyz(
            xyz_text=text,
            source_path=str(path),
            label=str(item.get("label") or "xyz"),
            manifest_path=str(manifest_path),
            smiles=resolved_smiles,
            n_frames=len(split_multixyz(text)),
            selected_index=1,
            metadata={"source_kind": "artifact", "bucket": "xyz", "note": "single geometry fallback"},
        )

    raise ArtifactResolutionError(
        "could not find conformer XYZ artifacts in buckets: "
        + ", ".join((*ENSEMBLE_DIRECT_BUCKETS, *ENSEMBLE_RECORD_BUCKETS, "xyz"))
    )


def resolve_geometry_source(
    source: str | Path,
    *,
    conformer_index: str | int | None = "best",
    smiles: str | None = None,
) -> ResolvedXyz:
    """Resolve source to a single-frame XYZ geometry."""

    direct = _path_from_source(source)
    if direct is not None and direct.is_file() and direct.suffix.lower() != ".json":
        text = ensure_xyz_text(direct)
        frame, selected_idx, n_frames = select_frame(text, conformer_index)
        return ResolvedXyz(
            xyz_text=frame,
            source_path=str(direct),
            label=direct.name,
            smiles=smiles,
            n_frames=1,
            selected_index=selected_idx,
            metadata={"source_kind": "direct_path", "original_n_frames": n_frames},
        )

    manifest, manifest_path = load_manifest_from_source(source)
    artifacts = manifest.get("artifacts") or {}
    if not isinstance(artifacts, Mapping):
        raise ArtifactResolutionError("manifest.artifacts is missing or not an object")
    resolved_smiles = smiles or find_upstream_smiles(manifest, manifest_path)

    token = "best" if conformer_index is None else str(conformer_index).strip().lower()

    # best/auto: use explicit xyz bucket first.
    if token in {"", "best", "auto", "first", "none", "null"}:
        xyz_items = _usable_xyz_items(manifest, manifest_path, "xyz")
        if xyz_items:
            # Prefer label=best when present; otherwise first by manifest order.
            best = None
            for item, path in xyz_items:
                if str(item.get("label") or "").lower() == "best":
                    best = (item, path)
                    break
            item, path = best or xyz_items[0]
            text = ensure_xyz_text(path)
            frame, selected_idx, original_n = select_frame(text, "first")
            return ResolvedXyz(
                xyz_text=frame,
                source_path=str(path),
                label=str(item.get("label") or "xyz"),
                manifest_path=str(manifest_path),
                smiles=resolved_smiles,
                n_frames=1,
                selected_index=selected_idx,
                metadata={"source_kind": "artifact", "bucket": "xyz", "original_n_frames": original_n},
            )

    # Numeric index or no xyz bucket: use the ensemble resolver then select.
    idx = conformer_index
    if token in {"", "best", "auto", "first", "none", "null"}:
        idx = "1"
    ensemble = resolve_ensemble_source(source, conformer_index=idx, smiles=resolved_smiles)
    frame, _frame_idx, original_n = select_frame(ensemble.xyz_text, "first")
    selected_idx = ensemble.selected_index or _frame_idx or 1
    return ResolvedXyz(
        xyz_text=frame,
        source_path=ensemble.source_path,
        label=f"{ensemble.label}:frame_{selected_idx}",
        manifest_path=ensemble.manifest_path,
        smiles=ensemble.smiles,
        n_frames=1,
        selected_index=selected_idx,
        metadata={"source_kind": "selected_from_ensemble", "original_n_frames": original_n, **(ensemble.metadata or {})},
    )
