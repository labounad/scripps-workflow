"""Manifest schema — the on-disk contract every node writes.

Each node call writes ``outputs/manifest.json`` (a ``wf.result.v1`` document)
to record what it ran, what it produced, and whether it succeeded. Downstream
nodes load the upstream manifest and look up artifacts by key.

Shape (chosen 2026-04-25 to match prism_pruner's manifest, which is the
cleanest in the existing zoo)::

    {
      "schema": "wf.result.v1",
      "ok": true,
      "step": "smiles_to_3d",
      "created_at_unix": 1775290816,
      "runtime_seconds": 12.34,
      "cwd": "/abs/path",
      "inputs": {
          "raw_argv": ["/...","..."],
          "<parsed_key>": <value>,
          ...
      },
      "environment": {
          "python": "3.11.14",
          "python_exe": "/gpfs/group/shenvi/envs/workflow/bin/python",
          "platform": "Linux-5.14.0-...",
          "host": "nodeb04092"
      },
      "upstream": {
          "pointer_schema": "wf.pointer.v1",
          "ok": true,
          "manifest_path": "/abs/path/upstream/outputs/manifest.json"
      },
      "artifacts": {
          "xyz": [{"label": "embedded", "path_abs": "...", "sha256": "..."}],
          "logs": [{"label": "stdout", "path_abs": "...", "sha256": "..."}],
          "files": [...],
          "accepted": [...],
          "rejected": [...],
          "xyz_ensemble": [...],
          "array": {"tasks_root_abs": "...", "n_tasks": 10}
      },
      "failures": []
    }

Key invariants enforced by :class:`scripps_workflow.node.Node`:
    * ``schema`` is always ``"wf.result.v1"`` (single value; no per-node
      variants).
    * ``inputs`` ALWAYS contains ``raw_argv`` (the verbatim ``sys.argv`` of
      the call) plus any parsed key/value pairs the node decided to surface.
    * ``environment`` is populated automatically by :class:`Node`; nodes do
      not need to fill it.
    * Artifact records always carry at least ``path_abs`` and a ``sha256``
      when the path is a file. Directory artifacts may omit ``sha256`` and
      should set it to ``None`` or leave it absent.
    * ``failures`` is a list of dicts; an empty list does not by itself imply
      ``ok=true`` (a node may legitimately succeed with non-fatal warnings,
      or fail without a structured failure record).

This module is **stdlib-only**. We use :mod:`dataclasses` rather than
:mod:`pydantic` so node code does not pick up a hard runtime dep. The GUI
round-trip layer in :mod:`scripps_workflow.gui` carries the matching
pydantic models for validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping

RESULT_SCHEMA: str = "wf.result.v1"


# -----------------------------------------------------------------
# Sub-records
# -----------------------------------------------------------------


@dataclass
class ArtifactRecord:
    """A single artifact (file or directory) produced by a node.

    Attributes:
        path_abs: Absolute filesystem path.
        label: Human-meaningful name (e.g., ``"best"``, ``"orca_stdout"``).
            Optional. Downstream code keys on the bucket name + index more
            often than on label.
        format: Media-type-ish hint (``"xyz"``, ``"out"``, ``"log"``, ...).
            Optional.
        sha256: SHA-256 hex digest of the file contents. ``None`` for
            directory artifacts.
        index: Optional integer index when the bucket carries an ordered
            sequence (e.g., conformer 1..N).
        extra: Free-form additional metadata (energies, num_atoms, etc.).
            Merged into the dict on serialization.
    """

    path_abs: str
    label: str | None = None
    format: str | None = None
    sha256: str | None = None
    index: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"path_abs": self.path_abs}
        if self.label is not None:
            d["label"] = self.label
        if self.format is not None:
            d["format"] = self.format
        if self.sha256 is not None:
            d["sha256"] = self.sha256
        if self.index is not None:
            d["index"] = self.index
        # Merge extras LAST so they cannot accidentally clobber the canonical
        # fields above.
        for k, v in self.extra.items():
            if k in d:
                continue
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ArtifactRecord":
        path_abs = d.get("path_abs") or d.get("path") or d.get("path_rel")
        if not isinstance(path_abs, str):
            raise ValueError("ArtifactRecord requires a string 'path_abs'")
        canonical = {"path_abs", "path", "path_rel", "label", "format", "sha256", "index"}
        extra = {k: v for k, v in d.items() if k not in canonical}
        return cls(
            path_abs=path_abs,
            label=d.get("label"),
            format=d.get("format"),
            sha256=d.get("sha256"),
            index=d.get("index"),
            extra=extra,
        )


@dataclass
class EnvironmentInfo:
    """Runtime environment metadata, populated by :class:`Node`."""

    python: str
    python_exe: str
    platform: str
    host: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "python": self.python,
            "python_exe": self.python_exe,
            "platform": self.platform,
        }
        if self.host is not None:
            d["host"] = self.host
        return d


@dataclass
class UpstreamRef:
    """Snapshot of the upstream pointer captured into the manifest."""

    pointer_schema: str | None = None
    ok: bool | None = None
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------


# Standard artifact bucket names. Nodes may add custom buckets, but these are
# the ones existing pipelines already key on, so framework-side helpers
# (e.g. conformer discovery) only look at these.
KNOWN_ARTIFACT_BUCKETS: tuple[str, ...] = (
    "logs",
    "xyz",
    "xyz_ensemble",
    "accepted",
    "rejected",
    "selected",
    "conformers",
    "files",
    "array",  # array nodes use a dict here, not a list
)


@dataclass
class Manifest:
    """The on-disk ``wf.result.v1`` document.

    Use :meth:`Manifest.skeleton` to build an empty one with all standard
    artifact buckets pre-allocated; then mutate ``inputs``, ``artifacts``,
    ``failures`` etc. in place before calling :meth:`write`.
    """

    schema: str
    ok: bool
    step: str
    created_at_unix: int
    runtime_seconds: float
    cwd: str
    inputs: dict[str, Any]
    environment: dict[str, Any]
    upstream: dict[str, Any]
    artifacts: dict[str, Any]
    failures: list[dict[str, Any]]

    # ---------------- construction ----------------

    @classmethod
    def skeleton(
        cls,
        *,
        step: str,
        cwd: str | Path,
        upstream: UpstreamRef | None = None,
    ) -> "Manifest":
        """Build a fresh manifest with sensible defaults and empty buckets.

        ``ok`` starts True; ``step`` is the only required identifier the node
        owns. The framework fills ``environment`` and ``runtime_seconds``
        automatically when the node finalizes.
        """
        artifacts: dict[str, Any] = {}
        for b in KNOWN_ARTIFACT_BUCKETS:
            artifacts[b] = {} if b == "array" else []

        return cls(
            schema=RESULT_SCHEMA,
            ok=True,
            step=str(step),
            created_at_unix=0,  # filled at finalize
            runtime_seconds=0.0,  # filled at finalize
            cwd=str(Path(cwd).resolve()),
            inputs={},
            environment={},
            upstream=(upstream.to_dict() if upstream else UpstreamRef().to_dict()),
            artifacts=artifacts,
            failures=[],
        )

    # ---------------- artifact helpers ----------------

    def add_artifact(self, bucket: str, record: ArtifactRecord | Mapping[str, Any]) -> None:
        """Append an artifact to the named bucket.

        Auto-creates the bucket if missing. ``record`` may be either an
        :class:`ArtifactRecord` or a dict-like (which is coerced).
        """
        if bucket == "array":
            raise ValueError(
                "use Manifest.set_array_info() for the 'array' bucket; it is "
                "a dict, not a list"
            )
        rec = (
            record
            if isinstance(record, ArtifactRecord)
            else ArtifactRecord.from_dict(record)
        )
        bucket_list = self.artifacts.setdefault(bucket, [])
        if not isinstance(bucket_list, list):
            raise TypeError(
                f"artifact bucket {bucket!r} is not a list (got {type(bucket_list).__name__})"
            )
        bucket_list.append(rec.to_dict())

    def set_array_info(self, *, tasks_root_abs: str | Path, n_tasks: int, **extra: Any) -> None:
        """Populate the ``artifacts.array`` dict for SLURM-array nodes."""
        info: dict[str, Any] = {
            "tasks_root_abs": str(Path(tasks_root_abs).resolve()),
            "n_tasks": int(n_tasks),
        }
        info.update(extra)
        self.artifacts["array"] = info

    def add_failure(self, error: str, **extra: Any) -> None:
        """Append a structured failure record. Does NOT flip ``ok`` — call
        sites should set ``self.ok = False`` explicitly so the intent is
        visible at the call site."""
        rec: dict[str, Any] = {"error": str(error)}
        rec.update(extra)
        self.failures.append(rec)

    # ---------------- I/O ----------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ok": bool(self.ok),
            "step": self.step,
            "created_at_unix": int(self.created_at_unix),
            "runtime_seconds": float(self.runtime_seconds),
            "cwd": self.cwd,
            "inputs": dict(self.inputs),
            "environment": dict(self.environment),
            "upstream": dict(self.upstream),
            "artifacts": dict(self.artifacts),
            "failures": list(self.failures),
        }

    def write(self, path: str | Path) -> Path:
        """Atomically write the manifest to ``path``.

        Writes to ``<path>.tmp`` then renames to avoid partially-written
        manifests being read by downstream pollers.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(target)
        return target

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Manifest":
        """Reconstruct a manifest from a parsed dict (lenient mode).

        Tolerates legacy ``wf.manifest.v1`` files (used by smiles_to_3d in
        7582) by silently upgrading the schema string. Other unknown fields
        are preserved on a best-effort basis but warnings are NOT raised —
        keep the reader permissive so old runs stay loadable.
        """
        schema = d.get("schema") or RESULT_SCHEMA
        return cls(
            schema=str(schema),
            ok=bool(d.get("ok", False)),
            step=str(d.get("step", "")),
            created_at_unix=int(d.get("created_at_unix", 0) or 0),
            runtime_seconds=float(d.get("runtime_seconds", 0.0) or 0.0),
            cwd=str(d.get("cwd", "")),
            inputs=dict(d.get("inputs") or {}),
            environment=dict(d.get("environment") or {}),
            upstream=dict(d.get("upstream") or {}),
            artifacts=dict(d.get("artifacts") or {}),
            failures=list(d.get("failures") or []),
        )

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        """Read a manifest from disk, applying lenient legacy upgrades."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


# -----------------------------------------------------------------
# Validation helpers (stdlib, no pydantic)
# -----------------------------------------------------------------


def validate_manifest_dict(d: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable validation problems.

    An empty list means the manifest is structurally fine. This is the
    lightweight, stdlib-only validator; the GUI module exposes a stricter
    pydantic-based check for tooling.
    """
    problems: list[str] = []
    if not isinstance(d, Mapping):
        return [f"manifest must be a dict, got {type(d).__name__}"]

    schema = d.get("schema")
    if schema not in (RESULT_SCHEMA, "wf.manifest.v1"):
        # legacy v1 still accepted, but flag for upgrade
        problems.append(f"unknown schema {schema!r}; expected {RESULT_SCHEMA!r}")

    for key in ("ok", "cwd", "inputs", "artifacts", "failures"):
        if key not in d:
            problems.append(f"missing required field {key!r}")

    inputs = d.get("inputs")
    if isinstance(inputs, MutableMapping) and "raw_argv" not in inputs:
        problems.append("inputs.raw_argv is missing; nodes must record sys.argv verbatim")

    artifacts = d.get("artifacts")
    if not isinstance(artifacts, Mapping):
        problems.append("artifacts must be a dict-of-buckets")
    else:
        for bucket, val in artifacts.items():
            if bucket == "array":
                if val and not isinstance(val, Mapping):
                    problems.append("artifacts.array must be a dict")
            else:
                if not isinstance(val, list):
                    problems.append(
                        f"artifacts.{bucket} must be a list (got {type(val).__name__})"
                    )

    failures = d.get("failures")
    if not isinstance(failures, list):
        problems.append("failures must be a list")

    return problems
