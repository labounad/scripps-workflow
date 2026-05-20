"""Cache lookup failures should never abort compute-capable nodes.

The NMR database is an optional cache/indexing layer. In production it may be
hosted on a login node, moved to a persistent service, or intentionally absent.
These tests lock the desired behavior: if cache lookup raises, the node treats it
like a cache miss and continues into the ordinary compute path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripps_workflow.node import NodeContext
from scripps_workflow.schema import Manifest
from scripps_workflow.nodes.orca_dft_array import OrcaDftArray
from scripps_workflow.nodes.orca_goat import OrcaGoat
from scripps_workflow.nodes.orca_thermo_array import OrcaThermoArray


def _ctx(tmp_path: Path, node: Any) -> NodeContext:
    outputs_dir = tmp_path / "outputs"
    return NodeContext(
        cwd=tmp_path,
        outputs_dir=outputs_dir,
        manifest_path=outputs_dir / "manifest.json",
        raw_argv=[],
        config=node.parse_config({}),
        upstream_pointer=None,
        upstream_manifest=None,
        manifest=Manifest.skeleton(step=node.step, cwd=tmp_path),
        started_at_unix=0,
        started_at_perf=0.0,
    )


def test_orca_dft_array_cache_lookup_exception_falls_through(tmp_path, monkeypatch):
    node = OrcaDftArray()
    ctx = _ctx(tmp_path, node)

    def raise_db_down(*_args, **_kwargs):
        raise RuntimeError("database is down")

    monkeypatch.setattr(node, "_maybe_emit_cached_manifest_dft", raise_db_down)

    node.run(ctx)

    assert ctx.manifest.ok is False
    assert ctx.manifest.failures[-1]["error"] == "no_upstream_manifest"


def test_orca_thermo_array_cache_lookup_exception_falls_through(tmp_path, monkeypatch):
    node = OrcaThermoArray()
    ctx = _ctx(tmp_path, node)

    def raise_db_down(*_args, **_kwargs):
        raise RuntimeError("database is down")

    monkeypatch.setattr(node, "_maybe_emit_cached_manifest_thermo", raise_db_down)

    node.run(ctx)

    assert ctx.manifest.ok is False
    assert ctx.manifest.failures[-1]["error"] == "no_upstream_manifest"


def test_orca_goat_cache_lookup_exception_falls_through(tmp_path, monkeypatch):
    node = OrcaGoat()
    ctx = _ctx(tmp_path, node)

    def raise_db_down(*_args, **_kwargs):
        raise RuntimeError("database is down")

    monkeypatch.setattr(node, "_maybe_emit_cached_manifest_goat", raise_db_down)

    node.run(ctx)

    assert ctx.manifest.ok is False
    assert ctx.manifest.failures[-1]["error"] in {
        "orca_not_found_on_PATH",
        "no_upstream_manifest",
    }
