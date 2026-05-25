"""End-to-end tests for the Node base class.

These verify the load-bearing engine contract:
    * stdout = exactly one wf.pointer.v1 line
    * outputs/manifest.json is always written, even on failure
    * inputs.raw_argv is preserved verbatim
    * ok=False propagates from upstream and into the emitted pointer
    * fail_policy="hard" returns exit 1 on failure
    * soft-fail (default) always returns 0
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from scripps_workflow.node import Node, NodeContext
from scripps_workflow.pointer import POINTER_SCHEMA
from scripps_workflow.schema import RESULT_SCHEMA, ArtifactRecord, Manifest


# -----------------------------------------------------------------
# Test doubles
# -----------------------------------------------------------------


class NoopNode(Node):
    """Source node that just records inputs and writes one fake artifact."""

    step = "noop"
    accepts_upstream = False
    requires_upstream = False

    def run(self, ctx: NodeContext) -> None:
        ctx.set_inputs(answer=42)
        # Record a synthetic artifact for round-trip testing.
        fake = ctx.outputs_dir / "fake.txt"
        fake.write_text("hello")
        ctx.add_artifact(
            "files",
            ArtifactRecord(
                path_abs=str(fake.resolve()),
                label="fake",
                format="txt",
            ),
        )


class ExplodingNode(Node):
    """Node whose run() raises — exercises the soft-fail path."""

    step = "exploding"
    accepts_upstream = False
    requires_upstream = False

    def run(self, ctx: NodeContext) -> None:
        raise RuntimeError("kaboom")


class StructuredFailNode(Node):
    """Node that uses ctx.fail() to record a failure and flip ok=False."""

    step = "structured_fail"
    accepts_upstream = False
    requires_upstream = False

    def run(self, ctx: NodeContext) -> None:
        ctx.fail("bad_thing_happened", detail="oops")


class ChainNode(Node):
    """Node that consumes upstream pointer + manifest."""

    step = "chain_node"
    accepts_upstream = True
    requires_upstream = True

    def run(self, ctx: NodeContext) -> None:
        # The framework should have populated upstream_manifest if the
        # pointer pointed at a real file.
        assert ctx.upstream_manifest is not None
        ctx.set_input("upstream_step", ctx.upstream_manifest.step)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def _invoke_in_dir(node: Node, argv: list[str], workdir: Path) -> tuple[int, str]:
    """Invoke a node with cwd=workdir and capture stdout.

    Returns (exit_code, stdout_text).
    """
    prev_cwd = Path.cwd()
    os.chdir(workdir)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            code = node.invoke(argv)
    finally:
        os.chdir(prev_cwd)
    return code, buf.getvalue()


def _parse_pointer_line(text: str) -> dict:
    """Parse the single-line pointer the node emitted on stdout."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got {len(lines)}: {text!r}"
    return json.loads(lines[0])


# -----------------------------------------------------------------
# Tests
# -----------------------------------------------------------------


class TestSourceNodeHappyPath:
    def test_writes_manifest_and_pointer(self, tmp_path):
        code, stdout = _invoke_in_dir(NoopNode(), ["script.py"], tmp_path)
        assert code == 0

        ptr = _parse_pointer_line(stdout)
        assert ptr["schema"] == POINTER_SCHEMA
        assert ptr["ok"] is True

        manifest_path = Path(ptr["manifest_path"])
        assert manifest_path.exists()
        m = Manifest.read(manifest_path)
        assert m.schema == RESULT_SCHEMA
        assert m.step == "noop"
        assert m.ok is True

    def test_raw_argv_is_preserved_verbatim(self, tmp_path):
        argv = ["script.py", "k1=v1", "k2=v2"]
        code, stdout = _invoke_in_dir(NoopNode(), argv, tmp_path)
        assert code == 0
        m = Manifest.read(Path(_parse_pointer_line(stdout)["manifest_path"]))
        assert m.inputs["raw_argv"] == argv

    def test_environment_block_is_populated(self, tmp_path):
        _, stdout = _invoke_in_dir(NoopNode(), ["script.py"], tmp_path)
        m = Manifest.read(Path(_parse_pointer_line(stdout)["manifest_path"]))
        assert m.environment["python"]
        assert m.environment["python_exe"] == sys.executable
        assert m.environment["platform"]

    def test_runtime_seconds_is_recorded(self, tmp_path):
        _, stdout = _invoke_in_dir(NoopNode(), ["script.py"], tmp_path)
        m = Manifest.read(Path(_parse_pointer_line(stdout)["manifest_path"]))
        assert m.runtime_seconds >= 0.0

    def test_set_inputs_appears_in_manifest(self, tmp_path):
        _, stdout = _invoke_in_dir(NoopNode(), ["script.py"], tmp_path)
        m = Manifest.read(Path(_parse_pointer_line(stdout)["manifest_path"]))
        assert m.inputs["answer"] == 42


class TestSoftFail:
    def test_run_exception_does_not_raise_to_engine(self, tmp_path):
        code, stdout = _invoke_in_dir(ExplodingNode(), ["script.py"], tmp_path)
        # Soft-fail by default: exit 0 even though run() raised.
        assert code == 0

        ptr = _parse_pointer_line(stdout)
        assert ptr["ok"] is False

        m = Manifest.read(Path(ptr["manifest_path"]))
        assert m.ok is False
        assert any("kaboom" in f.get("error", "") for f in m.failures)

    def test_structured_fail_records_extras(self, tmp_path):
        _, stdout = _invoke_in_dir(StructuredFailNode(), ["script.py"], tmp_path)
        m = Manifest.read(Path(_parse_pointer_line(stdout)["manifest_path"]))
        assert m.ok is False
        assert m.failures[0]["error"] == "bad_thing_happened"
        assert m.failures[0]["detail"] == "oops"

    def test_hard_fail_policy_returns_exit_1(self, tmp_path):
        code, stdout = _invoke_in_dir(
            ExplodingNode(), ["script.py", "fail_policy=hard"], tmp_path
        )
        assert code == 1
        # ...but pointer still emitted.
        ptr = _parse_pointer_line(stdout)
        assert ptr["ok"] is False


class TestUpstreamHandling:
    def test_chain_node_loads_upstream_manifest(self, tmp_path):
        # First, run a source node to produce an upstream manifest.
        upstream_dir = tmp_path / "upstream"
        upstream_dir.mkdir()
        _, upstream_stdout = _invoke_in_dir(NoopNode(), ["script.py"], upstream_dir)
        upstream_pointer_line = upstream_stdout.strip()

        # Now invoke the chain node with that pointer.
        chain_dir = tmp_path / "chain"
        chain_dir.mkdir()
        code, stdout = _invoke_in_dir(
            ChainNode(), ["script.py", upstream_pointer_line], chain_dir
        )
        assert code == 0
        m = Manifest.read(Path(_parse_pointer_line(stdout)["manifest_path"]))
        assert m.ok is True
        assert m.inputs["upstream_step"] == "noop"
        assert m.upstream["pointer_schema"] == POINTER_SCHEMA
        assert m.upstream["ok"] is True

    def test_missing_upstream_pointer_soft_fails(self, tmp_path):
        code, stdout = _invoke_in_dir(ChainNode(), ["script.py"], tmp_path)
        assert code == 0  # soft-fail
        ptr = _parse_pointer_line(stdout)
        assert ptr["ok"] is False

    def test_bad_upstream_pointer_soft_fails(self, tmp_path):
        code, stdout = _invoke_in_dir(
            ChainNode(), ["script.py", "not a pointer"], tmp_path
        )
        assert code == 0
        ptr = _parse_pointer_line(stdout)
        assert ptr["ok"] is False
        m = Manifest.read(Path(ptr["manifest_path"]))
        assert any("bad_pointer" in f["error"] for f in m.failures)

    def test_upstream_manifest_path_missing_soft_fails(self, tmp_path):
        bad_pointer = json.dumps(
            {
                "schema": POINTER_SCHEMA,
                "ok": True,
                "manifest_path": str(tmp_path / "does_not_exist.json"),
            }
        )
        code, stdout = _invoke_in_dir(ChainNode(), ["script.py", bad_pointer], tmp_path)
        assert code == 0
        m = Manifest.read(Path(_parse_pointer_line(stdout)["manifest_path"]))
        assert m.ok is False


class TestPointerOutputContract:
    def test_only_one_line_on_stdout(self, tmp_path):
        _, stdout = _invoke_in_dir(NoopNode(), ["script.py"], tmp_path)
        non_empty = [ln for ln in stdout.splitlines() if ln.strip()]
        assert len(non_empty) == 1

    def test_pointer_manifest_path_is_absolute(self, tmp_path):
        _, stdout = _invoke_in_dir(NoopNode(), ["script.py"], tmp_path)
        ptr = _parse_pointer_line(stdout)
        assert Path(ptr["manifest_path"]).is_absolute()

    def test_pointer_ok_matches_manifest_ok(self, tmp_path):
        _, stdout = _invoke_in_dir(ExplodingNode(), ["script.py"], tmp_path)
        ptr = _parse_pointer_line(stdout)
        m = Manifest.read(Path(ptr["manifest_path"]))
        assert ptr["ok"] == m.ok


class TestInvokeFactory:
    def test_factory_returns_callable(self):
        main = NoopNode.invoke_factory()
        assert callable(main)
        assert main.__name__ == "main"


# -----------------------------------------------------------------
# Node._try_register_to_nmr_data — producer-side self-registration
# -----------------------------------------------------------------


class TestTryRegisterToNmrData:
    """The helper must never raise: env unset, nmr_data missing, or a
    registry function raising should all map to a result dict with
    ``ok=False``. On success it normalizes RegistryResult → dict and
    stashes under ``manifest.inputs.registry[stage]``."""

    def _make_ctx(self, tmp_path: Path) -> NodeContext:
        """Minimal NodeContext sufficient for the helper."""
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        manifest = Manifest.skeleton(step="test", cwd=tmp_path)
        return NodeContext(
            cwd=tmp_path,
            outputs_dir=outputs_dir,
            manifest_path=outputs_dir / "manifest.json",
            raw_argv=[],
            config={},
            upstream_pointer=None,
            upstream_manifest=None,
            manifest=manifest,
            started_at_unix=0,
            started_at_perf=0.0,
        )

    def _install_fake_registry(self, monkeypatch, fake_fn):
        """Make ``from nmr_data import registry`` resolve to our fake.

        Both ``nmr_data`` and ``nmr_data.registry`` are swapped in
        sys.modules so the helper's lazy import picks up our stub
        even when the real nmr_data package is also installed.
        """
        import sys
        import types

        fake_registry = types.SimpleNamespace(
            register_ensemble=fake_fn,
            register_dft_run=fake_fn,
            register_thermo_run=fake_fn,
            register_predicted_run=fake_fn,
        )
        fake_nmr_data = types.ModuleType("nmr_data")
        fake_nmr_data.registry = fake_registry  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "nmr_data", fake_nmr_data)
        monkeypatch.setitem(sys.modules, "nmr_data.registry", fake_registry)

    def test_unknown_stage_raises(self, tmp_path):
        """Wrong stage name is a programmer error — surface it loudly."""
        node = NoopNode()
        ctx = self._make_ctx(tmp_path)
        with pytest.raises(ValueError, match="Unknown registration stage"):
            node._try_register_to_nmr_data("bogus", ctx=ctx)

    def test_env_unset_returns_skipped(self, tmp_path, monkeypatch):
        """No NMR_DATABASE_URL → skip without raising, with note."""
        monkeypatch.delenv("NMR_DATABASE_URL", raising=False)
        node = NoopNode()
        ctx = self._make_ctx(tmp_path)
        result = node._try_register_to_nmr_data(
            "ensemble", ctx=ctx, smiles="CO", ensemble_key=None,
        )
        assert result["ok"] is False
        assert result["status"] == "skipped"
        assert "NMR_DATABASE_URL" in result["reason"]
        # Manifest stash mirrors the returned dict.
        assert ctx.manifest.inputs["registry"]["ensemble"] is result

    def test_nmr_data_import_failure_returns_failed(
        self, tmp_path, monkeypatch,
    ):
        """If nmr_data isn't importable, helper logs + returns failed."""
        import sys

        monkeypatch.setenv("NMR_DATABASE_URL", "sqlite:///dummy")
        # Sabotage the import: setting nmr_data to None in sys.modules
        # makes ``from nmr_data import registry`` raise ImportError.
        monkeypatch.setitem(sys.modules, "nmr_data", None)
        monkeypatch.delitem(sys.modules, "nmr_data.registry", raising=False)

        node = NoopNode()
        ctx = self._make_ctx(tmp_path)
        result = node._try_register_to_nmr_data(
            "dft_run", ctx=ctx,
            parent_ensemble_fingerprint="x", dft_key=None,
        )
        assert result["ok"] is False
        assert result["status"] == "failed"
        assert "import" in result["reason"]

    def test_registry_function_raising_does_not_propagate(
        self, tmp_path, monkeypatch,
    ):
        """A bug in nmr_data.registry must not abort the producing node."""
        monkeypatch.setenv("NMR_DATABASE_URL", "sqlite:///dummy")

        def boom(**kwargs):
            raise RuntimeError("boom")

        self._install_fake_registry(monkeypatch, boom)
        node = NoopNode()
        ctx = self._make_ctx(tmp_path)
        result = node._try_register_to_nmr_data(
            "thermo_run", ctx=ctx,
            parent_dft_run_fingerprint="x", thermo_key=None,
        )
        assert result["ok"] is False
        assert result["status"] == "failed"
        assert "RuntimeError" in result["reason"]
        assert "boom" in result["reason"]

    def test_success_normalizes_result_and_stashes(
        self, tmp_path, monkeypatch,
    ):
        """Happy path: RegistryResult → dict with all fields, in manifest."""
        import uuid as _uuid

        monkeypatch.setenv("NMR_DATABASE_URL", "sqlite:///dummy")

        # The helper accepts anything with the RegistryResult attribute
        # shape — duck-typed normalization.
        class _FakeResult:
            ok = True
            status = "created"
            row_id = _uuid.UUID("00000000-0000-0000-0000-000000000001")
            fingerprint = "abc123"
            central_tree_path = "ABCDEF/ensembles/xyz"
            n_files_copied = 3
            notes: list[str] = []
            extra: dict = {"is_new_molecule": True}

        def ok_call(**kwargs):
            return _FakeResult()

        self._install_fake_registry(monkeypatch, ok_call)
        node = NoopNode()
        ctx = self._make_ctx(tmp_path)
        result = node._try_register_to_nmr_data(
            "ensemble", ctx=ctx, smiles="CO", ensemble_key=None,
        )
        assert result["ok"] is True
        assert result["status"] == "created"
        assert result["row_id"] == "00000000-0000-0000-0000-000000000001"
        assert result["fingerprint"] == "abc123"
        assert result["central_tree_path"] == "ABCDEF/ensembles/xyz"
        assert result["n_files_copied"] == 3
        assert result["extra"] == {"is_new_molecule": True}
        # Manifest stash uses the same dict instance.
        assert ctx.manifest.inputs["registry"]["ensemble"] is result

    def test_multiple_stages_coexist_in_manifest(
        self, tmp_path, monkeypatch,
    ):
        """Two calls on different stages produce two keys under registry."""
        monkeypatch.setenv("NMR_DATABASE_URL", "sqlite:///dummy")

        class _FakeResult:
            ok = True
            status = "reused"
            row_id = None
            fingerprint = None
            central_tree_path = None
            n_files_copied = 0
            notes: list[str] = []
            extra: dict = {}

        self._install_fake_registry(monkeypatch, lambda **k: _FakeResult())
        node = NoopNode()
        ctx = self._make_ctx(tmp_path)
        node._try_register_to_nmr_data(
            "ensemble", ctx=ctx, smiles="CO", ensemble_key=None,
        )
        node._try_register_to_nmr_data(
            "dft_run", ctx=ctx,
            parent_ensemble_fingerprint="x", dft_key=None,
        )
        bucket = ctx.manifest.inputs["registry"]
        assert set(bucket.keys()) == {"ensemble", "dft_run"}
