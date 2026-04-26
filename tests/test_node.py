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
