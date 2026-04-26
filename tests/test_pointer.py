"""Tests for the wf.pointer.v1 wire format.

The pointer is the engine's chain-glue, so even small breakage here cascades.
These tests pin its exact serialization, parsing, and validation behavior.
"""

from __future__ import annotations

import json

import pytest

from scripps_workflow.pointer import (
    POINTER_SCHEMA,
    Pointer,
    PointerError,
    dump_pointer,
    load_pointer,
)


class TestPointerLoad:
    def test_load_minimal_ok_pointer(self):
        text = (
            '{"schema": "wf.pointer.v1", "ok": true, '
            '"manifest_path": "/abs/path/manifest.json"}'
        )
        p = load_pointer(text)
        assert p.schema == POINTER_SCHEMA
        assert p.ok is True
        assert p.manifest_path == "/abs/path/manifest.json"

    def test_load_failed_pointer(self):
        text = json.dumps(
            {"schema": "wf.pointer.v1", "ok": False, "manifest_path": "/x"}
        )
        p = load_pointer(text)
        assert p.ok is False

    def test_load_strips_whitespace(self):
        text = (
            "  \n"
            '{"schema": "wf.pointer.v1", "ok": true, "manifest_path": "/x"}'
            "\n  "
        )
        p = load_pointer(text)
        assert p.manifest_path == "/x"

    def test_missing_ok_defaults_to_false(self):
        # Defensive default: if the field is missing, downstream should
        # treat it as "not ok" (fail-closed).
        text = '{"schema": "wf.pointer.v1", "manifest_path": "/x"}'
        p = load_pointer(text)
        assert p.ok is False

    def test_rejects_non_string(self):
        with pytest.raises(PointerError, match="must be a string"):
            load_pointer(42)  # type: ignore[arg-type]

    def test_rejects_empty_string(self):
        with pytest.raises(PointerError, match="empty"):
            load_pointer("   ")

    def test_rejects_invalid_json(self):
        with pytest.raises(PointerError, match="not valid JSON"):
            load_pointer("not json")

    def test_rejects_non_object(self):
        with pytest.raises(PointerError, match="must be a JSON object"):
            load_pointer("[1, 2, 3]")

    def test_rejects_wrong_schema(self):
        text = json.dumps({"schema": "wf.pointer.v2", "ok": True, "manifest_path": "/x"})
        with pytest.raises(PointerError, match="schema must be"):
            load_pointer(text)

    def test_rejects_missing_manifest_path(self):
        text = json.dumps({"schema": "wf.pointer.v1", "ok": True})
        with pytest.raises(PointerError, match="manifest_path"):
            load_pointer(text)

    def test_rejects_empty_manifest_path(self):
        text = json.dumps(
            {"schema": "wf.pointer.v1", "ok": True, "manifest_path": ""}
        )
        with pytest.raises(PointerError, match="non-empty"):
            load_pointer(text)


class TestPointerDump:
    def test_to_json_line_is_single_line(self):
        p = Pointer.of(ok=True, manifest_path="/abs/manifest.json")
        line = p.to_json_line()
        assert "\n" not in line
        # Should be compact (no spaces around separators).
        assert ", " not in line
        assert ": " not in line

    def test_dump_pointer_round_trip(self, tmp_path):
        # Resolve a path that actually exists so Path.resolve() is stable
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        p = Pointer.of(ok=False, manifest_path=manifest)
        dumped = dump_pointer(p)
        p2 = load_pointer(dumped)
        assert p2.schema == p.schema
        assert p2.ok == p.ok
        assert p2.manifest_path == p.manifest_path

    def test_to_dict_roundtrip(self):
        p = Pointer.of(ok=True, manifest_path="/abs/x")
        d = p.to_dict()
        assert d["schema"] == POINTER_SCHEMA
        assert d["ok"] is True
        assert d["manifest_path"] == "/abs/x"
