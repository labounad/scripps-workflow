"""Tests for wf.result.v1 manifest schema."""

from __future__ import annotations

import json

import pytest

from scripps_workflow.schema import (
    KNOWN_ARTIFACT_BUCKETS,
    RESULT_SCHEMA,
    ArtifactRecord,
    EnvironmentInfo,
    Manifest,
    UpstreamRef,
    validate_manifest_dict,
)


class TestArtifactRecord:
    def test_minimal_record_serializes_only_path(self):
        r = ArtifactRecord(path_abs="/abs/x.xyz")
        assert r.to_dict() == {"path_abs": "/abs/x.xyz"}

    def test_full_record_includes_all_fields(self):
        r = ArtifactRecord(
            path_abs="/abs/x.xyz",
            label="best",
            format="xyz",
            sha256="deadbeef",
            index=3,
            extra={"energy_kcal": 1.23},
        )
        d = r.to_dict()
        assert d["path_abs"] == "/abs/x.xyz"
        assert d["label"] == "best"
        assert d["format"] == "xyz"
        assert d["sha256"] == "deadbeef"
        assert d["index"] == 3
        assert d["energy_kcal"] == 1.23

    def test_extras_cannot_clobber_canonical_fields(self):
        # Defensive: even if a user puts ``label`` in extras, the explicit
        # field wins.
        r = ArtifactRecord(path_abs="/x", label="real", extra={"label": "evil"})
        assert r.to_dict()["label"] == "real"

    def test_from_dict_accepts_legacy_path_field(self):
        # Old smiles_to_3d manifests used `path` and `path_rel` instead of
        # path_abs. Reader should tolerate it.
        r = ArtifactRecord.from_dict({"path": "/abs/x.xyz", "label": "old"})
        assert r.path_abs == "/abs/x.xyz"
        assert r.label == "old"

    def test_from_dict_collects_unknown_fields_as_extras(self):
        r = ArtifactRecord.from_dict(
            {"path_abs": "/x", "energy_kcal": 1.0, "num_atoms": 12}
        )
        assert r.extra == {"energy_kcal": 1.0, "num_atoms": 12}

    def test_from_dict_rejects_missing_path(self):
        with pytest.raises(ValueError, match="path_abs"):
            ArtifactRecord.from_dict({"label": "x"})


class TestManifestSkeleton:
    def test_skeleton_has_all_known_buckets(self, tmp_path):
        m = Manifest.skeleton(step="my_step", cwd=tmp_path)
        for b in KNOWN_ARTIFACT_BUCKETS:
            if b == "array":
                assert m.artifacts[b] == {}
            else:
                assert m.artifacts[b] == []

    def test_skeleton_starts_ok_true(self, tmp_path):
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        assert m.ok is True
        assert m.failures == []

    def test_skeleton_records_resolved_cwd(self, tmp_path):
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        assert m.cwd == str(tmp_path.resolve())

    def test_skeleton_carries_upstream_ref(self, tmp_path):
        ref = UpstreamRef(
            pointer_schema="wf.pointer.v1",
            ok=True,
            manifest_path="/abs/upstream/manifest.json",
        )
        m = Manifest.skeleton(step="x", cwd=tmp_path, upstream=ref)
        assert m.upstream["manifest_path"] == "/abs/upstream/manifest.json"


class TestManifestArtifacts:
    def test_add_artifact_appends_to_bucket(self, tmp_path):
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        m.add_artifact("xyz", ArtifactRecord(path_abs="/abs/a.xyz"))
        m.add_artifact("xyz", ArtifactRecord(path_abs="/abs/b.xyz"))
        assert len(m.artifacts["xyz"]) == 2

    def test_add_artifact_accepts_dict(self, tmp_path):
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        m.add_artifact("xyz", {"path_abs": "/abs/a.xyz", "label": "best"})
        assert m.artifacts["xyz"][0]["label"] == "best"

    def test_add_artifact_creates_unknown_bucket(self, tmp_path):
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        m.add_artifact("custom_bucket", {"path_abs": "/x"})
        assert "custom_bucket" in m.artifacts

    def test_add_artifact_rejects_array_bucket(self, tmp_path):
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        with pytest.raises(ValueError, match="set_array_info"):
            m.add_artifact("array", {"path_abs": "/x"})

    def test_set_array_info_writes_dict(self, tmp_path):
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        m.set_array_info(tasks_root_abs=tmp_path, n_tasks=10, slurm_jobid=12345)
        assert m.artifacts["array"]["n_tasks"] == 10
        assert m.artifacts["array"]["slurm_jobid"] == 12345

    def test_add_failure_does_not_flip_ok(self, tmp_path):
        # Deliberate design: failures and ok are decoupled so the call site
        # has to make the decision visible.
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        m.add_failure("oh no")
        assert m.failures == [{"error": "oh no"}]
        assert m.ok is True


class TestManifestIO:
    def test_write_and_read_roundtrip(self, tmp_path):
        m = Manifest.skeleton(step="my_step", cwd=tmp_path)
        m.inputs = {"raw_argv": ["/path/script.py", "x=1"], "x": 1}
        m.add_artifact(
            "xyz",
            ArtifactRecord(path_abs="/abs/a.xyz", label="best", sha256="abc"),
        )
        target = tmp_path / "manifest.json"
        m.write(target)

        m2 = Manifest.read(target)
        assert m2.schema == RESULT_SCHEMA
        assert m2.step == "my_step"
        assert m2.inputs == {"raw_argv": ["/path/script.py", "x=1"], "x": 1}
        assert m2.artifacts["xyz"][0]["label"] == "best"

    def test_write_is_atomic_no_partial_file(self, tmp_path):
        m = Manifest.skeleton(step="x", cwd=tmp_path)
        target = tmp_path / "manifest.json"
        m.write(target)
        # Tmp file should not linger.
        assert not (tmp_path / "manifest.json.tmp").exists()
        assert target.exists()

    def test_legacy_v1_schema_is_tolerated(self, tmp_path):
        # smiles_to_3d in 7582 used wf.manifest.v1; reader should not blow up.
        target = tmp_path / "old_manifest.json"
        target.write_text(
            json.dumps(
                {
                    "schema": "wf.manifest.v1",
                    "ok": True,
                    "step": "smiles_to_3d",
                    "cwd": str(tmp_path),
                    "inputs": {"raw_argv": []},
                    "artifacts": {"xyz": []},
                    "failures": [],
                }
            )
        )
        m = Manifest.read(target)
        assert m.step == "smiles_to_3d"


class TestValidateManifestDict:
    def _good(self, tmp_path):
        return Manifest.skeleton(step="x", cwd=tmp_path).to_dict() | {
            "inputs": {"raw_argv": []},
        }

    def test_skeleton_with_raw_argv_is_valid(self, tmp_path):
        d = self._good(tmp_path)
        assert validate_manifest_dict(d) == []

    def test_missing_raw_argv_is_flagged(self, tmp_path):
        d = self._good(tmp_path)
        d["inputs"] = {}
        problems = validate_manifest_dict(d)
        assert any("raw_argv" in p for p in problems)

    def test_artifacts_must_be_dict(self, tmp_path):
        d = self._good(tmp_path)
        d["artifacts"] = []
        problems = validate_manifest_dict(d)
        assert any("dict-of-buckets" in p for p in problems)

    def test_artifact_buckets_must_be_lists(self, tmp_path):
        d = self._good(tmp_path)
        d["artifacts"] = {"xyz": "should-be-a-list"}
        problems = validate_manifest_dict(d)
        assert any("must be a list" in p for p in problems)

    def test_array_bucket_must_be_dict(self, tmp_path):
        d = self._good(tmp_path)
        d["artifacts"] = {"array": ["should-be-dict"]}
        problems = validate_manifest_dict(d)
        assert any("array must be a dict" in p for p in problems)

    def test_unknown_schema_is_flagged_but_not_fatal(self, tmp_path):
        d = self._good(tmp_path)
        d["schema"] = "wf.something.v3"
        problems = validate_manifest_dict(d)
        assert any("schema" in p for p in problems)


class TestEnvironmentInfo:
    def test_to_dict_omits_host_when_none(self):
        e = EnvironmentInfo(python="3.11.0", python_exe="/x", platform="Linux")
        assert "host" not in e.to_dict()

    def test_to_dict_includes_host_when_set(self):
        e = EnvironmentInfo(
            python="3.11.0", python_exe="/x", platform="Linux", host="node01"
        )
        assert e.to_dict()["host"] == "node01"
