"""Tests for the ``conformer_screen`` role contract validator.

The validator's purpose is to catch contract violations at the runtime
boundary so prism-screen and marc-screen can be swapped at the role
level without downstream surprises. Tests cover:

    * A canonical happy-path manifest passes cleanly.
    * Each required field, when missing, is reported.
    * Wrong types are reported with the actual + expected types.
    * Cross-field invariants: index ordering, count agreement,
      empty-accepted soft-fail requirement, ensemble label correctness.
    * Implementation-specific extensions (cluster_id from marc,
      energy_kcal from prism) coexist without false positives.
"""

from __future__ import annotations

import copy

import pytest

from scripps_workflow.contracts import (
    CONFORMER_SCREEN_VERSION,
    validate_conformer_screen,
)
from scripps_workflow.contracts.conformer_screen import (
    assert_valid_conformer_screen,
)


def _record(idx: int, *, energy_kcal: float | None = None,
            rel_kcal: float | None = None, **extra) -> dict:
    rec = {
        "index": idx,
        "path_abs": f"/tmp/conf_{idx:04d}.xyz",
        "sha256": "0" * 64,
    }
    if energy_kcal is not None:
        rec["energy_kcal"] = energy_kcal
    if rel_kcal is not None:
        rec["rel_energy_kcal"] = rel_kcal
    rec.update(extra)
    return rec


def _ensemble(label: str, name: str = "out") -> dict:
    return {
        "label": label,
        "path_abs": f"/tmp/{name}.xyz",
        "sha256": "0" * 64,
        "format": "xyz",
    }


def _good_prism_manifest(*, n_acc: int = 3, n_rej: int = 2) -> dict:
    """Canonical prism-flavored manifest. Used as a baseline for negative tests."""
    accepted = [
        _record(i + 1, energy_kcal=-100.0 + i * 0.1, rel_kcal=i * 0.1)
        for i in range(n_acc)
    ]
    rejected = [
        _record(n_acc + i + 1, energy_kcal=-99.0 + i * 0.2,
                rejected_reason="rmsd_dup")
        for i in range(n_rej)
    ]
    return {
        "schema": "wf.result.v1",
        "ok": True,
        "step": "prism_screen",
        "inputs": {
            "raw_argv": ["prism_screen", "<pointer>"],
            "method": "prism",
            "method_version": "0.3.1",
            "n_input": n_acc + n_rej,
            "n_accepted": n_acc,
            "n_rejected": n_rej,
            "best_chosen_by": "lowest_rel_energy_kcal",
        },
        "artifacts": {
            "accepted": accepted,
            "rejected": rejected,
            "xyz_ensemble": [_ensemble("accepted_ensemble", "ensemble")],
            "xyz": [_ensemble("best", "best")],
            "files": [],
        },
        "failures": [],
        "runtime_seconds": 1.23,
        "environment": {"python": "3.12.0", "python_exe": "/x", "platform": "Linux", "host": "h"},
    }


def _good_marc_manifest(*, n_acc: int = 3, n_rej: int = 1) -> dict:
    """marc-flavored manifest exercising cluster_id / cluster_distance fields."""
    accepted = [
        _record(i + 1, cluster_id=i, cluster_distance=0.0)
        for i in range(n_acc)
    ]
    rejected = [
        _record(n_acc + i + 1, cluster_id=i,
                cluster_distance=0.5 + i * 0.1,
                rejected_reason="cluster_dup")
        for i in range(n_rej)
    ]
    return {
        "schema": "wf.result.v1",
        "ok": True,
        "step": "marc_screen",
        "inputs": {
            "raw_argv": ["marc_screen", "<pointer>"],
            "method": "marc",
            "method_version": "git:abcdef0",
            "n_input": n_acc + n_rej,
            "n_accepted": n_acc,
            "n_rejected": n_rej,
            "best_chosen_by": "cluster_center",
        },
        "artifacts": {
            "accepted": accepted,
            "rejected": rejected,
            "xyz_ensemble": [_ensemble("accepted_ensemble")],
            "xyz": [_ensemble("best")],
            "files": [],
        },
        "failures": [],
        "runtime_seconds": 0.5,
        "environment": {"python": "3.11.7", "python_exe": "/x", "platform": "Linux", "host": "h"},
    }


# --------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------


class TestHappyPath:
    def test_prism_shape_passes(self):
        assert validate_conformer_screen(_good_prism_manifest()) == []

    def test_marc_shape_passes(self):
        assert validate_conformer_screen(_good_marc_manifest()) == []

    def test_assert_valid_does_not_raise(self):
        assert_valid_conformer_screen(_good_prism_manifest())

    def test_extra_fields_ignored(self):
        m = _good_prism_manifest()
        m["inputs"]["impl_specific_thing"] = "whatever"
        m["artifacts"]["accepted"][0]["metrics"] = {"foo": 1.0}
        m["artifacts"]["accepted"][0]["custom_field"] = "ok"
        assert validate_conformer_screen(m) == []

    def test_version_constant_exposed(self):
        assert CONFORMER_SCREEN_VERSION == 1

    def test_all_rejected_also_passes(self):
        # Edge: every conformer dropped. ok must be false with a failure
        # recorded; accepted bucket is empty; xyz/xyz_ensemble may be absent.
        m = _good_prism_manifest()
        m["ok"] = False
        m["artifacts"]["accepted"] = []
        del m["artifacts"]["xyz_ensemble"]
        del m["artifacts"]["xyz"]
        m["inputs"]["n_accepted"] = 0
        m["inputs"]["n_input"] = m["inputs"]["n_rejected"]
        del m["inputs"]["best_chosen_by"]
        m["failures"] = [{"error": "no_conformers_accepted"}]
        assert validate_conformer_screen(m) == []


# --------------------------------------------------------------------
# Required-field omissions
# --------------------------------------------------------------------


class TestMissingRequiredFields:
    @pytest.mark.parametrize(
        "field",
        ["method", "n_input", "n_accepted", "n_rejected"],
    )
    def test_missing_inputs_field(self, field):
        m = _good_prism_manifest()
        del m["inputs"][field]
        problems = validate_conformer_screen(m)
        assert any(field in p for p in problems), problems

    def test_missing_inputs_block_entirely(self):
        m = _good_prism_manifest()
        del m["inputs"]
        problems = validate_conformer_screen(m)
        assert any("inputs" in p for p in problems)

    def test_missing_artifacts_block_entirely(self):
        m = _good_prism_manifest()
        del m["artifacts"]
        problems = validate_conformer_screen(m)
        assert any("artifacts" in p for p in problems)

    def test_missing_accepted_bucket(self):
        m = _good_prism_manifest()
        del m["artifacts"]["accepted"]
        problems = validate_conformer_screen(m)
        assert any("accepted" in p for p in problems)

    def test_missing_rejected_bucket(self):
        m = _good_prism_manifest()
        del m["artifacts"]["rejected"]
        problems = validate_conformer_screen(m)
        assert any("rejected" in p for p in problems)

    def test_record_missing_required_field(self):
        m = _good_prism_manifest()
        del m["artifacts"]["accepted"][0]["sha256"]
        problems = validate_conformer_screen(m)
        assert any("sha256" in p and "accepted[0]" in p for p in problems)

    def test_record_missing_index(self):
        m = _good_prism_manifest()
        del m["artifacts"]["accepted"][1]["index"]
        problems = validate_conformer_screen(m)
        assert any("index" in p and "accepted[1]" in p for p in problems)


# --------------------------------------------------------------------
# Type checks
# --------------------------------------------------------------------


class TestTypeChecks:
    def test_n_input_must_be_int_not_str(self):
        m = _good_prism_manifest()
        m["inputs"]["n_input"] = "5"
        problems = validate_conformer_screen(m)
        assert any("n_input" in p and "wrong type" in p for p in problems)

    def test_bool_is_not_int(self):
        # bool is a subclass of int in Python, but ``"n_input": True``
        # is almost certainly a bug — explicitly reject.
        m = _good_prism_manifest()
        m["inputs"]["n_input"] = True
        problems = validate_conformer_screen(m)
        assert any("n_input" in p for p in problems)

    def test_method_must_be_string(self):
        m = _good_prism_manifest()
        m["inputs"]["method"] = 1
        problems = validate_conformer_screen(m)
        assert any("method" in p for p in problems)

    def test_record_must_be_object(self):
        m = _good_prism_manifest()
        m["artifacts"]["accepted"][0] = "not_a_dict"
        problems = validate_conformer_screen(m)
        assert any("must be an object" in p for p in problems)


# --------------------------------------------------------------------
# Cross-field invariants
# --------------------------------------------------------------------


class TestCrossFieldInvariants:
    def test_method_auto_must_be_resolved(self):
        m = _good_prism_manifest()
        m["inputs"]["method"] = "auto"
        problems = validate_conformer_screen(m)
        assert any("method" in p and "auto" in p for p in problems)

    def test_n_accepted_disagrees_with_bucket_length(self):
        m = _good_prism_manifest()
        m["inputs"]["n_accepted"] = 99
        problems = validate_conformer_screen(m)
        assert any("n_accepted" in p and "disagrees" in p for p in problems)

    def test_n_rejected_disagrees_with_bucket_length(self):
        m = _good_prism_manifest()
        m["inputs"]["n_rejected"] = 0
        problems = validate_conformer_screen(m)
        assert any("n_rejected" in p and "disagrees" in p for p in problems)

    def test_accepted_indices_must_be_unique(self):
        m = _good_prism_manifest()
        # Make two accepted records share an index.
        m["artifacts"]["accepted"][1]["index"] = m["artifacts"]["accepted"][0]["index"]
        m["inputs"]["n_accepted"] = 3  # keep count consistent
        problems = validate_conformer_screen(m)
        assert any("unique" in p for p in problems)

    def test_accepted_indices_must_be_sorted_ascending(self):
        m = _good_prism_manifest()
        m["artifacts"]["accepted"].reverse()
        problems = validate_conformer_screen(m)
        assert any("sorted" in p or "ascending" in p for p in problems)

    def test_index_must_be_one_based(self):
        m = _good_prism_manifest()
        m["artifacts"]["accepted"][0]["index"] = 0
        problems = validate_conformer_screen(m)
        assert any("1-based" in p for p in problems)

    def test_rel_energy_in_rejected_is_flagged(self):
        m = _good_prism_manifest()
        m["artifacts"]["rejected"][0]["rel_energy_kcal"] = 1.0
        problems = validate_conformer_screen(m)
        assert any("rel_energy_kcal" in p and "rejected" in p for p in problems)

    def test_empty_accepted_requires_ok_false(self):
        m = _good_prism_manifest()
        m["artifacts"]["accepted"] = []
        m["inputs"]["n_accepted"] = 0
        m["inputs"]["n_input"] = m["inputs"]["n_rejected"]
        # Leave ok=True deliberately.
        problems = validate_conformer_screen(m)
        assert any("ok=true" in p and "empty" in p for p in problems)

    def test_empty_accepted_requires_failure_record(self):
        m = _good_prism_manifest()
        m["ok"] = False
        m["artifacts"]["accepted"] = []
        m["inputs"]["n_accepted"] = 0
        m["inputs"]["n_input"] = m["inputs"]["n_rejected"]
        m["failures"] = []  # no structured failure
        problems = validate_conformer_screen(m)
        assert any("failures" in p for p in problems)


# --------------------------------------------------------------------
# Ensemble + best record requirements
# --------------------------------------------------------------------


class TestEnsembleRequirements:
    def test_xyz_ensemble_required_when_accepted_nonempty(self):
        m = _good_prism_manifest()
        m["artifacts"]["xyz_ensemble"] = []
        problems = validate_conformer_screen(m)
        assert any("xyz_ensemble" in p for p in problems)

    def test_xyz_best_required_when_accepted_nonempty(self):
        m = _good_prism_manifest()
        m["artifacts"]["xyz"] = []
        problems = validate_conformer_screen(m)
        assert any("xyz" in p and "best" in p for p in problems)

    def test_xyz_ensemble_label_must_be_accepted_ensemble(self):
        m = _good_prism_manifest()
        m["artifacts"]["xyz_ensemble"][0]["label"] = "wrong_label"
        problems = validate_conformer_screen(m)
        assert any("accepted_ensemble" in p for p in problems)

    def test_xyz_best_label_must_be_best(self):
        m = _good_prism_manifest()
        m["artifacts"]["xyz"][0]["label"] = "lowest_energy"  # close but wrong
        problems = validate_conformer_screen(m)
        assert any("'best'" in p for p in problems)

    def test_xyz_ensemble_only_one_record_allowed(self):
        m = _good_prism_manifest()
        m["artifacts"]["xyz_ensemble"].append(_ensemble("accepted_ensemble", "extra"))
        problems = validate_conformer_screen(m)
        assert any("xyz_ensemble" in p and "exactly one" in p for p in problems)

    def test_best_chosen_by_required_when_accepted_nonempty(self):
        m = _good_prism_manifest()
        del m["inputs"]["best_chosen_by"]
        problems = validate_conformer_screen(m)
        assert any("best_chosen_by" in p for p in problems)


# --------------------------------------------------------------------
# Rejected reason policy
# --------------------------------------------------------------------


class TestRejectedReasonPolicy:
    def test_rejected_reason_required_by_default(self):
        m = _good_prism_manifest()
        del m["artifacts"]["rejected"][0]["rejected_reason"]
        problems = validate_conformer_screen(m)
        assert any("rejected_reason" in p for p in problems)

    def test_rejected_reason_can_be_disabled(self):
        m = _good_prism_manifest()
        for r in m["artifacts"]["rejected"]:
            del r["rejected_reason"]
        problems = validate_conformer_screen(m, require_rejected_reason=False)
        # No rejected_reason problems with the flag off.
        assert all("rejected_reason" not in p for p in problems)

    def test_empty_rejected_does_not_require_reason(self):
        m = _good_prism_manifest(n_acc=3, n_rej=0)
        m["artifacts"]["rejected"] = []
        m["inputs"]["n_rejected"] = 0
        m["inputs"]["n_input"] = m["inputs"]["n_accepted"]
        # No rejected records at all → no reason required, no problems.
        assert validate_conformer_screen(m) == []


# --------------------------------------------------------------------
# Strict variant
# --------------------------------------------------------------------


class TestAssertVariant:
    def test_assert_raises_on_violation(self):
        m = _good_prism_manifest()
        del m["inputs"]["method"]
        with pytest.raises(ValueError, match="method"):
            assert_valid_conformer_screen(m)

    def test_assert_lists_all_problems(self):
        m = _good_prism_manifest()
        del m["inputs"]["method"]
        del m["artifacts"]["accepted"][0]["sha256"]
        with pytest.raises(ValueError) as exc_info:
            assert_valid_conformer_screen(m)
        msg = str(exc_info.value)
        # Both problems should appear so the user fixes them in one pass.
        assert "method" in msg
        assert "sha256" in msg


# --------------------------------------------------------------------
# Robustness: bad input shapes
# --------------------------------------------------------------------


class TestRobustness:
    def test_non_dict_manifest_returns_problem(self):
        problems = validate_conformer_screen("not_a_dict")  # type: ignore[arg-type]
        assert any("must be a dict" in p for p in problems)

    def test_validator_never_raises_on_garbage(self):
        # Pile of malformed inputs should still produce a list, not crash.
        for bad in [None, [], "x", 42, {"random": "garbage"}]:
            problems = validate_conformer_screen(bad)  # type: ignore[arg-type]
            assert isinstance(problems, list)

    def test_independent_baselines(self):
        # Sanity: the two baseline builders produce independent dicts so
        # mutations in one test don't leak into another.
        m1 = _good_prism_manifest()
        m2 = _good_prism_manifest()
        assert m1 is not m2
        m1["inputs"]["method"] = "MUTATED"
        assert m2["inputs"]["method"] == "prism"
        # And deepcopy is fine.
        m3 = copy.deepcopy(m1)
        m3["inputs"]["method"] = "AGAIN"
        assert m1["inputs"]["method"] == "MUTATED"
