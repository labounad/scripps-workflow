"""Stdlib-only validator for the ``conformer_screen`` role.

Source of truth for the contract is ``nodes/roles/conformer_screen.yaml``
in this repo. This module mirrors that spec in code so that:

    * Node implementations can self-test at the end of their ``run()``
      via ``validate_conformer_screen(manifest.to_dict())`` and append
      contract failures to the manifest.
    * Tests can pin both happy and unhappy shapes against the same code
      that the runtime uses.

The validator is deliberately permissive about *extra* fields (a
``cluster_id`` is fine even though the universal contract doesn't
mention it — the YAML lists it as an optional field). It is strict
about *missing* required fields and about type mismatches that would
break downstream consumers.

Returns a list of problem strings. Empty list = manifest satisfies the
contract. Never raises.
"""

from __future__ import annotations

from typing import Any, Iterable

#: Bumped when the contract's required fields change in a breaking way.
CONFORMER_SCREEN_VERSION: int = 1


# Fields that every conformer_record (in either accepted or rejected)
# MUST carry. ``sha256`` is required even for rejected because we want
# reproducible bookkeeping — rejected conformers are still real files.
_REQUIRED_RECORD_FIELDS: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("index", int),
    ("path_abs", str),
    ("sha256", str),
)

# Required ensemble_record fields (used for xyz_ensemble + xyz buckets).
_REQUIRED_ENSEMBLE_FIELDS: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("label", str),
    ("path_abs", str),
    ("sha256", str),
)

# Required keys in the manifest's top-level inputs block.
_REQUIRED_INPUTS_KEYS_ALWAYS: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("method", str),
    ("n_input", int),
    ("n_accepted", int),
    ("n_rejected", int),
)


def _check_field(
    container: dict[str, Any],
    name: str,
    expected_type: type | tuple[type, ...],
    where: str,
) -> str | None:
    """Return a problem string, or None if the field is present and well-typed."""
    if name not in container:
        return f"{where}: missing required field {name!r}"
    val = container[name]
    if isinstance(expected_type, tuple):
        ok = isinstance(val, expected_type)
    else:
        # bool is a subclass of int — reject bool when int is required, since
        # ``"n_input": True`` is almost certainly a bug.
        if expected_type is int and isinstance(val, bool):
            ok = False
        else:
            ok = isinstance(val, expected_type)
    if not ok:
        type_name = (
            expected_type.__name__
            if not isinstance(expected_type, tuple)
            else "/".join(t.__name__ for t in expected_type)
        )
        return (
            f"{where}: field {name!r} has wrong type "
            f"(got {type(val).__name__}, expected {type_name})"
        )
    return None


def _validate_record(
    rec: Any,
    bucket: str,
    idx: int,
    *,
    require_rejected_reason: bool,
) -> list[str]:
    """Validate one element of accepted/rejected against ``conformer_record``."""
    problems: list[str] = []
    where = f"artifacts.{bucket}[{idx}]"
    if not isinstance(rec, dict):
        return [f"{where}: must be an object, got {type(rec).__name__}"]

    for name, expected in _REQUIRED_RECORD_FIELDS:
        msg = _check_field(rec, name, expected, where)
        if msg is not None:
            problems.append(msg)

    # Index must be a positive 1-based ordinal.
    if isinstance(rec.get("index"), int) and rec["index"] < 1:
        problems.append(f"{where}: 'index' must be 1-based (>= 1), got {rec['index']}")

    # Optional but type-checked fields, when present.
    for name, expected in (
        ("label", str),
        ("format", str),
        ("energy_kcal", (int, float)),
        ("rel_energy_kcal", (int, float)),
        ("cluster_id", int),
        ("cluster_distance", (int, float)),
        ("rejected_reason", str),
        ("metrics", dict),
    ):
        if name in rec:
            msg = _check_field(rec, name, expected, where)
            if msg is not None:
                problems.append(msg)

    # rel_energy_kcal is meaningless in `rejected` — flag if present.
    if bucket == "rejected" and "rel_energy_kcal" in rec:
        problems.append(
            f"{where}: 'rel_energy_kcal' is only meaningful in 'accepted'"
        )

    # rejected_reason is required for rejected records only when keep_rejected
    # was requested AND records exist. Caller controls via require_rejected_reason.
    if bucket == "rejected" and require_rejected_reason and "rejected_reason" not in rec:
        problems.append(
            f"{where}: missing 'rejected_reason' (required when keep_rejected=true and rejected is non-empty)"
        )

    return problems


def _validate_ensemble_record(
    rec: Any,
    bucket: str,
    expected_label: str,
) -> list[str]:
    problems: list[str] = []
    where = f"artifacts.{bucket}[0]"
    if not isinstance(rec, dict):
        return [f"{where}: must be an object, got {type(rec).__name__}"]
    for name, expected in _REQUIRED_ENSEMBLE_FIELDS:
        msg = _check_field(rec, name, expected, where)
        if msg is not None:
            problems.append(msg)
    label = rec.get("label")
    if isinstance(label, str) and label != expected_label:
        problems.append(
            f"{where}: 'label' must be {expected_label!r}, got {label!r}"
        )
    return problems


def validate_conformer_screen(
    manifest: dict[str, Any],
    *,
    require_rejected_reason: bool = True,
) -> list[str]:
    """Validate a manifest against the ``conformer_screen`` role contract.

    Args:
        manifest: A dict — typically ``Manifest.to_dict()``. Must already
            satisfy the generic ``wf.result.v1`` envelope (use
            ``schema.validate_manifest_dict`` for that). This function
            checks only the role-specific layering on top.
        require_rejected_reason: When ``True`` (default), every record
            in the ``rejected`` bucket must carry a ``rejected_reason``
            field. Set ``False`` if the implementation legitimately can't
            attribute rejections to a single reason (rare).

    Returns:
        List of human-readable problem strings. Empty = contract
        satisfied. Never raises; even malformed manifests get reported
        as a list of problems.
    """
    problems: list[str] = []

    if not isinstance(manifest, dict):
        return [f"manifest must be a dict, got {type(manifest).__name__}"]

    # ---- inputs block ----
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        problems.append("inputs: missing or not an object")
        # Cannot continue checking inputs-level fields if it's missing; we
        # still continue to the artifacts checks below so the user sees
        # *all* the problems, not just the first.
    else:
        for name, expected in _REQUIRED_INPUTS_KEYS_ALWAYS:
            msg = _check_field(inputs, name, expected, "inputs")
            if msg is not None:
                problems.append(msg)

        method = inputs.get("method")
        if isinstance(method, str) and method == "auto":
            # `auto` means "wrapper picks", so the manifest must record
            # the resolved impl, never the placeholder.
            problems.append(
                "inputs.method: must be the resolved impl name, not 'auto'"
            )

        # n_accepted / n_rejected are also cross-checked against the
        # actual artifact bucket lengths below.

    # ---- artifacts.accepted ----
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        problems.append("artifacts: missing or not an object")
        return problems  # without artifacts, no further checking is meaningful

    accepted = artifacts.get("accepted")
    if not isinstance(accepted, list):
        problems.append("artifacts.accepted: missing or not a list")
        accepted = []
    for i, rec in enumerate(accepted):
        problems.extend(
            _validate_record(rec, "accepted", i, require_rejected_reason=False)
        )

    # ---- artifacts.rejected ----
    rejected = artifacts.get("rejected")
    if not isinstance(rejected, list):
        problems.append("artifacts.rejected: missing or not a list")
        rejected = []
    rejected_nonempty = bool(rejected)
    for i, rec in enumerate(rejected):
        problems.extend(
            _validate_record(
                rec,
                "rejected",
                i,
                require_rejected_reason=require_rejected_reason and rejected_nonempty,
            )
        )

    # ---- counts cross-check ----
    if isinstance(inputs, dict):
        n_acc = inputs.get("n_accepted")
        if isinstance(n_acc, int) and not isinstance(n_acc, bool):
            if n_acc != len(accepted):
                problems.append(
                    f"inputs.n_accepted={n_acc} disagrees with len(artifacts.accepted)={len(accepted)}"
                )
        n_rej = inputs.get("n_rejected")
        if isinstance(n_rej, int) and not isinstance(n_rej, bool):
            if n_rej != len(rejected):
                problems.append(
                    f"inputs.n_rejected={n_rej} disagrees with len(artifacts.rejected)={len(rejected)}"
                )

    # ---- accepted indices must be unique and ascending ----
    if accepted:
        idxs = [
            r["index"]
            for r in accepted
            if isinstance(r, dict) and isinstance(r.get("index"), int)
        ]
        if len(idxs) == len(accepted):  # only check if every record had a valid index
            if idxs != sorted(idxs):
                problems.append(
                    "artifacts.accepted: must be sorted by 'index' ascending"
                )
            if len(set(idxs)) != len(idxs):
                problems.append(
                    "artifacts.accepted: 'index' values must be unique"
                )

    # ---- xyz_ensemble + xyz/best are required when accepted is non-empty ----
    if accepted:
        ensemble = artifacts.get("xyz_ensemble")
        if not isinstance(ensemble, list) or len(ensemble) == 0:
            problems.append(
                "artifacts.xyz_ensemble: required (single record) when accepted is non-empty"
            )
        else:
            problems.extend(
                _validate_ensemble_record(ensemble[0], "xyz_ensemble", "accepted_ensemble")
            )
            if len(ensemble) > 1:
                problems.append(
                    f"artifacts.xyz_ensemble: must contain exactly one record, got {len(ensemble)}"
                )

        xyz = artifacts.get("xyz")
        if not isinstance(xyz, list) or len(xyz) == 0:
            problems.append(
                "artifacts.xyz: required (single 'best' record) when accepted is non-empty"
            )
        else:
            problems.extend(_validate_ensemble_record(xyz[0], "xyz", "best"))
            if len(xyz) > 1:
                problems.append(
                    f"artifacts.xyz: must contain exactly one record, got {len(xyz)}"
                )

        # best_chosen_by is required when accepted is non-empty.
        if isinstance(inputs, dict) and "best_chosen_by" not in inputs:
            problems.append(
                "inputs.best_chosen_by: required when accepted is non-empty"
            )
        elif isinstance(inputs, dict) and not isinstance(inputs.get("best_chosen_by"), str):
            problems.append(
                "inputs.best_chosen_by: must be a string"
            )

    # ---- empty accepted requires ok=false + a structured failure ----
    if not accepted:
        ok = manifest.get("ok")
        if ok is True:
            problems.append(
                "ok=true with empty accepted bucket: a conformer_screen impl that "
                "selects zero conformers must record ok=false and a structured failure"
            )
        failures = manifest.get("failures")
        if not isinstance(failures, list) or len(failures) == 0:
            problems.append(
                "failures: must contain at least one structured failure when accepted is empty"
            )

    return problems


def assert_valid_conformer_screen(manifest: dict[str, Any], **kwargs: Any) -> None:
    """Strict variant: raises ``ValueError`` listing all problems.

    Useful in tests and one-shot validations. In node ``run()`` code,
    prefer the non-raising :func:`validate_conformer_screen` and append
    problems to the manifest's failures.
    """
    problems = validate_conformer_screen(manifest, **kwargs)
    if problems:
        raise ValueError(
            "conformer_screen contract violations:\n  - " + "\n  - ".join(problems)
        )
