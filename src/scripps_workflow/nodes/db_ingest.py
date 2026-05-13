"""``wf-db-ingest`` — persist nmr_aggregate results into the nmr-data database.

Sits at the end of the NMR Predictor pipeline, after ``wf-nmr-aggregate``.
Reads the upstream nmr_aggregate manifest to locate the predicted_shifts.csv
and predicted_couplings.csv, walks one level further up to reach the
thermo_aggregate manifest for per-conformer energy data, then writes
everything to the nmr-data PostgreSQL database via ``nmr_data.ingest``.

The node is intentionally a thin orchestration wrapper — all DB logic lives
in ``nmr_data.ingest`` so it can be called directly from Python without
going through the pipeline.

Config keys (``key=value`` tokens or one JSON object)::

    database_url     PostgreSQL connection string. Falls back to
                     NMR_DATABASE_URL env var if omitted.          [None]
    source           Molecule provenance tag:
                     "virtual_library" | "lab_internal"            ["virtual_library"]
    cas_number       CAS registry number for the molecule.         [None]
    external_id      Any external identifier (compound ID, etc).   [None]
    hpc_data_root    Absolute HPC path under which .xyz and HDF5
                     files live. Falls back to NMR_HPC_DATA_ROOT
                     env var. Used to store relative file paths.   [None]
    dry_run          Parse everything, log what would be written,
                     but do NOT commit to the DB.                   [false]
    fail_policy      "soft" or "hard".                             ["soft"]

Requires:
    * ``nmr-data`` package installed in the same Python environment
      (``pip install nmr-data[ingest]``).
    * NMR_DATABASE_URL environment variable (or ``database_url`` config key).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..config_schema import ConfigField, NodeSchema, apply_schema
from ..node import Node, NodeContext
from ..parsing import normalize_optional_str, parse_bool
from ..schema import Manifest

# ---------------------------------------------------------------------------
# Optional nmr_data import — node fails gracefully if not installed
# ---------------------------------------------------------------------------

try:
    from nmr_data.db import get_session
    from nmr_data.ingest import ingest_nmr_aggregate_result
    _HAS_NMR_DATA = True
except ImportError:
    _HAS_NMR_DATA = False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = NodeSchema(
    step_name="db_ingest",
    cli_entrypoint="wf-db-ingest",
    module_path="scripps_workflow.nodes.db_ingest",
    overview=(
        "Persist nmr_aggregate outputs (shifts, couplings, conformer "
        "thermochemistry) into the nmr-data PostgreSQL database. "
        "Molecule rows are deduplicated by InChIKey — re-running the "
        "pipeline for the same compound is safe."
    ),
    fields=(
        ConfigField(
            name="database_url",
            type="str",
            default="",
            description=(
                "PostgreSQL connection string, e.g. "
                "postgresql://user:pass@host:5432/nmrdata. "
                "Falls back to NMR_DATABASE_URL environment variable."
            ),
        ),
        ConfigField(
            name="source",
            type="str",
            default="virtual_library",
            choices=("virtual_library", "lab_internal"),
            description=(
                "Molecule provenance tag stored in the molecules table."
            ),
        ),
        ConfigField(
            name="cas_number",
            type="str",
            default="",
            description="CAS registry number for the molecule (optional).",
        ),
        ConfigField(
            name="external_id",
            type="str",
            default="",
            description=(
                "Any external compound identifier (virtual library ID, "
                "internal compound number, etc.)."
            ),
        ),
        ConfigField(
            name="hpc_data_root",
            type="str",
            default="",
            description=(
                "Absolute path to the HPC data root directory. "
                ".xyz and HDF5 file paths stored in the DB are made "
                "relative to this root. Falls back to NMR_HPC_DATA_ROOT."
            ),
        ),
        ConfigField(
            name="dry_run",
            type="bool",
            default=False,
            description=(
                "Parse and validate all inputs, log what would be written, "
                "but do NOT commit any rows to the database."
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_artifact_by_label(
    manifest_dict: dict[str, Any],
    bucket: str,
    label: str,
) -> str | None:
    """Return path_abs of the first artifact in ``bucket`` whose label matches."""
    arts = manifest_dict.get("artifacts") or {}
    items = arts.get(bucket) or []
    for item in items:
        if isinstance(item, dict) and item.get("label") == label:
            return item.get("path_abs")
    return None


def _load_thermo_manifest(nmr_manifest_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Walk one level up from the nmr_aggregate manifest to thermo_aggregate."""
    upstream = nmr_manifest_dict.get("upstream") or {}
    manifest_path = upstream.get("manifest_path")
    if not manifest_path:
        return None
    p = Path(manifest_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _collect_conformer_records(thermo_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the conformers bucket out of the thermo_aggregate manifest dict."""
    arts = thermo_dict.get("artifacts") or {}
    confs = arts.get("conformers") or []
    return list(confs) if isinstance(confs, list) else []


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class DbIngest(Node):
    """Persist nmr_aggregate results into the nmr-data database."""

    step = "db_ingest"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        return apply_schema(raw, SCHEMA)

    def run(self, ctx: NodeContext) -> None:  # noqa: C901
        cfg = ctx.config

        # ---- 0) Dependency check ----
        if not _HAS_NMR_DATA:
            ctx.fail(
                "nmr_data_not_installed",
                detail=(
                    "The nmr-data package is not installed in this Python "
                    "environment. Install it with: pip install nmr-data[ingest]"
                ),
            )
            return

        # ---- 1) Resolve DB URL (not required for dry_run) ----
        dry_run = bool(cfg.get("dry_run", False))
        database_url = (
            cfg.get("database_url")
            or os.environ.get("NMR_DATABASE_URL")
            or ""
        )
        if not dry_run and not database_url:
            ctx.fail(
                "no_database_url",
                detail=(
                    "Set database_url config key or NMR_DATABASE_URL env var."
                ),
            )
            return

        # ---- 2) Read nmr_aggregate manifest ----
        if ctx.upstream_manifest is None:
            ctx.fail("no_upstream_manifest")
            return

        nmr_dict = ctx.upstream_manifest.to_dict()

        # SMILES is recorded by nmr_aggregate in inputs
        smiles = nmr_dict.get("inputs", {}).get("smiles")
        if not smiles:
            ctx.fail(
                "smiles_missing_from_upstream",
                detail=(
                    "nmr_aggregate manifest does not contain inputs.smiles. "
                    "Pass smiles=<SMILES> to wf-nmr-aggregate."
                ),
            )
            return

        # NMR method params (used to populate the predicted_runs row)
        nmr_inputs = nmr_dict.get("inputs", {})

        # ---- 3) Locate CSV files ----
        shifts_path_str = _find_artifact_by_label(nmr_dict, "files", "predicted_shifts_csv")
        couplings_path_str = _find_artifact_by_label(nmr_dict, "files", "predicted_couplings_csv")

        if not shifts_path_str:
            ctx.fail("shifts_csv_not_found_in_manifest")
            return
        if not couplings_path_str:
            ctx.fail("couplings_csv_not_found_in_manifest")
            return

        shifts_path = Path(shifts_path_str)
        couplings_path = Path(couplings_path_str)

        if not shifts_path.exists():
            ctx.fail("shifts_csv_missing_on_disk", path=shifts_path_str)
            return
        if not couplings_path.exists():
            ctx.fail("couplings_csv_missing_on_disk", path=couplings_path_str)
            return

        # ---- 4) Load thermo_aggregate manifest for conformer records ----
        thermo_dict = _load_thermo_manifest(nmr_dict)
        if thermo_dict is None:
            # Non-fatal: proceed without conformer energy data
            ctx.manifest.add_failure(
                "thermo_manifest_unavailable",
                detail="Conformers will be ingested without energy data.",
            )
            conformer_records: list[dict[str, Any]] = []
        else:
            conformer_records = _collect_conformer_records(thermo_dict)

        # ---- 5) Optional config values ----
        cas_number = normalize_optional_str(cfg.get("cas_number") or "")
        external_id = normalize_optional_str(cfg.get("external_id") or "")
        hpc_data_root = (
            cfg.get("hpc_data_root")
            or os.environ.get("NMR_HPC_DATA_ROOT")
            or None
        )
        dry_run = bool(cfg.get("dry_run", False))

        ctx.set_inputs(
            smiles=smiles,
            source=cfg["source"],
            cas_number=cas_number,
            external_id=external_id,
            dry_run=dry_run,
            n_conformer_records=len(conformer_records),
        )

        # ---- 6) Dry-run short-circuit ----
        if dry_run:
            from nmr_data.ingest import _compute_inchikey
            inchikey = _compute_inchikey(smiles)
            ctx.set_input("dry_run_inchikey", inchikey)
            ctx.set_input("dry_run_shifts_rows", _count_csv_rows(shifts_path))
            ctx.set_input("dry_run_couplings_rows", _count_csv_rows(couplings_path))
            return  # success, nothing written

        # ---- 7) Write to DB ----
        # Temporarily override the NMR_DATABASE_URL so nmr_data.config picks it up
        _orig_url = os.environ.get("NMR_DATABASE_URL")
        os.environ["NMR_DATABASE_URL"] = database_url

        try:
            with get_session() as session:
                summary = ingest_nmr_aggregate_result(
                    session=session,
                    smiles=smiles,
                    shifts_csv_path=shifts_path,
                    couplings_csv_path=couplings_path,
                    nmr_run_params=nmr_inputs,
                    conformer_records=conformer_records,
                    source=cfg["source"],
                    cas_number=cas_number,
                    external_id=external_id,
                    hpc_data_root=hpc_data_root,
                )
        finally:
            # Restore original env var
            if _orig_url is None:
                os.environ.pop("NMR_DATABASE_URL", None)
            else:
                os.environ["NMR_DATABASE_URL"] = _orig_url

        # ---- 8) Record summary ----
        ctx.set_inputs(**summary)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV (excludes header)."""
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

__all__ = ["DbIngest", "SCHEMA", "main"]

main = DbIngest.invoke_factory()

if __name__ == "__main__":
    raise SystemExit(main())
