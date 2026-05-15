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
                     NMR_DATABASE_URL env var if omitted. Required
                     unless ``dry_run`` is set.                     [""]
    source           Molecule provenance tag:
                     "virtual_library" | "lab_internal".            ["virtual_library"]
    external_id      Any external identifier (virtual-library ID,
                     internal compound number, etc.).               [""]
    hpc_data_root    Absolute HPC path under which .xyz and HDF5
                     files live; .xyz and HDF5 paths stored in the
                     DB are made relative to this root. Falls back
                     to NMR_HPC_DATA_ROOT env var.                  [""]
    dry_run          Parse the upstream manifest + locate CSVs,
                     log what would be written, but do NOT commit
                     any rows to the database.                      [false]

Upstream contract — wf-nmr-aggregate MUST carry a ``smiles`` field in
its parsed inputs, otherwise this node fails with
``smiles_missing_from_upstream``. The NMR Predictor recipe binds the
SMILES widget into both ``wf-embed.smiles`` and ``wf-nmr-aggregate.smiles``
for exactly this reason. If you wire the workflow by hand in the GUI,
remember to add the second binding.

Requires:
    * ``nmr-data`` Python package importable from the same env as
      this node. Pulled in via the ``db`` optional-deps group of
      ``scripps-workflow`` itself::

          pip install -e ".[db]"

      (which resolves ``nmr-data @ git+https://github.com/labounad/nmr-data.git``).
    * ``NMR_DATABASE_URL`` env var (or ``database_url`` config key)
      pointing at a PostgreSQL instance with the ``nmr-data`` schema
      already migrated. ``NMR_HPC_DATA_ROOT`` (or ``hpc_data_root``)
      controls path-relativization for stored file references.
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
        # cas_number is intentionally NOT a config port. It's resolved
        # automatically from the SMILES at ingest time via NCI's
        # Chemical Identifier Resolver (best-effort, 3-second timeout,
        # NULL on failure). See nmr_data.ingest._resolve_cas_from_smiles.
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


def _thermo_manifest_path(nmr_manifest_dict: dict[str, Any]) -> Path | None:
    """Return the on-disk path of the upstream thermo_aggregate manifest,
    or None if it isn't reachable."""
    upstream = nmr_manifest_dict.get("upstream") or {}
    manifest_path = upstream.get("manifest_path")
    if not manifest_path:
        return None
    p = Path(manifest_path)
    return p if p.exists() else None


def _node_script_path(manifest_path: Path) -> Path | None:
    """Locate the engine-rendered ``script.sh`` for the call that produced
    ``manifest_path``. Engine layout: ``<call_dir>/script.sh`` and
    ``<call_dir>/outputs/manifest.json`` — so the script sits one level
    up from outputs/. Returns None if it isn't there."""
    candidate = manifest_path.parent.parent / "script.sh"
    return candidate if candidate.exists() else None


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
        thermo_manifest_path: Path | None = _thermo_manifest_path(nmr_dict)
        if thermo_dict is None:
            # Non-fatal: proceed without conformer energy data
            ctx.manifest.add_failure(
                "thermo_manifest_unavailable",
                detail="Conformers will be ingested without energy data.",
            )
            conformer_records: list[dict[str, Any]] = []
        else:
            conformer_records = _collect_conformer_records(thermo_dict)

        # ---- 4a) Paths for the central-tree copy step ----
        # nmr_aggregate's manifest itself is the upstream pointer's
        # manifest_path. The outputs/ dir is its parent — that's where
        # the mnova XMLs, structure diagrams, and nmr_summary.json live.
        nmr_manifest_path: Path | None = None
        nmr_outputs_dir: Path | None = None
        nmr_script_path: Path | None = None
        if ctx.upstream_pointer is not None:
            upm = Path(ctx.upstream_pointer.manifest_path)
            if upm.exists():
                nmr_manifest_path = upm
                nmr_outputs_dir = upm.parent
                nmr_script_path = _node_script_path(upm)

        # thermo_aggregate's outputs/ dir similarly sits next to its
        # manifest — that's where conformer_thermo.csv lives.
        thermo_outputs_dir: Path | None = (
            thermo_manifest_path.parent if thermo_manifest_path else None
        )

        # ---- 5) Optional config values ----
        # cas_number is auto-resolved from the SMILES inside
        # get_or_create_molecule (NCI CIR, best-effort).
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
            external_id=external_id,
            dry_run=dry_run,
            n_conformer_records=len(conformer_records),
        )

        # ---- 6) Dry-run short-circuit ----
        if dry_run:
            from nmr_data.ingest import (
                _compute_inchikey,
                _resolve_cas_from_smiles,
            )
            inchikey = _compute_inchikey(smiles)
            # Fire the CAS resolver here too so the dry-run manifest
            # shows whatever the real ingest path would have stored.
            # Returns None on network failure / timeout / no match —
            # safe to surface either way.
            cas_number = _resolve_cas_from_smiles(smiles)
            ctx.set_input("dry_run_inchikey", inchikey)
            ctx.set_input("dry_run_cas_number", cas_number)
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
                    external_id=external_id,
                    hpc_data_root=hpc_data_root,
                    nmr_manifest_path=nmr_manifest_path,
                    thermo_manifest_path=thermo_manifest_path,
                    nmr_outputs_dir=nmr_outputs_dir,
                    thermo_outputs_dir=thermo_outputs_dir,
                    nmr_script_path=nmr_script_path,
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
