#!/usr/bin/env python3
"""Inspect the current nmr-data DB + central-tree state for one molecule.

Prints a per-stage summary: how many rows of each kind exist for the
given SMILES, what the central-tree dirs look like, and whether the
producing-node self-registration appears to have landed correctly.

Usage::

    python tools/inspect_registry_progress.py <SMILES>
    python tools/inspect_registry_progress.py CO
    python tools/inspect_registry_progress.py 'CC(=O)O'

Requires the same env as ``wf-db-ingest``: ``NMR_DATABASE_URL`` and
``NMR_HPC_DATA_ROOT`` must be set. Falls back to a friendly error
message if either is missing or if ``nmr-data`` isn't installed.

Used by ``docs/registry-verification.md`` for the four-claim
end-to-end verification protocol.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _inchikey_or_die(smiles: str) -> str:
    try:
        from nmr_data.cache import inchikey_from_smiles
    except Exception as e:
        _die(f"nmr_data not importable: {type(e).__name__}: {e}")
    ik = inchikey_from_smiles(smiles)
    if not ik:
        _die(f"could not compute InChIKey for SMILES {smiles!r}")
    return ik  # type: ignore[return-value]


def _list_dir(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if not p.name.startswith("."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("smiles", help="SMILES to inspect")
    args = ap.parse_args()

    db_url = os.environ.get("NMR_DATABASE_URL")
    hpc_root = os.environ.get("NMR_HPC_DATA_ROOT")
    if not db_url:
        _die("NMR_DATABASE_URL is unset")
    if not hpc_root:
        _die("NMR_HPC_DATA_ROOT is unset")
    hpc_root_path = Path(hpc_root)

    inchikey = _inchikey_or_die(args.smiles)
    print(f"SMILES:    {args.smiles}")
    print(f"InChIKey:  {inchikey}")
    print()

    try:
        from nmr_data.db import get_session
        from nmr_data.models import (
            ConformerEnsemble,
            ConformerThermo,
            DftRun,
            Molecule,
            PredictedCoupling,
            PredictedRun,
            PredictedShift,
            ThermoRun,
        )
    except Exception as e:
        _die(f"nmr_data import failed: {type(e).__name__}: {e}")

    with get_session() as s:
        mol = s.query(Molecule).filter_by(inchikey=inchikey).one_or_none()
        if mol is None:
            print(f"DB state:  molecule row absent (pipeline hasn't run yet)")
            tree = hpc_root_path / inchikey
            if tree.is_dir():
                print(f"⚠  central-tree dir present at {tree} despite no DB row")
            return 0

        print(f"Molecule:  id={mol.id} (created_at={mol.created_at})")
        print()

        # ---- Ensembles ----
        ens_rows = s.query(ConformerEnsemble).filter_by(molecule_id=mol.id).all()
        print(f"Ensembles  ({len(ens_rows)})")
        for ens in ens_rows:
            print(
                f"  - id={ens.id}  fp={ens.provenance_fingerprint[:12]}…"
                f"  search={ens.crest_search_method}"
                f"  solvent={ens.solvent}"
                f"  n_conformers={ens.n_conformers}"
            )
            if ens.ensemble_path:
                tree_dir = hpc_root_path / ens.ensemble_path
                files = _list_dir(tree_dir)
                confs = _list_dir(tree_dir / "conformers")
                print(
                    f"    central-tree: {ens.ensemble_path} "
                    f"{'OK' if tree_dir.is_dir() else 'MISSING'} "
                    f"({len(confs)} conf dirs, {len(files)} top-level entries)"
                )
            else:
                print(f"    central-tree: ⚠ ensemble_path is NULL")
        print()

        # ---- DftRuns ----
        ens_ids = [e.id for e in ens_rows]
        dft_rows = (
            s.query(DftRun).filter(DftRun.ensemble_id.in_(ens_ids)).all()
            if ens_ids else []
        )
        print(f"DftRuns    ({len(dft_rows)})")
        for d in dft_rows:
            print(
                f"  - id={d.id}  fp={d.provenance_fingerprint[:12]}…"
                f"  method={d.dft_opt_method}/{d.dft_opt_basis}"
                f"  solvent={d.solvent}"
            )
            if d.dft_run_path:
                tree_dir = hpc_root_path / d.dft_run_path
                confs = _list_dir(tree_dir / "conformers")
                print(
                    f"    central-tree: {d.dft_run_path} "
                    f"{'OK' if tree_dir.is_dir() else 'MISSING'} "
                    f"({len(confs)} conf dirs)"
                )
            else:
                print(f"    central-tree: ⚠ dft_run_path is NULL")
        print()

        # ---- ThermoRuns ----
        dft_ids = [d.id for d in dft_rows]
        thermo_rows = (
            s.query(ThermoRun).filter(ThermoRun.dft_run_id.in_(dft_ids)).all()
            if dft_ids else []
        )
        print(f"ThermoRuns ({len(thermo_rows)})")
        for t in thermo_rows:
            print(
                f"  - id={t.id}  fp={t.provenance_fingerprint[:12]}…"
                f"  method={t.thermo_method}/{t.thermo_basis}"
                f"  T={t.temperature_k}  state={t.standard_state}"
            )
            n_ct = (
                s.query(ConformerThermo)
                .filter_by(thermo_run_id=t.id)
                .count()
            )
            print(f"    conformer_thermo rows: {n_ct}")
            if t.thermo_run_path:
                tree_dir = hpc_root_path / t.thermo_run_path
                confs = _list_dir(tree_dir / "conformers")
                top = _list_dir(tree_dir)
                print(
                    f"    central-tree: {t.thermo_run_path} "
                    f"{'OK' if tree_dir.is_dir() else 'MISSING'} "
                    f"({len(confs)} conf dirs, {len(top)} top-level entries)"
                )
            else:
                print(f"    central-tree: ⚠ thermo_run_path is NULL")
        print()

        # ---- PredictedRuns ----
        pred_rows = (
            s.query(PredictedRun).filter_by(molecule_id=mol.id).all()
        )
        print(f"PredictedRuns ({len(pred_rows)})")
        for pr in pred_rows:
            n_shifts = s.query(PredictedShift).filter_by(run_id=pr.id).count()
            n_cpl = s.query(PredictedCoupling).filter_by(run_id=pr.id).count()
            print(
                f"  - id={pr.id}  shielding_h={pr.shielding_method_h}/{pr.shielding_basis_h}"
                f"  coupling={pr.coupling_method}/{pr.coupling_basis}"
                f"  solvent={pr.solvent}  T={pr.temperature_k}"
            )
            print(
                f"    n_conformers_used={pr.n_conformers_used}  "
                f"n_shifts={n_shifts}  n_couplings={n_cpl}"
            )
            if pr.run_root_path:
                tree_dir = hpc_root_path / pr.run_root_path
                top = _list_dir(tree_dir)
                print(
                    f"    central-tree: {pr.run_root_path} "
                    f"{'OK' if tree_dir.is_dir() else 'MISSING'} "
                    f"({len(top)} top-level entries)"
                )
            else:
                print(f"    central-tree: ⚠ run_root_path is NULL")
        print()

        # ---- Summary verdict ----
        n_ens = len(ens_rows)
        n_dft = len(dft_rows)
        n_thermo = len(thermo_rows)
        n_pred = len(pred_rows)
        print("Summary:")
        if n_ens and n_dft and n_thermo and n_pred:
            print("  ✓ all four stages registered")
        else:
            missing = []
            if not n_ens:    missing.append("ensemble")
            if not n_dft:    missing.append("dft_run")
            if not n_thermo: missing.append("thermo_run")
            if not n_pred:   missing.append("predicted_run")
            print(f"  partial state — missing stages: {', '.join(missing)}")
            print(
                "  re-running the pipeline should cache-hit the present "
                "stages and compute the missing ones."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
