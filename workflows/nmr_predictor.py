#!/usr/bin/env python3
"""NMR Predictor — SMILES -> predicted ¹H/¹³C NMR shifts + couplings.

Pipeline (left -> right):

    [SMILES widget]
        -> wf-embed                 (RDKit ETKDG + MMFF)
        -> wf-xtb                   (xTB pre-opt)
        -> wf-crest                 (conformer search)
        -> wf-prism                 (conformer pruning)
        -> wf-orca-dft-array        (DFT geometry opt, SLURM array)
        -> wf-prism                 (re-prune post-opt)
        -> wf-orca-thermo-array     (freq + high-level SP + GIAO + J-coupling, all chained per conformer)
        -> wf-thermo-aggregate      (Boltzmann weights + composite Gibbs)
        -> wf-nmr-aggregate         (Boltzmann-averaged shifts + couplings, cheshire / Bally-Rablen calibration)

Run::

    python workflows/nmr_predictor.py [--out dist/workflows]

Produces ``<out>/NMR Predictor.zip`` with the workflow JSON + every
NODE_*.zip the engine needs to import the workflow standalone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_workflow import Workflow  # noqa: E402


# Coordinate grid (engine uses pixel-ish units, 150 px per column / 90 px per row).
COL = 180
ROW = 90


def x(col: int, base: int = -90) -> int:
    return base + col * COL


def y(row: int, base: int = 105) -> int:
    return base + row * ROW


def build() -> Workflow:
    wf = Workflow(
        name="NMR Predictor",
        description=(
            "Predict ¹H/¹³C NMR shifts + ¹H–¹H couplings from a SMILES string.\n\n"
            "End-to-end SLURM-driven pipeline: RDKit embed -> xTB pre-opt -> CREST -> "
            "prism prune -> ORCA DFT opt array -> prism re-prune -> ORCA freq + high-level "
            "SP + GIAO shielding (¹H/¹³C) + ¹H–¹H J-coupling (compound jobs per conformer) -> "
            "Boltzmann-weighted thermochemistry -> Boltzmann-averaged NMR with "
            "cheshire / Bally-Rablen linear-scaling calibration."
        ),
    )

    # ------------------------------------------------------------------
    # GUI input widgets (left column).
    # ------------------------------------------------------------------
    smiles = wf.widget(
        "text", alias="SMILES", default="",
        help_text="One SMILES per line (optional ', name' suffix).",
        at=(x(0), y(1)),
    )

    charge   = wf.widget("number", alias="Charge",            default="0",  at=(x(0), y(2)))
    unpaired = wf.widget("number", alias="Unpaired electrons", default="0", at=(x(0), y(3)))
    solvent  = wf.widget("combo",  alias="Solvent",            default="CHCl3", at=(x(0), y(4)))

    xtb_theory   = wf.widget("combo", alias="xTB theory",      default="gfn2", at=(x(0), y(5)))
    crest_mode   = wf.widget("combo", alias="CREST mode",      default="standard", at=(x(0), y(6)))
    crest_ewin   = wf.widget("number", alias="CREST energy window (kcal/mol)", default="6", at=(x(0), y(7)))

    max_conc = wf.widget("number", alias="DFT max concurrency",   default="10",  at=(x(0), y(9)))
    nprocs   = wf.widget("number", alias="ORCA nprocs",           default="8",   at=(x(0), y(10)))
    partition= wf.widget("combo",  alias="SLURM partition",       default="highmem", at=(x(0), y(11)))
    time_lim = wf.widget("text",   alias="SLURM walltime",        default="12:00:00", at=(x(0), y(12)))

    temperature = wf.widget("number", alias="Temperature (K)", default="298.15", at=(x(0), y(13)))

    # NMR-side knobs (kept narrow — defaults already match nmr_aggregate's table).
    nmr_solvent = wf.widget(
        "combo", alias="NMR calibration solvent",
        default="CHCl3", at=(x(0), y(15)),
        help_text="Must match a key in nmr_calibration.NMR_CALIBRATION.",
    )

    # ------------------------------------------------------------------
    # Process pipeline (top row).
    # ------------------------------------------------------------------
    embed   = wf.process("wf-embed",              at=(x(2), y(1)))
    xtb     = wf.process("wf-xtb",                at=(x(3), y(1)))
    crest   = wf.process("wf-crest",              at=(x(4), y(1)))
    prune1  = wf.process("wf-prism", alias="prism (post-CREST)",     at=(x(5), y(1)))
    dftopt  = wf.process("wf-orca-dft-array",     at=(x(6), y(1)))
    prune2  = wf.process("wf-prism", alias="prism (post-DFT-opt)",   at=(x(7), y(1)))
    thermo  = wf.process("wf-orca-thermo-array",  at=(x(8), y(1)))
    aggT    = wf.process("wf-thermo-aggregate",   at=(x(9), y(1)))
    aggNmr  = wf.process("wf-nmr-aggregate",      at=(x(10), y(1)))

    # ------------------------------------------------------------------
    # Pointer chain (top row, left -> right).
    # ------------------------------------------------------------------
    wf.pointer(smiles, embed, port="smiles")    # widget -> source process
    wf.pointer(embed,  xtb)
    wf.pointer(xtb,    crest)
    wf.pointer(crest,  prune1)
    wf.pointer(prune1, dftopt)
    wf.pointer(dftopt, prune2)
    wf.pointer(prune2, thermo)
    wf.pointer(thermo, aggT)
    wf.pointer(aggT,   aggNmr)

    # ------------------------------------------------------------------
    # Config bindings (auto-tagged "key=value" -> argv tokens).
    # ------------------------------------------------------------------
    # Molecular state — fan out to every node that asks for it.
    wf.bind(charge,   to=[xtb, crest, dftopt, thermo],                key="charge")
    wf.bind(unpaired, to=[xtb, crest, dftopt, thermo],                key="unpaired_electrons")

    # Solvent: the QC nodes use it for SMD/ALPB; nmr-aggregate uses it
    # for the calibration-table lookup. nmr_solvent is a separate widget
    # so the user can decouple if needed.
    wf.bind(solvent,     to=[xtb, crest, dftopt, thermo],             key="solvent")
    wf.bind(nmr_solvent, to=aggNmr,                                   key="solvent")

    # xTB / CREST knobs.
    wf.bind(xtb_theory, to=[xtb, crest], key="theory")
    wf.bind(crest_mode, to=crest,        key="mode")
    wf.bind(crest_ewin, to=crest,        key="ewin_kcal")

    # SLURM / array knobs (shared across both ORCA arrays).
    wf.bind(max_conc,   to=[dftopt, thermo], key="max_concurrency")
    wf.bind(nprocs,     to=[dftopt, thermo], key="nprocs")
    wf.bind(partition,  to=[dftopt, thermo], key="partition")
    wf.bind(time_lim,   to=[dftopt, thermo], key="time_limit")

    # Aggregator knobs.
    wf.bind(temperature, to=aggT, key="temperature_k")

    return wf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("dist/workflows"))
    p.add_argument("--no-zip", action="store_true",
                   help="Stage files only; skip the .zip.")
    args = p.parse_args(argv)

    wf = build()
    out = wf.export(args.out, zip_=not args.no_zip)
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
