"""Demo: run nmr_aggregate end-to-end with synthetic data and inspect outputs.

Builds a single-conformer synthetic ``thermo_aggregate`` upstream manifest
for a molecule of your choosing (default: ethanol) and invokes the
``wf-nmr-aggregate`` node. Outputs land in a project-relative directory
you can browse with a text editor or open in your browser (the ``.html``
diagrams are 3Dmol.js viewers).

The shielding + coupling values are synthetic — the script doesn't run
ORCA and doesn't know real chemistry. They're shaped just enough to
exercise the aggregator's whole pipeline (calibration, equivalence
detection, mnova XML, SVG and HTML diagrams) so you can SEE the
artifacts produced. Don't draw chemical conclusions from them.

Usage::

    python tools/run_aggregate_demo.py
    python tools/run_aggregate_demo.py --smiles "CCO"           # ethanol
    python tools/run_aggregate_demo.py --smiles "C[C@H](Br)CC"  # 2-bromobutane (S)
    python tools/run_aggregate_demo.py --smiles "c1ccccc1"      # benzene
    python tools/run_aggregate_demo.py --smiles "CF"            # methyl fluoride — H-F coupling
    python tools/run_aggregate_demo.py --smiles "FC(F)F"        # fluoroform — 3 equivalent F's
    python tools/run_aggregate_demo.py --smiles "CCF"           # fluoroethane — multi-bond H-F
    python tools/run_aggregate_demo.py --outdir ./my_run

When the molecule contains ¹⁹F (or ³¹P), the demo auto-passes
``mnova_heteronuclear_partners=F,P`` to the aggregator so the ¹H XML
gains F/P groups + cross-element J's. The rendered ¹H spectrum in
mnova should show splitting from the heteronuclei. F atoms appear at
their raw σ values (out of the ¹H window in practice), so the J-only
splitting on H multiplets is what you'll see.

Then:

    open ./aggregate_demo/calls/nmr_aggregate/outputs/predicted_structure_1h.html

Requires the ``chem`` extra (rdkit + numpy).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# --------------------------------------------------------------------
# Synthetic upstream construction (ORCA-shaped output text + manifest)
# --------------------------------------------------------------------


def _shielding_block_text(rows: list[tuple[int, str, float, float]]) -> str:
    """Build a CHEMICAL SHIELDING SUMMARY block matching ORCA 6's format."""
    lines = [
        "                       CHEMICAL SHIELDING SUMMARY (ppm)",
        "   --------------------------------------------------------",
        "   Nucleus  Element     Isotropic     Anisotropy",
        "   --------  -------     ----------    ----------",
    ]
    for idx, el, iso, aniso in rows:
        lines.append(
            f"     {idx:3d}      {el:<2s}      {iso:10.3f}      {aniso:8.3f}"
        )
    return "\n".join(lines) + "\n"


def _coupling_block_text(
    pairs: list[tuple[int, int, str, str, float]],
) -> str:
    """Build ORCA-6-style J-coupling per-pair blocks (Total + Ramsey terms)."""
    out_lines: list[str] = []
    for (i, j, ei, ej, j_total) in pairs:
        out_lines.append(
            f" NUCLEUS A = {ei}    {i} NUCLEUS B = {ej}    {j}"
        )
        out_lines.append(f" J[{i},{j}](Total)            iso=    {j_total:.3f}")
        out_lines.append(f" J[{i},{j}](FC)               iso=    {j_total:.3f}")
        out_lines.append(f" J[{i},{j}](SD)               iso=      0.000")
        out_lines.append(f" J[{i},{j}](PSO)              iso=      0.000")
        out_lines.append(f" J[{i},{j}](DSO)              iso=      0.000")
        out_lines.append("")
    return "\n".join(out_lines) + "\n"


def _xyz_block_from_mol(mol) -> str:
    """Serialize a 3D-embedded RDKit Mol as XYZ format."""
    conf = mol.GetConformer()
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    lines = [str(len(atoms)), "demo"]
    for i, sym in enumerate(atoms):
        pos = conf.GetAtomPosition(i)
        lines.append(f"{sym:2s} {pos.x: .8f} {pos.y: .8f} {pos.z: .8f}")
    return "\n".join(lines) + "\n"


def _synthesize_shielding_data(mol):
    """Generate plausible-shaped (but not chemically-real) shielding values.

    Critical invariant: atoms in the same RDKit topological class get
    the SAME σ. Otherwise the aggregator's data-aware refinement (which
    splits classes whose DFT shifts spread above tol_shift_ppm = 0.05
    ppm by default — the right behavior for diastereotopic CH₂ in
    chiral environments) would falsely split, e.g., a methyl group
    just because we used per-atom variation.

    Per-class shifts are spread over a small window per nucleus type
    so the diagrams show distinguishable groups. Aromatic atoms land
    in a different window from aliphatic atoms so an aromatic-vs-
    aliphatic split is visible.
    """
    from rdkit import Chem

    h_rows: list[tuple[int, str, float, float]] = []
    c_rows: list[tuple[int, str, float, float]] = []

    # Canonical ranks: atoms with the same rank are topologically
    # equivalent and MUST share a synthetic shift.
    ranks = list(
        Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True)
    )

    # Cache per-rank shifts so all atoms with rank R get the same σ.
    h_sigma_by_rank: dict[int, float] = {}
    c_sigma_by_rank: dict[int, float] = {}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        rank = ranks[idx]
        sym = atom.GetSymbol()
        is_aromatic = atom.GetIsAromatic()

        if sym == "H":
            if rank not in h_sigma_by_rank:
                # Aromatic H: σ ≈ 24 ppm → δ ≈ 7 ppm post-WP04 cal.
                # Hydroxyl H: σ ≈ 29 ppm → δ ≈ 2.6 ppm.
                # Aliphatic H: σ ≈ 30.5 ppm → δ ≈ 1.3 ppm.
                parent = atom.GetNeighbors()[0] if atom.GetNeighbors() else None
                if parent and parent.GetIsAromatic():
                    base = 24.0
                elif parent and parent.GetSymbol() == "O":
                    base = 29.0
                else:
                    base = 30.5
                # Spread distinct ranks within the window by a small
                # offset so they're visually distinguishable.
                h_sigma_by_rank[rank] = base + 0.4 * len(h_sigma_by_rank)
            h_rows.append((idx, "H", h_sigma_by_rank[rank], 5.0))
        elif sym == "C":
            if rank not in c_sigma_by_rank:
                base = 60.0 if not is_aromatic else 60.0  # tweak windows here
                c_sigma_by_rank[rank] = base + 5.0 * len(c_sigma_by_rank)
            c_rows.append((idx, "C", c_sigma_by_rank[rank], 12.0))
    return h_rows, c_rows


def _synthesize_couplings(mol):
    """Generate J couplings for atom pairs within ~4 bonds.

    Topology-only heuristic with J magnitudes that look approximately
    chemically sane (so the rendered mnova spectrum splits in
    recognizable patterns):

    * Geminal (2-bond, same parent C) H-H: J ≈ -12 Hz
    * Vicinal (3-bond, across C-C) H-H:    J ≈ 7 Hz
    * 2-bond H-F (through one C):          J ≈ 46 Hz (²J)
    * 3-bond H-F (across C-C-F):           J ≈ 22 Hz (³J)
    * 2-bond F-F (geminal on same C):      J ≈ 155 Hz
    * Pairs farther than 4 bonds are omitted (mirrors ORCA's
      SpinSpinRThresh dropping long-range couplings).
    """
    pairs: list[tuple[int, int, str, str, float]] = []
    target_indices = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("H", "F")
    ]
    for i in target_indices:
        dist = _bfs_distances(mol, i, max_depth=4)
        ei = mol.GetAtomWithIdx(i).GetSymbol()
        for j in target_indices:
            if j <= i:
                continue
            d = dist.get(j)
            if d is None:
                continue
            ej = mol.GetAtomWithIdx(j).GetSymbol()
            j_value = _coupling_for(ei, ej, d)
            if j_value is not None:
                pairs.append((i, j, ei, ej, j_value))
    return pairs


def _coupling_for(elem_i: str, elem_j: str, n_bonds: int):
    """Pick a synthetic J value (Hz) for an (element pair, distance)."""
    pair = tuple(sorted([elem_i, elem_j]))
    if pair == ("H", "H"):
        if n_bonds == 2:
            return -12.0
        if n_bonds == 3:
            return 7.0
    if pair == ("F", "H"):
        if n_bonds == 2:
            return 46.0  # ²J(H,F) — large; visible doublet on H
        if n_bonds == 3:
            return 22.0  # ³J(H,F)
    if pair == ("F", "F"):
        if n_bonds == 2:
            return 155.0  # ²J(F,F) — typical geminal-difluoro magnitude
        if n_bonds == 3:
            return 5.0
    return None


def _bfs_distances(mol, start: int, max_depth: int) -> dict[int, int]:
    from collections import deque

    seen: dict[int, int] = {start: 0}
    q: deque[int] = deque([start])
    while q:
        cur = q.popleft()
        if seen[cur] >= max_depth:
            continue
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nb.GetIdx()
            if ni in seen:
                continue
            seen[ni] = seen[cur] + 1
            q.append(ni)
    return seen


def _build_upstream_manifest(
    *,
    outdir: Path,
    smiles: str,
    seed: int = 42,
):
    """3D-embed the SMILES, synthesize ORCA outputs, write the upstream."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from scripps_workflow.schema import Manifest

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise SystemExit(f"could not parse SMILES: {smiles!r}")
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)

    h_rows, c_rows = _synthesize_shielding_data(mol)
    couplings = _synthesize_couplings(mol)
    h_text = _shielding_block_text(h_rows)
    c_text = _shielding_block_text(c_rows)
    j_text = _coupling_block_text(couplings)
    xyz_text = _xyz_block_from_mol(mol)

    upstream_dir = outdir / "upstream"
    upstream_outputs = upstream_dir / "outputs"
    upstream_outputs.mkdir(parents=True, exist_ok=True)
    tasks_root = upstream_dir / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)

    # One synthetic conformer is enough for the demo. For a more
    # realistic feel, increase n_conformers and add small per-conformer
    # noise to the shielding values.
    confs: list[dict] = []
    n_conformers = 1
    for i in range(1, n_conformers + 1):
        d = tasks_root / f"task_{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "orca_nmr_h.out").write_text(h_text, encoding="utf-8")
        (d / "orca_nmr_c.out").write_text(c_text, encoding="utf-8")
        (d / "orca_nmr_j.out").write_text(j_text, encoding="utf-8")
        xyz_file = d / "input.xyz"
        xyz_file.write_text(xyz_text, encoding="utf-8")
        confs.append({
            "index": i,
            "label": f"conf_{i:04d}",
            "path_abs": str(xyz_file.resolve()),
            "task_dir_abs": str(d.resolve()),
            "boltzmann_weight": 1.0 / n_conformers,
        })

    m = Manifest.skeleton(step="thermo_aggregate", cwd=str(upstream_dir))
    m.artifacts["conformers"] = confs
    m_path = upstream_outputs / "manifest.json"
    m.write(m_path)
    return m_path, mol


def _run_aggregator(
    *,
    upstream_manifest: Path,
    smiles: str,
    outdir: Path,
    mol,
):
    """Invoke wf-nmr-aggregate against the synthetic upstream.

    Auto-detects heteronuclear partners (F, P) from the molecule's
    elements and threads them through as
    ``mnova_heteronuclear_partners=<csv>``, so the rendered ¹H XML
    includes cross-element J's (e.g., ¹H-¹⁹F splitting in mnova).
    """
    from scripps_workflow.nodes.nmr_aggregate import NmrAggregate
    from scripps_workflow.pointer import Pointer

    pointer_text = Pointer.of(
        ok=True, manifest_path=upstream_manifest
    ).to_json_line()

    # Auto-detect heteronuclear partners from the embedded molecule.
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    partners = sorted(elements & {"F", "P"})  # nuclei with calibration entries

    call_dir = outdir / "calls" / "nmr_aggregate"
    call_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(call_dir)
    config_tokens = [f"smiles={smiles}"]
    if partners:
        config_tokens.append(
            f"mnova_heteronuclear_partners={','.join(partners)}"
        )
        print(
            f"  auto-detected heteronuclear partners: {','.join(partners)} "
            f"(included in 1H XML's spin-system)"
        )
    try:
        rc = NmrAggregate().invoke(
            ["nmr_aggregate", pointer_text, *config_tokens]
        )
    finally:
        os.chdir(cwd)
    if rc != 0:
        print(f"WARNING: nmr_aggregate exited rc={rc} (soft-fail expected to be 0)")
    return call_dir / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smiles", default="CCO",
        help="SMILES of the molecule to demo (default: ethanol)",
    )
    parser.add_argument(
        "--outdir", default="./aggregate_demo",
        help="output directory (default: ./aggregate_demo)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        # Don't accumulate stale state across re-runs; clear and rebuild.
        import shutil
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    print(f"SMILES: {args.smiles}")
    print(f"output: {outdir}")
    print()

    upstream_manifest, mol = _build_upstream_manifest(
        outdir=outdir, smiles=args.smiles,
    )
    print(f"  built synthetic upstream: {upstream_manifest}")
    print(f"  atoms: {mol.GetNumAtoms()} "
          f"(C={sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')}, "
          f"H={sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'H')}, "
          f"other={sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in 'CH')})")

    aggregator_outputs = _run_aggregator(
        upstream_manifest=upstream_manifest,
        smiles=args.smiles,
        outdir=outdir,
        mol=mol,
    )
    print(f"  ran aggregator")
    print()

    # List artifacts.
    print(f"Artifacts in {aggregator_outputs}:")
    files = sorted(aggregator_outputs.glob("*"))
    for f in files:
        size = f.stat().st_size if f.is_file() else 0
        print(f"  {f.name:<48} {size:>8} bytes")

    # Print a summary peek.
    summary_path = aggregator_outputs / "nmr_summary.json"
    if summary_path.exists():
        print()
        print("nmr_summary.json:")
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            print(f"  {line}")

    print()
    print("Open the HTML viewers in a browser:")
    for nuc in ("1h", "13c"):
        p = aggregator_outputs / f"predicted_structure_{nuc}.html"
        if p.exists():
            print(f"  open {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
