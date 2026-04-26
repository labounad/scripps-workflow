"""``wf-orca-goat`` — ORCA GOAT global conformer search.

Chain node and CREST sibling. Where ``wf-crest`` runs an iMTD-GC
metadynamics search via the standalone CREST binary, ``wf-orca-goat``
runs ORCA's built-in GOAT (Global Optimization Algorithm by Topology,
ORCA 6.0+) directly. Both produce the same artifact shape
(``xyz_ensemble`` + ``conformers`` + ``xyz`` + ``files``), so the
downstream conformer_screen role is unaffected by which generator
produced the ensemble.

Why both? GOAT reaches conformers CREST's GFN-FF/XTB metadynamics
struggles with (e.g. anion stabilization, transition-metal complexes
where xTB parameters are weak), and conversely CREST is faster on the
small organics that dominate Lucas's NMR pipeline. Operators pick the
generator that matches the substrate; the workflow graph stays the
same.

Config keys (``key=value`` tokens, or one JSON object):

    theory              ORCA level-of-theory string. Anything ORCA
                        accepts in the simple-input line: composites
                        (``r2SCAN-3c``, ``B97-3c``), DFT functionals
                        (``B3LYP D3``, ``PBE0 D4``), or semiempirics
                        (``XTB``, ``XTB2``, ``GFN2-XTB``).
                        (default ``XTB``)
    charge              int                                  (default 0)
    unpaired_electrons  int. Multiplicity is computed as
                        ``2 * unpaired_electrons + 1`` to match the
                        crest node's UHF semantics.        (default 0)
    solvent             one of CPCM_SOLVENTS, or none/null   (default none)
    mode                regular | quick | explore | accurate
                        (GOAT speed knob)                  (default regular)
    ewin_kcal           float, must be > 0. Energy window for kept
                        conformers, passed via ``%goat maxEn`` or
                        equivalent (ORCA-version-dependent — we set
                        the GOAT block consistently; see _build_orca_input).
                                                            (default 6.0)
    max_conformers      int >= 0; 0 keeps all                (default 0)
    threads             int (>0) or 0/auto for SLURM autodetect
                                                             (default auto)
    maxcore_mb          int >= 100. ORCA per-process memory ceiling.
                                                             (default 2000)

The subprocess core is :func:`run_orca_goat`; tests monkeypatch it to
avoid needing a real ORCA install.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .. import logging_utils
from ..hashing import sha256_file
from ..node import Node, NodeContext
from ..parsing import parse_float, parse_int

# Reuse crest's xyz parsing helpers — split_multixyz / write_xyz_block
# are the canonical multi-xyz primitives in this package.
from .crest import XyzBlock, split_multixyz, write_xyz_block
from .xtb_calc import find_first_xyz_path, resolve_threads, slurm_threads_fallback


# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------

#: GOAT speed knobs. ``regular`` is ORCA's default and emits no
#: extra keyword. The others map to ``GOAT-QUICK`` etc. on the
#: simple-input line, which ORCA recognizes from 6.0 onward.
GOAT_MODES: frozenset[str] = frozenset(
    {"regular", "quick", "explore", "accurate"}
)

#: CPCM solvents recognized by ORCA's built-in CPCM (the common subset
#: shared with most QM packages — ORCA accepts a longer list, but
#: validating here catches the typo cases that would silently run in
#: vacuum). Lowercased for case-insensitive matching.
CPCM_SOLVENTS: frozenset[str] = frozenset(
    {
        "water",
        "acetone",
        "acetonitrile",
        "benzene",
        "chloroform",
        "ccl4",
        "ch2cl2",
        "dichloromethane",
        "dmf",
        "dmso",
        "ether",
        "diethylether",
        "ethanol",
        "hexane",
        "methanol",
        "octanol",
        "pyridine",
        "thf",
        "toluene",
        # Aliases the user is likely to type:
        "h2o",
    }
)

#: Aliases the user may type for ORCA theory strings. Lowercased on
#: lookup. The value is the canonical form passed to ORCA's input file
#: simple-input line; ORCA case-folds these but we keep the
#: ``r2SCAN-3c`` style for readability of the generated input.
_THEORY_ALIASES: dict[str, str] = {
    # xTB family (ORCA can drive xTB internally from 6.0).
    "xtb": "XTB",
    "xtb2": "XTB",
    "gfn2-xtb": "XTB",
    "gfn2xtb": "XTB",
    "gfn2": "XTB",
    "gfn1-xtb": "XTB1",
    "gfn1": "XTB1",
    "gfn-ff": "GFNFF",
    "gfnff": "GFNFF",
    # Composites — Lucas's lab default for "small + reasonably accurate".
    "r2scan-3c": "r2SCAN-3c",
    "r2scan3c": "r2SCAN-3c",
    "b97-3c": "B97-3c",
    "b973c": "B97-3c",
    "pbeh-3c": "PBEh-3c",
    "pbeh3c": "PBEh-3c",
    "hf-3c": "HF-3c",
    "hf3c": "HF-3c",
}


# --------------------------------------------------------------------
# Pure helpers (testable)
# --------------------------------------------------------------------


def normalize_theory(v: Any) -> str:
    """Canonicalize a user-supplied ORCA theory string.

    Known aliases (xTB / composites) get mapped to their canonical
    capitalization. Anything else is passed through verbatim after
    whitespace stripping — ORCA accepts a long tail of DFT functionals
    + basis-set combos (e.g. ``B3LYP D3 def2-SVP``) and we deliberately
    don't enumerate them here. ``None`` / empty string defaults to
    ``XTB`` (GOAT's fastest mode).

    Raises :class:`ValueError` only on tokens with shell-injection
    metacharacters; nothing else is rejected because ORCA itself is the
    authority on what's valid.
    """
    if v is None:
        return "XTB"
    s = str(v).strip()
    if s == "":
        return "XTB"
    key = s.lower()
    if key in _THEORY_ALIASES:
        return _THEORY_ALIASES[key]
    # Reject anything that could escape the simple-input line. ORCA
    # input files don't quote tokens — a stray ``\n`` or ``!`` in the
    # theory string would corrupt the input file.
    if re.search(r"[\n\r!*%&|;`$<>]", s):
        raise ValueError(
            f"theory string contains forbidden metacharacters: {v!r}"
        )
    return s


def normalize_mode(v: Any) -> str:
    """Canonicalize a GOAT speed-knob string."""
    if v is None:
        return "regular"
    s = str(v).strip().lower()
    if s in {"", "default"}:
        return "regular"
    if s in GOAT_MODES:
        return s
    raise ValueError(
        f"Unknown GOAT mode {v!r}; expected one of {sorted(GOAT_MODES)}"
    )


def normalize_solvent(v: Any) -> str | None:
    """Validate a CPCM solvent name. Returns ``None`` for vacuum."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in {"none", "null", "vacuum", "gas", "gas_phase"}:
        return None
    key = s.lower()
    if key not in CPCM_SOLVENTS:
        raise ValueError(
            f"Unknown CPCM solvent: {v!r} "
            f"(known: {sorted(CPCM_SOLVENTS)})"
        )
    # Map alias back to ORCA's canonical token.
    if key == "h2o":
        return "water"
    if key == "diethylether":
        return "ether"
    if key == "dichloromethane":
        return "ch2cl2"
    return key


def goat_simple_input_keyword(mode: str) -> str:
    """Return the GOAT keyword appropriate for ``mode``.

    ``regular`` → ``GOAT`` (no suffix). The other modes map to their
    speed-knob suffix, which ORCA 6.0+ recognizes on the simple-input
    line.
    """
    if mode == "regular":
        return "GOAT"
    if mode in {"quick", "explore", "accurate"}:
        return f"GOAT-{mode.upper()}"
    raise ValueError(f"Unknown GOAT mode: {mode!r}")


def build_orca_input(
    *,
    theory: str,
    mode: str,
    charge: int,
    multiplicity: int,
    solvent: str | None,
    ewin_kcal: float,
    max_conformers: int,
    threads: int,
    maxcore_mb: int,
    xyz_filename: str,
) -> str:
    """Build the text of an ``orca.inp`` file for a GOAT run.

    The simple-input line carries the GOAT keyword + theory + optional
    CPCM. The ``%goat`` block carries the energy window and the
    optional conformer cap. ``%pal`` and ``%maxcore`` are emitted
    unconditionally so the recorded input is reproducible.

    Args:
        theory: ORCA simple-input theory string (e.g. ``XTB``,
            ``r2SCAN-3c``, ``B3LYP D3``).
        mode: GOAT speed knob (one of :data:`GOAT_MODES`).
        charge: Net charge.
        multiplicity: Spin multiplicity (``2*S + 1``; the caller
            converts ``unpaired_electrons`` to this).
        solvent: CPCM solvent name, or ``None`` for vacuum.
        ewin_kcal: Energy window for kept conformers.
        max_conformers: Cap on conformers ORCA writes; ``0`` = no cap.
        threads: ``%pal nprocs`` value.
        maxcore_mb: ``%maxcore`` value (MB per process).
        xyz_filename: Filename of the staged input geometry, relative
            to the ORCA working directory.

    Returns:
        Full input file text, ready to be written to disk.
    """
    goat_kw = goat_simple_input_keyword(mode)
    simple_tokens = [f"! {goat_kw} {theory}"]
    if solvent is not None:
        simple_tokens.append(f"! CPCM({solvent})")

    parts: list[str] = []
    parts.extend(simple_tokens)
    parts.append("")
    parts.append(f"%pal nprocs {int(threads)} end")
    parts.append(f"%maxcore {int(maxcore_mb)}")
    parts.append("")
    # GOAT block: ewin is the energy window in kcal/mol that GOAT keeps
    # in its final ensemble. Keyword name has shifted slightly across
    # ORCA versions; ``maxEn`` is the 6.0+ spelling.
    goat_block = ["%goat", f"  maxEn {float(ewin_kcal):.6g}"]
    if max_conformers > 0:
        goat_block.append(f"  maxConfs {int(max_conformers)}")
    goat_block.append("end")
    parts.extend(goat_block)
    parts.append("")
    parts.append(f"* xyzfile {int(charge)} {int(multiplicity)} {xyz_filename}")
    parts.append("")
    return "\n".join(parts)


_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_goat_ensemble_energies(blocks: list[XyzBlock]) -> list[float | None]:
    """Pull a relative energy (kcal/mol) per frame from a GOAT
    ``finalensemble.xyz`` blob.

    GOAT writes the per-frame relative energy into the comment line.
    The exact text varies with ORCA version; common shapes include:

        ``"   0.000000000 kcal/mol"``
        ``"Erel:   1.234 kcal/mol Etot: -123.45 Eh"``
        ``"Energy: -123.456 Eh"`` (some pre-6.1 builds)

    Strategy: prefer an explicit ``Erel`` token when present; otherwise
    take the first float on the line. If the comment also carries an
    ``Eh`` / ``Hartree`` unit and we picked the wrong float (negative,
    huge magnitude), return ``None`` for that frame so the caller falls
    back to whatever upstream energies file is available.
    """
    out: list[float | None] = []
    for blk in blocks:
        out.append(_parse_one_comment_energy(blk.comment))
    return out


def _parse_one_comment_energy(comment: str) -> float | None:
    line = comment.strip()
    if not line:
        return None

    # Explicit Erel: <float>
    m = re.search(r"Erel\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None

    # Fallback: first float on the line. If "Eh" / "Hartree" is the
    # only unit hint and we read a negative-magnitude absolute energy,
    # we return None (we can't convert to relative without a reference).
    floats = _FLOAT_RE.findall(line)
    if not floats:
        return None
    try:
        v = float(floats[0])
    except ValueError:
        return None

    has_kcal = bool(re.search(r"\bkcal\b", line, flags=re.IGNORECASE))
    has_eh = bool(re.search(r"\b(Eh|Hartree)\b", line, flags=re.IGNORECASE))

    if has_kcal:
        return v
    if has_eh and not has_kcal:
        # Almost certainly an absolute energy. Caller decides what to
        # do; we don't try to subtract the minimum here because the
        # caller may also have the upstream energies file.
        return None
    # No explicit unit: assume the comment author meant kcal/mol
    # (GOAT's most common shape). This is the right default; if a
    # downstream sees garbage relative energies they'll catch it via
    # the prism / marc ewin filter or the contract validator.
    return v


def find_orca_outputs(run_dir: Path, basename: str) -> dict[str, Path | None]:
    """Locate GOAT output files in the ORCA run directory.

    Returns a dict with keys ``ensemble``, ``best``, ``out``, ``err``,
    ``property_json`` — each is the absolute path if present or
    ``None`` if missing.

    Filename conventions for ORCA 6.0+:

        ``<basename>.finalensemble.xyz``    multi-frame ensemble
        ``<basename>.globalminimum.xyz``    single-frame best
        ``<basename>.out``                  stdout-style log
        ``<basename>_property.json``        structured properties
                                            (optional, version-dep)
    """

    def _existing(p: Path) -> Path | None:
        return p if p.exists() else None

    return {
        "ensemble": _existing(run_dir / f"{basename}.finalensemble.xyz"),
        "best": _existing(run_dir / f"{basename}.globalminimum.xyz"),
        "out": _existing(run_dir / f"{basename}.out"),
        "err": _existing(run_dir / f"{basename}.err"),
        "property_json": _existing(run_dir / f"{basename}_property.json"),
    }


# --------------------------------------------------------------------
# Subprocess core (monkeypatch target for tests)
# --------------------------------------------------------------------


def run_orca_goat(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    """Single ORCA invocation. Returns ``(returncode, elapsed_seconds)``.

    ``stdout`` / ``stderr`` go to the given paths so they can be
    recorded as artifacts. Tests monkeypatch this whole function (it is
    the only place we shell out to ORCA).

    NB: ORCA must be invoked with its absolute path to enable the MPI /
    OpenMPI auto-detection. ``shutil.which("orca")`` is the canonical
    way to find it; the caller does that before constructing ``cmd``.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as fo, stderr_path.open(
        "w", encoding="utf-8"
    ) as fe:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=fo,
            stderr=fe,
            text=True,
            check=False,
        )
    elapsed = float(time.perf_counter() - t0)
    return int(proc.returncode), elapsed


# --------------------------------------------------------------------
# Node class
# --------------------------------------------------------------------


class OrcaGoat(Node):
    """Chain node: run ORCA GOAT on the first xyz artifact from upstream."""

    step = "orca_goat"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        ewin = parse_float(raw.get("ewin_kcal"), 6.0)
        if ewin <= 0:
            raise ValueError("ewin_kcal must be > 0")

        max_confs = parse_int(raw.get("max_conformers"), 0)
        if max_confs < 0:
            raise ValueError("max_conformers must be >= 0")

        maxcore_mb = parse_int(raw.get("maxcore_mb"), 2000)
        if maxcore_mb < 100:
            raise ValueError("maxcore_mb must be >= 100")

        return {
            "theory": normalize_theory(raw.get("theory", "XTB")),
            "charge": parse_int(raw.get("charge"), 0),
            "unpaired_electrons": parse_int(raw.get("unpaired_electrons"), 0),
            "solvent": normalize_solvent(raw.get("solvent")),
            "mode": normalize_mode(raw.get("mode", "regular")),
            "ewin_kcal": float(ewin),
            "max_conformers": int(max_confs),
            "threads": parse_int(raw.get("threads"), 0),
            "maxcore_mb": int(maxcore_mb),
        }

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config
        multiplicity = 2 * int(cfg["unpaired_electrons"]) + 1

        # Echo resolved config into manifest.inputs early, so even an
        # ORCA-not-on-PATH failure leaves a useful audit trail.
        ctx.set_inputs(
            theory=cfg["theory"],
            charge=cfg["charge"],
            unpaired_electrons=cfg["unpaired_electrons"],
            multiplicity=multiplicity,
            solvent=cfg["solvent"],
            mode=cfg["mode"],
            ewin_kcal=cfg["ewin_kcal"],
            max_conformers=cfg["max_conformers"],
            threads_requested=cfg["threads"],
            maxcore_mb=cfg["maxcore_mb"],
        )

        orca_exe = shutil.which("orca")
        if not orca_exe:
            ctx.fail("orca_not_found_on_PATH")
            return
        ctx.manifest.environment["orca"] = orca_exe

        if ctx.upstream_manifest is None:
            ctx.fail("no_upstream_manifest")
            return
        try:
            upstream_xyz = find_first_xyz_path(ctx.upstream_manifest)
        except Exception as e:
            ctx.fail(f"upstream_xyz_error: {e}")
            return

        outputs_dir = ctx.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = outputs_dir / "run"
        goat_dir = outputs_dir / "goat"
        conf_dir = outputs_dir / "conformers"
        run_dir.mkdir(parents=True, exist_ok=True)
        goat_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

        # Stage the upstream xyz alongside the ORCA input file. ORCA
        # rewrites the input geometry into its own output filenames, so
        # the staging dir effectively owns the run.
        staged_input = run_dir / "input.xyz"
        shutil.copy2(upstream_xyz, staged_input)
        ctx.add_artifact(
            "files",
            {
                "label": "input_xyz",
                "path_abs": str(staged_input.resolve()),
                "sha256": sha256_file(staged_input),
                "format": "xyz",
            },
        )

        threads = resolve_threads(cfg["threads"])
        ctx.manifest.environment["threads"] = threads

        # Build and write the ORCA input file. The basename ``orca`` is
        # arbitrary but stable so output filenames are predictable.
        basename = "orca"
        inp_path = run_dir / f"{basename}.inp"
        inp_text = build_orca_input(
            theory=cfg["theory"],
            mode=cfg["mode"],
            charge=cfg["charge"],
            multiplicity=multiplicity,
            solvent=cfg["solvent"],
            ewin_kcal=cfg["ewin_kcal"],
            max_conformers=cfg["max_conformers"],
            threads=threads,
            maxcore_mb=cfg["maxcore_mb"],
            xyz_filename=staged_input.name,
        )
        inp_path.write_text(inp_text, encoding="utf-8")
        ctx.add_artifact(
            "files",
            {
                "label": "orca_input",
                "path_abs": str(inp_path.resolve()),
                "sha256": sha256_file(inp_path),
                "format": "txt",
            },
        )

        # ORCA expects an absolute path to the .inp file as argv[1].
        # We cwd into run_dir so all outputs land alongside the input.
        cmd = [orca_exe, inp_path.name]

        stdout_path = outputs_dir / "orca.stdout.txt"
        stderr_path = outputs_dir / "orca.stderr.txt"

        logging_utils.log_info(f"orca-goat: {' '.join(cmd)}")
        rc, elapsed = run_orca_goat(
            cmd, cwd=run_dir, stdout_path=stdout_path, stderr_path=stderr_path
        )

        for stream, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            if path.exists():
                ctx.add_artifact(
                    "logs",
                    {
                        "label": f"orca_{stream}",
                        "path_abs": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "format": "txt",
                        "stream": stream,
                    },
                )

        ctx.add_artifact(
            "operations",
            {
                "label": "orca_goat",
                "path_abs": str(run_dir.resolve()),
                "format": "dir",
                "op": "orca_goat",
                "cmd": list(cmd),
                "returncode": int(rc),
                "elapsed_seconds": float(elapsed),
                "input_geom_abs": str(staged_input.resolve()),
            },
        )

        if rc != 0:
            ctx.fail(f"orca_failed_returncode_{rc}")
            # Continue: collect whatever ORCA managed to write, so the
            # operator can debug from the manifest.

        # ----- collect outputs -----
        self._collect_outputs(
            ctx,
            run_dir=run_dir,
            goat_dir=goat_dir,
            conf_dir=conf_dir,
            basename=basename,
            max_confs=cfg["max_conformers"],
        )

        if not ctx.manifest.artifacts.get("xyz"):
            ctx.fail("no_best_xyz_artifact_produced")

    # -------------------- helpers --------------------

    def _collect_outputs(
        self,
        ctx: NodeContext,
        *,
        run_dir: Path,
        goat_dir: Path,
        conf_dir: Path,
        basename: str,
        max_confs: int,
    ) -> None:
        """Copy GOAT's outputs into the manifest-visible buckets."""
        located = find_orca_outputs(run_dir, basename)

        # Whole multi-xyz ensemble.
        if located["ensemble"] is not None:
            dst = goat_dir / f"{basename}.finalensemble.xyz"
            shutil.copy2(located["ensemble"], dst)
            ctx.add_artifact(
                "xyz_ensemble",
                {
                    "label": "goat_conformers",
                    "path_abs": str(dst.resolve()),
                    "sha256": sha256_file(dst),
                    "format": "xyz",
                },
            )

        # globalminimum.xyz: copy now; canonical ``best`` published below
        # (preferring this if present, else first conformer).
        best_src_dst: Path | None = None
        if located["best"] is not None:
            best_src_dst = goat_dir / f"{basename}.globalminimum.xyz"
            shutil.copy2(located["best"], best_src_dst)

        # Optional structured properties JSON: just record it.
        if located["property_json"] is not None:
            dst = goat_dir / f"{basename}_property.json"
            shutil.copy2(located["property_json"], dst)
            ctx.add_artifact(
                "files",
                {
                    "label": "orca_property_json",
                    "path_abs": str(dst.resolve()),
                    "sha256": sha256_file(dst),
                    "format": "json",
                },
            )

        # Split the ensemble into per-conformer files. Best effort —
        # missing file = fail invariant in run() will catch it.
        ensemble_dst = goat_dir / f"{basename}.finalensemble.xyz"
        if not ensemble_dst.exists():
            return

        text = ensemble_dst.read_text(encoding="utf-8", errors="replace")
        blocks = split_multixyz(text)
        if max_confs > 0 and len(blocks) > max_confs:
            blocks = blocks[:max_confs]

        energies = parse_goat_ensemble_energies(blocks)

        for i, blk in enumerate(blocks, start=1):
            p = conf_dir / f"conf_{i:04d}.xyz"
            write_xyz_block(p, blk)
            rec: dict[str, Any] = {
                "index": i,
                "label": f"conf_{i:04d}",
                "path_abs": str(p.resolve()),
                "sha256": sha256_file(p),
                "format": "xyz",
            }
            if i - 1 < len(energies) and energies[i - 1] is not None:
                rec["rel_energy_kcal"] = float(energies[i - 1])
            ctx.add_artifact("conformers", rec)

        # Publish the ``best`` representative.
        best_dst = conf_dir / "best.xyz"
        if best_src_dst is not None and best_src_dst.exists():
            shutil.copy2(best_src_dst, best_dst)
        elif blocks:
            shutil.copy2(conf_dir / "conf_0001.xyz", best_dst)

        if best_dst.exists():
            ctx.add_artifact(
                "xyz",
                {
                    "label": "best",
                    "path_abs": str(best_dst.resolve()),
                    "sha256": sha256_file(best_dst),
                    "format": "xyz",
                },
            )


__all__ = [
    "CPCM_SOLVENTS",
    "GOAT_MODES",
    "OrcaGoat",
    "build_orca_input",
    "find_orca_outputs",
    "goat_simple_input_keyword",
    "main",
    "normalize_mode",
    "normalize_solvent",
    "normalize_theory",
    "parse_goat_ensemble_energies",
    "resolve_threads",
    "run_orca_goat",
    "slurm_threads_fallback",
]


main = OrcaGoat.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
