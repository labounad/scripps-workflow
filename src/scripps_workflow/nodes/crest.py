"""``wf-crest`` — CREST conformer search starting from an upstream xyz.

Chain node. Consumes the first ``xyz`` artifact from the upstream manifest,
runs ``crest`` to perform an iMTD-GC (or one of the speed-knob variants)
conformer search, then publishes the resulting ensemble:

    artifacts.xyz            single ``best`` representative (crest_best.xyz,
                             fallback to ``conf_0001.xyz`` if crest didn't
                             write one)
    artifacts.xyz_ensemble   single multi-xyz file (``crest_conformers``)
    artifacts.conformers     one record per conformer with ``index``,
                             ``path_abs``, ``sha256``, optional
                             ``rel_energy_kcal``
    artifacts.files          ``crest_rotamers.xyz`` and ``crest.energies``
                             when present
    artifacts.logs           crest stdout + stderr
    artifacts.operations     single record for the crest run with ``cmd``,
                             ``returncode``, ``elapsed_seconds``

Note that this node *produces* an ensemble — it is **not** a
``conformer_screen`` itself; the prism / marc / cluster nodes are. The
``conformers`` bucket here is the screen's input.

Config keys (``key=value`` tokens, or one JSON object):

    theory              GFN-FF | GFN1-XTB | GFN2-XTB | GFN2//GFN-FF
                                                             (default GFN2-XTB)
    charge              int                                  (default 0)
    unpaired_electrons  int  (maps to crest ``--uhf``)       (default 0)
    solvent             one of ALPB_SOLVENTS, or none/null   (default none)
    mode                standard | quick | squick | mquick   (default standard)
    ewin_kcal           float, must be > 0                   (default 10.0)
    max_conformers      int >= 0; 0 keeps all                (default 0)
    threads             int (>0) or 0/auto for SLURM autodetect
                                                             (default auto)

The subprocess-using core is :func:`run_crest`; tests monkeypatch it to
avoid needing a real crest/xtb pair on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import logging_utils
from ..hashing import sha256_file
from ..node import Node, NodeContext
from ..parsing import parse_float, parse_int

# Import shared helpers from xtb_calc rather than duplicate. If a third
# node grows the same dependency, factor these out into a shared util
# module (e.g. ``runtime.py``).
from .xtb_calc import (
    ALPB_SOLVENTS,
    find_first_xyz_path,
    normalize_solvent,
    resolve_threads,
    slurm_threads_fallback,
)


# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------

#: CREST run modes. ``standard`` maps to no flag (crest's default
#: iMTD-GC pipeline); the others trade rigor for wall time.
CREST_MODES: frozenset[str] = frozenset({"standard", "quick", "squick", "mquick"})

#: Canonical theory names for CREST. Includes the ``GFN2//GFN-FF``
#: composite mode (single-point GFN2 on a GFN-FF metadynamics surface).
CREST_THEORIES: tuple[str, ...] = (
    "GFN2-XTB",
    "GFN1-XTB",
    "GFN-FF",
    "GFN2//GFN-FF",
)

#: Aliases the user may type for each theory. Lowercased on lookup.
_THEORY_ALIASES: dict[str, str] = {
    # GFN2-XTB
    "gfn2-xtb": "GFN2-XTB",
    "gfn2xtb": "GFN2-XTB",
    "gfn2": "GFN2-XTB",
    "2": "GFN2-XTB",
    # GFN1-XTB
    "gfn1-xtb": "GFN1-XTB",
    "gfn1xtb": "GFN1-XTB",
    "gfn1": "GFN1-XTB",
    "1": "GFN1-XTB",
    # GFN-FF
    "gfn-ff": "GFN-FF",
    "gfnff": "GFN-FF",
    "gff": "GFN-FF",
    "gfn-ff (gff)": "GFN-FF",
    # Composite
    "gfn2//gfnff": "GFN2//GFN-FF",
    "gfn2//gfn-ff": "GFN2//GFN-FF",
    "gfn2/gfnff": "GFN2//GFN-FF",
}


# --------------------------------------------------------------------
# Pure helpers (testable)
# --------------------------------------------------------------------


def normalize_theory(v: Any) -> str:
    """Coerce a user-supplied theory string to one of :data:`CREST_THEORIES`.

    Empty / ``None`` → ``"GFN2-XTB"`` (CREST's default + Lucas's lab default).
    Raises :class:`ValueError` on anything else.
    """
    if v is None:
        return "GFN2-XTB"
    s = str(v).strip()
    if s == "":
        return "GFN2-XTB"
    key = s.lower()
    if key in _THEORY_ALIASES:
        return _THEORY_ALIASES[key]
    raise ValueError(
        f"Unknown theory {v!r}; expected one of {sorted(_THEORY_ALIASES)}"
    )


def normalize_mode(v: Any) -> str:
    """Coerce a user-supplied mode string.

    Empty / ``None`` → ``"standard"`` (CREST's default iMTD-GC pipeline).
    Raises :class:`ValueError` on unknown values.
    """
    if v is None:
        return "standard"
    s = str(v).strip().lower()
    if s in {"", "default"}:
        return "standard"
    if s in CREST_MODES:
        return s
    raise ValueError(
        f"Unknown CREST mode {v!r}; expected one of {sorted(CREST_MODES)}"
    )


def crest_theory_flag(theory: str) -> str | None:
    """Map a normalized theory name to the crest CLI flag.

    Returns ``None`` for ``"GFN2-XTB"`` because that is crest's default
    and the absence of a flag is the canonical way to request it. Other
    theories explicitly emit their flag.
    """
    if theory == "GFN2-XTB":
        return None
    if theory == "GFN1-XTB":
        return "--gfn1"
    if theory == "GFN-FF":
        return "--gfnff"
    if theory == "GFN2//GFN-FF":
        return "--gfn2//gfnff"
    raise ValueError(f"Unknown theory {theory!r}")


def crest_mode_flag(mode: str) -> str | None:
    """Map a normalized mode to its crest flag.

    ``"standard"`` returns ``None`` (no flag = default iMTD-GC pipeline).
    """
    if mode == "standard":
        return None
    if mode in {"quick", "squick", "mquick"}:
        return f"--{mode}"
    raise ValueError(f"Unknown CREST mode {mode!r}")


def build_crest_cmd(
    *,
    crest_exe: str,
    input_xyz_name: str,
    theory: str,
    charge: int,
    uhf: int,
    solvent: str | None,
    mode: str,
    ewin_kcal: float,
    threads: int,
) -> list[str]:
    """Assemble a crest command. ``input_xyz_name`` is treated as a relative
    name (the caller is expected to ``cwd=run_dir`` for the subprocess)."""
    cmd: list[str] = [crest_exe, input_xyz_name]

    th_flag = crest_theory_flag(theory)
    if th_flag is not None:
        cmd.append(th_flag)

    # Match legacy: only emit charge / uhf flags when nonzero. crest's
    # default is 0 for both anyway, so this keeps the recorded cmd line
    # shorter for the common neutral / closed-shell case.
    if charge != 0:
        cmd += ["--chrg", str(int(charge))]
    if uhf != 0:
        cmd += ["--uhf", str(int(uhf))]

    if solvent is not None:
        cmd += ["--alpb", solvent]

    cmd += ["--ewin", f"{float(ewin_kcal):.6g}"]

    mode_flag = crest_mode_flag(mode)
    if mode_flag is not None:
        cmd.append(mode_flag)

    cmd += ["--T", str(int(threads))]
    return cmd


@dataclass
class XyzBlock:
    """One frame from a multi-xyz file (CREST's ensemble format)."""

    nat: int
    comment: str
    lines: list[str]


def split_multixyz(text: str) -> list[XyzBlock]:
    """Split a CREST-style multi-xyz blob into individual frames.

    Tolerant: stops at the first malformed frame rather than raising,
    because crest occasionally writes a truncated tail when interrupted.
    """
    blocks: list[XyzBlock] = []
    lines = text.splitlines()
    n = len(lines)
    i = 0

    while i < n:
        # Skip blank separator lines.
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break

        try:
            nat = int(lines[i].strip())
        except ValueError:
            break
        if nat <= 0:
            break

        if i + 1 >= n:
            break
        comment = lines[i + 1]

        end = i + 2 + nat
        if end > n:
            break

        block_lines = lines[i:end]
        blocks.append(XyzBlock(nat=nat, comment=comment, lines=list(block_lines)))
        i = end

    return blocks


def write_xyz_block(path: Path, blk: XyzBlock) -> None:
    """Write a single :class:`XyzBlock` back out as a standalone xyz file."""
    path.write_text("\n".join(blk.lines) + "\n", encoding="utf-8")


def parse_crest_energies(path: Path) -> list[float | None]:
    """Parse a ``crest.energies`` file into a list of relative energies (kcal).

    The file format is loose — crest writes one line per conformer, each
    line containing one or more whitespace-separated floats. The
    *relative* energy is the last float on each line; the others are
    indices / absolute energies which we discard here. Lines without
    parseable floats yield ``None`` (preserves alignment with the
    conformer index).
    """
    energies: list[float | None] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.replace(",", " ").split()
        floats: list[float] = []
        for t in toks:
            try:
                floats.append(float(t))
            except ValueError:
                continue
        if floats:
            energies.append(float(floats[-1]))
        else:
            energies.append(None)
    return energies


# --------------------------------------------------------------------
# Subprocess core (monkeypatch target for tests)
# --------------------------------------------------------------------


def run_crest(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    """Single crest invocation. Returns ``(returncode, elapsed_seconds)``.

    ``stdout`` / ``stderr`` go to the given paths so they can be recorded
    as artifacts. Tests monkeypatch this whole function (it is the only
    place we shell out to crest).
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


class CrestConformerSearch(Node):
    """Chain node: run crest on the first xyz artifact from upstream."""

    step = "crest"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        # Raise on unknown values up-front so a typo becomes
        # ``argv_parse_failed`` rather than a silent wrong-mode run.
        ewin = parse_float(raw.get("ewin_kcal"), 10.0)
        if ewin <= 0:
            raise ValueError("ewin_kcal must be > 0")

        max_confs = parse_int(raw.get("max_conformers"), 0)
        if max_confs < 0:
            raise ValueError("max_conformers must be >= 0")

        return {
            "theory": normalize_theory(raw.get("theory", "GFN2-XTB")),
            "charge": parse_int(raw.get("charge"), 0),
            "unpaired_electrons": parse_int(raw.get("unpaired_electrons"), 0),
            "solvent": normalize_solvent(raw.get("solvent")),
            "mode": normalize_mode(raw.get("mode", "standard")),
            "ewin_kcal": float(ewin),
            "max_conformers": int(max_confs),
            "threads": parse_int(raw.get("threads"), 0),
        }

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config

        # Echo the resolved config into manifest.inputs.
        ctx.set_inputs(
            theory=cfg["theory"],
            charge=cfg["charge"],
            unpaired_electrons=cfg["unpaired_electrons"],
            solvent=cfg["solvent"],
            mode=cfg["mode"],
            ewin_kcal=cfg["ewin_kcal"],
            max_conformers=cfg["max_conformers"],
            threads_requested=cfg["threads"],
        )

        # Locate both binaries — crest shells out to xtb internally, so a
        # missing xtb is just as fatal as a missing crest. The wrapper
        # script does ``module load crest xtb`` so both should be present
        # in production.
        crest_exe = shutil.which("crest")
        xtb_exe = shutil.which("xtb")
        if not crest_exe:
            ctx.fail("crest_not_found_on_PATH")
            return
        if not xtb_exe:
            ctx.fail("xtb_not_found_on_PATH")
            return
        ctx.manifest.environment["crest"] = crest_exe
        ctx.manifest.environment["xtb"] = xtb_exe

        # Pull first xyz artifact from upstream.
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
        crest_dir = outputs_dir / "crest"
        conf_dir = outputs_dir / "conformers"
        run_dir.mkdir(parents=True, exist_ok=True)
        crest_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

        # Stage a stable copy of the input geometry inside the run dir, so
        # crest's working files end up alongside their input.
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

        # Build and run the crest command.
        cmd = build_crest_cmd(
            crest_exe=crest_exe,
            input_xyz_name=staged_input.name,
            theory=cfg["theory"],
            charge=cfg["charge"],
            uhf=cfg["unpaired_electrons"],
            solvent=cfg["solvent"],
            mode=cfg["mode"],
            ewin_kcal=cfg["ewin_kcal"],
            threads=threads,
        )

        stdout_path = outputs_dir / "crest.stdout.txt"
        stderr_path = outputs_dir / "crest.stderr.txt"

        logging_utils.log_info(f"crest: {' '.join(cmd)}")
        rc, elapsed = run_crest(
            cmd, cwd=run_dir, stdout_path=stdout_path, stderr_path=stderr_path
        )

        # Record stdout / stderr as logs artifacts even on failure.
        for stream, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            if path.exists():
                ctx.add_artifact(
                    "logs",
                    {
                        "label": f"crest_{stream}",
                        "path_abs": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "format": "txt",
                        "stream": stream,
                    },
                )

        # Single ``operations`` record for this crest run, mirroring the
        # xtb_calc per-op record shape so downstream tooling can use the
        # same code path for both.
        ctx.add_artifact(
            "operations",
            {
                "label": "crest",
                "path_abs": str(run_dir.resolve()),
                "format": "dir",
                "op": "crest",
                "cmd": list(cmd),
                "returncode": int(rc),
                "elapsed_seconds": float(elapsed),
                "input_geom_abs": str(staged_input.resolve()),
            },
        )

        if rc != 0:
            ctx.fail(f"crest_failed_returncode_{rc}")
            # Continue: collect whatever crest did manage to write, so the
            # operator can debug from the manifest.

        # ----- collect outputs -----
        # Best effort: any of these files may legitimately be missing
        # (e.g. on a quick run, or if crest crashed mid-run).
        self._collect_outputs(
            ctx,
            run_dir=run_dir,
            crest_dir=crest_dir,
            conf_dir=conf_dir,
            max_confs=cfg["max_conformers"],
        )

        # Final sanity: must have produced AT LEAST a best xyz; otherwise
        # downstream nodes have nothing to chain off and we should mark
        # the run as failed even if crest itself returned 0.
        if not ctx.manifest.artifacts.get("xyz"):
            ctx.fail("no_best_xyz_artifact_produced")

    # -------------------- helpers --------------------

    def _collect_outputs(
        self,
        ctx: NodeContext,
        *,
        run_dir: Path,
        crest_dir: Path,
        conf_dir: Path,
        max_confs: int,
    ) -> None:
        """Copy crest's output files into the manifest-visible buckets."""
        crest_conformers_src = run_dir / "crest_conformers.xyz"
        crest_rotamers_src = run_dir / "crest_rotamers.xyz"
        crest_best_src = run_dir / "crest_best.xyz"
        crest_energies_src = run_dir / "crest.energies"

        # Whole multi-xyz ensemble.
        if crest_conformers_src.exists():
            dst = crest_dir / "crest_conformers.xyz"
            shutil.copy2(crest_conformers_src, dst)
            ctx.add_artifact(
                "xyz_ensemble",
                {
                    "label": "crest_conformers",
                    "path_abs": str(dst.resolve()),
                    "sha256": sha256_file(dst),
                    "format": "xyz",
                },
            )

        # Rotamer multi-xyz (informational; not the canonical ensemble).
        if crest_rotamers_src.exists():
            dst = crest_dir / "crest_rotamers.xyz"
            shutil.copy2(crest_rotamers_src, dst)
            ctx.add_artifact(
                "files",
                {
                    "label": "crest_rotamers",
                    "path_abs": str(dst.resolve()),
                    "sha256": sha256_file(dst),
                    "format": "xyz",
                },
            )

        # Energies file.
        if crest_energies_src.exists():
            dst = crest_dir / "crest.energies"
            shutil.copy2(crest_energies_src, dst)
            ctx.add_artifact(
                "files",
                {
                    "label": "crest_energies",
                    "path_abs": str(dst.resolve()),
                    "sha256": sha256_file(dst),
                    "format": "txt",
                },
            )

        # crest_best.xyz: copy now, but the canonical ``best`` artifact in
        # the ``xyz`` bucket is published below (preferring crest_best.xyz
        # if present, falling back to conf_0001).
        crest_best_dst: Path | None = None
        if crest_best_src.exists():
            crest_best_dst = crest_dir / "crest_best.xyz"
            shutil.copy2(crest_best_src, crest_best_dst)

        # Split the ensemble into per-conformer files. Best effort — if
        # crest_conformers.xyz is missing, we skip and rely on the
        # ``no_best_xyz`` invariant in run() to fail the manifest.
        crest_conformers_dst = crest_dir / "crest_conformers.xyz"
        if not crest_conformers_dst.exists():
            return

        text = crest_conformers_dst.read_text(encoding="utf-8", errors="replace")
        blocks = split_multixyz(text)

        # Truncate AFTER parsing so a non-positive max_conformers (=0) is
        # interpreted as "keep everything", matching the legacy node.
        if max_confs > 0 and len(blocks) > max_confs:
            blocks = blocks[:max_confs]

        # Energies: one per line, last-float-wins. Aligned to the
        # conformer index by position.
        energies: list[float | None] = []
        crest_energies_dst = crest_dir / "crest.energies"
        if crest_energies_dst.exists():
            energies = parse_crest_energies(crest_energies_dst)

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
                # crest reports energies in kcal/mol; name the field with
                # explicit units so downstream consumers (prism / marc)
                # don't have to guess.
                rec["rel_energy_kcal"] = float(energies[i - 1])
            ctx.add_artifact("conformers", rec)

        # Publish the ``best`` representative.
        best_dst = conf_dir / "best.xyz"
        if crest_best_dst is not None and crest_best_dst.exists():
            shutil.copy2(crest_best_dst, best_dst)
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
    "ALPB_SOLVENTS",
    "CREST_MODES",
    "CREST_THEORIES",
    "CrestConformerSearch",
    "XyzBlock",
    "build_crest_cmd",
    "crest_mode_flag",
    "crest_theory_flag",
    "main",
    "normalize_mode",
    "normalize_solvent",
    "normalize_theory",
    "parse_crest_energies",
    "resolve_threads",
    "run_crest",
    "slurm_threads_fallback",
    "split_multixyz",
    "write_xyz_block",
]


main = CrestConformerSearch.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
