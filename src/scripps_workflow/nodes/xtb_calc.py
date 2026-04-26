"""``wf-xtb`` — single-geometry xTB calculation (SP and/or optimization).

Chain node. Consumes the first ``xyz`` artifact from the upstream manifest
and runs one or more xTB operations on it (geometry optimization, single-
point energy/gradient/Hessian). Each operation runs in its own
``outputs/run/<op>/`` directory so the raw xtb files (xtbopt.xyz,
xtbout.json, wbo, charges, ...) don't trample each other.

If ``Geometry Optimization`` is requested it is always run first, and any
subsequent SP ops use the optimized geometry (``xtbopt.xyz``) when xtb
produces one. Otherwise the SP ops run on the upstream geometry directly.

Config keys (all key=value tokens, or one JSON object):

    theory              GFN-FF | GFN1-XTB | GFN2-XTB         (default GFN2-XTB)
    charge              int                                  (default 0)
    unpaired_electrons  int (maps to ``--uhf``)              (default 0)
    solvent             one of ALPB_SOLVENTS, or none/null   (default none)
    opt_level           crude|sloppy|loose|normal|tight|verytight|extreme
                                                             (default tight)
    calculations        which ops to run. Any of:
                          - JSON object with GUI labels:
                              {"SP Energy": true, "Geometry Optimization": true}
                          - JSON list / CSV of internal tokens:
                              ["optimize", "sp_energy"]
                              "optimize,sp_energy"
                          - Single token: "optimize"
                        Default: ["optimize"]
    threads             int (>0) or 0/auto for SLURM autodetect
                                                             (default auto)
    write_json          true|false (xtb ``--json`` -> xtbout.json)
                                                             (default true)

Manifest shape (``wf.result.v1``)::

    artifacts.xyz             [{"label": "xtbopt", ...}]   (when optimize ran)
    artifacts.files           [{"label": "input_xyz", ...}]  (staged input copy)
    artifacts.xtbout_json     [{"label": "xtbout_<op>", "op": ..., ...}, ...]
    artifacts.logs            [{"label": "<op>_stdout"|"_stderr",
                                "op": ..., "stream": ..., ...}, ...]
    artifacts.operations      [{"label": <op>, "path_abs": <run_dir>,
                                "cmd": [...], "returncode": int,
                                "elapsed_seconds": float}, ...]

The subprocess-using core is :func:`run_xtb`; tests monkeypatch it to avoid
needing xtb on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from .. import logging_utils
from ..hashing import sha256_file
from ..node import Node, NodeContext
from ..parsing import parse_bool, parse_int

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------

#: ALPB-supported solvents recognized by xtb. Anything outside this set is
#: rejected by :func:`normalize_solvent`. Keep the spelling lowercase.
ALPB_SOLVENTS: frozenset[str] = frozenset(
    {
        "acetone",
        "acetonitrile",
        "aniline",
        "benzaldehyde",
        "benzene",
        "ch2cl2",
        "chcl3",
        "cs2",
        "dioxane",
        "dmf",
        "dmso",
        "ether",
        "ethylacetate",
        "furane",
        "hexandecane",
        "hexane",
        "methanol",
        "nitromethane",
        "octanol",
        "woctanol",
        "phenol",
        "toluene",
        "thf",
        "water",
    }
)

#: xtb optimization levels (passed verbatim after ``--opt``).
OPT_LEVELS: frozenset[str] = frozenset(
    {"crude", "sloppy", "loose", "normal", "tight", "verytight", "extreme"}
)

#: Internal op tokens, in run order. ``optimize`` always runs first when
#: present; SP ops follow.
OPS_ORDER: tuple[str, ...] = ("optimize", "sp_energy", "sp_gradient", "sp_hessian")

#: GUI checkbox labels -> internal op tokens. Reverse of OP_TOKEN_TO_LABEL.
OP_LABEL_TO_TOKEN: dict[str, str] = {
    "Geometry Optimization": "optimize",
    "SP Energy": "sp_energy",
    "SP Gradient": "sp_gradient",
    "SP Hessian": "sp_hessian",
}

#: Aliases the user may type for each token. All lower-cased on lookup.
_OP_ALIASES: dict[str, str] = {
    # canonical
    "optimize": "optimize",
    "sp_energy": "sp_energy",
    "sp_gradient": "sp_gradient",
    "sp_hessian": "sp_hessian",
    # short forms
    "opt": "optimize",
    "optimization": "optimize",
    "geometry optimization": "optimize",
    "sp": "sp_energy",
    "energy": "sp_energy",
    "sp energy": "sp_energy",
    "grad": "sp_gradient",
    "gradient": "sp_gradient",
    "sp gradient": "sp_gradient",
    "hess": "sp_hessian",
    "hessian": "sp_hessian",
    "sp hessian": "sp_hessian",
}


# --------------------------------------------------------------------
# Pure helpers (testable)
# --------------------------------------------------------------------


def normalize_theory(v: Any) -> str:
    """Normalize a theory string to one of GFN2-XTB, GFN1-XTB, GFN-FF.

    Empty / None defaults to GFN2-XTB. Raises ``ValueError`` on unknown values
    so a typo surfaces as an ``argv_parse_failed`` rather than running with
    the wrong theory.
    """
    s = str(v if v is not None else "").strip().upper()
    if s in {"", "GFN2", "GFN2-XTB", "GFN2XTB"}:
        return "GFN2-XTB"
    if s in {"GFN1", "GFN1-XTB", "GFN1XTB"}:
        return "GFN1-XTB"
    if s in {"GFN-FF", "GFNFF", "GFF"}:
        return "GFN-FF"
    raise ValueError(
        f"Unknown theory: {v!r} (expected GFN-FF, GFN1-XTB, or GFN2-XTB)"
    )


def normalize_solvent(v: Any) -> str | None:
    """Normalize an ALPB solvent name. Returns ``None`` for vacuum/no-solvent."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in {"none", "null", "vacuum"}:
        return None
    key = s.lower()
    if key not in ALPB_SOLVENTS:
        raise ValueError(
            f"Unknown ALPB solvent: {v!r} (see scripps_workflow.nodes.xtb_calc.ALPB_SOLVENTS)"
        )
    return key


def normalize_opt_level(v: Any) -> str:
    """Normalize an xtb opt level. Defaults to ``tight``."""
    s = str(v if v is not None else "").strip().lower()
    if not s:
        return "tight"
    if s not in OPT_LEVELS:
        raise ValueError(
            f"Unknown opt level: {v!r} (expected one of {sorted(OPT_LEVELS)})"
        )
    return s


def _normalize_ops_tokens(tokens: Iterable[str]) -> list[str]:
    """Reduce arbitrary aliases to canonical tokens, dedupe, sort by run order."""
    canonical: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        t = str(tok).strip().lower()
        if not t:
            continue
        canon = _OP_ALIASES.get(t)
        if canon is None:
            raise ValueError(f"Unknown calculation token: {tok!r}")
        if canon not in seen:
            seen.add(canon)
            canonical.append(canon)
    # Stable run order: optimize first, then SP ops in their canonical order.
    return [op for op in OPS_ORDER if op in seen]


def parse_calculations(value: Any) -> list[str]:
    """Resolve the ``calculations`` config to a list of internal op tokens.

    Accepts (in order of precedence):

    * ``None`` / empty → ``["optimize"]`` (the most useful default).
    * ``list`` of tokens.
    * JSON object string (GUI checkbox payload using GUI labels as keys, e.g.
      ``{"SP Energy": true, "Geometry Optimization": true}``).
    * JSON list string.
    * CSV string (``"optimize,sp_energy"``).
    * Single token string.
    """
    if value is None:
        return ["optimize"]
    if isinstance(value, list):
        return _normalize_ops_tokens(str(x) for x in value)

    s = str(value).strip()
    if not s:
        return ["optimize"]

    if s.startswith("{") and s.endswith("}"):
        import json

        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("calculations object must be a JSON object")
        chosen: list[str] = []
        for label, token in OP_LABEL_TO_TOKEN.items():
            if bool(obj.get(label, False)):
                chosen.append(token)
        return _normalize_ops_tokens(chosen)

    if s.startswith("[") and s.endswith("]"):
        import json

        obj = json.loads(s)
        if not isinstance(obj, list):
            raise ValueError("calculations array must be a JSON array")
        return _normalize_ops_tokens(str(x) for x in obj)

    # CSV / single token.
    return _normalize_ops_tokens(t.strip() for t in s.split(","))


def base_xtb_cmd(
    *,
    xtb_exe: str,
    theory: str,
    charge: int,
    uhf: int,
    solvent: str | None,
    threads: int,
    write_json: bool,
    input_name: str = "input.xyz",
) -> list[str]:
    """Build the prefix of an xtb command shared across operations.

    Per-op extra flags (``--opt <level>``, ``--grad``, ``--hess``) are appended
    by the caller.
    """
    cmd: list[str] = [xtb_exe, input_name]
    if theory == "GFN2-XTB":
        cmd += ["--gfn", "2"]
    elif theory == "GFN1-XTB":
        cmd += ["--gfn", "1"]
    elif theory == "GFN-FF":
        cmd += ["--gfnff"]
    else:
        raise ValueError(f"Unsupported theory in base_xtb_cmd: {theory!r}")
    cmd += ["--chrg", str(int(charge))]
    cmd += ["--uhf", str(int(uhf))]
    if solvent is not None:
        cmd += ["--alpb", solvent]
    if write_json:
        cmd += ["--json"]
    cmd += ["-P", str(int(threads))]
    return cmd


def slurm_threads_fallback() -> int | None:
    """Return a SLURM-derived thread count, or ``None`` if not set.

    Used when the user passes ``threads=auto`` (or 0). Honors
    ``SLURM_CPUS_PER_TASK`` first, then ``SLURM_CPUS_ON_NODE``, then
    ``OMP_NUM_THREADS``.
    """
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
        v = os.environ.get(key)
        if not v:
            continue
        try:
            n = int(v)
            if n > 0:
                return n
        except ValueError:
            continue
    return None


def resolve_threads(requested: int) -> int:
    """``requested`` if positive, else SLURM-derived fallback, else 1."""
    if requested > 0:
        return requested
    return slurm_threads_fallback() or 1


def find_first_xyz_path(upstream_manifest: Any) -> Path:
    """Pull the first xyz artifact path out of an upstream manifest.

    ``upstream_manifest`` is the :class:`~scripps_workflow.schema.Manifest`
    dataclass loaded by the framework. Raises if the bucket is empty, the
    record lacks a path, or the path doesn't exist on disk.
    """
    arts = getattr(upstream_manifest, "artifacts", None) or {}
    xyz_list = arts.get("xyz") or []
    if not isinstance(xyz_list, list) or not xyz_list:
        raise ValueError('upstream manifest has no "xyz" artifacts')

    a0 = xyz_list[0]
    if isinstance(a0, dict):
        path_str = a0.get("path_abs") or a0.get("path") or a0.get("path_rel")
    elif isinstance(a0, str):
        path_str = a0
    else:
        raise ValueError(
            f'upstream "xyz" artifact has unexpected type: {type(a0).__name__}'
        )

    if not path_str:
        raise ValueError('upstream "xyz" artifact missing path')

    p = Path(str(path_str))
    if not p.is_absolute():
        raise ValueError(f"upstream xyz path is not absolute: {path_str!r}")
    if not p.exists():
        raise FileNotFoundError(f"upstream xyz path does not exist: {p}")
    return p


# --------------------------------------------------------------------
# Subprocess core (monkeypatch target for tests)
# --------------------------------------------------------------------


def run_xtb(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    """Single xtb invocation. Returns ``(returncode, elapsed_seconds)``.

    ``stdout`` and ``stderr`` are written to the given paths so they can be
    recorded as artifacts. Tests monkeypatch this whole function (it is the
    only place we shell out to xtb).
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


_SP_FLAGS: dict[str, list[str]] = {
    "sp_energy": [],
    "sp_gradient": ["--grad"],
    "sp_hessian": ["--hess"],
}


class XtbCalc(Node):
    """Chain node: run xTB ops on the first xyz artifact from upstream."""

    step = "xtb_calc"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        # Raise on unknown values up-front so a typo becomes
        # ``argv_parse_failed`` rather than a silent wrong-theory run.
        return {
            "theory": normalize_theory(raw.get("theory", "GFN2-XTB")),
            "charge": parse_int(raw.get("charge"), 0),
            "unpaired_electrons": parse_int(raw.get("unpaired_electrons"), 0),
            "solvent": normalize_solvent(raw.get("solvent")),
            "opt_level": normalize_opt_level(raw.get("opt_level", "tight")),
            "calculations": parse_calculations(raw.get("calculations")),
            "threads": parse_int(raw.get("threads"), 0),
            "write_json": parse_bool(raw.get("write_json"), True),
        }

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config

        # Echo resolved config into manifest.inputs (sorted-ish for stable diffs).
        ctx.set_inputs(
            theory=cfg["theory"],
            charge=cfg["charge"],
            unpaired_electrons=cfg["unpaired_electrons"],
            solvent=cfg["solvent"],
            opt_level=cfg["opt_level"],
            calculations=list(cfg["calculations"]),
            threads_requested=cfg["threads"],
            write_json=cfg["write_json"],
        )

        if not cfg["calculations"]:
            ctx.fail("no_operations_requested")
            return

        # Locate xtb. The wrapper script does ``module load xtb`` so this
        # should always succeed in production; locally / in tests it tells
        # us when the env is wrong.
        xtb_exe = shutil.which("xtb")
        if not xtb_exe:
            ctx.fail("xtb_not_found_on_PATH")
            return
        ctx.manifest.environment["xtb"] = xtb_exe

        # Pull first xyz artifact from upstream manifest.
        if ctx.upstream_manifest is None:
            ctx.fail("no_upstream_manifest")
            return
        try:
            upstream_xyz = find_first_xyz_path(ctx.upstream_manifest)
        except Exception as e:
            ctx.fail(f"upstream_xyz_error: {e}")
            return

        # Stage a stable copy of the input geometry (so the per-op subdirs
        # don't all reach back to the upstream node's outputs).
        outputs_dir = ctx.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        staged_input = outputs_dir / "input.xyz"
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

        # Optimize first (if requested), then SP ops on the (possibly
        # optimized) current geometry.
        current_geom = staged_input

        if "optimize" in cfg["calculations"]:
            opt_ok, xtbopt_path = self._run_optimize(
                ctx, xtb_exe=xtb_exe, threads=threads, input_geom=current_geom
            )
            if opt_ok and xtbopt_path is not None:
                current_geom = xtbopt_path

        for op_key in ("sp_energy", "sp_gradient", "sp_hessian"):
            if op_key not in cfg["calculations"]:
                continue
            self._run_op(
                ctx,
                op_key=op_key,
                extra_flags=list(_SP_FLAGS[op_key]),
                input_geom=current_geom,
                xtb_exe=xtb_exe,
                threads=threads,
            )

    # -------------------- per-op helpers --------------------

    def _run_optimize(
        self,
        ctx: NodeContext,
        *,
        xtb_exe: str,
        threads: int,
        input_geom: Path,
    ) -> tuple[bool, Path | None]:
        """Run ``optimize``, publish ``xtbopt.xyz`` if produced.

        Returns ``(rc==0 and xtbopt produced, published_xtbopt_path or None)``.
        """
        cfg = ctx.config
        rc_ok = self._run_op(
            ctx,
            op_key="optimize",
            extra_flags=["--opt", cfg["opt_level"]],
            input_geom=input_geom,
            xtb_exe=xtb_exe,
            threads=threads,
        )

        run_dir = ctx.outputs_dir / "run" / "optimize"
        xtbopt_in_run = run_dir / "xtbopt.xyz"
        if not xtbopt_in_run.exists():
            ctx.fail("optimize_no_xtbopt: optimization did not produce xtbopt.xyz")
            return False, None

        published = ctx.outputs_dir / "xtbopt.xyz"
        shutil.copy2(xtbopt_in_run, published)
        ctx.add_artifact(
            "xyz",
            {
                "label": "xtbopt",
                "path_abs": str(published.resolve()),
                "sha256": sha256_file(published),
                "format": "xyz",
            },
        )
        return rc_ok, published

    def _run_op(
        self,
        ctx: NodeContext,
        *,
        op_key: str,
        extra_flags: list[str],
        input_geom: Path,
        xtb_exe: str,
        threads: int,
    ) -> bool:
        """Run a single xtb op in its own ``run/<op>/`` subdirectory.

        Records logs, optional xtbout.json, and an ``operations`` artifact
        record with the cmd / rc / elapsed metadata. Returns whether the op
        exited cleanly (rc == 0).
        """
        cfg = ctx.config
        run_dir = ctx.outputs_dir / "run" / op_key
        run_dir.mkdir(parents=True, exist_ok=True)
        run_input = run_dir / "input.xyz"
        shutil.copy2(input_geom, run_input)

        cmd = base_xtb_cmd(
            xtb_exe=xtb_exe,
            theory=cfg["theory"],
            charge=cfg["charge"],
            uhf=cfg["unpaired_electrons"],
            solvent=cfg["solvent"],
            threads=threads,
            write_json=cfg["write_json"],
        ) + list(extra_flags)

        stdout_path = run_dir / "xtb.stdout.txt"
        stderr_path = run_dir / "xtb.stderr.txt"

        logging_utils.log_info(f"xtb_calc: {op_key}: {' '.join(cmd)}")
        rc, elapsed = run_xtb(
            cmd, cwd=run_dir, stdout_path=stdout_path, stderr_path=stderr_path
        )

        # Record stdout/stderr as logs artifacts (so downstream / debugging
        # tools can grab them by label).
        for stream, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            if path.exists():
                ctx.add_artifact(
                    "logs",
                    {
                        "label": f"{op_key}_{stream}",
                        "path_abs": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "format": "txt",
                        "op": op_key,
                        "stream": stream,
                    },
                )

        # Publish xtbout.json (one per op) into outputs/ for easy downstream
        # access — keeps the per-op subdir as the raw xtb scratch space.
        xtbout = run_dir / "xtbout.json"
        if xtbout.exists():
            published = ctx.outputs_dir / f"xtbout_{op_key}.json"
            shutil.copy2(xtbout, published)
            ctx.add_artifact(
                "xtbout_json",
                {
                    "label": f"xtbout_{op_key}",
                    "path_abs": str(published.resolve()),
                    "sha256": sha256_file(published),
                    "format": "json",
                    "op": op_key,
                },
            )

        # Per-op metadata record. The ``path_abs`` field points at the run
        # directory so downstream tooling can rummage through the raw xtb
        # files without us having to enumerate every one as an artifact.
        ctx.add_artifact(
            "operations",
            {
                "label": op_key,
                "path_abs": str(run_dir.resolve()),
                "format": "dir",
                "op": op_key,
                "cmd": list(cmd),
                "returncode": int(rc),
                "elapsed_seconds": float(elapsed),
                "input_geom_abs": str(Path(input_geom).resolve()),
            },
        )

        if rc != 0:
            ctx.fail(f"xtb_op_failed: {op_key}: returncode={rc}")
            return False
        return True


# Re-export for engine entrypoint and tests that build raw-config dicts.
__all__ = [
    "ALPB_SOLVENTS",
    "OPT_LEVELS",
    "OPS_ORDER",
    "OP_LABEL_TO_TOKEN",
    "XtbCalc",
    "base_xtb_cmd",
    "find_first_xyz_path",
    "main",
    "normalize_opt_level",
    "normalize_solvent",
    "normalize_theory",
    "parse_calculations",
    "resolve_threads",
    "run_xtb",
    "slurm_threads_fallback",
]


main = XtbCalc.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
