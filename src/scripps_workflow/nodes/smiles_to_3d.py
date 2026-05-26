"""``wf-embed`` — single-SMILES 3D embed via RDKit ETKDGv3.

Source node (no upstream pointer). Reads a SMILES string from config,
embeds it in 3D using RDKit's ETKDGv3 with a deterministic seed, optionally
runs MMFF or UFF cleanup, and writes a single-frame ``.xyz`` file plus a
``wf.result.v1`` manifest.

Config keys (all key=value tokens, or one JSON object):

    smiles               (required)  SMILES string. Whitespace is stripped.
    name                 (optional)  Output file stem. Sanitized to
                                    ``[A-Za-z0-9._-]``; spaces become ``_``;
                                    empty/illegal → ``molecule``.
    opt                  (optional)  Geometry optimization: ``none|uff|mmff``.
                                    Default ``mmff`` runs MMFF only when
                                    parameters are available for every atom;
                                    otherwise RDKit force-field cleanup is
                                    skipped rather than falling back to UFF.
    seed                 (optional)  ETKDG random seed base. Default 0.
    max_embed_attempts   (optional)  How many seeds to try with deterministic
                                    coords before falling back to random
                                    coords. Default 50.
    max_opt_iters        (optional)  Optimizer iteration cap. Default 500.

Manifest shape: standard ``wf.result.v1`` envelope. The xyz artifact bucket
gets one record:

    {"label": "embed_xyz", "path_abs": ..., "sha256": ..., "format": "xyz",
     "name": ..., "smiles": ..., "num_atoms": ..., "num_heavy_atoms": ...}

The RDKit-using code is split out into :func:`build_3d_mol` and
:func:`mol_to_xyz_block` so that tests can monkeypatch the embed step
without needing RDKit installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .. import logging_utils
from ..config_schema import ConfigField, NodeSchema, apply_schema
from ..hashing import sha256_file
from ..node import Node, NodeContext

# Lazy-imported RDKit handles. Populated on first call to ``ensure_rdkit``.
# Kept module-level so tests can monkeypatch them.
_Chem: Any = None
_AllChem: Any = None
_RDKIT_PATH: str | None = None


def ensure_rdkit() -> None:
    """Import RDKit lazily, with a clear diagnostic on the env-mismatch case.

    The Scripps cluster has historically suffered from a Python-3.11 process
    importing a Python-3.8 RDKit build via stale ``PYTHONPATH``. The strict
    env guard in ``render_wrapper`` should prevent that, but we double-check
    here because the failure mode (silent garbage geometries) is sneaky.
    """
    global _Chem, _AllChem, _RDKIT_PATH
    if _Chem is not None and _AllChem is not None:
        return

    try:
        import rdkit  # type: ignore[import-not-found]
        from rdkit import Chem as _C  # type: ignore[import-not-found]
        from rdkit.Chem import AllChem as _AC  # type: ignore[import-not-found]
    except Exception as e:
        import sys as _sys

        raise RuntimeError(
            "Failed to import RDKit. Most likely cause: the workflow node is "
            "running under the wrong Python environment, or PYTHONPATH/PYTHONHOME "
            "leaked from the engine shell. The strict env guard in "
            "scripps_workflow.env.render_wrapper is supposed to prevent this; "
            "if it's still happening, check the wrapper script being used.\n"
            f"Python executable: {_sys.executable}\n"
            f"Original import error: {e}"
        ) from e

    rdkit_path = str(Path(rdkit.__file__).resolve())
    if "/opt/applications/rdkit/" in rdkit_path and "python3.8" in rdkit_path:
        import sys as _sys

        raise RuntimeError(
            "RDKit import mismatch detected: a Python-3.8 RDKit build was "
            "loaded into a different Python runtime. This produces silently "
            "wrong geometries. Re-launch via a wrapper that unsets PYTHONPATH "
            "and PYTHONHOME before invoking the workflow Python.\n"
            f"Python executable: {_sys.executable}\n"
            f"Imported RDKit from: {rdkit_path}"
        )

    _Chem = _C
    _AllChem = _AC
    _RDKIT_PATH = rdkit_path


def get_rdkit_path() -> str | None:
    """Return the resolved RDKit module path (set after :func:`ensure_rdkit`)."""
    return _RDKIT_PATH


# --------------------------------------------------------------------
# Filename sanitization
# --------------------------------------------------------------------


_FILENAME_BAD = re.compile(r"[^A-Za-z0-9._-]+")
_FILENAME_DEDUP = re.compile(r"_+")


def sanitize_filename(stem: str | None) -> str:
    """Reduce arbitrary user input to a safe filename stem.

    Spaces become underscores; everything outside ``[A-Za-z0-9._-]`` becomes
    a single underscore; runs of underscores collapse; leading/trailing
    ``._-`` are stripped. Empty/all-illegal input collapses to ``molecule``.
    """
    s = (stem or "").strip()
    if not s:
        return "molecule"
    s = s.replace(" ", "_")
    s = _FILENAME_BAD.sub("_", s)
    s = _FILENAME_DEDUP.sub("_", s).strip("._-")
    return s or "molecule"


def _unique_xyz_path(xyz_dir: Path, stem: str) -> Path:
    """Return ``<xyz_dir>/<stem>.xyz`` or ``<stem>_2.xyz``, ``..._3.xyz``, ... if taken."""
    candidate = xyz_dir / f"{stem}.xyz"
    if not candidate.exists():
        return candidate
    k = 2
    while True:
        candidate = xyz_dir / f"{stem}_{k}.xyz"
        if not candidate.exists():
            return candidate
        k += 1


# --------------------------------------------------------------------
# RDKit-using core (split out so tests can monkeypatch)
# --------------------------------------------------------------------


def _embed_with_etkdg(mol: Any, seed: int, use_random_coords: bool) -> int:
    """Single ETKDGv3 attempt. Returns RDKit's int status (0 = success)."""
    ensure_rdkit()
    assert _AllChem is not None

    params = _AllChem.ETKDGv3()
    if hasattr(params, "randomSeed"):
        params.randomSeed = int(seed)
    if hasattr(params, "useRandomCoords"):
        params.useRandomCoords = bool(use_random_coords)

    try:
        return int(_AllChem.EmbedMolecule(mol, params))
    except TypeError:
        # Fall back to old-style kwargs API on older RDKit builds.
        kwargs: dict[str, Any] = {"randomSeed": int(seed)}
        if use_random_coords:
            kwargs["useRandomCoords"] = True
        return int(_AllChem.EmbedMolecule(mol, **kwargs))


def build_3d_mol(
    smiles: str,
    *,
    seed: int = 0,
    max_embed_attempts: int = 50,
    opt: str = "mmff",
    max_opt_iters: int = 500,
) -> Any:
    """Parse SMILES, embed to 3D via ETKDGv3, optionally optimize.

    Two-pass embed: deterministic coords first (seeds ``[seed, seed+N)``);
    if every attempt fails, retry with random coords (seeds ``[seed+1e5,
    seed+1e5+N)``). RuntimeError if both passes fail.

    ``opt`` choices:
        * ``none`` — skip the cleanup step.
        * ``uff`` — UFF optimization with ``max_opt_iters`` cap.
        * ``mmff`` — MMFF if parameters are available for every atom in the
          molecule (RDKit's ``MMFFHasAllMoleculeParams``); otherwise skip
          RDKit force-field cleanup. This is the default because UFF fallback
          can badly distort metal complexes or molecules with unsupported
          atom types.

    The function returns the RDKit ``Mol`` with one embedded conformer. It
    is the caller's job to turn that into a coordinate file; see
    :func:`mol_to_xyz_block`.

    Tests can monkeypatch this whole function to return a stub Mol, since
    the orchestration layer (``SmilesTo3D.run``) treats it as opaque.
    """
    ensure_rdkit()
    assert _Chem is not None
    assert _AllChem is not None

    mol = _Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    mol = _Chem.AddHs(mol)

    status = 1
    for j in range(int(max_embed_attempts)):
        status = _embed_with_etkdg(mol, seed=seed + j, use_random_coords=False)
        if status == 0:
            break
    if status != 0:
        # Random-coord fallback — slow but more robust on stubborn systems.
        for j in range(int(max_embed_attempts)):
            status = _embed_with_etkdg(
                mol, seed=seed + 100_000 + j, use_random_coords=True
            )
            if status == 0:
                break
    if status != 0:
        raise RuntimeError(f"3D embedding failed for SMILES: {smiles!r}")

    opt_norm = (opt or "").lower().strip()
    if opt_norm not in {"none", "uff", "mmff"}:
        raise ValueError(f"opt must be one of: none, uff, mmff (got {opt!r})")

    _apply_forcefield_cleanup(
        mol,
        opt=opt_norm,
        max_opt_iters=int(max_opt_iters),
    )

    return mol


def _apply_forcefield_cleanup(
    mol: Any,
    *,
    opt: str,
    max_opt_iters: int,
) -> str:
    """Apply the requested RDKit force-field cleanup to an embedded molecule.

    Returns a short method label for tests/logging:

    * ``"none"`` when cleanup was explicitly disabled.
    * ``"mmff"`` when MMFF was actually run.
    * ``"mmff_skipped_missing_params"`` when ``opt=mmff`` but RDKit reports
      incomplete MMFF coverage. This is intentionally *not* a UFF fallback,
      because UFF cleanup can severely distort metal-containing complexes.
    * ``"uff"`` when UFF was explicitly requested.
    """
    assert _AllChem is not None

    opt_norm = (opt or "").lower().strip()
    if opt_norm == "none":
        return "none"

    if opt_norm == "mmff":
        try:
            has_mmff = bool(_AllChem.MMFFHasAllMoleculeParams(mol))
        except Exception as exc:
            logging_utils.log_info(
                "smiles_to_3d: MMFF parameter check failed; "
                f"skipping RDKit force-field cleanup ({exc})"
            )
            return "mmff_skipped_missing_params"

        if has_mmff:
            _AllChem.MMFFOptimizeMolecule(mol, maxIters=int(max_opt_iters))
            return "mmff"

        logging_utils.log_info(
            "smiles_to_3d: MMFF parameters unavailable for at least one atom; "
            "skipping RDKit force-field cleanup instead of falling back to UFF"
        )
        return "mmff_skipped_missing_params"

    if opt_norm == "uff":
        _AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_opt_iters))
        return "uff"

    raise ValueError(f"opt must be one of: none, uff, mmff (got {opt!r})")


def mol_to_xyz_block(mol: Any, comment: str = "") -> str:
    """Serialize an RDKit ``Mol`` (one conformer) as an XYZ-format block.

    Standard XYZ: line 1 = atom count, line 2 = comment, then one line per
    atom with ``element x y z`` (8-decimal fixed-point).
    """
    conf = mol.GetConformer()
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    lines = [str(len(atoms)), comment]
    for i, sym in enumerate(atoms):
        pos = conf.GetAtomPosition(i)
        lines.append(f"{sym:2s} {pos.x: .8f} {pos.y: .8f} {pos.z: .8f}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------
# Schema (source of truth for parse_config + auto-generated docs)
# --------------------------------------------------------------------


def _name_coercer(value: Any) -> Any:
    """Strip and treat whitespace-only / empty as None.

    apply_schema already short-circuits raw ``""``/``None`` to the field
    default, but ``"   "`` (whitespace-only) slips through as "present".
    Convert that to ``None`` so callers can ``cfg["name"] or "molecule"``.
    """
    s = str(value).strip()
    return s if s else None


def _opt_coercer(value: Any) -> str:
    """Lowercase + validate the ETKDG/MMFF/UFF optimization mode."""
    s = str(value).strip().lower()
    if s not in {"none", "uff", "mmff"}:
        raise ValueError(
            f"must be one of 'none', 'uff', 'mmff', got {value!r}"
        )
    return s


SCHEMA = NodeSchema(
    step_name="smiles_to_3d",
    cli_entrypoint="wf-embed",
    module_path="scripps_workflow.nodes.smiles_to_3d",
    overview=(
        "Source node — embeds a single SMILES into 3D coordinates via "
        "RDKit ETKDGv3 with a deterministic seed and (optionally) MMFF "
        "or UFF cleanup. Emits a single ``.xyz`` file and a "
        "``wf.result.v1`` manifest."
    ),
    fields=(
        ConfigField(
            name="smiles",
            type="str",
            description=(
                "SMILES string. Whitespace-stripped; must be non-empty."
            ),
        ),
        ConfigField(
            name="name",
            type="str",
            default=None,
            coercer=_name_coercer,
            description=(
                "Output filename stem. Sanitized to "
                "``[A-Za-z0-9._-]``; spaces become ``_``. "
                "Whitespace-only or empty becomes ``None`` and the "
                "node falls back to ``molecule``."
            ),
        ),
        ConfigField(
            name="opt",
            type="str",
            default="mmff",
            choices=("none", "uff", "mmff"),
            coercer=_opt_coercer,
            description=(
                "Geometry optimization mode. ``mmff`` (default) runs "
                "MMFF only when parameters are available for every atom; "
                "otherwise RDKit force-field cleanup is skipped rather "
                "than falling back to UFF. Use ``uff`` only when you "
                "explicitly want UFF cleanup."
            ),
        ),
        ConfigField(
            name="seed",
            type="int",
            default=0,
            description=(
                "ETKDGv3 random seed base. Bumped deterministically "
                "across embed attempts."
            ),
        ),
        ConfigField(
            name="max_embed_attempts",
            type="int",
            default=50,
            min_value=1,
            description=(
                "Number of seeds to try with deterministic coords "
                "before falling back to random coordinates."
            ),
        ),
        ConfigField(
            name="max_opt_iters",
            type="int",
            default=500,
            min_value=0,
            description="Optimizer iteration cap.",
        ),
    ),
)


# --------------------------------------------------------------------
# Node class
# --------------------------------------------------------------------


class SmilesTo3D(Node):
    """Source node: SMILES → 3D ``.xyz`` via RDKit ETKDGv3."""

    step = "smiles_to_3d"
    accepts_upstream = False
    requires_upstream = False

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        return apply_schema(raw, SCHEMA)

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config

        smiles = cfg.get("smiles")
        if not smiles:
            ctx.fail("missing_required_input: 'smiles' config key is required")
            return

        ctx.set_inputs(
            smiles=smiles,
            name=cfg.get("name"),
            opt=cfg.get("opt", "mmff"),
            seed=cfg.get("seed", 0),
            max_embed_attempts=cfg.get("max_embed_attempts", 50),
            max_opt_iters=cfg.get("max_opt_iters", 500),
        )

        stem = sanitize_filename(cfg.get("name") or "molecule")
        xyz_dir = ctx.outputs_dir / "xyz"
        xyz_dir.mkdir(parents=True, exist_ok=True)
        xyz_path = _unique_xyz_path(xyz_dir, stem)

        try:
            mol = build_3d_mol(
                smiles,
                seed=int(cfg.get("seed", 0)),
                max_embed_attempts=int(cfg.get("max_embed_attempts", 50)),
                opt=str(cfg.get("opt", "mmff")),
                max_opt_iters=int(cfg.get("max_opt_iters", 500)),
            )
        except Exception as e:
            ctx.fail(f"embed_failed: {e}")
            return

        rdkit_path = get_rdkit_path()
        if rdkit_path is not None:
            ctx.manifest.environment.setdefault("rdkit_path", rdkit_path)
            logging_utils.log_info(f"smiles_to_3d: rdkit={rdkit_path}")

        comment = f"{xyz_path.stem} | {smiles}"
        xyz_text = mol_to_xyz_block(mol, comment=comment)
        xyz_path.write_text(xyz_text, encoding="utf-8")

        ctx.add_artifact(
            "xyz",
            {
                "label": "embed_xyz",
                "path_abs": str(xyz_path.resolve()),
                "sha256": sha256_file(xyz_path),
                "format": "xyz",
                "name": xyz_path.stem,
                "smiles": smiles,
                "num_atoms": int(mol.GetNumAtoms()),
                "num_heavy_atoms": int(mol.GetNumHeavyAtoms()),
            },
        )

        logging_utils.log_info(f"smiles_to_3d: wrote {xyz_path.resolve()}")


main = SmilesTo3D.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
