"""``wf-prism`` — RMSD / MoI conformer screening via Prism Pruner.

Chain node. Consumes a conformer ensemble from upstream and prunes
duplicate / equivalent geometries down to a non-redundant set, then
publishes the result in the shape required by the ``conformer_screen``
role contract (so this node and ``marc_screen`` can be swapped without
downstream changes).

Conformer source discovery — the upstream's pointer is followed and the
manifest's artifact buckets are tried in priority order::

    accepted -> selected -> conformers -> xyz_ensemble -> xyz

The first non-empty bucket whose paths exist on disk wins. This makes
prism work both directly after ``crest`` (which fills ``conformers``)
and after another screen (which fills ``accepted``).

Energy harvesting is two-stage: per-conformer artifact items are checked
for ``rel_energy_kcal`` / ``rel_energy``; missing values are filled from
an upstream ``artifacts.files`` entry whose label contains "energies" or
whose filename ends in ``.energies``.

Config keys (``key=value`` tokens or one JSON object):

    min_conformers          int. Skip pruning and accept all when n_input
                            < this. (default 3)
    moi_pruning             bool. Run the MoI (moment-of-inertia) stage.
                            (default true)
    rmsd_pruning            bool. Run the RMSD stage. (default true)
    rot_corr_rmsd_pruning   bool. Run the rotation-corrected RMSD stage.
                            (default true)
    use_energies            auto | true | false. ``auto`` uses energies
                            iff every input has one. (default auto)
    max_dE_kcal             float. prism-pruner's *internal* energy gate
                            for the energy-aware geometric comparison.
                            (default 0.5)
    ewin_kcal               float >= 0. *Pre-pruning* energy cutoff:
                            drop conformers more than this many kcal/mol
                            above the lowest-energy input before the
                            pruner sees them. Mirrors the navicat-marc
                            / crest ``ewin_kcal`` convention. Silently
                            skipped when ``use_energies`` resolves to
                            false (no energy reference). (default 5.0)
    timeout_s               int >= 5. Per-stage timeout. (default 120)
    keep_rejected           bool. Copy rejected conformers to
                            ``outputs/rejected/``. (default true)

The subprocess-equivalent core is :func:`run_prism_pruner`; tests
monkeypatch it to avoid needing prism-pruner installed (it requires
Python >= 3.12 and numpy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .. import logging_utils
from ..contracts.conformer_screen import validate_conformer_screen
from ..hashing import sha256_file
from ..node import Node, NodeContext
from ..parsing import parse_bool, parse_float, parse_int

# TODO(factor): when a fourth caller appears, lift split_multixyz /
# write_xyz_block / XyzBlock into ``scripps_workflow.xyz``. For now the
# crest module is the source of truth and prism imports them.
from .crest import XyzBlock, split_multixyz, write_xyz_block

#: Identifier recorded into ``inputs.method`` so a downstream consumer
#: can branch on which screen impl produced the manifest.
METHOD_NAME = "prism_pruner"

#: ``rejected_reason`` value used for conformers prism-pruner deduplicated.
#: prism-pruner returns a single mask without per-conformer reasons, so the
#: best we can do is a generic label that satisfies the contract while
#: honestly representing the lack of granularity.
REJECTED_REASON = "pruned_by_prism_pruner"

#: ``rejected_reason`` value for conformers dropped by the pre-pruning
#: ``ewin_kcal`` filter (above the energy window vs the lowest-energy
#: input). Distinct from :data:`REJECTED_REASON` so downstream consumers
#: can tell duplicate-by-geometry rejections apart from energy-window
#: rejections.
REJECTED_REASON_EWIN = "above_energy_window"


# --------------------------------------------------------------------
# Pure helpers (testable)
# --------------------------------------------------------------------


import shutil


def normalize_use_energies(v: Any) -> str:
    """Tri-state coercion. Returns one of ``"auto" | "true" | "false"``.

    ``None`` / empty / unrecognized → ``"auto"``.
    """
    if v is None:
        return "auto"
    s = str(v).strip().lower()
    if s in {"", "auto"}:
        return "auto"
    if s in {"true", "1", "yes", "y", "on"}:
        return "true"
    if s in {"false", "0", "no", "n", "off"}:
        return "false"
    return "auto"


def _artifact_items(upstream_artifacts: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return list-of-dict items from an artifact bucket, normalizing
    string entries to ``{"path_abs": s}``."""
    items = upstream_artifacts.get(key)
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(it)
        elif isinstance(it, str):
            out.append({"path_abs": it})
    return out


def _item_path(it: dict[str, Any]) -> str | None:
    p = it.get("path_abs") or it.get("path") or it.get("path_rel")
    return p if isinstance(p, str) else None


def discover_conformer_sources(
    upstream_artifacts: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Pick the highest-priority non-empty conformer source from an
    upstream artifacts dict.

    Returns ``(mode, items)`` where ``mode`` is one of:

        ``"many"``      one record per conformer (accepted/selected/conformers)
        ``"ensemble"``  a single multi-xyz file (xyz_ensemble)
        ``"single"``    a single xyz frame (xyz)
        ``"none"``      nothing usable found

    Only items whose path exists on disk are kept; ``path_abs`` is
    resolved to an absolute path.
    """
    for key in ("accepted", "selected", "conformers"):
        items = _artifact_items(upstream_artifacts, key)
        kept: list[dict[str, Any]] = []
        for it in items:
            p = _item_path(it)
            if not p or not p.endswith(".xyz") or not Path(p).exists():
                continue
            it2 = dict(it)
            it2["path_abs"] = str(Path(p).resolve())
            kept.append(it2)
        if kept:
            return "many", kept

    for it in _artifact_items(upstream_artifacts, "xyz_ensemble"):
        p = _item_path(it)
        if p and p.endswith(".xyz") and Path(p).exists():
            return "ensemble", [{"path_abs": str(Path(p).resolve())}]

    for it in _artifact_items(upstream_artifacts, "xyz"):
        p = _item_path(it)
        if p and p.endswith(".xyz") and Path(p).exists():
            return "single", [{"path_abs": str(Path(p).resolve())}]

    return "none", []


def extract_energy_kcal_from_item(it: dict[str, Any]) -> float | None:
    """Pull ``rel_energy_kcal`` (preferred) or ``rel_energy`` from a
    per-conformer item. Returns ``None`` if absent or unparseable."""
    for key in ("rel_energy_kcal", "rel_energy"):
        if key in it:
            try:
                return float(it[key])
            except (TypeError, ValueError):
                pass
    return None


def find_energy_file(upstream_artifacts: dict[str, Any]) -> Path | None:
    """Look for a *.energies file in ``artifacts.files`` (label contains
    'energies' or filename ends with '.energies')."""
    for it in _artifact_items(upstream_artifacts, "files"):
        p = _item_path(it)
        if not p or not Path(p).exists():
            continue
        label = str(it.get("label") or "").lower()
        pp = Path(p)
        if "energies" in label or pp.name.endswith(".energies"):
            return pp.resolve()
    return None


def parse_generic_energies_kcal(path: Path) -> list[float]:
    """Parse a crest-style or orca-style energies file: last float on
    each non-empty line. Lines without floats are dropped (this differs
    from the crest helper which yields ``None`` for those — for prism
    purposes we only consume the file when it has at least n_input
    parseable rows)."""
    out: list[float] = []
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
            out.append(float(floats[-1]))
    return out


def concat_xyz_files(paths: Iterable[Path], out_path: Path) -> None:
    """Concatenate per-conformer xyz files into a single multi-xyz blob.

    Each input is appended verbatim with a trailing newline if missing.
    """
    chunks: list[str] = []
    for p in paths:
        t = Path(p).read_text(encoding="utf-8", errors="replace")
        if not t.endswith("\n"):
            t += "\n"
        chunks.append(t)
    out_path.write_text("".join(chunks), encoding="utf-8")


def resolve_use_energies(mode: str, *, have_energies: bool) -> bool:
    """Map the tri-state string + availability to a concrete bool.

    ``"true"`` requires energies (caller must check; this function does
    not raise). ``"false"`` ignores. ``"auto"`` uses iff available.
    """
    if mode == "true":
        return True
    if mode == "false":
        return False
    return have_energies


def apply_ewin_filter(
    energies_kcal: list[float | None],
    ewin_kcal: float,
) -> list[bool]:
    """Pre-pruning energy cutoff. Returns a length-N mask where ``True``
    means "conformer i is within ``ewin_kcal`` of the lowest-energy
    member of ``energies_kcal``" (closed interval, so a value exactly at
    the threshold is kept).

    Conformers with missing energies (``None``) are kept (mask ``True``)
    on the principle that absent energies are not grounds for *energy*-
    based exclusion — that decision belongs to the caller (typically by
    requiring use_energies to resolve True before applying this filter).
    If every entry is missing or the list is empty, the filter is a
    no-op.

    The reference is the *minimum* over present energies, not the first
    one — caller doesn't have to pre-sort. ``ewin_kcal`` is taken at
    face value; negatives produce an all-False mask (except possibly the
    minimum), which is almost always a config bug. The :class:`PrismScreen`
    node validates ``ewin_kcal >= 0`` in ``parse_config`` so the bug
    surfaces as ``argv_parse_failed`` before this is reached.
    """
    floats = [e for e in energies_kcal if isinstance(e, float)]
    if not floats:
        return [True] * len(energies_kcal)
    e_min = min(floats)
    out: list[bool] = []
    for e in energies_kcal:
        if isinstance(e, float):
            out.append((e - e_min) <= float(ewin_kcal))
        else:
            out.append(True)
    return out


# --------------------------------------------------------------------
# Library-using core (monkeypatch target for tests)
# --------------------------------------------------------------------


def run_prism_pruner(
    *,
    ensemble_path: Path,
    energies_kcal: list[float] | None,
    moi_pruning: bool,
    rmsd_pruning: bool,
    rot_corr_rmsd_pruning: bool,
    max_dE_kcal: float,
    timeout_s: int,
    log_fn,
) -> list[bool]:
    """Run prism-pruner and return the accept mask as a Python list of bools.

    Imports prism-pruner internally — the package requires Python >= 3.12
    and numpy, neither of which is required at the framework's test
    runtime. The whole function is the monkeypatch target in tests.

    Args:
        ensemble_path: Path to the multi-xyz file fed to
            :class:`prism_pruner.conformer_ensemble.ConformerEnsemble`.
        energies_kcal: Per-conformer relative energies (kcal/mol), or
            ``None`` to disable energy-gated comparisons.
        moi_pruning, rmsd_pruning, rot_corr_rmsd_pruning: Stage toggles
            forwarded to ``prism_pruner.pruner.prune``.
        max_dE_kcal: Energy window for energy-gated comparisons. Ignored
            when ``energies_kcal is None``.
        timeout_s: Per-stage timeout, forwarded to ``prune``.
        log_fn: Callback used as prism-pruner's ``debugfunction``. We
            wrap stderr logging there.

    Returns:
        ``list[bool]`` of length ``n_input``; ``True`` means "kept".
    """
    # Imports are inside the function so the framework / tests can
    # operate without prism-pruner installed.
    from prism_pruner.conformer_ensemble import ConformerEnsemble  # type: ignore
    from prism_pruner.pruner import prune  # type: ignore

    import numpy as np  # type: ignore

    ensemble = ConformerEnsemble.from_xyz(str(ensemble_path), read_energies=False)

    energies_arr = None
    max_dE = 0.0
    if energies_kcal is not None:
        energies_arr = np.array([float(e) for e in energies_kcal], dtype=float)
        max_dE = float(max_dE_kcal)

    _, mask = prune(
        ensemble.coords,
        ensemble.atoms,
        moi_pruning=bool(moi_pruning),
        rmsd_pruning=bool(rmsd_pruning),
        rot_corr_rmsd_pruning=bool(rot_corr_rmsd_pruning),
        energies=energies_arr,
        max_dE=float(max_dE),
        timeout_s=int(timeout_s),
        debugfunction=log_fn,
    )
    return [bool(x) for x in mask.tolist()]


# --------------------------------------------------------------------
# Node class
# --------------------------------------------------------------------


class PrismScreen(Node):
    """Chain node: prune duplicate conformers via prism-pruner."""

    step = "prism_screen"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        # Normalize-and-validate up front so typos surface as
        # ``argv_parse_failed`` rather than mid-run errors.
        min_conformers = parse_int(raw.get("min_conformers"), 3)
        if min_conformers < 1:
            raise ValueError("min_conformers must be >= 1")

        max_dE = parse_float(raw.get("max_dE_kcal", raw.get("energy_window_kcal")), 0.5)
        if max_dE < 0:
            raise ValueError("max_dE_kcal must be >= 0")

        timeout_s = parse_int(raw.get("timeout_s"), 120)
        if timeout_s < 5:
            raise ValueError("timeout_s must be >= 5")

        ewin_kcal = parse_float(raw.get("ewin_kcal"), 5.0)
        if ewin_kcal < 0:
            raise ValueError("ewin_kcal must be >= 0")

        return {
            "min_conformers": min_conformers,
            "moi_pruning": parse_bool(raw.get("moi_pruning"), True),
            "rmsd_pruning": parse_bool(raw.get("rmsd_pruning"), True),
            "rot_corr_rmsd_pruning": parse_bool(raw.get("rot_corr_rmsd_pruning"), True),
            "use_energies": normalize_use_energies(raw.get("use_energies")),
            "max_dE_kcal": float(max_dE),
            "ewin_kcal": float(ewin_kcal),
            "timeout_s": int(timeout_s),
            "keep_rejected": parse_bool(raw.get("keep_rejected"), True),
        }

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config

        if ctx.upstream_manifest is None:
            ctx.fail("no_upstream_manifest")
            return

        # ---------- 1. discover conformer source ----------
        up_arts = getattr(ctx.upstream_manifest, "artifacts", None) or {}
        mode, items = discover_conformer_sources(up_arts)
        if mode == "none" or not items:
            ctx.fail("no_xyz_inputs_found_in_upstream_manifest")
            return

        # ---------- 2. stage inputs into outputs/input_conformers/ ----------
        outputs_dir = ctx.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        staged_dir = outputs_dir / "input_conformers"
        staged_dir.mkdir(parents=True, exist_ok=True)

        staged_paths, energies_kcal = self._stage_inputs(
            mode=mode, items=items, staged_dir=staged_dir
        )

        n_input = len(staged_paths)
        if n_input == 0:
            ctx.fail("no_staged_conformers")
            return

        # Fill missing energies from an upstream energies file if all
        # per-item values were absent.
        if all(e is None for e in energies_kcal):
            energy_file = find_energy_file(up_arts)
            if energy_file is not None:
                parsed = parse_generic_energies_kcal(energy_file)
                if len(parsed) >= n_input:
                    energies_kcal = list(parsed[:n_input])
                    ctx.add_artifact(
                        "files",
                        {
                            "label": "upstream_energies_used",
                            "path_abs": str(energy_file),
                            "sha256": sha256_file(energy_file),
                            "format": "txt",
                        },
                    )
                    logging_utils.log_info(
                        f"prism_screen: loaded {n_input} energies from {energy_file}"
                    )
                else:
                    logging_utils.log_warn(
                        f"prism_screen: energy file {energy_file} had only "
                        f"{len(parsed)} values (need >= {n_input}); ignoring."
                    )

        have_energies = all(isinstance(e, float) for e in energies_kcal)
        use_energies = resolve_use_energies(
            cfg["use_energies"], have_energies=have_energies
        )
        if cfg["use_energies"] == "true" and not have_energies:
            ctx.fail(
                "use_energies_required_but_unavailable: "
                f"only {sum(1 for e in energies_kcal if isinstance(e, float))}/"
                f"{n_input} conformers had energies"
            )
            return

        # ---------- 3. ewin filter (pre-pruning energy cutoff) ----------
        # Drop conformers more than ewin_kcal above the lowest-energy
        # input BEFORE feeding them to prism-pruner. Mirrors the
        # navicat-marc / crest convention. Only applied when energies
        # are actually flowing through this run -- if use_energies
        # resolved to False there's no energy reference to compare
        # against, and the filter is silently skipped (consistent with
        # the user's stated intent).
        ewin_applied = bool(use_energies and have_energies)
        if ewin_applied:
            ewin_keep_mask = apply_ewin_filter(
                [e if isinstance(e, float) else None for e in energies_kcal],
                cfg["ewin_kcal"],
            )
        else:
            ewin_keep_mask = [True] * n_input
        n_above_ewin = sum(1 for k in ewin_keep_mask if not k)
        if ewin_applied and n_above_ewin:
            logging_utils.log_info(
                f"prism_screen: ewin_kcal={cfg['ewin_kcal']:.3f} dropped "
                f"{n_above_ewin}/{n_input} conformers above the energy window"
            )

        # The pruner only sees conformers that survived the ewin filter.
        # We keep the original 1..n_input index space so per-conformer
        # artifact records (and the rejection-reason split) stay stable.
        within_paths = [p for p, k in zip(staged_paths, ewin_keep_mask) if k]
        within_energies = [
            e for e, k in zip(energies_kcal, ewin_keep_mask) if k
        ]
        n_within = len(within_paths)

        # ---------- 4. concat ensemble (within-window only) ----------
        run_dir = outputs_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        ensemble_path = run_dir / "input_ensemble.xyz"
        if n_within > 0:
            concat_xyz_files(within_paths, ensemble_path)
            ctx.add_artifact(
                "files",
                {
                    "label": "input_ensemble",
                    "path_abs": str(ensemble_path.resolve()),
                    "sha256": sha256_file(ensemble_path),
                    "format": "xyz",
                },
            )

        # ---------- 5. prune (or skip if too few survived ewin) ----------
        if n_within == 0:
            # Everyone was filtered out. Pruner has nothing to do; we'll
            # rely on the empty-accepted check below to mark ok=false.
            prune_mask: list[bool] = []
        elif n_within < cfg["min_conformers"]:
            logging_utils.log_info(
                f"prism_screen: n_within={n_within} < min_conformers="
                f"{cfg['min_conformers']}; skipping pruning, accepting all "
                f"survivors of the ewin filter."
            )
            prune_mask = [True] * n_within
        else:
            try:
                prune_mask = run_prism_pruner(
                    ensemble_path=ensemble_path,
                    energies_kcal=(
                        [float(e) for e in within_energies]
                        if use_energies and have_energies
                        else None
                    ),
                    moi_pruning=cfg["moi_pruning"],
                    rmsd_pruning=cfg["rmsd_pruning"],
                    rot_corr_rmsd_pruning=cfg["rot_corr_rmsd_pruning"],
                    max_dE_kcal=cfg["max_dE_kcal"],
                    timeout_s=cfg["timeout_s"],
                    log_fn=lambda m: logging_utils.log_info(f"[prism] {m}"),
                )
            except ImportError as e:
                ctx.fail(f"import_prism_pruner_failed: {e}")
                prune_mask = [True] * n_within
            except Exception as e:
                ctx.fail(f"pruning_failed_fallback_accept_all: {e}")
                prune_mask = [True] * n_within

            if len(prune_mask) != n_within:
                ctx.fail(
                    f"pruning_returned_unexpected_mask_length: "
                    f"expected {n_within}, got {len(prune_mask)}"
                )
                prune_mask = [True] * n_within

        # ---------- 6. materialize accepted / rejected ----------
        # Combine masks: a conformer is accepted iff it survived BOTH
        # the ewin filter AND the pruner. Rejection reason is recorded
        # so downstream consumers can tell ewin-rejects from pruner-rejects.
        accepted_dir = outputs_dir / "accepted"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir = outputs_dir / "rejected"
        if cfg["keep_rejected"]:
            rejected_dir.mkdir(parents=True, exist_ok=True)

        prune_iter = iter(prune_mask)
        accepted_paths: list[Path] = []
        rejected_records: list[tuple[Path, str]] = []
        for i, (src, ewin_ok) in enumerate(
            zip(staged_paths, ewin_keep_mask), start=1
        ):
            if not ewin_ok:
                if cfg["keep_rejected"]:
                    dst = rejected_dir / f"conf_{i:04d}.xyz"
                    shutil.copy2(src, dst)
                    rejected_records.append((dst, REJECTED_REASON_EWIN))
                continue
            keep = next(prune_iter)
            if keep:
                dst = accepted_dir / f"conf_{i:04d}.xyz"
                shutil.copy2(src, dst)
                accepted_paths.append(dst)
            elif cfg["keep_rejected"]:
                dst = rejected_dir / f"conf_{i:04d}.xyz"
                shutil.copy2(src, dst)
                rejected_records.append((dst, REJECTED_REASON))

        # Synthesize an accept_mask for the downstream blocks that still
        # use it (energy harvest + index lookup). This is the COMBINED
        # mask spanning the original 1..n_input space.
        accept_mask: list[bool] = []
        prune_iter2 = iter(prune_mask)
        for ewin_ok in ewin_keep_mask:
            if not ewin_ok:
                accept_mask.append(False)
            else:
                accept_mask.append(next(prune_iter2))

        # Compute accepted-only relative energies.
        accepted_indices = [
            i for i, keep in enumerate(accept_mask, start=1) if keep
        ]
        accepted_energies_abs: list[float | None] = [
            energies_kcal[i - 1] for i in accepted_indices
        ]
        e_min: float | None = None
        if any(isinstance(e, float) for e in accepted_energies_abs):
            e_min = min(e for e in accepted_energies_abs if isinstance(e, float))

        for p in accepted_paths:
            idx = int(p.stem.split("_")[-1])
            rec: dict[str, Any] = {
                "index": idx,
                "label": p.stem,
                "path_abs": str(p.resolve()),
                "sha256": sha256_file(p),
                "format": "xyz",
            }
            e = energies_kcal[idx - 1] if idx - 1 < len(energies_kcal) else None
            if isinstance(e, float):
                # 'energy_kcal' is the absolute (or upstream-relative)
                # value; 'rel_energy_kcal' is shifted to the lowest
                # accepted, which is what downstream consumers
                # (Boltzmann weighting, screen-of-screens) expect.
                rec["energy_kcal"] = float(e)
                if e_min is not None:
                    rec["rel_energy_kcal"] = float(e - e_min)
            ctx.add_artifact("accepted", rec)

        for p, reason in rejected_records:
            idx = int(p.stem.split("_")[-1])
            ctx.add_artifact(
                "rejected",
                {
                    "index": idx,
                    "label": p.stem,
                    "path_abs": str(p.resolve()),
                    "sha256": sha256_file(p),
                    "format": "xyz",
                    "rejected_reason": reason,
                },
            )

        # Sort buckets by index so consumers don't have to.
        ctx.manifest.artifacts["accepted"].sort(key=lambda d: d["index"])
        ctx.manifest.artifacts["rejected"].sort(key=lambda d: d["index"])

        # ---------- 7. accepted ensemble + best xyz ----------
        best_chosen_by: str | None = None
        if accepted_paths:
            accepted_ens = outputs_dir / "accepted_ensemble.xyz"
            concat_xyz_files(accepted_paths, accepted_ens)
            ctx.add_artifact(
                "xyz_ensemble",
                {
                    "label": "accepted_ensemble",
                    "path_abs": str(accepted_ens.resolve()),
                    "sha256": sha256_file(accepted_ens),
                    "format": "xyz",
                },
            )

            # Best: lowest rel_energy_kcal among accepted, else first by index.
            accepted_records = ctx.manifest.artifacts["accepted"]
            if any("rel_energy_kcal" in r for r in accepted_records):
                best = min(
                    accepted_records,
                    key=lambda r: float(r.get("rel_energy_kcal", float("inf"))),
                )
                best_chosen_by = "lowest_rel_energy_kcal"
            else:
                best = accepted_records[0]
                best_chosen_by = "first_accepted_by_index"
            best_dst = outputs_dir / "best.xyz"
            shutil.copy2(Path(best["path_abs"]), best_dst)
            ctx.add_artifact(
                "xyz",
                {
                    "label": "best",
                    "path_abs": str(best_dst.resolve()),
                    "sha256": sha256_file(best_dst),
                    "format": "xyz",
                },
            )

        # ---------- 8. crest-ish energies file (accepted only) ----------
        accepted_records = ctx.manifest.artifacts["accepted"]
        if accepted_records and any("rel_energy_kcal" in r for r in accepted_records):
            energies_out = outputs_dir / "prism.energies"
            lines: list[str] = []
            for r in accepted_records:
                rk = r.get("rel_energy_kcal")
                if rk is None:
                    continue
                lines.append(f"{int(r['index']):6d}  {float(rk): .6f}")
            energies_out.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            ctx.add_artifact(
                "files",
                {
                    "label": "prism_energies",
                    "path_abs": str(energies_out.resolve()),
                    "sha256": sha256_file(energies_out),
                    "format": "txt",
                },
            )

        # ---------- 9. inputs block + final ok decision ----------
        n_accepted = len(accepted_records)
        n_rejected = len(ctx.manifest.artifacts["rejected"])
        inputs: dict[str, Any] = {
            "method": METHOD_NAME,
            "n_input": n_input,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "min_conformers": cfg["min_conformers"],
            "moi_pruning": cfg["moi_pruning"],
            "rmsd_pruning": cfg["rmsd_pruning"],
            "rot_corr_rmsd_pruning": cfg["rot_corr_rmsd_pruning"],
            "use_energies": cfg["use_energies"],
            "use_energies_resolved": use_energies and have_energies,
            "max_dE_kcal": cfg["max_dE_kcal"],
            "ewin_kcal": cfg["ewin_kcal"],
            "ewin_applied": ewin_applied,
            "n_above_ewin": n_above_ewin,
            "timeout_s": cfg["timeout_s"],
            "keep_rejected": cfg["keep_rejected"],
            "source_mode": mode,
        }
        if best_chosen_by is not None:
            inputs["best_chosen_by"] = best_chosen_by
        ctx.set_inputs(**inputs)

        if n_accepted == 0:
            ctx.fail("no_conformers_accepted")

        # ---------- 10. self-test against the conformer_screen contract ----------
        # This is the contract's intended use pattern: a non-raising
        # validator returns a list of human-readable problems which we
        # append to ``failures`` with a stable error key.
        problems = validate_conformer_screen(
            ctx.manifest.to_dict(),
            require_rejected_reason=cfg["keep_rejected"],
        )
        for p in problems:
            ctx.fail(f"contract_violation: {p}")

    # -------------------- helpers --------------------

    def _stage_inputs(
        self,
        *,
        mode: str,
        items: list[dict[str, Any]],
        staged_dir: Path,
    ) -> tuple[list[Path], list[float | None]]:
        """Materialize per-conformer input files in ``staged_dir`` and
        align a parallel list of per-conformer energies (``None`` when
        unavailable)."""
        staged_paths: list[Path] = []
        energies: list[float | None] = []

        if mode == "many":
            for i, it in enumerate(items, start=1):
                src = Path(it["path_abs"])
                dst = staged_dir / f"conf_{i:04d}.xyz"
                shutil.copy2(src, dst)
                staged_paths.append(dst)
                energies.append(extract_energy_kcal_from_item(it))
            return staged_paths, energies

        # ensemble or single
        src = Path(items[0]["path_abs"])
        text = src.read_text(encoding="utf-8", errors="replace")
        blocks = split_multixyz(text)
        if len(blocks) >= 2:
            for i, blk in enumerate(blocks, start=1):
                dst = staged_dir / f"conf_{i:04d}.xyz"
                write_xyz_block(dst, blk)
                staged_paths.append(dst)
                energies.append(None)
        else:
            dst = staged_dir / "conf_0001.xyz"
            shutil.copy2(src, dst)
            staged_paths.append(dst)
            energies.append(None)

        return staged_paths, energies


__all__ = [
    "METHOD_NAME",
    "REJECTED_REASON",
    "REJECTED_REASON_EWIN",
    "PrismScreen",
    "apply_ewin_filter",
    "concat_xyz_files",
    "discover_conformer_sources",
    "extract_energy_kcal_from_item",
    "find_energy_file",
    "main",
    "normalize_use_energies",
    "parse_generic_energies_kcal",
    "resolve_use_energies",
    "run_prism_pruner",
]


main = PrismScreen.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
