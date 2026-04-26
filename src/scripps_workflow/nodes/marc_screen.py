"""``wf-marc`` — clustering-based conformer screening via navicat-marc.

Chain node. Sibling impl of :mod:`scripps_workflow.nodes.prism_screen`
satisfying the same ``conformer_screen`` role contract — a workflow that
uses the role can swap ``wf-prism`` for ``wf-marc`` (or vice-versa)
without rewiring its graph or changing its downstream consumers.

Where prism returns a single accept mask (and we have to invent a
generic ``rejected_reason``), marc clusters the ensemble and returns
per-conformer cluster assignments + distances. We surface those as the
optional ``cluster_id`` / ``cluster_distance`` fields the contract
defines, which makes marc-produced manifests strictly more informative
than prism-produced ones for the same workflow position.

Conformer source discovery — the upstream's pointer is followed and the
manifest's artifact buckets are tried in priority order::

    accepted -> selected -> conformers -> xyz_ensemble -> xyz

The first non-empty bucket whose paths exist on disk wins. This makes
marc work both directly after ``crest`` (which fills ``conformers``)
and after another screen (which fills ``accepted``).

Energy harvesting is two-stage: per-conformer artifact items are checked
for ``rel_energy_kcal`` / ``rel_energy``; missing values are filled from
an upstream ``artifacts.files`` entry whose label contains "energies" or
whose filename ends in ``.energies``.

Config keys (``key=value`` tokens or one JSON object):

    min_conformers          int. Skip clustering and accept all when
                            n_input < this. (default 3)
    metric                  str. Distance metric. One of ``rmsd``,
                            ``moi``, ``rotcorr_rmsd``, ``mix``. (default
                            ``mix``; navicat-marc default)
    n_clusters              int >= 1, or ``auto`` / empty. ``auto`` lets
                            navicat-marc pick by silhouette score.
                            (default ``auto``)
    clustering              str. Clustering algorithm: ``kmeans``,
                            ``agglomerative``, ``dbscan``, or ``auto``.
                            (default ``auto``)
    use_energies            auto | true | false. ``auto`` uses energies
                            iff every input has one. (default ``auto``)
    ewin_kcal               float >= 0. *Pre-clustering* energy cutoff:
                            drop conformers more than this many kcal/mol
                            above the lowest-energy input before
                            clustering. Mirrors the navicat-marc /
                            crest ``ewin_kcal`` convention. Silently
                            skipped when ``use_energies`` resolves to
                            false. (default 5.0)
    timeout_s               int >= 5. Per-stage timeout. (default 120)
    keep_rejected           bool. Copy rejected conformers to
                            ``outputs/rejected/``. (default true)

The library-using core is :func:`run_marc`; tests monkeypatch it to
avoid needing navicat-marc installed.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import logging_utils
from ..contracts.conformer_screen import validate_conformer_screen
from ..hashing import sha256_file
from ..node import Node, NodeContext
from ..parsing import parse_bool, parse_float, parse_int

# Reuse the conformer_screen role helpers that prism_screen factored out.
# These are pure functions and contract-aware. When a third sibling impl
# appears (e.g. a future ``crest_cluster_screen``), lift them into a
# dedicated ``scripps_workflow.nodes._conformer_screen_common`` module
# and update both prism_screen and marc_screen to import from there.
from .crest import split_multixyz, write_xyz_block
from .prism_screen import (
    apply_ewin_filter,
    concat_xyz_files,
    discover_conformer_sources,
    extract_energy_kcal_from_item,
    find_energy_file,
    normalize_use_energies,
    parse_generic_energies_kcal,
    resolve_use_energies,
)

#: Identifier recorded into ``inputs.method`` so a downstream consumer
#: can branch on which screen impl produced the manifest.
METHOD_NAME = "navicat_marc"

#: ``rejected_reason`` value used for conformers dropped during the
#: clustering stage (i.e. their cluster's representative is somewhere
#: else in ``accepted``). Distinct from the prism reason so a downstream
#: consumer reading the rejected bucket can tell which impl was used.
REJECTED_REASON = "cluster_dup"

#: ``rejected_reason`` value for conformers dropped by the pre-clustering
#: ``ewin_kcal`` filter. Same string as prism by design — the semantics
#: ("above_energy_window" relative to lowest input) match exactly.
REJECTED_REASON_EWIN = "above_energy_window"

#: Allowed ``metric`` values. ``mix`` is navicat-marc's hybrid RMSD+MoI
#: distance; ``rotcorr_rmsd`` is the rotation-corrected RMSD that's
#: common in Sterling/Aleyna-style conformer libraries.
_ALLOWED_METRICS: tuple[str, ...] = ("rmsd", "moi", "rotcorr_rmsd", "mix")

#: Allowed ``clustering`` values. ``auto`` defers algorithm selection to
#: navicat-marc (which uses silhouette + BIC heuristics).
_ALLOWED_CLUSTERINGS: tuple[str, ...] = ("auto", "kmeans", "agglomerative", "dbscan")


# --------------------------------------------------------------------
# Pure helpers (testable)
# --------------------------------------------------------------------


def normalize_metric(v: Any) -> str:
    """Validate ``metric`` config, returning the canonical lowercase form.

    Raises :class:`ValueError` for unrecognized values so the caller's
    ``parse_config`` reports the bug as ``argv_parse_failed`` rather
    than letting an unknown metric reach navicat-marc and produce a
    less-readable error mid-run.
    """
    if v is None:
        return "mix"
    s = str(v).strip().lower()
    if s == "":
        return "mix"
    # Common synonyms — keep them lenient at the boundary so engine
    # operators don't have to memorize the exact spelling.
    if s in {"rmsd", "rmsd_pruning"}:
        return "rmsd"
    if s in {"moi", "moi_pruning"}:
        return "moi"
    if s in {"rotcorr", "rotcorr_rmsd", "rot_corr_rmsd", "rot_corr_rmsd_pruning"}:
        return "rotcorr_rmsd"
    if s in {"mix", "auto"}:
        return "mix"
    raise ValueError(
        f"metric must be one of {_ALLOWED_METRICS!r}, got {v!r}"
    )


def normalize_clustering(v: Any) -> str:
    """Validate ``clustering`` config, returning the canonical lowercase form."""
    if v is None:
        return "auto"
    s = str(v).strip().lower()
    if s in {"", "auto"}:
        return "auto"
    if s in _ALLOWED_CLUSTERINGS:
        return s
    raise ValueError(
        f"clustering must be one of {_ALLOWED_CLUSTERINGS!r}, got {v!r}"
    )


def normalize_n_clusters(v: Any) -> int | None:
    """Validate ``n_clusters`` config. Returns ``None`` for ``auto`` /
    unset, or a positive int otherwise.

    Negative or zero values raise :class:`ValueError` so the caller's
    ``parse_config`` rejects them up front."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"", "auto"}:
        return None
    try:
        n = int(s)
    except ValueError:
        raise ValueError(f"n_clusters must be an int or 'auto', got {v!r}")
    if n < 1:
        raise ValueError(f"n_clusters must be >= 1 or 'auto', got {n}")
    return n


# --------------------------------------------------------------------
# Library-using core (monkeypatch target for tests)
# --------------------------------------------------------------------


@dataclass(frozen=True)
class MarcResult:
    """Return shape of :func:`run_marc`.

    Attributes:
        accept_mask: Length-N bool list. ``True`` means "kept as a
            cluster representative".
        cluster_ids: Length-N int list. The cluster each conformer was
            assigned to. Cluster IDs are zero-based dense integers
            (i.e. ``range(n_clusters)``).
        cluster_distances: Length-N float list. Distance from each
            conformer to its cluster's representative under the chosen
            metric. The representative itself has distance ``0.0``.
        n_clusters: Number of clusters navicat-marc actually used. For
            ``n_clusters="auto"`` this is the silhouette-chosen value
            and is recorded into the manifest's inputs block.
        algorithm_used: Concrete clustering algorithm name navicat-marc
            picked. For ``clustering="auto"`` this is the resolved name.
        method_version: navicat-marc's own version string, when
            available. ``None`` if the library doesn't expose one.
    """

    accept_mask: list[bool]
    cluster_ids: list[int]
    cluster_distances: list[float]
    n_clusters: int
    algorithm_used: str
    method_version: str | None


def run_marc(
    *,
    ensemble_path: Path,
    energies_kcal: list[float] | None,
    metric: str,
    n_clusters: int | None,
    clustering: str,
    timeout_s: int,
    log_fn,
) -> MarcResult:
    """Run navicat-marc on a multi-xyz ensemble and return a typed result.

    Imports navicat-marc internally — the package pulls in scikit-learn
    and is therefore not part of the framework's test runtime. The whole
    function is the monkeypatch target in tests.

    Args:
        ensemble_path: Path to the multi-xyz file.
        energies_kcal: Per-conformer relative energies (kcal/mol) used
            for energy-aware ranking inside clusters, or ``None`` to
            ignore energies entirely.
        metric: Distance metric. One of :data:`_ALLOWED_METRICS`.
        n_clusters: Number of clusters, or ``None`` for navicat-marc's
            silhouette/BIC auto-pick.
        clustering: Clustering algorithm name. ``"auto"`` lets
            navicat-marc choose.
        timeout_s: Best-effort per-stage timeout (navicat-marc is
            mostly numpy/sklearn; this is a soft wall).
        log_fn: Callback used for navicat-marc's progress messages.

    Returns:
        :class:`MarcResult` with the accept mask + per-conformer
        cluster info.
    """
    # All imports inside the function so the framework / tests can
    # operate without navicat-marc installed.
    from navicat_marc import __version__ as marc_version  # type: ignore
    from navicat_marc.molecule import Molecule  # type: ignore
    from navicat_marc.clustering import cluster_molecules  # type: ignore
    from navicat_marc.exceptions import MarcError  # type: ignore  # noqa: F401

    import numpy as np  # type: ignore

    # navicat-marc consumes a list of Molecule objects, one per frame.
    # We split the ensemble back into frames using crest's helper so
    # there's a single source of truth for multi-xyz parsing.
    text = ensemble_path.read_text(encoding="utf-8", errors="replace")
    blocks = split_multixyz(text)
    if not blocks:
        raise ValueError(f"run_marc: empty ensemble at {ensemble_path}")

    molecules = []
    tmp_dir = ensemble_path.parent / "_marc_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, blk in enumerate(blocks, start=1):
            frame = tmp_dir / f"frame_{i:04d}.xyz"
            write_xyz_block(frame, blk)
            molecules.append(Molecule(str(frame)))

        energies_arr = None
        if energies_kcal is not None:
            energies_arr = np.array([float(e) for e in energies_kcal], dtype=float)

        # navicat-marc's ``cluster_molecules`` returns ``(labels, centers,
        # distances)`` where ``labels`` is per-molecule, ``centers`` is the
        # set of representative indices, and ``distances`` is per-molecule
        # distance to its center. The exact API has shifted across navicat
        # releases — adapt here to insulate the rest of the node.
        result = cluster_molecules(
            molecules=molecules,
            metric=metric,
            n_clusters=n_clusters,
            clustering=clustering,
            energies=energies_arr,
            timeout_s=int(timeout_s),
            log=log_fn,
        )
    finally:
        # Clean up the per-frame staging dir; the upstream multixyz
        # remains on disk as the canonical input.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Adapt the navicat-marc return into our typed result. We accept
    # either a tuple ``(labels, centers, distances)`` or a dict with
    # those keys, matching the two shapes navicat-marc has shipped.
    if isinstance(result, dict):
        labels = list(result["labels"])
        centers = list(result["centers"])
        distances = list(result["distances"])
        algorithm_used = str(result.get("algorithm", clustering))
    else:
        labels, centers, distances = result
        labels = list(labels)
        centers = list(centers)
        distances = list(distances)
        algorithm_used = str(clustering)

    n = len(molecules)
    if len(labels) != n or len(distances) != n:
        raise ValueError(
            f"run_marc: navicat-marc returned mismatched arrays "
            f"(labels={len(labels)}, distances={len(distances)}, n={n})"
        )

    accept_set = set(int(c) for c in centers)
    accept_mask = [i in accept_set for i in range(n)]

    return MarcResult(
        accept_mask=accept_mask,
        cluster_ids=[int(x) for x in labels],
        cluster_distances=[float(x) for x in distances],
        n_clusters=len(accept_set),
        algorithm_used=algorithm_used,
        method_version=str(marc_version),
    )


# --------------------------------------------------------------------
# Node class
# --------------------------------------------------------------------


class MarcScreen(Node):
    """Chain node: cluster a conformer ensemble via navicat-marc."""

    step = "marc_screen"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        # Normalize-and-validate up front so typos / out-of-range values
        # surface as ``argv_parse_failed`` rather than mid-run errors.
        min_conformers = parse_int(raw.get("min_conformers"), 3)
        if min_conformers < 1:
            raise ValueError("min_conformers must be >= 1")

        timeout_s = parse_int(raw.get("timeout_s"), 120)
        if timeout_s < 5:
            raise ValueError("timeout_s must be >= 5")

        ewin_kcal = parse_float(raw.get("ewin_kcal"), 5.0)
        if ewin_kcal < 0:
            raise ValueError("ewin_kcal must be >= 0")

        metric = normalize_metric(raw.get("metric"))
        clustering = normalize_clustering(raw.get("clustering"))
        n_clusters = normalize_n_clusters(raw.get("n_clusters"))

        return {
            "min_conformers": min_conformers,
            "metric": metric,
            "clustering": clustering,
            "n_clusters": n_clusters,  # int or None (auto)
            "use_energies": normalize_use_energies(raw.get("use_energies")),
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
                        f"marc_screen: loaded {n_input} energies from {energy_file}"
                    )
                else:
                    logging_utils.log_warn(
                        f"marc_screen: energy file {energy_file} had only "
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

        # ---------- 3. ewin filter (pre-clustering energy cutoff) ----------
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
                f"marc_screen: ewin_kcal={cfg['ewin_kcal']:.3f} dropped "
                f"{n_above_ewin}/{n_input} conformers above the energy window"
            )

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

        # ---------- 5. cluster (or skip if too few survived ewin) ----------
        # Default values that get overwritten when the library actually
        # ran, but kept as sane fallbacks for the trivial paths.
        cluster_ids_within: list[int] = list(range(n_within))
        cluster_distances_within: list[float] = [0.0] * n_within
        marc_n_clusters: int = n_within
        marc_algorithm_used: str = cfg["clustering"]
        marc_method_version: str | None = None

        if n_within == 0:
            cluster_mask: list[bool] = []
        elif n_within < cfg["min_conformers"]:
            logging_utils.log_info(
                f"marc_screen: n_within={n_within} < min_conformers="
                f"{cfg['min_conformers']}; skipping clustering, accepting all "
                f"survivors of the ewin filter."
            )
            cluster_mask = [True] * n_within
        else:
            try:
                marc_result = run_marc(
                    ensemble_path=ensemble_path,
                    energies_kcal=(
                        [float(e) for e in within_energies]
                        if use_energies and have_energies
                        else None
                    ),
                    metric=cfg["metric"],
                    n_clusters=cfg["n_clusters"],
                    clustering=cfg["clustering"],
                    timeout_s=cfg["timeout_s"],
                    log_fn=lambda m: logging_utils.log_info(f"[marc] {m}"),
                )
            except ImportError as e:
                ctx.fail(f"import_navicat_marc_failed: {e}")
                cluster_mask = [True] * n_within
            except Exception as e:
                ctx.fail(f"clustering_failed_fallback_accept_all: {e}")
                cluster_mask = [True] * n_within
            else:
                cluster_mask = list(marc_result.accept_mask)
                cluster_ids_within = list(marc_result.cluster_ids)
                cluster_distances_within = list(marc_result.cluster_distances)
                marc_n_clusters = int(marc_result.n_clusters)
                marc_algorithm_used = str(marc_result.algorithm_used)
                marc_method_version = marc_result.method_version

            if len(cluster_mask) != n_within:
                ctx.fail(
                    f"clustering_returned_unexpected_mask_length: "
                    f"expected {n_within}, got {len(cluster_mask)}"
                )
                cluster_mask = [True] * n_within
                # Reset the per-conformer arrays to safe defaults so we
                # don't index out of range below.
                cluster_ids_within = list(range(n_within))
                cluster_distances_within = [0.0] * n_within

        # ---------- 6. materialize accepted / rejected ----------
        # Combine masks: a conformer is accepted iff it survived BOTH
        # the ewin filter AND clustering. Rejection reason is recorded
        # so downstream consumers can tell ewin-rejects from cluster-dups.
        accepted_dir = outputs_dir / "accepted"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir = outputs_dir / "rejected"
        if cfg["keep_rejected"]:
            rejected_dir.mkdir(parents=True, exist_ok=True)

        # Build full-length per-conformer cluster info aligned with
        # ``staged_paths``. ewin-rejected slots get ``None`` since
        # navicat-marc never saw them.
        full_cluster_ids: list[int | None] = []
        full_cluster_dists: list[float | None] = []
        within_iter = iter(zip(cluster_ids_within, cluster_distances_within))
        for ewin_ok in ewin_keep_mask:
            if ewin_ok:
                cid, cdist = next(within_iter)
                full_cluster_ids.append(int(cid))
                full_cluster_dists.append(float(cdist))
            else:
                full_cluster_ids.append(None)
                full_cluster_dists.append(None)

        cluster_iter = iter(cluster_mask)
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
            keep = next(cluster_iter)
            if keep:
                dst = accepted_dir / f"conf_{i:04d}.xyz"
                shutil.copy2(src, dst)
                accepted_paths.append(dst)
            elif cfg["keep_rejected"]:
                dst = rejected_dir / f"conf_{i:04d}.xyz"
                shutil.copy2(src, dst)
                rejected_records.append((dst, REJECTED_REASON))

        # Synthesize an accept_mask spanning the original 1..n_input
        # space for the energy-harvest block below.
        accept_mask: list[bool] = []
        cluster_iter2 = iter(cluster_mask)
        for ewin_ok in ewin_keep_mask:
            if not ewin_ok:
                accept_mask.append(False)
            else:
                accept_mask.append(next(cluster_iter2))

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
            cid = full_cluster_ids[idx - 1] if idx - 1 < len(full_cluster_ids) else None
            cdist = full_cluster_dists[idx - 1] if idx - 1 < len(full_cluster_dists) else None
            if isinstance(cid, int):
                rec["cluster_id"] = int(cid)
            if isinstance(cdist, (int, float)):
                rec["cluster_distance"] = float(cdist)
            e = energies_kcal[idx - 1] if idx - 1 < len(energies_kcal) else None
            if isinstance(e, float):
                rec["energy_kcal"] = float(e)
                if e_min is not None:
                    rec["rel_energy_kcal"] = float(e - e_min)
            ctx.add_artifact("accepted", rec)

        for p, reason in rejected_records:
            idx = int(p.stem.split("_")[-1])
            rec_r: dict[str, Any] = {
                "index": idx,
                "label": p.stem,
                "path_abs": str(p.resolve()),
                "sha256": sha256_file(p),
                "format": "xyz",
                "rejected_reason": reason,
            }
            cid = full_cluster_ids[idx - 1] if idx - 1 < len(full_cluster_ids) else None
            cdist = full_cluster_dists[idx - 1] if idx - 1 < len(full_cluster_dists) else None
            if isinstance(cid, int):
                rec_r["cluster_id"] = int(cid)
            if isinstance(cdist, (int, float)):
                rec_r["cluster_distance"] = float(cdist)
            ctx.add_artifact("rejected", rec_r)

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

            # Best-pick policy is informative but never load-bearing —
            # if energies are present we use them, otherwise we fall
            # back to the cluster-center with the smallest cluster_id
            # (deterministic, doesn't hide the ranking from the user).
            accepted_records = ctx.manifest.artifacts["accepted"]
            if any("rel_energy_kcal" in r for r in accepted_records):
                best = min(
                    accepted_records,
                    key=lambda r: float(r.get("rel_energy_kcal", float("inf"))),
                )
                best_chosen_by = "lowest_rel_energy_kcal"
            else:
                best = min(
                    accepted_records,
                    key=lambda r: (
                        int(r.get("cluster_id", 1 << 30)),
                        int(r["index"]),
                    ),
                )
                best_chosen_by = "first_cluster_center"
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
            energies_out = outputs_dir / "marc.energies"
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
                    "label": "marc_energies",
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
            "metric": cfg["metric"],
            "clustering": cfg["clustering"],
            "clustering_resolved": marc_algorithm_used,
            "n_clusters": cfg["n_clusters"] if cfg["n_clusters"] is not None else "auto",
            "n_clusters_resolved": marc_n_clusters,
            "use_energies": cfg["use_energies"],
            "use_energies_resolved": use_energies and have_energies,
            "ewin_kcal": cfg["ewin_kcal"],
            "ewin_applied": ewin_applied,
            "n_above_ewin": n_above_ewin,
            "timeout_s": cfg["timeout_s"],
            "keep_rejected": cfg["keep_rejected"],
            "source_mode": mode,
        }
        if marc_method_version is not None:
            inputs["method_version"] = marc_method_version
        if best_chosen_by is not None:
            inputs["best_chosen_by"] = best_chosen_by
        ctx.set_inputs(**inputs)

        if n_accepted == 0:
            ctx.fail("no_conformers_accepted")

        # ---------- 10. self-test against the conformer_screen contract ----------
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
        unavailable).

        This is intentionally a copy of prism_screen's helper rather than
        a shared import — the staging policy (per-frame split for
        ensemble inputs, single-frame for ``mode == "single"``) is part
        of the conformer_screen role contract's behavioral guarantee,
        and we want the two impls to deviate only on the *clustering /
        pruning* step, not the staging step.
        """
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
    "MarcResult",
    "MarcScreen",
    "main",
    "normalize_clustering",
    "normalize_metric",
    "normalize_n_clusters",
    "run_marc",
]


main = MarcScreen.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
