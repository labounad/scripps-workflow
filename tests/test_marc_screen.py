"""Tests for the ``wf-marc`` conformer-screening node.

navicat-marc pulls in scikit-learn and is not available in the test
sandbox, so we monkeypatch the entire library-using core
(:func:`scripps_workflow.nodes.marc_screen.run_marc`) to return a chosen
:class:`MarcResult`. That isolates the node's wiring (source discovery,
staging, ewin filter, contract self-test, manifest shape) from the
library's internals.

These tests intentionally mirror the prism_screen test surface so the
two impls of the conformer_screen role contract are validated from the
same angles. Where they diverge: the marc result carries per-conformer
``cluster_id`` and ``cluster_distance`` which we surface through the
manifest's optional contract fields.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripps_workflow.contracts.conformer_screen import validate_conformer_screen
from scripps_workflow.nodes import marc_screen as mc
from scripps_workflow.nodes.marc_screen import (
    METHOD_NAME,
    REJECTED_REASON,
    REJECTED_REASON_EWIN,
    MarcResult,
    MarcScreen,
    normalize_clustering,
    normalize_metric,
    normalize_n_clusters,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------


class TestNormalizeMetric:
    def test_default_when_unset(self) -> None:
        assert normalize_metric(None) == "mix"
        assert normalize_metric("") == "mix"

    def test_canonical_passthrough(self) -> None:
        for v in ("rmsd", "moi", "rotcorr_rmsd", "mix"):
            assert normalize_metric(v) == v

    def test_synonyms_collapsed(self) -> None:
        assert normalize_metric("RMSD_pruning") == "rmsd"
        assert normalize_metric("moi_pruning") == "moi"
        assert normalize_metric("rotcorr") == "rotcorr_rmsd"
        assert normalize_metric("rot_corr_rmsd") == "rotcorr_rmsd"
        assert normalize_metric("auto") == "mix"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_metric("hausdorff")


class TestNormalizeClustering:
    def test_default_when_unset(self) -> None:
        assert normalize_clustering(None) == "auto"
        assert normalize_clustering("") == "auto"
        assert normalize_clustering("auto") == "auto"

    def test_canonical_passthrough(self) -> None:
        for v in ("kmeans", "agglomerative", "dbscan"):
            assert normalize_clustering(v) == v

    def test_case_insensitive(self) -> None:
        assert normalize_clustering("KMeans") == "kmeans"
        assert normalize_clustering("AGGLOMERATIVE") == "agglomerative"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_clustering("hierarchical")


class TestNormalizeNClusters:
    def test_auto_when_unset(self) -> None:
        assert normalize_n_clusters(None) is None
        assert normalize_n_clusters("") is None
        assert normalize_n_clusters("auto") is None

    def test_positive_int_passes(self) -> None:
        assert normalize_n_clusters(3) == 3
        assert normalize_n_clusters("5") == 5

    def test_zero_or_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_n_clusters(0)
        with pytest.raises(ValueError):
            normalize_n_clusters("-1")

    def test_non_integer_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_n_clusters("two")


# --------------------------------------------------------------------
# End-to-end fixtures
# --------------------------------------------------------------------


def _make_xyz(path: Path, *, atoms: int = 3, label: str = "x") -> Path:
    lines = [f"{atoms}", label]
    coords = [
        ("C", 0.0, 0.0, 0.0),
        ("O", 0.0, 0.0, 1.4),
        ("H", 0.0, 0.0, -1.0),
    ]
    for sym, x, y, z in coords[:atoms]:
        lines.append(f"{sym} {x:.3f} {y:.3f} {z:.3f}")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return path


def _make_multixyz(path: Path, n_frames: int) -> Path:
    chunks: list[str] = []
    for i in range(1, n_frames + 1):
        chunks.append(
            f"3\nframe {i}\nC {0.01*i:.3f} 0 0\nO 0 0 1.4\nH 0 0 -1.0"
        )
    path.write_text("\n".join(chunks) + "\n")
    return path


def _upstream_with_conformers(
    tmp_path: Path,
    n: int = 3,
    *,
    with_per_item_energies: bool = True,
) -> Path:
    """Build an upstream tree with `n` per-conformer xyz files in the
    'conformers' bucket, plus an xyz_ensemble. Returns manifest path."""
    up = tmp_path / "upstream"
    out = up / "outputs"
    out.mkdir(parents=True)
    confs_dir = out / "conformers"
    confs_dir.mkdir()

    m = Manifest.skeleton(step="crest", cwd=str(up))
    confs: list[dict] = []
    for i in range(1, n + 1):
        p = _make_xyz(confs_dir / f"conf_{i:04d}.xyz", label=f"conf{i}")
        rec: dict = {
            "index": i,
            "label": f"conf_{i:04d}",
            "path_abs": str(p.resolve()),
            "sha256": "f" * 64,
            "format": "xyz",
        }
        if with_per_item_energies:
            rec["rel_energy_kcal"] = float(0.5 * (i - 1))  # 0.0, 0.5, 1.0...
        confs.append(rec)
    m.artifacts["conformers"] = confs

    ens = _make_multixyz(out / "ens.xyz", n)
    m.artifacts["xyz_ensemble"] = [
        {
            "label": "crest_conformers",
            "path_abs": str(ens.resolve()),
            "sha256": "f" * 64,
            "format": "xyz",
        }
    ]
    m_path = out / "manifest.json"
    m.write(m_path)
    return m_path


def _upstream_with_ensemble(tmp_path: Path, n: int = 3) -> Path:
    """Upstream with NO 'conformers' bucket — only a multi-xyz ensemble."""
    up = tmp_path / "upstream"
    out = up / "outputs"
    out.mkdir(parents=True)
    ens = _make_multixyz(out / "ens.xyz", n)
    m = Manifest.skeleton(step="crest", cwd=str(up))
    m.artifacts["xyz_ensemble"] = [
        {
            "label": "crest_conformers",
            "path_abs": str(ens.resolve()),
            "sha256": "f" * 64,
            "format": "xyz",
        }
    ]
    m_path = out / "manifest.json"
    m.write(m_path)
    return m_path


def _upstream_with_single_xyz(tmp_path: Path) -> Path:
    up = tmp_path / "upstream"
    out = up / "outputs"
    xyz_dir = out / "xyz"
    xyz_dir.mkdir(parents=True)
    p = _make_xyz(xyz_dir / "best.xyz", label="best")
    m = Manifest.skeleton(step="embed", cwd=str(up))
    m.artifacts["xyz"] = [
        {
            "label": "embedded",
            "path_abs": str(p.resolve()),
            "sha256": "f" * 64,
            "format": "xyz",
        }
    ]
    m_path = out / "manifest.json"
    m.write(m_path)
    return m_path


def _empty_upstream(tmp_path: Path) -> Path:
    up = tmp_path / "upstream"
    out = up / "outputs"
    out.mkdir(parents=True)
    m = Manifest.skeleton(step="upstream", cwd=str(up))
    m_path = out / "manifest.json"
    m.write(m_path)
    return m_path


def _pointer_text(manifest_path: Path, ok: bool = True) -> str:
    return Pointer.of(ok=ok, manifest_path=manifest_path).to_json_line()


def _split_n(ensemble_path: Path) -> int:
    """Count frames in a multi-xyz file (uses the same helper the node uses)."""
    from scripps_workflow.nodes.crest import split_multixyz

    text = Path(ensemble_path).read_text(encoding="utf-8")
    return len(split_multixyz(text))


def _stub_run_marc_factory(result_factory):
    """Build a stub for ``run_marc`` that returns a chosen MarcResult.

    ``result_factory`` is a callable ``(n_input) -> MarcResult`` so tests
    can express results relative to the number of conformers actually
    fed to navicat-marc without recomputing them.
    """

    def _stub(*, ensemble_path, energies_kcal, **kwargs):
        n = _split_n(Path(ensemble_path))
        return result_factory(n)

    return _stub


def _trivial_result(n: int, *, accept_mask: list[bool] | None = None) -> MarcResult:
    """Build a MarcResult that pretends every conformer is its own cluster.

    If ``accept_mask`` is provided, only the True positions are surfaced
    as cluster representatives; the rest get cluster_id = the index of
    the previous accepted conformer (so the per-conformer cluster_id
    field is always populated). cluster_distance is 0.0 for accepted,
    1.0 for non-accepted (just so tests can verify it round-trips).
    """
    if accept_mask is None:
        accept_mask = [True] * n
    cluster_ids: list[int] = []
    cluster_distances: list[float] = []
    cur_cluster = -1
    for keep in accept_mask:
        if keep:
            cur_cluster += 1
            cluster_ids.append(cur_cluster)
            cluster_distances.append(0.0)
        else:
            # Belongs to whatever cluster center came before. If no
            # accepted yet, fall back to cluster 0.
            cluster_ids.append(max(cur_cluster, 0))
            cluster_distances.append(1.0)
    n_clusters = sum(1 for k in accept_mask if k)
    return MarcResult(
        accept_mask=list(accept_mask),
        cluster_ids=cluster_ids,
        cluster_distances=cluster_distances,
        n_clusters=n_clusters,
        algorithm_used="kmeans",
        method_version="0.0.0-test",
    )


@pytest.fixture
def stub_marc_keep_all(monkeypatch):
    """Default stub: each conformer is its own cluster."""
    monkeypatch.setattr(
        mc,
        "run_marc",
        _stub_run_marc_factory(lambda n: _trivial_result(n)),
    )


def _run_node(
    tmp_path: Path,
    upstream_manifest: Path,
    *config_tokens: str,
    ok_pointer: bool = True,
) -> dict:
    """Invoke MarcScreen against an existing upstream manifest and
    return the parsed manifest dict."""
    pointer_text = _pointer_text(upstream_manifest, ok=ok_pointer)

    call_dir = tmp_path / "calls" / "marc"
    call_dir.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = MarcScreen().invoke(["marc_screen", pointer_text, *config_tokens])
    finally:
        os.chdir(cwd)
    assert rc == 0, "soft-fail invariant violated"
    m_path = call_dir / "outputs" / "manifest.json"
    assert m_path.exists()
    return json.loads(m_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------


class TestHappyPath:
    def test_many_mode_keep_all(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["ok"] is True
        assert m["step"] == "marc_screen"

        accepted = m["artifacts"]["accepted"]
        rejected = m["artifacts"]["rejected"]
        assert [a["index"] for a in accepted] == [1, 2, 3]
        assert rejected == []
        assert m["inputs"]["method"] == METHOD_NAME
        assert m["inputs"]["n_input"] == 3
        assert m["inputs"]["n_accepted"] == 3
        assert m["inputs"]["n_rejected"] == 0
        assert m["inputs"]["source_mode"] == "many"

    def test_partial_rejection_writes_rejected(self, tmp_path, monkeypatch):
        # mask = [True, False, True] -> conf 2 rejected as cluster_dup.
        monkeypatch.setattr(
            mc,
            "run_marc",
            _stub_run_marc_factory(
                lambda n: _trivial_result(n, accept_mask=[True, False, True][:n])
            ),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["ok"] is True

        accepted = m["artifacts"]["accepted"]
        rejected = m["artifacts"]["rejected"]
        assert [a["index"] for a in accepted] == [1, 3]
        assert [r["index"] for r in rejected] == [2]
        assert rejected[0]["rejected_reason"] == REJECTED_REASON
        assert Path(rejected[0]["path_abs"]).exists()

    def test_cluster_fields_surface_into_records(self, tmp_path, monkeypatch):
        # Build a 4-conformer scenario where conf 1 & 3 are cluster
        # centers, conf 2 belongs to cluster 0 (with distance 1.0), and
        # conf 4 belongs to cluster 1 (with distance 1.0).
        def _factory(n):
            assert n == 4
            return MarcResult(
                accept_mask=[True, False, True, False],
                cluster_ids=[0, 0, 1, 1],
                cluster_distances=[0.0, 0.42, 0.0, 0.99],
                n_clusters=2,
                algorithm_used="kmeans",
                method_version="0.0.0-test",
            )

        monkeypatch.setattr(mc, "run_marc", _stub_run_marc_factory(_factory))
        up = _upstream_with_conformers(tmp_path, n=4)
        m = _run_node(tmp_path, up)
        accepted = m["artifacts"]["accepted"]
        rejected = m["artifacts"]["rejected"]
        # Accepted records carry cluster_id + cluster_distance.
        a1 = next(r for r in accepted if r["index"] == 1)
        a3 = next(r for r in accepted if r["index"] == 3)
        assert a1["cluster_id"] == 0 and a1["cluster_distance"] == 0.0
        assert a3["cluster_id"] == 1 and a3["cluster_distance"] == 0.0
        # Rejected (cluster_dup) records also carry them — distance to
        # the cluster center is exactly the data point a downstream
        # consumer would want for triage.
        r2 = next(r for r in rejected if r["index"] == 2)
        r4 = next(r for r in rejected if r["index"] == 4)
        assert r2["cluster_id"] == 0 and r2["cluster_distance"] == pytest.approx(0.42)
        assert r4["cluster_id"] == 1 and r4["cluster_distance"] == pytest.approx(0.99)

    def test_rel_energy_kcal_shifted_to_min(self, tmp_path, monkeypatch):
        # Reject conf 1 (the lowest), keep 2 and 3. Then min energy
        # within accepted is 0.5 (conf 2), and rel values shift to that.
        monkeypatch.setattr(
            mc,
            "run_marc",
            _stub_run_marc_factory(
                lambda n: _trivial_result(n, accept_mask=[False, True, True][:n])
            ),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        accepted = m["artifacts"]["accepted"]
        rels = [a["rel_energy_kcal"] for a in accepted]
        assert rels == pytest.approx([0.0, 0.5])
        assert accepted[0]["energy_kcal"] == pytest.approx(0.5)
        assert accepted[1]["energy_kcal"] == pytest.approx(1.0)

    def test_best_picked_by_rel_energy(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["inputs"]["best_chosen_by"] == "lowest_rel_energy_kcal"
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1 and xyz[0]["label"] == "best"
        assert "conf1" in Path(xyz[0]["path_abs"]).read_text()

    def test_best_falls_back_to_first_cluster_center_when_no_energies(
        self, tmp_path, stub_marc_keep_all
    ):
        # Upstream has no per-item energies; auto -> use_energies=false ->
        # best chosen by lowest cluster_id then lowest index.
        up = _upstream_with_conformers(tmp_path, n=3, with_per_item_energies=False)
        m = _run_node(tmp_path, up)
        assert m["inputs"]["best_chosen_by"] == "first_cluster_center"

    def test_ensemble_mode_splits_multixyz(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_ensemble(tmp_path, n=4)
        m = _run_node(tmp_path, up)
        assert m["ok"] is True
        assert m["inputs"]["source_mode"] == "ensemble"
        accepted = m["artifacts"]["accepted"]
        assert [a["index"] for a in accepted] == [1, 2, 3, 4]
        for a in accepted:
            assert "rel_energy_kcal" not in a

    def test_single_mode_passes_through(self, tmp_path, monkeypatch):
        # n_input=1 < default min_conformers=3 -> auto-accept.
        # run_marc should NOT be called.
        called = {"n": 0}

        def _explode(**kwargs):
            called["n"] += 1
            return _trivial_result(1)

        monkeypatch.setattr(mc, "run_marc", _explode)
        up = _upstream_with_single_xyz(tmp_path)
        m = _run_node(tmp_path, up)
        assert m["ok"] is True
        assert m["inputs"]["source_mode"] == "single"
        assert m["inputs"]["n_input"] == 1
        assert m["inputs"]["n_accepted"] == 1
        assert called["n"] == 0, "should skip clusterer when below min_conformers"

    def test_below_min_conformers_skips_clusterer(self, tmp_path, monkeypatch):
        called = {"n": 0}

        def _explode(**kwargs):
            called["n"] += 1
            return _trivial_result(2)

        monkeypatch.setattr(mc, "run_marc", _explode)
        up = _upstream_with_conformers(tmp_path, n=2)
        m = _run_node(tmp_path, up, "min_conformers=3")
        assert m["ok"] is True
        assert called["n"] == 0
        assert [a["index"] for a in m["artifacts"]["accepted"]] == [1, 2]

    def test_use_energies_passed_to_marc_when_available(
        self, tmp_path, monkeypatch
    ):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _trivial_result(_split_n(Path(kwargs["ensemble_path"])))

        monkeypatch.setattr(mc, "run_marc", _capture)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=1", "use_energies=auto")
        assert captured["energies_kcal"] == [0.0, 0.5, 1.0]
        assert m["inputs"]["use_energies_resolved"] is True

    def test_use_energies_false_blocks_energy_passthrough(
        self, tmp_path, monkeypatch
    ):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _trivial_result(_split_n(Path(kwargs["ensemble_path"])))

        monkeypatch.setattr(mc, "run_marc", _capture)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=1", "use_energies=false")
        assert captured["energies_kcal"] is None
        assert m["inputs"]["use_energies_resolved"] is False

    def test_metric_and_clustering_passed_to_marc(self, tmp_path, monkeypatch):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _trivial_result(_split_n(Path(kwargs["ensemble_path"])))

        monkeypatch.setattr(mc, "run_marc", _capture)
        up = _upstream_with_conformers(tmp_path, n=3)
        _run_node(
            tmp_path,
            up,
            "min_conformers=1",
            "metric=rotcorr_rmsd",
            "clustering=agglomerative",
            "n_clusters=2",
        )
        assert captured["metric"] == "rotcorr_rmsd"
        assert captured["clustering"] == "agglomerative"
        assert captured["n_clusters"] == 2

    def test_n_clusters_auto_passes_none_to_marc(self, tmp_path, monkeypatch):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _trivial_result(_split_n(Path(kwargs["ensemble_path"])))

        monkeypatch.setattr(mc, "run_marc", _capture)
        up = _upstream_with_conformers(tmp_path, n=3)
        _run_node(tmp_path, up, "min_conformers=1", "n_clusters=auto")
        assert captured["n_clusters"] is None

    def test_upstream_energies_file_used_when_per_item_missing(
        self, tmp_path, monkeypatch
    ):
        """When per-item energies are absent, fall back to a *.energies file
        in artifacts.files."""
        up_root = tmp_path / "upstream"
        out = up_root / "outputs"
        confs_dir = out / "conformers"
        confs_dir.mkdir(parents=True)
        m_up = Manifest.skeleton(step="crest", cwd=str(up_root))
        confs: list[dict] = []
        for i in range(1, 4):
            p = _make_xyz(confs_dir / f"conf_{i:04d}.xyz", label=f"c{i}")
            confs.append(
                {
                    "index": i,
                    "label": f"conf_{i:04d}",
                    "path_abs": str(p.resolve()),
                    "sha256": "f" * 64,
                    "format": "xyz",
                }
            )
        m_up.artifacts["conformers"] = confs

        e_file = out / "crest.energies"
        e_file.write_text("1 -1.234 0.0\n2 -1.230 0.5\n3 -1.225 1.0\n")
        m_up.artifacts["files"] = [
            {
                "label": "crest_energies",
                "path_abs": str(e_file.resolve()),
                "sha256": "f" * 64,
                "format": "txt",
            }
        ]
        m_up_path = out / "manifest.json"
        m_up.write(m_up_path)

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _trivial_result(_split_n(Path(kwargs["ensemble_path"])))

        monkeypatch.setattr(mc, "run_marc", _capture)

        m = _run_node(tmp_path, m_up_path, "min_conformers=1")
        assert captured["energies_kcal"] == [0.0, 0.5, 1.0]
        labels = {f["label"] for f in m["artifacts"]["files"]}
        assert "upstream_energies_used" in labels

    def test_accepted_ensemble_and_best_files_on_disk(
        self, tmp_path, stub_marc_keep_all
    ):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        ens = m["artifacts"]["xyz_ensemble"]
        assert len(ens) == 1 and ens[0]["label"] == "accepted_ensemble"
        assert Path(ens[0]["path_abs"]).exists()

        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1 and xyz[0]["label"] == "best"
        assert Path(xyz[0]["path_abs"]).exists()

    def test_marc_energies_file_emitted(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        files = m["artifacts"]["files"]
        labels = {f["label"] for f in files}
        assert "marc_energies" in labels

    def test_inputs_block_typed(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(
            tmp_path,
            up,
            "min_conformers=2",
            "metric=rmsd",
            "clustering=kmeans",
            "n_clusters=auto",
            "use_energies=auto",
            "ewin_kcal=3.0",
            "timeout_s=30",
            "keep_rejected=true",
        )
        ins = m["inputs"]
        assert ins["min_conformers"] == 2
        assert ins["metric"] == "rmsd"
        assert ins["clustering"] == "kmeans"
        assert ins["n_clusters"] == "auto"  # echo of input form
        assert ins["n_clusters_resolved"] == 3  # what the stub used
        assert ins["use_energies"] == "auto"
        assert ins["ewin_kcal"] == 3.0
        assert ins["ewin_applied"] is True
        assert ins["n_above_ewin"] == 0
        assert ins["timeout_s"] == 30
        assert ins["keep_rejected"] is True
        # method_version surfaces from the stub.
        assert ins["method_version"] == "0.0.0-test"

    def test_clustering_resolved_records_actual_algorithm(
        self, tmp_path, monkeypatch
    ):
        # Stub returns an algorithm name distinct from the config token —
        # the manifest must expose what was actually used, not what was
        # asked for, so a downstream consumer can audit ``auto`` choices.
        def _factory(n):
            r = _trivial_result(n)
            return MarcResult(
                accept_mask=r.accept_mask,
                cluster_ids=r.cluster_ids,
                cluster_distances=r.cluster_distances,
                n_clusters=r.n_clusters,
                algorithm_used="dbscan",
                method_version=r.method_version,
            )

        monkeypatch.setattr(mc, "run_marc", _stub_run_marc_factory(_factory))
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=1", "clustering=auto")
        assert m["inputs"]["clustering"] == "auto"
        assert m["inputs"]["clustering_resolved"] == "dbscan"

    def test_upstream_block_filled(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["upstream"]["pointer_schema"] == "wf.pointer.v1"
        assert m["upstream"]["ok"] is True
        assert m["upstream"]["manifest_path"].endswith("manifest.json")


# --------------------------------------------------------------------
# Contract self-test
# --------------------------------------------------------------------


class TestContractCompliance:
    def test_happy_manifest_satisfies_contract(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        problems = validate_conformer_screen(m, require_rejected_reason=True)
        assert problems == [], f"contract violations: {problems}"

    def test_partial_rejection_manifest_satisfies_contract(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            mc,
            "run_marc",
            _stub_run_marc_factory(
                lambda n: _trivial_result(n, accept_mask=[True, False, True][:n])
            ),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        problems = validate_conformer_screen(m, require_rejected_reason=True)
        assert problems == [], f"contract violations: {problems}"

    def test_keep_rejected_false_drops_rejected_bucket(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            mc,
            "run_marc",
            _stub_run_marc_factory(
                lambda n: _trivial_result(n, accept_mask=[True, False, True][:n])
            ),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "keep_rejected=false")
        assert m["artifacts"]["rejected"] == []
        # Counts still reflect what was accepted; rejected bucket is empty.
        assert m["inputs"]["n_accepted"] == 2
        assert m["inputs"]["n_rejected"] == 0
        # Contract still satisfied (require_rejected_reason mirrors keep_rejected).
        problems = validate_conformer_screen(m, require_rejected_reason=False)
        assert problems == [], f"contract violations: {problems}"


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_no_upstream_manifest(self, tmp_path):
        up = _empty_upstream(tmp_path)
        # Empty upstream -> no usable conformer bucket -> contract failure.
        m = _run_node(tmp_path, up)
        assert m["ok"] is False
        keys = {f.get("error", "") for f in m["failures"]}
        assert any(
            k.startswith("no_xyz_inputs_found_in_upstream_manifest") for k in keys
        )

    def test_navicat_marc_import_failure_recorded(self, tmp_path, monkeypatch):
        def _raise_import(**kwargs):
            raise ImportError("navicat-marc not installed in this env")

        monkeypatch.setattr(mc, "run_marc", _raise_import)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=1")
        # On import failure we fall back to accepting all + record a failure.
        # Manifest.ok depends on whether n_accepted > 0, so we check the
        # explicit failure key.
        keys = {f.get("error", "") for f in m["failures"]}
        assert any("import_navicat_marc_failed" in k for k in keys)

    def test_marc_raises_falls_back_and_records_failure(
        self, tmp_path, monkeypatch
    ):
        def _raise(**kwargs):
            raise RuntimeError("clustering exploded mid-run")

        monkeypatch.setattr(mc, "run_marc", _raise)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=1")
        keys = {f.get("error", "") for f in m["failures"]}
        assert any("clustering_failed_fallback_accept_all" in k for k in keys)

    def test_mask_length_mismatch_recorded(self, tmp_path, monkeypatch):
        # Stub returns a mask shorter than n_input -> caller must fail.
        def _short(*, ensemble_path, energies_kcal, **kwargs):
            return MarcResult(
                accept_mask=[True],  # wrong length: only 1 vs n_input=3
                cluster_ids=[0],
                cluster_distances=[0.0],
                n_clusters=1,
                algorithm_used="kmeans",
                method_version=None,
            )

        monkeypatch.setattr(mc, "run_marc", _short)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=1")
        keys = {f.get("error", "") for f in m["failures"]}
        assert any("clustering_returned_unexpected_mask_length" in k for k in keys)

    def test_zero_accepted_marks_failure(self, tmp_path, monkeypatch):
        # Reject everything -> contract requires ok=false + a structured failure.
        monkeypatch.setattr(
            mc,
            "run_marc",
            _stub_run_marc_factory(
                lambda n: MarcResult(
                    accept_mask=[False] * n,
                    cluster_ids=[0] * n,
                    cluster_distances=[1.0] * n,
                    n_clusters=0,
                    algorithm_used="kmeans",
                    method_version=None,
                )
            ),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=1")
        assert m["ok"] is False
        keys = {f.get("error", "") for f in m["failures"]}
        assert "no_conformers_accepted" in keys
        # No xyz_ensemble / xyz when nothing accepted.
        assert m["artifacts"].get("xyz_ensemble", []) == []
        assert m["artifacts"].get("xyz", []) == []

    def test_use_energies_true_without_energies(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3, with_per_item_energies=False)
        m = _run_node(tmp_path, up, "use_energies=true")
        assert m["ok"] is False
        keys = {f.get("error", "") for f in m["failures"]}
        assert any("use_energies_required_but_unavailable" in k for k in keys)

    def test_bad_min_conformers_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=0")
        assert m["ok"] is False
        keys = {f.get("error", "") for f in m["failures"]}
        assert any(k.startswith("argv_parse_failed") for k in keys)

    def test_bad_ewin_kcal_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "ewin_kcal=-1.0")
        assert m["ok"] is False
        keys = {f.get("error", "") for f in m["failures"]}
        assert any(k.startswith("argv_parse_failed") for k in keys)

    def test_bad_metric_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "metric=hausdorff")
        assert m["ok"] is False
        keys = {f.get("error", "") for f in m["failures"]}
        assert any(k.startswith("argv_parse_failed") for k in keys)

    def test_bad_clustering_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "clustering=hierarchical")
        assert m["ok"] is False
        keys = {f.get("error", "") for f in m["failures"]}
        assert any(k.startswith("argv_parse_failed") for k in keys)

    def test_bad_n_clusters_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "n_clusters=0")
        assert m["ok"] is False
        keys = {f.get("error", "") for f in m["failures"]}
        assert any(k.startswith("argv_parse_failed") for k in keys)

    def test_hard_fail_policy_returns_one(self, tmp_path):
        up = _empty_upstream(tmp_path)
        pointer_text = _pointer_text(up, ok=True)
        call_dir = tmp_path / "calls" / "marc"
        call_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = MarcScreen().invoke(
                ["marc_screen", pointer_text, "fail_policy=hard"]
            )
        finally:
            os.chdir(cwd)
        # hard failure: missing upstream conformers -> rc=1.
        assert rc == 1


# --------------------------------------------------------------------
# Ewin filter (pre-clustering energy cutoff)
# --------------------------------------------------------------------


def _upstream_with_explicit_energies(
    tmp_path: Path, energies: list[float]
) -> Path:
    """Build an upstream with caller-controlled rel_energy_kcal values."""
    up = tmp_path / "upstream"
    out = up / "outputs"
    confs_dir = out / "conformers"
    confs_dir.mkdir(parents=True)
    m = Manifest.skeleton(step="crest", cwd=str(up))
    confs: list[dict] = []
    for i, e in enumerate(energies, start=1):
        p = _make_xyz(confs_dir / f"conf_{i:04d}.xyz", label=f"c{i}")
        confs.append(
            {
                "index": i,
                "label": f"conf_{i:04d}",
                "path_abs": str(p.resolve()),
                "sha256": "f" * 64,
                "format": "xyz",
                "rel_energy_kcal": float(e),
            }
        )
    m.artifacts["conformers"] = confs
    m_path = out / "manifest.json"
    m.write(m_path)
    return m_path


class TestEwinFilter:
    def test_default_5kcal_drops_above_window(self, tmp_path, monkeypatch):
        # Energies [0.0, 2.0, 8.0]; default ewin=5.0 drops conf 3.
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _trivial_result(_split_n(Path(kwargs["ensemble_path"])))

        monkeypatch.setattr(mc, "run_marc", _capture)
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        m = _run_node(tmp_path, up, "min_conformers=1")

        # Conf 3 dropped by ewin BEFORE marc saw the ensemble.
        accepted = m["artifacts"]["accepted"]
        rejected = m["artifacts"]["rejected"]
        assert [a["index"] for a in accepted] == [1, 2]
        assert [r["index"] for r in rejected] == [3]
        assert rejected[0]["rejected_reason"] == REJECTED_REASON_EWIN
        # marc only saw the within-window energies.
        assert captured["energies_kcal"] == [0.0, 2.0]
        # Inputs block reflects ewin telemetry.
        assert m["inputs"]["ewin_applied"] is True
        assert m["inputs"]["n_above_ewin"] == 1

    def test_explicit_ewin_kcal_token(self, tmp_path, monkeypatch):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _trivial_result(_split_n(Path(kwargs["ensemble_path"])))

        monkeypatch.setattr(mc, "run_marc", _capture)
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        m = _run_node(tmp_path, up, "min_conformers=1", "ewin_kcal=1.0")
        accepted = m["artifacts"]["accepted"]
        assert [a["index"] for a in accepted] == [1]
        assert captured["energies_kcal"] == [0.0]
        assert m["inputs"]["n_above_ewin"] == 2

    def test_wide_ewin_keeps_all(self, tmp_path, monkeypatch):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _trivial_result(_split_n(Path(kwargs["ensemble_path"])))

        monkeypatch.setattr(mc, "run_marc", _capture)
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        m = _run_node(tmp_path, up, "min_conformers=1", "ewin_kcal=100.0")
        accepted = m["artifacts"]["accepted"]
        assert [a["index"] for a in accepted] == [1, 2, 3]
        assert m["inputs"]["n_above_ewin"] == 0

    def test_ewin_skipped_when_energies_unavailable(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3, with_per_item_energies=False)
        m = _run_node(tmp_path, up, "min_conformers=1")
        # Without energies, ewin filter cannot apply.
        assert m["inputs"]["ewin_applied"] is False
        assert m["inputs"]["n_above_ewin"] == 0
        assert [a["index"] for a in m["artifacts"]["accepted"]] == [1, 2, 3]

    def test_ewin_skipped_when_use_energies_false(self, tmp_path, stub_marc_keep_all):
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        m = _run_node(tmp_path, up, "min_conformers=1", "use_energies=false")
        # use_energies=false short-circuits ewin even though energies exist.
        assert m["inputs"]["ewin_applied"] is False
        assert [a["index"] for a in m["artifacts"]["accepted"]] == [1, 2, 3]

    def test_ewin_runs_before_clusterer_distinct_reasons(
        self, tmp_path, monkeypatch
    ):
        # End-to-end with both rejection types: ewin drops conf 4,
        # clusterer rejects conf 2 from the [1,2,3] survivors.
        def _factory(n):
            assert n == 3, "ewin should filter conf 4 before marc sees it"
            return _trivial_result(n, accept_mask=[True, False, True])

        monkeypatch.setattr(mc, "run_marc", _stub_run_marc_factory(_factory))
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 0.5, 1.0, 8.0])
        m = _run_node(tmp_path, up, "min_conformers=1", "ewin_kcal=2.0")

        rejected = m["artifacts"]["rejected"]
        rejection_map = {r["index"]: r["rejected_reason"] for r in rejected}
        assert rejection_map == {
            2: REJECTED_REASON,
            4: REJECTED_REASON_EWIN,
        }
        accepted = m["artifacts"]["accepted"]
        assert [a["index"] for a in accepted] == [1, 3]


# --------------------------------------------------------------------
# Node wiring
# --------------------------------------------------------------------


class TestNodeWiring:
    def test_method_constants_exposed(self):
        assert mc.METHOD_NAME == "navicat_marc"
        assert mc.REJECTED_REASON == "cluster_dup"
        assert mc.REJECTED_REASON_EWIN == "above_energy_window"

    def test_node_class_attrs(self):
        assert MarcScreen.step == "marc_screen"
        assert MarcScreen.accepts_upstream is True
        assert MarcScreen.requires_upstream is True

    def test_invoke_factory_main_callable(self):
        # The module exposes ``main`` which is the engine entrypoint.
        assert callable(mc.main)

    def test_marcresult_dataclass_shape(self):
        r = _trivial_result(3)
        assert r.accept_mask == [True, True, True]
        assert r.cluster_ids == [0, 1, 2]
        assert r.cluster_distances == [0.0, 0.0, 0.0]
        assert r.n_clusters == 3
        assert r.algorithm_used == "kmeans"
        assert r.method_version == "0.0.0-test"
