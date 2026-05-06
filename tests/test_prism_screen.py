"""Tests for the ``wf-prism`` conformer-screening node.

prism-pruner requires Python >= 3.12 + numpy and is not available in the
test sandbox, so we monkeypatch the entire library-using core
(:func:`scripps_workflow.nodes.prism_screen.run_prism_pruner`) to return
a chosen accept mask. That isolates the node's wiring (source
discovery, staging, contract self-test, manifest shape) from the
library's internals.

Coverage:

    * Pure helpers: normalize_use_energies, discover_conformer_sources,
      extract_energy_kcal_from_item, find_energy_file,
      parse_generic_energies_kcal, concat_xyz_files, resolve_use_energies.
    * Happy paths through each source mode (many / ensemble / single).
    * Energy harvesting: per-item, fallback to upstream energies file,
      use_energies tri-state.
    * Contract self-test: validate_conformer_screen returns no problems
      on the produced manifest.
    * Failure paths: no upstream, missing inputs, prism import fails,
      prune raises, zero accepted, use_energies=true without energies,
      bad config tokens, hard-fail policy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripps_workflow.contracts.conformer_screen import (
    assert_valid_conformer_screen,
    validate_conformer_screen,
)
from scripps_workflow.nodes import prism_screen as ps
from scripps_workflow.nodes.prism_screen import (
    METHOD_NAME,
    REJECTED_REASON,
    REJECTED_REASON_EWIN,
    PrismScreen,
    apply_ewin_filter,
    concat_xyz_files,
    discover_conformer_sources,
    extract_energy_kcal_from_item,
    find_energy_file,
    normalize_use_energies,
    parse_generic_energies_kcal,
    resolve_use_energies,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------


class TestNormalizeUseEnergies:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, "auto"),
            ("", "auto"),
            ("auto", "auto"),
            ("AUTO", "auto"),
            ("true", "true"),
            ("True", "true"),
            ("1", "true"),
            ("yes", "true"),
            ("on", "true"),
            ("false", "false"),
            ("FALSE", "false"),
            ("0", "false"),
            ("no", "false"),
            ("off", "false"),
            ("garbage", "auto"),  # unknown -> auto, not raise
        ],
    )
    def test_aliases(self, raw, expected):
        assert normalize_use_energies(raw) == expected


class TestDiscoverConformerSources:
    def _xyz(self, p: Path, comment: str = "x") -> Path:
        p.write_text(f"1\n{comment}\nC 0 0 0\n")
        return p

    def test_accepted_wins_over_conformers(self, tmp_path):
        a = self._xyz(tmp_path / "a.xyz")
        c = self._xyz(tmp_path / "c.xyz")
        artifacts = {
            "accepted": [{"path_abs": str(a), "index": 1}],
            "conformers": [{"path_abs": str(c), "index": 1}],
        }
        mode, items = discover_conformer_sources(artifacts)
        assert mode == "many"
        assert len(items) == 1
        assert Path(items[0]["path_abs"]).name == "a.xyz"

    def test_selected_wins_over_conformers(self, tmp_path):
        s = self._xyz(tmp_path / "s.xyz")
        c = self._xyz(tmp_path / "c.xyz")
        artifacts = {
            "selected": [{"path_abs": str(s)}],
            "conformers": [{"path_abs": str(c)}],
        }
        mode, items = discover_conformer_sources(artifacts)
        assert mode == "many"
        assert Path(items[0]["path_abs"]).name == "s.xyz"

    def test_falls_through_to_conformers(self, tmp_path):
        c = self._xyz(tmp_path / "c.xyz")
        artifacts = {"conformers": [{"path_abs": str(c)}]}
        mode, items = discover_conformer_sources(artifacts)
        assert mode == "many"
        assert len(items) == 1

    def test_xyz_ensemble_fallback(self, tmp_path):
        e = self._xyz(tmp_path / "ens.xyz")
        artifacts = {"xyz_ensemble": [{"path_abs": str(e), "label": "ens"}]}
        mode, items = discover_conformer_sources(artifacts)
        assert mode == "ensemble"
        assert len(items) == 1

    def test_xyz_single_fallback(self, tmp_path):
        x = self._xyz(tmp_path / "best.xyz")
        artifacts = {"xyz": [{"path_abs": str(x), "label": "best"}]}
        mode, items = discover_conformer_sources(artifacts)
        assert mode == "single"
        assert len(items) == 1

    def test_none_when_no_buckets(self, tmp_path):
        mode, items = discover_conformer_sources({})
        assert mode == "none"
        assert items == []

    def test_filters_missing_files(self, tmp_path):
        # path is well-formed but file does not exist on disk.
        artifacts = {
            "conformers": [
                {"path_abs": str(tmp_path / "missing.xyz"), "index": 1},
            ],
        }
        mode, items = discover_conformer_sources(artifacts)
        assert mode == "none"
        assert items == []

    def test_skips_non_xyz_paths(self, tmp_path):
        # A real file but wrong extension.
        p = tmp_path / "wrong.txt"
        p.write_text("not xyz")
        artifacts = {"conformers": [{"path_abs": str(p)}]}
        mode, _ = discover_conformer_sources(artifacts)
        assert mode == "none"

    def test_string_entries_normalized(self, tmp_path):
        # Tolerate plain string entries the way _artifact_items promises.
        p = self._xyz(tmp_path / "p.xyz")
        artifacts = {"conformers": [str(p)]}
        mode, items = discover_conformer_sources(artifacts)
        assert mode == "many"
        assert items[0]["path_abs"] == str(p.resolve())

    def test_non_list_bucket_ignored(self, tmp_path):
        # If a downstream consumer fed us garbage, don't crash.
        artifacts = {"conformers": "not-a-list"}
        mode, _ = discover_conformer_sources(artifacts)
        assert mode == "none"


class TestExtractEnergyKcalFromItem:
    def test_prefers_rel_energy_kcal(self):
        assert extract_energy_kcal_from_item(
            {"rel_energy_kcal": 1.5, "rel_energy": 9.9}
        ) == 1.5

    def test_falls_back_to_rel_energy(self):
        assert extract_energy_kcal_from_item({"rel_energy": 0.5}) == 0.5

    def test_int_coerced_to_float(self):
        assert extract_energy_kcal_from_item({"rel_energy_kcal": 2}) == 2.0

    def test_string_float_parsed(self):
        # External JSON sometimes carries numbers as strings.
        assert extract_energy_kcal_from_item({"rel_energy_kcal": "0.7"}) == 0.7

    def test_missing_returns_none(self):
        assert extract_energy_kcal_from_item({"label": "x"}) is None

    def test_unparseable_returns_none(self):
        assert extract_energy_kcal_from_item({"rel_energy_kcal": "abc"}) is None
        assert extract_energy_kcal_from_item({"rel_energy_kcal": None}) is None


class TestFindEnergyFile:
    def test_label_contains_energies(self, tmp_path):
        p = tmp_path / "blob.txt"
        p.write_text("0.0\n0.5\n")
        artifacts = {
            "files": [{"path_abs": str(p), "label": "crest_energies"}],
        }
        assert find_energy_file(artifacts) == p.resolve()

    def test_filename_endswith_energies(self, tmp_path):
        p = tmp_path / "crest.energies"
        p.write_text("0.0\n0.5\n")
        artifacts = {"files": [{"path_abs": str(p), "label": "anything"}]}
        assert find_energy_file(artifacts) == p.resolve()

    def test_returns_none_when_absent(self, tmp_path):
        p = tmp_path / "irrelevant.txt"
        p.write_text("hello")
        artifacts = {"files": [{"path_abs": str(p), "label": "log"}]}
        assert find_energy_file(artifacts) is None

    def test_skips_missing_files(self, tmp_path):
        artifacts = {
            "files": [
                {"path_abs": str(tmp_path / "missing.energies"), "label": "x"},
            ],
        }
        assert find_energy_file(artifacts) is None


class TestParseGenericEnergiesKcal:
    def test_last_float_per_line(self, tmp_path):
        p = tmp_path / "e"
        p.write_text("1 -1.234 0.0\n2 -1.230 0.5\n3 -1.225 1.0\n")
        assert parse_generic_energies_kcal(p) == [0.0, 0.5, 1.0]

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "e"
        p.write_text("\n0.0\n\n0.5\n\n")
        assert parse_generic_energies_kcal(p) == [0.0, 0.5]

    def test_unparseable_lines_dropped(self, tmp_path):
        # Differs from crest helper: this one drops, doesn't yield None.
        p = tmp_path / "e"
        p.write_text("0.0\nNA\n0.5\n# header\n1.0\n")
        assert parse_generic_energies_kcal(p) == [0.0, 0.5, 1.0]

    def test_csv_tokens(self, tmp_path):
        p = tmp_path / "e"
        p.write_text("1, -1.234, 0.0\n2, -1.230, 0.5\n")
        assert parse_generic_energies_kcal(p) == [0.0, 0.5]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "e"
        p.write_text("")
        assert parse_generic_energies_kcal(p) == []


class TestConcatXyzFiles:
    def test_round_trip(self, tmp_path):
        a = tmp_path / "a.xyz"
        a.write_text("1\nA\nC 0 0 0\n")
        b = tmp_path / "b.xyz"
        b.write_text("1\nB\nO 0 0 1\n")
        out = tmp_path / "ens.xyz"
        concat_xyz_files([a, b], out)
        text = out.read_text()
        assert "A" in text and "B" in text
        # Both blocks present in order.
        assert text.index("A") < text.index("B")

    def test_appends_missing_trailing_newline(self, tmp_path):
        a = tmp_path / "a.xyz"
        a.write_text("1\nA\nC 0 0 0")  # no trailing newline
        b = tmp_path / "b.xyz"
        b.write_text("1\nB\nO 0 0 1\n")
        out = tmp_path / "ens.xyz"
        concat_xyz_files([a, b], out)
        # B's header line is on its own line, not glued to "C 0 0 0".
        assert "C 0 0 0\n1\nB\n" in out.read_text()


class TestResolveUseEnergies:
    def test_true_forces_true(self):
        assert resolve_use_energies("true", have_energies=False) is True

    def test_false_forces_false(self):
        assert resolve_use_energies("false", have_energies=True) is False

    def test_auto_with_energies(self):
        assert resolve_use_energies("auto", have_energies=True) is True

    def test_auto_without_energies(self):
        assert resolve_use_energies("auto", have_energies=False) is False


class TestApplyEwinFilter:
    def test_keeps_within_window(self):
        # 0.0 is the min, threshold 5.0 -> 0.0 and 2.0 kept; 6.0 dropped.
        assert apply_ewin_filter([0.0, 2.0, 6.0], 5.0) == [True, True, False]

    def test_threshold_inclusive_at_boundary(self):
        # Closed interval: equality at the threshold is kept.
        assert apply_ewin_filter([0.0, 5.0], 5.0) == [True, True]

    def test_zero_threshold_keeps_only_min_and_ties(self):
        # Edge case: ewin=0 means "only the lowest (and ties) survive".
        assert apply_ewin_filter([0.0, 0.0, 0.0001, 1.0], 0.0) == [
            True, True, False, False,
        ]

    def test_min_finds_lowest_not_first(self):
        # Reference is the minimum across the list, not the first entry.
        assert apply_ewin_filter([10.0, 0.0, 4.0, 100.0], 5.0) == [
            False, True, True, False,
        ]

    def test_no_energies_returns_all_true(self):
        # Filter is a no-op when nothing to compare.
        assert apply_ewin_filter([None, None, None], 5.0) == [True, True, True]

    def test_partial_missing_energies_kept(self):
        # Missing energies are not grounds for *energy*-based exclusion.
        assert apply_ewin_filter([0.0, None, 100.0], 5.0) == [True, True, False]

    def test_empty_list(self):
        assert apply_ewin_filter([], 5.0) == []


# --------------------------------------------------------------------
# End-to-end fixtures
# --------------------------------------------------------------------


def _make_xyz(path: Path, *, atoms: int = 3, label: str = "x") -> Path:
    """Write a tiny but well-formed xyz file."""
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
    """Write a multi-frame xyz blob."""
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

    # Also publish an xyz_ensemble for completeness (some upstreams do).
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
    """Upstream with NO 'conformers' bucket — only a multi-xyz ensemble.

    Forces the node into ``mode == 'ensemble'``.
    """
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
    """Upstream with only an artifacts.xyz single record."""
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


def _stub_run_prism_factory(mask_factory):
    """Build a stub for ``run_prism_pruner`` that returns a chosen mask.

    ``mask_factory`` is a callable ``(n_input) -> list[bool]`` so tests
    can express masks relative to the number of conformers actually fed
    to the pruner without recomputing them.
    """

    def _stub(*, ensemble_path, energies_kcal, **kwargs):
        # Read the ensemble to learn n_input. Counting "atom-count
        # header" lines is fragile, but split_multixyz handles it
        # correctly and we already trust it (separately tested).
        from scripps_workflow.nodes.crest import split_multixyz

        text = Path(ensemble_path).read_text(encoding="utf-8")
        n = len(split_multixyz(text))
        return mask_factory(n)

    return _stub


@pytest.fixture
def stub_prism_keep_all(monkeypatch):
    """Default stub: accept every conformer."""
    monkeypatch.setattr(
        ps,
        "run_prism_pruner",
        _stub_run_prism_factory(lambda n: [True] * n),
    )


def _run_node(
    tmp_path: Path,
    upstream_manifest: Path,
    *config_tokens: str,
    ok_pointer: bool = True,
) -> dict:
    """Invoke PrismScreen against an existing upstream manifest and
    return the parsed manifest dict."""
    pointer_text = _pointer_text(upstream_manifest, ok=ok_pointer)

    call_dir = tmp_path / "calls" / "prism"
    call_dir.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = PrismScreen().invoke(["prism_screen", pointer_text, *config_tokens])
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
    def test_many_mode_keep_all(self, tmp_path, stub_prism_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["ok"] is True
        assert m["step"] == "prism_screen"

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
        # mask = [True, False, True] -> conf 2 rejected.
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True, False, True][:n]),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["ok"] is True

        accepted = m["artifacts"]["accepted"]
        rejected = m["artifacts"]["rejected"]
        assert [a["index"] for a in accepted] == [1, 3]
        assert [r["index"] for r in rejected] == [2]
        # Rejected reason recorded so the contract stays happy.
        assert rejected[0]["rejected_reason"] == REJECTED_REASON
        # Files actually copied to outputs/rejected/ on disk.
        assert Path(rejected[0]["path_abs"]).exists()

    def test_rel_energy_kcal_shifted_to_min(self, tmp_path, monkeypatch):
        # Reject conf 1 (the lowest), keep 2 and 3. Then the min energy
        # within the accepted set should be 0.5 (conf 2), and rel values
        # should be 0.0 for conf 2 and 0.5 for conf 3.
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [False, True, True][:n]),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        accepted = m["artifacts"]["accepted"]
        # Verify shifted to min within accepted.
        rels = [a["rel_energy_kcal"] for a in accepted]
        assert rels == pytest.approx([0.0, 0.5])
        # Absolute (input-relative) value preserved alongside.
        assert accepted[0]["energy_kcal"] == pytest.approx(0.5)
        assert accepted[1]["energy_kcal"] == pytest.approx(1.0)

    def test_best_picked_by_rel_energy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        # Conf 1 has rel 0.0 -> chosen as best.
        assert m["inputs"]["best_chosen_by"] == "lowest_rel_energy_kcal"
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1 and xyz[0]["label"] == "best"
        assert "conf1" in Path(xyz[0]["path_abs"]).read_text()

    def test_best_falls_back_to_first_index_when_no_energies(
        self, tmp_path, monkeypatch
    ):
        # Upstream has no per-item energies; auto -> use_energies=false ->
        # best chosen by first accepted index.
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        up = _upstream_with_conformers(tmp_path, n=2, with_per_item_energies=False)
        m = _run_node(tmp_path, up)
        assert m["inputs"]["best_chosen_by"] == "first_accepted_by_index"

    def test_ensemble_mode_splits_multixyz(self, tmp_path, stub_prism_keep_all):
        up = _upstream_with_ensemble(tmp_path, n=4)
        m = _run_node(tmp_path, up)
        assert m["ok"] is True
        assert m["inputs"]["source_mode"] == "ensemble"
        accepted = m["artifacts"]["accepted"]
        assert [a["index"] for a in accepted] == [1, 2, 3, 4]
        # No per-item energies in this mode -> rel_energy_kcal not set.
        for a in accepted:
            assert "rel_energy_kcal" not in a

    def test_single_mode_passes_through(self, tmp_path, monkeypatch):
        # n_input=1 < default min_conformers=3 -> auto-accept.
        # run_prism_pruner should NOT be called; if it is, blow up loudly.
        called = {"n": 0}

        def _explode(**kwargs):
            called["n"] += 1
            return [True]

        monkeypatch.setattr(ps, "run_prism_pruner", _explode)
        up = _upstream_with_single_xyz(tmp_path)
        m = _run_node(tmp_path, up)
        assert m["ok"] is True
        assert m["inputs"]["source_mode"] == "single"
        assert m["inputs"]["n_input"] == 1
        assert m["inputs"]["n_accepted"] == 1
        assert called["n"] == 0, "should skip pruner when below min_conformers"

    def test_below_min_conformers_skips_pruner(self, tmp_path, monkeypatch):
        called = {"n": 0}

        def _explode(**kwargs):
            called["n"] += 1
            return [True, True]

        monkeypatch.setattr(ps, "run_prism_pruner", _explode)
        up = _upstream_with_conformers(tmp_path, n=2)
        m = _run_node(tmp_path, up, "min_conformers=3")
        assert m["ok"] is True
        assert called["n"] == 0
        assert [a["index"] for a in m["artifacts"]["accepted"]] == [1, 2]

    def test_use_energies_passed_to_pruner_when_available(
        self, tmp_path, monkeypatch
    ):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            from scripps_workflow.nodes.crest import split_multixyz

            n = len(split_multixyz(Path(kwargs["ensemble_path"]).read_text()))
            return [True] * n

        monkeypatch.setattr(ps, "run_prism_pruner", _capture)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "use_energies=auto")
        # Auto + per-item energies present -> pruner gets the list.
        assert captured["energies_kcal"] == [0.0, 0.5, 1.0]
        assert m["inputs"]["use_energies_resolved"] is True

    def test_use_energies_false_blocks_energy_passthrough(
        self, tmp_path, monkeypatch
    ):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            from scripps_workflow.nodes.crest import split_multixyz

            n = len(split_multixyz(Path(kwargs["ensemble_path"]).read_text()))
            return [True] * n

        monkeypatch.setattr(ps, "run_prism_pruner", _capture)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "use_energies=false")
        assert captured["energies_kcal"] is None
        assert m["inputs"]["use_energies_resolved"] is False

    def test_upstream_energies_file_used_when_per_item_missing(
        self, tmp_path, monkeypatch
    ):
        """When per-item energies are absent, fall back to a *.energies file
        in artifacts.files."""
        # Build an upstream WITHOUT per-item energies but WITH a parallel
        # crest.energies file in artifacts.files.
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
        # last column is rel kcal: 0.0, 0.5, 1.0
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
            return [True, True, True]

        monkeypatch.setattr(ps, "run_prism_pruner", _capture)

        m = _run_node(tmp_path, m_up_path)
        assert captured["energies_kcal"] == [0.0, 0.5, 1.0]
        # Manifest records that we ingested the upstream file.
        labels = {f["label"] for f in m["artifacts"]["files"]}
        assert "upstream_energies_used" in labels

    def test_accepted_ensemble_and_best_files_on_disk(
        self, tmp_path, stub_prism_keep_all
    ):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        ens = m["artifacts"]["xyz_ensemble"]
        assert len(ens) == 1 and ens[0]["label"] == "accepted_ensemble"
        assert Path(ens[0]["path_abs"]).exists()

        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1 and xyz[0]["label"] == "best"
        assert Path(xyz[0]["path_abs"]).exists()

    def test_prism_energies_file_emitted(self, tmp_path, stub_prism_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        files = m["artifacts"]["files"]
        labels = {f["label"] for f in files}
        assert "prism_energies" in labels
        # File parsable round-trip with our own helper.
        e = next(f for f in files if f["label"] == "prism_energies")
        rels = parse_generic_energies_kcal(Path(e["path_abs"]))
        assert rels == pytest.approx([0.0, 0.5, 1.0])

    def test_inputs_block_typed(self, tmp_path, stub_prism_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(
            tmp_path,
            up,
            "min_conformers=2",
            "moi_pruning=false",
            "rmsd_pruning=true",
            "rot_corr_rmsd_pruning=false",
            "use_energies=auto",
            "max_dE_kcal=2.5",
            "ewin_kcal=3.0",
            "timeout_s=30",
            "keep_rejected=true",
        )
        ins = m["inputs"]
        assert ins["min_conformers"] == 2
        assert ins["moi_pruning"] is False
        assert ins["rmsd_pruning"] is True
        assert ins["rot_corr_rmsd_pruning"] is False
        assert ins["use_energies"] == "auto"
        assert ins["max_dE_kcal"] == 2.5
        assert ins["ewin_kcal"] == 3.0
        # All three upstream energies (0.0, 0.5, 1.0) within a 3.0 window.
        assert ins["ewin_applied"] is True
        assert ins["n_above_ewin"] == 0
        assert ins["timeout_s"] == 30
        assert ins["keep_rejected"] is True

    def test_upstream_block_filled(self, tmp_path, stub_prism_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["upstream"]["pointer_schema"] == "wf.pointer.v1"
        assert m["upstream"]["ok"] is True
        assert m["upstream"]["manifest_path"].endswith("manifest.json")


# --------------------------------------------------------------------
# Contract self-test
# --------------------------------------------------------------------


class TestContractCompliance:
    def test_happy_manifest_satisfies_contract(self, tmp_path, stub_prism_keep_all):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        # Validator returns no problems on the produced manifest.
        problems = validate_conformer_screen(m, require_rejected_reason=True)
        assert problems == [], f"unexpected contract problems: {problems}"

    def test_partial_rejection_satisfies_contract(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True, False, True][:n]),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        # The strict raising form mirrors what some downstream callers
        # use in their own self-tests.
        assert_valid_conformer_screen(m, require_rejected_reason=True)

    def test_no_contract_violation_in_failures(
        self, tmp_path, stub_prism_keep_all
    ):
        # Belt-and-suspenders: run() appends contract problems as
        # 'contract_violation: ...' failures. None should be present in
        # a healthy run.
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        for f in m["failures"]:
            assert not f["error"].startswith("contract_violation:"), f


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_no_upstream_manifest(self, tmp_path):
        # Pointer is well-formed but the manifest path doesn't exist.
        ghost = tmp_path / "nope" / "manifest.json"
        # Pointer.of validates on construction; build one by hand.
        ptr = '{"schema":"wf.pointer.v1","ok":true,"manifest_path":"' \
            + str(ghost) + '"}'
        call_dir = tmp_path / "call"
        call_dir.mkdir()
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = PrismScreen().invoke(["prism_screen", ptr])
        finally:
            os.chdir(cwd)
        assert rc == 0
        m = json.loads((call_dir / "outputs" / "manifest.json").read_text())
        assert m["ok"] is False

    def test_no_xyz_inputs_in_upstream(self, tmp_path):
        up = _empty_upstream(tmp_path)
        m = _run_node(tmp_path, up)
        assert m["ok"] is False
        assert any(
            "no_xyz_inputs_found_in_upstream_manifest" in f["error"]
            for f in m["failures"]
        )

    def test_prism_import_failure(self, tmp_path, monkeypatch):
        def _raises(**kwargs):
            raise ImportError("prism_pruner not installed")

        monkeypatch.setattr(ps, "run_prism_pruner", _raises)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["ok"] is False
        assert any(
            "import_prism_pruner_failed" in f["error"] for f in m["failures"]
        )

    def test_pruner_raises_other_exception(self, tmp_path, monkeypatch):
        def _raises(**kwargs):
            raise RuntimeError("solver blew up")

        monkeypatch.setattr(ps, "run_prism_pruner", _raises)
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["ok"] is False
        assert any(
            "pruning_failed_fallback_accept_all" in f["error"]
            for f in m["failures"]
        )
        # Fallback: all accepted, so manifest still has accepted records.
        assert len(m["artifacts"]["accepted"]) == 3

    def test_pruner_returns_wrong_length_mask(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True, False]),  # always 2
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up)
        assert m["ok"] is False
        assert any(
            "pruning_returned_unexpected_mask_length" in f["error"]
            for f in m["failures"]
        )

    def test_zero_accepted_marks_failure(self, tmp_path, monkeypatch):
        # Force every conformer rejected. First, override min_conformers
        # so the pruner actually runs at n_input=3.
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [False] * n),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=2")
        assert m["ok"] is False
        assert any(
            "no_conformers_accepted" in f["error"] for f in m["failures"]
        )
        # Contract expects ok=false + structured failure when accepted
        # is empty -- both must be present here.
        problems = validate_conformer_screen(m, require_rejected_reason=True)
        assert problems == [], f"unexpected contract problems: {problems}"

    def test_use_energies_true_without_energies(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        up = _upstream_with_conformers(tmp_path, n=3, with_per_item_energies=False)
        m = _run_node(tmp_path, up, "use_energies=true")
        assert m["ok"] is False
        assert any(
            "use_energies_required_but_unavailable" in f["error"]
            for f in m["failures"]
        )

    def test_bad_min_conformers_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "min_conformers=0")
        assert m["ok"] is False
        assert any(
            "argv_parse_failed" in f["error"] for f in m["failures"]
        )

    def test_bad_max_dE_kcal_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "max_dE_kcal=-0.1")
        assert m["ok"] is False
        assert any(
            "argv_parse_failed" in f["error"] for f in m["failures"]
        )

    def test_bad_timeout_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "timeout_s=2")
        assert m["ok"] is False
        assert any(
            "argv_parse_failed" in f["error"] for f in m["failures"]
        )

    def test_bad_ewin_kcal_is_argv_parse_failed(self, tmp_path):
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "ewin_kcal=-1.0")
        assert m["ok"] is False
        assert any(
            "argv_parse_failed" in f["error"] for f in m["failures"]
        )

    def test_hard_fail_policy_returns_one(self, tmp_path):
        # With fail_policy=hard on a soft failure, invoke() returns 1.
        up = _empty_upstream(tmp_path)
        pointer_text = _pointer_text(up, ok=True)
        call_dir = tmp_path / "calls" / "prism"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = PrismScreen().invoke(
                ["prism_screen", pointer_text, "fail_policy=hard"]
            )
        finally:
            os.chdir(cwd)
        assert rc == 1

    def test_keep_rejected_false_skips_rejected_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True, False, True][:n]),
        )
        up = _upstream_with_conformers(tmp_path, n=3)
        m = _run_node(tmp_path, up, "keep_rejected=false")
        # Rejected bucket is empty when keep_rejected=false.
        assert m["artifacts"]["rejected"] == []
        # n_rejected counter still reflects the *bucket* length (the
        # contract cross-checks against artifacts.rejected, not the
        # logical count). This documents current behavior.
        assert m["inputs"]["n_rejected"] == 0
        # Validator should not complain about missing rejected_reason
        # because the bucket is empty.
        problems = validate_conformer_screen(m, require_rejected_reason=False)
        assert problems == []


# --------------------------------------------------------------------
# Post-pruning energy cutoff (ewin_kcal) — end to end
# --------------------------------------------------------------------


def _upstream_with_explicit_energies(
    tmp_path: Path, energies: list[float | None]
) -> Path:
    """Like _upstream_with_conformers but with caller-controlled energies.

    Used by the ewin tests where we need to position conformers at
    specific kcal/mol values relative to the lowest one.
    """
    n = len(energies)
    up = tmp_path / "upstream"
    out = up / "outputs"
    confs_dir = out / "conformers"
    confs_dir.mkdir(parents=True)
    m = Manifest.skeleton(step="crest", cwd=str(up))
    confs: list[dict] = []
    for i, e in enumerate(energies, start=1):
        p = _make_xyz(confs_dir / f"conf_{i:04d}.xyz", label=f"c{i}")
        rec: dict = {
            "index": i,
            "label": f"conf_{i:04d}",
            "path_abs": str(p.resolve()),
            "sha256": "f" * 64,
            "format": "xyz",
        }
        if isinstance(e, float):
            rec["rel_energy_kcal"] = e
        confs.append(rec)
    m.artifacts["conformers"] = confs
    m_path = out / "manifest.json"
    m.write(m_path)
    return m_path


class TestEwinFilter:
    def test_default_5kcal_drops_above_window(self, tmp_path, monkeypatch):
        """Default ewin_kcal=5.0; energies 0/2/8 -> conf 3 dropped post-pruning.

        ``min_conformers=1`` so the pruner runs (otherwise the framework
        would short-circuit and the stub would never fire, leaving us
        unable to inspect what reached the pruner). The pruner now sees
        the full input ensemble — the ewin filter is applied AFTER, on
        survivors only.
        """
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            from scripps_workflow.nodes.crest import split_multixyz

            n = len(split_multixyz(Path(kwargs["ensemble_path"]).read_text()))
            return [True] * n

        monkeypatch.setattr(ps, "run_prism_pruner", _capture)
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        m = _run_node(tmp_path, up, "min_conformers=1")
        # The pruner sees the FULL input ensemble (post-pruning ewin).
        assert captured["energies_kcal"] == [0.0, 2.0, 8.0]
        # Conf 3 in rejected with the ewin reason.
        rejected = m["artifacts"]["rejected"]
        assert [r["index"] for r in rejected] == [3]
        assert rejected[0]["rejected_reason"] == REJECTED_REASON_EWIN
        # Accepted = conf 1 + conf 2.
        assert [a["index"] for a in m["artifacts"]["accepted"]] == [1, 2]
        # Inputs block reflects the filter.
        ins = m["inputs"]
        assert ins["ewin_applied"] is True
        assert ins["n_above_ewin"] == 1
        assert ins["ewin_kcal"] == 5.0

    def test_explicit_ewin_kcal_token(self, tmp_path, monkeypatch):
        """Tighter window: ewin_kcal=1.0 drops both conf 2 (2.0) and conf 3 (8.0)."""
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        # min_conformers=1 forces the pruner to run on the full ensemble
        # (post-pruning ewin then drops conf 2 and conf 3).
        m = _run_node(tmp_path, up, "ewin_kcal=1.0", "min_conformers=1")
        rejected = m["artifacts"]["rejected"]
        assert [r["index"] for r in rejected] == [2, 3]
        for r in rejected:
            assert r["rejected_reason"] == REJECTED_REASON_EWIN
        assert m["inputs"]["n_above_ewin"] == 2

    def test_wide_ewin_kcal_keeps_all(self, tmp_path, monkeypatch):
        """ewin_kcal=100 effectively disables the filter."""
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        m = _run_node(tmp_path, up, "ewin_kcal=100.0")
        assert [a["index"] for a in m["artifacts"]["accepted"]] == [1, 2, 3]
        assert m["inputs"]["n_above_ewin"] == 0

    def test_ewin_skipped_when_energies_unavailable(self, tmp_path, monkeypatch):
        """No per-item energies AND no upstream energies file -> filter
        silently skipped, ewin_applied=False."""
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        up = _upstream_with_conformers(
            tmp_path, n=3, with_per_item_energies=False
        )
        m = _run_node(tmp_path, up)
        assert m["inputs"]["ewin_applied"] is False
        assert m["inputs"]["n_above_ewin"] == 0
        assert [a["index"] for a in m["artifacts"]["accepted"]] == [1, 2, 3]

    def test_ewin_skipped_when_use_energies_false(self, tmp_path, monkeypatch):
        """Explicit use_energies=false suppresses ewin even when energies
        ARE present — the user's stated intent wins."""
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        m = _run_node(tmp_path, up, "use_energies=false")
        assert m["inputs"]["ewin_applied"] is False
        # Conf 3 NOT rejected — filter skipped.
        assert [a["index"] for a in m["artifacts"]["accepted"]] == [1, 2, 3]

    def test_ewin_runs_after_pruner_distinct_reasons(
        self, tmp_path, monkeypatch
    ):
        """End-to-end: pruner drops conf 2 from the FULL ensemble, then
        ewin drops conf 3 from the survivors. Final state has both
        reasons recorded distinctly.

        ``min_conformers=1`` so the pruner actually runs (otherwise the
        framework's "skip pruner if too few" guard would accept
        everything and we couldn't observe the pruner-rejection branch).
        """
        captured: dict = {}

        def _stub(**kwargs):
            captured.update(kwargs)
            # Pruner sees the full input ensemble; rejects conf 2 as a
            # duplicate. Conf 3 survives the pruner but the ewin filter
            # will drop it for being above the energy window.
            return [True, False, True]

        monkeypatch.setattr(ps, "run_prism_pruner", _stub)
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 2.0, 8.0])
        m = _run_node(tmp_path, up, "min_conformers=1")
        # Pruner saw the full ensemble (no pre-pruning ewin filter).
        assert captured["energies_kcal"] == [0.0, 2.0, 8.0]

        accepted = m["artifacts"]["accepted"]
        rejected = m["artifacts"]["rejected"]
        assert [a["index"] for a in accepted] == [1]
        assert [r["index"] for r in rejected] == [2, 3]
        reasons_by_idx = {r["index"]: r["rejected_reason"] for r in rejected}
        assert reasons_by_idx == {
            2: REJECTED_REASON,
            3: REJECTED_REASON_EWIN,
        }

    def test_ewin_zero_keeps_only_lowest_min_conformers_one(
        self, tmp_path, monkeypatch
    ):
        """ewin_kcal=0 keeps only the lowest-energy conformer (and ties).
        With min_conformers=1, that's enough to satisfy the contract."""
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 5.0, 10.0])
        m = _run_node(tmp_path, up, "ewin_kcal=0", "min_conformers=1")
        accepted = m["artifacts"]["accepted"]
        rejected = m["artifacts"]["rejected"]
        assert [a["index"] for a in accepted] == [1]
        assert [r["index"] for r in rejected] == [2, 3]
        for r in rejected:
            assert r["rejected_reason"] == REJECTED_REASON_EWIN
        assert m["inputs"]["n_above_ewin"] == 2
        # Contract still satisfied (single accepted, both ewin-rejects).
        problems = validate_conformer_screen(m, require_rejected_reason=True)
        assert problems == [], f"unexpected contract problems: {problems}"

    def test_ewin_drops_everything_marks_failure(self, tmp_path, monkeypatch):
        """Pathological config: tighter-than-zero window via every energy
        being far from the min. All ewin-rejected -> ok=false +
        no_conformers_accepted. Contract still satisfied (empty accepted
        + structured failure)."""
        monkeypatch.setattr(
            ps,
            "run_prism_pruner",
            _stub_run_prism_factory(lambda n: [True] * n),
        )
        # Min is 0.0. With ewin=1.0, only conf 1 survives... unless we
        # spread them all >1 from min. Use [0.0, 5.0, 10.0] with ewin=2.0.
        # That keeps conf 1 only, not "everything dropped". To test
        # "everything dropped" we need a programmatic path: stub the
        # mask to enforce empty within set despite having energies.
        # Easier path: monkeypatch apply_ewin_filter directly.
        monkeypatch.setattr(
            ps, "apply_ewin_filter", lambda energies, ewin: [False] * len(energies)
        )
        up = _upstream_with_explicit_energies(tmp_path, [0.0, 1.0, 2.0])
        m = _run_node(tmp_path, up)
        assert m["ok"] is False
        assert any(
            "no_conformers_accepted" in f["error"] for f in m["failures"]
        )
        # Every conformer recorded as rejected with the ewin reason.
        rejected = m["artifacts"]["rejected"]
        assert [r["index"] for r in rejected] == [1, 2, 3]
        for r in rejected:
            assert r["rejected_reason"] == REJECTED_REASON_EWIN
        # Contract: empty accepted needs ok=false + structured failure (both present).
        problems = validate_conformer_screen(m, require_rejected_reason=True)
        assert problems == [], f"unexpected contract problems: {problems}"


# --------------------------------------------------------------------
# Node wiring
# --------------------------------------------------------------------


class TestNodeWiring:
    def test_class_attributes(self):
        assert PrismScreen.step == "prism_screen"
        assert PrismScreen.accepts_upstream is True
        assert PrismScreen.requires_upstream is True

    def test_method_constants_exposed(self):
        assert METHOD_NAME == "prism_pruner"
        assert REJECTED_REASON == "pruned_by_prism_pruner"
        assert REJECTED_REASON_EWIN == "above_energy_window"

    def test_main_factory_returns_callable(self):
        # main is the entrypoint installed in pyproject's [project.scripts].
        from scripps_workflow.nodes.prism_screen import main as prism_main

        assert callable(prism_main)
