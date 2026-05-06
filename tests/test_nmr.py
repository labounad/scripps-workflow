"""Tests for the NMR-prediction surface area.

Three concerns live here, each in its own class:

1. ``nmr_calibration`` — the linear-scaling helpers and the case-insensitive
   lookup. Pure math + a tiny dict, so the tests are formula-correctness
   parameterized cases.

2. ``orca`` NMR parsers and the functional-alias resolver. The shielding
   summary parser, the J-coupling parser (which has to handle three header
   formats — ORCA 6 same-line, modern multi-line, legacy comma — and the
   ``J[i,j](TERM) iso=`` value form), and the WP04 / wB97X-D / mPW1PW91
   alias chain. We use minimal synthetic outputs that exercise the format
   shapes without re-creating a 30 KB ORCA log.

3. ``nmr_aggregate`` end-to-end. The relaxed H-H coupling failure contract
   (sparse pair tables are normal, zero-with-≥2-H is still a failure)
   gets a dedicated regression test pinning the new behavior. Plus one
   happy-path check that the CSVs and summary land with the expected
   shape from a synthetic ``thermo_aggregate`` upstream.

Tests are deliberately lean — one assertion per behavior, no breadth-
first coverage of incidental fields. Lucas wants "precise" tests.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripps_workflow.nmr_calibration import (
    NMR_CALIBRATION,
    lookup_calibration,
    predict_chemical_shift,
    predict_coupling_constant,
)
from scripps_workflow.nodes.nmr_aggregate import NmrAggregate
from scripps_workflow.orca import (
    parse_orca_couplings,
    parse_orca_shieldings,
    resolve_functional_alias,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# nmr_calibration: predict_chemical_shift / predict_coupling_constant
# --------------------------------------------------------------------


class TestPredictChemicalShift:
    """cheshire convention: σ_calc = slope·δ_exp + intercept ⇒
    δ_predicted = (σ_calc − intercept) / slope. Negative slope is the
    common case (regressing σ on δ — they trend opposite)."""

    @pytest.mark.parametrize(
        "sigma, slope, intercept, expected",
        [
            # WP04 / 1H: δ=2.0 ↔ σ = -1.0698*2 + 31.8447 = 29.7051
            (29.7051, -1.0698, 31.8447, 2.0),
            # ωB97X-D / 13C: δ=120 ↔ σ = -1.0501*120 + 187.25 = 61.238
            (61.238, -1.0501, 187.25, 120.0),
            # Identity: slope=1, intercept=0
            (5.0, 1.0, 0.0, 5.0),
            # Negative δ falls out of the formula like any other point.
            (32.9145, -1.0698, 31.8447, -1.0),
        ],
    )
    def test_formula(self, sigma, slope, intercept, expected):
        got = predict_chemical_shift(sigma, slope=slope, intercept=intercept)
        assert got == pytest.approx(expected, rel=1e-6)

    def test_slope_zero_raises(self):
        with pytest.raises(ValueError, match="slope cannot be zero"):
            predict_chemical_shift(10.0, slope=0.0, intercept=5.0)


class TestPredictCouplingConstant:
    """Bally/Rablen convention: J_predicted = slope·J_calc + intercept.
    Direct prediction shape — no inversion."""

    @pytest.mark.parametrize(
        "j_calc, slope, intercept, expected",
        [
            # mPW1PW91 / pcJ-2 defaults
            (10.0, 0.9105, 0.21, 10.0 * 0.9105 + 0.21),
            (-3.5, 0.9105, 0.21, -3.5 * 0.9105 + 0.21),
            (0.0, 0.9105, 0.21, 0.21),
        ],
    )
    def test_formula(self, j_calc, slope, intercept, expected):
        assert predict_coupling_constant(
            j_calc, slope=slope, intercept=intercept
        ) == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------
# nmr_calibration: lookup_calibration
# --------------------------------------------------------------------


class TestLookupCalibration:
    def test_exact_match(self):
        cal = lookup_calibration(
            functional="WP04", basis="6-311++G(2d,p)",
            solvent="CHCl3", nucleus="1H",
        )
        assert cal is not None
        assert cal["slope"] == pytest.approx(-1.0698)
        assert cal["intercept"] == pytest.approx(31.8447)

    @pytest.mark.parametrize(
        "functional",
        ["wB97X-D", "wb97x-d", "WB97X-D", "  wB97X-D  "],
    )
    def test_case_and_whitespace_insensitive(self, functional):
        cal = lookup_calibration(
            functional=functional, basis="6-31G(d,p)",
            solvent="CHCl3", nucleus="13C",
        )
        assert cal is not None
        assert cal["slope"] == pytest.approx(-1.0501)

    def test_miss_returns_none(self):
        assert lookup_calibration(
            functional="not-a-functional", basis="x",
            solvent="x", nucleus="13C",
        ) is None

    def test_custom_table(self):
        # Caller can pass a custom table — the default global is left alone.
        custom = {
            ("X", "Y", "Z", "1H"): {"slope": 1.0, "intercept": 0.0, "source": "x"}
        }
        cal = lookup_calibration(
            functional="x", basis="y", solvent="z", nucleus="1H",
            table=custom,
        )
        assert cal is not None and cal["source"] == "x"
        # Default table untouched.
        assert "X" not in {k[0] for k in NMR_CALIBRATION}


# --------------------------------------------------------------------
# orca: resolve_functional_alias
# --------------------------------------------------------------------


class TestResolveFunctionalAlias:
    """Three calibration-table labels need translation for ORCA 6:
    WP04 has no built-in keyword (assembled from B3LYP/G + %method),
    wB97X-D is a renamed ORCA-6 keyword, mPW1PW91 lost its '91' suffix.
    """

    def test_wp04_emits_method_block(self):
        kw, blocks = resolve_functional_alias("WP04")
        assert kw == "B3LYP/G"
        assert len(blocks) == 1
        assert "%method" in blocks[0].lower()

    @pytest.mark.parametrize(
        "alias, expected_kw",
        [
            ("wB97X-D", "wB97X-D3"),
            ("wb97x-d", "wB97X-D3"),  # case-insensitive
            ("mPW1PW91", "mPW1PW"),
            ("MPW1PW91", "mPW1PW"),
        ],
    )
    def test_renamed_keywords(self, alias, expected_kw):
        kw, blocks = resolve_functional_alias(alias)
        assert kw == expected_kw
        assert blocks == []

    @pytest.mark.parametrize(
        "passthrough", ["B3LYP", "wB97X-V", "M06-2X", "r2scan-3c"]
    )
    def test_native_keywords_pass_through(self, passthrough):
        kw, blocks = resolve_functional_alias(passthrough)
        assert kw == passthrough
        assert blocks == []


# --------------------------------------------------------------------
# orca: parse_orca_shieldings
# --------------------------------------------------------------------


_SHIELDING_BLOCK = """\
                       CHEMICAL SHIELDING SUMMARY (ppm)
   ---------------------------------------------------------
   Nucleus  Element     Isotropic     Anisotropy
   --------  -------     ----------    ----------
       0      C         175.234         12.345
       1      H          29.700          5.000
       2      H          30.100          5.100
"""


class TestParseShieldings:
    def test_basic_table(self, tmp_path):
        out = tmp_path / "x.out"
        out.write_text(_SHIELDING_BLOCK, encoding="utf-8")
        rows = parse_orca_shieldings(out)
        assert [r["atom_index"] for r in rows] == [0, 1, 2]
        assert rows[0]["element"] == "C"
        assert rows[0]["sigma_iso_ppm"] == pytest.approx(175.234)
        assert rows[1]["sigma_iso_ppm"] == pytest.approx(29.700)

    def test_multiple_summaries_last_wins(self, tmp_path):
        # Compound output with TWO shielding summaries; the second one
        # should overwrite the first (per the parser's documented
        # "last-occurrence-wins" convention).
        out = tmp_path / "x.out"
        first = _SHIELDING_BLOCK
        second = first.replace("29.700", "29.999")
        out.write_text(first + "\n...\n" + second, encoding="utf-8")
        rows = parse_orca_shieldings(out)
        h0 = next(r for r in rows if r["atom_index"] == 1)
        assert h0["sigma_iso_ppm"] == pytest.approx(29.999)

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_orca_shieldings(tmp_path / "nope.out") == []

    def test_no_summary_marker(self, tmp_path):
        out = tmp_path / "x.out"
        out.write_text("just some unrelated ORCA noise\n", encoding="utf-8")
        assert parse_orca_shieldings(out) == []


# --------------------------------------------------------------------
# orca: parse_orca_couplings (three header formats)
# --------------------------------------------------------------------


# ORCA 6: single-line header with element-then-index ordering, then
# the J terms appear ~30 lines below as J[i,j](TERM) iso=value.
_ORCA6_PAIR_BLOCK = """\
 NUCLEUS A = H    8 NUCLEUS B = H    9
... 30 lines of intermediate orbital integrals ...
... padding line 1
... padding line 2
... padding line 3
... padding line 4
... padding line 5
... padding line 6
... padding line 7
... padding line 8
... padding line 9
... padding line 10
 J[8,9](Total)                           iso=    -14.978
 J[8,9](FC)                              iso=    -15.001
 J[8,9](SD)                              iso=      0.012
 J[8,9](PSO)                             iso=      0.034
 J[8,9](DSO)                             iso=     -0.022
"""

# Modern multi-line: separate "Nucleus A:" / "Nucleus B:" lines + spelled-out term names.
_MULTILINE_PAIR_BLOCK = """\
 Nucleus A: 0 C
 Nucleus B: 1 H
 Fermi contact contribution     :    143.456 Hz
 Spin-dipolar contribution      :      0.123 Hz
 Paramagnetic contribution      :      1.234 Hz
 Diamagnetic contribution       :      0.310 Hz
 Total                          :    145.123 Hz
"""

# Legacy comma-separated single-line header.
_LEGACY_PAIR_BLOCK = """\
 NUCLEUS A = 2 H, NUCLEUS B = 3 H
 Total                          :     12.345 Hz
 FC                             :     12.300 Hz
 SD                             :      0.020 Hz
 PSO                            :      0.030 Hz
 DSO                            :     -0.005 Hz
"""


class TestParseCouplings:
    def test_orca6_same_line_header_with_iso_form(self, tmp_path):
        out = tmp_path / "j.out"
        out.write_text(_ORCA6_PAIR_BLOCK, encoding="utf-8")
        pairs = parse_orca_couplings(out)
        assert len(pairs) == 1
        p = pairs[0]
        # Pair canonicalized to (min, max).
        assert (p["i"], p["j"]) == (8, 9)
        assert p["elem_i"] == "H" and p["elem_j"] == "H"
        # Total comes from the printed J[8,9](Total) iso= line directly.
        assert p["J_total_hz"] == pytest.approx(-14.978)
        assert p["J_FC_hz"] == pytest.approx(-15.001)
        assert p["J_DSO_hz"] == pytest.approx(-0.022)

    def test_multiline_header_with_spelled_out_terms(self, tmp_path):
        out = tmp_path / "j.out"
        out.write_text(_MULTILINE_PAIR_BLOCK, encoding="utf-8")
        pairs = parse_orca_couplings(out)
        assert len(pairs) == 1
        p = pairs[0]
        assert (p["i"], p["j"]) == (0, 1)
        assert p["J_total_hz"] == pytest.approx(145.123)
        assert p["J_FC_hz"] == pytest.approx(143.456)

    def test_legacy_comma_header(self, tmp_path):
        out = tmp_path / "j.out"
        out.write_text(_LEGACY_PAIR_BLOCK, encoding="utf-8")
        pairs = parse_orca_couplings(out)
        assert len(pairs) == 1
        p = pairs[0]
        assert (p["i"], p["j"]) == (2, 3)
        assert p["J_total_hz"] == pytest.approx(12.345)

    def test_total_synthesized_when_missing(self, tmp_path):
        # Drop the printed Total line; parser should synthesize from the
        # four Ramsey terms (FC + SD + PSO + DSO).
        block = "\n".join(
            ln
            for ln in _ORCA6_PAIR_BLOCK.splitlines()
            if "(Total)" not in ln
        )
        out = tmp_path / "j.out"
        out.write_text(block, encoding="utf-8")
        pairs = parse_orca_couplings(out)
        synthesized = (-15.001) + 0.012 + 0.034 + (-0.022)
        assert pairs[0]["J_total_hz"] == pytest.approx(synthesized, rel=1e-9)

    def test_multiple_pairs_in_one_file(self, tmp_path):
        out = tmp_path / "j.out"
        out.write_text(
            _ORCA6_PAIR_BLOCK + "\n" + _MULTILINE_PAIR_BLOCK,
            encoding="utf-8",
        )
        pairs = parse_orca_couplings(out)
        keys = {(p["i"], p["j"]) for p in pairs}
        assert keys == {(0, 1), (8, 9)}


# --------------------------------------------------------------------
# nmr_aggregate end-to-end
# --------------------------------------------------------------------


def _shielding_block(rows: list[tuple[int, str, float, float]]) -> str:
    """Build a CHEMICAL SHIELDING SUMMARY block from explicit rows."""
    lines = [
        "                       CHEMICAL SHIELDING SUMMARY (ppm)",
        "   --------------------------------------------------------",
        "   Nucleus  Element     Isotropic     Anisotropy",
        "   --------  -------     ----------    ----------",
    ]
    for idx, el, iso, aniso in rows:
        lines.append(f"     {idx:3d}      {el:<2s}      {iso:10.3f}      {aniso:8.3f}")
    return "\n".join(lines) + "\n"


def _coupling_block_orca6(pairs: list[tuple[int, int, str, str, float]]) -> str:
    """Build ORCA-6-style J-coupling blocks for the listed (i, j, ei, ej, J)."""
    out_lines: list[str] = []
    for (i, j, ei, ej, j_total) in pairs:
        out_lines.append(f" NUCLEUS A = {ei}    {i} NUCLEUS B = {ej}    {j}")
        out_lines.append(f" J[{i},{j}](Total)            iso=    {j_total:.3f}")
        out_lines.append(f" J[{i},{j}](FC)               iso=    {j_total:.3f}")
        out_lines.append(f" J[{i},{j}](SD)               iso=      0.000")
        out_lines.append(f" J[{i},{j}](PSO)              iso=      0.000")
        out_lines.append(f" J[{i},{j}](DSO)              iso=      0.000")
        out_lines.append("")
    return "\n".join(out_lines) + "\n"


def _build_upstream(
    tmp_path: Path,
    *,
    n_conformers: int,
    weights: list[float],
    h_shieldings: list[tuple[int, str, float, float]],
    c_shieldings: list[tuple[int, str, float, float]],
    couplings: list[tuple[int, int, str, str, float]] | None,
) -> Path:
    """Build a synthetic thermo_aggregate upstream manifest with task dirs.

    Each conformer's task_dir gets the dedicated NMR output files
    (orca_nmr_h.out / _c.out / _j.out) populated with the supplied
    tables — the same shielding/coupling values per conformer so the
    Boltzmann average is just the input value.
    """
    up_dir = tmp_path / "upstream"
    out_dir = up_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_root = up_dir / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)

    h_text = _shielding_block(h_shieldings)
    c_text = _shielding_block(c_shieldings)
    j_text = (
        _coupling_block_orca6(couplings)
        if couplings is not None
        else ""
    )

    confs: list[dict[str, Any]] = []
    for i in range(1, n_conformers + 1):
        d = tasks_root / f"task_{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "orca_nmr_h.out").write_text(h_text, encoding="utf-8")
        (d / "orca_nmr_c.out").write_text(c_text, encoding="utf-8")
        if couplings is not None:
            (d / "orca_nmr_j.out").write_text(j_text, encoding="utf-8")
        confs.append({
            "index": i,
            "label": f"conf_{i:04d}",
            "path_abs": str(d.resolve()),  # ContentRecord requires str
            "task_dir_abs": str(d.resolve()),
            "boltzmann_weight": float(weights[i - 1]),
        })

    m = Manifest.skeleton(step="thermo_aggregate", cwd=str(up_dir))
    m.artifacts["conformers"] = confs
    m_path = out_dir / "manifest.json"
    m.write(m_path)
    return m_path


def _run_aggregate(
    tmp_path: Path, upstream_manifest: Path, *config_tokens: str,
) -> dict:
    pointer_text = Pointer.of(
        ok=True, manifest_path=upstream_manifest
    ).to_json_line()
    call_dir = tmp_path / "calls" / "nmr_aggregate"
    call_dir.mkdir(parents=True)
    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = NmrAggregate().invoke(
            ["nmr_aggregate", pointer_text, *config_tokens]
        )
    finally:
        os.chdir(cwd)
    assert rc == 0, "soft-fail invariant violated"
    m_path = call_dir / "outputs" / "manifest.json"
    assert m_path.exists()
    return json.loads(m_path.read_text(encoding="utf-8"))


class TestNmrAggregateHappyPath:
    def test_writes_csvs_and_summary(self, tmp_path):
        # 2 conformers, equal weights, same shieldings + couplings →
        # the Boltzmann average is the input value itself.
        up = _build_upstream(
            tmp_path,
            n_conformers=2,
            weights=[0.5, 0.5],
            h_shieldings=[(1, "H", 29.7051, 5.0), (2, "H", 30.1, 5.0)],
            c_shieldings=[(0, "C", 61.238, 12.0)],
            couplings=[(1, 2, "H", "H", 7.5)],
        )
        m = _run_aggregate(tmp_path, up)

        # No structural failures (the ones we removed).
        failure_codes = {f["error"] for f in m.get("failures", [])}
        assert "incomplete_hh_coupling_table" not in failure_codes
        assert "no_hh_couplings_parsed" not in failure_codes

        # Shifts CSV: H rows scaled with WP04, C row scaled with wB97X-D.
        shifts_path = next(
            Path(a["path_abs"])
            for a in m["artifacts"]["files"]
            if a["label"] == "predicted_shifts_csv"
        )
        with shifts_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3  # 2 H + 1 C
        h1 = next(r for r in rows if r["atom_index"] == "1")
        # δ = (29.7051 − 31.8447) / -1.0698 ≈ 2.0
        assert float(h1["delta_predicted_ppm"]) == pytest.approx(2.0, abs=1e-3)
        c0 = next(r for r in rows if r["atom_index"] == "0")
        # δ = (61.238 − 187.25) / -1.0501 ≈ 120.0
        assert float(c0["delta_predicted_ppm"]) == pytest.approx(120.0, abs=1e-2)

        # Couplings CSV: one H-H pair, scaled with mPW1PW91 / pcJ-2.
        cps_path = next(
            Path(a["path_abs"])
            for a in m["artifacts"]["files"]
            if a["label"] == "predicted_couplings_csv"
        )
        with cps_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert (rows[0]["elem_i"], rows[0]["elem_j"]) == ("H", "H")
        # J_pred = 7.5 * 0.9105 + 0.21 ≈ 7.039
        assert float(rows[0]["J_predicted_hz"]) == pytest.approx(
            7.5 * 0.9105 + 0.21, rel=1e-6
        )

        # Summary: row counts on artifacts, both calibrations used.
        artifacts_by_label = {
            a["label"]: a for a in m["artifacts"]["files"]
        }
        assert artifacts_by_label["predicted_shifts_csv"]["h_row_count"] == 2
        assert artifacts_by_label["predicted_shifts_csv"]["c_row_count"] == 1
        assert artifacts_by_label["predicted_couplings_csv"]["hh_row_count"] == 1


class TestNmrAggregateHHContract:
    """Pins the relaxed H-H failure contract:

    * Sparse H-H table (some pairs present, fewer than n_h*(n_h-1)/2)
      is NORMAL — no failure record. ORCA's SpinSpinRThresh deliberately
      excludes long-range pairs.
    * Empty H-H table with ≥2 H atoms IS a failure — at least one
      short-range pair always exists in any real molecule.
    """

    def test_sparse_table_is_not_a_failure(self, tmp_path):
        # 4 H atoms ⇒ complete graph would be 6 pairs. We only emit 2,
        # mimicking what ORCA does when SpinSpinRThresh excludes the
        # other 4 long-range pairs.
        h_rows = [
            (1, "H", 29.7, 5.0),
            (2, "H", 29.8, 5.0),
            (3, "H", 29.9, 5.0),
            (4, "H", 30.0, 5.0),
        ]
        up = _build_upstream(
            tmp_path,
            n_conformers=1,
            weights=[1.0],
            h_shieldings=h_rows,
            c_shieldings=[(0, "C", 61.238, 12.0)],
            couplings=[
                (1, 2, "H", "H", 7.5),
                (3, 4, "H", "H", 7.7),
            ],
        )
        m = _run_aggregate(tmp_path, up)

        failure_codes = [f["error"] for f in m.get("failures", [])]
        # Pre-fix this would have flagged "incomplete_hh_coupling_table"
        # for 2 < 6 pairs. Post-fix it's a normal sparse run.
        assert "incomplete_hh_coupling_table" not in failure_codes
        assert "no_hh_couplings_parsed" not in failure_codes
        assert m["ok"] is True

        # Summary still reports the sparse count for diagnostics.
        s_path = next(
            Path(a["path_abs"])
            for a in m["artifacts"]["files"]
            if a["label"] == "nmr_summary_json"
        )
        summary = json.loads(s_path.read_text(encoding="utf-8"))
        assert summary["n_hh_pairs"] == 2
        assert summary["n_h_atoms"] == 4

    def test_empty_table_with_h_atoms_is_a_failure(self, tmp_path):
        # 2 H atoms, parser/ORCA returns ZERO H-H pairs — that's the
        # genuine "something broke" signal we keep.
        up = _build_upstream(
            tmp_path,
            n_conformers=1,
            weights=[1.0],
            h_shieldings=[(1, "H", 29.7, 5.0), (2, "H", 29.8, 5.0)],
            c_shieldings=[(0, "C", 61.238, 12.0)],
            couplings=[],  # no pairs in orca_nmr_j.out
        )
        m = _run_aggregate(tmp_path, up)

        failure_codes = [f["error"] for f in m.get("failures", [])]
        assert "no_hh_couplings_parsed" in failure_codes
        assert m["ok"] is False
