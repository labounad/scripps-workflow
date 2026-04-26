"""Tests for the thermo_aggregate node.

The node is a downstream pure-aggregator — no SLURM, no ORCA, no
external binaries. It walks a fake ``orca_thermo_array`` upstream's
``task_XXXX/orca_thermo.out`` files, parses the thermochem block,
computes composite Gibbs energies + Boltzmann weights, and writes a CSV.

Pattern: build a synthetic upstream manifest pointing at a tasks_root
dir of our own creation, pre-populate the .out files with controlled
values, and check the resulting CSV + manifest.

Coverage:
    * Pure helpers: parse_task_dir, composite_gibbs_for_record,
      aggregate_thermo_records, compute_relative_gibbs, write_thermo_csv,
      locate_tasks_root.
    * Happy path end-to-end: 3 conformers, default config, CSV columns
      typed, manifest conformer bucket populated.
    * Standard-state correction: 1M shifts deltaG_kcal but not weights.
    * Custom temperature flows through.
    * Failure modes: missing task dir, no .out, missing thermo (G_low
      fallback), totally missing G (failure record), bad config.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Optional

import pytest

from scripps_workflow.nodes import thermo_aggregate as ta
from scripps_workflow.nodes.thermo_aggregate import (
    CSV_COLUMNS,
    ThermoAggregate,
    aggregate_thermo_records,
    composite_gibbs_for_record,
    compute_relative_gibbs,
    locate_tasks_root,
    parse_task_dir,
    write_thermo_csv,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# Synthetic ORCA output fixtures
# --------------------------------------------------------------------


def _compound_out(
    *,
    e_high: Optional[float] = -76.5,
    g_minus_e_el: Optional[float] = -0.04,
    g_low: Optional[float] = -76.123,
    h_low: Optional[float] = -76.10,
    s_corr_low: Optional[float] = -0.034,
    t_low: Optional[float] = 298.15,
    terminated_normally: bool = True,
) -> str:
    """Build a synthetic compound (freq + SP) ORCA .out body."""
    lines: list[str] = ["...lots of header noise...", ""]
    if t_low is not None:
        lines.append(f"THERMOCHEMISTRY AT T = {t_low} K")
    if h_low is not None:
        lines.append(f"Total enthalpy                ...    {h_low:.9f} Eh")
    if s_corr_low is not None:
        lines.append(
            f"Total entropy correction      ...    {s_corr_low:.9f} Eh"
        )
    if g_low is not None:
        lines.append(
            f"Final Gibbs free energy       ...    {g_low:.9f} Eh"
        )
    if g_minus_e_el is not None:
        lines.append(
            f"G-E(el)                       ...    {g_minus_e_el:.9f} Eh"
        )
    lines.append("")
    lines.append("------------------------- $new_job ----------------------")
    lines.append("")
    if e_high is not None:
        lines.append(f"FINAL SINGLE POINT ENERGY    {e_high:.9f}")
    if terminated_normally:
        lines.append("                     ****ORCA TERMINATED NORMALLY****")
    lines.append("")
    return "\n".join(lines)


def _populate_tasks(
    tasks_root: Path,
    *,
    n: int,
    out_name: str = "orca_thermo.out",
    e_highs: list[Optional[float]] | None = None,
    g_corrs: list[Optional[float]] | None = None,
    g_lows: list[Optional[float]] | None = None,
) -> None:
    """Create ``task_0001..task_{n:04d}`` under ``tasks_root`` with synthetic outs.

    Defaults make every task a happy-path: composite G_i = -76.50 - i*0.001
    so the sequence is monotonically decreasing (task 1 highest G, task n
    lowest). Caller can override per-task values.
    """
    tasks_root.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        d = tasks_root / f"task_{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        e_high = (
            e_highs[i - 1]
            if e_highs is not None
            else -76.500 - 0.001 * i
        )
        g_corr = g_corrs[i - 1] if g_corrs is not None else -0.04
        g_low = g_lows[i - 1] if g_lows is not None else (
            (e_high or 0.0) + (g_corr or 0.0) + 0.001
        )
        body = _compound_out(
            e_high=e_high, g_minus_e_el=g_corr, g_low=g_low
        )
        (d / out_name).write_text(body)


def _build_upstream_manifest(
    tmp_path: Path, *, tasks_root: Path, n_tasks: int
) -> Path:
    """Write a fake orca_thermo_array manifest pointing at tasks_root."""
    up_dir = tmp_path / "upstream"
    out_dir = up_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    m = Manifest.skeleton(step="orca_thermo_array", cwd=str(up_dir))
    m.set_array_info(
        tasks_root_abs=str(tasks_root.resolve()), n_tasks=n_tasks
    )
    m_path = out_dir / "manifest.json"
    m.write(m_path)
    return m_path


def _pointer_text(manifest_path: Path, ok: bool = True) -> str:
    return Pointer.of(ok=ok, manifest_path=manifest_path).to_json_line()


def _run_node(
    tmp_path: Path,
    *config_tokens: str,
    n_conformers: int = 3,
    populate_kwargs: dict[str, Any] | None = None,
    tasks_root: Optional[Path] = None,
) -> dict:
    """End-to-end harness: build a fake upstream + invoke the node."""
    if tasks_root is None:
        tasks_root = tmp_path / "fake_tasks"
    if populate_kwargs is None:
        populate_kwargs = {}
    _populate_tasks(tasks_root, n=n_conformers, **populate_kwargs)

    up_manifest_path = _build_upstream_manifest(
        tmp_path, tasks_root=tasks_root, n_tasks=n_conformers
    )
    pointer_text = _pointer_text(up_manifest_path)

    call_dir = tmp_path / "calls" / "thermo_aggregate"
    call_dir.mkdir(parents=True)
    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = ThermoAggregate().invoke(
            ["thermo_aggregate", pointer_text, *config_tokens]
        )
    finally:
        os.chdir(cwd)
    assert rc == 0, "soft-fail invariant violated"
    m_path = call_dir / "outputs" / "manifest.json"
    assert m_path.exists()
    return json.loads(m_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------
# parse_task_dir
# --------------------------------------------------------------------


class TestParseTaskDir:
    def test_full_compound_output(self, tmp_path):
        d = tmp_path / "task_0001"
        d.mkdir()
        (d / "orca_thermo.out").write_text(
            _compound_out(
                e_high=-76.987654321,
                g_minus_e_el=-0.041234567,
                g_low=-76.158024679,
                h_low=-76.123456789,
                s_corr_low=-0.034567890,
                t_low=298.15,
            )
        )

        rec = parse_task_dir(d)
        assert rec["E_high_eh"] == pytest.approx(-76.987654321)
        assert rec["G_minus_Eel_low_eh"] == pytest.approx(-0.041234567)
        assert rec["G_low_eh"] == pytest.approx(-76.158024679)
        assert rec["H_low_eh"] == pytest.approx(-76.123456789)
        assert rec["TS_corr_low_eh"] == pytest.approx(-0.034567890)
        assert rec["T_low_k"] == pytest.approx(298.15)
        assert rec["thermo_out"].endswith("orca_thermo.out")
        assert rec["sp_out"].endswith("orca_thermo.out")

    def test_missing_dir_returns_none_slots(self, tmp_path):
        rec = parse_task_dir(tmp_path / "nope")
        # Function never raises — None across the board.
        for k in (
            "thermo_out",
            "sp_out",
            "E_high_eh",
            "G_minus_Eel_low_eh",
            "G_low_eh",
            "H_low_eh",
            "TS_corr_low_eh",
            "T_low_k",
        ):
            assert rec[k] is None

    def test_dir_with_no_outs(self, tmp_path):
        d = tmp_path / "task_0001"
        d.mkdir()
        rec = parse_task_dir(d)
        assert rec["thermo_out"] is None
        assert rec["sp_out"] is None
        assert rec["E_high_eh"] is None


# --------------------------------------------------------------------
# composite_gibbs_for_record
# --------------------------------------------------------------------


class TestCompositeGibbsForRecord:
    def test_canonical_path(self):
        rec = {
            "E_high_eh": -76.5,
            "G_minus_Eel_low_eh": -0.04,
            "G_low_eh": -76.123,
        }
        # Composite = -76.5 + (-0.04) = -76.54
        assert composite_gibbs_for_record(rec) == pytest.approx(-76.54)

    def test_falls_back_to_g_low(self):
        rec = {
            "E_high_eh": None,
            "G_minus_Eel_low_eh": None,
            "G_low_eh": -76.158,
        }
        assert composite_gibbs_for_record(rec) == pytest.approx(-76.158)

    def test_returns_none_when_nothing(self):
        rec = {
            "E_high_eh": None,
            "G_minus_Eel_low_eh": None,
            "G_low_eh": None,
        }
        assert composite_gibbs_for_record(rec) is None

    def test_e_high_alone_uses_g_low_fallback(self):
        # E_high alone is NOT enough to compute composite G — needs the
        # thermal correction. Falls through to G_low if available.
        rec = {
            "E_high_eh": -76.5,
            "G_minus_Eel_low_eh": None,
            "G_low_eh": -76.158,
        }
        assert composite_gibbs_for_record(rec) == pytest.approx(-76.158)

    def test_corr_alone_uses_g_low_fallback(self):
        # Mirror case: thermal correction without high-level SP →
        # G_low fallback, NOT a half-cooked composite.
        rec = {
            "E_high_eh": None,
            "G_minus_Eel_low_eh": -0.04,
            "G_low_eh": -76.158,
        }
        assert composite_gibbs_for_record(rec) == pytest.approx(-76.158)


# --------------------------------------------------------------------
# aggregate_thermo_records
# --------------------------------------------------------------------


class TestAggregateThermoRecords:
    def test_three_happy_tasks(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        _populate_tasks(tasks_root, n=3)

        records, failures = aggregate_thermo_records(
            tasks_root=tasks_root, n_tasks=3
        )
        assert len(records) == 3
        assert [r["index"] for r in records] == [1, 2, 3]
        for r in records:
            assert r["G_composite_eh"] is not None
            assert r["E_high_eh"] is not None
        assert failures == []

    def test_missing_task_dir(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        _populate_tasks(tasks_root, n=2)
        # n_tasks=3 but only 2 task dirs exist.
        records, failures = aggregate_thermo_records(
            tasks_root=tasks_root, n_tasks=3
        )
        assert len(records) == 3
        # Task 3 has no G_composite_eh and a missing_task_dir failure.
        assert records[2]["G_composite_eh"] is None
        errs = [(f["error"], f["index"]) for f in failures]
        assert ("missing_task_dir", 3) in errs

    def test_no_out_files(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        _populate_tasks(tasks_root, n=2)
        # Wipe the .out from task 2.
        (tasks_root / "task_0002" / "orca_thermo.out").unlink()

        records, failures = aggregate_thermo_records(
            tasks_root=tasks_root, n_tasks=2
        )
        # Task 1 still happy, task 2 surfaces no_orca_out_files_found.
        assert records[0]["G_composite_eh"] is not None
        assert records[1]["G_composite_eh"] is None
        errs = [(f["error"], f["index"]) for f in failures]
        assert ("no_orca_out_files_found", 2) in errs

    def test_freq_only_falls_back_to_g_low(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        # No high-level SP — just freq.
        _populate_tasks(
            tasks_root,
            n=1,
            e_highs=[None],
            g_lows=[-76.158024679],
        )
        records, failures = aggregate_thermo_records(
            tasks_root=tasks_root, n_tasks=1
        )
        # The .out STILL has the freq job's own SCF "FINAL SINGLE POINT
        # ENERGY" line — but the composite branch needs both E_high AND
        # G_minus_Eel_low. With g_corrs default (-0.04), the canonical
        # branch fires using the freq SCF energy as a low-quality
        # E_high. That's the legacy behavior: aggregator can't tell the
        # difference between "intentional freq-only" and "compound with
        # SP" by parsing alone. The user opts out via
        # singlepoint_keywords=none upstream and gets a different on-
        # disk shape (the freq SCF E IS the only SP signal).
        assert records[0]["G_composite_eh"] is not None
        # No failures — we got SOME composite G.
        assert failures == []

    def test_truly_missing_g(self, tmp_path):
        # Outfile parses but every thermo marker is absent (only a
        # placeholder line that doesn't match any pattern).
        tasks_root = tmp_path / "tasks"
        d = tasks_root / "task_0001"
        d.mkdir(parents=True)
        (d / "orca_thermo.out").write_text(
            "junk that doesn't match any of the regexes\n"
        )
        records, failures = aggregate_thermo_records(
            tasks_root=tasks_root, n_tasks=1
        )
        assert records[0]["G_composite_eh"] is None
        errs = [f["error"] for f in failures]
        assert "missing_thermochem_or_energy_for_G" in errs


# --------------------------------------------------------------------
# compute_relative_gibbs
# --------------------------------------------------------------------


class TestComputeRelativeGibbs:
    def test_simple_three(self):
        # G values 1 mEh apart → ΔG values 0.627509 kcal/mol apart.
        gs = [-76.500, -76.501, -76.502]
        dG, g_min, ss = compute_relative_gibbs(
            g_composite=gs, standard_state="1atm", temperature_k=298.15
        )
        assert g_min == pytest.approx(-76.502)
        assert ss == 0.0
        assert dG[0] == pytest.approx(0.002 * 627.509474)
        assert dG[1] == pytest.approx(0.001 * 627.509474)
        assert dG[2] == pytest.approx(0.0)

    def test_none_propagates(self):
        gs = [-76.500, None, -76.502]
        dG, g_min, _ = compute_relative_gibbs(
            g_composite=gs, standard_state="1atm", temperature_k=298.15
        )
        assert dG[0] is not None
        assert dG[1] is None
        assert dG[2] == pytest.approx(0.0)
        assert g_min == pytest.approx(-76.502)

    def test_all_none_returns_no_min(self):
        dG, g_min, _ = compute_relative_gibbs(
            g_composite=[None, None],
            standard_state="1atm",
            temperature_k=298.15,
        )
        assert dG == [None, None]
        assert g_min is None

    def test_1m_correction_applied(self):
        # 1M correction at 298.15 K is +1.894 kcal/mol; it shifts ΔG
        # uniformly but doesn't change the *relative* ordering.
        gs = [-76.500, -76.502]
        dG_1atm, _, ss_1atm = compute_relative_gibbs(
            g_composite=gs, standard_state="1atm", temperature_k=298.15
        )
        dG_1m, _, ss_1m = compute_relative_gibbs(
            g_composite=gs, standard_state="1M", temperature_k=298.15
        )
        assert ss_1atm == 0.0
        assert ss_1m == pytest.approx(1.8941975683244532)
        # Each dG is shifted by +ss_1m.
        for a, b in zip(dG_1atm, dG_1m):
            assert b - a == pytest.approx(ss_1m)
        # Differences between conformers are unchanged.
        assert (dG_1m[0] - dG_1m[1]) == pytest.approx(dG_1atm[0] - dG_1atm[1])

    def test_1m_aliases_normalized(self):
        # Accept "1M", "1m", "1mol", "1molar" as the same correction.
        gs = [-76.500, -76.501]
        ss_values = []
        for token in ("1m", "1M", "1mol", "1molar"):
            _, _, ss = compute_relative_gibbs(
                g_composite=gs,
                standard_state=token,
                temperature_k=298.15,
            )
            ss_values.append(ss)
        assert all(s == pytest.approx(ss_values[0]) for s in ss_values)


# --------------------------------------------------------------------
# locate_tasks_root
# --------------------------------------------------------------------


class TestLocateTasksRoot:
    def test_reads_array_block(self, tmp_path):
        tasks_root = tmp_path / "real_tasks"
        tasks_root.mkdir()
        upm = {
            "artifacts": {
                "array": {
                    "tasks_root_abs": str(tasks_root.resolve()),
                    "n_tasks": 7,
                }
            },
            "cwd": str(tmp_path),
        }
        located, n = locate_tasks_root(upstream_manifest=upm)
        assert located == tasks_root.resolve()
        assert n == 7

    def test_falls_back_to_cwd_outputs_array_tasks(self, tmp_path):
        # No tasks_root_abs in array block — fall back to the
        # conventional <upstream_cwd>/outputs/array/tasks layout.
        legacy = tmp_path / "up" / "outputs" / "array" / "tasks"
        legacy.mkdir(parents=True)
        for i in range(1, 4):
            (legacy / f"task_{i:04d}").mkdir()
        upm = {
            "artifacts": {"array": {}},
            "cwd": str(tmp_path / "up"),
        }
        located, n = locate_tasks_root(upstream_manifest=upm)
        assert located == legacy
        # n_tasks counted by globbing.
        assert n == 3

    def test_bogus_manifest(self, tmp_path):
        upm = {"artifacts": {}, "cwd": str(tmp_path / "doesnt_exist")}
        located, n = locate_tasks_root(upstream_manifest=upm)
        assert located is None
        assert n is None

    def test_array_block_path_doesnt_exist(self, tmp_path):
        # tasks_root_abs is set but points at a non-existent path AND
        # the cwd fallback also doesn't help.
        upm = {
            "artifacts": {
                "array": {
                    "tasks_root_abs": "/no/such/dir",
                    "n_tasks": 5,
                }
            },
            "cwd": str(tmp_path / "doesnt_exist"),
        }
        located, n = locate_tasks_root(upstream_manifest=upm)
        assert located is None


# --------------------------------------------------------------------
# write_thermo_csv
# --------------------------------------------------------------------


class TestWriteThermoCsv:
    def test_columns_and_row_count(self, tmp_path):
        records = [
            {
                "index": 1,
                "task_dir": "/abs/task_0001",
                "thermo_out": "/abs/task_0001/orca_thermo.out",
                "sp_out": "/abs/task_0001/orca_thermo.out",
                "E_high_eh": -76.987654321,
                "G_minus_Eel_low_eh": -0.041234567,
                "G_low_eh": -76.158024679,
                "G_composite_eh": -77.028888888,
                "H_low_eh": -76.123456789,
                "TS_corr_low_eh": -0.034567890,
                "T_low_k": 298.15,
            },
            {
                "index": 2,
                "task_dir": "/abs/task_0002",
                "thermo_out": None,
                "sp_out": None,
                "E_high_eh": None,
                "G_minus_Eel_low_eh": None,
                "G_low_eh": None,
                "G_composite_eh": None,
                "H_low_eh": None,
                "TS_corr_low_eh": None,
                "T_low_k": None,
            },
        ]
        out_path = tmp_path / "out" / "csv" / "conformer_thermo.csv"

        write_thermo_csv(
            out_path=out_path,
            records=records,
            delta_g_kcal=[0.0, None],
            weights=[1.0, None],
            cumulative=[1.0, None],
            standard_state="1atm",
            ss_corr_kcal=0.0,
            temperature_k=298.15,
        )

        assert out_path.exists()
        with out_path.open(encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        assert tuple(rows[0]) == CSV_COLUMNS
        # Header + 2 rows.
        assert len(rows) == 3
        # Percent column is 100 * weight.
        assert rows[1][CSV_COLUMNS.index("boltzmann_percent")] == "100.0"
        # None columns become empty strings in the CSV.
        assert (
            rows[2][CSV_COLUMNS.index("boltzmann_weight")] == ""
        )

    def test_length_mismatch_raises(self, tmp_path):
        out_path = tmp_path / "x.csv"
        with pytest.raises(ValueError, match="equal length"):
            write_thermo_csv(
                out_path=out_path,
                records=[{"index": 1}],
                delta_g_kcal=[0.0, 1.0],
                weights=[0.5, 0.5],
                cumulative=[0.5, 1.0],
                standard_state="1atm",
                ss_corr_kcal=0.0,
                temperature_k=298.15,
            )


# --------------------------------------------------------------------
# parse_config
# --------------------------------------------------------------------


class TestParseConfig:
    def test_defaults(self):
        cfg = ThermoAggregate().parse_config({})
        assert cfg["temperature_k"] == 298.15
        assert cfg["standard_state"] == "1atm"
        assert cfg["output_csv"] == "conformer_thermo.csv"

    def test_explicit_temperature(self):
        cfg = ThermoAggregate().parse_config({"temperature_k": "310.0"})
        assert cfg["temperature_k"] == pytest.approx(310.0)

    def test_normalize_1m_aliases(self):
        for token in ("1M", "1m", "1mol", "1molar"):
            cfg = ThermoAggregate().parse_config({"standard_state": token})
            assert cfg["standard_state"] in {"1m", "1mol", "1molar"}

    def test_negative_temperature_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ThermoAggregate().parse_config({"temperature_k": "-5"})

    def test_bogus_standard_state_rejected(self):
        with pytest.raises(ValueError, match="standard_state"):
            ThermoAggregate().parse_config({"standard_state": "bogus"})

    def test_path_traversal_csv_rejected(self):
        with pytest.raises(ValueError, match="basename"):
            ThermoAggregate().parse_config({"output_csv": "../out.csv"})


# --------------------------------------------------------------------
# End-to-end happy path
# --------------------------------------------------------------------


class TestHappyPath:
    def test_writes_csv_and_summary(self, tmp_path):
        m = _run_node(tmp_path)
        assert m["ok"] is True
        assert m["step"] == "thermo_aggregate"

        files = {f["label"]: f for f in m["artifacts"]["files"]}
        assert "conformer_thermo_csv" in files
        assert "thermo_summary_json" in files

        csv_path = Path(files["conformer_thermo_csv"]["path_abs"])
        assert csv_path.exists()
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        assert tuple(rows[0]) == CSV_COLUMNS
        assert len(rows) == 4  # header + 3 conformers

        # Summary JSON shape.
        summary_path = Path(files["thermo_summary_json"]["path_abs"])
        s = json.loads(summary_path.read_text())
        assert s["n_conformers"] == 3
        assert s["n_with_G"] == 3
        assert s["temperature_k"] == 298.15
        assert s["standard_state"] == "1atm"
        assert s["std_state_corr_kcal"] == 0.0
        assert s["Gmin_eh"] is not None

    def test_conformers_bucket_populated(self, tmp_path):
        m = _run_node(tmp_path)
        confs = m["artifacts"]["conformers"]
        assert len(confs) == 3
        assert [c["index"] for c in confs] == [1, 2, 3]
        # Default tasks have monotonically decreasing G_composite (lowest
        # at task 3), so deltaG is highest for task 1.
        dGs = [c["deltaG_kcal"] for c in confs]
        assert dGs[0] > dGs[1] > dGs[2]
        assert dGs[2] == pytest.approx(0.0)
        # Boltzmann weights sum to 1.
        ws = [c["boltzmann_weight"] for c in confs]
        assert sum(ws) == pytest.approx(1.0)
        # Cumulative weight for the lowest-ΔG conformer is its own weight.
        assert confs[2]["cumulative_weight"] == pytest.approx(ws[2])

    def test_inputs_block_recorded(self, tmp_path):
        m = _run_node(tmp_path, "temperature_k=310.0", "standard_state=1M")
        ins = m["inputs"]
        assert ins["temperature_k"] == pytest.approx(310.0)
        assert ins["standard_state"] == "1m"
        assert ins["n_tasks"] == 3

    def test_1m_shifts_csv_dg_by_constant(self, tmp_path):
        # Run twice — same fake upstream — once with 1atm, once with 1M.
        # The CSV dG values should differ by exactly RT*ln(24.46).
        from scripps_workflow.thermo import rt_ln_24_46_kcal

        m1 = _run_node(tmp_path / "a")
        m2 = _run_node(tmp_path / "b", "standard_state=1M")
        files_a = {f["label"]: f for f in m1["artifacts"]["files"]}
        files_b = {f["label"]: f for f in m2["artifacts"]["files"]}

        with Path(files_a["conformer_thermo_csv"]["path_abs"]).open() as fa:
            rows_a = list(csv.DictReader(fa))
        with Path(files_b["conformer_thermo_csv"]["path_abs"]).open() as fb:
            rows_b = list(csv.DictReader(fb))

        delta_idx = "deltaG_kcal"
        shift = rt_ln_24_46_kcal(298.15)
        for ra, rb in zip(rows_a, rows_b):
            dG_a = float(ra[delta_idx])
            dG_b = float(rb[delta_idx])
            assert (dG_b - dG_a) == pytest.approx(shift, rel=1e-9)

        # Boltzmann weights are unaffected by the constant shift.
        for ra, rb in zip(rows_a, rows_b):
            assert float(ra["boltzmann_weight"]) == pytest.approx(
                float(rb["boltzmann_weight"])
            )


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_missing_upstream_array_block(self, tmp_path):
        # Upstream manifest present but no array.tasks_root_abs AND no
        # conventional fallback dir.
        up_dir = tmp_path / "upstream"
        out_dir = up_dir / "outputs"
        out_dir.mkdir(parents=True)
        m = Manifest.skeleton(step="orca_thermo_array", cwd=str(up_dir))
        m_path = out_dir / "manifest.json"
        m.write(m_path)

        pointer_text = _pointer_text(m_path)
        call_dir = tmp_path / "calls" / "thermo"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = ThermoAggregate().invoke(["thermo_aggregate", pointer_text])
        finally:
            os.chdir(cwd)
        assert rc == 0
        result = json.loads(
            (call_dir / "outputs" / "manifest.json").read_text()
        )
        assert result["ok"] is False
        errs = [f["error"] for f in result["failures"]]
        assert "could_not_locate_tasks_root_abs_in_upstream_manifest" in errs

    def test_zero_n_tasks(self, tmp_path):
        tasks_root = tmp_path / "real_but_empty"
        tasks_root.mkdir()
        up_manifest_path = _build_upstream_manifest(
            tmp_path, tasks_root=tasks_root, n_tasks=0
        )
        pointer_text = _pointer_text(up_manifest_path)

        call_dir = tmp_path / "calls" / "thermo"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = ThermoAggregate().invoke(["thermo_aggregate", pointer_text])
        finally:
            os.chdir(cwd)
        assert rc == 0
        result = json.loads(
            (call_dir / "outputs" / "manifest.json").read_text()
        )
        assert result["ok"] is False
        errs = [f["error"] for f in result["failures"]]
        assert "n_tasks_invalid_or_zero" in errs

    def test_partial_failure_still_writes_csv(self, tmp_path):
        # 3 tasks, but task 2 has no .out — node should still write the
        # CSV, surface the failure, and be ok=False.
        tasks_root = tmp_path / "tasks"
        _populate_tasks(tasks_root, n=3)
        (tasks_root / "task_0002" / "orca_thermo.out").unlink()

        up_manifest_path = _build_upstream_manifest(
            tmp_path, tasks_root=tasks_root, n_tasks=3
        )
        pointer_text = _pointer_text(up_manifest_path)

        call_dir = tmp_path / "calls" / "thermo"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = ThermoAggregate().invoke(["thermo_aggregate", pointer_text])
        finally:
            os.chdir(cwd)
        assert rc == 0
        m = json.loads(
            (call_dir / "outputs" / "manifest.json").read_text()
        )
        assert m["ok"] is False
        errs = [f["error"] for f in m["failures"]]
        assert "no_orca_out_files_found" in errs
        # CSV STILL written — task 2 row has empty Gibbs columns.
        files = {f["label"]: f for f in m["artifacts"]["files"]}
        assert "conformer_thermo_csv" in files
        with Path(files["conformer_thermo_csv"]["path_abs"]).open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3
        # Task 2 row has empty Gibbs cells (CSV-empty-string for None).
        assert rows[1]["G_composite_eh"] == ""
        assert rows[1]["boltzmann_weight"] == ""

    def test_bad_config_triggers_argv_parse_failed(self, tmp_path):
        # Negative temperature → ValueError in parse_config.
        tasks_root = tmp_path / "tasks"
        _populate_tasks(tasks_root, n=1)
        up_manifest_path = _build_upstream_manifest(
            tmp_path, tasks_root=tasks_root, n_tasks=1
        )
        pointer_text = _pointer_text(up_manifest_path)

        call_dir = tmp_path / "calls" / "thermo"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = ThermoAggregate().invoke(
                ["thermo_aggregate", pointer_text, "temperature_k=-5"]
            )
        finally:
            os.chdir(cwd)
        # Soft-fail invariant: rc=0 even with a bad config.
        assert rc == 0
        m = json.loads(
            (call_dir / "outputs" / "manifest.json").read_text()
        )
        assert m["ok"] is False
        errs = [f["error"] for f in m["failures"]]
        assert any("argv_parse_failed" in e for e in errs)


# --------------------------------------------------------------------
# Node wiring
# --------------------------------------------------------------------


class TestNodeWiring:
    def test_step_name(self):
        assert ThermoAggregate.step == "thermo_aggregate"

    def test_requires_upstream(self):
        assert ThermoAggregate.requires_upstream is True
        assert ThermoAggregate.accepts_upstream is True

    def test_main_factory_attached(self):
        assert callable(ta.main)

    def test_constants(self):
        assert ta.DEFAULT_TEMPERATURE_K == 298.15
        assert ta.DEFAULT_STANDARD_STATE == "1atm"
        assert ta.DEFAULT_OUTPUT_CSV == "conformer_thermo.csv"

    def test_csv_columns_locked(self):
        # The CSV column order is part of the public artifact contract.
        # Downstream tooling (a plotting node, marc-screen filtering)
        # may rely on positional access, so a refactor that reorders
        # columns gets caught here.
        assert CSV_COLUMNS == (
            "index",
            "task_dir",
            "thermo_out",
            "sp_out",
            "E_high_eh",
            "G_minus_Eel_low_eh",
            "G_low_eh",
            "G_composite_eh",
            "deltaG_kcal",
            "boltzmann_weight",
            "boltzmann_percent",
            "cumulative_weight",
            "cumulative_percent",
            "standard_state",
            "std_state_corr_kcal",
            "temperature_k_used",
            "T_low_k_parsed",
            "H_low_eh",
            "TS_corr_low_eh",
        )
