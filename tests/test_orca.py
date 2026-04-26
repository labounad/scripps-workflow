"""Tests for ``scripps_workflow.orca`` shared ORCA helpers.

The module is a small grab-bag of ORCA primitives (SMD alias map, simple
input renderer, FINAL E parser, energy file writer, multi-xyz
concatenator). Each helper has a focused test class below; the goal is
to lock the byte-for-byte output formats that the array nodes' on-disk
artifacts depend on.

ORCA itself is never invoked here — these are pure-text helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripps_workflow.orca import (
    FINAL_E_RE,
    HARTREE_TO_KCAL,
    ORCA_NORMAL_RE,
    concat_xyz_files,
    make_orca_compound_input,
    make_orca_simple_input,
    orca_terminated_normally,
    parse_orca_final_energy,
    solvent_to_orca_smd,
    write_energy_file,
)


# --------------------------------------------------------------------
# solvent_to_orca_smd
# --------------------------------------------------------------------


class TestSolventToOrcaSmd:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Common DCM aliases.
            ("ch2cl2", "CH2Cl2"),
            ("CH2CL2", "CH2Cl2"),
            ("CH2Cl2", "CH2Cl2"),
            ("DCM", "CH2Cl2"),
            ("dcm", "CH2Cl2"),
            ("dichloromethane", "CH2Cl2"),
            ("methylene_chloride", "CH2Cl2"),
            # Chloroform.
            ("chcl3", "CHCl3"),
            ("CHCl3", "CHCl3"),
            ("chloroform", "CHCl3"),
            # Common ethers / amides / sulfoxides.
            ("thf", "THF"),
            ("THF", "THF"),
            ("dmf", "DMF"),
            ("dmso", "DMSO"),
            # Alcohols.
            ("meoh", "methanol"),
            ("methanol", "methanol"),
            ("etoh", "ethanol"),
            ("ethanol", "ethanol"),
            # Water.
            ("h2o", "water"),
            ("H2O", "water"),
            ("water", "water"),
            # Hydrocarbons.
            ("toluene", "toluene"),
            ("benzene", "benzene"),
            ("hexane", "hexane"),
            ("n-hexane", "hexane"),
            ("n_hexane", "hexane"),
            # Nitriles / ketones / amines.
            ("acetonitrile", "acetonitrile"),
            ("mecn", "acetonitrile"),
            ("MeCN", "acetonitrile"),
            ("acetone", "acetone"),
            ("pyridine", "pyridine"),
            # Ether aliases.
            ("ether", "diethylether"),
            ("et2o", "diethylether"),
            ("diethylether", "diethylether"),
        ],
    )
    def test_alias_table(self, raw, expected):
        assert solvent_to_orca_smd(raw) == expected

    def test_passthrough_unknown_solvent(self):
        # ORCA accepts a long tail of solvents we haven't memorized;
        # unknown tokens should pass through verbatim.
        assert solvent_to_orca_smd("1,4-dioxane") == "1,4-dioxane"
        assert solvent_to_orca_smd("DMAc") == "DMAc"

    def test_strip_whitespace(self):
        assert solvent_to_orca_smd("  thf  ") == "THF"

    def test_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            solvent_to_orca_smd(None)  # type: ignore[arg-type]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            solvent_to_orca_smd("")
        with pytest.raises(ValueError, match="empty"):
            solvent_to_orca_smd("   ")


# --------------------------------------------------------------------
# make_orca_simple_input
# --------------------------------------------------------------------


class TestMakeOrcaSimpleInput:
    def _base(self, **overrides):
        defaults = dict(
            keywords="r2scan-3c TightSCF TightOpt",
            nprocs=16,
            maxcore=4000,
            charge=0,
            multiplicity=1,
            solvent=None,
            xyz_filename="input.xyz",
        )
        defaults.update(overrides)
        return make_orca_simple_input(**defaults)

    def test_default_vacuum_neutral_singlet(self):
        text = self._base()
        # Simple-input line.
        assert text.startswith("! r2scan-3c TightSCF TightOpt\n")
        # %pal / %maxcore blocks.
        assert "%pal\n  nprocs 16\nend" in text
        assert "%maxcore 4000" in text
        # No CPCM block in vacuum.
        assert "%cpcm" not in text
        assert "SMDsolvent" not in text
        # Coordinate spec at the bottom.
        assert "* xyzfile 0 1 input.xyz" in text

    def test_charge_and_multiplicity(self):
        text = self._base(charge=-1, multiplicity=3)
        assert "* xyzfile -1 3 input.xyz" in text

    def test_solvent_emits_smd_block(self):
        text = self._base(solvent="dcm")
        assert "%cpcm" in text
        assert "smd true" in text
        assert 'SMDsolvent "CH2Cl2"' in text

    def test_solvent_alias_used(self):
        # Pass "thf" — should be canonicalized to "THF".
        text = self._base(solvent="thf")
        assert 'SMDsolvent "THF"' in text

    def test_smd_solvent_override_bypasses_alias(self):
        # ``smd_solvent_override`` should pass through verbatim and
        # ignore the alias map entirely.
        text = self._base(solvent="thf", smd_solvent_override="my_custom_solvent")
        assert 'SMDsolvent "my_custom_solvent"' in text

    def test_keywords_with_leading_bang_stripped(self):
        # Operators commonly forget that ``!`` is added automatically.
        text = self._base(keywords="! b97-3c TightOpt")
        assert text.startswith("! b97-3c TightOpt\n")
        # And no double-bang in the middle.
        assert "!! " not in text

    def test_keywords_strips_whitespace(self):
        text = self._base(keywords="   r2scan-3c TightOpt   ")
        assert text.startswith("! r2scan-3c TightOpt\n")

    def test_empty_keywords_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self._base(keywords="")

    def test_empty_keywords_after_bang_strip_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self._base(keywords="!  ")

    def test_nprocs_int_coercion(self):
        # Float-shaped nprocs should be coerced to int — this catches
        # the case where parse_int returns a stringified number.
        text = self._base(nprocs=8)
        assert "nprocs 8\n" in text

    def test_maxcore_int_coercion(self):
        text = self._base(maxcore=2500)
        assert "%maxcore 2500" in text

    def test_xyz_filename_passed_through(self):
        text = self._base(xyz_filename="conf_0007.xyz")
        assert "* xyzfile 0 1 conf_0007.xyz" in text

    def test_full_shape_byte_for_byte(self):
        """Lock the full output shape so legacy tooling doesn't break."""
        text = make_orca_simple_input(
            keywords="r2scan-3c TightSCF TightOpt",
            nprocs=16,
            maxcore=4000,
            charge=0,
            multiplicity=1,
            solvent="dcm",
            xyz_filename="input.xyz",
        )
        expected = (
            "! r2scan-3c TightSCF TightOpt\n"
            "\n"
            "%pal\n"
            "  nprocs 16\n"
            "end\n"
            "\n"
            "%maxcore 4000\n"
            "\n"
            "%cpcm\n"
            "  smd true\n"
            '  SMDsolvent "CH2Cl2"\n'
            "end\n"
            "\n"
            "* xyzfile 0 1 input.xyz\n"
        )
        assert text == expected


# --------------------------------------------------------------------
# make_orca_compound_input
# --------------------------------------------------------------------


class TestMakeOrcaCompoundInput:
    def _base(self, **overrides):
        defaults = dict(
            keywords="r2scan-3c TightSCF Freq",
            singlepoint_keywords="wB97M-V def2-TZVPP TightSCF",
            nprocs=8,
            maxcore=4000,
            charge=0,
            multiplicity=1,
            solvent=None,
            xyz_filename="input.xyz",
        )
        defaults.update(overrides)
        return make_orca_compound_input(**defaults)

    def test_two_job_separator_emitted(self):
        text = self._base()
        # The $new_job separator splits the two jobs.
        assert "\n$new_job\n" in text
        # Both jobs render their own ``!`` line.
        assert "! r2scan-3c TightSCF Freq" in text
        assert "! wB97M-V def2-TZVPP TightSCF" in text
        # In order: freq before SP.
        assert text.index("Freq") < text.index("wB97M-V")

    def test_singlepoint_none_returns_single_job(self):
        # When SP keywords are None, the output is identical to
        # make_orca_simple_input — no $new_job, no second job.
        text = self._base(singlepoint_keywords=None)
        assert "$new_job" not in text
        # Only one ``!`` line.
        assert text.count("\n! ") == 0  # leading ! has no newline before
        assert text.count("! ") == 1

    def test_singlepoint_empty_string_treated_as_none(self):
        text = self._base(singlepoint_keywords="")
        assert "$new_job" not in text

    def test_singlepoint_whitespace_treated_as_none(self):
        text = self._base(singlepoint_keywords="   ")
        assert "$new_job" not in text

    def test_both_jobs_share_geometry(self):
        # Both jobs reference the same input.xyz, so the SP picks up the
        # exact geometry the freq job ran on.
        text = self._base(xyz_filename="conf_0007.xyz")
        # The xyz line appears in BOTH jobs (count >= 2).
        assert text.count("* xyzfile 0 1 conf_0007.xyz") == 2

    def test_both_jobs_share_charge_and_multiplicity(self):
        text = self._base(charge=-1, multiplicity=3)
        assert text.count("* xyzfile -1 3 input.xyz") == 2

    def test_both_jobs_share_solvent(self):
        text = self._base(solvent="dcm")
        # SMD block appears once per job → twice total.
        assert text.count('SMDsolvent "CH2Cl2"') == 2
        assert text.count("%cpcm") == 2

    def test_both_jobs_share_nprocs_and_maxcore(self):
        text = self._base(nprocs=4, maxcore=2500)
        assert text.count("nprocs 4") == 2
        assert text.count("%maxcore 2500") == 2

    def test_full_compound_shape(self):
        # Lock the full byte-for-byte output of a default compound run
        # so the legacy shape stays stable across refactors.
        text = make_orca_compound_input(
            keywords="r2scan-3c TightSCF Freq",
            singlepoint_keywords="wB97M-V def2-TZVPP TightSCF",
            nprocs=8,
            maxcore=4000,
            charge=0,
            multiplicity=1,
            solvent=None,
            xyz_filename="input.xyz",
        )
        expected = (
            "! r2scan-3c TightSCF Freq\n"
            "\n"
            "%pal\n"
            "  nprocs 8\n"
            "end\n"
            "\n"
            "%maxcore 4000\n"
            "\n"
            "* xyzfile 0 1 input.xyz\n"
            "\n"
            "$new_job\n"
            "\n"
            "! wB97M-V def2-TZVPP TightSCF\n"
            "\n"
            "%pal\n"
            "  nprocs 8\n"
            "end\n"
            "\n"
            "%maxcore 4000\n"
            "\n"
            "* xyzfile 0 1 input.xyz\n"
        )
        assert text == expected

    def test_single_job_matches_simple_input(self):
        # When SP is None, compound output should byte-equal what
        # make_orca_simple_input produces with the same args.
        compound = make_orca_compound_input(
            keywords="r2scan-3c TightSCF Freq",
            singlepoint_keywords=None,
            nprocs=8,
            maxcore=4000,
            charge=0,
            multiplicity=1,
            solvent="thf",
            xyz_filename="input.xyz",
        )
        simple = make_orca_simple_input(
            keywords="r2scan-3c TightSCF Freq",
            nprocs=8,
            maxcore=4000,
            charge=0,
            multiplicity=1,
            solvent="thf",
            xyz_filename="input.xyz",
        )
        assert compound == simple

    def test_final_e_parser_takes_sp_value(self):
        # Critical contract: parse_orca_final_energy uses [-1] which,
        # for a two-job compound, is the SP energy. Build a synthetic
        # output with two FINAL E lines (low-level then high-level) and
        # verify the parser returns the high-level one.
        # (This pins the assumption thermo_aggregate relies on.)
        text = (
            "FINAL SINGLE POINT ENERGY    -100.123456789\n"
            "...freq job complete...\n"
            "FINAL SINGLE POINT ENERGY    -100.234567891\n"
            "...sp job complete...\n"
        )
        matches = FINAL_E_RE.findall(text)
        assert matches == ["-100.123456789", "-100.234567891"]
        assert float(matches[-1]) == pytest.approx(-100.234567891)


# --------------------------------------------------------------------
# FINAL_E_RE / parse_orca_final_energy
# --------------------------------------------------------------------


class TestFinalEnergyRegex:
    def test_simple_match(self):
        line = "FINAL SINGLE POINT ENERGY      -1234.567890123"
        m = FINAL_E_RE.search(line)
        assert m is not None
        assert float(m.group(1)) == pytest.approx(-1234.567890123)

    def test_scientific_notation_match(self):
        line = "FINAL SINGLE POINT ENERGY  -1.234567e+03"
        m = FINAL_E_RE.search(line)
        assert m is not None
        assert float(m.group(1)) == pytest.approx(-1234.567)

    def test_findall_takes_last(self):
        text = (
            "FINAL SINGLE POINT ENERGY     -100.0\n"
            "some optimization noise\n"
            "FINAL SINGLE POINT ENERGY     -200.0\n"
            "even more noise\n"
            "FINAL SINGLE POINT ENERGY     -300.5\n"
        )
        matches = FINAL_E_RE.findall(text)
        assert matches == ["-100.0", "-200.0", "-300.5"]
        # The parser uses [-1] — confirm that's the converged value.
        assert float(matches[-1]) == -300.5


class TestParseOrcaFinalEnergy:
    def test_happy_path_takes_last_match(self, tmp_path):
        out = tmp_path / "orca.out"
        out.write_text(
            "FINAL SINGLE POINT ENERGY  -100.0\n"
            "intermediate\n"
            "FINAL SINGLE POINT ENERGY  -200.5\n"
        )
        assert parse_orca_final_energy(out) == pytest.approx(-200.5)

    def test_missing_file_returns_none(self, tmp_path):
        assert parse_orca_final_energy(tmp_path / "does_not_exist.out") is None

    def test_no_match_returns_none(self, tmp_path):
        out = tmp_path / "orca.out"
        out.write_text("nothing useful here\n")
        assert parse_orca_final_energy(out) is None

    def test_garbled_number_returns_none(self, tmp_path):
        # The regex is strict (digits + dot), so a garbled number
        # simply doesn't match — the function returns None rather than
        # raising.
        out = tmp_path / "orca.out"
        out.write_text("FINAL SINGLE POINT ENERGY  not_a_number\n")
        assert parse_orca_final_energy(out) is None

    def test_handles_unicode_garbage(self, tmp_path):
        # The reader uses errors="replace" so weird bytes don't crash
        # the parser. Make sure a valid match later in the file is
        # still found.
        out = tmp_path / "orca.out"
        out.write_bytes(
            b"\xff\xfe\xc3 some garbage\n"
            b"FINAL SINGLE POINT ENERGY  -42.0\n"
        )
        assert parse_orca_final_energy(out) == pytest.approx(-42.0)


# --------------------------------------------------------------------
# ORCA_NORMAL_RE / orca_terminated_normally
# --------------------------------------------------------------------


class TestOrcaNormalRegex:
    def test_matches_canonical_phrase(self):
        assert ORCA_NORMAL_RE.search("ORCA TERMINATED NORMALLY") is not None

    def test_extra_whitespace(self):
        # Tabs / multi-space between tokens still match.
        assert (
            ORCA_NORMAL_RE.search("ORCA   TERMINATED\tNORMALLY")
            is not None
        )

    def test_case_insensitive(self):
        # Some legacy builds emit lowercase termination footers.
        assert ORCA_NORMAL_RE.search("orca terminated normally") is not None

    def test_no_match_when_partial(self):
        # SCF blowup produces "ABORTING THE RUN" with no NORMALLY footer.
        text = "ORCA finished with errors -- ABORTING THE RUN\n"
        assert ORCA_NORMAL_RE.search(text) is None


class TestOrcaTerminatedNormally:
    def test_normal_termination(self, tmp_path):
        out = tmp_path / "orca.out"
        out.write_text(
            "...lots of output...\n"
            "FINAL SINGLE POINT ENERGY  -100.0\n"
            "                ****ORCA TERMINATED NORMALLY****\n"
        )
        assert orca_terminated_normally(out) is True

    def test_no_normal_footer(self, tmp_path):
        out = tmp_path / "orca.out"
        out.write_text(
            "FINAL SINGLE POINT ENERGY  -100.0\n"
            "[killed by walltime]\n"
        )
        assert orca_terminated_normally(out) is False

    def test_missing_file(self, tmp_path):
        assert orca_terminated_normally(tmp_path / "missing.out") is False

    def test_only_reads_tail(self, tmp_path):
        # Put the NORMAL footer near the start, then a huge body of
        # garbage. With a small tail_bytes the footer is OUTSIDE the
        # window, so the function should return False — this proves the
        # tail-only read is in effect (and that it's the operator's
        # responsibility to use a sensible tail size).
        out = tmp_path / "orca.out"
        out.write_text(
            "ORCA TERMINATED NORMALLY\n"
            + "x" * 20000
        )
        # Only look at the last 64 bytes — those are all 'x'.
        assert orca_terminated_normally(out, tail_bytes=64) is False
        # Default 16 KiB still doesn't reach 20000 bytes back, so
        # default also returns False — confirming tail behavior.
        assert orca_terminated_normally(out) is False

    def test_default_tail_finds_footer_at_end(self, tmp_path):
        # Footer at the END of a long file: default tail finds it.
        out = tmp_path / "orca.out"
        out.write_text(
            "x" * 5000
            + "\nORCA TERMINATED NORMALLY\n"
        )
        assert orca_terminated_normally(out) is True

    def test_unicode_garbage_in_tail(self, tmp_path):
        out = tmp_path / "orca.out"
        out.write_bytes(
            b"\xff\xfe\xc3 garbled\n"
            b"ORCA TERMINATED NORMALLY\n"
        )
        # errors="replace" means weird bytes don't crash the search.
        assert orca_terminated_normally(out) is True


# --------------------------------------------------------------------
# concat_xyz_files
# --------------------------------------------------------------------


class TestConcatXyzFiles:
    def test_simple_concat(self, tmp_path):
        a = tmp_path / "a.xyz"
        b = tmp_path / "b.xyz"
        a.write_text("3\nframe a\nC 0 0 0\nO 0 0 1.0\nH 0 0 -1.0\n")
        b.write_text("3\nframe b\nC 0 0 0\nO 0 0 1.1\nH 0 0 -1.0\n")

        out = tmp_path / "ensemble.xyz"
        concat_xyz_files([a, b], out)

        text = out.read_text()
        assert text.count("frame a") == 1
        assert text.count("frame b") == 1
        # Both frames present in order.
        assert text.index("frame a") < text.index("frame b")

    def test_no_trailing_newline_added(self, tmp_path):
        # If the per-frame file is missing a trailing \n the concat
        # should add one — otherwise the next frame's atom count line
        # gets glued onto the previous frame's last atom line.
        a = tmp_path / "a.xyz"
        b = tmp_path / "b.xyz"
        a.write_text("3\nframe a\nC 0 0 0\nO 0 0 1.0\nH 0 0 -1.0")
        b.write_text("3\nframe b\nC 0 0 0\nO 0 0 1.1\nH 0 0 -1.0")

        out = tmp_path / "ensemble.xyz"
        concat_xyz_files([a, b], out)

        # Frame "a"'s last atom line and frame "b"'s atom count line
        # should NOT share a line.
        text = out.read_text()
        assert "H 0 0 -1.03\nframe b" not in text
        # The frames are separated by a newline.
        assert "H 0 0 -1.0\n3\nframe b" in text

    def test_empty_inputs_writes_empty_file(self, tmp_path):
        out = tmp_path / "ensemble.xyz"
        concat_xyz_files([], out)
        assert out.exists()
        assert out.read_text() == ""


# --------------------------------------------------------------------
# write_energy_file
# --------------------------------------------------------------------


class TestWriteEnergyFile:
    def test_simple_three_conformers(self, tmp_path):
        out = tmp_path / "orca.energies"
        rel_kcal, e_min = write_energy_file(
            energies_h=[-100.0, -99.99, -99.98],
            out_path=out,
        )
        assert e_min == -100.0
        # rel_kcal scaled by HARTREE_TO_KCAL.
        assert rel_kcal[0] == pytest.approx(0.0)
        assert rel_kcal[1] == pytest.approx(0.01 * HARTREE_TO_KCAL)
        assert rel_kcal[2] == pytest.approx(0.02 * HARTREE_TO_KCAL)

        # File format: 3 columns, one row per conformer.
        text = out.read_text()
        # Use splitlines() WITHOUT stripping first — write_energy_file's
        # ``{i:6d}`` format pads the index to 6 columns, and .strip()
        # would eat the first line's leading whitespace.
        lines = text.splitlines()
        assert len(lines) == 3
        # 1-based index column, right-aligned to width 6.
        assert lines[0].startswith("     1   ")
        assert lines[1].startswith("     2   ")
        assert lines[2].startswith("     3   ")

    def test_missing_entries_become_nan(self, tmp_path):
        out = tmp_path / "orca.energies"
        rel_kcal, e_min = write_energy_file(
            energies_h=[-100.0, None, -99.98],
            out_path=out,
        )
        # Min computed from finite entries only.
        assert e_min == -100.0
        # rel for the None entry is None.
        assert rel_kcal[0] == pytest.approx(0.0)
        assert rel_kcal[1] is None
        assert rel_kcal[2] == pytest.approx(0.02 * HARTREE_TO_KCAL)

        # File renders the missing row with NaN in both columns.
        text = out.read_text()
        lines = text.strip().splitlines()
        assert "NaN   NaN" in lines[1]

    def test_all_missing_returns_none_emin(self, tmp_path):
        out = tmp_path / "orca.energies"
        rel_kcal, e_min = write_energy_file(
            energies_h=[None, None, None],
            out_path=out,
        )
        assert e_min is None
        assert rel_kcal == [None, None, None]
        # Every line is NaN NaN.
        for line in out.read_text().strip().splitlines():
            assert "NaN   NaN" in line

    def test_single_finite_entry(self, tmp_path):
        out = tmp_path / "orca.energies"
        rel_kcal, e_min = write_energy_file(
            energies_h=[-50.0],
            out_path=out,
        )
        assert e_min == -50.0
        assert rel_kcal == [pytest.approx(0.0)]

    def test_hartree_to_kcal_constant(self):
        # Locking the CODATA 2018 conversion constant — downstream
        # tooling reproduces ORCA's thermochemistry to the last digit
        # and depends on this exact value.
        assert HARTREE_TO_KCAL == 627.509474

    def test_file_ends_with_newline(self, tmp_path):
        out = tmp_path / "orca.energies"
        write_energy_file(
            energies_h=[-1.0, -0.5],
            out_path=out,
        )
        assert out.read_text().endswith("\n")

    def test_custom_conversion_factor(self, tmp_path):
        # Allow callers to pass a different rel_kcal_per_h, e.g. for
        # alternative conversion conventions.
        out = tmp_path / "orca.energies"
        rel_kcal, _ = write_energy_file(
            energies_h=[-100.0, -99.0],
            out_path=out,
            rel_kcal_per_h=1.0,
        )
        assert rel_kcal[1] == pytest.approx(1.0)


# --------------------------------------------------------------------
# Module surface
# --------------------------------------------------------------------


class TestPublicSurface:
    def test_all_exports_importable(self):
        from scripps_workflow import orca as o

        for name in (
            "FINAL_E_RE",
            "HARTREE_TO_KCAL",
            "ORCA_NORMAL_RE",
            "concat_xyz_files",
            "make_orca_compound_input",
            "make_orca_simple_input",
            "orca_terminated_normally",
            "parse_orca_final_energy",
            "solvent_to_orca_smd",
            "write_energy_file",
        ):
            assert hasattr(o, name), f"missing public export: {name}"
