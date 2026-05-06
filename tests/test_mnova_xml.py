"""Tests for ``scripps_workflow.mnova_xml``.

Pure-renderer tests — no RDKit, no chemistry. We construct
:class:`EquivalenceGroup` objects by hand to feed the renderer and
verify byte-exact output structure plus selected formatting details.

Three concerns:

1. Number formatting (the ``_fmt_decimals`` 6-decimal rule for shifts /
   J's, the ``_fmt_g`` trim-trailing-zeros rule for frequency / from /
   to / population / line width).

2. Single ``<spin-system>`` render — verify ``<group>`` order, the
   lower-triangular ``<summary>``, jCoupling/dCoupling pairing,
   missing-J fallback to ``0``, and overall envelope (``<?xml…?>``,
   ``<mnova-spinsim>`` root, terminating newline).

3. Multi ``<spin-system>`` render (per-conformer mode) — multiple
   sibling spin-systems with distinct ``<population>`` values, single
   ``<spectrum>`` at the top level.

A round-trip parse via the stdlib ``xml.etree.ElementTree`` confirms
the output is well-formed XML with the expected element counts.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from scripps_workflow.equivalence import EquivalenceGroup, Tier
from scripps_workflow.mnova_xml import (
    SpectrumConfig,
    SpinSystem,
    render_mnova_xml,
)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _methyl_group(name: str = "A", shift: float = 2.42) -> EquivalenceGroup:
    """A 3-H methyl (HARD class). J's filled in at the call site."""
    return EquivalenceGroup(
        name=name,
        element="H",
        atom_indices=(0, 1, 2),
        shift_avg_ppm=shift,
        tier=Tier.HARD,
        j_couplings={},
    )


def _single_h(name: str, atom_idx: int, shift: float) -> EquivalenceGroup:
    return EquivalenceGroup(
        name=name,
        element="H",
        atom_indices=(atom_idx,),
        shift_avg_ppm=shift,
        tier=Tier.NONE,
        j_couplings={},
    )


def _two_group_system_with_j() -> SpinSystem:
    """A methyl A coupled to a single H called B (J=7.0 Hz). Keeps the
    test XML small enough to inspect line-by-line."""
    a = EquivalenceGroup(
        name="A",
        element="H",
        atom_indices=(0, 1, 2),
        shift_avg_ppm=2.420000,
        tier=Tier.HARD,
        j_couplings={"B": 7.000000},
    )
    b = EquivalenceGroup(
        name="B",
        element="H",
        atom_indices=(3,),
        shift_avg_ppm=4.500000,
        tier=Tier.NONE,
        j_couplings={"A": 7.000000},
    )
    return SpinSystem(groups=(a, b), population=1.0)


# --------------------------------------------------------------------
# Envelope + structure
# --------------------------------------------------------------------


class TestEnvelopeAndStructure:
    def test_starts_with_xml_declaration(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>\n')

    def test_root_is_mnova_spinsim(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        root = ET.fromstring(xml)
        assert root.tag == "mnova-spinsim"

    def test_terminates_with_newline(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        assert xml.endswith("</mnova-spinsim>\n")

    def test_empty_spin_systems_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            render_mnova_xml([], spectrum=SpectrumConfig())

    def test_round_trip_parses_cleanly(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        # Stdlib ElementTree should accept the document without raising.
        ET.fromstring(xml)


# --------------------------------------------------------------------
# Group rendering
# --------------------------------------------------------------------


class TestGroupRendering:
    def test_methyl_attributes(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig(line_width_hz=1.0))
        # name + spinByTwo + lineWidth + number all present on group A.
        assert (
            '<group name="A" spinByTwo="1" lineWidth="1" number="3">'
            in xml
        )
        assert (
            '<group name="B" spinByTwo="1" lineWidth="1" number="1">'
            in xml
        )

    def test_shift_is_six_decimal(self):
        # shift_avg_ppm=2.42 should render with trailing zeros to 6 places.
        ss = SpinSystem(
            groups=(_methyl_group("A", shift=2.42),),
            population=1.0,
        )
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        assert "<shift>2.420000</shift>" in xml

    def test_jcoupling_six_decimal_and_dcoupling_zero(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        # Group A has a jCoupling to B at 7.0 — formatted as 7.000000.
        assert '<jCoupling name="B">7.000000</jCoupling>' in xml
        # Every jCoupling has a paired dCoupling at literal 0.
        assert '<dCoupling name="B">0</dCoupling>' in xml
        # Symmetric direction too.
        assert '<jCoupling name="A">7.000000</jCoupling>' in xml
        assert '<dCoupling name="A">0</dCoupling>' in xml

    def test_missing_jcoupling_renders_as_zero(self):
        # Three groups; A has J's only to B, not to C. Missing J should
        # appear as 0.000000 in A's <jCoupling name="C"> element rather
        # than being omitted.
        a = EquivalenceGroup(
            name="A", element="H", atom_indices=(0,),
            shift_avg_ppm=1.0, tier=Tier.NONE,
            j_couplings={"B": 5.0},
        )
        b = EquivalenceGroup(
            name="B", element="H", atom_indices=(1,),
            shift_avg_ppm=2.0, tier=Tier.NONE,
            j_couplings={"A": 5.0},
        )
        c = EquivalenceGroup(
            name="C", element="H", atom_indices=(2,),
            shift_avg_ppm=3.0, tier=Tier.NONE,
            j_couplings={},
        )
        xml = render_mnova_xml(
            [SpinSystem(groups=(a, b, c), population=1.0)],
            spectrum=SpectrumConfig(),
        )
        assert '<jCoupling name="C">0.000000</jCoupling>' in xml
        # And each missing pair still has its dCoupling 0 partner.
        assert '<dCoupling name="C">0</dCoupling>' in xml

    def test_qconst_always_zero(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        # qConst is hardcoded 0 (reserved for quadrupolar nuclei,
        # which the current pipeline doesn't model).
        assert xml.count("<qConst>0</qConst>") == 2  # one per group

    def test_group_order_preserved(self):
        # Render groups in caller-supplied order. We pass C, A, B and
        # expect that order in the output (the caller is responsible
        # for sorting if needed; the orchestrator already does so).
        a = _single_h("A", atom_idx=0, shift=1.0)
        b = _single_h("B", atom_idx=1, shift=2.0)
        c = _single_h("C", atom_idx=2, shift=3.0)
        ss = SpinSystem(groups=(c, a, b), population=1.0)
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        c_pos = xml.index('name="C"')
        a_pos = xml.index('name="A"')
        b_pos = xml.index('name="B"')
        # First <group> tag for each name should appear in the supplied order.
        assert c_pos < a_pos < b_pos


# --------------------------------------------------------------------
# Summary block (lower-triangular matrix)
# --------------------------------------------------------------------


class TestSummaryBlock:
    def test_header_row_is_tab_indented(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        # The summary header line starts with a tab, then group names.
        assert "\tA\tB\n" in xml

    def test_row_has_priors_then_diagonal(self):
        # Three groups with asymmetric J's: A↔B = 5, A↔C = 10, B↔C
        # missing. The summary's lower triangle should encode all three
        # plus each group's diagonal shift.
        a = EquivalenceGroup(
            name="A", element="H", atom_indices=(0,),
            shift_avg_ppm=1.0, tier=Tier.NONE,
            j_couplings={"B": 5.0, "C": 10.0},
        )
        b = EquivalenceGroup(
            name="B", element="H", atom_indices=(1,),
            shift_avg_ppm=2.0, tier=Tier.NONE,
            j_couplings={"A": 5.0},
        )
        c = EquivalenceGroup(
            name="C", element="H", atom_indices=(2,),
            shift_avg_ppm=3.0, tier=Tier.NONE,
            j_couplings={"A": 10.0},
        )
        ss = SpinSystem(groups=(a, b, c), population=1.0)
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        # Row 1 (A): "A\t" + diagonal shift = "A\t1.000000"
        # Row 2 (B): "B\t" + J(B,A) + diagonal = "B\t5.000000\t2.000000"
        # Row 3 (C): "C\t" + J(C,A) + J(C,B) + diagonal
        #           = "C\t10.000000\t0\t3.000000"  (J(C,B) missing → 0)
        assert "\nA\t1.000000\n" in xml
        assert "\nB\t5.000000\t2.000000\n" in xml
        assert "\nC\t10.000000\t0\t3.000000\n" in xml

    def test_summary_tags_are_indented(self):
        # The <summary> opening and closing tags sit at 8-space indent.
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        assert "        <summary>" in xml
        assert "        </summary>" in xml


# --------------------------------------------------------------------
# Spectrum block + number formatting
# --------------------------------------------------------------------


class TestSpectrumAndFormatting:
    def test_spectrum_fields(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml(
            [ss],
            spectrum=SpectrumConfig(
                frequency_mhz=400.13, points=16384,
                from_ppm=0.0, to_ppm=8.5, line_width_hz=1.0,
            ),
        )
        assert "<frequency>400.13</frequency>" in xml
        assert "<points>16384</points>" in xml
        assert "<from>0</from>" in xml
        assert "<to>8.5</to>" in xml

    def test_population_integer_renders_without_decimal(self):
        ss = _two_group_system_with_j()
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        assert "<population>1</population>" in xml

    def test_population_fractional_renders_g_style(self):
        ss = SpinSystem(
            groups=(_methyl_group("A"),),
            population=0.234,
        )
        xml = render_mnova_xml([ss], spectrum=SpectrumConfig())
        assert "<population>0.234</population>" in xml

    @pytest.mark.parametrize(
        "freq, expected",
        [
            (400.0, "400"),
            (400.13, "400.13"),
            (100.0, "100"),
            (600.5, "600.5"),
        ],
    )
    def test_frequency_g_style(self, freq, expected):
        xml = render_mnova_xml(
            [_two_group_system_with_j()],
            spectrum=SpectrumConfig(frequency_mhz=freq),
        )
        assert f"<frequency>{expected}</frequency>" in xml


# --------------------------------------------------------------------
# Multi-spin-system (per-conformer) mode
# --------------------------------------------------------------------


class TestMultiSpinSystem:
    def test_two_systems_with_distinct_populations(self):
        # Two spin systems, weights 0.7 and 0.3.
        ss1 = SpinSystem(
            groups=(_methyl_group("A", shift=2.40),), population=0.7,
        )
        ss2 = SpinSystem(
            groups=(_methyl_group("A", shift=2.42),), population=0.3,
        )
        xml = render_mnova_xml([ss1, ss2], spectrum=SpectrumConfig())
        # Both populations appear.
        assert "<population>0.7</population>" in xml
        assert "<population>0.3</population>" in xml
        # ElementTree confirms the structure: 2 <spin-system> children
        # of root, plus 1 <spectrum>.
        root = ET.fromstring(xml)
        assert len(root.findall("spin-system")) == 2
        assert len(root.findall("spectrum")) == 1

    def test_spectrum_appears_once_at_top_level(self):
        # Even with N spin-systems, exactly one <spectrum> block.
        systems = [
            SpinSystem(groups=(_methyl_group("A"),), population=w)
            for w in (0.5, 0.3, 0.2)
        ]
        xml = render_mnova_xml(systems, spectrum=SpectrumConfig())
        assert xml.count("<spectrum>") == 1
        assert xml.count("</spectrum>") == 1

    def test_per_system_groups_are_isolated(self):
        # Each spin-system carries its own <group> blocks; they don't
        # leak across systems. We use distinct group names per system
        # to verify (in practice the same molecule's groups would
        # share names, but the renderer is name-agnostic).
        ss1 = SpinSystem(groups=(_methyl_group("A"),), population=0.5)
        ss2 = SpinSystem(groups=(_methyl_group("X"),), population=0.5)
        xml = render_mnova_xml([ss1, ss2], spectrum=SpectrumConfig())
        # Each name appears exactly once in <group name="..."> form
        # (not counting jCoupling references — there are none here).
        assert xml.count('<group name="A"') == 1
        assert xml.count('<group name="X"') == 1
