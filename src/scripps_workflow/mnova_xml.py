"""mnova-spinsim XML emitter for NMR spin-system simulation files.

Pure rendering — given a list of :class:`SpinSystem` (each carrying
:class:`scripps_workflow.equivalence.EquivalenceGroup` records) plus a
:class:`SpectrumConfig`, produces a string in the byte-exact format
that mnova's spin-simulation engine expects. No chemistry, no RDKit.

The mnova format is straightforward but has a few quirks worth pinning
in tests:

1. **Lower-triangular ``<summary>`` block.** A redundant tab-separated
   matrix that mnova displays as a quick reference. Header row starts
   with a tab, then group names. Each data row is ``name + J's to
   prior groups + diagonal shift``. The summary's content lines are
   NOT XML-indented — they're raw text between the opening/closing
   tags. Matches the reference file ``SF_M_001.xml`` byte-for-byte.

2. **Every ``<jCoupling>`` is paired with a ``<dCoupling>``.** Dipolar
   couplings are always 0 in solution NMR, but the schema requires the
   tag. Missing J's are emitted as ``0`` (not omitted) — matches the
   reference and lets mnova's parser stay happy without conditional
   logic per row.

3. **Number formatting.**

   * shifts and J/D couplings: 6 decimal places (``f"{val:.6f}"``).
   * integer-y fields (``points``, ``qConst=0``, ``dCoupling=0``):
     plain int.
   * frequency / from / to / population / lineWidth: ``%g``-style —
     trims trailing zeros so ``1.0`` renders as ``"1"`` and ``0.0``
     as ``"0"`` (matches the reference's ``<from>0</from>`` and
     ``<population>1</population>`` shape).

4. **Multi-spin-system mode.** Multiple ``<spin-system>`` siblings
   each with their own ``<population>`` value. The ``<spectrum>``
   block stays at the top level, applied uniformly to all systems.
   Used for per-conformer rendering where each conformer's
   spin-system carries its Boltzmann weight as ``<population>``.

5. **Quadrupolar coupling (``<qConst>``)** is hardcoded to 0. mnova
   uses it for spin-1+ nuclei (²H, ¹⁴N), which the current pipeline
   doesn't model. If we ever add quadrupolar support, this is the
   one field that needs a real value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .equivalence import EquivalenceGroup


# --------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumConfig:
    """Simulation parameters that fill the ``<spectrum>`` block.

    Defaults target a typical ¹H NMR run on a 400 MHz spectrometer.
    For ¹³C, a 100 MHz frequency + 0–220 ppm window is more typical;
    callers should pass element-specific values rather than relying on
    these defaults.

    ``line_width_hz`` is applied uniformly to every ``<group>`` — mnova
    accepts a per-group attribute, but the current pipeline doesn't
    have a use case for varying line width by atom (the reference file
    uses a single value across all 15 groups).
    """

    frequency_mhz: float = 400.13
    points: int = 16384
    from_ppm: float = 0.0
    to_ppm: float = 12.0
    line_width_hz: float = 1.0


@dataclass(frozen=True)
class SpinSystem:
    """One ``<spin-system>`` block: equivalence groups + Boltzmann population.

    For pre-averaged single-system mode, ``population=1.0`` and one
    SpinSystem suffices. For per-conformer mode, build one SpinSystem
    per conformer with ``population`` set to that conformer's
    Boltzmann weight; the weights need NOT sum to 1 (mnova normalizes
    internally), but the convention is to feed normalized weights.

    ``groups`` is a tuple (not list) so SpinSystem stays hashable and
    immutable. Order matters — it controls the order ``<group>`` blocks
    appear in the rendered XML and the column order in the
    ``<summary>`` matrix. Convention: lowest-named group first
    (alphabetical, which equals atom-index order from
    :func:`scripps_workflow.equivalence.compute_equivalence_groups`).
    """

    groups: tuple[EquivalenceGroup, ...]
    population: float = 1.0


# --------------------------------------------------------------------
# Number formatting helpers
# --------------------------------------------------------------------


def _fmt_decimals(val: float, decimals: int = 6) -> str:
    """Fixed-decimal float — matches the reference file's shift / J format."""
    return f"{float(val):.{decimals}f}"


def _fmt_g(val: float) -> str:
    """``%g``-style float — trims trailing zeros, integer-when-exact.

    ``0.0`` → ``"0"``; ``8.5`` → ``"8.5"``; ``1.0`` → ``"1"``;
    ``400.13`` → ``"400.13"``. Matches the reference file's frequency,
    population, from/to, and lineWidth fields. Note that very small
    values (< ~1e-4) tip into scientific notation under ``%g``; the
    pipeline shouldn't see those in practice (Boltzmann weights below
    that threshold contribute negligibly to the simulated spectrum).
    """
    f = float(val)
    if f == int(f):
        return str(int(f))
    return f"{f:g}"


# --------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------


def _render_summary(groups: tuple[EquivalenceGroup, ...]) -> str:
    """Render the ``<summary>`` block: a lower-triangular tab-separated matrix.

    Header row: ``\\t`` + tab-joined group names.
    Each data row: name + N-1 J couplings to prior groups + diagonal shift.
    Empty group list collapses to an empty summary tag pair.

    The content lines (header + data rows) are NOT XML-indented; they
    sit flush left between the indented opening and closing tags. This
    matches the reference file's formatting exactly — mnova reads the
    summary as raw text rather than nested elements.
    """
    if not groups:
        return "        <summary>\n        </summary>"
    lines: list[str] = ["        <summary>"]
    # Header row.
    lines.append("\t" + "\t".join(g.name for g in groups))
    # Data rows.
    for i, g in enumerate(groups):
        row: list[str] = [g.name]
        for prior in groups[:i]:
            j = g.j_couplings.get(prior.name)
            row.append(_fmt_decimals(j, 6) if j is not None else "0")
        row.append(_fmt_decimals(g.shift_avg_ppm, 6))
        lines.append("\t".join(row))
    lines.append("        </summary>")
    return "\n".join(lines)


def _render_group(
    g: EquivalenceGroup,
    all_groups: tuple[EquivalenceGroup, ...],
    *,
    line_width_hz: float,
) -> str:
    """Render one ``<group>`` element.

    Emits a ``<jCoupling>`` + ``<dCoupling>`` pair for every OTHER group
    in ``all_groups``. Missing J entries (e.g., when ORCA's
    SpinSpinRThresh excluded the pair) fall back to ``0`` rather than
    being omitted, keeping the rendered group structure uniform across
    the file.
    """
    lines: list[str] = []
    lines.append(
        f'        <group name="{g.name}" spinByTwo="{g.spin_by_two}" '
        f'lineWidth="{_fmt_g(line_width_hz)}" number="{g.number}">'
    )
    lines.append(
        f"            <shift>{_fmt_decimals(g.shift_avg_ppm, 6)}</shift>"
    )
    lines.append("            <qConst>0</qConst>")
    for other in all_groups:
        if other.name == g.name:
            continue
        j = g.j_couplings.get(other.name, 0.0)
        lines.append(
            f'            <jCoupling name="{other.name}">'
            f"{_fmt_decimals(j, 6)}</jCoupling>"
        )
        lines.append(f'            <dCoupling name="{other.name}">0</dCoupling>')
    lines.append("        </group>")
    return "\n".join(lines)


def _render_spin_system(ss: SpinSystem, *, line_width_hz: float) -> str:
    """Render one ``<spin-system>`` block (summary + population + groups)."""
    lines: list[str] = ["    <spin-system>"]
    lines.append(_render_summary(ss.groups))
    lines.append(f"        <population>{_fmt_g(ss.population)}</population>")
    for g in ss.groups:
        lines.append(
            _render_group(g, ss.groups, line_width_hz=line_width_hz)
        )
    lines.append("    </spin-system>")
    return "\n".join(lines)


def _render_spectrum(spec: SpectrumConfig) -> str:
    """Render the ``<spectrum>`` block."""
    lines: list[str] = ["    <spectrum>"]
    lines.append(f"        <frequency>{_fmt_g(spec.frequency_mhz)}</frequency>")
    lines.append(f"        <points>{int(spec.points)}</points>")
    lines.append(f"        <from>{_fmt_g(spec.from_ppm)}</from>")
    lines.append(f"        <to>{_fmt_g(spec.to_ppm)}</to>")
    lines.append("    </spectrum>")
    return "\n".join(lines)


def render_mnova_xml(
    spin_systems: Iterable[SpinSystem],
    spectrum: SpectrumConfig,
) -> str:
    """Render a complete mnova-spinsim XML document as a string.

    Output starts with the XML declaration, opens ``<mnova-spinsim>``,
    emits each ``<spin-system>`` in order, then a single ``<spectrum>``
    block, then closes the root with a trailing newline. The trailing
    newline matches the reference file (which terminates with
    ``</mnova-spinsim>\\n``).

    Raises ``ValueError`` when ``spin_systems`` is empty — mnova
    requires at least one system, and an empty file has no plausible
    interpretation.
    """
    systems = list(spin_systems)
    if not systems:
        raise ValueError("render_mnova_xml: spin_systems must be non-empty")

    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append("<mnova-spinsim>")
    for ss in systems:
        parts.append(
            _render_spin_system(ss, line_width_hz=spectrum.line_width_hz)
        )
    parts.append(_render_spectrum(spectrum))
    parts.append("</mnova-spinsim>")
    return "\n".join(parts) + "\n"


__all__ = [
    "SpectrumConfig",
    "SpinSystem",
    "render_mnova_xml",
]
