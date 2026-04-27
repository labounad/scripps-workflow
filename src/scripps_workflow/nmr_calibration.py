"""NMR linear-scaling calibration tables.

DFT NMR predictions tend to have small random error but a meaningful
systematic bias that shifts ALL calculated shieldings by a roughly
constant amount (basis-set incompleteness, missing dynamic correlation,
PCM modeling errors). Empirical linear scaling absorbs that bias and
brings predictions to within ~0.1 ppm for ¹H, ~1–2 ppm for ¹³C, and
~1–2 Hz for ¹H–¹H J-couplings.

The defaults below target Lucas's chosen recipe:

    ¹H shieldings:    GIAO-PCM-WP04 / 6-311++G(2d,p) / CHCl3   (cheshire)
    ¹³C shieldings:   GIAO-PCM-ωB97X-D / 6-31G(d,p) / CHCl3    (cheshire)
    ¹H-¹H couplings:  PCM-mPW1PW91 / pcJ-2 / CHCl3             (Bally/Rablen 2011)

Reference values come from the cheshire NMR repository
(http://cheshirenmr.info/Recommendations.htm) and Bally & Rablen,
*J. Org. Chem.* **76**, 4818 (2011). cheshire occasionally updates the
fitted parameters, so values here should be treated as a starting
point — verify against the upstream tables before publication. The
node-side ``calibration_overrides`` config knob lets users plug in
their own fits without editing this module.

Convention for shieldings (cheshire): the published ``slope`` and
``intercept`` come from regressing σ_calc on δ_exp, i.e.

    σ_calc = slope · δ_exp + intercept

So to predict δ from a new σ, invert:

    δ_predicted = (σ_calc − intercept) / slope

Convention for couplings (Bally/Rablen): the published parameters are
the direct prediction formula:

    J_predicted = slope · J_calc + intercept

The two helpers ``predict_chemical_shift`` and
``predict_coupling_constant`` apply each convention.
"""

from __future__ import annotations

from typing import Any, Optional


#: Linear-scaling table.
#:
#: Key: ``(functional, basis, solvent, nucleus)`` — exact strings, but
#: lookup is case-insensitive (see :func:`lookup_calibration`).
#: Value dict carries:
#:
#:     * ``slope`` (float)
#:     * ``intercept`` (float)
#:     * ``source`` (str) — provenance string for the manifest
#:     * ``valid_range_ppm`` (optional 2-tuple) — chemical-shift range
#:       within which the linear fit was validated; predictions outside
#:       this range should be flagged in downstream output.
#:
#: TODO(labounad): verify the exact slope/intercept values against the
#: current cheshire web table before publication. Numbers here are the
#: published cheshire values as of writing, but they are periodically
#: refit when new datasets are added.
NMR_CALIBRATION: dict[tuple[str, str, str, str], dict[str, Any]] = {
    ("WP04", "6-311++G(2d,p)", "CHCl3", "1H"): {
        "slope": -1.0698,
        "intercept": 31.8447,
        "source": "cheshire (Wiitala/Hoye/Cramer 2006)",
        "valid_range_ppm": (0.0, 12.0),
    },
    ("wB97X-D", "6-31G(d,p)", "CHCl3", "13C"): {
        "slope": -1.0501,
        "intercept": 187.25,
        "source": "cheshire",
        "valid_range_ppm": (0.0, 220.0),
    },
    ("mPW1PW91", "pcJ-2", "CHCl3", "1H-1H_J"): {
        "slope": 0.9105,
        "intercept": 0.21,
        "source": "Bally Rablen 2011 (J. Org. Chem. 76, 4818)",
        "valid_range_ppm": None,
    },
}


# --------------------------------------------------------------------
# Application formulas
# --------------------------------------------------------------------


def predict_chemical_shift(
    sigma_calc_ppm: float, *, slope: float, intercept: float
) -> float:
    """Convert a calculated absolute shielding (ppm) to a predicted δ (ppm).

    Uses the cheshire convention: stored ``(slope, intercept)`` come
    from regressing σ_calc on δ_exp, so::

        δ_predicted = (σ_calc − intercept) / slope

    Raises ``ValueError`` if ``slope`` is zero (would divide by zero).
    """
    if slope == 0.0:
        raise ValueError("predict_chemical_shift: slope cannot be zero")
    return (float(sigma_calc_ppm) - float(intercept)) / float(slope)


def predict_coupling_constant(
    j_calc_hz: float, *, slope: float, intercept: float
) -> float:
    """Apply linear scaling to a calculated J-coupling (Hz).

    Uses the Bally/Rablen convention: stored ``(slope, intercept)``
    are the direct prediction parameters::

        J_predicted = slope · J_calc + intercept
    """
    return float(slope) * float(j_calc_hz) + float(intercept)


# --------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------


def lookup_calibration(
    *,
    functional: str,
    basis: str,
    solvent: str,
    nucleus: str,
    table: Optional[dict[tuple[str, str, str, str], dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """Look up a calibration entry by ``(functional, basis, solvent, nucleus)``.

    Matching is case-insensitive after stripping leading/trailing
    whitespace, so e.g. ``"wb97x-d"`` matches ``"wB97X-D"``.

    Returns ``None`` if no match — the aggregator decides whether to
    fall back to raw σ output or to surface a structured failure.
    """
    src = table if table is not None else NMR_CALIBRATION
    key = (
        str(functional).strip(),
        str(basis).strip(),
        str(solvent).strip(),
        str(nucleus).strip(),
    )
    if key in src:
        return src[key]
    lc_key = tuple(s.lower() for s in key)
    for k, v in src.items():
        if tuple(s.lower() for s in k) == lc_key:
            return v
    return None


__all__ = [
    "NMR_CALIBRATION",
    "lookup_calibration",
    "predict_chemical_shift",
    "predict_coupling_constant",
]
