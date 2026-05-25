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
        "valid_range_ppm": (-2.0, 14.0),
    },
    ("WP04", "6-311++G(2d,p)", "PCM_CHCl3_DELTA50", "1H"): {
        "slope": -1.0311,
        "intercept": 32.2654,
        "source": "DELTA50 high-accuracy: GIAO-PCM-WP04/6-311++G(2d,p)//PCM-B3LYP-D3/6-311G(d,p)",
        "valid_range_ppm": (-2.0, 14.0),
    },
    # ¹³C default: lab-fitted recalibration of cheshire's slope/intercept
    # against organopalladium experimental data (R²=0.9986, RMSE=1.51 ppm
    # on the original fit set). Composed from the cheshire baseline
    # (slope=-1.0501, intercept=187.25) plus the residual linear fit
    # δ_exp ≈ 1.073598 · δ_pred + 10.651605:
    #
    #     slope_new     = -1.0501 / 1.073598            = -0.9781
    #     intercept_new = 187.25  - slope_new × 10.6516 = 197.67
    #
    # Replaces the cheshire default, which had systematic affine bias on
    # this lab's Pd-containing test set. To force the original cheshire
    # row, override solvent to e.g. ``CHCl3_CHESHIRE_2006`` (same
    # solvent-token-as-discriminator trick used for the DELTA50 ¹H entry).
    ("wB97X-D", "6-31G(d,p)", "CHCl3", "13C"): {
        "slope": -0.9781,
        "intercept": 197.67,
        "source": "labounad 2026 (organopd recalibration; baseline=cheshire)",
        "valid_range_ppm": (-20.0, 240.0),
    },
    ("mPW1PW91", "pcJ-2", "CHCl3", "1H-1H_J"): {
        "slope": 0.9105,
        "intercept": 0.21,
        "source": "Bally Rablen 2011 (J. Org. Chem. 76, 4818)",
        "valid_range_ppm": None,
    },
    # ----------------------------------------------------------------
    # Heteronuclear J-coupling calibrations.
    #
    # The published references for ¹H-¹⁹F, ¹³C-¹⁹F, and ¹⁹F-¹⁹F
    # couplings on mPW1PW91/pcJ-2 are sparse / sample-dependent
    # (Bally/Rablen explicitly only fit ¹H-¹H), so we ship identity
    # scaling (slope=1.0, intercept=0.0) as a reasonable starting
    # point. Operators with a published coefficient pair should
    # override these via the calibration_overrides config knob.
    # TODO(labounad): collect literature references (Bagno's mPW1PW91
    # ¹H-¹⁹F work is a candidate) and replace identity entries.
    # ----------------------------------------------------------------
    ("mPW1PW91", "pcJ-2", "CHCl3", "1H-19F_J"): {
        "slope": 1.0,
        "intercept": 0.0,
        "source": "identity (no published mPW1PW91/pcJ-2 ¹H-¹⁹F calibration)",
        "valid_range_ppm": None,
    },
    ("mPW1PW91", "pcJ-2", "CHCl3", "19F-19F_J"): {
        "slope": 1.0,
        "intercept": 0.0,
        "source": "identity (no published mPW1PW91/pcJ-2 ¹⁹F-¹⁹F calibration)",
        "valid_range_ppm": None,
    },
    ("mPW1PW91", "pcJ-2", "CHCl3", "1H-13C_J"): {
        "slope": 1.0,
        "intercept": 0.0,
        "source": "identity (no published mPW1PW91/pcJ-2 ¹H-¹³C calibration)",
        "valid_range_ppm": None,
    },
}


# --------------------------------------------------------------------
# Canonical isotope notation for J-coupling calibration lookups
# --------------------------------------------------------------------


#: Default isotope label per element. NMR coupling calibrations are
#: indexed by (functional, basis, solvent, "<isoA>-<isoB>_J") where
#: <isoA>/<isoB> are the natural-abundance major isotopes for the
#: chemistry the lab cares about. ¹⁴N is the natural-abundance major
#: but most NMR work uses ¹⁵N (enriched), so we default to ¹⁵N for
#: lookups; the operator can override by passing an explicit nucleus
#: string to :func:`lookup_calibration`.
_DEFAULT_ISOTOPE_BY_ELEMENT: dict[str, str] = {
    "H": "1H",
    "C": "13C",
    "N": "15N",
    "O": "17O",
    "F": "19F",
    "P": "31P",
    "Si": "29Si",
}

#: Mass number used for ordering pair labels (lighter isotope first).
#: This convention makes ``1H-19F`` canonical regardless of which
#: side ORCA happens to print first in its coupling output.
_MASS_BY_ELEMENT: dict[str, int] = {
    "H": 1,
    "C": 13,
    "N": 15,
    "O": 17,
    "F": 19,
    "P": 31,
    "Si": 29,
}


def canonical_j_nucleus_label(elem_a: str, elem_b: str) -> str:
    """Return the calibration-table nucleus label for an element pair.

    Maps element symbols (e.g. ``"H"``, ``"F"``) to the corresponding
    isotope-based label used in :data:`NMR_CALIBRATION` keys (``"1H"``,
    ``"19F"``), then joins them lighter-first as ``"<isoA>-<isoB>_J"``.

    Examples::

        canonical_j_nucleus_label("H", "H")  -> "1H-1H_J"
        canonical_j_nucleus_label("H", "F")  -> "1H-19F_J"
        canonical_j_nucleus_label("F", "H")  -> "1H-19F_J"
        canonical_j_nucleus_label("C", "F")  -> "13C-19F_J"

    Unknown elements pass through to the bare element symbol with a
    ``?`` mass marker (so a typo is visible in the manifest rather
    than silently looking up the wrong calibration). The lookup will
    then miss and the caller can decide whether to fall back to
    identity or fail.
    """
    iso_a = _DEFAULT_ISOTOPE_BY_ELEMENT.get(elem_a, f"?{elem_a}")
    iso_b = _DEFAULT_ISOTOPE_BY_ELEMENT.get(elem_b, f"?{elem_b}")
    mass_a = _MASS_BY_ELEMENT.get(elem_a, 999)
    mass_b = _MASS_BY_ELEMENT.get(elem_b, 999)
    if mass_a > mass_b:
        iso_a, iso_b = iso_b, iso_a
    return f"{iso_a}-{iso_b}_J"


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
    "canonical_j_nucleus_label",
    "lookup_calibration",
    "predict_chemical_shift",
    "predict_coupling_constant",
]
