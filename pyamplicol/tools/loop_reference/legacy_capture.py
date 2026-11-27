# SPDX-License-Identifier: 0BSD
"""The control set and conventions of the frozen legacy-engine oracles.

The per-diagram legacy one-loop engine was removed (user ruling 2026-10-28,
M-QCD Q3). Before that, this tool captured its outputs once as the frozen
oracles of the v2 engine (DESIGN.md section 14) below
``tests/fixtures/loop/legacy/``. The O1/O2 capture code is in the git history;
the captures of the former live comparisons (``reduction/``,
``ew_families_*``, ``gg_gg_non_default_parameters``, ``gg_ttx_model_defaults``,
``gg_gg_qcd_qed_sm``; each file states its source and precision) were taken
once, on slice 321efdb4, by the removed tests' own legacy calls, in a scratch
script that was not committed. What stays is what the frozen files are read
with:

* O1, ``<name>.json``: Born and the six virtual components (virtual, bare, uv,
  r2, uv_local, uv_external_fields) per squared coupling order and Laurent
  coefficient, at every point, in binary64 and in 200-bit arbitrary precision;
  for the controls with ``colour_projections`` also the leading- and
  next-to-leading colour projections with their colour degrees;
* O2, ``<name>.families.json.gz``: per denominator family the Born-weighted,
  helicity- and colour-summed numerator coefficients that the legacy engine
  handed to its reducer, at the first point, in both carriers;
* projections only, ``<name>.<accuracy>.json`` (``PROJECTION_CONTROLS``).

The legacy path was the generic source compiler on MG5_aMC 3.7.0 ``loop_sm``
with ``MB`` fixed to zero (the providers' ``loop_sm-no_b_mass`` scheme);
Drell-Yan used its photon-only control program (MadLoop's
``u u~ > e- e+ / z``). The binary64 captures are only accurate to the legacy
reducer's binary64 precision (up to about 9e-12 relative on the five-leg
controls): tighter comparisons take the ``arbitrary-200`` values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests/fixtures/loop/legacy"
COMPONENTS = ("virtual", "bare", "uv", "r2", "uv_local", "uv_external_fields")
# carrier label -> (arithmetic, decimal digits, reduction engine, coefficient engine);
# 60 decimal digits are exactly 200 bits (ceil(60 log2 10)).
CARRIERS = {
    "double": ("double", None, "fast", "jit"),
    "arbitrary-200": ("arbitrary", 60, "generic", "interpreted"),
}
PARAMETERS = {"aS": "0.118", "MT": "173", "MU_R": "1000"}
FIXED_PARAMETERS = {"MB": "0"}
DRELL_YAN_MODEL_VALUES = ("0.118", "132.507")  # alpha_s, 1/alpha
MODEL = "MG5_aMC 3.7.0 loop_sm, MB fixed to 0 (provider scheme loop_sm-no_b_mass)"
COLOUR_DECLARATION = ROOT / "examples/loop/source_colour/loop_sm_qcd_subset.json"
CONVENTION = (
    "MadLoop/BLHA C_eps=(4pi)^eps/Gamma(1-eps); physical couplings; initial "
    "spin/colour average and final identical-particle factor included. Values are "
    "[real, imaginary] decimal strings; Laurent coefficients are ordered "
    "finite, single pole, double pole."
)
FAMILY_CONVENTION = (
    "Legacy reducer input per family: denominators D_i = (q + V_i)^2 - m_i^2 with "
    "V_i = sum_j shifts[i][j] p_j over the external momenta p_j of the point "
    "(incoming momenta as given, positive energy); the numerator is "
    "sum_a c_a q0^a0 q1^a1 q2^a2 q3^a3 mu2^a4 eps^a5 for the monomial powers a. "
    "Coefficients are Born-weighted (Born colour metric over the initial average), "
    "helicity- and colour-summed, one row per virtual order sector; the legacy "
    "virtual is its postprocess of the summed family reductions."
)


@dataclass(frozen=True, slots=True)
class Control:
    """One control process: pdgs, Born (QCD, QED) powers, points, program kind."""

    process: str
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]
    born: tuple[int, int]
    points: str
    drell_yan: bool = False
    colour_projections: tuple[str, ...] = ()

    @property
    def born_order_bounds(self):
        return dict(zip(("QCD", "QED"), self.born, strict=True))

    @property
    def loop_order_bounds(self):
        return {"QCD": self.born[0] + 2, "QED": self.born[1]}


# Points: tools/loop_reference/<file>.json (mu_R = 1000 GeV) or the committed
# three-provider references tests/fixtures/loop/required/<name>.json (own mu_R).
CONTROLS = {
    "gg_gg": Control(
        "g g > g g",
        (21, 21),
        (21, 21),
        (2, 0),
        "gluon_scattering_qcd",
        colour_projections=("lc", "nlc"),
    ),
    "uux_ddxg": Control(
        "u u~ > d d~ g",
        (2, -2),
        (1, -1, 21),
        (3, 0),
        "required:uux_ddxg",
        colour_projections=("lc", "nlc"),
    ),
    "uux_ggg": Control(
        "u u~ > g g g",
        (2, -2),
        (21, 21, 21),
        (3, 0),
        "quark_three_gluon_qcd",
        colour_projections=("lc", "nlc"),
    ),
    "dy": Control(
        "u u~ > e- e+ / z",
        (2, -2),
        (11, -11),
        (0, 2),
        "drell_yan_qcd",
        drell_yan=True,
        colour_projections=("lc", "nlc"),
    ),
    "uudd": Control("u u~ > d d~", (2, -2), (1, -1), (2, 0), "uudd_qcd"),
    "tt": Control(
        "u u~ > t t~",
        (2, -2),
        (6, -6),
        (2, 0),
        "top_pair_qcd",
        colour_projections=("lc", "nlc"),
    ),
    "uux_uux": Control("u u~ > u u~", (2, -2), (2, -2), (2, 0), "required:uux_uux"),
    "uux_uuxg": Control(
        "u u~ > u u~ g",
        (2, -2),
        (2, -2, 21),
        (3, 0),
        "required:uux_uuxg",
        colour_projections=("lc", "nlc"),
    ),
}
# Controls whose full capture is out of reach: the legacy program of one colour
# accuracy takes over an hour at six legs (u u~ > u u~ u u~ LC: 3987 s and
# 8.4 GB, then 543 s and 22.5 GB at 200 bits, one thread, 2026-10-27), so only
# their colour projections at the first point.
PROJECTION_CONTROLS = {
    "uux_uuxuux": Control(
        "u u~ > u u~ u u~",
        (2, -2),
        (2, -2, 2, -2),
        (4, 0),
        "required:uux_uuxuux",
        colour_projections=("lc", "nlc"),
    ),
}


def control_points(control: Control) -> tuple[list, list[str]]:
    """Exact decimal points and mu_R^2 per point, as decimal strings."""
    from decimal import Decimal

    if control.points.startswith("required:"):
        name = control.points.split(":", 1)[1]
        path = ROOT / "tests/fixtures/loop/required" / f"{name}.json"
    else:
        path = ROOT / "tools/loop_reference" / f"{control.points}.json"
    data = json.loads(path.read_text(), parse_int=str, parse_float=str)
    points = data["points"]
    mu_r = data.get("mu_r", ["1000"] * len(points))
    scales = [format(Decimal(mu) ** 2, "f") for mu in mu_r]
    return points, scales
