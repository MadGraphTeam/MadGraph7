# SPDX-License-Identifier: 0BSD
"""Independent N_c oracles of the shipped loop_sm colour declaration (MQ-NC).

LC and NLC rest on the formal-N_c continuation of the counterterm and field
couplings (``models.loop_color_declarations``, design §6.7). The projection
tests (``test_loop_v2_fermion_lines``) show that LC and NLC are the kept N_c
powers of full colour as a function of N_c, which the N_c scan
(``tests/unit/loop_nc_scan.py``: full colour with N_c = N wherever the
engine substitutes it, the declaration's couplings included) evaluates. The
oracles here check that function against references that read no colour
declaration (m-qcd PLAN §3.2 MQ-NC):

* (a) NJet's ``NJetNc``: the full-colour virtual of massless five-flavour
  QCD at N_c = 2, 4, 5, 6 (``tests/fixtures/loop/nc/njet_nc_scan.json``,
  NJet 2.1.1; ``tools/loop_reference/njet/run_njet.py run --nc``) against
  the scan of the top-free request (top vetoed, MB = 0), Born and every
  Laurent coefficient within 1e-10 in binary64. Rows A01 (g g > g g) and A08
  (g g > g g g) in NJet's direct double-double mode, whose normalisation
  keeps the N_c = 3 colour average; rows A09 (u u~ > g g g), A12 (u u~ > d
  d~ g) and A13 (u u~ > u u~ g) through NJet's BLHA interface (its own
  process mapping, every point re-evaluated in double-double), whose output
  divides by the colour average at N_c: the fixture stores it times
  avg(N_c)/avg(3), the scan's N_c = 3 average (the same correction brings
  BLHA's gluon rows to the direct values within 3.2e-15).
* (c) Catani at N_c = N: the poles of the renormalised virtual against
  Catani's operator with C_A = N, C_F = (N^2-1)/(2N), beta_0 at N and the
  colour-correlated Borns at N (``loop_nc_scan``'s ``catani`` patch). The
  double pole is the loop's own; the single pole tests the UV counterterms'
  N_c dependence (delta g_s ~ beta_0, the mass and field counterterms ~ C_F,
  the heavy-quark decoupling ~ T_F): a declaration with C_A = 3 fixed fails
  it. Rows B03, C01, D05, E01, F01 (massive and electroweak Borns) at 32
  digits within 1e-24 (§1.4 rule 4's double-double band), g g > g g in
  binary64 within 1e-12 (default tier).
* (d) the gauge-Ward check at N_c = N (``loops.v2.ward.ward_check``, one
  gluon polarisation replaced by its momentum, double-double): the R2
  couplings' N_c dependence must cancel the bare loop's violation at every
  N; a declaration with N_col = 3 fixed fails it on u u~ > g g g (1.7e-4 of
  the physical scale at N_c = 2). Rows B03 and D05 (nightly; on D05 the
  R2's own Ward identity holds without the bare part, so D05 is a
  consistency check, not a discriminating one).
* (b) BDK's O14 for u u~ > g g g is ``test_loop_v2_colour.py::
  test_bdk_five_point_colour_sum_o14`` and ``test_loop_v2_observables_nightly
  .py::test_bdk_partial_amplitudes_reproduce_the_colour_sum_o14``.

Each N runs in its own process (``loop_nc_scan``, Python tree lanes), with
its own ``PYAMPLICOL_CACHE_DIR``. The other half of the campaign's rule 7
(LC and NLC equal the kept N_c powers of the scan, and Catani at N on every
nc row) is ``test_loop_v2_nc_projection.py``, whose ``NC_COLUMN`` lists
each row's ids here.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from .loop_nc_scan import exact_point
from .loop_nc_scan import request_spec as _spec
from .loop_nc_scan import scan as _scan

ROOT = Path(__file__).resolve().parents[2]
NJET = ROOT / "tests/fixtures/loop/nc/njet_nc_scan.json"
COEFFICIENTS = ("born", "finite", "single_pole", "double_pole")
OTHER_NC = ("2", "4")


# A t t~ h point (massless initial state, exact decimals; the generator has
# no construction for an unpaired massive leg): e+ e- > t t~ h's first.
TTH_POINT = json.loads(
    (ROOT / "tests/fixtures/loop/top-pair-higgs-model.json").read_text()
)["points"][0]


# A g g > b b~ point at MB = 47/10 exactly (sqrt(s) = 470.047, the b at
# |p| = 234.9765 along (3/5, 0, 4/5)): the generator's search for a massive
# pair of this mass does not end.
BBX_POINT = [
    ["235.0235", "0", "0", "235.0235"],
    ["235.0235", "0", "0", "-235.0235"],
    ["235.0235", "140.9859", "0", "187.9812"],
    ["235.0235", "-140.9859", "0", "-187.9812"],
]
POINTS = {"tth": TTH_POINT, "bbx": BBX_POINT}


def _point(masses):
    """A point of ``POINTS`` by key, or the first exact point of the
    external masses."""

    return POINTS[masses] if isinstance(masses, str) else exact_point(masses)


def _relative(a: Decimal, b: Decimal) -> Decimal:
    scale = max(abs(a), abs(b))
    return Decimal(0) if scale == 0 else abs(a - b) / scale


def _pole_errors(result) -> list[Decimal]:
    """Per predicted sector, the single and double poles' deviations from
    Catani, strictly relative per coefficient (m-qcd PLAN §1.4 rule 4)."""

    errors = []
    assert result["ir_poles"], "no Catani prediction"
    for sector, (single, double) in result["ir_poles"].items():
        virtual = result["components"]["virtual"][sector]
        predicted = (Decimal(single), Decimal(double))
        for got, want in zip((virtual[1], virtual[2]), predicted, strict=True):
            errors.append(_relative(Decimal(got), want))
    return errors


# ------------------------------------------------------------- (a) NJetNc


def _njet_errors(model, name, index, values, cache):
    reference = json.loads(NJET.read_text())["processes"][name]
    spec = _spec(
        reference["process"],
        reference["born_orders"],
        reference["points"][index],
        fixed={"MB": "0"},
        veto=(6,),
        mu_r2=str(Fraction(reference["mu_r"][index]) ** 2),
    )
    scans = _scan(model, spec, values, "double", cache)
    out = {}
    for n, result in scans.items():
        (row,) = [r for r in reference["nc"][n] if r["index"] == index]
        (virtual,) = result["components"]["virtual"].values()
        mine = [sum(Decimal(v) for v in result["born"].values())]
        mine += [Decimal(v) for v in virtual]
        out[n] = [
            _relative(a, Decimal(row[key]))
            for a, key in zip(mine, COEFFICIENTS, strict=True)
        ]
    return out


@pytest.mark.loop_fast
def test_njet_fixture_values_carry_the_n_c_3_colour_average():
    """The fixture's BLHA rows are NJet's raw values times avg(N_c)/avg(3)
    (``run_njet.colour_average`` of the BLHA process line's incoming legs:
    N_c per quark, N_c^2 - 1 per gluon); the direct rows need none."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_njet", ROOT / "tools/loop_reference/njet/run_njet.py"
    )
    run_njet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_njet)
    assert run_njet.colour_average([2, -2], Fraction(2)) == 4
    assert run_njet.colour_average([21, 21], Fraction(4)) == 225
    assert run_njet.colour_average([21, 2], Fraction(5)) == 120
    processes = json.loads(NJET.read_text())["processes"]
    blha = 0
    with localcontext() as context:
        context.prec = 40
        for name, entry in processes.items():
            options = entry["provider"]["options"]
            if options["mode"] != "blha":
                assert "colour_average" not in entry, name
                continue
            blha += 1
            legs = len(entry["points"][0])
            initial, _ = run_njet.blha_process(options["process"], legs)
            for n, rows in entry["nc"].items():
                factor = run_njet.colour_average(initial, Fraction(n))
                factor /= run_njet.colour_average(initial, Fraction(3))
                assert Fraction(entry["colour_average"][n]["factor"]) == factor
                scale = Decimal(factor.numerator) / Decimal(factor.denominator)
                for row in rows:
                    for key in COEFFICIENTS:
                        raw = (
                            Decimal(row["njet"][key])
                            if factor != 1
                            else Decimal(row[key])
                        )
                        assert _relative(Decimal(row[key]), raw * scale) < Decimal(
                            "1e-30"
                        )
    assert blha == 3


@pytest.mark.loop_physics
def test_full_colour_at_other_nc_equals_njet_on_gg_gg(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """A01 at its first point, N_c = 2, 4, 5, 6: Born and every Laurent
    coefficient of the scan equal NJet's within 1e-10 (measured <= 1.2e-14)."""

    errors = _njet_errors(
        loop_sm_compiled_model, "gg_gg", 0, ("2", "4", "5", "6"), tmp_path
    )
    assert max(max(row) for row in errors.values()) <= Decimal("1e-10"), errors


@pytest.mark.loop_nightly
@pytest.mark.parametrize("index", (0, 1))
def test_full_colour_at_other_nc_equals_njet_on_gg_ggg(
    loop_sm_compiled_model, loop_native, tmp_path, index
):
    """A08 at two points, N_c = 2, 4, 5, 6, within 1e-10 (point 0 measured
    <= 1.8e-13)."""

    errors = _njet_errors(
        loop_sm_compiled_model, "gg_ggg", index, ("2", "4", "5", "6"), tmp_path
    )
    assert max(max(row) for row in errors.values()) <= Decimal("1e-10"), errors


# Quark rows of the fixture (m-qcd PLAN §4.2 row ids): NJet's BLHA values
# at the N_c = 3 colour average, two massless exact points each.
NJET_QUARK_ROWS = {"A09": "uux_ggg", "A12": "uux_ddxg", "A13": "uux_uuxg"}


@pytest.mark.loop_physics
def test_full_colour_at_other_nc_equals_njet_on_uux_ddxg(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """A12 at its first point, N_c = 2 and 5: the scan equals NJet's
    (colour average corrected) within 1e-10 (measured <= 2.4e-13); with
    four quarks and a gluon it reaches the quark-line R2 and UV counterterms
    and the U(1)-subtraction weights at N."""

    errors = _njet_errors(loop_sm_compiled_model, "uux_ddxg", 0, ("2", "5"), tmp_path)
    assert max(max(row) for row in errors.values()) <= Decimal("1e-10"), errors


@pytest.mark.loop_nightly
@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("row", sorted(NJET_QUARK_ROWS))
def test_full_colour_at_other_nc_equals_njet_on_quark_rows(
    loop_sm_compiled_model, loop_native, tmp_path, row, index
):
    """A09, A12 and A13 at two points, N_c = 2, 4, 5, 6, within 1e-10
    (measured <= 3.5e-12, A13 point 1 at N_c = 2; the others <= 6.1e-13)."""

    errors = _njet_errors(
        loop_sm_compiled_model,
        NJET_QUARK_ROWS[row],
        index,
        ("2", "4", "5", "6"),
        tmp_path,
    )
    assert max(max(r) for r in errors.values()) <= Decimal("1e-10"), (row, errors)


# ------------------------------------------------------- (c) Catani at N


@pytest.mark.loop_physics
def test_poles_at_other_nc_equal_catani_on_gg_gg(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """g g > g g at N_c = 2 and 4 in binary64: the poles equal Catani's at N
    within 1e-12 (measured <= 4e-15); with C_A fixed to 3 in the
    declaration's identities the single pole at N_c = 2 is 46 % off."""

    spec = _spec("g g > g g", {"QCD": 2, "QED": 0}, _point([0] * 4))
    scans = _scan(loop_sm_compiled_model, spec, OTHER_NC, "double", tmp_path)
    for n, result in scans.items():
        assert max(_pole_errors(result)) <= Decimal("1e-12"), n
    control = _scan(
        loop_sm_compiled_model,
        {**spec, "identities": {"CA": "3"}},
        ("2",),
        "double",
        tmp_path / "control",
    )
    assert max(_pole_errors(control["2"])) > Decimal("0.1")


CATANI_ROWS = {
    # Row: process, Born orders, external masses, fixed parameters.
    "B03": ("g g > t t~ g", {"QCD": 3, "QED": 0}, (0, 0, 173, 173, 0), {"WT": "0"}),
    # MB = 4.7, the default restriction's card value.
    "C01": ("g g > b b~", {"QCD": 2, "QED": 0}, "bbx", {}),
    "D05": ("u u~ > e- e+ g", {"QCD": 1, "QED": 2}, (0,) * 5, {}),
    "E01": ("g g > t t~ h", {"QCD": 2, "QED": 1}, "tth", {"WT": "0"}),
    # 5FS on the default restriction (MB = ymb = 0) until MQ-5F.
    "F01": (
        "u b > d t",
        {"QCD": 0, "QED": 2},
        (0, 0, 0, 173),
        {"MB": "0", "ymb": "0", "WT": "0"},
    ),
}


@pytest.mark.loop_nightly
@pytest.mark.parametrize("row", sorted(CATANI_ROWS))
def test_poles_at_other_nc_equal_catani(
    loop_sm_compiled_model, loop_native, tmp_path, row
):
    """The nc rows' poles at N_c = 2 and 4 on the 32-digit Python path equal
    Catani's at N within 1e-24 (m-qcd PLAN §1.4 rule 4's double-double band)."""

    process, born, masses, fixed = CATANI_ROWS[row]
    spec = _spec(process, born, _point(masses), fixed=fixed)
    scans = _scan(loop_sm_compiled_model, spec, OTHER_NC, "arbitrary:32", tmp_path)
    for n, result in scans.items():
        assert max(_pole_errors(result)) <= Decimal("1e-24"), (row, n)


# --------------------------------------------------------- (d) Ward at N


def _ward(model, spec, values, cache):
    return _scan(model, {**spec, "ward": True}, values, "double-double", cache)


@pytest.mark.loop_physics
def test_ward_identity_at_other_nc_needs_the_declared_r2(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """u u~ > g g g at N_c = 2 in double-double: every gluon leg's Ward check
    passes with the shipped declaration (the virtual below 1e-25 of the
    physical scale; measured <= 3e-30) and fails with N_col fixed to 3 in its
    identities (the R2 then misses the bare violation: 1.7e-4)."""

    spec = _spec("u u~ > g g g", {"QCD": 3, "QED": 0}, _point([0] * 5))
    # The check and its control run concurrently (one process each).
    with ThreadPoolExecutor(2) as pool:
        good, bad = (
            future.result()["2"]
            for future in [
                pool.submit(_ward, loop_sm_compiled_model, spec, ("2",), tmp_path),
                pool.submit(
                    _ward,
                    loop_sm_compiled_model,
                    {**spec, "identities": {"Ncol": "3"}},
                    ("2",),
                    tmp_path / "control",
                ),
            ]
        )
    assert good["passed"] is True, good["rows"]
    assert bad["passed"] is False
    assert max(max(row["virtual"]) for row in bad["rows"]) > 1e-6


WARD_ROWS = ("B03", "D05")


@pytest.mark.loop_nightly
@pytest.mark.parametrize("row", WARD_ROWS)
def test_ward_identity_at_other_nc(loop_sm_compiled_model, loop_native, tmp_path, row):
    """B03 and D05 at N_c = 2 and 4: every gluon leg's Ward check passes in
    double-double (``ward.ward_row``)."""

    process, born, masses, fixed = CATANI_ROWS[row]
    spec = _spec(process, born, _point(masses), fixed=fixed)
    for n, result in _ward(loop_sm_compiled_model, spec, OTHER_NC, tmp_path).items():
        assert result["passed"] is True, (row, n, result["rows"])
