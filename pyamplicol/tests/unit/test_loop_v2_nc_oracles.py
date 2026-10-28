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
  NJet 2.1.1 in double-double; ``tools/loop_reference/njet/run_njet.py run
  --nc``) against the scan of the top-free request (top vetoed, MB = 0), Born
  and every Laurent coefficient within 1e-10 in binary64. Both keep the
  N_c = 3 colour average, so no correction enters. Rows A01 (g g > g g) and
  A08 (g g > g g g); the quark rows need NJet's quark amplitudes in the
  driver (open).
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
its own ``PYAMPLICOL_CACHE_DIR``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NJET = ROOT / "tests/fixtures/loop/nc/njet_nc_scan.json"
COEFFICIENTS = ("born", "finite", "single_pole", "double_pole")
OTHER_NC = ("2", "4")


def _point(masses):
    from tools.loop_reference import exact_points

    point = exact_points.generate(list(masses), count=1)[0]
    return [[exact_points.decimal_string(x) for x in p] for p in point]


def _spec(process, born, point, *, fixed=None, veto=(), mu_r2="1000000", **extra):
    return {
        "process": process,
        "born": born,
        "loop": {**born, "QCD": born["QCD"] + 2},
        "fixed_parameters": dict(fixed or {}),
        "particle_veto": list(veto),
        "point": point,
        "mu_r2": mu_r2,
        "parameters": {"aS": "0.118"},
        **extra,
    }


def _scan(model, spec, values, precision, cache) -> dict[str, dict]:
    """``loop_nc_scan request`` of ``spec`` at each N of ``values`` (one
    process each, concurrently): N -> its JSON."""

    def run(n):
        environment = dict(
            os.environ,
            PYAMPLICOL_CACHE_DIR=str(cache / f"nc-{n.replace('/', '_')}"),
        )
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.unit.loop_nc_scan",
                "request",
                str(model),
                json.dumps(spec),
                n,
                precision,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert process.returncode == 0, (n, process.stderr[-4000:])
        return json.loads(process.stdout.strip().splitlines()[-1])

    with ThreadPoolExecutor(len(values)) as pool:
        return dict(zip(values, pool.map(run, values), strict=True))


def _relative(a: Decimal, b: Decimal) -> Decimal:
    scale = max(abs(a), abs(b))
    return Decimal(0) if scale == 0 else abs(a - b) / scale


def _pole_errors(result) -> list[Decimal]:
    """Per predicted sector, the single and double poles' deviations from
    Catani, relative to the larger prediction."""

    errors = []
    assert result["ir_poles"], "no Catani prediction"
    for sector, (single, double) in result["ir_poles"].items():
        virtual = result["components"]["virtual"][sector]
        predicted = (Decimal(single), Decimal(double))
        scale = max(map(abs, predicted))
        for got, want in zip((virtual[1], virtual[2]), predicted, strict=True):
            errors.append(abs(Decimal(got) - want) / scale)
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
    # MB = 4.7 exactly (the default restriction's card value).
    "C01": (
        "g g > b b~",
        {"QCD": 2, "QED": 0},
        (0, 0, Fraction(47, 10), Fraction(47, 10)),
        {},
    ),
    "D05": ("u u~ > e- e+ g", {"QCD": 1, "QED": 2}, (0,) * 5, {}),
    "E01": ("g g > t t~ h", {"QCD": 2, "QED": 1}, (0, 0, 173, 173, 125), {"WT": "0"}),
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
    good = _ward(loop_sm_compiled_model, spec, ("2",), tmp_path)["2"]
    assert good["passed"] is True, good["rows"]
    bad = _ward(
        loop_sm_compiled_model,
        {**spec, "identities": {"Ncol": "3"}},
        ("2",),
        tmp_path / "control",
    )["2"]
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
