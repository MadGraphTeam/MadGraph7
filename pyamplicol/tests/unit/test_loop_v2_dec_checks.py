# SPDX-License-Identifier: 0BSD
"""One-loop decays (M-FINAL MF-DEC-2): the internal checks beside the
MadLoop rows of ``tests/fixtures/loop/dec/manifest.json`` (m-final/PLAN.md
§2.2 "Further checks"; the rows' delegated columns).

* ``ff``: z > u u~ [QCD], the time-like massless quark form factor: per Born
  V/B = (alpha_s / 2 pi) C_F (-2/eps^2 - (3 + 2L)/eps - 8 + pi^2 - L^2 - 3L),
  L = log(mu_R^2 / M_Z^2), in the (4 pi)^eps / Gamma(1 - eps) normalisation
  ('t Hooft-Veltman; massless quarks need no field renormalisation), at
  XX04's rest-frame points (mu_R = M_Z and M_Z / 2), double-double within
  max(1e-24, 10 E_DD), strictly relative (as ``mqcd.rule_ff`` for D01's
  incoming pair: the pair is outgoing here, the invariant the same).
* ``nc``: LC and NLC of the loop_sm QCD decays XX01 (t > b w+) and XX07
  (t > b w+ g) are the kept powers of the full-colour N_c scan, MQ-NC's rule
  7 (``test_loop_v2_nc_projection``: native MPFR and double-double against
  the exact N_c fit, the fit's own checks, the leading degrees, Catani at
  integer N), at the decay references' points.

The Ward identity on XX07's external gluon is the row's ``ward`` extra (the
nightly row test).
"""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from tools.loop_benchmark import mqcd

from . import test_loop_v2_nc_projection as nc
from .loop_nc_scan import request_spec, scan

ROOT = Path(__file__).resolve().parents[2]
DEC = ROOT / "tests/fixtures/loop/required/dec"
MANIFEST = ROOT / "tests/fixtures/loop/dec/manifest.json"

# XX04's request with QCD corrections only, on loop_sm (5F): its points,
# mu_R and parameters (MW is loop_sm's derived parameter).
FF_ROW = {
    "id": "XX04@ff",
    "class": "D-QCD",
    "process": "z > u u~",
    "born_orders": {"QCD": 0, "QED": 1},
    "loop_orders": {"QCD": 2, "QED": 1},
    "corrections": ["QCD"],
    "restriction": "no_b_mass",
    "restriction_code": "5F",
    "fixed_parameters": {
        "MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0",
        "WT": "0", "WW": "0", "WZ": "0", "WH": "0",
    },
    "ignored_parameters": ["MW"],
    "tier": "default",
    "status": "active",
    "reference": {
        "format": "required",
        "path": "tests/fixtures/loop/required/dec/z_uux_ew.json",
        "providers": ["madloop"],
    },
    "points": {"default": [0, 3]},
    "predicted_sectors": [{"QCD": 2, "QED": 2}],
}  # fmt: skip
FF_POINTS = (0, 1, 2, 3)
# The nc rows: manifest row -> its reference (the scan's points, fixed
# parameters and parameters).
NC_ROWS = {"XX01": "t_bwp_qcd", "XX07": "t_bwpg_qcd"}
NC_DEFAULT = {"XX01": (0, 1)}
ALL_POINTS = (0, 1, 2, 3)


def _ff_checks(evaluated, point, mu2: str, alpha_s: Decimal) -> list[mqcd.Check]:
    """The form factor's three coefficients of one double-double point."""

    ((sector, laurent),) = evaluated.virtual.items()
    born = evaluated.total()[0]
    q, qbar = ([Fraction(x) for x in leg] for leg in point[1:3])
    total = [x + y for x, y in zip(q, qbar, strict=True)]
    s = total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2
    with localcontext() as context:
        context.prec = mqcd.TREE_DIGITS
        log = mqcd._decimal(Fraction(mu2) / s).ln()
        factor = alpha_s / (2 * mqcd.PI) * mqcd._decimal(mqcd.CF) * born
        expected = (
            factor * (-8 + mqcd.PI * mqcd.PI - log * log - 3 * log),
            factor * (-3 - 2 * log),
            factor * -2,
        )
    checks = []
    estimate = evaluated.estimate[sector]
    for k, name in enumerate(mqcd.LAURENT):
        error = mqcd.relative(laurent[k], expected[k])
        e_rel = estimate[k] / abs(float(laurent[k]))
        bound = max(mqcd.DD_FLOOR, mqcd.BRACKET * e_rel)
        checks.append(
            mqcd.Check(
                "ff", None, evaluated.carrier, mqcd.sector_text(sector), name,
                str(laurent[k]), str(expected[k]), error, bound, error <= bound,
                note=f"time-like quark form factor; E_DD = {e_rel:.3g}",
            )
        )  # fmt: skip
    return checks


@pytest.mark.loop_physics
def test_z_to_u_ubar_qcd_is_the_massless_quark_form_factor(
    loop_sm_ufo, loop_native, tmp_path
):
    """``ff`` (module docstring) at XX04's four rest-frame points."""

    campaign = mqcd.Campaign(
        loop_sm_ufo, loop_native, tmp_path, manifest={"rows": [FF_ROW]}
    )
    run = campaign.run(FF_ROW["id"])
    reference = run.reference
    alpha_s = Decimal(reference.parameters["aS"])
    checks = []
    for index, evaluated in zip(
        FF_POINTS, run.evaluate(FF_POINTS, "double-double"), strict=True
    ):
        checks += _ff_checks(
            evaluated, reference.points[index], reference.scales[index], alpha_s
        )
    failures = [c for c in checks if not c.passed]
    assert len(checks) == 3 * len(FF_POINTS) and not failures, failures
    # The points hold both scales: L = 0 and L = log(1/4).
    assert len({reference.scales[i] for i in FF_POINTS}) == 2


def nc_spec(row: str, indices) -> dict:
    """The ``loop_nc_scan request`` JSON of a decay row at its reference's
    points ``indices`` (its process, Born orders, fixed parameters and
    runtime parameters)."""

    fixture = json.loads((DEC / f"{NC_ROWS[row]}.json").read_text())
    points = [
        (fixture["points"][i], str(Fraction(fixture["mu_r"][i]) ** 2))
        for i in indices
    ]
    born = {"QCD": 0, **fixture["born_orders"]}
    spec = request_spec(
        " ".join(t for t in fixture["process"].split("[")[0].split() if "=" not in t),
        born,
        None,
        fixed=dict(fixture["fixed_parameters"]),
        points=[list(p) for p in points],
    )
    spec["parameters"] = dict(fixture["parameters"])
    spec["dressing"] = "native"
    return spec


def decay_rule_seven(model, compiled, native, row: str, indices, cache) -> dict:
    """``test_loop_v2_nc_projection.rule_seven`` on a decay row's spec."""

    indices = tuple(indices)
    spec = nc_spec(row, indices)
    alpha_s = nc._exact(spec["parameters"]["aS"])
    found = nc.native_projections(compiled, native, spec)
    lc = found["lc"][0]
    degrees = nc.sector_degrees(lc)
    assert degrees == nc.sector_degrees(found["nlc"][0])
    top = max(v for v in degrees.values() if v is not None)
    span = 2 * sum(1 for leg in lc.basis.legs if leg.representation != 1)
    values = (*nc.FIT_NODES[: span + 1], *nc.CHECK_NODES)
    scanned = scan(model, spec, values, f"arbitrary:{nc.SCAN_DIGITS}", cache)
    report = {"row": row, "top": top, "span": span, "points": []}
    for j, index in enumerate(indices):
        records = {n: scanned[n]["results"][j] for n in values}
        data = {n: nc._entries(r["born"], r["components"]) for n, r in records.items()}
        coefficients, miss = nc.fit(data, span, top, alpha_s)
        at_three = data["3"]
        expected = nc.expected_projections(coefficients, degrees)
        errors = {
            "fit": miss,
            **nc.power_errors(coefficients, degrees, at_three, alpha_s),
        }
        errors["catani"] = nc.catani_errors(records, alpha_s)
        mpfr = nc.carrier_errors(
            expected, {a: found[a][1]["mpfr"][j] for a in expected}, at_three, alpha_s
        )
        dd = nc.carrier_errors(
            expected,
            {a: found[a][1]["dd"][j] for a in expected},
            at_three,
            alpha_s,
            nc._born_or_virtual,
        )
        beyond = []
        for accuracy in expected:
            errors[f"mpfr-{accuracy}"] = max(e for e, *_rest in mpfr[accuracy])
            errors[f"dd-{accuracy}"] = max(e for e, *_rest in dd[accuracy])
            estimates = found[accuracy][1]["e-dd"][j]
            beyond += [
                {"accuracy": accuracy, **entry}
                for entry in nc.dd_beyond(dd[accuracy], estimates)
            ]
        lc_mpfr, full_mpfr = found["lc"][1]["mpfr"][j], found["full"][1]["mpfr"][j]
        report["points"].append(
            {
                "index": index,
                "errors": errors,
                "dd_beyond": beyond,
                "vanishing_gap": nc.vanishing_gap(expected, at_three, alpha_s),
                # The largest relative LC - FC difference of the virtual: LC
                # drops the subleading powers (C_F's -1/(2 N_c) at least), so
                # the projections are not FC.
                "lc_vs_full": float(
                    max(
                        abs(lc_mpfr.get(key, 0) - value) / abs(value)
                        for key, value in full_mpfr.items()
                        if key[0] == "virtual" and value
                    )
                ),
            }
        )
    return report


def _nc_run(model, native, tmp_path, row, indices):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(model)
    report = decay_rule_seven(model, compiled, native, row, indices, tmp_path)
    print(nc._printable(report), [p["lc_vs_full"] for p in report["points"]])
    failures = nc.verdict(report)
    assert not failures, (row, failures)
    assert all(point["lc_vs_full"] > 1e-3 for point in report["points"]), report


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            row,
            marks=pytest.mark.loop_physics
            if row in NC_DEFAULT
            else pytest.mark.loop_nightly,
        )
        for row in sorted(NC_ROWS)
    ],
)
def test_decay_lc_and_nlc_are_the_kept_powers_of_the_nc_scan(
    loop_sm_compiled_model, loop_native, tmp_path, row
):
    """Rule 7 (module docstring ``nc``) on the QCD decay row at its points
    (XX01: points 0 and 1; XX07: all four)."""

    _nc_run(
        loop_sm_compiled_model,
        loop_native,
        tmp_path,
        row,
        NC_DEFAULT.get(row, ALL_POINTS),
    )


@pytest.mark.loop_nightly
@pytest.mark.parametrize("row", sorted(NC_DEFAULT))
def test_decay_lc_and_nlc_at_the_nightly_points(
    loop_sm_compiled_model, loop_native, tmp_path, row
):
    """Rule 7 at the other points of the default nc rows (point 3: point 1
    at mu_R = M / 2)."""

    indices = tuple(i for i in ALL_POINTS if i not in NC_DEFAULT[row])
    _nc_run(loop_sm_compiled_model, loop_native, tmp_path, row, indices)


@pytest.mark.loop_fast
def test_delegated_columns_name_this_file():
    """The manifest's delegated columns of the decay rows name this file's
    tests (``nc`` on XX01 and XX07, ``ff`` on XX04)."""

    rows = mqcd.rows(mqcd.load_manifest(MANIFEST))
    here = "tests/unit/test_loop_v2_dec_checks.py::"
    columns = {
        name: dict(row.delegated) for name, row in rows.items() if row.delegated
    }
    assert columns == {
        "XX01": {
            "nc": (
                f"{here}test_decay_lc_and_nlc_are_the_kept_powers_of_the_nc_scan[XX01]",
                f"{here}test_decay_lc_and_nlc_at_the_nightly_points[XX01]",
            )
        },
        "XX04": {"ff": (f"{here}test_z_to_u_ubar_qcd_is_the_massless_quark_form_factor",)},
        "XX07": {
            "nc": (
                f"{here}test_decay_lc_and_nlc_are_the_kept_powers_of_the_nc_scan[XX07]",
            )
        },
    }  # fmt: skip
