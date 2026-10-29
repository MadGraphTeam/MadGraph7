# SPDX-License-Identifier: 0BSD
"""The campaign's ``nc`` column: m-qcd PLAN §1.4 rule 7 (MQ-NC).

Rule 7 asks two things of every ``nc`` row: the formal projection equals the
N_c scan in double-double within max(1e-24, 10 E_DD) per coefficient, and
the MQ-NC independent oracles hold at their tolerances. ``NC_COLUMN``
names, per row, the test ids that make up its column: this file's
projection ids and the row's oracle ids in ``test_loop_v2_nc_oracles``
(NJet's NJetNc, Catani and Ward at N_c != 3) and ``test_loop_v2_colour`` /
``test_loop_v2_observables_nightly`` (BDK O14).

The formal projection is the engine's LC and NLC: ``accuracy="lc"`` and
``"nlc"`` with the shipped loop_sm colour declaration (resolved from the
model, ``resolve_loop_color_declaration``, and handed to the request; the
path where a request names no declaration is
``test_loop_v2_nc_declaration``'s), on the production native path (native
point inputs: the plan's tree program, as a stored artifact runs it). They
are the N_c^P part (LC) and the N_c^P, N_c^(P-1), N_c^(P-2) parts (NLC) of
full colour at N_c = 3, P being the leading degree of each squared-order
sector (``LoopSlice.degrees``; design §7.4-§7.6).

The N_c scan is full colour as a function of N_c (``loop_nc_scan
request``): N_c = N wherever the engine substitutes it, one process per N
(one generation for all of a row's points), the Python point inputs (trees,
Born, colour cells, counterterm and field scalars at N, in Python lanes)
dressed and reduced by the native executor at ``SCAN_DIGITS`` digits
(``"dressing": "native"``). The Python reference path's own dressing equals
it within 7e-53 at N = 2 and 1/2 on A08, A12, A13 and B03's point 0
(``test_native_dressing_of_the_scan_equals_the_python_path``), but carries a
precision-independent error at B03's point 1 (the bare loop 4e-25 off
Catani at 60, 80 and 100 digits alike, where the native carriers equal
Catani to 1e-65; below the user's ~1e-5 threshold, a HANDOFF §9 note only).
Full colour has no grade split, so the scan does
not rest on the grade bookkeeping that LC and NLC use. FC(N) = sum_p c_p
N^p, with the external colour average of N_c = 3 at every N. The window p =
P_top - span .. P_top (P_top the request's largest leading degree, span
twice the number of coloured external legs) is solved exactly from the
first span + 1 ``FIT_NODES``.

Points: the campaign's (PLAN §4.2), read from the row's binding reference
fixture (``tests/fixtures/loop/required/mqcd_references.json``): points 0,
1 and 2 at their mu_R and point 3, point 0 at mu_R / 2. The default tier
runs points 0 and 1 of the default rows (``DEFAULT_ROWS``), the nightly
tier the others (every point of the nightly rows; points 2 and 3 of the
default rows). The fixture's fixed parameters (zero widths; MB = ymb = 0,
MTA = ymtau = 0 for the 5F rows, which run on the default restriction:
'5F*' until MQ-5F) and alpha_s.

Every check is per point, per squared-order sector and per coefficient
(PLAN §1.4), Born included: rel(a, r) = |a - r| / max(|a|, |r|), strictly
relative. A value r that vanishes identically, |r| <= 1e-28 A (four orders
under the absolute band: the scan's structural zeros and the analytically
vanishing projections, e.g. the leading colour of u u~ > e- e+ g's ``uv``
finite part), is held absolutely, |a - r| <= 1e-24 A, with A = F_s =
|B_s| alpha_s / 2 pi of a component's sector (PLAN's F_s rule at rule 7's
tolerance) and A = |B_s| for a Born, the summed |B| where B_s vanishes.
Within 1e-24:

* the fit reproduces the scan at both ``CHECK_NODES`` (nodes inside the
  window's node range), and the window's two lowest powers vanish
  (|c_p 3^p| <= 1e-24 A): a power outside the window fails one or the other;
* c_p = 0 for every p above the sector's own P (the leading degrees are
  the scan's);
* on native MPFR (``AP_DIGITS``), every component (``virtual``, ``bare``,
  the counterterm and field parts): LC = c_P 3^P, NLC = sum over
  p = P-2 .. P of c_p 3^p and FC = FC(3), the projection identity itself;
* in double-double, the same for the Born and the virtual, the plan's
  wording of rule 7, in rules 4 and 6's band (coordinator ruling 2026-10-29
  ~10:15): rel <= max(1e-24, 10 E_DD) per coefficient, E_DD the
  double-double stability estimate of that coefficient with the O1 rule's
  checks (``stability.RECOMMENDED_CHECKS``: one generic exact rotation and
  the rescaled reduction check) relative to the same denominator as the
  deviation (A for a value held absolutely). The Born carries no estimate
  and is held within 1e-24. MPFR already proves the identity exact, so a
  double-double miss within 10 E_DD is double-double's own conditioning (A18:
  the full-colour finite part, about 7e-24 at points 2 and 3).

The scan's integer N other than 3 also give the poles of the renormalised
virtual at N, compared with Catani's operator at N (C_A = N, C_F = (N^2-1)
/ (2N), beta_0 and the colour-correlated Borns at N; no colour declaration
enters it) per sector and pole within 1e-24, rule 4's double-double band.
This is MQ-NC oracle (c) on every row, not only the five rows PLAN §3.2
lists.

Tiers. PLAN §4.2 marks A01, A09, A12, C01 and D05 as default rows with nc.
The default tier keeps A12 (quark lines: the U(1)-subtraction weights and
the quark-line counterterms at N) and D05 (an electroweak Born: the only
default-tier LC/NLC request reaching the declared counterterms with
electroweak legs) at points 0 and 1, about 100 s each; A01, A09 and C01 run
their projection nightly (A01 and A09 keep their default-tier NJet, Catani
and Ward oracle ids), the user's fast-default-tier rule (DECISIONS
2026-10-28 ~17:05).
"""

from __future__ import annotations

import ast
import json
from decimal import Decimal
from fractions import Fraction
from functools import cache
from pathlib import Path

import pytest

from .loop_nc_scan import request_spec, scan

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "tests/fixtures/loop/required/mqcd_references.json"
FIXTURES = ROOT / "tests/fixtures/loop"

# The nc rows of PLAN §4.2 (``mqcd_references.json`` gives each row's
# process, Born orders and binding fixture).
NC_ROWS = (
    *("A01", "A08", "A09", "A12", "A13", "A15", "A18", "A21"),
    *("B03", "C01", "D05", "E01", "F01"),
)
DEFAULT_ROWS = frozenset({"A12", "D05"})
DEFAULT_POINTS = (0, 1)
# Point 3 of every campaign fixture is point 0 at mu_R / 2.
ALL_POINTS = (0, 1, 2, 3)

# The rows' nc columns (rule 7): this file's projection ids, then the
# independent oracles' ids.
_ORACLES = "tests/unit/test_loop_v2_nc_oracles.py::"
_BDK = (
    "tests/unit/test_loop_v2_colour.py::test_bdk_five_point_colour_sum_o14",
    "tests/unit/test_loop_v2_observables_nightly.py::"
    "test_bdk_partial_amplitudes_reproduce_the_colour_sum_o14",
)
_PROJECTION = "tests/unit/test_loop_v2_nc_projection.py::"


def _njet_quark(row: str) -> tuple[str, ...]:
    return tuple(
        f"{_ORACLES}test_full_colour_at_other_nc_equals_njet_on_quark_rows"
        f"[{row}-{index}]"
        for index in (0, 1)
    )


NC_COLUMN_ORACLES = {
    "A01": (
        f"{_ORACLES}test_full_colour_at_other_nc_equals_njet_on_gg_gg",
        f"{_ORACLES}test_poles_at_other_nc_equal_catani_on_gg_gg",
    ),
    "A08": tuple(
        f"{_ORACLES}test_full_colour_at_other_nc_equals_njet_on_gg_ggg[{index}]"
        for index in (0, 1)
    ),
    "A09": (
        *_njet_quark("A09"),
        f"{_ORACLES}test_ward_identity_at_other_nc_needs_the_declared_r2",
        *_BDK,
    ),
    "A12": (
        f"{_ORACLES}test_full_colour_at_other_nc_equals_njet_on_uux_ddxg",
        *_njet_quark("A12"),
    ),
    "A13": _njet_quark("A13"),
    "A15": (),
    "A18": (),
    "A21": (),
    **{
        row: (f"{_ORACLES}test_poles_at_other_nc_equal_catani[{row}]",)
        for row in ("C01", "E01", "F01")
    },
    **{
        row: (
            f"{_ORACLES}test_poles_at_other_nc_equal_catani[{row}]",
            f"{_ORACLES}test_ward_identity_at_other_nc[{row}]",
        )
        for row in ("B03", "D05")
    },
}


def _projection_ids(row: str) -> tuple[str, ...]:
    ids = (f"{_PROJECTION}test_lc_and_nlc_are_the_kept_powers_of_the_nc_scan[{row}]",)
    if row in DEFAULT_ROWS:
        ids += (
            f"{_PROJECTION}test_lc_and_nlc_at_the_nightly_points_of_default_rows"
            f"[{row}]",
        )
    return ids


NC_COLUMN = {
    row: (*_projection_ids(row), *oracles) for row, oracles in NC_COLUMN_ORACLES.items()
}

# The scan's fit nodes are geometric about 1, which keeps the exact Laurent
# fit well conditioned; any rational N is exact on the Python reference
# path. A window of span + 1 powers takes the first span + 1 nodes. The
# span, twice the number of coloured external legs, covers the measured
# supports with room to spare: 3 powers below P for g g > g g, 4 for
# u u~ > e- e+ g, 6 for u u~ > d d~ g, 8 for u u~ > g g g, and 10 for
# u u~ > u u~ u u~ (test_loop_v2_fermion_lines). The two check nodes lie
# inside the node range of every window (it holds 1/2 .. 4 from span 4).
# On the Python reference path, 40 digits lose 17 of them on g g > b b~'s
# massive loop (6.9e-23 against 60 digits) and 60 digits leave a floor near
# 1e-31 there; the native dressing at 60 digits keeps the massless rows'
# floor below 1e-50.
FIT_NODES = (
    "3",
    "2",
    "4",
    "1",
    "8",
    "1/2",
    "16",
    "1/4",
    "32",
    "1/8",
    "64",
    "1/16",
    "128",
    "1/32",
    "256",
)
CHECK_NODES = ("3/2", "2/3")
SCAN_DIGITS = 60
AP_DIGITS = 50
TOLERANCE = Fraction(1, 10**24)
VANISHING = Fraction(1, 10**28)
BRACKET = 10
CATANI_NC = frozenset({"2", "4", "8", "16", "32"})
# alpha_s / 2 pi of F_s (a scale only: 30 digits of pi suffice).
_PI = Fraction(Decimal("3.14159265358979323846264338328"))


def _exact(value) -> Fraction:
    return Fraction(Decimal(str(value)))


@cache
def _campaign() -> dict:
    return {row["row"]: row for row in json.loads(CAMPAIGN.read_text())["rows"]}


def campaign_row(row: str) -> dict:
    """Process, Born orders, fixed parameters, alpha_s and the points
    [(point, mu_R^2)] of a campaign row (its first binding reference
    fixture)."""

    entry = _campaign()[row]
    reference = entry["references"][0]
    fixture = json.loads((FIXTURES / reference["fixture"]).read_text())
    return {
        "process": entry["process"],
        "born": dict(reference["born_orders"]),
        "fixed": dict(fixture["fixed_parameters"]),
        "alpha_s": fixture["parameters"]["aS"],
        "points": [
            (point, str(_exact(mu) ** 2))
            for point, mu in zip(fixture["points"], fixture["mu_r"], strict=True)
        ],
    }


def row_spec(row: str, indices) -> dict:
    """The ``loop_nc_scan request`` JSON of the row at its campaign points
    ``indices``."""

    data = campaign_row(row)
    spec = request_spec(
        data["process"],
        data["born"],
        None,
        fixed=data["fixed"],
        points=[list(data["points"][i]) for i in indices],
    )
    spec["parameters"] = {"aS": data["alpha_s"]}
    spec["dressing"] = "native"
    return spec


def _entries(born, components) -> dict:
    """("born", order, 0) and (component, sector, k) -> exact value (orders
    and sectors by repr)."""

    out = {("born", order, 0): _exact(v) for order, v in born.items()}
    for component, sectors in components.items():
        for sector, coefficients in sectors.items():
            for k, c in enumerate(coefficients):
                out[component, sector, k] = _exact(c)
    return out


def _result_entries(result) -> dict:
    return _entries(
        {repr(k): v for k, v in result.born.items()},
        {
            c: {repr(s): laurent.coefficients for s, laurent in sectors.items()}
            for c, sectors in result.components.items()
        },
    )


def _born_order(sector: str) -> str:
    """The Born order (repr) of a QCD-corrected sector (repr)."""

    orders = dict(ast.literal_eval(sector))
    orders["QCD"] = orders.get("QCD", 0) - 2
    return repr(tuple(sorted((k, v) for k, v in orders.items() if v)))


def absolute_scale(entries: dict, key, alpha_s: Fraction) -> Fraction:
    """A of an entry (module docstring) from the Born of ``entries``."""

    borns = {k[1]: abs(v) for k, v in entries.items() if k[0] == "born"}
    total = sum(borns.values(), Fraction(0))
    if key[0] == "born":
        return borns.get(key[1]) or total
    flux = alpha_s / (2 * _PI)
    return (borns.get(_born_order(key[1])) or total) * flux


def denominator(got: Fraction, want: Fraction, scale: Fraction) -> Fraction:
    """max(|got|, |want|), or A where ``want`` vanishes identically (|want|
    <= 1e-28 A)."""

    if abs(want) <= VANISHING * scale:
        return scale or Fraction(1)
    return max(abs(got), abs(want))


def deviation(got: Fraction, want: Fraction, scale: Fraction) -> Fraction:
    """rel(got, want), or |got - want| / A where ``want`` vanishes
    identically: within 1e-24 passes either way."""

    return abs(got - want) / denominator(got, want, scale)


def _solve(matrix, rhs):
    """The exact solution of the square system ``matrix x = rhs``."""

    rows = [[*row, b] for row, b in zip(matrix, rhs, strict=True)]
    size = len(rows)
    for column in range(size):
        pivot = next(r for r in range(column, size) if rows[r][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(size):
            if r != column and rows[r][column]:
                factor = rows[r][column] / rows[column][column]
                rows[r] = [
                    x - factor * y for x, y in zip(rows[r], rows[column], strict=True)
                ]
    return [rows[r][size] / rows[r][r] for r in range(size)]


def native_projections(compiled, native, spec) -> dict:
    """The formal projections of ``spec`` at its points: accuracy -> (slice,
    {carrier: [per point]}) for LC, NLC and FC with the model's shipped
    colour declaration on the production native path (native point inputs,
    default seed contraction): ``"dd"``, the double-double entries evaluated
    with the O1 rule's checks (``RECOMMENDED_CHECKS``), ``"e-dd"`` their
    absolute estimates per virtual key, and ``"mpfr"``, the entries in MPFR
    at ``AP_DIGITS``."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
        source_loop_color_declaration,
    )

    from .loop_nc_scan import slice_request

    assert source_loop_color_declaration(compiled.raw_loop_model) is None
    declaration = resolve_loop_color_declaration(compiled.raw_loop_model)
    assert declaration is not None
    points = [point for point, _scale in spec["points"]]
    scales = [scale for _point, scale in spec["points"]]
    out = {}
    for accuracy in ("lc", "nlc", "full"):
        slice_ = LoopSlice.generate(
            compiled, slice_request(spec, declaration, accuracy, "auto"), native
        )
        executor = NativeExecutor(
            slice_,
            forward=False,
            exported=export_plan(slice_, forward=False, trees=True, born_squared=True),
            inputs="native",
        )

        def run(precision, stability=None, executor=executor):
            return executor.evaluate(
                points,
                scales,
                parameters=spec["parameters"],
                precision=precision,
                stability=stability,
            )

        estimated = run(Precision.double_double(), RECOMMENDED_CHECKS)
        found = {
            "dd": [_result_entries(r) for r in estimated],
            "e-dd": [_absolute_estimates(r) for r in estimated],
            "mpfr": [_result_entries(r) for r in run(Precision.arbitrary(AP_DIGITS))],
        }
        out[accuracy] = (slice_, found)
    return out


def _absolute_estimates(result) -> dict:
    """("virtual", sector, k) -> the absolute double-double estimate E of
    the coefficient."""

    return {
        ("virtual", repr(sector), k): abs(_exact(e))
        for sector, budget in result.stability.absolute.items()
        for k, e in enumerate(budget.coefficients)
    }


def sector_degrees(slice_) -> dict:
    """(component or "born", sector repr) -> leading degree P."""

    degrees = slice_.degrees()
    out = {("born", repr(k)): v for k, v in degrees["born"].items()}
    out.update({("virtual", repr(k)): v for k, v in degrees["virtual"].items()})
    return out


def sector_of(key) -> tuple:
    """The degree key of an entry: Born order, or the virtual sector of any
    component."""

    return ("born", key[1]) if key[0] == "born" else ("virtual", key[1])


def fit(data: dict, span: int, top: int, alpha_s: Fraction):
    """The exact c_p (p = top - span .. top) of every entry from the first
    span + 1 ``FIT_NODES`` of ``data`` (N -> entries at one point), and the
    largest deviation of the fit from the scan at the ``CHECK_NODES``."""

    powers = list(range(top - span, top + 1))
    used = FIT_NODES[: span + 1]
    assert len(used) == span + 1, (span, len(FIT_NODES))
    keys = set().union(*data.values())
    matrix = [[Fraction(n) ** p for p in powers] for n in used]
    coefficients = {
        key: dict(
            zip(
                powers,
                _solve(matrix, [data[n].get(key, 0) for n in used]),
                strict=True,
            )
        )
        for key in keys
    }
    miss = Fraction(0)
    for n in CHECK_NODES:
        x = Fraction(n)
        for key, row in coefficients.items():
            scale = absolute_scale(data[n], key, alpha_s)
            predicted = sum(c * x**p for p, c in row.items())
            miss = max(miss, deviation(predicted, data[n].get(key, 0), scale))
    return coefficients, miss


def expected_projections(coefficients: dict, degrees: dict) -> dict:
    """accuracy -> key -> the scan's LC, NLC and FC(3)."""

    three = Fraction(3)
    out = {"lc": {}, "nlc": {}, "full": {}}
    for key, row in coefficients.items():
        degree = degrees[sector_of(key)]
        out["full"][key] = sum(c * three**p for p, c in row.items())
        for accuracy, depth in (("lc", 0), ("nlc", 2)):
            out[accuracy][key] = sum(
                c * three**p
                for p, c in row.items()
                if degree is not None and degree - depth <= p <= degree
            )
    return out


def power_errors(coefficients: dict, degrees: dict, at_three: dict, alpha_s):
    """The window's two lowest powers ("floor") and the powers above each
    sector's P ("above"): |c_p 3^p| / A, which must vanish."""

    three = Fraction(3)
    floor = min(min(row) for row in coefficients.values())
    errors = {"floor": Fraction(0), "above": Fraction(0)}
    for key, row in coefficients.items():
        scale = absolute_scale(at_three, key, alpha_s)
        degree = degrees[sector_of(key)]
        for p, c in row.items():
            term = abs(c * three**p) / scale if scale else abs(c)
            if degree is None or p > degree:
                errors["above"] = max(errors["above"], term)
            if p <= floor + 1:
                errors["floor"] = max(errors["floor"], term)
    return errors


def carrier_errors(expected: dict, mine: dict, at_three: dict, alpha_s, keys=None):
    """accuracy -> [(deviation, key, its denominator)] of ``mine`` (accuracy
    -> entries) against the scan's projections, over ``keys`` (a predicate;
    all keys by default)."""

    out = {}
    for accuracy, values in expected.items():
        rows = []
        for key in set(values) | set(mine[accuracy]):
            if keys is not None and not keys(key):
                continue
            scale = absolute_scale(at_three, key, alpha_s)
            got, want = mine[accuracy].get(key, 0), values.get(key, 0)
            size = denominator(got, want, scale)
            rows.append((abs(got - want) / size, key, size))
        out[accuracy] = rows
    return out


def dd_beyond(rows, estimates: dict) -> list[dict]:
    """The double-double coefficients of ``rows`` (``carrier_errors``) beyond
    1e-24, each with E_DD (its absolute estimate over the deviation's
    denominator; None for the Born) and whether it lies in rules 4 and 6's
    band max(1e-24, 10 E_DD)."""

    out = []
    for error, key, size in rows:
        if error <= TOLERANCE:
            continue
        estimate = estimates.get(key)
        e_dd = None if estimate is None else estimate / size
        out.append(
            {
                "key": list(key),
                "error": float(error),
                "e_dd": None if e_dd is None else float(e_dd),
                "within": e_dd is not None and error <= BRACKET * e_dd,
            }
        )
    return out


def vanishing_gap(expected: dict, at_three: dict, alpha_s) -> list:
    """[the largest nonzero |r| / A held absolutely, the smallest |r| / A held
    relatively] over the scan's projections (a diagnostic: the values that
    vanish identically lie far below 1e-28 A, the others far above)."""

    absolute, relative = Fraction(0), None
    for values in expected.values():
        for key, value in values.items():
            scale = absolute_scale(at_three, key, alpha_s)
            if not value or not scale:
                continue
            size = abs(value) / scale
            if size <= VANISHING:
                absolute = max(absolute, size)
            elif relative is None or size < relative:
                relative = size
    return [float(absolute), None if relative is None else float(relative)]


def catani_errors(scan_point: dict, alpha_s) -> Fraction:
    """The largest deviation of the poles from Catani at the integer N of
    ``CATANI_NC`` (N -> one point's scan record), per sector and pole."""

    worst = Fraction(0)
    for n, result in scan_point.items():
        if n not in CATANI_NC:
            continue
        assert result["ir_poles"], (n, "no Catani prediction")
        entries = _entries(result["born"], result["components"])
        for sector, (single, double) in result["ir_poles"].items():
            virtual = result["components"]["virtual"][sector]
            for k, want in ((1, single), (2, double)):
                scale = absolute_scale(entries, ("virtual", sector, k), alpha_s)
                worst = max(worst, deviation(_exact(virtual[k]), _exact(want), scale))
    return worst


def _born_or_virtual(key) -> bool:
    return key[0] in ("born", "virtual")


def rule_seven(model, compiled, native, row: str, indices, cache):
    """Rule 7's projection part and the scan's Catani poles of a campaign
    row at its points ``indices``: per point the largest deviation of every
    check (``TOLERANCE`` to pass; double-double in its band instead) and the
    double-double coefficients beyond 1e-24 with their E_DD."""

    indices = tuple(indices)
    spec = row_spec(row, indices)
    alpha_s = _exact(spec["parameters"]["aS"])
    found = native_projections(compiled, native, spec)
    lc = found["lc"][0]
    degrees = sector_degrees(lc)
    assert degrees == sector_degrees(found["nlc"][0])
    top = max(v for v in degrees.values() if v is not None)
    span = 2 * sum(1 for leg in lc.basis.legs if leg.representation != 1)
    values = (*FIT_NODES[: span + 1], *CHECK_NODES)
    scanned = scan(model, spec, values, f"arbitrary:{SCAN_DIGITS}", cache)
    report = {"row": row, "top": top, "span": span, "points": []}
    for j, index in enumerate(indices):
        records = {n: scanned[n]["results"][j] for n in values}
        data = {n: _entries(r["born"], r["components"]) for n, r in records.items()}
        coefficients, miss = fit(data, span, top, alpha_s)
        at_three = data["3"]
        expected = expected_projections(coefficients, degrees)
        errors = {"fit": miss, **power_errors(coefficients, degrees, at_three, alpha_s)}
        errors["catani"] = catani_errors(records, alpha_s)
        mpfr = carrier_errors(
            expected,
            {a: found[a][1]["mpfr"][j] for a in expected},
            at_three,
            alpha_s,
        )
        dd = carrier_errors(
            expected,
            {a: found[a][1]["dd"][j] for a in expected},
            at_three,
            alpha_s,
            _born_or_virtual,
        )
        beyond = []
        for accuracy in expected:
            errors[f"mpfr-{accuracy}"] = max(e for e, *_rest in mpfr[accuracy])
            errors[f"dd-{accuracy}"] = max(e for e, *_rest in dd[accuracy])
            estimates = found[accuracy][1]["e-dd"][j]
            beyond += [
                {"accuracy": accuracy, **entry}
                for entry in dd_beyond(dd[accuracy], estimates)
            ]
        report["points"].append(
            {
                "index": index,
                "errors": errors,
                "dd_beyond": beyond,
                "vanishing_gap": vanishing_gap(expected, at_three, alpha_s),
            }
        )
    return report


def verdict(report: dict) -> list[str]:
    """The failures of a ``rule_seven`` report (empty: rule 7's projection
    part holds at every point)."""

    failures = []
    for point in report["points"]:
        index = point["index"]
        for name, error in point["errors"].items():
            # Double-double is held in its band, coefficient by coefficient.
            if error <= TOLERANCE or name.startswith("dd-"):
                continue
            failures.append(f"point {index}: {name} {float(error):.3g}")
        failures += [
            f"point {index}: dd {b['accuracy']} {b['key']} {b['error']:.3g} "
            f"beyond max(1e-24, 10 E_DD), E_DD = {b['e_dd']}"
            for b in point["dd_beyond"]
            if not b["within"]
        ]
    return failures


def _printable(report: dict) -> str:
    return json.dumps(
        {
            **{k: report[k] for k in ("row", "top", "span")},
            "points": [
                {
                    **{k: p[k] for k in ("index", "dd_beyond", "vanishing_gap")},
                    **{k: float(v) for k, v in p["errors"].items()},
                }
                for p in report["points"]
            ],
        }
    )


@pytest.mark.loop_fast
def test_nc_column_names_the_plan_rows_and_existing_oracles():
    """Every nc row of PLAN §4.2 has a column, the column's first id is this
    file's, every oracle id names a test of the oracle files (the
    parametrised ids among the declared parameters), and every row has a
    campaign fixture with the four points (point 3 = point 0 at mu_R / 2)."""

    from . import test_loop_v2_nc_oracles as oracles

    assert set(NC_COLUMN) == set(NC_ROWS) == set(NC_COLUMN_ORACLES)
    assert set(NC_ROWS) >= DEFAULT_ROWS
    parameters = {
        "test_full_colour_at_other_nc_equals_njet_on_gg_ggg": {"0", "1"},
        "test_full_colour_at_other_nc_equals_njet_on_quark_rows": {
            f"{row}-{index}" for row in oracles.NJET_QUARK_ROWS for index in (0, 1)
        },
        "test_poles_at_other_nc_equal_catani": set(oracles.CATANI_ROWS),
        "test_ward_identity_at_other_nc": set(oracles.WARD_ROWS),
    }
    for row, ids in NC_COLUMN.items():
        assert ids[0].endswith(f"_scan[{row}]"), row
        projection = [i for i in ids if i.startswith(_PROJECTION)]
        assert len(projection) == (2 if row in DEFAULT_ROWS else 1), row
        for test_id in ids[len(projection) :]:
            path, name = test_id.split("::")
            if path.endswith("test_loop_v2_nc_oracles.py"):
                base, _, parameter = name.partition("[")
                assert callable(getattr(oracles, base)), test_id
                if parameter:
                    assert parameter[:-1] in parameters[base], test_id
            else:
                assert test_id in _BDK, test_id
        data = campaign_row(row)
        assert len(data["points"]) == len(ALL_POINTS), row
        (p0, mu0), (p3, mu3) = data["points"][0], data["points"][3]
        assert p3 == p0 and _exact(mu3) * 4 == _exact(mu0), row


def _run(model, native, tmp_path, row, indices):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(model)
    report = rule_seven(model, compiled, native, row, indices, tmp_path)
    # The measured deviations, one JSON line (shown with -s or -rP).
    print(_printable(report))
    failures = verdict(report)
    assert not failures, (row, failures)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            row,
            marks=pytest.mark.loop_physics
            if row in DEFAULT_ROWS
            else pytest.mark.loop_nightly,
        )
        for row in sorted(NC_ROWS)
    ],
)
def test_lc_and_nlc_are_the_kept_powers_of_the_nc_scan(
    loop_sm_compiled_model, loop_native, tmp_path, row
):
    """Rule 7 on the row (module docstring) at its campaign points (default
    rows: points 0 and 1; nightly rows: all four): LC, NLC and FC on native
    MPFR and in double-double against the exact N_c fit of the scan, the
    fit's own checks, the leading degrees, and Catani at N, each per
    coefficient within 1e-24 (double-double within max(1e-24, 10 E_DD))."""

    indices = DEFAULT_POINTS if row in DEFAULT_ROWS else ALL_POINTS
    _run(loop_sm_compiled_model, loop_native, tmp_path, row, indices)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("row", sorted(DEFAULT_ROWS))
def test_lc_and_nlc_at_the_nightly_points_of_default_rows(
    loop_sm_compiled_model, loop_native, tmp_path, row
):
    """Rule 7 at points 2 and 3 (point 0 at mu_R / 2) of the default rows."""

    indices = tuple(i for i in ALL_POINTS if i not in DEFAULT_POINTS)
    _run(loop_sm_compiled_model, loop_native, tmp_path, row, indices)


@pytest.mark.loop_nightly
def test_native_dressing_of_the_scan_equals_the_python_path(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """The scan's native dressing of the Python point inputs equals the
    Python reference path's own dressing at N_c = 2 and 1/2 (A12 at its
    campaign points 0 and 1: U(1)-subtraction weights, quark-line
    counterterms and colour cells at N): every component and Catani's poles
    within 1e-50 (measured <= 3e-56)."""

    spec = row_spec("A12", DEFAULT_POINTS)
    runs = {
        dressing: scan(
            loop_sm_compiled_model,
            {**spec, "dressing": dressing},
            ("2", "1/2"),
            f"arbitrary:{SCAN_DIGITS}",
            tmp_path / dressing,
        )
        for dressing in ("python", "native")
    }
    worst = Fraction(0)
    for n, python in runs["python"].items():
        native = runs["native"][n]
        assert python["rescaled"] == native["rescaled"] > 0, n
        for mine, theirs in zip(python["results"], native["results"], strict=True):
            a = _entries(mine["born"], mine["components"])
            b = _entries(theirs["born"], theirs["components"])
            assert set(a) == set(b), n
            for key, value in a.items():
                size = max(abs(value), abs(b[key]))
                worst = max(worst, abs(value - b[key]) / size if size else 0)
            assert mine["ir_poles"].keys() == theirs["ir_poles"].keys(), n
            for sector, poles in mine["ir_poles"].items():
                for x, y in zip(poles, theirs["ir_poles"][sector], strict=True):
                    x, y = _exact(x), _exact(y)
                    worst = max(worst, abs(x - y) / max(abs(x), abs(y)))
    assert worst <= Fraction(1, 10**50), float(worst)
