# SPDX-License-Identifier: 0BSD
"""The campaign's ``nc`` column: m-qcd PLAN §1.4 rule 7 (MQ-NC).

Rule 7 asks two things of every ``nc`` row: the formal projection equals the
N_c scan within 1e-24 in double-double, and the MQ-NC independent oracles
hold at their tolerances. ``NC_COLUMN`` names, per row, the test ids that
make up its column: this file's projection id and the row's oracle ids in
``test_loop_v2_nc_oracles`` (NJet's NJetNc, Catani and Ward at N_c != 3)
and ``test_loop_v2_colour`` / ``test_loop_v2_observables_nightly`` (BDK
O14).

The formal projection is the engine's LC and NLC: ``accuracy="lc"`` and
``"nlc"`` with the shipped loop_sm colour declaration (the request names
none), the production native path, double-double. They are the N_c^P part
(LC) and the N_c^P, N_c^(P-1), N_c^(P-2) parts (NLC) of full colour at
N_c = 3, P being the leading degree of each squared-order sector
(``LoopSlice.degrees``; design §7.4-§7.6).

The N_c scan is full colour as a function of N_c (``loop_nc_scan
request``): N_c = N wherever the engine substitutes it, one process per N,
on the Python reference path at ``SCAN_DIGITS`` digits. Full colour has no
grade split, so the scan does not rest on the grade bookkeeping that LC and
NLC use. FC(N) = sum_p c_p N^p, with the external colour average of
N_c = 3 at every N. The window p = P_top - span .. P_top (P_top the
request's largest leading degree, span twice the number of coloured
external legs) is solved exactly from the first span + 1 ``FIT_NODES``.
Then, per Born order and per component, sector and Laurent coefficient,
within 1e-24 of the sector's largest term |c_p 3^p|:

* the fit reproduces the scan at both ``CHECK_NODES`` (nodes inside the
  window's node range), and the window's two lowest powers vanish: a power
  outside the window fails one or the other;
* c_p = 0 for every p above the sector's own P (the leading degrees are
  the scan's);
* LC = c_P 3^P and NLC = sum over p = P-2 .. P of c_p 3^p;
* the native double-double full colour equals the scan's FC(3).

The scan's integer N other than 3 also give the poles of the renormalised
virtual at N. They are compared with Catani's operator at N (C_A = N,
C_F = (N^2-1)/(2N), beta_0 and the colour-correlated Borns at N; no colour
declaration enters it) within 1e-24, rule 4's double-double band. This is
MQ-NC oracle (c) on every row, not only the five rows PLAN §3.2 lists.

Rows (PLAN §4.2), one exact point each. The A rows run on the default
restriction with a massless b (MB = ymb = 0) and zero widths ('5F*'); the
others are configured as ``test_loop_v2_nc_oracles.CATANI_ROWS``. Default
tier: A12 and D05. The others are nightly: 50-90 s per 2->2/2->3 row, and
2->4 scan processes of 1-6 min each.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from .test_loop_v2_nc_oracles import CATANI_ROWS, _point, _scan, _spec

FIVE_F = {"MB": "0", "ymb": "0", "WT": "0", "WZ": "0", "WW": "0", "WH": "0"}

# Row: process, Born orders, external masses (or a point key of
# ``test_loop_v2_nc_oracles.POINTS``), fixed parameters.
NC_ROWS = {
    "A01": ("g g > g g", {"QCD": 2, "QED": 0}, (0,) * 4, FIVE_F),
    "A08": ("g g > g g g", {"QCD": 3, "QED": 0}, (0,) * 5, FIVE_F),
    "A09": ("u u~ > g g g", {"QCD": 3, "QED": 0}, (0,) * 5, FIVE_F),
    "A12": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, (0,) * 5, FIVE_F),
    "A13": ("u u~ > u u~ g", {"QCD": 3, "QED": 0}, (0,) * 5, FIVE_F),
    "A15": ("u u~ > g g g g", {"QCD": 4, "QED": 0}, (0,) * 6, FIVE_F),
    "A18": ("g g > u u~ u u~", {"QCD": 4, "QED": 0}, (0,) * 6, FIVE_F),
    "A21": ("u u~ > u u~ u u~", {"QCD": 4, "QED": 0}, (0,) * 6, FIVE_F),
    **CATANI_ROWS,
}
DEFAULT_ROWS = frozenset({"A12", "D05"})

# The rows' nc columns (rule 7): this file's projection id, then the
# independent oracles' ids.
_ORACLES = "tests/unit/test_loop_v2_nc_oracles.py::"
_BDK = (
    "tests/unit/test_loop_v2_colour.py::test_bdk_five_point_colour_sum_o14",
    "tests/unit/test_loop_v2_observables_nightly.py::"
    "test_bdk_partial_amplitudes_reproduce_the_colour_sum_o14",
)


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
NC_COLUMN = {
    row: (
        "tests/unit/test_loop_v2_nc_projection.py::"
        f"test_lc_and_nlc_are_the_kept_powers_of_the_nc_scan[{row}]",
        *oracles,
    )
    for row, oracles in NC_COLUMN_ORACLES.items()
}

# The scan's fit nodes are geometric about 1, which keeps the exact Laurent
# fit well conditioned; any rational N is exact on the Python reference
# path. A window of span + 1 powers takes the first span + 1 nodes. The
# span, twice the number of coloured external legs, covers the measured
# supports with room to spare: 3 powers below P for g g > g g, 4 for
# u u~ > e- e+ g, 6 for u u~ > d d~ g, 8 for u u~ > g g g, and 10 for
# u u~ > u u~ u u~ (test_loop_v2_fermion_lines). The two check nodes lie
# inside the node range of every window (it holds 1/2 .. 4 from span 4).
# 40 digits lose 17 of them on g g > b b~'s massive loop (6.9e-23 against
# 60 digits); 60 digits leave a floor near 1e-31 there and below 1e-50 on
# the massless rows.
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
TOLERANCE = Fraction(1, 10**24)
CATANI_NC = frozenset({"2", "4", "8", "16", "32"})


def _exact(value) -> Fraction:
    return Fraction(Decimal(str(value)))


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


def row_spec(row: str) -> dict:
    process, born, masses, fixed = NC_ROWS[row]
    return _spec(process, born, _point(masses), fixed=fixed)


def projections(compiled, native, spec) -> dict:
    """The formal projections of ``spec``: accuracy -> (slice, exact entries)
    for LC, NLC and FC with the model's (shipped) colour declaration, on the
    production native path (default seed contraction) in double-double."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
        source_loop_color_declaration,
    )

    from .loop_nc_scan import slice_request

    assert source_loop_color_declaration(compiled.raw_loop_model) is None
    declaration = resolve_loop_color_declaration(compiled.raw_loop_model)
    assert declaration is not None
    out = {}
    for accuracy in ("lc", "nlc", "full"):
        slice_ = LoopSlice.generate(
            compiled, slice_request(spec, declaration, accuracy, "auto"), native
        )
        executor = NativeExecutor(
            slice_, forward=False, exported=export_plan(slice_, forward=False)
        )
        (result,) = executor.evaluate(
            [spec["point"]],
            [spec["mu_r2"]],
            parameters=spec["parameters"],
            precision=Precision.double_double(),
        )
        out[accuracy] = (slice_, _result_entries(result))
    return out


def sector_degrees(slice_) -> dict:
    """(component or "born", sector repr) -> leading degree P."""

    degrees = slice_.degrees()
    out = {("born", repr(k)): v for k, v in degrees["born"].items()}
    out.update({("virtual", repr(k)): v for k, v in degrees["virtual"].items()})
    return out


def fit(scan: dict, span: int, top: int):
    """The exact c_p (p = top - span .. top) of every entry from the first
    span + 1 ``FIT_NODES`` of ``scan`` (N -> loop_nc_scan JSON), and the
    relative miss of the fit at each of the ``CHECK_NODES`` (relative to the
    entry's sector's largest |c_p N^p|)."""

    powers = list(range(top - span, top + 1))
    used = FIT_NODES[: span + 1]
    assert len(used) == span + 1, (span, len(FIT_NODES))
    data = {n: _entries(scan[n]["born"], scan[n]["components"]) for n in scan}
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
    misses = {}
    for n in CHECK_NODES:
        x = Fraction(n)
        terms = {
            key: [c * x**p for p, c in row.items()] for key, row in coefficients.items()
        }
        size: dict = {}
        for key, row in terms.items():
            size[key[:2]] = max(size.get(key[:2], 0), *map(abs, row))
        misses[n] = max(
            abs(data[n].get(key, 0) - sum(row)) / (size[key[:2]] or 1)
            for key, row in terms.items()
        )
    return coefficients, misses


def sector_of(key) -> tuple:
    """The degree key of an entry: Born order, or the virtual sector of any
    component."""

    return ("born", key[1]) if key[0] == "born" else ("virtual", key[1])


def projection_errors(coefficients, mine, degrees) -> dict:
    """Per check, the largest error relative to the sector's largest term
    |c_p 3^p|: the window's two lowest powers, the powers above P, LC, NLC
    and FC(3)."""

    three = Fraction(3)
    size: dict = {}
    for key, row in coefficients.items():
        largest = max(abs(c * three**p) for p, c in row.items())
        size[key[:2]] = max(size.get(key[:2], 0), largest)
    floor = min(min(row) for row in coefficients.values())
    errors = dict.fromkeys(("floor", "above", "lc", "nlc", "fc"), Fraction(0))
    keys = set(coefficients) | set().union(*(m for m in mine.values()))
    for key in keys:
        row = coefficients.get(key, {})
        scale = size.get(key[:2]) or 1
        degree = degrees[sector_of(key)]
        for p, c in row.items():
            term = abs(c * three**p) / scale
            if degree is None or p > degree:
                errors["above"] = max(errors["above"], term)
            if p <= floor + 1:
                errors["floor"] = max(errors["floor"], term)
        for accuracy, depth in (("lc", 0), ("nlc", 2), ("fc", None)):
            if accuracy == "fc":
                expected = sum(c * three**p for p, c in row.items())
                got = mine["full"].get(key, 0)
            else:
                expected = sum(
                    c * three**p
                    for p, c in row.items()
                    if degree is not None and degree - depth <= p <= degree
                )
                got = mine[accuracy].get(key, 0)
            errors[accuracy] = max(errors[accuracy], abs(got - expected) / scale)
    return errors


def catani_errors(scan: dict) -> dict:
    """N -> the largest relative deviation of the poles from Catani at N."""

    out = {}
    for n, result in scan.items():
        if n not in CATANI_NC:
            continue
        errors = []
        assert result["ir_poles"], (n, "no Catani prediction")
        for sector, (single, double) in result["ir_poles"].items():
            virtual = result["components"]["virtual"][sector]
            predicted = (_exact(single), _exact(double))
            scale = max(map(abs, predicted))
            for got, want in zip((virtual[1], virtual[2]), predicted, strict=True):
                errors.append(abs(_exact(got) - want) / scale)
        out[n] = max(errors)
    return out


def rule_seven(model, compiled, native, spec, cache) -> dict:
    """Rule 7's projection part and the scan's Catani poles for ``spec``:
    the errors of every check (all within ``TOLERANCE`` to pass), the
    window and the fitted coefficients."""

    found = projections(compiled, native, spec)
    lc = found["lc"][0]
    degrees = sector_degrees(lc)
    assert degrees == sector_degrees(found["nlc"][0])
    top = max(v for v in degrees.values() if v is not None)
    span = 2 * sum(1 for leg in lc.basis.legs if leg.representation != 1)
    values = (*FIT_NODES[: span + 1], *CHECK_NODES)
    scan = _scan(model, spec, values, f"arbitrary:{SCAN_DIGITS}", cache)
    coefficients, misses = fit(scan, span, top)
    mine = {accuracy: entries for accuracy, (_s, entries) in found.items()}
    errors = projection_errors(coefficients, mine, degrees)
    errors["fit"] = max(misses.values())
    errors["catani"] = max(catani_errors(scan).values())
    return {
        "errors": errors,
        "top": top,
        "span": span,
        "coefficients": coefficients,
        "misses": misses,
    }


def test_nc_column_names_the_plan_rows_and_existing_oracles():
    """Every nc row of PLAN §4.2 has a column, the column's first id is this
    file's, and every oracle id names a test of the oracle files (the
    parametrised ids among the declared parameters)."""

    from . import test_loop_v2_nc_oracles as oracles

    assert (
        set(NC_COLUMN)
        == set(NC_ROWS)
        == {
            *("A01", "A08", "A09", "A12", "A13", "A15", "A18", "A21"),
            *("B03", "C01", "D05", "E01", "F01"),
        }
    )
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
        for test_id in ids[1:]:
            path, name = test_id.split("::")
            if path.endswith("test_loop_v2_nc_oracles.py"):
                base, _, parameter = name.partition("[")
                assert callable(getattr(oracles, base)), test_id
                if parameter:
                    assert parameter[:-1] in parameters[base], test_id
            else:
                assert test_id in _BDK, test_id


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
    """Rule 7 on the row (module docstring): LC, NLC and FC in double-double
    against the exact N_c fit of the scan, the fit's own checks, the leading
    degrees, and Catani at N, each within 1e-24."""

    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    found = rule_seven(
        loop_sm_compiled_model, compiled, loop_native, row_spec(row), tmp_path
    )
    errors = found["errors"]
    assert all(e <= TOLERANCE for e in errors.values()), (
        row,
        {k: float(v) for k, v in errors.items()},
    )
