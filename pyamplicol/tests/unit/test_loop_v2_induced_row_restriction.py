# SPDX-License-Identifier: 0BSD
"""Run-time helicity selections of loop-induced processes evaluate their row
restriction only (m-lit lit-dress (H); USER 2026-11-08 ~22:34, DECISIONS
2026-11-08 22:34 (2) 'ADDED').

A selection keeping some helicity rows of a loop-induced (dressed-row,
route R2) plan must dress, close, scatter and reduce only what the plan
whose columns keep those rows evaluates (the "row-restricted plan": the
same plan JSON with every column keeping the selected rows only, pruned by
the native loader, ``plan/live.rs``). Each call's native counts (modelled
dressing work, steps, families, fused factors read, closure columns,
reduction lanes) equal the row-restricted plan's, and so do its values
(bitwise, binary64):

* on the executor (``NativeExecutor.row_mask``, the path of every entry
  point) for a single row, two rows and all but one row of g g > h h;
* through the Python API: ``evaluate(helicities=...)`` and
  ``evaluate_resolved(helicities=...)`` on the stored g g > h g g artifact,
  for one, two and all but one helicity;
* through the C ABI (``rusticol_virtual_evaluate_selected``): a global
  selection of the helicity rows TIRust never refuses on the 128 stream
  points serves every point (an unselected row's refusal no longer fails
  a point: its lanes are not reduced), and the per-point selection
  (``helicity_by_point``) equals the Python API's per-point selections.

At the base (before lit-dress) a family keeping more than half of its row
links ran whole (``KEPT_SHARE``): the two-row and all-but-one selections of
g g > h h and the all-but-one selection of g g > h g g dressed and reduced
every row, a single row closed and stored every row's columns (no
"closure columns" count existed), and the C ABI selection above failed 19
of the 128 points through rows it did not select.
"""

from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.unit.loop_v2_induced_observables_rows import ROWS, generate
from tests.unit.test_loop_v2_induced import _Rows

pytestmark = pytest.mark.loop_physics

COUNTS = (
    "dressing work (modelled complex MAC per lane)",
    "steps",
    "families",
    "factors",
    "closure columns",
    "reduction lanes",
)


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li"))


@pytest.fixture(scope="module")
def artifacts(loop_sm_ufo, loop_native, tmp_path_factory):
    return {
        name: generate(tmp_path_factory.mktemp(name), name, expression)
        for name, expression in ROWS.items()
        if name == "li_gg_hgg"
    }


def _restricted(executor, kept):
    """``executor``'s plan with every column keeping the Born rows ``kept``
    only, loaded (and pruned) by the native loader: the row-restricted plan
    of the selection ``kept``."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor

    # A loaded artifact's executor holds its stored plan member (decoded to
    # the plan JSON by ``text()``); a generated slice's exports it.
    stored = executor._plan_json
    plan = json.loads(
        stored.text() if hasattr(stored, "text") else executor.exported.json()
    )
    every = range(plan["rows"])
    plan["column_rows"] = [
        [row for row in (every if rows is None else rows) if row in kept]
        for rows in (plan.get("column_rows") or [None] * plan["columns"])
    ]
    return NativeExecutor(
        executor.slice,
        exported=executor.exported,
        plan_json=json.dumps(plan, separators=(",", ":")),
        inputs=executor.inputs,
        lanes=executor.lanes,
        threads=executor.threads,
        forward=False,
    )


def _counts(executor):
    counts = executor.last_report["counts"]
    return {name: counts[name] for name in COUNTS}


def _values(results):
    return [
        [
            float(complex(laurent.finite).real).hex()
            for laurent in result.components["virtual"].values()
        ]
        for result in results
    ]


def _selections(rows):
    return {
        "single": {rows - 1},
        "two": {0, rows // 2},
        "all but one": set(range(rows)) - {rows // 3},
    }


def test_executor_selections_run_their_row_restriction(li):
    """g g > h h, three fixture points: each selection's counts and values
    are the row-restricted plan's (at the base the two-row and all-but-one
    selections ran every family whole)."""

    from pyamplicol.loops.v2.assemble import Precision

    name = "li_gg_hh"
    executor = li.executor(name, "native")
    rows = len(executor.exported.rows)
    points, scales, parameters = [], [], None
    for index in range(3):
        point, scale, parameters = li.point(name, index)
        points.append(point)
        scales.append(scale)
    whole = executor.evaluate(
        points, scales, parameters=parameters, precision=Precision.double()
    )
    whole_counts = _counts(executor)
    for label, kept in _selections(rows).items():
        executor.row_mask = tuple(row in kept for row in range(rows))
        try:
            selected = executor.evaluate(
                points, scales, parameters=parameters, precision=Precision.double()
            )
            counts = _counts(executor)
        finally:
            executor.row_mask = None
        restricted = _restricted(executor, kept)
        # The row-restricted plan dresses the kept rows only; its row mask
        # squares them only (every row's counterterm kets are formed).
        restricted.row_mask = tuple(row in kept for row in range(rows))
        try:
            reference = restricted.evaluate(
                points, scales, parameters=parameters, precision=Precision.double()
            )
        finally:
            restricted.row_mask = None
        assert counts == _counts(restricted), label
        assert _values(selected) == _values(reference), label
        for count in (
            "dressing work (modelled complex MAC per lane)",
            "reduction lanes",
        ):
            assert counts[count] < whole_counts[count], (label, count)
    assert _values(whole) != _values(selected)


def _finite(results):
    return [
        sum(
            float(complex(laurent.finite).real)
            for laurent in r.components["virtual"].values()
        )
        for r in results
    ]


def test_executor_selections_square_their_rows_only(li):
    """u u~ > h g g, two fixture points (eight of its 16 helicity rows vanish
    exactly): a row selection squares its kept rows only, so every single
    row's value lies in [0, the summed value], the single rows sum to the
    summed value and a two-row selection to its two rows (identities: a
    relative 1e-12 for the summation order). Before lit-dress the executor's
    row mask squared every row with its counterterm kets (formed for every
    row), so an exactly vanishing row returned the other rows' counterterm
    squares (up to 12x the summed value on the stream points)."""

    import math

    from pyamplicol.loops.v2.assemble import Precision

    name = "li_uux_hgg"
    executor = li.executor(name, "native")
    rows = len(executor.exported.rows)
    points, scales, parameters = [], [], None
    for index in range(2):
        point, scale, parameters = li.point(name, index)
        points.append(point)
        scales.append(scale)

    def evaluate(kept):
        executor.row_mask = (
            None if kept is None else tuple(r in kept for r in range(rows))
        )
        try:
            return _finite(
                executor.evaluate(
                    points, scales, parameters=parameters, precision=Precision.double()
                )
            )
        finally:
            executor.row_mask = None

    summed = evaluate(None)
    singles = [evaluate({row}) for row in range(rows)]
    assert any(all(v == 0.0 for v in single) for single in singles)
    for p, total in enumerate(summed):
        assert all(0.0 <= single[p] <= total * (1 + 1e-12) for single in singles)
        assert math.isclose(sum(single[p] for single in singles), total, rel_tol=1e-12)
    live = [row for row in range(rows) if any(singles[row])]
    pair = {live[0], live[-1]}
    for p, value in enumerate(evaluate(pair)):
        expected = sum(singles[row][p] for row in pair)
        assert math.isclose(value, expected, rel_tol=1e-12), (p, value, expected)


def _runtime(artifacts, name):
    from pyamplicol import Runtime

    return Runtime.load(artifacts[name], process="p")


def test_api_selections_run_their_row_restriction(artifacts):
    """evaluate(helicities=...) and evaluate_resolved(helicities=...) on the
    stored g g > h g g: the call's native counts are the row-restricted
    plan's, for one, two and all but one helicity (at the base the
    all-but-one selection ran every family whole, and every selection closed
    every row)."""

    from pyamplicol.loops.v2.assemble import Precision
    from tests.unit.test_loop_v2_induced import _fixture

    name = "li_gg_hgg"
    runtime = _runtime(artifacts, name)
    fixture = _fixture(name)
    points = fixture["points"][:2]
    scales = [str(Fraction(mu) ** 2) for mu in fixture["mu_r"][:2]]
    observables = runtime._backend._observable_runtime()
    executor = observables._observables.executor
    catalogue = observables.selection_catalogue()
    ids = [f"h{k}" for k in range(len(catalogue.rows))]
    exact = [
        [[Fraction(c) for c in momentum] for momentum in point] for point in points
    ]
    rows = executor.exported.rows
    for label, chosen in _selections(len(ids)).items():
        helicities = [ids[k] for k in sorted(chosen)]
        selected, _basis = catalogue.resolve(helicities, None)
        kept = {rows.index(row) for row in selected}
        restricted = _restricted(executor, kept)
        restricted.evaluate(
            exact,
            [Fraction(s) for s in scales],
            precision=Precision.double(),
        )
        reference = _counts(restricted)
        for call in ("evaluate", "evaluate_resolved"):
            getattr(runtime, call)(points, mu_r2=scales, helicities=helicities)
            assert _counts(executor) == reference, (label, call)


def test_c_abi_selections_serve_points_refused_outside_them(artifacts):
    """rusticol_virtual_evaluate_selected on g g > h g g with TIRust on the
    128 stream points: a global selection of the helicity rows whose single
    selections TIRust serves on every point serves every point (no lane of
    an unselected row is reduced, so no refusal of one fails a point; at the
    base five families ran whole there and 19 points failed), its values
    bitwise the Python API's; the per-point selection (``helicity_by_point``,
    each point a row of that set) serves every point too, each value the
    Python API's selection of that point's row."""

    from tests.unit.test_loop_v2_sdk_extras import _sdk
    from tests.unit.test_loop_v2_sdk_selection import (
        SUCCESS_STATUS,
        _call,
        _json,
        _open,
        _python_values,
    )
    from tools.loop_benchmark import streams

    name = "li_gg_hgg"
    runtime = _runtime(artifacts, name)
    stream = streams.load(name)
    count = len(stream["points"])
    exact = [streams.exact(stream, i) for i in range(count)]
    points = [[[float(c) for c in momentum] for momentum in point] for point in exact]
    scales = [float(Fraction(mu)) ** 2 for mu in stream["mu_r"][:count]]
    exact = [[[str(Decimal(c)) for c in p] for p in point] for point in points]
    exact_scales = [str(Decimal(s)) for s in scales]
    sdk = _sdk()
    handle = _open(sdk, artifacts[name])
    try:
        assert sdk.rusticol_virtual_set_reduction(handle, b"tirust") == 0
        rows = len(
            _json(sdk.rusticol_virtual_capabilities_json, handle)["helicity_rows"]
        )
        served = [
            row
            for row in range(rows)
            if all(
                s == SUCCESS_STATUS
                for s in _call(sdk, handle, points, scales, ids=[f"h{row}"])["statuses"]
            )
        ]
        assert 1 < len(served) < rows, served
        ids = [f"h{row}" for row in served]
        selected = _call(sdk, handle, points, scales, ids=ids)
        assert selected["statuses"] == [SUCCESS_STATUS] * count
        python = runtime.evaluate(
            exact,
            mu_r2=exact_scales,
            helicities=ids,
            reduction="tirust",
            on_failure="return",
        )
        expected = _python_values(python, False)
        assert [x.hex() for x in selected["virtual"]] == [x.hex() for x in expected]
        by_point = [served[p % len(served)] for p in range(count)]
        per_point = _call(sdk, handle, points, scales, by_point=by_point)
        assert per_point["statuses"] == [SUCCESS_STATUS] * count
        width = len(per_point["virtual"]) // count
        for p in range(0, count, 17):
            one = runtime.evaluate(
                exact[p : p + 1],
                mu_r2=exact_scales[p : p + 1],
                helicities=[f"h{by_point[p]}"],
                reduction="tirust",
                on_failure="return",
            )
            value = _python_values(one, False)
            assert [
                x.hex() for x in per_point["virtual"][p * width : (p + 1) * width]
            ] == [x.hex() for x in value]
    finally:
        sdk.rusticol_virtual_free(handle)
