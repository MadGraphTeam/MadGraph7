# SPDX-License-Identifier: 0BSD
"""The v2 NinjaRS adapter against the existing ``PolynomialPlan`` path.

A recording backend captures every family reduction of the inherited engine
(shifts, masses, invariants, scales and the projected family coefficients) and
the adapter re-reduces the same lanes as dense tensor numerators.
"""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops import physical_process

pytestmark = pytest.mark.loop_physics

REFERENCES = Path(__file__).resolve().parents[2] / "tools/loop_reference"


@pytest.fixture(scope="module")
def native(loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native generation requires a licensed Symbolica runtime")
    return loop_native


class _RecordingPlan:
    def __init__(self, plan, meta, calls):
        self._plan, self._meta, self._calls = plan, meta, calls

    def __getattr__(self, name):
        return getattr(self._plan, name)

    def evaluate_batch(self, shifts, masses, scales, coefficients, s_matrix=None):
        result = self._plan.evaluate_batch(
            shifts, masses, scales, coefficients, s_matrix=s_matrix
        )
        self._calls.append(
            (self._meta, shifts, masses, scales, s_matrix, coefficients, result)
        )
        return result


class _RecordingBackend:
    """The native loop module, recording each family reduction call."""

    def __init__(self, native):
        self._native, self.calls = native, []

    def __getattr__(self, name):
        return getattr(self._native, name)

    def PolynomialPlan(self, propagators, sectors, parameters, **options):
        plan = self._native.PolynomialPlan(propagators, sectors, parameters, **options)
        return _RecordingPlan(plan, (propagators, sectors, parameters), self.calls)


def _dense(native, meta, coefficients, arithmetic, precision):
    """Scatter the legacy monomial coefficients into the NinjaRS tensor order."""
    _, sectors, parameters = meta
    assert tuple(sectors[1:]) == ("0", "0")
    column = {name: index for index, name in enumerate(parameters)}
    terms = []
    for term in sectors[0].split("+"):
        name, *factors = term.split("*")
        powers = [0, 0, 0, 0]
        for factor in factors:
            variable, exponent = factor.split("^")
            powers[("q0", "q1", "q2", "q3").index(variable)] = int(exponent)
        terms.append((column[name], tuple(powers)))
    rank = max(sum(powers) for _, powers in terms)
    order = {tuple(m): i for i, m in enumerate(native.v2_tensor_monomials(rank))}
    rows, width = coefficients.rows, coefficients.columns
    values = coefficients.to_decimal()
    dense = [["0", "0"] for _ in range(rows * len(order))]
    for row in range(rows):
        for source, powers in terms:
            dense[row * len(order) + order[powers]] = values[row * width + source]
    return native.NativeBatch.from_decimal(
        dense, rows, len(order), arithmetic, precision, coefficients.row_ids
    )


def _capture(native, factory, fixture, arithmetic, precision, engine):
    points = np.asarray(
        json.loads((REFERENCES / fixture).read_text(), parse_float=str, parse_int=str)[
            "points"
        ],
        dtype=str,
    )
    backend = _RecordingBackend(native)
    factory(
        backend=backend,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    ).evaluate_decimal(points, ["1000000"] * len(points), model_values=["0.118", "173"])
    groups = [
        (
            meta[0],
            shifts,
            masses,
            scales,
            matrix,
            _dense(native, meta, coefficients, arithmetic, precision),
        )
        for meta, shifts, masses, scales, matrix, coefficients, _ in backend.calls
    ]
    return groups, [call[-1] for call in backend.calls]


_GLUONS = (physical_process.gluon_scattering_qcd_program, "gluon_scattering_qcd.json")
# Unequal-mass (gluon, top) bubbles reach the bubble masters with both orders.
_TOPS = (
    partial(
        physical_process._qcd_process_program,
        ("g", "g"),
        ("t", "t~"),
        ("epsilon", "epsilon", "u_bar", "v"),
        256,
    ),
    "top_pair_qcd.json",
)


@pytest.fixture(scope="module")
def captured(native):
    cache = {}

    def capture(process, arithmetic="double", precision=None, engine="fast"):
        key = (process, arithmetic, precision, engine)
        if key not in cache:
            factory, fixture = {"gg_gg": _GLUONS, "gg_ttx": _TOPS}[process]
            cache[key] = _capture(
                native, factory, fixture, arithmetic, precision, engine
            )
        return cache[key]

    return capture


def _decimals(batch):
    return [tuple(Decimal(x) for x in z) for z in batch.values.to_decimal()]


def _lane_relative(actual, expected, columns):
    """Largest lane deviation relative to the lane's largest component."""
    worst = Decimal(0)
    with localcontext() as context:
        context.prec = 80
        a, b = _decimals(actual), _decimals(expected)
        assert len(a) == len(b)
        for start in range(0, len(b), columns):
            norm = [abs(complex(*z)) for z in b[start : start + columns]]
            scale = Decimal(max(norm)) or Decimal(1)
            for x, y in zip(
                a[start : start + columns], b[start : start + columns], strict=True
            ):
                difference = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
                worst = max(worst, difference.sqrt() / scale)
    return worst


def _statuses_succeed(batches):
    return all(status == "success" for b in batches for _, status in b.statuses)


@pytest.mark.parametrize("process", ["gg_gg", "gg_ttx"])
@pytest.mark.parametrize("legacy_engine", ["fast", "generic"])
def test_v2_equals_the_polynomial_plan_on_legacy_family_numerators(
    native, captured, process, legacy_engine
):
    groups, legacy = captured(process, engine=legacy_engine)
    direct, check = native.V2Reducer("double", None, "fast").reduce(groups)
    assert check is None and _statuses_succeed(direct)
    for new, old in zip(direct, legacy, strict=True):
        assert new.values.row_ids == old.values.row_ids
        assert _lane_relative(new, old, 3) < Decimal("1e-12")
    if process == "gg_ttx":
        unequal = [
            g
            for g in groups
            if g[0] == 2 and len(set(map(tuple, g[2].to_decimal()))) == 2
        ]
        assert unequal


@pytest.mark.parametrize("process", ["gg_gg", "gg_ttx"])
def test_cached_masters_equal_uncached_bitwise(native, captured, process):
    groups, _ = captured(process)
    cached = native.V2Reducer("double", None, "fast", cache_masters=True)
    uncached = native.V2Reducer("double", None, "fast", cache_masters=False)
    direct, check = cached.reduce(groups, check=True)
    plain, plain_check = uncached.reduce(groups, check=True)
    for a, b in zip(direct + check, plain + plain_check, strict=True):
        assert a.values.to_decimal() == b.values.to_decimal()
        assert a.statuses == b.statuses
    statistics = cached.take_master_statistics()
    requests = sum(s["requests"] for s in statistics.values())
    hits = sum(s["hits"] for s in statistics.values())
    assert 0 < hits < requests
    assert uncached.take_master_statistics()["D0"]["hits"] == 0
    # Lanes of other families and points in the same bucket change nothing.
    alone = native.V2Reducer("double", None, "fast")
    for group, batched in zip(groups, direct, strict=True):
        assert (
            alone.reduce([group])[0][0].values.to_decimal()
            == batched.values.to_decimal()
        )


@pytest.mark.parametrize("process", ["gg_gg", "gg_ttx"])
def test_reduction_direction_check_agrees_and_poles_are_optional(
    native, captured, process
):
    groups, _ = captured(process)
    reducer = native.V2Reducer("double", None, "fast")
    direct, check = reducer.reduce(groups, check=True)
    assert _statuses_succeed(direct + check)
    for a, b in zip(direct, check, strict=True):
        assert _lane_relative(b, a, 3) < Decimal("1e-12")
    finite, finite_check = reducer.reduce(groups, poles=False, check=True)
    for full, only, full_check, only_check in zip(
        direct, finite, check, finite_check, strict=True
    ):
        assert only.values.columns == 1
        assert only.values.to_decimal() == full.values.take_columns([0]).to_decimal()
        assert (
            only_check.values.to_decimal()
            == full_check.values.take_columns([0]).to_decimal()
        )


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [("double-double", None, "1e-28"), ("arbitrary", 40, "1e-36")],
)
def test_generic_engine_from_exponent_arrays_equals_legacy_extended_precision(
    native, captured, arithmetic, precision, tolerance
):
    groups, legacy = captured("gg_ttx", arithmetic, precision, "generic")
    reducer = native.V2Reducer(arithmetic, precision, "generic")
    assert reducer.take_master_statistics() is None
    direct, check = reducer.reduce(groups, check=True)
    assert _statuses_succeed(direct + check)
    for new, reversed_, old in zip(direct, check, legacy, strict=True):
        assert _lane_relative(new, old, 3) < Decimal(tolerance)
        assert _lane_relative(reversed_, new, 3) < Decimal(tolerance)
