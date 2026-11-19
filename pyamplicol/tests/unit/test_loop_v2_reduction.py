# SPDX-License-Identifier: 0BSD
"""The v2 NinjaRS adapter against the removed legacy ``PolynomialPlan`` path.

Frozen captures (``tests/fixtures/loop/legacy/reduction/``, recorded once
from the legacy engine before its removal) hold every family reduction of the
legacy engine (shifts, masses, invariants, scales and the projected family
numerators in the NinjaRS tensor order) and its Laurent results; the adapter
re-reduces the same lanes.
"""

from __future__ import annotations

import gzip
import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

CAPTURES = Path(__file__).resolve().parents[1] / "fixtures/loop/legacy/reduction"


def _batch(native, record, arithmetic, precision):
    return native.NativeBatch.from_decimal(
        [tuple(value) for value in record["values"]],
        record["rows"],
        record["columns"],
        arithmetic,
        precision,
        record["row_ids"],
    )


class _Frozen:
    """A frozen legacy reduction result (its Laurent values)."""

    def __init__(self, values):
        self.values = values


def _load(native, process, arithmetic, precision, engine):
    suffix = f"{process}_{engine}_{arithmetic}" + (str(precision) if precision else "")
    data = json.loads(gzip.decompress((CAPTURES / f"{suffix}.json.gz").read_bytes()))
    assert (data["arithmetic"], data["precision"], data["engine"]) == (
        arithmetic,
        precision,
        engine,
    )

    def batch(record):
        return None if record is None else _batch(native, record, arithmetic, precision)

    groups = [
        (
            group["propagators"],
            batch(group["shifts"]),
            batch(group["masses"]),
            batch(group["scales"]),
            batch(group["s_matrix"]),
            batch(group["numerator"]),
        )
        for group in data["groups"]
    ]
    legacy = [_Frozen(batch(record)) for record in data.get("results", ())]
    return groups, legacy


@pytest.fixture(scope="module")
def captured(loop_native):
    cache = {}

    def capture(process, arithmetic="double", precision=None, engine="fast"):
        key = (process, arithmetic, precision, engine)
        if key not in cache:
            cache[key] = _load(loop_native, process, arithmetic, precision, engine)
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
    loop_native, captured, process, legacy_engine
):
    groups, legacy = captured(process, engine=legacy_engine)
    direct, check = loop_native.V2Reducer("double", None, "fast").reduce(groups)
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
        # B2 masters are keyed by the ordered mass tuple (DESIGN §11.2), so the
        # swapped orders of the reversed frame are new masters, not cache hits.
        reducer = loop_native.V2Reducer("double", None, "fast")
        reducer.reduce(groups)
        alone = reducer.take_master_statistics()["B2"]["evaluated"]
        reducer.reduce(groups, check=True)
        assert reducer.take_master_statistics()["B2"]["evaluated"] > alone


@pytest.mark.parametrize("process", ["gg_gg", "gg_ttx"])
def test_cached_masters_equal_uncached_bitwise(loop_native, captured, process):
    groups, _ = captured(process)
    cached = loop_native.V2Reducer("double", None, "fast", cache_masters=True)
    uncached = loop_native.V2Reducer("double", None, "fast", cache_masters=False)
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
    alone = loop_native.V2Reducer("double", None, "fast")
    for group, batched in zip(groups, direct, strict=True):
        assert (
            alone.reduce([group])[0][0].values.to_decimal()
            == batched.values.to_decimal()
        )


@pytest.mark.parametrize("process", ["gg_gg", "gg_ttx"])
def test_reduction_direction_check_agrees_and_poles_are_optional(
    loop_native, captured, process
):
    groups, _ = captured(process)
    reducer = loop_native.V2Reducer("double", None, "fast")
    direct, check = reducer.reduce(groups, check=True)
    assert _statuses_succeed(direct + check)
    for a, b in zip(direct, check, strict=True):
        assert _lane_relative(b, a, 3) < Decimal("1e-12")
    finite, finite_check = reducer.reduce(groups, poles="none", check=True)
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
    loop_native, captured, arithmetic, precision, tolerance
):
    groups, legacy = captured("gg_ttx", arithmetic, precision, "generic")
    reducer = loop_native.V2Reducer(arithmetic, precision, "generic")
    assert reducer.take_master_statistics() is None
    direct, check = reducer.reduce(groups, check=True)
    assert _statuses_succeed(direct + check)
    for new, reversed_, old in zip(direct, check, legacy, strict=True):
        assert _lane_relative(new, old, 3) < Decimal(tolerance)
        assert _lane_relative(reversed_, new, 3) < Decimal(tolerance)


@pytest.mark.loop_nightly
def test_master_hit_rate_on_five_gluons(loop_native, captured, record_property):
    """Report the cache hit rate of one gg -> ggg batch (frozen groups)."""
    groups, _ = captured("gg_ggg")
    reducer = loop_native.V2Reducer("double", None, "fast")
    uncached = loop_native.V2Reducer("double", None, "fast", cache_masters=False)
    rates = {}
    for check in (False, True):
        direct, reversed_ = reducer.reduce(groups, check=check)
        plain, plain_reversed = uncached.reduce(groups, check=check)
        for a, b in zip(
            direct + (reversed_ or []), plain + (plain_reversed or []), strict=True
        ):
            assert a.values.to_decimal() == b.values.to_decimal()
        statistics = reducer.take_master_statistics()
        uncached.take_master_statistics()
        requests = sum(s["requests"] for s in statistics.values())
        evaluated = sum(s["evaluated"] for s in statistics.values())
        assert evaluated == reducer.cached_masters
        rates["with_check" if check else "direct"] = {
            "requests": requests,
            "distinct": evaluated,
            "hit_rate": 1 - evaluated / requests,
            "per_family": statistics,
        }
    record_property("gg_ggg_master_cache", rates)
    print(f"gg->ggg master cache, {len(groups)} families: {rates}")
    assert rates["direct"]["hit_rate"] > 0.9
    # The reversed frame finds every master but swapped unequal-mass bubbles.
    assert rates["with_check"]["distinct"] == rates["direct"]["distinct"]
