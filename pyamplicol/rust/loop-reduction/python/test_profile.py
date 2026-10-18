# SPDX-License-Identifier: 0BSD
"""Optional timings delegate unchanged numerators/masters and reset per sample."""

import math

import pytest
from _loop_reduction import NativeBatch, PolynomialPlan


def triangle(rows):
    def batch(values, width):
        return NativeBatch.from_decimal(
            values * rows, rows, width, row_ids=[601 - 7 * i for i in range(rows)]
        )

    shifts = [
        [str(x), "0"] for p in [(0, 0, 0, 0), (1, 0, 0, 1), (0, 0, 0, 2)] for x in p
    ]
    return (
        [
            batch(shifts, 12),
            batch([["0", "0"]] * 3, 3),
            batch([["1", "0"]], 1),
            batch([["2", "0"], ["3", "0"], ["5", "0"]], 3),
        ],
        batch([[str(x), "0"] for x in [0, 0, -4, 0, 0, 0, -4, 0, 0]], 9),
    )


@pytest.mark.parametrize("engine", ["fast", "fast-tensor"])
@pytest.mark.parametrize(
    "sectors,active", [(["a*mu2", "b", "c"], 3), (["a", "0", "0"], 1)]
)
def test_profile_is_opt_in_preserves_values_and_resets_across_batches(
    engine, sectors, active
):
    plan = PolynomialPlan(3, sectors, ["a", "b", "c"], engine=engine)
    for rows in [3, 33, 0, 1]:
        inputs, matrix = triangle(rows)
        baseline = plan.evaluate_batch(*inputs, s_matrix=matrix)
        assert all(value == 0 for value in plan.take_profile().values())
        plan.set_profiling(True)
        profiled = plan.evaluate_batch(*inputs, s_matrix=matrix)
        assert profiled.statuses == baseline.statuses
        assert profiled.values.to_decimal() == baseline.values.to_decimal()
        profile = plan.take_profile()
        assert all(math.isfinite(value) and value >= 0 for value in profile.values())
        assert profile["engine_calls"] == active
        assert profile["engine_seconds"] >= (
            profile["masters_seconds"] + profile["numerator_callback_seconds"]
        )
        assert profile["reduction_exclusive_seconds"] == pytest.approx(
            profile["engine_seconds"]
            - profile["masters_seconds"]
            - profile["numerator_callback_seconds"],
            abs=1e-15,
        )
        if rows:
            for key in [
                "master_calls",
                "master_rows",
                "numerator_calls",
                "numerator_rows",
            ]:
                assert profile[key] > 0
        assert all(value == 0 for value in plan.take_profile().values())
        plan.evaluate_batch(*inputs, s_matrix=matrix)
        assert plan.take_profile()["engine_calls"] == active
        plan.evaluate_batch(*inputs, s_matrix=matrix)
        plan.set_profiling(False)
        assert all(value == 0 for value in plan.take_profile().values())


@pytest.mark.parametrize("engine", ["fast", "fast-tensor"])
def test_profile_preserves_failed_lane_status_and_sparse_values(engine):
    plan = PolynomialPlan(3, ["a*mu2", "b", "c"], ["a", "b", "c"], engine=engine)
    inputs, matrix = triangle(2)
    inputs[1] = NativeBatch.from_decimal(
        [["NaN", "0"]] * 3 + [["0", "0"]] * 3, 2, 3, row_ids=[601, 594]
    )
    baseline = plan.evaluate_batch(*inputs, s_matrix=matrix)
    plan.set_profiling(True)
    measured = plan.evaluate_batch(*inputs, s_matrix=matrix)
    assert measured.statuses == baseline.statuses
    assert measured.statuses[0][1] != "success"
    assert measured.successful_values().row_ids == [594]
    assert (
        measured.successful_values().to_decimal()
        == baseline.successful_values().to_decimal()
    )
    assert plan.take_profile()["engine_calls"] == 3


def test_profile_counts_completed_preparation_even_when_dense_head_fails():
    plan = PolynomialPlan(1, ["c^2", "0", "0"], ["c"], engine="fast-tensor")
    plan.set_profiling(True)
    with pytest.raises(ValueError, match="nonfinite polynomial coefficient"):
        plan.evaluate_decimal(
            [7], [[["0", "0"]] * 4], [["1", "0"]], [["1", "0"]], [["1e200", "0"]]
        )
    profile = plan.take_profile()
    assert profile["numerator_preparation_seconds"] > 0
    assert (
        profile["engine_calls"]
        == profile["master_calls"]
        == profile["numerator_calls"]
        == 0
    )
    assert all(value == 0 for value in plan.take_profile().values())


def test_profile_does_not_claim_generic_provider_instrumentation():
    plan = PolynomialPlan(1, ["1", "0", "0"], [])
    with pytest.raises(ValueError, match="requires fast"):
        plan.set_profiling(True)
    with pytest.raises(ValueError, match="requires fast"):
        plan.take_profile()
