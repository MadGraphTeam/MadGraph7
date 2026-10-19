# SPDX-License-Identifier: 0BSD
"""Opaque caller labels over actual retained producer state/current/reduction data."""

from dataclasses import replace

import numpy as np
import pytest

from pyamplicol.loops.kernels import require_finite_batch
from pyamplicol.loops.numerical_lanes import LaneFailures

RETAINED_CONTROL = None


@pytest.fixture(scope="module")
def retained():
    if RETAINED_CONTROL is None:
        pytest.skip("supply an actual retained complete source owner")
    return RETAINED_CONTROL


def _view(retained):
    return retained["ordinary"].bind(
        backend=retained["backend"],
        arithmetic="double",
        parameters=retained["parameters"],
    )


def _bad_column(backend, result, lane):
    bad = backend.NativeBatch.from_exact_decimal([("1e400", "0")], 1, 1, row_ids=[lane])
    selected = backend.NativeBatch.merge_rows(
        result.row_ids,
        [
            result.take_columns([0]).take_rows(
                [i for i, row in enumerate(result.row_ids) if row != lane]
            ),
            bad,
        ],
    )
    return backend.NativeBatch.concat_columns(
        [selected, result.take_columns(range(1, result.columns))]
    )


def _agrees(result, reference, ids):
    for name in (
        "born",
        "bare",
        "uv",
        "r2",
        "uv_local",
        "uv_external_fields",
        "totals",
    ):
        actual, expected = getattr(result, name), getattr(reference, name)
        expected = expected.take_rows(list(range(len(ids))), row_ids=ids)
        assert actual.row_ids == ids
        assert actual.agreement_rows(expected, "1e-8", "1e-12") == [
            (point, True) for point in ids
        ]


def test_opaque_labels_survive_tail_permutation_and_strict_state_error(retained):
    program = _view(retained)
    ids = [7, 2**64 - 1, 19]
    points = np.repeat(retained["points"][:1], len(ids), axis=0)
    result = program.evaluate_ordered_partial_native(
        points, ["1000000"] * len(ids), point_ids=ids
    )
    assert result.batch.point_ids == ids and not result.failures
    assert result.batch.lane_ids == list(range(result.batch.rows))
    one = program.evaluate_ordered_partial_native(
        points[:1], ["1000000"], point_ids=[ids[1]]
    )
    _agrees(one, retained["reference"], [ids[1]])
    bad = points[:2].copy()
    bad[1, retained["massive_vector_leg"], 1:] = "0"
    with pytest.raises(ValueError, match=f"{ids[1]}.*longitudinal"):
        program.bare.prepare_currents_native(bad, ["1000000"] * 2, point_ids=ids[:2])
    for value in (result.born, result.totals):
        expected = value.take_rows([1, 0, 2], row_ids=[ids[1], ids[0], ids[2]])
        repeated = program.evaluate_ordered_partial_native(
            points[[1, 0, 2]], ["1000000"] * 3, point_ids=[ids[1], ids[0], ids[2]]
        )
        actual = repeated.born if value is result.born else repeated.totals
        assert actual.agreement_rows(expected, "1e-12", "1e-14") == [
            (point, True) for point in expected.row_ids
        ]


@pytest.mark.parametrize("stage", ["loop", "counterterm"])
def test_current_failure_maps_local_rows_after_state_compaction(retained, stage):
    program = _view(retained)
    ids = [19, 2**64 - 1, 7]
    points = np.repeat(retained["points"][:1], 3, axis=0)
    points[0, retained["massive_vector_leg"], 1:] = "0"
    failures = LaneFailures()
    currents = program.bare.prepare_currents_native(
        points, ["1000000"] * 3, point_ids=ids, _failures=failures
    )
    assert currents.batch.point_ids == ids[1:]
    lane = currents.batch.lane_ids[0]
    assert lane != 0 and currents.batch.point_by_lane[lane] == ids[1]
    if stage == "loop":
        bad = _bad_column(retained["backend"], currents.loops[0], lane)
        with pytest.raises(ArithmeticError, match=str(ids[1])):
            require_finite_batch(
                bad, "loop current", row_points=currents.batch.point_by_lane
            )
        currents = replace(currents, loops=(bad, *currents.loops[1:]))
        result = program.bare._evaluate_currents_native(currents, failures)
        assert result.batch.point_ids == [7]
        assert [(f.point_id, f.stage) for f in failures.entries][-1] == (
            ids[1],
            "loop current 0",
        )
    else:
        current = program.currents[0]
        original = current.executor

        class FailedCurrent:
            def evaluate_batch(self, inputs):
                return _bad_column(
                    retained["backend"], original.evaluate_batch(inputs), lane
                )

        current.executor = FailedCurrent()
        try:
            program.evaluate_counterterm_currents_native(
                currents.batch, _failures=failures
            )
        finally:
            current.executor = original
        assert [(f.point_id, f.stage) for f in failures.entries][-1] == (
            ids[1],
            "counterterm current",
        )


@pytest.mark.parametrize("nonfinite", [False, True])
def test_reduction_failure_keeps_original_point_order_and_family(retained, nonfinite):
    program = _view(retained)
    ids = [2**64 - 1, 7]
    family = program.bare.families[0]
    original = family.reducer
    seen = []

    class FailedReduction:
        def evaluate_batch(self, shifts, masses, scales, coefficients, **kwargs):
            assert shifts.row_ids == list(range(shifts.rows))
            reduced = original.evaluate_batch(
                shifts, masses, scales, coefficients, **kwargs
            )
            failed_row = 0
            seen.append((family.index, family.sector_indices[0]))

            class Result:
                statuses = tuple(
                    (
                        row,
                        "injected numerical failure"
                        if row == failed_row and not nonfinite
                        else status,
                    )
                    for row, status in reduced.statuses
                )

                def successful_values(self):
                    values = reduced.successful_values()
                    if nonfinite:
                        return _bad_column(retained["backend"], values, failed_row)
                    return values.take_rows(
                        [i for i, row in enumerate(values.row_ids) if row != failed_row]
                    )

            return Result()

    family.reducer = FailedReduction()
    try:
        result = program.evaluate_ordered_partial_native(
            retained["points"], retained["scales"], point_ids=ids
        )
    finally:
        family.reducer = original
    assert seen == [(family.index, family.sector_indices[0])]
    assert result.batch.point_ids == [7]
    _agrees(result, retained["reference"], [7])
    assert [(f.point_id, f.order_index, f.family_index) for f in result.failures] == [
        (ids[0], family.sector_indices[0], family.index)
    ]
    assert all(
        point == 7 and status == "success" for point, status in result.statuses[0]
    )


@pytest.mark.parametrize("class_index", [0, 2])
def test_literal_sources_and_both_directed_terms_keep_opaque_ids(retained, class_index):
    from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

    model = _view(retained)
    original = retained["source_classes"][class_index]
    layout = replace(original.bare.spin_layout, owner=model.bare.prepared)
    prepared = replace(
        original,
        plan=model.prepared,
        bare=replace(original.bare, plan=model.bare.prepared, spin_layout=layout),
    )
    directed = prepared.bind(model)
    ids = [2**64 - 1, 7]
    sources = NativeSpinSourcePlan(model.bare, (layout.legs,))
    vectors = {
        leg: [("0", "0"), ("1", "2"), ("3", "-1"), ("0", "0")] for leg in layout.legs
    }
    literal = sources.prepare_decimal(vectors, ids)
    actual = directed.evaluate_native(
        retained["points"], retained["scales"], point_ids=ids, spin_sources=literal
    )
    dense = directed.evaluate_native(
        retained["points"],
        retained["scales"],
        point_ids=[0, 1],
        spin_sources=sources.prepare_decimal(vectors, [0, 1]),
    )
    for name in (
        "born",
        "virtual",
        "bare",
        "uv",
        "r2",
        "uv_local",
        "uv_external_fields",
        "first",
        "second",
    ):
        result = getattr(actual, name)
        reference = getattr(dense, name).take_rows([0, 1], row_ids=ids)
        assert result.row_ids == ids
        assert result.agreement_rows(reference, "1e-12", "1e-14") == [
            (point, True) for point in ids
        ]
    assert all(point in ids for status in actual.statuses for point, _ in status)
