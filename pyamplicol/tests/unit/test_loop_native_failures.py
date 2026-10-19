# SPDX-License-Identifier: 0BSD
"""Actual compact reduction failures preserve physical and resolved identities."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from .test_loop_artifact_writer import source_process, target_metadata, write_artifact

__all__ = ["target_metadata"]

NEAR = [
    ["5000000000.5", "0", "0", "5000000000.5"],
    ["5000000000.5", "0", "0", "-5000000000.5"],
    ["5000000000.5", "100000", "0", "4999999999.5"],
    ["5000000000.5", "-100000", "0", "-4999999999.5"],
]


@pytest.fixture(scope="module")
def source_plan():
    pytest.importorskip("_loop_reduction")
    source = os.environ.get("PYAMPLICOL_RESOLVED_SOURCE_PACBIN")
    if not source:
        pytest.skip("set PYAMPLICOL_RESOLVED_SOURCE_PACBIN to an actual gg control")
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    return Path(source), json.loads(
        fixture.read_text(), parse_float=str, parse_int=str
    )["points"]


@pytest.mark.parametrize("resolved", [False, True])
def test_actual_reduction_failure_compacts_whole_points(source_plan, resolved):
    import _loop_reduction as native

    path, points = source_plan
    runtime = native.NativeLoopProgram(
        str(path), arithmetic="double", reduction_engine="generic"
    )
    plan = json.loads(runtime.metadata_json())
    h = len(plan["helicities"]) if resolved else 1
    batch = [points[0], NEAR, points[1]]
    result = runtime.evaluate(
        batch, ["1000000"] * 3, point_ids=[7, 91, 4], resolved=resolved, partial=True
    )
    assert result.requested_point_ids == [7, 91, 4]
    assert result.point_ids == [
        point * h + helicity for point in (7, 4) for helicity in range(h)
    ]
    assert result.failures
    for failure in result.failures:
        assert failure.point_id == 91
        assert failure.helicity is None if not resolved else 0 <= failure.helicity < h
        assert failure.order_index == 0
        assert failure.family_index in {family["index"] for family in plan["families"]}
        assert failure.stage == "ordinary numerator reduction"
        assert failure.detail != "success"
    expected = runtime.evaluate(
        [points[0], points[1]], ["1000000"] * 2, point_ids=[7, 4], resolved=resolved
    )
    for actual, reference in zip(
        [result.born, *result.component_batches],
        [expected.born, *expected.component_batches],
        strict=True,
    ):
        assert all(ok for _, ok in actual.agreement_rows(reference, "1e-10", "1e-12"))
    with pytest.raises(ValueError, match="point_id: 91"):
        runtime.evaluate(
            batch, ["1000000"] * 3, point_ids=[7, 91, 4], resolved=resolved
        )
    failed = runtime.evaluate(
        [NEAR], ["1000000"], point_ids=[91], resolved=resolved, partial=True
    )
    assert failed.requested_point_ids == [91] and failed.point_ids == []
    assert failed.born.rows == failed.totals.rows == 0 and failed.failures
    empty = runtime.evaluate([], [], resolved=resolved, partial=True)
    assert empty.requested_point_ids == empty.point_ids == empty.failures == []
    assert empty.born.columns == failed.born.columns


def test_public_error_carries_native_failure_fields(
    source_plan, tmp_path, target_metadata
):
    from pyamplicol import EvaluationError, Runtime, VirtualFailureDiagnostic

    path, points = source_plan
    process = source_process(path)
    process = replace(process, metadata=replace(process.metadata, resolved=True))
    destination = tmp_path / "artifact"
    write_artifact(destination, [process])
    runtime = Runtime.load(destination, arithmetic="double-double")
    for operation in (runtime.evaluate, runtime.evaluate_resolved):
        with pytest.raises(EvaluationError) as error:
            operation([points[0], NEAR, points[1]], mu_r2="1000000")
        assert error.value.failures
        assert all(
            isinstance(item, VirtualFailureDiagnostic) for item in error.value.failures
        )
        assert {item.point_id for item in error.value.failures} == {1}
        assert all(item.order_index == 0 for item in error.value.failures)
        assert "inspect error.failures" in str(error.value)
