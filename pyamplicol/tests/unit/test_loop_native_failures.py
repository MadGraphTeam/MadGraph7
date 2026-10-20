# SPDX-License-Identifier: 0BSD
"""Actual compact reduction failures preserve physical and resolved identities."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from .test_loop_artifact_writer import source_process, target_metadata, write_artifact

pytestmark = pytest.mark.loop_physics

__all__ = ["target_metadata"]

NEAR = [
    ["5000000000.5", "0", "0", "5000000000.5"],
    ["5000000000.5", "0", "0", "-5000000000.5"],
    ["5000000000.5", "100000", "0", "4999999999.5"],
    ["5000000000.5", "-100000", "0", "-4999999999.5"],
]


@pytest.fixture(scope="module")
def source_plan(saved_loop_fixture, loop_native):
    source = str(saved_loop_fixture("source-portable-v3/gg-resolved.pacbin"))
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    return Path(source), json.loads(
        fixture.read_text(), parse_float=str, parse_int=str
    )["points"]


@pytest.mark.parametrize("resolved", [False, True])
def test_actual_reduction_failure_compacts_whole_points(
    loop_native, source_plan, resolved
):
    native = loop_native

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
    assert result.point_ids == [7, 4]
    assert result.born.row_ids == list(range(2 * h))
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


@pytest.mark.parametrize("arithmetic", ["double", "double-double"])
@pytest.mark.parametrize("resolved", [False, True])
def test_early_state_and_scale_failures_keep_exact_original_ids(
    loop_native, source_plan, arithmetic, resolved
):
    native = loop_native

    path, points = source_plan
    runtime = native.NativeLoopProgram(
        str(path),
        arithmetic=arithmetic,
        reduction_engine="fast" if arithmetic == "double" else "generic",
    )
    h = len(json.loads(runtime.metadata_json())["helicities"]) if resolved else 1
    huge = [[str(int(x) * 10**400) for x in leg] for leg in points[0]]
    for batch, scales, stage in (
        ([points[0], huge, points[1]], ["1000000"] * 3, "external state preparation"),
        (
            [points[0], points[0], points[1]],
            ["1000000", "1e400", "1000000"],
            "renormalization scales",
        ),
        (
            [points[0], points[0], points[1]],
            ["1000000", "1e-400", "1000000"],
            "positive real renormalization scales",
        ),
    ):
        result = runtime.evaluate(
            batch, scales, point_ids=[7, 91, 4], resolved=resolved, partial=True
        )
        assert result.requested_point_ids == [7, 91, 4]
        assert result.point_ids == [7, 4]
        assert result.born.row_ids == list(range(2 * h))
        assert result.failures and {f.point_id for f in result.failures} == {91}
        assert all(stage in f.stage for f in result.failures)
        assert all(
            f.order_index is None and f.family_index is None for f in result.failures
        )
        if stage == "external state preparation":
            assert all(
                f.external_leg in range(4) and f.helicity in range(16)
                for f in result.failures
            )
        else:
            assert all(f.external_leg is f.helicity is None for f in result.failures)
        expected = runtime.evaluate(
            [points[0], points[1]], ["1000000"] * 2, point_ids=[7, 4], resolved=resolved
        )
        for actual, reference in zip(
            [result.born, *result.component_batches],
            [expected.born, *expected.component_batches],
            strict=True,
        ):
            assert all(
                ok for _, ok in actual.agreement_rows(reference, "1e-25", "1e-50")
            )
        failed = runtime.evaluate(
            [batch[1]], [scales[1]], point_ids=[91], resolved=resolved, partial=True
        )
        assert failed.point_ids == [] and failed.totals.rows == 0 and failed.failures
    # Every original literal is checked, including a row whose state fails.
    for bad_scale in ("NaN", "-1", "0"):
        with pytest.raises(ValueError):
            runtime.evaluate([huge], [bad_scale], partial=True)
    invalid = [[list(leg) for leg in huge]]
    invalid[0][0][0] = "NaN"
    with pytest.raises(ValueError):
        runtime.evaluate(invalid, ["1000000"], partial=True)


@pytest.mark.parametrize("resolved", [False, True])
def test_native_replay_preserves_failures_from_distinct_stages(
    loop_native, source_plan, tmp_path, resolved
):
    import io

    native = loop_native
    from symbolica import E, S

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )
    from pyamplicol.loops.physical import _saved

    path, points = source_plan
    plan_member = "schedule/loop-runtime-v3.json"
    with PacbinReader.open(path) as reader:
        member = next(m for m in reader.members if m.logical_path == plan_member)
        plan = json.loads(reader.read_member(plan_member, length=member.length))
        # Explicit test-only fault: one CT schedule divides by E_in - 500.
        # It fails after reduction at the first ordinary point, while the
        # distinct E_in=650 point survives and supplies a separate-run oracle.
        energy_column = plan["bound_positions"]["loop_recurrence::p0_0"]
        current = next(
            c for c in plan["currents"] if energy_column in c["input_columns"]
        )
        schedule = plan["schedules"][current["schedule"]]
        count = schedule["parameter_count"]
        width = len(schedule["output_slots"])
        parameters = [S(f"stage_failure_test::x{i}") for i in range(count)]
        energy = parameters[current["input_columns"].index(energy_column)]
        faulty = _saved([E("1") / (energy - 500), *[E("0")] * (width - 1)], parameters)
        state = "evaluators/injected-stage-failure.state"
        schedule.update(
            states=[state],
            input_slots=[list(range(count))],
            output_slots=list(range(count, count + width)),
        )
        members = [
            PacbinMemberSource(
                m.logical_path,
                m.kind,
                io.BytesIO(
                    json.dumps(plan).encode()
                    if m.logical_path == plan_member
                    else reader.read_member(m.logical_path, length=m.length)
                ),
            )
            for m in reader.members
        ]
        members.append(
            PacbinMemberSource(
                state, PacbinMemberKind.SYMBOLICA_EXACT_STATE, io.BytesIO(faulty)
            )
        )
    destination = tmp_path / "injected-stage-failure.pacbin"
    write_pacbin_atomic(destination, members)
    runtime = native.NativeLoopProgram(str(destination), reduction_engine="generic")
    snapshot = runtime.source_state.snapshot()
    runtime.source_state.update({"aS": ["0.12", "0"]})
    huge = [[str(int(x) * 10**400) for x in leg] for leg in points[0]]
    result = runtime.evaluate(
        [points[0], NEAR, points[1], huge, points[0]],
        ["1000000"] * 4 + ["1e400"],
        point_ids=[2**64 - 1, 2**63, 2**64 - 2, 0, 91],
        resolved=resolved,
        partial=True,
        snapshot=snapshot,
    )
    assert result.source_revision == snapshot.revision == 0
    assert runtime.source_state.snapshot().revision == 1
    assert result.requested_point_ids == [2**64 - 1, 2**63, 2**64 - 2, 0, 91]
    h = len(plan["helicities"]) if resolved else 1
    assert result.point_ids == [2**64 - 2]
    assert result.born.row_ids == list(range(h))
    by_point = {
        p: {f.stage for f in result.failures if f.point_id == p}
        for p in (2**64 - 1, 2**63, 0, 91)
    }
    assert "counterterm current" in by_point[2**64 - 1]
    assert "ordinary numerator reduction" in by_point[2**63]
    assert "external state preparation" in by_point[0]
    assert "renormalization scales" in by_point[91]
    expected = runtime.evaluate(
        [points[1]],
        ["1000000"],
        point_ids=[2**64 - 2],
        resolved=resolved,
        snapshot=snapshot,
    )
    for a, b in zip(
        [result.born, *result.component_batches],
        [expected.born, *expected.component_batches],
        strict=True,
    ):
        assert all(ok for _, ok in a.agreement_rows(b, "1e-25", "1e-50"))
    assert all(status == "success" for row in result.statuses for _, status in row)
