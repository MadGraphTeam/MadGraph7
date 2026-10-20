# SPDX-License-Identifier: 0BSD
"""Saved signed Born anchors preserve physical, helicity and order identities."""

import io
import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_native_failures import NEAR

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def component_plan(saved_loop_fixture, loop_native):
    source = str(saved_loop_fixture("source-colour-components/gg-lc.pacbin"))
    expected = str(saved_loop_fixture("source-colour-components/gg-lc-expected.json"))
    return Path(source), json.loads(Path(expected).read_text())


def _batches(result):
    return (result.born, *result.component_batches)


def _close(actual, expected, tolerance):
    with localcontext() as context:
        context.prec = 100
        actual, expected = Decimal(actual), Decimal(expected)
        assert abs(actual - expected) <= Decimal(tolerance) * max(
            abs(expected), Decimal(1)
        )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "3e-10"),
        ("double-double", None, "generic", "3e-24"),
        ("arbitrary", 70, "generic", "1e-58"),
    ],
)
def test_actual_component_axes_and_all_laurent_sums(
    loop_native, component_plan, monkeypatch, arithmetic, precision, engine, tolerance
):
    native = loop_native

    from .test_loop_source_portable import _forbid_source

    path, control = component_plan
    _forbid_source(monkeypatch)
    runtime = native.NativeLoopProgram(
        str(path), arithmetic=arithmetic, precision=precision, reduction_engine=engine
    )
    plan = json.loads(runtime.metadata_json())
    h = len(plan["helicities"])
    c = len(plan["born_color_components"]["ids"])
    assert c > 1 and h > 1
    assert plan["born_color_components"] == control["components"]
    points, scales = control["points"], control["scales"]
    inclusive = runtime.evaluate(points, scales, point_ids=[7, 4])
    resolved = runtime.evaluate(points, scales, point_ids=[7, 4], resolved=True)
    assert resolved.point_ids == [7, 4]
    assert resolved.born.row_ids == list(range(2 * h * c))
    assert resolved.requested_point_ids == [7, 4]
    with localcontext() as context:
        context.prec = 100
        for name, single, summed in zip(
            control["expected"], _batches(resolved), _batches(inclusive), strict=True
        ):
            width = single.columns
            actual, reference = single.to_decimal(), summed.to_decimal()
            assert single.rows == 2 * h * c and summed.rows == 2
            for point in range(2):
                for column in range(width):
                    # Sum each order/pole independently; never flatten axes into
                    # a scalar checksum that could hide a transposition.
                    for part in range(2):
                        value = sum(
                            Decimal(
                                actual[((point * h + hi) * c + ci) * width + column][
                                    part
                                ]
                            )
                            for hi in range(h)
                            for ci in range(c)
                        )
                        _close(
                            value, reference[point * width + column][part], tolerance
                        )
                    assert Decimal(reference[point * width + column][1]) == 0
                    # Independent retained producer values use binary64; the
                    # high-precision reconstruction above has its own tighter
                    # arithmetic-specific tolerance.
                    _close(
                        reference[point * width + column][0],
                        control["expected"][name][point * width + column][0],
                        "3e-10",
                    )
    empty = runtime.evaluate([], [], resolved=True)
    assert empty.point_ids == [] and empty.born.rows == 0
    assert empty.born.columns == inclusive.born.columns


def test_component_reduction_and_early_state_failure_ids(loop_native, component_plan):
    native = loop_native

    path, control = component_plan
    runtime = native.NativeLoopProgram(str(path), reduction_engine="generic")
    plan = json.loads(runtime.metadata_json())
    h, c = len(plan["helicities"]), len(plan["born_color_components"]["ids"])
    points = control["points"]
    success_ids = [2**64 - 1, 4]
    failed_id = 2**64 - 2
    huge = [[str(int(x) * 10**400) for x in leg] for leg in points[0]]
    expected = runtime.evaluate(
        points, ["1000000"] * 2, point_ids=success_ids, resolved=True
    )
    for bad, stage in (
        (NEAR, "ordinary numerator reduction"),
        (huge, "external state preparation"),
    ):
        result = runtime.evaluate(
            [points[0], bad, points[1]],
            ["1000000"] * 3,
            point_ids=[success_ids[0], failed_id, success_ids[1]],
            resolved=True,
            partial=True,
        )
        assert result.requested_point_ids == [success_ids[0], failed_id, success_ids[1]]
        assert result.point_ids == expected.point_ids == success_ids
        assert result.born.row_ids == list(range(2 * h * c))
        assert all(
            status_id == row
            for row, statuses in enumerate(result.statuses)
            for status_id, _ in statuses
        )
        assert result.failures and {f.point_id for f in result.failures} == {failed_id}
        for failure in result.failures:
            assert failure.stage == stage
            assert failure.helicity in range(h)
            if stage == "ordinary numerator reduction":
                assert failure.color_component_index in range(c)
                assert failure.order_index == 0
                assert failure.family_index in {f["index"] for f in plan["families"]}
            else:
                # States are evaluated once per physical point/helicity, before
                # anchor expansion. Dividing this ID by H*C would be incorrect.
                assert failure.color_component_index is None
                assert failure.order_index is failure.family_index is None
                assert failure.external_leg in range(4)
        for actual, reference in zip(_batches(result), _batches(expected), strict=True):
            assert all(
                ok for _, ok in actual.agreement_rows(reference, "1e-10", "1e-12")
            )
        failed = runtime.evaluate(
            [bad], ["1000000"], point_ids=[failed_id], resolved=True, partial=True
        )
        assert failed.requested_point_ids == [failed_id] and failed.point_ids == []
        assert failed.born.rows == failed.totals.rows == 0 and failed.failures
        with pytest.raises(ValueError, match=f"point_id: {failed_id}"):
            runtime.evaluate([bad], ["1000000"], point_ids=[failed_id], resolved=True)

    with pytest.raises(ValueError, match="identities are not unique"):
        runtime.evaluate(points, ["1000000"] * 2, point_ids=[failed_id] * 2)


def test_component_subset_rescue_preserves_opaque_ids(loop_native, component_plan):
    native = loop_native

    path, control = component_plan
    runtime = native.NativeLoopProgram(str(path), reduction_engine="fast")
    ids = [2**64 - 1, 0]
    scales = ["1e-400", "1000000"]
    verified = runtime.evaluate_verified(
        control["points"],
        scales,
        [("arbitrary", 60), ("arbitrary", 80)],
        "1e-8",
        "1e-9",
        point_ids=ids,
        resolved=True,
    )
    result = verified.result
    assert result.point_ids == result.requested_point_ids == ids
    assert not result.failures
    diagnostics = json.loads(verified.diagnostics_json)
    assert [entry["point_id"] for entry in diagnostics] == ids
    assert [entry["status"] for entry in diagnostics] == ["rescued", "success"]
    assert {
        entry["point_id"] for entry in diagnostics[0]["attempts"][0]["failures"]
    } == {ids[0]}
    reference = runtime.rebind(arithmetic="arbitrary", precision=80).evaluate(
        control["points"][:1], scales[:1], point_ids=ids[:1], resolved=True
    )
    rows = reference.born.rows
    assert result.born.row_ids == list(range(2 * rows))
    assert len(result.statuses) == 2 * rows
    assert all(
        status_id == row
        for row, statuses in enumerate(result.statuses)
        for status_id, _ in statuses
    )
    assert all(status == "success" for row in result.statuses for _, status in row)
    for actual, expected in zip(_batches(result), _batches(reference), strict=True):
        assert all(
            ok
            for _, ok in actual.take_rows(list(range(rows))).agreement_rows(
                expected, "1e-8", "1e-9"
            )
        )


def test_one_component_projection_failure_keeps_component_identity(
    loop_native, component_plan, tmp_path
):
    native = loop_native
    from symbolica import E, S

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )
    from pyamplicol.loops.physical import _saved

    path, control = component_plan
    plan = json.loads(native.PortableLoopPlan(str(path)).metadata_json())
    width = sum(
        len(plan["schedules"][i]["output_slots"]) for i in plan["born"]["schedules"]
    )
    c = len(plan["born_color_components"]["ids"])
    sectors = len(plan["order_layout"]["born"])
    parameters = [S(f"component_failure_test::born{i}") for i in range(width)]
    outputs = [E("0")] * (c * sectors)
    outputs[(c - 1) * sectors] = E("1") / parameters[0]
    faulty = _saved(outputs, parameters)
    state = plan["helicity_projection"]["born"]
    destination = tmp_path / "one-component-failure.pacbin"
    with PacbinReader.open(path) as reader:
        members = [
            PacbinMemberSource(
                member.logical_path,
                member.kind,
                io.BytesIO(
                    faulty
                    if member.logical_path == state
                    else reader.read_member(member.logical_path, length=member.length)
                ),
            )
            for member in reader.members
        ]
    write_pacbin_atomic(destination, members)
    runtime = native.NativeLoopProgram(
        str(destination), reduction_engine="fast", parameters={"aS": ["0", "0"]}
    )
    result = runtime.evaluate(
        control["points"][:1],
        ["1000000"],
        point_ids=[2**64 - 1],
        resolved=True,
        partial=True,
    )
    assert result.point_ids == [] and result.born.rows == 0
    assert result.failures and {f.point_id for f in result.failures} == {2**64 - 1}
    assert all(
        f.stage == "Born projection"
        and f.helicity in range(len(plan["helicities"]))
        and f.color_component_index == c - 1
        for f in result.failures
    )


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda p: p["born_color_components"]["members"].append([0]), "declaration"),
        (lambda p: p["born_color_components"]["ids"].pop(), "declaration"),
        (lambda p: p["born_color_components"]["members"][0].append(900), "partition"),
        (lambda p: p["color_projection"].update(accuracy="full"), "declaration"),
    ],
)
def test_saved_partition_is_checked_before_execution(
    loop_native, component_plan, tmp_path, mutation, message
):
    native = loop_native

    from .test_loop_portable import _rewrite

    path, _ = component_plan
    destination = tmp_path / "malformed.pacbin"
    _rewrite(path, destination, mutation)
    with pytest.raises(ValueError, match=message):
        native.PortableLoopPlan(str(destination))
