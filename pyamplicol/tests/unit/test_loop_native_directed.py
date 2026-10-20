# SPDX-License-Identifier: 0BSD
"""One native call evaluates saved complex histories and exact-input rescue."""

import io
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from .test_loop_native_failures import NEAR
from .test_loop_physical_correlations import _close, _conjugate, _decimals
from .test_loop_physical_counterterms import _uudd_born_uv

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def directed_plan(saved_loop_fixture, loop_native):
    path = str(saved_loop_fixture("source-portable-directed/uudd-source.pacbin"))
    fixture = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    return Path(path), json.loads(fixture.read_text(), parse_float=str, parse_int=str)[
        "points"
    ][:2]


def _blocks(result):
    return [
        result.born,
        *result.component_batches,
        result.directed_first,
        result.directed_second,
        result.totals,
    ]


def _values(batch, shape):
    return _decimals(np.asarray(batch.to_decimal(), dtype=object).reshape(*shape, 2))


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "4e-11", 20),
        ("double-double", None, "generic", "4e-25", 70),
        ("arbitrary", 70, "generic", "4e-57", 170),
    ],
)
def test_native_directed_complex_components_exact_owner_and_mode(
    loop_native,
    directed_plan,
    monkeypatch,
    arithmetic,
    precision,
    engine,
    tolerance,
    bit,
):
    native = loop_native

    from .test_loop_source_portable import _forbid_source

    path, points = directed_plan
    _forbid_source(monkeypatch)
    runtime = native.NativeLoopProgram(
        str(path), arithmetic=arithmetic, precision=precision, reduction_engine=engine
    )
    metadata = json.loads(runtime.metadata_json())
    b, v = (len(metadata["order_layout"][key]) for key in ("born", "virtual"))
    ids = [2**64 - 1, 7]
    scales = ["1000000", "900000"]
    snapshot = runtime.source_state.snapshot()
    ordinary = runtime.evaluate(points, scales, point_ids=ids)
    result = runtime.evaluate_directed(points, scales, point_ids=ids)
    assert ordinary.observable_kind == "ordinary-interference"
    assert ordinary.directed_first is ordinary.directed_second is None
    assert ordinary.history_ids == []
    assert result.observable_kind == "directed-complex"
    assert result.history_ids == ["identity", "casimir", "ordered", "swapped"]
    assert result.point_ids == result.requested_point_ids == ids
    assert result.source_revision == snapshot.revision
    assert result.totals.columns == 100
    assert all(block.row_ids == [0, 1] for block in _blocks(result))
    assert all(
        status_id == row and status == "success"
        for row, statuses in enumerate(result.statuses)
        for status_id, status in statuses
    )
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        born = _values(result.born, (2, 4, b))
        values = [_values(block, (2, 4, v, 3)) for block in result.component_batches]
        first, second = (
            _values(block, (2, 4, v, 3))
            for block in (result.directed_first, result.directed_second)
        )
        for actual, reference in zip(
            [born, *values], [ordinary.born, *ordinary.component_batches], strict=True
        ):
            _close(actual[:, 0], _values(reference, actual[:, 0].shape[:-1]), tolerance)
            _close(actual[:, 1], actual[:, 0] * Decimal(4) / 3, tolerance)
            _close(actual[:, 3], _conjugate(actual[:, 2]), tolerance)
        _close(values[0], first + second, tolerance)
        _close(first[:, 2], _conjugate(second[:, 3]), tolerance)
        _close(values[0], values[1] + values[2] + values[3], tolerance)
        assert abs(values[0][0, 2, 0, 0, 1]) > Decimal("1e-4")
        summed = _values(result.totals, (2, 4, 25))
        _close(summed[:, :, 0], born.sum(axis=2), tolerance)
        for index, block in enumerate([*values, first, second]):
            _close(
                summed[:, :, 1 + 3 * index : 4 + 3 * index],
                block.sum(axis=2),
                tolerance,
            )
        for index, (point, scale) in enumerate(zip(points, scales, strict=True)):
            expected = _uudd_born_uv(point, scale)
            actual = [
                born[index, 0, 0, 0],
                values[2][index, 0, 0, 0, 0],
                values[2][index, 0, 0, 1, 0],
            ]
            _close(actual, expected, tolerance)
        alpha = Decimal(snapshot.overrides["aS"][0])
        updated_alpha = alpha * (1 + Decimal(2) ** -bit)
        updated = runtime.source_state.update({"aS": [str(updated_alpha), "0"]})
        current = runtime.evaluate_directed(points[:1], scales[:1], point_ids=ids[:1])
        assert current.source_revision == updated.revision
        for block, original, power in zip(
            [
                current.born,
                *current.component_batches,
                current.directed_first,
                current.directed_second,
            ],
            [
                result.born,
                *result.component_batches,
                result.directed_first,
                result.directed_second,
            ],
            [2, *([3] * 8)],
            strict=True,
        ):
            _close(
                _values(block, (1, block.columns)),
                _values(original.take_rows([0]), (1, original.columns))
                * (updated_alpha / alpha) ** power,
                tolerance,
            )
        assert current.born.to_decimal() != result.born.take_rows([0]).to_decimal()
    old = runtime.evaluate_directed(
        points[:1], scales[:1], point_ids=ids[:1], snapshot=snapshot
    )
    assert old.source_revision == snapshot.revision
    assert (
        old.directed_first.to_decimal()
        == result.directed_first.take_rows([0]).to_decimal()
    )
    empty = runtime.evaluate_directed([], [], partial=True)
    assert empty.point_ids == empty.failures == empty.statuses == []
    assert all(batch.rows == 0 for batch in _blocks(empty))
    assert empty.history_ids == result.history_ids


def test_directed_partial_failure_channels_and_replay(loop_native, directed_plan):
    native = loop_native

    path, points = directed_plan
    runtime = native.NativeLoopProgram(str(path), reduction_engine="generic")
    ids = [2**64 - 1, 2**63, 0, 91]
    batch = [points[0], points[0], NEAR, points[1]]
    scales = ["1000000", "1e-400", "1000000", "900000"]
    result = runtime.evaluate_directed(batch, scales, point_ids=ids, partial=True)
    assert result.point_ids == [ids[0], ids[3]]
    assert result.requested_point_ids == ids
    assert {failure.point_id for failure in result.failures} == {ids[1], ids[2]}
    reductions = [
        failure for failure in result.failures if failure.family_index is not None
    ]
    assert reductions
    assert all(
        f.point_id == ids[2]
        and f.history_index in range(4)
        and f.direction in (0, 1)
        and f.order_index == 0
        for f in reductions
    )
    assert all(f.helicity is f.color_component_index is None for f in reductions)
    assert all(
        status_id == row
        for row, statuses in enumerate(result.statuses)
        for status_id, _ in statuses
    )
    reference = runtime.evaluate_directed(
        points, ["1000000", "900000"], point_ids=[ids[0], ids[3]]
    )
    for actual, expected in zip(_blocks(result), _blocks(reference), strict=True):
        assert all(ok for _, ok in actual.agreement_rows(expected, "1e-10", "1e-12"))
    with pytest.raises(ValueError, match="unrecovered loop numerical channels"):
        runtime.evaluate_directed(batch, scales, point_ids=ids)
    failed = runtime.evaluate_directed(
        [NEAR], ["1000000"], point_ids=[ids[2]], partial=True
    )
    assert failed.point_ids == [] and failed.statuses == [] and failed.failures
    assert all(block.rows == 0 for block in _blocks(failed))


def test_directed_subset_rescue_keeps_every_complex_cross_term(
    loop_native, directed_plan
):
    native = loop_native

    path, points = directed_plan
    runtime = native.NativeLoopProgram(str(path), reduction_engine="fast")
    ids, scales = [2**64 - 1, 0], ["1e-400", "900000"]
    verified = runtime.evaluate_directed_verified(
        points,
        scales,
        [("arbitrary", 60), ("arbitrary", 80)],
        "1e-8",
        "1e-9",
        point_ids=ids,
    )
    result = verified.result
    assert result.point_ids == ids and result.failures == []
    diagnostics = json.loads(verified.diagnostics_json)
    assert [row["status"] for row in diagnostics] == ["rescued", "success"]
    assert diagnostics[0]["attempts"][-1]["agreement"]
    assert result.directed_first.precision_bits == 53
    reference = runtime.rebind(arithmetic="arbitrary", precision=80).evaluate_directed(
        points[:1], scales[:1], point_ids=ids[:1]
    )
    for actual, expected in zip(_blocks(result), _blocks(reference), strict=True):
        assert all(
            ok
            for _, ok in actual.take_rows([0]).agreement_rows(expected, "1e-8", "1e-9")
        )
    empty = runtime.evaluate_directed_verified(
        [], [], [("arbitrary", 60)], "1e-8", "1e-9"
    )
    assert empty.result.point_ids == [] and json.loads(empty.diagnostics_json) == []
    assert empty.result.history_ids == result.history_ids


def test_saved_history_labels_select_their_complex_output_columns(
    loop_native, directed_plan, tmp_path
):
    """A distinguishable saved assembly checks labels, not correlated physics.

    Swapping the ordered/swapped columns or a first/second output block fails
    this control even though conjugation-only physical checks might still pass.
    The original currents/reducer execute; only final assembly bytes are replaced.
    """
    native = loop_native
    from symbolica import E, S

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )
    from pyamplicol.loops.physical import _saved

    path, points = directed_plan
    plan = json.loads(native.PortableLoopPlan(str(path)).metadata_json())
    histories = plan["directed_projection"]["histories"]
    sectors = len(plan["order_layout"]["virtual"])
    width = len(histories) * sectors * 3
    pairs = [
        (
            1000 * block + 100 * history + 10 * order + pole + 1,
            -1000 * block - 100 * history - 10 * order - pole - 7,
        )
        for block in range(7)
        for history in range(len(histories))
        for order in range(sectors)
        for pole in range(3)
    ]
    parameters = [S(f"directed_column_control::input{i}") for i in range(8 * width)]
    state = _saved([E(str(re)) + E("1i") * E(str(im)) for re, im in pairs], parameters)
    destination = tmp_path / "labeled-assembly.pacbin"
    with PacbinReader.open(path) as reader:
        members = [
            PacbinMemberSource(
                member.logical_path,
                member.kind,
                io.BytesIO(
                    state
                    if member.logical_path == plan["directed_projection"]["assemble"]
                    else reader.read_member(member.logical_path, length=member.length)
                ),
            )
            for member in reader.members
        ]
    write_pacbin_atomic(destination, members)
    runtime = native.NativeLoopProgram(str(destination), reduction_engine="fast")
    result = runtime.evaluate_directed(points[:1], ["1000000"], point_ids=[2**64 - 1])
    assert result.history_ids == [item["id"] for item in histories]
    outputs = [result.component_batches[index] for index in (0, 2, 3, 4, 5)]
    outputs += [result.directed_first, result.directed_second]
    for block, output in enumerate(outputs):
        actual = [(Decimal(re), Decimal(im)) for re, im in output.to_decimal()]
        assert actual == pairs[block * width : (block + 1) * width]
