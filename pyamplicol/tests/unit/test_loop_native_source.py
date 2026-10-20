# SPDX-License-Identifier: 0BSD
"""One shared Rust call executes complete source-owned saved plans."""

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from .test_loop_portable import _batches
from .test_loop_source_portable import _forbid_source, source_control  # noqa: F401

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def native_source_control(request):
    """A retained ordinary PACBIN avoids repeating expensive producer setup."""
    retained = os.environ.get("PYAMPLICOL_NATIVE_SOURCE_PACBIN")
    if retained:
        path = Path(retained)
        fixture = (
            Path(__file__).resolve().parents[2]
            / "tools/loop_reference/gluon_scattering_qcd.json"
        )
        return path, np.asarray(json.loads(fixture.read_text())["points"], dtype=str)
    _, path, points = request.getfixturevalue("source_control")
    return path, points


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "2e-12", 20),
        ("double-double", None, "generic", "1e-27", 70),
        ("arbitrary", 70, "generic", "1e-59", 160),
    ],
)
def test_native_source_complete_snapshot_and_rebind(
    loop_native,
    native_source_control,
    monkeypatch,
    arithmetic,
    precision,
    engine,
    tolerance,
    bit,
):
    backend = loop_native
    from pyamplicol.loops.portable import PortableLoopPlan

    path, points = native_source_control
    _forbid_source(monkeypatch)
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    native = backend.NativeLoopProgram(str(path), **options)
    oracle = PortableLoopPlan.open(path).bind(**options)
    state = native.source_state
    initial = state.snapshot()
    assert initial.revision == 0
    assert initial.overrides["MT"] == ["173", "0"]
    assert initial.overrides["aS"] == ["0.118000000000000000000000001", "0"]
    scales = ["1000000", "900000", "2100000"]
    ids = [91, 7, 30]
    actual = native.evaluate(points.tolist(), scales, point_ids=ids)
    expected = oracle.evaluate_ordered_native(points, scales, point_ids=ids)
    assert actual.source_revision == initial.revision
    assert actual.r2_strategy == "model"
    assert actual.point_ids == ids
    assert actual.statuses == [
        [(row, status) for _, status in statuses]
        for row, statuses in enumerate(expected.statuses)
    ]
    for value, reference in zip(_batches(actual), _batches(expected), strict=True):
        assert all(
            ok
            for _, ok in value.agreement_rows(
                reference.take_rows(list(range(reference.rows)), value.row_ids),
                tolerance,
                tolerance,
            )
        )
    assert any(
        Decimal(value[0]) != 0 for value in actual.component_batches[5].to_decimal()
    )
    retained = [value.to_decimal() for value in _batches(actual)]

    with localcontext() as context:
        context.prec = 110
        alpha = str(Decimal(initial.overrides["aS"][0]) * (1 + Decimal(2) ** -bit))
    update = {"aS": [alpha, "0"], "MT": ["190.000000000000000000000001", "0"]}
    current = state.update(update)
    assert current.revision == 1
    assert initial.overrides["MT"] == ["173", "0"]
    # Retained snapshots remain valid after updates. They are not re-read from
    # the live owner between ordinary, local CT, and external-field stages.
    old = native.evaluate(
        points[:1].tolist(), scales[:1], point_ids=ids[:1], snapshot=initial
    )
    assert old.source_revision == 0
    for value, reference in zip(_batches(old), _batches(actual), strict=True):
        assert all(
            ok
            for _, ok in value.agreement_rows(
                reference.take_rows([0]), tolerance, tolerance
            )
        )
    changed = native.evaluate(points[:1].tolist(), scales[:1], point_ids=ids[:1])
    oracle.update_parameters(update)
    target = oracle.evaluate_ordered_native(points[:1], scales[:1], point_ids=ids[:1])
    assert changed.source_revision == 1
    assert changed.born.to_decimal() != actual.born.take_rows([0]).to_decimal()
    with localcontext() as context:
        context.prec = 110
        ratio = Decimal(changed.born.to_decimal()[0][0]) / Decimal(
            actual.born.to_decimal()[0][0]
        )
        exact = (Decimal(alpha) / Decimal(initial.overrides["aS"][0])) ** 2
        assert abs(ratio - exact) < Decimal(tolerance)
    for value, reference in zip(_batches(changed), _batches(target), strict=True):
        assert all(
            ok
            for _, ok in value.agreement_rows(
                reference.take_rows(list(range(reference.rows)), value.row_ids),
                tolerance,
                tolerance,
            )
        )
    assert [value.to_decimal() for value in _batches(actual)] == retained

    rebound = native.rebind(**options, parameters={"MU_R": ["1500", "0"]})
    other = rebound.source_state.snapshot()
    assert other.revision == 0
    assert other.overrides == current.overrides | {"MU_R": ["1500", "0"]}
    state.update({"aS": ["0.12", "0"]})
    assert rebound.source_state.snapshot().overrides == other.overrides
    # Even an empty call rejects foreign owners before numerical work.
    with pytest.raises(ValueError, match=r"different|owner"):
        rebound.evaluate([], [], snapshot=initial)
    assert rebound.evaluate([], []).born.rows == 0
    with pytest.raises(ValueError, match="positional"):
        native.evaluate([], [], model_values=[])

    before = state.snapshot()
    for bad in (
        {"aS": ["0.1", "1e-1000"]},
        {"MB": ["1", "0"]},
        {"unknown": ["1", "0"]},
        {"aS": ["NaN", "0"]},
    ):
        with pytest.raises(ValueError):
            state.update(bad)
        after = state.snapshot()
        assert after.revision == before.revision
        assert after.overrides == before.overrides
        assert after.values.to_decimal() == before.values.to_decimal()
    with pytest.raises(ValueError, match=r"scale|finite|positive"):
        native.evaluate(points[:1].tolist(), ["-1"], point_ids=[59])
    assert native.evaluate([], [], snapshot=before).source_revision == before.revision
