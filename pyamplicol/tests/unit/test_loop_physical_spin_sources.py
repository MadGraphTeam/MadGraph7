# SPDX-License-Identifier: 0BSD
"""Complete source-vector observables with ordered complex colour histories."""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.color.connections import ColorConnection, EmitGluon

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def source(loop_sm_ufo, loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed dependency generation required")
    path = str(loop_sm_ufo)
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    return build_loop_model(load_raw_loop_model(path))


def _values(values):
    return np.asarray([Decimal(str(x)) for x in values.flat], dtype=object).reshape(
        values.shape
    )


def _close(left, right, tolerance):
    left, right = np.asarray(left, dtype=object), np.asarray(right, dtype=object)
    assert left.shape == right.shape
    assert all(
        abs(a - b) < tolerance * max(Decimal(1), abs(b))
        for a, b in zip(left.flat, right.flat, strict=True)
    )


def _conjugate(values):
    result = values.copy()
    result[..., 1] *= -1
    return result


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,exponent",
    [
        ("double", None, "fast", Decimal("8e-10"), -30),
        ("double-double", None, "generic", Decimal("3e-23"), -65),
        ("arbitrary", 60, "generic", Decimal("3e-48"), -120),
    ],
)
def test_complete_joint_sources_polarization_sum_complex_scaling_and_ward(
    source, arithmetic, precision, engine, tolerance, exponent
):
    from pyamplicol.loops.physical_correlations import directed_color_legs
    from pyamplicol.loops.physical_counterterms import quark_gluon_model_virtual_program
    from pyamplicol.loops.spin_sources import NativeSpinSourceProgram
    from pyamplicol.models import BuiltinSMModel

    model = quark_gluon_model_virtual_program(
        source,
        source_contracts={21: BuiltinSMModel()._source_ir(21)},
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
        coefficient_engine="jit" if arithmetic == "double" else "interpreted",
    )
    legs = directed_color_legs(model.bare)
    identity = ColorConnection(legs)
    charged = ColorConnection(legs, (EmitGluon(2, -1),))
    alpha = ColorConnection(legs, (EmitGluon(0, -1),))
    beta = ColorConnection(legs, (EmitGluon(1, -1),))
    program = NativeSpinSourceProgram(
        model,
        ((identity, identity), (charged, charged), (alpha, beta), (beta, alpha)),
        spin_correlations=((3,), (3, 4)),
    )
    path = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/quark_gluon_qcd.json"
    )
    point = np.asarray(json.loads(path.read_text())["points"], dtype=str)[:1]
    options = {"model_values": ["0.118", "173"]}
    baseline = model.evaluate_ordered_decimal(point, ["1000000"], **options)
    restored = program.evaluate_decimal(point, ["1000000"], **options)
    names = ("born", "virtual", "bare", "uv", "r2", "uv_local", "uv_external_fields")
    with localcontext() as context:
        context.prec = 110
        for name in names:
            _close(
                _values(getattr(restored, name)[:, 0]),
                _values(getattr(baseline, name)),
                tolerance,
            )
        assert restored.spin_legs == ()
        a = [
            (Decimal(0), Decimal(0)),
            (Decimal("0.8"), Decimal(0)),
            (Decimal(0), Decimal(0)),
            (Decimal("-0.6"), Decimal(0)),
        ]
        b = [
            (Decimal(0), Decimal(0)),
            (Decimal(0), Decimal(0)),
            (Decimal(1), Decimal(0)),
            (Decimal(0), Decimal(0)),
        ]
        v = [(x[0], y[0]) for x, y in zip(a, b, strict=True)]

        def scale(vector, real, imag):
            return [(real * x - imag * y, real * y + imag * x) for x, y in vector]

        def shift(vector, momentum):
            return [
                (x + Decimal(p) / 1000, y + Decimal(p) / 500)
                for (x, y), p in zip(vector, momentum, strict=True)
            ]

        tiny = Decimal(2) ** exponent
        vectors = [
            v,
            scale(v, 2, -3),
            shift(v, point[0, 2]),
            a,
            b,
            scale(v, 1 + tiny, 0),
        ]
        single = program.evaluate_decimal(
            np.repeat(point, len(vectors), axis=0),
            ["1000000"] * len(vectors),
            spin_vectors={3: vectors},
            **options,
        )
        assert single.spin_legs == (3,)
        for name in names:
            value = _values(getattr(single, name))
            _close(value[1], 13 * value[0], tolerance)
            _close(value[3] + value[4], _values(getattr(restored, name)[0]), tolerance)
            _close(value[:, 1], 3 * value[:, 0], tolerance)
            _close(value[:, 3], _conjugate(value[:, 2]), tolerance)
        # A gauge shift preserves the complete complex observable with a nonzero
        # Born vector; this cannot pass merely because a Ward-replaced Born is zero.
        for name in ("born", "virtual", "first", "second"):
            value = _values(getattr(single, name))
            _close(value[2], value[0], tolerance)
        born = _values(single.born)
        delta = born[5, 0, 0, 0] - born[0, 0, 0, 0]
        expected = born[0, 0, 0, 0] * (2 * tiny + tiny * tiny)
        assert abs(delta - expected) < abs(expected) * Decimal("1e-4")
        assert abs(Decimal(single.first[0, 2, 0, 0, 1])) > Decimal("1e-6")

        first = [v, scale(v, 2, -3), shift(v, point[0, 2]), a, a, b, b]
        second = [v, scale(v, 1, 2), v, a, b, a, b]
        joint = program.evaluate_decimal(
            np.repeat(point, len(first), axis=0),
            ["1000000"] * len(first),
            spin_vectors={3: first, 4: second},
            **options,
        )
        assert joint.spin_legs == (3, 4)
        for name in names:
            value = _values(getattr(joint, name))
            _close(value[1], 65 * value[0], tolerance)
            _close(
                sum(value[i] for i in range(3, 7)),
                _values(getattr(restored, name)[0]),
                tolerance,
            )
        for name in ("born", "virtual", "first", "second"):
            value = _values(getattr(joint, name))
            _close(value[2], value[0], tolerance)
        # Actual layout sizes remain immutable, including after an empty call.
        assert len(program._views[(3,)].helicities) == 8
        assert len(program._views[(3, 4)].helicities) == 4
        assert len(model.bare.helicities) == 16
        empty = program.evaluate_decimal(point[:0], [], spin_vectors={3: v}, **options)
        assert empty.virtual.shape == (0, 4, 1, 3, 2) and empty.statuses == ()
        native_sources = program.sources.prepare_decimal({3: v, 4: v}, [])
        native_empty = program.evaluate_native(
            point[:0], [], spin_vectors=native_sources, **options
        )
        assert native_empty.batch.helicity_count == 4
        assert native_empty.batch.rows == 0 and native_empty.virtual.rows == 0
        full_empty = model.bare.prepare_currents_native(point[:0], [], **options)
        with pytest.raises(ValueError, match="different helicity/source layout"):
            program._views[(3,)].evaluate_currents_native(full_empty)
        for i, row in enumerate(joint.statuses):
            for (lane, status), _channel in zip(row, joint.channels, strict=True):
                assert lane == i and status == "success"
