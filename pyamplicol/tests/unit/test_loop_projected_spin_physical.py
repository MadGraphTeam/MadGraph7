# SPDX-License-Identifier: 0BSD
"""Reusable controls over an actual retained formally projected source owner.

The campaign supplies its already generated owner; these controls never regenerate
or manufacture a colour projection. Polarization completeness compares with the
existing ordinary complete evaluator, including each graded counterterm piece.
"""

from decimal import Decimal, localcontext

import numpy as np

from pyamplicol.color.connections import ColorConnection
from pyamplicol.loops.physical_correlations import directed_color_legs
from pyamplicol.loops.spin_sources import NativeSpinSourceProgram

from .test_loop_physical_spin_sources import _close, _values


def assert_projected_identity_components(
    source, *, arithmetic="double", precision=None, tolerance="8e-9"
):
    rebound = source.rebind(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
    )
    model = rebound.virtual
    bare = model.bare
    assert model.color_projection is not None
    degree_snapshot = (bare.born_color_degrees, bare.virtual_color_degrees)
    identity = ColorConnection(directed_color_legs(bare))
    spin = NativeSpinSourceProgram(
        model, ((identity, identity),), spin_correlations=((3,), (3, 4))
    )
    point = np.asarray(
        [
            [
                ["500", "0", "0", "500"],
                ["500", "0", "0", "-500"],
                ["500", "300", "0", "400"],
                ["500", "-300", "0", "-400"],
            ]
        ]
    )
    scale = "1000000"
    names = ("born", "virtual", "bare", "uv", "r2", "uv_local", "uv_external_fields")
    baseline = model.evaluate_ordered_decimal(point, [scale])
    ordinary = spin.evaluate_decimal(point, [scale], model_values=None)
    a = [["0", "0"], ["0.8", "0"], ["0", "0"], ["-0.6", "0"]]
    b = [["0", "0"], ["0", "0"], ["1", "0"], ["0", "0"]]
    single = spin.evaluate_decimal(
        np.repeat(point, 2, axis=0),
        [scale] * 2,
        model_values=None,
        spin_vectors={3: [a, b]},
    )
    joint = spin.evaluate_decimal(
        np.repeat(point, 4, axis=0),
        [scale] * 4,
        model_values=None,
        spin_vectors={3: [a, a, b, b], 4: [a, b, a, b]},
    )
    with localcontext() as context:
        context.prec = 110
        tol = Decimal(tolerance)
        for name in names:
            target = _values(getattr(baseline, name))
            _close(_values(getattr(ordinary, name))[:, 0], target, tol)
            _close(_values(getattr(single, name))[:, 0].sum(axis=0), target[0], tol)
            _close(_values(getattr(joint, name))[:, 0].sum(axis=0), target[0], tol)
        for name in ("first", "second"):
            target = _values(getattr(ordinary, name))[0, 0]
            _close(_values(getattr(single, name))[:, 0].sum(axis=0), target, tol)
            _close(_values(getattr(joint, name))[:, 0].sum(axis=0), target, tol)
        scaled = [[str(2 * Decimal(x)), str(-3 * Decimal(x))] for x, _ in a]
        zero = [["0", "0"]] * 4
        literal = spin.evaluate_decimal(
            np.repeat(point, 3, axis=0),
            [scale] * 3,
            model_values=None,
            spin_vectors={3: [a, scaled, zero], 4: b},
        )
        for name in (*names, "first", "second"):
            values = _values(getattr(literal, name))
            _close(values[1], 13 * values[0], tol)
            _close(values[2], np.zeros(values[2].shape, dtype=object), tol)
    assert degree_snapshot == (bare.born_color_degrees, bare.virtual_color_degrees)
    print(
        f"PROJECTED_SPIN_PASSED {model.color_accuracy} {arithmetic} {precision}",
        flush=True,
    )
    return rebound, spin
