# SPDX-License-Identifier: 0BSD
"""Physical precision regression, including the full-colour normalization."""

from __future__ import annotations

import importlib.util
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)


def _exact_frames():
    with localcontext() as context:
        context.prec = 120
        point = [
            [Decimal(x) for x in p]
            for p in (
                (500, 0, 0, 500),
                (500, 0, 0, -500),
                (500, 300, 0, 400),
                (500, -300, 0, -400),
            )
        ]
        axis = [Decimal(x) / 7 for x in (2, 3, 6)]
        beta, gamma = Decimal(3) / 5, Decimal(5) / 4
        boosted = []
        for p in point:
            longitudinal = sum(
                (x * n for x, n in zip(p[1:], axis, strict=True)), Decimal(0)
            )
            shift = (gamma - 1) * longitudinal + gamma * beta * p[0]
            boosted.append(
                [gamma * (p[0] + beta * longitudinal)]
                + [x + shift * n for x, n in zip(p[1:], axis, strict=True)]
            )
        return [point, boosted]


@pytest.mark.parametrize("execution_mode", ("recurrence", "eager", "compiled"))
@pytest.mark.parametrize("color_accuracy", ("lc", "full"))
def test_gg_helicity_sum_has_consistent_normalization_and_precision(
    tmp_path,
    execution_mode,
    color_accuracy,
):
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=color_accuracy),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    output = tmp_path / "gg"
    Generator(config).generate("g g > g g", output, model=ModelSource.built_in_sm())
    runtime = Runtime.load(
        output,
        model_parameters={"normalization.alpha_s_me_check": 0.125},
    )
    frames = _exact_frames()
    native_frames = [[[float(x) for x in p] for p in frame] for frame in frames]
    native = runtime.evaluate(native_frames)
    separate = [runtime.evaluate([frame])[0] for frame in native_frames]
    assert native == pytest.approx(separate, rel=3e-13, abs=0)
    exact = runtime.evaluate(frames, precision=80)
    with localcontext() as context:
        context.prec = 90
        assert exact[0] > 0
        assert abs(exact[0] - exact[1]) / abs(exact[0]) < Decimal("1e-70")
        # In particular, full colour must not acquire the extra LC factor81.
        for value, reference in zip(exact, native, strict=True):
            assert float(value) == pytest.approx(reference.real, rel=3e-13, abs=0)

    if execution_mode == "eager" and color_accuracy == "full":
        # Exercise both the ordinary and profiled topology-replay gathers.
        # Repeated points exposed an amplitude/point stride mix-up even when
        # each separate evaluation and the arbitrary-precision path agreed.
        for point_count in (2, 3, 65):
            batch = [native_frames[(index // 2) % 2] for index in range(point_count)]
            expected = [separate[(index // 2) % 2] for index in range(point_count)]
            assert runtime.evaluate(batch) == pytest.approx(expected, rel=3e-13, abs=0)
            resolved = runtime.evaluate_resolved(batch)
            totals = [
                sum(value for helicity in point for value in helicity)
                for point in resolved.values
            ]
            assert totals == pytest.approx(expected, rel=3e-13, abs=0)
            profiled = runtime._backend.profile(batch, include_values=True)
            assert profiled["values"] == pytest.approx(expected, rel=3e-13, abs=0)


def _four_quark_points():
    # (E, a, b, c) with a^2 + b^2 + c^2 = E^2: exact massless 2 -> 2 points.
    return [
        [[e, 0, 0, e], [e, 0, 0, -e], [e, a, b, c], [e, -a, -b, -c]]
        for e, a, b, c in ((3, 1, 2, 2), (7, 2, 3, 6), (9, 1, 4, 8), (11, 2, 6, 9))
    ]


@pytest.mark.parametrize("execution_mode", ("recurrence", "eager", "compiled"))
@pytest.mark.parametrize("model", ("built-in-sm", "built-in-sm-heft"))
@pytest.mark.parametrize(
    ("color_accuracy", "color_factor"), (("full", Fraction(1)), ("lc", Fraction(5, 4)))
)
def test_u1_subtraction_piece_is_exact_beyond_binary64(
    tmp_path, execution_mode, model, color_accuracy, color_factor
):
    """u d > u d through the U(1)-subtraction current: the -1/N_c piece of the
    gluon exchange enters with the exact weight 1/3, so a 100-digit evaluation
    meets the colour sum far below the binary64 rounding of 1/3 (before: a
    relative error of 3.9e-34 at full colour, and 1.1e-17 at LC in the UFO
    model). Full colour sums to (N_c^2 - 1)/4 (ESW Table 7.1); the LC accuracy
    keeps the U(1)-subtraction pieces at N_c = 3 and sums to 5/4 of it."""

    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    orders = {"QCD": 2, "QED": 0}
    if model == "built-in-sm-heft":
        orders["HEFT"] = 0
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            coupling_order_policy="explicit", max_coupling_orders=orders
        ),
        color=ColorConfig(accuracy=color_accuracy),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    source = (
        ModelSource.built_in_sm_heft()
        if model == "built-in-sm-heft"
        else ModelSource.built_in_sm()
    )
    output = tmp_path / "udud"
    Generator(config).generate("u d > u d", output, model=source)
    runtime = Runtime.load(output)
    points = _four_quark_points()
    values = runtime.evaluate(points, precision=100)
    with localcontext() as context:
        context.prec = 110
        pi = Decimal(
            "3.14159265358979323846264338327950288419716939937510582097494459"
            "2307816406286208998628034825342117067982148086513282306647093844"
        )
        # alpha_s = 0.118, the default of both models.
        coupling = (4 * pi * Decimal("0.118")) ** 2
        for point, value in zip(points, values, strict=True):
            energy, z = Fraction(point[0][0]), Fraction(point[2][3])
            s = 4 * energy**2
            t = -2 * energy * (energy - z)
            u = -s - t
            # Averaged |M|^2 of q q' > q q': (4/9) g^4 (s^2 + u^2) / t^2.
            ratio = Fraction(4, 9) * (s * s + u * u) / (t * t) * color_factor
            analytic = coupling * ratio.numerator / ratio.denominator
            assert abs(value / analytic - 1) < Decimal("1e-60")
