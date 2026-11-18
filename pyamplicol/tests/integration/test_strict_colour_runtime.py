# SPDX-License-Identifier: 0BSD
"""Strict (formal-N_c) tree colour and graded output against analytic sums.

For the 2 -> 2 trees below the colour-summed |M|^2 is known in closed form
for any N_c (Ellis-Stirling-Webber Table 7.1 with the SU(N_c) colour factors
kept symbolic), with s = (p1 + p2)^2, t = (p1 - p3)^2, u = (p1 - p4)^2 and
X = (t^2 + u^2)/s^2:

* u d > u d:   (N^2 - 1)/4 * K,                 degrees N^2/4 and -1/4;
* u u~ > g g:  [N (N^2 - 1) X - (N^2 - 1)/N] * K', degrees N^3 X,
  -N (X + 1), 1/N;
* g g > g g:   N^2 (N^2 - 1) * K'',               degrees N^4 and -N^2,

where K, K', K'' do not depend on N_c. Strict LC keeps the leading degree
P = n_g + n_qqbar, strict NLC the degrees P, P-1 and P-2, and the graded full
colour every degree; N_c = 3 is inserted after the truncation. The tests
compare ratios of graded degrees to the full-colour total of an independent
full-colour artifact, so K and the normalisation cancel.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)


def _configuration(accuracy: str, *, graded: bool = False) -> RunConfig:
    return RunConfig(
        action="generate",
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 2, "QED": 0},
        ),
        color=ColorConfig(accuracy=accuracy, graded=graded),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="compiled",
            optimization=EvaluatorOptimizationConfig(cores=1, horner_iterations=1),
            jit=JITConfig(optimization_level=0),
        ),
    )


def _point(seed: int) -> tuple[tuple[float, ...], ...]:
    rng = random.Random(seed)
    energy = 50.0 + 450.0 * rng.random()
    cos = -0.9 + 1.8 * rng.random()
    phi = 2.0 * math.pi * rng.random()
    sin = math.sqrt(1.0 - cos * cos)
    n = (energy * sin * math.cos(phi), energy * sin * math.sin(phi), energy * cos)
    return (
        (energy, 0.0, 0.0, energy),
        (energy, 0.0, 0.0, -energy),
        (energy, *n),
        (energy, -n[0], -n[1], -n[2]),
    )


def _mandelstam(point) -> tuple[float, float, float]:
    def square(p):
        return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2

    p1, p2, p3, _ = point
    s = square(tuple(a + b for a, b in zip(p1, p2, strict=True)))
    t = square(tuple(a - b for a, b in zip(p1, p3, strict=True)))
    u = -s - t
    return s, t, u


def _degrees(process: str, point) -> dict[int, float]:
    """Analytic N_c-degree terms at N_c = 3, relative to the full colour sum."""

    n = 3.0
    if process == "u d > u d":
        full = (n * n - 1.0) / 4.0
        return {2: n * n / 4.0 / full, 1: 0.0, 0: -0.25 / full}
    if process == "u u~ > g g":
        s, t, u = _mandelstam(point)
        x = (t * t + u * u) / (s * s)
        full = n * (n * n - 1.0) * x - (n * n - 1.0) / n
        return {
            3: n**3 * x / full,
            2: 0.0,
            1: -n * (x + 1.0) / full,
            0: 0.0,
            -1: (1.0 / n) / full,
        }
    if process == "g g > g g":
        full = n * n * (n * n - 1.0)
        return {4: n**4 / full, 3: 0.0, 2: -(n**2) / full}
    raise AssertionError(process)


_PROCESSES = ("u d > u d", "u u~ > g g", "g g > g g")
_DEPTH = {"strict-lc": 0, "strict-nlc": 2}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    pytest.importorskip("pyamplicol._rusticol")
    model = ModelSource.built_in_sm()
    root = tmp_path_factory.mktemp("strict-colour")
    result = {}
    for process in _PROCESSES:
        for accuracy, graded in (
            ("full", False),
            ("full", True),
            ("strict-lc", False),
            ("strict-nlc", False),
        ):
            label = f"{accuracy}{'-graded' if graded else ''}"
            path = root / f"{process.replace(' ', '_').replace('~', 'x')}-{label}"
            Generator(_configuration(accuracy, graded=graded)).generate(
                process, path, model=model
            )
            result[process, label] = Runtime.load(path)
    return result


@pytest.mark.parametrize("process", _PROCESSES)
@pytest.mark.parametrize("label", ("full-graded", "strict-lc", "strict-nlc"))
def test_graded_degrees_match_the_formal_expansion(artifacts, process, label):
    full = artifacts[process, "full"]
    runtime = artifacts[process, label]
    points = tuple(_point(seed) for seed in range(5))
    totals = full.evaluate(points)
    graded = runtime.evaluate_graded(points)
    reference = {"u d > u d": 2, "u u~ > g g": 3, "g g > g g": 4}[process]
    assert graded.reference_degree == reference
    depth = _DEPTH.get(label)
    for point, total, values, value_total in zip(
        points, totals, graded.values, graded.total, strict=True
    ):
        expected = {
            degree: ratio
            for degree, ratio in _degrees(process, point).items()
            if depth is None or reference - depth <= degree
        }
        if depth is not None:
            expected = {
                degree: expected.get(degree, 0.0)
                for degree in range(reference, reference - depth - 1, -1)
            }
            assert graded.degrees == tuple(sorted(expected, reverse=True))
        else:
            # The graded full colour reports every degree its colour weights
            # reach; degrees absent from the closed form vanish through the
            # amplitude relations (e.g. the U(1) decoupling of gluon traces).
            assert graded.degrees[0] == reference
            assert set(expected) <= set(graded.degrees)
            expected = {degree: expected.get(degree, 0.0) for degree in graded.degrees}
        for degree, value in zip(graded.degrees, values, strict=True):
            assert abs(complex(value).imag) <= 1e-12 * abs(complex(total))
            assert complex(value).real / complex(total).real == pytest.approx(
                expected[degree], rel=1e-12, abs=1e-12
            )
        assert complex(value_total).real == pytest.approx(
            sum(complex(value).real for value in values), rel=1e-13
        )
        if label == "full-graded":
            assert complex(value_total).real == pytest.approx(
                complex(total).real, rel=1e-12
            )


@pytest.mark.parametrize("process", _PROCESSES)
@pytest.mark.parametrize("label", ("strict-lc", "strict-nlc"))
def test_strict_totals_are_their_degrees_of_the_graded_full_colour(
    artifacts, process, label
):
    points = tuple(_point(seed) for seed in range(10, 13))
    strict = artifacts[process, label]
    graded = artifacts[process, "full-graded"].evaluate_graded(points)
    totals = strict.evaluate(points)
    depth = _DEPTH[label]
    for total, values in zip(totals, graded.values, strict=True):
        kept = sum(
            complex(value).real
            for degree, value in zip(graded.degrees, values, strict=True)
            if degree >= graded.reference_degree - depth
        )
        assert complex(total).real == pytest.approx(kept, rel=1e-12)


@pytest.mark.parametrize("process", _PROCESSES)
def test_resolved_graded_axis_sums_to_the_graded_totals(artifacts, process):
    runtime = artifacts[process, "full-graded"]
    points = tuple(_point(seed) for seed in range(20, 22))
    resolved = runtime.evaluate_resolved(points)
    graded = runtime.evaluate_graded(points)
    assert resolved.color_ids == graded.color_ids
    assert len(resolved.color_ids) == len(graded.degrees) > 1
    for point, values in zip(resolved.values, graded.values, strict=True):
        for index, value in enumerate(values):
            assert complex(value).real == pytest.approx(
                math.fsum(complex(helicity[index]).real for helicity in point),
                rel=1e-13,
                abs=1e-300,
            )
    physics = runtime.physics
    assert physics.color_accuracy == "full"
    assert tuple(item.nc_degree for item in physics.contracted_color_components) == (
        graded.degrees
    )


def test_graded_output_at_higher_precision(artifacts):
    # u d > u d: the degree ratios are the rationals 9/8, 0 and -1/8 at every
    # point. The degrees sum to the independent full-colour artifact to the
    # working precision. Their ratios are compared with 1e-30: the built-in
    # model's U(1)-subtraction vertex is the stored coupling 1/3 (``Vertex(8,
    # ..., (1.0 / 3.0, 0.0))``), whose exact-path value is accurate to about
    # 1e-33, an amplitude property shared by the full colour sum.
    graded_runtime = artifacts["u d > u d", "full-graded"]
    full_runtime = artifacts["u d > u d", "full"]
    point = (
        (500, 0, 0, 500),
        (500, 0, 0, -500),
        (500, 300, 0, 400),
        (500, -300, 0, -400),
    )
    graded = graded_runtime.evaluate_graded((point,), precision=40)
    full = full_runtime.evaluate((point,), precision=40)[0]
    assert graded.degrees == (2, 1, 0)
    (leading,), (odd,), (subleading,) = (graded.for_degree(d) for d in (2, 1, 0))
    with localcontext() as context:
        context.prec = 45
        total = graded.total[0]
        assert isinstance(total, Decimal) and isinstance(full, Decimal)
        assert abs(total / full - 1) < Decimal("1e-36")
        assert abs(odd / total) < Decimal("1e-36")
        assert abs(leading / total - Decimal(9) / Decimal(8)) < Decimal("1e-30")
        assert abs(subleading / total + Decimal(1) / Decimal(8)) < Decimal("1e-30")
    assert Fraction(9, 8) - Fraction(1, 8) == 1


def test_ungraded_artifacts_refuse_graded_output(artifacts):
    from pyamplicol.api.errors import CompatibilityError

    with pytest.raises(CompatibilityError, match="graded output"):
        artifacts["u d > u d", "full"].evaluate_graded((_point(0),))
