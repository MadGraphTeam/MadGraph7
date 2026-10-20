# SPDX-License-Identifier: 0BSD
"""Partial Python output over retained physical artifacts; no graph generation."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol import CorrelatedRequest, Runtime
from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.api.errors import CompatibilityError, EvaluationError

from .test_runtime_source_binding_rescue import POINT, _close

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def retained(saved_loop_fixtures):
    directory = str(saved_loop_fixtures)
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    return Path(directory)


@pytest.mark.parametrize(
    "arithmetic,precision", [("double", 16), ("double-double", 16), ("arbitrary", 70)]
)
def test_actual_partial_resolved_preserves_c6_and_exact_carriers(
    retained, arithmetic, precision
):
    root = retained / "source-colour-components"
    data = json.loads((root / "gg-lc-expected.json").read_text())
    runtime = Runtime.load(
        root / "public-artifact",
        arithmetic=arithmetic,
        precision=precision,
        model_parameters={"aS": "0.11800000000000000000000000000000000001"},
    )
    points = data["points"]
    baseline = runtime.evaluate_resolved(points, mu_r2="1000000")
    # Exactly singular external states exercise typed numerical isolation in
    # every carrier, without malformed input or a synthetic successful zero.
    singular = [["0"] * 4 for _ in POINT]
    batch = [points[1], singular, points[0]]
    result = runtime.evaluate_resolved_with_prec(
        batch, precision, mu_r2="1000000", on_failure="return"
    )
    assert result.shape == (3, 1, 16, 6, 3)
    assert result.valid == (True, False, True)
    assert result.born_values[1] is result.values[1] is None
    assert {failure.point_id for failure in result.diagnostics[1].failures} == {1}
    for output, reference in ((0, 1), (2, 0)):
        _close(result.born_values[output], baseline.born_values[reference])
        _close(result.values[output], baseline.values[reference])
    expected_type = float if arithmetic == "double" else Decimal
    assert isinstance(result.values[0][0][0][0][0], expected_type)
    total = result.total()
    assert total.valid == result.valid and total.diagnostics == result.diagnostics
    assert (
        total.sum_orders()[1]
        is total.finite[1]
        is total.single_pole[1]
        is total.double_pole[1]
        is None
    )
    assert total.for_order(total.sectors[0].order).diagnostics == result.diagnostics
    empty = runtime.evaluate_resolved([], mu_r2=[], on_failure="return")
    assert empty.shape == (0, 1, 16, 6, 3) and empty.total().valid == ()
    if arithmetic == "double":
        inclusive = runtime.evaluate_with_prec(
            batch, precision, mu_r2="1000000", on_failure="return"
        )
        assert inclusive.valid == result.valid
        for point in (0, 2):
            _close(inclusive.values[point], total.values[point])
        with pytest.raises(EvaluationError) as caught:
            runtime.evaluate(batch, mu_r2="1000000")
        assert caught.value.failures == inclusive.diagnostics[1].failures
        failed = runtime.evaluate_resolved(
            [singular], mu_r2="1000000", on_failure="return"
        )
        assert failed.values == failed.born_values == (None,)
        with pytest.raises(ValueError):
            runtime.evaluate([[["NaN", 0, 0, 0]] * 4], mu_r2=1, on_failure="return")
        with pytest.raises(CompatibilityError):
            runtime.evaluate([], mu_r2=[], helicities=["h0"], on_failure="return")


def test_actual_grouped_source_masks_keep_complex_values_and_real_zero(retained):
    runtime = Runtime.load(retained / "projected-spin-v43ff063/lc-artifact")
    a = [0, ("0.800000000000000000000000000000000001", "-0.16"), 0, ("-0.6", "0.12")]
    b = [0, 0, ("1", "0.2"), 0]
    good = {3: a, 4: b}
    runtime.set_spin_correlation_vectors(good)
    baseline = runtime.evaluate_correlated([POINT], mu_r2="1000000")
    overflow = [0, ("1e400", "0"), 0, 0]
    result = runtime.evaluate_correlated_many(
        [POINT] * 3,
        {
            "default": CorrelatedRequest("born"),
            "joint": CorrelatedRequest("born", {3: [a, overflow, a], 4: b}),
            "zero": CorrelatedRequest("born", {3: [0, 0, 0, 0], 4: b}),
        },
        mu_r2="1000000",
        on_failure="return",
    )
    assert result["default"].valid == result["zero"].valid == (True, True, True)
    assert result["joint"].valid == (True, False, True)
    assert result["joint"].values[1] is None
    assert {
        failure.point_id for failure in result["joint"].diagnostics[1].failures
    } == {1}
    assert all(
        value == 0 for row in result["zero"].values for order in row for value in order
    )
    assert all(value == 0 for row in result["zero"].born_values for value in row)
    for label in ("default", "joint"):
        for point in (0, 2):
            _close(result[label].values[point], baseline.values[0])
            assert isinstance(result[label].values[point][0][0], complex)
    assert runtime.evaluate_correlated([], mu_r2=[], on_failure="return").valid == ()
    with pytest.raises(CompatibilityError):
        runtime.evaluate_correlated(
            [], color_correlation="missing", mu_r2=[], on_failure="return"
        )
    # A failing explicit assignment did not mutate the finite retained setter.
    assert (
        runtime.evaluate_correlated([POINT], mu_r2="1000000").values == baseline.values
    )


def test_configured_source_admission_cannot_publish_one_successful_attempt(retained):
    policy = VirtualStabilityPolicy(
        (VirtualArithmeticSpec("arbitrary", 60),), "1e-8", "1e-9"
    )
    runtime = Runtime.load(
        retained / "source-binding-rescue/resolved-artifact", stability=policy
    )
    result = runtime.evaluate([POINT], mu_r2="1000000", on_failure="return")
    assert result.values == result.born_values == (None,)
    diagnostic = result.diagnostics[0]
    assert diagnostic.status == "accuracy-not-established" and not diagnostic.valid
    assert len(diagnostic.verification.attempts) == 2
    assert diagnostic.verification.attempts[-1].numerical_success
    assert not diagnostic.verification.attempts[0].numerical_success
    assert any(
        failure.stage == "precision verification" for failure in diagnostic.failures
    )
    assert runtime.evaluate([], mu_r2=[], on_failure="return").valid == ()
    with pytest.raises(ValueError):
        runtime.evaluate([POINT], mu_r2=-1, on_failure="return")


def test_partial_return_preserves_complete_joint_source_rescue(retained):
    policy = VirtualStabilityPolicy(
        (
            VirtualArithmeticSpec("arbitrary", 60),
            VirtualArithmeticSpec("arbitrary", 80),
        ),
        "1e-8",
        "1e-9",
    )
    runtime = Runtime.load(retained / "projected-spin-v43ff063/lc-artifact")
    runtime.set_spin_correlation_vectors(
        {3: [0, "0.8", 0, "-0.6"], 4: [0, 0, ("1", "0.2"), 0]}
    )
    result = runtime.evaluate_correlated(
        [POINT], mu_r2="1e-400", stability=policy, on_failure="return"
    )
    diagnostic = result.diagnostics[0]
    assert diagnostic.status == "rescued" and diagnostic.accuracy_established
    assert diagnostic.failures == ()
    assert len(diagnostic.verification.attempts) == 3
    assert diagnostic.verification.attempts[-1].agreement
    assert diagnostic.verification.attempts[-1].conversion_agreement
    assert result.values[0] is not None and isinstance(result.values[0][0][0], complex)
