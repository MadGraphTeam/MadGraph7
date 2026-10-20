# SPDX-License-Identifier: 0BSD
"""Public configured binding rescue over unchanged saved physical instructions.

Fixtures only add a scalar overflow diagnostic to an existing input guard. They
exercise failure routing; they are not new physical models or reference numbers.
"""

from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol import Runtime
from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.api.errors import EvaluationError
from pyamplicol.api.virtual_results import LaurentCoefficients

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def controls(saved_loop_fixture):
    directory = str(saved_loop_fixture("source-binding-rescue-v43ff063"))
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    return Path(directory)


def _policy():
    return VirtualStabilityPolicy(
        (
            VirtualArithmeticSpec("arbitrary", 60),
            VirtualArithmeticSpec("arbitrary", 80),
        ),
        "1e-8",
        "1e-9",
    )


POINT = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    ["500", "300", "0", "400"],
    ["500", "-300", "0", "-400"],
]
PARAMETERS = {"aS": "0.11800000000000000000000000000000000001"}


def _close(actual, expected):
    if isinstance(actual, LaurentCoefficients):
        actual, expected = tuple(actual), tuple(expected)
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _close(a, b)
        return
    if (
        hasattr(actual, "real")
        and hasattr(actual, "imag")
        and not isinstance(actual, (float, Decimal))
    ):
        _close(actual.real, expected.real)
        _close(actual.imag, expected.imag)
        return
    with localcontext() as context:
        context.prec = 100
        a, b = Decimal(actual), Decimal(expected)
        assert abs(a - b) <= Decimal("2e-8") * max(
            abs(a), abs(b), Decimal("1e-12")
        ) + Decimal("1e-9")


@pytest.mark.parametrize("kind", ["resolved", "spin"])
def test_public_retained_policy_returns_complete_rescued_observables(controls, kind):
    root = controls.parent
    pristine = (
        root / "source-colour-components/public-artifact"
        if kind == "resolved"
        else root / "projected-spin-v43ff063/lc-artifact"
    )
    with pytest.raises(EvaluationError):
        Runtime.load(controls / f"{kind}-artifact", model_parameters=PARAMETERS)
    runtime = Runtime.load(
        controls / f"{kind}-artifact", model_parameters=PARAMETERS, stability=_policy()
    )
    reference = Runtime.load(pristine, model_parameters=PARAMETERS)
    if kind == "spin":
        sources = {
            3: [0, ("0.8", "-0.16"), 0, ("-0.6", "0.12")],
            4: [0, 0, ("1", "0.2"), 0],
        }
        runtime.set_spin_correlation_vectors(sources)
        reference.set_spin_correlation_vectors(sources)
        evaluate, compare = runtime.evaluate_correlated, reference.evaluate_correlated
    else:
        evaluate, compare = runtime.evaluate_resolved, reference.evaluate_resolved
    result = evaluate([POINT], mu_r2="1000000")
    baseline = compare([POINT], mu_r2="1000000")
    _close(result.born_values, baseline.born_values)
    _close(result.values, baseline.values)
    diagnostic = result.diagnostics[0]
    assert diagnostic.status == "rescued" and diagnostic.accuracy_established
    attempts = diagnostic.verification.attempts
    assert len(attempts) == 3
    assert attempts[0].failures[0].stage == "real model input type"
    assert not attempts[0].numerical_success
    assert attempts[-1].agreement and attempts[-1].conversion_agreement
    assert runtime.inspect()["parameter_overrides"]["aS"] == [PARAMETERS["aS"], "0"]
    before = runtime.inspect()
    for invalid in ({"aS": "NaN"}, {"MT": ("173", "1e-1000")}):
        with pytest.raises((ValueError, EvaluationError)):
            runtime.set_model_parameters(invalid)
        assert runtime.inspect()["source_revision"] == before["source_revision"]
        assert runtime.inspect()["parameter_overrides"] == before["parameter_overrides"]
    runtime.set_model_parameters({"aS": "0.12"})
    assert runtime.inspect()["source_revision"] == before["source_revision"] + 1
    runtime.clear()
    assert evaluate([], mu_r2=[]).valid == ()
    assert runtime.inspect()["stability"] is not None
