# SPDX-License-Identifier: 0BSD
"""The best available value of a refused point (m-lit ux-besteffort; USER
2026-11-11 items 7-9).

With the reduction backends' own acceptance checks on (the default), an
explicitly selected backend and no configured alternative left, a point the
backend refuses is returned by default (``on_failure="best-effort"``) with
the value the same backend gives WITHOUT its own checks
(``reduction_unchecked=True``, bitwise), lane status ``"best-effort"`` (never
a success), the refusal kept and an uncertainty: the call's own stability
estimate when it ran checks, else a rotation re-test of that point alone; the
result's ``warnings`` metadata lists it. Nothing is printed, logged or warned
by default; ``emit_warnings()`` is the opt-in single Python warning;
``"return"`` keeps the failed lanes without a value; ``"raise"`` raises.
"""

from __future__ import annotations

import json
import math
import warnings
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.unit.loop_v2_induced_observables_rows import generate


def _tirust(native) -> bool:
    return "tirust" in json.loads(native.reduction_backends())


@pytest.fixture(scope="module")
def aag(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > a a g (LI-33) and its first frozen stream points: they include
    points TIRust refuses structurally at the default acceptance."""

    from tools.loop_benchmark import streams

    if not _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    artifact = generate(tmp_path_factory.mktemp("aag"), "li_gg_aag", "g g > a a g")
    stream = streams.load("li_gg_aag")
    count = 6
    points = [
        [[str(Decimal(float(Fraction(c)))) for c in leg] for leg in point]
        for point in stream["points"][:count]
    ]
    scales = [str(Decimal(float(Fraction(mu) ** 2))) for mu in stream["mu_r"][:count]]
    return artifact, points, scales


# ------------------------------------------------------------ the model


def test_the_failure_policy_has_one_definition_and_best_effort_is_the_default():
    from pyamplicol.config.models import (
        DEFAULT_ON_FAILURE,
        ON_FAILURE,
        EvaluationConfig,
    )

    assert ON_FAILURE == ("best-effort", "return", "raise")
    assert DEFAULT_ON_FAILURE == "best-effort"
    assert EvaluationConfig().on_failure is None


def test_a_best_effort_record_states_its_refusal_and_uncertainty():
    from pyamplicol.api.virtual_results import (
        VirtualBestEffort,
        VirtualLaneDiagnostic,
        VirtualReduction,
        best_effort_detail,
    )

    record = VirtualBestEffort("TIRust Unstable: x", 3e-4, "rotation", 2e-3)
    assert "3.00e-04" in best_effort_detail(record)
    none = VirtualBestEffort("TIRust Unstable: x", None, "none")
    assert "no uncertainty could be estimated" in best_effort_detail(none)
    with pytest.raises(ValueError):
        VirtualBestEffort("x", None, "rotation")
    with pytest.raises(ValueError):
        VirtualBestEffort("", 1e-3, "rotation")
    reduction = VirtualReduction(
        "tirust", "tirust", "best effort", unchecked=True, best_effort=record
    )
    lane = VirtualLaneDiagnostic(
        "best-effort", "double", 53, False, "d", reduction=reduction
    )
    assert lane.has_value and not lane.valid and lane.best_effort is record
    with pytest.raises(ValueError):
        VirtualLaneDiagnostic("success", "double", 53, False, "d", reduction=reduction)
    with pytest.raises(ValueError):
        VirtualReduction("tirust", "tirust", "x", unchecked=False, best_effort=record)


def test_numerical_warnings_summarise_the_lanes_and_emit_only_on_request():
    from pyamplicol.api.result_warnings import (
        NumericalAccuracyWarning,
        ResultWarning,
        emit,
        merge_warnings,
        numerical_warnings,
        summary_line,
    )
    from pyamplicol.api.virtual_results import (
        VirtualBestEffort,
        VirtualFailureDiagnostic,
        VirtualLaneDiagnostic,
        VirtualReduction,
    )

    best = VirtualReduction(
        "tirust",
        "tirust",
        "r",
        unchecked=True,
        best_effort=VirtualBestEffort("refused A", 1e-3, "rotation"),
    )
    lanes = (
        VirtualLaneDiagnostic("success", "double", 53, False, "ok"),
        VirtualLaneDiagnostic("best-effort", "double", 53, False, "b", reduction=best),
        VirtualLaneDiagnostic(
            "failed",
            "double",
            53,
            False,
            "f",
            failures=(
                VirtualFailureDiagnostic(2, None, None, None, "v2", "no value B"),
            ),
        ),
        VirtualLaneDiagnostic("best-effort", "double", 53, False, "b", reduction=best),
    )
    found = numerical_warnings(lanes, "evaluate")
    assert [(w.kind, w.count, w.points, w.message) for w in found] == [
        ("best-effort", 2, (1, 3), "refused A"),
        ("failed", 1, (2,), "no value B"),
    ]
    assert ResultWarning.from_json(found[0].to_json()) == found[0]
    merged = merge_warnings(found, found[:1])
    assert merged[0].count == 4 and merged[0].points == (1, 3, 1, 3)
    assert summary_line(found).startswith("3 captured warning(s)")
    assert summary_line(()) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert emit(found)
    assert [w.category for w in caught] == [NumericalAccuracyWarning]
    assert not emit(())


# ------------------------------------------------------------ the engine


def _quiet(capfd, call):
    """``call()`` with every Python warning recorded and the process's
    stdout / stderr captured: the evaluation path emits nothing."""

    capfd.readouterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    out, err = capfd.readouterr()
    assert not caught, [str(w.message) for w in caught]
    assert out == "" and err == "", (out[-500:], err[-500:])
    return result


@pytest.mark.loop_physics
def test_refused_points_carry_the_unchecked_value_by_default(aag, capfd):
    """Default policy on explicit TIRust: accepted points bitwise the
    "return" call's, refused points the unchecked TIRust call's values with
    status "best-effort", the refusal and a rotation re-test uncertainty;
    nothing emitted; ``"raise"`` raises; ``"return"`` keeps failed lanes."""

    from pyamplicol import EvaluationError, Runtime

    artifact, points, scales = aag
    options = {"mu_r2": scales, "poles": "none", "reduction": "tirust"}
    runtime = Runtime.load(artifact, process="p")
    returned = runtime.evaluate(points, on_failure="return", **options)
    refused = [p for p, ok in enumerate(returned.valid) if not ok]
    assert refused, "the fixture points include refused ones"
    unchecked = runtime.evaluate(
        points, on_failure="return", reduction_unchecked=True, **options
    )
    found = _quiet(capfd, lambda: runtime.evaluate(points, **options))
    assert found.has_warnings
    assert found.statuses == tuple(
        "best-effort" if p in refused else "success" for p in range(len(points))
    )
    for p, lane in enumerate(found.diagnostics):
        if p not in refused:
            assert repr(found.values[p]) == repr(returned.values[p]), p
            assert lane.best_effort is None
            continue
        assert repr(found.values[p]) == repr(unchecked.values[p]), p
        record = lane.best_effort
        assert record is not None and not lane.valid and lane.has_value
        assert record.refusal and "TIRust" in record.refusal
        assert lane.reduction.effective == "tirust" and lane.reduction.unchecked
        if record.uncertainty is None:
            assert record.source == "none"
        else:
            assert record.source == "rotation"
            assert math.isfinite(record.uncertainty) and record.uncertainty >= 0
        assert lane.failures and lane.failures[0].detail == record.refusal
    # The call ran no checks: the public result carries no estimate.
    assert found.stability is None
    (warning,) = found.warnings
    assert (warning.source, warning.kind, warning.points) == (
        "numerical",
        "best-effort",
        tuple(refused),
    )
    with pytest.raises(EvaluationError):
        runtime.evaluate(points, on_failure="raise", **options)


@pytest.mark.loop_physics
def test_a_checked_call_gives_its_own_estimate_as_the_uncertainty(aag):
    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    artifact, points, scales = aag
    checks = VirtualStabilityChecks(rotations=1)
    options = {"mu_r2": scales, "poles": "none", "reduction": "tirust"}
    runtime = Runtime.load(artifact, process="p")
    unchecked = runtime.evaluate(
        points, on_failure="return", reduction_unchecked=True, **options
    )
    found = runtime.evaluate(points, stability_checks=checks, **options)
    served = [p for p, s in enumerate(found.statuses) if s == "best-effort"]
    assert served
    for p in served:
        record = found.diagnostics[p].best_effort
        assert repr(found.values[p]) == repr(unchecked.values[p]), p
        assert record.source == "stability-checks"
        assert record.uncertainty == found.stability[p].relative


@pytest.mark.loop_physics
def test_the_resolved_call_serves_best_effort_values(aag):
    from pyamplicol import Runtime

    artifact, points, scales = aag
    options = {"mu_r2": scales, "poles": "none", "reduction": "tirust"}
    runtime = Runtime.load(artifact, process="p")
    returned = runtime.evaluate_resolved(points, on_failure="return", **options)
    found = runtime.evaluate_resolved(points, **options)
    refused = [p for p, ok in enumerate(returned.valid) if not ok]
    assert refused
    for p in range(len(points)):
        if p in refused:
            assert found.statuses[p] == "best-effort"
            assert found.values[p] is not None
        else:
            assert repr(found.values[p]) == repr(returned.values[p])
