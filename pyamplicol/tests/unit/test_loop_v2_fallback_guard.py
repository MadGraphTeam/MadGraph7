# SPDX-License-Identifier: 0BSD
"""The guard of the loop-induced ``"auto"`` per-point NinjaRS fallback (m-lit
lit-fallback-guard; DECISIONS 2026-11-09 05:56 'NINJARS DEFECT CONFIRMED'
ruling (1), USER 2026-11-08 ~20:55 'Same precision as call').

NinjaRS (pin 5b6a8f57) returns finite, badly wrong values as success at
near-planar 2 -> 3 points of rank-n pentagon families (g g > z z z stream
points 44 and 57 in double-double: 1.3e7 and 0.48 off the truth). Every
point the per-point fallback serves therefore runs the cached
reduction-direction re-reduction at the call's carrier and fails typed when
its relative estimate exceeds the call's target; the estimate is recorded
in the point's record (``VirtualReduction.guard_estimate``), an accepted
point keeps the fallback's value bitwise, and points TIRust serves run
nothing more.

* the guard's rule and reason (one definition in Rust ``reduction_policy``)
  and its logic in ``loops.v2.reduction.with_fallback`` with a scripted
  batch runner (reuse of the call's cached check, the extra pass beside the
  rescaled variant, the public estimate, the failure paths);
* (a') ``"auto"`` double-double at the z z z points 44 and 57 never returns
  a value more than 1e-5 off the truth;
* (b') explicit ``reduction="ninjars"`` binary64 with the recommended checks
  at the tilt-1e-11 point: a served value more than 1e-5 off the truth
  carries an estimate above 1e-5 (one rule that holds before and after
  lit-ninja-planar's re-pin: a refusal or an accurate value pass too);
* (c') default binary64 ``"auto"`` at the 13 tilt points stays within 1e-5
  of the truth;
* the guard at binary64, double-double and arbitrary precision through the
  public API, and the loop-induced default rescue stack at 44 and 57.

The fixture ``tests/fixtures/loop/fallback_guard_zzz.json`` holds the
binary64 texts, mu_R^2 and truths (its ``provenance``). The SDK guard is in
``test_loop_v2_sdk_selection.py``.
"""

from __future__ import annotations

import ctypes
import json
import math
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.unit.loop_v2_induced_observables_rows import generate

FIXTURE = Path(__file__).parents[1] / "fixtures" / "loop" / "fallback_guard_zzz.json"
TARGET = 1e-5


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _tirust(native) -> bool:
    return "tirust" in json.loads(native.reduction_backends())


def _error(value: Any, truth: str) -> float:
    """Relative deviation of a one-sector finite value from the truth."""

    found = Decimal(str(value[0][0]))
    exact = Decimal(truth)
    return float(abs(found - exact) / abs(exact))


# ------------------------------------------------------------ rule and text


@pytest.mark.loop_fast
def test_the_guard_has_one_rule_and_one_reason(loop_native):
    """The guard's acceptance rule and its one reason text are the native
    policy's (``reduction_policy::fallback_accepted`` /
    ``fallback_rejected``), read by Python."""

    from pyamplicol.loops.v2.reduction import fallback_accepted, text

    assert fallback_accepted(1e-5, 1e-5) and fallback_accepted(0.0, 1e-5)
    assert not fallback_accepted(1.5e-5, 1e-5)
    assert not fallback_accepted(math.nan, 1e-5)
    assert not fallback_accepted(math.inf, 1e-5)
    reason = text(
        "rejected",
        "tirust",
        "ninjars",
        "TIRust refused the row",
        carrier="double-double",
        estimate=142.4,
        target=1e-5,
    )
    assert reason == (
        "auto: TIRust refused or failed the point (TIRust refused the row); the "
        "per-point NinjaRS fallback's value failed its reduction-direction check: "
        "estimate 1.4e2 > target 1e-5 (double-double)"
    )
    assert loop_native.reduction_text(
        "rejected",
        "tirust",
        "ninjars",
        "x",
        carrier="arbitrary",
        estimate=3.0,
        target=1e-7,
    ).endswith("estimate 3.0e0 > target 1e-7 (arbitrary)")


# ------------------------------------------------- with_fallback, scripted


@dataclass(frozen=True)
class _Point:
    """A scripted point result (the fields ``with_fallback`` reads)."""

    value: float
    stability: Any = None
    finiteness: float | None = None
    tag: str = ""


def _estimate(value: float, budgets: dict[str, float]) -> Any:
    from pyamplicol.loops.v2.stability import Budget, StabilityEstimate

    return StabilityEstimate.from_checks(
        {"s": Budget((value,))},
        {name: {"s": Budget((budget,))} for name, budget in budgets.items()},
        scale={"s": Budget((0.0,))},
        roundoff=2.0**-53,
    )


@dataclass
class _Runner:
    """A batch runner: TIRust fails ``refused``; NinjaRS serves every point
    with value 1 and per check its budget (``cached`` for the reduction
    direction, ``rescaled``, ``rotation``) unless the point is in
    ``ninja_fails``; ``guard_fails``: points the cached check fails."""

    refused: set[int]
    cached: dict[int, float]
    ninja_fails: set[int] = field(default_factory=set)
    guard_fails: set[int] = field(default_factory=set)
    calls: list[tuple[str, tuple[int, ...] | None, Any]] = field(default_factory=list)

    def __call__(self, indices, checks):
        from pyamplicol.loops.v2.reduction import current
        from pyamplicol.loops.v2.stability import REDUCTION, RESCALED, ROTATION

        backend = current()
        self.calls.append(
            (backend, None if indices is None else tuple(indices), checks)
        )
        points = list(range(8)) if indices is None else list(indices)
        results: list[Any] = []
        failures: dict[int, list[str]] = {}
        for local, index in enumerate(points):
            if backend == "tirust" and index in self.refused:
                results.append(None)
                failures[local] = [f"TIRust refused point {index}"]
                continue
            if backend == "ninjars" and index in self.ninja_fails:
                results.append(None)
                failures[local] = ["MasterFailed"]
                continue
            budgets: dict[str, float] = {}
            enabled = checks is not None and (
                checks.rotations or checks.reduction_direction
            )
            if enabled and checks.rotations:
                budgets[ROTATION.format(0)] = 1e-14
            if enabled and checks.reduction_direction:
                if checks.rescaled:
                    budgets[RESCALED] = 1e-13
                elif backend == "ninjars" and index in self.guard_fails:
                    results.append(None)
                    failures[local] = ["re-test of family 0: Unstable"]
                    continue
                else:
                    budgets[REDUCTION] = (
                        self.cached.get(index, 1e-12) if backend == "ninjars" else 1e-14
                    )
            results.append(
                _Point(
                    1.0 + index,
                    _estimate(1.0 + index, budgets) if budgets else None,
                    0.5 if budgets else None,
                    backend,
                )
            )
        return results, failures


def _choice(target: float | None = TARGET):
    from pyamplicol.api.virtual_results import VirtualReduction

    return VirtualReduction(
        "auto", "tirust", "the loop-induced default", fallback="ninjars", target=target
    )


@pytest.mark.loop_fast
def test_a_served_point_is_guarded_by_the_cached_check_without_call_checks(
    loop_native,
):
    """Without the call's checks the fallback's evaluation runs the cached
    reduction-direction check (one pass, no rescaled variant); an accepted
    point keeps the fallback's value with no public estimate or finiteness
    (what an evaluation without checks returns) and its estimate recorded,
    a point above the target fails typed with the one reason; TIRust's
    points run once, unguarded."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import text, with_fallback

    runner = _Runner(refused={2, 5, 6}, cached={2: 3e-7, 5: 2.0}, ninja_fails={6})
    none = VirtualStabilityChecks()
    results, failures, records = with_fallback(runner, _choice(), 8, none, "double")
    assert [call[:2] for call in runner.calls] == [
        ("tirust", None),
        ("ninjars", (2, 5, 6)),
    ]
    guard = runner.calls[1][2]
    assert isinstance(guard, VirtualStabilityChecks)
    assert (guard.rotations, guard.reduction_direction, guard.rescaled) == (
        (),
        True,
        False,
    )
    # Accepted: the fallback's value, no estimate (the call ran no checks).
    assert results[2] == _Point(3.0, None, None, "ninjars")
    assert records[2].effective == "ninjars" and records[2].fallback is None
    assert records[2].guard_estimate == pytest.approx(2 * 3e-7 / 3.0, rel=1e-12)
    assert records[2].reason == text(
        "served", "tirust", "ninjars", "TIRust refused point 2"
    )
    # Rejected: one reason naming the fallback, TIRust's failure, the check,
    # the estimate, the target and the carrier.
    estimate = 2 * 2.0 / 6.0
    assert results[5] is None
    assert failures[5] == [
        text(
            "rejected",
            "tirust",
            "ninjars",
            "TIRust refused point 5",
            carrier="double",
            estimate=estimate,
            target=TARGET,
        )
    ]
    assert (
        "NinjaRS fallback's value failed its reduction-direction check"
        in failures[5][0]
    )
    assert records[5].guard_estimate == pytest.approx(estimate, rel=1e-12)
    assert records[5].reason == failures[5][0]
    # The fallback fails too: both reasons, no estimate.
    assert failures[6] == ["TIRust refused point 6", "NinjaRS fallback: MasterFailed"]
    assert records[6].guard_estimate is None
    # TIRust's points: its results, no record.
    assert all(results[i].tag == "tirust" for i in (0, 1, 3, 4, 7))
    assert set(records) == {2, 5, 6}


@pytest.mark.loop_fast
def test_the_calls_cached_check_is_reused_and_a_rescaled_one_is_not(loop_native):
    """The call's own cached reduction check is reused (no second pass; the
    estimate of that check alone, ``StabilityEstimate.subset``); beside the
    rescaled variant (the recommended checks, every default rescue level)
    the served points run the cached check once more, the decision is on
    the cached estimate and the public estimate stays the call's (rescaled)
    one; the cross-backend check is not run on fallback points (R6)."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import with_fallback
    from pyamplicol.loops.v2.stability import REDUCTION, RESCALED

    # Reused: rotation + cached reduction check.
    cached = VirtualStabilityChecks(1, True, cross_backend=True)
    runner = _Runner(refused={1, 3}, cached={1: 1e-9, 3: 1.0})
    results, failures, records = with_fallback(runner, _choice(), 8, cached, "double")
    assert [call[:2] for call in runner.calls] == [
        ("tirust", None),
        ("ninjars", (1, 3)),
    ]
    assert runner.calls[0][2] is cached
    assert runner.calls[1][2] == replace(cached, cross_backend=False)
    assert set(results[1].stability.budgets) == {"rotation-0", REDUCTION}
    assert records[1].guard_estimate == pytest.approx(
        results[1].stability.subset([REDUCTION]).relative, rel=0, abs=0
    )
    assert records[1].guard_estimate < TARGET and failures.get(1) is None
    assert 3 in failures and records[3].guard_estimate > TARGET

    # Rescaled: one more pass of the served points with the cached check.
    rescaled = VirtualStabilityChecks(1, True, rescaled=True)
    runner = _Runner(refused={1, 3, 4}, cached={1: 1e-9, 3: 1.0}, guard_fails={4})
    results, failures, records = with_fallback(
        runner, _choice(), 8, rescaled, "double-double"
    )
    assert [call[:2] for call in runner.calls] == [
        ("tirust", None),
        ("ninjars", (1, 3, 4)),
        ("ninjars", (1, 3, 4)),
    ]
    assert runner.calls[1][2] is rescaled
    guard = runner.calls[2][2]
    assert (guard.rotations, guard.reduction_direction, guard.rescaled) == (
        (),
        True,
        False,
    )
    # The public estimate is the call's own (rescaled) one.
    assert set(results[1].stability.budgets) == {"rotation-0", RESCALED}
    assert records[1].guard_estimate == pytest.approx(2 * 1e-9 / 2.0, rel=1e-12)
    assert failures[3][0].endswith("(double-double)") and len(failures[3]) == 1
    # The guard's check fails the point: the fallback failed it.
    assert results[4] is None
    assert failures[4] == [
        "TIRust refused point 4",
        "NinjaRS fallback: re-test of family 0: Unstable",
    ]
    assert records[4].guard_estimate is None

    # No served point: no extra pass.
    runner = _Runner(refused={2}, cached={}, ninja_fails={2})
    with_fallback(runner, _choice(), 8, rescaled, "double")
    assert len(runner.calls) == 2


@pytest.mark.loop_fast
def test_the_target_nan_and_calls_without_fallback(loop_native):
    """The only threshold is the call's target (the default target when the
    choice has none); a NaN estimate is rejected; a choice without a
    per-point fallback (explicit and Born-interfered calls) runs once."""

    from pyamplicol.api.virtual_results import VirtualReduction
    from pyamplicol.loops.v2.reduction import default_target, with_fallback

    runner = _Runner(refused={0, 1}, cached={0: 1e-4, 1: math.nan})
    _results, failures, records = with_fallback(runner, _choice(1e-3), 8)
    assert 0 not in failures and records[0].guard_estimate == pytest.approx(2e-4)
    assert math.isnan(records[1].guard_estimate) and "estimate NaN" in failures[1][0]
    runner = _Runner(refused={0}, cached={0: 4e-6})
    _results, failures, records = with_fallback(runner, _choice(None), 8)
    assert default_target() == TARGET
    assert 0 in failures and records[0].guard_estimate == pytest.approx(8e-6)
    explicit = VirtualReduction("ninjars", "ninjars", "requested")
    runner = _Runner(refused=set(), cached={})
    with_fallback(runner, explicit, 8)
    assert len(runner.calls) == 1
    with pytest.raises(ValueError, match="guard_estimate is a per-point"):
        replace(_choice(), guard_estimate=1.0)
    with pytest.raises(TypeError):
        VirtualReduction("auto", "ninjars", "x", guard_estimate="1")


# ------------------------------------------- g g > z z z through the public API


@pytest.fixture(scope="module")
def zzz(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > z z z (LI-34) through the public Generator (process ``p``)."""

    return generate(tmp_path_factory.mktemp("zzz"), "li_gg_zzz", "g g > z z z")


def _stream(keys=("44", "57")):
    found = _fixture()["stream"]
    return (
        [found[key]["point"] for key in keys],
        [found[key]["mu_r2"] for key in keys],
        [found[key]["truth"] for key in keys],
    )


def _tilt(keys=None):
    fixture = _fixture()
    keys = list(fixture["tilt"]) if keys is None else list(keys)
    return (
        keys,
        [fixture["tilt"][key]["point"] for key in keys],
        [fixture["tilt_mu_r2"]] * len(keys),
        [fixture["tilt"][key]["truth"] for key in keys],
    )


ESTIMATE = r"estimate [0-9.]+e-?[0-9]+ > target "


def _guarded_failure(lane, target: float = TARGET) -> None:
    """A typed failure of a point the fallback evaluated: its record names
    the fallback; where the fallback served it, the guard's estimate is
    above the target and the one reason names it."""

    import re

    record = lane.reduction
    assert lane.status == "failed", lane
    assert record is not None and record.requested == "auto", record
    assert record.effective == "ninjars", record
    assert "NinjaRS" in lane.detail, lane.detail
    if record.guard_estimate is not None:
        assert not record.guard_estimate <= target, record
        assert re.search(ESTIMATE, lane.detail), lane.detail
        assert "fallback's value failed its reduction-direction check" in lane.detail


@pytest.mark.loop_physics
def test_a_auto_double_double_returns_no_wrong_fallback_value(zzz, loop_native):
    """(a'): ``"auto"`` double-double at g g > z z z stream points 44 and 57
    (TIRust refuses both at its Gram gate; NinjaRS returned 1.3e7 and 0.48
    off the truth as success before the guard, ninjars-dd-zzz B.md §2.1)
    returns no value more than 1e-5 off the truth: each point is within it
    or a typed failure naming the fallback (and, served, the estimate)."""

    from pyamplicol import Runtime

    points, scales, truths = _stream()
    runtime = Runtime.load(zzz, process="p")
    found = runtime.evaluate(
        points,
        mu_r2=scales,
        arithmetic="double-double",
        poles="none",
        on_failure="return",
    )
    if not _tirust(loop_native):
        # Without TIRust "auto" is NinjaRS for the whole call, with no
        # per-point fallback to guard (explicit NinjaRS's documented hazard).
        assert all(lane.reduction.fallback is None for lane in found.diagnostics)
        return
    for lane, value, truth in zip(found.diagnostics, found.values, truths, strict=True):
        if value is not None:
            assert _error(value, truth) <= TARGET, (
                lane.reduction,
                _error(value, truth),
            )
            continue
        _guarded_failure(lane)


@pytest.mark.loop_physics
def test_b_explicit_ninjars_with_the_recommended_checks_flags_its_wrong_value(
    zzz, loop_native
):
    """(b'): explicit ``reduction="ninjars"`` binary64 with the recommended
    checks at the tilt-1e-11 point: ONE RULE, a served value more than 1e-5
    off the truth carries an estimate above 1e-5 (at pin 5b6a8f57 it is
    8.9e13 off with estimate 2.4e2, ninjars-dd-zzz C/out/tilt.json
    'nj_b64_chk'); an accurate value or a typed refusal (lit-ninja-planar's
    conditioning bound) passes too."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    _keys, points, scales, truths = _tilt(["1e-11"])
    runtime = Runtime.load(zzz, process="p")
    found = runtime.evaluate(
        points,
        mu_r2=scales,
        reduction="ninjars",
        stability_checks=VirtualStabilityChecks(1, True, rescaled=True),
        poles="none",
        on_failure="return",
    )
    value = found.values[0]
    if value is None:
        assert found.diagnostics[0].status == "failed"
        return
    error = _error(value, truths[0])
    estimate = found.stability[0].relative
    assert not (error > TARGET and estimate <= TARGET), (error, estimate)


@pytest.mark.loop_physics
def test_c_default_auto_binary64_stays_accurate_on_the_tilt_points(zzz, loop_native):
    """(c'): default binary64 ``"auto"`` at the 13 tilt points (eps 0, the
    double-double window <= 1e-14, the binary64 window 1e-11..1e-7 and the
    regular control 1e-2) returns every value within 1e-5 of the truth (a
    typed failure fails this test)."""

    from pyamplicol import Runtime

    keys, points, scales, truths = _tilt()
    runtime = Runtime.load(zzz, process="p")
    found = runtime.evaluate(points, mu_r2=scales, poles="none", on_failure="return")
    errors = {}
    for key, lane, value, truth in zip(
        keys, found.diagnostics, found.values, truths, strict=True
    ):
        assert value is not None, (key, lane.detail[:300])
        errors[key] = _error(value, truth)
    assert max(errors.values()) <= TARGET, errors


def _ratio(reason: str) -> float | None:
    """The largest deviation / term scale of TIRust's method-choice
    refusals in a failure text (``v2_tir``'s acceptance check), or None."""

    import re

    found = [
        float(deviation) / float(scale)
        for deviation, scale in re.findall(
            r"deviation ([0-9.eE+-]+) of term scale ([0-9.eE+-]+)", reason
        )
    ]
    return max(found) if found else None


@pytest.mark.loop_physics
def test_the_guard_rejects_a_wrong_binary64_fallback_value(zzz, loop_native):
    """Binary64 reject: at the tilt-1e-11 point TIRust's method-choice lanes
    deviate by r of their term scale; at the target whose calibrated
    acceptance is r / 2 TIRust refuses the point, and the per-point fallback
    serves NinjaRS's binary64 value (8.9e13 off the truth at pin 5b6a8f57,
    C/out/tilt.json 'nj_b64'). The guard's cached reduction-direction
    estimate (175 there) exceeds the target: the point fails typed with the
    one reason and its estimate recorded; the public result carries no
    estimate (no checks requested)."""

    from pyamplicol import Runtime
    from pyamplicol.api.errors import EvaluationError

    _keys, points, scales, truths = _tilt(["1e-11"])
    runtime = Runtime.load(zzz, process="p")
    options = {"mu_r2": scales, "poles": "none", "on_failure": "return"}
    if not _tirust(loop_native):
        found = runtime.evaluate(points, **options)
        assert found.diagnostics[0].reduction.fallback is None
        return
    refused = runtime.evaluate(
        points, reduction="tirust", reduction_target=1e-300, **options
    )
    ratio = _ratio(refused.diagnostics[0].detail)
    assert ratio is not None and ratio > 0, refused.diagnostics[0].detail
    target = ratio / 2 / loop_native.tir_acceptance(1.0)
    explicit = runtime.evaluate(
        points, reduction="tirust", reduction_target=target, **options
    )
    assert explicit.values[0] is None
    found = runtime.evaluate(points, reduction_target=target, **options)
    assert found.stability is None
    lane = found.diagnostics[0]
    value = found.values[0]
    if value is not None:
        # A served value (lit-ninja-planar's re-pin may make it accurate).
        assert lane.reduction.guard_estimate <= target
        assert _error(value, truths[0]) <= TARGET
        return
    _guarded_failure(lane, target)
    assert lane.detail.endswith("(double)"), lane.detail
    assert lane.reduction.target == target
    with pytest.raises(EvaluationError, match="reduction-direction check: estimate"):
        runtime.evaluate(
            points, reduction_target=target, **{**options, "on_failure": "raise"}
        )


@pytest.mark.loop_physics
def test_the_default_rescue_stack_escalates_the_guarded_fallback(zzz, loop_native):
    """The loop-induced default stack (TIRust b64 -> NinjaRS b64 ->
    double-double -> 57 digits) at z z z points 44 and 57: before the guard
    both escalated past the double-double level on that level's rescaled
    estimates (64 / 31, ninjars-dd-zzz B.md §2.1). The double-double level is
    ``"auto"`` with the per-point fallback; its rescaled checks run no cached
    re-reduction, so the guard runs its own on the served points and rejects
    the NinjaRS value (cached estimates 142 / 26 on record, C/out/checks.json
    'nj_dd_dir'): the attempt is the guard's typed failure with its
    estimate, and the stack escalates it to the 57-digit level as before,
    whose value is within 1e-5 of the truth."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualRescuePolicy

    points, scales, truths = _stream()
    runtime = Runtime.load(zzz, process="p")
    found = runtime.evaluate(
        points,
        mu_r2=scales,
        poles="none",
        on_failure="return",
        stability=VirtualRescuePolicy(),
    )
    assert found.rescue is not None
    for record, value, truth in zip(found.rescue, found.values, truths, strict=True):
        assert value is not None and _error(value, truth) <= TARGET, record
        assert record.arithmetic == "arbitrary", record
        (attempt,) = [a for a in record.attempts if a.arithmetic == "double-double"]
        assert attempt.estimate is None and attempt.failure is not None, attempt
        if not _tirust(loop_native):
            continue
        reduction = attempt.reduction
        assert (reduction.requested, reduction.effective) == ("auto", "ninjars")
        assert "NinjaRS" in attempt.failure
        if reduction.guard_estimate is not None:
            # The fallback served the point: the guard's typed failure.
            assert not reduction.guard_estimate <= TARGET, reduction
            assert "fallback's value failed its reduction-direction check" in (
                attempt.failure
            )
            assert attempt.failure.endswith("(double-double)"), attempt.failure


# ------------------------------------------------------------------ the SDKs


def _c_inputs(points, scales):
    """The C ABI's binary64 inputs of the fixture's binary64 texts (exact)."""

    return (
        [[[float(c) for c in leg] for leg in point] for point in points],
        [float(scale) for scale in scales],
    )


def _python_guard(found) -> dict:
    """The Python API's per-point fallback record of each guard-rejected or
    accepted point, as the SDK's diagnostics "fallbacks" name it."""

    return {
        str(index): {
            "effective": None if found.values[index] is None else "ninjars",
            "reason": lane.reduction.reason,
            "guard_estimate": lane.reduction.guard_estimate,
        }
        for index, lane in enumerate(found.diagnostics)
        if lane.reduction is not None and lane.reduction.effective == "ninjars"
    }


@pytest.mark.loop_physics
def test_the_sdk_guard_equals_the_python_guard(zzz, loop_native, tmp_path):
    """The SDK's ``with_fallback`` guard through the C ABI and the C++
    wrapper (m-lit lit-fallback-guard Done 2): binary64 at the tilt-1e-11
    point and the target that refuses TIRust there, double-double at the
    stream points 44 and 57: the points fail with status Failed (not
    AccuracyNotEstablished), the one reason and the guard's estimate in the
    "fallbacks" record, equal to the Python API's (binary64 bitwise; the
    extended carriers' estimate read in binary64 from the carrier's re-test,
    as the Python assembly reads it); the C++ wrapper returns the C ABI's
    record."""

    import subprocess

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_extras import SDK, _compiler, _sdk, _staged_sdk
    from tests.unit.test_loop_v2_sdk_induced import (
        FAILED_STATUS,
        FALLBACK_CLIENT,
        _evaluate,
        _open,
    )

    runtime = Runtime.load(zzz, process="p")
    _keys, tilt, tilt_scales, _truths = _tilt(["1e-11"])
    stream, stream_scales, _ = _stream()
    sdk = _sdk()
    if not _tirust(loop_native):
        handle = _open(sdk, zzz)
        try:
            found = _evaluate(sdk, handle, *_c_inputs(tilt, tilt_scales))
            assert found["diagnostics"]["reduction"]["fallback"] is None
        finally:
            sdk.rusticol_virtual_free(handle)
        return
    refused = runtime.evaluate(
        tilt,
        mu_r2=tilt_scales,
        reduction="tirust",
        reduction_target=1e-300,
        poles="none",
        on_failure="return",
    )
    target = _ratio(refused.diagnostics[0].detail) / 2 / loop_native.tir_acceptance(1.0)
    binary64 = runtime.evaluate(
        tilt,
        mu_r2=tilt_scales,
        reduction_target=target,
        poles="none",
        on_failure="return",
    )
    double_double = runtime.evaluate(
        stream,
        mu_r2=stream_scales,
        arithmetic="double-double",
        poles="none",
        on_failure="return",
    )
    expected64 = _python_guard(binary64)
    expected_dd = _python_guard(double_double)
    assert set(expected64) == {"0"} and expected64["0"]["effective"] is None
    assert expected_dd and all(item["guard_estimate"] for item in expected_dd.values())

    handle = _open(sdk, zzz)
    try:
        assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0
        assert (
            sdk.rusticol_virtual_set_reduction_target(handle, ctypes.c_double(target))
            == 0
        )
        found = _evaluate(sdk, handle, *_c_inputs(tilt, tilt_scales))
        assert found["statuses"] == [FAILED_STATUS]
        assert found["diagnostics"]["fallbacks"] == expected64
        assert found["diagnostics"]["failures"]["0"] == [expected64["0"]["reason"]]
        assert (
            sdk.rusticol_virtual_set_reduction_target(handle, ctypes.c_double(TARGET))
            == 0
        )
        assert sdk.rusticol_virtual_set_arithmetic(handle, b"double-double", 0) == 0
        found = _evaluate(sdk, handle, *_c_inputs(stream, stream_scales))
    finally:
        sdk.rusticol_virtual_free(handle)
    record = found["diagnostics"]["fallbacks"]
    assert set(record) == set(expected_dd)
    for key, item in expected_dd.items():
        mine = record[key]
        assert mine["effective"] == item["effective"], key
        # The estimate in binary64 from the carrier's re-test: the Python
        # assembly's operations, its colour-metric norms formed in binary64
        # instead of the carrier (a relative difference at binary64 rounding).
        assert math.isclose(
            mine["guard_estimate"], item["guard_estimate"], rel_tol=1e-12
        ), (key, mine, item)
        if item["effective"] is None:
            assert found["statuses"][int(key)] == FAILED_STATUS
            assert "reduction-direction check" in mine["reason"]
            assert mine["reason"].endswith("(double-double)")

    # The C++ wrapper: the C ABI's binary64 record.
    build = _staged_sdk()
    (tmp_path / "client.cpp").write_text(FALLBACK_CLIENT)
    subprocess.run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-O2",
            f"-I{SDK / 'include'}",
            str(tmp_path / "client.cpp"),
            f"-L{build}",
            "-lrusticol_loop_sdk",
            f"-Wl,-rpath,{build}",
            "-lm",
            "-o",
            str(tmp_path / "client"),
        ],
        check=True,
        cwd=tmp_path,
    )
    points, scales = _c_inputs(tilt, tilt_scales)
    lines = [f"{len(points)} {len(points[0])}"]
    for point, scale in zip(points, scales, strict=True):
        lines.append(" ".join(v.hex() for v in [*(c for p in point for c in p), scale]))
    (tmp_path / "input.txt").write_text("\n".join(lines) + "\n")
    done = subprocess.run(
        [
            str(tmp_path / "client"),
            str(zzz),
            "p",
            str(tmp_path / "input.txt"),
            repr(target),
            "double",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    statuses, _values, diagnostics = done.stdout.splitlines()
    assert statuses == f"statuses {FAILED_STATUS}"
    assert json.loads(diagnostics.split(" ", 1)[1])["fallbacks"] == expected64
