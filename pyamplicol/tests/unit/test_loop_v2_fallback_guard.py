# SPDX-License-Identifier: 0BSD
"""The guard of the loop-induced ``"auto"`` per-point NinjaRS fallback (m-lit
lit-fallback-guard; DECISIONS 2026-11-09 05:56 'NINJARS DEFECT CONFIRMED'
ruling (1) and 11:53, USER 2026-11-08 ~20:55 'Same precision as call' and
2026-11-09 ~11:53 'Double-double+ only, but also it's fine to relax the
target precision to 1e-5 or so').

NinjaRS (pin 5b6a8f57) returns finite, badly wrong values as success at
near-planar 2 -> 3 points of rank-n pentagon families (g g > z z z stream
points 44 and 57 in double-double: 1.3e7 and 0.48 off the truth). In the
extended carriers (double-double, arbitrary precision) every point the
per-point fallback serves therefore runs the cached reduction-direction
re-reduction at the call's carrier and fails typed when its relative
estimate exceeds the guard's limit, the call's target never below 1e-5
(max(target, 1e-5)); the estimate is recorded in the point's record
(``VirtualReduction.guard_estimate``), an accepted point keeps the
fallback's value bitwise, and points TIRust serves run nothing more. A
binary64 fallback point is served unguarded, as before the guard.

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
* the guard at double-double and arbitrary precision (and its absence in
  binary64) through the public API and the SDKs, on the summed route and on
  the resolved, correlated and selected routes, where it decides on the
  outputs the call returns (DECISIONS 2026-11-09 14:30, RULING Q7: each
  output against its own magnitude, a vanishing one against the largest
  returned output; ``reduction_policy::fallback_estimate``), and
  ``on_failure="raise"``;
* the loop-induced default rescue stack at 44 and 57 (loop_nightly: its
  57-digit level takes minutes).

The fixture ``tests/fixtures/loop/fallback_guard_zzz.json`` holds the
binary64 texts, mu_R^2 and truths (its ``provenance``). The module's
artifacts are generated each in a fresh interpreter (``_generated``).
"""

from __future__ import annotations

import ctypes
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parents[1] / "fixtures" / "loop" / "fallback_guard_zzz.json"
TARGET = 1e-5
# The guard's limit is the call's target, never below 1e-5 (USER 2026-11-09
# ~11:53; ``reduction_policy::FALLBACK_FLOOR``).
FLOOR = 1e-5


def _limit(target: float) -> float:
    return max(target, FLOOR)


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
    """The guard's carriers, its acceptance rule and its one reason text are
    the native policy's (``reduction_policy::fallback_guarded`` /
    ``fallback_accepted`` / ``fallback_rejected``), read by Python: the
    extended carriers only, the limit max(target, 1e-5)."""

    from pyamplicol.loops.v2.reduction import (
        CARRIERS,
        fallback_accepted,
        fallback_guarded,
        text,
    )

    assert [fallback_guarded(c) for c in CARRIERS] == [c != "double" for c in CARRIERS]
    assert fallback_accepted(1e-5, 1e-5) and fallback_accepted(0.0, 1e-5)
    assert not fallback_accepted(1.5e-5, 1e-5)
    assert not fallback_accepted(math.nan, 1e-5)
    assert not fallback_accepted(math.inf, 1e-5)
    # Never stricter than 1e-5; a looser target is the limit.
    assert fallback_accepted(9e-6, 1e-8) and not fallback_accepted(1.1e-5, 1e-8)
    assert fallback_accepted(1e-3, 1e-3) and not fallback_accepted(1.1e-3, 1e-3)
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
        "estimate 1.4e2 > 1e-5 = max(target 1e-5, 1e-5) (double-double)"
    )
    assert loop_native.reduction_text(
        "rejected",
        "tirust",
        "ninjars",
        "x",
        carrier="arbitrary",
        estimate=3.0,
        target=1e-7,
    ).endswith("estimate 3.0e0 > 1e-5 = max(target 1e-7, 1e-5) (arbitrary)")


# ------------------------------------------------- with_fallback, scripted


@dataclass(frozen=True)
class _Point:
    """A scripted point result (the fields ``with_fallback`` reads)."""

    value: float
    stability: Any = None
    finiteness: float | None = None
    tag: str = ""


def _estimate(
    value: float, budgets: dict[str, float], scale: float = 0.0, roundoff=2.0**-53
) -> Any:
    from pyamplicol.loops.v2.stability import Budget, StabilityEstimate

    return StabilityEstimate.from_checks(
        {"s": Budget((value,))},
        {name: {"s": Budget((budget,))} for name, budget in budgets.items()},
        scale={"s": Budget((scale,))},
        roundoff=roundoff,
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
    """In an extended carrier, without the call's checks the fallback's
    evaluation runs the cached reduction-direction check (one pass, no
    rescaled variant); an accepted point keeps the fallback's value with no
    public estimate or finiteness (what an evaluation without checks
    returns) and its estimate recorded, a point above the limit fails typed
    with the one reason; TIRust's points run once, unguarded."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import text, with_fallback

    runner = _Runner(refused={2, 5, 6}, cached={2: 3e-7, 5: 2.0}, ninja_fails={6})
    none = VirtualStabilityChecks()
    results, failures, records = with_fallback(
        runner, _choice(), 8, none, "double-double"
    )
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
            carrier="double-double",
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
def test_a_binary64_fallback_point_is_served_unguarded(loop_native):
    """In binary64 the guard does not run (USER 2026-11-09 ~11:53, 'Double-
    double+ only'; DECISIONS 11:53): the fallback's evaluation runs with the
    call's checks only (none: no check at all; the rescaled variant: no
    extra pass), every point it serves keeps its value and the call's
    checks' estimate, whatever its cached estimate would be, and its record
    carries no guard estimate, as before the guard."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import text, with_fallback

    for checks in (None, VirtualStabilityChecks(1, True, rescaled=True)):
        runner = _Runner(refused={2, 5, 6}, cached={2: 3e-7, 5: 2.0}, ninja_fails={6})
        results, failures, records = with_fallback(
            runner, _choice(), 8, checks, "double"
        )
        assert [call[:2] for call in runner.calls] == [
            ("tirust", None),
            ("ninjars", (2, 5, 6)),
        ]
        assert runner.calls[1][2] is checks
        for index in (2, 5):
            assert results[index].value == 1.0 + index and index not in failures
            assert results[index].tag == "ninjars"
            assert records[index].guard_estimate is None
            assert records[index].reason == text(
                "served", "tirust", "ninjars", f"TIRust refused point {index}"
            )
        assert failures[6] == [
            "TIRust refused point 6",
            "NinjaRS fallback: MasterFailed",
        ]


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
    results, failures, records = with_fallback(
        runner, _choice(), 8, cached, "arbitrary"
    )
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
    with_fallback(runner, _choice(), 8, rescaled, "double-double")
    assert len(runner.calls) == 2


@pytest.mark.loop_fast
def test_the_target_nan_and_calls_without_fallback(loop_native):
    """The only threshold is the guard's limit: the call's target (the
    default target when the choice has none), never below 1e-5 (USER
    2026-11-09 ~11:53); a NaN estimate is rejected; a choice without a
    per-point fallback (explicit and Born-interfered calls) runs once."""

    from pyamplicol.api.virtual_results import VirtualReduction
    from pyamplicol.loops.v2.reduction import default_target, with_fallback

    dd = "double-double"
    runner = _Runner(refused={0, 1}, cached={0: 1e-4, 1: math.nan})
    _results, failures, records = with_fallback(runner, _choice(1e-3), 8, None, dd)
    assert 0 not in failures and records[0].guard_estimate == pytest.approx(2e-4)
    assert math.isnan(records[1].guard_estimate) and "estimate NaN" in failures[1][0]
    runner = _Runner(refused={0}, cached={0: 6e-6})
    _results, failures, records = with_fallback(runner, _choice(None), 8, None, dd)
    assert default_target() == TARGET
    assert 0 in failures and records[0].guard_estimate == pytest.approx(1.2e-5)
    # A target below 1e-5: the limit stays 1e-5 (the floor).
    runner = _Runner(refused={0, 1}, cached={0: 3e-6, 1: 1.2e-5})
    _results, failures, records = with_fallback(runner, _choice(1e-8), 8, None, dd)
    assert 0 not in failures and records[0].guard_estimate == pytest.approx(6e-6)
    assert records[0].target == 1e-8
    assert records[1].guard_estimate == pytest.approx(1.2e-5)
    assert failures[1][0].endswith(
        "estimate 1.2e-5 > 1e-5 = max(target 1e-8, 1e-5) (double-double)"
    ), failures[1]
    explicit = VirtualReduction("ninjars", "ninjars", "requested")
    runner = _Runner(refused=set(), cached={})
    with_fallback(runner, explicit, 8, None, dd)
    assert len(runner.calls) == 1
    with pytest.raises(ValueError, match="guard_estimate is a per-point"):
        replace(_choice(), guard_estimate=1.0)
    with pytest.raises(TypeError):
        VirtualReduction("auto", "ninjars", "x", guard_estimate="1")


@dataclass
class _Outputs:
    """A batch runner of a call with several outputs per point (the
    resolved, correlated and selected routes: a mapping of outputs per
    point): TIRust refuses point 1; NinjaRS serves it with the outputs
    ``outputs`` (key -> (value, cached budget, term scale))."""

    outputs: dict[str, tuple[float, float, float]]
    roundoff: float = 2.0**-104
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def __call__(self, indices, checks):
        from pyamplicol.loops.v2.reduction import current
        from pyamplicol.loops.v2.stability import REDUCTION

        backend = current()
        self.calls.append((backend, checks))
        points = [0, 1] if indices is None else list(indices)
        results: list[Any] = []
        failures: dict[int, list[str]] = {}
        for local, index in enumerate(points):
            if backend == "tirust" and index == 1:
                results.append(None)
                failures[local] = ["TIRust refused point 1"]
                continue
            checked = checks is not None and checks.reduction_direction
            results.append(
                {
                    key: _Point(
                        value,
                        _estimate(value, {REDUCTION: budget}, scale, self.roundoff)
                        if checked
                        else None,
                        None,
                        backend,
                    )
                    for key, (value, budget, scale) in self.outputs.items()
                }
            )
        return results, failures


@pytest.mark.loop_fast
def test_the_guard_decides_on_the_outputs_the_call_returns(loop_native):
    """DECISIONS 2026-11-09 14:30 (RULING Q7): on the resolved, correlated
    and selected routes the guard decides on the outputs the call returns,
    from the fallback's evaluation of those outputs alone: each output's
    cached estimate against its own magnitude; an output that vanishes (its
    magnitude within the estimate's rounding floor, 16 u S) against the
    largest returned output of the point. The one rule is the native
    policy's (``reduction_policy::fallback_estimate``); for one output it is
    the "relative" metric of the output's own estimate."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import fallback_estimate, with_fallback

    u = 2.0**-104
    # One output: its own relative estimate (the summed route's metric).
    assert fallback_estimate([[(4.0, 1e-3, 4.0)]], u) == 1e-3 / 4.0
    # A small output on its own magnitude: rejected, although its error is
    # 1e-8 of the point's largest output (a decision on the sum would pass).
    small = {"h0": (2.0, 2e-12 / 2, 2.0), "h1": (1e-6, 1e-8 / 2, 1.0)}
    runner = _Outputs(small)
    results, failures, records = with_fallback(
        runner, _choice(), 2, VirtualStabilityChecks(), "double-double"
    )
    # One evaluation of the served point, carrying the cached check.
    assert [call[0] for call in runner.calls] == ["tirust", "ninjars"]
    assert runner.calls[1][1].reduction_direction and not runner.calls[1][1].rescaled
    assert results[1] is None and "reduction-direction check" in failures[1][0]
    assert records[1].guard_estimate == pytest.approx(1e-8 / 1e-6, rel=1e-12)
    # A vanishing output (a numerical zero: 1e-31 <= 16 u S): judged against
    # the largest returned output, accepted, its error 3e-30 of it.
    zero = {"h0": (2.0, 2e-12 / 2, 2.0), "h1": (1e-31, 3e-30 / 2, 2.0)}
    assert 16 * u * 2.0 >= 1e-31
    results, failures, records = with_fallback(
        _Outputs(zero), _choice(), 2, None, "double-double"
    )
    assert 1 not in failures, failures
    # (Each estimate is SAFETY x budget + 16 u S.)
    floor = 16 * u * 2.0
    assert records[1].guard_estimate == pytest.approx(
        max((2e-12 + floor) / 2.0, (3e-30 + floor) / 2.0), rel=1e-12
    )
    # The accepted outputs are those of an evaluation without checks.
    assert results[1] == {
        "h0": _Point(2.0, None, None, "ninjars"),
        "h1": _Point(1e-31, None, None, "ninjars"),
    }
    # Alone, the same output is its own largest one: rejected.
    results, failures, records = with_fallback(
        _Outputs({"h1": zero["h1"]}), _choice(), 2, None, "double-double"
    )
    assert results[1] is None
    assert records[1].guard_estimate == pytest.approx(
        (3e-30 + floor) / 1e-31, rel=1e-12
    )


@pytest.mark.loop_fast
def test_the_guard_stays_on_under_the_unchecked_setting(loop_native):
    """Q1 (ruled, DECISIONS 2026-11-09 06:21 (2): the unchecked setting
    governs only the backends' own checks; pyAmpliCol's checks stay
    separate settings acting on the unchecked values): a call under
    ``reduction_unchecked=True`` whose fallback serves a point (a residual
    class of TIRust's) guards it as without the setting, with the same
    decisions, estimates and reasons."""

    from pyamplicol.loops.v2.reduction import with_fallback

    outcomes = []
    for unchecked in (False, True):
        runner = _Runner(refused={1, 4}, cached={1: 1e-9, 4: 3.0})
        choice = replace(_choice(), unchecked=unchecked)
        results, failures, records = with_fallback(
            runner, choice, 8, None, "double-double"
        )
        assert [call[:2] for call in runner.calls] == [
            ("tirust", None),
            ("ninjars", (1, 4)),
        ]
        assert results[1] is not None and results[4] is None
        assert "reduction-direction check" in failures[4][0]
        outcomes.append(
            (
                results,
                failures,
                {k: (r.guard_estimate, r.reason) for k, r in records.items()},
            )
        )
    assert outcomes[0] == outcomes[1]


# ------------------------------------------- g g > z z z through the public API

# ``generate`` (tests/unit/loop_v2_induced_observables_rows.py) in a fresh
# interpreter on this process's import path: argv = the path (JSON), root,
# fixture name, process; the last line of its output is the artifact's path.
_GENERATE = r"""
import json
import sys
from pathlib import Path

sys.path[:] = json.loads(sys.argv[1])
from tests.unit.loop_v2_induced_observables_rows import generate

artifact = generate(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
print(json.dumps(str(artifact)))
"""


def _generated(root: Path, name: str, expression: str) -> Path:
    """Fixture row ``name``'s artifact of ``expression``, generated through
    the public Generator (``generate``, the same helper and options) in a
    fresh interpreter, one per artifact; the test process only loads it.

    With PyPI symbolica 3.0.3 (the CI release lane) the Python API's results
    depend on what the same process generated before (the process-history
    defect fixed by symbolica-dev/symbolica#163, not in 3.0.3; ~1e-30
    relative normally): g g > z z z generated and then g g > h h in one
    process (29 "Imported symbol ... was previously defined with
    user-defined functions" warnings) moves the Python API's guard
    estimates at the near-forward g g > h h points (NinjaRS amplifies the
    difference ~1/Gram there) from the SDK's, which only loads the artifact,
    by 8 % at point 0 (CI run 42758711721, shard 6) up to a factor 19, and
    one resolved-route estimate across the guard's limit (4.4e-5 against
    the SDK's 1.5e-7). A fresh process per artifact gives the SDK's
    estimates bitwise, as the Community build does in one process; the test
    process loading both artifacts is clean (no such warning), so every
    comparison below stays strict."""

    done = subprocess.run(
        [
            sys.executable,
            "-c",
            _GENERATE,
            json.dumps(sys.path),
            str(root),
            name,
            expression,
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    artifact = Path(json.loads(done.stdout.strip().splitlines()[-1]))
    assert artifact == root / "artifact" and artifact.is_dir(), artifact
    return artifact


@pytest.fixture(scope="module")
def zzz(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > z z z (LI-34) through the public Generator (process ``p``)."""

    return _generated(tmp_path_factory.mktemp("zzz"), "li_gg_zzz", "g g > z z z")


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


ESTIMATE = r"estimate [0-9.]+e-?[0-9]+ > [0-9.]+e-?[0-9]+ = max\(target "


def _guarded_failure(lane, target: float = TARGET) -> None:
    """A typed failure of a point the fallback evaluated: its record names
    the fallback; where the fallback served it, the guard's estimate is
    above the guard's limit at the call's ``target`` and the one reason
    names it."""

    import re

    record = lane.reduction
    assert lane.status == "failed", lane
    assert record is not None and record.requested == "auto", record
    assert record.effective == "ninjars", record
    assert "NinjaRS" in lane.detail, lane.detail
    if record.guard_estimate is not None:
        assert not record.guard_estimate <= _limit(target), record
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
    if not _tirust(loop_native):
        # Without TIRust default "auto" is NinjaRS binary64 for the whole
        # call, with no per-point fallback: explicit NinjaRS's documented
        # near-planar hazard (8.9e13 off at eps 1e-11, ninjars-dd-zzz
        # C/out/tilt.json 'nj_b64'), which lit-ninja-planar's conditioning
        # bound addresses; (c') is the default (TIRust) path's.
        found = runtime.evaluate(
            points[:1], mu_r2=scales[:1], poles="none", on_failure="return"
        )
        assert found.diagnostics[0].reduction.fallback is None
        return
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
def test_a_binary64_fallback_value_is_served_unguarded(zzz, loop_native):
    """Binary64 (USER 2026-11-09 ~11:53, 'Double-double+ only'; DECISIONS
    11:53): at the tilt-1e-11 point TIRust's method-choice lanes deviate by
    r of their term scale; at the target whose calibrated acceptance is
    r / 2 TIRust refuses the point and the per-point fallback serves
    NinjaRS's binary64 value as before the guard (8.9e13 off the truth at
    pin 5b6a8f57, ninjars-dd-zzz C/out/tilt.json 'nj_b64': the near-planar
    hazard that lit-ninja-planar's conditioning bound in NinjaRS's fast
    engine addresses): bitwise the explicit NinjaRS value at that target,
    its record NinjaRS with the served reason and no guard estimate; where
    NinjaRS refuses the point (that bound), a failure with both backends'
    reasons."""

    from pyamplicol import Runtime
    from pyamplicol.loops.v2.reduction import text

    _keys, points, scales, _truths = _tilt(["1e-11"])
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
    ninjars = runtime.evaluate(
        points, reduction="ninjars", reduction_target=target, **options
    )
    found = runtime.evaluate(points, reduction_target=target, **options)
    lane = found.diagnostics[0]
    assert lane.reduction.effective == "ninjars", lane.reduction
    assert lane.reduction.guard_estimate is None, lane.reduction
    if ninjars.values[0] is None:
        assert found.values[0] is None and "NinjaRS fallback: " in lane.detail
        return
    assert repr(found.values[0]) == repr(ninjars.values[0])
    served = text("served", "tirust", "ninjars", "")
    assert lane.reduction.reason.startswith(served[:-1]), lane.reduction.reason
    assert "method-choice" in lane.reduction.reason, lane.reduction.reason


@pytest.mark.loop_nightly
def test_the_default_rescue_stack_escalates_the_guarded_fallback(zzz, loop_native):
    """(loop_nightly: ~340 s, its 57-digit level; DECISIONS 2026-11-09 14:30:
    the CI tier keeps the guard's rejections on every route below.) The
    loop-induced default stack (TIRust b64 -> NinjaRS b64 ->
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
        # The levels as before the guard (the base's record: TIRust and
        # NinjaRS binary64 fail both points, the double-double level misses
        # the target, the 57-digit level serves them).
        assert [
            (a.arithmetic, a.reduction.requested, a.failure is None)
            for a in record.attempts
        ] == [
            ("double", "tirust", False),
            ("double", "ninjars", False),
            ("double-double", "auto", False),
            ("arbitrary", "auto", True),
        ], record.attempts
        reduction = attempt.reduction
        assert (reduction.requested, reduction.effective) == ("auto", "ninjars")
        assert "NinjaRS" in attempt.failure
        if reduction.guard_estimate is not None:
            # The fallback served the point: the guard's typed failure.
            assert not reduction.guard_estimate <= _limit(TARGET), reduction
            assert "fallback's value failed its reduction-direction check" in (
                attempt.failure
            )
            assert attempt.failure.endswith("(double-double)"), attempt.failure


# ------------------------------------- g g > h h: double-double and arbitrary


@pytest.fixture(scope="module")
def hh(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > h h through the public Generator (process ``p``) and its
    near-forward stress points (the fixture's ``hh_forward``)."""

    artifact = _generated(tmp_path_factory.mktemp("hh"), "li_gg_hh", "g g > h h")
    found = _fixture()["hh_forward"]
    return artifact, found["points"], [found["mu_r2"]] * len(found["points"])


def _guard_split(found, explicit, target: float) -> tuple[list[int], list[int]]:
    """The points the per-point fallback served (accepted: the explicit
    NinjaRS value bitwise, its estimate within the guard's limit at
    ``target``) and those its guard rejected (typed, the estimate above the
    limit in the reason)."""

    accepted, rejected = [], []
    for p, lane in enumerate(found.diagnostics):
        record = lane.reduction
        if record.effective != "ninjars":
            assert record.guard_estimate is None, p
            continue
        assert record.guard_estimate is not None, (p, lane.detail)
        if found.values[p] is not None:
            assert record.guard_estimate <= _limit(target), (p, record)
            assert repr(found.values[p]) == repr(explicit.values[p]), p
            accepted.append(p)
        else:
            _guarded_failure(lane, target)
            rejected.append(p)
    return accepted, rejected


@pytest.mark.loop_physics
def test_the_guard_at_double_double_and_arbitrary_precision(hh, loop_native):
    """The guard in the extended carriers through the public API (Done 1):
    at the near-forward g g > h h points TIRust's gates refuse all three
    points in double-double (points 1 and 2 also at 16-20 digits); the
    fallback serves them on NinjaRS in the call's own carrier and the guard
    runs its check there. Double-
    double and 20 digits accept them (estimates within 1e-5, values bitwise
    the explicit NinjaRS ones, no public estimate without checks); at 16
    digits and the target 1e-6 (estimates 2.3e-6 and 1.9e-2 at pin
    5b6a8f57) the guard's limit is 1e-5, never stricter (USER 2026-11-09
    ~11:53): it accepts the point whose estimate lies between the target and
    1e-5 and rejects the other typed, naming the arithmetic.

    These decisions were derived on the fixture's binary64 texts as given:
    they are asserted with the kinematics improvement off (m-lit
    lit-psimprove (G8), DECISIONS 2026-11-10 01:30 (F)); the improved points
    are ``test_the_guard_on_improved_near_forward_points``'s."""

    from pyamplicol import Runtime

    artifact, points, scales = hh
    runtime = Runtime.load(artifact, process="p")
    options = {
        "mu_r2": scales,
        "poles": "none",
        "on_failure": "return",
        "kinematics_improvement": "off",
    }
    if not _tirust(loop_native):
        found = runtime.evaluate(points, arithmetic="double-double", **options)
        assert all(lane.reduction.fallback is None for lane in found.diagnostics)
        return
    for carrier, digits, target, expect in (
        ("double-double", {}, TARGET, "accept"),
        ("arbitrary", {"precision": 20}, TARGET, "accept"),
        ("arbitrary", {"precision": 16}, 1e-6, "both"),
    ):
        call = {**options, "arithmetic": carrier, **digits, "reduction_target": target}
        explicit = runtime.evaluate(points, reduction="ninjars", **call)
        found = runtime.evaluate(points, **call)
        assert found.stability is None
        accepted, rejected = _guard_split(found, explicit, target)
        assert accepted if expect == "accept" else rejected, (carrier, digits)
        if expect == "both":
            # The floor: accepted above the call's target, within 1e-5.
            assert any(
                target < found.diagnostics[p].reduction.guard_estimate <= FLOOR
                for p in accepted
            ), [lane.reduction for lane in found.diagnostics]
        for p in rejected:
            assert found.diagnostics[p].detail.endswith(f"({carrier})")


@pytest.mark.loop_physics
def test_the_guard_on_improved_near_forward_points(hh, loop_native):
    """The g g > h h near-forward points with the kinematics improvement
    (m-lit lit-psimprove (G8); DECISIONS 2026-11-10 01:30 (F), 01:56 (4)).
    The fixture's frame has its beams off the z axis.

    (i) ``kinematics_improvement="projection"``: every point is projected
    exactly (its record), and every value the double-double / 20- / 32-digit
    ``"auto"`` call returns on the summed route lies within max(target,
    1e-5) of the consistent truth of the projected point (TIRust at 80
    digits, the improvement off, on that exact point), or the point fails
    typed: no GUARD MISS. The resolved route (the route of the DECISIONS
    2026-11-10 01:08 misses, points 0 / 1 at 1.06e-4 / 2.13e4 off on these
    texts without the improvement) in double-double: every returned row
    within that limit of the projected point's resolved truth, or the point
    fails typed (measured: point 0 served 1.0e-12 off, points 1 / 2
    rejected typed at estimates 2.7e-5 / 1.0). At 20 / 32 digits the
    resolved route is recorded, not asserted (lit-psimprove fix round 1,
    ``tmp/lit-psimprove/fgh/xt-table.txt``; items for the coordinator, not
    input consistency, the projected points being exact): TIRust, the
    primary, serves point 0 0.39 off at 20 digits and point 1 0.26 off at
    32 (the TIRust low-digit side finding, DECISIONS 01:30 ROUTING (3)); at
    20 digits the NinjaRS fallback serves point 2's suppressed rows 1.5e-3
    off at estimate 1.6e-10 (NinjaRS's rounding at 40 working digits,
    amplified ~1/Gram on rows ~1e-21 of the largest, which the cached
    reduction-direction check does not see; 1.2e-15 at 32 digits, where the
    guard rejects it at 5.7e-2).

    (ii) the default (MadLoop's dispatch, DECISIONS 2026-11-10 01:56 (1)):
    ORIG refuses the frame (code 200) and PSMC produces the point, which
    keeps PSMC's (x^2 - x) p1 conservation residual: the record carries the
    typed exactness warning naming PSMC and the residual (01:56 (3)). Its
    values and the guard's decisions there are recorded, not asserted
    correct: lit-ninja-fwd's input-consistency refusal (E_in) is the pending
    defence in depth for the default chain in such frames (01:56 (4)).
    Recorded on these decimal texts (fix round 1, against the TIRust-120
    truth of each PSMC-improved point; ``fgh/xt-table.txt``): the summed
    route within 1e-5 at every carrier (double-double 1.8e-17 / 6.4e-16 /
    2.9e-12); the resolved route in double-double serves point 0 2.6e-4 off
    at estimate 6.0e-10 and point 1 4.77 off at 7.8e-6 (both GUARD MISSES:
    the 01:08 misses persist at the default on the texts) and rejects point
    2 typed (1.3e-3); at 20 digits TIRust serves point 0 0.42 off and the
    fallback serves points 1 / 2 4.8 / 3.7e8 off at 1.6e-10 / 6.8e-10; at 32
    digits TIRust serves point 1 1.2 off and the fallback point 2 3.7e8 off
    at 1.4e-9. On C's exact binary64 values of the same points (another
    exact input, (G7)) the double-double resolved route serves point 0
    3.7e-7 off and rejects point 1: the default's outcome in such a frame
    depends on the input's rounding."""

    from pyamplicol import Runtime

    artifact, points, scales = hh
    runtime = Runtime.load(artifact, process="p")
    backend = runtime._backend
    options = {"mu_r2": scales, "poles": "none", "on_failure": "return"}
    for carrier, digits in (
        ("double-double", {}),
        ("arbitrary", {"precision": 20}),
        ("arbitrary", {"precision": 32}),
    ):
        call = {**options, "arithmetic": carrier, **digits, "reduction_target": TARGET}
        projected, records = backend.improved_points(
            points, variant="projection", mu_r2=scales, arithmetic=carrier, **digits
        )
        assert all(r.produced == "projection" and r.exact for r in records), records
        found = runtime.evaluate(points, kinematics_improvement="projection", **call)
        if _tirust(loop_native):
            truth = runtime.evaluate(
                projected,
                mu_r2=scales,
                poles="none",
                arithmetic="arbitrary",
                precision=80,
                reduction="tirust",
                kinematics_improvement="off",
            )
        for p, lane in enumerate(found.diagnostics):
            assert lane.kinematics.produced == "projection", lane.kinematics
            if found.values[p] is None:
                assert lane.status == "failed" and lane.detail, lane
            elif _tirust(loop_native):
                exact = str(truth.values[p][0].finite)
                value = ((found.values[p][0].finite,),)
                assert _error(value, exact) <= _limit(TARGET), (
                    carrier,
                    digits,
                    p,
                    lane,
                )
        if carrier == "double-double" and _tirust(loop_native):
            # The resolved route of the 01:08 misses: every returned row
            # within the guard's limit of the projected point's truth, or
            # the point fails typed.
            resolved = runtime.evaluate_resolved(
                points, kinematics_improvement="projection", **call
            )
            rows_truth = runtime.evaluate_resolved(
                projected,
                mu_r2=scales,
                poles="none",
                arithmetic="arbitrary",
                precision=80,
                reduction="tirust",
                kinematics_improvement="off",
            )
            for p, lane in enumerate(resolved.diagnostics):
                assert lane.kinematics.produced == "projection", lane.kinematics
                if resolved.values[p] is None:
                    assert lane.status == "failed" and lane.detail, lane
                    continue
                for row, exact in zip(
                    resolved.values[p][0], rows_truth.values[p][0], strict=True
                ):
                    value = ((row[0].finite,),)
                    assert _error(value, str(exact[0].finite)) <= _limit(TARGET), (
                        p,
                        lane,
                    )
        found = runtime.evaluate(points, **call)
        for lane in found.diagnostics:
            record = lane.kinematics
            assert record.attempts == (("z-shift", 200), ("rescaling", 0)), record
            assert record.produced == "rescaling" and not record.exact, record
            assert record.warning == "kinematics-inexact", record
            assert "rescaling" in record.warning_text


# ------------------------------------------------------------------ the SDKs


def _c_inputs(points, scales):
    """The C ABI's binary64 inputs of the fixture's binary64 texts (each
    text's nearest double)."""

    return (
        [[[float(c) for c in leg] for leg in point] for point in points],
        [float(scale) for scale in scales],
    )


def _same_inputs(points, scales):
    """The Python API's texts of the C ABI's inputs (``_c_inputs``): the
    exact decimal expansions of the doubles. The SDK reads every binary64
    input as its exact value, the Python API a text exactly; a shortest
    decimal text (the fixture's) lies up to half an ulp from its double
    (docs/user/native-apis.md), so only these texts give both APIs the same
    point."""

    c_points, c_scales = _c_inputs(points, scales)
    return (
        [[[str(Decimal(c)) for c in leg] for leg in point] for point in c_points],
        [str(Decimal(scale)) for scale in c_scales],
    )


ULPS = 4


def _per_point(found: dict) -> list:
    """The C ABI's values per point (None: a failed point, which returns
    none)."""

    from tests.unit.test_loop_v2_sdk_induced import FAILED_STATUS

    served = [status != FAILED_STATUS for status in found["statuses"]]
    width = len(found["virtual"]) // max(1, sum(served))
    values = iter(found["virtual"])
    return [[next(values) for _ in range(width)] if ok else None for ok in served]


def _ulps(a: float, b: float) -> float:
    """The distance of two estimates in units in the last place."""

    return abs(a - b) / math.ulp(max(abs(a), abs(b)))


def _same_guard(sdk_record: dict, python, arithmetic: str) -> None:
    """The SDK's per-point fallback record (diagnostics "fallbacks") against
    the Python API's on the same exact inputs: the same points, backend and
    decision, the guard's estimates equal within ``ULPS`` units in the last
    place (the colour-metric norms of the amplitude vectors: the Python
    assembly forms them in the carrier, the SDK in binary64,
    ``induced.quadratic_budget`` / ``InducedShape::g_norm``; bitwise or one
    ulp apart on record) and none in binary64, where the guard does not
    run, and the same reason where the estimates agree bitwise."""

    expected = {
        str(index): lane.reduction
        for index, lane in enumerate(python.diagnostics)
        if lane.reduction is not None and lane.reduction.effective == "ninjars"
    }
    assert set(sdk_record) == set(expected), (arithmetic, sdk_record, expected)
    for key, record in expected.items():
        mine = sdk_record[key]
        served = python.values[int(key)] is not None
        assert mine["effective"] == ("ninjars" if served else None), (arithmetic, key)
        theirs = record.guard_estimate
        if arithmetic == "double" or theirs is None or mine["guard_estimate"] is None:
            assert theirs is None and mine["guard_estimate"] is None, (key, mine)
        else:
            assert _ulps(mine["guard_estimate"], theirs) <= ULPS, (key, mine, theirs)
        if mine["guard_estimate"] == theirs:
            assert mine["reason"] == record.reason, (arithmetic, key)
        if not served:
            assert mine["reason"].endswith(f"({arithmetic})") or theirs is None, key


@pytest.mark.loop_physics
def test_the_sdk_guard_fails_typed_and_records_its_estimate(
    zzz, hh, loop_native, tmp_path
):
    """The SDK's ``with_fallback`` guard through the C ABI and the C++
    wrapper (m-lit lit-fallback-guard Done 2) against the Python API on the
    SAME exact inputs (``_same_inputs``: the doubles' decimal expansions;
    given the shortest decimal texts instead, the Python API evaluates
    points up to half an ulp away, docs/user/native-apis.md):

    * binary64, the tilt-1e-11 point at the target that refuses TIRust
      there: no guard (USER 2026-11-09 ~11:53, the extended carriers only):
      the fallback's value as the Python API serves it, no guard estimate in
      either record;
    * double-double, z z z points 44 and 57 (the defect): status Failed
      (not AccuracyNotEstablished) as in the Python API: at NinjaRS
      5b6a8f57 the guard's rejection naming the carrier, its estimates in
      the "fallbacks" record equal to the Python API's (within ``ULPS``; on
      record 23.14776330598882 vs 23.147763305988818 and 31.352350506280747
      on both sides); at 71eddb15 NinjaRS's conditioning bound refuses the
      points, no estimate on either side;
    * the g g > h h near-forward points in double-double, at 20 digits and
      at 16 digits with the target 1e-6: the Python API's decisions and
      estimates per point, every accepted value bitwise the SDK's explicit
      NinjaRS value (the summed call's fallback evaluation carries the
      guard's check: one evaluation per served point) and equal to the
      Python API's;
    * the C++ wrapper returns the C ABI's double-double record of g g > h h.
    """

    import subprocess

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_extras import SDK, _compiler, _sdk, _staged_sdk
    from tests.unit.test_loop_v2_sdk_induced import (
        FAILED_STATUS,
        FALLBACK_CLIENT,
        SUCCESS_STATUS,
        _evaluate,
        _open,
    )

    sdk = _sdk()
    set_target = sdk.rusticol_virtual_set_reduction_target
    set_target.argtypes = [ctypes.c_void_p, ctypes.c_double]
    runtime = Runtime.load(zzz, process="p")
    _keys, tilt, tilt_scales, _truths = _tilt(["1e-11"])
    stream, stream_scales, _ = _stream()
    if not _tirust(loop_native):
        handle = _open(sdk, zzz)
        try:
            found = _evaluate(sdk, handle, *_c_inputs(tilt, tilt_scales))
            assert found["diagnostics"]["reduction"]["fallback"] is None
        finally:
            sdk.rusticol_virtual_free(handle)
        return
    options = {"poles": "none", "on_failure": "return"}
    same_tilt, same_tilt_scales = _same_inputs(tilt, tilt_scales)
    refused = runtime.evaluate(
        same_tilt,
        mu_r2=same_tilt_scales,
        reduction="tirust",
        reduction_target=1e-300,
        **options,
    )
    target = _ratio(refused.diagnostics[0].detail) / 2 / loop_native.tir_acceptance(1.0)

    def call(artifact, points, scales, arithmetic=b"double", digits=0, **settings):
        handle = _open(sdk, artifact)
        try:
            assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0
            for name, value in settings.items():
                assert getattr(sdk, f"rusticol_virtual_set_{name}")(*value(handle)) == 0
            assert sdk.rusticol_virtual_set_arithmetic(handle, arithmetic, digits) == 0
            return _evaluate(sdk, handle, *_c_inputs(points, scales))
        finally:
            sdk.rusticol_virtual_free(handle)

    # Binary64: no guard; the fallback's value as the Python API's.
    python = runtime.evaluate(
        same_tilt, mu_r2=same_tilt_scales, reduction_target=target, **options
    )
    found = call(zzz, tilt, tilt_scales, reduction_target=lambda h: (h, target))
    (record,) = found["diagnostics"]["fallbacks"].values()
    assert record["guard_estimate"] is None
    _same_guard(found["diagnostics"]["fallbacks"], python, "double")
    if python.values[0] is None:
        assert found["statuses"] == [FAILED_STATUS]
    else:
        assert found["statuses"] == [SUCCESS_STATUS]
        assert record["effective"] == "ninjars"
        assert _per_point(found)[0][0] == float(python.values[0][0].finite)

    # Double-double at the defect's points: rejected typed, as the Python
    # API rejects them.
    same_stream, same_stream_scales = _same_inputs(stream, stream_scales)
    python = runtime.evaluate(
        same_stream,
        mu_r2=same_stream_scales,
        arithmetic="double-double",
        **options,
    )
    found = call(zzz, stream, stream_scales, b"double-double")
    assert found["statuses"] == [FAILED_STATUS, FAILED_STATUS]
    for key, item in found["diagnostics"]["fallbacks"].items():
        assert item["effective"] is None, key
        assert item["guard_estimate"] is None or not item["guard_estimate"] <= TARGET
    _same_guard(found["diagnostics"]["fallbacks"], python, "double-double")

    # The extended carriers on g g > h h: the Python API's decisions and
    # estimates; accepted values bitwise the explicit NinjaRS ones.
    artifact, points, scales = hh
    same_points, same_scales = _same_inputs(points, scales)
    hh_runtime = Runtime.load(artifact, process="p")
    records = {}
    for arithmetic, digits, chosen in (
        ("double-double", 0, TARGET),
        ("arbitrary", 20, TARGET),
        ("arbitrary", 16, 1e-6),
    ):
        python = hh_runtime.evaluate(
            same_points,
            mu_r2=same_scales,
            arithmetic=arithmetic,
            reduction_target=chosen,
            **options,
            **({"precision": digits} if digits else {}),
        )

        def chosen_target(h, chosen=chosen):
            return (h, chosen)

        found = call(
            artifact,
            points,
            scales,
            arithmetic.encode(),
            digits,
            reduction_target=chosen_target,
        )
        explicit = call(
            artifact,
            points,
            scales,
            arithmetic.encode(),
            digits,
            reduction_target=chosen_target,
            reduction=lambda h: (h, b"ninjars"),
        )
        record = found["diagnostics"]["fallbacks"]
        records[arithmetic, digits] = record
        _same_guard(record, python, arithmetic)
        mine, plain = _per_point(found), _per_point(explicit)
        for p, value in enumerate(python.values):
            status = FAILED_STATUS if value is None else SUCCESS_STATUS
            assert found["statuses"][p] == status, (arithmetic, digits, p)
            if value is None:
                continue
            assert mine[p][0] == float(value[0].finite), (arithmetic, digits, p)
            if str(p) in record:
                # Served: bitwise the unguarded (explicit NinjaRS) value.
                assert [repr(x) for x in mine[p]] == [repr(x) for x in plain[p]]

    # The C++ wrapper: the C ABI's double-double record.
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
    c_points, c_scales = _c_inputs(points, scales)
    lines = [f"{len(c_points)} {len(c_points[0])}"]
    for point, scale in zip(c_points, c_scales, strict=True):
        lines.append(" ".join(v.hex() for v in [*(c for p in point for c in p), scale]))
    (tmp_path / "input.txt").write_text("\n".join(lines) + "\n")
    done = subprocess.run(
        [
            str(tmp_path / "client"),
            str(artifact),
            "p",
            str(tmp_path / "input.txt"),
            repr(TARGET),
            "double-double",
            "none",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    _statuses, _values, diagnostics = done.stdout.splitlines()
    wrapper = json.loads(diagnostics.split(" ", 1)[1])["fallbacks"]
    assert wrapper == records["double-double", 0]
    assert any(item["guard_estimate"] is not None for item in wrapper.values())


# ------------------------- the resolved, correlated and selected routes (Q7)

# The spin source of gluon 1 (test_loop_v2_sdk_induced.SPIN: a literal
# complex vector, not transverse).
SPIN = [(0.5, 0.25), (0.75, -0.125), (0.375, 1.0), (-0.625, 0.5)]


def _lanes(found) -> list:
    """Per point of a Python result: (failed, its per-point fallback
    record)."""

    return [(lane.status == "failed", lane.reduction) for lane in found.diagnostics]


def _correlated_lanes(found: dict) -> list:
    """``_lanes`` of an ``evaluate_correlated_many`` call (one batch: every
    request carries the same per-point records)."""

    lanes = [_lanes(evaluation) for evaluation in found.values()]
    assert all(item == lanes[0] for item in lanes), lanes
    return lanes[0]


def _sdk_correlated(sdk, handle, points, scales, spin=None) -> dict:
    """One ``rusticol_virtual_evaluate_correlated`` call (with ``spin``: the
    ``_with_sources`` one, one broadcast row): statuses, the decimal record
    and the diagnostics."""

    from tests.unit.test_loop_v2_sdk_extras import _error, _request, _Sources
    from tests.unit.test_loop_v2_sdk_induced import _json

    request, _keep = _request(points, scales, False)
    result = ctypes.c_void_p()
    if spin is None:
        status = sdk.rusticol_virtual_evaluate_correlated(
            handle, ctypes.byref(request), ctypes.byref(result)
        )
    else:
        legs = sorted(spin)
        row = [x for leg in legs for pair in spin[leg] for x in pair]
        values = (ctypes.c_double * len(row))(*row)
        sources = _Sources(
            (ctypes.c_size_t * len(legs))(*legs), len(legs), values, len(row) // 2
        )
        status = sdk.rusticol_virtual_evaluate_correlated_with_sources(
            handle, ctypes.byref(request), ctypes.byref(sources), ctypes.byref(result)
        )
    assert status == 0, _error(sdk)
    try:
        statuses = (ctypes.c_uint32 * len(points))()
        assert (
            sdk.rusticol_virtual_correlated_result_copy_statuses(
                result, statuses, len(points)
            )
            == 0
        )
        return {
            "statuses": list(statuses),
            "decimal": _json(
                sdk, sdk.rusticol_virtual_correlated_result_decimal_json, result
            ),
            "diagnostics": _json(
                sdk, sdk.rusticol_virtual_correlated_result_diagnostics_json, result
            ),
        }
    finally:
        sdk.rusticol_virtual_correlated_result_free(result)


def _decimal_per_point(found: dict) -> list:
    """The C ABI's decimal values per point (None: a failed point, which
    has none), from its decimal record (the copy layout over the successful
    points)."""

    from tests.unit.test_loop_v2_sdk_induced import FAILED_STATUS

    served = [status != FAILED_STATUS for status in found["statuses"]]
    flat = found["decimal"]["virtual"]
    width = len(flat) // max(1, sum(served))
    values = iter(flat)
    return [[next(values) for _ in range(width)] if ok else None for ok in served]


def _route_guard(route: str, sdk_found: dict, lanes: list, arithmetic: str, target):
    """The guard of one route in both APIs on the same exact inputs: the
    same points evaluated again on the fallback, the same decisions, the
    guard's estimates equal within ``ULPS`` (none where the fallback failed
    the point); a rejected point fails typed (status Failed in the SDK) with
    the one reason naming the arithmetic and an estimate above the limit, an
    accepted one is within it. Returns (served, rejected) point indices."""

    from tests.unit.test_loop_v2_sdk_induced import FAILED_STATUS

    record = sdk_found["diagnostics"]["fallbacks"]
    served, rejected = [], []
    for p, (failed, reduction) in enumerate(lanes):
        if reduction is None or reduction.effective != "ninjars":
            assert str(p) not in record, (route, p, record)
            continue
        mine, theirs = record[str(p)], reduction.guard_estimate
        assert (sdk_found["statuses"][p] == FAILED_STATUS) == failed, (route, p)
        if theirs is None:
            # The fallback failed the point too.
            assert failed and mine["guard_estimate"] is None, (route, p, mine)
            continue
        assert mine["guard_estimate"] is not None, (route, p, mine)
        assert _ulps(mine["guard_estimate"], theirs) <= ULPS, (route, p, mine, theirs)
        if failed:
            assert not theirs <= _limit(target), (route, p, reduction)
            assert re.search(ESTIMATE, reduction.reason), reduction.reason
            for reason in (reduction.reason, mine["reason"]):
                assert reason.endswith(f"({arithmetic})"), (route, p, reason)
            assert mine["effective"] is None, (route, p, mine)
            rejected.append(p)
        else:
            assert theirs <= _limit(target), (route, p, reduction)
            assert mine["effective"] == "ninjars", (route, p, mine)
            served.append(p)
    return served, rejected


@pytest.mark.loop_physics
def test_a_helicity_selection_at_the_defect_returns_no_value(zzz, loop_native):
    """DECISIONS 2026-11-09 14:30 (RULING Q7) at the defect: g g > z z z
    stream point 44 in double-double (TIRust refuses it at its Gram gate), a
    one-helicity selection of row h1 in both APIs on the same exact inputs
    (Python ``evaluate(helicities=["h1"])``, the C ABI's
    ``rusticol_virtual_evaluate_selected``): the fallback evaluates that row
    only, and the point fails typed in both APIs with the same record (SDK
    status Failed), never with a value. Before NinjaRS's conditioning bound
    (pin 5b6a8f57) NinjaRS served the row 8.6e4 times its truth and the
    guard rejected it on its estimate 1.4e2
    (``tmp/lit-fallback-guard/fix3/probe10.json``); with the bound (NinjaRS
    71eddb15, lit-ninja-planar) NinjaRS refuses it (``CutFailed {
    SingularParameter }``): either way a typed failure, and
    ``on_failure="raise"`` raises ``EvaluationError`` naming the NinjaRS
    fallback."""

    from pyamplicol import Runtime
    from pyamplicol.api.errors import EvaluationError
    from tests.unit.test_loop_v2_sdk_extras import _sdk
    from tests.unit.test_loop_v2_sdk_induced import _open
    from tests.unit.test_loop_v2_sdk_selection import _call

    points, scales, _truths = _stream(("44",))
    runtime = Runtime.load(zzz, process="p")
    same, same_scales = _same_inputs(points, scales)
    c_points, c_scales = _c_inputs(points, scales)
    options = {
        "mu_r2": same_scales,
        "arithmetic": "double-double",
        "poles": "none",
        "helicities": ["h1"],
    }
    python = runtime.evaluate(same, on_failure="return", **options)
    if not _tirust(loop_native):
        assert all(lane.reduction.fallback is None for lane in python.diagnostics)
        return
    sdk = _sdk()
    handle = _open(sdk, zzz)
    try:
        assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0
        assert sdk.rusticol_virtual_set_arithmetic(handle, b"double-double", 0) == 0
        found = _call(sdk, handle, c_points, c_scales, ids=["h1"])
    finally:
        sdk.rusticol_virtual_free(handle)
    assert found["status"] == 0, found
    served, _rejected = _route_guard(
        "selected", found, _lanes(python), "double-double", TARGET
    )
    assert not served and python.values == (None,), _lanes(python)
    assert "NinjaRS" in python.diagnostics[0].detail
    with pytest.raises(EvaluationError, match="NinjaRS"):
        runtime.evaluate(same, on_failure="raise", **options)


@pytest.mark.loop_physics
def test_the_routes_guard_in_double_double_and_arbitrary_precision(hh, loop_native):
    """The guard on the resolved, correlated and selected routes in both
    extended carriers and both APIs (DECISIONS 2026-11-09 14:30, RULING
    Q7), on the same exact inputs: at the near-forward g g > h h points
    TIRust's gates refuse points (all three in double-double, 1 and 2 at 16
    digits) and the fallback serves them in the call's own arithmetic on a
    helicity selection, the resolved call, the correlated call and the
    spin-correlated call (a literal source on gluon 1). In both APIs the
    guard's estimate of the call's outputs is the same (bitwise on record,
    ``ULPS`` allowed) and so is the decision. Double-double: the accepted
    outputs are the explicit NinjaRS call's bitwise (Python ``repr``, the C
    ABI's decimal record); the resolved call rejects point 2 on its
    suppressed +- rows' own estimates, and ``on_failure="raise"`` raises
    ``EvaluationError`` with the guard's reason. (Accuracy is not this
    test's claim: on these binary64 texts NinjaRS's +- rows move 1.1e-4 ..
    9.3e11 from the exact point's in response to the texts' shell and
    momentum-conservation defect, while TIRust's at 80 digits stay within
    2.3e-6; on the exact points nearby NinjaRS at 60 and 80 digits agrees
    with TIRust at 80 digits; the cause is under investigation (F-GH),
    ``tmp/lit-fallback-guard/fix5/probe15.json``, report §0.) 16
    digits at the target 1e-6 (the limit stays 1e-5): point 1 is accepted
    and point 2 rejected typed on every route
    (estimates 1.6e-6..3.7e-6 and 3.8e-3..1.8e-2 at NinjaRS 5b6a8f57 and 71eddb15,
    ``tmp/lit-fallback-guard/fix3/probe9.json``). These decisions were
    derived on the texts as given: both APIs run with the kinematics
    improvement off (m-lit lit-psimprove (G8), DECISIONS 2026-11-10 01:30
    (F); the improved points are
    ``test_the_guard_on_improved_near_forward_points``'s)."""

    from pyamplicol import Runtime
    from pyamplicol.api.errors import EvaluationError
    from pyamplicol.correlators import CorrelatedRequest
    from tests.unit.test_loop_v2_sdk_extras import _sdk
    from tests.unit.test_loop_v2_sdk_induced import _open
    from tests.unit.test_loop_v2_sdk_selection import _call

    artifact, points, scales = hh
    runtime = Runtime.load(artifact, process="p")
    same, same_scales = _same_inputs(points, scales)
    c_points, c_scales = _c_inputs(points, scales)
    if not _tirust(loop_native):
        found = runtime.evaluate_resolved(
            same, mu_r2=same_scales, arithmetic="double-double", poles="none"
        )
        assert all(lane.reduction.fallback is None for lane in found.diagnostics)
        return
    histories = [item.id for item in runtime.available_color_correlations()]
    assert histories[0] == "born" and len(histories) > 1
    vector = {1: [(str(Decimal(re)), str(Decimal(im))) for re, im in SPIN]}

    def python(options):
        return {
            "selected": runtime.evaluate(same, helicities=["h0"], **options),
            "resolved": runtime.evaluate_resolved(same, **options),
            "correlated": runtime.evaluate_correlated_many(
                same, {h: CorrelatedRequest(h) for h in histories}, **options
            ),
            "spin": runtime.evaluate_correlated_many(
                same, {h: CorrelatedRequest(h, vector) for h in histories}, **options
            ),
        }

    def sdk_calls(arithmetic, digits, target, reduction=None):
        sdk = _sdk()
        sdk.rusticol_virtual_set_reduction_target.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
        ]
        handle = _open(sdk, artifact)
        try:
            assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0
            assert sdk.rusticol_virtual_set_reduction_target(handle, target) == 0
            assert sdk.rusticol_virtual_set_arithmetic(handle, arithmetic, digits) == 0
            assert sdk.rusticol_virtual_set_kinematics_improvement(handle, b"off") == 0
            if reduction is not None:
                assert sdk.rusticol_virtual_set_reduction(handle, reduction) == 0
            return {
                "selected": _call(sdk, handle, c_points, c_scales, ids=["h0"]),
                "resolved": _call(sdk, handle, c_points, c_scales, resolved=True),
                "correlated": _sdk_correlated(sdk, handle, c_points, c_scales),
                "spin": _sdk_correlated(sdk, handle, c_points, c_scales, {1: SPIN}),
            }
        finally:
            sdk.rusticol_virtual_free(handle)

    def lanes_of(result):
        return _correlated_lanes(result) if isinstance(result, dict) else _lanes(result)

    def by_key(result):
        return result if isinstance(result, dict) else {"": result}

    # Double-double: the accepted outputs are the unguarded ones.
    options = {
        "mu_r2": same_scales,
        "arithmetic": "double-double",
        "poles": "none",
        "on_failure": "return",
        "kinematics_improvement": "off",
    }
    found, explicit = python(options), python({**options, "reduction": "ninjars"})
    sdk_found = sdk_calls(b"double-double", 0, TARGET)
    sdk_explicit = sdk_calls(b"double-double", 0, TARGET, b"ninjars")
    for route, result in found.items():
        served, rejected = _route_guard(
            route, sdk_found[route], lanes_of(result), "double-double", TARGET
        )
        assert served, (route, lanes_of(result))
        if route == "resolved":
            # Point 2's suppressed +- rows (1.3e-21 of its ++ rows) have
            # estimates 2.9e-5 / 2.9e-5 on their own magnitudes at the
            # folded artifact (fix5/probe15.json; 3.2e-5 / 3.1e-5 before
            # lit-fold, fix3/probe9.json; the summed estimate is 2.7e-11):
            # rejected.
            assert rejected == [2], lanes_of(result)
        theirs = by_key(explicit[route])
        for key, evaluation in by_key(result).items():
            for p in served:
                assert repr(evaluation.values[p]) == repr(theirs[key].values[p]), (
                    route,
                    key,
                    p,
                )
        mine = _decimal_per_point(sdk_found[route])
        plain = _decimal_per_point(sdk_explicit[route])
        for p in served:
            assert mine[p] is not None and mine[p] == plain[p], (route, p)

    with pytest.raises(EvaluationError, match=ESTIMATE):
        runtime.evaluate_resolved(same, **{**options, "on_failure": "raise"})

    # 16 digits at the target 1e-6: one point accepted, one rejected typed.
    options = {
        "mu_r2": same_scales,
        "arithmetic": "arbitrary",
        "precision": 16,
        "reduction_target": 1e-6,
        "poles": "none",
        "on_failure": "return",
        "kinematics_improvement": "off",
    }
    found, sdk_found = python(options), sdk_calls(b"arbitrary", 16, 1e-6)
    for route, result in found.items():
        decided = _route_guard(
            route, sdk_found[route], lanes_of(result), "arbitrary", 1e-6
        )
        assert decided == ([1], [2]), (route, lanes_of(result))
