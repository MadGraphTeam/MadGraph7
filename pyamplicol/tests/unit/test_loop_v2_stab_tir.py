# SPDX-License-Identifier: 0BSD
"""Stability and rescue with TIR, and per-level backends (m-lit LIT-STAB-T,
PLAN §2.b.4, §2.b.6, §2.a.8; RULED Q17 / Q21 / Q22; DECISIONS 2026-11-07
21:50 R6 / R7).

* ``VirtualRescueLevel.reduction`` (``"auto"`` | ``"ninjars"`` |
  ``"tirust"``, default ``"auto"``) is validated by the one request
  validator; a level's ``"auto"`` takes the call's request; each level
  resolves its backend once, for its carrier, through the one resolver
  (``reduction.resolve``) before any point is evaluated, and every
  ``VirtualRescueAttempt`` records the level's ``VirtualReduction``
  (requested, effective, reason).
* Before lit-tir-mp (capability-driven: the carriers the native extension
  reports for TIRust): an ``"auto"`` level of a loop-induced call whose
  request is ``"auto"`` takes TIRust, the loop-induced default, where the
  native serves TIRust at the level's carrier (binary64; the m-lit F3
  switch, which supersedes R3) and NinjaRS with its reason elsewhere
  (double-double); an explicit ``"tirust"`` double-double level, or an
  ``"auto"`` one inheriting an explicit call-level ``"tirust"``, is refused
  typed (``ReductionUnavailable``) before any point. After it (a native reporting
  TIRust at every carrier, here a capability-reporting stub): those levels
  resolve to TIRust and stay there, and so does every ``"auto"`` level; a
  route whose ``auto`` set leaves TIRust out keeps NinjaRS on them with its
  reason.
* The cross-backend check (``VirtualStabilityChecks.cross_backend``): every
  point again on the other backend, its lane deviation an additive budget
  (``stability.CROSS``) of the estimate; the values are bitwise those of
  the call without it; the budget bounds the two backends' deviation; a
  carrier the other backend does not serve is refused typed before any
  point; a point the other backend fails keeps the other checks' estimate
  bitwise, with the check and its reason in ``not_run`` (R6); it runs on the
  summed route and on the resolved outputs of both process types.
* TIR single-point estimates on the loop-induced calibration rows (g g > h h,
  g g > z; TIRust in its shipped default, rotation + rescaled re-test)
  hold the true error to 10 E, per sector and Laurent coefficient, on the
  points TIRust serves (none served: the row waits on lit-tir-mp, a skip,
  never a vacuous pass).
* The committed TIR calibration (``tools/loop_v2_stability_calibration.py
  --reduction tirust --cross-backend``,
  ``tests/fixtures/loop/v2/stability_calibration_tir.json``): no served point
  beyond 10 E in any configuration; served fractions recorded.
* After lit-tir-mp (ready to run; skipped while the native extension does not
  report TIRust at double-double): a double-double rescue level on TIRust is
  served there, within L6 of NinjaRS's.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = (
    ROOT / "tests" / "fixtures" / "loop" / "v2" / "stability_calibration_tir.json"
)
# USER DECISION (2026-10-23; DESIGN §10.4): a single-point test of a
# stability estimate E holds the true error to at most 10 E.
ESTIMATE_MARGIN = 10
TRUTH_DIGITS = 57
WAITS = "WAITS ON lit-tir-mp"


# "auto"'s reason on a loop-induced call where TIRust and NinjaRS serve
# it (the native policy, m-lit F3: TIRust with the per-point NinjaRS
# fallback).
AUTO_LOOP_INDUCED = (
    "auto: TIRust, the loop-induced default; a point it refuses or fails is "
    "evaluated again on NinjaRS (the per-point fallback, recorded per point)"
)


def _li():
    from pyamplicol._internal import selftest_virtual_li

    return selftest_virtual_li


def _ew():
    from pyamplicol._internal import selftest_virtual_ew

    return selftest_virtual_ew


def _load(module):
    from pyamplicol import Runtime

    root = module.fixture_root()
    expected = json.loads((root / "expected.json").read_text())
    runtime = Runtime.load(
        root / "artifact",
        process=module.NAME,
        model_parameters=expected.get("parameters") or None,
    )
    return runtime, expected["points"], expected["mu_r2"]


def _tirust(native) -> frozenset[str]:
    """The carriers the native extension reports for TIRust."""

    from pyamplicol.loops.v2.reduction import native_backends

    return native_backends(native).get("tirust", frozenset())


def _forced(*levels):
    """A policy whose first level accepts no point (every point climbs)."""

    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy

    first, *rest = levels
    return VirtualRescuePolicy(
        (
            VirtualRescueLevel(
                first.arithmetic,
                first.precision,
                first.rotations,
                first.reduction_check,
                1e-300,
                first.metric,
                first.reduction,
            ),
            *rest,
        )
    )


class _Native:
    """A capability-reporting stub of the native extension (lit-api-b's
    precedent): ``reduction_backends`` only."""

    def __init__(self, record):
        self.record = record

    def reduction_backends(self):
        return json.dumps(self.record)


# ------------------------------------------------------------------- records


@pytest.mark.loop_fast
def test_a_rescue_level_names_its_reduction_backend():
    from pyamplicol.api.virtual_results import (
        VirtualReduction,
        VirtualRescueAttempt,
        VirtualRescueLevel,
        VirtualRescuePolicy,
    )

    assert VirtualRescueLevel("double").reduction == "auto"
    assert all(level.reduction == "auto" for level in VirtualRescuePolicy().levels)
    level = VirtualRescueLevel("double-double", reduction="tirust")
    assert level.reduction == "tirust"
    assert level.backend_request("ninjars") == "tirust"
    assert VirtualRescueLevel("double").backend_request("tirust") == "tirust"
    assert VirtualRescueLevel("double").backend_request("auto") == "auto"
    with pytest.raises(ValueError, match="unknown reduction backend"):
        VirtualRescueLevel("double", reduction="ninja")
    with pytest.raises(ValueError, match="unknown reduction backend"):
        VirtualRescueLevel("double").backend_request("engine")
    choice = VirtualReduction("auto", "ninjars", "auto: NinjaRS")
    attempt = VirtualRescueAttempt("double", 53, 1e-9, reduction=choice)
    assert attempt.reduction is choice
    with pytest.raises(TypeError, match="VirtualReduction"):
        VirtualRescueAttempt("double", 53, 1e-9, reduction="ninjars")  # type: ignore[arg-type]


@pytest.mark.loop_fast
def test_the_cross_backend_check_combines_with_another_check():
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    with pytest.raises(ValueError, match="combine it with a rotation"):
        VirtualStabilityChecks(cross_backend=True)
    with pytest.raises(TypeError, match="cross_backend"):
        VirtualStabilityChecks(1, cross_backend=1)  # type: ignore[arg-type]
    for checks in (
        VirtualStabilityChecks(1, cross_backend=True),
        VirtualStabilityChecks(0, True, cross_backend=True),
    ):
        assert checks.enabled and checks.to_checks().cross_backend
    assert not VirtualStabilityChecks(1).to_checks().cross_backend


@pytest.mark.loop_fast
def test_the_cross_backend_budget_adds_to_the_estimate():
    """E = SAFETY x (max rotation + reduction + cross) + FLOOR_UNITS u S;
    ``not_run`` survives ``subset``; the public record validates it."""

    from pyamplicol.api.virtual_results import VirtualStabilityEstimate
    from pyamplicol.loops.v2.stability import (
        CROSS,
        FLOOR_UNITS,
        RESCALED,
        ROTATION,
        SAFETY,
        Budget,
        CrossRuns,
        StabilityEstimate,
    )

    reference = {(): Budget((1.0,))}
    budgets = {
        ROTATION.format(0): {(): Budget((3e-9,))},
        ROTATION.format(1): {(): Budget((5e-9,))},
        RESCALED: {(): Budget((2e-9,))},
        CROSS: {(): Budget((7e-9,))},
    }
    estimate = StabilityEstimate.from_checks(
        reference, budgets, scale={(): Budget((2.0,))}, roundoff=2.0**-53
    )
    expected = SAFETY * (5e-9 + 2e-9 + 7e-9) + FLOOR_UNITS * 2.0**-53 * 2.0
    assert estimate.absolute[()].coefficients[0] == pytest.approx(expected, rel=1e-15)
    assert estimate.checks[CROSS] == pytest.approx(7e-9)
    skipped = StabilityEstimate.from_checks(
        reference,
        {RESCALED: budgets[RESCALED]},
        scale={(): Budget((2.0,))},
        roundoff=2.0**-53,
        not_run={CROSS: "TIRust failed the point: gates"},
    )
    assert skipped.not_run == {CROSS: "TIRust failed the point: gates"}
    assert skipped.subset([RESCALED]).not_run == skipped.not_run
    runs = CrossRuns("TIRust", ["a", None], {1: ["refused by the gates"]})
    assert runs.of(0) == ("a", "")
    assert runs.of(1) == (None, "TIRust failed the point: refused by the gates")
    refused = CrossRuns("TIRust", [None, None], {}, "not built")
    assert refused.of(1) == (None, "TIRust refused the call: not built")
    record = dict(
        relative=1e-9,
        finite=1e-9,
        single_pole=None,
        double_pole=None,
        checks=(("rotation-0", 1e-9),),
    )
    assert VirtualStabilityEstimate(**record, not_run=((CROSS, "x"),)).not_run == (
        (CROSS, "x"),
    )
    with pytest.raises(ValueError, match="states its reason"):
        VirtualStabilityEstimate(**record, not_run=((CROSS, ""),))
    with pytest.raises(ValueError, match="not as run"):
        VirtualStabilityEstimate(**record, not_run=(("rotation-0", "x"),))


@pytest.mark.loop_fast
def test_the_cross_backend_is_the_other_backend():
    from pyamplicol.loops.v2.reduction import BACKENDS, cross, current, running

    assert {cross(backend) for backend in BACKENDS} == set(BACKENDS)
    assert all(cross(backend) != backend for backend in BACKENDS)
    with pytest.raises(ValueError):
        cross("auto")
    assert current() == "ninjars"
    with running("tirust"):
        assert current() == "tirust"


# --------------------------------------------------- per-level resolution


@pytest.mark.loop_fast
def test_levels_resolve_once_per_level_capability_driven(loop_native, monkeypatch):
    """The per-level resolution on the loop-induced self-test (g g > h),
    'before' on the real native and 'after' on a stub reporting TIRust at
    every carrier (never a constant or a version check)."""

    from pyamplicol import ReductionUnavailable
    from pyamplicol.api.virtual_results import (
        VirtualRescueLevel,
        VirtualRescuePolicy,
        VirtualStabilityChecks,
    )
    from pyamplicol.loops.v2 import reduction
    from pyamplicol.loops.v2.reduction import CARRIERS, SUMMED

    runtime, _points, _scales = _load(_li())
    backend = runtime._backend
    auto = VirtualRescuePolicy(
        (VirtualRescueLevel("double"), VirtualRescueLevel("double-double"))
    )
    explicit = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", reduction="ninjars"),
            VirtualRescueLevel("double-double", reduction="tirust"),
        )
    )

    def choices(policy, request=None, route=SUMMED):
        return backend._reduction_choices(request, None, None, policy, route)

    served = _tirust(loop_native)
    if "double-double" not in served:
        # Before lit-tir-mp: "auto" falls back with its reason, explicit
        # requests (own or inherited) are refused typed.
        found = choices(auto)
        binary64 = "tirust" if "double" in served else "ninjars"
        assert [found[level].effective for level in auto.levels] == [
            binary64,
            "ninjars",
        ]
        reason = found[auto.levels[1]].reason
        assert "TIRust, the loop-induced default, is unavailable" in reason
        assert "double-double" in reason
        with pytest.raises(ReductionUnavailable, match="double-double"):
            choices(explicit)
        if "double" in served:
            with pytest.raises(ReductionUnavailable, match="double-double"):
                choices(auto, "tirust")
    # After lit-tir-mp (the stub reports TIRust at every carrier).
    monkeypatch.setattr(
        backend, "_native", _Native({name: list(CARRIERS) for name in reduction.NAMES})
    )
    monkeypatch.setattr(backend, "_reductions", {})
    found = choices(explicit)
    assert [
        (found[level].requested, found[level].effective) for level in explicit.levels
    ] == [
        ("ninjars", "ninjars"),
        ("tirust", "tirust"),
    ]
    inherited = choices(auto, "tirust")
    assert {choice.effective for choice in inherited.values()} == {"tirust"}
    assert {choice.reason for choice in inherited.values()} == {"requested"}
    # The m-lit F3 switch (superseding R3): the route's "auto" set holds TIRust, so the
    # "auto" levels resolve to TIRust, the loop-induced default; a route
    # whose "auto" set leaves it out keeps NinjaRS with that reason.
    found = choices(auto)
    assert {choice.effective for choice in found.values()} == {"tirust"}
    assert all(choice.reason == AUTO_LOOP_INDUCED for choice in found.values())
    kept = choices(
        auto,
        None,
        reduction.Route(SUMMED.serves, SUMMED.backends, frozenset({"ninjars"})),
    )
    assert {choice.effective for choice in kept.values()} == {"ninjars"}
    assert all("on explicit request only" in choice.reason for choice in kept.values())
    # The cross-backend check is admitted at every carrier the stub serves
    # both backends at, and refused typed by a native without TIRust (a
    # tir-off build: USER 2026-11-06 ~13:35).
    checks = VirtualStabilityChecks(1, cross_backend=True)
    for arithmetic in ("double", "double-double"):
        backend._reduction_choices(None, arithmetic, None, None, SUMMED, checks)
    monkeypatch.setattr(backend, "_native", _Native({"ninjars": list(CARRIERS)}))
    monkeypatch.setattr(backend, "_reductions", {})
    with pytest.raises(ReductionUnavailable, match=r"cross-backend check.*'tir'"):
        backend._reduction_choices(None, "double", None, None, SUMMED, checks)
    assert backend._reduction_choices(None, "double", None, None, SUMMED, None)


@pytest.mark.loop_fast
def test_levels_resolving_to_one_precision_and_backend_are_refused(
    loop_native, monkeypatch
):
    """m-lit F3 (b): the policy refuses a repeated (precision bits,
    reduction) pair of its levels' requests, and a call refuses, before any
    point, levels whose backends resolve to one (precision bits, backend)
    pair: a binary64 "auto" level beside a binary64 "tirust" one on a
    loop-induced "auto" call (both TIRust in binary64), or beside a binary64
    "ninjars" one on a call naming NinjaRS. On a stub reporting TIRust at
    every carrier, so the test decides in every build."""

    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy
    from pyamplicol.loops.v2 import reduction
    from pyamplicol.loops.v2.reduction import CARRIERS, SUMMED

    runtime, points, scales = _load(_li())
    backend = runtime._backend
    monkeypatch.setattr(
        backend, "_native", _Native({name: list(CARRIERS) for name in reduction.NAMES})
    )
    monkeypatch.setattr(backend, "_reductions", {})
    beside_tirust = VirtualRescuePolicy(
        (VirtualRescueLevel("double"), VirtualRescueLevel("double", reduction="tirust"))
    )
    beside_ninjars = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double"),
            VirtualRescueLevel("double", reduction="ninjars"),
        )
    )

    def effective(policy, request=None):
        found = backend._reduction_choices(request, None, None, policy, SUMMED)
        return [found[level].effective for level in policy.levels]

    with pytest.raises(ValueError, match=r"repeated \(precision bits, backend\)"):
        effective(beside_tirust)
    assert effective(beside_tirust, "ninjars") == ["ninjars", "tirust"]
    with pytest.raises(ValueError, match=r"repeated \(precision bits, backend\)"):
        effective(beside_ninjars, "ninjars")
    assert effective(beside_ninjars) == ["tirust", "ninjars"]
    # Through the public call: refused before any point.
    with pytest.raises(ValueError, match=r"repeated \(precision bits, backend\)"):
        runtime.evaluate(points, mu_r2=scales, stability=beside_tirust)


@pytest.mark.loop_physics
def test_rescue_attempts_record_each_level_backend(loop_native):
    """MadLoop's ``MLReductionLib`` order: binary64 on TIRust, then
    double-double on NinjaRS; every point climbs (level 0 accepts none) and
    records both levels' backends; the kept double-double values are
    bitwise NinjaRS's double-double values of the same points. A loop-
    induced "auto" ladder records the resolver's choices: TIRust, the
    loop-induced default, at binary64 (the m-lit F3 switch) and, before
    lit-tir-mp, NinjaRS with its reason at double-double."""

    from pyamplicol.api.virtual_results import VirtualRescueLevel

    if "double" not in _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    runtime, points, scales = _load(_li())
    policy = _forced(
        VirtualRescueLevel("double", reduction="tirust"),
        VirtualRescueLevel("double-double", reduction="ninjars"),
    )
    result = runtime.evaluate(points, mu_r2=scales, stability=policy)
    direct = runtime.evaluate(
        points,
        mu_r2=scales,
        arithmetic="double-double",
        stability_checks=policy.levels[1].checks,
        reduction="ninjars",
    )
    for k, record in enumerate(result.rescue):
        assert record.level == 1 and not record.failed
        first, second = record.attempts
        assert (first.reduction.requested, first.reduction.effective) == (
            "tirust",
            "tirust",
        )
        assert (second.reduction.requested, second.reduction.effective) == (
            "ninjars",
            "ninjars",
        )
        assert record.reduction == second.reduction == result.diagnostics[k].reduction
        assert repr(result.values[k]) == repr(direct.values[k])
    ladder = _forced(VirtualRescueLevel("double"), VirtualRescueLevel("double-double"))
    result = runtime.evaluate(points[:1], mu_r2=scales[:1], stability=ladder)
    first, second = result.rescue[0].attempts
    assert first.reduction.effective == "tirust"
    assert first.reduction.reason == AUTO_LOOP_INDUCED
    if "double-double" not in _tirust(loop_native):
        assert second.reduction.effective == "ninjars"
        assert "is unavailable" in second.reduction.reason


@pytest.mark.loop_physics
def test_explicit_tirust_levels_are_refused_before_any_point(loop_native, monkeypatch):
    """An explicit TIRust level at a carrier TIRust does not serve, its own
    or inherited from the call by "auto", is refused typed when the policy
    is resolved: no point is evaluated. (A native serving TIRust at
    double-double, m-lit lit-tir-mp, reports binary64 only through a
    capability stub here.)"""

    from pyamplicol import ReductionUnavailable
    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy
    from pyamplicol.loops.v2.reduction import CARRIERS, SUMMED

    served = _tirust(loop_native)
    runtime, points, scales = _load(_li())
    backend = runtime._backend
    if "double-double" in served:
        # After lit-tir-mp (capability-driven, never a skip: the CI loop tier
        # admits none): the explicit double-double level resolves to TIRust;
        # the refusal itself is checked below with the capability stub.
        own = VirtualRescuePolicy(
            (
                VirtualRescueLevel("double", reduction="ninjars"),
                VirtualRescueLevel("double-double", reduction="tirust"),
            )
        )
        found = backend._reduction_choices(None, None, None, own, SUMMED)
        assert found[own.levels[1]].effective == "tirust"
        # A fresh runtime (the choices are cached per runtime) on the stub.
        runtime, points, scales = _load(_li())
        backend = runtime._backend
        served = frozenset({"double"})
        monkeypatch.setattr(
            backend,
            "_native",
            _Native({"ninjars": list(CARRIERS), "tirust": sorted(served)}),
        )

    def never(*_args, **_kwargs):
        raise AssertionError("a point was evaluated")

    monkeypatch.setattr(backend, "_evaluate_level", never)
    own = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", reduction="ninjars"),
            VirtualRescueLevel("double-double", reduction="tirust"),
        )
    )
    with pytest.raises(ReductionUnavailable, match="double-double"):
        runtime.evaluate(points, mu_r2=scales, stability=own)
    inherited = VirtualRescuePolicy(
        (VirtualRescueLevel("double"), VirtualRescueLevel("double-double"))
    )
    if "double" in served:
        with pytest.raises(ReductionUnavailable, match="double-double"):
            runtime.evaluate(
                points, mu_r2=scales, stability=inherited, reduction="tirust"
            )


# ------------------------------------------------------- cross-backend check


def _finite(value, sector=0) -> float:
    return float(value[sector].finite)


@pytest.mark.loop_physics
@pytest.mark.parametrize("module", ["li", "ew"])
@pytest.mark.parametrize("main", ["ninjars", "tirust"])
def test_the_cross_backend_check_on_the_summed_route(loop_native, module, main):
    """The values are bitwise those without the check; the estimate names
    ``cross-backend``; its absolute budget bounds the deviation of the two
    backends' values; the other checks' budgets are unchanged."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.stability import CROSS

    if "double" not in _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    runtime, points, scales = _load(_li() if module == "li" else _ew())
    other = {"ninjars": "tirust", "tirust": "ninjars"}[main]
    plain = VirtualStabilityChecks(1, True)
    with_cross = VirtualStabilityChecks(1, True, cross_backend=True)
    base = runtime.evaluate(
        points, mu_r2=scales, stability_checks=plain, reduction=main
    )
    found = runtime.evaluate(
        points, mu_r2=scales, stability_checks=with_cross, reduction=main
    )
    others = runtime.evaluate(points, mu_r2=scales, reduction=other)
    assert repr(found.values) == repr(base.values)
    for k in range(len(points)):
        estimate, reference = found.stability[k], base.stability[k]
        checks = dict(estimate.checks)
        assert CROSS in checks and not estimate.not_run
        assert {n: v for n, v in checks.items() if n != CROSS} == dict(reference.checks)
        for sector in range(len(found.values[k])):
            deviation = abs(
                _finite(found.values[k], sector) - _finite(others.values[k], sector)
            )
            # E holds 2 x the cross budget (SAFETY) on top of the others'.
            gained = estimate.absolute[sector][0] - reference.absolute[sector][0]
            assert gained >= 0
            assert deviation <= gained, (k, sector, deviation, gained)


@pytest.mark.loop_physics
def test_the_cross_backend_check_is_refused_where_the_other_backend_is_not(
    loop_native, monkeypatch
):
    """Before any point: a carrier the other backend does not serve (a
    native serving TIRust at double-double, m-lit lit-tir-mp, reports
    binary64 only through a capability stub here)."""

    from pyamplicol import ReductionUnavailable
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import CARRIERS

    runtime, points, scales = _load(_li())
    if "double-double" in _tirust(loop_native):
        # After lit-tir-mp the check is admitted at double-double (asserted by
        # test_double_double_tir_levels_stay_on_tir_after_lit_tir_mp; never a
        # skip here: the CI loop tier admits none); the refusal is checked
        # with the capability stub.
        monkeypatch.setattr(
            runtime._backend,
            "_native",
            _Native({"ninjars": list(CARRIERS), "tirust": ["double"]}),
        )
    monkeypatch.setattr(
        runtime._backend,
        "_evaluate_level",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("evaluated")),
    )
    with pytest.raises(ReductionUnavailable, match="cross-backend check"):
        runtime.evaluate(
            points,
            mu_r2=scales,
            arithmetic="double-double",
            stability_checks=VirtualStabilityChecks(1, True, cross_backend=True),
        )


@pytest.mark.loop_physics
def test_a_point_the_other_backend_fails_keeps_the_other_estimate(
    loop_native, monkeypatch
):
    """R6: the check is 'not run' for a point the other backend fails, with
    the reason; the point keeps the other checks' estimate bitwise."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2 import plan_export
    from pyamplicol.loops.v2.reduction import current
    from pyamplicol.loops.v2.stability import CROSS

    if "double" not in _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    runtime, points, scales = _load(_li())
    plain = VirtualStabilityChecks(1, True)
    # NinjaRS named as the call's backend, so that the other backend is
    # TIRust whatever "auto" takes (TIRust, the loop-induced default, where it
    # serves the carrier: the m-lit F3 switch).
    base = runtime.evaluate(
        points, mu_r2=scales, stability_checks=plain, reduction="ninjars"
    )
    original = plan_export.NativeExecutor._evaluated

    def failing(self, *args, **kwargs):
        precision, evaluations, failed = original(self, *args, **kwargs)
        if current() == "tirust":
            failed = {**failed, 0: ["refused by a conditioning gate (forced)"]}
        return precision, evaluations, failed

    monkeypatch.setattr(plan_export.NativeExecutor, "_evaluated", failing)
    found = runtime.evaluate(
        points,
        mu_r2=scales,
        stability_checks=VirtualStabilityChecks(1, True, cross_backend=True),
        reduction="ninjars",
    )
    assert repr(found.values) == repr(base.values)
    first = found.stability[0]
    ((name, reason),) = first.not_run
    assert name == CROSS
    assert reason == "TIRust failed the point: refused by a conditioning gate (forced)"
    assert first.checks == base.stability[0].checks
    assert first.absolute == base.stability[0].absolute
    assert all(CROSS in dict(e.checks) for e in found.stability[1:])


@pytest.mark.loop_physics
@pytest.mark.parametrize("module", ["li", "ew"])
def test_the_cross_backend_check_on_the_resolved_outputs(loop_native, module):
    """The resolved outputs of both process types (loop-induced observables;
    Born-interfered stored columns) carry the check per entry, values
    bitwise those without it."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.stability import CROSS

    if "double" not in _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    runtime, points, scales = _load(_li() if module == "li" else _ew())
    plain = runtime.evaluate_resolved(
        points, mu_r2=scales, stability_checks=VirtualStabilityChecks(1, True)
    )
    found = runtime.evaluate_resolved(
        points,
        mu_r2=scales,
        stability_checks=VirtualStabilityChecks(1, True, cross_backend=True),
    )
    assert repr(found.values) == repr(plain.values)
    estimates = [e for e in _leaves(found.stability) if e is not None]
    assert estimates
    for estimate in estimates:
        assert CROSS in dict(estimate.checks) or dict(estimate.not_run).get(CROSS)


def _leaves(value):
    if value is None or hasattr(value, "checks"):
        return [value]
    return [x for item in value for x in _leaves(item)]


# ------------------------------------------------- TIR single-point estimates


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    from .test_loop_v2_induced import _Rows

    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li-stab-tir"))


def _errors(result, truth):
    exact = truth.components["virtual"]
    return {
        sector: [
            abs(Decimal(str(value)) - Decimal(str(reference)))
            for value, reference in zip(
                laurent.coefficients, exact[sector].coefficients, strict=True
            )
        ]
        for sector, laurent in result.components["virtual"].items()
    }


def _tir_single_points(li, name, loop_native):
    """Every fixture point of a calibration row on TIRust (recommended
    checks) against NinjaRS at Float(256) on the same exact point: the
    served count and the worst error / E."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.plan_export import FailedPoints
    from pyamplicol.loops.v2.reduction import running
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    from .test_loop_v2_induced import _fixture

    if "double" not in _tirust(loop_native):
        pytest.skip("the staged native extension is built without TIRust")
    count = len(_fixture(name)["points"])
    served = 0
    for index in range(count):
        try:
            with running("tirust"):
                result = li.native_value(
                    name,
                    Precision.double(),
                    inputs="native",
                    index=index,
                    stability=RECOMMENDED_CHECKS,
                )
        except FailedPoints:
            continue
        served += 1
        truth = li.native_value(
            name, Precision.arbitrary(TRUTH_DIGITS), inputs="native", index=index
        )
        for sector, errors in _errors(result, truth).items():
            estimate = result.stability.absolute[sector].coefficients
            for k, error in enumerate(errors):
                assert float(error) <= ESTIMATE_MARGIN * estimate[k], (
                    index,
                    sector,
                    k,
                    float(error),
                    estimate[k],
                )
    if not served:
        pytest.skip(f"{WAITS}: TIRust serves 0/{count} points of {name}")
    return served


@pytest.mark.loop_physics
def test_tir_single_point_estimates_bracket_the_error_gg_hh(li, loop_native):
    _tir_single_points(li, "li_gg_hh", loop_native)


@pytest.mark.loop_nightly
def test_tir_single_point_estimates_bracket_the_error_gg_z(li, loop_native):
    _tir_single_points(li, "li_gg_z", loop_native)


@pytest.mark.loop_fast
def test_tir_calibration_has_no_served_miss_beyond_the_margin():
    """The recorded TIR calibration (TIRust's shipped default, against
    NinjaRS at Float(256) on the same exact points; 400 points of g g > g g g
    and u u~ > e- e+ d d~ g, the first 100 also in double-double, 2000 of
    g g > h h and g g > z, the first 400 also in double-double; re-run by
    m-lit lit-tir-mp with the binary64 method choice on and TIRust serving
    double-double, and by lit-stab-t-b with the reduction-direction check on
    TIRust's rotation copy): on every served point of either carrier, every
    check configuration with a reduction re-test (TIR's frames: the
    relabelling by one, TIRust's rotation copy, or the rotation by one with
    the origin shift at rescaled inputs) or the cross-backend check holds
    the absolute error within 10 E, per sector and Laurent coefficient, and
    the recommended one (rescaled re-test + a rotation) misses none at all.
    Rotations alone are no TIR estimate (lit-tir finding (B)): they miss
    served points by more than 10 E, which every re-test frame sees. Served
    fractions, refusal reasons and the check carrying each estimate are
    recorded; the cross-backend check leaves every virtual bitwise; a carrier
    TIRust serves no point of is recorded as waiting (none since
    lit-tir-mp)."""

    from pyamplicol.loops.v2.stability import CROSS, FRAMES

    data = json.loads(CALIBRATION.read_text())
    assert data["reduction"] == "tirust"
    rows = data["processes"]
    assert set(rows) == {"gg_ggg", "uux_emepddxg", "li_gg_hh", "li_gg_z"}
    for name, carriers in rows.items():
        for carrier, record in carriers.items():
            assert 0 <= record["served_fraction"] <= 1
            assert record["failed_truth"] == 0, name
            if not record["points"]:
                assert record["status"] == WAITS, (name, carrier)
                continue
            assert carrier in ("double", "double-double"), carrier
            for configuration, rates in record["configurations"].items():
                checks = set(configuration.split(" + "))
                if checks & {*FRAMES, CROSS}:
                    assert rates["absolute"]["beyond_margin"] == 0, (
                        name,
                        configuration,
                    )
            recommended = record["configurations"]["reduction-rescaled + rotation-0"]
            assert recommended["absolute"]["false_negatives"] == 0, name
            assert sum(record["carried_by"].values()) == record["points"]
            cost = record["cross_backend_cost"]
            assert cost["virtual_bitwise"] == cost["compared"] == record["points"]
    assert rows["li_gg_hh"]["double"]["points"] >= 1900
    assert rows["li_gg_z"]["double"]["points"] == 2000
    for name in ("li_gg_hh", "li_gg_z"):
        assert rows[name]["double-double"]["points"] == 400, name
    for name in ("gg_ggg", "uux_emepddxg"):
        assert rows[name]["double-double"]["points"] >= 90, name


# ------------------------------------------- after lit-tir-mp (ready to run)


@pytest.mark.loop_physics
def test_double_double_tir_levels_stay_on_tir_after_lit_tir_mp(loop_native):
    """Runs once the native extension reports TIRust at double-double
    (lit-tir-mp; capability-driven): an explicit TIRust double-double level
    is served on TIRust, its values within L6 (10 max(E_N, E_T)) of a
    NinjaRS double-double level; the cross-backend check is admitted
    there."""

    from pyamplicol.api.virtual_results import (
        VirtualRescueLevel,
        VirtualStabilityChecks,
    )
    from pyamplicol.loops.v2.stability import CROSS

    runtime, points, scales = _load(_li())
    if "double-double" not in _tirust(loop_native):
        # Before lit-tir-mp (capability-driven, never a skip: the CI loop tier
        # admits none): the explicit TIRust double-double level is refused
        # typed; the full check below runs once the native serves it.
        from pyamplicol import ReductionUnavailable

        policy = _forced(
            VirtualRescueLevel("double", reduction="ninjars"),
            VirtualRescueLevel("double-double", reduction="tirust"),
        )
        with pytest.raises(ReductionUnavailable, match="double-double"):
            runtime.evaluate(points, mu_r2=scales, stability=policy)
        return
    found = {}
    for backend in ("tirust", "ninjars"):
        policy = _forced(
            VirtualRescueLevel("double", reduction="ninjars"),
            VirtualRescueLevel("double-double", reduction=backend),
        )
        found[backend] = runtime.evaluate(points, mu_r2=scales, stability=policy)
    for k, record in enumerate(found["tirust"].rescue):
        assert record.level == 1 and record.reduction.effective == "tirust"
        e = max(found[b].stability[k].absolute[0][0] for b in found)
        deviation = abs(
            _finite(found["tirust"].values[k]) - _finite(found["ninjars"].values[k])
        )
        assert deviation <= ESTIMATE_MARGIN * e, (k, deviation, e)
    checked = runtime.evaluate(
        points,
        mu_r2=scales,
        arithmetic="double-double",
        stability_checks=VirtualStabilityChecks(1, True, cross_backend=True),
    )
    assert all(CROSS in dict(e.checks) for e in checked.stability)
