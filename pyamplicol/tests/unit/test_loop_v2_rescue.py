# SPDX-License-Identifier: 0BSD
"""The built-in stability rescue of the Python native evaluate (M-FINAL
MF-RESCUE; user ruling 2026-10-31 Q9; ``loops.v2.rescue``).

Pure unit tests on a fake level evaluator: only failed or unconverged points
reach the next level, in the caller's order; an accepted point is never
evaluated again; a final-level miss keeps its value with ``met=False``; a
point that fails at every level is a failed lane (raised only under
``on_failure="raise"``); policy validation; the exact ``euler:`` rotation.

Integration on public v2 artifacts (g g > g g and u u~ > d d~ g at full
colour, g g > g g at leading colour):

* (a) without a policy ``evaluate`` returns the native executor's binary64
  values bitwise and no rescue records;
* (b) a level-0 target of 1e-300 sends every point to double-double, whose
  lanes equal a direct double-double call with the same checks bitwise;
* (c) a target between two points' binary64 estimates escalates only the
  worse point; the other lanes equal the plain binary64 call bitwise, in
  every pole selection (with (e): ``poles="uv"`` pole parts merged per lane);
* (d) g g > g g with a light top (MT = 1e-6): binary64 fails (D0 not
  finite) and the default policy returns it rescued at double-double, within
  1e-17 of 32-digit arbitrary precision;
* (f) the leading-colour selection path;
* (g) one typed refusal per refused combination, and the load-time default;
* (h) escalated subsets of 1, 3 and 5 points build one double-double
  executor.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from pyamplicol.api.errors import CompatibilityError, EvaluationError
from pyamplicol.api.virtual_results import (
    VirtualEvaluation,
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
    VirtualRescueLevel,
    VirtualRescuePolicy,
    VirtualStabilityChecks,
    VirtualStabilityEstimate,
    parse_rotation,
)
from pyamplicol.config import R2Strategy
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
from pyamplicol.loops.strategy import R2Selection
from pyamplicol.loops.v2.rescue import rescue, rescue_policy
from pyamplicol.loops.v2.stability import DEFAULT_RESCUE, Rotation

from .test_loop_v2_artifact import GG_PARAMETERS, _card, _fixture, _generate

# ------------------------------------------------------------- fake evaluator

QCD2 = (("QCD", 2),)
BORN_ORDERS = ((("QCD", 4),),)
SECTORS = (VirtualOrderSector((("QCD", 6),), (BornLoopPair(QCD2, (("QCD", 4),)),)),)
R2 = R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "model-supplied R2")
FAIL = "fail"


def _value(level: int, point: int, arithmetic: str):
    x = 100 * level + point + 0.5
    return x if arithmetic == "double" else Decimal(str(x))


class _Fake:
    """A level evaluator: ``table[level][point]`` is the point's estimate
    at that level, ``FAIL`` or ``None`` (no estimate); records every call."""

    def __init__(self, policy: VirtualRescuePolicy, table):
        self.policy, self.table, self.calls = policy, table, []

    def __call__(self, indices, level) -> VirtualEvaluation:
        k = self.policy.levels.index(level)
        self.calls.append((k, indices))
        bits = level.precision_bits
        born, values, diagnostics, estimates = [], [], [], []
        for lane, point in enumerate(indices):
            outcome = self.table[k][point]
            if outcome == FAIL:
                reason = f"failure {k}:{point}"
                born.append(None)
                values.append(None)
                estimates.append(None)
                diagnostics.append(
                    VirtualLaneDiagnostic(
                        "failed",
                        level.arithmetic,
                        bits,
                        False,
                        reason,
                        failures=(
                            VirtualFailureDiagnostic(
                                lane, None, None, None, "v2-evaluation", reason
                            ),
                        ),
                    )
                )
                continue
            x = _value(k, point, level.arithmetic)
            born.append((x,))
            values.append(((x, x, x),))
            diagnostics.append(
                VirtualLaneDiagnostic("success", level.arithmetic, bits, False)
            )
            estimates.append(
                None
                if outcome is None
                else VirtualStabilityEstimate(
                    outcome, outcome, outcome, outcome, (("rotation-0", outcome),)
                )
            )
        return VirtualEvaluation(
            tuple(born),
            tuple(values),
            BORN_ORDERS,
            SECTORS,
            tuple(diagnostics),
            R2,
            stability=tuple(estimates) if level.checks.enabled else None,
        )


THREE = VirtualRescuePolicy()


@pytest.mark.loop_fast
def test_only_failed_or_unconverged_points_escalate_in_order():
    table = (
        (1e-8, 1e-3, FAIL, 1e-9, 1e-2),
        (None, 1e-12, FAIL, None, 1e-4),
        (None, None, 1e-20, None, 1e-3),
    )
    fake = _Fake(THREE, table)
    result = rescue(THREE, 5, fake)
    # Level 0 on the batch; escalated levels one point per call, in order.
    assert fake.calls == [
        (0, (0, 1, 2, 3, 4)),
        (1, (1,)),
        (1, (2,)),
        (1, (4,)),
        (2, (2,)),
        (2, (4,)),
    ]
    # Accepted points are never evaluated again.
    assert all(0 not in batch and 3 not in batch for k, batch in fake.calls if k)
    levels = [record.level for record in result.rescue]
    assert levels == [0, 1, 2, 0, 2]
    assert [record.met for record in result.rescue] == [True, True, True, True, False]
    assert [len(record.attempts) for record in result.rescue] == [1, 2, 3, 1, 3]
    assert [item.status for item in result.diagnostics] == [
        "success",
        "rescued",
        "rescued",
        "success",
        "rescued",
    ]
    assert [item.accuracy_established for item in result.diagnostics] == [
        True,
        True,
        True,
        True,
        False,
    ]
    # Each lane is the accepted level's evaluation, in the caller's order.
    for point, level in enumerate(levels):
        arithmetic = THREE.levels[level].arithmetic
        assert result.values[point][0].finite == _value(level, point, arithmetic)
        assert result.diagnostics[point].arithmetic == arithmetic
        assert result.stability[point].relative == table[level][point]
    assert result.rescue[2].attempts[1].failure == "failure 1:2"
    assert result.rescue[1].attempts[0].estimate == 1e-3
    assert result.rescue_counts == {
        0: 2,
        1: 1,
        2: 1,
        "not met": 1,
        "best-effort": 0,
        "failed": 0,
    }
    # A final-level miss keeps the value, with the target marked as missed.
    assert "not met at the final level" in result.diagnostics[4].detail
    assert result.values[4][0].finite == Decimal("204.5")


@pytest.mark.loop_fast
def test_a_point_failing_at_every_level_is_failed_and_raises_only_on_request():
    table = ((1e-9, FAIL), (None, FAIL), (None, FAIL))
    fake = _Fake(THREE, table)
    result = rescue(THREE, 2, fake, on_failure="return")
    assert result.valid == (True, False)
    assert result.values[1] is None and result.stability[1] is None
    failed = result.diagnostics[1]
    assert failed.status == "failed" and failed.arithmetic == "arbitrary"
    assert "level 0 (double): failure 0:1" in failed.detail
    assert "level 2 (arbitrary): failure 2:1" in failed.detail
    # Failure diagnostics name the caller's point, not the subset's lane.
    assert [item.point_id for item in failed.failures] == [1]
    assert result.rescue[1].failed and not result.rescue[1].met
    assert result.rescue_counts == {
        0: 1,
        1: 0,
        2: 0,
        "not met": 0,
        "best-effort": 0,
        "failed": 1,
    }
    with pytest.raises(EvaluationError, match="point 1 at every rescue level"):
        rescue(THREE, 2, _Fake(THREE, table))


@pytest.mark.loop_fast
def test_a_check_free_last_level_keeps_every_point_without_an_estimate():
    policy = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", rotations=1, reduction_check="none"),
            VirtualRescueLevel("arbitrary", 40, rotations=0, reduction_check="none"),
        )
    )
    fake = _Fake(policy, ((1e-9, 1.0), (None, None)))
    result = rescue(policy, 2, fake)
    assert result.stability[0].relative == 1e-9 and result.stability[1] is None
    assert result.rescue[1].level == 1 and not result.rescue[1].met
    assert result.diagnostics[1].status == "rescued"
    assert "no stability check" in result.diagnostics[1].detail
    # An empty batch runs level 0 once and returns no lane.
    empty = rescue(THREE, 0, _Fake(THREE, ((), (), ())))
    assert empty.values == () and empty.rescue == ()


@pytest.mark.loop_fast
def test_policies_are_validated():
    assert [(level.arithmetic, level.precision) for level in THREE.levels] == list(
        DEFAULT_RESCUE
    )
    for level in THREE.levels:
        assert level.target == 1e-5 and level.metric == "relative"
        assert level.rotations == (parse_rotation("euler:0.1,0.2,0.3"),)
        assert level.checks == VirtualStabilityChecks(
            level.rotations, True, rescaled=True
        )
    assert [level.precision_bits for level in THREE.levels] == [53, 106, 190]
    assert len(VirtualRescueLevel("double", rotations=3).rotations) == 3
    VirtualRescuePolicy((VirtualRescueLevel("double-double"),))
    VirtualRescuePolicy(
        (VirtualRescueLevel("double-double"), VirtualRescueLevel("arbitrary", 32))
    )
    for levels in (
        (),
        (VirtualRescueLevel("double"), VirtualRescueLevel("double")),
        # 31 digits are 103 bits, fewer than double-double's 106.
        (VirtualRescueLevel("double-double"), VirtualRescueLevel("arbitrary", 31)),
        (
            VirtualRescueLevel("double", rotations=0, reduction_check="none"),
            VirtualRescueLevel("double-double"),
        ),
    ):
        with pytest.raises(ValueError):
            VirtualRescuePolicy(levels)
    with pytest.raises(TypeError):
        VirtualRescuePolicy(VirtualRescueLevel("double"))
    for kwargs in (
        {"arithmetic": "arbitrary"},
        {"arithmetic": "double", "precision": 20},
        {"arithmetic": "quad"},
        {"arithmetic": "double", "target": 0},
        {"arithmetic": "double", "target": -1e-5},
        {"arithmetic": "double", "target": math.inf},
        {"arithmetic": "double", "target": math.nan},
        {"arithmetic": "double", "reduction_check": "forward"},
        {"arithmetic": "double", "metric": "absolute"},
        {"arithmetic": "double", "rotations": -1},
    ):
        with pytest.raises(ValueError):
            VirtualRescueLevel(**kwargs)
    with pytest.raises(TypeError):
        VirtualRescueLevel("double", target=True)
    # The policy owns the carriers: refused with explicit ones, the load-time
    # default applies only to calls that name none.
    for kwargs in (
        {"arithmetic": "double"},
        {"precision": 40, "default_precision": 16},
        {"stability_checks": VirtualStabilityChecks(1)},
    ):
        with pytest.raises(CompatibilityError):
            rescue_policy(THREE, **kwargs)
    assert rescue_policy(THREE, precision=16, default_precision=16) is THREE
    assert rescue_policy(None, default=THREE) is THREE
    assert rescue_policy(None, default=THREE, arithmetic="double") is None
    with pytest.raises(TypeError):
        rescue_policy("default")


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    "angles", [(0.1, 0.2, 0.3), (-2.5, 1.3, 3.0), (math.pi / 2, 0.0, 0.0)]
)
def test_euler_rotation_is_exact_and_matches_gammaloop(angles):
    """``euler:a,b,g`` is Rz(g) Ry(b) Rx(a), gammaloop's matrix
    (``ThreeMomentum::rotate_mut``), exactly orthogonal (``Rotation``
    checks it) and within 1e-9 of the floating-point matrix."""

    alpha, beta, gamma = angles
    rotation = parse_rotation("euler:{},{},{}".format(*angles))
    assert rotation == Rotation.from_euler(*angles)
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    gammaloop = (
        (cg * cb, -ca * sg + sa * sb * cg, sa * sg + ca * sb * cg),
        (sg * cb, ca * cg + sa * sb * sg, -sa * cg + ca * sb * sg),
        (-sb, cb * sa, ca * cb),
    )
    for row, expected in zip(rotation.matrix, gammaloop, strict=True):
        for x, y in zip(row, expected, strict=True):
            assert abs(float(x) - y) <= 1e-9
    with pytest.raises(ValueError):
        parse_rotation("euler:0.1,0.2")


# ------------------------------------------------------------ public artifacts

ENTRIES = (("gg", "g g > g g"), ("uux_ddxg", "u u~ > d d~ g"))
CHECKS = VirtualStabilityChecks(("euler:0.1,0.2,0.3",), True, rescaled=True)
# g g > g g with a light top (``test_loop_v2_onshell_threshold``).
LIGHT = {"aS": "0.118", "MT": "0.000001"}
LIGHT_POINT = [
    ["500.0", "0.0", "0.0", "500.0"],
    ["500.0", "0.0", "0.0", "-500.0"],
    ["500.0", "300.0", "240.0", "320.0"],
    ["500.0", "-300.0", "-240.0", "-320.0"],
]
LIGHT_SCALE = "250000"
# DoubleFloat on the light box against MPFR(32) (measured 2.05e-18).
LIGHT_BOX_DOUBLE_DOUBLE = 1e-17


@pytest.fixture(scope="module")
def artifact(loop_sm_compiled_model, loop_native, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v2-rescue")
    return _generate(
        _card(loop_sm_compiled_model, tmp / "artifact", ENTRIES), tmp, ENTRIES
    )


def _load(artifact, process, parameters=GG_PARAMETERS, **options):
    from pyamplicol import Runtime

    return Runtime.load(
        artifact, process=process, model_parameters=parameters, **options
    )


def _points(name: str, count: int = 3):
    fixture = _fixture(name)
    return fixture["points"][:count], fixture["scales_squared"][:count]


def _bits(value):
    """Every number of a (nested) result exactly: binary64 as hex."""

    if isinstance(value, float):
        return value.hex()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_bits(item) for item in value)
    if value is None:
        return None
    if hasattr(value, "origin"):  # VirtualPoleParts
        return tuple(
            tuple((name, _bits(part)) for name, part in value.origin(o).items())
            for o in ("uv", "ir")
        )
    return value


def _lane(result: VirtualEvaluation, index: int):
    return (
        _bits(result.born_values[index]),
        _bits(result.values[index]),
        None if result.stability is None else result.stability[index],
        None if result.pole_parts is None else _bits(result.pole_parts[index]),
    )


def _to_dd(level: VirtualRescueLevel) -> VirtualRescuePolicy:
    """``level`` in binary64 with an unreachable target, then double-double
    with the same checks."""

    return VirtualRescuePolicy(
        (
            level,
            VirtualRescueLevel(
                "double-double",
                rotations=level.rotations,
                reduction_check=level.reduction_check,
            ),
        )
    )


FORCED = _to_dd(VirtualRescueLevel("double", target=1e-300))


@pytest.mark.loop_physics
def test_without_a_policy_evaluate_is_the_native_binary64_path(artifact):
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.runtime.virtual import _inputs

    points, scales = _points("uux_ddxg")
    runtime = _load(artifact, "uux_ddxg")
    result = runtime.evaluate(points, mu_r2=scales)
    assert result.rescue is None and result.rescue_counts is None
    backend = runtime._backend
    native = backend._executor.evaluate(
        *_inputs(points, scales, 5),
        parameters=dict(backend._parameters),
        precision=Precision.double(),
        poles="all",
        stability=None,
    )
    for lane, point in zip(result.values, native, strict=True):
        ((laurent,),) = (lane,)
        expected = next(iter(point.components["virtual"].values())).coefficients
        assert _bits(tuple(laurent)) == _bits(tuple(expected))
    assert result.diagnostics[0].status == "success"


@pytest.mark.loop_physics
def test_an_unreachable_binary64_target_sends_every_point_to_double_double(artifact):
    points, scales = _points("gg_gg")
    runtime = _load(artifact, "gg")
    result = runtime.evaluate(points, mu_r2=scales, stability=FORCED)
    direct = runtime.evaluate(
        points, mu_r2=scales, arithmetic="double-double", stability_checks=CHECKS
    )
    assert result.rescue_counts == {
        0: 0,
        1: 3,
        "not met": 0,
        "best-effort": 0,
        "failed": 0,
    }
    for index in range(len(points)):
        assert _lane(result, index) == _lane(direct, index)
        record = result.rescue[index]
        assert record.level == 1 and record.met
        assert record.attempts[0].arithmetic == "double"
        assert record.attempts[0].estimate > 1e-300
        assert result.diagnostics[index].status == "rescued"
        assert result.diagnostics[index].accuracy_established
        assert result.diagnostics[index].precision_bits == 106


@pytest.mark.loop_physics
@pytest.mark.parametrize("poles", ["all", "none", "uv"])
def test_a_mixed_batch_escalates_only_the_worse_points(artifact, poles):
    points, scales = _points("uux_ddxg")
    runtime = _load(artifact, "uux_ddxg")
    plain = runtime.evaluate(points, mu_r2=scales, poles=poles, stability_checks=CHECKS)
    estimates = [item.relative for item in plain.stability]
    low, high = min(estimates), max(estimates)
    assert low < high, estimates
    target = math.sqrt(low * high)
    policy = _to_dd(VirtualRescueLevel("double", target=target))
    result = runtime.evaluate(points, mu_r2=scales, poles=poles, stability=policy)
    direct = runtime.evaluate(
        points,
        mu_r2=scales,
        poles=poles,
        arithmetic="double-double",
        stability_checks=CHECKS,
    )
    escalated = [index for index, e in enumerate(estimates) if e > target]
    assert 0 < len(escalated) < len(points)
    for index in range(len(points)):
        reference = direct if index in escalated else plain
        assert _lane(result, index) == _lane(reference, index)
        assert result.rescue[index].level == (1 if index in escalated else 0)
    assert (result.pole_parts is None) == (poles != "uv")
    assert result.poles == plain.poles


@pytest.mark.loop_physics
def test_a_light_top_fails_in_binary64_and_is_rescued_in_double_double(artifact):
    runtime = _load(artifact, "gg", LIGHT)
    plain = runtime.evaluate([LIGHT_POINT], mu_r2=LIGHT_SCALE, on_failure="return")
    assert not plain.valid[0] and "NonFiniteResult" in plain.diagnostics[0].detail
    result = runtime.evaluate(
        [LIGHT_POINT], mu_r2=LIGHT_SCALE, stability=VirtualRescuePolicy()
    )
    (record,) = result.rescue
    assert record.level == 1 and record.met and record.arithmetic == "double-double"
    assert "NonFiniteResult" in record.attempts[0].failure
    assert result.diagnostics[0].status == "rescued"
    reference = runtime.evaluate(
        [LIGHT_POINT], mu_r2=LIGHT_SCALE, arithmetic="arbitrary", precision=32
    )
    ((mine,),), ((exact,),) = result.values[0:1], reference.values[0:1]
    scale = max(abs(x) for x in exact)
    error = max(abs(a - b) for a, b in zip(mine, exact, strict=True)) / scale
    assert error < LIGHT_BOX_DOUBLE_DOUBLE, error


@pytest.mark.loop_physics
def test_the_leading_colour_selection_path_is_rescued(
    loop_sm_colour_model, loop_native, tmp_path
):
    entries = (("gg_gg", "g g > g g"),)
    output = _generate(
        _card(
            loop_sm_colour_model,
            tmp_path / "artifact",
            entries,
            accuracy="lc",
            declared=False,
        ),
        tmp_path,
        entries,
    )
    points, scales = _points("gg_gg", 1)
    runtime = _load(output, "gg_gg")
    result = runtime.evaluate(points, mu_r2=scales, color_flows=["1"], stability=FORCED)
    direct = runtime.evaluate(
        points,
        mu_r2=scales,
        color_flows=["1"],
        arithmetic="double-double",
        stability_checks=CHECKS,
    )
    assert result.color_accuracy == "lc" and result.rescue[0].level == 1
    assert _lane(result, 0) == _lane(direct, 0)
    assert result.diagnostics[0].status == "rescued"


@pytest.mark.loop_physics
def test_refused_combinations_and_the_load_time_default(artifact):
    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest

    points, scales = _points("gg_gg", 1)
    runtime = _load(artifact, "gg")
    policy = VirtualRescuePolicy()
    for call in (
        lambda: runtime.evaluate(
            points, mu_r2=scales, stability=policy, stability_checks=CHECKS
        ),
        lambda: runtime.evaluate(
            points, mu_r2=scales, stability=policy, arithmetic="double-double"
        ),
        lambda: runtime.evaluate(points, mu_r2=scales, stability=policy, precision=40),
        lambda: runtime.evaluate_resolved(points, mu_r2=scales, stability=policy),
        lambda: runtime.evaluate_correlated(points, mu_r2=scales, stability=policy),
        lambda: runtime.evaluate_correlated_many(
            points,
            {"born": CorrelatedRequest("born", {})},
            mu_r2=scales,
            stability=policy,
        ),
        lambda: Runtime.load(artifact, process="gg", stability=policy, precision=40),
        lambda: Runtime.load(
            artifact, process="gg", stability=policy, arithmetic="double-double"
        ),
    ):
        with pytest.raises(CompatibilityError):
            call()
    with pytest.raises(TypeError):
        runtime.evaluate(points, mu_r2=scales, stability="default")
    # The load-time default runs for calls that name no carrier or checks.
    loaded = _load(artifact, "gg", stability=FORCED)
    assert loaded.inspect()["default_rescue_levels"] == 2
    assert loaded.evaluate(points, mu_r2=scales).rescue[0].level == 1
    single = loaded.evaluate(points, mu_r2=scales, arithmetic="double")
    assert single.rescue is None and single.stability is None


@pytest.mark.loop_physics
def test_escalated_subsets_share_one_double_double_executor(artifact):
    points, scales = _points("gg_gg", 3)
    runtime = _load(artifact, "gg")
    native = runtime._backend._executor._native
    for size in (1, 3, 5):
        batch = [points[k % 3] for k in range(size)]
        result = runtime.evaluate(batch, mu_r2=scales[0], stability=FORCED)
        assert {record.level for record in result.rescue} == {1}
    assert [key for key in native if key[0] == "double-double"] == [
        ("double-double", None, 1)
    ]
