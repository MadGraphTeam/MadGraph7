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
from pyamplicol.loops.v2.plan_export import DEFAULT_LANES
from pyamplicol.loops.v2.reduction import default_arbitrary_digits
from pyamplicol.loops.v2.rescue import rescue, rescue_policy
from pyamplicol.loops.v2.stability import (
    DEFAULT_RESCUE,
    RESCUE_INTERMEDIATE_DIGITS,
    Rotation,
    default_rescue,
)

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


# The mechanics run on a three-level ladder of the default kind (binary64,
# double-double, 57 digits); the default itself is checked below.
THREE = VirtualRescuePolicy(
    (
        VirtualRescueLevel("double"),
        VirtualRescueLevel("double-double"),
        VirtualRescueLevel("arbitrary", 57),
    )
)


@pytest.mark.loop_fast
def test_only_failed_or_unconverged_points_escalate_in_order():
    table = (
        (1e-8, 1e-3, FAIL, 1e-9, 1e-2),
        (None, 1e-12, FAIL, None, 1e-4),
        (None, None, 1e-20, None, 1e-3),
    )
    fake = _Fake(THREE, table)
    result = rescue(THREE, 5, fake)
    # Level 0 on the batch; every escalated level on its points in one call.
    assert fake.calls == [
        (0, (0, 1, 2, 3, 4)),
        (1, (1, 2, 4)),
        (2, (2, 4)),
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
def test_only_a_numerical_zero_is_not_escalated():
    """The stopping rule (review B3): a point is not escalated when its
    estimate grew, against the next lower precision, at least by the square
    root of the two levels' unit-roundoff ratio (binary64 -> double-double:
    2^25.5 = 4.7e7), a numerical zero's signature; it stays at that level
    with the target not met. A smaller growth (a ruined point's flat, noisy
    estimate above 1), a failed attempt, or a level of the same precision on
    the other backend never stops a point; the last level keeps every
    point."""

    table = (
        (1e-2, 1e-2, FAIL, 1e-2),
        (1e7, 1e-1, 1e-1, FAIL),
        (1e-9, 1e-9, 1e-9, 5e-2),
    )
    fake = _Fake(THREE, table)
    result = rescue(THREE, 4, fake)
    assert fake.calls == [(0, (0, 1, 2, 3)), (1, (0, 1, 2, 3)), (2, (1, 2, 3))]
    assert [record.level for record in result.rescue] == [1, 2, 2, 2]
    assert [record.met for record in result.rescue] == [False, True, True, False]
    stopped = result.diagnostics[0]
    assert stopped.status == "rescued" and not stopped.accuracy_established
    assert "not escalated: its estimate grew from the double level's by 1.0e+09 " in (
        stopped.detail
    )
    assert "(at least 4.7e+07, the square root" in stopped.detail
    assert result.values[0][0].finite == _value(1, 0, "double-double")
    # Point 1 grew by 10 only (no zero): escalated, and served.
    assert "not escalated" not in result.diagnostics[1].detail
    # Point 3 failed at level 1 and reaches the last level, which keeps it.
    assert "not met at the final level" in result.diagnostics[3].detail
    # HC05 #40 point 29 on the review's ladder (50, 57, 500 digits): 13.96
    # then 48.8, a growth of 3.5 against the sqrt(10^7) = 3.2e3 needed:
    # escalated to 500 digits, which serves it.
    digits = VirtualRescuePolicy(
        tuple(VirtualRescueLevel("arbitrary", d) for d in (50, 57, 500))
    )
    fake = _Fake(digits, ((13.96,), (48.8,), (1e-9,)))
    (record,) = rescue(digits, 1, fake).rescue
    assert record.level == 2 and record.met
    # ... while a growth like g g > z's (a numerical zero) stops at 57.
    fake = _Fake(digits, ((13.96,), (13.96 * 3.2e3,), (1e-9,)))
    (record,) = rescue(digits, 1, fake).rescue
    assert record.level == 1 and not record.met
    # Two levels at the same precision (the other backend) never stop a point.
    backends = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", reduction="tirust"),
            VirtualRescueLevel("double", reduction="ninjars"),
            VirtualRescueLevel("double-double"),
        )
    )
    fake = _Fake(backends, ((1e-2,), (1e9,), (1e-9,)))
    (record,) = rescue(backends, 1, fake).rescue
    assert record.level == 2 and record.met
    # Review F1: two backends at binary64 (TIRust nearly meets the target,
    # NinjaRS is ruined), then double-double: the growth is measured from
    # the larger binary64 estimate (1e3 -> 1e4: 10, not 1e8 from 1e-4), so
    # the point escalates and 57 digits serves it.
    stack = VirtualRescuePolicy((*backends.levels, VirtualRescueLevel("arbitrary", 57)))
    fake = _Fake(stack, ((1e-4,), (1e3,), (1e4,), (1e-9,)))
    (record,) = rescue(stack, 1, fake).rescue
    assert record.level == 3 and record.met


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
    # The default ladder (USER 2026-11-18): binary64, double-double, 57 digits
    # (the intermediate level) and the default arbitrary-precision digits
    # (500, the native policy's one definition) last.
    default = VirtualRescuePolicy()
    assert default_rescue() == DEFAULT_RESCUE
    assert [(level.arithmetic, level.precision) for level in default.levels] == [
        ("double", None),
        ("double-double", None),
        ("arbitrary", RESCUE_INTERMEDIATE_DIGITS),
        ("arbitrary", default_arbitrary_digits()),
    ]
    assert (RESCUE_INTERMEDIATE_DIGITS, default_arbitrary_digits()) == (57, 500)
    assert [level.precision_bits for level in default.levels] == [53, 106, 190, 1661]
    for level in default.levels:
        assert level.target == 1e-5 and level.metric == "relative"
        assert level.rotations == (parse_rotation("euler:0.1,0.2,0.3"),)
        assert level.checks == VirtualStabilityChecks(
            level.rotations, True, rescaled=True
        )
    assert THREE.levels == default.levels[:3]
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
        # (Arbitrary precision without digits takes the default, below.)
        {"arithmetic": "arbitrary", "precision": 0},
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
# The binary64 call whose non-finite lane the lane rescue served (binary64
# coefficients, double-double reduction) against MPFR(32).
LIGHT_BOX_LANE_RESCUE = 1e-12


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
    # The references at the level's target, which drives NinjaRS's
    # cancellation rescue too (its lane target is the level's target / 64).
    plain = runtime.evaluate(
        points,
        mu_r2=scales,
        poles=poles,
        stability_checks=CHECKS,
        reduction_target=target,
    )
    estimates = [item.relative for item in plain.stability]
    direct = runtime.evaluate(
        points,
        mu_r2=scales,
        poles=poles,
        arithmetic="double-double",
        stability_checks=CHECKS,
        reduction_target=target,
    )
    escalated = [index for index, e in enumerate(estimates) if e > target]
    # At the level's target the rescue may lift every point below it: the
    # policy escalates exactly the points whose estimate there exceeds it.
    assert len(escalated) < len(points)
    for index in range(len(points)):
        reference = direct if index in escalated else plain
        assert _lane(result, index) == _lane(reference, index)
        assert result.rescue[index].level == (1 if index in escalated else 0)
    assert (result.pole_parts is None) == (poles != "uv")
    assert result.poles == plain.poles


@pytest.mark.loop_physics
def test_a_light_top_lane_is_rescued_in_binary64_and_the_point_without_it(artifact):
    """The light box's binary64 lane is not finite (NonFiniteResult): the lane
    rescue reduces it again in double-double inside the binary64 call (the
    point is served in binary64 with its record); without the lane rescue
    the point fails in binary64 and the rescue stack serves it in
    double-double."""

    runtime = _load(artifact, "gg", LIGHT)
    reference = runtime.evaluate(
        [LIGHT_POINT], mu_r2=LIGHT_SCALE, arithmetic="arbitrary", precision=32
    )
    ((exact,),) = reference.values[0:1]
    scale = max(abs(x) for x in exact)

    def error(result):
        ((mine,),) = result.values[0:1]
        return (
            max(abs(Decimal(str(a)) - b) for a, b in zip(mine, exact, strict=True))
            / scale
        )

    plain = runtime.evaluate([LIGHT_POINT], mu_r2=LIGHT_SCALE, on_failure="return")
    lane = plain.diagnostics[0]
    assert plain.valid[0] and lane.arithmetic == "double", lane.detail
    assert lane.lane_rescue is not None and lane.lane_rescue.carrier == "double-double"
    assert error(plain) < LIGHT_BOX_LANE_RESCUE, error(plain)
    plain = runtime.evaluate(
        [LIGHT_POINT], mu_r2=LIGHT_SCALE, on_failure="return", lane_rescue=False
    )
    assert not plain.valid[0] and "NonFiniteResult" in plain.diagnostics[0].detail
    result = runtime.evaluate(
        [LIGHT_POINT],
        mu_r2=LIGHT_SCALE,
        stability=VirtualRescuePolicy(),
        lane_rescue=False,
    )
    (record,) = result.rescue
    assert record.level == 1 and record.met and record.arithmetic == "double-double"
    assert "NonFiniteResult" in record.attempts[0].failure
    assert result.diagnostics[0].status == "rescued"
    assert error(result) < LIGHT_BOX_DOUBLE_DOUBLE, error(result)


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
def test_escalated_subsets_run_together_on_one_double_double_executor(artifact):
    points, scales = _points("gg_gg", 3)
    runtime = _load(artifact, "gg")
    native = runtime._backend._executor._native
    alone = [
        runtime.evaluate([point], mu_r2=scales[0], stability=FORCED) for point in points
    ]
    for size in (1, 3, 5):
        batch = [points[k % 3] for k in range(size)]
        result = runtime.evaluate(batch, mu_r2=scales[0], stability=FORCED)
        assert {record.level for record in result.rescue} == {1}
        # Each point's value and estimate are those of a call of its own.
        for k in range(size):
            assert result.values[k] == alone[k % 3].values[0]
            assert result.stability[k] == alone[k % 3].stability[0]
    # One double-double executor (the executor fills its chunks).
    assert [key for key in native if key[0] == "double-double"] == [
        ("double-double", None, DEFAULT_LANES)
    ]


@pytest.mark.loop_fast
def test_arbitrary_precision_without_digits_takes_the_default_digits():
    """USER 2026-11-18 ("500 digits"): an arbitrary-precision carrier named
    without digits takes the default arbitrary-precision digits (the native
    policy's one definition); a number given is kept."""

    from pyamplicol.api.virtual_results import VirtualArithmeticSpec
    from pyamplicol.config import EvaluationConfig

    default = default_arbitrary_digits()
    assert default == 500
    assert VirtualArithmeticSpec("arbitrary").precision == default
    assert VirtualArithmeticSpec("arbitrary", 40).precision == 40
    assert VirtualArithmeticSpec("double").precision is None
    level = VirtualRescueLevel("arbitrary")
    assert level.precision == default and level.precision_bits == 1661
    assert VirtualRescueLevel("arbitrary", 80).precision == 80
    # The configuration leaves it unset (the carrier's default).
    assert EvaluationConfig().precision is None
    assert EvaluationConfig(precision=70).precision == 70


@pytest.mark.loop_physics
def test_an_arbitrary_call_without_digits_runs_at_the_default_digits(artifact):
    """``evaluate(..., arithmetic="arbitrary")`` and ``Runtime.load(...,
    arithmetic="arbitrary")`` without digits resolve to the default
    arbitrary-precision digits (500); given digits are kept (evaluated at
    40 digits); an unset precision without arithmetic stays binary64. (No
    500-digit evaluation here: it shifts later double-double guard estimates
    of the same process, pre-existing; FINDINGS 2026-11-18 "ladder".)"""

    points, scales = _points("gg_gg", 1)
    runtime = _load(artifact, "gg")
    assert runtime._call_precision(None, "arbitrary") == default_arbitrary_digits()
    assert runtime._call_precision(None, None) == 16
    assert runtime._call_precision(40, "arbitrary") == 40
    given = runtime.evaluate(points, mu_r2=scales, arithmetic="arbitrary", precision=40)
    plain = runtime.evaluate(points, mu_r2=scales)
    assert given.diagnostics[0].precision_bits == 133
    assert plain.diagnostics[0].precision_bits == 53
    loaded = _load(artifact, "gg", arithmetic="arbitrary")
    assert loaded._backend.default_precision == default_arbitrary_digits()
    assert loaded._call_precision(None, None) == default_arbitrary_digits()
    # A load naming arbitrary precision keeps it for a call naming digits
    # only, 16 included (the summed evaluation on the artifact runtime, the
    # resolved one on its observables runtime), as before the default.
    for call in (loaded.evaluate, loaded.evaluate_resolved):
        found = call(points, mu_r2=scales, precision=16).diagnostics[0]
        assert (found.arithmetic, found.precision_bits) == ("arbitrary", 54), call
    kept = _load(artifact, "gg", arithmetic="arbitrary", precision=40)
    assert kept._call_precision(None, "arbitrary") == 40
    assert kept.evaluate(points, mu_r2=scales).diagnostics[0].precision_bits == 133


# ------------------------------------- lanes served without accuracy evidence

BELOW = (
    "a reduction lane vanishes within the resolution of its own terms"
    " (no symmetry proof)"
)


class _Uncertified(_Fake):
    """``_Fake`` whose diagnostics at the (level, point) pairs of
    ``uncertified`` carry a lane served without accuracy evidence."""

    def __init__(self, policy, table, uncertified):
        super().__init__(policy, table)
        self.uncertified = set(uncertified)

    def __call__(self, indices, level):
        from dataclasses import replace

        result = super().__call__(indices, level)
        k = self.policy.levels.index(level)
        diagnostics = tuple(
            replace(item, uncertified=(BELOW,))
            if (k, point) in self.uncertified and item.valid
            else item
            for item, point in zip(result.diagnostics, indices, strict=True)
        )
        return replace(result, diagnostics=diagnostics)


@pytest.mark.loop_fast
def test_an_uncertified_level_is_never_met_and_escalates():
    """A level whose point carries an uncertified lane is not met even when
    its estimate passes: the policy escalates, and the point is met at the
    next, clean level; at a last level it stays not established with the
    reason, and nothing raises."""

    table = ((1e-12, 1e-12), (1e-20, None), (None, None))
    fake = _Uncertified(THREE, table, {(0, 0)})
    result = rescue(THREE, 2, fake)
    assert fake.calls[:2] == [(0, (0, 1)), (1, (0,))]
    assert [record.level for record in result.rescue] == [1, 0]
    assert [record.met for record in result.rescue] == [True, True]
    assert result.diagnostics[0].status == "rescued"
    assert result.diagnostics[0].accuracy_established
    assert result.diagnostics[0].uncertified == ()
    # A single-level policy: the uncertified point is kept, not established.
    one = VirtualRescuePolicy((VirtualRescueLevel("double"),))
    result = rescue(one, 1, _Uncertified(one, ((1e-12,),), {(0, 0)}))
    (diagnostic,) = result.diagnostics
    assert result.rescue[0].met is False
    assert diagnostic.has_value and not diagnostic.accuracy_established
    assert diagnostic.uncertified == (BELOW,)
    assert "accuracy not established: " + BELOW in diagnostic.detail


@pytest.mark.loop_fast
def test_the_selected_helicity_diagnostic_keeps_the_uncertified_reasons():
    """The diagnostic of the selected-helicity, resolved and correlated
    routes (``V2ObservablesRuntime._diagnostic``) keeps the native
    uncertified reasons as the summed route does, and a rescue policy with a
    passing rotation metric does not establish its accuracy."""

    from dataclasses import replace
    from types import SimpleNamespace

    from pyamplicol.api.virtual_results import UNCERTIFIED
    from pyamplicol.runtime.virtual import V2ObservablesRuntime

    owner = SimpleNamespace(_executor="native")
    diagnostic = V2ObservablesRuntime._diagnostic(
        owner, "double", None, True, lane_rescue=((UNCERTIFIED + BELOW, 1),)
    )
    assert diagnostic.status == "success" and diagnostic.uncertified == (BELOW,)
    assert not diagnostic.accuracy_established
    assert "accuracy not established: " + BELOW in diagnostic.detail
    one = VirtualRescuePolicy((VirtualRescueLevel("double"),))
    fake = _Fake(one, ((1e-12,),))

    def evaluate(indices, level):
        result = fake(indices, level)
        return replace(result, diagnostics=(diagnostic,))

    result = rescue(one, 1, evaluate)
    assert result.rescue[0].met is False
    assert not result.diagnostics[0].accuracy_established
    assert result.diagnostics[0].uncertified == (BELOW,)


@pytest.mark.loop_fast
def test_the_python_reference_executor_never_certifies_ninjars_levels():
    """The Python reference executor (``executor="python"``) does not carry
    the reduction's record of lanes served without accuracy evidence: its
    NinjaRS-reduced points are returned with the reason and a one-level
    NinjaRS policy whose estimate passes is not met; a TIRust point keeps
    its evidence."""

    from dataclasses import replace
    from types import SimpleNamespace

    from pyamplicol.api.virtual_results import VirtualReduction
    from pyamplicol.runtime.virtual import (
        PYTHON_REFERENCE_UNCERTIFIED,
        V2ObservablesRuntime,
    )

    owner = SimpleNamespace(_executor="python")
    ninjars = V2ObservablesRuntime._diagnostic(owner, "double", None, True)
    assert ninjars.uncertified == (PYTHON_REFERENCE_UNCERTIFIED,)
    assert "accuracy not established: " in ninjars.detail
    tirust = VirtualReduction("tirust", "tirust", "requested")
    other = V2ObservablesRuntime._diagnostic(
        owner, "double", None, True, reductions={"double": tirust}
    )
    assert other.uncertified == ()
    one = VirtualRescuePolicy((VirtualRescueLevel("double", target=1e-5),))
    fake = _Fake(one, ((3.3069954e-8,),))

    def evaluate(indices, level):
        return replace(fake(indices, level), diagnostics=(ninjars,))

    result = rescue(one, 1, evaluate)
    assert result.rescue[0].met is False
    assert not result.diagnostics[0].accuracy_established
    assert result.diagnostics[0].uncertified == (PYTHON_REFERENCE_UNCERTIFIED,)
