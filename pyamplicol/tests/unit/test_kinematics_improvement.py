# SPDX-License-Identifier: 0BSD
"""The kinematics improvement of the input points (m-lit lit-psimprove; USER
2026-11-10 ~00:56 / ~01:56; DECISIONS 2026-11-10 01:56 and 02:29).

Unit tests of the Python surface of the one Rust implementation
(``rusticol_core::kinematics_improvement`` through ``pyamplicol.kinematics``):
the native policy (MadLoop's default variant, the binary64 switch off, the
carrier key), the validator's one refusal text (also through the CLI and
the run-card keys), MadLoop 1 / 2 against MadLoop's own ``improve_ps.inc``
built standalone in quadruple precision
(``tests/fixtures/loop/kinematics_improvement_madloop.json``), the
projection and the records.

Route tests on public v2 artifacts (g g > g g, Born-interfered, full colour;
g g > h h, loop-induced): with the improvement on, each double-double and
arbitrary-precision route (summed, also 'arbitrary' at 16 digits by the
carrier key; resolved; colour-correlated; the Born-interfered and the
loop-induced helicity selections; a rescue stack binary64 -> double-double
-> arbitrary precision, the loop-induced one with a NinjaRS level and the
per-point NinjaRS fallback) records the variant that produced its point and
returns the values of the explicitly pre-improved exact point evaluated with
the improvement off, bitwise; the stability rotations act on the improved
point (the summed and resolved routes and a rescue level's checks);
binary64 is untouched unless switched on; ``"off"`` records ``"off"``; an
exactly physical point is passed unchanged; a frame whose beams are not
along z takes MadLoop's fallback (PSMC) under the default and THE FINISH
(the projection continues from PSMC's inexact point, USER 2026-11-10
~19:20, DECISIONS 19:20 (F1)): exact, as with ``"projection"``.
The loop SDK (C ABI) in double-double and at 32 digits gives the Python
API's records and values; the tree runtime's higher-precision modes, its
correlated default and the binary64 switch through the native tree runtime.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol import kinematics

ROOT = Path(__file__).resolve().parents[2]
MADLOOP = ROOT / "tests" / "fixtures" / "loop" / "kinematics_improvement_madloop.json"

# -------------------------------------------------------------- the policy


@pytest.mark.loop_fast
def test_the_policy_is_madloops_default_and_the_carrier_key(loop_native):
    """DECISIONS 2026-11-10 01:56 (1), (2) and 02:29 (G3) (a): MadLoop's
    default variant above binary64, off in binary64, keyed on the carrier
    kind (an arbitrary-precision call at 16 digits is improved)."""

    policy = kinematics.policy()
    assert policy["default"] == "z-shift"
    assert policy["binary64_default"] is False
    assert policy["key"] == "carrier"
    assert list(policy["settings"]) == ["z-shift", "rescaling", "projection", "off"]
    settings = kinematics.resolve(None, None)
    assert settings.variant("double") is None
    assert settings.variant("double-double") == "z-shift"
    assert settings.variant("arbitrary") == "z-shift"
    switched = kinematics.resolve("projection", True)
    assert switched.variant("double") == "projection"
    assert kinematics.resolve("off", True).variant("arbitrary") is None
    with pytest.raises(ValueError) as refused:
        kinematics.setting_value("orig")
    assert str(refused.value) == policy["setting_refusal"]
    with pytest.raises(ValueError) as refused:
        kinematics.binary64_value(1)
    assert str(refused.value) == policy["binary64_refusal"]


@pytest.mark.loop_fast
def test_the_cli_and_run_card_keys_resolve_through_the_native_validator(loop_native):
    """``evaluate``, ``profile`` and ``check`` take the two settings
    (``--kinematics-improvement``, ``--[no-]kinematics-improvement-binary64``)
    into the run card's ``evaluation`` keys; an unknown variant is refused
    when the configuration resolves, with the native policy's one refusal
    text (no argparse ``choices`` restating the values). Here, not in
    test_cli_parser.py: resolving validates through the native extension,
    which the source-contracts job does not build."""

    from pyamplicol.cli import parse_cli
    from pyamplicol.config import ConfigurationError

    for command, value, switch, expected in (
        ("evaluate", "projection", "--kinematics-improvement-binary64", True),
        ("profile", "off", "--no-kinematics-improvement-binary64", False),
        ("check", "rescaling", "--kinematics-improvement-binary64", True),
    ):
        effective = (
            parse_cli((command, "--kinematics-improvement", value, switch))
            .resolve()
            .effective
        )
        assert effective.evaluation.kinematics_improvement == value, command
        assert effective.evaluation.kinematics_improvement_binary64 is expected
    default = parse_cli(("evaluate",)).resolve().effective
    assert default.evaluation.kinematics_improvement is None
    assert default.evaluation.kinematics_improvement_binary64 is None
    with pytest.raises(ConfigurationError) as refused:
        parse_cli(("evaluate", "--kinematics-improvement", "orig")).resolve()
    assert kinematics.policy()["setting_refusal"] in str(refused.value)


# -------------------------------------------------- MadLoop's improve_ps.inc


def _fixture() -> dict:
    return json.loads(MADLOOP.read_text())


@pytest.mark.loop_fast
def test_madloop_variants_match_madloops_quadruple_build(loop_native):
    """MadLoop 1 (PSMC) and 2 (ORIG), each alone (no fallback), through the
    Python binding at 40 digits against MadLoop 3.7.0's ``improve_ps.inc``
    compiled standalone in quadruple precision on the same binary64 inputs
    (the fixture's provenance): equal component-wise to MadLoop's own
    quadruple rounding (relative to the point's scale) wherever MadLoop
    improves the point, and the same refusal code where it refuses. The one
    tolerance is MadLoop's quadruple precision (2^-113 per operation, a few
    hundred operations), not a loosened agreement."""

    data = _fixture()
    for case in data["cases"]:
        for mode in ("orig", "psmc"):
            expected = case[mode]
            variant = "z-shift" if mode == "orig" else "rescaling"
            masses = [Fraction(m) ** 2 for m in case["masses"]]
            (out,), (record,) = kinematics.improve(
                [case["input"]],
                case["n_initial"],
                masses,
                "arbitrary",
                40,
                variant,
                fallback=False,
            )
            if expected["code"] != 0:
                assert record["status"] == "not-improved", (case["name"], mode)
                assert record["attempts"][-1]["code"] == expected["code"], (
                    case["name"],
                    mode,
                    record,
                )
                continue
            assert record["produced"] == variant, (case["name"], mode, record)
            scale = sum(abs(Fraction(leg[0])) for leg in case["input"])
            worst = max(
                abs(Fraction(a) - Fraction(b)) / scale
                for leg, ref in zip(out, expected["point"], strict=True)
                for a, b in zip(leg, ref, strict=True)
            )
            assert worst < Fraction(1, 10**30), (case["name"], mode, float(worst))


@pytest.mark.loop_fast
def test_the_projection_is_exact_and_moves_by_the_inputs_defect(loop_native):
    """The frame-independent projection (DECISIONS 01:30 (A)) on the
    fixture's generic-frame inputs: exact to the carrier's rounding in
    double-double and at 50 / 100 digits, the move of the input's own
    defect, iterations recorded."""

    for case in _fixture()["cases"]:
        masses = [Fraction(m) ** 2 for m in case["masses"]]
        for arithmetic, digits in (
            ("double-double", None),
            ("arbitrary", 50),
            ("arbitrary", 100),
        ):
            (_out,), (record,) = kinematics.improve(
                [case["input"]],
                case["n_initial"],
                masses,
                arithmetic,
                digits,
                "projection",
            )
            assert record["produced"] == "projection", (case["name"], record)
            assert record["exactness"]["exact"], (case["name"], arithmetic, record)
            assert record["warning"] is None
            assert record["moved"] < 1e-14, record
            assert record["iterations"] >= 1


# ------------------------------------------------------------ route tests


def _generate(tmp: Path, entries, compiled_model) -> Path:
    from .test_loop_v2_artifact import _card
    from .test_loop_v2_artifact import _generate as generate

    return generate(_card(compiled_model, tmp / "artifact", entries), tmp, entries)


@pytest.fixture(scope="module")
def gg(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """g g > g g (full colour) through the public Generator."""

    tmp = tmp_path_factory.mktemp("ki-gg")
    return _generate(tmp, (("gg", "g g > g g"),), loop_sm_compiled_model)


def _beam_point() -> list[list[str]]:
    """A 2 -> 2 point with beams along z whose final states are formed in
    binary64 (their shells missed by binary64 rounding)."""

    c, phi, e = 0.37, 1.1, 500.0
    s = math.sqrt(1 - c * c)
    p3 = [e, e * s * math.cos(phi), e * s * math.sin(phi), e * c]
    p4 = [e, -p3[1], -p3[2], -p3[3]]
    return [
        [str(x) for x in leg] for leg in ([e, 0.0, 0.0, e], [e, 0.0, 0.0, -e], p3, p4)
    ]


def _rotated(point: list[list[str]]) -> list[list[str]]:
    """The point rotated in binary64 (beams no longer along z)."""

    a = 0.61
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for leg in point:
        e, x, y, z = (float(v) for v in leg)
        out.append([str(v) for v in (e, ca * x + sa * z, y, -sa * x + ca * z)])
    return out


EXACT = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    ["500", "300", "0", "400"],
    ["500", "-300", "0", "-400"],
]

CALL = {"mu_r2": "8315.06", "poles": "none", "on_failure": "return"}


def _recommended():
    """``RECOMMENDED_CHECKS``: one rotation, the rescaled reduction-direction
    check."""

    from pyamplicol.api import VirtualStabilityChecks

    return VirtualStabilityChecks(1, True, rescaled=True)


def _same(a, b) -> None:
    assert repr(a.values) == repr(b.values)


@pytest.mark.loop_physics
def test_the_summed_route_in_double_double_and_arbitrary_precision(gg):
    """The default (MadLoop's ORIG on beams along z) in double-double, at 32
    digits and 'arbitrary' at 16 digits (above binary64 by the carrier key,
    DECISIONS 2026-11-10 02:29 (G3) (a)): the record names the producer, the
    point is exact, and the value is that of the pre-improved point with the
    improvement off, bitwise. Binary64 is untouched (no record, the
    unimproved values) unless the binary64 switch is on."""

    from pyamplicol import Runtime

    runtime = Runtime.load(gg, process="gg")
    backend = runtime._backend
    point = _beam_point()
    for carrier, digits in (
        ("double-double", {}),
        ("arbitrary", {"precision": 32}),
        ("arbitrary", {"precision": 16}),
    ):
        call = {**CALL, "arithmetic": carrier, **digits}
        found = runtime.evaluate([point], **call)
        (record,) = [lane.kinematics for lane in found.diagnostics]
        assert record.status == "improved" and record.produced == "z-shift", record
        assert record.attempts == (("z-shift", 0),) and record.exact, record
        assert record.warning is None
        improved, _ = backend.improved_points(
            [point], arithmetic=carrier, **digits, mu_r2=CALL["mu_r2"]
        )
        _same(found, runtime.evaluate(improved, kinematics_improvement="off", **call))
        off = runtime.evaluate([point], kinematics_improvement="off", **call)
        assert off.diagnostics[0].kinematics.status == "off"
        # The move is the inputs' own defect (relative ~1e-16): the values
        # move at that level.
        a, b = found.values[0][0].finite, off.values[0][0].finite
        assert abs(Decimal(a) - Decimal(b)) <= Decimal("1e-12") * abs(Decimal(b))
    # The stability rotations act on the improved point (never re-improved):
    # the checked call's values and estimates equal the pre-improved point's.
    from pyamplicol.api import VirtualStabilityChecks

    checks = VirtualStabilityChecks(("z:3/5:4/5",), True)
    call = {**CALL, "arithmetic": "double-double", "stability_checks": checks}
    found = runtime.evaluate([point], **call)
    improved, _ = backend.improved_points(
        [point], arithmetic="double-double", mu_r2=CALL["mu_r2"]
    )
    reference = runtime.evaluate(improved, kinematics_improvement="off", **call)
    assert repr(found.values) == repr(reference.values)
    assert (
        repr(found.stability) == repr(reference.stability)
        and found.stability[0] is not None
    )
    # Binary64: off by default (no record), bitwise the unimproved values.
    found = runtime.evaluate([point], arithmetic="double", **CALL)
    assert found.diagnostics[0].kinematics is None
    _same(
        found,
        runtime.evaluate(
            [point], arithmetic="double", kinematics_improvement="off", **CALL
        ),
    )
    # The binary64 switch.
    switched = runtime.evaluate(
        [point], arithmetic="double", kinematics_improvement_binary64=True, **CALL
    )
    assert switched.diagnostics[0].kinematics.status in ("improved", "exact")
    improved, _ = backend.improved_points(
        [point], arithmetic="double", mu_r2=CALL["mu_r2"]
    )
    _same(switched, runtime.evaluate(improved, arithmetic="double", **CALL))


@pytest.mark.loop_physics
def test_an_exact_point_is_passed_unchanged_and_a_rotated_frame_is_finished(gg):
    """An exactly physical input is evaluated as given (status "exact",
    bitwise the improvement off); in a frame whose beams are not along z the
    default chain records ORIG's refusal (200), PSMC producing a point that
    misses exactness (its (x^2 - x) p1 residual, far above double-double's
    rounding) and THE FINISH, the projection continuing from PSMC's point
    (USER 2026-11-10 ~19:20 'Projection after MadLoop'; DECISIONS 19:20
    (F1)): exact, no warning, the value of that point with the improvement
    off; PSMC alone (``fallback=False``) keeps the typed exactness warning.
    "projection" is exact there too."""

    from pyamplicol import Runtime

    runtime = Runtime.load(gg, process="gg")
    backend = runtime._backend
    call = {**CALL, "arithmetic": "double-double"}
    found = runtime.evaluate([EXACT], **call)
    assert found.diagnostics[0].kinematics.status == "exact"
    _same(found, runtime.evaluate([EXACT], kinematics_improvement="off", **call))
    rotated = _rotated(_beam_point())
    found = runtime.evaluate([rotated], **call)
    record = found.diagnostics[0].kinematics
    assert record.attempts == (
        ("z-shift", 200),
        ("rescaling", 0),
        ("projection", 0),
    ), record
    assert record.produced == "projection" and record.exact, record
    assert record.warning is None
    assert found.values[0] is not None and found.diagnostics[0].status == "success"
    improved, _ = backend.improved_points(
        [rotated], arithmetic="double-double", mu_r2=CALL["mu_r2"]
    )
    _same(found, runtime.evaluate(improved, kinematics_improvement="off", **call))
    _psmc, (alone,) = backend.improved_points(
        [rotated],
        arithmetic="double-double",
        mu_r2=CALL["mu_r2"],
        variant="rescaling",
        fallback=False,
    )
    assert alone.produced == "rescaling" and not alone.exact, alone
    assert alone.warning == "kinematics-inexact" and "rescaling" in alone.warning_text
    projected = runtime.evaluate([rotated], kinematics_improvement="projection", **call)
    record = projected.diagnostics[0].kinematics
    assert record.produced == "projection" and record.exact and record.warning is None


@pytest.mark.loop_physics
def test_the_resolved_correlated_and_rescue_routes(gg):
    """Resolved and colour-correlated in double-double and at 32 digits, and
    a rescue stack (binary64 -> double-double -> 40 digits): each level
    improves the original inputs in its own carrier (its record), the values
    equal the pre-improved point's with the improvement off."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy
    from pyamplicol.correlators import CorrelatedRequest

    runtime = Runtime.load(gg, process="gg")
    backend = runtime._backend
    point = _beam_point()
    requests = {"born": CorrelatedRequest("born", {})}
    for carrier, digits in (("double-double", {}), ("arbitrary", {"precision": 32})):
        call = {**CALL, "arithmetic": carrier, **digits}
        improved, _ = backend.improved_points(
            [point], arithmetic=carrier, **digits, mu_r2=CALL["mu_r2"]
        )
        found = runtime.evaluate_resolved([point], **call)
        assert found.diagnostics[0].kinematics.produced == "z-shift", carrier
        assert repr(found.values) == repr(
            runtime.evaluate_resolved(
                improved, kinematics_improvement="off", **call
            ).values
        )
        # The resolved route's stability rotations act on the improved
        # point, never on the binary64 one rotated then lifted (DECISIONS
        # 2026-11-10 18:54 (b)): values and estimates are the pre-improved
        # point's with the same checks.
        checked = {**call, "stability_checks": _recommended()}
        found = runtime.evaluate_resolved([point], **checked)
        reference = runtime.evaluate_resolved(
            improved, kinematics_improvement="off", **checked
        )
        assert repr(found.values) == repr(reference.values)
        assert found.stability[0] is not None
        assert repr(found.stability) == repr(reference.stability)
        many = runtime.evaluate_correlated_many([point], requests, **call)
        reference = runtime.evaluate_correlated_many(
            improved, requests, kinematics_improvement="off", **call
        )
        assert many["born"].diagnostics[0].kinematics.produced == "z-shift"
        assert repr(many["born"].values) == repr(reference["born"].values)
    # A rescue stack whose binary64 level never meets its target: every
    # level's attempt records its own carrier's improvement.
    policy = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", target=1e-300),
            VirtualRescueLevel("double-double", target=1e-300),
            VirtualRescueLevel("arbitrary", precision=40),
        )
    )
    rescued = runtime.evaluate([point], stability=policy, **CALL)
    attempts = rescued.rescue[0].attempts
    by_carrier = {a.arithmetic: a.kinematics for a in attempts}
    assert by_carrier["double"] is None
    assert by_carrier["double-double"].produced == "z-shift"
    assert by_carrier["arbitrary"].produced == "z-shift"
    improved40, _ = backend.improved_points(
        [point], arithmetic="arbitrary", precision=40, mu_r2=CALL["mu_r2"]
    )
    direct = runtime.evaluate(
        improved40,
        arithmetic="arbitrary",
        precision=40,
        kinematics_improvement="off",
        **{k: v for k, v in CALL.items()},
    )
    assert repr(rescued.values[0]) == repr(direct.values[0])
    # Each level's checks (one rotation and the rescaled re-reduction) act on
    # the point improved in that level's carrier (DECISIONS 2026-11-10 18:54
    # (b)): the level's estimate is that of its pre-improved point with the
    # level's checks, the improvement off.
    for level, digits in (
        (policy.levels[1], {}),
        (policy.levels[2], {"precision": 40}),
    ):
        (attempt,) = [a for a in attempts if a.arithmetic == level.arithmetic]
        at, _ = backend.improved_points(
            [point], arithmetic=level.arithmetic, **digits, mu_r2=CALL["mu_r2"]
        )
        checked = runtime.evaluate(
            at,
            arithmetic=level.arithmetic,
            **digits,
            stability_checks=level.checks,
            kinematics_improvement="off",
            **CALL,
        )
        assert attempt.estimate == checked.stability[0].relative, level


@pytest.mark.loop_physics
def test_the_born_interfered_helicity_selection(gg):
    """The Born-interfered helicity selection (lit-fc-hel, full colour; the
    ordinary and the resolved route) in double-double and at 32 digits: the
    record names the producer and the selected rows' values are those of
    the pre-improved point with the improvement off, bitwise (the
    improvement adds kinematics only: the same rows are formed, USER
    2026-11-08 ~22:34)."""

    from pyamplicol import Runtime

    runtime = Runtime.load(gg, process="gg")
    backend = runtime._backend
    point = _beam_point()
    for carrier, digits in (("double-double", {}), ("arbitrary", {"precision": 32})):
        call = {**CALL, "arithmetic": carrier, **digits}
        improved, _ = backend.improved_points(
            [point], arithmetic=carrier, **digits, mu_r2=CALL["mu_r2"]
        )
        found = runtime.evaluate([point], helicities=["h1"], **call)
        assert found.diagnostics[0].kinematics.produced == "z-shift", carrier
        _same(
            found,
            runtime.evaluate(
                improved, helicities=["h1"], kinematics_improvement="off", **call
            ),
        )
        resolved = runtime.evaluate_resolved([point], helicities=["h1"], **call)
        assert resolved.diagnostics[0].kinematics.produced == "z-shift"
        assert repr(resolved.values) == repr(
            runtime.evaluate_resolved(
                improved, helicities=["h1"], kinematics_improvement="off", **call
            ).values
        )


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > h h (loop-induced) through the public Generator (process ``p``),
    the helper and options of the loop-induced observables tests."""

    from tests.unit.loop_v2_induced_observables_rows import generate

    return generate(tmp_path_factory.mktemp("ki-li"), "li_gg_hh", "g g > h h")


def _higgs_point() -> list[list[str]]:
    """g g > h h with beams along z, the Higgs momenta formed in binary64
    (their shells missed by binary64 rounding)."""

    e, mh, c, phi = 300.0, 125.0, 0.43, 2.1
    q = math.sqrt(e * e - mh * mh)
    s = math.sqrt(1 - c * c)
    p3 = [e, q * s * math.cos(phi), q * s * math.sin(phi), q * c]
    p4 = [e, -p3[1], -p3[2], -p3[3]]
    return [
        [str(x) for x in leg] for leg in ([e, 0.0, 0.0, e], [e, 0.0, 0.0, -e], p3, p4)
    ]


@pytest.mark.loop_physics
def test_the_loop_induced_selection_and_rescue_stack(li):
    """Loop-induced g g > h h: the helicity-selected route in double-double
    and at 32 digits (the record; the selected row's value of the
    pre-improved point with the improvement off, bitwise), and a rescue
    stack binary64 "auto" (TIRust where served) -> binary64 NinjaRS (the
    loop-induced default stack's NinjaRS level) -> double-double "auto"
    (TIRust with the per-point NinjaRS fallback at that carrier) -> 40
    digits: the binary64
    levels have no record (the switch is off), every higher level improves
    the ORIGINAL inputs in its own carrier, and the last level's value is
    the pre-improved point's at 40 digits."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy

    runtime = Runtime.load(li, process="p")
    backend = runtime._backend
    point = _higgs_point()
    base = {"mu_r2": "15625", "poles": "none", "on_failure": "return"}
    for carrier, digits in (("double-double", {}), ("arbitrary", {"precision": 32})):
        call = {**base, "arithmetic": carrier, **digits}
        improved, _ = backend.improved_points(
            [point], arithmetic=carrier, **digits, mu_r2=base["mu_r2"]
        )
        found = runtime.evaluate([point], helicities=["h0"], **call)
        record = found.diagnostics[0].kinematics
        assert record.produced == "z-shift" and record.exact, (carrier, record)
        _same(
            found,
            runtime.evaluate(
                improved, helicities=["h0"], kinematics_improvement="off", **call
            ),
        )
    policy = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", target=1e-300),
            VirtualRescueLevel("double", reduction="ninjars", target=1e-300),
            VirtualRescueLevel("double-double", target=1e-300),
            VirtualRescueLevel("arbitrary", precision=40),
        )
    )
    rescued = runtime.evaluate([point], stability=policy, **base)
    attempts = rescued.rescue[0].attempts
    assert [a.arithmetic for a in attempts] == [
        "double",
        "double",
        "double-double",
        "arbitrary",
    ], attempts
    assert attempts[0].kinematics is None and attempts[1].kinematics is None
    assert attempts[2].kinematics.produced == "z-shift"
    assert attempts[3].kinematics.produced == "z-shift"
    improved40, _ = backend.improved_points(
        [point], arithmetic="arbitrary", precision=40, mu_r2=base["mu_r2"]
    )
    direct = runtime.evaluate(
        improved40,
        arithmetic="arbitrary",
        precision=40,
        kinematics_improvement="off",
        **base,
    )
    assert repr(rescued.values[0]) == repr(direct.values[0])


# ------------------------------------------------------------- tree routes


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """u u~ > t t~ (built-in SM, recurrence) through the public Generator."""

    from pyamplicol import ColorCorrelator, CorrelatorConfig, Generator, ModelSource
    from tests.integration.test_builtin_sm_normalization import _config

    artifact = tmp_path_factory.mktemp("ki-tree") / "artifact"
    Generator(_config()).generate(
        "u u~ > t t~",
        artifact,
        model=ModelSource.built_in_sm(),
        correlators=CorrelatorConfig(
            color_correlations=(ColorCorrelator.dipole("T13", 1, 3),)
        ),
    )
    return artifact


def _top_point() -> list[list[float]]:
    """u u~ > t t~ with beams along z, the tops formed in binary64."""

    e, mt, c, phi = 400.0, 173.0, 0.31, 0.8
    q = math.sqrt(e * e - mt * mt)
    s = math.sqrt(1 - c * c)
    p3 = [e, q * s * math.cos(phi), q * s * math.sin(phi), q * c]
    return [[e, 0.0, 0.0, e], [e, 0.0, 0.0, -e], p3, [e, -p3[1], -p3[2], -p3[3]]]


@pytest.mark.loop_physics
def test_the_tree_higher_precision_modes_and_the_binary64_switch(tree):
    """The tree runtime's higher-precision modes (precision 32 / 50, the
    resolved route, the correlated default 'arbitrary' at 16 digits: the
    carrier key, DECISIONS 2026-11-10 02:29 (G3)) evaluate the improved point
    (the value of the pre-improved point with the improvement off, bitwise),
    the record goes through the warnings channel; precision 16 is bitwise
    unchanged unless the binary64 switch is on."""

    from pyamplicol import Runtime

    runtime = Runtime.load(tree)
    backend = runtime._backend
    point = _top_point()
    texts = [[str(x) for x in leg] for leg in point]
    for precision in (32, 50):
        found = runtime.evaluate([point], precision=precision)
        notes = runtime.take_warnings()
        assert any(
            note.startswith("kinematics-improvement: arbitrary") for note in notes
        ), notes
        # The record of the most recent call only (bounded), consumed.
        assert len(notes) == 1 and runtime.take_warnings() == (), notes
        from pyamplicol.runtime.symbolica_exact import _working_precision

        masses, initial = backend._tree_kinematics_masses(texts, precision)
        assert initial == 2 and masses[2] == masses[3] != 0
        (improved,), (record,) = kinematics.improve(
            [texts], 2, masses, "arbitrary", _working_precision(precision), "z-shift"
        )
        assert record["produced"] == "z-shift" and record["exactness"]["exact"], record
        assert found == runtime.evaluate(
            [improved], precision=precision, kinematics_improvement="off"
        )
        resolved = runtime.evaluate_resolved([point], precision=precision)
        assert (
            resolved.values
            == runtime.evaluate_resolved(
                [improved], precision=precision, kinematics_improvement="off"
            ).values
        )
    # Binary64: bitwise the unimproved point unless switched on.
    plain = runtime.evaluate([point])
    assert plain == runtime.evaluate([point], kinematics_improvement="off")
    switched = runtime.evaluate([point], kinematics_improvement_binary64=True)
    assert runtime.take_warnings() == (
        "kinematics-improvement: double at precision 16 (z-shift): z-shift 1",
    )
    # The native tree runtime's binary64 implementation (the tree C ABI's),
    # toward the binary64 masses of its external wavefunctions.
    (b64,), (text,) = backend._runtime.improve_f64([point], "z-shift")
    record = json.loads(text)
    assert record["produced"] == "z-shift" and record["exactness"]["exact"], record
    assert b64 != point
    assert switched == runtime.evaluate([b64])
    # Muted: no record.
    runtime.mute_warnings()
    runtime.evaluate([point], kinematics_improvement_binary64=True)
    runtime.unmute_warnings()
    assert runtime.take_warnings() == ()


@pytest.mark.loop_physics
def test_the_tree_correlated_default_is_improved_at_16_digits(tree):
    """``evaluate_correlated``'s default carrier ('arbitrary' at precision
    16) is above binary64 by the carrier key (DECISIONS 2026-11-10 02:29
    (G3) (a)): its point is improved (the value of the pre-improved point
    with the improvement off), and ``"off"`` restores the given point."""

    from pyamplicol import Runtime
    from pyamplicol.runtime.symbolica_exact import _working_precision

    runtime = Runtime.load(tree)
    backend = runtime._backend
    point = _top_point()
    texts = [[str(x) for x in leg] for leg in point]
    masses, _ = backend._tree_kinematics_masses(texts, 16)
    (improved,), (record,) = kinematics.improve(
        [texts], 2, masses, "arbitrary", _working_precision(16), "z-shift"
    )
    assert record["produced"] == "z-shift"
    found = runtime.evaluate_correlated([point])
    assert found == runtime.evaluate_correlated(
        [improved], kinematics_improvement="off"
    )
    assert runtime.evaluate_correlated([point], kinematics_improvement="off") == (
        runtime.evaluate_correlated([texts], kinematics_improvement="off")
    )


# ------------------------------------------------------------ the loop SDK


def _json_of(getter, result) -> dict:
    import ctypes

    needed = ctypes.c_size_t()
    assert getter(result, None, ctypes.c_size_t(0), ctypes.byref(needed)) == 0
    buffer = ctypes.create_string_buffer(needed.value)
    assert getter(result, buffer, needed, ctypes.byref(needed)) == 0
    return json.loads(buffer.value)


@pytest.mark.loop_physics
def test_the_loop_sdk_improves_like_the_python_api(gg):
    """The C ABI (``rusticol_virtual_set_kinematics_improvement[_binary64]``,
    additive, ABI version 2 kept) improves each call's points in its carrier
    exactly as the Python API: in double-double and at 32 digits the
    diagnostics record
    (``"kinematics"``) names the producer and the decimal values equal the
    Python API's on the same exact inputs; ``"off"`` records ``"off"``; a
    binary64 call has no record unless the binary64 switch is on; another
    value is refused typed with the one refusal text."""

    import ctypes

    from pyamplicol import Runtime

    from .test_loop_v2_sdk_extras import (
        _agree,
        _error,
        _laurent,
        _open,
        _Request,
        _ResultShape,
        _sdk,
    )

    sdk = _sdk()
    point = _beam_point()
    exact = [[str(Decimal(float(c))) for c in leg] for leg in point]
    runtime = Runtime.load(gg, process="gg")
    python = runtime.evaluate(
        [exact],
        arithmetic="double-double",
        # The exact binary64 value of the SDK's mu_R^2 argument.
        mu_r2=str(Decimal(8315.06)),  # noqa: RUF032
        poles="all",
    )

    def call(setting=None, binary64=None, arithmetic="double-double", digits=0):
        handle = _open(sdk, gg, "gg")
        try:
            assert (
                sdk.rusticol_virtual_set_arithmetic(
                    handle, arithmetic.encode(), ctypes.c_uint32(digits)
                )
                == 0
            ), _error(sdk)
            if setting is not None:
                status = sdk.rusticol_virtual_set_kinematics_improvement(
                    handle, setting.encode()
                )
                if status != 0:
                    return status, _error(sdk)
            if binary64 is not None:
                status = sdk.rusticol_virtual_set_kinematics_improvement_binary64(
                    handle, ctypes.c_int(binary64)
                )
                if status != 0:
                    return status, _error(sdk)
            flat = [float(c) for leg in point for c in leg]
            momenta = (ctypes.c_double * len(flat))(*flat)
            mu = (ctypes.c_double * 1)(8315.06)
            request = _Request(momenta, len(flat), 1, None, mu, 1)
            result = ctypes.c_void_p()
            status = sdk.rusticol_virtual_evaluate(
                handle, ctypes.byref(request), ctypes.byref(result)
            )
            assert status == 0, _error(sdk)
            try:
                shape = _ResultShape()
                assert (
                    sdk.rusticol_virtual_result_shape(result, ctypes.byref(shape)) == 0
                )
                diagnostics = _json_of(
                    sdk.rusticol_virtual_result_diagnostics_json, result
                )
                decimal = _json_of(sdk.rusticol_virtual_result_decimal_json, result)
            finally:
                sdk.rusticol_virtual_result_free(result)
        finally:
            sdk.rusticol_virtual_free(handle)
        return diagnostics, decimal

    diagnostics, decimal = call()
    (record,) = diagnostics["kinematics"]
    assert record["produced"] == "z-shift" and record["exactness"]["exact"], record
    # The same improved point as the Python API's (its record, every
    # exactness number included) and the same values.
    mine = python.diagnostics[0].kinematics
    assert (record["status"], record["produced"]) == (mine.status, mine.produced)
    assert [
        tuple(a)
        for a in ((attempt["step"], attempt["code"]) for attempt in record["attempts"])
    ] == list(mine.attempts)
    assert record["exactness"]["shell"] == mine.shell
    assert record["exactness"]["conservation"] == mine.conservation
    assert record["moved"] == mine.moved
    want = [x for lc in python.values[0] for x in _laurent(lc)]
    assert _agree(want, decimal["virtual"]), (want, decimal["virtual"])
    # Arbitrary precision (32 digits): the same improved point and values as
    # the Python API's at that carrier.
    python32 = runtime.evaluate(
        [exact],
        arithmetic="arbitrary",
        precision=32,
        mu_r2=str(Decimal(8315.06)),  # noqa: RUF032
        poles="all",
    )
    diagnostics, decimal = call(arithmetic="arbitrary", digits=32)
    (record,) = diagnostics["kinematics"]
    mine = python32.diagnostics[0].kinematics
    assert (record["status"], record["produced"]) == (mine.status, mine.produced)
    assert record["exactness"]["shell"] == mine.shell
    assert record["exactness"]["conservation"] == mine.conservation
    assert record["moved"] == mine.moved
    want = [x for lc in python32.values[0] for x in _laurent(lc)]
    assert _agree(want, decimal["virtual"]), (want, decimal["virtual"])
    diagnostics, _ = call("off")
    assert diagnostics["kinematics"][0]["status"] == "off"
    diagnostics, _ = call(arithmetic="double")
    assert "kinematics" not in diagnostics
    diagnostics, _ = call(binary64=1, arithmetic="double")
    assert diagnostics["kinematics"][0]["status"] in ("improved", "exact")
    status, message = call("orig")
    assert status != 0 and kinematics.policy()["setting_refusal"] in message, message
