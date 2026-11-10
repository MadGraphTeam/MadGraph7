# SPDX-License-Identifier: 0BSD
"""The kinematics improvement of the input points (m-lit lit-psimprove; USER
2026-11-10 ~00:56 / ~01:56; DECISIONS 2026-11-10 01:56 and 02:29).

Unit tests of the Python surface of the one Rust implementation
(``rusticol_core::kinematics_improvement`` through ``pyamplicol.kinematics``):
the native policy (MadLoop's default variant, the binary64 switch off, the
carrier key), the validator's one refusal text, MadLoop 1 / 2 against
MadLoop's own ``improve_ps.inc`` built standalone in quadruple precision
(``tests/fixtures/loop/kinematics_improvement_madloop.json``), the projection
and the records.

Route tests on public v2 artifacts (g g > g g; loop-induced g g > h h): with
the improvement on, each double-double and arbitrary-precision route
(summed, resolved, colour-correlated, helicity-selected, a rescue stack)
records the variant that produced its point and returns the values of the
explicitly pre-improved exact point evaluated with the improvement off,
bitwise; binary64 is untouched unless switched on; ``"off"`` records
``"off"``; an exactly physical point is passed unchanged; a frame whose beams
are not along z takes MadLoop's fallback (PSMC, with the typed exactness
warning) under the default and is exact with ``"projection"``.
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


# -------------------------------------------------- MadLoop's improve_ps.inc


def _fixture() -> dict:
    return json.loads(MADLOOP.read_text())


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


def test_the_projection_is_exact_and_moves_by_the_inputs_defect(loop_native):
    """The frame-independent projection (DECISIONS 01:30 (A)) on the
    fixture's generic-frame inputs: exact to the carrier's rounding in
    double-double and at 50 / 100 digits, the move of the input's own
    defect, iterations recorded."""

    for case in _fixture()["cases"]:
        masses = [Fraction(m) ** 2 for m in case["masses"]]
        for arithmetic, digits in (("double-double", None), ("arbitrary", 50), ("arbitrary", 100)):
            (out,), (record,) = kinematics.improve(
                [case["input"]], case["n_initial"], masses, arithmetic, digits, "projection"
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
        [str(x) for x in leg]
        for leg in ([e, 0.0, 0.0, e], [e, 0.0, 0.0, -e], p3, p4)
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


EXACT = [["500", "0", "0", "500"], ["500", "0", "0", "-500"], ["500", "300", "0", "400"], ["500", "-300", "0", "-400"]]

CALL = {"mu_r2": "8315.06", "poles": "none", "on_failure": "return"}


def _same(a, b) -> None:
    assert repr(a.values) == repr(b.values)


@pytest.mark.loop_physics
def test_the_summed_route_in_double_double_and_arbitrary_precision(gg):
    """The default (MadLoop's ORIG on beams along z) in double-double and at
    32 digits: the record names the producer, the point is exact, and the
    value is that of the pre-improved point with the improvement off,
    bitwise. Binary64 is untouched (no record, the unimproved values)
    unless the binary64 switch is on."""

    from pyamplicol import Runtime

    runtime = Runtime.load(gg, process="gg")
    backend = runtime._backend
    point = _beam_point()
    for carrier, digits in (("double-double", {}), ("arbitrary", {"precision": 32})):
        call = {**CALL, "arithmetic": carrier, **digits}
        found = runtime.evaluate([point], **call)
        (record,) = [lane.kinematics for lane in found.diagnostics]
        assert record.status == "improved" and record.produced == "z-shift", record
        assert record.attempts == (("z-shift", 0),) and record.exact, record
        assert record.warning is None
        improved, _ = backend.improved_points([point], arithmetic=carrier, **digits, mu_r2=CALL["mu_r2"])
        _same(found, runtime.evaluate(improved, kinematics_improvement="off", **call))
        off = runtime.evaluate([point], kinematics_improvement="off", **call)
        assert off.diagnostics[0].kinematics.status == "off"
        # The move is the inputs' own defect (relative ~1e-16): the values
        # move at that level.
        a, b = found.values[0][0][0].finite, off.values[0][0][0].finite
        assert abs(Decimal(a) - Decimal(b)) <= Decimal("1e-12") * abs(Decimal(b))
    # Binary64: off by default (no record), bitwise the unimproved values.
    found = runtime.evaluate([point], arithmetic="double", **CALL)
    assert found.diagnostics[0].kinematics is None
    _same(found, runtime.evaluate([point], arithmetic="double", kinematics_improvement="off", **CALL))
    # The binary64 switch.
    switched = runtime.evaluate(
        [point], arithmetic="double", kinematics_improvement_binary64=True, **CALL
    )
    assert switched.diagnostics[0].kinematics.status in ("improved", "exact")
    improved, _ = backend.improved_points([point], arithmetic="double", mu_r2=CALL["mu_r2"])
    _same(switched, runtime.evaluate(improved, arithmetic="double", **CALL))


@pytest.mark.loop_physics
def test_an_exact_point_is_passed_unchanged_and_a_rotated_frame_falls_back(gg):
    """An exactly physical input is evaluated as given (status "exact",
    bitwise the improvement off); in a frame whose beams are not along z the
    default chain records ORIG's refusal (200) and PSMC producing the point
    with the typed exactness warning (a record, not a failure: the value is
    returned), and "projection" is exact there."""

    from pyamplicol import Runtime

    runtime = Runtime.load(gg, process="gg")
    call = {**CALL, "arithmetic": "double-double"}
    found = runtime.evaluate([EXACT], **call)
    assert found.diagnostics[0].kinematics.status == "exact"
    _same(found, runtime.evaluate([EXACT], kinematics_improvement="off", **call))
    rotated = _rotated(_beam_point())
    found = runtime.evaluate([rotated], **call)
    record = found.diagnostics[0].kinematics
    assert record.attempts == (("z-shift", 200), ("rescaling", 0)), record
    assert record.produced == "rescaling"
    if not record.exact:
        assert record.warning == "kinematics-inexact"
        assert "rescaling" in record.warning_text
    assert found.values[0] is not None and found.diagnostics[0].status == "success"
    projected = runtime.evaluate([rotated], kinematics_improvement="projection", **call)
    record = projected.diagnostics[0].kinematics
    assert record.produced == "projection" and record.exact and record.warning is None


@pytest.mark.loop_physics
def test_the_resolved_correlated_and_rescue_routes(gg):
    """Resolved and colour-correlated in double-double, and a rescue stack
    (binary64 -> double-double -> 40 digits): each level improves the
    original inputs in its own carrier (its record), the values equal the
    pre-improved point's with the improvement off."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy

    runtime = Runtime.load(gg, process="gg")
    backend = runtime._backend
    point = _beam_point()
    call = {**CALL, "arithmetic": "double-double"}
    improved, _ = backend.improved_points([point], arithmetic="double-double", mu_r2=CALL["mu_r2"])
    found = runtime.evaluate_resolved([point], **call)
    assert found.diagnostics[0].kinematics.produced == "z-shift"
    assert repr(found.values) == repr(
        runtime.evaluate_resolved(improved, kinematics_improvement="off", **call).values
    )
    from pyamplicol.api import CorrelatedRequest

    requests = {"born": CorrelatedRequest("born", {})}
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
    attempts = rescued.diagnostics[0].rescue.attempts
    by_carrier = {a.arithmetic: a.kinematics for a in attempts}
    assert by_carrier["double"] is None
    assert by_carrier["double-double"].produced == "z-shift"
    assert by_carrier["arbitrary"].produced == "z-shift"
    improved40, _ = backend.improved_points([point], arithmetic="arbitrary", precision=40, mu_r2=CALL["mu_r2"])
    direct = runtime.evaluate(
        improved40, arithmetic="arbitrary", precision=40, kinematics_improvement="off", **{k: v for k, v in CALL.items()}
    )
    assert repr(rescued.values[0]) == repr(direct.values[0])
