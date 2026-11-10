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
        a, b = found.values[0][0].finite, off.values[0][0].finite
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
    from pyamplicol.correlators import CorrelatedRequest

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
    attempts = rescued.rescue[0].attempts
    by_carrier = {a.arithmetic: a.kinematics for a in attempts}
    assert by_carrier["double"] is None
    assert by_carrier["double-double"].produced == "z-shift"
    assert by_carrier["arbitrary"].produced == "z-shift"
    improved40, _ = backend.improved_points([point], arithmetic="arbitrary", precision=40, mu_r2=CALL["mu_r2"])
    direct = runtime.evaluate(
        improved40, arithmetic="arbitrary", precision=40, kinematics_improvement="off", **{k: v for k, v in CALL.items()}
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
        notes = runtime.take_warnings() if hasattr(runtime, "take_warnings") else backend.take_warnings()
        assert any(note.startswith("kinematics-improvement: arbitrary") for note in notes), notes
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
        assert resolved.values == runtime.evaluate_resolved(
            [improved], precision=precision, kinematics_improvement="off"
        ).values
    # Binary64: bitwise the unimproved point unless switched on.
    plain = runtime.evaluate([point])
    assert plain == runtime.evaluate([point], kinematics_improvement="off")
    switched = runtime.evaluate([point], kinematics_improvement_binary64=True)
    (b64,), (record,) = kinematics.improve(
        [texts], 2, backend._tree_kinematics_masses(texts, 16)[0], "double", None, "z-shift"
    )
    assert switched == runtime.evaluate([[[float(x) for x in leg] for leg in b64]])


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
    assert found == runtime.evaluate_correlated([improved], kinematics_improvement="off")
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
    exactly as the Python API: in double-double the diagnostics record
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
        [exact], arithmetic="double-double", mu_r2=str(Decimal(8315.06)), poles="all"
    )

    def call(setting=None, binary64=None, arithmetic="double-double"):
        handle = _open(sdk, gg, "gg")
        try:
            assert sdk.rusticol_virtual_set_arithmetic(
                handle, arithmetic.encode(), ctypes.c_uint32(0)
            ) == 0, _error(sdk)
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
                assert sdk.rusticol_virtual_result_shape(result, ctypes.byref(shape)) == 0
                diagnostics = _json_of(sdk.rusticol_virtual_result_diagnostics_json, result)
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
    assert [tuple(a) for a in (
        (attempt["step"], attempt["code"]) for attempt in record["attempts"]
    )] == list(mine.attempts)
    assert record["exactness"]["shell"] == mine.shell
    assert record["exactness"]["conservation"] == mine.conservation
    assert record["moved"] == mine.moved
    want = [x for lc in python.values[0] for x in _laurent(lc)]
    assert _agree(want, decimal["virtual"]), (want, decimal["virtual"])
    diagnostics, _ = call("off")
    assert diagnostics["kinematics"][0]["status"] == "off"
    diagnostics, _ = call(arithmetic="double")
    assert "kinematics" not in diagnostics
    diagnostics, _ = call(binary64=1, arithmetic="double")
    assert diagnostics["kinematics"][0]["status"] in ("improved", "exact")
    status, message = call("orig")
    assert status != 0 and kinematics.policy()["setting_refusal"] in message, message
