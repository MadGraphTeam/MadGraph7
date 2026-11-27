# SPDX-License-Identifier: 0BSD
"""NinjaRS's cancellation rescue as pyAmpliCol records and drives it: the
lanes the reduction backend re-reduced within its own reduction (the native
executor's ``local_rescued``, prefixed ``virtual_results.LOCAL`` in a point's
merged counts) are recorded apart from pyAmpliCol's lane rescue, never add
the lane rescue's extra check, reach ``profile``'s per-point records and
totals; the call's target reaches the executor. The executor's side (the
refinement source, the records) is covered by its Rust tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_the_reductions_own_rescue_is_a_record_of_its_own():
    """``VirtualLaneRescue`` keeps its meaning (pyAmpliCol's lane rescue
    served a lane; ``carrier`` a string): the reduction's own rescue
    (``LOCAL`` entries and their largest ``LOCAL_ESTIMATE``) is the separate
    ``VirtualReductionRescue``."""

    from pyamplicol.api.virtual_results import (
        LOCAL,
        LOCAL_ESTIMATE,
        MOMENTS,
        VirtualLaneRescue,
        VirtualReductionRescue,
        reduction_rescue_text,
        rescue_text,
        served_by_lane_rescue,
    )

    counts = (
        ("double-double", 2),
        (MOMENTS + "arbitrary", 1),
        (LOCAL + "double-double", 3),
        (LOCAL_ESTIMATE, 2.5e-9),
    )
    lane = VirtualLaneRescue.of(counts)
    assert (lane.lanes, lane.moments) == ((("double-double", 2),), (("arbitrary", 1),))
    assert (lane.count, lane.carrier) == (2, "arbitrary")
    assert rescue_text(lane, False) == (
        "lane rescue: 2 lanes in double-double, 1 term-scale moment in arbitrary"
    )
    own = VirtualReductionRescue.of(counts)
    assert own is not None
    assert (own.lanes, own.estimate) == ((("double-double", 3),), 2.5e-9)
    assert (own.count, own.precision) == (3, "double-double")
    assert (
        reduction_rescue_text(own) == "reduction's own rescue: 3 lanes in double-double"
    )
    only_own = ((LOCAL + "double-double", 1), (LOCAL_ESTIMATE, 1e-10))
    assert not served_by_lane_rescue(only_own)
    assert served_by_lane_rescue(counts)
    assert not served_by_lane_rescue(())
    assert VirtualReductionRescue.of((("double-double", 1),)) is None
    with pytest.raises(ValueError):
        VirtualLaneRescue.of(only_own)  # a lane rescue serves a lane
    with pytest.raises(ValueError):
        VirtualReductionRescue((("double", 1),))


def test_lanes_served_without_accuracy_evidence_mark_their_point():
    """A point's lanes served without accuracy evidence (``UNCERTIFIED``
    entries of its merged counts, ``plan_export.point_rescued``) name their
    reasons, are no lane-rescue or reduction-rescue lanes, and a lane
    diagnostic that carries them cannot establish accuracy."""

    import pytest

    from pyamplicol.api.virtual_results import (
        LOCAL,
        UNCERTIFIED,
        VirtualLaneDiagnostic,
        VirtualReductionRescue,
        served_by_lane_rescue,
        uncertified_reasons,
    )
    from pyamplicol.loops.v2.plan_export import point_rescued

    below = (
        "a reduction lane vanishes within the resolution of its own terms"
        " (no symmetry proof)"
    )
    output = {
        "rescued": [[]],
        "rescued_moments": [[]],
        "local_rescued": [[((0, 0, 0), "double-double", 1e-9)]],
        "uncertified": [[((1, 0, 0), below), ((2, 0, 0), below)]],
    }
    counts = point_rescued(output, None, 0)
    assert (UNCERTIFIED + below, 2) in counts
    assert uncertified_reasons(counts) == (below,)
    assert not served_by_lane_rescue(counts)
    rescue = VirtualReductionRescue.of(counts)
    assert rescue is not None and rescue.lanes == (("double-double", 1),)
    assert uncertified_reasons(((LOCAL + "double-double", 1),)) == ()
    lane = VirtualLaneDiagnostic(
        "success",
        "double",
        53,
        False,
        "accuracy not established: " + below,
        uncertified=(below,),
    )
    assert lane.valid and lane.has_value and lane.uncertified == (below,)
    with pytest.raises(ValueError):
        VirtualLaneDiagnostic("success", "double", 53, True, "x", uncertified=(below,))


def test_the_reductions_own_rescue_reaches_its_readers():
    """``benchmarking``: no "reduction lanes rescued" cause for the
    reduction's own rescue, its own record instead."""

    from pyamplicol.api.virtual_results import VirtualLaneRescue, VirtualReductionRescue
    from pyamplicol.benchmarking import lane_rescue_reason, reduction_rescue

    own = SimpleNamespace(
        lane_rescue=None,
        reduction_rescue=VirtualReductionRescue((("double-double", 4),), 1e-9),
    )
    both = SimpleNamespace(
        lane_rescue=VirtualLaneRescue((("double-double", 1),)),
        reduction_rescue=VirtualReductionRescue((("double-double", 2),)),
    )
    assert lane_rescue_reason(own) is None
    assert lane_rescue_reason(both) == "reduction lanes rescued in double-double"
    assert reduction_rescue(own) == (4, "double-double")
    assert reduction_rescue(both) == (2, "double-double")
    assert reduction_rescue(SimpleNamespace(reduction_rescue=None)) is None


def test_the_point_counts_merge_the_executors_local_rescues():
    from pyamplicol.api.virtual_results import LOCAL, LOCAL_ESTIMATE
    from pyamplicol.loops.v2.plan_export import merged_rescue, point_rescued

    output = {
        "rescued": [[((0, 0, 0), "double-double")]],
        "local_rescued": [
            [((1, 0, 0), "double-double", 1e-10), ((2, 0, 0), "double-double", 3e-9)]
        ],
    }
    extra = {"local_rescued": [[((1, 0, 0), "double-double", 2e-9)]]}
    point = point_rescued(output, extra, 0)
    assert dict(point) == {
        "double-double": 1,
        LOCAL + "double-double": 3,
        LOCAL_ESTIMATE: 3e-9,
    }
    # An output without the field (an older executor) counts none.
    assert point_rescued({"rescued": [[]]}, None, 0) == ()
    # Over a point's evaluations: counts add, the estimate is the largest.
    own = tuple((key, count) for key, count in point if key != "double-double")
    merged = merged_rescue(
        [SimpleNamespace(lane_rescue=point), SimpleNamespace(lane_rescue=own)]
    )
    assert dict(merged) == {
        "double-double": 1,
        LOCAL + "double-double": 6,
        LOCAL_ESTIMATE: 3e-9,
    }


def test_the_calls_target_reaches_the_executor_only_when_not_default():
    from pyamplicol.loops.v2 import reduction

    assert "target" not in reduction.executor_options()
    with reduction.running("ninjars", target=1e-7):
        assert reduction.executor_options()["target"] == 1e-7
    with reduction.running("ninjars", target=reduction.default_target()):
        assert "target" not in reduction.executor_options()


def test_profile_records_carry_the_reductions_own_rescue():
    from pyamplicol import loop_profile as lp

    row = lp.BackendStability(
        "ninjars",
        per_point=(
            lp.StabilityPoint(0, "measured", 12.0, 11.0),
            lp.StabilityPoint(1, "measured", 9.0, 8.5),
            lp.StabilityPoint(2, "binary64-failed", None, None, "failed"),
        ),
    )
    records = {
        0: (
            0,
            None,
            None,
            None,
            "success",
            None,
            None,
            None,
            None,
            None,
            "a reduction lane vanishes within the resolution of its own terms",
        ),
        1: (
            1,
            None,
            None,
            None,
            "success",
            None,
            None,
            None,
            None,
            (5, "double-double"),
        ),
        # A shorter record (no tenth entry) has none.
        2: (2, None, None, "failed", "failed", None, None, None),
    }
    out = lp._with_reduction_rescues(row, records)
    assert [p.reduction_rescue_lanes for p in out.per_point] == [0, 5, 0]
    assert [p.reduction_rescue_carrier for p in out.per_point] == [
        None,
        "double-double",
        None,
    ]
    assert (out.reduction_rescued_points, out.reduction_rescued_lanes) == (1, 5)
    assert out.reduction_rescue_carriers == (("double-double", 5),)
    # Lanes served without accuracy evidence: the point's reasons and the
    # backend's count.
    assert [p.uncertified for p in out.per_point] == [
        "a reduction lane vanishes within the resolution of its own terms",
        None,
        None,
    ]
    assert out.uncertified_points == 1
    table = out.cli_json()["per_point"]
    assert table["fields"][-4:] == [
        "reduction_rescue_lanes",
        "reduction_rescue_carrier",
        "uncertified",
        "reference_uncertified",
    ]
    assert lp.SAMPLE_FIELDS[-4:] == (
        "reduction_rescue_lanes",
        "reduction_rescue_carrier",
        "uncertified",
        "reference_uncertified",
    )
    causes, _ = lp._stability_causes([out])
    assert (
        "ninjars",
        "stability",
        "reduction rescue",
        "lanes re-reduced in double-double within the reduction",
    ) in causes


@pytest.mark.loop_fast
def test_a_reference_with_uncertified_lanes_is_not_certified():
    """``profile --stability``: a reference served with lanes without
    accuracy evidence (its record's sixth entry, ``harvest``) is not
    certified: the sample's ``reference_certified`` is False whatever its
    estimate or agreement said, ``reference_uncertified`` keeps the reason,
    the true digits are reported all the same, ``per_point`` flags it and
    the backend counts it (``uncertified_reference_points``); a point whose
    reference is clean is unchanged bit for bit in the record."""

    import json
    from fractions import Fraction

    from pyamplicol import loop_profile as lp

    reason = "a reduction lane vanishes within the resolution of its own terms"

    def b64(index):
        # index, value (per sector [finite]), summary, reason, status, ...
        return (index, [[1.0 + 1e-14]], None, None, "success")

    def reference(index, unsure=None):
        exact = Fraction(1)
        head = (index, [[exact]], None, "1", "agreement")
        return head if unsure is None else (*head, unsure)

    records = {0: b64(0), 1: b64(1)}
    clean = {0: reference(0), 1: reference(1)}
    marked = {0: reference(0), 1: reference(1, reason)}

    def rows(references):
        found = lp._sample_records(
            "ninjars",
            [0, 1],
            records,
            references,
            1e-5,
            {},
            lp.GroundTruth("arbitrary", 40),
        )
        return found["fields"], json.loads(found["rows"].text)

    fields, before = rows(clean)
    _fields, after = rows(marked)
    at = {name: k for k, name in enumerate(fields)}
    assert after[0] == before[0]
    assert before[1][at["reference_certified"]] == "agreement"
    assert after[1][at["reference_certified"]] is False
    assert after[1][at["reference_uncertified"]] == reason
    assert before[1][at["reference_uncertified"]] is None
    assert after[1][at["true_digits"]] == before[1][at["true_digits"]] is not None
    changed = {at["reference_certified"], at["reference_uncertified"]}
    assert [x for k, x in enumerate(after[1]) if k not in changed] == [
        x for k, x in enumerate(before[1]) if k not in changed
    ]
    row = lp.BackendStability(
        "ninjars",
        per_point=(
            lp.StabilityPoint(0, "measured", 14.0, None),
            lp.StabilityPoint(1, "measured", 14.0, None),
        ),
    )
    assert lp._with_reference_uncertified(row, clean) is row
    out = lp._with_reference_uncertified(row, marked)
    assert [p.reference_uncertified for p in out.per_point] == [None, reason]
    assert out.uncertified_reference_points == 1
    assert out.per_point[0] == row.per_point[0]
    table = out.cli_json()["per_point"]
    assert table["fields"][-1] == "reference_uncertified"


@pytest.mark.loop_fast
def test_a_flagged_reference_certifies_no_other_backend():
    """``profile --stability``'s agreement (``_agreement_dispositions``): a
    clean backend's reference is certified by agreement only with another
    CLEAN reference; a reference served with lanes without accuracy
    evidence is no comparator, so a clean one that only matches it goes to
    its checked re-run, and the agreement's spread is formed over clean
    references only; with every reference clean nothing changes."""

    from fractions import Fraction

    from pyamplicol import loop_profile as lp

    one, near = Fraction(1), Fraction(1) + Fraction(1, 10**40)
    clean, spread = lp._agreement_dispositions({"ninjars": one, "tirust": near}, set())
    assert clean == {"ninjars": "agreement", "tirust": "agreement"}
    assert spread == Fraction(1, 10**40) / near
    flagged, spread = lp._agreement_dispositions(
        {"ninjars": one, "tirust": near}, {"tirust"}
    )
    assert flagged["ninjars"] == "check"
    assert spread is None
    # The flagged one may agree with the clean one; its own sample reads
    # reference_certified False all the same (``_sample_records``).
    assert flagged["tirust"] == "agreement"
    three, _ = lp._agreement_dispositions(
        {"ninjars": one, "tirust": near, "other": one}, {"tirust"}
    )
    assert three == {
        "ninjars": "agreement",
        "tirust": "agreement",
        "other": "agreement",
    }
    none, _ = lp._agreement_dispositions({"ninjars": None, "tirust": one}, set())
    assert none == {"ninjars": "none", "tirust": "check"}


from tests.unit.test_loop_v2_best_effort import aag  # noqa: E402,F401  (fixture)


def test_the_c_abi_evaluates_a_supplied_refined_point(aag, loop_native):  # noqa: F811
    """``rusticol_virtual_evaluate_refined``: zero correction words are
    ``rusticol_virtual_evaluate`` bit for bit; a caller's double-double point
    (``momenta`` its binary64 words, one correction word) is evaluated at
    that point exactly, bitwise the Python API given it as decimal strings;
    momenta that are not the binary64 roundings of the refined point are
    refused, naming the point, leg and component, and nothing is
    evaluated."""

    import ctypes
    from decimal import Decimal, localcontext
    from fractions import Fraction

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_extras import _error, _sdk
    from tests.unit.test_loop_v2_sdk_induced import (
        STATUS_INVALID_ARGUMENT,
        _evaluate,
        _open,
        _python_flat,
        _Request,
        _same,
    )

    artifact, exact_points, exact_scales = aag
    # The caller's point: the stream points improved in double-double (any
    # point physical to double-double's digits would do).
    legs = len(exact_points[0])
    improved, _ = loop_native.improve_kinematics(
        exact_points, 2, ["0"] * legs, "double-double"
    )
    refined = [[[Fraction(c) for c in leg] for leg in point] for point in improved]
    hi = [[[float(c) for c in leg] for leg in point] for point in refined]
    lo = [
        float(c - Fraction(h))
        for point, words in zip(refined, hi, strict=True)
        for leg, hleg in zip(point, words, strict=True)
        for c, h in zip(leg, hleg, strict=True)
    ]
    assert any(lo), "the double-double point refines its binary64 words"
    with localcontext() as context:
        context.prec = 400
        texts = [
            [
                [str(Decimal(c.numerator) / Decimal(c.denominator)) for c in leg]
                for leg in point
            ]
            for point in refined
        ]
    scales = [float(Fraction(s)) for s in exact_scales]
    call = dict(
        mu_r2=exact_scales, poles="none", reduction="ninjars", on_failure="return"
    )
    expected = Runtime.load(artifact, process="p").evaluate(texts, **call)

    sdk = _sdk()
    sdk.rusticol_virtual_evaluate_refined.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Request),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    def refined_call(handle, words, corrections):
        flat = [c for point in hi for leg in point for c in leg]
        momenta = (ctypes.c_double * len(flat))(*flat)
        mu = (ctypes.c_double * len(scales))(*scales)
        request = _Request(momenta, len(flat), len(hi), None, mu, len(scales))
        extra = (ctypes.c_double * max(len(corrections), 1))(*corrections)
        result = ctypes.c_void_p()
        status = sdk.rusticol_virtual_evaluate_refined(
            handle, ctypes.byref(request), extra, words, ctypes.byref(result)
        )
        if status == 0:
            sdk.rusticol_virtual_result_free(result)
        return status, result

    handle = _open(sdk, artifact)
    try:
        assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0, _error(sdk)
        assert sdk.rusticol_virtual_set_reduction(handle, b"ninjars") == 0
        plain = _evaluate(sdk, handle, hi, scales)
        zero = _evaluate_refined(sdk, handle, hi, scales, [0.0] * len(lo))
        assert _same(zero["virtual"], plain["virtual"])
        found = _evaluate_refined(sdk, handle, hi, scales, lo)
        assert _same(found["virtual"], _python_flat(expected))
        # One component's correction beyond half an ulp of its binary64 word.
        far = list(lo)
        far[4 * legs + 6] = 1e-3 * (abs(hi[1][1][2]) + 1.0)
        status, result = refined_call(handle, 1, far)
        assert status == STATUS_INVALID_ARGUMENT and not result.value
        message = _error(sdk)
        assert (
            "point 1, leg 1, component py" in message and "binary64 rounding" in message
        )
    finally:
        sdk.rusticol_virtual_free(handle)


def _evaluate_refined(sdk, handle, points, scales, corrections) -> dict:
    """``_evaluate`` through ``rusticol_virtual_evaluate_refined`` with one
    correction word per component."""

    import ctypes

    from tests.unit.test_loop_v2_sdk_extras import _error
    from tests.unit.test_loop_v2_sdk_induced import _Request, _ResultShape

    flat = [c for point in points for p in point for c in p]
    momenta = (ctypes.c_double * len(flat))(*flat)
    mu = (ctypes.c_double * len(scales))(*scales)
    request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
    words = (ctypes.c_double * len(corrections))(*corrections)
    result = ctypes.c_void_p()
    status = sdk.rusticol_virtual_evaluate_refined(
        handle, ctypes.byref(request), words, 1, ctypes.byref(result)
    )
    assert status == 0, _error(sdk)
    try:
        shape = _ResultShape()
        assert sdk.rusticol_virtual_result_shape(result, ctypes.byref(shape)) == 0
        loop = (ctypes.c_double * max(shape.virtual_values, 1))()
        assert (
            sdk.rusticol_virtual_result_copy_virtual(result, loop, shape.virtual_values)
            == 0
        )
        return {"virtual": list(loop)[: shape.virtual_values]}
    finally:
        sdk.rusticol_virtual_result_free(result)


# A target no carrier reaches: NinjaRS flags every lane (its binary64
# estimate is at least 2^-53 of the lane) and every re-reduction misses it
# (its own estimate is at least 2^-104 of the lane), so each lane is refused
# by the cancellation rescue, deterministically.
UNREACHABLE = 1e-40
# A target binary64 cannot reach (the fast engine's re-reduction keeps its
# binary64 coefficients' 2^-53) but double-double can: every flagged lane is
# refused by NinjaRS's rescue and served by pyAmpliCol's lane rescue, whose
# reductions take the call's target too (at UNREACHABLE they would refuse as
# well, after climbing every rung of their own ladder).
DOUBLE_DOUBLE_REACHABLE = 1e-20


def test_a_point_refused_by_the_rescue_is_served_best_effort(aag, loop_native):  # noqa: F811
    """A point a lane refused by NinjaRS's cancellation rescue leaves
    without a value (lane rescue off) is ``best-effort`` under the default
    policy: its value is the best-effort pass's (``RESCUE_IN_BEST_EFFORT``:
    the unchecked reduction with NinjaRS's rescue, which serves a refused
    lane the better of its binary64 and re-reduced values; here within 1e-9
    of the unchecked value: this process's lanes move by about 1e-11 with the
    point's last bits, as for the lane rescue below), its record names the refusal
    and carries an uncertainty; ``"return"`` keeps it failed. With the lane
    rescue on, at a target double-double reaches, the refused lanes are
    served and the point is a success."""

    import math

    from pyamplicol import Runtime

    artifact, points, scales = aag
    runtime = Runtime.load(artifact, process="p")
    call = dict(mu_r2=scales, poles="none", reduction="ninjars")
    truth = runtime.evaluate(points, reduction_unchecked=True, **call)
    forced = dict(call, reduction_target=UNREACHABLE)
    strict = runtime.evaluate(points, lane_rescue=False, on_failure="return", **forced)
    assert all(value is None for value in strict.values)
    assert all("cancellation rescue" in lane.detail for lane in strict.diagnostics)
    found = runtime.evaluate(points, lane_rescue=False, **forced)
    assert found.statuses == ("best-effort",) * len(points)
    for p, lane in enumerate(found.diagnostics):
        record = lane.best_effort
        assert record is not None and lane.has_value and not lane.valid, p
        assert "cancellation rescue" in record.refusal, p
        assert record.uncertainty is not None and math.isfinite(record.uncertainty), p
        value = float(found.values[p][0].finite)
        exact = float(truth.values[p][0].finite)
        assert abs(value - exact) <= 1e-9 * abs(exact), p
    rescued = runtime.evaluate(
        points, **dict(call, reduction_target=DOUBLE_DOUBLE_REACHABLE)
    )
    assert rescued.statuses == ("success",) * len(points)
    for p, lane in enumerate(rescued.diagnostics):
        assert lane.lane_rescue is not None and lane.lane_rescue.count > 0, p
        # Every lane re-reduced in double-double at the improved point: the
        # binary64 values themselves differ from it at their own rounding,
        # amplified over the point's lanes (measured 9e-12 relative).
        value = float(rescued.values[p][0].finite)
        exact = float(truth.values[p][0].finite)
        assert abs(value - exact) <= 1e-9 * abs(exact), p


def test_the_c_abi_serves_a_rescue_refused_point_best_effort(aag, loop_native):  # noqa: F811
    """The same through the C ABI (default policy best-effort, lane rescue
    off, the unreachable target): every point ``RUSTICOL_VIRTUAL_BEST_EFFORT``
    with its value, the refusal and the uncertainty in the diagnostics;
    ``"return"`` keeps them ``FAILED``."""

    import ctypes
    import math
    from fractions import Fraction

    from tests.unit.test_loop_v2_sdk_extras import _error, _sdk
    from tests.unit.test_loop_v2_sdk_induced import FAILED_STATUS, _evaluate, _open

    artifact, exact_points, exact_scales = aag
    points = [[[float(c) for c in leg] for leg in point] for point in exact_points]
    scales = [float(Fraction(s)) for s in exact_scales]
    sdk = _sdk()
    sdk.rusticol_virtual_set_lane_rescue.argtypes = [ctypes.c_void_p, ctypes.c_int]
    sdk.rusticol_virtual_set_on_failure.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    sdk.rusticol_virtual_set_reduction_target.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    handle = _open(sdk, artifact)
    try:
        assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0, _error(sdk)
        assert sdk.rusticol_virtual_set_reduction(handle, b"ninjars") == 0
        assert sdk.rusticol_virtual_set_lane_rescue(handle, 0) == 0
        assert sdk.rusticol_virtual_set_reduction_target(handle, UNREACHABLE) == 0
        found = _evaluate(sdk, handle, points, scales)
        assert found["statuses"] == [5] * len(points), found["statuses"]
        for p in range(len(points)):
            record = found["diagnostics"]["best_effort"][str(p)]
            assert "cancellation rescue" in record["refusal"], p
            uncertainty = record["uncertainty"]
            assert uncertainty is not None and math.isfinite(uncertainty), p
        assert sdk.rusticol_virtual_set_on_failure(handle, b"return") == 0
        strict = _evaluate(sdk, handle, points, scales)
        assert strict["statuses"] == [FAILED_STATUS] * len(points)
    finally:
        sdk.rusticol_virtual_free(handle)


REFINED_RUST = """
#[path = "@WRAPPER@"]
#[allow(dead_code)]
mod rusticol;
use rusticol::virtual_amplitudes::Runtime;

fn print(name: &str, values: &[f64]) {
    println!("call {name}");
    let words: Vec<String> =
        values.iter().map(|v| format!("b:{:016x}", v.to_bits())).collect();
    println!("virtual {}", words.join(" "));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[3])?;
    let mut lines = text.lines();
    let head: Vec<usize> = lines
        .next()
        .ok_or("empty input")?
        .split_whitespace()
        .map(|v| v.parse())
        .collect::<Result<_, _>>()?;
    let (points, legs) = (head[0], head[1]);
    let (mut hi, mut lo, mut scales) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines.take(points) {
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|v| v.parse::<i64>().map(|bits| f64::from_bits(bits as u64)))
            .collect::<Result<_, _>>()?;
        hi.extend_from_slice(&values[..4 * legs]);
        lo.extend_from_slice(&values[4 * legs..8 * legs]);
        scales.push(values[8 * legs]);
    }
    let mut runtime = Runtime::open(&args[1], Some(args[2].as_str()), None, None)?;
    runtime.set_poles("none")?;
    runtime.set_reduction("ninjars")?;
    let plain = runtime.evaluate(&hi, points, None, &scales, false, None)?;
    print("plain", &plain.virtual_values);
    let zero = vec![0.0; hi.len()];
    let found =
        runtime.evaluate_refined(&hi, &zero, points, None, &scales, false, None)?;
    print("zero", &found.virtual_values);
    let found = runtime.evaluate_refined(&hi, &lo, points, None, &scales, false, None)?;
    print("refined", &found.virtual_values);
    let mut far = lo.clone();
    far[4 * legs + 6] = 1e-3 * (hi[4 * legs + 6].abs() + 1.0);
    println!("call inconsistent");
    match runtime.evaluate_refined(&hi, &far, points, None, &scales, false, None) {
        Ok(_) => println!("accepted"),
        Err(error) => println!("refused 1 {}", error.message()),
    }
    Ok(())
}
"""


def test_the_cpp_fortran_and_rust_wrappers_evaluate_a_refined_point(
    aag,  # noqa: F811
    loop_native,
    tmp_path,
):
    """``VirtualRuntime::evaluate_refined`` (C++), ``evaluate(...,
    corrections=)`` (Fortran) and the packaged Rust wrapper's
    ``Runtime::evaluate_refined``, compiled against the staged SDK
    (``examples/refined_v4.{cpp,f90}``, ``REFINED_RUST``): zero corrections
    give the plain
    values bit for bit, a double-double point gives the Python API's values
    at that point given as decimal strings, an inconsistent one is refused
    with the point, leg and component named."""

    import struct
    import subprocess
    from decimal import Decimal, localcontext
    from fractions import Fraction

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_extras import SDK, WRAPPER, _compiler, _staged_sdk
    from tests.unit.test_loop_v2_sdk_induced import _python_flat, _same

    artifact, exact_points, exact_scales = aag
    legs = len(exact_points[0])
    improved, _ = loop_native.improve_kinematics(
        exact_points, 2, ["0"] * legs, "double-double"
    )
    refined = [[Fraction(c) for leg in point for c in leg] for point in improved]
    hi = [[float(c) for c in point] for point in refined]
    lo = [
        [float(c - Fraction(h)) for c, h in zip(point, words, strict=True)]
        for point, words in zip(refined, hi, strict=True)
    ]
    scales = [float(Fraction(s)) for s in exact_scales]
    with localcontext() as context:
        context.prec = 400
        texts = [
            [
                [
                    str(Decimal(c.numerator) / Decimal(c.denominator))
                    for c in point[4 * k : 4 * k + 4]
                ]
                for k in range(legs)
            ]
            for point in refined
        ]
    expected = _python_flat(
        Runtime.load(artifact, process="p").evaluate(
            texts,
            mu_r2=exact_scales,
            poles="none",
            reduction="ninjars",
            on_failure="return",
        )
    )
    build = _staged_sdk()
    link = [f"-L{build}", "-lrusticol_loop_sdk", f"-Wl,-rpath,{build}", "-lm"]
    examples = SDK / "examples"
    cpp, fortran = tmp_path / "refined-cpp", tmp_path / "refined-fortran"
    subprocess.run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-O2",
            f"-I{SDK / 'include'}",
            str(examples / "refined_v4.cpp"),
            *link,
            "-o",
            str(cpp),
        ],
        check=True,
        cwd=tmp_path,
    )
    subprocess.run(
        [
            _compiler("FC", "gfortran"),
            "-std=f2018",
            "-O2",
            f"-J{tmp_path}",
            str(SDK / "fortran" / "rusticol_virtual.f90"),
            str(examples / "refined_v4.f90"),
            *link,
            "-o",
            str(fortran),
        ],
        check=True,
        cwd=tmp_path,
    )

    rust = tmp_path / "refined-rust"
    (tmp_path / "client.rs").write_text(REFINED_RUST.replace("@WRAPPER@", str(WRAPPER)))
    subprocess.run(
        [
            _compiler("RUSTC", "rustc"),
            "--edition=2021",
            "-D",
            "warnings",
            str(tmp_path / "client.rs"),
            "-o",
            str(rust),
            "-L",
            f"native={build}",
            "-l",
            "dylib=rusticol_loop_sdk",
            "-C",
            f"link-arg=-Wl,-rpath,{build}",
        ],
        check=True,
        cwd=tmp_path,
    )

    def bits(x: float) -> str:
        return str(struct.unpack("<q", struct.pack("<d", x))[0])

    def feed(path, text) -> None:
        lines = [f"{len(hi)} {legs}"]
        for h, low, s in zip(hi, lo, scales, strict=True):
            lines.append(" ".join(text(x) for x in (*h, *low, s)))
        path.write_text("\n".join(lines) + "\n")

    feed(tmp_path / "hex.txt", float.hex)
    feed(tmp_path / "bits.txt", bits)

    def double(token: str) -> float:
        if token.startswith("b:"):
            return struct.unpack(">d", bytes.fromhex(token[2:]))[0]
        return float.fromhex(token)

    for client, data in ((cpp, "hex.txt"), (fortran, "bits.txt"), (rust, "bits.txt")):
        run = subprocess.run(
            [str(client), str(artifact), "p", str(tmp_path / data)],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        calls: dict = {}
        name = None
        for line in run.stdout.splitlines():
            kind, _, rest = line.strip().partition(" ")
            if kind == "call":
                name = rest
            elif kind == "virtual":
                calls[name] = [double(t) for t in rest.split()]
            else:
                calls[name] = line.strip()
        assert _same(calls["zero"], calls["plain"]), client
        assert _same(calls["refined"], expected), client
        refusal = calls["inconsistent"]
        assert refusal.startswith("refused 1 "), (client, refusal)
        assert "point 1, leg 1, component py" in refusal, (client, refusal)


def test_the_umami_provider_returns_a_best_effort_point(aag, loop_native, tmp_path):  # noqa: F811
    """The UMAMI provider (``examples/best_effort_v4_umami.c``, lane rescue
    off and the unreachable target on the SDK handle): every point NinjaRS's
    rescue refuses comes back with status ``RUSTICOL_VIRTUAL_BEST_EFFORT``
    (5) and its best value (within 1e-9 of the unchecked one); the provider no
    longer aborts the bulk call on a best-effort row."""

    import subprocess
    from fractions import Fraction

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_extras import (
        SDK,
        UMAMI_HEADERS,
        _compiler,
        _staged_sdk,
    )

    artifact, exact_points, exact_scales = aag
    truth = Runtime.load(artifact, process="p").evaluate(
        exact_points,
        mu_r2=exact_scales,
        poles="none",
        reduction="ninjars",
        reduction_unchecked=True,
    )
    build = _staged_sdk()
    client = tmp_path / "best-effort-umami"
    subprocess.run(
        [
            _compiler("CC", "cc"),
            "-std=c11",
            "-O2",
            f"-I{SDK / 'include'}",
            f"-I{SDK / 'umami'}",
            f"-I{UMAMI_HEADERS}",
            str(SDK / "umami" / "pyamplicol_umami_virtual_v1.c"),
            str(SDK / "examples" / "best_effort_v4_umami.c"),
            f"-L{build}",
            "-lrusticol_loop_sdk",
            f"-Wl,-rpath,{build}",
            "-lm",
            "-o",
            str(client),
        ],
        check=True,
        cwd=tmp_path,
    )
    legs = len(exact_points[0])
    lines = [f"{len(exact_points)} {legs}"]
    for point, scale in zip(exact_points, exact_scales, strict=True):
        values = [float(Fraction(c)) for leg in point for c in leg] + [
            float(Fraction(scale))
        ]
        lines.append(" ".join(v.hex() for v in values))
    (tmp_path / "in.txt").write_text("\n".join(lines) + "\n")
    run = subprocess.run(
        [str(client), str(artifact), "p", str(tmp_path / "in.txt"), repr(UNREACHABLE)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    statuses: list[int] = []
    finite: dict[int, list[float]] = {}
    for line in run.stdout.splitlines():
        kind, _, rest = line.partition(" ")
        if kind == "status":
            statuses = [int(s) for s in rest.split()]
        elif kind == "finite":
            event, *values = rest.split()
            finite[int(event)] = [float.fromhex(v) for v in values]
        else:
            raise AssertionError(line)
    assert statuses == [5] * len(exact_points), statuses
    assert sorted(finite) == list(range(len(exact_points)))
    for event, values in finite.items():
        expected = [float(order.finite) for order in truth.values[event]]
        for got, want in zip(values, expected, strict=True):
            assert abs(got - want) <= 1e-9 * max(abs(want), 1e-300), (event, got, want)


def _bubble(native, coefficients):
    """One bubble lane (shifts 0 and (1, 0, 0, 0), squared masses 4, mu^2 = 1)
    with the rank-1 numerator ``c0 + c1 q0`` (``coefficients`` = (c0, c1),
    complex), as a ``V2Reducer.reduce`` group in binary64."""

    def batch(values, columns):
        return native.NativeBatch.from_decimal(
            [(str(complex(v).real), str(complex(v).imag)) for v in values],
            1,
            columns,
            "double",
            None,
            [0],
        )

    c0, c1 = coefficients
    return (
        2,
        batch([0, 0, 0, 0, 1, 0, 0, 0], 8),
        batch([4, 4], 2),
        batch([1], 1),
        batch([0, 1, 1, 0], 4),
        batch([c0, c1, 0, 0, 0], 5),
    )


@pytest.mark.loop_fast
def test_the_python_binding_reports_the_reductions_uncertified_lanes(loop_native):
    """``V2Reducer.take_uncertified`` (the Python binding of
    ``V2Output::uncertified``): a lane zero by symmetry (``N = 1 + 2 q0`` on
    the bubble, whose integral is exactly 0), which NinjaRS's rescue refuses
    in binary64 and whose re-reduced value is consistent with zero within the
    re-reduction's own bound, is served with status "success" and its
    reason; an ordinary lane (``N = i``) carries none; the record is taken."""

    vanishing = (
        "a reduction lane vanishes within the resolution of its own terms at "
        "double-double; no symmetry proof"
    )
    reducer = loop_native.V2Reducer("double", None, "fast")
    (direct,), check = reducer.reduce([_bubble(loop_native, (1, 2))])
    assert check is None
    assert [status for _row, status in direct.statuses] == ["success"]
    (value,) = [complex(float(re), float(im)) for re, im in direct.values.to_decimal()][
        :1
    ]
    assert abs(value) < 1e-14
    assert reducer.take_uncertified() == [(0, 0, vanishing)]
    assert reducer.take_uncertified() == []
    (direct,), _check = reducer.reduce([_bubble(loop_native, (1j, 0))])
    assert [status for _row, status in direct.statuses] == ["success"]
    assert reducer.take_uncertified() == []
