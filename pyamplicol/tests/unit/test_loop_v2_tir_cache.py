# SPDX-License-Identifier: 0BSD
"""TIR throughput levers for loop-induced processes (m-lit lit-cache, PLAN
§3.4, §2.b.2-§2.b.3).

* the aligned family export frame (``plan_export.ALIGNED_FAMILY_FRAMES``,
  opt-in, amplitude mode only): every family of a loop-induced slice is read
  with the process's highest leg in its last entry, so a propagator's shift
  mask is the side not holding that leg in every family (shared
  propagators meet with equal masks, in inclusion order), and
  the value is the default frame's to rounding;
* the amplitude-mode numerator chunks (``V2Options::input_budget``): a
  budget that puts every family in its own bucket chunk leaves every value
  bitwise and lowers the peak of one chunk's family numerators;
* TIRust's profile (``V2Executor.prepare_reduction(profile=True)``): its
  stages join the report and the values stay bitwise.
"""

from __future__ import annotations

import itertools

import pytest

from .test_loop_v2_induced import _fixture, _Rows

# The default-frame and aligned-frame values sum the same integrals in
# other frames and orders (rounding only; far inside the lax rule).
FRAMES = 1e-10


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li-cache"))


def _value(executor, li, name, *, reduction=None):
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.reduction import running

    point, scale, parameters = li.point(name)
    with running(reduction):
        (result,) = executor.evaluate(
            [point], [scale], parameters=parameters, precision=Precision.double()
        )
    return result


def _shift_masks(reading) -> list[int]:
    total, out = 0, []
    for mask, _mass in reading:
        total |= mask
        out.append(total)
    return out


@pytest.mark.loop_physics
def test_aligned_family_frames_share_the_shift_masks(li, monkeypatch):
    from pyamplicol.loops.v2 import plan_export
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.plan_export import NativeExecutor, aligned_frame

    from .test_loop_v2_induced import _request

    name = "li_gg_zz"
    fixture = _fixture(name)
    compiled = li.model(fixture["scheme_data"].get("restriction") or "default")
    default = li.slice(name)
    with monkeypatch.context() as patch:
        patch.setattr(plan_export, "ALIGNED_FAMILY_FRAMES", True)
        aligned = LoopSlice.generate(compiled, _request(fixture, compiled), li.native)
    assert aligned.dressed_frames["aligned"] is True
    assert default.dressed_frames["aligned"] is False
    for key, schedule in aligned.plan.schedules.items():
        assert aligned_frame(key, schedule.trie.frame), key
        reading = schedule.trie.frame.apply(key)
        union = _shift_masks(reading)[-1]
        top = 1 << (union.bit_length() - 1)
        masks = _shift_masks(reading)[:-1]
        # Every shift mask but the last (the full mask, momentum 0) lacks
        # the top leg, and they grow by inclusion.
        assert all(not mask & top for mask in masks), (key, masks)
        assert all(a & b == a for a, b in itertools.pairwise(masks)), masks
    first = _value(NativeExecutor(default, forward=False, inputs="native"), li, name)
    second = _value(NativeExecutor(aligned, forward=False, inputs="native"), li, name)
    for a, b in zip(first.totals, second.totals, strict=True):
        assert abs(float(a) - float(b)) <= FRAMES * abs(float(first.totals[1])), (
            first.totals,
            second.totals,
        )


def _native_run(executor, li, name, **options):
    """(families, failures, report) of one native evaluation of the row's
    first fixture point with ``options`` (the executor's own call)."""

    from pyamplicol.loops.v2.assemble import Precision

    point, scale, parameters = li.point(name)
    evaluations = executor._native_points(
        [point], [scale], parameters, Precision.double()
    )
    native = executor.native("double", None, 1)
    output = native.evaluate(
        executor._inputs(evaluations, "double"), raw_lanes=True, **options
    )
    return output["families"], output["failures"], output["report"]


@pytest.mark.loop_physics
@pytest.mark.parametrize("reduction", ["ninjars", "tirust"])
def test_amplitude_numerator_chunks_bound_the_peak_bitwise(li, loop_native, reduction):
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    if reduction == "tirust" and not loop_native.tirust_available():
        pytest.skip("the staged native extension is built without TIRust")
    name = "li_gg_zz"
    executor = NativeExecutor(li.slice(name), forward=False, inputs="native")
    if reduction == "tirust":
        executor.native("double", None, 1).prepare_reduction("tirust")
    whole = _native_run(executor, li, name, reduction=reduction)
    # One byte: every family is a bucket chunk of its own.
    alone = _native_run(executor, li, name, reduction=reduction, input_budget=1)
    assert whole[1] == alone[1] == [[]]
    assert whole[0] == alone[0]
    peak = "family numerators (one bucket chunk)"
    assert alone[2]["reals_per_lane"][peak] < whole[2]["reals_per_lane"][peak]


@pytest.mark.loop_physics
def test_tirust_profile_joins_the_report_and_keeps_the_values(li, loop_native):
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    if not loop_native.tirust_available():
        pytest.skip("the staged native extension is built without TIRust")
    name = "li_gg_hh"
    executor = NativeExecutor(li.slice(name), forward=False, inputs="native")
    native = executor.native("double", None, 1)
    with pytest.raises(Exception, match="TIRust's settings"):
        native.prepare_reduction("ninjars", profile=True)
    native.prepare_reduction("tirust")
    plain = _native_run(executor, li, name, reduction="tirust")
    native.prepare_reduction("tirust", profile=True)
    profiled = _native_run(executor, li, name, reduction="tirust")
    assert plain[0] == profiled[0] and plain[1] == profiled[1] == [[]]
    assert "reduction: TIR masters" not in plain[2]["seconds"]
    for stage in ("geometry", "masters", "recursion", "gather", "output"):
        assert f"reduction: TIR {stage}" in profiled[2]["seconds"]
    assert profiled[2]["counts"]["TIR master evaluations"] > 0
    native.prepare_reduction("tirust")
    assert _native_run(executor, li, name, reduction="tirust")[0] == plain[0]
