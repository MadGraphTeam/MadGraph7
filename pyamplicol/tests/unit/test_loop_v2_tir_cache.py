# SPDX-License-Identifier: 0BSD
"""TIR throughput levers for loop-induced processes (m-lit lit-cache, PLAN
§3.4, §2.b.2-§2.b.3).

* the aligned family export frame (``LoopSlice.generate(aligned_frames=True)``,
  opt-in, amplitude mode only; on g g > h g, where it moves families): every
  family of a loop-induced slice is read
  with the process's highest leg in its last entry, so a propagator's shift
  mask is the side not holding that leg in every family (shared
  propagators meet with equal masks, in inclusion order), and
  the value is the default frame's to rounding;
* the amplitude-mode numerator chunks (``V2Options::input_budget``), end
  to end: a budget that puts every family in its own bucket chunk leaves
  every value bitwise and lowers the peak of one chunk's family numerators
  (the split the numerator term alone causes is the Rust unit test
  ``loop_execution::v2::tests::
  amplitude_numerators_count_against_the_chunk_budget``);
* TIRust's profile (``V2Executor.prepare_reduction(profile=True)``): its
  stages join the report and the values stay bitwise;
* TIRust's tensor store (``prepare_reduction(store=True)``, opt-in): the
  families of a point chunk share TIRust's per-subset results, the values
  stay bitwise and the reuse is counted;
* the exit codes of the L6 both-backends check
  (``tools/loop_v2_observables_backends.py``, extended by lit-cache for the
  lit-amp rows), on a stubbed runtime.
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
def test_aligned_family_frames_share_the_shift_masks(li):
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.plan_export import NativeExecutor, aligned_frame

    from .test_loop_v2_induced import _request

    # g g > h g: a row whose default dressed frames are not all aligned
    # (g g > z z's are), so the option moves families and the value check
    # sees the moved ones.
    name = "li_gg_hg"
    fixture = _fixture(name)
    compiled = li.model(fixture["scheme_data"].get("restriction") or "default")
    default = li.slice(name)
    aligned = LoopSlice.generate(
        compiled, _request(fixture, compiled), li.native, aligned_frames=True
    )
    assert aligned.dressed_frames["aligned"] is True
    assert default.dressed_frames["aligned"] is False
    moved = [
        key
        for key, schedule in aligned.plan.schedules.items()
        if schedule.trie.frame != default.plan.schedules[key].trie.frame
    ]
    assert moved, "no family moved to an aligned frame"
    assert any(
        not aligned_frame(key, default.plan.schedules[key].trie.frame) for key in moved
    )
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
    assert profiled[2]["counts"]["TIR master requests"] > 0
    native.prepare_reduction("tirust")
    assert _native_run(executor, li, name, reduction="tirust")[0] == plain[0]


@pytest.mark.loop_physics
def test_tirust_tensor_store_keeps_the_values_and_counts_its_reuse(li, loop_native):
    """``prepare_reduction(store=True)`` (``TirOptions::store``, m-lit
    lit-cache STEP 3): the families of g g > z z share TIRust's per-subset
    results within a point chunk; every lane is bitwise the store-off value
    and the store's reuse is reported; NinjaRS refuses the keyword."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor

    if not loop_native.tirust_available():
        pytest.skip("the staged native extension is built without TIRust")
    name = "li_gg_zz"
    executor = NativeExecutor(li.slice(name), forward=False, inputs="native")
    native = executor.native("double", None, 1)
    with pytest.raises(Exception, match="TIRust's settings"):
        native.prepare_reduction("ninjars", store=True)
    native.prepare_reduction("tirust")
    plain = _native_run(executor, li, name, reduction="tirust")
    native.prepare_reduction("tirust", store=True)
    stored = _native_run(executor, li, name, reduction="tirust")
    assert plain[0] == stored[0] and plain[1] == stored[1] == [[]]
    assert not any(key.startswith("TIR store") for key in plain[2]["counts"])
    counts = stored[2]["counts"]
    assert counts["TIR store masters reused"] > 0
    assert counts["TIR store masters computed"] > 0
    # TIRust's own count of the master rows its calls request is the same
    # (the store serves some of them).
    evaluated = "TIR master requests"
    assert counts[evaluated] == plain[2]["counts"][evaluated] > 0
    # The masters census counts the store's misses: refused without it.
    with pytest.raises(Exception, match="masters_census"):
        native.prepare_reduction("tirust", masters_census=True)
    native.prepare_reduction("tirust", store=True, masters_census=True)
    census = _native_run(executor, li, name, reduction="tirust")
    assert census[0] == plain[0] and census[1] == [[]]
    families = ("tadpole", "bubble", "triangle", "box")
    for count in ("store misses", "canonical repeats"):
        assert census[2]["counts"][f"TIR masters census: {count}"] == sum(
            census[2]["counts"][f"TIR masters census: {family} {count}"]
            for family in families
        )


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("agree", 0),
        ("tirust-refused", 2),
        ("tirust-unserved", 2),
        ("tirust-failed-closed", 2),
        ("bitwise-equal", 2),
        ("ninjars-refused", 1),
        ("disagree", 1),
    ],
)
def test_the_backends_check_exit_codes(case, code, tmp_path, monkeypatch):
    """The exit-code contract of the L6 both-backends check
    (``tools/loop_v2_observables_backends.py``, which lit-tir-mp and lit-val
    re-run), on a stubbed runtime: 0 when every compared output passes; 2
    when the TIRust arm was not exercised (a TIRust refusal, a fixture point
    NinjaRS serves that TIRust does not, points failed closed, or arms
    bitwise equal); 1 when an output fails or the NinjaRS arm does not run."""

    import json
    import sys
    from decimal import Decimal

    import pyamplicol
    from pyamplicol.api.errors import ReductionUnavailable
    from tools import loop_v2_observables_backends as tool

    def entry(value, error):
        return [[("finite", (Decimal(value), Decimal(0)), error)]]

    def served(name, runtime, backend, carrier):
        if backend == "ninjars" and case == "ninjars-refused":
            raise ReductionUnavailable("no NinjaRS")
        if backend == "tirust" and case == "tirust-refused":
            raise ReductionUnavailable("no TIRust")
        unserved = backend == "tirust" and case == "tirust-unserved"
        return {"p0": "success", "p1": "unstable" if unserved else "success"}

    def entries(name, runtime, backend, stability, carrier, failed, observables):
        if backend == "ninjars":
            return {"virtual/p0": entry("1.0", 1e-3), "virtual/p1": entry("2.0", 1e-3)}
        value = {"disagree": "1.5", "bitwise-equal": "1.0"}.get(case, "1.0001")
        out = {"virtual/p0": entry(value, 1e-3)}
        if case == "tirust-failed-closed":
            failed["virtual/p1"] = "TIRust Unstable: ill-conditioned Gram matrix"
        else:
            out["virtual/p1"] = entry(
                "2.0" if case == "bitwise-equal" else "2.0001", 1e-3
            )
        return out

    class Runtime:
        @staticmethod
        def load(path, process):
            return object()

    monkeypatch.setattr(
        tool, "_artifact", lambda work, name, expression, observables: work
    )
    monkeypatch.setattr(tool, "_served", served)
    monkeypatch.setattr(tool, "_entries", entries)
    monkeypatch.setattr(tool, "checks", lambda: None)
    monkeypatch.setattr(pyamplicol, "Runtime", Runtime)
    output = tmp_path / "record.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool",
            "--work-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--set",
            "lit-amp",
            "--rows",
            "li_gg_hh",
        ],
    )
    assert tool.main() == code
    record = json.loads(output.read_text())
    assert ("not_exercised" in record) == (code == 2)
    if case == "disagree":
        assert record["passed"] is False
    if case == "agree":
        assert record["passed"] is True and len(record["rows"]["li_gg_hh"]) == 2
