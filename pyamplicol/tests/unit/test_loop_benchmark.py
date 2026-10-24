# SPDX-License-Identifier: 0BSD
"""The gate-G1 benchmark harness (``tools/loop_benchmark``) and its native
instrumentation.

* the frozen streams are exact on-shell, momentum-conserving and distinct,
  start with the campaign's points where a committed reference exists, and
  their decimal text for the providers is the binary64 rounding of the
  exact points;
* the speed gate: strictly faster than MadLoop and at most 1.30x each
  Recola/OpenLoops time; no matched provider means no verdict;
* the cost model's G1 recalibration on a synthetic document;
* the native report counts the modelled dressing work and times the master
  integrals inside the reduction (one 2 -> 2 evaluation); the work is counted
  per lane chunk in schedules B and M, M's equals B's for a quark line and
  stays at or below it for gluons;
* the particle names of the slice request come from the model's PDG codes,
  and the gate statistic has its uncertainty band (pass, fail, inconclusive).
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loop_benchmark import gate, streams  # noqa: E402

pytestmark = pytest.mark.loop_fast


@pytest.mark.parametrize("key", streams.keys())
def test_streams_are_exact_distinct_and_start_with_the_campaign(key):
    stream = streams.load(key)
    exact_points, required_spec = streams._toolkit()
    masses = [Fraction(m) for m in required_spec.masses(streams.spec(key))]
    points = [streams.exact(stream, index) for index in range(len(stream["points"]))]
    assert len(points) == streams.STREAM_LENGTH
    assert len({tuple(x for p in point for x in p) for point in points}) == len(points)
    for point in points:
        exact_points.check(point, masses, 2)
    assert streams.campaign_agreement(key, stream) in (True, None)
    text = streams.binary64(stream, 3)
    for exact, decimal in zip(stream["points"][:3], text, strict=True):
        for leg, values in zip(exact, decimal, strict=True):
            assert [float(Fraction(x)) for x in leg] == [float(x) for x in values]


def test_gate_verdict():
    times = {"madloop": 1.0, "openloops": 0.5, "recola": 0.8}
    passed = gate.gate_verdict(0.65, times)
    assert passed["status"] == "pass"
    assert passed["fastest_provider"] == "openloops"
    assert passed["binding_limit"] == pytest.approx(0.65)
    assert gate.gate_verdict(0.66, times)["status"] == "fail"
    # Strictly faster than MadLoop; equal to 1.30x OpenLoops passes.
    assert gate.gate_verdict(1.0, {"madloop": 1.0})["status"] == "fail"
    assert gate.gate_verdict(1.3, {"openloops": 1.0})["status"] == "pass"
    assert gate.gate_verdict(1.0, {}) == {"status": "no matched provider timing"}


def test_gate_band_decides_and_inconclusive_straddles_the_limit():
    times = {"openloops": 1.0}
    reference = {"openloops": (0.02, False)}
    # 1.20 / 1.30 = 0.923; a 5 % + 2 % band stays below 1: pass.
    verdict = gate.gate_verdict(
        1.20, times, uncertainty=0.05, reference_uncertainty=reference
    )
    assert verdict["status"] == "pass"
    assert verdict["band_to_binding_limit"][1] < 1
    # 1.29 / 1.30 = 0.992: the band straddles the limit.
    verdict = gate.gate_verdict(
        1.29, times, uncertainty=0.05, reference_uncertainty=reference
    )
    assert verdict["status"] == "inconclusive"
    assert verdict["central"] == "pass"
    assert (
        gate.gate_verdict(
            1.50, times, uncertainty=0.05, reference_uncertainty=reference
        )["status"]
        == "fail"
    )
    # Few blocks: half the range; many: the median's standard error.
    assert gate.median_uncertainty([1.0, 1.2, 1.1]) == pytest.approx(0.1 / 1.1)
    assert gate.median_uncertainty([1.0] * 9) == 0.0
    assert gate.half_range([2.0]) is None


def test_process_text_from_the_model_pdg_codes():
    from tools.loop_benchmark import pyamplicol_v2

    class Record:
        def __init__(self, name, antiname, pdg, ghost=0, auxiliary=None):
            self.name, self.antiname, self.pdg_code = name, antiname, pdg
            self.ghost_number, self.auxiliary_kind = ghost, auxiliary

    class IR:
        particles = (
            Record("gh", "gh~", 82, ghost=1),
            Record("u", "u~", 2),
            Record("e-", "e+", 11),
            Record("g", "g", 21),
        )

    class Compiled:
        ir = IR()

    text = pyamplicol_v2.process_text("uux_emepgg", Compiled())
    assert text == "u u~ > e- e+ g g"
    with pytest.raises(KeyError):
        pyamplicol_v2.process_text("gg_ttxgg", Compiled())
    assert gate.label("uux_emepg") == "u u~ > e- e+ g"


def test_g1_recalibration_sets_measurements_beside_the_recipe():
    from tools import loop_v2_costmodel as model

    batch = {
        "median_seconds_per_point": 0.5,
        "stages_per_point": {
            "dressing": 0.2,
            "factors": 0.05,
            "reduction": 0.1,
            "reduction: masters": 0.02,
            "seeds": 0.01,
            "scatter": 0.04,
        },
        "throughput": {"kernel_cmac_per_second": 1.3e9},
    }
    document = {
        "processes": [
            {
                "key": "gg_gggg",
                "pyamplicol": {
                    "native/full": {"accuracy": "full", "batches": {"128": batch}}
                },
            }
        ]
    }
    (row,) = model.g1_recalibration(document)
    assert row["batch"] == 128
    assert row["dressing"] == pytest.approx(0.25)
    assert row["native_loop_rest"] == pytest.approx(0.05)
    assert row["outside_native_loop"] == pytest.approx(0.1)
    assert row["outer_factor"] == pytest.approx(0.5 / 0.35)
    low, high = row["design"]["recipe_finite_only"]
    assert 0 < low < high
    assert row["design"]["g0_dressing"] == pytest.approx(0.171)


def test_native_report_counts_dressing_work_and_master_time(
    loop_sm_compiled_model, loop_native
):
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "u u~ > d d~",
        {"QCD": 2, "QED": 0},
        {"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    executor = NativeExecutor(slice_, forward=False, threads=1)
    point = [
        ["500", "0", "0", "500"],
        ["500", "0", "0", "-500"],
        ["500", "300", "0", "400"],
        ["500", "-300", "0", "-400"],
    ]
    executor.evaluate(
        [point, point], ["1000000"] * 2, parameters={"aS": "0.118"}, poles="none"
    )
    report = executor.last_report
    work = report["counts"]["dressing work (modelled complex MAC per lane)"]
    assert work > 0
    assert report["counts"]["master requests"] >= report["counts"]["master evaluations"]
    assert report["counts"]["master evaluations"] > 0
    seconds = report["seconds"]
    masters = seconds["reduction: masters"]
    assert 0 < seconds["reduction: masters (oneloopmaster)"] <= masters
    assert masters <= seconds["reduction"]


@pytest.mark.loop_physics
def test_native_dressing_work_is_per_chunk_and_schedule_m_counts_its_states(
    loop_sm_compiled_model, loop_native
):
    """The modelled dressing work is counted once per lane chunk (one 16-lane
    chunk for 1 and for 16 points, two for 17) in schedule B and in schedule M,
    whose count adds its merged states to the B families it leaves. For a
    quark line M's entries are B's leaves, so its work equals B's exactly; in
    a pure-gluon process M merges seed entries and dresses no more than B
    (DESIGN §5.6: M/B work 1.00 with a quark line, 0.88-0.97 for gluons)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    work = {}
    for text, born, key in (
        ("u u~ > d d~ g", 3, "uux_ddxg"),
        ("g g > g g", 2, "gg_gg"),
    ):
        request = SliceRequest(
            text,
            {"QCD": born, "QED": 0},
            {"QCD": born + 2, "QED": 0},
            fixed_parameters={"MB": "0"},
        )
        slice_ = LoopSlice.generate(compiled, request, loop_native)
        executor = NativeExecutor(slice_, forward=False, threads=1, merged=True)
        stream = streams.load(key)
        points = [streams.exact(stream, i) for i in range(17)]
        scales = [str(Fraction(stream["mu_r"][i]) ** 2) for i in range(17)]
        for schedule in ("B", "M"):
            counts = {}
            for size in (1, 16, 17):
                executor.evaluate(
                    points[:size],
                    scales[:size],
                    parameters={"aS": "0.118"},
                    poles="none",
                    schedule=schedule,
                )
                report = executor.last_report
                assert report["lanes"] == 16
                counts[size] = report["counts"][
                    "dressing work (modelled complex MAC per lane)"
                ]
            assert counts[1] == counts[16] > 0
            assert counts[17] == 2 * counts[1]
            work[key, schedule] = counts[1]
    assert work["uux_ddxg", "M"] == work["uux_ddxg", "B"]
    assert 0.85 * work["gg_gg", "B"] <= work["gg_gg", "M"] <= work["gg_gg", "B"]
