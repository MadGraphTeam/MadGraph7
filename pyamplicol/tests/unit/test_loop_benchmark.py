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
  and the gate statistic has its uncertainty band (pass, fail, inconclusive);
* a matched provider without a production-mode timing (HANDOFF §0.1) makes a
  passing verdict "partial", and an imported campaign timing outside the
  protocol stays beside the verdict; imported timings must be the stream's
  points;
* the Ward rows are divided by the physical scale of their point and judged
  with the O10 test's binary64 bound;
* below 5 ms per point the gate does not block (the user's rule of
  2026-10-24): the verdict says so and keeps the checks;
* provider files are resolved relative to the reference root only: the
  absolute paths stored in the records (an old root) are never touched;
* the universal double pole of the physical rows at full and leading colour,
  and the top-free g g > g g g g g row's NJet reference at the stream's
  first points.
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


def test_gate_partial_without_a_matched_provider_and_outside_the_protocol():
    times = {"madloop": 100.0}
    missing = {"recola": "no production-mode timing (HANDOFF §0.1)"}
    outside = {"recola": {"seconds": 20.0, "mode": "compute_ir_poles 1"}}
    verdict = gate.gate_verdict(5.0, times, missing=missing, outside=outside)
    assert verdict["status"] == "partial"
    assert verdict["missing"] == missing
    assert verdict["outside_protocol"]["recola"]["ratio"] == pytest.approx(0.25)
    # The timing outside the protocol never decides: a failed check fails.
    assert (
        gate.gate_verdict(150.0, times, missing=missing, outside=outside)["status"]
        == "fail"
    )
    assert gate.gate_verdict(1.0, {}, missing=missing)["status"] == (
        "no matched provider timing"
    )


def test_missing_providers_follow_the_specification():
    # A provider without a library is not matched; one beyond HANDOFF §0.1 is.
    assert gate.missing_providers("uux_emepg", None) == {}
    assert set(gate.missing_providers("gg_ggggg", None)) == {"madloop", "recola"}
    assert set(gate.missing_providers("uux_gggg", None)) == {"madloop"}
    record = {
        "providers": {
            "madloop": {"median_seconds_per_point": 500.0, "protocol_mode": True},
            "recola": {
                "median_seconds_per_point": 180.0,
                "protocol_mode": False,
                "mode": "compute_ir_poles 1",
            },
        }
    }
    assert set(gate.missing_providers("uux_ggggg", record)) == {"recola"}
    assert gate._times(record, "median_seconds_per_point") == {"madloop": 500.0}
    assert gate.outside_protocol(record) == {
        "recola": {"seconds": 180.0, "mode": "compute_ir_poles 1"}
    }


def test_imported_campaign_timings_are_the_stream_points(tmp_path):
    import json

    from tools.loop_benchmark import providers

    stream = streams.load("uux_ggggg")
    run = tmp_path / "runs" / "required" / "uux_ggggg"
    run.mkdir(parents=True)
    points = {"points": stream["points"][:3], "mu_r": stream["mu_r"][:3]}
    (run / "points.json").write_text(json.dumps(points))
    rows = [
        {"born": 1.0, "finite": 2.0, "seconds": t, "raw": {"finite": 1.5}}
        for t in (3.0, 1.0, 2.0)
    ]
    (run / "recola-result.json").write_text(
        json.dumps(
            {
                "provider": "recola",
                "process_name": "uux_ggggg",
                "cpu": 7,
                "settings": {"compute_ir_poles": 1, "collier_cache": "off"},
                "generation_seconds": 10.0,
                "rows": rows,
            }
        )
    )

    class Options:
        reference_root = tmp_path
        mg5 = recola = openloops = None

    trees = providers.Trees(Options())
    result = providers.imported(streams.spec("uux_ggggg"), "recola", trees)
    assert result["protocol_mode"] is False
    assert result["median_seconds_per_point"] == 2.0
    assert result["cpu"] == 7
    assert result["setup"] == {"generation_seconds": 10.0}
    points["points"] = points["points"][::-1]
    (run / "points.json").write_text(json.dumps(points))
    with pytest.raises(RuntimeError):
        providers.imported(streams.spec("uux_ggggg"), "recola", trees)


def test_ward_rows_against_the_physical_scale():
    physical = [
        {"index": 0, "components": {"virtual": [2.0, 4.0, 1.0], "r2": [8.0, 0, 0]}}
    ]
    small = {"virtual": [1e-12, 0.0, 0.0], "r2": [0.0, 0.0, 0.0]}
    rows = gate.ward_rows(
        [
            {"index": 0, "components": small, "lanes": [10.0, 0, 0]},
            {"index": 0, "components": {"virtual": [1e-9, 0, 0]}, "lanes": [0, 0, 0]},
            {"index": 1, "components": small},
        ],
        physical,
    )
    # The scale is the largest physical component per coefficient.
    assert rows[0]["scale"] == [8.0, 4.0, 1.0]
    assert rows[0]["bound"][0] == pytest.approx(1e-11 * 8.0 + 1e-12 * 10.0)
    assert rows[0]["passed"] is True
    assert rows[0]["ratios"]["virtual"][0] == pytest.approx(1e-12 / 8.0)
    assert rows[1]["passed"] is False
    assert rows[2]["scale"] is None and "passed" not in rows[2]


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


def test_gate_floor_below_five_milliseconds():
    times = {"madloop": 0.004, "openloops": 0.001}
    verdict = gate.gate_verdict(0.003, times, uncertainty=0.01)
    assert verdict["status"] == gate.GATE_FLOOR_STATUS
    assert verdict["status"] == "below 5 ms: not gated (user 2026-10-24)"
    # The checks and ratios stay; what they alone say is kept beside.
    assert verdict["measured_status"] == "fail"
    assert verdict["ratio_to_binding_limit"] == pytest.approx(0.003 / 0.0013)
    assert set(verdict["checks"]) == {"madloop", "openloops"}
    assert gate.gate_verdict(0.003, {})["measured_status"] == (
        "no matched provider timing"
    )
    # At 5 ms and above the gate applies.
    assert gate.gate_verdict(0.005, {"openloops": 0.001})["status"] == "fail"
    assert gate.gate_verdict(0.006, {"openloops": 0.005})["status"] == "pass"
    # The batched time decides (floor_basis): a B = 1 time above 5 ms of a
    # configuration batched below it is not gated either, and conversely.
    one = gate.gate_verdict(0.009, times, floor_basis=0.003)
    assert one["status"] == gate.GATE_FLOOR_STATUS
    assert one["floor_basis_seconds"] == 0.003
    assert gate.gate_verdict(0.004, times, floor_basis=0.006)["status"] == "fail"


def test_floor_basis_is_the_largest_batch_and_the_default_configuration():
    def config(inputs, accuracy, batches):
        return {
            "inputs": inputs,
            "accuracy": accuracy,
            "batches": {
                str(b): {"median_seconds_per_point": t} for b, t in batches.items()
            },
        }

    bases = gate.floor_bases(
        {
            "native/full": config("native", "full", {1: 0.009, 32: 0.003, 128: 0.0027}),
            "native/full/L1": config("native", "full", {1: 0.004}),
            "python/full": config("python", "full", {1: 0.1}),
        }
    )
    assert bases == {
        "native/full": 0.0027,
        "native/full/L1": 0.0027,
        "python/full": None,
    }


OLD_ROOT = "/nonexistent-old-root/workspace/reference"


@pytest.fixture
def no_old_root(monkeypatch):
    """Fail on any filesystem query of a path under ``OLD_ROOT``."""

    import builtins

    touched = []
    for name in ("exists", "stat", "resolve", "read_text", "open", "is_file"):
        original = getattr(Path, name)

        def guarded(self, *args, _original=original, **kwargs):
            if str(self).startswith("/nonexistent-old-root"):
                touched.append(str(self))
                raise AssertionError(f"old root touched: {self}")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, name, guarded)
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if str(file).startswith("/nonexistent-old-root"):
            touched.append(str(file))
            raise AssertionError(f"old root touched: {file}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    return touched


def test_madloop_exports_resolve_under_the_reference_root_only(tmp_path, no_old_root):
    import json

    export = tmp_path / "runs" / "required" / "uux_ddxg" / "madloop-export"
    export.mkdir(parents=True)
    times = {"graph_generation_s": 1.0, "source_export_s": 2.0, "compile_s": 3.0}
    (export.parent / "madloop-export.generation.json").write_text(json.dumps(times))
    recorded = f"{OLD_ROOT}/runs/required/uux_ddxg/madloop-export"
    assert gate._export_of("uux_ddxg", recorded, tmp_path) == (
        export,
        "runs/required/uux_ddxg/madloop-export",
    )
    assert gate._madloop_generation(export)["total_s"] == 6.0
    # An export outside a reference root's runs (a scratch copy) that moved:
    # the specification's madloop.export.
    moved = export.parents[2] / "g1" / "gg_ggg" / "madloop-export"
    moved.mkdir(parents=True)
    scratch = "/nonexistent-old-root/workspace/tmp/measure-g1/gg_ggg/madloop-export"
    assert gate._export_of("gg_ggg", scratch, tmp_path) == (
        moved,
        "runs/g1/gg_ggg/madloop-export",
    )
    # Without a root, or with no export under it, only the name stays.
    assert gate._export_of("uux_ddxg", recorded, None) == (
        None,
        "runs/required/uux_ddxg/madloop-export",
    )
    assert gate._export_of("uux_uuxg", recorded.replace("ddxg", "uuxg"), tmp_path) == (
        None,
        "runs/required/uux_uuxg/madloop-export",
    )
    assert gate._madloop_generation(None) is None
    assert no_old_root == []


def _pyamplicol_record(key: str, seconds: list[float], **extra) -> dict:
    summary = {
        "blocks": len(seconds),
        "points_per_block": 128,
        "median_seconds_per_point": sorted(seconds)[len(seconds) // 2],
        "min_seconds_per_point": min(seconds),
        "max_seconds_per_point": max(seconds),
        "samples_seconds_per_point": seconds,
        "stages_per_point": {"dressing": seconds[0] / 2},
        "counts_per_block": {},
        "reals_per_lane": {},
        "throughput": None,
        "values": {"0": [1.0, 0.5], "1": [2.0, 1.0]},
    }
    return {
        "kind": "pyamplicol-v2",
        "process": key,
        "accuracy": "full",
        "inputs": "native",
        "lanes": 16,
        "schedule": "B",
        "revision": "0" * 40,
        "branch": "work/test",
        "dirty": False,
        "batches": {"128": summary},
        **extra,
    }


def test_gate_document_applies_the_floor_and_reads_no_old_root(tmp_path, no_old_root):
    """``gate.build`` on records whose MadLoop export points into an old
    root: the export is found under the reference root, the old root is
    never touched, and the 5 ms floor sets the verdict of a fast row while
    a slow row is gated."""

    import json

    export = tmp_path / "ref" / "runs" / "required" / "uux_ddxg" / "madloop-export"
    export.mkdir(parents=True)
    (export.parent / "madloop-export.generation.json").write_text(
        json.dumps({"graph_generation_s": 1, "source_export_s": 2, "compile_s": 3})
    )
    rows = [{"born": 1.0, "finite": 0.5, "seconds": 0.002}] * 2
    providers = {
        "kind": "providers",
        "process": "uux_ddxg",
        "cpu": 130,
        "points": 2,
        "providers": {
            "madloop": {
                "points": 2,
                "mean_seconds_per_point": 0.003,
                "median_seconds_per_point": 0.003,
                "rows": rows,
                "export": f"{OLD_ROOT}/runs/required/uux_ddxg/madloop-export",
            },
            "openloops": {
                "points": 2,
                "mean_seconds_per_point": 0.0015,
                "median_seconds_per_point": 0.0015,
                "rows": rows,
            },
        },
    }
    records = [
        providers,
        {**providers, "process": "gg_ggg"},
        _pyamplicol_record("uux_ddxg", [0.0026, 0.0027, 0.0028]),
        _pyamplicol_record("gg_ggg", [0.0060, 0.0061, 0.0062]),
    ]
    records[1]["providers"] = {
        name: {k: v for k, v in result.items() if k != "export"}
        for name, result in providers["providers"].items()
    }
    paths = []
    for index, record in enumerate(records):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    document, _ = gate.build(paths, tmp_path / "ref")
    rows = {row["key"]: row for row in document["processes"]}
    fast = rows["uux_ddxg"]["pyamplicol"]["native/full"]["batches"]["128"]["gate"]
    assert fast["status"] == gate.GATE_FLOOR_STATUS
    assert fast["measured_status"] == "fail"
    madloop = rows["uux_ddxg"]["providers"]["madloop"]
    assert madloop["export"] == "runs/required/uux_ddxg/madloop-export"
    assert madloop["generation"]["total_s"] == 6
    slow = rows["gg_ggg"]["pyamplicol"]["native/full"]["batches"]["128"]["gate"]
    assert slow["status"] == "fail" and "measured_status" not in slow
    assert no_old_root == []


def test_a_repeat_at_another_batch_keeps_the_earlier_checks(tmp_path):
    """The seven-gluon leading-colour row: a B = 64 run without physical rows
    after a B = 16 run with them decides the timing, keeps the B = 16 times
    as an earlier run and the physical rows with their revision."""

    import json

    first = _pyamplicol_record(
        "gg_ggggg", [15.7, 15.9, 16.2], accuracy="lc", physical=[{"index": 0}]
    )
    first["batches"] = {"16": first["batches"]["128"]}
    second = _pyamplicol_record("gg_ggggg", [15.5, 15.6, 15.8], accuracy="lc")
    second["batches"] = {"64": second["batches"]["128"]}
    second["revision"] = "1" * 40
    paths = []
    for index, record in enumerate((first, second)):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    document, _ = gate.build(paths, None)
    (row,) = document["processes"]
    config = row["pyamplicol"]["native/lc"]
    assert set(config["batches"]) == {"64"} and config["revision"] == "1" * 40
    assert set(config["earlier_runs"][0]["batches"]) == {"16"}
    assert config["physical"] == [{"index": 0}]
    assert config["checks_from_earlier_run"] == {"physical": "0" * 40}
    assert "partial" not in config
    # A checkpoint of a run that did not finish is marked.
    second["partial"] = True
    paths[1].write_text(json.dumps(second))
    document, _ = gate.build(paths, None)
    assert document["processes"][0]["pyamplicol"]["native/lc"]["partial"] is True


def test_universal_double_pole_at_full_and_leading_colour():
    import math

    from tools.loop_benchmark import pyamplicol_v2

    alpha = 0.118 / (2 * math.pi)
    # u u~ > d d~ g: four quarks and a gluon; leading colour C_F -> N_c/2.
    assert pyamplicol_v2.universal_double_pole("uux_ddxg", "full") == (
        pytest.approx(-alpha * (4 * 4 / 3 + 3))
    )
    assert pyamplicol_v2.universal_double_pole("uux_ddxg", "lc") == pytest.approx(
        -alpha * (4 * 3 / 2 + 3)
    )
    # Massive tops add no double pole; the leptons none.
    assert pyamplicol_v2.universal_double_pole("gg_ttxemepg_qcd", "full") == (
        pytest.approx(-alpha * 9)
    )
    assert pyamplicol_v2.universal_double_pole("gg_ggggg", "nlc") is None


def test_top_free_seven_gluon_row_is_checked_against_njet():
    import json

    spec = streams.spec("gg_ggggg")
    assert spec["particle_veto"] == [6]
    name = streams.fixture_name("gg_ggggg")
    assert name == "gg_ggggg_njet" and streams.fixture_name("uux_ddxg") == "uux_ddxg"
    fixture = json.loads((streams.FIXTURES / f"{name}.json").read_text())
    assert fixture["scheme"] == "massless-qcd-nf5"
    stream = streams.load("gg_ggggg")
    # NJet's first points are the stream's (at NJet's own scale).
    assert fixture["points"][:3] == stream["points"][:3]
    # Its Catani entry states the Born and the poles, no finite part: only the
    # quantities an entry states are compared.
    from tools.loop_benchmark import pyamplicol_v2

    row = fixture["rows"][0]["providers"]
    primary = row[fixture["primary_provider"]]
    errors = pyamplicol_v2.reference_errors(
        float(primary["born"]), float(primary["finite"]), row, 0.118
    )
    assert "finite" not in row["catani"] and set(errors["catani"]) == {"born"}
    assert set(errors[fixture["primary_provider"]]) == {"born", "finite"}
    assert max(errors[fixture["primary_provider"]].values()) == 0.0
