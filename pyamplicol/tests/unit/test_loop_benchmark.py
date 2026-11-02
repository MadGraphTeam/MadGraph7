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
* the gate is taken on the available providers: a matched provider without
  a production-mode timing (HANDOFF §0.1) is listed beside the verdict, and
  an imported campaign timing outside the protocol stays beside it;
  imported timings must be the stream's points;
* a repeat of a configuration keeps the earlier run's checks it lacks, and a
  checkpoint record is marked partial;
* the timing runs on a frozen setup heap: the generation's objects are out
  of the cyclic collector's scans, and the record states their count;
* the Ward rows are divided by the physical scale of their point and judged
  with the O10 test's binary64 bound;
* below 5 ms per point the gate does not block (the user's rule of
  2026-10-24): the verdict says so and keeps the checks;
* provider files are resolved relative to the reference root only: the
  absolute paths stored in the records (an old root) are never touched; an
  imported MadLoop timing takes its generation from the specification's
  export, an imported Recola run keeps its generation, and reference runs
  that did not finish are reported as such;
* the universal double pole of the physical rows at full and leading colour
  (one source each: the toolkit's at full colour, the generator's
  leading-colour table), and the top-free g g > g g g g g row's NJet
  reference at the stream's first points; stability rows at the reference
  fixture's own points (``stability --fixture``) carry their deviations from
  its references, stay apart from the stream's rows in the gate and give the
  report its estimate at the fixture's worst point;
* a stored process (``--store``) loads as the public runtime loads an
  artifact, regenerates nothing and times bitwise the generating run's
  values; a load of another configuration or native build is refused, and
  a generate-only record is no gate input;
* the report builder labels the gate tool's 5 ms floor with the registry's
  threshold rule and stops on a disagreement, and gives a row without a
  gate its rotation and Ward checks (addendum H);
* a user acceptance of a check's points above its tolerance marks exactly
  those points on the validation cell (the check and its tolerance stay)
  and stops the build when its claims do not hold;
* the accuracy check asserts binary64 against the providers only where the
  finite part's O1 estimate E <= 1e-9, a double-double evaluation of the
  point above it (the user's ruling of 2026-10-29 on E1), recording E and
  the binary64 errors.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loop_benchmark import gate, providers, streams  # noqa: E402

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


def test_gate_on_the_available_providers_with_the_missing_ones_beside():
    times = {"madloop": 100.0}
    missing = {"recola": "no production-mode timing (HANDOFF §0.1)"}
    outside = {"recola": {"seconds": 20.0, "mode": "compute_ir_poles 1"}}
    verdict = gate.gate_verdict(5.0, times, missing=missing, outside=outside)
    # Addendum D: each available provider; the missing ones are listed.
    assert verdict["status"] == "pass" and set(verdict["checks"]) == {"madloop"}
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
    # The import's upper-bound reason travels with the timing (never a limit).
    record["providers"]["recola"]["production_upper_bound"] = "poles on top"
    assert gate.outside_protocol(record)["recola"]["production_upper_bound"] == (
        "poles on top"
    )
    assert gate._times(record, "median_seconds_per_point") == {"madloop": 500.0}


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
                "version": "Recola2 (2.3.0 tree) / SM model 2.2.3 / Collier 1.2.8",
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
    # IR poles on top of the production work, without the Collier cache that
    # production keeps on (COLLIER 1.2.8 has no seven-leg defect): the time
    # bounds the production time from above.
    assert "compute_ir_poles 1" in result["production_upper_bound"]
    assert "without the Collier cache" in result["production_upper_bound"]
    assert "the protocol's: on" in result["mode"]
    record = json.loads((run / "recola-result.json").read_text())
    # (poles, cache, COLLIER release) -> protocol mode, upper bound
    cases = {
        (1, "on", "1.2.8"): (False, True),  # the protocol's cache, poles on top
        (1, "on", "1.2.9"): (False, False),  # a cache the defect forbids
        (1, "off", "1.2.9"): (False, True),  # the protocol's cache (off)
        (1, "off", None): (False, True),  # no cache: at least production work
        (0, "on", "1.2.8"): (True, False),  # the protocol itself
        (0, "off", "1.2.9"): (True, False),
        (0, "off", "1.2.8"): (False, False),  # no poles margin: no bound
    }
    for (poles, cache, collier), (protocol, bound) in cases.items():
        record["settings"] = {"compute_ir_poles": poles, "collier_cache": cache}
        record["version"] = f"Recola2 / Collier {collier}" if collier else "Recola2"
        (run / "recola-result.json").write_text(json.dumps(record))
        other = providers.imported(streams.spec("uux_ggggg"), "recola", trees)
        assert other["protocol_mode"] is protocol, (poles, cache, collier)
        assert ("production_upper_bound" in other) is bound, (poles, cache, collier)
    points["points"] = points["points"][::-1]
    (run / "points.json").write_text(json.dumps(points))
    with pytest.raises(RuntimeError):
        providers.imported(streams.spec("uux_ggggg"), "recola", trees)


def test_collier_cache_is_production_unless_the_release_has_the_defect():
    """ADDENDUM D: caches as in production; the global cache is off only at
    seven legs or more with a COLLIER release of the documented N = 7
    defect (1.2.9, 1.2.10), and a timing run refuses an unknown release
    there."""

    from tools.loop_benchmark import providers

    assert providers.collier_cache(6, None)["on"] is True
    assert providers.collier_cache(6, "1.2.9")["on"] is True
    assert providers.collier_cache(7, "1.2.8")["on"] is True
    assert providers.collier_cache(8, "1.2.8")["on"] is True
    for release in ("1.2.9", "1.2.10"):
        decision = providers.collier_cache(7, release)
        assert decision["on"] is False and release in decision["reason"]
    assert providers.collier_cache(7, None)["on"] is None
    assert providers._collier_version("Recola2 / libcollier.so 1.2.8 / x 2") == "1.2.8"
    assert providers._collier_version("SM model 2.2.3 / Collier 1.2.10") == "1.2.10"
    assert providers._collier_version("MadLoop/MG5_aMC 3.7.0") is None


def test_files_of_a_moved_root_are_taken_under_the_current_root(tmp_path):
    """A provider tree built before its root moved names files of the old
    root; the harness takes them from the same place under the current root
    and never looks at the original (no stat outside the root)."""

    from tools.loop_benchmark import providers

    root = tmp_path / "reference"
    collier = root / "sources" / "COLLIER-1.2.8"
    collier.mkdir(parents=True)
    (collier / "libcollier.a").write_text("")
    old = "/no/such/old-root/workspace/reference/sources/COLLIER-1.2.8"
    assert providers.relocated(old, root) == collier
    assert providers.relocated(old + "/libcollier.a", root) == collier / "libcollier.a"
    assert providers.relocated(str(collier), root) == collier
    store = "/nix/store/abc-glibc/lib"
    assert providers.relocated(store, root) == Path(store)
    with pytest.raises(RuntimeError, match="outside the reference root"):
        providers.relocated("/no/such/old-root/sources/NinjaRS/lib", root)
    # A MadLoop export's static reducers (LOOP_LIBS) with an old-root path.
    export = root / "runs" / "required" / "p" / "export"
    (export / "SubProcesses").mkdir(parents=True)
    (export / "lib").mkdir()
    (export / "lib" / "libcts.a").write_text("")
    (export / "SubProcesses" / "MadLoop_makefile_definitions").write_text(
        f"LINK_LOOP_LIBS = -L$(LIBDIR) -lcts\n"
        f"LOOP_LIBS = $(LIBDIR)libcts.$(libext) {old}/libcollier.$(libext)\n"
    )
    assert providers._madloop_libraries(export, root) == [
        export / "lib" / "libcts.a",
        collier / "libcollier.a",
    ]


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
    # The B = 128 time decides (floor_basis, ``floor_bases``), and conversely.
    one = gate.gate_verdict(0.009, times, floor_basis=0.003)
    assert one["status"] == gate.GATE_FLOOR_STATUS
    assert one["floor_basis_seconds"] == 0.003
    assert gate.gate_verdict(0.004, times, floor_basis=0.006)["status"] == "fail"


def test_only_b128_is_gated():
    """The user's rule of 2026-10-28: any batch but B = 128 is reported with
    its checks, ratios and band, the checks' own verdict as measured_status,
    and neither the gate nor the 5 ms floor sets its status."""

    times = {"madloop": 0.004, "openloops": 0.001}
    assert gate.GATE_BATCH == 128
    at128 = gate.gate_verdict(0.009, times, uncertainty=0.01)
    assert at128 == gate.gate_verdict(0.009, times, uncertainty=0.01, batch=128)
    assert at128["status"] == "fail"
    for batch in (1, 16, 256):
        out = gate.gate_verdict(0.009, times, uncertainty=0.01, batch=batch)
        assert out["status"] == gate.GATE_REPORTED_STATUS
        assert out["measured_status"] == "fail" and out["gate_batch"] == 128
        assert {
            k: v for k, v in out.items() if k not in ("measured_status", "gate_batch")
        } == {
            **at128,
            "status": gate.GATE_REPORTED_STATUS,
        }
    below = gate.gate_verdict(0.003, times, floor_basis=0.003, batch=1)
    assert below["status"] == gate.GATE_REPORTED_STATUS
    assert below["measured_status"] == "fail" and "floor_seconds" not in below
    assert gate.gate_verdict(0.003, {}, batch=16)["measured_status"] == (
        "no matched provider timing"
    )


def test_floor_basis_is_the_b128_time():
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
            "native/full": config(
                "native", "full", {1: 0.009, 32: 0.003, 128: 0.0027, 256: 0.002}
            ),
            "native/full/L1": config("native", "full", {1: 0.004}),
            "native/lc": config("native", "lc", {16: 0.004}),
            "python/full": config("python", "full", {1: 0.1}),
        }
    )
    # Only B = 128 decides: no other batch and no other configuration.
    assert bases == {
        "native/full": 0.0027,
        "native/full/L1": None,
        "native/lc": None,
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


def test_gate_document_takes_the_verdict_at_b128_only(tmp_path):
    """``gate.build``: a B = 1 fail beside a B = 128 pass is a passing
    configuration, a configuration without a B = 128 timing has no verdict
    (missing, with the reason), and the 5 ms floor is the B = 128 time's."""

    import json

    rows = [{"born": 1.0, "finite": 0.5, "seconds": 0.004}] * 2
    providers = {
        "kind": "providers",
        "process": "gg_ggg",
        "cpu": 130,
        "points": 2,
        "providers": {
            "openloops": {
                "points": 2,
                "mean_seconds_per_point": 0.004,
                "median_seconds_per_point": 0.004,
                "rows": rows,
            }
        },
    }

    def record(key, accuracy, batches, **extra):
        out = _pyamplicol_record(key, [1.0], accuracy=accuracy, **extra)
        summary = out["batches"]["128"]
        out["batches"] = {
            str(size): {
                **summary,
                "median_seconds_per_point": t,
                "min_seconds_per_point": t,
                "max_seconds_per_point": t,
                "samples_seconds_per_point": [t, t, t],
            }
            for size, t in batches.items()
        }
        return out

    records = [
        providers,
        {**providers, "process": "uux_ddxg"},
        # Limit 1.3 x 4 ms = 5.2 ms: B = 1 fails, B = 128 passes.
        record("gg_ggg", "full", {1: 0.009, 128: 0.005}),
        # One lane, B = 1 only: no verdict, whatever B = 1 says.
        record("gg_ggg", "full", {1: 0.003}, lanes=1),
        record("gg_ggg", "full", {16: 0.003}, schedule="M"),
        record("gg_ggg", "lc", {16: 0.001}),
        # B = 1 above 5 ms, B = 128 below: the floor is B = 128's.
        record("uux_ddxg", "full", {1: 0.009, 128: 0.0045}),
        record("uux_ddxg", "full", {1: 0.003, 128: 0.006}, schedule="M"),
    ]
    paths = []
    for index, data in enumerate(records):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(data))
        paths.append(path)
    document, _ = gate.build(paths, None)
    rows = {row["key"]: row["pyamplicol"] for row in document["processes"]}
    full = rows["gg_ggg"]["native/full"]
    assert full["gate"] == {"batch": 128, "status": "pass"}
    one = full["batches"]["1"]["gate"]
    assert one["status"] == gate.GATE_REPORTED_STATUS
    assert one["measured_status"] == "fail"
    assert one["ratio_to_binding_limit"] == pytest.approx(0.009 / 0.0052)
    assert full["batches"]["128"]["gate"]["status"] == "pass"
    for name, timed in (("native/full/L1", "1"), ("native/full/M", "16")):
        verdict = rows["gg_ggg"][name]["gate"]
        assert verdict["status"] == gate.GATE_MISSING_STATUS
        assert f"timed at B = {timed}" in verdict["reason"]
        assert "B = 128 only" in verdict["reason"]
        batch = rows["gg_ggg"][name]["batches"][timed]["gate"]
        assert batch["status"] == gate.GATE_REPORTED_STATUS
        assert batch["measured_status"] == "pass"
    lc = rows["gg_ggg"]["native/lc"]
    assert lc["gate"] == {"status": "no gate (colour approximation)"}
    assert lc["batches"]["16"]["gate"] == lc["gate"]
    floored = rows["uux_ddxg"]["native/full"]
    assert floored["gate"] == {"batch": 128, "status": gate.GATE_FLOOR_STATUS}
    assert floored["batches"]["128"]["gate"]["floor_basis_seconds"] == 0.0045
    assert floored["batches"]["1"]["gate"]["status"] == gate.GATE_REPORTED_STATUS
    gated = rows["uux_ddxg"]["native/full/M"]
    assert gated["gate"] == {"batch": 128, "status": "fail"}
    assert gated["batches"]["128"]["gate"]["status"] == "fail"
    assert "floor_seconds" not in gated["batches"]["1"]["gate"]
    assert gated["batches"]["1"]["gate"]["measured_status"] == "pass"


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
    # No provider timing: reported without a gate (addenda F and H).
    assert row["gated"] is False and row["gate_note"] == gate.NO_PROVIDER_NOTE
    # A checkpoint of a run that did not finish is marked.
    second["partial"] = True
    paths[1].write_text(json.dumps(second))
    document, _ = gate.build(paths, None)
    assert document["processes"][0]["pyamplicol"]["native/lc"]["partial"] is True


def test_the_timing_runs_on_a_frozen_setup_heap():
    import gc
    import types

    from tools.loop_benchmark import pyamplicol_v2

    run = pyamplicol_v2.Run.__new__(pyamplicol_v2.Run)
    run.options = types.SimpleNamespace(
        load=None, generate_only=False, physical_points=None, stability_points=None
    )
    run.record = {}
    order: list[str] = []
    seen: list[object] = []
    # A cycle, tracked by the collector, standing for the generated slice.
    setup: dict[str, list] = {"slice": []}
    setup["slice"].append(setup)

    def generate():
        order.append("generate")
        run.held = setup

    def timing():
        order.append("timing")
        seen.append(gc.get_freeze_count())
        seen.append(any(o is setup for o in gc.get_objects()))

    run.generate = generate
    run.accuracy = lambda: order.append("accuracy")
    run.timing = timing
    run.close = lambda: None
    try:
        record = run.run()
    finally:
        gc.unfreeze()
    assert order == ["generate", "accuracy", "timing"]
    # Frozen objects are in the permanent generation, which no collection
    # (and gc.get_objects) visits.
    assert seen[0] > 0 and seen[1] is False
    assert record["gc_frozen_objects"] == seen[0]


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
    # One source each: full colour is the providers' conversion (the
    # toolkit's double_pole_factor), and without quarks leading colour
    # equals it bitwise (C_A leads with N_c).
    for key in ("uux_ddxg", "gg_ggggg"):
        assert pyamplicol_v2.universal_double_pole(key, "full") == (
            providers.universal_double_pole(streams.spec(key))
        )
    assert pyamplicol_v2.universal_double_pole("gg_ggggg", "lc") == (
        pyamplicol_v2.universal_double_pole("gg_ggggg", "full")
    )


def test_universal_double_pole_refuses_a_non_qcd_correction(monkeypatch):
    from tools.loop_benchmark import pyamplicol_v2

    qcd = streams.spec("uux_ddxg")
    monkeypatch.setattr(streams, "spec", lambda key: {**qcd, "correction": "EW"})
    for accuracy in ("full", "lc", "nlc"):
        with pytest.raises(ValueError, match="QCD corrections only"):
            pyamplicol_v2.universal_double_pole("uux_ddxg", accuracy)


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


def test_stability_rows_at_the_reference_fixtures_points():
    """``stability --fixture``: the rows are the fixture's points at its own
    scales (NJet's ten seven-gluon points at mu_R = 1000 GeV, not the
    stream's scales), each with its finite part's deviation from every
    reference of the fixture that states one (not Catani's, which has none)."""

    import types

    from tools.loop_benchmark import pyamplicol_v2

    name, fixture = pyamplicol_v2.reference_fixture("gg_ggggg")
    assert name == "gg_ggggg_njet"
    seen = {}

    def evaluate(points, scales, **options):
        seen.update(points=points, scales=scales, options=options)
        out = []
        for index in (4, 8):
            ref = fixture["rows"][index]["providers"]["njet_dd_oneloop_R1"]
            finite = float(ref["finite"]) * (1 + 3e-8)
            out.append(
                types.SimpleNamespace(
                    born={"total": float(ref["born"])},
                    components={
                        "virtual": {"total": types.SimpleNamespace(finite=finite)}
                    },
                    stability=types.SimpleNamespace(
                        relative=1e-6, coefficients=[1e-6], checks={"rotation-0": 4e-7}
                    ),
                )
            )
        return out

    run = types.SimpleNamespace(
        key="gg_ggggg",
        options=types.SimpleNamespace(lanes=16),
        parameters={"aS": "0.118"},
        executor=types.SimpleNamespace(evaluate=evaluate),
    )
    rows = pyamplicol_v2.stability_rows(run, [4, 8], ["double"], (name, fixture))
    points, scales = pyamplicol_v2.fixture_batch(fixture, [4, 8])
    assert seen["points"] == points and seen["scales"] == scales
    assert scales == [str(Fraction(1000) ** 2)] * 2
    assert [r["index"] for r in rows] == [4, 8]
    for row in rows:
        assert row["fixture"] == name and row["mu_r"] == "1000"
        deviations = row["deviation_vs_references"]
        assert "catani" not in deviations
        assert deviations["njet_dd_oneloop_R1"] == pytest.approx(3e-8, rel=1e-6)
        assert row["estimate"] == 1e-6


def test_fixture_stability_rows_stay_apart_and_decide_the_worst_point(tmp_path):
    """The gate keeps a ``stability --fixture`` record's rows apart from the
    stream's (``fixture_stability``: their indices are the fixture's); the
    report builder takes the estimate at the fixture's worst point from them,
    never from the stream's rows of the same index."""

    import json

    check_rows = [
        {
            "index": i,
            "mu_r": "1000",
            "born": 1.0,
            "finite": 2.0,
            "errors": {
                "njet_dd_oneloop_R1": {"born": 1e-13, "finite": dev},
                "njet_double_R2": {"born": 1e-13, "finite": 2 * dev},
            },
        }
        for i, dev in enumerate([1e-9, 2e-9, 3.7e-8, 5e-10])
    ]
    timing = _pyamplicol_record(
        "gg_ggggg",
        [7.0, 7.1, 7.2],
        accuracy_check={
            "fixture": "gg_ggggg_njet",
            "points": 4,
            "points_are_stream_points": [0, 1, 2],
            "metric": "test",
            "tolerance": 2e-8,
            "tolerance_source": "required",
            "max_error": 7.4e-8,
            "passed": False,
            "rows": check_rows,
        },
    )

    def stability(fixture, rows):
        return {
            "kind": "stability",
            "process": "gg_ggggg",
            "accuracy": "full",
            "inputs": "native",
            "lanes": 16,
            "schedule": "B",
            "revision": "0" * 40,
            **({"fixture": fixture} if fixture else {}),
            "rows": rows,
        }

    stream_row = {"index": 2, "arithmetic": "double", "born": 1.0, "finite": 1.0}
    fixture_rows = [
        {
            "index": 2,
            "arithmetic": "double",
            "born": 1.0,
            "finite": 2.0,
            "estimate": 1.3e-6,
        },
        {
            "index": 2,
            "arithmetic": "double-double",
            "born": 1.0,
            "finite": 2.0,
            "estimate": 1e-20,
            "deviation_vs_references": {
                "njet_dd_oneloop_R1": 4e-13,
                "njet_double_R2": 4e-8,
            },
        },
    ]
    paths = []
    for index, record in enumerate(
        (
            timing,
            stability(None, [{**stream_row, "estimate": 1e-9}]),
            stability("gg_ggggg_njet", fixture_rows),
        )
    ):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    document, _ = gate.build(paths, None)
    (proc,) = document["processes"]
    assert [r["estimate"] for r in proc["stability"]] == [1e-9]
    assert {r["fixture"] for r in proc["fixture_stability"]} == {"gg_ggggg_njet"}
    assert {r["configuration"] for r in proc["fixture_stability"]} == {"native/full"}
    builder = _report_builder()
    cfg = proc["pyamplicol"]["native/full"]
    out = builder._validation_g1(proc, cfg, "native/full", "src", "native/full")
    (ref,) = out["references"]
    assert ref["key"] == "njet_dd_oneloop_R1" and out["max_finite"] == 3.7e-8
    wp = out["worst_point"]
    assert wp["index"] == 2 and wp["fixture"] == "gg_ggggg_njet"
    assert wp["estimate"] == 1.3e-6 and wp["within_estimate"] is True
    # The double-double deviation from the shown references only (NJet DD).
    assert wp["dd_closest"] == 4e-13
    assert wp["source"].endswith(".fixture_stability (native/full)")
    # Without fixture rows, the stream's row of the same index is no estimate.
    del proc["fixture_stability"]
    out = builder._validation_g1(proc, cfg, "native/full", "src", "native/full")
    assert "worst_point" not in out


def test_a_user_acceptance_marks_exactly_the_points_above_the_tolerance():
    """A user acceptance (milestone.json validation_decisions; g g > g g g g g
    against NJet, 2026-10-25) marks the accepted points on the validation
    cell; the check keeps its tolerance and passed: false. Accepted points
    other than those above the tolerance, a point outside its binary64
    estimate, or a DD value off the reference stop the build."""

    builder = _report_builder()
    errors = [1e-9, 3.7e-8, 5e-10, 2.2e-8]
    check = {
        "fixture": "gg_ggggg_njet",
        "tolerance": 2e-8,
        "max_error": 4.3e-8,
        "passed": False,
        "rows": [
            {
                "index": i,
                "errors": {
                    "njet_dd_oneloop_R1": {"born": 1e-13, "finite": dev},
                    "njet_double_R2": {"born": 1e-13, "finite": dev * 1.1},
                },
            }
            for i, dev in enumerate(errors)
        ],
    }

    def stab(index, arithmetic, estimate, dd=0.0):
        return {
            "index": index,
            "arithmetic": arithmetic,
            "configuration": "native/full",
            "fixture": "gg_ggggg_njet",
            "estimate": estimate,
            "deviation_vs_references": {
                "njet_dd_oneloop_R1": dd,
                "njet_double_R2": 5e-9,
            },
        }

    def proc(estimate=1.3e-6, dd=0.0):
        return {
            "key": "gg_ggggg",
            "pyamplicol": {"native/full": {"accuracy_check": check}},
            "fixture_stability": [
                stab(1, "double", estimate),
                stab(1, "double-double", 1e-23, dd),
                stab(3, "double", 2.5e-7),
                stab(3, "double-double", 1e-23),
            ],
        }

    entry = {
        "process": "gg_ggggg",
        "configuration": "native/full",
        "fixture": "gg_ggggg_njet",
        "points": [1, 3],
        "tolerance": 2e-8,
        "label": "accepted (user 2026-10-25): O(1e-8) binary64, within the"
        " estimate, DD = NJet",
        "short": "accepted (user 2026-10-25)",
    }
    out = builder._accepted(proc(), entry)
    assert out["status"] == entry["label"] and out["passed"] is False
    assert out["tolerance"] == 2e-8 and [p["index"] for p in out["points"]] == [1, 3]
    first = out["points"][0]
    assert first["finite"] == {"njet_dd_oneloop_R1": 3.7e-8}
    assert first["max_error"] == pytest.approx(3.7e-8 * 1.1)
    assert first["dd_deviation"] == {"njet_dd_oneloop_R1": 0.0}
    # The cell carries it only where it validates with that check.
    source = "g1.json processes[gg_ggggg].pyamplicol['native/full'].accuracy_check"
    rows = [{"key": "gg_ggggg", "validation": {"source": source}}]
    g1 = {"processes": [proc()]}
    builder._accept(rows, g1, {"accepted": [entry]})
    assert rows[0]["validation"]["accepted"]["short"] == entry["short"]
    with pytest.raises(ValueError, match="not the validation cell"):
        builder._accept(
            [{"key": "gg_ggggg", "validation": {}}], g1, {"accepted": [entry]}
        )
    with pytest.raises(ValueError, match="points above the tolerance"):
        builder._accepted(proc(), {**entry, "points": [1]})
    with pytest.raises(ValueError, match="outside its estimate"):
        builder._accepted(proc(estimate=1e-8), entry)
    with pytest.raises(ValueError, match="DD value"):
        builder._accepted(proc(dd=3e-8), entry)
    # 'DD = NJet' means equal to binary64 rounding, not within the tolerance.
    with pytest.raises(ValueError, match="DD value"):
        builder._accepted(proc(dd=1e-9), entry)
    with pytest.raises(ValueError, match="not the check"):
        builder._accepted(proc(), {**entry, "tolerance": 5e-8})


def test_a_worst_deviation_the_double_double_value_repeats_is_the_references():
    """Beyond the binary64 estimate E, a worst deviation that pyAmpliCol's
    double-double value repeats is the reference's own (u u~ > g g g g g,
    Recola at point 0); one the DD value removes stays pyAmpliCol's (no
    attribution), and so does a deviation within E."""

    builder = _report_builder()

    def vs(dev):
        return {
            "points": 3,
            "max_finite_deviation": dev,
            "median_finite_deviation": dev,
            "worst_point": 0,
        }

    cfg = {"values_vs_providers": {"madloop": vs(1.2e-12), "recola": vs(2.9e-10)}}

    def row(arithmetic, estimate, devs):
        return {
            "index": 0,
            "arithmetic": arithmetic,
            "configuration": "native/full",
            "estimate": estimate,
            "deviation_vs_providers": devs,
        }

    def worst(dd_recola, b64_estimate=1.9e-11, dd_madloop=1.4e-12):
        proc = {
            "key": "uux_ggggg",
            "stability": [
                row("double", b64_estimate, {"madloop": 1.2e-12, "recola": 2.9e-10}),
                row(
                    "double-double",
                    1e-27,
                    {"madloop": dd_madloop, "recola": dd_recola},
                ),
            ],
        }
        out = builder._validation_g1(proc, cfg, "native/full", "src", "native/full")
        return out["worst_point"]

    wp = worst(2.9e-10)
    assert wp["within_estimate"] is False and wp["dd_closest"] == 1.4e-12
    assert wp["reference_own"] == {
        "key": "recola",
        "short": "Rc",
        "dd_deviation": 2.9e-10,
        "dd_estimate": 1e-27,
        "independent": {"key": "madloop", "dd_deviation": 1.4e-12},
    }
    # No independent reference within a hundredth of the deviation: a
    # systematic error of ours would repeat in DD too, so no attribution.
    assert "reference_own" not in worst(2.9e-10, dd_madloop=1e-11)
    # The DD value agrees with the reference: the binary64 loss is ours.
    assert "reference_own" not in worst(1e-15)
    # Within E: nothing to attribute.
    assert "reference_own" not in worst(2.9e-10, b64_estimate=1e-9)


def _accuracy_fixture():
    """A two-provider fixture of three points (the required fixtures'
    shape), at the 2e-8 tolerance."""

    point = [["1", "0", "0", "1"], ["1", "0", "0", "-1"]]
    return {
        "error_metric": "test",
        "tolerance": 2e-8,
        "points": [point] * 3,
        "mu_r": ["100"] * 3,
        "rows": [
            {
                "providers": {
                    "recola": {"born": 1.0, "finite": 2.0},
                    "openloops": {"born": 1.0, "finite": 2.0 * (1 + 1e-12)},
                }
            }
        ]
        * 3,
    }


def test_the_accuracy_check_asserts_binary64_only_within_the_o1_scope():
    """The user's ruling of 2026-10-29 on E1: binary64 is asserted against the
    providers where the finite part's estimate E <= 1e-9 (mqcd.O1_CAP), a
    double-double evaluation of the point above it; E and the binary64
    errors are recorded either way, and ``max_error`` / ``passed`` are the
    asserted carriers'."""

    from tools.loop_benchmark import mqcd, pyamplicol_v2

    fixture = _accuracy_fixture()
    tolerance = fixture["tolerance"]
    assert pyamplicol_v2.O1_CAP is mqcd.O1_CAP == 1e-9
    carrier = pyamplicol_v2.asserted_carrier
    assert carrier(1e-12) == carrier(1e-9) == "double"
    assert carrier(1.01e-9) == carrier(float("inf")) == "double-double"
    assert carrier(float("nan")) == "double-double"

    def record(b64_finite, estimates, dd_finite=2.0):
        binary64 = {i: (1.0, f) for i, f in enumerate(b64_finite)}
        dd = {
            i: (1.0, dd_finite)
            for i, e in estimates.items()
            if carrier(e) == "double-double"
        }
        return pyamplicol_v2.accuracy_record(
            "fx", fixture, binary64, estimates, dd, 0.118
        )

    # E1's shape: point 1 off by 2.4e-7 in binary64 with E = 2.5e-6.
    off = 2.0 * (1 + 2.4e-7)
    check = record([2.0, off, 2.0], {0: 1e-12, 1: 2.5e-6, 2: 1e-9})
    assert check["passed"] is True and check["tolerance"] == tolerance
    assert check["double_double_points"] == [1] and check["estimate_cap"] == 1e-9
    assert check["max_error"] < 1e-11
    assert check["max_error_binary64"] == pytest.approx(2.4e-7, rel=1e-6)
    rows = check["rows"]
    assert [r["asserted"]["finite"] for r in rows] == [
        "double",
        "double-double",
        "double",
    ]
    assert all(r["asserted"]["born"] == "double" for r in rows)
    assert rows[1]["estimate"] == {"finite": 2.5e-6}
    # The binary64 errors stay where the report builder reads them.
    assert rows[1]["errors"]["recola"]["finite"] == pytest.approx(2.4e-7, rel=1e-6)
    assert rows[1]["double_double"]["errors"]["recola"]["finite"] == 0.0
    assert "double_double" not in rows[0] and "double_double" not in rows[2]
    # Within the O1 scope binary64 is asserted: a 3e-8 miss fails.
    worse = 2.0 * (1 + 3e-8)
    failed = record([2.0, 2.0, worse], {0: 1e-12, 1: 1e-12, 2: 1e-9})
    assert failed["passed"] is False and failed["double_double_points"] == []
    assert failed["max_error"] == pytest.approx(3e-8, rel=1e-6)
    # Beyond it the double-double value is asserted at the same tolerance.
    assert (
        record([2.0, off, 2.0], {0: 1e-12, 1: 2.5e-6, 2: 1e-12}, worse)["passed"]
        is False
    )
    # The Born is asserted in binary64 at every point.
    fixture["rows"] = [
        {"providers": {"recola": {"born": 1.0 + 1e-7, "finite": 2.0}}}
    ] * 3
    born = record([2.0, 2.0, 2.0], {0: 2.5e-6, 1: 2.5e-6, 2: 2.5e-6})
    assert born["passed"] is False
    assert born["max_error"] == pytest.approx(1e-7, rel=1e-6)


def test_the_accuracy_check_evaluates_with_the_o1_estimate_and_dd_beyond_it(
    monkeypatch,
):
    """``Run.accuracy`` evaluates the fixture's points once in binary64 with
    the recommended checks (the O1 estimate, ``mqcd.check_configuration``)
    and the points with E > 1e-9 again in double-double, with the timed
    configuration's options; E is the finite part's absolute budget over
    the finite part, summed over the sectors."""

    import types

    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS
    from tools.loop_benchmark import pyamplicol_v2

    fixture = _accuracy_fixture()
    monkeypatch.setattr(pyamplicol_v2, "reference_fixture", lambda key: ("fx", fixture))
    # Per point: the binary64 finite part's deviation and its absolute budget
    # split over two sectors (E = 5e-8 / 2 at point 1).
    deviation = [0.0, 2.4e-7, 1e-12]
    budget = [(1e-12, 1e-12), (2e-8, 3e-8), (5e-10, 5e-10)]
    calls = []

    def result(finite, budgets):
        laurent = types.SimpleNamespace(finite=finite, coefficients=(finite,))
        half = types.SimpleNamespace(finite=0.0, coefficients=(0.0,))
        return types.SimpleNamespace(
            born={"a": 1.0},
            components={"virtual": {"a": laurent, "b": half}},
            ir_poles=None,
            stability=None
            if budgets is None
            else types.SimpleNamespace(
                absolute={
                    s: types.SimpleNamespace(coefficients=(e,))
                    for s, e in zip("ab", budgets, strict=True)
                }
            ),
        )

    def evaluate(points, scales, **options):
        calls.append((len(points), options))
        if options.get("precision") is not None:
            return [result(2.0, None) for _ in points]
        return [
            result(2.0 * (1 + d), b) for d, b in zip(deviation, budget, strict=True)
        ]

    run = types.SimpleNamespace(
        key="gg_gggg",
        options=types.SimpleNamespace(accuracy="full", lanes=16, schedule="B"),
        parameters={"aS": "0.118"},
        executor=types.SimpleNamespace(evaluate=evaluate),
        stream={"points": fixture["points"][:2]},
        record={},
    )
    pyamplicol_v2.Run.accuracy(run)
    (n64, first), (ndd, second) = calls
    assert n64 == 3 and first["stability"] is RECOMMENDED_CHECKS
    assert first["poles"] == second["poles"] == pyamplicol_v2._finite_only()
    assert ndd == 1 and second["precision"].arithmetic == "double-double"
    assert "stability" not in second
    check = run.record["accuracy_check"]
    assert check["points"] == 3 and check["points_are_stream_points"] == [0, 1]
    assert check["double_double_points"] == [1] and check["passed"] is True
    estimates = [r["estimate"]["finite"] for r in check["rows"]]
    assert estimates[0] == pytest.approx(1e-12)
    assert estimates[1] == pytest.approx(2.5e-8, rel=1e-6)
    assert estimates[2] == pytest.approx(5e-10, rel=1e-6)


@pytest.mark.loop_physics
def test_the_gg_gggg_accuracy_check_asserts_double_double_at_point_1(
    tmp_path, loop_native
):
    """E1 (g g > g g g g, native inputs, full colour; about 1.5 min): point 1
    is a pinched two-mass triangle whose binary64 estimate exceeds 1e-9, so
    its double-double value is asserted, and it equals MadLoop's
    quadruple-precision one (m-qcd PLAN §1.4 rule 1); every point's carrier
    follows its estimate, and the binary64 values evaluated with the checks
    are the timed configuration's (bitwise those without checks)."""

    import types

    from tools.loop_benchmark import pyamplicol_v2

    options = types.SimpleNamespace(
        process="gg_gggg",
        accuracy="full",
        inputs="native",
        lanes=16,
        schedule="B",
        native_build=None,
        scratch=tmp_path,
        store=None,
    )
    run = pyamplicol_v2.Run(options)
    try:
        run.generate()
        run.accuracy()
        check = run.record["accuracy_check"]
        _, fixture = pyamplicol_v2.reference_fixture("gg_gggg")
        points, scales = pyamplicol_v2.fixture_batch(fixture, [0, 1, 2])
        plain = run.executor.evaluate(
            points,
            scales,
            parameters=run.parameters,
            poles=pyamplicol_v2._finite_only(),
        )
    finally:
        run.close()
    rows = check["rows"]
    assert [pyamplicol_v2._totals(r) for r in plain] == [
        (row["born"], row["finite"]) for row in rows
    ]
    for row in rows:
        estimate = row["estimate"]["finite"]
        assert row["asserted"]["finite"] == pyamplicol_v2.asserted_carrier(estimate)
    point = rows[1]
    assert point["estimate"]["finite"] > pyamplicol_v2.O1_CAP
    assert 1 in check["double_double_points"]
    assert point["double_double"]["errors"]["madloop"]["finite"] <= 1e-12
    assert check["passed"] is True and check["max_error"] <= check["tolerance"]
    assert check["max_error_binary64"] >= max(
        e["finite"] for e in point["errors"].values()
    )


@pytest.mark.loop_physics
def test_a_stored_process_loads_and_evaluates_bitwise_equal(tmp_path, loop_native):
    import json

    from tools.loop_benchmark.__main__ import main

    build = tmp_path / "build"
    build.mkdir()
    info = {
        "native_build_inputs_sha256": "a" * 64,
        "source_revision": "0" * 40,
        "profile": {"name": "release"},
    }
    (build / "build-info.json").write_text(json.dumps(info))
    common = ["gg_gg", "--inputs", "native", "--native-build", str(build)]
    common += ["--batches", "2", "--reps", "1", "--min-points", "1"]
    store = tmp_path / "store"
    generated_json, loaded_json = tmp_path / "generated.json", tmp_path / "loaded.json"
    generate = ["--scratch", str(tmp_path), "--store", str(store)]
    assert main(["pyamplicol", *common, *generate, "--json", str(generated_json)]) == 0
    load = ["--load", str(store), "--json", str(loaded_json)]
    assert main(["pyamplicol", *common, *load]) == 0
    generated = json.loads(generated_json.read_text())
    loaded = json.loads(loaded_json.read_text())
    assert loaded["batches"]["2"]["values"] == generated["batches"]["2"]["values"]
    assert loaded["accuracy_check"] == generated["accuracy_check"]
    assert loaded["generation"] == generated["generation"]
    assert loaded["structure"] == generated["structure"]
    stored = loaded["stored"]
    from pyamplicol.generation.loop_artifact import LOOP_V2_NATIVE_POINT_INPUTS

    assert stored["point_inputs"] == LOOP_V2_NATIVE_POINT_INPUTS == "native-v2"
    assert "store_label" not in stored
    stored_meta = json.loads((store / "generation.json").read_text())
    assert stored_meta["point_inputs"] == LOOP_V2_NATIVE_POINT_INPUTS
    assert stored["generated_revision"] == generated["revision"]
    assert str(tmp_path) not in json.dumps(stored)
    # Another configuration or another native build is refused; a stored
    # process is never overwritten.
    with pytest.raises(SystemExit, match="is not this run"):
        main(["pyamplicol", *common, "--accuracy", "lc", "--load", str(store)])
    meta = json.loads((store / "generation.json").read_text())
    meta["native"]["digest"] = "b" * 64
    (store / "generation.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="another native build"):
        main(["pyamplicol", *common, "--load", str(store)])
    with pytest.raises(SystemExit, match="never overwritten"):
        main(["pyamplicol", *common, "--store", str(store), "--generate-only"])
    # A storing run's record times nothing: no gate input.
    only = {**generated, "batches": {}, "generate_only": True}
    path = tmp_path / "only.json"
    path.write_text(json.dumps(only))
    with pytest.raises(ValueError, match="generate-only"):
        gate.build([path], None)


def _report_builder():
    import importlib.util

    path = (
        ROOT / "docs" / "performance_reports" / "loop" / "scripts" / "build_summary.py"
    )
    spec = importlib.util.spec_from_file_location("loop_report_build_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_builder_labels_the_floor_and_reports_addendum_h_checks():
    builder = _report_builder()
    rule = {
        "below_seconds_per_point": gate.GATE_FLOOR_SECONDS,
        "verdict": "not gated",
        "label": gate.GATE_FLOOR_STATUS,
        "short": "<5 ms: not gated",
        "source": "test",
    }
    decisions = {"decisions": [], "threshold_rule": rule, "source": "test"}
    times = {"madloop": 0.0040, "openloops": 0.0010}

    def cell(seconds: float, verdict: dict) -> dict:
        summary = _pyamplicol_record("uux_ddxg", [seconds] * 3)["batches"]["128"]
        summary["uncertainty"] = 0.01
        summary["gate"] = verdict
        config = {"revision": "0" * 40, "inputs": "native", "batches": {"128": summary}}
        return builder._timing_of(
            {"key": "uux_ddxg"}, "native/full", config, "src", None, decisions
        )

    fast = gate.gate_verdict(0.003, times, uncertainty=0.01)
    out = cell(0.003, fast)["gate"]
    assert out["status"] == "not gated" and out["rule"] == gate.GATE_FLOOR_STATUS
    assert out["measured_status"] == "fail" and out["floor_seconds"] == 0.005
    gated = cell(0.006, gate.gate_verdict(0.006, times, uncertainty=0.01))["gate"]
    assert gated["status"] == "fail" and "rule" not in gated
    # The tool's floor and the registry's rule must agree on the cell.
    with pytest.raises(ValueError, match="disagrees"):
        cell(0.006, fast)
    # A row without a gate: its rotation rows per configuration and, without
    # Ward rows of its own, the registry's reason with the lower ones.
    rotation = {"rotation-0": 2e-8, "reduction-direction": 8e-9}
    g1 = {
        "processes": [
            {
                "key": "gg_gggg",
                "outgoing": 4,
                "ward": [{"passed": True, "ratios": {"virtual": [3e-10, 4e-11]}}],
            },
            {
                "key": "gg_ggggg",
                "outgoing": 5,
                "stability": [
                    {
                        "index": i,
                        "arithmetic": "double",
                        "configuration": "native/full",
                        "estimate": 6e-8,
                        "checks": rotation,
                        "revision": "1" * 40,
                    }
                    for i in range(3)
                ],
            },
        ]
    }
    checks = builder._sanity_checks(g1, "gg_ggggg", {"ward": "not run: reason"})
    full = checks["rotation"]["configurations"]["native/full"]
    assert full["points"] == [0, 1, 2] and full["max_rotation"] == 2e-8
    ward = checks["ward"]
    assert ward["status"] == "not run" and ward["reason"] == "not run: reason"
    (lower,) = ward["lower_multiplicity"]
    assert lower["key"] == "gg_gggg" and lower["passed"] is True
    assert lower["max_virtual_ratio"] == 3e-10


def test_an_upper_bound_outside_the_protocol_bounds_the_fastest_provider():
    """A matched provider's timing outside the protocol that does the
    production work and more (production_upper_bound) bounds its production
    time from above: below the fastest protocol-mode time it is the fastest
    as a bound, and the ratio to it is a lower bound (u u~ > g g g g g:
    Recola's reference run with the IR poles, faster than MadLoop's
    production build). Above it, or without the reason, it changes nothing."""

    builder = _report_builder()
    recola = {
        "unavailable": "no production-mode timing",
        "median_seconds_per_point": 182.0,
        "points": 3,
        "mode": "reference campaign: compute_ir_poles 1, Collier cache off",
        "production_upper_bound": "poles on top of the production work",
    }
    madloop = {"median_seconds_per_point": 569.0, "points": 3, "mode": "production"}
    proc = {"providers": {"madloop": madloop, "recola": dict(recola)}}
    runtime = builder._provider_runtime(proc)
    fastest = runtime["fastest"]
    assert fastest["key"] == "recola" and fastest["bound"] == "upper"
    assert fastest["seconds_per_point"] == 182.0
    assert fastest["protocol_fastest"]["key"] == "madloop"
    out = runtime["providers"]["recola"]["outside_protocol"]
    assert out["production_upper_bound"] == recola["production_upper_bound"]
    summary = _pyamplicol_record("uux_ggggg", [1.8] * 3)["batches"]["128"]
    summary["uncertainty"] = 0.01
    config = {"revision": "0" * 40, "inputs": "native", "batches": {"128": summary}}
    decisions = {"decisions": [], "source": "test"}
    cell = builder._timing_of(
        {"key": "uux_ggggg"}, "native/full", config, "src", fastest, decisions
    )
    assert cell["ratio_to_fastest"] == pytest.approx(1.8 / 182.0)
    assert cell["ratio_to_fastest_bound"] == "lower"
    # A bound above the fastest protocol-mode time, or a timing outside the
    # protocol without the reason, leaves the protocol-mode fastest.
    proc["providers"]["madloop"] = dict(madloop, median_seconds_per_point=100.0)
    assert builder._provider_runtime(proc)["fastest"] == {
        "name": "MadLoop",
        "key": "madloop",
        "seconds_per_point": 100.0,
    }
    del proc["providers"]["recola"]["production_upper_bound"]
    proc["providers"]["madloop"] = madloop
    assert "bound" not in builder._provider_runtime(proc)["fastest"]
    plain = builder._timing_of(
        {"key": "uux_ggggg"},
        "native/full",
        config,
        "src",
        builder._provider_runtime(proc)["fastest"],
        decisions,
    )
    assert "ratio_to_fastest_bound" not in plain


def test_a_campaign_timed_on_tool_revisions_of_its_slice_revision(monkeypatch):
    """A G1 campaign whose configurations are all recorded at its tool
    revisions (slice_tool_revisions: benchmark-tool commits on top of
    slice_revision, whose engine _check_tool_revisions compares) is
    consistent; one with no configuration at slice_revision or at a tool
    revision (an older campaign under a new label) stops the build."""

    builder = _report_builder()
    monkeypatch.setattr(builder, "_check_tool_revisions", lambda g1: None)
    cfg = {"revision": "b" * 40, "branch": "HEAD"}
    g1 = {
        "slice_revision": "a" * 8,
        "slice_tool_revisions": "cccccccc,bbbbbbbb",
        "processes": [{"key": "gg_gg", "pyamplicol": {"native/full": cfg}}],
    }
    sides = builder._sides(g1)
    assert sides["predicates"]["integrated"](cfg)
    assert sides["revisions"] == {"integrated": "a" * 8}
    g1["slice_tool_revisions"] = "cccccccc"
    with pytest.raises(ValueError, match="no configuration is recorded"):
        builder._sides(g1)


def test_an_imported_madloop_timing_takes_the_specified_exports_generation(
    tmp_path, no_old_root
):
    """A MadLoop timing imported from the reference campaign names no export
    (u u~ > g g g g g): its generation is read from the specification's
    ``madloop.export`` under the reference root, and an imported Recola run
    without a production-mode timing keeps its generation in the report."""

    import json

    times = {"graph_generation_s": 5.0, "source_export_s": 7.0, "compile_s": 1.0}
    export = tmp_path / "ref" / gate._specified_export("uux_ggggg")
    export.mkdir(parents=True)
    (export.parent / f"{export.name}.generation.json").write_text(json.dumps(times))
    rows = [{"born": 1.0, "finite": 0.5, "seconds": 500.0}] * 2
    imported = {
        "provider": "madloop",
        "protocol_mode": True,
        "imported_from": "runs/required/uux_ggggg/madloop-production-result.json",
        "points": 2,
        "mean_seconds_per_point": 500.0,
        "median_seconds_per_point": 500.0,
        "rows": rows,
    }
    providers = {
        "kind": "providers",
        "process": "uux_ggggg",
        "cpu": 172,
        "points": 2,
        "providers": {"madloop": imported},
    }
    paths = []
    for index, record in enumerate(
        [providers, _pyamplicol_record("uux_ggggg", [3.0, 3.1, 3.2])]
    ):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    document, _ = gate.build(paths, tmp_path / "ref")
    (row,) = document["processes"]
    madloop = row["providers"]["madloop"]
    assert madloop["export"] == gate._specified_export("uux_ggggg")
    assert madloop["generation"]["total_s"] == 13.0
    # Without the reference root: the name only, no generation.
    document, _ = gate.build(paths, None)
    assert document["processes"][0]["providers"]["madloop"]["generation"] is None
    assert no_old_root == []

    builder = _report_builder()
    recola = {
        "unavailable": "no production-mode timing",
        "imported_from": "runs/required/uux_ggggg/recola-result.json",
        "setup": {"generation_seconds": 11.0},
    }
    gens = builder._provider_generation_g1(
        {
            "providers": {
                "madloop": row["providers"]["madloop"],
                "recola": recola,
                "openloops": {"unavailable": "no library"},
            }
        }
    )
    assert gens["madloop"]["seconds"] == 13.0 and gens["recola"] == {"seconds": 11.0}
    assert gens["openloops"] is None
    # A provider that never ran (no imported run) has no generation.
    stopped = builder._provider_generation_g1(
        {"providers": {"recola": {"unavailable": "stopped"}}}
    )
    assert stopped["recola"] is None
    # Reference runs that did not finish are reported as such.
    gens, unfinished = builder._reference_generation(
        {
            "madloop": {"status": "exporting", "graph_generation_s": 60.0},
            "recola": {"status": "running", "updated": "2026-10-20"},
            "openloops": None,
        }
    )
    assert gens == {"madloop": None, "recola": None, "openloops": None}
    assert unfinished == {
        "madloop": {"status": "exporting", "graph_generation_s": 60.0},
        "recola": {"status": "running", "since": "2026-10-20"},
    }
