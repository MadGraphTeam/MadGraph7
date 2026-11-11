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
  gate its rotation and Ward checks (addendum H); it carries each summary
  table's provenance line (engine revision, native build and digest, host,
  date);
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
    # Supplied points (mass patterns exact_points cannot represent, e.g. the
    # loop-induced g g > h h h): momentum conservation exact, mass shells to
    # the printed 30 digits.
    supplied = stream["construction"].startswith("supplied_points")
    # A decay (loop-induced h > g g, z > g g g, ...): one incoming particle.
    incoming = stream.get("initial", 2)
    for point in points:
        if not supplied:
            exact_points.check(point, masses, incoming)
            continue
        for mu in range(4):
            assert sum(p[mu] for p in point[:incoming]) == sum(
                p[mu] for p in point[incoming:]
            )
        for p, m in zip(point, masses, strict=True):
            assert abs(exact_points._dot(p, p) - m * m) <= p[0] ** 2 / 10**27
    assert streams.campaign_agreement(key, stream) in (True, None)
    text = streams.binary64(stream, 3)
    for exact, decimal in zip(stream["points"][:3], text, strict=True):
        for leg, values in zip(exact, decimal, strict=True):
            assert [float(Fraction(x)) for x in leg] == [float(x) for x in values]


@pytest.mark.parametrize(
    "key",
    [k for k in streams.keys() if "two-body construction" in streams.load(k)["construction"]],
)
def test_pair_streams_hold_no_beam_collinear_point(key):
    """lit-report: the exact centre-of-mass points of a massive 2 -> 2 pair
    never put a final momentum exactly along the beam (zero transverse
    momentum: the boxes' Gram determinants vanish exactly there; both
    backends failed such points of g g > z z in binary64)."""

    stream = streams.load(key)
    prefix = stream["campaign_prefix"]
    for index in range(prefix, len(stream["points"])):
        point = streams.exact(stream, index)
        assert all(p[1] or p[2] for p in point[2:]), index


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
        # A Born-interfered run of the harness names its backend (a string).
        _pyamplicol_record("gg_ggg", [0.0060, 0.0061, 0.0062], reduction="ninjars"),
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
    # Single provider runs: this run's host spread is 0; another campaign's
    # (``--host-spread``, the EW row gated beside G1) is assumed where larger.
    assert document["host_provider_spread"] == 0.0
    assert "host_provider_spread_of_run" not in document
    wide, _ = gate.build(paths, tmp_path / "ref", host_spread=0.08)
    assert wide["host_provider_spread"] == 0.08
    assert wide["host_provider_spread_of_run"] == 0.0
    row = {r["key"]: r for r in wide["processes"]}["gg_ggg"]
    assert row["provider_uncertainty"]["openloops"] == {
        "relative": 0.08,
        "assumed": True,
    }
    band = row["pyamplicol"]["native/full"]["batches"]["128"]["gate"]
    assert band["ratio_to_binding_limit"] == slow["ratio_to_binding_limit"]
    assert band["band_to_binding_limit"][0] < slow["band_to_binding_limit"][0]
    assert band["band_to_binding_limit"][1] > slow["band_to_binding_limit"][1]
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


def test_the_rotation_block_times_exact_rotated_copies_and_records_the_contraction():
    """``--rotation-samples K --rotations R``: after the warm-up, one timed
    batch of the K stream points after the warm-up's, each as R copies (copy 0
    the point, copy r its ``default_rotation(r - 1)``); the record keeps each
    sample's finite parts and spread and the seed contraction that ran."""

    import types

    from pyamplicol.loops.v2.plan_export import FFT_SEED_JOBS
    from pyamplicol.loops.v2.stability import default_rotation
    from tools.loop_benchmark import pyamplicol_v2

    calls: list[tuple[list, list]] = []

    def evaluate(points, scales, **options):
        calls.append((points, scales))
        executor.last_report = {
            "seconds": {"dressing": 0.5},
            "counts": {FFT_SEED_JOBS: 7},
            "lanes": 16,
            "threads": 1,
            "reals_per_lane": {},
        }
        # Rotation invariant: |p1|^2 of the first final-state leg (exact).
        out = []
        for point in points:
            value = float(sum(x * x for x in point[2][1:]))
            out.append(
                types.SimpleNamespace(
                    born={"total": 2.0 * value},
                    components={
                        "virtual": {"total": types.SimpleNamespace(finite=value)}
                    },
                    colour_contraction="symmetric-group-fft",
                )
            )
        return out

    executor = types.SimpleNamespace(
        evaluate=evaluate,
        contraction="fft",
        exported=types.SimpleNamespace(fourier="{}"),
    )
    run = pyamplicol_v2.Run.__new__(pyamplicol_v2.Run)
    run.options = types.SimpleNamespace(
        batches=[16],
        warm_up=16,
        schedule="B",
        rotation_samples=4,
        rotations=4,
        json=None,
    )
    run.key = "gg_ggg"
    run.stream = streams.load("gg_ggg")
    run.parameters = {}
    run.timers = pyamplicol_v2.Timers()
    run.executor = executor
    run.record = {}
    run.timing()
    assert [len(points) for points, _ in calls] == [16, 16]
    points, scales = calls[1]
    for k, index in enumerate(range(16, 20)):
        point = streams.exact(run.stream, index)
        copies = points[4 * k : 4 * k + 4]
        assert copies[0] == point
        for r in range(1, 4):
            assert copies[r] == [list(m) for m in default_rotation(r - 1).apply(point)]
            assert copies[r] != point
        assert len(set(scales[4 * k : 4 * k + 4])) == 1
    block = run.record["rotation_timing"]
    assert block["stream_indices"] == [16, 17, 18, 19] and block["points"] == 16
    assert [len(row["finite"]) for row in block["rows"]] == [4] * 4
    assert block["max_relative_spread"] < 1e-15
    assert block["colour_contraction"] == {
        "policy": "fft",
        "certified_fourier_record": True,
        "evaluated": ["symmetric-group-fft"],
        "fft_seed_jobs": 7,
    }
    assert list(run.record["batches"]) == ["16"]
    assert sorted(run.record["batches"]["16"]["values"]) == ["16", "17", "18", "19"]
    assert "skipped_batches" not in run.record


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
    # Each measured provider names its timing's settings (lit-report G5): a
    # record without a protocol is a production run.
    proc["providers"]["openloops"] = {
        "median_seconds_per_point": 50.0,
        "points": 16,
        "mode": "most optimistic",
        "protocol": "optimistic",
        "library": {"name": "ppllll"},
    }
    runtime = builder._provider_runtime(proc)["providers"]
    assert runtime["openloops"]["protocol"] == "optimistic"
    assert runtime["madloop"]["protocol"] == "production"
    assert "protocol" not in runtime["recola"]


def test_table_1_names_its_provider_protocols_and_marks_kept_production_times():
    """lit-report G5: summary.json meta lists the settings of Table 1's
    provider timings and why a provider kept its production record (g1.json
    meta ``provider_kept``); the report marks a production time beside
    optimistic ones and names the protocol in its legend."""

    import json

    source = (
        ROOT / "docs/performance_reports/loop/scripts/build_summary.py"
    ).read_text()
    assert '"provider_protocols": sorted(' in source
    assert '{"provider_kept": g1["provider_kept"]}' in source
    typ = (ROOT / "docs/performance_reports/loop/benchmarks.typ").read_text()
    assert '#let t1-optimistic = "optimistic" in S.meta.at("provider_protocols"' in typ
    assert "#t1-mark(x)" in typ and "#t1-mark(c.providers.at(f.key))" in typ
    assert "#if t1-optimistic [ The providers run in their most optimistic" in typ
    summary = json.loads(
        (ROOT / "docs/performance_reports/loop/data/summary.json").read_text()
    )
    protocols = summary["meta"]["provider_protocols"]
    shown = {
        x["protocol"]
        for g in summary["groups"]
        for r in g["rows"]
        for x in (r["full_colour"].get("providers") or {}).values()
        if "protocol" in x
    }
    assert protocols == sorted(shown) and protocols


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


def test_report_builder_gives_each_summary_table_its_provenance(tmp_path, monkeypatch):
    """The report's essentials rule (user 2026-11-06 ~13:25): each summary
    table states its provenance in one line (engine revision, native build
    and digest, host, date), so the builder carries them: Table 1's
    configurations the G1 campaign's native build and digest beside its
    revision and date (meta: host); the loop-induced table the engine's
    native build while no record of its own names one, each record's
    native build and digest on its cell, and each provider timing record's
    host on its provider entries."""

    import json

    builder = _report_builder()
    monkeypatch.setattr(builder, "_check_tool_revisions", lambda g1: None)
    cfg = {
        "revision": "a" * 40,
        "branch": "HEAD",
        "inputs": "native",
        "accuracy": "full",
        "schedule": "B",
        "lanes": 16,
    }
    g1 = {
        "date": "2026-11-06",
        "slice_revision": "a" * 8,
        "slice_build": "loop-native-test",
        "slice_native_digest": "d" * 64,
        "processes": [{"key": "gg_gg", "pyamplicol": {"native/full": cfg}}],
    }
    ms = {"development_records": {"conditions": "test", "short": "test"}}
    meas = {"required_native": {"branch": "work/test", "processes": {}}}
    recency = builder.Recency(tmp_path, {"revisions": [], "ancestors": []})
    configurations = builder._configurations(g1, ms, meas, [], recency)
    integrated = configurations["integrated"]
    assert integrated["revision"] == "a" * 8
    assert integrated["native_build"] == "loop-native-test"
    assert integrated["native_digest"] == "d" * 64
    assert integrated["date"] == "2026-11-06"

    ms = _loop_induced_milestone(_loop_induced_registry())
    ms["native_build"] = {"name": "loop-native-li", "digest": "e" * 8, "source": "x"}
    data = tmp_path / "data"
    data.mkdir()
    record = {
        "key": "gg_hgg",
        "revision": "bbbbbbbb",
        "backend": "tirust",
        "batch": 128,
        "seconds_per_point": 0.003,
        "host": "testhost",
        "generation": {"seconds": 12.0, "probes_seconds": 3.0, "peak_gib": 0.5},
        "native_build": "loop-native-li-run",
        "native_digest": "f" * 64,
        "source": "test/gg_hgg",
    }
    (data / "li-timing.json").write_text(json.dumps({"records": [record]}))
    timing = tmp_path / "tests" / "fixtures" / "loop" / "required" / "li" / "t.json"
    timing.parent.mkdir(parents=True)
    timing.write_text(
        json.dumps(
            {
                "kind": "providers",
                "host": "testhost",
                "providers": {"openloops": {"median_seconds_per_point": 0.01}},
            }
        )
    )
    # lit-ref's manifest: every registry row, one reference run with a
    # provider timing record for g g > h g g.
    rows_m = [
        {"row": f"LI-{r['key']}", "processes": r["process"], "runs": []}
        for r in ms["loop_induced"]["rows"]
    ]
    hgg = next(m for m in rows_m if m["row"] == "LI-gg_hgg")
    hgg["runs"] = [
        {
            "process": "li_gg_hgg",
            "madloop_process": "g g > h g g [sqrvirt=QCD]",
            "provider_timing_records": ["required/li/t.json"],
        }
    ]
    manifest = tmp_path / builder.LI_CAMPAIGN
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"rows": rows_m}))
    recency = builder.Recency(tmp_path, ms["ancestry"])
    out = builder.build_loop_induced(ms, recency, data=data, repo=tmp_path)
    assert out["meta"]["native_build_on_loop"] == {
        "name": "loop-native-li",
        "digest": "e" * 8,
    }
    assert out["meta"]["date"] == "2026-11-06"
    rows = {r["key"]: r for g in out["groups"] for r in g["rows"]}
    cell = rows["gg_hgg"]["timing"]
    assert cell["revision"] == "bbbbbbbb"
    assert cell["native_build"] == "loop-native-li-run"
    assert cell["native_digest"] == "f" * 64
    assert cell["host"] == rows["gg_hgg"]["generation"]["host"] == "testhost"
    openloops = rows["gg_hgg"]["providers"]["providers"]["openloops"]
    assert openloops["host"] == "testhost"
    # The report's own line (running header, page-1 subtitle): its date and
    # newest measured engine, beside each table's own (Table 1's
    # configuration labelled as such): Table 2's bbbbbbbb is newer than
    # Table 1's cccccccc by the recorded ancestry; without a Table 2 record
    # the head is Table 1's.
    summary = {
        "groups": [
            {
                "rows": [
                    {
                        kind: {"main": {"status": "measured", "revision": "c" * 8}}
                        for kind in ("generation", "full_colour", "leading_colour")
                    }
                ]
            }
        ]
    }
    ms["ancestry"] = {
        "revisions": [*ms["ancestry"]["revisions"], "c" * 8],
        "ancestors": [*ms["ancestry"]["ancestors"], ["c" * 8, "bbbbbbbb"]],
    }
    recency = builder.Recency(tmp_path, ms["ancestry"])
    report = builder.report_meta(summary, out, recency, ms)
    assert (report["date"], report["head"]) == ("2026-11-06", "bbbbbbbb")
    assert (report["table1"], report["table2"]) == ("c" * 8, "bbbbbbbb")
    empty = {"groups": []}
    report = builder.report_meta(summary, empty, recency, ms)
    assert report["head"] == report["table1"] == "c" * 8 and report["table2"] is None


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


def _loop_induced_registry(**changes) -> dict:
    """A small loop-induced row registry (milestone.json loop_induced)."""

    def row(key, process, group, gate_class, served, backends=("ninjars", "tirust")):
        return {
            "key": key,
            "process": process,
            "ids": [f"LI-{key}"],
            "corrections": "QCD",
            "legs": len(process.replace(">", "").split()),
            "group": group,
            "gate_class": gate_class,
            "served_by": served,
            "backends": list(backends),
            "references": ["madloop", "openloops", "recola"],
        }

    registry = {
        "source": "test",
        "default_backend": "tirust",
        "landed": ["lit-amp", "lit-r2"],
        "groups": ["2->2", "2->3"],
        "gate_classes": {"reported": "test", "gated": "test"},
        "validation_only": {"ids": ["LI-Z1"], "reason": "test"},
        "rows": [
            row("gg_hh", "g g > h h", "2->2", "reported", "lit-amp"),
            row("aa_hh", "a a > h h", "2->2", "reported", "lit-ew"),
            row("gg_hgg", "g g > h g g", "2->3", "gated", "lit-r2"),
            row("gg_zzg", "g g > z z g", "2->3", "gated", "lit-r2"),
            row("gg_hhh", "g g > h h h", "2->3", "gated", "lit-r2"),
            row("gg_zzz", "g g > z z z", "2->3", "gated", "lit-r2"),
        ],
    }
    registry.update(changes)
    return registry


def _loop_induced_milestone(registry: dict) -> dict:
    rule = {
        "below_seconds_per_point": gate.GATE_FLOOR_SECONDS,
        "verdict": "not gated",
        "label": gate.GATE_FLOOR_STATUS,
        "short": "<5 ms: not gated",
        "source": "test",
    }
    return {
        "date": "2026-11-06",
        "engine_revision": "bbbbbbbb",
        "gate_decisions": {"decisions": [], "threshold_rule": rule, "source": "test"},
        "ancestry": {
            "revisions": ["aaaaaaaa", "bbbbbbbb"],
            "ancestors": [["aaaaaaaa", "bbbbbbbb"]],
        },
        "loop_induced": registry,
    }


def test_report_builder_loop_induced_table_markers_backend_floor_and_reports(
    tmp_path,
):
    """The loop-induced section of the report builder (m-lit/PLAN.md §3.7
    'Pushes'): absent input files read as pending, a package not landed
    leaves its rows pending, a landed package without a record is 'not run';
    the cell takes the newest revision's B = 128 record, the default backend
    (TIRust) first with the other backend beside it, and NinjaRS per
    amplitude labelled where only NinjaRS ran at B = 128; B = 1 or other
    batches alone give no batched cell (with the reason), a repeated record
    is refused; four legs or fewer are reported, below 5 ms not gated, no
    verdict before lit-perf's records, Table 1's chip from them; no matched
    provider means no gate; the registry's legs and gate classes agree."""

    import json

    builder = _report_builder()
    ms = _loop_induced_milestone(_loop_induced_registry())
    data = tmp_path / "data"
    data.mkdir()

    def build():
        recency = builder.Recency(tmp_path, ms["ancestry"])
        out = builder.build_loop_induced(ms, recency, data=data, repo=tmp_path)
        return {r["key"]: r for g in out["groups"] for r in g["rows"]}, out

    # No input file yet: every cell pending, the validation and providers too.
    rows, out = build()
    assert [g["name"] for g in out["groups"]] == ["2->2", "2->3"]
    assert not any(x["present"] for x in out["meta"]["inputs"].values())
    assert out["meta"]["campaign"]["checked"] is False
    for r in rows.values():
        assert r["timing"]["status"] == r["generation"]["status"] == "pending"
        assert r["validation"]["status"] == r["providers"]["status"] == "pending"
    assert "lit-ew" in rows["aa_hh"]["timing"]["reason"]
    assert "li-timing.json" in rows["gg_hh"]["timing"]["reason"]

    def rec(key, backend, seconds, revision="bbbbbbbb", **extra):
        return {
            "key": key,
            "revision": revision,
            "backend": backend,
            "batch": 128,
            "seconds_per_point": seconds,
            "host": "testhost",
            "generation": {"seconds": 12.0, "probes_seconds": 3.0, "peak_gib": 0.5},
            "source": f"test/{key}",
            **extra,
        }

    timing = {
        "records": [
            rec("gg_hh", "ninjars", 0.002),
            rec("aa_hh", "ninjars", 0.002),
            rec("gg_hgg", "tirust", 0.010, revision="aaaaaaaa"),
            rec("gg_hgg", "ninjars", 0.004),
            rec("gg_hgg", "tirust", 0.003),
            # B = 1 (reported in the detailed sections, PLAN §5.1) before
            # B = 128 in the file: the cell is B = 128's.
            rec("gg_zzg", "tirust", 0.050, batch=1),
            rec("gg_zzg", "tirust", 0.020),
            rec("gg_zzz", "tirust", 0.030),
        ]
    }
    # lit-ref's provider timings: the manifest's reference runs name their
    # timing records (below tests/fixtures/loop/, the gate tool's providers
    # format) and their unavailable providers; MadLoop counts at its
    # production card 7|6|1 only (the 6|7|1 OPP column is not the summary's).
    fixtures = tmp_path / "tests" / "fixtures" / "loop"

    def timing_record(name, **providers):
        path = fixtures / "required" / "li" / "timing" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": "providers", "providers": providers}))
        return str(path.relative_to(fixtures))

    def ml(t, lib="7|6|1"):
        return {
            "median_seconds_per_point": t,
            "madloop_params": {"MLReductionLib": lib},
        }

    production = timing_record(
        "production.json",
        madloop=ml(0.04),
        openloops={"median_seconds_per_point": 0.01},
        recola={
            "median_seconds_per_point": 0.05,
            "setup": {"generation_seconds": 60.0},
        },
    )
    no_recola = timing_record(
        "no-recola.json", madloop=ml(0.04), openloops={"median_seconds_per_point": 0.01}
    )
    opp = timing_record("opp.json", madloop=ml(0.001, lib="6|7|1"))
    process = {r["key"]: r["process"] for r in ms["loop_induced"]["rows"]}

    def run(key, records=(), **unavailable):
        return {
            "process": f"li_{key}",
            "madloop_process": f"{process[key]} [sqrvirt=QCD]",
            "provider_timing_records": list(records),
            "unavailable": unavailable,
        }

    def write_manifest(runs):
        rows_m = [
            {"row": f"LI-{key}", "processes": p, "runs": runs.get(key, [])}
            for key, p in process.items()
        ]
        manifest = tmp_path / builder.LI_CAMPAIGN
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"rows": rows_m}))

    runs = {
        "gg_hh": [run("gg_hh", [production, opp])],
        "gg_hgg": [run("gg_hgg", [production, opp])],
        "gg_zzg": [run("gg_zzg", [no_recola], recola="beyond 45 min")],
        "gg_zzz": [run("gg_zzz", madloop="no", openloops="no library", recola="no")],
    }
    write_manifest(runs)
    (data / "li-timing.json").write_text(json.dumps(timing))
    rows, out = build()
    assert out["meta"]["inputs"]["providers"]["present"] is True
    hgg_ml = rows["gg_hgg"]["providers"]["providers"]["madloop"]
    assert hgg_ml["seconds_per_point"] == 0.04
    assert hgg_ml["setting"] == "MLReductionLib 7|6|1"
    assert rows["gg_zzg"]["providers"]["providers"]["recola"] == {
        "status": "unavailable",
        "reason": "beyond 45 min",
        "source": "li_campaign.json runs[li_gg_zzg]",
    }
    # A package not landed stays pending even with a record.
    assert rows["aa_hh"]["timing"]["status"] == "pending"
    # NinjaRS per amplitude, labelled, where only NinjaRS ran; four legs: reported.
    hh = rows["gg_hh"]["timing"]
    assert hh["backend_label"] == "NinjaRS per amplitude" and not hh["default_backend"]
    assert hh["gate"] == {"status": "reported", "label": "no gate (2->2)"}
    assert hh["ratio_to_fastest"] == pytest.approx(0.002 / 0.01)
    assert rows["gg_hh"]["providers"]["fastest"]["key"] == "openloops"
    # Generation: the engine's generation (generate + filter + plan export),
    # its multiplier against the provider's generation formed on it; the
    # optional generation-time probes (validation, off by default since
    # lit-probe) apart, with their carrier (from their split).
    gen = rows["gg_hh"]["generation"]
    assert gen["status"] == "measured" and gen["seconds"] == 12.0
    assert gen["validation"] == {"seconds": 3.0, "carrier": "double-double"}
    assert "probe_share" not in gen and "probes_seconds" not in gen
    assert gen["ratio_to_fastest"] == pytest.approx(12.0 / 60.0)
    assert gen["host"] == hh["host"] == "testhost"
    # The newest revision's default backend, the other backend beside it;
    # below 5 ms not gated.
    hgg = rows["gg_hgg"]["timing"]
    assert hgg["backend"] == "tirust" and hgg["revision"] == "bbbbbbbb"
    assert hgg["seconds_per_point"] == 0.003
    assert hgg["other_backend"]["backend_label"] == "NinjaRS per amplitude"
    assert hgg["gate"]["status"] == "not gated"
    assert hgg["gate"]["label"] == "<5 ms: not gated"
    # Gated, above 5 ms, no lit-perf record: no verdict yet; the B = 1
    # record listed first neither hides the B = 128 one nor is the other
    # backend.
    zzg = rows["gg_zzg"]["timing"]
    assert zzg["status"] == "measured" and zzg["batch"] == 128
    assert zzg["seconds_per_point"] == 0.020 and "other_backend" not in zzg
    assert zzg["gate"]["label"] == "no verdict yet"
    # No matched provider: no gate, no ratio.
    zzz = rows["gg_zzz"]["timing"]
    assert zzz["gate"]["label"] == "no matched provider: no gate"
    assert "ratio_to_fastest" not in zzz
    # A landed package without a record of the row: not run.
    assert rows["gg_hhh"]["timing"]["status"] == "not run"
    assert rows["gg_hhh"]["providers"]["status"] == "not run"
    # lit-perf's verdict becomes Table 1's chip (inconclusive -> open).
    zzg128 = timing["records"][6]
    assert (zzg128["key"], zzg128["batch"]) == ("gg_zzg", 128)
    zzg128["gate"] = {
        "status": "inconclusive",
        "ratio_to_limit": 1.0,
        "band": [0.9, 1.1],
    }
    (data / "li-timing.json").write_text(json.dumps(timing))
    rows, _ = build()
    assert rows["gg_zzg"]["timing"]["gate"]["status"] == "open"
    # A row whose providers are not all unavailable but none timed yet: no
    # matched-provider exemption, no verdict.
    runs["gg_hhh"] = [run("gg_hhh", openloops="no library")]
    write_manifest(runs)
    timing["records"].append(rec("gg_hhh", "tirust", 0.040))
    (data / "li-timing.json").write_text(json.dumps(timing))
    rows, _ = build()
    hhh = rows["gg_hhh"]
    assert hhh["providers"]["status"] == "recorded"
    assert hhh["providers"]["providers"]["madloop"]["status"] == "not timed"
    assert hhh["timing"]["gate"]["label"] == "no provider time yet"
    # A record without the probes (off by default) is measured: the engine's
    # generation, no validation; one with them names their carrier from
    # their split (the engine's per-carrier passes, or lit-perf's double-
    # double "probe runs"); a record without its host is refused.
    generation = timing["records"][-1]["generation"]
    for split, carrier in (
        ({"double": 1.0, "probe_slice": 2.0}, "binary64"),
        ({"double": 1.0, "double-double": 2.0}, "binary64, double-double fallback"),
        ({"double-double": 2.0}, "double-double"),
        ({"probe runs (double-double)": 2.0, "probe slice": 1.0}, "double-double"),
        ({}, "no probe ran"),
    ):
        generation["probes_split_seconds"] = split
        (data / "li-timing.json").write_text(json.dumps(timing))
        rows, _ = build()
        assert rows["gg_hhh"]["generation"]["validation"]["carrier"] == carrier
    generation["probes_seconds"] = None
    (data / "li-timing.json").write_text(json.dumps(timing))
    rows, _ = build()
    off = rows["gg_hhh"]["generation"]
    assert off["status"] == "measured" and off["seconds"] == 12.0
    assert "validation" not in off
    del generation["probes_seconds"], generation["probes_split_seconds"]
    (data / "li-timing.json").write_text(json.dumps(timing))
    rows, _ = build()
    assert rows["gg_hhh"]["generation"]["status"] == "measured"
    assert "validation" not in rows["gg_hhh"]["generation"]
    del timing["records"][-1]["host"]
    (data / "li-timing.json").write_text(json.dumps(timing))
    with pytest.raises(ValueError, match="names no host"):
        build()
    timing["records"].pop()
    del runs["gg_hhh"]
    # Two production timings of one provider are ambiguous.
    runs["gg_hh"] = [run("gg_hh", [production, no_recola])]
    write_manifest(runs)
    with pytest.raises(ValueError, match="two production timings of madloop"):
        build()
    # A provider whose record excludes its values as a reference: timed and
    # shown, not a matched provider (lit-perf: never the fastest matched one
    # nor a limit), the ratio to it beside the cell.
    excluded = timing_record(
        "excluded.json",
        madloop=ml(0.04),
        openloops={"median_seconds_per_point": 0.01},
        recola={"median_seconds_per_point": 0.004, "values_excluded": "widths"},
    )
    runs["gg_hh"] = [run("gg_hh", [excluded])]
    write_manifest(runs)
    (data / "li-timing.json").write_text(json.dumps(timing))
    rows, out = build()
    prov = rows["gg_hh"]["providers"]
    assert prov["providers"]["recola"]["match"] == builder.LI_NOT_MATCHED
    assert prov["providers"]["recola"]["values_excluded"] == "widths"
    assert "match" not in prov["providers"]["openloops"]
    assert prov["fastest"]["key"] == "openloops" and "match" not in prov["fastest"]
    assert prov["fastest_unmatched"]["key"] == "recola"
    hh = rows["gg_hh"]["timing"]
    assert hh["ratio_to_fastest"] == pytest.approx(0.002 / 0.01)
    assert hh["ratio_to_fastest_unmatched"] == pytest.approx(0.002 / 0.004)
    drawn = {f["key"]: f for f in builder.li_figure(out, 1.3)["rows"]}
    assert drawn["gg_hh"]["not_matched"] == ["Recola"]
    assert drawn["gg_hgg"]["not_matched"] == []
    assert "fastest_unmatched" not in rows["gg_hgg"]["providers"]
    assert "ratio_to_fastest_unmatched" not in rows["gg_hgg"]["timing"]
    # Figure 1's gate limit: MadLoop and 1.3 x each matched provider (the
    # unmatched Recola gives none); none while a matched reference has no
    # production time yet.
    assert builder._li_limit(prov, 1.3) == {
        "seconds": pytest.approx(1.3 * 0.01),
        "binding": "OpenLoops",
    }
    pending = {
        "providers": {
            "madloop": {"seconds_per_point": 0.04},
            "openloops": {"status": "not timed", "reason": "no production timing yet"},
        }
    }
    assert builder._li_limit(pending, 1.3) is None
    runs["gg_hh"] = [run("gg_hh", [production, opp])]
    write_manifest(runs)
    # The batch is chosen before the backend: TIRust timed at B = 1 only and
    # NinjaRS at B = 128 give NinjaRS's B = 128 cell, labelled, no other
    # backend; without a B = 128 record at the newest revision no batched
    # cell (Table 1's 'no gate batch' / 'not batched', with the reason).
    batches = {
        "records": [
            rec("gg_hgg", "tirust", 0.009, batch=1),
            rec("gg_hgg", "ninjars", 0.006),
            rec("gg_zzg", "tirust", 0.060, batch=1),
            rec("gg_zzg", "tirust", 0.025, batch=16),
            rec("gg_zzg", "tirust", 0.020, revision="aaaaaaaa"),
            rec("gg_zzz", "tirust", 0.070, batch=1),
        ]
    }
    (data / "li-timing.json").write_text(json.dumps(batches))
    rows, _ = build()
    hgg = rows["gg_hgg"]["timing"]
    assert hgg["status"] == "measured" and hgg["backend"] == "ninjars"
    assert hgg["batch"] == 128 and not hgg["default_backend"]
    assert hgg["seconds_per_point"] == 0.006 and "other_backend" not in hgg
    assert rows["gg_zzg"]["timing"] == {
        "status": "no gate batch",
        "reason": "timed B = 1, 16 only",
        "timed_batches": [1, 16],
        "revision": "bbbbbbbb",
        "source": "li-timing.json records[gg_zzg, tirust, bbbbbbbb]",
    }
    assert rows["gg_zzg"]["generation"]["status"] == "measured"
    assert rows["gg_zzz"]["timing"]["status"] == "not batched"
    assert rows["gg_zzz"]["timing"]["reason"] == "timed B = 1 only"
    # Two records of one backend and batch at a revision are ambiguous.
    batches["records"].append(rec("gg_hgg", "ninjars", 0.007))
    (data / "li-timing.json").write_text(json.dumps(batches))
    with pytest.raises(ValueError, match="two records of gg_hgg"):
        build()
    (data / "li-timing.json").write_text(json.dumps(timing))
    # The registry: a 2->3 row cannot be 'reported', a backend record must
    # be one of the row's, and the manifest check stops on an unknown row.
    bad = _loop_induced_registry()
    bad["rows"][2]["gate_class"] = "reported"
    with pytest.raises(ValueError, match="four legs or fewer"):
        builder._li_registry({"loop_induced": bad})
    tir_only = _loop_induced_registry()
    tir_only["rows"][0]["backends"] = ["tirust"]
    ms["loop_induced"] = tir_only
    with pytest.raises(ValueError, match="not one of the row's"):
        build()
    ms["loop_induced"] = _loop_induced_registry(
        validation_only={"ids": ["LI-Z1", "LI-Z2"], "reason": "test"}
    )
    # lit-ref's manifest format: a row's reference runs (madloop_process)
    # and its free-text process list (annotations, 'or', 'vs'); one row may
    # carry several ids ("LI-Z1/Z2").
    manifest = tmp_path / builder.LI_CAMPAIGN
    rows_m = [
        {"row": f"LI-{r['key']}", "processes": f"{r['process']} [QCD] (t loop)"}
        for r in ms["loop_induced"]["rows"]
    ]
    rows_m[0] = {
        "row": "LI-gg_hh",
        "processes": "large m_t: g g > h g vs HEFT; g g > h h vs the LET",
    }
    rows_m[1] = {
        "row": "LI-aa_hh",
        "processes": "a a > h h h",
        "runs": [{"madloop_process": "a a > h h [sqrvirt=QED]"}],
    }
    identities = {"row": "LI-Z1/Z2", "processes": "zeros", "runs": []}
    manifest.write_text(json.dumps({"rows": [*rows_m, identities]}))
    _, out = build()
    campaign = out["meta"]["campaign"]
    assert campaign["checked"] is True
    assert (campaign["rows"], campaign["ids"]) == (len(rows_m) + 1, len(rows_m) + 2)
    # A process is an item of the text, not a prefix of a longer one.
    rows_m[0]["processes"] = "g g > h h h, g g > h h g"
    manifest.write_text(json.dumps({"rows": [*rows_m, identities]}))
    with pytest.raises(ValueError, match="'g g > h h' is not a process"):
        build()
    rows_m[0]["processes"] = "g g > h h or g g > h h g"
    manifest.write_text(
        json.dumps({"rows": [*rows_m, identities, {"row": "LI-99", "processes": ""}]})
    )
    with pytest.raises(ValueError, match="neither registry rows"):
        build()


def test_report_builder_loop_induced_untimed_row_prints_its_marker(tmp_path):
    """A registry row declared ``untimed`` (LI-X1: an execution check without
    a reference, USER 2026-11-06 ~14:55) prints its explicit marker with the
    registry's reason in both cells once its package has landed, and a row
    without any reference expects no provider: never 'pending'."""

    builder = _report_builder()
    registry = _loop_induced_registry()
    registry["groups"].append("EFT")
    registry["gate_classes"]["eft"] = "test"
    registry["landed"].append("lit-eft")
    registry["rows"].append(
        {
            "key": "li_x1",
            "process": None,
            "ids": ["LI-X1"],
            "corrections": "EFT",
            "legs": None,
            "group": "EFT",
            "gate_class": "eft",
            "served_by": "lit-eft",
            "backends": ["tirust"],
            "references": [],
            "label": "rank > N+2 EFT row",
            "untimed": "execution check only (test)",
        }
    )
    ms = _loop_induced_milestone(registry)
    data = tmp_path / "data"
    data.mkdir()
    recency = builder.Recency(tmp_path, ms["ancestry"])
    out = builder.build_loop_induced(ms, recency, data=data, repo=tmp_path)
    row = {r["key"]: r for g in out["groups"] for r in g["rows"]}["li_x1"]
    marker = {"status": "not timed", "reason": "execution check only (test)"}
    assert row["generation"] == marker and row["timing"] == marker
    assert row["providers"] == {
        "status": "no reference",
        "reason": "no reference provider",
        "expected": [],
    }
    # Not landed: pending like every other row of a package not landed.
    registry["landed"].remove("lit-eft")
    out = builder.build_loop_induced(ms, recency, data=data, repo=tmp_path)
    row = {r["key"]: r for g in out["groups"] for r in g["rows"]}["li_x1"]
    assert row["timing"]["status"] == "pending"


def test_report_builder_backend_table_and_missing_box(tmp_path):
    """The backend-comparison table (m-lit Q40): TIRust and NinjaRS under
    the unchecked setting per loop-induced row (time, stages grouped with the
    backend's own reduction stages, ratio, accuracy, the L6 verdict of
    lit.json) and per Born-interfered 2->3 / 2->4 row of Table 1 (the
    records of backends-born.json, else the 'not timed' marker), LI-B1's L6
    tally; the "Missing or incomplete" box lists the cells without a value,
    a failing gate with its dominating stage, a partial validation (its
    partial checks and TIRust's failed fixture points) and an accuracy note,
    rows sharing a reason on one line."""

    import json

    builder = _report_builder()
    ti_stages = {
        "inputs": 0.1,
        "numerator": 0.5,
        "scatter": 0.1,
        "moments": 0.2,
        "contraction": 0.05,
        "glue": 0.05,
        "untimed": 0.0,
    }
    nj_stages = {
        "inputs": 0.1,
        "numerator": 0.5,
        "scatter": 0.1,
        "ninjars": 0.6,
        "masters": 0.2,
        "glue": 0.05,
        "untimed": 0.0,
    }
    acc = {
        "points": 128,
        "max_relative": 2e-3,
        "lax": 1e-5,
        "above": 3,
        "worst": 2e-3,
        "served_by": ["last resort"],
        "failed": 0,
        "failed_classes": [],
        "note": True,
    }
    li_row = {
        "key": "gg_hgg",
        "math": "g g -> h g g",
        "label": "g g > h g g",
        "generation": {"status": "measured"},
        "providers": {"status": "recorded", "fastest": {"key": "openloops"}},
        "validation": {
            "status": "partial",
            "note": "li_gg_hgg: L3 partial (x); li_gg_hgg: L6 partial"
            " (TIRust fails 2 of 3 binary64 points closed (typed))",
        },
        "timing": {
            "status": "measured",
            "unchecked": True,
            "backend": "tirust",
            "backend_label": "TIRust",
            "seconds_per_point": 1.0,
            "revision": "aaaaaaaa",
            "stages": ti_stages,
            "accuracy": acc,
            "source": "test",
            "gate": {
                "status": "fail",
                "ratio_to_limit": 1.2,
                "backend_label": "TIRust",
            },
            "other_backend": {
                "backend": "ninjars",
                "backend_label": "NinjaRS per amplitude",
                "status": "served",
                "seconds_per_point": 2.5,
                "stages": nj_stages,
            },
        },
    }
    untimed = {
        "key": "li_x1",
        "math": None,
        "label": "EFT row",
        "generation": {"status": "not timed", "reason": "execution check only"},
        "timing": {"status": "not timed", "reason": "execution check only"},
        "providers": {"status": "no reference"},
        "validation": {"status": "pass"},
    }
    loop_induced = {
        "groups": [
            {"name": "2->3", "rows": [li_row]},
            {"name": "EFT", "rows": [untimed]},
        ]
    }

    def g1(key, math, gate=None, lc="measured"):
        return {
            "key": key,
            "math": math,
            "generation": {"main": {"status": "measured"}},
            "leading_colour": {"main": {"status": lc, "reason": "refused by design"}},
            "full_colour": {
                "main": {"status": "measured", "gate": gate, "stages": nj_stages},
                "gate_rule": "G1",
            },
        }

    summary = {
        "groups": [
            {"name": "2->2", "rows": [g1("gg_gg", "g g -> g g")]},
            {
                "name": "2->3",
                "rows": [
                    g1(
                        "gg_ggg",
                        "g g -> g g g",
                        {"status": "open", "ratio_to_limit": 1.0},
                    ),
                    g1("uux_ggg", "u overline(u) -> g g g", lc="not supported"),
                ],
            },
        ]
    }
    data = tmp_path / "data"
    data.mkdir()
    (data / "lit.json").write_text(
        json.dumps(
            {
                "rows": {
                    "gg_hgg": {
                        "runs": {
                            "li_gg_hgg": {
                                "L6": {
                                    "double": {
                                        "status": "partial",
                                        "tirust_served": 1,
                                        "of": 3,
                                        "finite_max_relative": 3e-12,
                                    }
                                }
                            }
                        }
                    }
                },
                "ids": {
                    "LI-B1": {
                        "status": "partial",
                        "campaigns": {
                            "m": {
                                "rows": {
                                    "A": {"status": "pass"},
                                    "B": {"status": "partial"},
                                    "C": {"status": "pass"},
                                }
                            }
                        },
                    }
                },
            }
        )
    )
    table = builder.build_backend_table(summary, loop_induced, data=data)
    (li,) = table["loop_induced"]["rows"]
    assert li["ratio"] == 2.5 and li["tirust"]["accuracy"] == acc
    assert (
        li["tirust"]["stages"]["moments"] == 0.2
        and "ninjars" in li["ninjars"]["stages"]
    )
    assert li["l6"]["runs"] == [
        {
            "run": "li_gg_hgg",
            "status": "partial",
            "tirust_served": 1,
            "of": 3,
            "finite_max_relative": 3e-12,
        }
    ]
    assert table["loop_induced"]["not_unchecked"] == ["li_x1"]
    born = table["born_interfered"]
    assert [r["key"] for r in born["rows"]] == ["gg_ggg", "uux_ggg"]
    assert born["l6"]["rows"] == {"pass": 2, "partial": 1} and not born["present"]
    assert all(
        r["tirust"]["status"] == r["ninjars"]["status"] == "not timed"
        for r in born["rows"]
    )
    (data / "backends-born.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "key": "gg_ggg",
                        "backend": "tirust",
                        "seconds_per_point": 0.01,
                        "revision": "b" * 40,
                        "stages_per_point": {
                            "reduction": 0.004,
                            "reduction: tensor moments": 0.003,
                            "reduction: contraction": 0.0005,
                            "dressing": 0.005,
                        },
                        "agreement": {"max_relative": 1e-12, "points": 128},
                    },
                    {
                        "key": "gg_ggg",
                        "backend": "ninjars",
                        "seconds_per_point": 0.02,
                        "revision": "b" * 40,
                        "stages_per_point": {
                            "reduction": 0.012,
                            "reduction: masters": 0.002,
                            "dressing": 0.005,
                        },
                    },
                    {
                        "key": "uux_ggg",
                        "backend": "tirust",
                        "missing": "outside the row budget",
                    },
                ]
            }
        )
    )
    born = builder.build_backend_table(summary, loop_induced, data=data)[
        "born_interfered"
    ]
    ggg = born["rows"][0]
    assert ggg["ratio"] == 2.0 and ggg["agreement"]["max_relative"] == 1e-12
    assert ggg["tirust"]["stages"]["moments"] == 0.003
    assert ggg["tirust"]["stages"]["glue"] == pytest.approx(0.0005)
    assert ggg["ninjars"]["stages"]["ninjars"] == 0.01
    assert born["rows"][1]["tirust"] == {
        "status": "not run",
        "reason": "outside the row budget",
    }
    missing = builder.build_missing(summary, loop_induced)
    texts = {(e["table"], e["kind"], e["text"]): e for e in missing}
    assert (
        "1",
        "no value",
        "leading colour: not supported (refused by design)",
    ) in texts
    gate1 = texts[("1", "gate", "open")]
    assert (
        gate1["rows"] == ["g g -> g g g"] and gate1["dominant"][0]["stage"] == "ninjars"
    )
    gate2 = texts[("2", "gate", "fail on TIRust")]
    assert gate2["ratio"] == [1.2] and gate2["dominant"][0]["stage"] == "numerator"
    assert (
        "2",
        "validation",
        "partial: L3, L6; TIRust fails 2 of 3 binary64 fixture points"
        " closed (checked default; lit.json)",
    ) in texts
    assert (
        "2",
        "accuracy",
        "TIRust unchecked: 3/128 points above 1e-05, worst 2.0e-03",
    ) in texts
    assert ("2", "no value", "not timed: execution check only") in texts
    assert [e["table"] for e in missing] == sorted(e["table"] for e in missing)


def test_report_builder_missing_box_labels_a_cross_backend_difference_once():
    """A row without a 256-bit truth (lit-report R2) gives ONE labelled
    Missing-box line for the two backends' difference, never an error line
    per backend."""

    builder = _report_builder()
    cross = {
        "points": 128,
        "max_relative": 7e-2,
        "lax": 1e-5,
        "above": 23,
        "worst": 7e-2,
        "served_by": [],
        "failed": None,
        "not_compared": 0,
        "failed_classes": [],
        "note": True,
        "cross_backend": True,
        "truth_at_points": {
            "points": [104, 107],
            "tirust_max_relative": 1.3e-4,
            "ninjars_max_relative": 7.6e-2,
        },
    }
    row = {
        "math": "g g -> a a g g",
        "label": "li_gg_aagg",
        "generation": {"status": "measured"},
        "timing": {
            "status": "measured",
            "backend_label": "TIRust",
            "accuracy": cross,
            "other_backend": {"backend_label": "NinjaRS", "accuracy": dict(cross)},
        },
        "providers": {"status": "no reference"},
        "validation": {"status": "pass"},
    }
    missing = builder.build_missing({"groups": []}, {"groups": [{"rows": [row]}]})
    acc = [e for e in missing if e["kind"] == "accuracy"]
    assert len(acc) == 1 and acc[0]["rows"] == ["g g -> a a g g"]
    assert "cross-backend, not a truth" in acc[0]["text"]
    assert "TIRust unchecked" not in acc[0]["text"]
    assert acc[0]["text"].endswith(
        "at its 2 largest-difference points against the 256-bit truth:"
        " TIRust 1.3e-04, NinjaRS 7.6e-02"
    )


def test_report_builder_missing_box_other_cell_and_born_note():
    """The Missing box names the other backend's cell without a value when
    no accuracy line names its failures (g g > z z z z's NinjaRS), and a
    Born-interfered TIRust-vs-NinjaRS difference above the lax precision
    (Table 3), labelled cross-backend."""

    builder = _report_builder()
    row = {
        "math": "g g -> z z z z",
        "label": "li_gg_zzzz",
        "generation": {"status": "measured"},
        "timing": {
            "status": "measured",
            "backend_label": "TIRust",
            "other_backend": {
                "backend_label": "NinjaRS",
                "status": "failed closed",
                "label": "16/128 failed closed, lane status Unstable",
                "accuracy": {
                    "cross_backend": True,
                    "note": False,
                    "failed": None,
                },
            },
        },
        "providers": {"status": "no reference"},
        "validation": {"status": "pass"},
    }
    born = {
        "rows": [
            {
                "math": "g g -> g g g",
                "agreement": {
                    "max_relative": 2.1e-3,
                    "points": 128,
                    "lax": 1e-5,
                    "note": True,
                },
            },
            {
                "math": "u u~ -> g g g",
                "agreement": {
                    "max_relative": 1.8e-9,
                    "points": 128,
                    "lax": 1e-5,
                    "note": False,
                },
            },
        ]
    }
    loop_induced = {
        "groups": [{"rows": [row]}],
        "backends": {"born_interfered": born},
    }
    missing = builder.build_missing({"groups": []}, loop_induced)
    texts = {(e["table"], e["kind"], e["text"]): e["rows"] for e in missing}
    other = "NinjaRS time: failed closed (16/128 failed closed, lane status Unstable)"
    assert texts[("2", "no value", other)] == ["g g -> z z z z"]
    born_lines = [e for e in missing if e["table"] == "3"]
    assert len(born_lines) == 1 and born_lines[0]["rows"] == ["g g -> g g g"]
    assert "cross-backend, not a truth" in born_lines[0]["text"]
    assert "2.1e-03" in born_lines[0]["text"]


def test_report_builder_accuracy_line_without_a_truth_is_the_labelled_cross_backend_difference():
    """A row whose 256-bit truth is not obtainable within its row budget
    (lit-report R2) carries the two backends' difference at the timed
    points, labelled, with the 1e-5 note rule; a row with a truth keeps
    the truth's line per backend."""

    builder = _report_builder()
    acc = {
        "lax": 1e-5,
        "rows": {
            "li_gg_aagg": {
                "points": 128,
                "revision": "bbbbbbbb",
                "cross_backend": {
                    "points": 126,
                    "failed": 2,
                    "max_relative": 3e-5,
                    "above_1e-5": 1,
                    "reason": "256-bit truth 405 s per point",
                },
            },
            "li_gg_hh": {
                "points": 128,
                "revision": "bbbbbbbb",
                "tirust": {"max_relative": 1e-9, "above_1e-5": [], "failed": []},
            },
        },
    }
    for backend in ("tirust", "ninjars"):
        line = builder._li_accuracy(acc, "li_gg_aagg", backend)
        assert line["cross_backend"] and line["note"] and line["above"] == 1
        assert line["worst"] == 3e-5 and line["not_compared"] == 2
        assert line["failed"] is None
        assert (
            line["source"]
            == "li-unchecked-accuracy.json rows[li_gg_aagg].cross_backend"
        )
    truth = builder._li_accuracy(acc, "li_gg_hh", "tirust")
    assert truth["max_relative"] == 1e-9 and "cross_backend" not in truth
    assert builder._li_accuracy(acc, "li_gg_hh", "ninjars") is None


def test_report_builder_figure_one_draws_loop_induced_rows_with_a_time_and_a_provider():
    """Figure 1's loop-induced rows (USER 2026-11-07 ~10:40): every row of the
    loop-induced table with a batched pyAmpliCol time and a matched provider
    time, in the table's order, with its providers' times and, from 2 -> 3
    upward, the gate limit (MadLoop's time and the gate factor times each
    other matched provider's, the smallest; none while a matched reference
    has no production time yet); the other rows are not drawn, listed by
    reason. The committed summary-li.json holds the figure of its own rows."""

    import json

    builder = _report_builder()

    def row(key, gate_class, timing, providers=None):
        recorded = providers is not None
        timed = {k: {"seconds_per_point": t} for k, t in (providers or {}).items()}
        fastest = min(timed, key=lambda k: timed[k]["seconds_per_point"], default=None)
        return {
            "key": key,
            "math": key,
            "corrections": "QCD",
            "gate_class": gate_class,
            "timing": timing,
            "providers": {
                "status": "recorded" if recorded else "not run",
                "providers": {**timed, "recola": {"status": "not timed"}}
                if recorded and "recola" not in timed
                else timed,
                "fastest": None if fastest is None else {"key": fastest},
            },
        }

    def measured(t, providers, gate):
        cell = {
            "status": "measured",
            "seconds_per_point": t,
            "backend_label": "TIRust",
            "gate": gate,
        }
        if providers:
            cell["ratio_to_fastest"] = t / min(providers.values())
        return cell

    reported = {"status": "reported", "label": "no gate (2->2)"}
    verdict = {"status": "no verdict", "label": "no verdict yet"}
    small = {"status": "not gated", "label": "<5 ms: not gated"}
    table = {
        "groups": [
            {
                "name": "2->2",
                "rows": [
                    row(
                        "hh",
                        "reported",
                        measured(0.002, {"madloop": 0.001}, reported),
                        {"madloop": 0.001},
                    ),
                    row("zz", "reported", measured(0.03, {}, reported)),
                    row("aa", "reported", {"status": "pending", "reason": "lit-x"}),
                ],
            },
            {
                "name": "2->3",
                "rows": [
                    row(
                        "hhh",
                        "gated",
                        measured(0.007, {"x": 0.0056}, verdict),
                        {"madloop": 0.009, "recola": 0.0056, "openloops": 0.0229},
                    ),
                    row(
                        "hgg",
                        "gated",
                        measured(0.003, {"x": 0.0035}, small),
                        {"madloop": 0.004, "openloops": 0.0035},
                    ),
                    row("zzg", "gated", {"status": "no gate batch", "reason": "B = 1"}),
                ],
            },
        ]
    }
    figure = builder.li_figure(table, 1.3)
    assert [f["key"] for f in figure["rows"]] == ["hh", "hhh", "hgg"]
    hh, hhh, hgg = figure["rows"]
    assert hh["limit"] is None and hh["gate"] == reported
    assert hh["providers"] == {"madloop": 0.001} and hh["fastest"] == "madloop"
    assert hhh["limit"] == {"seconds": pytest.approx(1.3 * 0.0056), "binding": "Recola"}
    assert hhh["providers"] == {"madloop": 0.009, "recola": 0.0056, "openloops": 0.0229}
    # Recola, a reference of the row, has no production time yet: no
    # complete gate limit (Table 1's rule).
    assert hgg["limit"] is None
    assert hgg["gate"]["status"] == "not gated" and hgg["group"] == "2->3"
    assert figure["not_drawn"] == {
        builder.LI_NO_TIME: ["aa", "zzg"],
        builder.LI_NO_PROVIDER: ["zz"],
    }
    # The committed data: the figure is the one of the committed table.
    data = ROOT / "docs" / "performance_reports" / "loop" / "data"
    committed = json.loads((data / "summary-li.json").read_text())
    factor = json.loads((data / "summary.json").read_text())["meta"]["gate_factor"]
    assert committed["figure"] == builder.li_figure(committed, factor)


def test_report_builder_loop_induced_cell_is_the_auto_record_never_a_failed_one(
    tmp_path,
):
    """The loop-induced cell is the record of the backend "auto" serves at
    the row's newest revision (m-lit DECISIONS 2026-11-07 21:50 R9: "auto"
    is NinjaRS until lit-cache's switch), whatever the registry's default
    and whichever backend is faster; it is never a record with failed
    points: a TIRust call that fails points closed (ruling 17:20 (d)) is
    timed whole, carries no time per point and is shown beside the cell
    with its served fraction and recorded causes only, its time per point
    only where it served every point; an "auto" record that failed points
    leaves the row a marker, never another backend's record; a refused
    request is named with its reason. Inconsistent records are refused:
    an explicit request run on another backend, a time per point on a
    record with failed points, a served fraction its failed points do not
    give. The figure draws the cell's backend only."""

    import json

    builder = _report_builder()
    ms = _loop_induced_milestone(_loop_induced_registry())
    data = tmp_path / "data"
    data.mkdir()
    gen = {"seconds": 12.0, "probes_seconds": 3.0, "peak_gib": 0.5}
    ruling_d = "ruling (d): conditioning-gate refusals, method choice off"

    def rec(key, requested, backend, *, failed=None, seconds=None, **extra):
        r = {
            "key": key,
            "revision": "bbbbbbbb",
            "requested": requested,
            "backend": backend,
            "effective": backend,
            "batch": 128,
            "host": "testhost",
            "failed_points": failed or [0] * 7,
            "source": f"test/{key}",
            **extra,
        }
        if seconds is not None:
            r["seconds_per_point"] = seconds
        if failed and any(failed):
            r["served_fraction"] = 1 - sum(failed) / (128 * len(failed))
            r["whole_call_seconds_per_point"] = 0.001
        return r

    auto = "auto: NinjaRS; TIRust ... on explicit request"
    records = [
        rec("gg_hh", "auto", "ninjars", seconds=0.002, generation=gen, reason=auto),
        rec(
            "gg_hh", "tirust", "tirust", failed=[43] * 7, failure_causes={ruling_d: 43}
        ),
        rec("gg_hgg", "auto", "ninjars", seconds=0.04, generation=gen, reason=auto),
        rec(
            "gg_hgg",
            "tirust",
            "tirust",
            failed=[128] * 7,
            failure_causes={ruling_d: 128},
        ),
        # TIRust faster and served: the cell stays "auto"'s.
        rec("gg_zzg", "auto", "ninjars", seconds=0.5, generation=gen, reason=auto),
        rec("gg_zzg", "tirust", "tirust", seconds=0.4),
        # "auto" itself failed points: a marker, not the served TIRust record.
        rec(
            "gg_hhh",
            "auto",
            "ninjars",
            failed=[3] * 7,
            generation=gen,
            failure_causes={"a lane status": 3},
        ),
        rec("gg_hhh", "tirust", "tirust", seconds=0.01),
        rec("gg_zzz", "auto", "ninjars", seconds=0.3, generation=gen, reason=auto),
        rec("gg_zzz", "tirust", "tirust", refused="ReductionUnavailable: rank"),
    ]
    (data / "li-timing.json").write_text(json.dumps({"records": records}))
    recency = builder.Recency(tmp_path, ms["ancestry"])
    out = builder.build_loop_induced(ms, recency, data=data, repo=tmp_path)
    rows = {r["key"]: r for g in out["groups"] for r in g["rows"]}

    hgg = rows["gg_hgg"]["timing"]
    assert hgg["status"] == "measured" and hgg["backend"] == "ninjars"
    assert hgg["backend_label"] == "NinjaRS per amplitude"
    assert hgg["requested"] == "auto" and hgg["reason"] == auto
    assert hgg["seconds_per_point"] == 0.04 and not hgg["default_backend"]
    other = hgg["other_backend"]
    assert other["status"] == "failed closed" and other["every_point"]
    assert other["failed_points"] == [128, 128] and other["served_fraction"] == 0.0
    assert other["causes"] == {ruling_d: 128} and "seconds_per_point" not in other
    assert other["label"] == f"failed closed on every point (128/128), {ruling_d}"
    # The engine's generation; the probes' 3 s apart (lit-probe).
    assert rows["gg_hgg"]["generation"]["seconds"] == 12.0
    assert rows["gg_hgg"]["generation"]["validation"]["seconds"] == 3.0
    hh = rows["gg_hh"]["timing"]["other_backend"]
    assert hh["failed_points"] == [43, 43] and not hh["every_point"]
    assert hh["served_fraction"] == pytest.approx(85 / 128)
    assert hh["label"] == f"43/128 failed closed, {ruling_d}"
    zzg = rows["gg_zzg"]["timing"]
    assert zzg["backend"] == "ninjars" and zzg["seconds_per_point"] == 0.5
    assert zzg["other_backend"] == {
        "backend": "tirust",
        "backend_label": "TIRust",
        "requested": "tirust",
        "status": "served",
        "served_fraction": 1.0,
        "label": "128/128",
        "seconds_per_point": 0.4,
    }
    hhh = rows["gg_hhh"]["timing"]
    assert hhh["status"] == "failed closed" and hhh["requested"] == "auto"
    assert hhh["failed_points"] == [3, 3] and "seconds_per_point" not in hhh
    assert rows["gg_hhh"]["generation"]["status"] == "measured"
    zzz = rows["gg_zzz"]["timing"]["other_backend"]
    assert zzz["status"] == "refused" and zzz["reason"] == "ReductionUnavailable: rank"
    figure = builder.li_figure(out, 1.3)
    assert all(f["backend_label"] != "TIRust" for f in figure["rows"])

    def refused(record, match):
        (data / "li-timing.json").write_text(json.dumps({"records": [record]}))
        with pytest.raises(ValueError, match=match):
            builder.build_loop_induced(ms, recency, data=data, repo=tmp_path)

    # A row not generated or timed: its measured reason on both cells; the
    # run stopped before its request was resolved names no backend.
    why = "generation beyond the 45-min rule (probes running at 45 min)"
    stopped = rec("gg_zzz", "auto", "ninjars", missing=why)
    del stopped["backend"], stopped["effective"], stopped["failed_points"]
    (data / "li-timing.json").write_text(json.dumps({"records": [stopped]}))
    out = builder.build_loop_induced(ms, recency, data=data, repo=tmp_path)
    rows = {r["key"]: r for g in out["groups"] for r in g["rows"]}
    zzz = rows["gg_zzz"]
    assert zzz["timing"]["status"] == zzz["generation"]["status"] == "not run"
    assert zzz["timing"]["reason"] == zzz["generation"]["reason"] == why
    assert zzz["timing"]["requested"] == "auto"
    assert zzz["timing"]["effective"] == "not resolved"
    assert "backend" not in zzz["timing"]
    assert zzz["timing"]["source"] == "li-timing.json records[gg_zzz, auto, bbbbbbbb]"
    refused(rec("gg_zzz", "auto", "ninjars", missing=why), "resolved no backend")
    refused(rec("gg_hgg", "tirust", "ninjars", seconds=0.04), "was requested but")
    refused(
        rec("gg_hgg", "tirust", "tirust", failed=[1] * 7, seconds=0.04),
        "has no time per point",
    )
    refused(rec("gg_hgg", "tirust", "tirust"), "without its time per point")
    bad = rec("gg_hgg", "tirust", "tirust", failed=[64] * 7)
    bad["served_fraction"] = 1.0
    refused(bad, "served fraction")


def test_madloop_columns_share_the_production_run_setup(tmp_path, monkeypatch):
    # The production column and the one-helicity column build and prepare
    # their run directories alike (Born-less driver for a loop-induced
    # process); the toolkit is replaced by stand-ins.
    import contextlib
    import types

    built = []

    class References:
        def madloop_build_flags(self, export, mg5):
            return {"production_build": True, "export_make_opts": {"GLOBAL_FLAG": "-O"}}

        def build_madloop(self, export, run, flag):
            built.append(("born-interfered", run.name, flag))
            return run / "driver", []

        def point_lines(self, points):
            return ["point"] * len(points["points"])

        def run_native(self, command, stdin, run, log, cpu):
            return "", 1.0

        def parse_rows(self, stdout, provider, count):
            return [{"seconds": 0.5}] * count, {}

        def madloop_helicities(self, export):
            return [(1, 1), (1, -1)]

        def madloop_process_dir(self, export):
            return export

    class Li:
        def build_madloop(self, export, run, flag):
            built.append(("loop-induced", run.name, flag))
            return run / "driver", ["repair"]

        def helavg_factor(self, process):
            return 4, None

        def parse_madloop(self, stdout, count, helavg):
            kept = {"seconds": 0.1, "value": 1.0}
            filtered = {"seconds": 0.0, "value": 0.0}
            return [{"seconds": 0.5, "per_helicity": [kept, filtered]}] * count, None

    card = {"MadLoopParams": {"MLReductionLib": "7|6|1"}}
    monkeypatch.setattr(providers, "_references", References)
    monkeypatch.setattr(providers, "_li_references", Li)
    monkeypatch.setattr(providers, "_relocated_madloop_build", contextlib.nullcontext)
    monkeypatch.setattr(
        providers,
        "_prepare_madloop",
        lambda *args: (card, {"on": True, "reason": "test"}),
    )
    monkeypatch.setattr(
        providers, "_madloop_header", lambda spec, run, count: f"{count}\n"
    )
    trees = types.SimpleNamespace(mg5=tmp_path, root=tmp_path)
    points = {"points": [[], []]}
    loop_induced = {"loop_induced": True}
    result = providers.madloop(loop_induced, points, tmp_path / "a", 0, tmp_path, trees)
    one = providers.madloop_one_helicity(
        loop_induced, points, tmp_path / "a", 0, tmp_path, trees
    )
    providers.madloop({}, points, tmp_path / "b", 0, tmp_path, trees)
    assert built == [
        ("loop-induced", "madloop-run", "-O"),
        ("loop-induced", "madloop-one-helicity-run", "-O"),
        ("born-interfered", "madloop-run", "-O"),
    ]
    assert (
        result["qp_model_repairs"] == ["repair"]
        and result["madloop_params"] == card["MadLoopParams"]
    )
    assert (one["calls"], one["helicities"], one["kept_helicities_per_point"]) == (
        2,
        2,
        [1, 1],
    )
    assert one["madloop_params"] == card["MadLoopParams"]


def test_provider_records_carry_the_specification_value_annotations(
    tmp_path, monkeypatch
):
    # A provider whose values the specification excepts at some stream
    # points (or excludes) keeps that annotation beside its timing.
    import json
    import types

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(providers, "Trees", lambda options: None)
    monkeypatch.setattr(
        providers,
        "recola",
        lambda *args: {"provider": "recola", "mean_seconds_per_point": 1.0},
    )
    options = types.SimpleNamespace(
        process="li_gg_zzzz",
        points=2,
        work=tmp_path / "work",
        json=tmp_path / "record.json",
        cpu=0,
        providers=["recola"],
    )
    assert providers.main(options) == 0
    entry = json.loads(options.json.read_text())["providers"]["recola"]
    assert (
        entry["value_exceptions"]
        == streams.spec("li_gg_zzzz")["recola"]["value_exceptions"]
    )
    assert entry["value_exceptions"]["points"] == [7, 13]


def test_the_most_optimistic_protocol_sets_each_provider_up_for_speed(
    tmp_path, monkeypatch
):
    # m-lit F4: OpenLoops without its stability system, MadLoop without its
    # stability test, Recola without dynamic settings (alpha_s and the scales
    # of the generation, no per-point setter in the timer), its build
    # recorded; loop-induced rows only.
    import json
    import types

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        providers,
        "Trees",
        lambda options: types.SimpleNamespace(
            madloop_export=lambda spec, override: tmp_path
        ),
    )
    seen = {}

    def provider(name):
        def run(*args):
            seen[name] = args
            return {"provider": name, "mean_seconds_per_point": 1.0}

        return run

    monkeypatch.setattr(providers, "openloops", provider("openloops"))
    monkeypatch.setattr(providers, "recola", provider("recola"))
    monkeypatch.setattr(providers, "madloop", provider("madloop"))
    options = types.SimpleNamespace(
        process="li_gg_hh",
        points=2,
        work=tmp_path / "work",
        json=tmp_path / "record.json",
        cpu=0,
        providers=["openloops", "recola", "madloop"],
        protocol="optimistic",
        madloop_export=None,
        madloop_param=["MLReductionLib=6|7|1"],
        madloop_stages=False,
    )
    assert providers.main(options) == 0
    assert json.loads(options.json.read_text())["protocol"] == "optimistic"
    assert seen["openloops"][-1] == seen["recola"][-1] == "optimistic"
    assert seen["madloop"][-1] == {"CTModeRun": "1", "MLReductionLib": "6|7|1"}
    # Born-interfered rows (lit-report G5): the same settings; Recola, whose
    # scale is the generation's, times the stream points at the first
    # point's mu_R (their indices recorded); a production record of the
    # same tool checks every provider's values at the timed points.
    seen.clear()
    born = types.SimpleNamespace(
        **{**vars(options), "process": "gg_ggg", "points": 4, "madloop_param": []}
    )

    def recola(spec, points, *args):
        seen["recola"] = (points, *args)
        return {
            "provider": "recola",
            "mean_seconds_per_point": 1.0,
            "rows": [{"finite": 2.0 + 1e-9}] * len(points["points"]),
        }

    def rows(name, values):
        def run(*args):
            seen[name] = args
            return {
                "provider": name,
                "mean_seconds_per_point": 1.0,
                "rows": [{"finite": v} for v in values],
            }

        return run

    monkeypatch.setattr(providers, "recola", recola)
    monkeypatch.setattr(providers, "openloops", rows("openloops", [1.0, 2.0, 3.0, 4.0]))
    monkeypatch.setattr(providers, "madloop", rows("madloop", [1.0, 2.0, 3.0, 4.5]))
    production = tmp_path / "production.json"
    production.write_text(
        json.dumps(
            {
                "providers": {
                    name: {"rows": [{"finite": v} for v in (2.0, 9.0, 9.0, 2.0)]}
                    for name in ("recola", "openloops", "madloop")
                }
            }
        )
    )
    born.production_json = production
    born.json = tmp_path / "born.json"
    assert providers.main(born) == 0
    stream = streams.load("gg_ggg")
    first = stream["mu_r"][0]
    kept = [i for i in range(4) if stream["mu_r"][i] == first]
    assert seen["recola"][0]["mu_r"] == [first] * len(kept)
    assert seen["openloops"][-1] == "optimistic" and seen["recola"][-1] == "optimistic"
    assert seen["madloop"][-1] == {"CTModeRun": "1"}
    record = json.loads(born.json.read_text())["providers"]
    assert record["recola"]["stream_points"] == kept == [0, 3]
    assert record["recola"]["production_check"]["passed"]
    assert record["recola"]["production_check"]["points"] == 2
    assert not record["openloops"]["production_check"]["passed"]
    assert record["madloop"]["production_check"]["max_relative"] == pytest.approx(7 / 9)
    # The same through the command line (--production-json).
    from tools.loop_benchmark.__main__ import main as cli

    argv = ["providers", "gg_ggg", "--cpu", "0", "--points", "4"]
    argv += ["--work", str(tmp_path / "cli"), "--protocol", "optimistic"]
    argv += ["--production-json", str(production), "--json", str(tmp_path / "c.json")]
    assert cli(argv) == 0
    again = json.loads((tmp_path / "c.json").read_text())["providers"]
    assert again["recola"]["production_check"] == record["recola"]["production_check"]
    assert providers.same_scale(
        {"points": [1, 2, 3], "mu_r": ["7", "5", "7"], "pdgs": []}
    ) == ({"points": [1, 3], "mu_r": ["7", "7"], "pdgs": []}, [0, 2])

    # The Recola driver: dynamic settings and the per-point setters follow
    # the ``dynamic`` command (1, the production default, keeps them).
    built = []
    monkeypatch.setattr(
        providers.subprocess, "run", lambda command, **kw: built.append(command)
    )
    providers._recola_driver(tmp_path, tmp_path, [])
    source = (tmp_path / "recola_driver_production.f90").read_text()
    assert "integer :: dynamic = 1" in source
    assert "call set_dynamic_settings_rcl(dynamic)" in source
    assert "set_dynamic_settings_rcl(1)" not in source
    assert (
        "    if (dynamic == 1) then\n"
        "      call set_alphas_rcl(stored_alpha_s, scales(i), stored_nf)\n"
    ) in source

    # The build of the library a Recola run links, from the root's CMake builds.
    root, tree = tmp_path / "root", tmp_path / "root/sources/recola"
    build = root / "builds/recola"
    (build / "src/CMakeFiles/recola.dir").mkdir(parents=True)
    tree.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text(
        f"//comment\nCMAKE_BUILD_TYPE:STRING=\nCMAKE_HOME_DIRECTORY:INTERNAL={tree}\n"
    )
    (build / "src/CMakeFiles/recola.dir/flags.make").write_text(
        "Fortran_FLAGS = -cpp -J/abs/include -fPIC\n"
        "# Custom flags: src/CMakeFiles/recola.dir/a.f90.o_FLAGS = -O2\n"
    )
    assert providers.cmake_build(tree, root, "recola") == {
        "build_dir": "builds/recola",
        "build_type": "(none)",
        "fortran_flags": "-cpp -fPIC",
        "source_flags": ["-O2"],
    }
    assert providers.cmake_build(tmp_path, root, "recola") is None


def test_a_provider_subset_of_the_stream_keeps_its_stream_indices():
    """lit-report G5: a provider timing a subset of the stream (Recola's
    Born-interfered most optimistic arm, ``stream_points``) is compared with
    pyAmpliCol and written to the values document at those stream points,
    not at the first n."""

    born = [{"born": 1.0, "finite": float(i), "seconds": 0.1} for i in (0, 3)]
    record = {
        "providers": {
            "openloops": {"rows": [{"born": 1.0, "finite": 9.0}] * 2},
            "madloop": {"rows": born, "stream_points": [0, 3]},
        }
    }
    values = gate._provider_values("gg_ggg", record)
    assert sorted(values["madloop"]) == [0, 3]
    assert values["madloop"][3] == (1.0, 3.0)
    assert sorted(values["openloops"]) == [0, 1]
    assert list(gate.stream_indices({"rows": [{}, {}, {}]})) == [0, 1, 2]
    assert gate.stream_indices({"rows": [{}], "stream_points": [5]}) == [5]


def test_one_helicity_arms_select_one_row_for_one_call():
    """PLAN §5.1 (Q39): a loop-induced one-helicity arm dresses the plan's
    first helicity row alone (``NativeExecutor.row_mask``) for its calls and
    restores every row after; its label and record name the row."""

    import types

    from tools.loop_benchmark import pyamplicol_v2

    rows = [((1, -1), (2, -1)), ((1, -1), (2, 1)), ((1, 1), (2, 1))]
    executor = types.SimpleNamespace(exported=types.SimpleNamespace(rows=rows))
    executor.row_mask = None
    run = types.SimpleNamespace(executor=executor)
    backend = {"requested": "tirust"}
    arm = {"backend": backend, "size": 128, "row": 0}
    with pyamplicol_v2.InducedRun._one_row(run, arm):
        assert executor.row_mask == (True, False, False)
    assert executor.row_mask is None
    assert arm["row_helicities"] == [-1, -1]
    assert pyamplicol_v2._arm_label(arm) == "tirust B=128 helicity row 0"
    with pyamplicol_v2.InducedRun._one_row(run, {"backend": backend, "size": 1}):
        assert executor.row_mask is None


def test_the_harness_probe_stage_follows_the_option_with_the_engines_record(
    monkeypatch,
):
    """lit-probe (DECISIONS 2026-11-08 22:34 (3)): the loop-induced harness's
    generation records the probes off by default (no time, never added to
    the generation's); with the option the engine's own probe record gives
    their time, split, carriers and fallbacks, and no timer wraps the probes'
    internals; the gate tool reads the off stage as no probe time."""

    import types

    from pyamplicol.loops.v2 import goldstone, induced, ward
    from tools.loop_benchmark import pyamplicol_v2

    internals = [
        (induced, "probe_point"),
        (induced, "_probe_runs"),
        (induced, "_goldstone_probes"),
        (ward, "ward_slice"),
        (goldstone, "goldstone_slice"),
    ]
    originals = [getattr(owner, name) for owner, name in internals]
    calls = []
    record = {
        "status": "run",
        "carriers": ["double", "double-double"],
        "fallbacks": 1,
        "seconds": {
            "probe_slice": 4.0,
            "double": 1.0,
            "double-double": 2.0,
            "total": 7.5,
        },
    }

    def probes(slice_, compiled, native, candidates):
        assert [getattr(o, n) for o, n in internals] == originals
        calls.append(candidates)
        return record

    monkeypatch.setattr(induced, "generation_probes", probes)
    process = types.SimpleNamespace(slice=object())
    off, error = pyamplicol_v2.probe_stage(
        types.SimpleNamespace(), process, None, None, [[0.0]]
    )
    assert off == {"status": "off", "seconds": None} and error is None
    assert not calls
    on, error = pyamplicol_v2.probe_stage(
        types.SimpleNamespace(loop_induced_probes=True), process, None, None, [[0.0]]
    )
    assert error is None and calls == [[[0.0]]]
    assert on["seconds"] == 7.5 and on["record"] is record
    assert on["split_seconds"] == {
        "probe_slice": 4.0,
        "double": 1.0,
        "double-double": 2.0,
    }
    assert on["carriers"] == ["double", "double-double"] and on["fallbacks"] == 1
    # The gate tool's li-timing records: the generation's time never holds
    # the probes'; off reads as no probe time (the builder: measured, no
    # validation), on as the engine record's time and split.
    batch = {
        "blocks": 1,
        "median_seconds_per_point": 0.01,
        "samples_seconds_per_point": [0.01],
        "stages_per_point": {},
        "gate": {"status": "pass"},
    }
    for probes_stage, expected in ((off, None), (on, 7.5)):
        config = {
            "reduction": {"requested": "auto", "effective": "ninjars", "reason": "r"},
            "revision": "0" * 40,
            "host": "testhost",
            "generation": {
                "generate": {"seconds": 1.0, "peak_gib": 0.5},
                "export": {"seconds": 0.5},
                "filter": {"seconds": 0.25},
                "probes": probes_stage,
            },
            "batches": {"128": batch},
        }
        document = {
            "processes": [
                {
                    "key": "li_gg_emepmupmum",
                    "loop_induced": True,
                    "pyamplicol": {"native/full/auto": config},
                }
            ]
        }
        (converted,) = gate.li_timing_records(
            document,
            [
                {
                    "key": "gg_emepmupmum",
                    "ids": ["LI-43"],
                    "process": "g g > e+ e- mu+ mu-",
                }
            ],
        )
        generation = converted["generation"]
        assert generation["seconds"] == 1.75
        assert generation["probes_seconds"] == expected
        if expected is not None:
            assert generation["probes_split_seconds"] == on["split_seconds"]


def _induced_run(seconds: list[float], failed: list[int], values: dict) -> dict:
    """One backend run of a loop-induced harness record (``InducedRun``)."""

    summary = {
        "blocks": len(seconds),
        "points_per_block": 128,
        "median_seconds_per_point": sorted(seconds)[len(seconds) // 2],
        "min_seconds_per_point": min(seconds),
        "max_seconds_per_point": max(seconds),
        "samples_seconds_per_point": seconds,
        "stages_per_point": {"dressing": seconds[0] / 2, "squaring": 1e-5},
        "counts_per_block": {"TIR gate refusals": 3},
        "reals_per_lane": {},
        "throughput": None,
        "values": values,
        "failed_points": failed,
        "failure_causes": {"x": max(failed)} if any(failed) else {},
    }
    return summary


def test_loop_induced_gate_on_tirust_with_the_born_free_metric(tmp_path):
    """``gate.build`` on a loop-induced row (m-lit lit-perf, PLAN §5.2): one
    configuration per requested backend; the verdict on TIRust; NinjaRS per
    amplitude reported as the second mode with its ratios; a TIRust run that
    failed points closed has no verdict (the gap), its served points' time
    beside it; the value metric is Born-free; a provider whose values are
    excluded is no matched provider (reported beside the verdict, its values
    not compared) and an excepted point is left out; the li-timing records
    carry the request, per block the failed points, the served fraction and
    the verdict, under the registry key."""

    import json

    key = "li_gg_emepmupmum"
    rows = [{"born": 0.0, "finite": 1.0 + 1e-6 * i, "seconds": 0.01} for i in range(2)]
    providers = {
        "kind": "providers",
        "process": key,
        "cpu": 130,
        "points": 2,
        "providers": {
            "madloop": {
                "points": 2,
                "mean_seconds_per_point": 0.040,
                "median_seconds_per_point": 0.040,
                "rows": rows,
            },
            "openloops": {
                "points": 2,
                "mean_seconds_per_point": 0.010,
                "median_seconds_per_point": 0.010,
                "rows": rows,
            },
            "recola": {
                "points": 2,
                "mean_seconds_per_point": 0.004,
                "median_seconds_per_point": 0.004,
                "rows": [{**r, "raw": {"finite": r["finite"]}} for r in rows],
                "values_excluded": "width convention",
            },
        },
    }
    values = {"0": [0.0, 1.0], "1": [0.0, 1.0]}
    record = {
        **_pyamplicol_record(key, [0.006]),
        "contribution": "loop-induced",
        "host": "testhost",
        "host_cpus": [7],
        "native": {"directory": "loop-native-x", "digest": "d" * 64},
        "generation": {
            "generate": {"seconds": 1.0, "peak_gib": 0.5},
            "export": {"seconds": 0.5},
            "filter": {"seconds": 0.1, "peak_gib": 0.5, "dropped_rows": 0},
            "probes": {"seconds": 9.0, "peak_gib": 0.7, "split_seconds": {}},
            "plan_mb": 0.1,
        },
        "batches": {},
        "backend_runs": {
            "auto": {
                "requested": "auto",
                "effective": "ninjars",
                "reason": "auto: NinjaRS; test",
                "batches": {"128": _induced_run([0.008] * 5, [0] * 5, values)},
                "served": {},
                "profiled": {},
            },
            "tirust": {
                "requested": "tirust",
                "effective": "tirust",
                "reason": "requested",
                "batches": {"128": _induced_run([0.006] * 5, [0] * 5, values)},
                "served": {},
                "profiled": {},
            },
        },
    }
    failing = json.loads(json.dumps(record))
    failing["revision"] = "1" * 40
    run = failing["backend_runs"]["tirust"]
    run["batches"]["128"] = _induced_run([0.007] * 5, [3] * 5, values)
    run["served"] = {
        "128": {**_induced_run([0.0065] * 5, [0] * 5, values), "points_per_block": 125}
    }
    paths = []
    for index, item in enumerate([providers, record]):
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(item))
        paths.append(path)
    document, _ = gate.build(paths)
    (row,) = document["processes"]
    assert row["loop_induced"] and row["gate_backend"] == "tirust"
    assert set(row["not_matched"]) == {"recola"}
    tirust = row["pyamplicol"]["native/full/tirust"]
    auto = row["pyamplicol"]["native/full/auto"]
    verdict = tirust["batches"]["128"]["gate"]
    # The limit: MadLoop 40 ms and 1.3 x OpenLoops 10 ms; Recola beside it.
    assert verdict["status"] == "pass" and verdict["binding_provider"] == "openloops"
    assert verdict["not_matched"]["recola"]["ratio"] == 0.006 / 0.004
    assert auto["batches"]["128"]["gate"]["status"] == gate.SECOND_MODE_STATUS
    assert auto["batches"]["128"]["gate"]["measured_status"] == "pass"
    assert tirust["reduction"]["requested"] == "tirust"
    compared = tirust["values_vs_providers"]
    assert compared["recola"] == {"excluded": "width convention"}
    assert compared["madloop"]["metric"] == gate.INDUCED_METRIC
    worst = abs(1.0 - (1.0 + 1e-6)) / (1.0 + 1e-6)
    assert compared["madloop"]["max_finite_deviation"] == worst
    assert "squaring" in tirust["stages"]["128"]
    # TIRust failed points closed: no verdict, the gap with its numbers.
    paths[1].write_text(json.dumps(failing))
    document, _ = gate.build(paths)
    (row,) = document["processes"]
    gap = row["pyamplicol"]["native/full/tirust"]["batches"]["128"]["gate"]
    assert gap["status"] == gate.FAILED_CLOSED_STATUS
    assert gap["failed_points"] == [3] * 5 and "ratio_to_binding_limit" not in gap
    assert gap["served_points"]["points"] == 125
    assert gap["served_fraction"] == 1 - 3 / 128
    records = gate.li_timing_records(
        document,
        [{"key": "gg_emepmupmum", "ids": ["LI-43"], "process": "g g > e+ e- mu+ mu-"}],
    )
    by = {(r["requested"], r["batch"]): r for r in records}
    tir = by[("tirust", 128)]
    assert tir["key"] == "gg_emepmupmum" and tir["host"] == "testhost"
    assert tir["failed_points"] == [3] * 5 and "seconds_per_point" not in tir
    assert tir["whole_call_seconds_per_point"] == 0.007
    assert tir["gate"]["status"] == gate.FAILED_CLOSED_STATUS
    assert (
        tir["generation"]["seconds"] == 1.6
        and tir["generation"]["probes_seconds"] == 9.0
    )
    nin = by[("auto", 128)]
    assert nin["seconds_per_point"] == 0.008 and nin["effective"] == "ninjars"
    assert nin["reason"] == "auto: NinjaRS; test"
    target = tmp_path / "li-timing.json"
    target.write_text(json.dumps({"records": [{"key": "other", "revision": "2" * 40}]}))
    gate.write_li_timing(target, records)
    gate.write_li_timing(target, records)
    merged = json.loads(target.read_text())["records"]
    assert len(merged) == 1 + len(records)


def test_value_exceptions_leave_their_points_out():
    """``compare_values`` honours a provider's ``value_exceptions``: its
    listed stream points are not compared (named in the result)."""

    mine = {0: (0.0, 1.0), 7: (0.0, 1.0), 9: (0.0, 2.0)}
    theirs = {"recola": {0: (0.0, 1.0), 7: (0.0, 5.0), 9: (0.0, 2.0)}}
    notes = {"recola": {"value_exceptions": {"points": [7], "reason": "unstable"}}}
    found = gate.compare_values("li_gg_zzzz", mine, theirs, notes)["recola"]
    assert found["max_finite_deviation"] == 0.0 and found["points"] == 2
    assert found["excepted_points"] == [7] and found["exception"] == "unstable"
    plain = gate.compare_values("li_gg_zzzz", mine, theirs)["recola"]
    assert plain["max_finite_deviation"] == 0.8


def test_blocks_without_their_core_are_left_out():
    """``gate.valid_blocks``: a loop-induced block whose process had less than
    the minimal CPU share of its wall time (a shared core, or a job a host
    guard stopped) is left out of the median and listed; all blocks kept
    without shares."""

    summary = {
        "samples_seconds_per_point": [1.0, 1.1, 9.0, 1.2],
        "median_seconds_per_point": 1.15,
        "min_seconds_per_point": 1.0,
        "max_seconds_per_point": 9.0,
        "blocks": 4,
        "failed_points": [0, 1, 2, 3],
        "cpu_share": [0.99, 0.99, 0.12, 0.98],
    }
    kept = gate.valid_blocks(summary, gate.MIN_CPU_SHARE)
    assert kept["samples_seconds_per_point"] == [1.0, 1.1, 1.2]
    assert kept["median_seconds_per_point"] == 1.1 and kept["blocks"] == 3
    assert kept["failed_points"] == [0, 1, 3]
    assert kept["dropped_blocks"] == {"2": 0.12}
    # No block at the minimum: kept whole and marked.
    shared = {**summary, "cpu_share": [0.8, 0.7, 0.12, 0.85]}
    marked = gate.valid_blocks(shared, gate.MIN_CPU_SHARE)
    assert marked["samples_seconds_per_point"] == summary["samples_seconds_per_point"]
    assert marked["below_min_cpu_share"] == gate.MIN_CPU_SHARE
    del summary["cpu_share"]
    assert gate.valid_blocks(summary, gate.MIN_CPU_SHARE) is summary


def test_report_builder_pairs_providers_of_a_row_without_a_fixture_record():
    """lit-report: a loop-induced row whose providers were timed only in its
    unchecked re-time unit (no lit-ref fixture timing record) takes the
    paired times; each provider is matched when the row's lit.json validation
    agrees with it within 1e-5, else shown unmatched with the reason."""

    builder = _report_builder()
    prov = {"status": "not run", "reason": "no record", "expected": ["ML", "OL", "Rc"]}
    rec = {
        "key": "gg_zz",
        "backend": "tirust",
        "revision": "524e6cfaa78c",
        "host": "test host",
        "providers": {
            "times": {
                "madloop": {"seconds_per_point": 3e-3, "rounds": [1, 2]},
                "openloops": {"seconds_per_point": 2e-3, "rounds": [1, 2]},
                "recola": {"seconds_per_point": 1e-3, "rounds": [1, 2]},
            }
        },
    }
    validation = {
        "references": [
            {"key": "madloop", "short": "ML", "max_relative": 1e-13},
            {"key": "openloops", "short": "OL", "max_relative": 2e-4},
        ]
    }
    out = builder._li_paired(prov, rec, validation)
    assert out["status"] == "recorded" and out["paired"]
    assert not out["providers"]["madloop"].get("match")
    assert "2.0e-04" in out["providers"]["openloops"]["match"]
    assert "no agreement" in out["providers"]["recola"]["match"]
    # The fastest matched provider is MadLoop (the faster two are unmatched).
    assert out["fastest"]["key"] == "madloop"
    assert out["fastest_unmatched"]["key"] == "recola"
    # A row with a fixture record keeps its entries' match status.
    kept = builder._li_paired(
        {"status": "recorded", "providers": {"recola": {"match": "x"}}}, rec, validation
    )
    assert kept["providers"]["recola"]["match"] == "x"


def test_report_builder_states_the_unchecked_runs_host_state():
    """lit-report: Table 2's provenance line carries the runs' concurrency,
    load, MemAvailable, smallest median CPU share and the re-timed rows."""

    builder = _report_builder()
    state = {"loadavg": ["300.5", "1", "1"], "mem_available_gib": 250.0}
    recs = [
        {
            "configuration": "unchecked",
            "revision": "524e6cfaa78c",
            "key": "a",
            "concurrency": "40 rows in parallel units",
            "host_state": {"start": state, "end": dict(state, loadavg=["350.0"])},
            "cpu_share": {"min": 0.9, "median": 0.97},
            "retimed": {"reason": "disturbed"},
        },
        {
            "configuration": "unchecked",
            "revision": "0000000",
            "key": "b",
            "host_state": {"start": dict(state, loadavg=["999"])},
        },
    ]
    hs = builder._li_host_state({"records": recs}, {"524e6cfa"})
    assert hs == {
        "concurrency_max": 40,
        "load_1min": [300.5, 350.0],
        "mem_available_gib_min": 250.0,
        "cpu_share_min": 0.97,
        "re_timed": 1,
    }
