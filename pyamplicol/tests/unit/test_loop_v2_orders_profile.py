# SPDX-License-Identifier: 0BSD
"""Partial weights per squared coupling order of virtual runtimes and the
public loop profiling route (m-lit lit-examples G4, G5).

* ``Runtime.evaluate_orders`` on a Born-interfered runtime with several
  orders (u u~ > d d~, NLO QCD, gluon and photon Borns) is a view of ONE
  ``evaluate`` call: its sectors and Born orders as they are, the explicit
  sums, and the sum equal to the total of an independent public path (the
  resolved evaluation summed over helicities, colour components and
  sectors) to rounding: within gamma_n S, the summation bound of its n
  resolved entries against the pre-cancellation scale S; a loop-induced
  runtime's view has no Born. The CLI
  ``--orders`` prints the explicit sums apart.
* ``pyamplicol profile`` / ``BenchmarkRunner`` of a loop artifact returns the
  per-point time and its split into trees, dressing, scatter, reduction and
  rest from the executor's stage report of the timed calls, for the checked
  default and the unchecked setting; the CLI on both reduction backends, and
  ``--stability`` the stable-digit statistics over flat points.
"""

from __future__ import annotations

import io
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"


def _artifact(
    root: Path,
    expression: str,
    extra: str = "",
    fixed: str = "",
    generation: str = "",
) -> Path:
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(
        f"""action = "generate"
[model]
source = "loop_sm"
restriction = "no_b_mass"
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = ["QCD"]
{extra}
[loop.preparation]
fixed_parameters = {{{fixed or 'WT = "0"'}}}
[loop]
r2_strategy = "model"
[generation]
output = "artifact"
emit_api_bundle = false
{generation}
"""
    )
    config = resolve_config(card)
    output = Path(config.effective.generation.output)
    Generator(config).generate(
        ProcessSet((ProcessRequest.parse(expression, name="p"),)), output
    )
    return output


def _points(artifact: Path) -> list:
    record = json.loads(
        (artifact / "processes" / "p" / "validation-momenta.json").read_text()
    )
    return [[entry["momentum"] for entry in point] for point in record["points"]]


@pytest.fixture(scope="module")
def uux_ddx(tmp_path_factory):
    return _artifact(
        tmp_path_factory.mktemp("orders"),
        "u u~ > d d~",
        'coupling_order_policy = "explicit"\n'
        "max_coupling_orders = { QCD = 2, QED = 2 }",
    )


@pytest.mark.loop_fast
def test_the_stage_groups_split_the_wall_time():
    from pyamplicol.runtime.virtual import PROFILE_GROUPS, profile_stages

    seconds = {
        "trees": 1.0,
        "trees: programs": 0.5,
        "seeds": 0.25,
        "factors": 0.25,
        "dressing": 1.0,
        "scatter": 0.5,
        "reduction": 2.0,
        "reduction: masters": 1.0,
        "seed contraction": 0.2,
    }
    virtual = profile_stages(seconds, 6.0, "virtual")
    assert tuple(virtual) == PROFILE_GROUPS
    assert virtual == {
        "trees": 1.0,
        "dressing": 1.5,
        "scatter": 0.5,
        "reduction": 2.0,
        "rest": 1.0,
    }
    induced = profile_stages(seconds, 6.0, "loop-induced")
    assert induced["dressing"] == 1.0 and induced["rest"] == 1.5
    assert profile_stages(seconds, 1.0, "virtual")["rest"] == 0.0


@pytest.mark.loop_fast
def test_the_profile_counts_every_native_batch_of_a_call():
    from pyamplicol.runtime.virtual import collected_stage_seconds

    class Executor:
        def __init__(self) -> None:
            self.last_report = None

        def evaluate(self, batches: int) -> None:
            # A call of several native batches (a per-point fallback, a
            # rescue level, one batch per scale), each with its report.
            for _ in range(batches):
                self.last_report = {"seconds": {"trees": 1.0, "reduction": 2.0}}

    class Backend:
        def __init__(self) -> None:
            self._executor = Executor()

    class Runtime:
        def __init__(self) -> None:
            self._backend = Backend()

    runtime, other = Runtime(), Runtime()
    with collected_stage_seconds(runtime) as seconds:
        runtime._backend._executor.evaluate(3)
        other._backend._executor.evaluate(1)
        assert runtime._backend._executor.last_report["seconds"]["trees"] == 1.0
    assert seconds == {"trees": 3.0, "reduction": 6.0}
    # Only the profiled runtime's executor recorded, and only while active.
    assert type(runtime._backend._executor) is Executor
    assert type(other._backend._executor) is Executor
    runtime._backend._executor.evaluate(1)
    assert seconds == {"trees": 3.0, "reduction": 6.0}
    assert "_collected_stage_seconds" not in vars(runtime._backend._executor)
    with collected_stage_seconds(object()) as nothing:
        assert nothing is None


@pytest.mark.loop_physics
def test_evaluate_orders_is_a_view_of_one_evaluation(uux_ddx, monkeypatch):
    from pyamplicol import Runtime
    from pyamplicol.api import VirtualOrderEvaluation, VirtualStabilityChecks

    runtime = Runtime.load(uux_ddx)
    points = _points(uux_ddx)
    calls = []
    evaluate = runtime._backend.evaluate

    def counted(*args, **kwargs):
        calls.append(1)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(runtime._backend, "evaluate", counted)
    view = runtime.evaluate_orders(
        points,
        mu_r2="10000",
        stability_checks=VirtualStabilityChecks(1, True, rescaled=True),
    )
    assert calls == [1]
    assert isinstance(view, VirtualOrderEvaluation)
    assert len(view.orders) == 3 and len(view.born_orders) == 3
    assert view.orders == tuple(sector.order for sector in view.evaluation.sectors)
    for index, order in enumerate(view.orders):
        assert view.for_order(order) == tuple(
            point[index] for point in view.evaluation.values
        )
    for index, order in enumerate(view.born_orders):
        assert view.born_for_order(order) == tuple(
            point[index] for point in view.evaluation.born_values
        )
    with pytest.raises(ValueError, match="absent from the catalogue"):
        view.for_order((("QCD", 99),))
    monkeypatch.undo()
    # The explicit sum against an independent public path: the resolved
    # evaluation summed over helicities and colour components (``total``)
    # and over its sectors. An identity held to rounding: within the
    # summation bound gamma_n S (n the resolved entries summed, u the
    # binary64 unit roundoff, S the pre-cancellation scale summed over the
    # sectors), which is far below 10 x the summed estimate here.
    resolved = runtime.evaluate_resolved(points, mu_r2="10000")
    total = resolved.total()
    entries = (
        len(resolved.helicity_ids)
        * len(resolved.virtual_color_ids)
        * len(resolved.sectors)
    )
    unit = 2.0**-53
    gamma = entries * unit / (1 - entries * unit)
    for point, summed in enumerate(view.sum_orders()):
        assert summed == tuple(
            math.fsum(float(sector[k]) for sector in view.evaluation.values[point])
            for k in range(3)
        )
        independent = [
            math.fsum(float(sector[k]) for sector in total.values[point])
            for k in range(3)
        ]
        estimate = view.evaluation.stability[point]
        for k in range(3):
            scale = math.fsum(float(m[k]) for m in estimate.magnitude)
            deviation = abs(float(summed[k]) - independent[k])
            assert deviation <= gamma * scale, (point, k, deviation / (unit * scale))
        born = math.fsum(float(x) for x in total.born_values[point])
        assert view.sum_born_orders()[point] == pytest.approx(born, rel=1e-13)


@pytest.mark.loop_physics
def test_the_cli_orders_and_profile_of_a_loop_artifact(uux_ddx, tmp_path):
    from pyamplicol.benchmarking import VirtualProfileResult
    from pyamplicol.cli.main import run_cli
    from pyamplicol.runtime.virtual import PROFILE_GROUPS

    momenta = tmp_path / "momenta.json"
    momenta.write_text(json.dumps(_points(uux_ddx)))
    stdout = io.StringIO()
    code = run_cli(
        [
            "evaluate",
            str(uux_ddx),
            "--momenta",
            str(momenta),
            "--mu-r2",
            "10000",
            "--orders",
            "--json",
            "--progress",
            "off",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert len(payload["orders"]) == 3
    assert len(payload["sum_orders"]) == len(_points(uux_ddx))
    assert payload["evaluation"]["contribution_kind"] == "virtual"
    human = io.StringIO()
    assert (
        run_cli(
            [
                "evaluate",
                str(uux_ddx),
                "--momenta",
                str(momenta),
                "--mu-r2",
                "10000",
                "--orders",
                "--progress",
                "off",
                "--color",
                "never",
            ],
            stdout=human,
            stderr=io.StringIO(),
        )
        == 0
    )
    assert "Explicit sums over the squared coupling orders" in human.getvalue()
    for setting in ([], ["--reduction-unchecked"], ["--checked"]):
        checked = setting == ["--checked"]
        stdout = io.StringIO()
        code = run_cli(
            [
                "profile",
                str(uux_ddx),
                "--batch-size",
                "4",
                "--minimum-samples",
                "2",
                "--warmup-runs",
                "0",
                "--target-runtime",
                "1e-6",
                "--json",
                "--progress",
                "off",
                *setting,
            ],
            stdout=stdout,
            stderr=io.StringIO(),
        )
        assert code == 0
        report = json.loads(stdout.getvalue())
        # Both reduction backends side by side, the type's default first;
        # the optimistic pass times each backend alone, the checked mode the
        # default backend as "auto"; the stability not asked.
        assert [row["backend"] for row in report["runtime"]] == ["ninjars", "tirust"]
        assert report["runtime"][0]["requested"] == ("auto" if checked else "ninjars")
        setup = report["runtime_setup"]
        assert ("checked reduction, refused points served" in setup) is checked
        assert ("optimistic: unchecked reduction, no checks" in setup) is not checked
        assert report["stability"] is None and report["default_backend"] == "ninjars"
        other = report["runtime"][1]
        assert (other["profile"] is None) is (other["refusal"] is not None)
        profile = report["runtime"][0]["profile"]
        assert set(profile["stages"]) == set(PROFILE_GROUPS)
        assert profile["batch"] == 4 and profile["samples"] == 2
        assert profile["unchecked"] is not checked
        assert profile["reduction"]["effective"] == "ninjars"
        # Every lane of every timed call counts (none falls back here).
        assert set(profile["reduction"]["lanes"]) == {"ninjars"}
        assert profile["reduction"]["lanes"]["ninjars"] >= 2
        assert profile["seconds_per_point"] > 0 and profile["label"] is None
        # The stages add up to the reported time per point.
        assert sum(profile["stages"].values()) == pytest.approx(
            profile["seconds_per_point"]
        )
        # One core: the timed process's pools run one thread each; the
        # reference (double-double by default) is one unchecked pass on the
        # same backend, its chunk what its budget affords (here 1e-6 s: one
        # point, one call).
        assert profile["threads"] == 1
        assert "on one core" in report["runtime_setup"]
        assert report["ground_truth"]["label"] == "dd"
        reference = profile["reference"]
        assert reference["arithmetic"] == "double-double"
        assert reference["digits"] is None and 1 <= reference["batch"] <= 4
        assert reference["samples"] >= 1 and reference["warmup"] == 0
    # The checked mode and the unchecked reduction contradict each other.
    stderr = io.StringIO()
    code = run_cli(
        ["profile", str(uux_ddx), "--checked", "--reduction-unchecked"],
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert code != 0 and "benchmark.checked" in stderr.getvalue()
    # --stability on both backends over three flat points (binary64 with the
    # default rotation against a double-double re-run); human and JSON.
    for output in (["--json"], ["--color", "never"]):
        stdout = io.StringIO()
        code = run_cli(
            [
                "profile",
                str(uux_ddx),
                "--stability",
                "--stability-points",
                "3",
                "--sqrt-s",
                "500",
                "--progress",
                "off",
                *output,
            ],
            stdout=stdout,
            stderr=io.StringIO(),
        )
        assert code == 0
        if output == ["--json"]:
            report = json.loads(stdout.getvalue())
            assert report["runtime"] is None
            assert report["stability_setup"]["points"] == 3
            assert report["stability_setup"]["sqrt_s"] == 500.0
            first = report["stability"][0]
            assert first["backend"] == "ninjars" and first["refusal"] is None
            assert first["points"] + first["failed_reference"] == 3
            # Every flat point is recorded by index; binary64 failures rank
            # among the measured points at 0 digits.
            per_point = _records(first["per_point"])
            assert [p["index"] for p in per_point] == [0, 1, 2]
            assert first["failed_binary64"] == sum(
                p["status"] == "binary64-failed" for p in per_point
            )
            assert len(first["percentiles"]) == 4
            assert all(0.0 <= d <= 16.0 for d in first["true_digits"])
        else:
            text = stdout.getvalue()
            assert "true digits" in text and "NinjaRS" in text and "50 %" in text
            assert max(len(line) for line in text.splitlines()) <= 100
    from pyamplicol.api import BenchmarkRunner
    from pyamplicol.config import BenchmarkConfig

    result = BenchmarkRunner(BenchmarkConfig(target_runtime=1e-6)).run(uux_ddx)
    assert isinstance(result, VirtualProfileResult)
    assert result.contribution == "virtual" and result.failed_points == 0


@pytest.mark.loop_physics
def test_the_runtime_profile_times_distinct_points(uux_ddx, monkeypatch):
    """Every warm-up and timed call of the loop runtime profile takes a new
    batch of distinct flat points at the validation point's energy (no
    point repeated within or across calls), deterministic in the seed: the
    minimum calls when they reach the time budget, and further calls on the
    following points of the same sequence until they reach it. The first
    warm-up probes one point of its batch first (``WARMUP_CALL_SECONDS``: a
    dear full call keeps that point as the warm-up)."""

    from pyamplicol.api import Runtime
    from pyamplicol.benchmarking import profile_points, profile_virtual
    from pyamplicol.config import BenchmarkConfig

    bench = BenchmarkConfig(
        batch_size=4, warmup_runs=1, minimum_samples=2, target_runtime=1e-6
    )
    batches: list = []
    evaluate = Runtime.evaluate

    def recorded(self, points, *args, **kwargs):
        batches.append([tuple(map(tuple, point)) for point in points])
        return evaluate(self, points, *args, **kwargs)

    monkeypatch.setattr(Runtime, "evaluate", recorded)
    result = profile_virtual(uux_ddx, bench)
    assert result.batch == 4 and result.samples == 2 and result.warmup == 1
    assert [len(batch) for batch in batches] == [1, 4, 4, 4]
    assert batches[0] == batches[1][:1]
    flat = [point for batch in batches[1:] for point in batch]
    assert len(set(flat)) == len(flat) == 12
    validation = _points(uux_ddx)[0]
    total = [sum(float(leg[i]) for leg in validation[:2]) for i in range(4)]
    energy = math.sqrt(total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2)
    assert all(math.isclose(p[0][0] + p[1][0], energy) for p in flat)
    assert profile_points(uux_ddx, "p", bench) == profile_points(uux_ddx, "p", bench)
    assert [tuple(map(tuple, p)) for p in profile_points(uux_ddx, "p", bench)] == flat
    # A budget the minimum calls do not reach: more calls, the points
    # continuing the sequence (the first calls' points unchanged).
    batches.clear()
    longer = replace(bench, target_runtime=10 * 4 * result.high)
    again = profile_virtual(uux_ddx, longer)
    assert again.samples > 2
    more = [point for batch in batches[1:] for point in batch]
    assert more[:12] == flat and len(set(more)) == len(more)


def test_the_runtime_profile_reports_peak_ram_and_the_reference(uux_ddx):
    """The runtime profile's peak RAM over its binary64 calls, and with
    ``reference=`` the reference carrier timed after them: one unchecked
    pass on the profiled backend alone (no checks, no rescue, no
    best-effort re-evaluation), calls of one SIMD chunk on the first points
    of the binary64 calls' batches, the binary64 protocol's warm-up and
    minimum calls; nothing extra runs without it."""

    from pyamplicol.api import Runtime
    from pyamplicol.benchmarking import profile_virtual
    from pyamplicol.config import BenchmarkConfig

    bench = BenchmarkConfig(
        batch_size=2, warmup_runs=2, minimum_samples=2, target_runtime=1e-6
    )
    runtime = Runtime.load(uux_ddx)
    points = _points(uux_ddx) * 6
    calls: list = []
    evaluate = Runtime.evaluate

    def recorded(self, batch, *args, **kwargs):
        calls.append((len(batch), {k: kwargs.get(k) for k in _KEYS}))
        return evaluate(self, batch, *args, **kwargs)

    Runtime.evaluate = recorded
    try:
        plain = profile_virtual(runtime, bench, points=points, reduction="tirust")
        # The one-point probe, two warm-ups and two timed calls.
        assert plain.reference is None and len(calls) == 5
        assert [c[0] for c in calls] == [1, 2, 2, 2, 2]
        assert plain.peak_memory is not None
        assert plain.peak_memory.peak_rss_bytes >= plain.peak_memory.start_rss_bytes
        calls.clear()
        both = profile_virtual(
            runtime,
            bench,
            points=points,
            reduction="tirust",
            reference={"arithmetic": "double-double"},
        )
    finally:
        Runtime.evaluate = evaluate
    # The probe, two binary64 warm-ups and two timed calls; one reference
    # warm-up of one point, then one call of what the 1e-6 s budget affords
    # (a point).
    assert [c[1]["arithmetic"] for c in calls] == [None] * 5 + ["double-double"] * 2
    assert [c[0] for c in calls[5:]] == [1, 1]
    assert all(
        c[1]
        == {
            "arithmetic": "double-double",
            "on_failure": "return",
            "reduction_unchecked": True,
            "reduction": "tirust",
            "stability_checks": None,
        }
        for c in calls[5:]
    )
    reference = both.reference
    assert reference is not None and reference.arithmetic == "double-double"
    assert (reference.batch, reference.samples, reference.warmup) == (1, 1, 1)
    assert both.warmup == 2
    assert reference.points == 1 and reference.digits is None
    assert reference.low <= reference.seconds_per_point <= reference.high
    assert reference.failed_points == 0 and reference.peak_memory is not None
    assert both.samples == plain.samples and both.batch == plain.batch


_KEYS = (
    "arithmetic",
    "on_failure",
    "reduction_unchecked",
    "reduction",
    "stability_checks",
)


@pytest.mark.loop_physics
def test_the_stability_tables_do_not_depend_on_the_worker_count(uux_ddx):
    """profile --stability on 1 (in-process) and 3 worker processes: the
    same per-point records and statistics (ux-cli C5); only the times and
    the worker count differ."""

    from pyamplicol.cli.main import run_cli

    reports = {}
    for cores in ("1", "3"):
        stdout = io.StringIO()
        code = run_cli(
            [
                "profile",
                str(uux_ddx),
                "--stability",
                "--stability-points",
                "7",
                "--n_core" if cores == "1" else "--n-core",
                cores,
                "--progress",
                "off",
                "--json",
            ],
            stdout=stdout,
            stderr=io.StringIO(),
        )
        assert code == 0
        reports[cores] = json.loads(stdout.getvalue())
    one, three = reports["1"], reports["3"]
    assert one["stability_setup"]["n_core"] == 1
    assert three["stability_setup"]["n_core"] == 3
    for report in (one, three):
        assert report["stability_setup"]["interrupted"] is False
        assert report["stability_setup"]["requested"] == 7
        report["stability_setup"].pop("n_core")
        for row in report["stability"]:
            row.pop("seconds")
            # Every sample's record (the per-sample dump) reproduces the
            # aggregate numbers; its task timings depend on the workers.
            target = report["stability_setup"]["target"]
            table = row["samples"]
            tasks = table["tasks"]
            samples = _records(table)
            assert [s["index"] for s in samples] == list(range(7))
            # Every point's binary64 and reference task (seconds, points).
            assert all(
                tasks[s["binary64_task"]][1] >= 1 and tasks[s["reference_task"]][1] >= 1
                for s in samples
            )
            measured = [
                0.0 if s["value"] is None else s["true_digits"]
                for s in samples
                if s["value"] is None or s["true_digits"] is not None
            ]
            # The records' digits carry 3 decimals.
            assert sorted(measured) == sorted(round(d, 3) for d in row["true_digits"])
            assert sum(s["flagged"] for s in samples) == row["flagged"]
            assert (
                sum(
                    1
                    for s in samples
                    if s["true_relative_error"] is not None
                    and float(s["true_relative_error"]) > target
                    and not s["flagged"]
                )
                == row["unflagged_inaccurate"]
            )
            for sample in samples:
                assert sample["reference_value"] is not None
                # A double-double value is reproducible to its own rounding
                # only (a worker's engine state depends on what it evaluated
                # before: the last digits move at ~1e-28 relative).
                sample["reference_value"] = f"{float(sample['reference_value']):.15e}"
                assert any(
                    name.startswith("rotation") for name in sample["estimate_checks"]
                )
                sample.pop("binary64_task")
                sample.pop("reference_task")
            row["samples"] = samples
            # Timings, with the point counts per precision (all of them).
            timing = row.pop("precision_timing")
            assert {k: v["points"] for k, v in timing.items()} == {
                "binary64": 7,
                "double-double": 7,
            }
    assert one["stability"] == three["stability"]
    inputs = _records(one["stability_setup"]["samples"])
    assert [s["index"] for s in inputs] == list(range(7))
    assert all(len(s["momenta"]) == 4 and len(s["momenta"][0]) == 4 for s in inputs)
    assert one["stability_setup"] == three["stability_setup"]
    assert one["failure_causes"] == three["failure_causes"]


@pytest.mark.loop_fast
def test_arbitrary_precision_references_are_certified_on_any_worker_count(uux_ddx):
    """profile --stability --ground-truth arb: every reference is certified,
    by the other backend's agreeing reference or by its checked re-run,
    with the same records on one and on three workers; with one backend
    every reference is the checked one."""

    from pyamplicol.cli.main import run_cli

    def profile(*extra):
        stdout = io.StringIO()
        code = run_cli(
            [
                "profile",
                str(uux_ddx),
                "--stability",
                "--stability-points",
                "6",
                "--ground-truth",
                "arb:30",
                "--progress",
                "off",
                "--json",
                *extra,
            ],
            stdout=stdout,
            stderr=io.StringIO(),
        )
        assert code == 0
        return json.loads(stdout.getvalue())

    certified = {}
    for cores in ("1", "3"):
        report = profile("--n_core", cores)
        assert "certified" in report["stability_setup"]["reference"]
        certified[cores] = {
            row["backend"]: [
                (s["index"], s["reference_certified"], s["true_digits"])
                for s in _records(row["samples"])
            ]
            for row in report["stability"]
        }
        for row in report["stability"]:
            timing = row["precision_timing"]
            assert {k: v["points"] for k, v in timing.items()} == {
                "binary64": 6,
                "arbitrary": 6,
            }
            for sample in _records(row["samples"]):
                if sample["reference_value"] is None:
                    assert sample["reference_certified"] is None
                else:
                    assert sample["reference_certified"] in ("agreement", "checks")
    assert certified["1"] == certified["3"]
    # Both backends served every point here: they agree everywhere.
    assert {c for rows in certified["1"].values() for _, c, _ in rows} == {"agreement"}
    alone = profile("--n_core", "1", "--loop-reduction", "ninjars")
    (row,) = alone["stability"]
    assert {s["reference_certified"] for s in _records(row["samples"])} == {"checks"}


@pytest.mark.loop_fast
def test_disagreeing_references_are_rerun_checked_on_the_worker_pool(
    uux_ddx, monkeypatch
):
    """References that do not agree go to their checked re-run, queued
    after both backends' plain references and submitted to the pool before
    any further chunk: every reference ends certified by its checks, the
    records equal on one and three workers. (The agreement is decided in
    the parent process, so forcing it off holds for spawned workers.)"""

    from pyamplicol import loop_profile
    from pyamplicol.cli.main import run_cli

    def plain_disagree(a, b, tolerance=loop_profile._AGREEMENT):
        # The cross-check's agreement off; the checked references' own
        # comparison (to binary64's unit roundoff) as it is.
        return tolerance is not loop_profile._AGREEMENT and abs(a - b) <= (
            tolerance * max(abs(a), abs(b))
        )

    monkeypatch.setattr(loop_profile, "_references_agree", plain_disagree)
    found = {}
    for cores in ("1", "3"):
        stdout = io.StringIO()
        code = run_cli(
            [
                "profile",
                str(uux_ddx),
                "--stability",
                "--stability-points",
                "5",
                "--ground-truth",
                "arb:30",
                "--n_core",
                cores,
                "--progress",
                "off",
                "--json",
            ],
            stdout=stdout,
            stderr=io.StringIO(),
        )
        assert code == 0
        report = json.loads(stdout.getvalue())
        assert report["stability_setup"]["interrupted"] is False
        assert report["stability_setup"]["n_core"] == int(cores)
        for row in report["stability"]:
            tasks = row["samples"]["tasks"]
            for sample in _records(row["samples"]):
                # The re-run's own task (seconds, points) is recorded.
                assert tasks[sample["reference_checked_task"]][1] >= 1
        found[cores] = {
            row["backend"]: [
                (s["index"], s["reference_certified"], s["true_digits"])
                for s in _records(row["samples"])
            ]
            for row in report["stability"]
        }
        for row in report["stability"]:
            # The re-runs' time is the reference stream's, not new points.
            assert row["precision_timing"]["arbitrary"]["points"] == 5
    assert found["1"] == found["3"]
    assert {c for rows in found["1"].values() for _, c, _ in rows} == {"checks"}
    assert all(len(rows) == 5 for rows in found["1"].values())


def _profile_json(artifact, *extra):
    from pyamplicol.cli.main import run_cli

    stdout = io.StringIO()
    code = run_cli(
        [
            "profile",
            str(artifact),
            "--stability",
            "--progress",
            "off",
            "--json",
            *extra,
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    return json.loads(stdout.getvalue())


@pytest.mark.loop_fast
def test_checked_references_that_contradict_each_other_are_no_reference(
    uux_ddx, monkeypatch
):
    """Two checked re-runs that both pass their checks but still differ by
    more than binary64's unit roundoff contradict their estimates: neither
    is kept, the point has no reference, its reason says so."""

    from pyamplicol import loop_profile

    monkeypatch.setattr(loop_profile, "_references_agree", lambda a, b, t=None: False)
    report = _profile_json(
        uux_ddx, "--stability-points", "4", "--ground-truth", "arb:30", "--n_core", "1"
    )
    for row in report["stability"]:
        assert row["failed_reference"] == 4
        for sample in _records(row["samples"]):
            assert sample["reference_value"] is None
            assert sample["reference_certified"] is None
            assert sample["reference_reason"].startswith(
                "failed: stability checks: the backends' checked references differ by"
            )


@pytest.mark.loop_fast
def test_the_deadline_admits_checked_reruns_and_ends_in_time(uux_ddx, monkeypatch):
    """Under --completion-time on three workers, with every reference sent to
    its checked re-run: no chunk is admitted on an unmeasured stream, the
    re-runs of the admitted points are all done, the run ends within the
    plan's tolerance."""

    import time

    from pyamplicol import loop_profile
    from pyamplicol.profile_plan import BUDGET_TOLERANCE, STABILITY_MINIMUM

    monkeypatch.setattr(
        loop_profile,
        "_references_agree",
        lambda a, b, t=loop_profile._AGREEMENT: t is not loop_profile._AGREEMENT,
    )
    begin = time.perf_counter()
    report = _profile_json(
        uux_ddx,
        "--ground-truth",
        "arb:30",
        "--completion-time",
        "40s",
        "--n_core",
        "3",
    )
    assert time.perf_counter() - begin <= BUDGET_TOLERANCE * 40
    setup = report["stability_setup"]
    assert setup["interrupted"] is False and setup["n_core"] == 3
    for row in report["stability"]:
        samples = _records(row["samples"])
        assert len(samples) >= STABILITY_MINIMUM
        assert {s["reference_certified"] for s in samples} == {"checks"}


@pytest.mark.loop_fast
def test_a_backend_refused_mid_run_leaves_the_other_certified(uux_ddx, monkeypatch):
    """TIRust refused from the second chunk on (in process): its row is
    refused, NinjaRS's references of the first chunk stay certified by the
    agreement, the later ones by their checks (one live backend)."""

    from pyamplicol import loop_profile
    from pyamplicol.api.errors import ReductionUnavailable

    task = loop_profile._stability_task

    def refusing(backend, precision, indices, batch, scales):
        if backend == "tirust" and min(indices) >= 16:
            raise ReductionUnavailable("refused for the test")
        return task(backend, precision, indices, batch, scales)

    monkeypatch.setattr(loop_profile, "_stability_task", refusing)
    report = _profile_json(
        uux_ddx,
        "--stability-points",
        "20",
        "--ground-truth",
        "arb:30",
        "--n_core",
        "1",
    )
    rows = {row["backend"]: row for row in report["stability"]}
    assert rows["tirust"]["refusal"] == "refused for the test"
    labels = {
        s["index"]: s["reference_certified"]
        for s in _records(rows["ninjars"]["samples"])
    }
    assert sorted(labels) == list(range(20))
    assert {labels[i] for i in range(16)} == {"agreement"}
    assert {labels[i] for i in range(16, 20)} == {"checks"}


def _records(table):
    """A columnar --json table ({"fields": ..., "rows": ...}) as records."""

    return [dict(zip(table["fields"], row, strict=True)) for row in table["rows"]]


def test_a_loop_induced_view_has_no_born(tmp_path):
    from pyamplicol import Runtime

    fixture = json.loads((FIXTURES / "li_gg_aa_nf5.json").read_text())
    fixed = ", ".join(
        f"{key} = {json.dumps(value)}"
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
    )
    artifact = _artifact(
        tmp_path,
        "g g > a a",
        "",
        fixed,
        'particle_veto = ["t"]',
    )
    runtime = Runtime.load(artifact)
    view = runtime.evaluate_orders(_points(artifact)[:1], mu_r2="10000")
    assert view.contribution == "loop-induced"
    assert view.born_orders == () and view.sum_born_orders() == (0.0,)
    assert len(view.orders) == 1
    (only,) = view.orders
    assert view.sum_orders()[0] == view.for_order(only)[0]


def test_interrupted_stability_notice_names_the_common_points():
    from pyamplicol.reporting.loop_profile import interrupted_line

    assert interrupted_line({"interrupted": False}, color=False) is None
    setup = {
        "interrupted": True,
        "requested": 1000,
        "measured_points": 68,
        "evaluated_per_backend": {"ninjars": 68, "tir": 107},
    }
    notice = interrupted_line(setup, color=False)
    # Wrapped to the line width.
    assert max(len(part) for part in notice.splitlines()) <= 100
    line = " ".join(notice.split())
    assert line.startswith("interrupted: 68 of 1000 requested points evaluated")
    assert "by every backend (" in line and "107" in line
    assert line.endswith("the tables compare those points")
    setup["evaluated_per_backend"] = {"ninjars": 68, "tir": 68}
    assert " ".join(interrupted_line(setup, color=False).split()).endswith(
        "evaluated; the tables hold those points"
    )


@pytest.mark.loop_physics
def test_the_stability_run_without_a_ground_truth(uux_ddx):
    """profile --stability --ground-truth none: no reference evaluation, the
    estimate-only tables (the header says so, no true digits), every
    sample's record with its estimate and no reference, the field
    descriptions of those tables."""

    from pyamplicol.cli.main import run_cli

    common = [
        "profile",
        str(uux_ddx),
        "--stability",
        "--stability-points",
        "6",
        "--n_core",
        "1",
        "--ground-truth",
        "none",
        "--progress",
        "off",
    ]
    stdout = io.StringIO()
    assert run_cli([*common, "--json"], stdout=stdout, stderr=io.StringIO()) == 0
    report = json.loads(stdout.getvalue())
    assert report["ground_truth"]["arithmetic"] == "none"
    for row in report["stability"]:
        assert row["estimate_only"] is True and row["true_digits"] == []
        samples = _records(row["samples"])
        assert len(samples) == 6
        for sample in samples:
            assert sample["reference_value"] is None and sample["true_digits"] is None
            assert sample["estimate_relative"] is not None
            assert sample["reference_task"] is None
        assert set(row["precision_timing"]) == {"binary64"}
    text = io.StringIO()
    assert (
        run_cli(
            [*common, "--fields-description-help", "--color", "never"],
            stdout=text,
            stderr=io.StringIO(),
        )
        == 0
    )
    rendered = text.getvalue()
    assert "none (estimates only)" in rendered
    # The counts row names what was given (only the stability points here).
    assert "6 stability points as given" in rendered
    assert "Estimated digits of binary64 (no ground truth)" in rendered
    assert "Default checks, estimates only" in rendered
    assert "true digits" not in rendered.split("Field descriptions")[0]
    assert "'none (estimates only)': no reference pass" in " ".join(rendered.split())
    assert max(len(line) for line in rendered.splitlines()) <= 100


@pytest.mark.loop_physics
def test_the_completion_time_overrules_given_counts_end_to_end(uux_ddx):
    """profile --completion-time with --stability-points and --batch-size
    given (verifier D2): the counts are planned from the time, the given
    ones named as overruled. The stability run has a deadline and its own
    point count; the timed batches are the plan's whole SIMD chunks (up to
    the default size), never the given 24, which a run with the counts
    given times as is (no planned batch can be 24, on any machine; that the
    given size does not cap the plan either is
    test_profile_plan.test_effective_benchmark_resets_every_count_and_size).
    The run takes about the time asked: within half of it to three times it
    (or the minimums when they do not fit), loose enough for a slow or busy
    runner, tight enough for a run on the default budgets instead (a
    minute of stability alone) or one that misses the time tenfold."""

    from pyamplicol.cli.main import run_cli
    from pyamplicol.config.models import BenchmarkConfig
    from pyamplicol.loops.v2.plan_export import DEFAULT_LANES
    from pyamplicol.profile_plan import STABILITY_MINIMUM

    completion = 12.0
    given_batch = 24
    assert given_batch % DEFAULT_LANES != 0
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_cli(
        [
            "profile",
            str(uux_ddx),
            "--runtime",
            "--stability",
            "--stability-points",
            "7",
            "--batch-size",
            str(given_batch),
            "--n_core",
            "1",
            "--completion-time",
            f"{completion:g}s",
            "--progress",
            "off",
            "--json",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    report = json.loads(stdout.getvalue())
    plan = report["plan"]
    assert set(plan["overruled"]) == {"--stability-points", "--batch-size"}
    assert plan["rule"] == "completion time"
    assert plan["given"] == {"runtime": False, "stability": False}
    setup = report["stability_setup"]
    assert setup["deadline"] is not None
    assert setup["points"] >= STABILITY_MINIMUM
    largest = int(BenchmarkConfig().batch_size)
    for row in report["runtime"]:
        batch = row["profile"]["batch"]
        assert batch % DEFAULT_LANES == 0 and DEFAULT_LANES <= batch <= largest, row
    asked = max(completion, plan["minimum_seconds"])
    assert 0.5 * completion <= report["seconds"] <= 3.0 * asked, (
        report["seconds"],
        plan,
    )


@pytest.mark.loop_physics
def test_the_samples_output_holds_every_record(uux_ddx, tmp_path):
    """--samples-output PATH: every sample's record in one columnar JSON
    file (the inputs, per backend its rows); --json names the file instead
    of holding them, and the human tables write it too."""

    from pyamplicol.cli.main import run_cli

    path = tmp_path / "records" / "samples.json"
    common = [
        "profile",
        str(uux_ddx),
        "--stability",
        "--stability-points",
        "6",
        "--n_core",
        "1",
        "--progress",
        "off",
        "--samples-output",
        str(path),
    ]
    stdout = io.StringIO()
    assert run_cli([*common, "--json"], stdout=stdout, stderr=io.StringIO()) == 0
    report = json.loads(stdout.getvalue())
    where = {"output": str(path), "points": 6}
    assert report["stability_setup"]["samples"] == where
    assert all(row["samples"] == where for row in report["stability"])
    records = json.loads(path.read_text())
    assert records["schema"] == "pyamplicol.profile-samples/1"
    assert len(_records(records["inputs"])) == 6
    backends = {row["backend"] for row in report["stability"]}
    assert set(records["backends"]) == backends
    for table in records["backends"].values():
        rows = _records(table)
        assert [r["index"] for r in rows] == list(range(6))
        assert all(r["reference_value"] is not None for r in rows)
    path.unlink()
    text = io.StringIO()
    assert run_cli(common, stdout=text, stderr=io.StringIO()) == 0
    assert "true digits" in text.getvalue() and path.is_file()


class _Events:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


@pytest.mark.loop_physics
def test_a_second_profile_estimates_the_whole_run_from_its_start(
    uux_ddx, tmp_path, monkeypatch
):
    """A profile remembers the artifact's costs and its total (the history
    in the pyAmpliCol cache, written with progress only): the next profile
    with the same settings has a whole-run ETA before the artifact is even
    loaded, near the first one's time; costs as small as 0.1 ms survive."""

    from pyamplicol.config import resolve_config
    from pyamplicol.loop_profile import GroundTruth, _ProfileHistory, profile_loop
    from pyamplicol.reporting.progress import ProgressStart, ProgressUpdate

    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(tmp_path / "cache"))
    config = resolve_config(
        {
            "action": "benchmark",
            "evaluation": {"artifact": str(uux_ddx)},
            "benchmark": {"runtime": True, "target_runtime": 0.5, "n_core": 1},
        }
    ).effective

    def first_root_eta(sink):
        for event in sink.events:
            if isinstance(event, ProgressStart) and event.task_id != "loop-profile":
                return None  # a phase started before any estimate
            if (
                isinstance(event, ProgressUpdate)
                and event.task_id == "loop-profile"
                and event.details.get("eta_seconds") is not None
            ):
                return float(event.details["eta_seconds"])
        return None

    first, second = _Events(), _Events()
    before = time.perf_counter()
    profile_loop(uux_ddx, config, progress=first)
    took = time.perf_counter() - before
    assert first_root_eta(first) is None
    remembered = _ProfileHistory(uux_ddx, None, GroundTruth("double-double"))
    assert remembered.cost("tirust", "binary64") is not None
    assert 0.0 < remembered.cost("tirust", "binary64") < 0.1
    profile_loop(uux_ddx, config, progress=second)
    estimate = first_root_eta(second)
    assert estimate is not None and 0.3 * took <= estimate <= 3.0 * took
    # Without progress (a library call): nothing is written.
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(tmp_path / "quiet"))
    profile_loop(uux_ddx, config)
    assert not (tmp_path / "quiet").exists()


@pytest.mark.loop_physics
def test_a_stale_history_only_seeds_the_estimate(uux_ddx, tmp_path, monkeypatch):
    """recheck2 N3: a history 100x off seeds the measurement's priors until
    this run's first measured step; from then on the costs expected of the
    steps to come are rescaled by how far off it was (a 100x-stale history
    showed ~3 h for a 5 min run)."""

    from pyamplicol.api import Runtime
    from pyamplicol.config import resolve_config
    from pyamplicol.loop_profile import GroundTruth, _measure, _Monitor
    from pyamplicol.profile_plan import Costs

    true = {("tirust", None): 0.004, ("tirust", "double-double"): 0.02}
    true |= {("ninjars", None): 0.006, ("ninjars", "double-double"): 0.06}
    runtime = Runtime.load(uux_ddx)

    class Timed:
        physics = runtime.physics

        def evaluate(self, batch, **options):
            key = (options["reduction"], options.get("arithmetic"))
            time.sleep(true[key] * len(batch))

    class Stale:
        backends = ("tirust", "ninjars")

        def cost(self, backend, precision):
            arithmetic = None if precision == "binary64" else "double-double"
            return 100.0 * true[(backend, arithmetic)]

    config = resolve_config(
        {"action": "benchmark", "evaluation": {"artifact": str(uux_ddx)}}
    ).effective
    seen: list = []
    costs = _measure(
        Timed(),
        uux_ddx,
        runtime.physics,
        config,
        ("tirust", "ninjars"),
        GroundTruth("double-double"),
        _Monitor(None, "x", None),
        0.5,
        history=Stale(),
        provisional=lambda expected, left: seen.append((dict(expected), left)),
    )
    # Before any step: the history alone (100x). After the first: rescaled.
    first, after = seen[0][0], seen[1][0]
    assert first["ninjars"].reference == pytest.approx(6.0)
    for name, cost in after.items():
        assert isinstance(cost, Costs)
        assert cost.binary64 <= 3 * true[(name, None)]
        assert cost.reference <= 3 * true[(name, "double-double")]
    assert seen[1][1] <= 3 * sum(16 * t for t in true.values()) + 1.0
    assert costs["ninjars"].binary64 == pytest.approx(0.006, rel=0.5)


def _normalised(text: str) -> str:
    """A rendered profile without what timing changes (numbers, units,
    bars, padding)."""

    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"[0-9]+(\.[0-9]+)?(e[-+][0-9]+)?", "N", text)
    text = re.sub(r"\b(s|ms|us|KiB|MiB|GiB)\b", "U", text)
    text = re.sub(r"#+", "B", text)
    text = re.sub(r"-+", "-", text)
    return re.sub(r" +", " ", text)


@pytest.mark.loop_physics
def test_the_runtime_threads_are_in_force(uux_ddx, monkeypatch):
    """--runtime-threads N: the timed calls report N threads (and the
    header says so), a one-thread timing of the same calls gives the gain;
    N is capped by the usable cores with a plain line; the values do not
    depend on the threads."""

    import pyamplicol.loop_profile as lp
    from pyamplicol.cli.main import run_cli

    monkeypatch.setattr(lp, "usable_cores", lambda: 2)
    stdout, stderr = io.StringIO(), io.StringIO()
    args = [
        "profile",
        str(uux_ddx),
        "--runtime",
        "--target-runtime",
        "0.4",
        "--runtime-threads",
        "64",
        "--progress",
        "log",
    ]
    assert run_cli([*args, "--json"], stdout=stdout, stderr=stderr) == 0
    assert "--runtime-threads 64 reduced to 2, the usable cores" in stderr.getvalue()
    report = json.loads(stdout.getvalue())
    assert "on 2 threads;" in report["runtime_setup"]
    for row in report["runtime"]:
        profile = row["profile"]
        assert profile["threads"] == 2
        assert profile["one_thread_seconds_per_point"] > 0.0
        assert profile["one_thread_calls"] >= 1


@pytest.mark.loop_physics
def test_values_do_not_depend_on_the_threads(uux_ddx):
    """A runtime loaded with its pools at 4 threads evaluates bit-identically
    to one at one thread (binary64 and double-double, both backends)."""

    import os

    from pyamplicol.api import Runtime
    from pyamplicol.benchmarking import _initial_invariant
    from pyamplicol.loop_profile import _WORKER_THREADS, pin_threads

    points = _points(uux_ddx) * 8
    scales = [_initial_invariant(point, 2) for point in points]
    saved = {name: os.environ.get(name) for name in _WORKER_THREADS}
    results = {}
    try:
        for threads in (1, 4):
            pin_threads(threads)
            runtime = Runtime.load(uux_ddx)
            results[threads] = [
                repr(
                    runtime.evaluate(
                        points,
                        mu_r2=scales,
                        reduction=backend,
                        poles="none",
                        **precision,
                    ).values
                )
                for backend in ("tirust", "ninjars")
                for precision in ({}, {"arithmetic": "double-double"})
            ]
            del runtime
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert results[1] == results[4]


@pytest.mark.loop_physics
def test_one_runtime_thread_is_the_default(uux_ddx):
    """--runtime-threads 1 prints what the default prints (the tables, the
    field descriptions, the JSON's fields), timings aside."""

    from pyamplicol.cli.main import run_cli

    def keys(value, path=""):
        if isinstance(value, dict):
            found = set()
            for key, entry in value.items():
                found |= {f"{path}/{key}"} | keys(entry, f"{path}/{key}")
            return found
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return keys(value[0], f"{path}[]")
        return set()

    outputs = {}
    for extra in ((), ("--runtime-threads", "1")):
        args = [
            "profile",
            str(uux_ddx),
            "--runtime",
            "--target-runtime",
            "0.3",
            "--progress",
            "off",
            *extra,
        ]
        text, machine = io.StringIO(), io.StringIO()
        assert (
            run_cli(
                [*args, "--fields-description-help", "--color", "never"],
                stdout=text,
                stderr=io.StringIO(),
            )
            == 0
        )
        assert run_cli([*args, "--json"], stdout=machine, stderr=io.StringIO()) == 0
        outputs[extra] = (text.getvalue(), json.loads(machine.getvalue()))
    (default_text, default_json), (one_text, one_json) = outputs.values()
    assert _normalised(default_text) == _normalised(one_text)
    assert "on one core;" in default_json["runtime_setup"]
    assert "vs 1 thread" not in default_text
    assert keys(default_json) == keys(one_json)
    assert all(row["profile"]["threads"] == 1 for row in one_json["runtime"])
    assert "one_thread_seconds_per_point" not in json.dumps(one_json)


@pytest.mark.parametrize("rule", ["reference", "measurement"])
def test_a_cost_that_cannot_change_the_counts_is_not_measured(
    uux_ddx, monkeypatch, rule
):
    """Under the default budget the measurement times only what can change
    the counts (``profile_plan.reference_affects_counts``,
    ``measurement_affects_counts``; forced here, as on an expensive process):
    without the reference's the parent evaluates binary64 only and each
    backend still times its reference; without any it evaluates nothing and
    each backend runs the minimum counts with one reference point, its costs
    measured by its own runtime timing (the plan's record)."""

    import pyamplicol.loop_profile as lp
    import pyamplicol.profile_plan as plan_module
    from pyamplicol.api import Runtime
    from pyamplicol.config import resolve_config
    from pyamplicol.loop_profile import profile_loop

    name = f"{rule}_affects_counts"
    monkeypatch.setattr(plan_module, name, lambda *args, **kwargs: False)
    # An absurd reference prior: the record and the counts row must come
    # from the reference the runtime timing measured (ptimereview B3).
    monkeypatch.setattr(lp, "_MULTIPLIER_PRIOR", {"tirust": 1e7, "ninjars": 1e7})
    calls: list = []
    evaluate = Runtime.evaluate

    def recorded(self, batch, *args, **kwargs):
        calls.append(kwargs.get("arithmetic"))
        return evaluate(self, batch, *args, **kwargs)

    monkeypatch.setattr(Runtime, "evaluate", recorded)
    config = resolve_config(
        {
            "action": "benchmark",
            "evaluation": {"artifact": str(uux_ddx)},
            # Given: the reference is timed whatever the absurd prior says.
            "benchmark": {
                "runtime": True,
                "n_core": 1,
                "ground_truth": "double-double",
            },
        }
    ).effective
    report = profile_loop(uux_ddx, config)
    assert report.runtime is not None
    for row in report.runtime:
        assert row.profile is not None and row.profile.reference is not None
        # The expected seconds from the measured costs (ptimereview m4).
        expected = report.plan["runtime"][row.backend]["expected_seconds"]
        assert expected > row.profile.batch * row.profile.seconds_per_point
    assert report.seconds is not None
    assert report.plan["minimum_seconds"] < 10.0 * report.seconds
    assert report.plan["feasible"] is True
    if rule == "reference":
        # Binary64 only in this process; the reference budget is the share's
        # quarter alone (the estimate sets none).
        assert calls and set(calls) == {None}
        return
    assert calls == []
    for row in report.runtime:
        assert (row.profile.batch, row.profile.samples) == (16, 1)
        assert row.profile.reference.points == 1
        counts = report.plan["runtime"][row.backend]
        assert (counts["batch"], counts["minimum"], counts["budget_seconds"]) == (
            16,
            1,
            0.0,
        )


def test_a_one_task_stability_run_after_the_runtime_timing(uux_ddx, monkeypatch):
    """The parent keeps (or loads again) the runtime an in-process stability
    run needs: with the points given and a single stream (one backend, no
    reference) its one task runs in this process although two workers were
    asked for (``_stability`` lowers the count to the tasks), after the
    runtime timing (ptimereview B1: AttributeError on a released runtime)."""

    import pyamplicol.loop_profile as lp
    from pyamplicol.config import resolve_config

    monkeypatch.setattr(lp, "usable_cores", lambda: 2)
    config = resolve_config(
        {
            "action": "benchmark",
            "evaluation": {"artifact": str(uux_ddx), "reduction": "tirust"},
            "benchmark": {
                "runtime": True,
                "stability": True,
                "n_core": 2,
                "ground_truth": "none",
                "stability_points": 5,
            },
        }
    ).effective
    report = lp.profile_loop(uux_ddx, config)
    assert report.runtime is not None and report.stability is not None
    (row,) = report.stability
    assert row.backend == "tirust" and len(row.per_point) == 5


def test_a_refusing_backend_takes_no_share_of_the_budget(uux_ddx, monkeypatch):
    """The no-measurement rule shares the budget among the backends that can
    serve (``_serving``: admitted by the artifact, present in the native
    build), not every requested one (ptimereview B2): NinjaRS absent from
    the native build and a 5 s load leave TIRust the whole 10 s less its
    start-up, so the parent measures and TIRust gets budgeted counts, as
    before the rule."""

    import time

    import pyamplicol.loop_profile as lp
    import pyamplicol.loops.v2.reduction as reduction
    from pyamplicol.api import Runtime
    from pyamplicol.config import resolve_config

    native_backends = reduction.native_backends

    def without_ninjars(native):
        found = dict(native_backends(native))
        found.pop("ninjars", None)
        return found

    monkeypatch.setattr(reduction, "native_backends", without_ninjars)
    load = Runtime.load.__func__

    def slow(cls, *args, **kwargs):
        time.sleep(5.0)
        return load(cls, *args, **kwargs)

    monkeypatch.setattr(Runtime, "load", classmethod(slow))
    runtime = load(Runtime, uux_ddx)
    assert lp._serving(runtime, ("ninjars", "tirust")) == ["tirust"]
    config = resolve_config(
        {
            "action": "benchmark",
            "evaluation": {"artifact": str(uux_ddx)},
            "benchmark": {"runtime": True, "n_core": 1},
        }
    ).effective
    report = lp.profile_loop(uux_ddx, config)
    rows = {row.backend: row for row in report.runtime or ()}
    tirust = rows["tirust"].profile
    assert tirust is not None
    assert (tirust.batch, tirust.samples) != (16, 1)
    assert report.plan["runtime"]["tirust"]["budget_seconds"] > 0.0


@pytest.mark.parametrize("point_seconds", [None, 1e-3, 1e3])
def test_the_warm_entry_replaces_a_dear_binary64_warm_up(uux_ddx, point_seconds):
    """With a warm entry on the runtime's backend (``prepare``, loadfast) and
    a dear binary64 point (above ``MEASURE_REPEAT_SECONDS``) the optimistic
    timing warms through it, in its backend and unchecked mode, instead of
    the untimed warm-up point: no warm-up call, the timed calls unchanged;
    an unknown or cheap point keeps the warm-up point (inert without the
    entry, as on this base)."""

    from pyamplicol.api import Runtime
    from pyamplicol.benchmarking import profile_virtual
    from pyamplicol.config import BenchmarkConfig

    runtime = Runtime.load(uux_ddx)
    prepared: list = []
    runtime._backend.prepare = lambda **kwargs: prepared.append(kwargs)
    calls: list = []
    evaluate = Runtime.evaluate

    def recorded(self, batch, *args, **kwargs):
        calls.append(len(batch))
        return evaluate(self, batch, *args, **kwargs)

    bench = BenchmarkConfig(
        batch_size=4, warmup_runs=1, minimum_samples=1, target_runtime=1e-6
    )
    Runtime.evaluate = recorded
    try:
        result = profile_virtual(
            runtime,
            bench,
            points=_points(uux_ddx) * 8,
            reduction="tirust",
            optimistic=True,
            point_seconds=point_seconds,
        )
    finally:
        Runtime.evaluate = evaluate
    if point_seconds == 1e3:
        assert prepared == [{"reduction": "tirust", "unchecked": True}]
        assert result.warmup == 0 and result.prepared_seconds is not None
        assert calls == [4]
        assert "prepared_seconds" in result.cli_json()
    else:
        assert prepared == [] and result.prepared_seconds is None
        assert result.warmup == 1 and calls == [1, 4, 4]
        assert "prepared_seconds" not in result.cli_json()
    assert result.samples == 1 and result.batch == 4


@pytest.mark.parametrize(
    ("given", "measured"),
    [
        ({}, True),
        ({}, False),
        ({"all_backends": True}, True),
        ({"ground_truth": "double-double"}, True),
        ({"ground_truth": "double-double"}, False),
        ({"stability": True, "stability_points": 4}, True),
        ({"target_runtime": 1e-6}, True),
    ],
)
def test_an_exceeded_budget_times_the_default_backend_alone(
    uux_ddx, monkeypatch, given, measured
):
    """A runtime budget the minimums exceed (forced: 1 ms) on the runtime
    timing alone (USER 2026-11-19): no reference pass and the default
    backend only, from the measurement or, without one (``measured`` False:
    the start-up takes the share), from the runtime timing's own costs; the
    record and the counts row name what was left out. --all-backends keeps
    every backend, an explicit --ground-truth the multiplier, --stability
    both (its true digits need the reference), and counts given are
    honoured as given."""

    import pyamplicol.profile_plan as plan_module
    from pyamplicol.config import resolve_config
    from pyamplicol.loop_profile import profile_loop
    from pyamplicol.reporting.loop_profile import _counts_cell

    monkeypatch.setattr(plan_module, "DEFAULT_RUNTIME_SECONDS", 1e-3)
    if not measured:
        monkeypatch.setattr(
            plan_module, "measurement_affects_counts", lambda *a, **k: False
        )
    config = resolve_config(
        {
            "action": "benchmark",
            "evaluation": {"artifact": str(uux_ddx)},
            "benchmark": {"runtime": True, "n_core": 1, **given},
        }
    ).effective
    report = profile_loop(uux_ddx, config)
    rows = {row.backend: row.profile for row in report.runtime or ()}
    # Counts given (--target-runtime) are honoured as given.
    stability = given.get("stability", False) or "target_runtime" in given
    forced = "ground_truth" in given
    every = given.get("all_backends") or stability
    assert set(rows) == ({"ninjars", "tirust"} if every else {"ninjars"})
    assert report.backends == tuple(rows)
    for profile in rows.values():
        assert profile is not None
        assert (profile.reference is not None) == (forced or stability)
    plan = report.plan
    assert plan["omitted_backends"] == ([] if every else ["tirust"])
    assert plan["omitted_reference"] == (None if forced or stability else "dd")
    text = _counts_cell(report, False) or ""
    assert ("TIRust not profiled (--all-backends)" in text) == (not every)
    assert ("no dd multiplier (--ground-truth)" in text) == (not (forced or stability))


def test_a_default_budget_keeps_every_backend_and_the_multiplier(uux_ddx):
    """On a cheap process with the default budget nothing is left out."""

    from pyamplicol.config import resolve_config
    from pyamplicol.loop_profile import profile_loop

    config = resolve_config(
        {
            "action": "benchmark",
            "evaluation": {"artifact": str(uux_ddx)},
            "benchmark": {"runtime": True, "n_core": 1},
        }
    ).effective
    report = profile_loop(uux_ddx, config)
    rows = {row.backend: row.profile for row in report.runtime or ()}
    assert set(rows) == {"ninjars", "tirust"}
    assert all(p is not None and p.reference is not None for p in rows.values())
    assert report.plan["omitted_backends"] == []
    assert report.plan["omitted_reference"] is None
