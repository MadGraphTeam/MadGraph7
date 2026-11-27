#!/usr/bin/env python3
"""Separate full-stream accepted-precision diagnostics; never speed acceptance.

Preparation only copies qualified objects. Explicit ``run --execute`` compiles
the diagnostic driver, links its private executable and executes amplitudes.
Every timed traversal contains all 100 points in order. Classification occurs
after execution; neither subsets nor forced precision are supported. Instrumented
per-call wall times include point/scale setup and clock overhead. No correction,
outlier removal, confidence-bound gate or automatic retry is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import fcntl
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

import tirust_benchmark as bench
import tirust_benchmark_chain_probe as probe
import tirust_benchmark_cohorts as cohorts
from tirust_benchmark_compiler import capture_effective_flags
import tirust_benchmark_local as local
from tirust_benchmark_worker import BACKEND_CODES, Worker

SCHEMA = "madloop-tirust-precision-stream-v1"
TEMPLATE = Path(__file__).with_suffix(".f")
LANES = ("baseline-1", "baseline-chain", "candidate-chain",
         *(f"candidate-{b}" for b in (1, 6, 7, 8, 9)))
PROTOCOLS = (Path(__file__).name, TEMPLATE.name, "tirust_benchmark.py",
             "tirust_benchmark_chain_probe.py", "tirust_benchmark_cohorts.py",
             "tirust_benchmark_local.py", "tirust_benchmark_worker.py",
             "tirust_benchmark_worker.f", "tirust_benchmark_compiler.py",
             "tirust_benchmark_campaign.py", "tirust_benchmark_warmup.py")
RAW = ("stream-clock.dat", "stream-calls.dat", "stream-values.dat",
       "clock-before.dat", "clock-after.dat", "timing.dat", "values.dat",
       "timed_status.dat", "dimensions.dat", "warmup.dat")
SCOPE = {"performance_acceptance_eligible": False,
         "timing_scope": "Instrumented full-amplitude wall time including point/scale setup and clock overhead",
         "precision_scope": "Final accepted backend/precision and MadLoop T flag; internal attempts are unobserved",
         "clock_overhead_subtracted": False,
         "full_stream": "Every loop evaluates original points 1..100; no subset classification or precision forcing",
         "abrupt_exit": "Buffered calls can be unavailable after abrupt termination; missing calls remain incomplete"}


def write(path, value):
    cohorts.exclusive_json(path, value)


def read(path):
    return json.loads(Path(path).read_text())


def roots(measurement, label):
    local.campaign._id(label)
    report, work = local.roots(measurement)
    return (local.within(report, "precision-stream/" + label),
            local.within(work, "precision-stream/" + label))


@contextmanager
def measurement_lock(measurement):
    with (local.roots(measurement)[0] / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sealed_prerequisite(value):
    """Integrity-only local.verify is insufficient: replay the complete seal."""
    proof = local.verify_numerical(value["cpu"])
    seal = bench.REPORT.parent / "numerical-prerequisites.json"
    if not seal.is_file() or read(seal) != proof:
        raise ValueError("Complete unchanged numerical prerequisite seal is required")
    return bench.digest(seal)


def lane_chain(lane, case):
    if lane not in LANES:
        raise ValueError("Unsupported diagnostic lane")
    source = lane.split("-", 1)[0]
    if lane.endswith("-chain"):
        baseline = read(bench.output_dir("baseline", case) / "generation.json")
        chain = bench.parse_chain(baseline["default_chain"])
        if source == "candidate":
            chain = [9, *(b for b in chain if b != 9)]
    else:
        chain = [int(lane.rsplit("-", 1)[1])]
    return source, chain


def object_inventory(output):
    return {key: value for key, value in probe.objects(output).items()
            if Path(key).name != "check_sa.o"}


def resource_inventory(output):
    """Original cards and filter files must never be mutated by a clone."""
    output = Path(output)
    return {p.relative_to(output).as_posix(): bench.digest(p)
            for p in sorted(output.rglob("*")) if p.is_file()
            and (p.name == "param_card.dat" or p.name == "MadLoopParams.dat"
                 or p.name.endswith(("HelFilter.dat", "LoopFilter.dat")))}


def immutable_inventory(output):
    directory = next(Path(output).glob("SubProcesses/P*_*"))
    return {"sources": bench.generated_sources(output),
            "objects": object_inventory(output), "resources": resource_inventory(output),
            "compilation": bench.digest(Path(output) / "compilation.json"),
            "generation": bench.digest(Path(output) / "generation.json"),
            "executable": bench.digest(directory / "check")}


def writable_targets(output):
    """Check all paths this adapter or SETPARA may write, before any edit."""
    output = Path(output).resolve()
    probe.mutable_paths_are_local(output)
    for p in output.rglob("*"):
        if (p.name in {"check_sa.f", "check_sa.o", "check", "param_card.dat"}
                or p.name in RAW or p.name.endswith(("HelFilter.dat", "LoopFilter.dat"))):
            if not p.resolve().is_relative_to(output):
                raise ValueError("Diagnostic writable path escapes private clone")


def validate_t_semantics(source):
    """T reflects this amplitude's MP construction, not a historical latch."""
    compact = "".join(line for line in source.splitlines()
                      if not line or line[0] not in "Cc*!").upper()
    import re
    compact = re.sub(r"\s+|\$", "", compact)
    reset = "MP_DONE=.FALSE.MP_DONE_ONCE=.FALSE.MP_PS_SET=.FALSE.STAB_INDEX=0DOING_QP_EVALS=.FALSE."
    final = "IF(MP_DONE_ONCE)THENRET_CODE_T=2ELSERET_CODE_T=1ENDIF"
    if compact.count(reset) != 1 or compact.count(final) != 1 or compact.index(reset) > compact.index(final):
        raise ValueError("Generated source does not establish the qualified call-local T semantics")


def render(template, parameters, numerical_source):
    validate_t_semantics(numerical_source)
    import re
    compact = re.sub(r"\s+|\$", "", "".join(
        line for line in numerical_source.splitlines() if not line or line[0] not in "Cc*!").upper())
    prefix = parameters["prefix"]
    expected = f"COMMON/{prefix}FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION".upper()
    observed = expected in compact
    declaration = ("      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION\n"
                   f"      COMMON/{prefix}FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,\n"
                   "     $ FPE_IN_QP_REDUCTION") if observed else ""
    before = ("            CALL_FPE(1:2,NCALL)=(/\n"
              "     $         MERGE(1,0,FPE_IN_DP_REDUCTION),\n"
              "     $         MERGE(1,0,FPE_IN_QP_REDUCTION)/)") if observed else "            CALL_FPE(1:2,NCALL)=-1"
    after = before.replace("1:2", "3:4")
    text = template.replace("@FPE_DECLARATIONS@", declaration).replace("@FPE_BEFORE@", before).replace("@FPE_AFTER@", after)
    return bench.render_worker(text, parameters), observed


def refuse_duplicate(report, case, lane):
    for p in report.parent.glob("*/specification.json"):
        previous = read(p)
        if previous["case"]["id"] == case["id"] and previous["lane"] == lane:
            raise ValueError("This measurement/case/lane already has a retained diagnostic; a new label cannot retry it")


def prepare(args):
    report, workspace = roots(args.measurement, args.label)
    with local.isolated(args.measurement) as value, measurement_lock(args.measurement):
        seal = sealed_prerequisite(value)
        cases = {c["id"]: c for c in bench.load_config()["cases"]}
        if args.case not in cases:
            raise ValueError("Unknown frozen process")
        case = cases[args.case]
        source, chain = lane_chain(args.lane, case)
        if report.exists() or workspace.exists():
            raise FileExistsError("Diagnostic label exists; all attempts are immutable")
        report.parent.mkdir(parents=True, exist_ok=True)
        with (report.parent / ".registry.lock").open("a") as registry:
            fcntl.flock(registry, fcntl.LOCK_EX)
            refuse_duplicate(report, case, args.lane)
            report.mkdir()
            workspace.mkdir(parents=True)
            protocols = report / "protocol-sources"
            protocols.mkdir()
            for name in PROTOCOLS:
                shutil.copy2(bench.ROOT / "tools" / name, protocols / name)
            spec = {"schema": SCHEMA, **SCOPE, "measurement": args.measurement,
                    "label": args.label, "case": case, "lane": args.lane,
                    "source": source, "chain": chain, "loops": args.loops,
                    "batches": args.batches, "timeout_seconds": args.timeout,
                    "cpu": value["cpu"], "created_utc": bench.utc_now(),
                    "local_manifest_sha256": bench.digest(local.roots(args.measurement)[0] / "measurement.json"),
                    "numerical_prerequisite_sha256": seal,
                    "protocols": {n: bench.digest(protocols / n) for n in PROTOCOLS}}
            write(report / "specification.json", spec)
        try:
            original = bench.output_dir(source, case)
            bench.campaign_guard("precision-prepare", inputs=read(original / "generation.json"))
            before = immutable_inventory(original)
            output = workspace / "outputs" / source / case["id"]
            prepared = probe.clone_export(original, output, protocols / "tirust_benchmark_worker.f")
            writable_targets(output)
            settings = cohorts.configure_clone(workspace, source, case, chain, bench.load_config())
            bench.validate_worker_modes(settings)
            lane_inputs = (bench.REPORT / "experiments/numerical" / ("ordinary-" + case["id"])
                           / (args.lane if not args.lane.endswith("-chain") else source + "-1")
                           / (source + "-inputs.json"))
            expected_settings = dict(read(lane_inputs)["settings"])
            expected_settings["mlreductionlib"] = "|".join(map(str, chain))
            if settings != expected_settings:
                raise ValueError("Diagnostic policy differs from sealed numerical lane")
            directory = next(output.glob("SubProcesses/P*_*"))
            validate_t_semantics((directory / "loop_matrix.f").read_text())
            points = bench.REPORT / "points" / (case["id"] + ".dat")
            metadata = read(points.with_suffix(".json"))
            if metadata["inputs"]["count"] != 100 or bench.digest(points) != metadata["sha256"]:
                raise ValueError("Expected exact registered 100-point stream")
            shutil.copy2(points, report / "points.dat")
            shutil.copy2(points.with_suffix(".json"), report / "points.json")
            # A singleton reference qualifies coefficients, not chain status identity.
            reference_lane = args.lane if not args.lane.endswith("-chain") else "baseline-1"
            reference = bench.REPORT / "experiments/numerical" / ("ordinary-" + case["id"]) / reference_lane / "result.json"
            write(report / "reference.json", read(reference)["observation"])
            if immutable_inventory(original) != before:
                raise ValueError("Preparation mutated the original qualified output")
            write(report / "preparation.json", {"status": "prepared", "original": str(original),
                  "output": str(output), "original_before": before, "clone": prepared,
                  "settings": settings, "resources": resource_inventory(output),
                  "specification_sha256": bench.digest(report / "specification.json"),
                  "reference_lane": reference_lane,
                  "points_sha256": bench.digest(report / "points.dat"),
                  "reference_sha256": bench.digest(report / "reference.json"),
                  "machine": bench.machine_identity(value["cpu"])})
        except BaseException as error:
            write(report / "preparation-failure.json", {**SCOPE, "error": repr(error)})
            raise
    return {"status": "prepared", "report": str(report), **SCOPE}


def verify_preparation(report, spec, value, *, compiled=False):
    if (spec["schema"] != SCHEMA or any(spec.get(k) != v for k, v in SCOPE.items())
            or spec["cpu"] != value["cpu"]
            or spec["local_manifest_sha256"] != bench.digest(local.roots(spec["measurement"])[0] / "measurement.json")
            or spec["numerical_prerequisite_sha256"] != sealed_prerequisite(value)):
        raise ValueError("Diagnostic registration/prerequisites changed")
    for name, checksum in spec["protocols"].items():
        if (name not in PROTOCOLS or bench.digest(report / "protocol-sources" / name) != checksum
                or bench.digest(bench.ROOT / "tools" / name) != checksum):
            raise ValueError("Diagnostic protocol source changed")
    if set(spec["protocols"]) != set(PROTOCOLS):
        raise ValueError("Incomplete diagnostic protocol inventory")
    prep = read(report / "preparation.json")
    if (bench.digest(report / "specification.json") != prep["specification_sha256"]
            or immutable_inventory(Path(prep["original"])) != prep["original_before"]
            or bench.digest(report / "points.dat") != prep["points_sha256"]
            or bench.digest(report / "reference.json") != prep["reference_sha256"]):
        raise ValueError("Qualified original or retained inputs changed")
    output = Path(prep["output"])
    expected = roots(spec["measurement"], spec["label"])[1] / "outputs" / spec["source"] / spec["case"]["id"]
    if output != expected or output.is_symlink():
        raise ValueError("Private output path changed")
    writable_targets(output)
    if object_inventory(output) != prep["original_before"]["objects"]:
        raise ValueError("Private clone numerical objects changed")
    expected_cards = {k: v for k, v in prep["resources"].items()
                      if Path(k).name in ("param_card.dat", "MadLoopParams.dat")}
    if card_inventory(output) != expected_cards:
        raise ValueError("Private clone physical/numerical cards changed")
    if compiled:
        expected_sources = dict(prep["clone"]["generated_sources"])
        driver = report / "check_sa.f"
        if driver.is_file():
            directory = next(output.glob("SubProcesses/P*_*"))
            expected_sources[(directory / "check_sa.f").relative_to(output).as_posix()] = bench.digest(driver)
        if bench.generated_sources(output) != expected_sources:
            raise ValueError("Private clone changed sources beyond the recorded driver")
    if not compiled:
        probe.verify_prepared_clone(output, prep["clone"])
        if resource_inventory(output) != prep["resources"]:
            raise ValueError("Prepared numerical resources changed")
    return prep


def build_driver(report, prep, spec):
    output = Path(prep["output"])
    directory = next(output.glob("SubProcesses/P*_*"))
    writable_targets(output)
    objects_before = object_inventory(output)
    sources_before = bench.generated_sources(output)
    deps = bench.dependency_inputs(directory)
    flags = capture_effective_flags(directory, env=bench.environment())
    original = read(output / "compilation.json")
    rendered, fpe_observed = render((report / "protocol-sources" / TEMPLATE.name).read_text(),
                                  original["worker_render_parameters"], (directory / "loop_matrix.f").read_text())
    (directory / "check_sa.f").write_text(rendered)
    (report / "check_sa.f").write_text(rendered)
    plan = cohorts.link_commands(output, directory, bench.environment(), report)
    write(report / "driver-build-plan.json", plan)
    with (report / "driver-build.log").open("x") as log:
        for command in plan["commands"]:
            subprocess.run(command, cwd=directory, env=bench.environment(), stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=spec["timeout_seconds"])
    expected = dict(sources_before)
    expected[(directory / "check_sa.f").relative_to(output).as_posix()] = bench.digest(directory / "check_sa.f")
    if (object_inventory(output) != objects_before or bench.generated_sources(output) != expected
            or bench.dependency_inputs(directory) != deps
            or not bench.same_compiler_policy(flags, capture_effective_flags(directory, env=bench.environment()))
            or immutable_inventory(Path(prep["original"])) != prep["original_before"]):
        raise ValueError("Driver build mutated numerical objects/sources/libraries/flags/original")
    receipt = {"sources": expected, "objects": objects_before, "dependencies": deps,
               "compiler_flags": flags, "executable_sha256": bench.digest(directory / "check"),
               "runtime": bench.runtime_identity(directory), "fpe_common_observed": fpe_observed,
               "cards": card_inventory(output)}
    write(report / "driver-build.json", receipt)
    return directory, receipt


def tick_delta(start, end, maximum):
    if any(type(x) is not int for x in (start, end, maximum)) or maximum <= 0 or not (0 <= start <= maximum and 0 <= end <= maximum):
        raise ValueError("Invalid clock ticks/max")
    return end - start if end >= start else maximum - start + 1 + end


def clock_pairs(path):
    lines = Path(path).read_text().splitlines()
    rate, maximum, first, last, count = map(int, lines[0].split())
    if rate <= 0 or count != 10000 or len(lines) != count + 1:
        raise ValueError("Malformed clock-pair control")
    outer = tick_delta(first, last, maximum)
    durations, previous, offset = [], first, 0
    for line in lines[1:]:
        start, end = map(int, line.split())
        offset += tick_delta(previous, start, maximum)
        delta = tick_delta(start, end, maximum)
        offset += delta
        if offset > outer:
            raise ValueError("Clock-pair chronology exceeds outer interval")
        durations.append(delta)
        previous = end
    if offset + tick_delta(previous, last, maximum) != outer:
        raise ValueError("Clock-pair chronology wraps more than once")
    return {"rate": rate, "maximum": maximum, "count": count, "outer_ticks": outer,
            "minimum_ticks": min(durations), "median_ticks": statistics.median(durations),
            "maximum_ticks": max(durations), "zero_count": durations.count(0),
            "median_seconds": statistics.median(durations) / rate,
            "trace_seconds_per_pair": outer / (rate * count), "subtracted": False}


def classify(code, chain, finite, checked):
    h, t, u = code // 100, code // 10 % 10, code % 10
    actual = BACKEND_CODES.get((h, u)) if code >= 0 else None
    result = {"status": code, "h": h, "t": t, "u": u,
              "actual_backend": actual[0] if actual else None,
              "accepted_precision": actual[1] if actual else None,
              "fallback_backend": actual[0] != chain[0] if actual else None,
              "qp_observed_via_return_code": t in (2, 4), "finite": finite, "accuracy_reported": checked}
    if not finite:
        category = "failed_nonfinite"
    elif h == 1 and t in (1, 2):
        category = "unchecked"
    elif not checked or actual is None or actual[0] not in chain or t not in (1, 2):
        category = "failed_or_inconsistent"
    elif actual[1] == "QP":
        category = "accepted_qp" if t == 2 else "failed_or_inconsistent"
    else:
        category = "accepted_dp_no_observed_qp" if t == 1 else "accepted_dp_after_qp"
    return dict(result, category=category)


def parse_batch(directory, spec, reference, fpe_observed=None):
    directory = Path(directory)
    rate, maximum, first, last = map(int, (directory / "stream-clock.dat").read_text().split())
    if rate <= 0:
        raise ValueError("Nonpositive clock rate")
    outer = tick_delta(first, last, maximum)
    ndim, nso, nborn = map(int, (directory / "dimensions.dat").read_text().split())
    if min(nso, nborn) < 0 or ndim != max(nso, nborn):
        raise ValueError("Invalid answer dimensions")
    calls = [list(map(int, line.split())) for line in (directory / "stream-calls.dat").read_text().splitlines()]
    expected = [(i, p) for i in range(1, spec["loops"] + 1) for p in range(1, 101)]
    if (any(len(r) != 9 or any(x not in (-1, 0, 1) for x in r[5:]) for r in calls)
            or [tuple(r[:2]) for r in calls] != expected):
        raise ValueError("Missing, reordered or substituted full-stream calls")
    if (any(not (all(x == -1 for x in r[5:]) or all(x in (0, 1) for x in r[5:])) for r in calls)
            or fpe_observed is True and any(-1 in r[5:] for r in calls)
            or fpe_observed is False and any(r[5:] != [-1] * 4 for r in calls)):
        raise ValueError("Software FPE observations differ from driver layout")
    lines = (directory / "stream-values.dat").read_text().splitlines()
    if len(lines) != len(calls) * (ndim + 1):
        raise ValueError("Missing all-call numerical answer columns")
    samples, final, offset, previous = [], [], 0, first
    for index, (iteration, point, code, start, end, *fpe) in enumerate(calls):
        elapsed = tick_delta(start, end, maximum)
        offset += tick_delta(previous, start, maximum) + elapsed
        if offset > outer:
            raise ValueError("Per-call clock chronology exceeds batch interval")
        previous = end
        rows, finite, checked = [], True, True
        for order in range(ndim + 1):
            fields = lines[index * (ndim + 1) + order].split()
            if len(fields) != 9 or tuple(map(int, fields[:4])) != (iteration, point, order, code):
                raise ValueError("Per-call numerical/status/order mismatch")
            numbers = list(map(float, fields[4:]))
            finite = finite and all(math.isfinite(x) for x in numbers)
            checked = checked and math.isfinite(numbers[-1]) and numbers[-1] >= 0
            rows.append({"point": point, "order": order, "status": code,
                         "coefficients": [n if math.isfinite(n) else None for n in numbers[:4]],
                         "accuracy": numbers[-1] if math.isfinite(numbers[-1]) else None})
        sample = classify(code, spec["chain"], finite, checked)
        sample.update(iteration=iteration, point=point, ticks=elapsed, seconds=elapsed / rate,
                      clock_start=start, clock_end=end, software_fpe_before=fpe[:2], software_fpe_after=fpe[2:])
        samples.append(sample)
        if iteration == spec["loops"]:
            final.extend(rows)
    if offset + tick_delta(previous, last, maximum) != outer:
        raise ValueError("Batch chronology wraps more than once")
    histogram = Counter(str(c[2]) for c in calls if 0 <= c[2] <= 499)
    recorded = {}
    for line in (directory / "timed_status.dat").read_text().splitlines():
        code, count = map(int, line.split())
        if str(code) in recorded or count <= 0:
            raise ValueError("Malformed status histogram")
        recorded[str(code)] = count
    if recorded != dict(histogram):
        raise ValueError("Status histogram differs from per-call evidence")
    ordinary = []
    for line in (directory / "values.dat").read_text().splitlines():
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Malformed final values.dat")
        p, order, code = map(int, fields[:3])
        numbers = list(map(float, fields[3:]))
        ordinary.append({"point": p, "order": order, "status": code,
                         "coefficients": [n if math.isfinite(n) else None for n in numbers[:4]],
                         "accuracy": numbers[-1] if math.isfinite(numbers[-1]) else None})
    if ordinary != final:
        raise ValueError("Final values.dat differs from all-call final traversal")
    timing = list(map(float, (directory / "timing.dat").read_text().split()))
    if (len(timing) != 2 or not all(math.isfinite(x) for x in timing) or timing[0] <= 0
            or not math.isclose(timing[1], outer / rate, rel_tol=2e-15, abs_tol=1e-18)):
        raise ValueError("timing.dat differs from raw outer ticks")
    before, after = (clock_pairs(directory / name) for name in ("clock-before.dat", "clock-after.dat"))
    if any((c["rate"], c["maximum"]) != (rate, maximum) for c in (before, after)):
        raise ValueError("Clock rate/max changed between controls and stream")
    valid_final = all(None not in r["coefficients"] and r["accuracy"] is not None for r in final)
    comparison = bench.compare_rows(reference["rows"], final) if valid_final else {"passed": False, "reason": "Nonfinite final values"}
    differences = [{"point": r["point"], "order": r["order"],
                    "reference_status": old["status"], "observed_status": r["status"],
                    "reference_accuracy": old["accuracy"], "observed_accuracy": r["accuracy"]}
                   for old, r in zip(reference["rows"], final)
                   if old["status"] != r["status"] or old["accuracy"] != r["accuracy"]]
    groups = {}
    for category in sorted({s["category"] for s in samples}):
        selected = [s for s in samples if s["category"] == category]
        groups[category] = {"calls": len(selected), "total_seconds": sum(s["seconds"] for s in selected),
                            "median_seconds": statistics.median(s["seconds"] for s in selected),
                            "zero_tick_calls": sum(s["ticks"] == 0 for s in selected)}
    return {**SCOPE, "calls": samples, "groups": groups, "clock_rate": rate,
            "clock_maximum": maximum, "outer_ticks": outer,
            "outer_seconds": outer / rate, "sum_call_ticks": sum(s["ticks"] for s in samples),
            "coefficient_comparison": comparison, "status_accuracy_differences": differences,
            "final_rows": final, "status_counts": recorded,
            "clock_before": before, "clock_after": after, "cold_seconds": timing[0],
            "all_calls_checked": all(s["category"].startswith("accepted_") for s in samples)}


def capture(directory, destination):
    cohorts.capture(directory, destination, RAW)


def card_inventory(output):
    return {k: v for k, v in resource_inventory(output).items()
            if Path(k).name in ("MadLoopParams.dat", "param_card.dat")}


def check_built(output, receipt):
    directory = next(Path(output).glob("SubProcesses/P*_*"))
    runtime = bench.runtime_identity(directory)
    if (bench.generated_sources(output) != receipt["sources"]
            or object_inventory(output) != receipt["objects"]
            or bench.digest(directory / "check") != receipt["executable_sha256"]
            or bench.dependency_inputs(directory) != receipt["dependencies"]
            or card_inventory(output) != receipt["cards"]
            or any(runtime[k] != receipt["runtime"][k] for k in ("libraries", "archives", "metadata"))
            or not bench.same_compiler_policy(receipt["compiler_flags"], capture_effective_flags(directory, env=bench.environment()))):
        raise ValueError("Diagnostic numerical build/runtime changed")


def validate_closed_batches(report, spec, rows, result):
    expected_names = [f"batch-{i:04d}" for i in range(spec["batches"])]
    if [r["directory"] for r in rows] != expected_names or any(r["status"] != "complete" for r in rows):
        raise ValueError("Closed diagnostic does not match all registered batches")
    expected = [{"directory": name, "result_sha256": bench.digest(report / name / "result.json")}
                for name in expected_names]
    if result["batches"] != expected:
        raise ValueError("Closed diagnostic batch receipt changed")


def validate_controller(seconds, observation):
    period = (observation["clock_maximum"] + 1) / observation["clock_rate"]
    if (not math.isfinite(seconds) or seconds <= 0 or seconds >= period
            or observation["outer_seconds"] > seconds + 1e-6 + 1 / observation["clock_rate"]):
        raise ValueError("Controller interval cannot establish batch timing/single clock wrap")


def run(args):
    report, workspace = roots(args.measurement, args.label)
    spec = read(report / "specification.json")
    # Exclusive marker is retained even if preflight/build fails. No relabel retry.
    write(report / "execution-started.json", {"utc": bench.utc_now(), **SCOPE})
    try:
        with local.isolated(args.measurement) as value, measurement_lock(args.measurement), ExitStack() as stack:
            if any(b in (8, 9) for b in spec["chain"]):
                lock = stack.enter_context((bench.ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX)
            prep = verify_preparation(report, spec, value)
            directory, receipt = build_driver(report, prep, spec)
            output = Path(prep["output"])
            reference = read(report / "reference.json")
            command = ["taskset", "-c", str(spec["cpu"]), "./check", str(report / "points.dat")]
            try:
                worker = Worker(command, directory, report / "worker.log", env=bench.environment(), timeout=spec["timeout_seconds"])
                stack.enter_context(worker)
            finally:
                capture(directory, report / "initialization")
            if worker.points != 100:
                raise ValueError("Worker did not initialize all 100 points")
            write(report / "ready.json", {"points": worker.points, "ndim": worker.ndim,
                  "nso": worker.nso, "nborn": worker.nborn, "cold_seconds": worker.cold_seconds,
                  "resources": resource_inventory(output)})
            batches = []
            previous_resources = resource_inventory(output)
            for index in range(spec["batches"]):
                destination = report / f"batch-{index:04d}"
                destination.mkdir()
                current_resources = resource_inventory(output)
                if current_resources != previous_resources:
                    raise ValueError("Resources changed between diagnostic batches")
                check_built(output, receipt)
                write(destination / "request.json", {"loops": spec["loops"], "host_before": bench.host_load_snapshot(),
                      "resources_before": current_resources})
                for name in RAW:
                    if name not in ("warmup.dat", "dimensions.dat", "clock-before.dat"):
                        (directory / name).unlink(missing_ok=True)
                try:
                    controller_seconds = worker.run(spec["loops"])
                finally:
                    capture(directory, destination / "raw")
                    write(destination / "after.json", {"resources_after": resource_inventory(output),
                          "host_after": bench.host_load_snapshot()})
                previous_resources = resource_inventory(output)
                observation = parse_batch(destination / "raw", spec, reference, receipt["fpe_common_observed"])
                validate_controller(controller_seconds, observation)
                write(destination / "controller.json", {"seconds": controller_seconds,
                      "scope": "Whole RUN protocol including output I/O and clock controls; not amplitude timing"})
                if observation["cold_seconds"] != worker.cold_seconds:
                    raise ValueError("Initialization timing differs from READY")
                write(destination / "result.json", observation)
                batches.append({"directory": destination.name, "result_sha256": bench.digest(destination / "result.json")})
                check_built(output, receipt)
                if immutable_inventory(Path(prep["original"])) != prep["original_before"]:
                    raise ValueError("Diagnostic mutated qualified original")
            # Context exit verifies graceful STOP/BYE and reaps the worker.
        with local.isolated(args.measurement) as value, measurement_lock(args.measurement):
            verify_preparation(report, spec, value, compiled=True)
            check_built(output, receipt)
        if resource_inventory(output) != previous_resources:
            raise ValueError("Resource history changed during worker exit")
        write(report / "results.json", {"status": "closed", **SCOPE, "batches": batches,
              "initialization_seconds": worker.cold_seconds,
              "all_batch_results_retained": True, "resources_after_exit": resource_inventory(output)})
    except BaseException as error:
        write(report / "execution-failure.json", {**SCOPE, "error": repr(error), "utc": bench.utc_now(),
              "trace_status": "Retained raw files may be incomplete; buffered calls are never reconstructed"})
        raise
    return verify(args)


def verify(args):
    report, _ = roots(args.measurement, args.label)
    spec = read(report / "specification.json")
    with local.isolated(args.measurement) as value, measurement_lock(args.measurement):
        if spec["measurement"] != args.measurement or spec["label"] != args.label:
            raise ValueError("Diagnostic label/measurement changed")
        prep = verify_preparation(report, spec, value, compiled=(report / "execution-started.json").exists())
        if not (report / "execution-started.json").exists():
            return {"status": "prepared", **SCOPE}
        if (report / "driver-build.json").exists():
            check_built(Path(prep["output"]), read(report / "driver-build.json"))
        rows = []
        previous_resources = read(report / "ready.json")["resources"] if (report / "ready.json").exists() else None
        for directory in sorted(report.glob("batch-*")):
            try:
                request = read(directory / "request.json")
                if request["loops"] != spec["loops"] or request["resources_before"] != previous_resources:
                    raise ValueError("Batch resource history differs from READY/previous batch")
                previous_resources = read(directory / "after.json")["resources_after"]
                observed = parse_batch(directory / "raw", spec, read(report / "reference.json"),
                                       read(report / "driver-build.json")["fpe_common_observed"])
                controller = read(directory / "controller.json")["seconds"]
                validate_controller(controller, observed)
                if observed["cold_seconds"] != read(report / "ready.json")["cold_seconds"]:
                    raise ValueError("Initialization timing differs from READY")
                if read(directory / "result.json") != observed:
                    raise ValueError("Stored result differs from complete raw replay")
                rows.append({"directory": directory.name, "status": "complete",
                             "all_calls_checked": observed["all_calls_checked"],
                             "coefficients_pass": observed["coefficient_comparison"]["passed"]})
            except (ValueError, OSError, KeyError) as error:
                rows.append({"directory": directory.name, "status": "incomplete_or_invalid", "error": str(error)})
        closed = (report / "results.json").is_file() and not (report / "execution-failure.json").exists()
        if closed:
            result = read(report / "results.json")
            validate_closed_batches(report, spec, rows, result)
            if resource_inventory(Path(prep["output"])) != result["resources_after_exit"]:
                raise ValueError("Closed diagnostic resources changed after worker exit")
            if previous_resources != result["resources_after_exit"]:
                raise ValueError("Final batch resource history differs from worker exit")
        return {"status": "closed" if closed else "failed_or_incomplete", **SCOPE, "batches": rows,
                "failed_calls_remain_visible": True,
                "numerical_diagnostic_passed": closed and all(r.get("all_calls_checked") and r.get("coefficients_pass") for r in rows)}


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Copy qualified objects and freeze a diagnostic; no compilation or amplitude calls")
    preparation.add_argument("--case", required=True)
    preparation.add_argument("--lane", choices=LANES, required=True)
    preparation.add_argument("--loops", type=int, default=1, help="Complete 100-point traversals per batch; fixed before running")
    preparation.add_argument("--batches", type=int, default=7)
    preparation.add_argument("--timeout", type=float, default=3600., help="Driver-build and per-command cap in seconds")
    execution = commands.add_parser("run", help="Explicit native driver build/link and full-stream amplitudes; never acceptance")
    execution.add_argument("--execute", action="store_true", required=True)
    commands.add_parser("verify", help="Replay retained raw files; no native execution")
    for command in commands.choices.values():
        command.add_argument("--measurement", required=True)
        command.add_argument("--label", required=True)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if Path.cwd().resolve() != bench.ROOT:
        raise ValueError(f"Run from {bench.ROOT}")
    if arguments.command == "prepare" and (arguments.loops < 1 or arguments.loops > 10000
            or arguments.batches < 1 or arguments.batches > 1000
            or not math.isfinite(arguments.timeout) or arguments.timeout <= 0):
        raise ValueError("Use 1..10000 loops, 1..1000 batches, and a positive finite timeout")
    outcome = {"prepare": prepare, "run": run, "verify": verify}[arguments.command](arguments)
    print(json.dumps(outcome, indent=2, sort_keys=True, allow_nan=False))
    return 0 if (outcome["status"] in ("prepared", "closed")
                 and outcome.get("numerical_diagnostic_passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
