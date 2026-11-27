#!/usr/bin/env python3
"""Bounded full-100-point shared-host observations, never performance acceptance.

Preparation does no native evaluation. Execution needs a separately reviewed
prepared specification and an explicit --execute. Original exports and formal
campaign runs remain untouched. All timing uses the existing worker unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import ctypes
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import statistics
import subprocess
import sys
import time

import tirust_benchmark as bench
from tirust_benchmark_campaign import activate_campaign
from tirust_benchmark_chain_probe import clone_export, verify_prepared_clone, validate_shared_runtime
from tirust_benchmark_compiler import validate_compiler_equality
from tirust_benchmark_worker import parse_outputs, RAW_OUTPUTS
from tirust_benchmark_warmup import parse_warmup

SCHEMA = "madloop-tirust-provisional-full100-v1"
CAMPAIGN = "split-dp-terminal-1"
MANIFEST_SHA = "1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31"
PREFLIGHT_SHA = "ee42b1968c429b0224483a5cfddac79622f8c57acfe87f05e7601b33b429ec29"
CLEANUP_SHA = "240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84"
CASE_ORDER = ("qcd_n1_uux_z", "ew_n1_udx_wp", "qcd_n2_gg_ttx", "ew_n2_epem_mupmum",
              "li_gg_h", "mixed_uux_ga", "li_gg_hh", "qcd_n3_uux_zgg", "ew_n3_epem_ttxa",
              "qcd_n4_uux_zzgg", "ew_n4_epem_mupmumaa")
BACKENDS = (1, 6, 7, 8, 9)
PROTOCOLS = ("tirust_benchmark_provisional.py", "tirust_benchmark.py", "tirust_benchmark_campaign.py",
             "tirust_benchmark_chain_probe.py", "tirust_benchmark_compiler.py",
             "tirust_benchmark_worker.py", "tirust_benchmark_worker.f", "tirust_benchmark_warmup.py")
TOTAL_SECONDS, CHAIN_SECONDS, STANDALONE_SECONDS = 1800., 120., 180.
CLEANUP_RESERVE, PAIRS, TARGET_SECONDS = 10., 3, 1.


class BudgetExpired(TimeoutError):
    pass


@contextmanager
def alarm_deadline(deadline):
    """Interrupt synchronous Python/IO work too; nested timers never extend a cap."""
    previous = signal.getsignal(signal.SIGALRM)
    before = time.monotonic()
    old_seconds, old_interval = signal.getitimer(signal.ITIMER_REAL)
    old_deadline = before + old_seconds if old_seconds else None
    def expired(number, frame):
        raise BudgetExpired("Preregistered all-in wall deadline expired")
    signal.signal(signal.SIGALRM, expired)
    effective = min(deadline, old_deadline) if old_deadline is not None else deadline
    signal.setitimer(signal.ITIMER_REAL, max(.000001, effective - time.monotonic()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if old_deadline is not None:
            signal.setitimer(signal.ITIMER_REAL, max(.000001, old_deadline - time.monotonic()), old_interval)


class Subreaper:
    """Adopt detached descendants even when their immediate parent exits fast.

    This dedicated controller creates one comparison child at a time. Existing
    direct children are excluded. Linux adoption establishes ownership for new
    direct children; process-start/session checks still guard every signal.
    """
    def children(self):
        result = set()
        for path in Path(f"/proc/{os.getpid()}/task").glob("*/children"):
            try:
                result.update(map(int, path.read_text().split()))
            except FileNotFoundError:
                pass
        return result

    def __enter__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.previous = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(self.previous), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot query Linux child subreaper")
        self.existing = self.children()
        if self.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot enable Linux child subreaper")
        return self

    def adopt(self, owned, identity):
        for pid in self.children() - self.existing:
            record = identity(pid)
            if record is not None and record["parent_pid"] == os.getpid():
                owned.records[pid] = record

    def reap(self, owned):
        for pid in owned.records:
            if pid == owned.process.pid:
                continue
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass

    def stop_unassigned(self, cleanup):
        """Cover interruption after fork but before Popen returns its handle."""
        records = {}
        for sig in (signal.SIGTERM, signal.SIGKILL):
            until = time.monotonic() + 1.
            while True:
                live = []
                for pid in self.children() - self.existing:
                    record = cleanup.identity(pid)
                    if record is None or record["parent_pid"] != os.getpid():
                        continue
                    records[pid] = record
                    if cleanup.matches(record):
                        live.append(record)
                        try:
                            os.kill(pid, sig)
                        except ProcessLookupError:
                            pass
                    else:
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                if not live or time.monotonic() >= until:
                    break
                time.sleep(.02)
        survivors = [record for record in records.values() if cleanup.matches(record)]
        return {"observed_owned_processes": list(records.values()), "survivors": survivors,
                "scope": "New Linux-adopted/direct children of this dedicated supervisor; guarded individual PID cleanup"}

    def __exit__(self, *error):
        self.libc.prctl(36, self.previous.value, 0, 0, 0)


def read(path):
    return json.loads(Path(path).read_text())


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def checked_module(name, path, checksum):
    if bench.digest(path) != checksum:
        raise ValueError(f"Reviewed helper changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locations(label):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,80}", label):
        raise ValueError("Use a new simple lowercase append-only label")
    return (bench.REPORT / "experiments/provisional-timing" / label,
            bench.WORK / "provisional-timing" / label)


def schedule(config, baseline_chains):
    if {c["id"] for c in config["cases"]} != set(CASE_ORDER):
        raise ValueError("Provisional schedule requires exactly the frozen eleven cases")
    result = []
    for case in CASE_ORDER:
        chain = bench.parse_chain(baseline_chains[case])
        result.append({"case": case, "kind": "chain", "backend": None,
                       "chains": {"baseline": chain, "candidate": [9, *chain]},
                       "cap_seconds": CHAIN_SECONDS})
    anchor_order = ("qcd_n1_uux_z", "ew_n2_epem_mupmum")
    for case in (*anchor_order, *(c for c in CASE_ORDER if c not in anchor_order)):
        for backend in BACKENDS:
            result.append({"case": case, "kind": "standalone", "backend": backend,
                           "chains": {"baseline": [1 if backend == 9 else backend], "candidate": [backend]},
                           "cap_seconds": STANDALONE_SECONDS})
    for number, row in enumerate(result, 1):
        row["id"] = f"{number:02d}-{row['kind']}-{row['case']}" + (f"-{row['backend']}" if row["backend"] else "")
    return result


@contextmanager
def private_execution(workspace):
    """Route file operations privately while original campaign guards stay live.

    This adapter does not suppress any guard: every guard sees the original
    registered WORK and the actual private worker inputs. Only build/card/file
    lookups use the isolated copies. The controller is single-threaded; its
    stdout draining thread does not call benchmark helpers.
    """
    original_work, original_guard = bench.WORK, bench.campaign_guard

    def guard(*args, **kwargs):
        private_work = bench.WORK
        try:
            bench.WORK = original_work
            return original_guard(*args, **kwargs)
        finally:
            bench.WORK = private_work

    bench.WORK, bench.campaign_guard = Path(workspace), guard
    try:
        yield
    finally:
        bench.WORK, bench.campaign_guard = original_work, original_guard


def descriptive(timings):
    if set(timings) != {"baseline", "candidate"} or any(len(v) != PAIRS for v in timings.values()):
        raise ValueError("A completed comparison requires all three pairs")
    if any(not math.isfinite(x) or x <= 0 for v in timings.values() for x in v):
        raise ValueError("Timing values must be finite and positive")
    ratios = [r / l for l, r in zip(timings["baseline"], timings["candidate"])]
    return {"median_seconds_per_point": {k: statistics.median(v) for k, v in timings.items()},
            "mad_seconds_per_point": {k: statistics.median(abs(x - statistics.median(v)) for x in v)
                                      for k, v in timings.items()},
            "paired_ratios": ratios, "geometric_mean_paired_ratio": math.exp(sum(map(math.log, ratios)) / PAIRS),
            "ratio_range": [min(ratios), max(ratios)], "paired_repetitions": PAIRS,
            "performance_acceptance_eligible": False,
            "interpretation": "Descriptive shared-host observations; no confidence-bound acceptance verdict"}


def prepare(label, cpu):
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("CPU affinity is outside the allowed set")
    report, work = locations(label)
    if report.exists() or work.exists():
        raise ValueError("Prepared labels are append-only")
    # This reviewed function validates the closed all66 numerical streams and
    # live source/runtime bindings; it does not invoke its returned timing CLI.
    preflight = checked_module("provisional_numerical_preflight",
        bench.REPORT / "timing-qualification-1/preflight.py", PREFLIGHT_SHA)
    prerequisites = preflight.verify_prerequisites()
    if prerequisites["ordinary_numerical_qualification_passed"] is not True:
        raise ValueError("All eleven ordinary numerical qualifications must pass first")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    protocols = report / "protocol-sources"
    protocols.mkdir()
    for name in PROTOCOLS:
        shutil.copy2(bench.ROOT / "tools" / name, protocols / name)
    cleanup = bench.REPORT / "numerical-qualification-2/process_cleanup.py"
    if bench.digest(cleanup) != CLEANUP_SHA:
        raise ValueError("Reviewed process cleanup changed")
    shutil.copy2(cleanup, protocols / "process_cleanup.py")
    shutil.copy2(bench.REPORT / "manifest.json", report / "campaign-manifest.json")
    shutil.copy2(bench.REPORT / "cases.json", report / "cases.json")
    shutil.copytree(bench.REPORT / "points", report / "points")
    exclusive(report / "prerequisites.json", prerequisites)
    config = bench.load_config()
    clones, chains = {}, {}
    for case in config["cases"]:
        clones[case["id"]] = {}
        for source in ("baseline", "candidate"):
            original = bench.output_dir(source, case)
            output = work / "outputs" / source / case["id"]
            clones[case["id"]][source] = clone_export(original, output, protocols / "tirust_benchmark_worker.f")
        chains[case["id"]] = read(bench.output_dir("baseline", case) / "generation.json")["default_chain"]
    exclusive(report / "preparation.json", {"clones": clones, "native_evaluation": False,
                                            "compiled_or_relinked": False})
    spec = {"schema": SCHEMA, **bench.campaign_fields(), "created_utc": bench.utc_now(),
            "purpose": "Bounded provisional shared-host timings; precise acceptance deferred locally",
            "performance_acceptance_eligible": False, "cpu_reservation": False,
            "machine": bench.machine_identity(cpu), "cpu": cpu, "schedule": schedule(config, chains),
            "total_cap_seconds": TOTAL_SECONDS, "cleanup_reserve_seconds": CLEANUP_RESERVE,
            "pairs": PAIRS, "target_batch_seconds": TARGET_SECONDS, "points": 100,
            "point_subset": None, "learning_history": "original full100 stream",
            "comparison_cleanup": {"cooperative_seconds": 2., "terminate_seconds": 2., "kill_seconds": 2.},
            "protocols": {p.name: bench.digest(p) for p in protocols.iterdir()},
            "frozen_files": {str(p.relative_to(report)): bench.digest(p) for p in sorted(report.rglob("*"))
                             if p.is_file() and "protocol-sources" not in p.parts}}
    exclusive(report / "specification.json", spec)
    verify_prepared(label)
    return report


def verify_prepared(label):
    report, work = locations(label)
    spec = read(report / "specification.json")
    config = read(report / "cases.json")
    if (spec.get("schema") != SCHEMA or spec.get("campaign_binding") != bench.campaign_binding()
            or spec.get("performance_acceptance_eligible") is not False
            or spec.get("pairs") != PAIRS or spec.get("points") != 100 or spec.get("point_subset") is not None
            or spec.get("target_batch_seconds") != TARGET_SECONDS
            or spec.get("total_cap_seconds") != TOTAL_SECONDS or spec.get("cleanup_reserve_seconds") != CLEANUP_RESERVE
            or spec.get("cpu_reservation") is not False
            or spec.get("comparison_cleanup") != {"cooperative_seconds": 2., "terminate_seconds": 2., "kill_seconds": 2.}):
        raise ValueError("Provisional scope/binding changed")
    if (bench.digest(report / "campaign-manifest.json") != MANIFEST_SHA
            or bench.digest(report / "cases.json") != bench.digest(bench.REPORT / "cases.json")):
        raise ValueError("Prepared campaign or numerical policy differs")
    for name, checksum in spec["protocols"].items():
        if bench.digest(report / "protocol-sources" / name) != checksum:
            raise ValueError("Retained protocol changed")
        if name != "process_cleanup.py" and bench.digest(bench.ROOT / "tools" / name) != checksum:
            raise ValueError("Live protocol changed after preparation")
    for name, checksum in spec["frozen_files"].items():
        if bench.digest(report / name) != checksum:
            raise ValueError("Frozen preparation/input changed")
    preparation = read(report / "preparation.json")
    chains = {case: read(work / "outputs/baseline" / case / "generation.json")["default_chain"] for case in CASE_ORDER}
    if spec["schedule"] != schedule(config, chains):
        raise ValueError("Ordered scope changed")
    bench.campaign_guard("provisional-verification")
    return spec, preparation


def capture_failure_raw(work, case, report):
    """Retain only this comparison's cleared-and-started worker streams."""
    for source in ("baseline", "candidate"):
        if not (report / (source + "-raw-cleared.json")).exists():
            continue
        output = work / "outputs" / source / case
        directories = list(output.glob("SubProcesses/P*_*"))
        if len(directories) != 1:
            continue
        destination = report / "last-raw" / source
        destination.mkdir(parents=True, exist_ok=True)
        for name in RAW_OUTPUTS:
            path = directories[0] / name
            if path.is_file():
                shutil.copy2(path, destination / name)


def comparison(label, number, deadline):
    report, work = locations(label)
    spec, preparation = verify_prepared(label)
    row = spec["schedule"][number]
    run = report / "comparisons" / row["id"]
    admission = read(run / "admission.json")
    if (admission["comparison"] != row or admission["work_deadline_monotonic"] != deadline
            or not (report / "run-start.json").exists()):
        raise ValueError("Comparison lacks its exact supervisor admission")
    config = read(report / "cases.json")
    case = next(c for c in config["cases"] if c["id"] == row["case"])
    chains = row["chains"]
    result = None
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Comparison work allowance exhausted")
        return value
    try:
        with ExitStack() as stack:
            if any(backend in (8, 9) for chain in chains.values() for backend in chain):
                lock = stack.enter_context((bench.ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                while True:
                    remaining()
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(min(.1, remaining()))
            workers, settings, inputs = {}, {}, {}
            with private_execution(work):
                for source in ("baseline", "candidate"):
                    output = bench.output_dir(source, case)
                    verify_prepared_clone(output, preparation["clones"][case["id"]][source])
                    directory = bench.subprocess_dir(source, case)
                    if (directory / "worker-protocol.fifo").exists():
                        raise RuntimeError("A private worker FIFO is still present")
                    for name in RAW_OUTPUTS:
                        (directory / name).unlink(missing_ok=True)
                    exclusive(run / (source + "-raw-cleared.json"), {"utc": bench.utc_now(), "case": case["id"]})
                    workers[source], settings[source] = bench.start_worker(stack, source, case, chains[source], config,
                        spec["cpu"], run, timeout=remaining(), points_path=report / "points" / (case["id"] + ".dat"),
                        diagnostic_template=report / "protocol-sources/tirust_benchmark_worker.f")
                    inputs[source] = read(run / (source + "-inputs.json"))
                    ready = {key: getattr(workers[source], key)
                             for key in ("points", "ndim", "nso", "nborn", "cold_seconds")}
                    exclusive(run / (source + "-ready.json"), ready)
                    learned = parse_warmup(run / (source + "-initialization"), chains[source], 100, ready)
                    exclusive(run / (source + "-learning.json"), learned)
                    shutil.copy2(output / "Cards/param_card.dat", run / (source + "-param_card.dat"))
                    shutil.copy2(output / "Cards/MadLoopParams.dat", run / (source + "-MadLoopParams.dat"))
                    for name in ("compilation", "generation"):
                        shutil.copy2(output / (name + ".json"), run / (source + "-" + name + ".json"))
                common = set(settings["baseline"]) & set(settings["candidate"]) - {"mlreductionlib"}
                if any(settings["baseline"][k] != settings["candidate"][k] for k in common):
                    raise ValueError("Numerical settings differ")
                if any(inputs["baseline"][key] != inputs["candidate"][key]
                       for key in ("param_card_sha256", "points_sha256")):
                    raise ValueError("Physical inputs differ")
                validate_compiler_equality(inputs["baseline"]["compilation"]["effective_compiler_flags"],
                                           inputs["candidate"]["compilation"]["effective_compiler_flags"])
                validate_shared_runtime(inputs)
                host, loops, observations, calibration = [], {}, {s: [] for s in chains}, {}
                calibration_records = {}
                def observe(source, count, tag):
                    workers[source].timeout = remaining()
                    before = bench.host_load_snapshot()
                    value = bench.observe(workers[source], chains[source], count, run / "raw" / source / tag, count=100)
                    host.append({"source": source, "phase": tag, "before": before, "after": bench.host_load_snapshot()})
                    exclusive(run / "host-snapshots" / (source + "-" + tag + ".json"), host[-1])
                    remaining()
                    return value
                for source in chains:
                    first = observe(source, 1, "calibrate-0")
                    count = max(1, math.ceil(TARGET_SECONDS / first["seconds"]))
                    sustained = observe(source, count, "calibrate-1")
                    loops[source] = max(1, math.ceil(count * TARGET_SECONDS / sustained["seconds"]))
                    calibration[source] = sustained
                    calibration_records[source] = {"initial": first, "sustained": sustained,
                                                   "sustained_loops": count}
                parity = [bench.compare_rows(calibration["baseline"]["rows"], calibration["candidate"]["rows"],
                                             config["relative_tolerance"])]
                exclusive(run / "calibration.json", {"loops": loops, "comparison": parity[0]})
                if not parity[0]["passed"]:
                    raise ValueError("Calibration numerical comparison failed")
                for repetition in range(PAIRS):
                    order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
                    for source in order:
                        observations[source].append(observe(source, loops[source], f"repetition-{repetition}"))
                    parity.append(bench.compare_rows(observations["baseline"][-1]["rows"],
                        observations["candidate"][-1]["rows"], config["relative_tolerance"]))
                timings = {source: [x["seconds_per_point"] for x in values] for source, values in observations.items()}
                result = {"schema": SCHEMA, "status": "complete", "performance_acceptance_eligible": False,
                          "comparison": row, "numerical_pass": all(x["passed"] for x in parity),
                          "parity": parity, "calibration_loops": loops, "calibration": calibration_records,
                          "timings": timings,
                          "statistics": descriptive(timings), "observations": observations,
                          "host_load": host, "actual_backend_precision_counts":
                              {s: bench.aggregate_usage(v) for s, v in observations.items()},
                          "usage_scope": "Final accepted amplitude results; internal attempts and FPE flags unobserved",
                          "initialization_and_warmup_seconds": {s: w.cold_seconds for s, w in workers.items()}}
            # Close workers before input/runtime postchecks and deadline completion.
        with private_execution(work):
            result["inputs_after"] = {s: bench.verify_observation_inputs(s, case, inputs[s]) for s in chains}
            for source in chains:
                verify_prepared_clone(bench.output_dir(source, case), preparation["clones"][case["id"]][source])
        remaining()
        result["evidence_files"] = {str(p.relative_to(run)): bench.digest(p) for p in sorted(run.rglob("*"))
                                    if p.is_file() and p.name != "console.log"}
        exclusive(run / "result.json", result)
        return result
    except BaseException as error:
        capture_failure_raw(work, case["id"], run)
        exclusive(run / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                  "performance_acceptance_eligible": False, "utc": bench.utc_now()})
        raise


def execute(label, started=None):
    started, started_utc = time.monotonic() if started is None else started, bench.utc_now()
    deadline = started + TOTAL_SECONDS
    report, work = locations(label)
    rows, spec, failure = [], None, None
    try:
        # The alarm includes synchronous hashing/copy/replay and activation,
        # rather than merely reporting an overrun after those operations.
        with alarm_deadline(deadline):
            exclusive(report / "run-start.json", {"utc": started_utc, "monotonic": started,
                      "total_deadline_monotonic": deadline, "performance_acceptance_eligible": False})
            spec, preparation = verify_prepared(label)
            cleanup = checked_module("provisional_owned_cleanup", report / "protocol-sources/process_cleanup.py", CLEANUP_SHA)
            with cleanup.SignalRequests() as requests, Subreaper() as reaper:
                class TrackedChild(cleanup.OwnedChild):
                    def sample(self):
                        super().sample()
                        reaper.adopt(self, cleanup.identity)
                for number, item in enumerate(spec["schedule"]):
                    requests.check()
                    if deadline - time.monotonic() <= CLEANUP_RESERVE + 1.:
                        rows.append({"comparison": item, "status": "not_run_budget", "performance_acceptance_eligible": False})
                        continue
                    finish_by = min(time.monotonic() + item["cap_seconds"], deadline)
                    work_until = finish_by - CLEANUP_RESERVE
                    # Emergency cleanup is reserved INSIDE the original cap.
                    cleanup_until = finish_by - min(2., CLEANUP_RESERVE / 3.)
                    run = report / "comparisons" / item["id"]
                    run.mkdir(parents=True, exist_ok=False)
                    exclusive(run / "admission.json", {"comparison": item, "utc": bench.utc_now(),
                        "work_deadline_monotonic": work_until, "normal_cleanup_deadline_monotonic": cleanup_until,
                        "finish_deadline_monotonic": finish_by,
                        "performance_acceptance_eligible": False})
                    command = [sys.executable, "-B", str(Path(__file__).resolve()), "_comparison", "--label", label,
                               "--number", str(number), "--deadline", repr(work_until)]
                    exclusive(run / "command.json", {"argv": command, "controller_sha256": bench.digest(__file__)})
                    owned, process, cleanup_record, status = None, None, None, "starting"
                    start = time.monotonic()
                    try:
                        with alarm_deadline(cleanup_until):
                            try:
                                with alarm_deadline(work_until):
                                    with (run / "console.log").open("w") as stream:
                                        process = subprocess.Popen(command, cwd=bench.ROOT, env=bench.environment(),
                                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                                        owned = TrackedChild(process)
                                        while True:
                                            # Adoption catches descendants reparented between samples,
                                            # including those whose controller exits before this poll.
                                            owned.sample()
                                            requests.check()
                                            if process.poll() is not None:
                                                owned.sample()
                                                break
                                            time.sleep(.1)
                                    if owned.live():
                                        cleanup_record = owned.stop(**spec["comparison_cleanup"])
                                        reaper.reap(owned)
                                        exclusive(run / "cleanup.json", cleanup_record)
                                        if cleanup_record["survivors"]:
                                            raise RuntimeError("Owned processes survived cleanup; admission blocked")
                                    capture_failure_raw(work, item["case"], run)
                                    if process.returncode != 0 or not (run / "result.json").exists():
                                        status = "unavailable_or_numerical_failure"
                                    else:
                                        result = verify_comparison(run)
                                        status = "complete" if result["numerical_pass"] else "numerical_failure"
                                    # Source/protocol drift stops the controller instead of becoming
                                    # dozens of apparent independent numerical failures.
                                    verify_prepared(label)
                                    for source in ("baseline", "candidate"):
                                        verify_prepared_clone(work / "outputs" / source / item["case"],
                                            preparation["clones"][item["case"]][source])
                            except BudgetExpired:
                                status = "unavailable_deadline"
                            finally:
                                if owned is not None:
                                    owned.sample()
                                    if owned.live():
                                        cleanup_record = owned.stop(**spec["comparison_cleanup"])
                                        bench.dump(run / "cleanup.json", cleanup_record)
                                    reaper.reap(owned)
                                    if owned.live():
                                        raise RuntimeError("Owned process cleanup is unresolved; admission blocked")
                                else:
                                    cleanup_record = reaper.stop_unassigned(cleanup)
                                    bench.dump(run / "cleanup-unassigned.json", cleanup_record)
                                    if cleanup_record["survivors"]:
                                        raise RuntimeError("Spawn-window cleanup is unresolved; admission blocked")
                            capture_failure_raw(work, item["case"], run)
                            requests.check()
                    except BaseException:
                        # No extension after the comparison deadline. The final
                        # emergency interval was reserved before finish_by.
                        with alarm_deadline(finish_by):
                            if owned is not None:
                                cleanup_record = owned.stop(cooperative_seconds=0.,
                                    terminate_seconds=0., kill_seconds=min(1., max(0., finish_by-time.monotonic())))
                                reaper.reap(owned)
                                bench.dump(run / "cleanup-interruption.json", cleanup_record)
                            else:
                                cleanup_record = reaper.stop_unassigned(cleanup)
                                bench.dump(run / "cleanup-unassigned-interruption.json", cleanup_record)
                        raise
                    row = {"comparison": item, "status": status,
                           "elapsed_seconds_including_cleanup": time.monotonic() - start,
                           "returncode": process.returncode if process is not None else None,
                           "cleanup": cleanup_record, "performance_acceptance_eligible": False}
                    with alarm_deadline(finish_by):
                        exclusive(run / "completion.json", row)
                        bench.dump(report / "progress.json", {"rows": [*rows, row], "performance_acceptance_eligible": False})
                    rows.append(row)
                requests.check()
    except BudgetExpired as error:
        failure = {"error_type": type(error).__name__, "error": str(error),
                   "scope": "Controller deadline; unresolved or uncompleted work remains unavailable"}
    if spec is not None:
        done = {row["comparison"]["id"] for row in rows}
        rows.extend({"comparison": item, "status": "not_completed_budget", "performance_acceptance_eligible": False}
                    for item in spec["schedule"] if item["id"] not in done)
    result = {"schema": SCHEMA, "status": "closed_budget" if failure else "closed", "rows": rows,
              "performance_acceptance_eligible": False, "precise_acceptance": "deferred_to_local",
              "elapsed_seconds": time.monotonic() - started, "started_utc": started_utc,
              "finished_utc": bench.utc_now(), "budget_seconds": TOTAL_SECONDS,
              "budget_exceeded": time.monotonic() > deadline, "failure": failure}
    try:
        with alarm_deadline(deadline):
            result["publication_scope"] = "within-original-overall-deadline"
            exclusive(report / "summary.json", result)
    except BudgetExpired:
        # Only a small failure/closure record may be published after exhaustion:
        # no native work, hashes, copies, replay or new sample admission. A
        # partially written summary is preserved rather than overwritten.
        result["publication_scope"] = "administrative-after-deadline; outside work cap; no measurement"
        result["status"] = "closed_budget"
        result["budget_exceeded"] = True
        result["failure"] = result["failure"] or {"error_type": "BudgetExpired",
            "error": "Overall deadline expired during closure publication", "scope": "administrative"}
        path = report / ("administrative-closure.json" if (report / "summary.json").exists() else "summary.json")
        exclusive(path, result)
    return result


def verify_comparison(run):
    """Portable raw replay of complete observations; no native execution."""
    run = Path(run)
    result = read(run / "result.json")
    if result.get("schema") != SCHEMA or result.get("performance_acceptance_eligible") is not False:
        raise ValueError("Unexpected provisional result scope")
    for name, checksum in result["evidence_files"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or bench.digest(path) != checksum:
            raise ValueError("Comparison evidence changed")
    spec = read(run.parent.parent / "specification.json")
    config = read(run.parent.parent / "cases.json")
    preparation = read(run.parent.parent / "preparation.json")
    if result["comparison"] not in spec["schedule"]:
        raise ValueError("Comparison is not in the frozen schedule")
    timings, observed = {}, {}
    calibration, inputs = {}, {}
    for source, chain in result["comparison"]["chains"].items():
        inputs[source] = read(run / (source + "-inputs.json"))
        prepared = preparation["clones"][result["comparison"]["case"]][source]
        for name in ("compilation", "generation"):
            path = run / (source + "-" + name + ".json")
            if bench.digest(path) != prepared[name + "_sha256"] or read(path) != inputs[source][name]:
                raise ValueError("Recorded build differs from the prepared clone")
        if inputs[source].get("campaign_binding") != spec.get("campaign_binding"):
            raise ValueError("Worker campaign binding differs")
        points = run.parent.parent / "points" / (result["comparison"]["case"] + ".dat")
        if (bench.digest(points) != inputs[source]["points_sha256"]
                or bench.digest(points.with_suffix(".json")) != inputs[source]["points_metadata_sha256"]):
            raise ValueError("Frozen physical point stream changed")
        options = dict(config["madloop_parameters"])
        if source == "candidate":
            options.update(config["candidate_parameters"])
        if (inputs[source]["settings"].get("mlreductionlib") != "|".join(map(str, chain))
                or any(inputs[source]["settings"].get(k.lower()) != v for k, v in options.items())):
            raise ValueError("Numerical policy differs from the frozen settings")
        ready = read(run / (source + "-ready.json"))
        if parse_warmup(run / (source + "-initialization"), chain, 100, ready) != read(run / (source + "-learning.json")):
            raise ValueError("Full100 learning replay differs")
        cal = result["calibration"][source]
        if (parse_outputs(run / "raw" / source / "calibrate-0", chain, 100, 1) != cal["initial"]
                or parse_outputs(run / "raw" / source / "calibrate-1", chain, 100, cal["sustained_loops"]) != cal["sustained"]):
            raise ValueError("Calibration raw replay differs")
        calibration[source] = cal["sustained"]
        after = result["inputs_after"][source]
        if (any(inputs[source]["runtime"][key] != after["runtime"][key] for key in ("libraries", "archives", "metadata"))
                or after["points_sha256"] != inputs[source]["points_sha256"]
                or after["executable_sha256"] != inputs[source]["compilation"]["executable_sha256"]
                or after["numerical_cards"] != inputs[source]["numerical_cards"]
                or bench.digest(run / (source + "-param_card.dat")) != inputs[source]["param_card_sha256"]
                or bench.digest(run / (source + "-MadLoopParams.dat")) != inputs[source]["numerical_cards"]["Cards/MadLoopParams.dat"]):
            raise ValueError("Before/after numerical input binding differs")
        observed[source] = []
        loops = result["calibration_loops"][source]
        for repetition in range(PAIRS):
            value = parse_outputs(run / "raw" / source / f"repetition-{repetition}", chain, 100, loops)
            if value != result["observations"][source][repetition]:
                raise ValueError("Raw timing/value/status replay differs")
            observed[source].append(value)
        timings[source] = [x["seconds_per_point"] for x in observed[source]]
    if timings != result["timings"] or descriptive(timings) != result["statistics"]:
        raise ValueError("Descriptive timing summary differs")
    parity = [bench.compare_rows(calibration["baseline"]["rows"], calibration["candidate"]["rows"], config["relative_tolerance"])]
    parity.extend(bench.compare_rows(observed["baseline"][i]["rows"], observed["candidate"][i]["rows"],
                                    config["relative_tolerance"]) for i in range(PAIRS))
    if parity != result["parity"] or result["numerical_pass"] != all(x["passed"] for x in parity):
        raise ValueError("Numerical comparison replay differs")
    if result["actual_backend_precision_counts"] != {s: bench.aggregate_usage(v) for s, v in observed.items()}:
        raise ValueError("Accepted backend/precision counts differ")
    if any(key in result["statistics"] for key in ("passed", "upper_95", "lower_95", "threshold", "acceptance_eligible")):
        raise ValueError("Provisional timing must not contain a confidence-bound speed gate")
    return result


def main():
    entry_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "_comparison"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--cpu", type=int, default=9)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--number", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    if args.command == "run":
        with alarm_deadline(entry_started + TOTAL_SECONDS):
            activate_campaign(bench, CAMPAIGN)
    else:
        activate_campaign(bench, CAMPAIGN)
    if bench.CAMPAIGN_MANIFEST_SHA256 != MANIFEST_SHA:
        parser.error("This proposal is bound to the reviewed source campaign")
    if args.command == "prepare":
        print(prepare(args.label, args.cpu))
    elif args.command == "verify":
        verify_prepared(args.label)
        print("prepared inputs verified; no native evaluation")
    elif args.command == "run":
        if not args.execute:
            parser.error("Runtime requires reviewed admission and explicit --execute")
        try:
            print(json.dumps(execute(args.label, started=entry_started)))
        except BaseException as error:
            report, _ = locations(args.label)
            exclusive(report / "controller-failure.json", {"error_type": type(error).__name__, "error": str(error),
                      "utc": bench.utc_now(), "performance_acceptance_eligible": False,
                      "status": "unavailable_controller_failure",
                      "publication_scope": ("administrative-after-deadline; outside work cap; no measurement"
                          if time.monotonic() >= entry_started + TOTAL_SECONDS else "administrative-failure-record")})
            raise
    else:
        if args.number is None or args.deadline is None:
            parser.error("Internal comparison requires its supervisor admission")
        comparison(args.label, args.number, args.deadline)


if __name__ == "__main__":
    main()
