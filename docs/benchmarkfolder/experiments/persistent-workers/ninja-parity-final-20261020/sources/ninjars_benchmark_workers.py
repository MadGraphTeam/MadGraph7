#!/usr/bin/env python3
"""Experimental persistent MadLoop workers, isolated from accepted measurements.

Each backend initializes and warms once. Commands run serially on the same CPU;
all other workers block on stdin. Final timed traversals retain numerical outputs.
The prototype writes only experiments/persistent-workers, never benchmark data/.
"""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time

import ninjars_benchmark as bench

ROOT = bench.ROOT
SCHEME = "persistent-worker-v1"
OUTPUTS = ("timing.dat", "values.dat", "timed_status.dat")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(work):
    result = os.environ.copy()
    result.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  XDG_CONFIG_HOME=str(work / "config"),
                  MAKEFLAGS="SHELL=" + shutil.which("bash"))
    libraries = [bench.BASELINE / "HEPTools/ninja/lib",
                 bench.BASELINE / "HEPTools/collier", ROOT / "HEPTools/ninjars/lib"]
    result["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries)) + (
        os.pathsep + result["LD_LIBRARY_PATH"] if result.get("LD_LIBRARY_PATH") else "")
    locks = ROOT / ".ninjars-build/symbolica-locks"
    locks.mkdir(parents=True, exist_ok=True, mode=0o700)
    result.setdefault("SYMBOLICA_LOCK_DIR", str(locks))
    return result


@contextmanager
def isolated_harness(work, report):
    """Reuse validators/configuration without writing into core output trees."""
    original = bench.WORK, bench.REPORT
    bench.WORK, bench.REPORT = work, report
    try:
        yield
    finally:
        bench.WORK, bench.REPORT = original


class ProtocolError(RuntimeError):
    pass


class Worker:
    """One child and one dedicated FIFO; drain diagnostic stdout continuously."""
    def __init__(self, command, directory, log, env=None, timeout=3600.):
        self.directory, self.log = Path(directory), Path(log)
        self.timeout, self.sequence = timeout, 0
        self.pipe = self.directory / "worker-protocol.fifo"
        self.pipe.unlink(missing_ok=True)
        os.mkfifo(self.pipe, mode=0o600)
        self.fd = os.open(self.pipe, os.O_RDWR | os.O_NONBLOCK)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.fd, selectors.EVENT_READ)
        self.buffer = bytearray()
        self.process = None
        self.thread = None
        self.stream = None
        self.closed = False
        try:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.log.open("wb")
            self.process = subprocess.Popen(
                [*command, str(self.pipe)], cwd=self.directory, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True)
            self.thread = threading.Thread(target=self._drain, daemon=True)
            self.thread.start()
            fields = self._message()
            if len(fields) != 7 or fields[:2] != ["READY", "1"]:
                raise ProtocolError(f"Invalid READY message: {fields}")
            self.points, self.ndim, self.nso, self.nborn = map(int, fields[2:6])
            self.cold_seconds = float(fields[6])
            if (self.points <= 0 or self.nso < 0 or self.nborn < 0 or
                    self.ndim != max(self.nso, self.nborn) or self.nborn > self.nso or
                    not math.isfinite(self.cold_seconds) or self.cold_seconds <= 0):
                raise ProtocolError(f"Invalid READY dimensions/timing: {fields}")
        except BaseException:
            self.close(graceful=False)
            raise

    def _drain(self):
        while True:
            chunk = self.process.stdout.read1(65536)
            if not chunk:
                return
            self.stream.write(chunk)
            self.stream.flush()

    def _message(self):
        deadline = time.monotonic() + self.timeout
        while True:
            if b"\n" in self.buffer:
                line, _, remainder = self.buffer.partition(b"\n")
                self.buffer = bytearray(remainder)
                try:
                    return line.decode("ascii").split()
                except UnicodeDecodeError as error:
                    raise ProtocolError("Non-ASCII worker protocol") from error
            if time.monotonic() >= deadline:
                raise ProtocolError(f"Worker protocol timeout; see {self.log}")
            ready = self.selector.select(min(.2, max(0., deadline-time.monotonic())))
            if ready:
                self.buffer.extend(os.read(self.fd, 65536))
                if len(self.buffer) > 4096:
                    raise ProtocolError("Oversized worker protocol message")
                continue
            status = self.process.poll()
            if status is not None:
                raise ProtocolError(f"Worker exited ({status}) before protocol reply; see {self.log}")

    def _send(self, line):
        try:
            self.process.stdin.write(line.encode("ascii") + b"\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise ProtocolError(f"Worker command pipe closed; see {self.log}") from error

    def run(self, loops):
        if isinstance(loops, bool) or not isinstance(loops, int) or loops <= 0:
            raise ValueError("loops must be a positive integer")
        for name in OUTPUTS:
            (self.directory / name).unlink(missing_ok=True)
        self.sequence += 1
        start = time.perf_counter()
        self._send(f"RUN {self.sequence} {loops}")
        message = self._message()
        if message != ["DONE", str(self.sequence), str(loops)]:
            raise ProtocolError(f"Unexpected worker completion: {message}")
        return time.perf_counter() - start

    def close(self, graceful=True):
        if self.closed:
            return
        self.closed = True
        shutdown_error = None
        def terminate_group(sig):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
        try:
            if (graceful and self.process is not None and self.process.poll() is not None):
                shutdown_error = ProtocolError(
                    f"Worker exited ({self.process.returncode}) before STOP/BYE")
            if self.process is not None and self.process.poll() is None:
                if graceful:
                    try:
                        self._send("STOP 0 0")
                        old_timeout, self.timeout = self.timeout, min(self.timeout, 10.)
                        try:
                            if self._message() != ["BYE"]:
                                raise ProtocolError("Missing graceful worker shutdown")
                        finally:
                            self.timeout = old_timeout
                        self.process.wait(timeout=10)
                        if self.process.returncode:
                            raise ProtocolError(
                                f"Worker exited ({self.process.returncode}) after BYE")
                    except (ProtocolError, subprocess.TimeoutExpired) as error:
                        shutdown_error = error
                        terminate_group(signal.SIGTERM)
                else:
                    terminate_group(signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    terminate_group(signal.SIGKILL)
                    self.process.wait()
        finally:
            if self.process is not None:
                # Reap any subprocess launched by this worker before a failure.
                # start_new_session isolates this group from every other worker.
                terminate_group(signal.SIGTERM)
                if self.process.stdin:
                    self.process.stdin.close()
                if self.thread:
                    self.thread.join(timeout=10)
                if self.process.stdout:
                    self.process.stdout.close()
            if self.stream:
                self.stream.close()
            self.selector.close()
            os.close(self.fd)
            self.pipe.unlink(missing_ok=True)
        if shutdown_error is not None:
            raise ProtocolError(f"Worker shutdown failed: {shutdown_error}") from shutdown_error


def parse_outputs(directory, backend, points, loops):
    """Parse every physical order and validate every observed status."""
    timing = list(map(float, (directory / "timing.dat").read_text().split()))
    if len(timing) != 2 or not all(math.isfinite(x) and x > 0 for x in timing):
        raise RuntimeError("Invalid worker timing")
    dimensions = list(map(int, (directory / "dimensions.dat").read_text().split()))
    if len(dimensions) != 3:
        raise RuntimeError("Invalid worker dimensions")
    ndim, nso, nborn = dimensions
    if ndim != max(nso, nborn) or nborn > nso:
        raise RuntimeError("Worker cannot omit Born-only split-order columns")
    rows, counts = [], {}
    for line in (directory / "values.dat").read_text().splitlines():
        fields = line.split()
        if len(fields) != 8:
            raise RuntimeError("Malformed worker numerical row")
        row = {"point": int(fields[0]), "order": int(fields[1]), "status": int(fields[2]),
               "coefficients": list(map(float, fields[3:7])), "accuracy": float(fields[7])}
        if not all(math.isfinite(x) for x in row["coefficients"] + [row["accuracy"]]):
            raise RuntimeError("Nonfinite worker numerical row")
        if row["accuracy"] < 0:
            raise RuntimeError("Missing worker stability estimate")
        bench.validate_status(row["status"], backend)
        if row["order"] == 0:
            key = str(row["status"])
            counts[key] = counts.get(key, 0) + 1
        rows.append(row)
    if [(row["point"], row["order"]) for row in rows] != [
            (p, order) for p in range(1, points+1) for order in range(nso+1)]:
        raise RuntimeError("Missing, duplicated or unordered worker numerical rows")
    timed = {}
    for line in (directory / "timed_status.dat").read_text().splitlines():
        code, count = map(int, line.split())
        bench.validate_status(code, backend)
        if count <= 0 or str(code) in timed:
            raise RuntimeError("Invalid worker status histogram")
        timed[str(code)] = count
    if sum(timed.values()) != loops*points:
        raise RuntimeError("Worker evaluation count differs from command")
    return {"cold_seconds": timing[0], "seconds": timing[1],
            "seconds_per_point": timing[1]/(loops*points), "rows": rows,
            "status_counts": counts, "timed_status_counts": timed,
            "answer_dimensions": {"allocated": ndim, "loop_orders": nso,
                                  "born_orders": nborn}}


def validate_param_card(directory, expected):
    """Follow LHA_READ.F's actual SETPARA('param_card.dat') search order."""
    candidates = [directory / "param_card.dat"] + [
        directory.joinpath(*([".."] * level), "Cards/param_card.dat") for level in range(6)]
    actual = next((path for path in candidates if path.is_file()), None)
    if actual is None or actual.read_bytes() != expected:
        raise RuntimeError("Worker subprocess reads a different parameter card")
    return actual


def prepare(source, case, backend, config, work, report, clean=False):
    original = bench.output_dir(source, case)
    original_directory = bench.subprocess_dir(source, case)
    original_compilation = json.loads((original / "compilation.json").read_text())
    if digest(original_directory / "check") != original_compilation["executable_sha256"]:
        raise RuntimeError("Core executable differs from compilation metadata")
    expected_manifest = original_compilation.get("generated_sources_sha256")
    if (expected_manifest is not None and
            bench.generated_source_manifest(original)["files_sha256"] != expected_manifest):
        raise RuntimeError("Core generated sources differ from compilation metadata")
    job = work / f"{case['id']}-{source}-{backend}"
    target = job / "outputs" / source / case["id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"Worker output already exists: {target}")
    subprocess.run(["cp", "-a", "--reflink=auto", str(original), str(target)], check=True)
    # Preserve shared read-only dependencies, but redirect any absolute links
    # into the source output so configuring/rebuilding a worker cannot alter it.
    for path in target.rglob("*"):
        if path.is_symlink():
            resolved = path.resolve()
            if resolved.is_relative_to(original):
                replacement = target / resolved.relative_to(original)
                path.unlink()
                path.symlink_to(os.path.relpath(replacement, path.parent))
    with isolated_harness(job, report):
        directory = bench.subprocess_dir(source, case)
        settings = bench.configure(source, case, backend, config=config)
        validate_param_card(directory, (target / "Cards/param_card.dat").read_bytes())
        # configure() intentionally changes cards; physics and build source must
        # still agree with the source manifest recorded for the core executable.
        shutil.copy2(directory / "check", directory / "check_fresh")
        fresh_hash = digest(directory / "check_fresh")
        fresh_driver_hash = digest(directory / "check_sa.f")
        original_manifest = bench.generated_source_manifest(target)
        prefix = re.search(r"CALL\s+(\w*)GET_ANSWER_DIMENSION",
                           (directory / "check_sa.f").read_text(), re.I).group(1)
        born = directory / "nsqso_born.inc"
        nborn = int(re.search(r"NSQSO_BORN\s*=\s*(\d+)", born.read_text(),
                             re.I).group(1)) if born.exists() else 0
        worker_source = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
        worker_source = worker_source.replace("@PREFIX@", prefix).replace(
            "@NEXTERNAL@", str(case["nfinal"]+2)).replace("@NBORN@", str(nborn))
        (directory / "check_sa.f").unlink()
        (directory / "check_sa.f").write_text(worker_source)
        (directory / "check_sa.o").unlink(missing_ok=True)
        if clean:
            for path in directory.glob("*.o"):
                path.unlink()
            (directory / "check").unlink(missing_ok=True)
        start = time.perf_counter()
        log = report / "logs" / f"compile-{case['id']}-{source}-{backend}.log"
        bench.run(["make", "-j4", "check"], directory, log)
        build_seconds = time.perf_counter()-start
        manifest = bench.generated_source_manifest(target)
        metadata = {
            "source": source, "case": case["id"], "backend": backend,
            "worker_scheme": SCHEME, "acceptance_eligible": False,
            "build_kind": "clean-subprocess" if clean else "driver-relink",
            "build_scope": "Generated subprocess Fortran; copied cached model/DHELAS/dependency libraries",
            "compilation_seconds": build_seconds,
            "compiler": subprocess.check_output(["gfortran", "--version"],
                                               text=True).splitlines()[0],
            "origin_compilation": original_compilation,
            "generation_source_identity": bench.generation_identity(source, case),
            "original_generated_sources_sha256": original_manifest["files_sha256"],
            "generated_sources_sha256": manifest["files_sha256"],
            "driver_sha256": digest(directory / "check_sa.f"),
            "executable_sha256": digest(directory / "check"),
            "fresh_executable_sha256": fresh_hash, "fresh_driver_sha256": fresh_driver_hash,
            "parameter_card_sha256": digest(target / "Cards/param_card.dat"),
            "madloop_card_sha256": digest(target / "Cards/MadLoopParams.dat"),
            "shared_numerical_settings": settings,
        }
        retained = report / "builds" / f"{case['id']}-{source}-{backend}"
        bench.dump(retained / "build.json", metadata)
        bench.dump(retained / "compiled-sources.json", manifest)
        shutil.copy2(directory / "check_sa.f", retained / "worker.f")
        for relative in ("Source/make_opts", "SubProcesses/MadLoop_makefile_definitions",
                         "Cards/param_card.dat", "Cards/MadLoopParams.dat"):
            shutil.copy2(target / relative, retained / Path(relative).name)
    return job, directory, metadata


def numerical_parity(worker_result, fresh_result, config):
    # Executables deliberately have distinct drivers. All numerical inputs,
    # settings, physical columns and CPU must still agree.
    comparison = dict(fresh_result, executable_sha256=worker_result["executable_sha256"])
    bench.validate_repetition(worker_result, comparison, config)
    exact = worker_result["rows"] == fresh_result["rows"]
    status_match = worker_result["status_counts"] == fresh_result["status_counts"]
    if not status_match:
        raise RuntimeError("Persistent and fresh workers disagree on backend/rescue counts")
    return {"passed": True, "exact_rows": exact, "status_counts_match": status_match}


def run_case(case, sources, backends, config, args, work, report, run_id):
    targets = [(source, backend) for source in sources for backend in backends
               if not (source == "baseline" and backend == 8)]
    pair = [("baseline", 6), ("candidate", 8)]
    if all(target in targets for target in pair):
        targets = pair + [target for target in targets if target not in pair]
    points = bench.REPORT / "points" / f"{case['id']}.dat"
    point_count = int(points.read_text().splitlines()[0])
    if point_count != config["points_per_case"]:
        raise RuntimeError("Unexpected retained point count")
    jobs, workers, calibrations, loops = {}, {}, {}, {}
    active, stage, active_repetition = None, "prepare", None
    try:
        # Compile all workers first. Initialization and all subsequent commands
        # are serial; idle workers remain blocked and consume no evaluation CPU.
        for source, backend in targets:
            active = source, backend
            jobs[source, backend] = prepare(source, case, backend, config, work, report,
                                            clean=args.clean_build)
        for source, backend in targets:
            key = source, backend
            active, stage = key, "initialize"
            job, directory, metadata = jobs[key]
            result_base = {**metadata, "cpu": args.cpu, "case": case["id"],
                           "points_sha256": digest(points),
                           "measurement_run_id": run_id,
                           "paired_baseline_candidate": all(target in targets for target in pair),
                           "worker_scheme": SCHEME, "acceptance_eligible": False,
                           "captured_during_timed_final_traversal": True}
            fresh = None
            if args.fresh_parity:
                stage = "fresh-parity"
                with isolated_harness(job, report):
                    bench.configure(source, case, backend, config=config)
                validate_param_card(directory, (
                    job / "outputs" / source / case["id"] / "Cards/param_card.dat").read_bytes())
                log = report / "logs" / f"fresh-{case['id']}-{source}-{backend}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w") as stream:
                    subprocess.run(["taskset", "-c", str(args.cpu), "./check_fresh",
                                    str(points), "1"], cwd=directory, env=environment(job),
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=args.timeout, check=True)
                fresh = {**result_base, **parse_outputs(directory, backend, point_count, 1),
                         "executable_sha256": metadata["fresh_executable_sha256"],
                         "repetition": "fresh-parity", "loops": 1}
                bench.dump(report / "data" / f"{case['id']}-{source}-{backend}-fresh.json", fresh)
            with isolated_harness(job, report):
                bench.configure(source, case, backend, config=config)
            validate_param_card(directory, (
                job / "outputs" / source / case["id"] / "Cards/param_card.dat").read_bytes())
            stage = "initialize"
            worker = Worker(["taskset", "-c", str(args.cpu), "./check", str(points)],
                            directory, report / "logs" / f"worker-{case['id']}-{source}-{backend}.log",
                            env=environment(job), timeout=args.timeout)
            workers[key] = worker
            if worker.points != point_count:
                raise ProtocolError("Worker initialized with a different point count")
            stage, active_repetition = "calibration", "calibration"
            wall = worker.run(1)
            result = {**result_base, **parse_outputs(directory, backend, point_count, 1),
                      "repetition": "calibration", "loops": 1, "command_wall_seconds": wall}
            if result["answer_dimensions"] != {
                    "allocated": worker.ndim, "loop_orders": worker.nso, "born_orders": worker.nborn}:
                raise ProtocolError("Worker READY dimensions differ from numerical output")
            bench.dump(report / "data" / f"{case['id']}-{source}-{backend}-calibration.json", result)
            if fresh:
                parity = numerical_parity(result, fresh, config)
                bench.dump(report / "data" / f"{case['id']}-{source}-{backend}-parity.json",
                           {**parity, "worker_executable_sha256": result["executable_sha256"],
                            "fresh_executable_sha256": fresh["executable_sha256"]})
            calibrations[key] = result
            loops[key] = max(1, math.ceil(args.minimum_seconds / result["seconds"]))
            print(case["id"], source, backend, "ready", result["seconds_per_point"], flush=True)
        for repetition in range(args.repetitions):
            order = targets if repetition % 2 == 0 else list(reversed(targets))
            for key in order:
                active, stage, active_repetition = key, "measurement", repetition
                source, backend = key
                worker = workers[key]
                wall = worker.run(loops[key])
                result = {**calibrations[key],
                          **parse_outputs(worker.directory, backend, point_count, loops[key]),
                          "repetition": repetition, "loops": loops[key],
                          "command_wall_seconds": wall, "worker_sequence": worker.sequence,
                          "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                bench.validate_repetition(result, calibrations[key], config)
                bench.dump(report / "data" / f"{case['id']}-{source}-{backend}-{repetition}.json", result)
                print(case["id"], source, backend, repetition, result["seconds_per_point"], flush=True)
    except BaseException as error:
        directory = jobs[active][1] if active in jobs else None
        worker = workers.get(active)
        failure = {
            "case": case["id"], "source": active[0] if active else None,
            "backend": active[1] if active else None, "stage": stage,
            "repetition": active_repetition,
            "command_sequence": worker.sequence if worker else None,
            "error_type": type(error).__name__, "error": str(error),
            "acceptance_eligible": False, "run_status": "failed",
            "diagnostic_log": str(worker.log) if worker else None,
            "raw_outputs": {name: (directory / name).read_text(errors="replace")
                            for name in (*OUTPUTS, "dimensions.dat")
                            if directory is not None and (directory / name).exists()},
        }
        bench.dump(report / "rejected" / f"{case['id']}-failure.json", failure)
        raise
    finally:
        cleanup_errors = []
        for worker in reversed(list(workers.values())):
            try:
                worker.close()
            except BaseException as error:
                cleanup_errors.append(str(error))
        if cleanup_errors:
            bench.dump(report / "rejected" / f"{case['id']}-cleanup-failure.json",
                       {"errors": cleanup_errors, "acceptance_eligible": False})
            if sys.exc_info()[0] is None:
                raise RuntimeError("Worker cleanup failed: " + "; ".join(cleanup_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("baseline", "candidate", "both"), default="both")
    parser.add_argument("--backends", default="1,6,7,8")
    parser.add_argument("--cases", default="qcd-1,qcd-2")
    parser.add_argument("--config", type=Path, default=bench.REPORT / "cases.json")
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--minimum-seconds", type=float, default=1.)
    parser.add_argument("--timeout", type=float, default=7200.)
    parser.add_argument("--fresh-parity", action="store_true")
    parser.add_argument("--clean-build", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if (args.cpu not in os.sched_getaffinity(0) or args.repetitions <= 0 or
            not math.isfinite(args.minimum_seconds) or args.minimum_seconds <= 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid CPU, repetition count, duration or timeout")
    backends = [int(value) for value in args.backends.split(",")]
    if not backends or len(backends) != len(set(backends)) or any(
            value not in (1, 6, 7, 8) for value in backends):
        parser.error("Backends must be distinct members of 1,6,7,8")
    if args.source == "baseline" and 8 in backends:
        parser.error("Original baseline does not provide NinjaRS")
    config = json.loads(args.config.read_text())
    selected = args.cases.split(",")
    cases = [case for case in config["cases"] if args.cases == "all" or case["id"] in selected]
    if not cases or (args.cases != "all" and len(cases) != len(set(selected))):
        parser.error("Unknown or empty case selection")
    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        parser.error("run-id must contain only letters, digits, underscores and hyphens")
    report = bench.REPORT / "experiments/persistent-workers" / run_id
    work = bench.WORK / "persistent-workers" / run_id
    if report.exists() or work.exists():
        parser.error("Use a new run-id; existing experimental data is never overwritten")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    (report / "sources").mkdir()
    for name in ("ninjars_benchmark_worker.f", "ninjars_benchmark_workers.py",
                 "ninjars_benchmark.py"):
        shutil.copy2(ROOT / "tools" / name, report / "sources" / name)
    bench.dump(report / "protocol.json", {
        "schema": SCHEME, "acceptance_eligible": False, "configuration": config,
        "worker_source_sha256": digest(ROOT / "tools/ninjars_benchmark_worker.f"),
        "controller_source_sha256": digest(Path(__file__)),
        "cpu": args.cpu, "repetitions": args.repetitions,
        "sources": args.source, "backends": backends,
        "backend_order": "Baseline Ninja and candidate NinjaRS adjacent, reverse order on odd repetitions",
        "initialization_policy": "One initialization plus full warmed stream per worker",
        "timing_policy": "Full evaluation, statuses every call, array copies on final timed traversal",
        "stdout_policy": "Continuously drained during initialization; suppressed during timing",
        "execution_policy": "One active command at a time; all workers pinned to one CPU",
        "acceptance_requires_fresh_process_parity": True,
        "uncertainty_scope": "Warmed blocks within retained processes; initialization variability is separate",
        "fresh_parity": args.fresh_parity, "clean_subprocess_build": args.clean_build})
    sources = ("baseline", "candidate") if args.source == "both" else (args.source,)
    status = {"status": "incomplete", "acceptance_eligible": False,
              "completed_cases": [], "requested_cases": [case["id"] for case in cases]}
    bench.dump(report / "run-status.json", status)
    try:
        for case in cases:
            run_case(case, sources, backends, config, args, work, report, run_id)
            status["completed_cases"].append(case["id"])
            bench.dump(report / "run-status.json", status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(report / "run-status.json", status)
        raise
    status["status"] = "complete"
    bench.dump(report / "run-status.json", status)


if __name__ == "__main__":
    main()
