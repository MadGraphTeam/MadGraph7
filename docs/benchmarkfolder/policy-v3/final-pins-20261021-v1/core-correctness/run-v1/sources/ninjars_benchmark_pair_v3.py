#!/usr/bin/env python3
"""Explicit original-Ninja/final-NinjaRS timing pair; no Cartesian targets.

The execution body is workers.run_case with exactly two explicit targets, no
fresh-parity subprocess, an append-only dispatch ledger and pre-parse raw-file
retention. Worker, prepare, environment, parse_outputs, configure, numerical
parity and the timed Fortran payload remain unchanged. Qualification of those
primitives/runtimes is bound separately, never inferred from this controller.
"""
import argparse
from contextlib import contextmanager
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as references
import ninjars_benchmark_workers as workers
import ninjars_supplement_v2 as supplement

ROOT = bench.ROOT
SCHEMA = "madloop-explicit-timing-pair-spec-v3"
PAIR_SCHEMA = "madloop-explicit-timing-pair-v3"
TARGETS = (("baseline", 6), ("candidate", 8))
ROLES = {"baseline": "original", "candidate": "candidate_corrected"}
CONFIG_SHA256 = "f7fdf15ae6e0b20852144ce484a02ddc98ce13ff84bddb57154e77e1e7fbbbd3"
GENERATOR_ID = "34aed45638a2e18b18135179e749317c3cf9a0fdd3bb404e43fe68d23837c6de"
CANDIDATE_REVISION = "b0ea105a04afa34304d437d2956e4d4672360670"
NUMERICAL_REVISION = "d05b59f3277ce0d727cb2816d8b7ad45107e58c4"
RUNTIME_ID = "8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff"
ARCHIVE_SHA256 = "800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS")


def read(path):
    return json.loads(Path(path).read_text())


def candidate_cohort(spec):
    """Use an explicit release freeze; keep historical specs bound to d05/800.

    A new dependency release must name all five identities together.  Omitting
    one cannot inherit an old archive or generator by accident.  The ordinary
    preflight then checks these identities against committed source bytes,
    actual CLI records, and the complete installed runtime descriptor.
    """
    if "candidate_cohort" not in spec:
        return {"numerical_revision": NUMERICAL_REVISION,
                "generation_revision": CANDIDATE_REVISION,
                "generator_identity_sha256": GENERATOR_ID,
                "runtime_identity_sha256": RUNTIME_ID,
                "archive_sha256": ARCHIVE_SHA256}
    value = spec["candidate_cohort"]
    expected = {"numerical_revision": 40, "generation_revision": 40,
                "generator_identity_sha256": 64,
                "runtime_identity_sha256": 64, "archive_sha256": 64}
    if (not isinstance(value, dict) or set(value) != set(expected) or
            any(not isinstance(value[key], str) or
                re.fullmatch("[0-9a-f]{%d}" % length, value[key]) is None
                for key, length in expected.items())):
        raise ValueError("Explicit candidate cohort requires complete commit/generator/runtime/archive identities")
    return dict(value)


def validate_spec(spec):
    policy.verified(spec, SCHEMA)
    cohort = candidate_cohort(spec)
    if (spec["pair_targets"] != [list(t) for t in TARGETS] or
            spec["role_map"] != ROLES or spec["cpu"] != 207 or
            type(spec["repetitions"]) is not int or spec["repetitions"] < 7):
        raise ValueError("Timing requires exactly original6/candidate8 on CPU207, at least seven pairs")
    if (not math.isfinite(spec["minimum_seconds"]) or spec["minimum_seconds"] < 1. or
            not math.isfinite(spec["timeout"]) or not 0 < spec["timeout"] <= 86400.):
        raise ValueError("Invalid full-traversal calibration or command timeout")
    signature = policy.verified(spec["candidate_runtime_signature"], "reducer-runtime-signature-v2")
    if (spec["literal_baseline_revision"] != bench.BASELINE_REVISION or
            spec["candidate_generator_identity_sha256"] != cohort["generator_identity_sha256"] or
            signature["identity_sha256"] != cohort["runtime_identity_sha256"] or
            cohort["archive_sha256"] not in signature["libraries"].values()):
        raise ValueError("Timing spec differs from the final literal-original/NinjaRS cohort")
    expected_sources = {"collector", *workers.SOURCE_NAMES}
    if set(spec["sources"]) != expected_sources or not spec["production_sources"]:
        raise ValueError("Missing actual controller/worker/production source inventory")
    if spec["sources"]["collector"]["sha256"] != workers.digest(Path(__file__)):
        raise ValueError("Spec does not bind the actual pair controller")
    if any(spec["sources"][name].get("sha256") != workers.digest(ROOT / "tools" / name)
           for name in workers.SOURCE_NAMES):
        raise ValueError("Spec does not bind the unchanged worker payload")
    return spec


def check_production(spec):
    """Freeze numerical bytes, allowing unrelated orchestration commits."""
    paths = (*bench.GENERATOR_PATHS, "vendor")
    names = subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=ROOT).decode().split("\0")
    actual = {name: workers.digest(ROOT / name) if (ROOT / name).is_file() else None
              for name in names if name}
    if actual != spec["production_sources"]:
        raise ValueError("Tracked generator/vendor numerical source changed")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--exit-code", candidate_cohort(spec)["numerical_revision"], "HEAD", "--", *paths], cwd=ROOT, check=True)
    additions = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=ROOT).decode().splitlines()
    if set(additions) - {"vendor/StdHEP/fail"}:
        raise ValueError("Untracked generator/numerical source is present")
    for record in spec["sources"].values():
        supplement.checked(record)


def actual_runtime(output, manifest, backend):
    return policy.reducer_signature({"backend": backend, "compiled_manifest": manifest,
        "libraries": policy.library_hashes(output), "runtime_metadata": {
            name: supplement.bound(path) for name, path in policy.runtime_metadata_files(output).items()}})


def validate_origin_records(source, output, generation, compilation, manifest):
    """Keep genuine old original metadata; never synthesize modern fields."""
    actual = bench.generated_source_manifest(output)
    if (read(output / "generation.json") != generation or
            read(output / "compilation.json") != compilation or
            actual["files"] != manifest["files"] or actual["files_sha256"] != manifest["files_sha256"] or
            bench.content_digest(manifest["files"]) != manifest["files_sha256"]):
        raise ValueError("Actual timing origin differs from its recorded generation/compilation/source manifest")
    identity = generation["source_identity"]
    if source == "candidate":
        if (compilation.get("generated_sources_sha256") != manifest["files_sha256"] or
                compilation.get("source_identity") != identity):
            raise ValueError("Final candidate requires modern source/compilation identity fields")
    elif source == "baseline":
        policy.original_identity(identity)
        # Optional modern fields, if present, may not contradict the old record.
        if (compilation.get("generated_sources_sha256", manifest["files_sha256"]) != manifest["files_sha256"] or
                compilation.get("source_identity", identity) != identity):
            raise ValueError("Literal original compilation contradicts its observed source")
    else:
        raise ValueError("Unknown timing origin role")
    directories = list((output / "SubProcesses").glob("P*_*"))
    if len(directories) != 1 or any(workers.digest(directories[0] / name) != compilation[key]
            for name, key in (("check", "executable_sha256"), ("check_sa.f", "driver_sha256"))):
        raise ValueError("Actual original/candidate executable or driver differs from its untouched compilation record")
    return identity


def preflight(spec, config, correction):
    check_production(spec)
    cohort = candidate_cohort(spec)
    if workers.digest(supplement.checked(spec["configuration"])) != CONFIG_SHA256:
        raise ValueError("The reviewed eight-case configuration changed")
    policy.validate_config(config)
    cases = {case["id"] for case in config["cases"]}
    if set(spec["points"]) != cases or set(spec["origins"]) != cases or set(spec["output_roots"]) != set(ROLES):
        raise ValueError("Missing/injected physical case or source")
    supplement.checked(spec["runtime_qualification"])
    for case in config["cases"]:
        cid = case["id"]
        points = supplement.checked(spec["points"][cid])
        if (int(points.read_text().splitlines()[0]) != 100 or
                points.read_bytes() != (ROOT / "docs/benchmarkfolder/points" / (cid + ".dat")).read_bytes()):
            raise ValueError("Exactly 100 fixed points are required")
        if set(spec["origins"][cid]) != set(ROLES):
            raise ValueError("A case omits/injects a timing target")
        cards = []
        for source, backend in TARGETS:
            anchor = spec["origins"][cid][source]
            output = policy.path(spec["output_roots"][source]) / cid
            generation = read(supplement.checked(anchor["generation"]))
            compilation = read(supplement.checked(anchor["compilation"]))
            manifest = read(supplement.checked(anchor["compiled_manifest"]))
            card = supplement.checked(anchor["physical_card"])
            identity = validate_origin_records(source, output, generation, compilation, manifest)
            if (output / "Cards/param_card.dat").read_bytes() != card.read_bytes():
                raise ValueError("Actual timing input differs from its source/card anchor")
            policy.validate_source_role(ROLES[source], cid, manifest, identity, correction)
            if source == "candidate" and (identity["revision"] != cohort["generation_revision"] or
                    identity["tree_state_sha256"] != cohort["generator_identity_sha256"] or
                    bench.content_digest(identity["tree_state"]) != cohort["generator_identity_sha256"] or
                    identity.get("dirty_patch_sha256") is not None or identity["tree_state"]["untracked"]):
                raise ValueError("Candidate CLI source differs from final numerical cohort")
            runtime = actual_runtime(output, manifest, backend)
            if runtime != anchor["reducer_runtime"] or (source == "candidate" and runtime != spec["candidate_runtime_signature"]):
                raise ValueError("Actual timing reduction runtime differs from the sealed full descriptor")
            cards.append(card.read_bytes())
        if cards[0] != cards[1]:
            raise ValueError("Original/candidate physical parameter cards differ")


class Dispatch:
    def __init__(self, report):
        self.path, self.index = report / "dispatch.jsonl", 0
        self.path.touch(exist_ok=False)

    def event(self, case, phase, event, key, repetition=None, **fields):
        value = {"index": self.index, "case": case["id"], "phase": phase,
                 "event": event, "source": key[0], "backend": key[1],
                 "repetition": repetition, "loops": None, "worker_sequence": None,
                 "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
        with self.path.open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        self.index += 1
        return value["index"]


def retain_raw(report, case, key, phase, repetition, directory):
    """Snapshot before parsing/acceptance; malformed/failed outputs survive."""
    destination = report / "raw" / f"{case['id']}-{key[0]}-{key[1]}" / f"{phase}-{repetition}"
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in (*workers.OUTPUTS, "dimensions.dat"):
        path = directory / name
        if path.exists():
            shutil.copy2(path, destination / name)
            files[name] = supplement.bound(destination / name)
    return files


@contextmanager
def defer_interruptions():
    """Finish a short ownership transfer/cleanup before delivering a signal."""
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous, pending = {sig: signal.getsignal(sig) for sig in signals}, []
    def defer(sig, _frame):
        pending.append(sig)
    try:
        for sig in signals:
            signal.signal(sig, defer)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if pending:
            handler = previous[pending[0]]
            if callable(handler):
                handler(pending[0], None)
            elif handler != signal.SIG_IGN:
                raise SystemExit("Deferred collector interruption " + str(pending[0]))


def process_event(report, value):
    with (report / "process-ledger.jsonl").open("a") as ledger:
        ledger.write(json.dumps(value) + "\n")


@contextmanager
def owned_preparation(report, label="pair", fresh_cpu=None):
    """Own prepare and Worker Popen children without changing either primitive.

    The separate registry closes both Popen->self.process and constructor->local
    assignment windows. Signals stay enabled during the potentially long READY
    wait; only creation/registration and final cleanup defer interruptions.
    """
    original_run, original_subprocess = bench.run, workers.subprocess
    serial, children = [0], []

    def reap(process):
        identity = next(item for child, item in children if child is process)
        if identity.get("reaped"):
            return

        def current_ticks():
            try:
                return int((Path("/proc") / str(process.pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
            except FileNotFoundError:
                return None

        def exited_group():
            current = current_ticks()
            if current is not None and (identity["start_ticks"] is None or current != identity["start_ticks"]):
                # Original exit status is already reaped. A replacement is never
                # part of our ownership; a surviving old PGID would reserve its ID.
                identity["replacement_ignored"] = True
                return
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            raise RuntimeError("Owned group survives its reaped leader; refusing an unverified group signal")

        def signal_live(sig):
            if identity["start_ticks"] is None or current_ticks() != identity["start_ticks"]:
                raise RuntimeError("Owned child identity unavailable or reused; refusing to signal")
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        if process.poll() is None:
            signal_live(signal.SIGTERM)
            try:
                process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                # No signal is sent after wait has reaped the original leader.
                if process.poll() is None:
                    signal_live(signal.SIGKILL)
                    process.wait()
        exited_group()
        identity["reaped"] = True

    def spawn(command, **kwargs):
        if kwargs.get("start_new_session") is not True:
            raise ValueError("Every owned child requires an isolated process group")
        with defer_interruptions():
            process = original_subprocess.Popen(command, **kwargs)
            # Register before any fallible identity/ledger operation.
            identity = {"pid": process.pid, "pgid": process.pid, "start_ticks": None}
            children.append((process, identity))
            try:
                fields = (Path("/proc") / str(process.pid) / "stat").read_text().rsplit(")", 1)[1].split()
                identity["start_ticks"] = int(fields[19])
                identity["pgid"] = int(fields[2])
            except FileNotFoundError:
                identity["exited_before_identity"] = process.poll()
            process_event(report, {"event": "spawn", **identity, "command": command,
                                   "cwd": str(kwargs.get("cwd", ROOT))})
        return process

    def run(command, cwd, log, **kwargs):
        timeout = kwargs.pop("timeout", 21600.)
        if kwargs or not 0 < timeout <= 21600.:
            raise ValueError("Unexpected preparation subprocess override")
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with log.open("x") as stream:
            process = None
            try:
                process = spawn(command, cwd=cwd, env=workers.environment(bench.WORK),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code = process.wait(timeout=timeout)
            finally:
                if process is not None:
                    with defer_interruptions():
                        reap(process)
        if code:
            raise RuntimeError(f"{command[0]} failed ({code}); see {log}")
        return time.perf_counter() - start

    class CopyProxy:
        def __getattr__(self, name):
            return getattr(original_subprocess, name)

        Popen = staticmethod(spawn)

        def run(self, command, **kwargs):
            if fresh_cpu is not None and command[:4] == ["taskset", "-c", str(fresh_cpu), "./check_fresh"]:
                required = {"cwd", "env", "stdout", "stderr", "timeout", "check"}
                if (len(command) != 6 or command[-1] != "1" or set(kwargs) != required or
                        kwargs["check"] is not True or kwargs["stderr"] != subprocess.STDOUT or
                        not 0 < kwargs["timeout"] <= 86400. or
                        not Path(kwargs["cwd"]).resolve().is_relative_to(ROOT / ".benchmark-work") or
                        any(kwargs["env"].get(name) != "1" for name in THREADS)):
                    raise ValueError("Unexpected representative fresh-process override")
                options = dict(kwargs); timeout = options.pop("timeout"); options.pop("check")
                process = None
                try:
                    process = spawn(command, **options, start_new_session=True)
                    code = process.wait(timeout=timeout)
                finally:
                    if process is not None:
                        with defer_interruptions():
                            reap(process)
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return subprocess.CompletedProcess(command, code)
            if command[:3] != ["cp", "-a", "--reflink=auto"] or kwargs != {"check": True}:
                raise ValueError("Unexpected frozen prepare subprocess")
            serial[0] += 1
            run(command, ROOT, report / "logs" / f"copy-{label}-{serial[0]}.log")
            return subprocess.CompletedProcess(command, 0)

    bench.run, workers.subprocess = run, CopyProxy()
    try:
        yield
    finally:
        errors = []
        try:
            with defer_interruptions():
                for process, identity in reversed(children):
                    try:
                        reap(process)
                    except BaseException as error:
                        errors.append(str(error))
                    try:
                        process_event(report, {"event": "reaped", **identity, "returncode": process.poll()})
                    except BaseException as error:
                        errors.append(str(error))
        finally:
            bench.run, workers.subprocess = original_run, original_subprocess
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError("Owned process cleanup failed: " + "; ".join(errors))


def run_pair_case(case, config, args, work, report, run_id, dispatch):
    """Narrow copy of frozen run_case; all timed primitives stay unchanged."""
    protocol = read(report / "protocol.json")
    if protocol["pair_targets"] != [list(t) for t in TARGETS]:
        raise ValueError("Injected timing target")
    points = report / "points" / (case["id"] + ".dat")
    point_count = int(points.read_text().splitlines()[0])
    if point_count != config["points_per_case"]:
        raise ValueError("Unexpected retained point count")
    jobs, instances, calibrations, loops = {}, {}, {}, {}
    active, stage, repetition, raw = None, "prepare", None, {}
    try:
        for key in TARGETS:
            active = key
            raw = {}
            dispatch.event(case, "prepare", "start", key)
            jobs[key] = workers.prepare(key[0], case, key[1], config, work, report, clean=True)
            dispatch.event(case, "prepare", "complete", key)
        for key in TARGETS:
            source, backend = key
            active, stage, repetition = key, "initialize", None
            raw = {}
            job, directory, metadata = jobs[key]
            base = {**metadata, "cpu": args.cpu, "case": case["id"],
                    "points_sha256": workers.digest(points), "measurement_run_id": run_id,
                    "paired_baseline_candidate": True, "worker_scheme": workers.SCHEME,
                    "acceptance_eligible": False, "benchmark_protocol": workers.SCHEME,
                    "protocol_identity_sha256": protocol["protocol_identity_sha256"],
                    "captured_during_timed_final_traversal": True}
            with workers.isolated_harness(job, report):
                bench.configure(source, case, backend, config=config)
            workers.validate_param_card(directory, (job / "outputs" / source / case["id"] / "Cards/param_card.dat").read_bytes())
            dispatch.event(case, stage, "start", key)
            worker = workers.Worker(["taskset", "-c", str(args.cpu), "./check", str(points)],
                directory, report / "logs" / f"worker-{case['id']}-{source}-{backend}.log",
                env=workers.environment(job), timeout=args.timeout)
            instances[key] = worker
            if worker.points != point_count:
                raise workers.ProtocolError("Worker initialized with a different point count")
            dispatch.event(case, stage, "complete", key, worker_sequence=worker.sequence)
            stage, repetition = "calibration", "calibration"
            index = dispatch.event(case, stage, "start", key, repetition, loops=1, worker_sequence=worker.sequence + 1)
            try:
                wall = worker.run(1)
            finally:
                raw = retain_raw(report, case, key, stage, repetition, directory)
            result = {**base, **workers.parse_outputs(directory, backend, point_count, 1),
                      "repetition": repetition, "loops": 1, "command_wall_seconds": wall,
                      "worker_sequence": worker.sequence, "dispatch_index": index, "raw_capture": raw}
            if result["answer_dimensions"] != {"allocated": worker.ndim, "loop_orders": worker.nso, "born_orders": worker.nborn}:
                raise workers.ProtocolError("READY dimensions differ from numerical output")
            supplement.comparable_numerical_settings(result)
            filename = report / "data" / f"{case['id']}-{source}-{backend}-calibration.json"
            bench.dump(filename, result)
            dispatch.event(case, stage, "complete", key, repetition, loops=1,
                           worker_sequence=worker.sequence, record=supplement.bound(filename))
            calibrations[key] = result
            loops[key] = max(1, math.ceil(args.minimum_seconds / result["seconds"]))
        for repetition in range(args.repetitions):
            order = TARGETS if repetition % 2 == 0 else tuple(reversed(TARGETS))
            for key in order:
                active, stage = key, "measurement"
                source, backend = key
                worker = instances[key]
                index = dispatch.event(case, stage, "start", key, repetition,
                    loops=loops[key], worker_sequence=worker.sequence + 1)
                try:
                    wall = worker.run(loops[key])
                finally:
                    raw = retain_raw(report, case, key, stage, repetition, worker.directory)
                result = {**calibrations[key], **workers.parse_outputs(worker.directory, backend, point_count, loops[key]),
                          "repetition": repetition, "loops": loops[key], "command_wall_seconds": wall,
                          "worker_sequence": worker.sequence, "dispatch_index": index, "raw_capture": raw,
                          "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                bench.validate_repetition(result, calibrations[key], config)
                filename = report / "data" / f"{case['id']}-{source}-{backend}-{repetition}.json"
                bench.dump(filename, result)
                dispatch.event(case, stage, "complete", key, repetition, loops=loops[key],
                               worker_sequence=worker.sequence, record=supplement.bound(filename))
    except BaseException as error:
        directory = jobs[active][1] if active in jobs else None
        worker = instances.get(active)
        failure = {"case": case["id"], "source": active[0] if active else None,
                   "backend": active[1] if active else None, "stage": stage, "repetition": repetition,
                   "error_type": type(error).__name__, "error": str(error),
                   "command_sequence": worker.sequence if worker else None,
                   "acceptance_eligible": False, "run_status": "failed", "raw_capture": raw,
                   "diagnostic_log": str(worker.log) if worker else None,
                   "raw_outputs": {name: (directory / name).read_text(errors="replace")
                       for name in (*workers.OUTPUTS, "dimensions.dat")
                       if directory is not None and (directory / name).exists()}}
        bench.dump(report / "rejected" / f"{case['id']}-failure.json", failure)
        dispatch.event(case, stage, "failed", active, repetition, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        cleanup_errors = []
        with defer_interruptions():
            for key, worker in reversed(list(instances.items())):
                try:
                    dispatch.event(case, "cleanup", "start", key)
                except BaseException as error:
                    cleanup_errors.append(str(error))
                try:
                    worker.close()
                    dispatch.event(case, "cleanup", "complete", key)
                except BaseException as error:
                    cleanup_errors.append(str(error))
                    try:
                        dispatch.event(case, "cleanup", "failed", key, error=str(error))
                    except BaseException as ledger_error:
                        cleanup_errors.append(str(ledger_error))
        if cleanup_errors:
            bench.dump(report / "rejected" / f"{case['id']}-cleanup-failure.json",
                       {"case": case["id"], "errors": cleanup_errors, "acceptance_eligible": False})
            if sys.exc_info()[0] is None:
                raise RuntimeError("Worker cleanup failed: " + "; ".join(cleanup_errors))


def bind_pair(report, work, spec, correction):
    from ninjars_benchmark_policy_v3 import checked_pair_series
    checked_pair_series(report)
    policy.require_no_rejections(report)
    targets, assets = {}, report / "runtime-assets"
    assets.mkdir()
    for case in read(report / "protocol.json")["configuration"]["cases"]:
        for source, backend in TARGETS:
            key = f"{case['id']}-{source}-{backend}"
            output = work / "workers" / key / "outputs" / source / case["id"]
            target = policy.bind_target(ROLES[source], case["id"], backend, output,
                report / "builds" / key / "build.json", correction)
            metadata = {}
            for name, path in policy.runtime_metadata_files(output).items():
                destination = assets / workers.digest(path)
                if not destination.exists():
                    shutil.copy2(path, destination)
                metadata[name] = supplement.bound(destination)
            target["runtime_metadata"] = metadata
            target["reducer_runtime"] = policy.reducer_signature(target)
            if target["reducer_runtime"] != spec["origins"][case["id"]][source]["reducer_runtime"]:
                raise ValueError("Built pair worker changes the sealed runtime")
            targets[key] = target
    inventory = {policy.relative(path): workers.digest(path) for path in report.rglob("*") if path.is_file()}
    result = policy.seal({"schema": "source-role-bindings-v2", "report": policy.relative(report),
        "run_files_sha256": inventory, "correction_identity_sha256": correction["identity_sha256"], "targets": targets})
    bench.dump(report / "role-bindings.json", result)
    return result


def collect(specification, handoff, execute_numerical=False):
    if not execute_numerical:
        raise ValueError("Pair collection requires explicit --execute-numerical and a root handoff")
    specification = policy.path(specification)
    spec = validate_spec(read(specification))
    authorization = read(handoff)
    if authorization != {"root_authorized": True, "phase": PAIR_SCHEMA, "cpu": 207,
                         "specification_sha256": workers.digest(specification)}:
        raise ValueError("Missing/mismatched exclusive timing handoff")
    if 207 not in os.sched_getaffinity(0):
        raise ValueError("CPU207 is unavailable")
    thread_environment = {name: os.environ.get(name) for name in THREADS}
    if any(value != "1" for value in thread_environment.values()):
        raise ValueError("OMP/OPENBLAS/RAYON/MKL thread counts must all be exactly 1")
    config = read(supplement.checked(spec["configuration"]))
    correction = policy.validate_correction(supplement.checked(spec["correction"]), live=True)
    preflight(spec, config, correction)
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    if report.exists() or work.exists():
        raise ValueError("Retain prior attempt; choose fresh report/work namespaces")
    report.mkdir(parents=True); work.mkdir(parents=True)
    for name in ("sources", "points"):
        (report / name).mkdir()
    for name, record in spec["sources"].items():
        shutil.copy2(supplement.checked(record), report / "sources" / (Path(__file__).name if name == "collector" else name))
    shutil.copy2(specification, report / "input-spec.json")
    shutil.copy2(handoff, report / "root-handoff.json")
    for case in config["cases"]:
        shutil.copy2(supplement.checked(spec["points"][case["id"]]), report / "points" / (case["id"] + ".dat"))
    descriptor = workers.protocol_descriptor()
    protocol = {"schema": workers.SCHEME, "pair_schema": PAIR_SCHEMA,
        "pair_targets": [list(t) for t in TARGETS], "configuration": config,
        "protocol_descriptor": descriptor, "protocol_identity_sha256": bench.content_digest(descriptor),
        "cpu": 207, "repetitions": spec["repetitions"], "sources": "both", "backends": [6, 8],
        "role_map": ROLES, "fresh_parity": False, "acceptance_eligible": False,
        "thread_environment": thread_environment,
        "numerical_revision": candidate_cohort(spec)["numerical_revision"],
        "candidate_cohort": candidate_cohort(spec),
        "input_adapter": supplement.bound(report / "sources" / Path(__file__).name),
        "input_spec": supplement.bound(report / "input-spec.json"),
        "orchestration_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dispatch_ledger": policy.relative(report / "dispatch.jsonl")}
    bench.dump(report / "protocol.json", protocol)
    status = {"status": "incomplete", "requested_cases": [c["id"] for c in config["cases"]], "completed_cases": []}
    bench.dump(report / "run-status.json", status)
    dispatch = Dispatch(report)
    args = SimpleNamespace(cpu=207, repetitions=spec["repetitions"], minimum_seconds=spec["minimum_seconds"], timeout=spec["timeout"])
    try:
        with workers.isolated_harness(work, report), references.input_roots(spec["output_roots"], work):
            for case in config["cases"]:
                with owned_preparation(report, case["id"]):
                    run_pair_case(case, config, args, work / "workers", report, spec["run_id"], dispatch)
                check_production(spec)
                status["completed_cases"].append(case["id"])
                bench.dump(report / "run-status.json", status)
        protocol["dispatch_ledger"] = supplement.bound(dispatch.path)
        bench.dump(report / "protocol.json", protocol)
        status["status"] = "complete"
        bench.dump(report / "run-status.json", status)
        return bind_pair(report, work, spec, correction)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(report / "run-status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--execute-numerical", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args.spec, args.handoff, args.execute_numerical), indent=2))


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise SystemExit("Collector interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    main()
