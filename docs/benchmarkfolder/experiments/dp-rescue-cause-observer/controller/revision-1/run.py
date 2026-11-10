#!/usr/bin/env python3
"""Separate preparation, build dispatch, and one six-arm private replay.

Source proposal only. Explicit invocation is required for every stage; preparing
or building never launches an amplitude. Root clearance is a reviewed message,
not an approval file synthesized by this program.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = PROPOSAL.parents[3]
PROPOSAL_SHA = "b0b0af81739be886420450459eeed94a839c71e8aeef22b17c79d3b510c2cd1b"
SNAPSHOT_SHA = "2aa5e87ead00c163d0f987dbef257654a8a22ec748f7d3f71be68aeaa64e7759"
ANALYZER_SHA = "473475dfdde70df2f721ca782f3ac1719ab08b92b72a9eadd4e8bbe216aed10b"
ORIGINAL_DSO_SHA = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
SUPERVISOR = ROOT / "docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py"
SUPERVISOR_SHA = "ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42"
LINEAGE_HELPER = ROOT / "docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-qualification-2/process_cleanup.py"
LINEAGE_SHA = "240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84"
CASES = {"qcd_n3_uux_zgg": "P0_uux_zgg", "ew_n3_epem_ttxa": "P0_epem_ttxa"}
ARMS = ("original", "disabled", "enabled")
CPU = 11
ARM_SECONDS = 600
TOTAL_SECONDS = 3900
MAX_STDOUT = 64 * 1024 * 1024
MAX_NATIVE = 512 * 1024 * 1024
MAX_PUBLIC = 2 * 1024 * 1024
SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
DIGEST = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Deadline:
    def __init__(self, seconds, outer=None):
        self.end = time.monotonic() + seconds
        if outer is not None:
            self.end = min(self.end, outer.end)

    def check(self):
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Private diagnostic deadline exhausted; no retry")
        return remaining


def within(path, root):
    return path == root or root in path.parents


def controlled(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Unsafe recorded absolute path")
    resolved = path.resolve(strict=True)
    allowed = (ROOT, ROOT.parent / "references", Path("/nix/store"))
    require(any(within(path, root) for root in allowed)
            and any(within(resolved, root) for root in allowed), "Input escaped declared roots")
    require(resolved.is_file(), "Pinned input is not a regular file: " + str(path))
    return resolved


def relative(base, name):
    part = PurePosixPath(name)
    require(isinstance(name, str) and not part.is_absolute() and name == part.as_posix()
            and part.parts and ".." not in part.parts, "Unsafe relative artifact path")
    return base.joinpath(*part.parts)


def sha(path, deadline=None):
    digest = hashlib.sha256()
    with controlled(path).open("rb") as stream:
        while True:
            if deadline is not None:
                deadline.check()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    if deadline is not None:
        deadline.check()
    return digest.hexdigest()


def read(path):
    return json.loads(controlled(path).read_text())


def pin(path, deadline=None):
    path = controlled(path)
    return {"path": str(path), "sha256": sha(path, deadline)}


def check_pin(value, deadline=None):
    require(isinstance(value, dict) and isinstance(value.get("path"), str)
            and isinstance(value.get("sha256"), str) and DIGEST.fullmatch(value["sha256"]), "Invalid file pin")
    require(sha(Path(value["path"]), deadline) == value["sha256"], "Pinned bytes changed: " + value["path"])


def collect_pins(value, output=None):
    output = {} if output is None else output
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            path, digest = value["path"], value["sha256"]
            require(isinstance(path, str) and isinstance(digest, str) and DIGEST.fullmatch(digest), "Malformed pin")
            require(path not in output or output[path] == digest, "Conflicting pin")
            output[path] = digest
        for name, item in value.items():
            if isinstance(name, str) and name.startswith("/") and isinstance(item, str) and DIGEST.fullmatch(item):
                require(name not in output or output[name] == item, "Conflicting absolute pin")
                output[name] = item
            elif isinstance(item, (dict, list)):
                collect_pins(item, output)
    elif isinstance(value, list):
        for item in value:
            collect_pins(item, output)
    return output


def verify_pins(pins, deadline=None):
    for name, digest in sorted(pins.items()):
        require(sha(Path(name), deadline) == digest, "Protected input changed: " + name)


def write(path, value):
    require(not path.is_symlink() and path.parent.resolve() == path.parent, "Unsafe private output path")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), "Stale metadata temporary file")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def record_failure(path, record, error):
    """Metadata failure must not replace the numerical/ownership exception."""
    record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["complete"] = False
    try:
        write(path, record)
    except BaseException as metadata_error:
        if hasattr(error, "add_note"):
            error.add_note("Failure metadata could not be saved: " + repr(metadata_error))


def proposal_inputs():
    require(Path.cwd().resolve() == ROOT, "Run only from the MadGraph repository root")
    require(sha(PROPOSAL / "index.json") == PROPOSAL_SHA, "Wrong frozen 44-file proposal")
    index = read(PROPOSAL / "index.json")
    require(len(index["files"]) == 43, "Expected 43 frozen payloads plus their index")
    pins = {str(PROPOSAL / "index.json"): PROPOSAL_SHA}
    for name, entry in index["files"].items():
        path = relative(PROPOSAL, name)
        require(not path.is_symlink() and path.stat().st_size == entry["bytes"], "Proposal file identity changed")
        pins[str(path)] = entry["sha256"]
    require(pins[str(PROPOSAL / "analyze.py")] == ANALYZER_SHA, "Analyzer identity changed")
    snapshot_path = PROPOSAL / "preparation/source-1/snapshot.json"
    require(pins[str(snapshot_path)] == SNAPSHOT_SHA, "Snapshot identity changed")
    verify_pins(pins)
    snapshot = read(snapshot_path)
    collect_pins(snapshot, pins)
    pins[str(SUPERVISOR)] = SUPERVISOR_SHA
    pins[str(LINEAGE_HELPER)] = LINEAGE_SHA
    pins[str(Path(__file__).resolve())] = sha(Path(__file__))
    verify_pins(pins)
    return snapshot, pins


def verify_build(path, expected_sha, snapshot, deadline=None):
    require(DIGEST.fullmatch(expected_sha) and sha(path, deadline) == expected_sha, "Build manifest hash mismatch")
    build = read(path)
    require(build.get("schema") == "dp-rescue-observer-build-v1" and build.get("status") == "complete",
            "Build is absent, incomplete, or from another protocol")
    require(build["source_index_sha256"] == PROPOSAL_SHA, "Build does not bind frozen proposal")
    require(build["build_plan_sha256"] == sha(HERE / "build-plan.json", deadline)
            and build["builder_sha256"] == sha(HERE / "build.py", deadline)
            and build["native_inputs_sha256"] == sha(HERE / "native-inputs.json", deadline),
            "Build controller inputs changed")
    require(build["protected_before"] == build["protected_after"], "Build changed protected inputs")
    require(build.get("commands") and build.get("artifacts"), "Build lacks actual command/artifact provenance")
    require(set(build["cases"]) == set(CASES), "Build case matrix differs")
    check_pin(build["original_dso"], deadline)
    check_pin(build["observed_dso"], deadline)
    require(build["original_dso"]["sha256"] == ORIGINAL_DSO_SHA, "Original runtime is not installed eb3d")
    require(build["observed_dso"]["path"] != build["original_dso"]["path"], "Private DSO aliases original")
    pins = collect_pins(build)
    for name in ("build.py", "build-plan.json", "native-inputs.json"):
        pins[str(HERE / name)] = sha(HERE / name, deadline)
    pins[str(path.resolve())] = expected_sha
    verify_pins(pins, deadline)
    for case_id, subprocess_name in CASES.items():
        old = snapshot["cases"][case_id]
        arms = build["cases"][case_id]["arms"]
        require(set(arms) == {"original", "observed"}, "Expected exactly two compiled binaries per case")
        for name, arm in arms.items():
            check_pin(arm["executable"], deadline)
            check_pin(arm["interpreter"], deadline)
            require(arm["launch_prefix"] == [arm["interpreter"]["path"]], "Unexpected loader command")
            expected_dso = build["original_dso"] if name == "original" else build["observed_dso"]
            require(isinstance(arm["runtime"], dict) and arm["runtime"], "Missing runtime closure")
            bridges = [p for p in arm["runtime"] if Path(p).name == "libmadloop_tirust.so"]
            require(bridges == [expected_dso["path"]] and arm["runtime"][bridges[0]] == expected_dso["sha256"],
                    "Runtime includes wrong or multiple numerical DSOs")
            loaders = [p for p in arm["runtime"] if Path(p).name == "ld-linux-x86-64.so.2"]
            require(loaders == [arm["interpreter"]["path"]], "Runtime must contain the actual interpreter only")
            changed = {f"SubProcesses/{subprocess_name}/check_sa.o"}
            if name == "observed":
                changed.update(f"SubProcesses/{subprocess_name}/{file}.o" for file in
                               ("loop_matrix", "TIR_interface", "TIRUST_interface"))
            require(set(old["existing_numerical_objects"]) <= set(arm["objects"]), "Build omits original object inventory")
            for relative_name, original in old["existing_numerical_objects"].items():
                current = arm["objects"][relative_name]
                check_pin(current, deadline)
                if relative_name not in changed:
                    require(current["sha256"] == original["sha256"], "Unexpected numerical object change: " + relative_name)
            for relative_name, original in old["all_sources_and_includes"].items():
                current = arm["sources"][relative_name]
                check_pin(current, deadline)
                source_changed = relative_name.removesuffix(".f") + ".o" in changed
                expected = original["sha256"]
                if source_changed:
                    expected = sha(PROPOSAL / "prepared" / case_id / Path(relative_name).name, deadline)
                require(current["sha256"] == expected, "Unexpected host source change: " + relative_name)
            require(set(arm["objects"]) == set(old["existing_numerical_objects"]), "Unexpected numerical object inventory")
            extras = {"host_observer.o", "observer_noop.o"} if name == "original" else {"host_observer.o"}
            require(set(arm["extra_objects"]) == extras, "Unexpected observer support object inventory")
            source_extras = {"host_observer.f90", "observer_noop.c"} if name == "original" else {"host_observer.f90"}
            require(set(arm["extra_sources"]) == source_extras, "Unexpected observer support source inventory")
            for source_name, value in arm["extra_sources"].items():
                require(value["sha256"] == sha(PROPOSAL / source_name, deadline), "Observer support source changed")
        require(arms["original"]["extra_objects"]["host_observer.o"]["sha256"]
                == arms["observed"]["extra_objects"]["host_observer.o"]["sha256"], "Common observer module objects differ")
    return build, pins


@contextmanager
def deferred_signals():
    """Adapted from the pinned reviewed supervisor; no byte-identity claim."""
    previous, pending = {}, []
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    try:
        try:
            for signum in SIGNALS:
                previous[signum] = signal.signal(signum, lambda number, frame: pending.append(number))
        except BaseException:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    try:
        yield pending
    finally:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        try:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)


class OwnedChild:
    """One session leader, reserved by WNOWAIT until final group cleanup."""
    def __init__(self, deadline, progress):
        self.deadline, self.progress = deadline, progress
        self.process = None
        self.owned = False

    def spawn(self, command, cwd, env, log):
        require(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, "Default SIGCHLD disposition required")
        self.deadline.check()
        with deferred_signals() as pending:
            self.process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.owned = True
            self.progress.update(process_spawned=True, pid=self.process.pid, process_reaped=False)
        if pending:
            raise InterruptedError("Interrupted during child creation: " + repr(pending))
        self.deadline.check()
        os.set_blocking(self.process.stdin.fileno(), False)

    def exited(self):
        self.deadline.check()
        status = os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        self.deadline.check()
        if status is not None:
            require(status.si_pid == self.process.pid, "Unexpected child exit identity")
        return status is not None

    def stop_command(self):
        pending = memoryview(b"STOP 1 0\n")
        while pending:
            self.deadline.check()
            try:
                count = os.write(self.process.stdin.fileno(), pending)
                require(count > 0, "Worker command pipe closed")
                pending = pending[count:]
            except BlockingIOError:
                time.sleep(min(.01, self.deadline.check()))
        self.deadline.check()

    def finish(self):
        if not self.owned:
            return self.process.returncode if self.process is not None else None
        with deferred_signals() as pending:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # The unreaped leader still reserves the group identity. Reaping
            # after SIGKILL is mandatory cleanup, never extra numerical work.
            code = self.process.wait()
            self.owned = False
            self.progress.update(process_reaped=True, returncode=code)
        if pending:
            raise InterruptedError("Interrupted during final group cleanup: " + repr(pending))
        return code

    def cleanup_after(self, error):
        try:
            self.finish()
        except BaseException as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note("Owned child cleanup error: " + repr(cleanup_error))


def interrupted(signum, frame):
    raise InterruptedError("Controller interrupted by signal " + str(signum))


def copy_pin(value, target, deadline):
    check_pin(value, deadline)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(target.parent.resolve() == target.parent and not target.exists(), "Private copy destination already exists")
    with controlled(value["path"]).open("rb") as source, target.open("xb") as destination:
        while True:
            deadline.check()
            block = source.read(1024 * 1024)
            if not block:
                break
            destination.write(block)
    require(sha(target, deadline) == value["sha256"], "Private copied input differs")


def stage_run(case, run_dir, deadline):
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "tmp").mkdir()
    copied = {}
    sources = {"points.dat": case["points"]["dat"], "points.json": case["points"]["metadata"],
               "Cards/param_card.dat": case["cards"]["Cards/param_card.dat"],
               "Cards/ident_card.dat": case["initial_resource_inventory"]["SubProcesses/MadLoop5_resources/ident_card.dat"]}
    for name, value in case["initial_resource_inventory"].items():
        basename = Path(name).name
        if basename.endswith(("HelFilter.dat", "LoopFilter.dat")):
            continue
        sources["MadLoop5_resources/" + basename] = value
    for name, value in sources.items():
        target = relative(run_dir, name)
        copy_pin(value, target, deadline)
        copied[str(target)] = value["sha256"]
    # Matches original configure() only in this fresh private run directory.
    for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
        require(not list(run_dir.rglob(pattern)), "Fresh private run contains warm filters")
    return copied


def runtime_environment(build, arm, mode, run_dir):
    allowed = {"PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TZ", "OMP_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"}
    env = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
    for supplied in (build.get("runtime_environment", {}), arm.get("runtime_environment", {})):
        require(isinstance(supplied, dict) and set(supplied) <= allowed
                and all(isinstance(value, str) for value in supplied.values()), "Unexpected runtime environment variable")
        env.update(supplied)
    for key in allowed:
        if key.endswith("_NUM_THREADS") or key == "VECLIB_MAXIMUM_THREADS":
            require(key not in env or env[key] == "1", "Thread count override is not one")
            env[key] = "1"
    bridge = next(path for path in arm["runtime"] if Path(path).name == "libmadloop_tirust.so")
    # Use the builder's exact loader search path, also used for PT_INTERP --list.
    directories = env.get("LD_LIBRARY_PATH", "").split(":")
    require(directories[0] == str(Path(bridge).parent)
            and all(part and Path(part).is_absolute() for part in directories), "Unexpected loader search path")
    env["TMPDIR"] = str(run_dir / "tmp")
    env["DCO_OBSERVE"] = str(mode)
    return env


def output_limits(directory, deadline):
    for name, limit in (("stdout.log", MAX_STDOUT), ("cause-native.jsonl", MAX_NATIVE),
                        ("cause-public.txt", MAX_PUBLIC), ("warmup.dat", MAX_PUBLIC), ("protocol.raw", 65536)):
        deadline.check()
        path = directory / name
        require(not path.is_symlink(), "Worker created a linked output")
        if path.exists():
            require(path.stat().st_size <= limit, "Worker output exceeded bound: " + name)


def actual_maps(child, arm, directory, deadline):
    deadline.check()
    pid = child.process.pid
    raw = Path(f"/proc/{pid}/maps").read_text()
    (directory / "process-maps.txt").write_text(raw)
    expected = arm["runtime"]
    mapped = set()
    for line in raw.splitlines():
        columns = line.split(None, 5)
        if len(columns) == 6 and columns[5].startswith("/"):
            require(not columns[5].endswith(" (deleted)"), "Deleted file in actual runtime mappings")
            mapped.add(str(Path(columns[5]).resolve(strict=True)))
    elves = set()
    for name in mapped:
        deadline.check()
        with controlled(name).open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                elves.add(name)
    executable = str(controlled(arm["executable"]["path"]))
    require(elves == set(expected) | {executable}, "Actual mapped ELF closure differs from declared runtime")
    bridges = [name for name in elves if Path(name).name == "libmadloop_tirust.so"]
    require(len(bridges) == 1 and bridges[0] in expected, "Wrong or multiple actual TIRust DSOs")
    actual = {name: sha(Path(name), deadline) for name in sorted(elves)}
    require({name: actual[name] for name in expected} == expected
            and actual[executable] == arm["executable"]["sha256"], "Actual mapped bytes differ")
    tasks = list(Path(f"/proc/{pid}/task").iterdir())
    task_records = []
    for task in tasks:
        deadline.check()
        status = (task / "status").read_text()
        affinity = [line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("Cpus_allowed_list:")]
        require(affinity == [str(CPU)], "Numerical worker task affinity changed")
        require(not (task / "children").read_text().strip(), "Numerical worker spawned a child process")
        task_records.append({"tid": int(task.name), "cpu": CPU})
    deadline.check()
    return {"pid": pid, "sha256": actual, "maps_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "cpu": CPU, "tasks": task_records, "thread_limits": 1, "captured_before_stop": True}


def warmup_complete(path, deadline):
    deadline.check()
    lines = path.read_text().splitlines()
    rows = [line.split() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    require(len(rows) == 202 and all(len(row) == 9 for row in rows), "Worker exceeded or omitted full101-call history")
    headers = [tuple(map(int, row[:3])) for row in rows]
    expected = [(call, max(call, 1), order) for call in range(101) for order in (0, 1)]
    require(headers == expected, "Warmup order/count changed; no pruning is allowed")
    deadline.check()


def run_arm(case, arm, arm_name, directory, build, taskset, overall):
    deadline = Deadline(ARM_SECONDS, overall)
    record = {"complete": False, "arm": arm_name, "process_spawned": False, "process_reaped": False}
    child = OwnedChild(deadline, record)
    fifo_fd = None
    try:
        inputs = stage_run(case, directory, deadline)
        fifo = directory / "protocol.fifo"
        os.mkfifo(fifo, 0o600)
        fifo_fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        env = runtime_environment(build, arm, int(arm_name == "enabled"), directory)
        command = [taskset["path"], "-c", str(CPU), *arm["launch_prefix"], arm["executable"]["path"],
                   str(directory / "points.dat"), str(fifo)]
        record.update(command=command, environment=env, copied_inputs=inputs,
                      executable=arm["executable"], runtime=arm["runtime"], seconds_cap=ARM_SECONDS)
        write(directory / "execution.json", record)
        with (directory / "stdout.log").open("xb") as log, (directory / "protocol.raw").open("xb") as protocol_log:
            child.spawn(command, directory, env, log)
            buffer = bytearray()
            ready = bye = False
            def drain_protocol():
                nonlocal buffer, ready, bye
                while True:
                    deadline.check()
                    output_limits(directory, deadline)
                    try:
                        chunk = os.read(fifo_fd, 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    protocol_log.write(chunk); protocol_log.flush()
                    buffer.extend(chunk)
                    require(len(buffer) <= 65536, "Protocol line exceeded bound")
                    while b"\n" in buffer:
                        line, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        fields = line.decode("ascii").split()
                        if not ready:
                            require(len(fields) == 7 and fields[:6] == ["READY", "1", "100", "1", "1", "1"],
                                    "Worker did not reach expected full-100 READY")
                            warmup_complete(directory / "warmup.dat", deadline)
                            record["actual_runtime"] = actual_maps(child, arm, directory, deadline)
                            ready = True
                            record["ready_record"] = fields
                            child.stop_command()
                        else:
                            require(not bye and fields == ["BYE"], "Unexpected worker protocol after STOP")
                            bye = True
            while True:
                drain_protocol()
                if child.exited():
                    # BYE can arrive between the preceding read and WNOWAIT.
                    # With the exited leader reserved, drain all queued bytes.
                    drain_protocol()
                    require(ready and bye and not buffer, "Worker exited before complete READY/STOP/BYE exchange")
                    break
                time.sleep(min(.01, deadline.check()))
            require(child.finish() == 0, "Native worker exited unsuccessfully")
        verify_pins(inputs, deadline)
        verify_pins(arm["runtime"], deadline)
        check_pin(arm["executable"], deadline)
        output_limits(directory, deadline)
        record["output_sha256"] = {
            file.relative_to(directory).as_posix(): sha(file, deadline)
            for file in sorted(directory.rglob("*"))
            if file.is_file() and file.name != "execution.json"
            and "tmp" not in file.relative_to(directory).parts}
        record["complete"] = True
        write(directory / "execution.json", record)
        return record
    except BaseException as error:
        child.cleanup_after(error)
        if directory.exists():
            record_failure(directory / "execution.json", record, error)
        raise
    finally:
        if fifo_fd is not None:
            os.close(fifo_fd)
        if child.process is not None and child.process.stdin is not None:
            child.process.stdin.close()


def analyzer_module():
    require(sha(PROPOSAL / "analyze.py") == ANALYZER_SHA, "Frozen analyzer changed")
    require("protocol" not in sys.modules, "Unexpected preloaded protocol module")
    specification = importlib.util.spec_from_file_location("protocol", PROPOSAL / "protocol.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules["protocol"] = module
    specification.loader.exec_module(module)
    specification = importlib.util.spec_from_file_location("dco_frozen_analyzer", PROPOSAL / "analyze.py")
    analyzer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(analyzer)
    return analyzer


def execute(args, snapshot, protected):
    manifest_path = controlled(args.build_manifest)
    build, build_pins = verify_build(manifest_path, args.build_manifest_sha256, snapshot)
    protected.update(build_pins)
    check_pin(args.taskset_pin)
    protected[args.taskset_pin["path"]] = args.taskset_pin["sha256"]
    work = manifest_path.parent.parent
    expected_work = ROOT / ".tirust-build/dp-rescue-cause-observer" / args.label
    require(work == expected_work and manifest_path == work / "build/build.json",
            "Build manifest is not in the designated private label directory")
    target = work / "controller-native/attempt1"
    require(not target.exists(), "This build's one native attempt has already been consumed")
    lock_path = ROOT / ".tirust-build/numerical-tests.lock"
    require(not lock_path.is_symlink(), "Numerical lock must not be a symlink")
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print("Shared numerical lane busy; no attempt created and no retry scheduled.")
        return 75
    record = {"schema": "dp-rescue-six-arm-execution-v1", "complete": False,
              "performance_eligible": False, "cause_attribution_qualified": False,
              "automatic_retry_count": 0, "process_budget": 6, "cpu": CPU,
              "seconds_per_arm": ARM_SECONDS, "total_seconds_cap": TOTAL_SECONDS,
              "controller_sha256": sha(Path(__file__)), "proposal_sha256": PROPOSAL_SHA,
              "build_manifest_sha256": args.build_manifest_sha256, "protected_before": protected, "arms": {}}
    handlers = {}
    try:
        overall = Deadline(TOTAL_SECONDS)
        target.mkdir(parents=True, exist_ok=False)
        for signum in SIGNALS:
            handlers[signum] = signal.signal(signum, interrupted)
        write(target / "execution.json", record)
        verify_pins(protected, overall)
        for case_id in CASES:
            for arm_name in ARMS:
                overall.check()
                arm = build["cases"][case_id]["arms"]["original" if arm_name == "original" else "observed"]
                verify_pins(collect_pins(arm), overall)
                key = case_id + "/" + arm_name
                record["arms"][key] = run_arm(snapshot["cases"][case_id], arm, arm_name,
                                              target / case_id / arm_name, build, args.taskset_pin, overall)
                write(target / "execution.json", record)
        overall.check()
        analyzer = analyzer_module()
        record["case_analysis"] = {}
        for case_id in CASES:
            overall.check()
            directories = [target / case_id / arm for arm in ARMS]
            result = analyzer.analyze_triplet(case_id, *(analyzer.read_arm(path) for path in directories))
            overall.check()
            write(target / case_id / "analysis.json", result)
            record["case_analysis"][case_id] = pin(target / case_id / "analysis.json", overall)
        verify_pins(protected, overall)
        record.update(complete=True, protected_after=dict(protected), runtime_provenance_validated=True,
                      numerical_parity_passed=True, remaining_gate="Review retained multi-cause observations; no physics or timing acceptance")
        write(target / "execution.json", record)
    except BaseException as error:
        if target.exists():
            record_failure(target / "execution.json", record, error)
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        lock.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "build", "execute"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--self-sha256", help="Reviewed controller hash; required before build/execute")
    parser.add_argument("--build-source-sha256", help="Reviewed build.py hash for explicit build dispatch")
    parser.add_argument("--build-plan-sha256", help="Reviewed build-plan.json hash for explicit build dispatch")
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--build-manifest-sha256")
    parser.add_argument("--taskset", type=Path)
    parser.add_argument("--taskset-sha256")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", args.label), "Unsafe work label")
    snapshot, protected = proposal_inputs()
    if args.phase == "prepare":
        print(json.dumps({"schema": "dp-rescue-controller-preparation-v1", "source_only": True,
                          "controller": pin(Path(__file__)), "proposal_sha256": PROPOSAL_SHA,
                          "snapshot_sha256": SNAPSHOT_SHA, "cleanup_reference_sha256": SUPERVISOR_SHA,
                          "owned_lineage_reference_sha256": LINEAGE_SHA,
                          "protected_inputs": protected, "cases": list(CASES), "arms": list(ARMS),
                          "cpu": CPU, "seconds_per_arm": ARM_SECONDS, "total_seconds": TOTAL_SECONDS,
                          "build_executed": False, "native_executed": False}, indent=2, sort_keys=True))
        return 0
    require(args.self_sha256 and sha(Path(__file__)) == args.self_sha256, "Reviewed controller hash is required")
    if args.phase == "build":
        builder = HERE / "build.py"
        require(args.build_source_sha256 and sha(builder) == args.build_source_sha256, "Reviewed builder hash is required")
        require(args.build_plan_sha256 and sha(HERE / "build-plan.json") == args.build_plan_sha256,
                "Reviewed build plan hash is required")
        plan = read(HERE / "build-plan.json")
        require(plan["lifecycle_helper_sha256"] == args.self_sha256
                and plan["native_inputs_sha256"] == sha(HERE / "native-inputs.json"),
                "Build plan does not bind this controller and native inputs")
        # Replace this process: the separately reviewed builder owns all of its
        # compiler/linker children. No numerical lock or amplitude is started.
        command = [sys.executable, "-I", "-B", str(builder), "--build", "--label", args.label,
                   "--work-root", str(ROOT / ".tirust-build/dp-rescue-cause-observer" / args.label),
                   "--expected-plan-sha", args.build_plan_sha256]
        require(not any(os.environ.get(name) for name in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG",
                    "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")), "Unsupported build environment interception")
        # The builder reconstructs its exact frozen compilation environment;
        # these variables only launch the reviewed isolated Python entrypoint.
        os.execve(sys.executable, command, {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
                                          "PYTHONDONTWRITEBYTECODE": "1"})
    require(args.build_manifest is not None and args.build_manifest_sha256 and args.taskset is not None
            and args.taskset_sha256, "Execution requires exact build manifest and taskset pins")
    args.taskset_pin = {"path": str(args.taskset.resolve(strict=True)), "sha256": args.taskset_sha256}
    return execute(args, snapshot, protected)


if __name__ == "__main__":
    raise SystemExit(main())
