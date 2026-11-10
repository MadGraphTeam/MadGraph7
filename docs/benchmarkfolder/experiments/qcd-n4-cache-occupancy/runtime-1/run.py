#!/usr/bin/env python3
"""One explicitly invoked, three-arm diagnostic; plan performs reads only.

Native execution needs a separate root instruction naming this reviewed source
hash and resources. This program neither creates nor infers that instruction.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = PROPOSAL.parents[3]
SNAPSHOT = PROPOSAL / "preparation/source-1/snapshot.json"
SNAPSHOT_SHA = "c7dba79d4d84e091840a910e079d8a8f48969b4c6a6b98995b966ad4c6b5f7e3"
BUILD = ROOT / ".tirust-build/qcd-n4-cache-occupancy/attempt1/build/build.json"
BUILD_SHA = "40dedfe9db3793d20f21fc97f6f7d1f4507268619d968ce0c64392bf5be58d8a"
LIFECYCLE = ROOT / "docs/benchmarkfolder/experiments/dp-rescue-cause-observer/controller/run.py"
LIFECYCLE_SHA = "6a0512b620f0f9723af8811bbbd128f3b1f0990aac64e3aaf1da60ec7e5cab10"
DSO_SHA = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
TARGET = BUILD.parent.parent / "runtime-1/attempt1"
LOCK = ROOT / ".tirust-build/numerical-tests.lock"
CPU, ARM_SECONDS, TOTAL_SECONDS = 12, 180, 600
ARMS = ("original", "disabled", "enabled")
LIMITS = {"stdout.log": 16 * 1024**2, "public.txt": 64 * 1024,
          "occupancy.txt": 80 * 1024**2}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path, deadline=None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if deadline is not None:
                deadline.check()
            block = stream.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def load_module(path, expected, name):
    need(digest(path) == expected, "Changed Python helper: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    need(not path.is_symlink() and path.parent.resolve() == path.parent, "Unsafe metadata destination")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def verify(pins, preparation, deadline=None):
    for name, expected in sorted(pins.items()):
        if deadline is not None:
            deadline.check()
        path = Path(name)
        preparation.controlled(path)
        need(digest(path, deadline) == expected["sha256"], "Protected bytes changed: " + name)
        for key, actual in (("resolved_path", str(path.resolve(strict=True))),
                            ("size_bytes", path.stat().st_size),
                            ("symlink", os.readlink(path) if path.is_symlink() else None)):
            need(key not in expected or expected[key] == actual, "Protected path identity changed: " + name)


def inputs():
    need(Path.cwd().resolve() == ROOT, "Run from the MadGraph repository root")
    need(digest(SNAPSHOT) == SNAPSHOT_SHA and digest(BUILD) == BUILD_SHA, "Wrong frozen snapshot or build")
    snapshot = json.loads(SNAPSHOT.read_text())
    build = json.loads(BUILD.read_text())
    sources = snapshot["private_sources"]
    preparation = load_module(PROPOSAL / "prepare.py", sources["prepare.py"]["sha256"], "occupancy_runtime_preparation")
    builder = load_module(PROPOSAL / "build.py", sources["build.py"]["sha256"], "occupancy_runtime_builder")
    lifecycle = load_module(LIFECYCLE, LIFECYCLE_SHA, "occupancy_runtime_lifecycle")
    need(snapshot["schema"] == "qcd-n4-cache-occupancy-preparation-v1"
         and build["schema"] == "qcd-n4-cache-occupancy-build-v1" and build["status"] == "complete",
         "Preparation/build is not complete for this protocol")
    need(build["snapshot_sha256"] == SNAPSHOT_SHA and build["builder_sha256"] == sources["build.py"]["sha256"]
         and build["protected_before"] == build["protected_after"], "Build provenance differs")
    need(build["numerical_execution_performed"] is False and build["runtime_provenance_validated"] is True
         and build["numerical_runtime_mapping_validated"] is False, "Unexpected build scope")
    need(build["disabled_enabled_same_executable"] is True and set(build["arms"]) == {"original", "observed"},
         "Wrong binary matrix")
    need(len(build["commands"]) == 7 and all(row.get("returncode") == 0 and row.get("process_reaped")
         for row in build["commands"]), "Build lacks seven successful owned commands")
    need(build["installed_dso"]["sha256"] == DSO_SHA, "Build does not use installed eb3d DSO")
    protected = builder.records(snapshot)
    builder.records(build, protected)
    for path in (SNAPSHOT, BUILD, LIFECYCLE, Path(__file__).resolve()):
        protected[str(path)] = preparation.pin(path)
    for arm in build["arms"].values():
        need(arm["launch_prefix"] == [arm["interpreter"]["path"]], "Unexpected launch prefix")
        for path, sha in arm["runtime"].items():
            protected[path] = preparation.pin(Path(path), sha)
        bridges = [p for p in arm["runtime"] if Path(p).name == "libmadloop_tirust.so"]
        need(bridges == [build["installed_dso"]["resolved_path"]]
             and arm["runtime"][bridges[0]] == DSO_SHA, "Wrong runtime bridge")
        need(set(arm["common_objects"]) == {"driver.o", "observer.o"}, "Wrong common objects")
        for token, obj in arm["objects"].items():
            if token == "check_sa.o":
                need(obj == arm["common_objects"]["driver.o"], "Driver object differs")
            elif arm is build["arms"]["original"] or token != "TIRUST_interface.o":
                key = "SubProcesses/" + snapshot["subprocess"] + "/" + token
                need(obj["sha256"] == snapshot["objects"][key]["sha256"], "Unexpected numerical object change")
    original, observed = (build["arms"][name] for name in ("original", "observed"))
    need(set(original["objects"]) == set(observed["objects"]) and len(original["objects"]) == 25
         and original["common_objects"] == observed["common_objects"]
         and original["runtime"] == observed["runtime"]
         and original["runtime_environment"] == observed["runtime_environment"], "Arm inputs differ unexpectedly")
    need(snapshot["protocol"]["arms"] == list(ARMS) and snapshot["protocol"]["public_calls"] == [
        {"phase": 0, "point": 1, "sequence": 0}, {"phase": 1, "point": 1, "sequence": 1}], "Public call protocol changed")
    verify(protected, preparation)
    analyzer = load_module(PROPOSAL / "analyze.py", sources["analyze.py"]["sha256"], "occupancy_runtime_analyzer")
    return snapshot, build, protected, preparation, lifecycle, analyzer


def stage(snapshot, directory, lifecycle, deadline):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tmp").mkdir()
    sources = {"points.dat": snapshot["point"]["dat"], **snapshot["cards"]}
    for name, value in snapshot["initial_resources"].items():
        need(Path(name).name == name, "Unsafe initial resource name")
        if not name.endswith(("HelFilter.dat", "LoopFilter.dat")):
            sources["MadLoop5_resources/" + name] = value
    copied = {}
    for name, value in sources.items():
        target = lifecycle.relative(directory, name)
        lifecycle.copy_pin(value, target, deadline)
        copied[str(target)] = {"sha256": value["sha256"]}
    need(not (directory / "param_card.dat").exists() and not (directory / "MadLoopParams.dat").exists(),
         "Root parameter cards break original fallback semantics")
    need(not any(directory.rglob("*HelFilter.dat")) and not any(directory.rglob("*LoopFilter.dat")),
         "Fresh run contains trained filters")
    return copied


def environment(arm, name, directory):
    env = dict(arm["runtime_environment"])
    allowed = {"PATH", "LANG", "LC_ALL", "TZ", "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}
    need(set(env) == allowed and all(isinstance(v, str) for v in env.values()), "Unexpected frozen environment")
    need(all(env[k] == "1" for k in allowed if k.endswith("_NUM_THREADS") or k == "VECLIB_MAXIMUM_THREADS"),
         "Thread limits differ")
    env.update(TMPDIR=str(directory / "tmp"), OCC_OBSERVE=str(int(name == "enabled")))
    return env


def output_limits(directory, deadline):
    total = count = 0
    for path in directory.rglob("*"):
        deadline.check()
        need(not path.is_symlink(), "Linked runtime output")
        if path.is_dir():
            continue
        info = path.stat()
        need(stat.S_ISREG(info.st_mode), "Nonregular runtime output")
        count += 1
        total += info.st_size
        need(info.st_size <= LIMITS.get(path.name, 16 * 1024**2), "Runtime output exceeded bound: " + str(path))
        need(count <= 128 and total <= 128 * 1024**2, "Runtime output inventory exceeded bound")


def sample_process(child, arm, directory, deadline, preparation, record):
    """A live /proc sample; incomplete loader startup may be sampled again."""
    pid = child.process.pid
    expected = dict(arm["runtime"])
    expected[arm["executable"]["resolved_path"]] = arm["executable"]["sha256"]
    deadline.check()
    proc = Path("/proc") / str(pid)
    tasks = []
    try:
        for task in sorted((proc / "task").iterdir()):
            try:
                affinity = sorted(os.sched_getaffinity(int(task.name)))
                children = (task / "children").read_text().strip()
            except (FileNotFoundError, ProcessLookupError):
                continue  # A worker thread may finish between these reads.
            need(affinity == [CPU] and not children, "Worker affinity changed or spawned a process")
            tasks.append({"tid": int(task.name), "affinity": affinity, "children": children})
        raw = (proc / "maps").read_text()
    except (FileNotFoundError, ProcessLookupError):
        need(child.exited(), "Live worker /proc state disappeared")
        return
    if not raw:
        need(child.exited(), "Live worker has empty mappings")
        return
    elves = {}
    for line in raw.splitlines():
        deadline.check()
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        need(not fields[5].endswith(" (deleted)"), "Deleted runtime mapping")
        path = Path(fields[5]).resolve(strict=True)
        if str(path) in elves:
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
            info = os.fstat(stream.fileno())
        need(str(path) in expected, "Unexpected mapped ELF: " + str(path))
        dev = ":".join(format(x, "x") for x in (os.major(info.st_dev), os.minor(info.st_dev)))
        mapped_dev = ":".join(format(int(x, 16), "x") for x in fields[3].split(":"))
        need(dev == mapped_dev and info.st_ino == int(fields[4]), "Mapped inode differs from pinned path")
        elves[str(path)] = {"device": dev, "inode": info.st_ino}
    record["process_samples"] = record.get("process_samples", 0) + 1
    if set(elves) != set(expected):
        need(not record.get("actual_runtime"), "Previously complete runtime closure disappeared")
        return
    need(tasks, "No task affinity sample for complete runtime")
    if not record.get("actual_runtime"):
        verify({p: {"sha256": sha} for p, sha in expected.items()}, preparation, deadline)
        (directory / "process-maps-first.txt").write_text(raw)
        record["actual_runtime"] = {"pid": pid, "sha256": expected, "mapped_files": elves,
                                    "first_tasks": tasks, "live_maps_captured": True,
                                    "first_maps_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    else:
        need(elves == record["actual_runtime"]["mapped_files"], "Mapped inode identity changed")
    (directory / "process-maps-last.txt").write_text(raw)
    record["actual_runtime"].update(last_tasks=tasks, last_maps_sha256=hashlib.sha256(raw.encode()).hexdigest())


def retained_outputs(directory, deadline=None):
    return {str(path.relative_to(directory)): {"sha256": digest(path, deadline), "size_bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*")) if path.is_file() and not path.is_symlink()
            and path.name != "execution.json" and "tmp" not in path.relative_to(directory).parts}


def run_arm(snapshot, arm, name, directory, record, preparation, lifecycle, analyzer, overall):
    deadline = lifecycle.Deadline(ARM_SECONDS, overall)
    child = lifecycle.OwnedChild(deadline, record)
    error, copied = None, {}
    try:
        copied = stage(snapshot, directory, lifecycle, deadline)
        env = environment(arm, name, directory)
        command = [*arm["launch_prefix"], arm["executable"]["path"], str(directory / "points.dat")]
        record.update(command=command, environment=env, copied_inputs=copied, executable=arm["executable"],
                      expected_runtime=arm["runtime"], seconds_cap=ARM_SECONDS)
        write(directory / "execution.json", record)
        with (directory / "stdout.log").open("xb") as log:
            child.spawn(command, directory, env, log)
            child.process.stdin.close()
            while not child.exited():
                output_limits(directory, deadline)
                sample_process(child, arm, directory, deadline, preparation, record)
                time.sleep(min(.01, deadline.check()))
            need(child.finish() == 0, "Numerical worker exited unsuccessfully")
        need(record.get("actual_runtime"), "Worker exited before complete live runtime capture; no retry")
        output_limits(directory, deadline)
        verify(copied, preparation, deadline)
        verify({arm["executable"]["path"]: arm["executable"],
                **{p: {"sha256": sha} for p, sha in arm["runtime"].items()}}, preparation, deadline)
        public = analyzer.public(directory / "public.txt")
        analyzer.occupancy(directory / "occupancy.txt", name == "enabled")
        deadline.check()
        record.update(complete=True, public_records=public, outputs=retained_outputs(directory, deadline))
        return public
    except BaseException as exc:
        error = exc
        record.update(complete=False, error=repr(exc))
        child.cleanup_after(exc)
        raise
    finally:
        if error is not None and copied:
            try:
                verify(copied, preparation)
                record["copied_inputs_unchanged_after_failure"] = True
            except BaseException as preservation_error:
                record["copied_inputs_after_error"] = repr(preservation_error)
                error.add_note("Arm preservation error: " + repr(preservation_error))
        try:
            if directory.exists():
                write(directory / "execution.json", record)
        except BaseException as metadata_error:
            if error is None:
                raise
            error.add_note("Arm preservation/metadata error: " + repr(metadata_error))


def execute(snapshot, build, protected, preparation, lifecycle, analyzer):
    need(not any(os.environ.get(k) for k in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG")), "Ambient loader interception")
    need(not TARGET.exists() and TARGET.resolve() == TARGET, "Native attempt already exists or path aliases")
    need(not LOCK.is_symlink(), "Numerical lock is a symlink")
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Shared numerical lane busy; no attempt created, no retry scheduled.")
            return 75
        need(not TARGET.exists(), "Native attempt was already consumed before lock acquisition")
        overall = lifecycle.Deadline(TOTAL_SECONDS)
        handlers, old_affinity, error, created = {}, os.sched_getaffinity(0), None, False
        record = {"schema": "qcd-n4-cache-occupancy-execution-v1", "complete": False,
                  "controller_sha256": digest(__file__), "snapshot_sha256": SNAPSHOT_SHA,
                  "build_manifest_sha256": BUILD_SHA, "analyzer": snapshot["private_sources"]["analyze.py"],
                  "cpu": CPU, "thread_limits": 1, "seconds_per_arm": ARM_SECONDS, "total_seconds_cap": TOTAL_SECONDS,
                  "process_budget": 3, "automatic_retry_count": 0, "performance_eligible": False,
                  "protected_before": protected, "arms": {}}
        try:
            for signum in lifecycle.SIGNALS:
                handlers[signum] = signal.signal(signum, lifecycle.interrupted)
            os.sched_setaffinity(0, {CPU})
            TARGET.mkdir(parents=True, exist_ok=False)
            created = True
            write(TARGET / "execution.json", record)
            verify(protected, preparation, overall)
            baseline = None
            for name in ARMS:
                overall.check()
                arm = build["arms"]["original" if name == "original" else "observed"]
                verify({arm["executable"]["path"]: arm["executable"],
                        **{p: {"sha256": sha} for p, sha in arm["runtime"].items()}}, preparation, overall)
                state = {"arm": name, "complete": False, "process_spawned": False, "process_reaped": False}
                record["arms"][name] = state
                public = run_arm(snapshot, arm, name, TARGET / name, state,
                                 preparation, lifecycle, analyzer, overall)
                if baseline is None:
                    baseline = public
                need(public == baseline, "Original/disabled/enabled raw or IEEE state mismatch")
                write(TARGET / "execution.json", record)
            overall.check()
            analysis = analyzer.summarize(TARGET)
            overall.check()
            write(TARGET / "analysis.json", analysis)
            record["analysis_sha256"] = digest(TARGET / "analysis.json", overall)
            record["numerical_parity_passed"] = True
            need(analysis["second_call_accepted"], "Point 1 was not accepted after the fixed bootstrap; no retry")
            verify(protected, preparation, overall)
            record.update(complete=True, protected_after=protected, numerical_parity_passed=True,
                          runtime_provenance_validated=True)
        except BaseException as exc:
            error = exc
            record.update(complete=False, error=repr(exc))
            raise
        finally:
            try:
                if error is not None:
                    try:
                        verify(protected, preparation)
                        record["protected_after"] = protected
                    except BaseException as preservation_error:
                        record["protected_after_error"] = repr(preservation_error)
                        error.add_note("Protected input verification error: " + repr(preservation_error))
                if created:
                    write(TARGET / "execution.json", record)
            except BaseException as metadata_error:
                if error is None:
                    raise
                error.add_note("Execution metadata error: " + repr(metadata_error))
            finally:
                os.sched_setaffinity(0, old_affinity)
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "execute"))
    parser.add_argument("--self-sha256", help="Root-reviewed source hash; required for execute")
    args = parser.parse_args()
    if args.phase == "execute":
        need(args.self_sha256 and digest(__file__) == args.self_sha256, "Reviewed controller hash required")
    bound = inputs()
    if args.phase == "plan":
        print(json.dumps({"source_only": True, "snapshot_sha256": SNAPSHOT_SHA, "build_manifest_sha256": BUILD_SHA,
                          "controller_sha256": digest(__file__), "cpu": CPU, "seconds_per_arm": ARM_SECONDS,
                          "total_seconds_cap": TOTAL_SECONDS, "arms": ARMS, "native_executed": False,
                          "attempt_directory": str(TARGET), "lock": str(LOCK)}, sort_keys=True))
        return 0
    return execute(*bound)


if __name__ == "__main__":
    raise SystemExit(main())
