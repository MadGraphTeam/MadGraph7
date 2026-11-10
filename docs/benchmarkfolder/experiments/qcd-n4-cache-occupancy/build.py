#!/usr/bin/env python3
"""Explicit bounded private Fortran build; no numerical-worker execution.

The plan subcommand reads inputs only. Build requires a separately reviewed root
instruction; this program neither creates nor infers that instruction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LIFECYCLE = ROOT / "docs/benchmarkfolder/experiments/dp-rescue-cause-observer/controller/run.py"
LIFECYCLE_SHA = "6a0512b620f0f9723af8811bbbd128f3b1f0990aac64e3aaf1da60ec7e5cab10"
CPU, SECONDS = 12, 180


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def module(path, expected, name):
    require(digest(path) == expected, "Python source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def records(value, result=None):
    result = {} if result is None else result
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            old = result.setdefault(value["path"], value)
            require(old["sha256"] == value["sha256"], "Conflicting input pins")
        for child in value.values():
            if isinstance(child, (dict, list)):
                records(child, result)
    elif isinstance(value, list):
        for child in value:
            records(child, result)
    return result


def verify(pins, preparation, deadline=None):
    for path, expected in sorted(pins.items()):
        if deadline is not None:
            deadline.check()
        actual = preparation.pin(Path(path), expected["sha256"])
        for key in ("resolved_path", "size_bytes", "symlink"):
            require(key not in expected or actual[key] == expected[key], "Input path identity changed: " + path)
    if deadline is not None:
        deadline.check()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


class Build:
    def __init__(self, work, lifecycle, preparation):
        self.work, self.lifecycle, self.preparation = work, lifecycle, preparation
        self.deadline = lifecycle.Deadline(SECONDS)
        self.commands = []

    def copy(self, source, target):
        self.deadline.check()
        target.parent.mkdir(parents=True, exist_ok=True)
        require(target.parent.resolve() == target.parent and not target.exists(), "Unsafe private source copy")
        with Path(source).open("rb") as src, target.open("xb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                self.deadline.check()
                dst.write(block)
        require(digest(source) == digest(target), "Private source copy differs")

    def command(self, argv, cwd, env, name):
        self.deadline.check()
        log = self.work / "logs" / (name + ".txt")
        log.parent.mkdir(exist_ok=True)
        event = {"argv": list(map(str, argv)), "cwd": str(cwd), "environment": env,
                 "log": str(log), "process_spawned": False, "process_reaped": False}
        self.commands.append(event)
        child = self.lifecycle.OwnedChild(self.deadline, event)
        error = None
        try:
            with log.open("xb") as stream:
                child.spawn(event["argv"], cwd, env, stream)
                child.process.stdin.close()
                while not child.exited():
                    require(log.stat().st_size <= 16 * 1024 * 1024, "Build log exceeded bound")
                    time.sleep(min(.05, self.deadline.check()))
                require(child.finish() == 0, "Private build command failed: " + name)
            self.deadline.check()
        except BaseException as exc:
            error = exc
            event["error"] = repr(exc)
            child.cleanup_after(exc)
            raise
        finally:
            try:
                if log.exists():
                    event["log_pin"] = self.preparation.pin(log)
                write(self.work / "commands.json", self.commands)
            except BaseException as metadata_error:
                if error is None:
                    raise
                error.add_note("Build-command metadata error: " + repr(metadata_error))
        return log.read_text(errors="strict")


def runtime(build, executable, expected, environment, name):
    interpreter = build.preparation.interpreter(executable)
    require(interpreter["sha256"] == expected["interpreter"]["sha256"], "Private ELF interpreter changed")
    listing = build.command([interpreter["path"], "--list", str(executable)], build.work,
                            environment, name + "-loader-list")
    closure = {}
    for line in listing.splitlines():
        require("not found" not in line, "Unresolved private runtime dependency")
        for path in re.findall(r"(?:=>\s+|^\s*)(/\S+)", line):
            resolved = Path(path).resolve(strict=True)
            closure[str(resolved)] = build.preparation.pin(resolved)["sha256"]
    require(closure and closure == {record["resolved_path"]: record["sha256"]
            for record in expected["retained_runtime"]["expected_elf_closure"].values()},
            "Private dependency listing differs from frozen expected closure")
    bridge = [path for path in closure if Path(path).name == "libmadloop_tirust.so"]
    require(len(bridge) == 1 and closure[bridge[0]] == build.preparation.DSO_SHA, "Private link did not retain eb3d")
    return closure, interpreter


def build_private(snapshot, snapshot_pin, preparation, lifecycle, work):
    build = Build(work, lifecycle, preparation)
    protected = records(snapshot)
    protected[snapshot_pin["path"]] = snapshot_pin
    protected[str(LIFECYCLE)] = preparation.pin(LIFECYCLE, LIFECYCLE_SHA)
    record = {"schema": "qcd-n4-cache-occupancy-build-v1", "status": "started",
              "snapshot_sha256": snapshot_pin["sha256"], "builder_sha256": digest(__file__),
              "lifecycle_helper": protected[str(LIFECYCLE)], "cpu": CPU, "jobs": 1,
              "seconds_cap": SECONDS, "numerical_execution_performed": False,
              "runtime_provenance_validated": False, "performance_eligible": False,
              "arms": {}, "protected_before": protected}
    handlers, old_affinity, error = {}, os.sched_getaffinity(0), None
    try:
        for signum in lifecycle.SIGNALS:
            handlers[signum] = signal.signal(signum, lifecycle.interrupted)
        os.sched_setaffinity(0, {CPU})
        verify(protected, preparation, build.deadline)
        clone = work / "generated"
        for collection in (snapshot["sources"], snapshot["modules"]):
            for relative_name, source in collection.items():
                build.copy(source["path"], preparation.relative(clone, relative_name))
        sub = clone / "SubProcesses" / snapshot["subprocess"]
        module_dir = work / "modules"
        module_dir.mkdir()
        observer = sub / "observer.f90"
        driver = sub / "occupancy_driver.f"
        build.copy(snapshot["rendered"]["common/observer.f90"]["path"], observer)
        build.copy(snapshot["rendered"]["common/driver.f"]["path"], driver)
        derived_adapter = sub / "occupancy_TIRUST_interface.f"
        build.copy(snapshot["rendered"]["observed/TIRUST_interface.f"]["path"], derived_adapter)
        record["private_sources"] = {p.name: preparation.pin(p) for p in (observer, driver, derived_adapter)}
        fc = snapshot["compile_argv"]["check_sa.f"][0]
        require(fc in snapshot["tools"], "Compiler is not in frozen tools")
        environment = {"PATH": str(Path(fc).parent) + os.pathsep + os.defpath,
                       "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "TMPDIR": str(work / "tmp"),
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        (work / "tmp").mkdir()
        extra_flags = ["-I" + str(module_dir), "-J" + str(module_dir)]
        for path in snapshot["include_directories"]:
            extra_flags.append("-I" + path)
        common_observer = sub / "observer.o"
        common_driver = sub / "occupancy_driver.o"
        adapter_object = sub / "occupancy_TIRUST_interface.o"
        build.command([fc, "-w", "-fPIC", *extra_flags, "-c", str(observer), "-o", str(common_observer)],
                      sub, environment, "compile-observer")
        for source_name, source, target, command_name in (
                ("check_sa.f", driver, common_driver, "compile-driver"),
                ("TIRUST_interface.f", derived_adapter, adapter_object, "compile-adapter")):
            argv = list(snapshot["compile_argv"][source_name])
            argv[argv.index(source_name)] = str(source)
            argv[argv.index("-o") + 1] = str(target)
            argv += ["-I" + str(module_dir), "-J" + str(module_dir)]
            build.command(argv, sub, environment, command_name)
        original_sub = Path(snapshot["generated_root"]) / "SubProcesses" / snapshot["subprocess"]
        numerical = [token for token in snapshot["link_argv"] if token.endswith(".o")]
        require(len(numerical) == 25 and numerical.count("check_sa.o") == 1
                and numerical.count("TIRUST_interface.o") == 1, "Unexpected original numerical link inventory")
        common = {"observer.o": preparation.pin(common_observer), "driver.o": preparation.pin(common_driver)}
        bridge = snapshot["installed_native"]["lib/libmadloop_tirust.so"]
        runtime_environment = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
                               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                               "RAYON_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                               "LD_LIBRARY_PATH": os.pathsep.join([str(Path(bridge["path"]).parent),
                                   str(ROOT / "HEPTools/ninja/lib"), str(ROOT / "HEPTools/collier"),
                                   str(ROOT / "HEPTools/ninjars/lib")])}
        for name in ("original", "observed"):
            binary = work / ("check-" + name)
            link_map = work / (name + "-link.map")
            argv = list(snapshot["link_argv"])
            argv[argv.index("-o") + 1] = str(binary)
            objects = {}
            for index, token in enumerate(argv):
                if token in numerical:
                    path = original_sub / token
                    if token == "check_sa.o":
                        path = common_driver
                    elif name == "observed" and token == "TIRUST_interface.o":
                        path = adapter_object
                    argv[index] = str(path)
                    objects[token] = preparation.pin(path)
            argv.insert(argv.index(next(token for token in argv if token.startswith("-L"))), str(common_observer))
            argv.append("-Wl,-Map," + str(link_map))
            build.command(argv, sub, environment, "link-" + name)
            actual_link_inputs = {}
            allowed_loads = {}
            for group in (snapshot["toolchain_link_inputs"], snapshot["link_inputs"]["archives"],
                          snapshot["retained_runtime"]["libraries"], objects, common):
                for value in group.values():
                    resolved = value["resolved_path"]
                    require(resolved not in allowed_loads or allowed_loads[resolved] == value["sha256"],
                            "Conflicting allowed linker input")
                    allowed_loads[resolved] = value["sha256"]
            for line in link_map.read_text().splitlines():
                if line.startswith("LOAD "):
                    path = Path(line[5:].strip())
                    if not path.is_absolute():
                        path = sub / path
                    path = path.resolve(strict=True)
                    require(str(path) in allowed_loads, "Unbound actual linker LOAD: " + str(path))
                    actual_link_inputs[str(path)] = preparation.pin(path, allowed_loads[str(path)])
            require(actual_link_inputs and bridge["resolved_path"] in actual_link_inputs,
                    "Actual linker map omits installed TIRust DSO")
            closure, elf_interpreter = runtime(build, binary, snapshot, runtime_environment, name)
            record["arms"][name] = {"executable": preparation.pin(binary), "objects": objects,
                                    "common_objects": common, "link_inputs": actual_link_inputs,
                                    "link_map": preparation.pin(link_map), "runtime": closure,
                                    "interpreter": elf_interpreter, "launch_prefix": [elf_interpreter["path"]],
                                    "runtime_environment": runtime_environment}
        require(record["arms"]["original"]["runtime"] == record["arms"]["observed"]["runtime"],
                "Original and observed runtime closures differ")
        verify(protected, preparation, build.deadline)
        record.update(status="complete", protected_after=protected,
                      disabled_enabled_same_executable=True, installed_dso=bridge,
                      runtime_provenance_validated=True,
                      runtime_provenance_scope="Bound interpreter, linker LOAD inputs, and dependency listing; no amplitude executed",
                      numerical_runtime_mapping_validated=False)
    except BaseException as exc:
        error = exc
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        try:
            if error is not None:
                try:
                    verify(protected, preparation)
                    record["protected_after"] = protected
                except BaseException as preservation_error:
                    record["protected_after_error"] = repr(preservation_error)
                    error.add_note("Protected-input verification error: " + repr(preservation_error))
            record["commands"] = build.commands
            record["artifacts"] = {p.relative_to(work).as_posix(): preparation.pin(p)
                                   for p in sorted(work.rglob("*")) if p.is_file() and p.name != "build.json"}
            write(work / "build.json", record)
        except BaseException as metadata_error:
            if error is None:
                raise
            error.add_note("Build-final metadata error: " + repr(metadata_error))
        finally:
            os.sched_setaffinity(0, old_affinity)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    return preparation.pin(work / "build.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "build"))
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", args.label),
            "Wrong working directory or unsafe label")
    snapshot_path = args.snapshot.resolve(strict=True)
    require(HERE / "preparation" in snapshot_path.parents and snapshot_path.name == "snapshot.json"
            and digest(snapshot_path) == args.snapshot_sha256, "Snapshot identity differs")
    snapshot = json.loads(snapshot_path.read_text())
    require(snapshot["schema"] == "qcd-n4-cache-occupancy-preparation-v1", "Wrong preparation schema")
    require(snapshot["private_sources"]["build.py"]["sha256"] == digest(__file__), "Builder differs from frozen preparation")
    preparation = module(HERE / "prepare.py", snapshot["private_sources"]["prepare.py"]["sha256"], "occupancy_preparation")
    lifecycle = module(LIFECYCLE, LIFECYCLE_SHA, "occupancy_lifecycle")
    verify(records(snapshot), preparation)
    if args.phase == "plan":
        print(json.dumps({"source_only": True, "snapshot": preparation.pin(snapshot_path),
                          "cpu": CPU, "jobs": 1, "seconds_cap": SECONDS,
                          "compile_commands": 3, "link_commands": 2, "loader_list_commands": 2,
                          "rust_build": False, "numerical_execution": False}, sort_keys=True))
        return
    require(not any(os.environ.get(key) for key in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG")),
            "Ambient loader interception is present")
    work = ROOT / ".tirust-build/qcd-n4-cache-occupancy" / args.label / "build"
    require(work.resolve() == work and not work.exists(), "Build attempt already exists or aliases another path")
    work.mkdir(parents=True, exist_ok=False)
    result = build_private(snapshot, preparation.pin(snapshot_path), preparation, lifecycle, work)
    print(json.dumps({"build_manifest": result, "numerical_execution": False}, sort_keys=True))


if __name__ == "__main__":
    main()
