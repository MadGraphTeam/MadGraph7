#!/usr/bin/env python3
"""Private four-lepton source preparation; default prints a plan, no workload."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE / "work"
REPORT = HERE / "report"
PLAN = HERE / "case-plan.json"
CLEANUP_SECONDS = 60.


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def inside(root, relative):
    name = Path(relative)
    target = Path(root) / name
    require(not name.is_absolute() and ".." not in name.parts
            and target.resolve().is_relative_to(Path(root).resolve()), "Path escapes its bound root")
    return target


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configurations():
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for p in sorted((ROOT / "input").glob("*configuration*.txt")) if p.is_file()}


def plan():
    value = read(PLAN)
    require(value["case"] == {"id": "qcd_n4_uux_epem_mupmum", "model": "loop_sm",
        "process": "u u~ > e+ e- mu+ mu- [virt=QCD]", "nfinal": 4, "role": "diagnostic"}, "Unexpected case")
    require(value["cpu"] == 12 and value["jobs"] == 1 and value["base_seed"] == 20261109
            and value["case_index"] == 0 and value["effective_seed"] == 20261109, "Preparation policy changed")
    require(value["stage_caps_seconds"] == {"generate": 900., "build": 1200.}
            and value["cleanup_reserve_seconds"] == CLEANUP_SECONDS, "Stage deadline policy changed")
    return value


def source_guard(value, native=False):
    """Source/config hashing only by default; no Git, compiler or loader query."""
    for name, checksum in value["files"].items():
        require(sha(inside(ROOT, name)) == checksum, "Bound source/policy changed: " + name)
    manifest = read(ROOT / value["manifest_path"])
    identity = manifest["candidate"]["exporter_identity"]
    require(identity["revision"] == value["source_revision"]
            and identity["files_sha256"] == value["source_files_sha256"]
            and content_sha(identity["files"]) == value["source_files_sha256"]
            and manifest["candidate"]["numerical_identity_sha256"] == value["numerical_identity_sha256"],
            "Current source/numerical manifest identity differs")
    for name, checksum in identity["files"].items():
        path = inside(ROOT, name)
        require((sha(path) if path.is_file() else None) == checksum, "Current numerical source changed: " + name)
    require(configurations() == value["configuration_files"], "Production configuration drift; no restoration")
    require(manifest["candidate"]["native"]["prefix"] == str(ROOT / value["tirust_prefix"])
            and manifest["candidate"]["native"]["files"]["lib/libmadloop_tirust.so"] == value["library_sha256"],
            "Manifest native identity differs")
    if native:
        for name, checksum in manifest["candidate"]["native"]["files"].items():
            require(sha(inside(ROOT / value["tirust_prefix"], name)) == checksum,
                    "Installed native file changed: " + name)
    return {"source_files_sha256": identity["files_sha256"], "configuration_files": configurations(),
            "native_file_hashes_checked": native, "native_queries_performed": False}


def custom_config(value):
    config = read(ROOT / value["policy_path"])
    require(config["seed"] == value["base_seed"] and config["points_per_case"] == 100
            and config["energy_gev"] == 1000., "Pinned sampling policy differs")
    config["cases"] = [value["case"]]
    require(config["seed"] + 0 == value["effective_seed"], "Effective point seed differs")
    return config


def fresh_benchmark(value):
    """Load an independent module; never activate or modify a frozen campaign."""
    sys.path.insert(0, str(ROOT / "tools"))
    bench = load_file("four_lepton_private_benchmark", ROOT / "tools/tirust_benchmark.py")
    require(bench.ROOT == ROOT, "Wrong repository root")
    bench.CAMPAIGN_ID = bench.CAMPAIGN_MANIFEST = bench.CAMPAIGN_MANIFEST_SHA256 = None
    bench.WORK, bench.REPORT, bench.MODEL_ROOT = WORK, REPORT, WORK / "models"
    bench.TIRUST_PREFIX = ROOT / value["tirust_prefix"]
    return bench


def schedule(value):
    return [{"id": name, "cap_seconds": value["stage_caps_seconds"][name],
             "command": ["taskset", "-c", str(value["cpu"]), sys.executable, "-B",
                         str(HERE / "run.py"), "--execute", "--child-stage", name]}
            for name in ("generate", "build")]


def load_support(value):
    sys.path.insert(0, str(ROOT / "tools"))
    cleanup = load_file("process_cleanup", ROOT / value["cleanup_source"])
    preparation = load_file("four_lepton_stage_support", ROOT / value["deadline_source"])
    return cleanup, preparation


def verify_generated(bench, value):
    output = bench.output_dir("candidate", value["case"])
    generation = read(output / "generation.json")
    require(generation["case"] == value["case"] and generation["source"] == "candidate"
            and generation.get("campaign_binding") is None and not generation.get("reused_from")
            and generation["source_identity"]["files_sha256"] == value["source_files_sha256"],
            "Fresh generation identity differs")
    require(sha(output / "Cards/param_card.dat") == generation["parameter_card_sha256"], "Physical card drift")
    commands = (REPORT / "commands" / ("candidate-" + value["case"]["id"] + ".mg5")).read_text().splitlines()
    require(commands == bench.cli_lines(ROOT, "candidate", value["case"], output)
            and all(line.endswith(" --no_save") for line in commands if line.startswith("set ")),
            "CLI commands or no_save policy differ")
    require(read(REPORT / "inputs/candidate" / value["case"]["id"] / "generation.json") == generation,
            "Retained generation differs")
    return generation


def verify_built(bench, value):
    generation = verify_generated(bench, value)
    output = bench.output_dir("candidate", value["case"])
    compiled = read(output / "compilation.json")
    require(compiled.get("campaign_binding") is None
            and compiled.get("numerical_build_from_sources") is True
            and compiled.get("full_numerical_rebuild_this_invocation") is True
            and compiled.get("previous_compilation_sha256") is None
            and compiled["process_build_command"] == ["make", "--always-make", "-j1", "check"]
            and set(compiled["build_logs"]) == {"MODEL", "DHELAS", "process"}, "Full fresh source rebuild not proven")
    require(bench.generated_sources(output) == compiled["sources"], "Generated sources changed after build")
    directory = bench.subprocess_dir("candidate", value["case"])
    require(sha(directory / "check") == compiled["executable_sha256"], "New executable changed")
    bench.verify_worker_template(compiled, ROOT / "tools/tirust_benchmark_worker.f")
    require(read(REPORT / "inputs/candidate" / value["case"]["id"] / "compilation.json") == compiled,
            "Retained compilation differs")
    points = REPORT / "points" / (value["case"]["id"] + ".dat")
    metadata = read(points.with_suffix(".json"))
    masses = compiled["masses"]
    require(len(masses) == 6 and all(math.isfinite(x) and x >= 0 for x in masses), "Invalid external masses")
    require(metadata["inputs"] == {"masses": masses[2:], "incoming_masses": masses[:2],
            "seed": value["effective_seed"], "count": 100, "energy": 1000.}
            and sha(points) == metadata["sha256"], "Point stream differs")
    return {"generation_sha256": sha(output / "generation.json"),
            "compilation_sha256": sha(output / "compilation.json"),
            "executable_sha256": compiled["executable_sha256"], "masses": masses,
            "points_sha256": sha(points), "points_metadata_sha256": sha(points.with_suffix(".json")),
            "full_numerical_rebuild": True, "mass_query_only": True,
            "amplitude_evaluations": 0, "performance_acceptance_eligible": False}


def child_stage(name, value):
    spec = read(HERE / "specification.json")
    require(spec["supervisor_pid"] == os.getppid() and spec["controller_sha256"] == sha(HERE / "run.py")
            and spec["case_plan_sha256"] == sha(PLAN) and spec["stages"] == schedule(value),
            "Child lacks its live controller specification")
    require(os.sched_getaffinity(0) == {value["cpu"]}, "Child affinity differs")
    source_guard(value, native=True)
    bench = fresh_benchmark(value)
    require(read(REPORT / "cases.json") == custom_config(value), "Private configuration differs")
    output = bench.output_dir("candidate", value["case"])
    if name == "generate":
        require(not output.exists(), "Generation destination already exists; no retry or reuse")
        bench.generate("candidate", value["case"])
        verify_generated(bench, value)
    else:
        verify_generated(bench, value)
        require(not (output / "compilation.json").exists() and not (REPORT / "points").exists(),
                "Build/point destination already exists; no retry or reuse")
        masses = bench.build("candidate", value["case"], jobs=1)
        bench.freeze_points(value["case"], masses, custom_config(value))
        write(HERE / "prepared.json", verify_built(bench, value), exclusive=True)
    source_guard(value, native=True)


def cleanup_owned(owned, end):
    budget = min(55., max(0., end - time.monotonic()))
    result = owned.stop(cooperative_seconds=budget * 45./55.,
                        terminate_seconds=budget * 5./55., kill_seconds=budget * 5./55.)
    require(not result["survivors"], "Observed owned descendants survived; no next stage")
    return result


def run_stage(item, value, initial, environment, requests, state, support):
    cleanup, preparation = support
    preparation.STAGE_SECONDS, preparation.CLEANUP_RESERVE = item["cap_seconds"], CLEANUP_SECONDS
    directory = HERE / "stages" / item["id"]
    directory.mkdir(parents=True, exist_ok=False)
    entry = {**item, "status": "running", "started_utc": now()}
    state["stages"].append(entry)
    process = owned = None
    cleaned = False
    try:
        with preparation.StageDeadline() as deadline:
            entry["deadline_monotonic"] = deadline.end
            requests.check()
            require(source_guard(value, native=True) == initial, "Inputs changed before stage")
            require(sha(HERE / "run.py") == state["controller_sha256"] and sha(PLAN) == state["case_plan_sha256"],
                    "Controller or plan changed")
            with (directory / "console.log").open("xb") as stream:
                deadline.handoff = True
                try:
                    process = subprocess.Popen(item["command"], cwd=ROOT, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    owned = cleanup.OwnedChild(process)
                finally:
                    deadline.handoff = False
                    if deadline.deferred:
                        deadline.expired(None, None)
                entry["child_pid"] = process.pid
                write(HERE / "status.json", state)
                entry["returncode"] = owned.wait(requests)
                requests.check()
                entry["cleanup"] = cleanup_owned(owned, deadline.end)
                cleaned = True
            require(entry["returncode"] == 0, "Preparation child failed; raw evidence retained")
            require(source_guard(value, native=True) == initial, "Inputs changed after stage")
            requests.check()
            entry.update(status="passed", finished_utc=now(), console_sha256=sha(directory / "console.log"))
    except BaseException as error:
        if owned is not None and not cleaned:
            try:
                entry["cleanup"] = cleanup_owned(owned, entry["deadline_monotonic"])
            except BaseException as cleanup_error:
                entry["cleanup_error"] = repr(cleanup_error)
        elif process is not None and owned is None:
            # Retain the Popen handle even if lineage acquisition failed. A new
            # positively matched handle may clean up; otherwise fail closed.
            try:
                owned = cleanup.OwnedChild(process)
                entry["cleanup"] = cleanup_owned(owned, entry["deadline_monotonic"])
            except BaseException as cleanup_error:
                entry["cleanup_error"] = "Ownership unresolved: " + repr(cleanup_error)
        entry.update(status="failed", finished_utc=now(), error=repr(error))
        raise
    finally:
        # Administrative closure only after expiry; no source/native queries.
        write(HERE / "status.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Launch preparation only after coordinator review")
    parser.add_argument("--child-stage", choices=("generate", "build"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(Path.cwd() == ROOT, "Run from the repository root")
    value = plan()
    if args.child_stage:
        require(args.execute, "Child stage requires explicit execution")
        child_stage(args.child_stage, value)
        return
    if not args.execute:
        source_guard(value)
        print(json.dumps({"status": "dry-run-no-execution", "case": value["case"], "stages": schedule(value),
            "work": str(WORK), "report": str(REPORT), "effective_seed": value["effective_seed"],
            "points": 100, "energy_gev": 1000., "cpu": 12, "jobs": 1, "nonexclusive_affinity": True,
            "cleanup_reserve_seconds": CLEANUP_SECONDS, "native_queries_performed": False,
            "amplitude_evaluations": 0, "performance_acceptance_eligible": False}, indent=2))
        return
    require(12 in os.sched_getaffinity(0), "Requested nonexclusive CPU12 is unavailable")
    require(not any(p.exists() for p in (WORK, REPORT, HERE / "specification.json", HERE / "status.json",
                                       HERE / "prepared.json", HERE / "stages")), "Append-only destination already exists")
    initial = source_guard(value, native=True)
    support = load_support(value)
    bench = fresh_benchmark(value)
    state = {"status": "running", "started_utc": now(), "supervisor_pid": os.getpid(), "stages": [],
             "controller_sha256": sha(HERE / "run.py"), "case_plan_sha256": sha(PLAN),
             "amplitude_evaluations": 0, "performance_acceptance_eligible": False}
    write(HERE / "specification.json", {**state, "stages": schedule(value), "initial": initial,
        "configuration": custom_config(value), "nonexclusive_affinity": True,
        "deadline_scope": "Per stage, including checks and 60-second cleanup reserve; observed descendant guarantee"}, True)
    write(REPORT / "cases.json", custom_config(value), True)
    write(HERE / "status.json", state, True)
    try:
        with support[0].SignalRequests() as requests:
            for item in schedule(value):
                run_stage(item, value, initial, bench.environment(), requests, state, support)
        state.update(status="prepared", finished_utc=now(), preparation=read(HERE / "prepared.json"))
    except BaseException as error:
        state.update(status="failed", finished_utc=now(), error=repr(error))
        raise
    finally:
        write(HERE / "status.json", state)


if __name__ == "__main__":
    main()
