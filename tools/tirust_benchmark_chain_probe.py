#!/usr/bin/env python3
"""Bounded, explicitly ineligible paired diagnostics on immutable old exports.

``prepare`` copies existing executables and objects without compiling or running
them. ``run`` is a separate, append-only operation under the numerical lock.
These probes guide optimization; they never enter the formal campaign gate.
"""
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import stat
import statistics
import time

import tirust_benchmark as bench
from tirust_benchmark_compiler import capture_effective_flags, validate_compiler_equality
from tirust_benchmark_worker import paired_statistics


SCHEMA = "madloop-tirust-chain-probe-v1"
CASES = {"qcd_n2_gg_ttx": "strict-policy-qcd-n2",
         "ew_n2_epem_mupmum": "strict-policy-ew-n2"}


@contextmanager
def private_workspace(path):
    previous = bench.WORK
    bench.WORK = Path(path)
    try:
        yield
    finally:
        bench.WORK = previous


def objects(output):
    return {str(path.relative_to(output)): bench.digest(path)
            for scope in ("Source", "SubProcesses")
            for path in sorted((output / scope).rglob("*"))
            if path.is_file() and path.suffix.lower() in (".o", ".mod", ".a")
            and "build-history" not in path.parts and "worker-raw" not in path.parts}


def mutable_paths_are_local(output):
    for path in [output / "Cards/MadLoopParams.dat", *(output / "SubProcesses").rglob("*")]:
        if (path.is_file() and (path.name == "MadLoopParams.dat"
                or path.name.endswith(("HelFilter.dat", "LoopFilter.dat")))
                and not path.resolve().is_relative_to(output.resolve())):
            raise ValueError("Mutable resource escapes the private probe output")


def clone_export(original, output, template):
    """Copy exact executable/object bytes, retaining immutable library paths."""
    original, output = Path(original).resolve(), Path(output).resolve()
    compilation = json.loads((original / "compilation.json").read_text())
    bench.verify_worker_template(compilation, template)
    if compilation.get("numerical_build_from_sources") is not True:
        raise ValueError("Original numerical objects lack source-build provenance")
    if bench.generated_sources(original) != compilation["sources"]:
        raise ValueError("Original generated sources changed")
    subdirs = sorted(original.glob("SubProcesses/P*_*"))
    if len(subdirs) != 1 or (subdirs[0] / "worker-protocol.fifo").exists():
        raise ValueError("Original output is active or has an ambiguous subprocess")
    if bench.digest(subdirs[0] / "check") != compilation["executable_sha256"]:
        raise ValueError("Original executable changed")
    source_objects = objects(original)

    def ignored(directory, names):
        return [name for name in names if name in ("build-history", "worker-raw")
                or (Path(directory) == original and name == "lib")
                or stat.S_ISFIFO((Path(directory) / name).lstat().st_mode)]

    shutil.copytree(original, output, symlinks=True, ignore=ignored)
    (output / "lib").symlink_to(original / "lib", target_is_directory=True)
    mutable_paths_are_local(output)
    directory = next(output.glob("SubProcesses/P*_*"))
    if (bench.generated_sources(output) != compilation["sources"]
            or objects(output) != source_objects
            or bench.digest(directory / "check") != compilation["executable_sha256"]):
        raise ValueError("Private clone differs from the original numerical build")
    dependencies = bench.dependency_inputs(directory)
    if any(compilation.get(key) != dependencies for key in ("link_inputs_before", "link_inputs_after")):
        raise ValueError("Original link-time dependencies changed")
    if not bench.same_compiler_policy(capture_effective_flags(directory, env=bench.environment()),
                                      compilation.get("effective_compiler_flags")):
        raise ValueError("Original effective compiler flags changed")
    return {"original_output": str(original), "private_output": str(output),
            "compilation_sha256": bench.digest(output / "compilation.json"),
            "generation_sha256": bench.digest(output / "generation.json"),
            "executable_sha256": compilation["executable_sha256"],
            "numerical_objects": source_objects, "generated_sources": compilation["sources"],
            "runtime": bench.runtime_identity(directory), "compiled_or_relinked": False}


def verify_prepared_clone(output, prepared):
    output = Path(output)
    directory = next(output.glob("SubProcesses/P*_*"))
    for name in ("compilation", "generation"):
        if bench.digest(output / (name + ".json")) != prepared[name + "_sha256"]:
            raise ValueError("Private build provenance changed after preparation")
    if (bench.digest(directory / "check") != prepared["executable_sha256"]
            or bench.generated_sources(output) != prepared["generated_sources"]
            or objects(output) != prepared["numerical_objects"]):
        raise ValueError("Private numerical build changed after preparation")
    runtime = bench.runtime_identity(directory)
    if any(runtime[key] != prepared["runtime"][key] for key in ("libraries", "archives", "metadata")):
        raise ValueError("Prepared native runtime changed")


def validate_shared_runtime(inputs):
    libraries = {source: {Path(name).name: checksum for name, checksum in
                          inputs[source]["runtime"]["libraries"].items()}
                 for source in ("baseline", "candidate")}
    left, right = libraries["baseline"], libraries["candidate"]
    if any(left[name] != right[name] for name in left.keys() & right.keys()):
        raise ValueError("Shared runtime library bytes differ")
    for kind in ("archives", "metadata"):
        left, right = inputs["baseline"]["runtime"][kind], inputs["candidate"]["runtime"][kind]
        if any(left[key] != right[key] for key in left.keys() & right.keys()):
            raise ValueError("Shared static dependencies differ")


def paths(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Probe labels must be simple append-only names")
    return (bench.LEGACY_REPORT / "experiments/chain-probes" / label,
            bench.LEGACY_WORK / "chain-probes" / label)


def prepare(args):
    report, work = paths(args.label)
    if report.exists() or work.exists():
        raise FileExistsError("Probe label already exists; previous attempts are immutable")
    config = bench.load_config()
    case = next(case for case in config["cases"] if case["id"] == args.case)
    if args.cpu not in bench.os.sched_getaffinity(0):
        raise ValueError("Requested CPU is outside allowed affinity")
    if (args.pairs < 7 or not all(math.isfinite(x) and x > 0 for x in
            (args.target_seconds, args.worker_timeout, args.wall_seconds))):
        raise ValueError("Need at least seven pairs and finite positive duration bounds")
    report.mkdir(parents=True)
    protocols = report / "protocol-sources"
    protocols.mkdir()
    for path in (Path(__file__), bench.ROOT / "tools/tirust_benchmark.py",
                 bench.ROOT / "tools/tirust_benchmark_worker.py", bench.ROOT / "tools/tirust_benchmark_compiler.py"):
        shutil.copy2(path, protocols / path.name)
    historical = (bench.LEGACY_REPORT / "experiments/numerical" / CASES[args.case]
                  / "tirust_benchmark_worker.f")
    shutil.copy2(historical, protocols / historical.name)
    inputs = report / "inputs"
    inputs.mkdir()
    shutil.copy2(bench.REPORT / "cases.json", inputs / "cases.json")
    for suffix in (".dat", ".json"):
        shutil.copy2(bench.REPORT / "points" / (case["id"] + suffix), inputs / ("points" + suffix))
    originals = {source: bench.output_dir(source, case) for source in ("baseline", "candidate")}
    generation = json.loads((originals["baseline"] / "generation.json").read_text())
    chain = bench.parse_chain(generation["default_chain"])
    spec = {"schema": SCHEMA, "case": case, "created_utc": bench.utc_now(),
            "label": args.label, "performance_acceptance_eligible": False,
            "purpose": "optimization scope diagnostic; no formal campaign acceptance",
            "baseline_chain": chain, "candidate_chain": [9, *chain],
            "pairs": args.pairs, "calibration_target_seconds": args.target_seconds,
            "worker_timeout_seconds": args.worker_timeout, "cooperative_wall_budget_seconds": args.wall_seconds,
            "wall_budget_scope": "checked before each worker start and observation after obtaining lock; preflight and bounded worker shutdown can extend this budget; not a hard total lifetime cap",
            "minimum_descriptive_batch_seconds": config["minimum_batch_seconds"],
            "machine": bench.machine_identity(args.cpu),
            "host_exclusivity": "Not assumed; diagnostic while other host work may be active",
            "historical_template_source": str(historical.relative_to(bench.LEGACY_REPORT)),
            "protocols": {path.name: bench.digest(path) for path in sorted(protocols.iterdir())},
            "inputs": {path.name: bench.digest(path) for path in sorted(inputs.iterdir())}}
    bench.dump(report / "specification.json", spec)
    try:
        with private_workspace(work):
            clones = {source: clone_export(original, bench.output_dir(source, case),
                        protocols / historical.name) for source, original in originals.items()}
        bench.dump(report / "preparation.json", {"status": "prepared", "clones": clones,
                    "performance_acceptance_eligible": False})
    except BaseException as error:
        bench.dump(report / "preparation-failure.json", {"error": repr(error), "utc": bench.utc_now()})
        raise
    return report


def descriptive_statistics(timings):
    stats = paired_statistics(timings["baseline"], timings["candidate"])
    # Its threshold decision belongs to the formal harness. Here retain only
    # descriptive uncertainty and absolute costs, never a performance pass.
    for key in ("passed", "acceptance_eligible", "threshold", "minimum_pairs"):
        stats.pop(key)
    stats["median_seconds_per_point"] = {source: statistics.median(values)
                                         for source, values in timings.items()}
    return stats


def run(label):
    report, work = paths(label)
    spec = json.loads((report / "specification.json").read_text())
    preparation = json.loads((report / "preparation.json").read_text())
    if spec["schema"] != SCHEMA or spec["performance_acceptance_eligible"] is not False:
        raise ValueError("This tool accepts diagnostic specifications only")
    protocols = report / "protocol-sources"
    for name, checksum in spec["protocols"].items():
        if bench.digest(protocols / name) != checksum:
            raise ValueError("Retained protocol changed after preparation")
        if name.endswith(".py") and bench.digest(bench.ROOT / "tools" / name) != checksum:
            raise ValueError("Live controller changed after preparation; retain this attempt and prepare another")
    for name, checksum in spec["inputs"].items():
        if bench.digest(report / "inputs" / name) != checksum:
            raise ValueError("Frozen input changed after preparation")
    # Exclusive creation records intent before lock acquisition or any child.
    with (report / "run-start.json").open("x") as stream:
        json.dump({"utc": bench.utc_now(), "performance_acceptance_eligible": False}, stream)
    config = json.loads((report / "inputs/cases.json").read_text())
    case = spec["case"]
    chains = {source: spec[source + "_chain"] for source in ("baseline", "candidate")}
    try:
        with private_workspace(work), ExitStack() as stack:
            lock = stack.enter_context((bench.ROOT / ".tirust-build/numerical-tests.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
            start = time.monotonic()
            deadline = start + spec["cooperative_wall_budget_seconds"]

            def remaining():
                seconds = deadline - time.monotonic()
                if seconds <= 0:
                    raise TimeoutError("Preregistered cooperative diagnostic wall budget exhausted")
                return min(spec["worker_timeout_seconds"], seconds)

            workers, settings, inputs = {}, {}, {}
            for source in chains:
                verify_prepared_clone(bench.output_dir(source, case), preparation["clones"][source])
                workers[source], settings[source] = bench.start_worker(stack, source, case, chains[source],
                    config, spec["machine"]["selected_cpu"], report, timeout=remaining(),
                    points_path=report / "inputs/points.dat",
                    diagnostic_template=protocols / "tirust_benchmark_worker.f")
                inputs[source] = json.loads((report / (source + "-inputs.json")).read_text())
            shared = settings["baseline"].keys() & settings["candidate"].keys() - {"mlreductionlib"}
            if any(settings["baseline"][key] != settings["candidate"][key] for key in shared):
                raise ValueError("Numerical settings differ between baseline and candidate")
            for key in ("param_card_sha256", "points_sha256"):
                if inputs["baseline"][key] != inputs["candidate"][key]:
                    raise ValueError("Baseline and candidate physics inputs differ")
            validate_compiler_equality(inputs["baseline"]["compilation"]["effective_compiler_flags"],
                                       inputs["candidate"]["compilation"]["effective_compiler_flags"])
            validate_shared_runtime(inputs)
            host_load, observations, loops = [], {source: [] for source in chains}, {}

            def observe(source, count, tag):
                workers[source].timeout = remaining()
                before = bench.host_load_snapshot()
                result = bench.observe(workers[source], chains[source], count,
                    report / "raw" / source / tag, count=config["points_per_case"])
                host_load.append({"source": source, "phase": tag, "before": before,
                                  "after": bench.host_load_snapshot()})
                return result

            calibration = {}
            for source in chains:
                first = observe(source, 1, "calibrate-0")
                trial_loops = max(1, math.ceil(spec["calibration_target_seconds"] / first["seconds"]))
                sustained = observe(source, trial_loops, "calibrate-1")
                loops[source] = max(1, math.ceil(trial_loops * spec["calibration_target_seconds"] / sustained["seconds"]))
                calibration[source] = sustained
            parity = [bench.compare_rows(calibration["baseline"]["rows"], calibration["candidate"]["rows"],
                                        config["relative_tolerance"])]
            bench.dump(report / "calibration.json", {"loops": loops, "comparison": parity[0]})
            if not parity[0]["passed"]:
                raise ValueError("Calibration numerical comparison failed")
            for repetition in range(spec["pairs"]):
                order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
                for source in order:
                    observations[source].append(observe(source, loops[source], f"repetition-{repetition}"))
                parity.append(bench.compare_rows(observations["baseline"][-1]["rows"],
                    observations["candidate"][-1]["rows"], config["relative_tolerance"]))
            timings = {source: [row["seconds_per_point"] for row in rows] for source, rows in observations.items()}
            result = {"schema": SCHEMA, "status": "complete", "performance_acceptance_eligible": False,
                "numerical_pass": all(row["passed"] for row in parity), "comparisons": parity,
                "statistics": descriptive_statistics(timings), "timings": timings,
                "raw_observations": observations, "host_load": host_load,
                "minimum_descriptive_batch_duration_met": all(row["seconds"] >= spec["minimum_descriptive_batch_seconds"]
                    for rows in observations.values() for row in rows),
                "actual_backend_precision_counts": {source: bench.aggregate_usage(rows) for source, rows in observations.items()},
                "usage_scope": "final accepted amplitude results; internal attempts are not counted",
                "initialization_and_warmup_seconds": {source: worker.cold_seconds for source, worker in workers.items()},
                "elapsed_seconds_before_shutdown": time.monotonic() - start}
        result["elapsed_seconds_through_shutdown"] = time.monotonic() - start
        with private_workspace(work):
            result["inputs_after"] = {source: bench.verify_observation_inputs(source, case, inputs[source]) for source in chains}
            for source in chains:
                verify_prepared_clone(bench.output_dir(source, case), preparation["clones"][source])
        bench.dump(report / "summary.json", result)
        return result
    except BaseException as error:
        bench.dump(report / "failure.json", {"status": "failed", "error": repr(error),
                    "utc": bench.utc_now(), "performance_acceptance_eligible": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("prepare")
    pre.add_argument("--case", choices=CASES, required=True)
    pre.add_argument("--label", required=True)
    pre.add_argument("--cpu", type=int, required=True)
    pre.add_argument("--pairs", type=int, default=7)
    pre.add_argument("--target-seconds", type=float, default=2.)
    pre.add_argument("--worker-timeout", type=float, default=120.)
    pre.add_argument("--wall-seconds", type=float, default=600.)
    native = commands.add_parser("run")
    native.add_argument("--label", required=True)
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    if args.command == "prepare":
        print(prepare(args))
    else:
        print(json.dumps(run(args.label)["statistics"], sort_keys=True))


if __name__ == "__main__":
    main()
