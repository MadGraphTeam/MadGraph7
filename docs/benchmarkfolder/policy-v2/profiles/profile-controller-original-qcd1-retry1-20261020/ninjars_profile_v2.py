#!/usr/bin/env python3
"""Sample a cloned, bound MadLoop worker; profiling is never acceptance timing."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ninjars_benchmark as bench
import ninjars_benchmark_workers as workers
import ninjars_benchmark_policy_v2 as policy

ROOT = bench.ROOT


def category(lines):
    stack = "\n".join(lines)
    owner = lambda pattern: any(re.match(r"\s*[0-9a-f]+ <"+pattern, line)
                                for line in lines)
    if "mp_compute_loop_coefs_" in stack or "mp_coef_construction_" in stack:
        return "fortran_qp_numerator"
    if "compute_loop_coefs_" in stack or "coef_construction_" in stack:
        return "fortran_dp_numerator"
    if owner(r"ninjars_symbolica::masters::SymbolicaMasters"):
        return "mp_scalar_masters"
    if owner(r"madloop_ninjars::tensor::TensorGenericNumerator"):
        return "mp_tensor"
    if owner(r"ninjars_symbolica::(?:cuts|high_cuts)::"):
        return "mp_geometry"
    if owner(r"ninjars::(?:boxes|triangles|bubbles|tadpoles|amplitude)::"):
        return "mp_reduction"
    if "madloop_ninjars::guarded::" in stack:
        return "compensated_dp_tensor"
    if owner(r"ninjars::fast::(?:tensor|tensor_jets)::"):
        return "dp_tensor"
    if any(name in stack for name in (
            "madloop_ninjars::parse_record", "madloop_ninjars::format_record",
            "madloop_ninjars::parse_records", "madloop_ninjars::pairs")):
        return "bridge_numeric_transport"
    return "other"


def partition(perf, raw, report):
    command = [perf, "script", "--no-inline", "-i", str(raw)]
    stacks = report / "perf-stacks.txt"
    with stacks.open("w") as stream, (report / "perf-script.log").open("w") as errors:
        subprocess.run(command, cwd=ROOT, stdout=stream,
                       stderr=errors, check=True)
    counts, leaves, examples = Counter(), {}, {}
    for block in stacks.read_text().split("\n\n"):
        lines = [line for line in block.splitlines()
                 if line.strip() and not line.startswith("#")]
        if not lines:
            continue
        label = category(lines)
        counts[label] += 1
        examples.setdefault(label, lines)
        leaf = re.sub(r"^\s*[0-9a-f]+ ", "", lines[1]) if len(lines) > 1 else "[no callchain]"
        leaves.setdefault(label, Counter())[leaf] += 1
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("Profiler produced no samples")
    return {
        "samples": total,
        "categories": {key: {"samples": value, "percent": 100*value/total}
                       for key, value in sorted(counts.items())},
        "classification": "First matching owner/ancestor; all unmatched work remains other. "
                          "Samples are CPU-clock observations, not exclusive stage timers.",
        "top_leaf_frames": {key: values.most_common(10) for key, values in leaves.items()},
        "examples": examples,
        "raw_perf_sha256": workers.digest(raw),
        "stacks": policy.bound_file(stacks),
    }


def run(args):
    if shutil.which(args.perf) is None:
        raise ValueError("Profiler executable is unavailable: "+args.perf)
    correction = policy.validate_correction(args.correction, live=True)
    binding = policy.checked_bindings(args.binding, correction, live=True)
    target = binding["targets"][args.target]
    policy.reducer_signature(target)
    measurement_path = policy.path(args.measurement)
    if binding["run_files_sha256"].get(str(measurement_path.relative_to(ROOT))) != workers.digest(measurement_path):
        raise RuntimeError("Measurement is not bound by this worker run")
    reference = json.loads(measurement_path.read_text())
    measurement_target = f"{reference['case']}-{reference['source']}-{reference['backend']}"
    if (args.target != measurement_target or reference["repetition"] < 0
            or reference["case"] != target["case"]
            or reference["backend"] != target["backend"]
            or reference["executable_sha256"] != target["executable_sha256"]
            or workers.digest(args.points) != reference["points_sha256"]):
        raise RuntimeError("Target, retained numerical stream or point file mismatch")
    if args.cpu not in os.sched_getaffinity(0) or args.blocks < 1 or args.loops < 1:
        raise ValueError("Invalid CPU or evaluation count")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Invalid run identifier")
    work = ROOT / ".benchmark-work/policy-v2/profiles" / args.run_id
    report = ROOT / "docs/benchmarkfolder/policy-v2/profiles" / args.run_id
    if work.exists() or report.exists():
        raise RuntimeError("Use a new profile namespace")
    original = policy.path(target["output"])
    output = work / "output"
    work.mkdir(parents=True)
    report.mkdir(parents=True)
    shutil.copy2(__file__, report / Path(__file__).name)
    subprocess.run(["cp", "-a", "--reflink=auto", str(original), str(output)], check=True)
    for link in output.rglob("*"):
        if link.is_symlink() and link.resolve().is_relative_to(original):
            replacement = output / link.resolve().relative_to(original)
            link.unlink()
            link.symlink_to(os.path.relpath(replacement, link.parent))
    directory = next(output.glob("SubProcesses/P*_*"))
    if workers.digest(directory / "check") != target["executable_sha256"]:
        raise RuntimeError("Copied executable changed")
    points = report / "points.dat"
    shutil.copy2(args.points, points)
    point_count = int(points.read_text().split()[0])
    for name, field in (("param_card.dat", "parameter_card_sha256"),
                        ("MadLoopParams.dat", "madloop_card_sha256")):
        if workers.digest(output / "Cards" / name) != reference[field]:
            raise RuntimeError("Copied physical or numerical card differs from measurement")
        shutil.copy2(output / "Cards" / name, report / name)
    effective_card = directory / "MadLoop5_resources/MadLoopParams.dat"
    if workers.digest(effective_card) != reference["madloop_card_sha256"]:
        raise RuntimeError("Effective subprocess reduction card differs from measurement")
    workers.validate_param_card(directory, (output / "Cards/param_card.dat").read_bytes())
    for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
        for path in (output / "SubProcesses").rglob(pattern):
            path.unlink()
    before = bench.generated_source_manifest(output)
    libraries = policy.library_hashes(output)
    if before != target["compiled_manifest"] or libraries != target["libraries"]:
        raise RuntimeError("Copied generated sources or libraries changed")
    command = ["taskset", "-c", str(args.cpu), "./check", str(points)]
    result = {
        "schema": "madloop-sampled-profile-v2", "complete": False,
        "acceptance_eligible": False, "target": args.target,
        "binding": policy.bound_file(args.binding),
        "measurement": policy.bound_file(measurement_path),
        "points": policy.bound_file(points),
        "parameter_card_sha256": reference["parameter_card_sha256"],
        "madloop_card_sha256": reference["madloop_card_sha256"],
        "executable_sha256": target["executable_sha256"],
        "generation_source_identity": target["generation_source_identity"],
        "reducer_runtime": target["reducer_runtime"],
        "compiled_sources_sha256": before["files_sha256"],
        "libraries": libraries, "cpu": args.cpu, "blocks": [],
        "scope": "Full fixed stream after initialization and warming; sampled execution "
                 "contains profiler overhead and is excluded from performance acceptance.",
        "command": command, "script_sha256": workers.digest(Path(__file__)),
    }
    worker = profiler = None
    try:
        env = workers.environment(work)
        worker = workers.Worker(command, directory, report / "worker.log",
                                env=env, timeout=args.timeout)
        raw = work / "profile.perf"
        perf_command = [args.perf, "record", "--no-buildid", "-e", "cpu-clock:u",
                        "-F", "99", "--call-graph", "dwarf,65528",
                        "-p", str(worker.process.pid), "-o", str(raw)]
        result["perf_command"] = perf_command
        result["perf_version"] = subprocess.check_output(
            [args.perf, "--version"], text=True).strip()
        with (report / "perf-record.log").open("w") as log:
            profiler = subprocess.Popen(perf_command, cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            time.sleep(.25)
            if profiler.poll() is not None:
                raise RuntimeError("Profiler failed to attach")
            for index in range(args.blocks):
                worker.run(args.loops)
                parsed = workers.parse_outputs(directory, target["backend"], point_count, args.loops)
                if parsed["rows"] != reference["rows"]:
                    raise RuntimeError("Profiled physical rows, accuracy or statuses changed")
                block = report / f"block-{index}"
                block.mkdir()
                for name in (*workers.OUTPUTS, "dimensions.dat"):
                    shutil.copy2(directory / name, block / name)
                result["blocks"].append(parsed)
            profiler.send_signal(signal.SIGINT)
            code = profiler.wait()
            profiler = None
            if code not in (0, -signal.SIGINT):
                raise RuntimeError(f"Profiler exited {code}")
        worker.close()
        worker = None
        result["partition"] = partition(args.perf, raw, report)
        if (bench.generated_source_manifest(output) != before
                or policy.library_hashes(output) != libraries):
            raise RuntimeError("Profiled numerical sources or libraries changed")
        policy.checked_bindings(args.binding, correction, live=True)
        result.update(complete=True, every_physical_row_exact=True,
                      original_worker_unchanged=True)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        try:
            if profiler is not None:
                if profiler.poll() is None:
                    profiler.send_signal(signal.SIGINT)
                profiler.wait()
        finally:
            try:
                if worker is not None:
                    worker.close(graceful=False)
            finally:
                bench.dump(report / "profile.json", result)
    print(json.dumps({"report": str(report.relative_to(ROOT)),
                      "samples": result["partition"]["samples"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binding", "correction", "measurement", "points"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--perf", default="perf")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=7200.)
    run(parser.parse_args())
