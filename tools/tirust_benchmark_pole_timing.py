#!/usr/bin/env python3
"""Bounded paired tensor timings against a pinned historical TIRust release.

This is a warm tensor diagnostic, never a physical-amplitude acceptance gate.
"""
import argparse
from decimal import Decimal, localcontext
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics

import tirust_benchmark as bench
from tirust_benchmark_compiler import capture_effective_flags, validate_compiler_equality

DSO_SHA256 = "49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12"
ARCHIVE_SHA256 = "04c2b1ac3b8275dd2d0060416c4147bebd5e9bd62b71509e4cb29ce3d5c47d08"


def fixtures():
    original = json.loads((bench.ROOT / "vendor/ninjars/tests/poles/tensor_cases.json").read_text())
    rank_two = json.loads((bench.REPORT / "experiments/poles-diagnostics/tensor-cases-r2.json").read_text())
    chosen = [dict(original[0], name="massive-tadpole-r6", rank=6, scope="public_fortran_bridge")]
    chosen += [dict(next(x for x in original if x["name"] == name), scope="generated_adapter")
               for name in ("massive-bubble-r4", "infrared-triangle-r3", "massive-box-r4")]
    chosen += [dict(next(x for x in rank_two if x["n"] == n), scope="generated_adapter") for n in (5, 6)]
    return [dict(fixture, ctmode=mode, pairs=7, target_seconds=1.1, max_calibrations=4, max_loops=10000000)
            for fixture in chosen for mode in (1, 4)]


def fixture_stream(fixture):
    scope = 0 if fixture["scope"] == "generated_adapter" else 1
    keys = ("n", "rank", "mu2", "ctmode", "pairs", "target_seconds", "max_calibrations", "max_loops")
    lines = [str(scope) + " " + " ".join(str(fixture[k]) for k in keys)]
    lines += [" ".join(format(float(v), ".17e") for v in row) for row in fixture["shifts"]]
    lines += [" ".join(format(float(v), ".17e") for v in row) for row in fixture["mass2"]]
    return "\n".join(lines) + "\n"


def paired_interval(ratios):
    logs = [math.log(value) for value in ratios]
    mean = statistics.mean(logs)
    # Seven predeclared pairs, df=6, conservative rounding of two-sided 95% t.
    radius = 2.447 * statistics.stdev(logs) / math.sqrt(len(logs))
    return {"geometric_mean": math.exp(mean), "confidence_interval_95":
            [math.exp(mean-radius), math.exp(mean+radius)], "ratios": ratios,
            "method": "paired log-ratio Student-t, df=6, critical value 2.447"}


def parse_results(text, fixture):
    frames, sequence, header, ended = {}, [], None, None
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if ended is not None:
            raise ValueError("Output record follows the completion footer")
        if fields[0] == "H":
            if header is not None or len(fields) != 9:
                raise ValueError("Malformed/duplicate timing header")
            header = fields[1:]
        elif fields[0] == "F":
            if len(fields) != 9:
                raise ValueError("Malformed timing frame")
            phase, repetition, finite, loops = map(int, fields[1:5])
            key = (phase, repetition, finite)
            if key in frames or phase not in (0, 1, 2) or finite not in (0, 1):
                raise ValueError("Duplicate or invalid timing frame")
            frames[key] = dict(phase=phase, repetition=repetition, finite=finite, loops=loops,
                stable=fields[5] == "T", count=int(fields[6]), wall_seconds=float(fields[7]),
                cpu_seconds=float(fields[8]), rows=[])
            sequence.append(key)
        elif fields[0] == "C":
            if len(fields) != 11:
                raise ValueError("Malformed timing coefficient")
            phase, repetition, finite, coefficient = map(int, fields[1:5])
            key = (phase, repetition, finite)
            if key not in frames or coefficient != len(frames[key]["rows"]):
                raise ValueError("Unordered tensor coefficient")
            frames[key]["rows"].append(fields[5:])
        elif fields[0] == "E":
            if ended is not None or len(fields) != 2:
                raise ValueError("Malformed completion record")
            ended = int(fields[1])
        else:
            raise ValueError("Unknown timing output record")
    scope = 0 if fixture["scope"] == "generated_adapter" else 1
    count = math.comb(fixture["rank"]+4, 4)
    if header is None or ended is None or list(map(int, header[:5])) != [scope, fixture["ctmode"], fixture["n"], fixture["rank"], count]:
        raise ValueError("Missing/mismatched timing metadata")
    if float(header[5]) != fixture["target_seconds"] or int(header[6]) != fixture["pairs"] or int(header[7]) != fixture["max_loops"]:
        raise ValueError("Timing controls changed")
    if fixture["pairs"] != 7 or not 0 <= ended <= fixture["pairs"]:
        raise ValueError("Protocol requires seven pairs and a bounded completion count")
    if (0, 1, 1) not in frames or (0, 1, 0) not in frames:
        raise ValueError("Both independent preflights are required")
    if sequence[:2] != [(0, 1, 1), (0, 1, 0)] or any(key[0] == 0 for key in sequence[2:]):
        raise ValueError("Preflight order differs from the declared protocol")
    if [key[0] for key in sequence] != sorted(key[0] for key in sequence):
        raise ValueError("Calibration and measurement phases are out of order")
    for frame in frames.values():
        if not 0 <= frame["loops"] <= fixture["max_loops"]:
            raise ValueError("Recorded call count exceeds the declared bound")
        if any(not math.isfinite(frame[k]) or frame[k] < 0 for k in ("wall_seconds", "cpu_seconds")):
            raise ValueError("Invalid frame duration")
    calibrations = {finite: [key for key in sequence if key[0] == 1 and key[2] == finite]
                    for finite in (1, 0)}
    expected_calibrations = []
    for finite, keys in calibrations.items():
        if len(keys) > fixture["max_calibrations"]:
            raise ValueError("Calibration exceeds its iteration bound")
        expected_keys = [(1, iteration, finite) for iteration in range(1, len(keys)+1)]
        if keys != expected_keys:
            raise ValueError("Calibration iteration numbers are missing or unordered")
        expected_calibrations.extend(expected_keys)
    if [key for key in sequence if key[0] == 1] != expected_calibrations:
        raise ValueError("Full and partial calibrations are out of order")
    threshold = Decimal("1e-10" if fixture["ctmode"] == 1 else "1e-24")
    failures, maximum = [], Decimal(0)
    full, partial = frames[(0, 1, 1)], frames[(0, 1, 0)]
    with localcontext() as context:
        context.prec = 80
        for key in sequence:
            frame = frames[key]
            if not frame["stable"]:
                if frame["rows"]:
                    raise ValueError("Failed reduction returned coefficients")
                continue
            if frame["count"] != count or len(frame["rows"]) != count or frame["loops"] <= 0:
                raise ValueError("Missing stable tensor coefficients or call count")
            expected = full if frame["finite"] else partial
            for row, truth in zip(frame["rows"], expected["rows"]):
                values, reference = list(map(Decimal, row)), list(map(Decimal, truth))
                if frame["finite"] == 0:
                    valid = (values[0].is_nan() and values[1] == 0) if scope == 0 else values[0] == values[1] == 0
                    if not valid:
                        failures.append("incorrect unevaluated finite placeholder")
                    values, reference = values[2:], reference[2:]
                for a, b in zip(values, reference):
                    if not a.is_finite() or not b.is_finite():
                        failures.append("nonfinite evaluated coefficient")
                        continue
                    residual = abs(a-b)/max(Decimal(1), abs(a), abs(b))
                    maximum = max(maximum, residual)
                    if residual > threshold:
                        failures.append("batch output differs from preflight")
        if full["stable"] and partial["stable"]:
            for left, right in zip(full["rows"], partial["rows"]):
                for a, b in zip(map(Decimal, left[2:]), map(Decimal, right[2:])):
                    residual = abs(a-b)/max(Decimal(1), abs(a), abs(b))
                    maximum = max(maximum, residual)
                    if residual > threshold:
                        failures.append("full/pole coefficient mismatch")
    base = dict(fixture=fixture, completed_pairs=ended, maximum_scaled_coefficient_residual=str(maximum),
        coefficient_threshold=str(threshold), correctness_failures=sorted(set(failures)),
        finite_placeholder="NaN+0i from generated adapter" if scope == 0 else "zero raw bridge placeholder; unevaluated",
        frames=[{k: v for k, v in frames[key].items() if k != "rows"} for key in sequence],
        performance_eligible=False, scope="warm tensor diagnostic; no physical-amplitude acceptance")
    refused = [key for key in sequence if not frames[key]["stable"]]
    if refused:
        return dict(base, status="refused", refusal_frames=refused, timing_qualified=False)
    if failures:
        return dict(base, status="coefficient_failure", timing_qualified=False)
    if ended != fixture["pairs"]:
        return dict(base, status="calibration_incomplete", timing_qualified=False)
    for finite, keys in calibrations.items():
        if not keys:
            raise ValueError("Measured pairs lack the required calibration")
        def calibrated(key):
            return 0.75*fixture["target_seconds"] <= frames[key]["wall_seconds"] <= 1.75*fixture["target_seconds"]
        if not calibrated(keys[-1]) or any(calibrated(key) for key in keys[:-1]):
            raise ValueError("Calibration does not end at its first accepted batch")
        if any(frames[(2, pair, finite)]["loops"] != frames[keys[-1]]["loops"] for pair in range(1, fixture["pairs"]+1)):
            raise ValueError("Timed call count differs from accepted calibration")
    expected = [(2, pair, finite) for pair in range(1, fixture["pairs"]+1)
                for finite in ((1, 0) if pair % 2 else (0, 1))]
    if [key for key in sequence if key[0] == 2] != expected:
        raise ValueError("Paired order is not the predeclared alternation")
    for finite in (1, 0):
        if len({frames[(2, pair, finite)]["loops"] for pair in range(1, fixture["pairs"]+1)}) != 1:
            raise ValueError("Loop counts changed within the paired measurements")
    timing_failures, wall_ratios, cpu_ratios = [], [], []
    for pair in range(1, fixture["pairs"]+1):
        left, right = frames[(2, pair, 1)], frames[(2, pair, 0)]
        for frame in (left, right):
            if not 0.75*fixture["target_seconds"] <= frame["wall_seconds"] <= 1.75*fixture["target_seconds"]:
                timing_failures.append("batch wall duration outside predeclared calibration interval")
            if any(not math.isfinite(frame[k]) or frame[k] <= 0 for k in ("wall_seconds", "cpu_seconds")):
                raise ValueError("Invalid measured duration")
        wall_ratios.append((right["wall_seconds"]/right["loops"])/(left["wall_seconds"]/left["loops"]))
        cpu_ratios.append((right["cpu_seconds"]/right["loops"])/(left["cpu_seconds"]/left["loops"]))
    return dict(base, status="measured" if not timing_failures else "timing_incomplete",
                timing_qualified=not timing_failures, timing_failures=sorted(set(timing_failures)),
                wall_poles_over_full=paired_interval(wall_ratios), cpu_poles_over_full=paired_interval(cpu_ratios))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cpu", type=int, default=9)
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("Run from repository root with a simple label")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU is outside allowed affinity")
    report = bench.REPORT / "experiments/poles-timing" / args.label
    work = bench.WORK / "poles-timing" / args.label
    report.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    original_prefix = bench.ROOT / "HEPTools/tirust"
    pinned = work / "pinned-tirust"
    if bench.digest(original_prefix / "lib/libmadloop_tirust.so") != DSO_SHA256:
        raise RuntimeError("Historical TIRust DSO is no longer installed; supply its preserved installation")
    shutil.copytree(original_prefix, pinned, symlinks=False)
    if (bench.digest(pinned / "lib/libmadloop_tirust.so") != DSO_SHA256
            or bench.digest(pinned / "lib/libtirust.a") != ARCHIVE_SHA256):
        raise RuntimeError("Historical DSO/archive identity differs")
    bench.TIRUST_PREFIX = pinned
    output = args.output.resolve()
    original_compilation = json.loads((output / "compilation.json").read_text())
    if original_compilation.get("numerical_build_from_sources") is not True or bench.generated_sources(output) != original_compilation["sources"]:
        raise RuntimeError("Generated numerical source provenance is missing or stale")
    copies = work / "process"
    shutil.copytree(output, copies, symlinks=True)
    for path in copies.rglob("*"):
        if path.is_symlink():
            resolved = path.resolve()
            if resolved.is_relative_to(original_prefix):
                relative = resolved.relative_to(original_prefix)
                path.unlink()
                path.symlink_to(pinned / relative)
    directories = sorted(copies.glob("SubProcesses/P*_*"))
    if len(directories) != 1:
        raise RuntimeError("Expected one generated subprocess")
    directory = directories[0]
    prefix = (directory / "proc_prefix.txt").read_text().strip()
    changes = {}
    for path in (directory / "makefile", directory.parent / "MadLoop_makefile_definitions"):
        before = path.read_text()
        after = before.replace(str(original_prefix), str(pinned))
        if after != before:
            # Do not write through a copied symlink into the original process.
            if path.is_symlink():
                path.unlink()
            path.write_text(after)
            changes[str(path.relative_to(copies))] = {"before": before, "after": after}
    protocols = [Path(__file__), bench.ROOT / "tools/tirust_benchmark_pole_timing.f",
                 bench.ROOT / "tools/tirust_benchmark.py", bench.ROOT / "tools/tirust_benchmark_compiler.py"]
    protocol = report / "protocol-sources"
    protocol.mkdir()
    for path in protocols:
        shutil.copy2(path, protocol / path.name)
    matrix = fixtures()
    bench.dump(report / "specification.json", {"fixtures": matrix, "cpu": args.cpu,
        "machine": bench.machine_identity(args.cpu), "scope": "warm tensor adapter/bridge diagnostic, not physical-amplitude acceptance",
        "performance_eligible": False, "library_sha256": DSO_SHA256,
        "pinned_installation": {str(p.relative_to(pinned)): bench.digest(p) for p in sorted(pinned.rglob("*")) if p.is_file()},
        "relocation_changes": changes, "protocols": {p.name: bench.digest(protocol / p.name) for p in protocols},
        "calibration_interval": [0.825, 1.925], "finite_work_omission_evidence": "qualification/poles-only/",
        "cache_policy": "Numerical caches disabled; immutable plans and allocated scratch remain warm."})
    (directory / "tirust_pole_timing.f").write_text((protocol / "tirust_benchmark_pole_timing.f").read_text().replace("@PREFIX@", prefix))
    (directory / "tirust_pole_timing.mk").write_text("include makefile\n"
        "tirust-pole-timing: tirust_pole_timing.o $(PROCESS) $(LIBS)\n"
        "\t$(FC) $(FFLAGS) -o $@ tirust_pole_timing.o $(PROCESS) $(LINKLIBS)\n")
    sources = bench.generated_sources(copies)
    flags = capture_effective_flags(directory, env=bench.environment())
    validate_compiler_equality(original_compilation["effective_compiler_flags"], flags)
    dependencies = bench.dependency_inputs(directory)
    bench.run(["make", "-f", "tirust_pole_timing.mk", "tirust-pole-timing"], directory, report / "build-output.txt")
    if bench.generated_sources(copies) != sources or bench.dependency_inputs(directory) != dependencies:
        raise RuntimeError("Diagnostic build changed sources or link inputs")
    runtime = bench.runtime_identity(directory, "tirust-pole-timing")
    tirust_resolved = {name: value for name, value in runtime["libraries"].items() if Path(name).name == "libmadloop_tirust.so"}
    if tirust_resolved != {str(pinned / "lib/libmadloop_tirust.so"): DSO_SHA256}:
        raise RuntimeError("Runtime does not resolve the pinned historical DSO")
    executable_sha = bench.digest(directory / "tirust-pole-timing")
    bench.dump(report / "compilation.json", {"generated_sources": sources, "original_compilation": original_compilation,
        "effective_compiler_flags": flags, "runtime": runtime, "executable_sha256": executable_sha,
        "parameter_card_sha256": bench.digest(copies / "Cards/param_card.dat")})
    results = []
    for fixture in matrix:
        label = fixture["name"] + ("-dp" if fixture["ctmode"] == 1 else "-qp")
        destination = report / label
        destination.mkdir()
        input_file = destination / "fixture.dat"
        input_file.write_text(fixture_stream(fixture))
        before, after_load = None, None
        raw = directory / "pole-timing-values.dat"
        raw.unlink(missing_ok=True)
        error = None
        try:
            with (bench.ROOT / ".tirust-build/numerical-tests.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                before = bench.host_load_snapshot()
                try:
                    bench.run(["taskset", "-c", str(args.cpu), "./tirust-pole-timing", str(input_file)],
                              directory, destination / "run-output.txt", timeout=90)
                finally:
                    after_load = bench.host_load_snapshot()
        except Exception as caught:
            error = repr(caught)
        finally:
            if raw.exists():
                shutil.copy2(raw, destination / raw.name)
            bench.dump(destination / "host-load.json", {"before": before, "after": after_load})
        if error is None:
            try:
                result = parse_results((destination / "pole-timing-values.dat").read_text(), fixture)
            except Exception as caught:
                result = {"fixture": fixture, "status": "analysis_failure", "error": repr(caught),
                          "timing_qualified": False, "performance_eligible": False}
        else:
            result = {"fixture": fixture, "status": "execution_failure", "error": error,
                      "timing_qualified": False, "performance_eligible": False}
        bench.dump(destination / "result.json", result)
        results.append(result)
        bench.dump(report / "results.json", {"results": results, "performance_eligible": False,
            "complete_requested_settings": len(results) == len(matrix)})
        print(json.dumps({"fixture": label, "status": result["status"],
            "ratio": result.get("wall_poles_over_full", {}).get("geometric_mean")}), flush=True)
    after = bench.runtime_identity(directory, "tirust-pole-timing")
    if (any(after[k] != runtime[k] for k in ("libraries", "archives", "metadata"))
            or bench.digest(directory / "tirust-pole-timing") != executable_sha
            or bench.generated_sources(copies) != sources):
        raise RuntimeError("Source, executable or library changed during measurement")
    bench.dump(report / "manifest.json", {"files": {str(p.relative_to(report)):
        {"sha256": bench.digest(p), "bytes": p.stat().st_size}
        for p in sorted(report.rglob("*")) if p.is_file() and p.name != "manifest.json"}})


if __name__ == "__main__":
    main()
