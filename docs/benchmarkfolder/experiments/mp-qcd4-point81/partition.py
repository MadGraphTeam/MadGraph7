#!/usr/bin/env python3
"""Summarize actual owner/method stack frames, avoiding generic type arguments."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
PERF = "/nix/store/7ccpnz8xkn5qsyw8nkz998vjmb6jpl49-perf-linux-6.19.6/bin/perf"


def category(lines):
    stack = "\n".join(lines)
    if "mp_compute_loop_coefs_" in stack or "mp_coef_construction_" in stack:
        return "fortran_qp_numerator"
    if any(re.match(r"\s*[0-9a-f]+ <ninjars_symbolica::masters::SymbolicaMasters", x)
           for x in lines):
        return "scalar_masters"
    if any(re.match(r"\s*[0-9a-f]+ <madloop_ninjars::tensor::TensorGenericNumerator", x)
           for x in lines):
        return "tensor_numerator"
    if any(re.match(r"\s*[0-9a-f]+ <ninjars_symbolica::cuts::", x) for x in lines):
        return "generic_cut_geometry"
    if "avh_olo" in stack:
        return "scalar_masters"
    if "quadninja" in stack:
        return "ninja_qp_reduction"
    if lines[0].startswith("madloop-ninjars"):
        return "other_mp_worker"
    return "other_main"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    args = parser.parse_args()
    report = ROOT / "docs/benchmarkfolder/experiments" / args.experiment
    data = ROOT / ".benchmark-work/logs" / f"{args.experiment}.perf"
    command = [PERF, "script", "--no-inline", "-i", str(data)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    counts = Counter()
    phases = {}
    durations = []
    if (report / "per-call.dat").exists():
        durations = [float(columns[-1]) for line in (report / "per-call.dat").read_text().splitlines()
                     if (columns := line.split())[1] == "0"]
    first_sample = None
    examples = {}
    block = []

    def consume(lines):
        nonlocal first_sample
        if not lines:
            return
        key = category(lines)
        counts[key] += 1
        examples.setdefault(key, lines.copy())
        if durations:
            stamp = float(re.search(r"\s(\d+\.\d+):", lines[0]).group(1))
            if first_sample is None:
                first_sample = stamp
            elapsed = stamp - first_sample
            phase = 1
            for duration in durations[:-1]:
                if elapsed < duration:
                    break
                elapsed -= duration
                phase += 1
            phases.setdefault(phase, Counter())[key] += 1

    for line in process.stdout:
        if line.strip():
            block.append(line.rstrip())
        else:
            consume(block)
            block.clear()
    consume(block)
    code = process.wait()
    if code:
        raise RuntimeError(f"perf script failed: {code}")
    total = sum(counts.values())
    result = {
        "command": command, "samples": total,
        "categories": {key: {"samples": count, "percent": 100*count/total}
                       for key, count in counts.items()},
        "classification": "Mutually exclusive first matching owner/method stack: Fortran "
                          "QP coefficient construction, SymbolicaMasters, tensor numerator, "
                          "generic cut geometry, avh_olo, quadninja, other worker/main. "
                          "Master and geometry categories include first-use evaluator setup."}
    if phases:
        result["approximate_call_partitions"] = {
            phase: {"samples": sum(values.values()), "categories": {
                key: {"samples": count, "percent": 100*count/sum(values.values())}
                for key, count in values.items()}}
            for phase, values in phases.items()}
        result["phase_boundary_note"] = (
            "Per-call durations anchor boundaries to the first sampled event. The initial "
            "perf attach delay (normally <0.1s) makes these partitions approximate; complete "
            "call timings/statuses/coefficients come from the exact Fortran counters.")
    (report / "perf-partition.json").write_text(json.dumps(result, indent=2) + "\n")
    (report / "perf-exemplars.txt").write_text("\n\n".join(
        key + "\n" + "\n".join(lines) for key, lines in examples.items()) + "\n")
    with (report / "perf-flat.txt").open("w") as stream:
        subprocess.run([PERF, "report", "-i", str(data), "--stdio", "--no-children",
                        "--no-inline", "--call-graph", "none", "--percent-limit", "0.5"],
                       stdout=stream, check=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
