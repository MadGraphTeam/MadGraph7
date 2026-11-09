#!/usr/bin/env python3
"""Reproduce frozen Fortran prefix/cache microbenchmarks (no physics reducer)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
VARIANTS = ("base", "current", "early", "linear", "ordered")
FLAGS = ("-fPIC", "-ffixed-line-length-132", "-ffree-line-length-none")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.25)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--suite", choices=("focused", "all"), default="focused")
    parser.add_argument("--cpu", type=int, default=max(os.sched_getaffinity(0)))
    parser.add_argument("--compiler", default="gfortran")
    args = parser.parse_args()
    if args.seconds <= 0 or args.pairs < 2:
        parser.error("positive duration and at least two pairs required")
    work = args.build_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "sources.json").read_text())
    for name, digest in manifest["sources_sha256"].items():
        source = HERE / "sources" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen source digest mismatch: " + name)
        shutil.copy2(source, work / name)
    commands = []

    def run(command, affinity=False):
        commands.append(command)
        result = subprocess.run(
            command, cwd=work, text=True, capture_output=True,
            preexec_fn=(lambda: os.sched_setaffinity(0, {args.cpu})) if affinity else None)
        if result.returncode:
            raise RuntimeError(str(command) + "\n" + result.stdout + result.stderr)
        return result.stdout

    result = dict(
        cpu=args.cpu, platform=platform.platform(),
        cpu_model=next((line.partition(":")[2].strip()
                        for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")), "unknown"),
        compiler=run([args.compiler, "--version"]).splitlines()[0],
        compiler_path=shutil.which(args.compiler), flags=list(FLAGS),
        seed=1234567, pairs=args.pairs, target_cpu_seconds_per_batch=args.seconds,
        timing="process CPU seconds; alternating paired order; one CPU affinity",
        limitations="Microkernels and stubbed reducer only; no physical amplitude throughput claim.",
        source_manifest=manifest, equivalence={}, runs=[], compile_commands=[])
    for optimization in ("-O0", "-O3"):
        objects = []
        for name in VARIANTS:
            for source in (name + ".f", name + "_helpers.f90"):
                target = source + ".o"
                command = [args.compiler, optimization, *FLAGS, "-c", source, "-o", target]
                run(command)
                result["compile_commands"].append(command)
                objects.append(target)
        command = [args.compiler, optimization, *FLAGS, "-c", "prefix.f90", "-o", "prefix.o"]
        run(command)
        result["compile_commands"].append(command)
        command = [args.compiler, optimization, *FLAGS, "driver.f90", "prefix.o", *objects, "-o", "bench"]
        run(command)
        result["compile_commands"].append(command)
        result["equivalence"][optimization] = run(["./bench", "verify"], True).strip()
        print(optimization, result["equivalence"][optimization], file=sys.stderr, flush=True)

        def measure(n, kind, domain, miss_period, left, right):
            def timed(variant, count):
                raw = run(["./bench", kind, str(n), domain, str(miss_period), variant, str(count)], True).split()
                return float(raw[0]), raw[1]
            seconds, _ = timed(left, 2000)
            count = max(2000, min(100000000, int(2000 * args.seconds / max(seconds, 1e-6))))
            pairs = []
            for pair in range(args.pairs):
                order = (left, right) if pair % 2 == 0 else (right, left)
                samples = {variant: timed(variant, count) for variant in order}
                if samples[left][1] != samples[right][1]:
                    raise RuntimeError("Unequal timed checksums: " + str(samples))
                pairs.append([samples[left][0] / count, samples[right][0] / count])
            logs = [math.log(b / a) for a, b in pairs]
            item = dict(opt=optimization, n=n, kind=kind, domain=domain,
                        miss_period=miss_period, left=left, right=right, count=count, pairs=pairs,
                        median_old_ns=statistics.median(a for a, _ in pairs) * 1e9,
                        median_new_ns=statistics.median(b for _, b in pairs) * 1e9,
                        median_ratio=statistics.median(b / a for a, b in pairs),
                        paired_logmean=statistics.mean(logs), paired_log_stderr=statistics.stdev(logs) / math.sqrt(args.pairs))
            result["runs"].append(item)
            print(f"{optimization} {kind}/{domain} N={n} miss_period={miss_period} "
                  f"{left}->{right}: {item['median_old_ns']:.1f}->{item['median_new_ns']:.1f} ns "
                  f"ratio={item['median_ratio']:.3f}", file=sys.stderr, flush=True)

        for n in ((3, 6) if args.suite == "focused" else range(2, 7)):
            for domain in ("dp", "qp"):
                measure(n, "prefix", domain, 0, "old", "new")
            for miss_period in ((0, 1) if args.suite == "focused" else (0, 2, 1)):
                measure(n, "tir", "cache", miss_period, "current", "early")
                measure(n, "tir", "prefix_only", miss_period, "current", "ordered")
            if args.suite == "all":
                measure(n, "tir", "combined", 1, "current", "linear")
                measure(n, "tir", "baseline", 1, "base", "current")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
