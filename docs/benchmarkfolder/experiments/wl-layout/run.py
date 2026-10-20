#!/usr/bin/env python3
"""Compare an order-preserving compact UPDATE_WL loop interchange in copied output."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = ROOT / ".benchmark-work/experiments/wl-layout"
HERE.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "scratch_experiment", ROOT / "docs/benchmarkfolder/experiments/ninja-scratch/run.py")
scratch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scratch)


def interchange(source):
    pattern = (
        r"            DO L=0,(\d+)\n"
        r"              NEW_POSITION = COMB_COEF_POS\(L,M\)\n"
        r"              DO I=1,LCUT_SIZE\n"
        r"(                OUT.*\n     \$.*\n)"
        r"              ENDDO\n"
        r"            ENDDO"
    )
    def replacement(match):
        return ("            DO I=1,LCUT_SIZE\n"
                f"              DO L=0,{match[1]}\n"
                "                NEW_POSITION = COMB_COEF_POS(L,M)\n" +
                match[2] + "              ENDDO\n            ENDDO")
    result, count = re.subn(pattern, replacement, source)
    if not count:
        raise ValueError("No compact wavefunction updates found")
    return result, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="qcd-2")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    for case in args.cases.split(","):
        source = ROOT / ".benchmark-work/outputs/baseline" / case
        points = ROOT / "docs/benchmarkfolder/points" / f"{case}.dat"
        dirs = {}
        for variant in ("original", "interchange"):
            output = HERE / case / variant
            if not output.exists():
                shutil.copytree(source, output, symlinks=True)
            directory = next(output.glob("SubProcesses/P*_*"))
            dirs[variant] = output, directory
            if variant == "interchange":
                path = directory / "polynomial.f"
                text = (next(source.glob("SubProcesses/P*_*")) / "polynomial.f").read_text()
                transformed, count = interchange(text)
                path.write_text(transformed)
                (HERE / case / "generated.patch").write_text("".join(difflib.unified_diff(
                    text.splitlines(True), transformed.splitlines(True),
                    fromfile="original/polynomial.f", tofile="interchange/polynomial.f")))
                print(case, "modified", count, "compact updates", flush=True)
            scratch.command(["make", f"SHELL={shutil.which('bash')}", "-j4", "check"],
                            directory, HERE / case / f"build-{variant}.log")
        output, directory = dirs["original"]
        scratch.configure(output, "dp")
        (directory / "timing.dat").unlink(missing_ok=True)
        scratch.command(["taskset", "-c", str(args.cpu), "./check", str(points), "1"],
                        directory, HERE / case / "calibration.log")
        elapsed = float((directory / "timing.dat").read_text().split()[1])
        loops = max(1, round(0.5 / elapsed))
        runs = {v: [] for v in dirs}
        expected = None
        for rep in range(args.repetitions):
            for variant in (tuple(dirs) if rep % 2 == 0 else tuple(reversed(dirs))):
                output, directory = dirs[variant]
                scratch.configure(output, "dp")
                for filename in ("timing.dat", "values.dat", "timed_status.dat"):
                    (directory / filename).unlink(missing_ok=True)
                scratch.command(["taskset", "-c", str(args.cpu), "./check", str(points), str(loops)],
                                directory, HERE / case / f"run-{variant}-{rep}.log")
                cold, elapsed = map(float, (directory / "timing.dat").read_text().split())
                values = (directory / "values.dat").read_bytes()
                counts = dict(map(int, line.split()) for line in
                              (directory / "timed_status.dat").read_text().splitlines())
                if set(counts) != {216} or sum(counts.values()) != 100 * loops:
                    raise ValueError(f"Unexpected status counts: {counts}")
                if expected is None:
                    expected = values
                if values != expected:
                    (HERE / case / f"mismatch-{variant}-{rep}.dat").write_bytes(values)
                    raise ValueError("Coefficients, accuracies or statuses changed")
                sample = {"seconds_per_point": elapsed / (100 * loops),
                          "cold_seconds": cold, "loops": loops, "status_counts": counts}
                runs[variant].append(sample)
                print(case, variant, rep, sample["seconds_per_point"], flush=True)
        medians = {v: statistics.median(r["seconds_per_point"] for r in samples)
                   for v, samples in runs.items()}
        evidence = {
            "case": case, "cpu": args.cpu, "exploratory_only": True,
            "concurrent_builds": True, "measurements": runs,
            "median_seconds_per_point": medians,
            "interchange_over_original": medians["interchange"] / medians["original"],
            "all_100_points_bitwise_identical_in_printed_output": True,
            "points_sha256": hashlib.sha256(points.read_bytes()).hexdigest(),
        }
        (HERE / case / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({k: v for k, v in evidence.items() if k != "measurements"}), flush=True)


if __name__ == "__main__":
    main()
