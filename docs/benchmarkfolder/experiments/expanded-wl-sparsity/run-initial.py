#!/usr/bin/env python3
"""Exploratory full-process zero-skip trials in isolated generated copies."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(HERE)]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam
from prototype import transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="qcd-3")
    parser.add_argument("--variants", default="compact")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    work = bench.WORK / "experiments/expanded-wl-sparsity" / args.run_id
    report = HERE / args.run_id
    report.mkdir(parents=True, exist_ok=False)
    variants = ["original", *args.variants.split(",")]
    assert len(set(variants)) == len(variants)
    source = bench.WORK / "outputs/candidate" / args.case
    points = bench.REPORT / "points" / f"{args.case}.dat"
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MAKEFLAGS="SHELL=" + shutil.which("bash"),
                       LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE / "HEPTools" / p)
                                                     for p in ("ninja/lib", "collier")))
    result = {"case": args.case, "backend": 6, "cpu": args.cpu,
              "exploratory_only": True, "acceptance_eligible": False,
              "source_manifest": bench.generated_source_manifest(source),
              "points_sha256": hashlib.sha256(points.read_bytes()).hexdigest(),
              "prototype_sha256": hashlib.sha256((HERE / "prototype.py").read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "builds": {}, "measurements": {name: [] for name in variants}}
    jobs = {}
    for variant in variants:
        output = work / variant
        output.parent.mkdir(parents=True, exist_ok=True)
        assert not output.exists()
        subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(output)], check=True)
        for path in output.rglob("*"):
            if path.is_symlink() and path.resolve().is_relative_to(source):
                target = output / path.resolve().relative_to(source)
                path.unlink()
                path.symlink_to(os.path.relpath(target, path.parent))
        directory = next(output.glob("SubProcesses/P*_*"))
        jobs[variant] = output, directory
        card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
        for key, value in {"MLReductionLib": "6", "CTModeRun": -1,
                           "MLStabThres": 1e-9, "COLLIERRequiredAccuracy": 1e-10}.items():
            card[key] = value
        for path in [output / "Cards/MadLoopParams.dat",
                     *(output / "SubProcesses").rglob("MadLoopParams.dat")]:
            card.write(str(path), template=str(
                ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
        polynomial = directory / "polynomial.f"
        before = polynomial.read_text()
        after = before
        if variant != "original":
            for mode in (["expanded", "compact"] if variant == "combined" else [variant]):
                after, _ = transform(after, mode)
            polynomial.write_text(after)
            (report / (variant + ".patch")).write_text("".join(difflib.unified_diff(
                before.splitlines(True), after.splitlines(True),
                fromfile="original/polynomial.f", tofile=f"{variant}/polynomial.f")))
        with (report / f"build-{variant}.log").open("w") as log:
            subprocess.run(["make", "-j2", "check"], cwd=directory, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result["builds"][variant] = {
            "executable_sha256": hashlib.sha256((directory / "check").read_bytes()).hexdigest(),
            "polynomial_sha256": hashlib.sha256(after.encode()).hexdigest()}
    expected = expected_status = None
    try:
        for rep in range(args.repetitions):
            for variant in variants if rep % 2 == 0 else list(reversed(variants)):
                output, directory = jobs[variant]
                for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
                    for path in (output / "SubProcesses").rglob(pattern):
                        path.unlink()
                with (report / f"run-{variant}-{rep}.log").open("w") as log:
                    subprocess.run(["./check", str(points), str(args.loops)], cwd=directory,
                                   env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   check=True)
                for name in ("values.dat", "timing.dat", "timed_status.dat"):
                    shutil.copy2(directory / name, report / f"{variant}-{rep}-{name}")
                values = (directory / "values.dat").read_bytes()
                statuses = (directory / "timed_status.dat").read_bytes()
                if expected is None:
                    expected, expected_status = values, statuses
                if values != expected or statuses != expected_status:
                    raise RuntimeError("Printed coefficients, accuracy, row statuses or timed statuses differ")
                cold, elapsed = map(float, (directory / "timing.dat").read_text().split())
                result["measurements"][variant].append({
                    "repetition": rep, "loops": args.loops,
                    "seconds_per_point": elapsed/(100*args.loops),
                    "cold_initialization_and_warm_pass_seconds": cold,
                    "row_sha256": hashlib.sha256(values).hexdigest(),
                    "timed_status_sha256": hashlib.sha256(statuses).hexdigest()})
                print(variant, rep, elapsed/(100*args.loops), flush=True)
        medians = {v: statistics.median(r["seconds_per_point"]
                                       for r in result["measurements"][v]) for v in variants}
        result.update(complete=True, all_printed_rows_and_statuses_exact=True,
                      median_seconds_per_point=medians,
                      ratios={v: medians[v]/medians["original"] for v in variants[1:]})
    except Exception as error:
        result.update(complete=False, error=repr(error))
        raise
    finally:
        (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["ratios"]), flush=True)


if __name__ == "__main__":
    main()
