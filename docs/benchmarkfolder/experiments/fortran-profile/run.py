#!/usr/bin/env python3
"""Profile copied generated Fortran; shared reducer time is not sampled by gprof."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = ROOT / ".benchmark-work/experiments/fortran-profile"
HERE.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "scratch_experiment", ROOT / "docs/benchmarkfolder/experiments/ninja-scratch/run.py")
scratch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scratch)


def run(command, directory, log, profile=False):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(ROOT / ".benchmark-work/original/HEPTools/ninja/lib"),
        str(ROOT / ".benchmark-work/original/HEPTools/collier"),
        env.get("LD_LIBRARY_PATH", ""),
    ])
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    if profile:
        env["FFLAGS"] = "-w -fPIC -g -pg"
        env["LDFLAGS"] = "-pg"
    with log.open("w") as stream:
        subprocess.run(command, cwd=directory, env=env, check=True,
                       stdout=stream, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="qcd-2,qcd-3")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    for case in args.cases.split(","):
        source = ROOT / ".benchmark-work/outputs/baseline" / case
        output = HERE / case
        if not output.exists():
            shutil.copytree(source, output, symlinks=True)
        directory = next(output.glob("SubProcesses/P*_*"))
        for object_file in directory.glob("*.o"):
            object_file.unlink()
        run(["make", f"SHELL={shutil.which('bash')}", "-j4", "check"],
            directory, HERE / f"build-{case}.log", profile=True)
        scratch.configure(output, "dp")
        points = ROOT / "docs/benchmarkfolder/points" / f"{case}.dat"
        (directory / "timing.dat").unlink(missing_ok=True)
        run(["taskset", "-c", str(args.cpu), "./check", str(points), "1"],
            directory, HERE / f"calibration-{case}.log")
        calibration = float((directory / "timing.dat").read_text().split()[1])
        loops = max(1, round(args.seconds / calibration))
        scratch.configure(output, "dp")
        for filename in ("gmon.out", "timing.dat"):
            (directory / filename).unlink(missing_ok=True)
        run(["taskset", "-c", str(args.cpu), "./check", str(points), str(loops)],
            directory, HERE / f"run-{case}.log")
        run(["gprof", "-b", "check", "gmon.out"], directory, HERE / f"{case}-functions.txt")
        run(["gprof", "-b", "-l", "check", "gmon.out"], directory, HERE / f"{case}-lines.txt")
        cold, seconds = map(float, (directory / "timing.dat").read_text().split())
        record = {
            "case": case, "cpu": args.cpu, "loops": loops,
            "initialization_seconds": cold, "timed_seconds": seconds,
            "exploratory_only": True, "concurrent_builds": True,
            "flags": "-w -fPIC -g -pg; existing polynomial routines retain -O3",
            "scope": "gprof samples generated executable Fortran; dynamic reducer/library samples excluded",
            "executable_sha256": hashlib.sha256((directory / "check").read_bytes()).hexdigest(),
            "status_histogram": (directory / "timed_status.dat").read_text(),
        }
        (HERE / f"{case}-metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
