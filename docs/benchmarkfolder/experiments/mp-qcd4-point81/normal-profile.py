#!/usr/bin/env python3
"""Profile normal point-81 rescue after DP-only initialization at point 1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ninjars", "ninja"), required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--parent", type=Path,
                        help="Explicit archived generated output, for historical reproduction")
    args = parser.parse_args()
    source = "candidate" if args.backend == "ninjars" else "baseline"
    backend = 8 if args.backend == "ninjars" else 6
    name = f"normal-qcd4-point81-{args.backend}"
    parent = ((ROOT / args.parent).resolve() if args.parent else
              bench.WORK / "outputs" / source / "qcd-4")
    output = bench.WORK / "experiments" / name
    report = bench.REPORT / "experiments" / name
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    shutil.copytree(parent, output, symlinks=True)
    report.mkdir(parents=True, exist_ok=True)
    directory = next(output.glob("SubProcesses/P*_*"))
    parent_directory = next(parent.glob("SubProcesses/P*_*"))
    card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
    card["MLReductionLib"] = str(backend)
    card["CTModeRun"] = -1
    card["MLStabThres"] = 1e-9
    card["CTModeInit"] = 1
    card["UseLoopFilter"] = False
    template_tree = ROOT if backend == 8 else bench.BASELINE
    for path in [output / "Cards/MadLoopParams.dat",
                 *(output / "SubProcesses").rglob("MadLoopParams.dat")]:
        card.write(str(path), template=str(
            template_tree / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
        for path in (output / "SubProcesses").rglob(pattern):
            path.unlink()
    driver = directory / "check_sa.f"
    text = driver.read_text()
    # Initialize at a stable point. This driver then evaluates point81 exactly
    # once, including any normal rescue and first-use MP initialization.
    text = text.replace("1+MOD(K-1,NPOINTS)", "1")
    text = text.replace("WARM_POINTS.GE.NPOINTS", "WARM_POINTS.GE.1")
    text = text.replace("WARM_POINTS.LT.NPOINTS", "WARM_POINTS.LT.1")
    begin = text.index("C     Save numerical outputs outside the timing interval.")
    end = text.index("C     Suppress any repeated Fortran diagnostics", begin)
    text = text[:begin] + text[end:]
    text = text.replace("      OPEN(6,FILE='/dev/null',STATUS='OLD')",
        "      OPEN(74,FILE='profile-ready.dat',STATUS='REPLACE')\n"
        "      WRITE(74,*) 'normal rescue, point81, after DP initialization'\n"
        "      CLOSE(74)\n"
        "      OPEN(6,FILE='/dev/null',STATUS='OLD')")
    text = text.replace("         DO K=1,NPOINTS", "         DO K=2,2")
    text = text.replace(
        "            IF(CODE.LT.0.OR.CODE.GT.499) STOP 4",
        "            VALUES(:,:,K)=ANS\n"
        "            ACCURACIES(:,K)=ACC\n"
        "            CODES(K)=CODE\n"
        "            IF(CODE.LT.0.OR.CODE.GT.499) STOP 4")
    text = text.replace("      DO K=1,NPOINTS\n         DO J=0,NSO",
                        "      DO K=2,2\n         DO J=0,NSO")
    driver.write_text(text)
    stream = bench.REPORT / "points/qcd-4.dat"
    lines = stream.read_text().splitlines()
    points = output / "points1-and81.dat"
    points.write_text("2\n" + "\n".join(
        lines[1:7] + lines[1 + 80*6:1 + 81*6]) + "\n")
    bench.run(["make", "-j2", "check"], directory,
              bench.WORK / "logs" / f"compile-{name}.log")
    for path in (points, output / "Cards/MadLoopParams.dat",
                 output / "Cards/param_card.dat", driver):
        shutil.copy2(path, report / path.name)
    bench.dump(report / "generated-sources.json", bench.generated_source_manifest(output))
    env = os.environ.copy()
    env.setdefault("SYMBOLICA_LOCK_DIR", str(ROOT / ".ninjars-build/symbolica-locks"))
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(bench.BASELINE / "HEPTools/collier/lib"),
        str(bench.BASELINE / "HEPTools/ninja/lib"), env.get("LD_LIBRARY_PATH", "")])
    command = ["taskset", "-c", str(args.cpu), "./check", str(points), "1"]
    ready = directory / "profile-ready.dat"
    ready.unlink(missing_ok=True)
    start = time.monotonic()
    with (report / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        print(f"{args.backend}: child {process.pid}", flush=True)
        while not ready.exists():
            if process.poll() is not None:
                raise RuntimeError("Check ended before the profiling marker")
            time.sleep(.05)
        warmup = time.monotonic() - start
        perf = shutil.which("perf") or (
            "/nix/store/7ccpnz8xkn5qsyw8nkz998vjmb6jpl49-perf-linux-6.19.6/bin/perf")
        data = bench.WORK / "logs" / f"{name}.perf"
        perf_command = [perf, "record", "-e", "cpu-clock:u", "-F", "49",
                        "--call-graph", "dwarf,65528",
                        "-p", str(process.pid), "-o", str(data)]
        with (report / "perf-record.log").open("w") as perf_log:
            profiler = subprocess.Popen(perf_command, cwd=ROOT, env=env,
                                        stdout=perf_log, stderr=subprocess.STDOUT)
            print(f"{args.backend}: normal rescue profile started after {warmup:.3f}s",
                  flush=True)
            code = process.wait()
            process_seconds = time.monotonic() - start
            profiler.send_signal(signal.SIGINT)
            perf_code = profiler.wait()
    for filename in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
        shutil.copy2(directory / filename, report / filename)
    bench.dump(report / "provenance.json", {
        "parent_output": str(parent.relative_to(ROOT)), "point_index": 81,
        "full_point_stream_sha256": digest(stream),
        "profile_points_sha256": digest(points),
        "parent_executable_sha256": digest(parent_directory / "check"),
        "profile_executable_sha256": digest(directory / "check"),
        "precision_bits": 256 if backend == 8 else 113,
        "CTModeRun": -1, "MLStabThres": 1e-9, "backend": backend,
        "cpu": args.cpu, "loops": 1, "command": command,
        "perf_command": perf_command, "perf_data": str(data.relative_to(ROOT)),
        "warmup_seconds": warmup, "process_seconds": process_seconds,
        "exit_code": code, "perf_exit_code": perf_code,
        "note": "One normal point81 evaluation after DP-only point1 initialization; "
                "includes first-use MP setup. Diagnostic, not a performance gate."})
    print(f"{args.backend}: complete check={code}, perf={perf_code}, "
          f"process={process_seconds:.3f}s", flush=True)
    if code or perf_code not in (0, -signal.SIGINT):
        raise RuntimeError(f"Profile failed: check={code}, perf={perf_code}")


if __name__ == "__main__":
    main()
