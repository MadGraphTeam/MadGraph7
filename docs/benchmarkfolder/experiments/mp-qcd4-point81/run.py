#!/usr/bin/env python3
"""Profile a warm, isolated full-QP evaluation of retained qcd-4 point 81."""
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
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--perf", default=shutil.which("perf") or
                        "/nix/store/7ccpnz8xkn5qsyw8nkz998vjmb6jpl49-perf-linux-6.19.6/bin/perf")
    args = parser.parse_args()
    source = "candidate" if args.backend == "ninjars" else "baseline"
    backend = 8 if args.backend == "ninjars" else 6
    name = "mp-qcd4-point81" + ("" if backend == 8 else "-ninja")
    parent = bench.WORK / "outputs" / source / "qcd-4"
    output = bench.WORK / "experiments" / name
    report = bench.REPORT / "experiments" / name
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment: {output}")
    shutil.copytree(parent, output, symlinks=True)
    report.mkdir(parents=True, exist_ok=True)
    directory = next(output.glob("SubProcesses/P*_*"))
    original_directory = next(parent.glob("SubProcesses/P*_*"))
    card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
    card["MLReductionLib"] = str(backend)
    card["CTModeRun"] = 4
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
    needle = "      OPEN(6,FILE='/dev/null',STATUS='OLD')"
    assert text.count(needle) == 1
    driver.write_text(text.replace(needle,
        "      OPEN(74,FILE='profile-ready.dat',STATUS='REPLACE')\n"
        "      WRITE(74,*) 'warm single-point QP profile'\n"
        "      CLOSE(74)\n" + needle))
    all_points = bench.REPORT / "points/qcd-4.dat"
    lines = all_points.read_text().splitlines()
    point = output / "point81.dat"
    point.write_text("1\n" + "\n".join(lines[1 + 80*6:1 + 81*6]) + "\n")
    bench.run(["make", "-j2", "check"], directory,
              bench.WORK / "logs" / f"compile-{name}.log")
    for path in (point, output / "Cards/MadLoopParams.dat",
                 output / "Cards/param_card.dat", driver):
        shutil.copy2(path, report / path.name)
    bench.dump(report / "generated-sources.json", bench.generated_source_manifest(output))

    env = os.environ.copy()
    env.setdefault("SYMBOLICA_LOCK_DIR", str(ROOT / ".ninjars-build/symbolica-locks"))
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(bench.BASELINE / "HEPTools/collier/lib"),
        str(bench.BASELINE / "HEPTools/ninja/lib"), env.get("LD_LIBRARY_PATH", "")])
    command = ["taskset", "-c", str(args.cpu), "./check", str(point), str(args.loops)]
    ready = directory / "profile-ready.dat"
    ready.unlink(missing_ok=True)
    start = time.monotonic()
    with (report / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        while not ready.exists():
            if process.poll() is not None:
                raise RuntimeError("Check ended before the warm profiling marker")
            time.sleep(.1)
        warmup_seconds = time.monotonic() - start
        data = bench.WORK / "logs" / f"{name}-full.perf"
        perf_command = [
            args.perf, "record", "-e", "cpu-clock:u", "-F", "49",
            "--call-graph", "dwarf,65528", "-p", str(process.pid), "-o", str(data)]
        with (report / "perf-full-record.log").open("w") as perf_log:
            profiler = subprocess.Popen(perf_command, cwd=ROOT, env=env,
                                        stdout=perf_log, stderr=subprocess.STDOUT)
            code = process.wait()
            profiler.send_signal(signal.SIGINT)
            perf_code = profiler.wait()
    for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
        shutil.copy2(directory / name, report / name)
    bench.dump(report / "provenance.json", {
        "parent_output": str(parent.relative_to(ROOT)), "point_index": 81,
        "full_point_stream_sha256": digest(all_points),
        "point_sha256": digest(point),
        "parent_executable_sha256": digest(original_directory / "check"),
        "profile_executable_sha256": digest(directory / "check"),
        "precision_bits": 256 if backend == 8 else 113,
        "CTModeRun": 4, "backend": backend, "cpu": args.cpu, "loops": args.loops,
        "command": command, "perf_command": perf_command,
        "perf_data": str(data.relative_to(ROOT)),
        "warmup_seconds": warmup_seconds, "process_seconds": time.monotonic() - start,
        "exit_code": code, "perf_exit_code": perf_code,
        "note": "Isolated output; whole warm QP phase. Diagnostic, not a performance gate."})
    if code or perf_code:
        raise RuntimeError(f"Profile failed: check={code}, perf={perf_code}")


if __name__ == "__main__":
    main()
