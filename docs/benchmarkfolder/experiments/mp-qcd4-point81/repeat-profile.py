#!/usr/bin/env python3
"""Compare first/subsequent normal rescues using the retained pre-optimization driver."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ninja", "ninjars"), required=True)
    args = parser.parse_args()
    name = f"repeat-qcd4-point81-{args.backend}"
    parent = bench.WORK / f"experiments/normal-qcd4-point81-{args.backend}"
    output = bench.WORK / "experiments" / name
    report = bench.REPORT / "experiments" / name
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    shutil.copytree(parent, output, symlinks=True)
    report.mkdir(parents=True, exist_ok=True)
    directory = next(output.glob("SubProcesses/P*_*"))
    # Older exports used an absolute process-output symlink, which is invalid
    # once the parent output is archived for final regeneration.
    collier = directory / "libcollier.so"
    if collier.is_symlink():
        collier.unlink()
        collier.symlink_to("../../lib/collier_lib/libcollier.so")
    driver = directory / "check_sa.f"
    text = driver.read_text()
    text = text.replace("      INTEGER*8 T0,T1,RATE", "      INTEGER*8 T0,T1,T2,T3,RATE")
    text = text.replace("      DO L=1,NLOOPS",
        "      OPEN(75,FILE='per-call.dat',STATUS='REPLACE')\n"
        "      DO L=1,NLOOPS\n"
        "         CALL SYSTEM_CLOCK(T2)")
    needle = "            TIMED_CODES(CODE)=TIMED_CODES(CODE)+1"
    assert text.count(needle) == 1
    text = text.replace(needle, needle + """
            CALL SYSTEM_CLOCK(T3)
            DO J=0,NSO
               WRITE(75,'(3I8,6ES26.17E3)') L,J,CODE,
     $              ANS(:,J),ACC(J),DBLE(T3-T2)/DBLE(RATE)
            ENDDO
            FLUSH(75)""")
    text = text.replace("      ELAPSED=DBLE(T1-T0)/DBLE(RATE)",
                        "      CLOSE(75)\n      ELAPSED=DBLE(T1-T0)/DBLE(RATE)")
    driver.write_text(text)
    if args.backend == "ninjars":
        definitions = output / "SubProcesses/MadLoop_makefile_definitions"
        text = definitions.read_text()
        assert text.count(" -lninjars ") == 1
        prefix = (ROOT / "HEPTools/ninjars/lib/ninjars-link-before.txt").read_text().strip()
        definitions.write_text(text.replace(" -lninjars ", f" {prefix} -lninjars "))
        shutil.copy2(ROOT / "HEPTools/ninjars/lib/ninjars-config.json",
                     report / "ninjars-config.json")
    bench.run(["make", "-j2", "check"], directory,
              bench.WORK / "logs" / f"compile-{name}.log")
    points = output / "points1-and81.dat"
    ready = directory / "profile-ready.dat"
    ready.unlink(missing_ok=True)
    for filename in ("timing.dat", "values.dat", "timed_status.dat", "per-call.dat"):
        (directory / filename).unlink(missing_ok=True)
    for path in (points, output / "Cards/MadLoopParams.dat",
                 output / "Cards/param_card.dat", driver):
        shutil.copy2(path, report / path.name)
    bench.dump(report / "generated-sources.json", bench.generated_source_manifest(output))
    env = os.environ.copy()
    env.setdefault("SYMBOLICA_LOCK_DIR", str(ROOT / ".ninjars-build/symbolica-locks"))
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(bench.BASELINE / "HEPTools/collier"),
        str(bench.BASELINE / "HEPTools/ninja/lib"), env.get("LD_LIBRARY_PATH", "")])
    command = ["taskset", "-c", "2", "./check", str(points), "2"]
    start = time.monotonic()
    with (report / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        print(f"{args.backend} repeat child {process.pid}", flush=True)
        while not ready.exists():
            if process.poll() is not None:
                raise RuntimeError("Check ended before marker")
            time.sleep(.05)
        warmup = time.monotonic() - start
        data = bench.WORK / "logs" / f"{name}.perf"
        perf = "/nix/store/7ccpnz8xkn5qsyw8nkz998vjmb6jpl49-perf-linux-6.19.6/bin/perf"
        perf_command = [perf, "record", "--no-buildid", "-e", "cpu-clock:u", "-F", "49",
                        "--call-graph", "dwarf,65528", "-p", str(process.pid),
                        "-o", str(data)]
        with (report / "perf-record.log").open("w") as perf_log:
            profiler = subprocess.Popen(perf_command, cwd=ROOT, env=env,
                                        stdout=perf_log, stderr=subprocess.STDOUT)
            print(f"{args.backend}: first/subsequent rescue profile started", flush=True)
            code = process.wait()
            elapsed = time.monotonic() - start
            profiler.send_signal(signal.SIGINT)
            perf_code = profiler.wait()
    for filename in ("values.dat", "timing.dat", "timed_status.dat",
                     "dimensions.dat", "per-call.dat"):
        shutil.copy2(directory / filename, report / filename)
    bench.dump(report / "provenance.json", {
        "parent_output": str(parent.relative_to(ROOT)),
        "point_index": 81, "backend": 8 if args.backend == "ninjars" else 6,
        "CTModeRun": -1, "MLStabThres": 1e-9, "loops": 2, "cpu": 2,
        "command": command, "perf_command": perf_command,
        "perf_data": str(data.relative_to(ROOT)),
        "warmup_seconds": warmup, "process_seconds": elapsed,
        "exit_code": code, "perf_exit_code": perf_code,
        "executable_sha256": hashlib.sha256((directory / "check").read_bytes()).hexdigest(),
        "points_sha256": hashlib.sha256(points.read_bytes()).hexdigest(),
        "parameter_card_sha256": hashlib.sha256(
            (output / "Cards/param_card.dat").read_bytes()).hexdigest(),
        "madloop_card_sha256": hashlib.sha256(
            (output / "Cards/MadLoopParams.dat").read_bytes()).hexdigest(),
        "note": "Pre-optimization generator retained; NinjaRS uses fixed GCC-runtime "
                "link order. Two normal same-point rescues after DP point1 initialization. "
                "Each coefficient/status/time retained in per-call.dat. Diagnostic only."})
    print(f"{args.backend}: check={code}, perf={perf_code}, wall={elapsed:.3f}s", flush=True)
    if code or perf_code not in (0, -signal.SIGINT):
        raise RuntimeError("Profile failed")


if __name__ == "__main__":
    main()
