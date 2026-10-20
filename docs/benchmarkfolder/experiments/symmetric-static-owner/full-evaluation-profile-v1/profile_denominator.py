#!/usr/bin/env python3
"""Sample only full evaluation traversals in an unchanged-polynomial copy."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench

PERF = "/nix/store/7ccpnz8xkn5qsyw8nkz998vjmb6jpl49-perf-linux-6.19.6/bin/perf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--loops", type=int, default=5)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    report = HERE / args.tag
    report.mkdir(exist_ok=False)
    source = ROOT / ".benchmark-work/policy-v2/generation-abi3-terminal-fixed/experiments/symmetric-static-owner/qcd3-operation-trace-v1/original"
    work = ROOT / ".benchmark-work/experiments/symmetric-static-owner" / args.tag
    assert not work.exists()
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(work)], check=True)
    for path in work.rglob("*"):
        if path.is_symlink() and path.resolve().is_relative_to(source):
            target = work / path.resolve().relative_to(source)
            path.unlink()
            path.symlink_to(os.path.relpath(target, path.parent))
    directory = next(work.glob("SubProcesses/P*_*"))
    driver = directory / "check_sa.f"
    before = driver.read_text()
    marker = "      OPEN(6,FILE='/dev/null',STATUS='OLD')\n      CALL SYSTEM_CLOCK(T0)"
    assert before.count(marker) == 1
    after = before.replace(marker, """      OPEN(6,FILE='/dev/null',STATUS='OLD')
      OPEN(75,FILE='profile.ready',STATUS='REPLACE')
      WRITE(75,*) 'READY'
      CLOSE(75)
      READ(5,*) J
      CALL SYSTEM_CLOCK(T0)""", 1)
    driver.write_text(after)
    (report / "driver.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="original/check_sa.f", tofile="profile/check_sa.f")))
    shutil.copy2(driver, report / "check_sa.f")
    shutil.copy2(Path(__file__), report / Path(__file__).name)
    manifest = bench.generated_source_manifest(work)
    bench.dump(report / "generated-sources.json", manifest)
    original_manifest = bench.generated_source_manifest(source)
    changed = {key for key in manifest["files"] if
               manifest["files"][key] != original_manifest["files"][key]}
    assert changed == {str(driver.relative_to(work))}
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MAKEFLAGS="SHELL=" + shutil.which("bash"),
        LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE / "HEPTools" / p)
                                       for p in ("ninja/lib", "collier")))
    with (report / "build.log").open("w") as log:
        subprocess.run(["make", "-j2", "check"], cwd=directory, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    points = bench.REPORT / "points/qcd-3.dat"
    record = dict(complete=False, acceptance_eligible=False, cpu=args.cpu,
        loops=args.loops, point_count=100, points_sha256=sha(points),
        executable_sha256=sha(directory / "check"),
        polynomial_sha256=sha(directory / "polynomial.f"),
        original_polynomial_sha256=sha(next(source.glob("SubProcesses/P*_*/polynomial.f"))),
        generated_source_sha256=manifest["files_sha256"],
        changed_sources=sorted(changed), perf_executable_sha256=sha(Path(PERF)),
        inputs_sha256={name: sha(work / name) for name in
            ("Cards/param_card.dat", "Cards/MadLoopParams.dat", "Source/make_opts")},
        scope="Diagnostic user-space cycle samples during complete full-point "
              "timed traversals only. Initialization and untimed values finish "
              "before the driver blocks for perf attachment. The sole source "
              "change is that pre-timer stdin gate; all original DP/QP/reduction "
              "bodies remain unchanged. No performance acceptance claim.")
    assert record["polynomial_sha256"] == record["original_polynomial_sha256"]
    ready = directory / "profile.ready"
    ready.unlink(missing_ok=True)
    process = profiler = None
    try:
        with (report / "run.log").open("w") as log, (report / "perf.log").open("w") as perf_log:
            process = subprocess.Popen(["./check", str(points), str(args.loops)],
                cwd=directory, env=environment, stdin=subprocess.PIPE,
                stdout=log, stderr=subprocess.STDOUT, text=True)
            deadline = time.monotonic() + 600
            while not ready.exists():
                if process.poll() is not None:
                    raise RuntimeError("Driver exited before pre-timer gate")
                if time.monotonic() > deadline:
                    raise TimeoutError("Driver pre-timer gate")
                time.sleep(.2)
            data = work / "perf.data"
            command = [PERF, "record", "-F", "199", "-e", "cycles:u",
                       "--call-graph", "dwarf,16384", "-p", str(process.pid),
                       "-o", str(data)]
            record["perf_command"] = command
            profiler = subprocess.Popen(command, stdout=perf_log, stderr=subprocess.STDOUT)
            time.sleep(1)
            if profiler.poll() is not None:
                raise RuntimeError("perf failed before driver release")
            process.stdin.write("0\n")
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=1200)
            if process.returncode:
                raise RuntimeError("Profiled driver failed")
            if profiler.poll() is None:
                profiler.send_signal(signal.SIGINT)
            profiler.wait(timeout=30)
            if profiler.returncode:
                raise RuntimeError("perf record failed")
        for filename in ("values.dat", "timed_status.dat", "timing.dat", "dimensions.dat"):
            shutil.copy2(directory / filename, report / filename)
        reference = HERE / "qcd3-operation-trace-v1/original-0-values.dat"
        assert (report / "values.dat").read_bytes() == reference.read_bytes()
        expected_codes = {216: 98 * args.loops, 328: 2 * args.loops}
        actual_codes = dict(map(int, line.split()) for line in
                            (report / "timed_status.dat").read_text().splitlines())
        assert actual_codes == expected_codes
        for name, command in {
            "perf-flat.txt": [PERF, "report", "-i", str(data), "--stdio",
                "--no-children", "--sort", "symbol", "--percent-limit", "0.1"],
            "perf-callgraph.txt": [PERF, "report", "-i", str(data), "--stdio",
                "--children", "--sort", "symbol", "--percent-limit", "0.1"],
            "perf-script.txt": [PERF, "script", "-i", str(data)],
        }.items():
            with (report / name).open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record.update(complete=True, all_printed_outputs_exact=True,
            timed_status_counts=actual_codes, perf_data_path=str(data),
            perf_data_sha256=sha(data),
            evidence_sha256={p.name: sha(p) for p in report.iterdir() if p.is_file()})
    finally:
        for child in (process, profiler):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        (report / "profile.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Full evaluation profile complete; every printed row and status exact.")


if __name__ == "__main__":
    main()
