#!/usr/bin/env python3
"""Count actual terminal fast/fallback calls in a separate diagnostic executable."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/"madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/"tools")]
import ninjars_benchmark as bench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original_run, run_id = sys.argv[1:3]
    os.sched_setaffinity(0, {2})
    experiment = ROOT/".benchmark-work/policy-v2/generation/experiments/fortran-qp-followup"
    source = experiment/original_run/"terminal"
    output = experiment/run_id/"terminal-counted"
    report = HERE/run_id
    report.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=False)
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(output)], check=True)
    for path in output.rglob("*"):
        if path.is_symlink() and path.resolve().is_relative_to(source):
            target = output/path.resolve().relative_to(source)
            path.unlink()
            path.symlink_to(os.path.relpath(target, path.parent))
    directory = next(output.glob("SubProcesses/P*_*"))
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MAKEFLAGS="SHELL="+shutil.which("bash"),
        LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE/"HEPTools"/p)
                                       for p in ("ninja/lib", "collier")))
    points = ROOT/"docs/benchmarkfolder/points/qcd-3.dat"
    expected = HERE/original_run/"original-0-values.dat"
    expected_status = HERE/original_run/"original-0-timed_status.dat"
    result = {"complete": False, "acceptance_eligible": False, "case": "qcd-3", "cpu": 2,
              "scope": "Integer trace counters in a separate executable. "
                       "Counts cover exactly the last 100-point traversal; "
                       "no runtime claims are made from this instrumented run.",
              "source_manifest": bench.generated_source_manifest(source),
              "reference_values_sha256": sha(expected),
              "reference_status_sha256": sha(expected_status),
              "points_sha256": sha(points), "script_sha256": sha(__file__)}
    shutil.copy2(__file__, report/Path(__file__).name)
    try:
        polynomial = (directory/"polynomial.f").read_text()
        start = re.search(r"(?m)^      SUBROUTINE MP_\w+TERMINAL_WL_", polynomial).start()
        body = polynomial[start:]
        body = body.replace("      LOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT\n",
            "      LOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT\n"
            "      INTEGER*8 TERMINAL_FAST,TERMINAL_FALLBACK\n"
            "      COMMON/TERMINAL_TRACE/TERMINAL_FAST,TERMINAL_FALLBACK\n")
        body, fallback_count = re.subn(
            r"(?m)^(      CALL MP_\w+UPDATE_WL_\d+_\d+\(A,)",
            r"      TERMINAL_FALLBACK=TERMINAL_FALLBACK+1\n\1", body)
        body, fast_count = re.subn(
            r"(      RETURN\n      ENDIF\n)(      WL_(?:SKIP=|AZERO\())",
            r"\1      TERMINAL_FAST=TERMINAL_FAST+1\n\2", body)
        if not fast_count or fast_count != fallback_count:
            raise RuntimeError("Not all private helpers received both trace counters")
        driver = (directory/"check_sa.f").read_text()
        driver = driver.replace("      IMPLICIT NONE\n", "      IMPLICIT NONE\n"
            "      INTEGER*8 TERMINAL_FAST,TERMINAL_FALLBACK\n"
            "      COMMON/TERMINAL_TRACE/TERMINAL_FAST,TERMINAL_FALLBACK\n", 1)
        driver = driver.replace("      CALL SETPARA", "      TERMINAL_FAST=0\n"
                               "      TERMINAL_FALLBACK=0\n      CALL SETPARA", 1)
        driver = driver.replace("      OPEN(6,FILE='/dev/null',STATUS='OLD')",
            "      TERMINAL_FAST=0\n      TERMINAL_FALLBACK=0\n"
            "      OPEN(6,FILE='/dev/null',STATUS='OLD')")
        driver = driver.replace("      END\n",
            "      OPEN(74,FILE='terminal-counts.dat',STATUS='REPLACE')\n"
            "      WRITE(74,*) TERMINAL_FAST,TERMINAL_FALLBACK\n"
            "      CLOSE(74)\n      END\n")
        patches = []
        for name, text in {"polynomial.f": polynomial[:start]+body, "check_sa.f": driver}.items():
            target = directory/name
            patches.extend(difflib.unified_diff(target.read_text().splitlines(True),
                text.splitlines(True), fromfile="uninstrumented/"+name, tofile="counted/"+name))
            target.write_text(text)
        (report/"counters.patch").write_text("".join(patches))
        with (report/"build.log").open("w") as log:
            subprocess.run(["make", "-j2", "check"], cwd=directory, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
            for path in (output/"SubProcesses").rglob(pattern):
                path.unlink()
        with (report/"run.log").open("w") as log:
            subprocess.run(["./check", str(points), "1"], cwd=directory, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        for name in ("values.dat", "timed_status.dat", "terminal-counts.dat", "dimensions.dat"):
            shutil.copy2(directory/name, report/name)
        if ((directory/"values.dat").read_bytes() != expected.read_bytes() or
                (directory/"timed_status.dat").read_bytes() != expected_status.read_bytes()):
            raise RuntimeError("Instrumented full-process rows/statuses differ")
        fast, fallback = map(int, (directory/"terminal-counts.dat").read_text().split())
        if fast <= 0:
            raise RuntimeError("No terminal fast path was exercised")
        result.update(complete=True, all_rows_and_statuses_exact=True,
            terminal_fast_calls=fast, terminal_fallback_calls=fallback,
            fast_fraction=fast/(fast+fallback), private_helpers_instrumented=fast_count,
            executable_sha256=sha(directory/"check"),
            compiled_manifest=bench.generated_source_manifest(output))
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        bench.dump(report/"result.json", result)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("source_manifest", "compiled_manifest")}, indent=2))


if __name__ == "__main__":
    main()
