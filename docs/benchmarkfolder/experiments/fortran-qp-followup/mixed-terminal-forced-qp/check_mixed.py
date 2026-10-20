#!/usr/bin/env python3
"""Untimed full-process mixed-order diagnostic with forced original-Ninja QP."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/"tools"), str(HERE)]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam
import terminal


def sha(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def main():
    run_id = sys.argv[1]
    os.sched_setaffinity(0, {3})
    work = ROOT/".benchmark-work/policy-v2/generation/experiments/fortran-qp-followup"/run_id
    report = HERE/run_id
    report.mkdir(parents=True, exist_ok=False)
    source = ROOT/".benchmark-work/policy-v2/generation/outputs/candidate/control-mixed"
    points = ROOT/"docs/benchmarkfolder/points/control-mixed.dat"
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MAKEFLAGS="SHELL="+shutil.which("bash"),
        LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE/"HEPTools"/p)
                                       for p in ("ninja/lib", "collier")))
    result = {"complete": False, "acceptance_eligible": False,
              "case": "control-mixed", "backend": 6, "CTModeRun": 4,
              "scope": "Forced-QP diagnostic: stability is intentionally unknown. "
                       "No timing claims; all physical split orders are compared.",
              "points_sha256": sha(points),
              "source_manifest": bench.generated_source_manifest(source),
              "script_sha256": sha(__file__), "terminal_transform_sha256": sha(terminal.__file__),
              "variants": {}}
    for filename in (__file__, terminal.__file__):
        shutil.copy2(filename, report/Path(filename).name)
    expected = None
    try:
        for variant in ("original", "terminal"):
            output = work/variant
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(output)], check=True)
            for path in output.rglob("*"):
                if path.is_symlink() and path.resolve().is_relative_to(source):
                    target = output/path.resolve().relative_to(source)
                    path.unlink()
                    path.symlink_to(os.path.relpath(target, path.parent))
            directory = next(output.glob("SubProcesses/P*_*"))
            card = MadLoopParam(str(output/"Cards/MadLoopParams.dat"))
            for key, value in {"MLReductionLib": "6", "CTModeRun": 4,
                               "MLStabThres": 1e-9,
                               "COLLIERRequiredAccuracy": 1e-12}.items():
                card[key] = value
            for path in [output/"Cards/MadLoopParams.dat",
                         *(output/"SubProcesses").rglob("MadLoopParams.dat")]:
                card.write(str(path), template=str(
                    ROOT/"Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
            driver = (directory/"check_sa.f").read_text()
            driver = driver.replace("      IMPLICIT NONE\n", "      IMPLICIT NONE\n"
                "      INTEGER*8 TERMINAL_FAST,TERMINAL_FALLBACK\n"
                "      COMMON/TERMINAL_TRACE/TERMINAL_FAST,TERMINAL_FALLBACK\n", 1)
            driver = driver.replace("      CALL SETPARA", "      TERMINAL_FAST=0\n"
                                   "      TERMINAL_FALLBACK=0\n      CALL SETPARA", 1)
            driver = driver.replace("      END\n", "      OPEN(74,FILE='terminal-counts.dat',STATUS='REPLACE')\n"
                "      WRITE(74,*) TERMINAL_FAST,TERMINAL_FALLBACK\n"
                "      CLOSE(74)\n      END\n")
            (directory/"check_sa.f").write_text(driver)
            if variant == "terminal":
                files, proof = terminal.transform(directory)
                bench.dump(report/"terminal-proof.json", proof)
                # Diagnostic-only counters; the full-process timing executable
                # from run.py contains no such instrumentation.
                original_polynomial = (directory/"polynomial.f").read_text()
                if not files["polynomial.f"].startswith(original_polynomial):
                    raise RuntimeError("Public polynomial routines changed")
                polynomial = files["polynomial.f"][len(original_polynomial):]
                polynomial = polynomial.replace(
                    "      LOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT\n",
                    "      LOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT\n"
                    "      INTEGER*8 TERMINAL_FAST,TERMINAL_FALLBACK\n"
                    "      COMMON/TERMINAL_TRACE/TERMINAL_FAST,TERMINAL_FALLBACK\n")
                for routine in proof["private_helpers"]:
                    call = "      CALL "+routine+"(A,LCUT_SIZE,B,IN_SIZE"
                    polynomial = polynomial.replace(call,
                        "      TERMINAL_FALLBACK=TERMINAL_FALLBACK+1\n"+call)
                polynomial = polynomial.replace(
                    "      RETURN\n      ENDIF\n      WL_SKIP=.FALSE.",
                    "      RETURN\n      ENDIF\n"
                    "      TERMINAL_FAST=TERMINAL_FAST+1\n      WL_SKIP=.FALSE.")
                polynomial = polynomial.replace(
                    "      RETURN\n      ENDIF\n      WL_AZERO",
                    "      RETURN\n      ENDIF\n"
                    "      TERMINAL_FAST=TERMINAL_FAST+1\n      WL_AZERO")
                files["polynomial.f"] = original_polynomial+polynomial
                patches = []
                for name, text in files.items():
                    path = directory/name
                    patches.extend(difflib.unified_diff(path.read_text().splitlines(True),
                        text.splitlines(True), fromfile="original/"+name, tofile="terminal/"+name))
                    path.write_text(text)
                (report/"terminal.patch").write_text("".join(patches))
            with (report/f"{variant}-build.log").open("w") as log:
                subprocess.run(["make", "-j2", "check"], cwd=directory, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
                for path in (output/"SubProcesses").rglob(pattern):
                    path.unlink()
            with (report/f"{variant}-run.log").open("w") as log:
                subprocess.run(["./check", str(points), "1"], cwd=directory, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            for name in ("values.dat", "timed_status.dat", "dimensions.dat", "terminal-counts.dat"):
                shutil.copy2(directory/name, report/(variant+"-"+name))
            values = (directory/"values.dat").read_bytes()
            statuses = (directory/"timed_status.dat").read_bytes()
            if expected is None:
                expected = values, statuses
            if (values, statuses) != expected:
                raise RuntimeError("Mixed-order coefficient/accuracy/status diagnostic differs")
            counts = list(map(int, (directory/"terminal-counts.dat").read_text().split()))
            if variant == "terminal" and counts[0] == 0:
                raise RuntimeError("Private terminal fast path was not exercised")
            result["variants"][variant] = {
                "executable_sha256": sha(directory/"check"),
                "driver_sha256": sha(directory/"check_sa.f"),
                "compiled_sources": bench.generated_source_manifest(output),
                "card_sha256": sha(output/"Cards/MadLoopParams.dat"),
                "values_sha256": sha(directory/"values.dat"),
                "statuses_sha256": sha(directory/"timed_status.dat"),
                "terminal_fast_fallback_counts": counts,
                "answer_dimensions": (directory/"dimensions.dat").read_text().strip()}
        result.update(complete=True, all_rows_and_statuses_exact=True)
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        bench.dump(report/"result.json", result)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("source_manifest", "variants")}, indent=2))


if __name__ == "__main__":
    main()
