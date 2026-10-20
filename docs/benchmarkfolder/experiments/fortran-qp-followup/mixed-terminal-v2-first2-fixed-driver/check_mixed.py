#!/usr/bin/env python3
"""Untimed full-process mixed-order diagnostic with forced original-Ninja QP."""
import difflib
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
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
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--cpu", type=int, default=3)
    args = parser.parse_args()
    run_id = args.run_id
    os.sched_setaffinity(0, {args.cpu})
    work = ROOT/".benchmark-work/policy-v2/generation/experiments/fortran-qp-followup"/run_id
    report = HERE/run_id
    report.mkdir(parents=True, exist_ok=False)
    source = ROOT/".benchmark-work/policy-v2/generation/outputs/candidate/control-mixed"
    parent_points = ROOT/"docs/benchmarkfolder/points/control-mixed.dat"
    lines = parent_points.read_text().splitlines(keepends=True)
    if not 1 <= args.points <= int(lines[0]):
        raise RuntimeError("Invalid diagnostic prefix length")
    points = report/"points.dat"
    points.write_text(str(args.points)+"\n"+"".join(lines[1:1+4*args.points]))
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
              "parent_points_sha256": sha(parent_points),
              "selected_point_indexes": list(range(1, args.points+1)), "cpu": args.cpu,
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
            polynomial_path = directory/"polynomial.f"
            before = polynomial_path.read_text()
            after = terminal.current_classifier(before)
            polynomial_path.write_text(after)
            (report/f"{variant}-classifier.patch").write_text("".join(
                difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                    fromfile="before/polynomial.f", tofile="after/polynomial.f")))
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
            forced = "FORCE_STABILITY_CHECK(.TRUE.)"
            if driver.count(forced) != 1:
                raise RuntimeError("Expected one forced-stability driver call")
            # Fixed CTModeRun=4 deliberately has unknown stability. Forcing a
            # test remaps every such return to initialization code140 forever;
            # this diagnostic must keep its unchecked-QP status explicit.
            driver = driver.replace(forced, "FORCE_STABILITY_CHECK(.FALSE.)")
            driver = driver.replace("      IMPLICIT NONE\n", "      IMPLICIT NONE\n"
                "      INTEGER*8 TERMINAL_FAST,TERMINAL_FALLBACK\n"
                "      COMMON/TERMINAL_TRACE/TERMINAL_FAST,TERMINAL_FALLBACK\n", 1)
            driver = driver.replace("      CALL SETPARA", "      TERMINAL_FAST=0\n"
                                   "      TERMINAL_FALLBACK=0\n      CALL SETPARA", 1)
            driver = driver.replace("         IF(WARM_POINTS.GE.NPOINTS) EXIT",
                "         WRITE(*,*) 'DIAGNOSTIC_WARM',K,CODE,WARM_POINTS\n"
                "         CALL FLUSH(6)\n"
                "         IF(WARM_POINTS.GE.NPOINTS) EXIT")
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
                polynomial, fallback_count = re.subn(
                    r"(?m)^( +CALL MP_\w+UPDATE_WL_\d+_\d+\(A,)",
                    r"      TERMINAL_FALLBACK=TERMINAL_FALLBACK+1\n\1", polynomial)
                polynomial, fast_count = re.subn(
                    r"( +RETURN\n      ENDIF\n)(      WL_(?:SKIP=|AZERO\())",
                    r"\1      TERMINAL_FAST=TERMINAL_FAST+1\n\2", polynomial)
                if fast_count != len(proof["private_helpers"]) or fast_count != fallback_count:
                    raise RuntimeError("Not all private helpers received both trace counters")
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
            dimensions = list(map(int, (directory/"dimensions.dat").read_text().split()))
            point_count = int(points.read_text().split()[0])
            rows = [line.split() for line in values.decode().splitlines()]
            if (dimensions[1] <= 1 or
                    [(int(row[0]), int(row[1])) for row in rows] !=
                    [(point, order) for point in range(1, point_count+1)
                     for order in range(dimensions[1]+1)]):
                raise RuntimeError("Physical mixed-order rows are incomplete")
            if any(int(row[2])//100 != 1 or int(row[2]) % 100//10 != 2
                   for row in rows):
                raise RuntimeError("Expected unchecked QP status for every diagnostic row")
            if any(not all(math.isfinite(float(x)) for x in row[3:]) for row in rows):
                raise RuntimeError("Nonfinite diagnostic output")
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
