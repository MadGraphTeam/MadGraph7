#!/usr/bin/env python3
"""Isolated generated-output experiment; never modifies baseline or templates."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = ROOT / ".benchmark-work/experiments/ninja-scratch"
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import MadLoopParam


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence: {old[:65]!r}")
    return text.replace(old, new, 1)


def optimize_generated(text):
    """Change only generated DP/QP Ninja coefficient preparation."""
    starts = list(re.finditer(r"^      SUBROUTINE (\w*(?:MP_)?NINJA_LOOP)\(", text, re.M))
    # Both names contain NINJA_LOOP; stop each at the next generated subroutine.
    for match in reversed(starts):
        start = match.start()
        next_routine = re.search(r"^      SUBROUTINE ", text[match.end():], re.M)
        stop = match.end() + next_routine.start() if next_routine else len(text)
        body = text[start:stop]
        prefix = re.search(r"SUBROUTINE (\w*?)(?:MP_)?NINJA_LOOP", body).group(1)
        # MadLoop's QP entry is PREFIX_MP_NINJA_LOOP.
        prefix = prefix.removesuffix("MP_")
        body = replace_once(body, "      USE MNINJA\n",
                            "      USE MNINJA\n"
                            f"      USE {prefix}POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,\n"
                            "     $ COEFTORANK_MAP\n")
        body = replace_once(
            body,
            "      CURR_MAXCOEF = 0\n      DO I=0,RANK\n"
            "        CURR_MAXCOEF=CURR_MAXCOEF+(3+I)*(2+I)*(1+I)/6\n      ENDDO",
            "      CURR_MAXCOEF = NCOEF_R(RANK)")
        if "MP_NINJA_LOOP" not in match.group(1):
            body = replace_once(body,
                "      COMPLEX*16, ALLOCATABLE :: TENSORCOEFS(:)",
                "      COMPLEX*16 TENSORCOEFS(0:LOOPMAXCOEFS-1)")
            body = replace_once(body,
                "C     It should never be allocated at this stage\n"
                "      IF (.NOT. ALLOCATED(TENSORCOEFS)) THEN\n"
                "        ALLOCATE(TENSORCOEFS(0:CURR_MAXCOEF-1))\n      ENDIF\n", "")
            body = replace_once(body,
                "        TENSORCOEFS(I) = LOOPCOEFS(I,SQSOINDEX,ID)\n",
                "        TENSORCOEFS(I) = LOOPCOEFS(I,SQSOINDEX,ID)\n"
                "        IF (MOD(COEFTORANK_MAP(I),2).EQ.1)\n"
                "     $    TENSORCOEFS(I)=-TENSORCOEFS(I)\n")
            body = replace_once(body,
                f"      CALL {prefix}INVERT_MOMENTA_IN_POLYNOMIAL(CURR_MAXCOEF,TENSORCOEFS)\n", "")
            body = replace_once(body,
                "C     Make sure to deallocate the tensor of coefficients\n"
                "      IF (ALLOCATED(TENSORCOEFS)) THEN\n"
                "        DEALLOCATE(TENSORCOEFS)\n      ENDIF\n", "")
        else:
            body = replace_once(body,
                "      COMPLEX*32, ALLOCATABLE :: MP_TENSORCOEFS(:)\n"
                "      COMPLEX(KI_QNIN), ALLOCATABLE :: MP_NINJA_TENSORCOEFS(:)",
                "      COMPLEX*32 MP_COEFFICIENT\n"
                "      COMPLEX(KI_QNIN) MP_NINJA_TENSORCOEFS(0:LOOPMAXCOEFS-1)")
            body = replace_once(body,
                "C     It should never be allocated at this stage\n"
                "      IF (.NOT. ALLOCATED(MP_TENSORCOEFS)) THEN\n"
                "        ALLOCATE(MP_TENSORCOEFS(0:CURR_MAXCOEF-1))\n      ENDIF\n"
                "      IF (.NOT. ALLOCATED(MP_NINJA_TENSORCOEFS)) THEN\n"
                "        ALLOCATE(MP_NINJA_TENSORCOEFS(0:CURR_MAXCOEF-1))\n      ENDIF\n", "")
            body = replace_once(body,
                "        MP_TENSORCOEFS(I) = MP_LOOPCOEFS(I,SQSOINDEX,ID)\n",
                "        MP_COEFFICIENT = MP_LOOPCOEFS(I,SQSOINDEX,ID)\n"
                "        IF (MOD(COEFTORANK_MAP(I),2).EQ.1)\n"
                "     $    MP_COEFFICIENT=-MP_COEFFICIENT\n"
                "        MP_NINJA_TENSORCOEFS(I)=CMPLX(MP_COEFFICIENT,\n"
                "     $    KIND=KI_QNIN)\n")
            body = replace_once(body,
                f"      CALL MP_{prefix}INVERT_MOMENTA_IN_POLYNOMIAL(CURR_MAXCOEF\n"
                "     $ ,MP_TENSORCOEFS)\n", "")
            body = replace_once(body,
                "      DO I=0,CURR_MAXCOEF-1\n"
                "        MP_NINJA_TENSORCOEFS(I)=CMPLX(MP_TENSORCOEFS(I),KIND=KI_QNIN)\n"
                "      ENDDO\n", "")
            body = replace_once(body,
                "C     Make sure to deallocate the tensor of coefficients\n"
                "      IF (ALLOCATED(MP_TENSORCOEFS)) THEN\n"
                "        DEALLOCATE(MP_TENSORCOEFS)\n      ENDIF\n"
                "      IF (ALLOCATED(MP_NINJA_TENSORCOEFS)) THEN\n"
                "        DEALLOCATE(MP_NINJA_TENSORCOEFS)\n      ENDIF\n", "")
        text = text[:start] + body + text[stop:]
    if len(starts) != 2:
        raise ValueError(f"Expected both Ninja wrappers, got {len(starts)}")
    return text


def command(args, cwd, log):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(ROOT / ".benchmark-work/original/HEPTools/collier"),
        str(ROOT / ".benchmark-work/original/HEPTools/ninja/lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ])
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    with log.open("w") as stream:
        subprocess.run(args, cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def configure(output, mode):
    card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
    card["MLReductionLib"] = "6"
    card["CTModeRun"] = -1 if mode == "dp" else 4
    card["CTModeInit"] = 1
    card["UseLoopFilter"] = False
    template = ROOT / ".benchmark-work/original/Template/loop_material/StandAlone/Cards/MadLoopParams.dat"
    card.write(str(output / "Cards/MadLoopParams.dat"), template=str(template))
    for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
        for path in (output / "SubProcesses").rglob(pattern):
            path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="qcd-1,qcd-2")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--loops", type=int, default=0, help="zero calibrates a half-second interval")
    args = parser.parse_args()
    for case in args.cases.split(","):
        source = ROOT / ".benchmark-work/outputs/baseline" / case
        points = ROOT / "docs/benchmarkfolder/points" / f"{case}.dat"
        dirs = {}
        for variant in ("original", "scratch"):
            output = HERE / case / variant
            if not output.exists():
                shutil.copytree(source, output, symlinks=True)
            directory = next(output.glob("SubProcesses/P*_*"))
            dirs[variant] = output, directory
            if variant == "scratch":
                path = directory / "CT_interface.f"
                original = next(source.glob("SubProcesses/P*_*")) / "CT_interface.f"
                text = original.read_text()
                optimized = optimize_generated(text)
                path.write_text(optimized)
                (HERE / case / "generated.patch").write_text("".join(difflib.unified_diff(
                    text.splitlines(True), optimized.splitlines(True),
                    fromfile="original/CT_interface.f", tofile="scratch/CT_interface.f")))
        evidence = {"case": case, "cpu": args.cpu,
                    "exploratory_only": True, "concurrent_builds": True,
                    "points_sha256": hashlib.sha256(points.read_bytes()).hexdigest(),
                    "measurements": {}}
        for mode in ("dp", "qp"):
            runs = {"original": [], "scratch": []}
            for variant, (output, directory) in dirs.items():
                driver = next(source.glob("SubProcesses/P*_*")) / "check_sa.f"
                text = driver.read_text()
                if mode == "qp":
                    text = text.replace("FORCE_STABILITY_CHECK(.TRUE.)",
                                        "FORCE_STABILITY_CHECK(.FALSE.)")
                (directory / "check_sa.f").write_text(text)
                command(["make", f"SHELL={shutil.which('bash')}", "-j1", "check"], directory,
                        HERE / case / f"build-{variant}-{mode}.log")
            expected = None
            repetitions = args.repetitions if mode == "dp" else 1
            loops = args.loops if mode == "dp" else 1
            if loops == 0:
                output, directory = dirs["original"]
                configure(output, mode)
                (directory / "timing.dat").unlink(missing_ok=True)
                command(["taskset", "-c", str(args.cpu), "./check", str(points), "1"],
                        directory, HERE / case / f"calibration-{mode}.log")
                elapsed = float((directory / "timing.dat").read_text().split()[1])
                loops = max(1, round(0.5 / elapsed))
            for rep in range(repetitions):
                for variant in (("original", "scratch") if rep % 2 == 0 else ("scratch", "original")):
                    output, directory = dirs[variant]
                    configure(output, mode)
                    for filename in ("timing.dat", "values.dat", "timed_status.dat"):
                        (directory / filename).unlink(missing_ok=True)
                    command(["taskset", "-c", str(args.cpu), "./check", str(points), str(loops)],
                            directory, HERE / case / f"run-{variant}-{mode}-{rep}.log")
                    cold, elapsed = map(float, (directory / "timing.dat").read_text().split())
                    values = (directory / "values.dat").read_bytes()
                    counts = dict(map(int, line.split()) for line in
                                  (directory / "timed_status.dat").read_text().splitlines())
                    required = 216 if mode == "dp" else 120
                    if set(counts) != {required} or sum(counts.values()) != 100 * loops:
                        raise RuntimeError(f"Unexpected {mode} statuses: {counts}")
                    if expected is None:
                        expected = values
                    if values != expected:
                        raise RuntimeError(f"Numerical coefficients/status changed: {variant}/{mode}/{rep}")
                    (HERE / case / f"values-{variant}-{mode}-{rep}.dat").write_bytes(values)
                    sample = {"seconds_per_point": elapsed / (100 * loops),
                              "cold_seconds": cold, "loops": loops, "statuses": counts}
                    runs[variant].append(sample)
                    print(case, mode, variant, rep, sample["seconds_per_point"], flush=True)
            medians = {v: statistics.median(r["seconds_per_point"] for r in rows)
                       for v, rows in runs.items()}
            evidence["measurements"][mode] = {
                "runs": runs, "median_seconds_per_point": medians,
                "scratch_over_original": medians["scratch"] / medians["original"],
                "all_100_points_bitwise_identical_in_printed_output": True,
            }
        (HERE / case / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print("Finished", case, json.dumps(evidence["measurements"]["dp"]["median_seconds_per_point"]),
              flush=True)


if __name__ == "__main__":
    main()
