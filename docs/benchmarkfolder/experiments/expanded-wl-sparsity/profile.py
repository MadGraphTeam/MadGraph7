#!/usr/bin/env python3
"""Measure real UPDATE_WL input sparsity and body cost in an isolated output."""
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

ROOT = Path(__file__).resolve().parents[4]
REPORT = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/expanded-wl-sparsity"
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


MODULE = """      MODULE WL_SPARSE_PROFILE
      INTEGER*8, SAVE :: WL_CALLS(32)=0,WL_BLOCKS(32)=0
      INTEGER*8, SAVE :: WL_ZEROS(32)=0,WL_MULTS(32)=0
      INTEGER*8, SAVE :: WL_ZERO_MULTS(32)=0,WL_TICKS(32)=0
      INTEGER*8, SAVE :: WL_NONFINITE_A(32)=0
      END MODULE
"""
HELPERS = """
      SUBROUTINE WL_RESET()
      USE WL_SPARSE_PROFILE
      WL_CALLS=0
      WL_BLOCKS=0
      WL_ZEROS=0
      WL_MULTS=0
      WL_ZERO_MULTS=0
      WL_TICKS=0
      WL_NONFINITE_A=0
      END

      SUBROUTINE WL_DUMP()
      USE WL_SPARSE_PROFILE
      INTEGER I
      INTEGER*8 RATE
      CALL SYSTEM_CLOCK(COUNT_RATE=RATE)
      OPEN(89,FILE='sparsity.dat',STATUS='REPLACE')
      WRITE(89,*) RATE
      DO I=1,32
        IF(WL_CALLS(I).GT.0) WRITE(89,*) I,WL_CALLS(I),
     $    WL_BLOCKS(I),WL_ZEROS(I),WL_MULTS(I),WL_ZERO_MULTS(I),
     $    WL_NONFINITE_A(I),WL_TICKS(I)
      ENDDO
      CLOSE(89)
      END
"""


def instrument(text):
    routines = []
    pattern = r"(?m)^      SUBROUTINE (\w*UPDATE_WL_(\d+)_(\d+))\(.*?^      END\s*$"

    def replace(match):
        original = match[0]
        name, r1, r2 = match[1], int(match[2]), int(match[3])
        index = len(routines) + 1
        na, nb = math.comb(r1 + 4, 4), math.comb(r2 + 4, 4)
        routines.append({"index": index, "name": name, "left_rank": r1,
                         "right_rank": r2, "qp": name.startswith("MP_"),
                         "expanded": "UPDATER_COEF" not in original})
        text = re.sub(r"(?m)^(      USE \w+POLYNOMIAL_CONSTANTS)$",
                      r"\1\n      USE WL_SPARSE_PROFILE\n"
                      r"      USE, INTRINSIC :: IEEE_ARITHMETIC", original)
        declaration = "      INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE\n"
        assert text.count(declaration) == 1
        text = text.replace(declaration, declaration +
                            "      INTEGER WL_J,WL_K\n"
                            "      INTEGER*8 WL_START,WL_STOP\n")
        executable = ("      DO I=1,LCUT_SIZE" if "UPDATER_COEF" not in original
                      else "      OUT(:,:,:)")
        position = text.index(executable)
        counters = f"""      WL_CALLS({index})=WL_CALLS({index})+1
      WL_BLOCKS({index})=WL_BLOCKS({index})+OUT_SIZE*IN_SIZE
      WL_MULTS({index})=WL_MULTS({index})+
     $  INT(LCUT_SIZE,8)*OUT_SIZE*IN_SIZE*{na * nb}
      IF(.NOT.ALL(IEEE_IS_FINITE(REAL(A(1:IN_SIZE,0:{na-1},
     $  1:LCUT_SIZE)))) .OR.
     $   .NOT.ALL(IEEE_IS_FINITE(AIMAG(A(1:IN_SIZE,0:{na-1},
     $  1:LCUT_SIZE))))) WL_NONFINITE_A({index})=
     $  WL_NONFINITE_A({index})+1
      DO WL_K=1,IN_SIZE
        DO WL_J=1,OUT_SIZE
          IF(ALL(B(WL_J,0:{nb-1},WL_K).EQ.0)) THEN
            WL_ZEROS({index})=WL_ZEROS({index})+1
            WL_ZERO_MULTS({index})=WL_ZERO_MULTS({index})+
     $        INT(LCUT_SIZE,8)*{na * nb}
          ENDIF
        ENDDO
      ENDDO
      CALL SYSTEM_CLOCK(WL_START)
"""
        text = text[:position] + counters + text[position:]
        ending = "      END"
        position = text.rfind(ending)
        text = text[:position] + f"""      CALL SYSTEM_CLOCK(WL_STOP)
      WL_TICKS({index})=WL_TICKS({index})+WL_STOP-WL_START
""" + text[position:]
        return text

    modified, count = re.subn(pattern, replace, text, flags=re.S | re.M)
    assert count and count == len(routines)
    return MODULE + modified + HELPERS, routines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="qcd-3")
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    source = bench.WORK / "outputs/candidate" / args.case
    output = WORK / args.case / "profile"
    assert not output.exists(), "Retain prior copies; choose a new case/work directory"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(output)], check=True)
    for path in output.rglob("*"):
        if path.is_symlink() and path.resolve().is_relative_to(source):
            target = output / path.resolve().relative_to(source)
            path.unlink()
            path.symlink_to(os.path.relpath(target, path.parent))
    directory = next(output.glob("SubProcesses/P*_*"))
    polynomial = directory / "polynomial.f"
    original = polynomial.read_text()
    modified, routines = instrument(original)
    polynomial.write_text(modified)
    (REPORT / f"{args.case}-profile-polynomial.f").write_text(modified)
    driver = directory / "check_sa.f"
    original_driver = driver.read_text()
    marker = "      OPEN(6,FILE='/dev/null',STATUS='OLD')\n"
    assert original_driver.count(marker) == 1
    modified_driver = original_driver.replace(marker, marker + "      CALL WL_RESET()\n")
    assert modified_driver.count("      END\n") == 1
    modified_driver = modified_driver.replace("      END\n", "      CALL WL_DUMP()\n      END\n")
    driver.write_text(modified_driver)
    (REPORT / f"{args.case}-profile-driver.f").write_text(modified_driver)
    card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
    for key, value in {"MLReductionLib": "6", "CTModeRun": -1,
                       "MLStabThres": 1e-9, "COLLIERRequiredAccuracy": 1e-10}.items():
        card[key] = value
    for path in [output / "Cards/MadLoopParams.dat",
                 *(output / "SubProcesses").rglob("MadLoopParams.dat")]:
        card.write(str(path), template=str(
            ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
        for path in (output / "SubProcesses").rglob(pattern):
            path.unlink()
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MAKEFLAGS="SHELL=" + shutil.which("bash"),
                       LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE / "HEPTools" / p)
                                                     for p in ("ninja/lib", "collier")))
    with (REPORT / f"{args.case}-build.log").open("w") as log:
        subprocess.run(["taskset", "-c", str(args.cpu), "make", "-j2", "check"],
                       cwd=directory, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    points = bench.REPORT / "points" / f"{args.case}.dat"
    with (REPORT / f"{args.case}-run.log").open("w") as log:
        subprocess.run(["taskset", "-c", str(args.cpu), "./check", str(points), "1"],
                       cwd=directory, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    for name in ("sparsity.dat", "values.dat", "timed_status.dat", "timing.dat"):
        shutil.copy2(directory / name, REPORT / f"{args.case}-{name}")
    lines = (directory / "sparsity.dat").read_text().splitlines()
    rate = int(lines[0])
    for row in lines[1:]:
        values = list(map(int, row.split()))
        item = routines[values[0]-1]
        item.update(dict(zip(("calls", "blocks", "zero_blocks", "multiplications",
                              "zero_block_multiplications", "nonfinite_a_calls",
                              "body_ticks"), values[1:])))
        item["zero_block_fraction"] = item["zero_blocks"] / item["blocks"]
        item["body_seconds"] = item["body_ticks"] / rate
    reference = json.loads((bench.REPORT /
        f"experiments/persistent-workers/preflight-qcd3-final-generator/data/{args.case}-candidate-6-0.json").read_text())
    observed = [line.split() for line in (directory / "values.dat").read_text().splitlines()]
    assert len(observed) == len(reference["rows"])
    for actual, expected in zip(observed, reference["rows"]):
        assert list(map(int, actual[:3])) == [expected["point"], expected["order"], expected["status"]]
        assert list(map(float, actual[3:7])) == expected["coefficients"]
        assert float(actual[7]) == expected["accuracy"]
    dump(REPORT / f"{args.case}-profile.json", {
        "scope": "Instrumented diagnostic only; timer excludes sparsity scans and covers each original updater body.",
        "case": args.case, "backend": 6, "cpu": args.cpu, "points": 100,
        "all_coefficients_accuracy_status_exact": True,
        "generated_polynomial_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "instrumented_polynomial_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "source_manifest": bench.generated_source_manifest(output),
        "executable_sha256": bench.sha256(directory / "check") if hasattr(bench, "sha256")
                             else hashlib.sha256((directory / "check").read_bytes()).hexdigest(),
        "clock_rate": rate, "routines": routines,
    })
    print(json.dumps([{k:v for k,v in r.items() if k in
                      ("name", "expanded", "calls", "zero_block_fraction", "body_seconds",
                       "nonfinite_a_calls")} for r in routines], indent=2), flush=True)


if __name__ == "__main__":
    main()
