#!/usr/bin/env python3
"""Count exact-zero A work after the compact B filter.

The retained qcd3 follow-up is complete. It uses Ninja6 only and preserves all
arithmetic; it adds diagnostic scans outside the per-updater body timers.
"""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("wl_profile", HERE / "profile.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_instrument = base.instrument


def instrument(source):
    text, routines = original_instrument(source)
    text = text.replace(
        "      END MODULE\n",
        "      INTEGER*8, SAVE :: WL_ACTUAL_MULT(32)=0\n"
        "      INTEGER*8, SAVE :: WL_AZERO_MULT(32)=0\n"
        "      INTEGER*8, SAVE :: WL_AZERO_BLOCK_MULT(32)=0\n"
        "      INTEGER*8, SAVE :: WL_BAD_B_MULT(32)=0\n"
        "      END MODULE\n", 1)
    for routine in routines:
        if routine["expanded"]:
            continue
        index = routine["index"]
        na = base.math.comb(routine["left_rank"]+4, 4)
        nb = base.math.comb(routine["right_rank"]+4, 4)
        pattern = rf"(?m)^      SUBROUTINE {routine['name']}\(.*?^      END\s*$"
        def replace(match):
            text = match[0].replace("      INTEGER WL_J,WL_K\n",
                                   "      INTEGER WL_J,WL_K,WL_I\n"
                                   "      INTEGER*8 WL_BNZ,WL_BFINITE,WL_AZERO\n")
            counters = f"""      DO WL_K=1,IN_SIZE
        WL_BNZ=COUNT(B(1:OUT_SIZE,0:{nb-1},WL_K).NE.0)
        WL_BFINITE=COUNT(B(1:OUT_SIZE,0:{nb-1},WL_K).NE.0
     $    .AND.IEEE_IS_FINITE(REAL(B(1:OUT_SIZE,0:{nb-1},WL_K)))
     $    .AND.IEEE_IS_FINITE(AIMAG(B(1:OUT_SIZE,0:{nb-1},WL_K))))
        WL_AZERO=COUNT(A(WL_K,0:{na-1},1:LCUT_SIZE).EQ.0)
        WL_ACTUAL_MULT({index})=WL_ACTUAL_MULT({index})+
     $    WL_BNZ*{na}*LCUT_SIZE
        WL_AZERO_MULT({index})=WL_AZERO_MULT({index})+
     $    WL_BFINITE*WL_AZERO
        WL_BAD_B_MULT({index})=WL_BAD_B_MULT({index})+
     $    (WL_BNZ-WL_BFINITE)*{na}*LCUT_SIZE
        DO WL_I=1,LCUT_SIZE
          IF(ALL(A(WL_K,0:{na-1},WL_I).EQ.0)) THEN
            WL_AZERO_BLOCK_MULT({index})=WL_AZERO_BLOCK_MULT({index})
     $        +WL_BFINITE*{na}
          ENDIF
        ENDDO
      ENDDO
"""
            return text.replace("      CALL SYSTEM_CLOCK(WL_START)\n",
                                counters + "      CALL SYSTEM_CLOCK(WL_START)\n")
        text, count = re.subn(pattern, replace, text, flags=re.S | re.M)
        assert count == 1
    text = text.replace("      WL_NONFINITE_A=0\n",
                        "      WL_NONFINITE_A=0\n"
                        "      WL_ACTUAL_MULT=0\n"
                        "      WL_AZERO_MULT=0\n"
                        "      WL_AZERO_BLOCK_MULT=0\n"
                        "      WL_BAD_B_MULT=0\n")
    text = text.replace("      CLOSE(89)\n", """      CLOSE(89)
      OPEN(89,FILE='compact-sparsity.dat',STATUS='REPLACE')
      DO I=1,32
        IF(WL_ACTUAL_MULT(I).GT.0) WRITE(89,*) I,WL_ACTUAL_MULT(I),
     $    WL_AZERO_MULT(I),WL_AZERO_BLOCK_MULT(I),WL_BAD_B_MULT(I)
      ENDDO
      CLOSE(89)
""")
    return text, routines


def main():
    base.instrument = instrument
    base.WORK = base.WORK / "compact-a-followup"
    base.REPORT = HERE / "compact-a-followup"
    base.REPORT.mkdir(exist_ok=True)
    base.main()
    # The command-line case is recorded in the just-created profile JSON.
    for profile in base.REPORT.glob("*-profile.json"):
        result = json.loads(profile.read_text())
        case = result["case"]
        directory = next((base.WORK / case / "profile").glob("SubProcesses/P*_*"))
        path = directory / "compact-sparsity.dat"
        shutil.copy2(path, base.REPORT / f"{case}-compact-sparsity.dat")
        rows = []
        for line in path.read_text().splitlines():
            index, actual, zeros, blocks, bad_b = map(int, line.split())
            routine = result["routines"][index-1]
            rows.append({
                "name": routine["name"], "qp": routine["qp"],
                "multiplications_after_existing_b_zero_skip": actual,
                "zero_a_multiplications_with_finite_b": zeros,
                "zero_a_fraction": zeros / actual,
                "whole_a_block_multiplications_with_finite_b": blocks,
                "nonfinite_b_multiplications": bad_b,
                "original_body_seconds": routine["body_seconds"],
            })
        base.dump(base.REPORT / f"{case}-compact-a-summary.json", rows)
        print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
