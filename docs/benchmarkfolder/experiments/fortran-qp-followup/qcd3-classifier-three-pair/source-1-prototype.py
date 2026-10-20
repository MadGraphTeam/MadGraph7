#!/usr/bin/env python3
"""Isolated finite-path simplification; production generator stays unchanged."""
from pathlib import Path
import re
import tempfile

ZERO_HELPER = """ELEMENTAL LOGICAL FUNCTION WL_IS_ZERO(Z)
  COMPLEX*32, INTENT(IN) :: Z
  INTEGER*8 RE_WORDS(2),IM_WORDS(2)
  INTEGER*8, PARAMETER :: ZERO_ONE_WORDS(2)=TRANSFER(1.0E0_16,(/0_8,0_8/))
  INTEGER, PARAMETER :: ZERO_HI=MERGE(1,2,ZERO_ONE_WORDS(1).NE.0_8)
  RE_WORDS=TRANSFER(REAL(Z,KIND=16),RE_WORDS)
  IM_WORDS=TRANSFER(AIMAG(Z),IM_WORDS)
  WL_IS_ZERO=IBCLR(RE_WORDS(ZERO_HI),63).EQ.0_8.AND.RE_WORDS(3-ZERO_HI).EQ.0_8.AND.IBCLR(IM_WORDS(ZERO_HI),63).EQ.0_8.AND.IM_WORDS(3-ZERO_HI).EQ.0_8
END FUNCTION
"""


def transform_logical(source):
    if "WL_NEAREST" not in source:
        return source
    body, contained = source.split("CONTAINS", 1)
    rank = re.search(r"SUBROUTINE \w+UPDATE_WL_(\d+)_(\d+)\(", body)
    left, right = map(int, rank.groups())
    from madgraph.various.q_polynomial import get_number_of_coefs_for_rank
    na, nb = (get_number_of_coefs_for_rank(r)-1 for r in (left, right))
    if "UPDATER_COEF" in body:
        pattern = r"WL_AZERO\(1:IN_SIZE,0:\d+,1:LCUT_SIZE\)=\.FALSE\..*?\nENDIF\n"
        replacement = f"""DO WL_I=1,LCUT_SIZE
  DO WL_L=0,{na}
    DO WL_K=1,IN_SIZE
      WL_AZERO(WL_K,WL_L,WL_I)=WL_IS_ZERO(A(WL_K,WL_L,WL_I))
    ENDDO
  ENDDO
ENDDO
"""
        body, count = re.subn(pattern, replacement, body, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError("Compact mask preparation not found")
        body = body.replace("  LOGICAL WL_SKIP_ZERO\n", "")
        body, count = re.subn(r"[ \t]*WL_SKIP_ZERO=\.FALSE\.\n"
                             r"[ \t]*IF\(WL_NEAREST\) WL_SKIP_ZERO=.*?\n",
                             "", body, count=1)
        if count != 1:
            raise RuntimeError("Redundant updater finite check not found")
        body, count = re.subn(r"IF\(WL_SKIP_ZERO\) THEN\n\s*"
                             r"IF\(WL_AZERO\(K,L,I\)\) CYCLE\n\s*ENDIF",
                             "IF(WL_AZERO(K,L,I)) CYCLE", body, count=1)
        if count != 1:
            raise RuntimeError("Compact skip guard not found")
        body, count = re.subn(r"IF \(UPDATER_COEF\.EQ\.CMPLX\([^\n]+\)\) CYCLE",
                             "IF (WL_IS_ZERO(UPDATER_COEF)) CYCLE", body, count=1)
        if count != 1:
            raise RuntimeError("Finite updater zero comparison not found")
    else:
        body, count = re.subn(r"WL_SKIP=\.FALSE\..*?\nENDIF\n",
                             f"""DO WL_K=1,IN_SIZE
  DO WL_J=1,OUT_SIZE
    WL_SKIP(WL_J,WL_K)=ALL(WL_IS_ZERO(B(WL_J,0:{nb},WL_K)))
  ENDDO
ENDDO
""", body, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError("Expanded skip preparation not found")
    contained = contained.replace("END FUNCTION\n", "END FUNCTION\n"+ZERO_HELPER, 1)
    return body + "CONTAINS" + contained


def logical_lines(fixed):
    lines = []
    for line in fixed.splitlines():
        if not line.strip():
            continue
        if line[0] in "cC*!":
            lines.append(line)
        elif len(line) > 5 and line[5] != " ":
            lines[-1] += line[6:].strip()
        else:
            lines.append(line[6:].strip())
    return "\n".join(lines) + "\n"


def transform_fixed(source):
    from madgraph.iolibs.file_writers import FortranWriter
    pattern = re.compile(r"(?m)^      SUBROUTINE MP_\w+UPDATE_WL_\d+_\d+\("
                         r".*?^      END\s*$", re.S | re.M)
    count = 0
    def replace(match):
        nonlocal count
        before = logical_lines(match.group())
        after = transform_logical(before)
        if before == after:
            return match.group()
        count += 1
        with tempfile.TemporaryDirectory(prefix="qp-classifier-") as temp:
            filename = Path(temp) / "routine.f"
            with FortranWriter(str(filename)) as writer:
                writer.writelines(after)
            return filename.read_text().rstrip()
    result = pattern.sub(replace, source)
    if count == 0:
        raise RuntimeError("No production sparse QP routines found")
    return result, count
