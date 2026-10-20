#!/usr/bin/env python3
"""Isolated changed-rounding QP factorization; never edits production sources."""
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "symmetric-accumulation"
sys.path.insert(0, str(PREVIOUS))
import gather

PROOF_SHA256 = "f5bc8c65680a028e79103f788fefc2356962350a9a7cffd8ffacbfad49510f42"


def proof(vertex):
    path = HERE / "analysis.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROOF_SHA256
    cases = {
        "MP_FFV1L1_2": "qcd-3", "MP_FFV1L2_1": "qcd-3",
        "MP_FFS1L1_2": "qcd-4", "MP_FFS1L2_1": "qcd-4",
    }
    assert vertex in cases
    result = json.loads(path.read_text())["cases"][cases[vertex]]["vertex_proofs"][vertex]
    assert result["passed"] and result["coefficient_identities_checked"] == 64
    return result


def rotate(value, phase):
    phase = tuple(phase)
    if phase == (1, 0):
        return value
    if phase == (-1, 0):
        return f"(-{value})"
    if phase == (0, 1):
        return f"CMPLX(-AIMAG({value}),REAL({value},KIND=16),KIND=16)"
    if phase == (0, -1):
        return f"CMPLX(AIMAG({value}),-REAL({value},KIND=16),KIND=16)"
    raise ValueError("Only exact signed permutations of real/imaginary components")


def generate(generator, rank, vertex="MP_FFV1L2_1", terminal=False,
             suffix="_FACTOR"):
    assert generator.coef_format == "complex*32" and 2 <= rank <= 4
    identity = proof(vertex)
    rows = identity["q_rows"]
    required = [set(entry["source_row"] for entry in [r[j] for r in rows])
                for j in range(4)]
    assert [sorted(r) for r in required] == identity["terminal_required_u_rows"]
    name = generator.sub_prefix + ("TERMINAL" if terminal else "UPDATE")
    name += f"_WL_{rank}_1"
    assert len(name + suffix) <= 63
    source = gather.generate(generator, rank, terminal)
    source = source.replace("SUBROUTINE " + name + "_GATHER(",
                            "SUBROUTINE " + name + suffix + "(", 1)
    if not terminal:
        before = "SUBROUTINE " + name + suffix + "(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)"
        assert source.count(before) == 1
        source = source.replace(before, before[:-1] + ",WL_ENTRY_INEXACT)", 1)
        source = source.replace("  implicit none",
                                "  implicit none\n  LOGICAL WL_ENTRY_INEXACT", 1)
    source, removed = re.subn(
        r"(?m)^  (?:INTEGER N|INTEGER, PARAMETER :: PREDECESSOR.*|"
        r"LOGICAL ACTIVE.*|complex\*32 ACC)\n", "", source)
    assert removed == 4
    ncoef = gather.get_number_of_coefs_for_rank(rank)
    mapping = ",".join(".TRUE." if r in required[j - 1] else ".FALSE."
                       for j in range(1, 5) for r in range(1, 5))
    source = source.replace("  INTEGER I,J,K,L,M\n",
        "  INTEGER I,J,K,L,M,R,HALT_INDEX\n"
        f"  COMPLEX*32 U(4,0:{ncoef-1},MAXLWFSIZE)\n"
        f"  LOGICAL, PARAMETER :: NEEDED(4,4)=RESHAPE((/{mapping}/),(/4,4/))\n", 1)
    extra = ",WL_ENTRY_INEXACT" if terminal else ""
    fallback = f"  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT{extra})\n  RETURN\n"
    guard = f"""IF(.NOT.WL_ENTRY_INEXACT.OR.IN_SIZE.NE.4.OR.OUT_SIZE.NE.4.OR.MAXLWFSIZE.NE.4.OR.LCUT_SIZE.LT.0.OR.LCUT_SIZE.GT.4) THEN
{fallback}ENDIF
DO HALT_INDEX=1,SIZE(IEEE_ALL)
  IF(.NOT.IEEE_SUPPORT_HALTING(IEEE_ALL(HALT_INDEX))) THEN
{fallback}  ENDIF
ENDDO
"""
    marker = "C Experimental coefficient-major gather\n"
    assert source.count(marker) == 1
    source = source.replace(marker,
        "C Experimental derivative-row factorization; changed rounding\n" + guard, 1)

    # Keep existing preparation/fallback. Runtime comparisons operate only on
    # bounded finite operands. Numerical equality ignores real/imaginary zero
    # signs, including a zero component of a nonzero complex value. This is part
    # of the changed-rounding numerical contract, not an output-bit promise.
    # Every nonzero component must agree exactly with the stored row relation.
    relation_guard = ""
    for m, row in enumerate(rows, 1):
        for j, entry in enumerate(row, 1):
            r = entry["source_row"]
            rotated = rotate(f"B({r},1,K)", entry["phase"])
            relation_guard += f"""DO K=1,4
  IF(B({j},{m},K).NE.{rotated}) THEN
{fallback}  ENDIF
ENDDO
"""
    start = source.index("OUT(:,:,:)=", source.index("C Experimental"))
    end = source.index("\nCONTAINS", start)
    iloop = "DO I=J,MIN(J,LCUT_SIZE)" if terminal else "DO I=1,LCUT_SIZE"
    # The original B0 multiplication and accumulation operations stay intact,
    # including K order and both original B/A zero guards.
    b0 = f"""DO J=1,OUT_SIZE
  M=0
  DO K=1,IN_SIZE
    UPDATER_COEF = B(J,M,K)
    IF (UPDATER_COEF.EQ.{generator.czero}) CYCLE
    WL_SKIP_ZERO=.FALSE.
    IF(WL_NEAREST) WL_SKIP_ZERO=WL_IS_FINITE(REAL(UPDATER_COEF,KIND=16)).AND.WL_IS_FINITE(AIMAG(UPDATER_COEF))
    {iloop}
      DO L=0,{ncoef-1}
        IF(WL_SKIP_ZERO) THEN
          IF(WL_AZERO(K,L,I)) CYCLE
        ENDIF
        NEW_POSITION = COMB_COEF_POS(L,M)
        OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + A(K,L,I)*UPDATER_COEF
      ENDDO
    ENDDO
  ENDDO
ENDDO
"""
    # In a terminal call I is the final diagonal J, and only the union of rows
    # needed by its four derivative shifts is formed for that cut component.
    u = f"""U={generator.czero}
DO I=1,LCUT_SIZE
  DO R=1,4
"""
    if terminal:
        u += "    IF(.NOT.NEEDED(R,I)) CYCLE\n"
    u += f"""    DO K=1,IN_SIZE
      UPDATER_COEF=B(R,1,K)
      IF(UPDATER_COEF.EQ.{generator.czero}) CYCLE
      DO L=0,{ncoef-1}
        IF(WL_AZERO(K,L,I)) CYCLE
        U(R,L,I)=U(R,L,I)+A(K,L,I)*UPDATER_COEF
      ENDDO
    ENDDO
  ENDDO
ENDDO
"""
    shifts = ""
    for j in range(1, 5):
        for m, row in enumerate(rows, 1):
            entry = row[j - 1]
            value = rotate(f"U({entry['source_row']},L,I)", entry["phase"])
            active_i = f"DO I={j},MIN({j},LCUT_SIZE)" if terminal else "DO I=1,LCUT_SIZE"
            shifts += f"""{active_i}
  DO L=0,{ncoef-1}
    NEW_POSITION=COMB_COEF_POS(L,{m})
    OUT({j},NEW_POSITION,I)=OUT({j},NEW_POSITION,I)+{value}
  ENDDO
ENDDO
"""
    body = ("C FACTOR_FAST_PATH\n" + f"OUT(:,:,:)={generator.czero}\n"
            + b0 + u + shifts)
    return source[:start] + relation_guard + body + source[end:]
