#!/usr/bin/env python3
"""Isolated binary64 derivative-row factorization; no production mutation.

This is a source-only draft until the independent actual-source binding,
compiled binary64 oracle, exception controls and process checks have passed.
Unlike the QP experiment, the initial DP target is the full compact updater.
There is no production DP terminal updater to use as a public fallback.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())
PROOF_SHA256 = "f5bc8c65680a028e79103f788fefc2356962350a9a7cffd8ffacbfad49510f42"
PARSER_SHA256 = "ce7c060c6c3f6a556ad1b518800030091dcc226a422bd17c1ef3ec5afe6d039a"
REVISION = "b44d20e725f1ec68589df20284e4ce9116e9e2ab"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proof(vertex):
    """Fail closed until independently retained DP expressions are bound."""
    assert vertex in ("FFV1L1_2", "FFV1L2_1")
    path = HERE / "oracle/source-parity.json"
    data = json.loads(path.read_text())
    assert data["schema"] == "binary64-actual-source-map-parity-v1"
    assert data["complete"] and data["passed"]
    assert data["source_revision"] == REVISION
    assert data["inherited_proof_sha256"] == PROOF_SHA256
    assert data["parser_sha256"] == PARSER_SHA256
    assert data["script_sha256"] == sha(HERE / "oracle/bind_sources.py")
    qp = HERE.parent / "symmetric-factorization"
    assert sha(qp / "analysis.json") == PROOF_SHA256
    assert sha(qp / "analyze.py") == PARSER_SHA256
    result = data["vertices"][vertex]
    assert result["passed"] and result["coefficient_identities_checked"] == 64
    assert result["all_coefficient_source_identities_checked"] == 80
    assert sha(ROOT / result["source"]) == result["source_sha256"]
    assert sha(ROOT / result["inherited_source"]) == result["inherited_source_sha256"]
    return result


def rotate(value, phase):
    phase = tuple(phase)
    if phase == (1, 0):
        return value
    if phase == (-1, 0):
        return f"(-{value})"
    if phase == (0, 1):
        return f"CMPLX(-AIMAG({value}),REAL({value},KIND=8),KIND=8)"
    if phase == (0, -1):
        return f"CMPLX(AIMAG({value}),-REAL({value},KIND=8),KIND=8)"
    raise ValueError("Only exact signed permutations of real/imaginary components")


def generate(generator, rank, vertex="FFV1L2_1", terminal=False,
             suffix="_FACTOR"):
    """Guarded changed-association full DP updater with unchanged fallback.

    B0 operations retain the public compact updater's original K ordering and
    B-zero predicate. No A-zero skip is introduced. Component zero signs may
    change on the factored path; its contract is finite coefficient accuracy,
    unchanged inputs and cumulative IEEE flags, rather than output bit identity.
    """
    assert generator.coef_format == "complex*16" and 2 <= rank <= 4
    assert not terminal, "The initial DP experiment supports only full updaters"
    identity = proof(vertex)
    rows = identity["q_rows"]
    assert len(rows) == 4 and all(len(row) == 4 for row in rows)
    assert all(entry["source_row"] == j and entry["phase"] == [1, 0]
               for j, entry in enumerate(rows[0], 1))
    name = generator.sub_prefix + f"UPDATE_WL_{rank}_1"
    assert len(name + suffix) <= 63
    source = generator.write_compact_wl_updater(rank, 1)
    assert "IEEE_" not in source and "\nCONTAINS" not in source
    signature = f"SUBROUTINE {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)"
    assert source.count(signature) == 1
    source = source.replace(signature,
        f"SUBROUTINE {name}{suffix}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT,WL_ENTRY_INEXACT)", 1)
    assert source.count("  implicit none") == 1
    source = source.replace("  implicit none", """  USE, INTRINSIC :: IEEE_ARITHMETIC
  USE, INTRINSIC :: IEEE_EXCEPTIONS
  implicit none
  LOGICAL WL_ENTRY_INEXACT
  TYPE(IEEE_ROUND_TYPE) WL_ROUNDING
  LOGICAL HALT_FLAGS(SIZE(IEEE_ALL))
  INTEGER HALT_INDEX,R""", 1)
    ncoef = (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24
    source = source.replace("  INTEGER I,J,K,L,M\n",
        f"  INTEGER I,J,K,L,M\n  COMPLEX*16 U(4,0:{ncoef - 1},MAXLWFSIZE)\n", 1)
    fallback = f"  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)\n  RETURN\n"
    guard = f"""C Experimental binary64 derivative-row factorization
IF(.NOT.WL_ENTRY_INEXACT.OR.IN_SIZE.NE.4.OR.OUT_SIZE.NE.4.OR.MAXLWFSIZE.NE.4.OR.LCUT_SIZE.LT.0.OR.LCUT_SIZE.GT.4) THEN
{fallback}ENDIF
DO HALT_INDEX=1,SIZE(IEEE_ALL)
  IF(.NOT.IEEE_SUPPORT_HALTING(IEEE_ALL(HALT_INDEX))) THEN
{fallback}  ENDIF
ENDDO
CALL IEEE_GET_HALTING_MODE(IEEE_ALL,HALT_FLAGS)
IF(ANY(HALT_FLAGS)) THEN
{fallback}ENDIF
CALL IEEE_GET_ROUNDING_MODE(WL_ROUNDING)
IF(WL_ROUNDING.NE.IEEE_NEAREST) THEN
{fallback}ENDIF
IF(.NOT.ALL(FACTOR_BOUNDED(REAL(A(1:IN_SIZE,0:{ncoef - 1},1:LCUT_SIZE),KIND=8))).OR..NOT.ALL(FACTOR_BOUNDED(AIMAG(A(1:IN_SIZE,0:{ncoef - 1},1:LCUT_SIZE)))).OR..NOT.ALL(FACTOR_BOUNDED(REAL(B(1:OUT_SIZE,0:4,1:IN_SIZE),KIND=8))).OR..NOT.ALL(FACTOR_BOUNDED(AIMAG(B(1:OUT_SIZE,0:4,1:IN_SIZE))))) THEN
{fallback}ENDIF
"""
    # Equality concerns the actual stored finite binary64 components. It ignores
    # signed-zero differences even in a nonzero complex coefficient. All nonzero
    # components must match the signed/permuted row exactly.
    # m=1 is the asserted identity row. Its self-comparisons add no information
    # after every input component has passed the integer finite/range check.
    for m, row in enumerate(rows[1:], 2):
        for j, entry in enumerate(row, 1):
            value = rotate(f"B({entry['source_row']},1,K)", entry["phase"])
            guard += f"""DO K=1,4
  IF(B({j},{m},K).NE.{value}) THEN
{fallback}  ENDIF
ENDDO
"""
    b0 = f"""C FACTOR_FAST_PATH
OUT(:,:,:)={generator.czero}
DO J=1,OUT_SIZE
  M=0
  DO K=1,IN_SIZE
    UPDATER_COEF=B(J,M,K)
    IF(UPDATER_COEF.EQ.{generator.czero}) CYCLE
    DO I=1,LCUT_SIZE
      DO L=0,{ncoef - 1}
        NEW_POSITION=COMB_COEF_POS(L,M)
        OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I)+A(K,L,I)*UPDATER_COEF
      ENDDO
    ENDDO
  ENDDO
ENDDO
"""
    u = f"""U={generator.czero}
DO I=1,LCUT_SIZE
  DO R=1,4
    DO K=1,IN_SIZE
      UPDATER_COEF=B(R,1,K)
      IF(UPDATER_COEF.EQ.{generator.czero}) CYCLE
      DO L=0,{ncoef - 1}
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
            shifts += f"""DO I=1,LCUT_SIZE
  DO L=0,{ncoef - 1}
    NEW_POSITION=COMB_COEF_POS(L,{m})
    OUT({j},NEW_POSITION,I)=OUT({j},NEW_POSITION,I)+{value}
  ENDDO
ENDDO
"""
    classifier = """CONTAINS
ELEMENTAL LOGICAL FUNCTION FACTOR_BOUNDED(X)
  REAL*8, INTENT(IN) :: X
  INTEGER*8, PARAMETER :: ONE(1)=TRANSFER(1.0_8,(/0_8/))
  FACTOR_BOUNDED=.FALSE.
  IF(STORAGE_SIZE(X).NE.64.OR.DIGITS(X).NE.53.OR.RADIX(X).NE.2.OR.MAXEXPONENT(X).NE.1024) RETURN
  IF(ONE(1).NE.ISHFT(1023_8,52)) RETURN
  ASSOCIATE(WORDS=>TRANSFER(X,ONE))
    FACTOR_BOUNDED=IBITS(WORDS(1),52,11).GE.823_8.AND.IBITS(WORDS(1),52,11).LE.1223_8
    IF(IBCLR(WORDS(1),63).EQ.0_8) FACTOR_BOUNDED=.TRUE.
  END ASSOCIATE
END FUNCTION
END
"""
    marker = "C Welcome to the computational heart of MadLoop..."
    assert source.count(marker) == 1
    return source[:source.index(marker)] + guard + b0 + u + shifts + classifier
