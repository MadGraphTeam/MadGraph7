"""Unmeasured generated-QP prototypes; no production generator is modified.

Both transforms preserve the order of every retained arithmetic statement.
The runtime checks deliberately leave non-nearest rounding and relevant
nonfinite operands on the original path. Compiled bit-pattern/IEEE tests and
isolated process measurements are required before considering production.
"""
import math
import re


def transform(source, mode):
    assert mode in ("expanded", "compact")
    count = 0

    def change(match):
        nonlocal count
        text = match[0]
        original = text
        name = match[1]
        r1, r2 = int(match[2]), int(match[3])
        compact = "UPDATER_COEF" in text
        if compact != (mode == "compact"):
            return text
        count += 1
        na, nb = math.comb(r1+4, 4), math.comb(r2+4, 4)
        text = re.sub(r"(?m)^(      USE \w+POLYNOMIAL_CONSTANTS)$",
                      r"\1\n      USE, INTRINSIC :: IEEE_ARITHMETIC", text)
        declarations = ("      TYPE(IEEE_ROUND_TYPE) WL_ROUNDING\n"
                        "      LOGICAL WL_NEAREST\n")
        if compact:
            declarations += (
                "      LOGICAL WL_SKIP_ZERO\n"
                "      LOGICAL WL_AZERO(MAXLWFSIZE,0:LOOPMAXCOEFS-1,\n"
                "     $  MAXLWFSIZE)\n"
                "      INTEGER WL_I,WL_K,WL_L\n")
        else:
            declarations += ("      LOGICAL WL_SKIP(MAXLWFSIZE,MAXLWFSIZE)\n"
                             "      INTEGER WL_J,WL_K\n")
        marker = "      INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE\n"
        assert text.count(marker) == 1
        text = text.replace(marker, marker + declarations)
        executable = "      OUT(:,:,:)" if compact else "      DO I=1,LCUT_SIZE"
        position = text.index(executable)
        preparation = ("      CALL IEEE_GET_ROUNDING_MODE(WL_ROUNDING)\n"
                       "      WL_NEAREST=WL_ROUNDING.EQ.IEEE_NEAREST\n")
        # An untouched helper preserves the compiler's original exceptional
        # complex-arithmetic behavior. Merely disabling individual skips changed
        # NaN signs in the v1 adversarial tests.
        preparation += f"""      IF(.NOT.WL_NEAREST.OR.
     $  .NOT.ALL(IEEE_IS_FINITE(REAL(A(1:IN_SIZE,0:{na-1},
     $    1:LCUT_SIZE),KIND=16))).OR.
     $  .NOT.ALL(IEEE_IS_FINITE(AIMAG(A(1:IN_SIZE,0:{na-1},
     $    1:LCUT_SIZE)))).OR.
     $  .NOT.ALL(IEEE_IS_FINITE(REAL(B(1:OUT_SIZE,0:{nb-1},
     $    1:IN_SIZE),KIND=16))).OR.
     $  .NOT.ALL(IEEE_IS_FINITE(AIMAG(B(1:OUT_SIZE,0:{nb-1},
     $    1:IN_SIZE))))) THEN
        CALL {name}_DENSE(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
        RETURN
      ENDIF
"""
        if compact:
            # Classify each A once. In particular, do not add floating-point
            # equality comparisons on quiet NaNs to the original arithmetic.
            preparation += f"""      WL_AZERO(1:IN_SIZE,0:{na-1},1:LCUT_SIZE)=.FALSE.
      IF(WL_NEAREST) THEN
        DO WL_K=1,IN_SIZE
          DO WL_I=1,LCUT_SIZE
            DO WL_L=0,{na-1}
              IF(IEEE_IS_FINITE(REAL(A(WL_K,WL_L,WL_I),KIND=16))
     $          .AND.IEEE_IS_FINITE(AIMAG(A(WL_K,WL_L,WL_I))))
     $          WL_AZERO(WL_K,WL_L,WL_I)=A(WL_K,WL_L,WL_I).EQ.0
            ENDDO
          ENDDO
        ENDDO
      ENDIF
"""
            text = text[:position] + preparation + text[position:]
            marker = re.search(r"(?m)^            IF \(UPDATER_COEF\.EQ\..*\) CYCLE$", text)
            assert marker
            insertion = """
            WL_SKIP_ZERO=.FALSE.
            IF(WL_NEAREST) WL_SKIP_ZERO=
     $        IEEE_IS_FINITE(REAL(UPDATER_COEF,KIND=16)).AND.
     $        IEEE_IS_FINITE(AIMAG(UPDATER_COEF))
"""
            text = text[:marker.end()] + insertion + text[marker.end():]
            marker = "                NEW_POSITION = COMB_COEF_POS(L,M)\n"
            assert text.count(marker) == 1
            text = text.replace(marker, """                IF(WL_SKIP_ZERO) THEN
                  IF(WL_AZERO(K,L,I)) CYCLE
                ENDIF
""" + marker)
        else:
            preparation += f"""      WL_SKIP=.FALSE.
      IF(WL_NEAREST) THEN
        DO WL_K=1,IN_SIZE
          IF(ALL(IEEE_IS_FINITE(REAL(A(WL_K,0:{na-1},
     $      1:LCUT_SIZE),KIND=16))).AND.
     $       ALL(IEEE_IS_FINITE(AIMAG(A(WL_K,0:{na-1},
     $      1:LCUT_SIZE))))) THEN
            DO WL_J=1,OUT_SIZE
              IF(ALL(IEEE_IS_FINITE(REAL(B(WL_J,0:{nb-1},WL_K),
     $          KIND=16))).AND.
     $          ALL(IEEE_IS_FINITE(AIMAG(B(WL_J,0:{nb-1},WL_K)))))
     $          WL_SKIP(WL_J,WL_K)=ALL(B(WL_J,0:{nb-1},WL_K).EQ.0)
            ENDDO
          ENDIF
        ENDDO
      ENDIF
"""
            text = text[:position] + preparation + text[position:]
            marker = "        DO K=1,IN_SIZE\n          DO J=1,OUT_SIZE\n"
            assert text.count(marker) == 1
            text = text.replace(marker, marker +
                                "            IF(WL_SKIP(J,K)) CYCLE\n")
        # The compiler's IEEE classification intrinsics can signal INVALID on
        # signaling NaNs, even when the original performs no arithmetic.
        # Inspect only integer representation bits on IEEE binary128; any other
        # REAL16 representation conservatively selects the untouched helper.
        text = text.replace("IEEE_IS_FINITE(", "WL_IS_FINITE(")
        ending = text.rfind("      END")
        text = text[:ending] + """      CONTAINS
      ELEMENTAL LOGICAL FUNCTION WL_IS_FINITE(X)
      REAL*16, INTENT(IN) :: X
      INTEGER*8 WORDS(2)
      INTEGER*8, PARAMETER :: ONE_WORDS(2)=
     $  TRANSFER(1.0_16,(/0_8,0_8/))
      INTEGER, PARAMETER :: HI=MERGE(1,2,ONE_WORDS(1).NE.0_8)
      WL_IS_FINITE=.FALSE.
      IF(STORAGE_SIZE(X).NE.128.OR.DIGITS(X).NE.113.OR.
     $  RADIX(X).NE.2.OR.MAXEXPONENT(X).NE.16384) RETURN
      WORDS=TRANSFER(X,WORDS)
      WL_IS_FINITE=IBITS(WORDS(HI),48,15).NE.32767_8
      END FUNCTION
""" + text[ending:]
        dense = original.replace("SUBROUTINE " + name + "(",
                                 "SUBROUTINE " + name + "_DENSE(", 1)
        return text + "\n\n" + dense

    pattern = r"(?m)^      SUBROUTINE (MP_\w*UPDATE_WL_(\d+)_(\d+))\(.*?^      END\s*$"
    modified = re.sub(pattern, change, source, flags=re.S | re.M)
    if not count:
        raise ValueError("No matching QP routines")
    return modified, count
