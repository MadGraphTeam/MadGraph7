"""Isolated per-batch color contraction cache; generated files only."""
import re


def polynomial(source):
    """Return a modified Born-interference polynomial file and metadata."""
    pattern = r"(?m)^      SUBROUTINE ((?:MP_)?(ML5_\d+_)CREATE_LOOP_COEFS)\(.*?^      END\s*$"
    routines = list(re.finditer(pattern, source, flags=re.S | re.M))
    assert len(routines) == 2, "Prototype requires the DP/QP Born-interference pair"
    original = routines[0][0]
    prefix = routines[0][2]
    dims = {name: int(re.search(rf"\b{name}=(\d+)", original)[1])
            for name in ("NBORNAMPS", "NCOLORROWS", "NAMPSO")}
    module = prefix + "COLOR_CACHE"
    module_text = f"""      MODULE {module}
      IMPLICIT NONE
      INTEGER, PARAMETER :: CC_NROW={dims['NCOLORROWS']}
      INTEGER, PARAMETER :: CC_NAMP={dims['NBORNAMPS']}
      INTEGER, PARAMETER :: CC_NSO={dims['NAMPSO']}
      INTEGER, SAVE :: CC_MAP(CC_NROW)=0,CC_NCLASS=0
      LOGICAL, SAVE :: CC_MAP_READY=.FALSE.
      LOGICAL, SAVE :: CC_DP_ACTIVE=.FALSE.,CC_QP_ACTIVE=.FALSE.
      LOGICAL, SAVE :: CC_DP_VALID(CC_NROW)=.FALSE.
      LOGICAL, SAVE :: CC_QP_VALID(CC_NROW)=.FALSE.
      COMPLEX*16, SAVE :: CC_DP(CC_NSO,CC_NROW)
      COMPLEX*32, SAVE :: CC_QP(CC_NSO,CC_NROW)
      CONTAINS
      SUBROUTINE CC_PREPARE_MAP()
      INTEGER CF_D(CC_NROW,CC_NAMP),CF_N(CC_NROW,CC_NAMP)
      COMMON/{prefix}CF/CF_D,CF_N
      INTEGER I,J,REP(CC_NROW)
      IF(CC_MAP_READY) RETURN
      CC_NCLASS=0
      DO I=1,CC_NROW
        DO J=1,CC_NCLASS
          IF(ALL(CF_N(I,:).EQ.CF_N(REP(J),:)).AND.
     $       ALL(CF_D(I,:).EQ.CF_D(REP(J),:))) EXIT
        ENDDO
        IF(J.GT.CC_NCLASS) THEN
          CC_NCLASS=J
          REP(J)=I
        ENDIF
        CC_MAP(I)=J
      ENDDO
      CC_MAP_READY=.TRUE.
      END SUBROUTINE
      END MODULE

      SUBROUTINE {prefix}COLOR_CACHE_INVALIDATE()
      USE {module}
      CC_MAP_READY=.FALSE.
      CC_DP_ACTIVE=.FALSE.
      CC_QP_ACTIVE=.FALSE.
      END
"""
    for precision, kind in (("DP", 8), ("QP", 16)):
        amp_common = prefix + ("MP_" if kind == 16 else "") + "AMPS"
        module_text += f"""
      SUBROUTINE {prefix}COLOR_CACHE_BEGIN_{precision}()
      USE {module}
      USE, INTRINSIC :: IEEE_ARITHMETIC
      COMPLEX*{2*kind} AMP(CC_NAMP)
      COMMON/{amp_common}/AMP
      CALL CC_PREPARE_MAP()
      CC_{precision}_VALID=.FALSE.
      CC_{precision}_ACTIVE=ALL(IEEE_IS_FINITE(REAL(AMP,KIND={kind})))
     $  .AND.ALL(IEEE_IS_FINITE(AIMAG(AMP)))
      END

      SUBROUTINE {prefix}COLOR_CACHE_END_{precision}()
      USE {module}
      CC_{precision}_ACTIVE=.FALSE.
      END
"""

    def modify(match):
        text = match[0]
        precision = "QP" if match[1].startswith("MP_") else "DP"
        use = f"      USE {prefix}POLYNOMIAL_CONSTANTS\n"
        assert text.count(use) == 1
        text = text.replace(use, use + f"""      USE {module}, ONLY: CC_MAP,CC_{precision},
     $  CC_{precision}_ACTIVE,CC_{precision}_VALID
""")
        declaration = "      INTEGER I,J\n"
        assert text.count(declaration) == 1
        text = text.replace(declaration, declaration +
                            "      INTEGER CC_SLOT\n      LOGICAL CC_HIT\n")
        start = text.index("      DO I=1,NAMPSO\n")
        stop = text.index("      DO I=1,NAMPSO\n", start+1)
        unchanged_loop = text[start:stop]
        # Keep every conversion, Born lookup and addition on the miss path
        # in its original order. Later weighting only changes local CONST.
        text = text[:start] + f"""      CC_HIT=.FALSE.
      IF(CC_{precision}_ACTIVE) THEN
        CC_SLOT=CC_MAP(COLOR_ID)
        IF(CC_{precision}_VALID(CC_SLOT)) THEN
          CONST=CC_{precision}(:,CC_SLOT)
          CC_HIT=.TRUE.
        ENDIF
      ENDIF
      IF(.NOT.CC_HIT) THEN
""" + unchanged_loop + f"""        IF(CC_{precision}_ACTIVE) THEN
          CC_{precision}(:,CC_SLOT)=CONST
          CC_{precision}_VALID(CC_SLOT)=.TRUE.
        ENDIF
      ENDIF

""" + text[stop:]
        return text

    changed = re.sub(pattern, modify, source, flags=re.S | re.M)
    return module_text + changed, {"prefix": prefix, **dims}


def batches(source, prefix):
    """Wrap each complete consecutive coefficient-construction chunk group."""
    pattern = (rf"(?m)^([ ]*)CALL {prefix}(MP_)?COEF_CONSTRUCTION_\d+\([^\n]*\)"
               rf"(?:\n[ ]*CALL {prefix}(?:MP_)?COEF_CONSTRUCTION_\d+\([^\n]*\))*")
    counts = {"DP": 0, "QP": 0}

    def wrap(match):
        precision = "QP" if match[2] else "DP"
        calls = re.findall(r"COEF_CONSTRUCTION_(\d+)", match[0])
        assert int(calls[0]) == 1 and list(map(int, calls)) == list(range(1, len(calls)+1))
        counts[precision] += 1
        indent = match[1]
        return (f"{indent}CALL {prefix}COLOR_CACHE_BEGIN_{precision}()\n"
                + match[0] + f"\n{indent}CALL {prefix}COLOR_CACHE_END_{precision}()")
    changed = re.sub(pattern, wrap, source)
    return changed, counts


def loop_matrix(source, prefix):
    source, counts = batches(source, prefix)
    assert counts == {"DP": 1, "QP": 0}, counts
    marker = " 107    CONTINUE\n        CLOSE(1)\n"
    assert source.count(marker) == 1
    source = source.replace(marker, marker +
                            f"        CALL {prefix}COLOR_CACHE_INVALIDATE()\n")
    return source


def mp_compute(source, prefix):
    changed, counts = batches(source, prefix)
    assert counts == {"DP": 1, "QP": 1}, counts
    return changed
