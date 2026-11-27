"""Scratch: switch PWT_NODOUBLE drops the doubled-line integrals (DP and QP)."""
import sys
p = sys.argv[1]
full = open(p).read()
cut = full.index('      SUBROUTINE ML5_0_MP_ONIA_CG(')
dp, mp = full[:cut], full[cut:]
DECL = """      INTEGER POWERS(NLOOPLINE), I, J, K, MU, NC, NC1, KK"""
SW = """
      LOGICAL PWT_NODOUBLE
      COMMON/ML5_0_PW_TEST2/PWT_NODOUBLE"""
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)
old = """      IF (ALL(PCOEFS(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.(0D0,0D0)))
     $  RETURN"""
dp = rep(dp, old, old + "\n      IF (PWT_NODOUBLE) RETURN")
dp = rep(dp, DECL, DECL + SW)
old = """      IF (ALL(PCOEFS(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.CZERO)) RETURN"""
mp = rep(mp, old, old + "\n      IF (PWT_NODOUBLE) RETURN")
mp = rep(mp, DECL, DECL + SW)
mp += """
      SUBROUTINE ML5_0_PW_SET_NODOUBLE(FLAG)
      LOGICAL FLAG
      LOGICAL PWT_NODOUBLE
      COMMON/ML5_0_PW_TEST2/PWT_NODOUBLE
      PWT_NODOUBLE=FLAG
      END
"""
open(p, 'w').write(dp + mp)
