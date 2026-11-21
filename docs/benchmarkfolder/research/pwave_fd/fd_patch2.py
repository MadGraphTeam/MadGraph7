import sys
p = sys.argv[1]
s = open(p).read()
old = """      IF (ALL(PCOEFS(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.(0D0,0D0)))
     $  RETURN"""
assert s.count(old) == 1
s = s.replace(old, old + """
      IF (PWT_NODOUBLE) RETURN""")
old = """      INTEGER POWERS(NLOOPLINE), I, J, K, MU, NC, NC1, KK"""
assert s.count(old) == 1
s = s.replace(old, old + """
      LOGICAL PWT_NODOUBLE
      COMMON/ML5_0_PW_TEST2/PWT_NODOUBLE""")
s += """
      SUBROUTINE ML5_0_PW_SET_NODOUBLE(FLAG)
      LOGICAL FLAG
      LOGICAL PWT_NODOUBLE
      COMMON/ML5_0_PW_TEST2/PWT_NODOUBLE
      PWT_NODOUBLE=FLAG
      END
"""
open(p, 'w').write(s)
