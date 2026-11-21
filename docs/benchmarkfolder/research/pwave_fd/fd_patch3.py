import sys
p = sys.argv[1]
s = open(p).read()
# switches: PWT_PART = 0 all, 1 CT only (no loop merges), 2 loops only (no CT)
old = """      INTEGER MU
      CALL ML5_0_PW_MERGE_LOOP(DWL"""
assert s.count(old) == 1
s = s.replace(old, """      INTEGER MU
      INTEGER PWT_PART
      COMMON/ML5_0_PW_TEST3/PWT_PART
      IF (PWT_PART.EQ.1) RETURN
      CALL ML5_0_PW_MERGE_LOOP(DWL""")
old = """      INTEGER I, K, COMP
      IF (PASS.EQ.1) THEN"""
assert s.count(old) == 1
s = s.replace(old, """      INTEGER I, K, COMP
      INTEGER PWT_PART
      COMMON/ML5_0_PW_TEST3/PWT_PART
      IF (PWT_PART.EQ.2) THEN
        ML5_0_ONIA_PW_CT_INPUT=.FALSE.
        RETURN
      ENDIF
      IF (PASS.EQ.1) THEN""")
s += """
      SUBROUTINE ML5_0_PW_SET_PART(PART)
      INTEGER PART
      INTEGER PWT_PART
      COMMON/ML5_0_PW_TEST3/PWT_PART
      PWT_PART=PART
      END
"""
open(p, 'w').write(s)
