"""Scratch: switch PWT_PART = 0 all, 1 CT only (no loop merges), 2 loops
only (no CT), in DP and QP."""
import sys
p = sys.argv[1]
full = open(p).read()
cut = full.index('      SUBROUTINE ML5_0_MP_ONIA_CG(')
dp, mp = full[:cut], full[cut:]
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)
SW = """      INTEGER PWT_PART
      COMMON/ML5_0_PW_TEST3/PWT_PART
"""
for pre, name in (('', 'dp'), ('MP_', 'mp')):
    t = dp if name == 'dp' else mp
    old = "      INTEGER MU\n      CALL %sML5_0_PW_MERGE_LOOP(DWL" % pre
    t = rep(t, old, "      INTEGER MU\n" + SW + "      IF (PWT_PART.EQ.1) RETURN\n"
            "      CALL %sML5_0_PW_MERGE_LOOP(DWL" % pre)
    fn = 'ML5_0_%sONIA_PW_CT_INPUT' % pre
    old = "      INTEGER I, K, COMP\n      IF (PASS.EQ.1) THEN"
    t = rep(t, old, "      INTEGER I, K, COMP\n" + SW + """      IF (PWT_PART.EQ.2) THEN
        %s=.FALSE.
        RETURN
      ENDIF
      IF (PASS.EQ.1) THEN""" % fn)
    if name == 'dp':
        dp = t
    else:
        mp = t
mp += """
      SUBROUTINE ML5_0_PW_SET_PART(PART)
      INTEGER PART
""" + SW + """      PWT_PART=PART
      END
"""
open(p, 'w').write(dp + mp)
