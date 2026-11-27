"""Patch an output's TIRUST_interface.f for the hybrid test: with
ONIA_DBG_DPTIR set, MadLoop's quadruple-precision run (CTModeRun 4) takes
TIRust's double-precision integrals, converted to quadruple precision,
instead of TIRust's binary128 path. ONIA_DBG_DPTIR=1 does this for the
calls with raised powers only, =2 for every call. Everything else (the
loop coefficients, the contractions, the projections) stays in quadruple
precision, so the error left against the plain CTModeRun 4 is the error
of the double-precision integrals alone.

    hybrid.py SUBPROCESS_DIR      (then rebuild ml_points_driver)"""
import os
import re
import sys

path = os.path.join(sys.argv[1], 'TIRUST_interface.f')
src = open(path).read()
if 'ONIA_DBG_DPTIR' in src:
    sys.exit('already patched')

DECL = '''      REAL*8 ONIA_DPL(0:3,NLOOPLINE)
      COMPLEX*16 ONIA_DCOEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER ONIA_DPTIR
      CHARACTER*8 ONIA_ENV
'''


def patch(src, name, call, level):
    start = src.index('      SUBROUTINE %s(' % name)
    end = src.index('\n      END\n', start)
    body = src[start:end]
    decl_at = body.index('      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS\n')
    body = body[:decl_at] + DECL + body[decl_at:]
    hook = '''      STABLE=.FALSE.
      TIRCOEFS=0
'''
    assert body.count(hook) == 1, name
    body = body.replace(hook, hook + '''      CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_DPTIR',ONIA_ENV)
      ONIA_DPTIR=0
      IF(LEN_TRIM(ONIA_ENV).GT.0) READ(ONIA_ENV,*) ONIA_DPTIR
      IF(ONIA_DPTIR.GE.%d) THEN
        ONIA_DPL=REAL(PL,KIND=8)
        %s
        TIRCOEFS=ONIA_DCOEFS
        RETURN
      ENDIF
''' % (level, call))
    return src[:start] + body + src[end:]


src = patch(src, 'ML5_0_MP_TIRUST_TENSOR_POWERS',
            '''CALL ML5_0_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK,POWERS
     $   ,ONIA_DPL,M2L,ONIA_DCOEFS,STABLE)''', 1)
src = patch(src, 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC',
            '''CALL ML5_0_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK
     $   ,ONIA_DPL,M2L,ONIA_DCOEFS,STABLE)''', 2)
open(path, 'w').write(src)
print('patched', path)
