"""Patch an output's TIRUST_interface.f for the low-word tests (TIRust with
EmbedderOptions::low_words and the bridge entry tirust_tensor_evaluate_powers_dd):

- adds ML5_0_TIRUST_TENSOR_POWERS_XP(CTMODE,N,RANK,POWERS,PL,M2L,TIRCOEFS,
  STABLE): the double-precision raised-power entry (binary64 inputs, as
  ML5_0_TIRUST_TENSOR_POWERS) whose result RAW + RAW_LO is summed in
  quadruple precision (COMPLEX*32), so that the double-double value of the
  tier that served the call survives. ep_retarget.py retargets it to kind 10
  with the rest of the quadruple-precision path;
- ONIA_DBG_LW=1 in the quadruple-precision run (CTModeRun 4): the calls with
  raised powers take that entry with the loop momenta rounded to double (the
  hybrid test of hybrid.py with RAW + RAW_LO instead of RAW);
- ONIA_DBG_LW=2: those calls stay on TIRust's binary128 path, with their
  loop momenta rounded to double first (PL is changed in place, which is
  safe because ONIA_PW_DOUBLED passes a copy). Not usable: binary128
  refuses the relation lanes of rounded inputs (return code 420);
- ONIA_DBG_LW=3: RAW + RAW_LO for the calls TIRust served on a precision
  tier (route 2 to 4), binary128 at the unrounded inputs for the others;
  =4: the converse, RAW only for the calls served by binary64 (route 0).
  The two split the error of =1 into that of the tier-served calls (their
  inputs' rounding and the tier's precision) and that of the binary64
  calls;
- ONIA_DBG_ROUTES=FILE: one line per call of the new entry (lines, rank,
  powers, TIRust route: 0 binary64, 2 double-double, 3 256 bits, 4 512 bits,
  5 none).

    lowwords.py SUBPROCESS_DIR      (then rebuild ml_points_driver)"""
import os
import re
import sys

path = os.path.join(sys.argv[1], 'TIRUST_interface.f')
src = open(path).read()
if 'ML5_0_TIRUST_TENSOR_POWERS_XP' in src:
    sys.exit('already patched')


def routine(src, name):
    start = src.index('      SUBROUTINE %s(' % name)
    end = src.index('\n      END\n', start) + len('\n      END\n')
    return start, end


# the new entry: a copy of the double-precision raised-power routine
s, e = routine(src, 'ML5_0_TIRUST_TENSOR_POWERS')
body = src[s:e]


def sub(old, new, count=1):
    global body
    assert body.count(old) == count, old
    body = body.replace(old, new)


sub('''      SUBROUTINE ML5_0_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK
     $ ,POWERS,PL,M2L,TIRCOEFS,STABLE)''', '''      SUBROUTINE ML5_0_TIRUST_TENSOR_POWERS_XP(CTMODE,NLOOPLINE,RANK
     $ ,POWERS,PL,M2L,TIRCOEFS,STABLE)
C     lowwords.py: RAW + RAW_LO of tirust_tensor_evaluate_powers_dd,
C     summed in quadruple precision''')
sub('''      COMPLEX*16 TIRCOEFS(0:LOOPMAXCOEFS-1,3)''',
    '''      COMPLEX*32 TIRCOEFS(0:LOOPMAXCOEFS-1,3)''')
sub('''      COMPLEX*16 RAW(LOOPMAXCOEFS,4)''',
    '''      COMPLEX*16 RAW(LOOPMAXCOEFS,4),RAW_LO(LOOPMAXCOEFS,4)
      COMPLEX*32 XQ(LOOPMAXCOEFS,4)
      INTEGER ROUTE,LWROUTE
      COMMON/ONIA_LW_LAST/LWROUTE''')
call = re.search(r'      CALL TIRUST_TENSOR_EVALUATE_POWERS\(NLOOPLINE,RANK,POW,SHIFTS\n'
                 r'     \$ ,INVARIANTS,MASSES,SCALE,RAW\(1:NC,1:4\),STATUS,MAX\(0\n'
                 r'     \$ ,LOOPPOLEMODE\),TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED\)\n', body).group(0)
sub(call, '''      CALL TIRUST_TENSOR_EVALUATE_POWERS_DD(NLOOPLINE,RANK,POW,SHIFTS
     $ ,INVARIANTS,MASSES,SCALE,RAW(1:NC,1:4),RAW_LO(1:NC,1:4),ROUTE
     $ ,STATUS,MAX(0,LOOPPOLEMODE),TIRUSTCOMPUTEFINITE,CHECK_PASS
     $ ,CACHE_ENABLED)
      CALL ONIA_DBG_ROUTE(NLOOPLINE,RANK,POW,ROUTE)
      LWROUTE=ROUTE
      XQ(1:NC,1:4)=CMPLX(RAW(1:NC,1:4),KIND=16)
     $ +CMPLX(RAW_LO(1:NC,1:4),KIND=16)
''')
sub('''          TIRCOEFS(I,1)=RAW(K,4)''', '''          TIRCOEFS(I,1)=XQ(K,4)''')
sub('''        TIRCOEFS(I,2)=RAW(K,1)+RAW(K,2)
        TIRCOEFS(I,3)=RAW(K,3)''', '''        TIRCOEFS(I,2)=XQ(K,1)+XQ(K,2)
        TIRCOEFS(I,3)=XQ(K,3)''')
init = '      STABLE=.FALSE.\n      TIRCOEFS=0\n'
assert body.count(init) == 1
xp = body.replace(init, init + '      LWROUTE=5\n')

# the hybrid hook in the binary128 raised-power routine
s2, e2 = routine(src, 'ML5_0_MP_TIRUST_TENSOR_POWERS')
mp = src[s2:e2]
decl = '      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS\n'
assert mp.count(decl) == 1
mp = mp.replace(decl, '''      REAL*8 ONIA_LWPL(0:3,NLOOPLINE)
      CHARACTER*8 ONIA_LWENV
      INTEGER LWROUTE
      LOGICAL LWKEEP
      COMMON/ONIA_LW_LAST/LWROUTE
''' + decl)
hook = '''      STABLE=.FALSE.
      TIRCOEFS=0
'''
assert mp.count(hook) == 1
mp = mp.replace(hook, hook + '''      CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_LW',ONIA_LWENV)
      IF(ONIA_LWENV(1:1).EQ.'1') THEN
        ONIA_LWPL=REAL(PL,KIND=8)
        CALL ML5_0_TIRUST_TENSOR_POWERS_XP(CTMODE,NLOOPLINE,RANK,POWERS
     $   ,ONIA_LWPL,M2L,TIRCOEFS,STABLE)
        RETURN
      ELSEIF(ONIA_LWENV(1:1).EQ.'2') THEN
        PL=REAL(REAL(PL,KIND=8),KIND=16)
      ELSEIF(ONIA_LWENV(1:1).EQ.'3'.OR.ONIA_LWENV(1:1).EQ.'4') THEN
        ONIA_LWPL=REAL(PL,KIND=8)
        CALL ML5_0_TIRUST_TENSOR_POWERS_XP(CTMODE,NLOOPLINE,RANK,POWERS
     $   ,ONIA_LWPL,M2L,TIRCOEFS,STABLE)
        IF(ONIA_LWENV(1:1).EQ.'3') THEN
          LWKEEP=STABLE.AND.LWROUTE.GE.2.AND.LWROUTE.LE.4
        ELSE
          LWKEEP=STABLE.AND.LWROUTE.EQ.0
        ENDIF
        IF(LWKEEP) RETURN
        STABLE=.FALSE.
        TIRCOEFS=0
      ENDIF
''')
src = src[:s2] + mp + src[e2:]
s, e = routine(src, 'ML5_0_TIRUST_TENSOR_POWERS')
src = src[:e] + '\n' + xp + src[e:] + '''
      SUBROUTINE ONIA_DBG_ROUTE(N,RANK,POW,ROUTE)
C     lowwords.py: the TIRust route of each raised-power call
      IMPLICIT NONE
      INTEGER N,RANK,POW(N),ROUTE,I,L
      INTEGER, SAVE :: U=-1
      CHARACTER*512 FNAME
      IF (U.LT.0) THEN
        U=0
        CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_ROUTES',FNAME,L)
        IF (L.GT.0) THEN
          U=74
          OPEN(UNIT=U,FILE=FNAME(1:L),STATUS='REPLACE')
        ENDIF
      ENDIF
      IF (U.GT.0) THEN
        WRITE(U,'(A,3I4,1X,8I2)') 'R',N,RANK,ROUTE,(POW(I),I=1,N)
        FLUSH(U)
      ENDIF
      END
'''
open(path, 'w').write(src)
print('patched', path)
