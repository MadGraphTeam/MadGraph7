"""Instrument TIRUST_interface.f of a standalone MadLoop output (diagnosis only):

    python3 instrument.py OUTPUT/SubProcesses/P*/TIRUST_interface.f

Every TIRust call (DP and QP, plain and with propagator powers) is numbered
in the order of the run. Environment switches of the instrumented code:
- ONIA_DBG_DUMP=FILE: the returned arrays (as doubles, hex) and a timestamp;
- ONIA_DBG_SUBST=FILE: replaces the arrays of the listed DP calls (the
  format written by run.py: count, then index, coefficients and rows);
- ONIA_DBG_INP=FILE: the inputs of the DP calls with powers;
- ONIA_DBG_NOSHIFT=1: no invariant shift in TIRust's check pass (CTMODE 2);
- ONIA_DBG_MIXED=N: DP calls with powers and at least N lines are evaluated
  by TIRust's QP path from the DP inputs (the prototype of the report).
Rebuild the driver afterwards (onia_loop_check.build_driver)."""
import re, sys

EXTRA = """

      SUBROUTINE ONIA_DBG_INP(CTMODE,N,RANK,POW,SHIFTS,INV,MASSES,SCALE)
      IMPLICIT NONE
      INTEGER CTMODE,N,RANK,POW(N),I,J,L
      REAL*8 SHIFTS(0:3,N),INV(N,N),SCALE
      COMPLEX*16 MASSES(N)
      INTEGER, SAVE :: U=-1, NCALL=0
      CHARACTER*512 FNAME
      IF (U.LT.0) THEN
        U=0
        CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_INP',FNAME,L)
        IF (L.GT.0) THEN
          U=76
          OPEN(UNIT=U,FILE=FNAME(1:L),STATUS='UNKNOWN')
        ENDIF
      ENDIF
      IF (U.GT.0) THEN
        WRITE(U,'(A,12I6)') 'P',NCALL,CTMODE,N,RANK,(POW(I),I=1,N)
        WRITE(U,'(A,ES25.17)') 'MU2',SCALE
        DO J=1,N
          WRITE(U,'(A,I3,2ES25.17)') 'M',J,MASSES(J)
        ENDDO
        DO J=1,N
          WRITE(U,'(A,I3,8ES25.17)') 'I',J,(INV(J,I),I=1,N)
        ENDDO
        DO J=1,N
          WRITE(U,'(A,I3,4ES25.17)') 'S',J,(SHIFTS(I,J),I=0,3)
        ENDDO
        FLUSH(U)
      ENDIF
      NCALL=NCALL+1
      END

      LOGICAL FUNCTION ONIA_DBG_NOSHIFT()
      IMPLICIT NONE
      INTEGER, SAVE :: V=-1
      INTEGER L
      CHARACTER*16 VAL
      IF (V.LT.0) THEN
        V=0
        CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_NOSHIFT',VAL,L)
        IF (L.GT.0) V=1
      ENDIF
      ONIA_DBG_NOSHIFT=V.EQ.1
      END

      INTEGER FUNCTION ONIA_DBG_MIXED_N()
      IMPLICIT NONE
      INTEGER, SAVE :: V=-1
      INTEGER L
      CHARACTER*16 VAL
      IF (V.LT.0) THEN
        V=1000
        CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_MIXED',VAL,L)
        IF (L.GT.0) READ(VAL(1:L),*) V
      ENDIF
      ONIA_DBG_MIXED_N=V
      END
"""
path = sys.argv[1]
src = open(path).read().split('\n')
out, i, sub = [], 0, None
while i < len(src):
    line = src[i]
    m = re.match(r'      SUBROUTINE ML5_0_(MP_)?TIRUST_TENSOR_(DIAGNOSTIC|POWERS)\(', line)
    if m:
        sub = ('QP' if m.group(1) else 'DP', 1 if m.group(2) == 'POWERS' else 0)
    if line.startswith('      SUBROUTINE') and not m:
        sub = None
    out.append(line)
    if sub and re.match(r'      CALL TIRUST_TENSOR_EVALUATE(_POWERS)?\(', line):
        while src[i + 1].startswith('     $'):
            i += 1
            out.append(src[i])
        out.append('      CALL ONIA_DBG_%s(%d,NLOOPLINE,RANK,NC,STATUS,RAW)' % (sub[0], sub[1]))
    i += 1
out.append('''
      SUBROUTINE ONIA_DBG_DP(KIND,N,RANK,NC,STATUS,RAW)
      USE MTIRUST, ONLY: TIRUST_SUCCESS
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER KIND,N,RANK,NC,STATUS
      COMPLEX*16 RAW(LOOPMAXCOEFS,4)
      CALL ONIA_DBG_WRITE('DP',KIND,N,RANK,NC,STATUS,RAW)
      CALL ONIA_DBG_SUBST(NC,RAW,STATUS)
      END

      SUBROUTINE ONIA_DBG_QP(KIND,N,RANK,NC,STATUS,RAW)
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER KIND,N,RANK,NC,STATUS,I,J
      COMPLEX*32 RAW(LOOPMAXCOEFS,4)
      COMPLEX*16 R(LOOPMAXCOEFS,4)
      DO J=1,4
        DO I=1,NC
          R(I,J)=DCMPLX(DBLE(REAL(RAW(I,J))),DBLE(AIMAG(RAW(I,J))))
        ENDDO
      ENDDO
      CALL ONIA_DBG_WRITE('QP',KIND,N,RANK,NC,STATUS,R)
      END

      SUBROUTINE ONIA_DBG_WRITE(PREC,KIND,N,RANK,NC,STATUS,RAW)
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      CHARACTER*2 PREC
      INTEGER KIND,N,RANK,NC,STATUS,I,J,L
      COMPLEX*16 RAW(LOOPMAXCOEFS,4)
      INTEGER, SAVE :: U=-1, NCALL=0
      INTEGER*8 TCNT,TRATE
      CHARACTER*512 FNAME
      IF (U.LT.0) THEN
        U=0
        CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_DUMP',FNAME,L)
        IF (L.GT.0) THEN
          U=79
          OPEN(UNIT=U,FILE=FNAME(1:L),STATUS='UNKNOWN')
        ENDIF
      ENDIF
      IF (U.GT.0) THEN
        CALL SYSTEM_CLOCK(TCNT,TRATE)
        WRITE(U,'(A,1X,A,6I8,1X,F22.9)') 'C',PREC,NCALL,KIND,N,RANK,NC,
     $   STATUS,DBLE(TCNT)/DBLE(TRATE)
        DO I=1,NC
          WRITE(U,'(8(Z16.16,1X))') (REAL(RAW(I,J)),AIMAG(RAW(I,J)),J=1,4)
        ENDDO
        FLUSH(U)
      ENDIF
      NCALL=NCALL+1
      END

      SUBROUTINE ONIA_DBG_SUBST(NC,RAW,STATUS)
      USE MTIRUST, ONLY: TIRUST_SUCCESS
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NC, STATUS
      COMPLEX*16 RAW(LOOPMAXCOEFS,4)
      INTEGER, SAVE :: NCALL=0, NSUB=-1
      INTEGER, ALLOCATABLE, SAVE :: SIDX(:), SNC(:)
      COMPLEX*16, ALLOCATABLE, SAVE :: SRAW(:,:,:)
      CHARACTER*512 FNAME
      INTEGER L, I, K, J, U
      INTEGER*8 B(8)
      REAL*8 V(8)
      IF (NSUB.LT.0) THEN
        NSUB=0
        CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_SUBST',FNAME,L)
        IF (L.GT.0) THEN
          U=78
          OPEN(UNIT=U,FILE=FNAME(1:L),STATUS='OLD')
          READ(U,*) NSUB
          ALLOCATE(SIDX(MAX(NSUB,1)),SNC(MAX(NSUB,1)))
          ALLOCATE(SRAW(LOOPMAXCOEFS,4,MAX(NSUB,1)))
          DO I=1,NSUB
            READ(U,*) SIDX(I), SNC(I)
            DO K=1,SNC(I)
              READ(U,'(8(Z16,1X))') B
              V=TRANSFER(B,V)
              DO J=1,4
                SRAW(K,J,I)=DCMPLX(V(2*J-1),V(2*J))
              ENDDO
            ENDDO
          ENDDO
          CLOSE(U)
        ENDIF
      ENDIF
      DO I=1,NSUB
        IF (SIDX(I).EQ.NCALL) THEN
          RAW(1:SNC(I),1:4)=SRAW(1:SNC(I),1:4,I)
          STATUS=TIRUST_SUCCESS
        ENDIF
      ENDDO
      NCALL=NCALL+1
      END''')
text = '\n'.join(out)
text = text.replace("      CALL ONIA_DBG_DP(1,NLOOPLINE,RANK,NC,STATUS,RAW)\n",
    "      CALL ONIA_DBG_DP(1,NLOOPLINE,RANK,NC,STATUS,RAW)\n"
    "      CALL ONIA_DBG_INP(CTMODE,NLOOPLINE,RANK,POW,SHIFTS,INVARIANTS,MASSES,\n"
    "     $ SCALE)\n")
text = text.replace("      CHECK_PASS=CTMODE.EQ.2.OR.CTMODE.EQ.5\n",
    "      CHECK_PASS=CTMODE.EQ.2.OR.CTMODE.EQ.5\n      IF (ONIA_DBG_NOSHIFT()) CHECK_PASS=.FALSE.\n")
text = text.replace("      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS\n",
    "      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS\n      LOGICAL ONIA_DBG_NOSHIFT\n")
start = text.index('      SUBROUTINE ML5_0_TIRUST_TENSOR_POWERS(')
end = text.index('      SUBROUTINE ML5_0_TIRUSTLOOP(')
body = text[start:end]
body = body.replace("      LOGICAL ONIA_DBG_NOSHIFT\n", "      LOGICAL ONIA_DBG_NOSHIFT\n      INTEGER ONIA_DBG_MIXED_N\n"
    "      REAL*16 MP_PL(0:3,NLOOPLINE)\n      COMPLEX*32 MP_TIRCOEFS(0:LOOPMAXCOEFS-1,3)\n", 1)
guard = re.search(r"      IF\(RANK\.LT\.0\.OR\.RANK\.GT\.\d\) RETURN\n", body).group(0)
body = body.replace(guard, guard +
    "      IF(NLOOPLINE.GE.ONIA_DBG_MIXED_N()) THEN\n"
    "        MP_PL=REAL(PL,KIND=16)\n"
    "        CALL ML5_0_MP_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK,POWERS\n"
    "     $   ,MP_PL,M2L,MP_TIRCOEFS,STABLE)\n"
    "        TIRCOEFS=CMPLX(MP_TIRCOEFS,KIND=8)\n"
    "        RETURN\n"
    "      ENDIF\n", 1)
assert 'ONIA_DBG_MIXED_N()) THEN' in body and 'LOGICAL ONIA_DBG_NOSHIFT' in body
text = text[:start] + body + text[end:] + EXTRA
open(path, 'w').write(text)
EXTRA_UNUSED = None
print('instrumented', sum(1 for l in out if 'CALL ONIA_DBG_' in l and 'SUBROUTINE' not in l and 'WRITE' not in l and 'SUBST(' not in l), 'calls')
