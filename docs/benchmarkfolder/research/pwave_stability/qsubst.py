"""Patch an output's onia_loop.f for the converse of hybrid.py: the
double-precision run with the raised-power integrals at full quadruple
precision.
- ONIA_DBG_QDUMP=FILE, in the quadruple-precision run (CTModeRun 4): writes
  the binary128 array of every call with a doubled line, in call order.
- ONIA_DBG_QSUBST=FILE with ONIA_DBG_QMODE=M, in the double-precision
  primary pass (CTModeRun 1): reads them back in the same order (the
  header of each call is checked) and contracts the doubled-line integrals
  in quadruple precision, then adds the result to the double-precision
  loop amplitude.
  - M=1: the numerator QC formed in double precision, as in production;
  - M=2: QC formed in quadruple precision from the double-precision loop
    coefficients;
  - M=3: as 1, with the arrays rounded to double first (only the
    summation in quadruple precision).
Everything else stays in double precision: the loop coefficients, the
calls with unit powers, the projections.

    qsubst.py SUBPROCESS_DIR      (then rebuild ml_points_driver)"""
import os
import sys

path = os.path.join(sys.argv[1], 'onia_loop.f')
src = open(path).read()
if 'ONIA_DBG_QSUBST' in src:
    sys.exit('already patched')


def routine(src, name):
    start = src.index('      SUBROUTINE %s(' % name)
    end = src.index('\n      END\n', start)
    return start, end


# quadruple precision: dump the arrays
s, e = routine(src, 'ML5_0_MP_ONIA_PW_DOUBLED')
body = src[s:e]
call = '''        CALL ML5_0_MP_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK+1
     $   ,POWERS,PLC,M2LC,
     &   TIRCOEFS,OK)
        IF (.NOT.OK) THEN
          STABLE=.FALSE.
          PW_REFUSED=.TRUE.
          RETURN
        ENDIF
'''
assert body.count(call) == 1
body = body.replace(call, call + '''        CALL ONIA_QDUMP(NLOOPLINE,RANK+1,J,NC1,TIRCOEFS)
''')
src = src[:s] + body + src[e:]

# double precision: read them back and contract in quadruple precision
s, e = routine(src, 'ML5_0_ONIA_PW_DOUBLED')
body = src[s:e]
decl = '''      LOGICAL OK
'''
assert body.count(decl) == 1
body = body.replace(decl, decl + '''      COMPLEX*32 QCQ(0:LOOPMAXCOEFS-1), TQ(0:LOOPMAXCOEFS-1,3), RQ, PCQ
      INTEGER ONIA_QMODE
      CHARACTER*8 ONIA_ENV
''')
contract = '''        DO I=1,3
          DO K=0,NC1-1
            RES(I)=RES(I)+QC(K)*TIRCOEFS(K,I)
          ENDDO
        ENDDO
'''
assert body.count(contract) == 1
body = body.replace(contract, '''        CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_QMODE',ONIA_ENV)
        ONIA_QMODE=0
        IF(LEN_TRIM(ONIA_ENV).GT.0) READ(ONIA_ENV,*) ONIA_QMODE
        IF (ONIA_QMODE.GT.0) THEN
          CALL ONIA_QREAD(NLOOPLINE,RANK+1,J,NC1,TQ)
          IF (ONIA_QMODE.EQ.3) THEN
            DO I=1,3
              DO K=0,NC1-1
                TQ(K,I)=CMPLX(CMPLX(TQ(K,I),KIND=8),KIND=16)
              ENDDO
            ENDDO
          ENDIF
          DO K=0,NC1-1
            QCQ(K)=QC(K)
          ENDDO
          IF (ONIA_QMODE.EQ.2) THEN
            DO K=0,NC1-1
              QCQ(K)=(0.0E0_16,0.0E0_16)
            ENDDO
            DO MU=0,3
              DO K=0,NC-1
                PCQ=PCOEFS(K,I_SQSO,I_LOOPGROUP,MU)
                IF (PCQ.EQ.(0.0E0_16,0.0E0_16)) CYCLE
                QCQ(K)=QCQ(K)-2.0E0_16*REAL(C,KIND=16)*REAL(G(MU),KIND=16)
     $           *REAL(SHIFT(MU,J),KIND=16)*PCQ
                KK=COMB_COEF_POS(K,MU+1)
                QCQ(KK)=QCQ(KK)+2.0E0_16*REAL(C,KIND=16)
     $           *REAL(G(MU),KIND=16)*PCQ
              ENDDO
            ENDDO
          ENDIF
          DO I=1,3
            RQ=(0.0E0_16,0.0E0_16)
            DO K=0,NC1-1
              RQ=RQ+QCQ(K)*TQ(K,I)
            ENDDO
            RES(I)=RES(I)+CMPLX(RQ,KIND=8)
          ENDDO
        ELSE
''' + contract + '''        ENDIF
''')
src = src[:s] + body + src[e:]

src += '''
C     ---- qsubst.py: dump and read the doubled-line arrays at full
C     ---- quadruple precision
      SUBROUTINE ONIA_QDUMP(N,RANK,J,NC,T)
      IMPLICIT NONE
      INTEGER N,RANK,J,NC,K,I,L
      COMPLEX*32 T(0:NC-1,3)
      CHARACTER*512 FNAME
      INTEGER U
      LOGICAL OPENED
      DATA OPENED/.FALSE./
      SAVE OPENED,U
      CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_QDUMP',FNAME,L)
      IF (L.EQ.0) RETURN
      IF (.NOT.OPENED) THEN
        OPEN(NEWUNIT=U,FILE=FNAME(1:L),STATUS='REPLACE')
        OPENED=.TRUE.
      ENDIF
      WRITE(U,'(A,4I6)') 'H',N,RANK,J,NC
      DO I=1,3
        DO K=0,NC-1
          WRITE(U,'(2ES48.38E4)') REAL(T(K,I)),AIMAG(T(K,I))
        ENDDO
      ENDDO
      FLUSH(U)
      END

      SUBROUTINE ONIA_QREAD(N,RANK,J,NC,T)
      IMPLICIT NONE
      INTEGER N,RANK,J,NC,K,I,L,N2,R2,J2,NC2
      COMPLEX*32 T(0:NC-1,3)
      REAL*16 RE,IM
      CHARACTER*512 FNAME
      CHARACTER*1 TAG
      INTEGER U
      LOGICAL OPENED
      DATA OPENED/.FALSE./
      SAVE OPENED,U
      IF (.NOT.OPENED) THEN
        CALL GET_ENVIRONMENT_VARIABLE('ONIA_DBG_QSUBST',FNAME,L)
        OPEN(NEWUNIT=U,FILE=FNAME(1:L),STATUS='OLD')
        OPENED=.TRUE.
      ENDIF
      READ(U,'(A,4I6)') TAG,N2,R2,J2,NC2
      IF (TAG.NE.'H'.OR.N2.NE.N.OR.R2.NE.RANK.OR.J2.NE.J.OR.NC2.NE.NC)
     $  THEN
        WRITE(*,*) 'ERROR:: QSUBST call mismatch',N,RANK,J,NC,' file',
     $   N2,R2,J2,NC2
        STOP 3
      ENDIF
      DO I=1,3
        DO K=0,NC-1
          READ(U,*) RE,IM
          T(K,I)=CMPLX(RE,IM,KIND=16)
        ENDDO
      ENDDO
      END
'''
open(path, 'w').write(src)
print('patched', path)
