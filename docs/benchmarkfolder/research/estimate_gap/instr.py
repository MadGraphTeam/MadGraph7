"""Instrument a generated output (in place) for the estimate-gap traces:
- SPLITDBG: per pass the counterterm part and every loop group (loop_matrix.f);
- QSBCMP: TIR_QCMP_N=N (-1 all): each double-precision TIRust call also in quadruple
  precision from the same inputs, the relative difference printed; TIR_QSUBST_USE=1 substitutes;
- LCQ / LC_SUBST: the QP pass prints the loop coefficients of every group (LCQ_DUMP=1); the DP
  pass replaces those of the groups listed in lcs_groups.txt by them (LC_SUBST=1).
    instr.py OUTDIR"""
import os, re, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
pdir = olc.subprocess_dir(sys.argv[1])
# loop_matrix.f
p = os.path.join(pdir, 'loop_matrix.f'); s = open(p).read()
if 'SPLITDBG' not in s:
    old = "C     Make sure that no NaN is present in the result\n"
    s = s.replace(old, """      WRITE(*,'(A,I3,4ES26.17)') 'SPLITDBG CT',CTMODE,BUFFR_BIS(1,0),
     $ BUFFR_BIS(2,0),BUFFR_BIS(3,0),ANS(1,0)
      DO I=1,NLOOPGROUPS
        WRITE(*,'(A,I3,I6,4ES26.17)') 'SPLITDBG LG',CTMODE,I,
     $   DBLE(LOOPRES(1,1,I)),DIMAG(LOOPRES(1,1,I)),
     $   DBLE(LOOPRES(2,1,I)),DBLE(LOOPRES(3,1,I))
      ENDDO
""" + old, 1)
    open(p, 'w').write(s)
# TIRUST_interface.f: QSBCMP in the DP routine, LCQ in the MP loop routine
p = os.path.join(pdir, 'TIRUST_interface.f'); s = open(p).read()
if 'TIR_QCMP_N' not in s:
    i = s.index('      SUBROUTINE ML5_0_TIRUST_TENSOR_DIAGNOSTIC(')
    j = s.index('\n      END\n', i)
    body = s[i:j]
    a = "      INCLUDE 'coupl.inc'\n"
    body = body.replace(a, """      REAL*16 QSB_PL(0:3,NLOOPLINE)
      COMPLEX*32 QSB_COEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER QSB_N,QSB_USE,QI,QJ
      CHARACTER*16 QSB_ENV
      LOGICAL QSB_INIT,QSB_ON,QSB_OK
      REAL*8 QSB_D(3),QSB_R(3)
      SAVE QSB_N,QSB_USE,QSB_INIT
      DATA QSB_INIT/.FALSE./
""" + a, 1)
    a = "      STABLE=.FALSE.\n      TIRCOEFS=0\n"
    body = body.replace(a, a + """      IF(.NOT.QSB_INIT) THEN
        QSB_N=0
        QSB_USE=0
        QSB_ENV=' '
        CALL GET_ENVIRONMENT_VARIABLE('TIR_QCMP_N',QSB_ENV)
        IF(LEN_TRIM(QSB_ENV).GT.0) READ(QSB_ENV,*) QSB_N
        QSB_ENV=' '
        CALL GET_ENVIRONMENT_VARIABLE('TIR_QSUBST_USE',QSB_ENV)
        IF(LEN_TRIM(QSB_ENV).GT.0) READ(QSB_ENV,*) QSB_USE
        QSB_INIT=.TRUE.
      ENDIF
      QSB_ON=QSB_N.EQ.-1.OR.QSB_N.EQ.NLOOPLINE
      IF(QSB_ON) THEN
        QSB_PL=REAL(PL,KIND=16)
        CALL ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,
     $   QSB_PL,M2L,QSB_COEFS,QSB_OK)
        STABLE=.FALSE.
        TIRCOEFS=0
      ENDIF
""", 1)
    a = "      STABLE=.TRUE."
    assert body.endswith(a)
    body += """
      IF(QSB_ON) THEN
        QSB_D=0
        QSB_R=0
        DO QI=0,NC-1
          DO QJ=1,3
            QSB_D(QJ)=MAX(QSB_D(QJ),DBLE(ABS(TIRCOEFS(QI,QJ)
     $       -CMPLX(QSB_COEFS(QI,QJ),KIND=8))))
            QSB_R(QJ)=MAX(QSB_R(QJ),DBLE(ABS(QSB_COEFS(QI,QJ))))
          ENDDO
        ENDDO
        WRITE(*,'(A,4I4,L2,6ES12.3)') 'QSBCMP',CTMODE,NLOOPLINE,RANK,
     $   NC,QSB_OK,(QSB_D(QJ)/MAX(QSB_R(QJ),1D-300),QJ=1,3),QSB_R
        IF(QSB_USE.EQ.1.AND.QSB_OK) THEN
          DO QI=0,NC-1
            DO QJ=1,3
              TIRCOEFS(QI,QJ)=CMPLX(QSB_COEFS(QI,QJ),KIND=8)
            ENDDO
          ENDDO
        ENDIF
      ENDIF"""
    s = s[:i] + body + s[j:]
    a = """      RESULT=0
      DO I=1,3
        DO J=0,NCOEF_R(RANK)-1
          RESULT(I)=RESULT(I)+LOOPCOEFS(J,SQSOINDEX,ID)*COEFS(J,I)"""
    assert s.count(a) == 1
    s = s.replace(a, """      DO J=0,NCOEF_R(RANK)-1
        WRITE(*,'(A,2I5,2ES46.36)') 'LCQ',ID,J,
     $   REAL(LOOPCOEFS(J,SQSOINDEX,ID),KIND=16),
     $   AIMAG(LOOPCOEFS(J,SQSOINDEX,ID))
      ENDDO
""" + a)
    open(p, 'w').write(s)
# TIR_interface.f: LC_SUBST for the groups of lcs_groups.txt
p = os.path.join(pdir, 'TIR_interface.f'); s = open(p).read()
if 'LC_SUBST' not in s:
    a = "      COMMON/ML5_0_LCOEFS/LOOPCOEFS\n"
    s = s.replace(a, a + """      COMPLEX*32 LCS_Q(0:LOOPMAXCOEFS-1)
      REAL*16 LCS_RE,LCS_IM
      INTEGER LCS_MODE,LCS_ID,LCS_J,LCS_K,LCS_G(1000),LCS_NG
      CHARACTER*16 LCS_ENV
      LOGICAL LCS_INIT,LCS_HIT
      SAVE LCS_MODE,LCS_INIT,LCS_G,LCS_NG
      DATA LCS_INIT/.FALSE./
""", 1)
    m = re.search(r"( +)DO I=1,3\n +RES\(I\)=\(0\.0D0,0\.0D0\)\n +DO J=0,NLOOPCOEFS-1\n +RES\(I\)=RES\(I\)\+LOOPCOEFS\(J,I_SQSO,I_LOOPGROUP\)\*TIRCOEFS\(J,I\)\n +ENDDO\n +ENDDO\n", s)
    assert m
    add = """      IF(.NOT.LCS_INIT) THEN
        LCS_MODE=0
        LCS_NG=0
        LCS_ENV=' '
        CALL GET_ENVIRONMENT_VARIABLE('LC_SUBST',LCS_ENV)
        IF(LEN_TRIM(LCS_ENV).GT.0) READ(LCS_ENV,*) LCS_MODE
        IF(LCS_MODE.GT.0) THEN
          OPEN(78,FILE='lcs_groups.txt',STATUS='OLD')
          DO LCS_K=1,1000
            READ(78,*,END=92) LCS_G(LCS_K)
            LCS_NG=LCS_K
          ENDDO
 92       CLOSE(78)
        ENDIF
        LCS_INIT=.TRUE.
      ENDIF
      LCS_HIT=.FALSE.
      DO LCS_K=1,LCS_NG
        IF(LCS_G(LCS_K).EQ.I_LOOPGROUP) LCS_HIT=.TRUE.
      ENDDO
      IF(LCS_HIT) THEN
        LCS_Q=0
        OPEN(77,FILE='lcq.txt',STATUS='OLD')
        DO LCS_K=1,10000000
          READ(77,*,END=91) LCS_ID,LCS_J,LCS_RE,LCS_IM
          IF(LCS_ID.EQ.I_LOOPGROUP) LCS_Q(LCS_J)=CMPLX(LCS_RE,LCS_IM
     $     ,KIND=16)
        ENDDO
 91     CLOSE(77)
        DO I=1,3
          RES(I)=(0.0D0,0.0D0)
          DO J=0,NLOOPCOEFS-1
            RES(I)=RES(I)+CMPLX(LCS_Q(J),KIND=8)*TIRCOEFS(J,I)
          ENDDO
        ENDDO
      ENDIF
"""
    s = s[:m.end()] + add + s[m.end():]
    open(p, 'w').write(s)
olc.build_driver(pdir)
print('instrumented', pdir)
