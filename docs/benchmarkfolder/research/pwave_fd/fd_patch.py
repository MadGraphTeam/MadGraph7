"""Scratch: add the finite-difference test modes to a generated P-wave
onia_loop.f (prefix ML5_0_), in double and in quadruple precision.
MODE 1: real test directions perpendicular to QREST, W1 = 0, direction 3
dropped and the weights W2 frozen; MODE 2 (direction D): the frozen
W2(:,:,D) as the value weight W1, no derivative (W2 = V = 0, zero dual
parts). Usage: fd_patch.py onia_loop.f NBORNAMPS NCOMB"""
import sys
path, NB, NC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = open(path).read()
cut = s.index('      SUBROUTINE ML5_0_MP_ONIA_CG(')
dp, mp = s[:cut], s[cut:]

def rep(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), old[:90])
    return text.replace(old, new)

MODES = """      INTEGER PWT_MODE, PWT_D
      COMMON/ML5_0_PW_TESTM/PWT_MODE,PWT_D
"""
# ---- double precision
dp = rep(dp, """      DO ILZ=1,3
        CALL VXXXXX(KD,CMASS(Q1)+CMASS(Q2),ILZ-2,+1,EPSW)
        DO I=0,3
          EPSLZ(I,ILZ)=EPSW%W(I+1)%COMP(0)
        ENDDO
      ENDDO
      END""", """      DO ILZ=1,3
        CALL VXXXXX(KD,CMASS(Q1)+CMASS(Q2),ILZ-2,+1,EPSW)
        DO I=0,3
          EPSLZ(I,ILZ)=EPSW%W(I+1)%COMP(0)
        ENDDO
      ENDDO
      IF (PWT_MODE.NE.0) THEN
        CALL ML5_0_PW_TEST_DIRS(PWT_E)
        DO ILZ=1,3
          ER(1:3)=PWT_E(1:3,ILZ)
          CALL ML5_0_ONIA_BOOST_REST_VECTOR(KR,ER,EL)
          EPSLZ(0:3,ILZ)=DCMPLX(EL(0:3),0D0)
        ENDDO
      ENDIF
      END""")
dp = rep(dp, """      TYPE(DUAL) KD(0:3)
      TYPE(ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2""", """      TYPE(DUAL) KD(0:3)
      TYPE(ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2
      REAL*8 KR(0:3), ER(3), EL(0:3), PWT_E(3,3)
""" + MODES.rstrip('\n'))
dp = rep(dp, """      DO I=0,3
        KD(I)=P(I,Q1)+P(I,Q2)
      ENDDO
      DO ILZ=1,3""", """      DO I=0,3
        KD(I)=P(I,Q1)+P(I,Q2)
        KR(I)=P(I,Q1)+P(I,Q2)
      ENDDO
      DO ILZ=1,3""")
ZERO_DUAL = """      DO I=0,3
        PD(I,ONIA_Q1(ONIA_IPW))%COMP(1)=EPSLZ(I,ILZ)
        PD(I,ONIA_Q2(ONIA_IPW))%COMP(1)=-EPSLZ(I,ILZ)
        PW_EPS(I)=EPSLZ(I,ILZ)
      ENDDO"""
ZERO_DUAL_NEW = """      DO I=0,3
        PD(I,ONIA_Q1(ONIA_IPW))%COMP(1)=EPSLZ(I,ILZ)
        PD(I,ONIA_Q2(ONIA_IPW))%COMP(1)=-EPSLZ(I,ILZ)
        IF (PWT_MODE.EQ.2) THEN
          PD(I,ONIA_Q1(ONIA_IPW))%COMP(1)=0
          PD(I,ONIA_Q2(ONIA_IPW))%COMP(1)=0
        ENDIF
        PW_EPS(I)=EPSLZ(I,ILZ)
      ENDDO"""
dp = rep(dp, ZERO_DUAL, ZERO_DUAL_NEW)
dp = rep(dp, """      COMMON/ML5_0_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, J
      DO J=1,NEXTERNAL""", """      COMMON/ML5_0_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, J
""" + MODES + """      DO J=1,NEXTERNAL""")
TEST_BLOCK = """C     (test modes)
      IF (PWT_MODE.EQ.1) THEN
        WEFF(1:NBORNAMPS,1:NCOMB)=0
        W2(1:NBORNAMPS,1:NCOMB,3)=0
        DO H=1,NCOMB
          VW(1:NBORNAMPS,0:3,H)=0
          DO ILZ=1,2
            DO MU=0,3
              VW(1:NBORNAMPS,MU,H)=VW(1:NBORNAMPS,MU,H)
     &         +W2(1:NBORNAMPS,H,ILZ)*CONJG(EPSLZ(MU,ILZ))
            ENDDO
          ENDDO
        ENDDO
        %(W2)s(1:NBORNAMPS,1:NCOMB,1:3)=W2(1:NBORNAMPS,1:NCOMB,1:3)
      ELSEIF (PWT_MODE.EQ.2) THEN
        WEFF(1:NBORNAMPS,1:NCOMB)=%(W2)s(1:NBORNAMPS,1:NCOMB,PWT_D)
        W2(1:NBORNAMPS,1:NCOMB,1:3)=0
        VW(1:NBORNAMPS,0:3,1:NCOMB)=0
      ENDIF
      DO H=1,NCOMB
        FIRSTLZ(H)=0"""
dp = rep(dp, """      DO H=1,NCOMB
        FIRSTLZ(H)=0""", TEST_BLOCK % {'W2': 'PWT_W2'})
dp = rep(dp, """      INTEGER H, T, K, I, J, O, Q1, Q2, ILZ, LZ, SZ, MU
      INCLUDE 'coupl.inc'""", """      INTEGER H, T, K, I, J, O, Q1, Q2, ILZ, LZ, SZ, MU
      COMPLEX*16 PWT_W2(NBORNAMPS,NCOMB,3)
      COMMON/ML5_0_PW_TEST/PWT_W2
""" + MODES + """      INCLUDE 'coupl.inc'""")
# ---- quadruple precision
mp = rep(mp, """      DO ILZ=1,3
        CALL MP_VXXXXX(KD,CMASS(Q1)+CMASS(Q2),ILZ-2,+1,EPSW)
        DO I=0,3
          EPSLZ(I,ILZ)=EPSW%W(I+1)%COMP(0)
        ENDDO
      ENDDO
      END""", """      DO ILZ=1,3
        CALL MP_VXXXXX(KD,CMASS(Q1)+CMASS(Q2),ILZ-2,+1,EPSW)
        DO I=0,3
          EPSLZ(I,ILZ)=EPSW%W(I+1)%COMP(0)
        ENDDO
      ENDDO
      IF (PWT_MODE.NE.0) THEN
        CALL ML5_0_PW_TEST_DIRS(PWT_E)
        DO ILZ=1,3
          ER(1:3)=PWT_E(1:3,ILZ)
          CALL ML5_0_MP_ONIA_BOOST_REST_VECTOR(KR,ER,EL)
          EPSLZ(0:3,ILZ)=CMPLX(EL(0:3),0.0E0_16,KIND=16)
        ENDDO
      ENDIF
      END""")
mp = rep(mp, """      TYPE(MP_DUAL) KD(0:3)
      TYPE(MP_ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2""", """      TYPE(MP_DUAL) KD(0:3)
      TYPE(MP_ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2
      REAL*16 KR(0:3), ER(3), EL(0:3)
      REAL*8 PWT_E(3,3)
""" + MODES.rstrip('\n'))
mp = rep(mp, """      DO I=0,3
        KD(I)=P(I,Q1)+P(I,Q2)
      ENDDO
      DO ILZ=1,3""", """      DO I=0,3
        KD(I)=P(I,Q1)+P(I,Q2)
        KR(I)=P(I,Q1)+P(I,Q2)
      ENDDO
      DO ILZ=1,3""")
mp = rep(mp, ZERO_DUAL, ZERO_DUAL_NEW)
mp = rep(mp, """      COMMON/ML5_0_MP_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, J
      DO J=1,NEXTERNAL""", """      COMMON/ML5_0_MP_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, J
""" + MODES + """      DO J=1,NEXTERNAL""")
mp = rep(mp, """      DO H=1,NCOMB
        FIRSTLZ(H)=0""", TEST_BLOCK % {'W2': 'MP_PWT_W2'})
mp = rep(mp, """      INTEGER H, T, K, I, O, Q1, Q2, ILZ, LZ, SZ, MU
      INCLUDE 'coupl.inc'""", """      INTEGER H, T, K, I, O, Q1, Q2, ILZ, LZ, SZ, MU
      COMPLEX*32 MP_PWT_W2(NBORNAMPS,NCOMB,3)
      COMMON/ML5_0_MP_PW_TEST/MP_PWT_W2
""" + MODES + """      INCLUDE 'coupl.inc'""")
s = dp + mp + """
      SUBROUTINE ML5_0_PW_TEST_DIRS(E)
C     two unit vectors perpendicular to the base relative momentum, and it
      IMPLICIT NONE
      REAL*8 E(3,3), N(3), A(3), X
      COMMON/ML5_0_PW_TEST4/N
      E(1:3,3)=N(1:3)
      A=(/1D0,0D0,0D0/)
      IF (ABS(N(1)).GT.0.9D0) A=(/0D0,1D0,0D0/)
      X=SUM(A*N)
      E(1:3,1)=A-X*N
      E(1:3,1)=E(1:3,1)/SQRT(SUM(E(1:3,1)**2))
      E(1,2)=N(2)*E(3,1)-N(3)*E(2,1)
      E(2,2)=N(3)*E(1,1)-N(1)*E(3,1)
      E(3,2)=N(1)*E(2,1)-N(2)*E(1,1)
      END

      SUBROUTINE ML5_0_PW_SET_BASE(Q0)
      IMPLICIT NONE
      REAL*8 Q0(3), N(3)
      COMMON/ML5_0_PW_TEST4/N
      N=Q0/SQRT(SUM(Q0**2))
      END

      SUBROUTINE ML5_0_PW_SET_TEST(MODE,D)
      IMPLICIT NONE
      INTEGER MODE, D
""" + MODES + """      PWT_MODE=MODE
      PWT_D=D
      END
"""
open(path, 'w').write(s)
print('patched')
