"""Scratch: add the finite-difference test mode to a generated P-wave
onia_loop.f (prefix ML5_0_). MODE 1: real test directions e_d, W1 = 0 and
the weights W2 frozen; MODE 2 (direction D): the frozen W2(:,:,D) as the
value weight W1, no derivative (W2 = V = 0, zero dual parts)."""
import sys
path = sys.argv[1]
s = open(path).read()
def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)

TEST_COMMON = """      INTEGER PWT_MODE, PWT_D
      %s PWT_W2(NBORNAMPS,NCOMB,3)
      COMMON/ML5_0_PW_TEST/PWT_W2,PWT_MODE,PWT_D
"""
# directions: real unit vectors of the rest frame in test mode
rep("""      DO ILZ=1,3
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
C       two directions perpendicular to the base relative momentum
        CALL ML5_0_PW_TEST_DIRS(PWT_E)
        DO ILZ=1,3
          ER(1:3)=PWT_E(1:3,ILZ)
          CALL ML5_0_ONIA_BOOST_REST_VECTOR(KR,ER,EL)
          EPSLZ(0:3,ILZ)=DCMPLX(EL(0:3),0D0)
        ENDDO
      ENDIF
      END""")
rep("""      TYPE(DUAL) KD(0:3)
      TYPE(ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2""", """      TYPE(DUAL) KD(0:3)
      TYPE(ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2
      REAL*8 KR(0:3), ER(3), EL(0:3), PWT_E(3,3)
      INTEGER PWT_MODE, PWT_D
      COMPLEX*16 PWT_W2(%d,%d,3)
      COMMON/ML5_0_PW_TEST/PWT_W2,PWT_MODE,PWT_D""" % (int(sys.argv[2]), int(sys.argv[3])))
rep("""      DO I=0,3
        KD(I)=P(I,Q1)+P(I,Q2)
      ENDDO
      DO ILZ=1,3""", """      DO I=0,3
        KD(I)=P(I,Q1)+P(I,Q2)
        KR(I)=P(I,Q1)+P(I,Q2)
      ENDDO
      DO ILZ=1,3""")
# zero dual parts in mode 2
rep("""      DO I=0,3
        PD(I,ONIA_Q1(ONIA_IPW))%COMP(1)=EPSLZ(I,ILZ)
        PD(I,ONIA_Q2(ONIA_IPW))%COMP(1)=-EPSLZ(I,ILZ)
        PW_EPS(I)=EPSLZ(I,ILZ)
      ENDDO""", """      DO I=0,3
        PD(I,ONIA_Q1(ONIA_IPW))%COMP(1)=EPSLZ(I,ILZ)
        PD(I,ONIA_Q2(ONIA_IPW))%COMP(1)=-EPSLZ(I,ILZ)
        IF (PWT_MODE.EQ.2) THEN
          PD(I,ONIA_Q1(ONIA_IPW))%COMP(1)=(0D0,0D0)
          PD(I,ONIA_Q2(ONIA_IPW))%COMP(1)=(0D0,0D0)
        ENDIF
        PW_EPS(I)=EPSLZ(I,ILZ)
      ENDDO""")
rep("""      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_dp_format)s PW_EPS(0:3)
      COMMON/ML5_0_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, J
      DO J=1,NEXTERNAL""".replace('%(complex_dp_format)s', 'COMPLEX*16'), """      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      COMPLEX*16 PW_EPS(0:3)
      COMMON/ML5_0_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, J
      INTEGER PWT_MODE, PWT_D
      COMPLEX*16 PWT_W2(%d,%d,3)
      COMMON/ML5_0_PW_TEST/PWT_W2,PWT_MODE,PWT_D
      DO J=1,NEXTERNAL""" % (int(sys.argv[2]), int(sys.argv[3])))
# the weights in test mode
rep("""      DO H=1,NCOMB
        FIRSTLZ(H)=0""", """C     (test modes)
      IF (PWT_MODE.EQ.1) THEN
        WEFF(1:NBORNAMPS,1:NCOMB)=(0D0,0D0)
        W2(1:NBORNAMPS,1:NCOMB,3)=(0D0,0D0)
        DO H=1,NCOMB
          VW(1:NBORNAMPS,0:3,H)=(0D0,0D0)
          DO ILZ=1,2
            DO MU=0,3
              VW(1:NBORNAMPS,MU,H)=VW(1:NBORNAMPS,MU,H)
     &         +W2(1:NBORNAMPS,H,ILZ)*DCONJG(EPSLZ(MU,ILZ))
            ENDDO
          ENDDO
        ENDDO
        PWT_W2(1:NBORNAMPS,1:NCOMB,1:3)=W2(1:NBORNAMPS,1:NCOMB,1:3)
      ELSEIF (PWT_MODE.EQ.2) THEN
        WEFF(1:NBORNAMPS,1:NCOMB)=PWT_W2(1:NBORNAMPS,1:NCOMB,PWT_D)
        W2(1:NBORNAMPS,1:NCOMB,1:3)=(0D0,0D0)
        VW(1:NBORNAMPS,0:3,1:NCOMB)=(0D0,0D0)
      ENDIF
      DO H=1,NCOMB
        FIRSTLZ(H)=0""")
rep("""      INTEGER H, T, K, I, J, O, Q1, Q2, ILZ, LZ, SZ, MU
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL ML5_0_PW_DUAL_INIT()""", """      INTEGER H, T, K, I, J, O, Q1, Q2, ILZ, LZ, SZ, MU
      INTEGER PWT_MODE, PWT_D
      COMPLEX*16 PWT_W2(NBORNAMPS,NCOMB,3)
      COMMON/ML5_0_PW_TEST/PWT_W2,PWT_MODE,PWT_D
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL ML5_0_PW_DUAL_INIT()""")
s += """
      SUBROUTINE ML5_0_PW_TEST_DIRS(E)
C     two unit vectors perpendicular to the relative momentum QREST, and it
      IMPLICIT NONE
      REAL*8 E(3,3), QREST(3), N(3), A(3), X
      COMMON/ML5_0_ONIA_RELMOM/QREST
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
      INTEGER PWT_MODE, PWT_D
      COMPLEX*16 PWT_W2(%d,%d,3)
      COMMON/ML5_0_PW_TEST/PWT_W2,PWT_MODE,PWT_D
      PWT_MODE=MODE
      PWT_D=D
      END
""" % (int(sys.argv[2]), int(sys.argv[3]))
open(path, 'w').write(s)
print('patched')
