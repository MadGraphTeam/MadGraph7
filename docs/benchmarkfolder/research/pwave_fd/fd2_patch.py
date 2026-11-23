"""Scratch: add the finite-difference test modes to a generated onia_loop.f
with two P-wave bound states (prefix ML5_0_, double precision only; the
two-P-wave outputs have no quadruple-precision path yet).

- Each P-wave state gets its own relative momentum (ML5_0_PW2_SET_RELMOM:
  rest-frame 3-vectors of ONIA_IPW and ONIA_IPW2).
- PWT_MODE 1: real test directions, perpendicular to the base relative
  momentum of each state (ML5_0_PW2_SET_BASE) for L_z index 1, 2 and along
  it for 3; W0 = W1 = W2 = 0 and the pairs with a direction 3 dropped, the
  mixed weights W3 frozen (PWT_W3). MadLoop then returns
  sum_(D1,D2) conj(W3(D1,D2)) d^2 M / d eps1(D1) d eps2(D2).
- PWT_MODE 2 (pair D): the frozen W3(:,:,D) as the value weight W0, every
  other weight zero and no dual parts: sum conj(W3(D)) M(q1, q2).
Usage: fd2_patch.py onia_loop.f NBORNAMPS NCOMB"""
import sys
path, NB, NC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = open(path).read()
assert 'PWT_MODE' not in s, 'already patched'


def routine(text, name):
    a = text.index('      SUBROUTINE %s(' % name)
    b = text.index('\n      END\n', a) + len('\n      END\n')
    return a, b


def rep(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), old[:100])
    return text.replace(old, new)


def patch(name, fn):
    global s
    a, b = routine(s, name)
    s = s[:a] + fn(s[a:b]) + s[b:]


MODES = """      INTEGER PWT_MODE, PWT_D
      COMMON/ML5_0_PW_TESTM/PWT_MODE,PWT_D
"""


def constituents(r):
    r = rep(r, "      COMMON/ML5_0_ONIA_RELMOM/QREST\n",
            "      COMMON/ML5_0_ONIA_RELMOM/QREST\n      REAL*8 QR2(3,2), QQ(3)\n"
            "      COMMON/ML5_0_PW2_RELMOM/QR2\n")
    old = """      IF (ANY(QREST.NE.0D0)) THEN
        DO K=1,ONIA_NONIA
          CALL ML5_0_ONIA_BOOST_REST_VECTOR(
     &     P_PHYS(0,ONIA_MAPPING(ONIA_Q1(K))),QREST,QLAB)
          P(0:3,ONIA_Q1(K))=P(0:3,ONIA_Q1(K))+QLAB(0:3)
          P(0:3,ONIA_Q2(K))=P(0:3,ONIA_Q2(K))-QLAB(0:3)
        ENDDO
      ENDIF
"""
    new = """      IF (ANY(QR2.NE.0D0)) THEN
        DO K=1,ONIA_NONIA
          IF (K.EQ.ONIA_IPW) THEN
            QQ(1:3)=QR2(1:3,1)
          ELSEIF (K.EQ.ONIA_IPW2) THEN
            QQ(1:3)=QR2(1:3,2)
          ELSE
            CYCLE
          ENDIF
          CALL ML5_0_ONIA_BOOST_REST_VECTOR(
     &     P_PHYS(0,ONIA_MAPPING(ONIA_Q1(K))),QQ,QLAB)
          P(0:3,ONIA_Q1(K))=P(0:3,ONIA_Q1(K))+QLAB(0:3)
          P(0:3,ONIA_Q2(K))=P(0:3,ONIA_Q2(K))-QLAB(0:3)
        ENDDO
      ENDIF
"""
    return rep(r, old, new)


patch('ML5_0_ONIA_CONSTITUENT_MOMENTA', constituents)


def epsilons(r):
    r = rep(r, "      INTEGER I, ILZ, Q1, Q2, S, K\n",
            "      INTEGER I, ILZ, Q1, Q2, S, K\n      REAL*8 KR(0:3), ER(3), EL(0:3), E(3,3)\n" + MODES)
    tail = "        ENDDO\n      ENDDO\n      END\n"
    assert r.endswith(tail)
    return r[:-len("      END\n")] + """      IF (PWT_MODE.NE.0) THEN
        DO S=1,2
          IF (S.EQ.1) THEN
            K=ONIA_IPW
          ELSE
            K=ONIA_IPW2
          ENDIF
          KR(0:3)=P(0:3,ONIA_Q1(K))+P(0:3,ONIA_Q2(K))
          CALL ML5_0_PW2_TEST_DIRS(S,E)
          DO ILZ=1,3
            ER(1:3)=E(1:3,ILZ)
            CALL ML5_0_ONIA_BOOST_REST_VECTOR(KR,ER,EL)
            IF (S.EQ.1) THEN
              EPS1(0:3,ILZ)=DCMPLX(EL(0:3),0D0)
            ELSE
              EPS2(0:3,ILZ)=DCMPLX(EL(0:3),0D0)
            ENDIF
          ENDDO
        ENDDO
      ENDIF
      END
"""


patch('ML5_0_ONIA_PW2_EPSILONS', epsilons)


def set_dual(r):
    r = rep(r, "      INTEGER I, J\n", "      INTEGER I, J\n" + MODES)
    old = "        PW_EPS2(I)=EPS2(I,PW_ILZ2)\n      ENDDO\n      END\n"
    new = """        PW_EPS2(I)=EPS2(I,PW_ILZ2)
        IF (PWT_MODE.EQ.2) THEN
          PD(I,ONIA_Q1(ONIA_IPW))%COMP(1)=0
          PD(I,ONIA_Q2(ONIA_IPW))%COMP(1)=0
          PD(I,ONIA_Q1(ONIA_IPW2))%COMP(2)=0
          PD(I,ONIA_Q2(ONIA_IPW2))%COMP(2)=0
        ENDIF
      ENDDO
      END
"""
    return rep(r, old, new)


patch('ML5_0_ONIA_PW2_SET_DUAL', set_dual)


def prepare(r):
    r = rep(r, "      INCLUDE 'coupl.inc'\n",
            "      COMPLEX*16 PWT_W3(NBORNAMPS,NCOMB,9)\n      COMMON/ML5_0_PW2_TEST/PWT_W3\n"
            + MODES + "      INCLUDE 'coupl.inc'\n")
    old = """        LIVE(H)=.FALSE.
        IF (.NOT.ACTIVE(H)) CYCLE
"""
    r = rep(r, old, """        LIVE(H)=.FALSE.
        IF (.NOT.ACTIVE(H).AND.PWT_MODE.NE.2) CYCLE
""")
    old = "        W3(1:NBORNAMPS,H,1:9)=FACT*W3(1:NBORNAMPS,H,1:9)\n"
    return rep(r, old, old + """C       (test modes)
        IF (PWT_MODE.EQ.1) THEN
          WEFF(1:NBORNAMPS,H)=(0D0,0D0)
          W1(1:NBORNAMPS,H,1:3)=(0D0,0D0)
          W2(1:NBORNAMPS,H,1:3)=(0D0,0D0)
          DO IDIR=1,9
            ILZ1=MOD(IDIR-1,3)+1
            ILZ2=(IDIR-1)/3+1
            IF (ILZ1.EQ.3.OR.ILZ2.EQ.3) W3(1:NBORNAMPS,H,IDIR)=(0D0,0D0)
          ENDDO
          PWT_W3(1:NBORNAMPS,H,1:9)=W3(1:NBORNAMPS,H,1:9)
        ELSEIF (PWT_MODE.EQ.2) THEN
          WEFF(1:NBORNAMPS,H)=PWT_W3(1:NBORNAMPS,H,PWT_D)
          W1(1:NBORNAMPS,H,1:3)=(0D0,0D0)
          W2(1:NBORNAMPS,H,1:3)=(0D0,0D0)
          W3(1:NBORNAMPS,H,1:9)=(0D0,0D0)
        ENDIF
""")


patch('ML5_0_ONIA_PREPARE', prepare)

s += """
      SUBROUTINE ML5_0_PW2_TEST_DIRS(S,E)
C     state S: two unit vectors perpendicular to its base relative
C     momentum, and the base direction itself
      IMPLICIT NONE
      INTEGER S
      REAL*8 E(3,3), N(3,2), A(3), X
      COMMON/ML5_0_PW2_TEST4/N
      E(1:3,3)=N(1:3,S)
      A=(/1D0,0D0,0D0/)
      IF (ABS(N(1,S)).GT.0.9D0) A=(/0D0,1D0,0D0/)
      X=SUM(A*N(1:3,S))
      E(1:3,1)=A-X*N(1:3,S)
      E(1:3,1)=E(1:3,1)/SQRT(SUM(E(1:3,1)**2))
      E(1,2)=N(2,S)*E(3,1)-N(3,S)*E(2,1)
      E(2,2)=N(3,S)*E(1,1)-N(1,S)*E(3,1)
      E(3,2)=N(1,S)*E(2,1)-N(2,S)*E(1,1)
      END

      SUBROUTINE ML5_0_PW2_SET_BASE(Q01,Q02)
      IMPLICIT NONE
      REAL*8 Q01(3), Q02(3), N(3,2)
      COMMON/ML5_0_PW2_TEST4/N
      N(1:3,1)=Q01/SQRT(SUM(Q01**2))
      N(1:3,2)=Q02/SQRT(SUM(Q02**2))
      END

      SUBROUTINE ML5_0_PW2_SET_RELMOM(Q1,Q2)
      IMPLICIT NONE
      REAL*8 Q1(3), Q2(3), QR2(3,2)
      COMMON/ML5_0_PW2_RELMOM/QR2
      QR2(1:3,1)=Q1
      QR2(1:3,2)=Q2
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
