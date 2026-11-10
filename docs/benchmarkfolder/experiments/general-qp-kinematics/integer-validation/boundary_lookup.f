      SUBROUTINE BOUNDARY_KINEMATIC_IDENTITY(G,CT,N,I,J,
     $ CLASS,LEG,SGN,CLOSURE,DECLARED_ONSHELL,KNOWN)
      IMPLICIT NONE
      INTEGER G,CT,N,I,J,CLASS,LEG,SGN,CLOSURE
      LOGICAL DECLARED_ONSHELL,KNOWN
      INTEGER A,B,K,DIR,NP,ORI,L
      INTEGER GOFF(3),GNUM(3),GVALID(3)
      INTEGER CLASSES(1),LEGS(1),SIGNS(1)
      INTEGER CLOSURES(1),SHELLS(1)
C     Identity only: declared on-shell is not a numerical certificate.
      DATA (GOFF(L),L=1,3) /
     $ 0,0,0/
      DATA (GNUM(L),L=1,3) /
     $ 1,0,2/
      DATA (GVALID(L),L=1,3) /
     $ 1,0,0/
      DATA (CLASSES(L),L=1,1) /
     $ 0/
      DATA (LEGS(L),L=1,1) /
     $ 0/
      DATA (SIGNS(L),L=1,1) /
     $ 0/
      DATA (CLOSURES(L),L=1,1) /
     $ 0/
      DATA (SHELLS(L),L=1,1) /
     $ 0/
      CLASS=0
      LEG=0
      SGN=0
      CLOSURE=0
      DECLARED_ONSHELL=.FALSE.
      KNOWN=.FALSE.
      IF(G.LT.1.OR.G.GT.3) RETURN
      IF(GVALID(G).NE.1.OR.N.NE.GNUM(G)) RETURN
      IF(I.LT.1.OR.J.LT.1.OR.I.GT.N.OR.J.GT.N) RETURN
      IF(I.EQ.J) RETURN
      DIR=0
      IF(CT.EQ.2.OR.CT.EQ.5) DIR=1
      IF(CT.NE.1.AND.CT.NE.2.AND.CT.NE.4.AND.CT.NE.5)
     $ RETURN
      A=MIN(I,J)
      B=MAX(I,J)
      ORI=1
      IF(I.GT.J) ORI=-1
      NP=N*(N-1)/2
      K=GOFF(G)+DIR*NP+(A-1)*N-(A-1)*A/2+B-A
      CLASS=CLASSES(K)
      LEG=LEGS(K)
      SGN=ORI*SIGNS(K)
      CLOSURE=ORI*CLOSURES(K)
      DECLARED_ONSHELL=SHELLS(K).EQ.1
      KNOWN=.TRUE.
      END
