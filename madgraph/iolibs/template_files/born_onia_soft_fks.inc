      SUBROUTINE BORN_ONIA_SOFT(P,VEC,TEN,BAUX,IONIUM)
C
%(info_lines)s
C
C The soft current of the P-wave quarkonium of the Born (arXiv:2402.19221,
C sect. 3.2; madgraph/fks/fks_onia.py), at the Born momenta P (physical
C numbering, one leg per onium):
C   VEC(:,K) = sum over the S-wave states S' of the cross vectors
C              M^mu_{J,K[C C']}(r, r') of Born leg K (the onium included)
C              of the paper (eqs. MlinkedOnium4J): with T_K . Q_eff(CC')
C   TEN(:,:) = sum over S' of C_eff(CC') M_J^{mu nu}(r')
C   BAUX(S') = -g_{mu nu} M_J^{mu nu}(r')
C with the LDME and average factors of the P-wave Born (its kernel factor
C and identical-particle denominator), without g_s. M_J^{mu nu} is built
C from the S-wave amplitudes A(S_z) of the same diagrams,
C   V^mu(J_z) = sum_{L_z,S_z} <1 L_z; S S_z|J J_z> eps*^mu(L_z) A(S_z).
C IONIUM is the position of the onium among the Born legs.
C
%(process_lines)s
C
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER    NB
      PARAMETER (NB=%(nb)d)
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER    NCOLP
      PARAMETER (NCOLP=%(ncolp)d)
      INTEGER    NAUX
      PARAMETER (NAUX=%(naux)d)
      INTEGER    MAXCOLS
      PARAMETER (MAXCOLS=%(maxcols)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    QQ, SPIN, JTOT
      PARAMETER (QQ=%(ionium)d, SPIN=%(spin)d, JTOT=%(jtot)d)
      INTEGER    NCONST
      PARAMETER (NCONST=%(nconst)d)
      INTEGER    IQ, IQBAR
      PARAMETER (IQ=%(nq)d, IQBAR=%(nqbar)d)
      REAL*8     ZERO, NC, CF
      PARAMETER (ZERO=0D0, NC=3D0, CF=4D0/3D0)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NB), VEC(0:3,NB), BAUX(NAUX)
      COMPLEX*16 TEN(0:3,0:3)
      INTEGER IONIUM
C
C LOCAL VARIABLES
C
      INTEGER I, IC, A, K, MU, NU, IA, IB, IB2, SZ, LZ, JZ
      INTEGER NHELS(NB,NCOMB), NHEL(NB)
      INTEGER NCOLS(NAUX), CAUX(NAUX)
      DATA NCOLS /%(ncols)s/
      DATA CAUX /%(aux_colours)s/
      CHARACTER*2 LABEL(NAUX)
      DATA LABEL /%(aux_labels)s/
      REAL*8 XR(NCOLP,MAXCOLS,NB,NAUX,2), XI(NCOLP,MAXCOLS,NB,NAUX,2)
      COMPLEX*16 XM(NCOLP,MAXCOLS,NB,NAUX)
      REAL*8 CFS(MAXCOLS,MAXCOLS,NAUX)
      LOGICAL LINKED(NB,NAUX)
      REAL*8 CEFF(NAUX), F1(NAUX), F2(NAUX), R, MSUM
      REAL*8 PMASS_ONIA(NCONST), MQQ, CG, FAC, DENOM
      COMPLEX*16 AMP(NGRAPHS), JP(NCOLP), JS(MAXCOLS)
      COMPLEX*16 V(0:3,MAXCOLS), Z, ZT(0:3,0:3)
      TYPE(ALOHA) EPS
      LOGICAL FIRSTTIME
      DATA FIRSTTIME /.TRUE./
      SAVE FIRSTTIME, XM, CFS, LINKED, CEFF, MQQ
%(helicity_lines)s
      DOUBLE PRECISION ONIA_SOFT_AMPPFAC
      EXTERNAL ONIA_SOFT_AMPPFAC
      INTEGER IDEN
      %(den_line)s
      INCLUDE 'coupl.inc'
C ----------
C BEGIN CODE
C ----------
      IF (FIRSTTIME) THEN
%(pmassonia_lines)s
        MQQ = PMASS_ONIA(IQ)+PMASS_ONIA(IQBAR)
C       the mass factors of Q_eff (r = m_Q/m_Qbar) and its Casimirs
        R = PMASS_ONIA(IQ)/PMASS_ONIA(IQBAR)
        MSUM = 2D0+R+1D0/R
        DO A = 1, NAUX
          IF (LABEL(A).EQ.'18') THEN
            F1(A) = MSUM/SQRT(2D0*NC)
            F2(A) = 0D0
            CEFF(A) = MSUM**2/(2D0*NC)
          ELSE IF (LABEL(A).EQ.'81') THEN
            F1(A) = MSUM/SQRT(2D0*NC)
            F2(A) = 0D0
            CEFF(A) = MSUM**2*CF
          ELSE
            F1(A) = MSUM/2D0
            F2(A) = (1D0/R-R)/2D0
            CEFF(A) = MSUM*(-2D0/NC+(NC**2-2D0)/(2D0*NC)*(R+1D0/R))
          ENDIF
        ENDDO
C       the colour transition matrices <P| T_K Q_eff |S'> (without the mass
C       factors), and the colour matrices of the S-wave states
        XR(:,:,:,:,:) = 0D0
        XI(:,:,:,:,:) = 0D0
        CFS(:,:,:) = 0D0
        LINKED(:,:) = .FALSE.
%(x_lines)s
%(cf_lines)s
%(link_lines)s
        DO A = 1, NAUX
          XM(:,:,:,A) = F1(A)*DCMPLX(XR(:,:,:,A,1),XI(:,:,:,A,1))
     &      +F2(A)*DCMPLX(XR(:,:,:,A,2),XI(:,:,:,A,2))
        ENDDO
        FIRSTTIME = .FALSE.
      ENDIF
      IONIUM = QQ
      VEC(:,:) = 0D0
      TEN(:,:) = (0D0,0D0)
      BAUX(:) = 0D0
      DO IC = 1, NCOMB
        NHEL(:) = NHELS(:,IC)
        JZ = NHEL(QQ)
        CALL ONIA_SOFT_AMPP(P,NHEL,AMP)
        CALL ONIA_SOFT_JAMPP(AMP,JP)
        DO A = 1, NAUX
C         V^mu(J_z) of the S-wave state, colour flow by colour flow
          V(:,:) = (0D0,0D0)
          DO SZ = -SPIN, SPIN
            LZ = JZ-SZ
            IF (ABS(LZ).GT.1) CYCLE
            CALL ONIA_SOFT_AMPPCG(1,LZ,SPIN,SZ,JTOT,JZ,CG)
            IF (CG.EQ.0D0) CYCLE
            NHEL(QQ) = SZ
            CALL ONIA_SOFT_AMPS(A,P,NHEL,JS)
            NHEL(QQ) = JZ
            CALL VXXXXX(P(0,QQ),MQQ,LZ,1,EPS)
            DO MU = 0, 3
              V(MU,1:NCOLS(A)) = V(MU,1:NCOLS(A))
     &          +CG*EPS%%W(MU+1)*JS(1:NCOLS(A))
            ENDDO
          ENDDO
C         the cross vectors -2 Re{ A_P^* T_K Q_eff V^mu }
          DO K = 1, NB
            IF (.NOT.LINKED(K,A)) CYCLE
            DO MU = 0, 3
              Z = (0D0,0D0)
              DO IA = 1, NCOLP
                DO IB = 1, NCOLS(A)
                  Z = Z+DCONJG(JP(IA))*XM(IA,IB,K,A)*V(MU,IB)
                ENDDO
              ENDDO
              VEC(MU,K) = VEC(MU,K)-2D0*DBLE(Z)
            ENDDO
          ENDDO
C         the self tensor sum_colours V^mu V^nu*
          DO MU = 0, 3
            DO NU = 0, 3
              Z = (0D0,0D0)
              DO IB = 1, NCOLS(A)
                DO IB2 = 1, NCOLS(A)
                  Z = Z+V(MU,IB)*CFS(IB,IB2,A)*DCONJG(V(NU,IB2))
                ENDDO
              ENDDO
              ZT(MU,NU) = Z
            ENDDO
          ENDDO
          TEN(:,:) = TEN(:,:)+CEFF(A)*ZT(:,:)
          BAUX(A) = BAUX(A)-DBLE(ZT(0,0)-ZT(1,1)-ZT(2,2)-ZT(3,3))
        ENDDO
      ENDDO
      FAC = ONIA_SOFT_AMPPFAC()/DBLE(IDEN)
      VEC(:,:) = VEC(:,:)*FAC
      TEN(:,:) = TEN(:,:)*FAC
      BAUX(:) = BAUX(:)*FAC
      END


      SUBROUTINE ONIA_SOFT_AMPS(A,P,NHEL,JS)
C     the colour flows of S-wave state A at the Born momenta P, helicities
C     NHEL (S_z for the onium)
      IMPLICIT NONE
      INTEGER    NB
      PARAMETER (NB=%(nb)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    MAXCOLS
      PARAMETER (MAXCOLS=%(maxcols)d)
      INTEGER A, NHEL(NB)
      REAL*8 P(0:3,NB)
      COMPLEX*16 JS(MAXCOLS), AMP(NGRAPHS)
      JS(:) = (0D0,0D0)
%(dispatch_lines)s
      END

