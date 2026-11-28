      SUBROUTINE BORN_ONIA_SOFT(P,VEC,TEN,BAUX,DD,VPAR)
C
%(info_lines)s
C
C The soft currents of the P-wave quarkonia of the Born (arXiv:2402.19221,
C sect. 3.2, and its extension to several P-wave pairs, JHEP_2Pwave,
C sects. 3-5; madgraph/fks/fks_onia.py), at the Born momenta P (physical
C numbering, one leg per onium). For each P-wave quarkonium H (the others
C are spectators in their own Fock states, P waves keeping their
C derivative):
C   VEC(:,K,H)   = sum over the S-wave daughters S' of the cross vectors
C                  M^mu_{J,K[C C']}(r, r') of Born leg K (the onium
C                  included): -2 Re{A^* T_K . Q_eff(CC') P^mu A(r')}
C   TEN(:,:,H)   = sum over S' of C_eff(CC') M_J^{mu nu}(r')
C   BAUX(S',H)   = -g_{mu nu} M_J^{mu nu}(r')
C and for each pair IP = (H1, H2), H1 < H2 (BORN_ONIA_INFO), mu of H1 and
C nu of H2:
C   DD(:,:,IP)   = sum over S'_1, S'_2 of the dipole-dipole interference
C                  -2 Re{[Q_eff,2 P_2^nu A(r'_2)]^* . [Q_eff,1 P_1^mu A(r'_1)]}
C                  (real emission, cross class, eq. Mdipdip)
C   VPAR(:,:,IP) = sum over S'_1, S'_2 of A^* Q_eff,1 . Q_eff,2
C                  P_1^mu P_2^nu A(r'_12)  (virtual, parallel class, eq. MV)
C with the LDME and average factors of the P-wave Born (its kernel factor
C and identical-particle denominator), without g_s. P^mu A(r') is built from
C the S-wave amplitudes A(S_z) of the same diagrams,
C   V^mu(J_z) = sum_{L_z,S_z} <1 L_z; S S_z|J J_z> eps*^mu(L_z) A(S_z),
C with eps(L_z) instead of eps*(L_z) for a decaying onium (ISV = -1, the
C incoming polarisation vectors of its P-wave kernel). The soft current of
C a decaying onium is that of the production with its momentum taken as
C -P (all momenta outgoing): the cross vectors (of degree -1 in P) change
C sign, and the dipole-dipole tensors with each decaying onium; the self
C tensor (degree -2) does not. These signs are in VEC, DD and VPAR, so
C that the soft, integrated and pole terms take the physical P.
C The results are kept for the last momenta P.
C
%(process_lines)s
C
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'born_onia.inc'
      INTEGER    NB
      PARAMETER (NB=%(nb)d)
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER    NCOLP
      PARAMETER (NCOLP=%(ncolp)d)
      INTEGER    MAXCOLS
      PARAMETER (MAXCOLS=%(maxcols)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCONST
      PARAMETER (NCONST=%(nconst)d)
      REAL*8     ZERO, NC, CF
      PARAMETER (ZERO=0D0, NC=3D0, CF=4D0/3D0)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NB), VEC(0:3,NB,NPQ), BAUX(NAUXMAX,NPQ)
      COMPLEX*16 TEN(0:3,0:3,NPQ)
      REAL*8 DD(0:3,0:3,NPAIRX)
      COMPLEX*16 VPAR(0:3,0:3,NPAIRX)
C
C LOCAL VARIABLES
C
      INTEGER I, IC, A, A1, A2, K, MU, NU, IA, IB, IB2, SZ, LZ, JZ
      INTEGER H, H1, H2, IP, KD1, KD2, SZ1, SZ2, LZ1, LZ2, JZ1, JZ2
      INTEGER NHELS(NB,NCOMB), NHEL(NB)
      INTEGER QQ(NPQ), SPIN(NPQ), JTOT(NPQ), NAUX(NPQ), IQ(NPQ)
      INTEGER IQBAR(NPQ), IPAIR(2,NPAIRX), ISV(NPQ)
      INTEGER NCOLS(NAUXMAX,NPQ), NCOLD(NAUXMAX,NAUXMAX,NPAIRX)
      CHARACTER*2 LABEL(NAUXMAX,NPQ)
%(data_decl_lines)s
      REAL*8 XR(NCOLP,MAXCOLS,NB,NAUXMAX,2,NPQ)
      REAL*8 XI(NCOLP,MAXCOLS,NB,NAUXMAX,2,NPQ)
      COMPLEX*16 XM(NCOLP,MAXCOLS,NB,NAUXMAX,NPQ)
      REAL*8 CFS(MAXCOLS,MAXCOLS,NAUXMAX,NPQ)
      LOGICAL LINKED(NB,NAUXMAX,NPQ)
      REAL*8 YR(MAXCOLS,MAXCOLS,NAUXMAX,NAUXMAX,2,2,NPAIRX)
      REAL*8 YI(MAXCOLS,MAXCOLS,NAUXMAX,NAUXMAX,2,2,NPAIRX)
      COMPLEX*16 YM(MAXCOLS,MAXCOLS,NAUXMAX,NAUXMAX,NPAIRX)
      REAL*8 ZR(NCOLP,MAXCOLS,NAUXMAX,NAUXMAX,2,2,NPAIRX)
      REAL*8 ZI(NCOLP,MAXCOLS,NAUXMAX,NAUXMAX,2,2,NPAIRX)
      COMPLEX*16 ZM(NCOLP,MAXCOLS,NAUXMAX,NAUXMAX,NPAIRX)
      LOGICAL ZNZ(NAUXMAX,NAUXMAX,NPAIRX)
      REAL*8 CEFF(NAUXMAX,NPQ), FK(2,NAUXMAX,NPQ), R, MSUM
      REAL*8 PMASS_ONIA(NCONST), MQQ(NPQ), CG, CG1, CG2, FAC
      COMPLEX*16 AMP(NGRAPHS), JP(NCOLP), JS(MAXCOLS), JD(MAXCOLS)
      COMPLEX*16 V(0:3,MAXCOLS,NAUXMAX,NPQ), Z, ZT(0:3,0:3)
      COMPLEX*16 W(0:3,0:3,MAXCOLS)
      TYPE(ALOHA) EPS, EPS1, EPS2
      REAL*8 PLAST(0:3,NB), VEC_S(0:3,NB,NPQ), BAUX_S(NAUXMAX,NPQ)
      COMPLEX*16 TEN_S(0:3,0:3,NPQ), VPAR_S(0:3,0:3,NPAIRX)
      REAL*8 DD_S(0:3,0:3,NPAIRX)
      LOGICAL FIRSTTIME
      DATA FIRSTTIME /.TRUE./
      SAVE FIRSTTIME, XM, CFS, LINKED, CEFF, MQQ, YM, ZM, ZNZ
      SAVE PLAST, VEC_S, BAUX_S, TEN_S, VPAR_S, DD_S
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
C       the mass factors of Q_eff (r = m_Q/m_Qbar) and its Casimirs, per
C       P-wave quarkonium: FK(1,.) multiplies Delta ([18], [81]) or d_abc
C       ([88]), FK(2,.) -i f_abc ([88])
        DO H = 1, NPQ
          MQQ(H) = PMASS_ONIA(IQ(H))+PMASS_ONIA(IQBAR(H))
          R = PMASS_ONIA(IQ(H))/PMASS_ONIA(IQBAR(H))
C         a decaying onium: crossed, its antiquark is the quark of the
C         pair (it carries the octet index), r -> 1/r: the -i f_abc part
C         (1/r-r)/2 of Q_eff(88) changes sign (the rest is symmetric)
          IF (ISV(H).LT.0) R = 1D0/R
          MSUM = 2D0+R+1D0/R
          DO A = 1, NAUX(H)
            IF (LABEL(A,H).EQ.'18') THEN
              FK(1,A,H) = MSUM/SQRT(2D0*NC)
              FK(2,A,H) = 0D0
              CEFF(A,H) = MSUM**2/(2D0*NC)
            ELSE IF (LABEL(A,H).EQ.'81') THEN
              FK(1,A,H) = MSUM/SQRT(2D0*NC)
              FK(2,A,H) = 0D0
              CEFF(A,H) = MSUM**2*CF
            ELSE
              FK(1,A,H) = MSUM/2D0
              FK(2,A,H) = (1D0/R-R)/2D0
              CEFF(A,H) = MSUM*(-2D0/NC+(NC**2-2D0)/(2D0*NC)*(R+1D0/R))
            ENDIF
          ENDDO
        ENDDO
C       the colour transition matrices <P| T_K Q_eff |S'> (without the
C       mass factors), the colour matrices of the S-wave daughters, and the
C       dipole-dipole colour matrices of the pairs (fks_onia.dipole_matrices)
        XR(:,:,:,:,:,:) = 0D0
        XI(:,:,:,:,:,:) = 0D0
        CFS(:,:,:,:) = 0D0
        LINKED(:,:,:) = .FALSE.
        YR(:,:,:,:,:,:,:) = 0D0
        YI(:,:,:,:,:,:,:) = 0D0
        ZR(:,:,:,:,:,:,:) = 0D0
        ZI(:,:,:,:,:,:,:) = 0D0
        ZNZ(:,:,:) = .FALSE.
%(x_lines)s
%(cf_lines)s
%(link_lines)s
%(y_lines)s
%(z_lines)s
        XM(:,:,:,:,:) = (0D0,0D0)
        DO H = 1, NPQ
          DO A = 1, NAUX(H)
            DO KD1 = 1, 2
              XM(:,:,:,A,H) = XM(:,:,:,A,H)+FK(KD1,A,H)
     &          *DCMPLX(XR(:,:,:,A,KD1,H),XI(:,:,:,A,KD1,H))
            ENDDO
          ENDDO
        ENDDO
        YM(:,:,:,:,:) = (0D0,0D0)
        ZM(:,:,:,:,:) = (0D0,0D0)
        DO IP = 1, NPAIR
          H1 = IPAIR(1,IP)
          H2 = IPAIR(2,IP)
          DO A1 = 1, NAUX(H1)
            DO A2 = 1, NAUX(H2)
              DO KD1 = 1, 2
                DO KD2 = 1, 2
                  YM(:,:,A1,A2,IP) = YM(:,:,A1,A2,IP)
     &              +FK(KD1,A1,H1)*FK(KD2,A2,H2)
     &              *DCMPLX(YR(:,:,A1,A2,KD1,KD2,IP),
     &              YI(:,:,A1,A2,KD1,KD2,IP))
                  ZM(:,:,A1,A2,IP) = ZM(:,:,A1,A2,IP)
     &              +FK(KD1,A1,H1)*FK(KD2,A2,H2)
     &              *DCMPLX(ZR(:,:,A1,A2,KD1,KD2,IP),
     &              ZI(:,:,A1,A2,KD1,KD2,IP))
                ENDDO
              ENDDO
            ENDDO
          ENDDO
        ENDDO
        PLAST(:,:) = -1D0
        FIRSTTIME = .FALSE.
      ENDIF
      IF (ALL(P(:,:).EQ.PLAST(:,:))) THEN
        VEC(:,:,:) = VEC_S(:,:,:)
        TEN(:,:,:) = TEN_S(:,:,:)
        BAUX(:,:) = BAUX_S(:,:)
        DD(:,:,:) = DD_S(:,:,:)
        VPAR(:,:,:) = VPAR_S(:,:,:)
        RETURN
      ENDIF
      VEC(:,:,:) = 0D0
      TEN(:,:,:) = (0D0,0D0)
      BAUX(:,:) = 0D0
      DD(:,:,:) = 0D0
      VPAR(:,:,:) = (0D0,0D0)
      DO IC = 1, NCOMB
        NHEL(:) = NHELS(:,IC)
        CALL ONIA_SOFT_AMPP(P,NHEL,AMP)
        CALL ONIA_SOFT_JAMPP(AMP,JP)
        DO H = 1, NPQ
          JZ = NHEL(QQ(H))
          DO A = 1, NAUX(H)
C           V^mu(J_z) of the S-wave daughter, colour flow by colour flow
            V(:,:,A,H) = (0D0,0D0)
            DO SZ = -SPIN(H), SPIN(H)
              LZ = JZ-SZ
              IF (ABS(LZ).GT.1) CYCLE
              CALL ONIA_SOFT_AMPPCG(1,LZ,SPIN(H),SZ,JTOT(H),JZ,CG)
              IF (CG.EQ.0D0) CYCLE
              NHEL(QQ(H)) = SZ
              CALL ONIA_SOFT_AMPS(H,A,P,NHEL,JS)
              NHEL(QQ(H)) = JZ
              CALL VXXXXX(P(0,QQ(H)),MQQ(H),LZ,ISV(H),EPS)
              DO MU = 0, 3
                V(MU,1:NCOLS(A,H),A,H) = V(MU,1:NCOLS(A,H),A,H)
     &            +CG*EPS%%W(MU+1)*JS(1:NCOLS(A,H))
              ENDDO
            ENDDO
C           the cross vectors -2 Re{ A_P^* T_K Q_eff V^mu }
            DO K = 1, NB
              IF (.NOT.LINKED(K,A,H)) CYCLE
              DO MU = 0, 3
                Z = (0D0,0D0)
                DO IA = 1, NCOLP
                  DO IB = 1, NCOLS(A,H)
                    Z = Z+DCONJG(JP(IA))*XM(IA,IB,K,A,H)*V(MU,IB,A,H)
                  ENDDO
                ENDDO
                VEC(MU,K,H) = VEC(MU,K,H)-2D0*DBLE(Z)
              ENDDO
            ENDDO
C           the self tensor sum_colours V^mu V^nu*
            DO MU = 0, 3
              DO NU = 0, 3
                Z = (0D0,0D0)
                DO IB = 1, NCOLS(A,H)
                  DO IB2 = 1, NCOLS(A,H)
                    Z = Z+V(MU,IB,A,H)*CFS(IB,IB2,A,H)
     &                *DCONJG(V(NU,IB2,A,H))
                  ENDDO
                ENDDO
                ZT(MU,NU) = Z
              ENDDO
            ENDDO
            TEN(:,:,H) = TEN(:,:,H)+CEFF(A,H)*ZT(:,:)
            BAUX(A,H) = BAUX(A,H)-DBLE(ZT(0,0)-ZT(1,1)-ZT(2,2)-ZT(3,3))
          ENDDO
        ENDDO
        DO IP = 1, NPAIR
          H1 = IPAIR(1,IP)
          H2 = IPAIR(2,IP)
          DO A1 = 1, NAUX(H1)
            DO A2 = 1, NAUX(H2)
C             the cross class: -2 Re{ V_2^nu^* Y V_1^mu }
              DO MU = 0, 3
                DO NU = 0, 3
                  Z = (0D0,0D0)
                  DO IA = 1, NCOLS(A2,H2)
                    DO IB = 1, NCOLS(A1,H1)
                      Z = Z+DCONJG(V(NU,IA,A2,H2))*YM(IA,IB,A1,A2,IP)
     &                  *V(MU,IB,A1,H1)
                    ENDDO
                  ENDDO
                  DD(MU,NU,IP) = DD(MU,NU,IP)-2D0*DBLE(Z)
                ENDDO
              ENDDO
C             the parallel class: A_P^* Z P_1^mu P_2^nu A(r'_12)
              IF (.NOT.ZNZ(A1,A2,IP)) CYCLE
              JZ1 = NHEL(QQ(H1))
              JZ2 = NHEL(QQ(H2))
              W(:,:,:) = (0D0,0D0)
              DO SZ1 = -SPIN(H1), SPIN(H1)
                LZ1 = JZ1-SZ1
                IF (ABS(LZ1).GT.1) CYCLE
                CALL ONIA_SOFT_AMPPCG(1,LZ1,SPIN(H1),SZ1,JTOT(H1),JZ1,
     &            CG1)
                IF (CG1.EQ.0D0) CYCLE
                CALL VXXXXX(P(0,QQ(H1)),MQQ(H1),LZ1,ISV(H1),EPS1)
                DO SZ2 = -SPIN(H2), SPIN(H2)
                  LZ2 = JZ2-SZ2
                  IF (ABS(LZ2).GT.1) CYCLE
                  CALL ONIA_SOFT_AMPPCG(1,LZ2,SPIN(H2),SZ2,JTOT(H2),
     &              JZ2,CG2)
                  IF (CG2.EQ.0D0) CYCLE
                  CALL VXXXXX(P(0,QQ(H2)),MQQ(H2),LZ2,ISV(H2),EPS2)
                  NHEL(QQ(H1)) = SZ1
                  NHEL(QQ(H2)) = SZ2
                  CALL ONIA_SOFT_AMPD(IP,A1,A2,P,NHEL,JD)
                  NHEL(QQ(H1)) = JZ1
                  NHEL(QQ(H2)) = JZ2
                  DO MU = 0, 3
                    DO NU = 0, 3
                      W(MU,NU,1:NCOLD(A1,A2,IP)) =
     &                  W(MU,NU,1:NCOLD(A1,A2,IP))+CG1*CG2
     &                  *EPS1%%W(MU+1)*EPS2%%W(NU+1)
     &                  *JD(1:NCOLD(A1,A2,IP))
                    ENDDO
                  ENDDO
                ENDDO
              ENDDO
              DO MU = 0, 3
                DO NU = 0, 3
                  Z = (0D0,0D0)
                  DO IA = 1, NCOLP
                    DO IB = 1, NCOLD(A1,A2,IP)
                      Z = Z+DCONJG(JP(IA))*ZM(IA,IB,A1,A2,IP)
     &                  *W(MU,NU,IB)
                    ENDDO
                  ENDDO
                  VPAR(MU,NU,IP) = VPAR(MU,NU,IP)+Z
                ENDDO
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      FAC = ONIA_SOFT_AMPPFAC()/DBLE(IDEN)
      DO H = 1, NPQ
        VEC(:,:,H) = VEC(:,:,H)*ISV(H)
      ENDDO
      DO IP = 1, NPAIR
        DD(:,:,IP) = DD(:,:,IP)*ISV(IPAIR(1,IP))*ISV(IPAIR(2,IP))
        VPAR(:,:,IP) = VPAR(:,:,IP)*ISV(IPAIR(1,IP))*ISV(IPAIR(2,IP))
      ENDDO
      VEC(:,:,:) = VEC(:,:,:)*FAC
      TEN(:,:,:) = TEN(:,:,:)*FAC
      BAUX(:,:) = BAUX(:,:)*FAC
      DD(:,:,:) = DD(:,:,:)*FAC
      VPAR(:,:,:) = VPAR(:,:,:)*FAC
      PLAST(:,:) = P(:,:)
      VEC_S(:,:,:) = VEC(:,:,:)
      TEN_S(:,:,:) = TEN(:,:,:)
      BAUX_S(:,:) = BAUX(:,:)
      DD_S(:,:,:) = DD(:,:,:)
      VPAR_S(:,:,:) = VPAR(:,:,:)
      END


      SUBROUTINE BORN_ONIA_INFO(IONIUM_OUT,NAUX_OUT,SPIN_OUT,JTOT_OUT,
     &  LABEL_OUT,CEFF_OUT,MQ_OUT,MQB_OUT,IPAIR_OUT)
C     the P-wave quarkonia of BORN_ONIA_SOFT: for each, its position among
C     the Born legs, its spin S and J, its S-wave daughters S' (label 18, 81
C     or 88, and C_eff of Q_eff(CC')), and the masses of its quark and
C     antiquark; the pairs (H1, H2) of the dipole-dipole terms
      IMPLICIT NONE
      INCLUDE 'born_onia.inc'
      INTEGER    NCONST
      PARAMETER (NCONST=%(nconst)d)
      REAL*8     ZERO, NC, CF
      PARAMETER (ZERO=0D0, NC=3D0, CF=4D0/3D0)
      INTEGER IONIUM_OUT(NPQ), NAUX_OUT(NPQ), SPIN_OUT(NPQ)
      INTEGER JTOT_OUT(NPQ), IPAIR_OUT(2,NPAIRX), A, H
      CHARACTER*2 LABEL_OUT(NAUXMAX,NPQ)
      REAL*8 CEFF_OUT(NAUXMAX,NPQ), MQ_OUT(NPQ), MQB_OUT(NPQ), R, MSUM
      INTEGER QQ(NPQ), SPIN(NPQ), JTOT(NPQ), NAUX(NPQ), IQ(NPQ)
      INTEGER IQBAR(NPQ), IPAIR(2,NPAIRX), ISV(NPQ)
      INTEGER NCOLS(NAUXMAX,NPQ), NCOLD(NAUXMAX,NAUXMAX,NPAIRX)
      CHARACTER*2 LABEL(NAUXMAX,NPQ)
%(data_decl_lines)s
      REAL*8 PMASS_ONIA(NCONST)
      INCLUDE 'coupl.inc'
%(pmassonia_lines)s
      IPAIR_OUT(:,:) = IPAIR(:,:)
      DO H = 1, NPQ
        R = PMASS_ONIA(IQ(H))/PMASS_ONIA(IQBAR(H))
        MSUM = 2D0+R+1D0/R
        IONIUM_OUT(H) = QQ(H)
        NAUX_OUT(H) = NAUX(H)
        SPIN_OUT(H) = SPIN(H)
        JTOT_OUT(H) = JTOT(H)
        MQ_OUT(H) = PMASS_ONIA(IQ(H))
        MQB_OUT(H) = PMASS_ONIA(IQBAR(H))
        DO A = 1, NAUX(H)
          LABEL_OUT(A,H) = LABEL(A,H)
          IF (LABEL(A,H).EQ.'18') THEN
            CEFF_OUT(A,H) = MSUM**2/(2D0*NC)
          ELSE IF (LABEL(A,H).EQ.'81') THEN
            CEFF_OUT(A,H) = MSUM**2*CF
          ELSE
            CEFF_OUT(A,H) = MSUM*(-2D0/NC+(NC**2-2D0)/(2D0*NC)
     &        *(R+1D0/R))
          ENDIF
        ENDDO
      ENDDO
      END


      SUBROUTINE ONIA_SOFT_AMPS(H,A,P,NHEL,JS)
C     the colour flows of S-wave daughter A of P-wave quarkonium H at the
C     Born momenta P, helicities NHEL (S_z for that onium)
      IMPLICIT NONE
      INTEGER    NB
      PARAMETER (NB=%(nb)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    MAXCOLS
      PARAMETER (MAXCOLS=%(maxcols)d)
      INTEGER H, A, NHEL(NB)
      REAL*8 P(0:3,NB)
      COMPLEX*16 JS(MAXCOLS), AMP(NGRAPHS)
      JS(:) = (0D0,0D0)
%(dispatch_lines)s
      END


      SUBROUTINE ONIA_SOFT_AMPD(IP,A1,A2,P,NHEL,JD)
C     the colour flows of pair IP with both P-wave quarkonia in their
C     S-wave daughters A1, A2 at the Born momenta P, helicities NHEL (S_z
C     for the two onia)
      IMPLICIT NONE
      INTEGER    NB
      PARAMETER (NB=%(nb)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    MAXCOLS
      PARAMETER (MAXCOLS=%(maxcols)d)
      INTEGER IP, A1, A2, NHEL(NB)
      REAL*8 P(0:3,NB)
      COMPLEX*16 JD(MAXCOLS), AMP(NGRAPHS)
      JD(:) = (0D0,0D0)
%(dispatch_d_lines)s
      END
