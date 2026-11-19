C ======================================================================
C     Double-distribution GPD model (Radyushkin; Musatov-Radyushkin):
C
C       H(x,xi) = int_-1^1 dbeta int dalpha delta(x - beta - xi alpha)
C                 h_b(beta,alpha) f(beta),
C       h_b = Gamma(2b+2)/(2^(2b+1) Gamma(b+1)^2)
C             [(1-|beta|)^2 - alpha^2]^b / (1-|beta|)^(2b+1),
C
C     on the forward PDFs at the scale MU (PDG2PDF of the proton):
C       gluon   f(beta) = |beta| g(|beta|), b = GPD_BG (even in x)
C       valence f(beta) = q(beta) - qbar(beta) for beta > 0, b = GPD_BV
C       sea     f(beta) = qbar(beta) (beta > 0), -qbar(-beta) (beta < 0),
C               b = GPD_BS; the two halves are integrated together (each
C               diverges at beta -> 0 in the ERBL region, their sum does not)
C     As in the Goloskokov-Kroll models (b = 2 for gluons and sea, 1 for
C     valence). E = Ht = Et = 0. No D-term. The t dependence (GPD_TMODEL):
C     1 exp(B t), 2 Regge exp(B t) |beta|^(-alpha' t) inside the integral.
C     For xi -> 0 it returns the forward PDFs: H^g = x g(x), H^q = q(x)
C     (x > 0), -qbar(-x) (x < 0).
C ======================================================================
      SUBROUTINE GPD_DD(IPART, X, XI, T, MU, G)
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER IPART
      DOUBLE PRECISION X, XI, T, MU, G(4)
      DOUBLE PRECISION GPD_DD_INT, B, AP, FT, XB
      G = 0D0
      IF (ABS(X).GE.1D0) RETURN
      IF (IPART.EQ.21) THEN
        B = GPD_TSLOPEG
        AP = GPD_ALPHAPG
      ELSE
        B = GPD_TSLOPEQ
        AP = GPD_ALPHAPQ
      ENDIF
      IF (XI.LT.1D-9) THEN
C       forward limit
        XB = ABS(X)
        CALL GPD_DD_F(IPART, 0, X, MU, G(1))
        FT = 1D0
        IF (GPD_TMODEL.GE.1) FT = EXP(B*T)
        IF (GPD_TMODEL.EQ.2) FT = FT*XB**(-AP*T)
        G(1) = G(1)*FT
        RETURN
      ENDIF
      IF (IPART.EQ.21) THEN
        G(1) = GPD_DD_INT(IPART, 0, X, XI, T, MU, GPD_BG, B, AP)
      ELSE
        G(1) = GPD_DD_INT(IPART, 1, X, XI, T, MU, GPD_BV, B, AP)
     &       + GPD_DD_INT(IPART, 2, X, XI, T, MU, GPD_BS, B, AP)
      ENDIF
      END

      SUBROUTINE GPD_DD_F(IPART, KIND, BETA, MU, F)
C     The forward function of the DD at BETA (kind 0: total, 1 valence,
C     2 sea), from PDG2PDF of a proton.
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER IPART, KIND
      DOUBLE PRECISION BETA, MU, F, AB, Q, QB, PDG2PDF
      EXTERNAL PDG2PDF
      AB = ABS(BETA)
      F = 0D0
      IF (AB.LE.0D0 .OR. AB.GE.1D0) RETURN
      IF (IPART.EQ.21) THEN
        F = AB*PDG2PDF(1, 21, GPD_BEAM, AB, MU)
        RETURN
      ENDIF
      Q = PDG2PDF(1, IPART, GPD_BEAM, AB, MU)
      QB = PDG2PDF(1, -IPART, GPD_BEAM, AB, MU)
      IF (KIND.EQ.1) THEN
        IF (BETA.GT.0D0) F = Q - QB
      ELSE IF (KIND.EQ.2) THEN
        IF (BETA.GT.0D0) THEN
          F = QB
        ELSE
          F = -QB
        ENDIF
      ELSE
        IF (BETA.GT.0D0) THEN
          F = Q
        ELSE
          F = -QB
        ENDIF
      ENDIF
      END

      DOUBLE PRECISION FUNCTION GPD_PROFILE(B, BETA, ALPHA)
      IMPLICIT NONE
      DOUBLE PRECISION B, BETA, ALPHA, OB, NB
      OB = 1D0 - ABS(BETA)
      GPD_PROFILE = 0D0
      IF (ABS(ALPHA).GE.OB) RETURN
      NB = EXP(LOG_GAMMA(2D0*B+2D0) - (2D0*B+1D0)*LOG(2D0)
     &          - 2D0*LOG_GAMMA(B+1D0))
      GPD_PROFILE = NB*(OB*OB - ALPHA*ALPHA)**B/OB**(2D0*B+1D0)
      END

      DOUBLE PRECISION FUNCTION GPD_DD_INT(IPART, KIND, X, XI, T, MU,
     &                                     B, BT, AP)
C     int dbeta/xi h_b(beta,(x-beta)/xi) f(beta) [t factor], beta > 0 and
C     beta < 0 halves together: the beta < 0 half is written at -beta,
C     with alpha = (x+beta)/xi and f(-beta). The support of each half is
C     an interval of beta in (0,1); the union is split at the ends of both
C     and at the threshold of the profile, and each piece is integrated by
C     48-point Gauss-Legendre in u = (beta - b0)^(1/6), which makes the
C     beta^(-delta) of the PDFs smooth (f ~ beta^(-1.2): the sea halves
C     cancel to beta^(-0.2) there).
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER IPART, KIND
      DOUBLE PRECISION X, XI, T, MU, B, BT, AP
      DOUBLE PRECISION PTS(6), LO1, HI1, LO2, HI2, A, C, TMP
      DOUBLE PRECISION S, U, BETA, W, JAC, F1, F2, H1, H2, FT
      DOUBLE PRECISION GPD_PROFILE
      INTEGER NP, I, J, K
      DOUBLE PRECISION XG(48), WG(48)
      LOGICAL INIT
      DATA INIT/.TRUE./
      SAVE INIT, XG, WG
      IF (INIT) THEN
        CALL GPD_GAULEG(48, XG, WG)
        INIT = .FALSE.
      ENDIF
C     supports in beta > 0: |x - beta| <= xi (1-beta) and |x + beta| <= xi (1-beta)
      LO1 = MAX(0D0, (X-XI)/(1D0-XI))
      HI1 = MIN(1D0, (X+XI)/(1D0+XI))
      LO2 = MAX(0D0, (-X-XI)/(1D0-XI))
      HI2 = MIN(1D0, (-X+XI)/(1D0+XI))
      NP = 0
      DO I = 1, 4
        IF (I.EQ.1) TMP = LO1
        IF (I.EQ.2) TMP = HI1
        IF (I.EQ.3) TMP = LO2
        IF (I.EQ.4) TMP = HI2
        IF (TMP.LT.0D0 .OR. TMP.GT.1D0) CYCLE
        NP = NP + 1
        PTS(NP) = TMP
      ENDDO
      NP = NP + 1
      PTS(NP) = 0D0
      DO I = 1, NP
        DO J = I+1, NP
          IF (PTS(J).LT.PTS(I)) THEN
            TMP = PTS(I)
            PTS(I) = PTS(J)
            PTS(J) = TMP
          ENDIF
        ENDDO
      ENDDO
      S = 0D0
      DO K = 1, NP-1
        A = PTS(K)
        C = PTS(K+1)
        IF (C-A.LE.1D-15) CYCLE
C       u = (beta - a)^(1/6): beta = a + u^6 (smooth for beta^(-delta))
        DO I = 1, 48
          U = 0.5D0*(1D0+XG(I))*(C-A)**(1D0/6D0)
          BETA = A + U**6
          JAC = 6D0*U**5*0.5D0*(C-A)**(1D0/6D0)*WG(I)
          IF (BETA.LE.0D0 .OR. BETA.GE.1D0) CYCLE
          FT = 1D0
          IF (GPD_TMODEL.GE.1) FT = EXP(BT*T)
          IF (GPD_TMODEL.EQ.2) FT = FT*BETA**(-AP*T)
          H1 = 0D0
          H2 = 0D0
          IF (BETA.GE.LO1 .AND. BETA.LE.HI1)
     &       H1 = GPD_PROFILE(B, BETA, (X-BETA)/XI)
          IF (BETA.GE.LO2 .AND. BETA.LE.HI2)
     &       H2 = GPD_PROFILE(B, BETA, (X+BETA)/XI)
          IF (H1.EQ.0D0 .AND. H2.EQ.0D0) CYCLE
          CALL GPD_DD_F(IPART, KIND, BETA, MU, F1)
          CALL GPD_DD_F(IPART, KIND, -BETA, MU, F2)
          S = S + JAC*FT*(H1*F1 + H2*F2)/XI
        ENDDO
      ENDDO
      GPD_DD_INT = S
      END

      SUBROUTINE GPD_GAULEG(N, X, W)
C     Gauss-Legendre nodes and weights on (-1,1).
      IMPLICIT NONE
      INTEGER N, I, J, M
      DOUBLE PRECISION X(N), W(N), Z, Z1, P1, P2, P3, PP, PI
      PARAMETER (PI=3.14159265358979323846D0)
      M = (N+1)/2
      DO I = 1, M
        Z = COS(PI*(I-0.25D0)/(N+0.5D0))
        DO
          P1 = 1D0
          P2 = 0D0
          DO J = 1, N
            P3 = P2
            P2 = P1
            P1 = ((2D0*J-1D0)*Z*P2 - (J-1D0)*P3)/J
          ENDDO
          PP = N*(Z*P1-P2)/(Z*Z-1D0)
          Z1 = Z
          Z = Z1 - P1/PP
          IF (ABS(Z-Z1).LT.1D-15) EXIT
        ENDDO
        X(I) = -Z
        X(N+1-I) = Z
        W(I) = 2D0/((1D0-Z*Z)*PP*PP)
        W(N+1-I) = W(I)
      ENDDO
      END
