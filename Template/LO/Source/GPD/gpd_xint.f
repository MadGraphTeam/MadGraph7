C ======================================================================
C     x integration of the GPD convolution of a hard amplitude.
C
C     For one GPD channel (parton IPART of the probed proton) the hard
C     amplitudes of a process are given by a kernel
C        KERNEL(X, N, AV, AA)
C     returning, at the momentum fraction X, the N projected amplitudes
C     (colour flows times helicity configurations of the other legs) for
C     the vector (H, E) and the axial (Ht, Et) projections. For a gluon
C     channel the light-cone gauge factor 1/((x+xi-i0)(x-xi+i0)) is applied
C     here. The amplitudes are rational functions of x with simple poles
C     at the a_i where a propagator carrying x goes on shell (linear
C     denominators: P^2 = 0). With G the GPD and R_i the residues,
C
C       int_-1^1 G B = int_-1^1 [G B - sum_i G(a_i) R_i/(x - a_i)]
C                      + sum_i G(a_i) R_i [ln|(1-a_i)/(1+a_i)| - i pi s_i]
C
C     (complex a_i of unstable propagators: ln((a_i-1)/(a_i+1))). The
C     first integrand is bounded: it is integrated by adaptive Gauss-Kronrod
C     quadrature (GPD_XQUAD) or sampled at one Monte Carlo point
C     (GPD_XMC). The residues are numerical (symmetric limit with
C     Richardson extrapolation); a double pole that does not cancel (an
C     end-point divergence) is reported by GPD_XSETUP (IERR=1).
C
C     Results: C(N,4) = the generalised Compton form factors for H, E, Ht,
C     Et (H, E with the vector, Ht, Et with the axial amplitudes).
C ======================================================================
      MODULE GPD_XINT_MOD
      IMPLICIT NONE
      INTEGER MAXPOLE
      PARAMETER (MAXPOLE=400)
      INTEGER NAMPS, NPOLES
      COMPLEX*16 APOLE(MAXPOLE)
      INTEGER SPOLE(MAXPOLE)
      LOGICAL SUBPOLE(MAXPOLE)
      DOUBLE PRECISION GPOLE(4,MAXPOLE)
      COMPLEX*16, ALLOCATABLE :: RESV(:,:), RESA(:,:)
      DOUBLE PRECISION XIC, TC, MUC, DPWORST
      INTEGER IPARTC, NCALLK
      LOGICAL GLUONC
C     breakpoints of the quadrature
      INTEGER NBREAK
      DOUBLE PRECISION BREAKS(MAXPOLE+6)
C     pole groups: the propagators (indices of the process list) of every
C     group; GRPFULL: the residue is taken on the whole amplitude (the
C     gluon prefactor poles)
      INTEGER GRPN(MAXPOLE), GRPM(MAXPOLE,MAXPOLE)
      LOGICAL GRPFULL(MAXPOLE)
      END MODULE GPD_XINT_MOD

C ----------------------------------------------------------------------
      SUBROUTINE GPD_XKERNEL(KERNEL, X, N, BV, BA)
C     The kernel times the gluon prefactor 1/((x+xi)(x-xi)).
      USE GPD_XINT_MOD
      IMPLICIT NONE
      EXTERNAL KERNEL
      INTEGER N
      DOUBLE PRECISION X
      COMPLEX*16 BV(N), BA(N)
      DOUBLE PRECISION PREF
      CALL KERNEL(X, N, BV, BA)
      NCALLK = NCALLK + 1
      IF (GLUONC) THEN
        PREF = 1D0/((X+XIC)*(X-XIC))
        BV = BV*PREF
        BA = BA*PREF
      ENDIF
      END

C ----------------------------------------------------------------------
      SUBROUTINE GPD_XSETUP(KERNEL, N, XI, T, MU, IPART, NP, AP, SP, IERR)
C     Prepare the x integration of one channel at one phase-space point:
C     the poles AP(NP) (complex) with their i0 signs SP(NP), the residues
C     of the kernel there, the GPD values at the poles, and the
C     double-pole check. The poles at -xi and +xi of the gluon prefactor
C     are added here. IERR=1: a double pole that does not cancel.
      USE GPD_XINT_MOD
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      EXTERNAL KERNEL
      INTEGER N, IPART, NP, IERR
      DOUBLE PRECISION XI, T, MU
      COMPLEX*16 AP(*)
      INTEGER SP(*)
      INTEGER I, J, K
      DOUBLE PRECISION AR, D, EPS1, EPS2, DMIN, G4(4), DREL
      COMPLEX*16, ALLOCATABLE :: VP1(:),AP1(:),VM1(:),AM1(:)
      COMPLEX*16, ALLOCATABLE :: VP2(:),AP2(:),VM2(:),AM2(:)
      COMPLEX*16 DD
      COMPLEX*16, ALLOCATABLE :: RV3(:,:), RA3(:,:), DV3(:,:), DA3(:,:)
      DOUBLE PRECISION SCALE, DDMAX
      INTEGER L
      LOGICAL DUP

      CALL GPD_INIT()
      IERR = 0
C     real poles at -+xi up to rounding (a propagator next to a GPD leg that
C     meets an on-shell leg of the same mass) are put at -+xi exactly: in a
C     boosted frame the rounding of K^2 can leave them ~1e-13 off, and the
C     window between the pole and the breakpoint at -+xi would then be as
C     narrow, with propagators evaluated where they are rounding noise
      DO I = 1, NP
        IF (DIMAG(AP(I)).NE.0D0) CYCLE
        IF (ABS(DBLE(AP(I))-XI).LT.1D-8*XI) AP(I) = DCMPLX(XI, 0D0)
        IF (ABS(DBLE(AP(I))+XI).LT.1D-8*XI) AP(I) = DCMPLX(-XI, 0D0)
      ENDDO
      NAMPS = N
      XIC = XI
      TC = T
      MUC = MU
      IPARTC = IPART
      GLUONC = (IPART.EQ.21)
      NCALLK = 0
      DPWORST = 0D0
      IF (ALLOCATED(RESV)) DEALLOCATE(RESV, RESA)
      ALLOCATE(RESV(N,MAXPOLE), RESA(N,MAXPOLE))
      ALLOCATE(VP1(N),AP1(N),VM1(N),AM1(N),VP2(N),AP2(N),VM2(N),AM2(N))
      ALLOCATE(RV3(N,3),RA3(N,3),DV3(N,3),DA3(N,3))
C     the pole groups: the given propagators grouped by position and i0
C     sign (coincident poles with opposite signs stay apart: their residues
C     are taken on their own diagrams), then, for a gluon, the poles of
C     1/((x+xi-i0)(x-xi+i0)), whose residues need the whole amplitude
      NPOLES = 0
      DO I = 1, NP + 2
        IF (I.GT.NP .AND. .NOT.GLUONC) CYCLE
        IF (I.LE.NP) THEN
          IF (ABS(DBLE(AP(I))).GE.1D0-1D-12) CYCLE
          IF (ABS(DIMAG(AP(I))).GT.0.25D0) CYCLE
        ENDIF
        DUP = .FALSE.
        DO J = 1, NPOLES
          IF (I.LE.NP) THEN
            IF (ABS(APOLE(J)-AP(I)).LT.1D-10*MAX(1D0,ABS(AP(I))) .AND.
     &          (SPOLE(J).EQ.SP(I) .OR. GRPFULL(J))) THEN
              DUP = .TRUE.
              GRPN(J) = GRPN(J) + 1
              GRPM(GRPN(J),J) = I
            ENDIF
          ELSE
            AR = -XI
            IF (I.EQ.NP+2) AR = XI
            IF (ABS(APOLE(J)-AR).LT.1D-10) THEN
              DUP = .TRUE.
              GRPFULL(J) = .TRUE.
              SPOLE(J) = -1
              IF (I.EQ.NP+2) SPOLE(J) = 1
            ENDIF
          ENDIF
        ENDDO
        IF (DUP) CYCLE
        NPOLES = NPOLES + 1
        IF (NPOLES.GT.MAXPOLE) STOP 'GPD_XSETUP: too many poles'
        GRPN(NPOLES) = 0
        GRPFULL(NPOLES) = .FALSE.
        IF (I.LE.NP) THEN
          APOLE(NPOLES) = AP(I)
          SPOLE(NPOLES) = SP(I)
          GRPN(NPOLES) = 1
          GRPM(1,NPOLES) = I
        ELSE IF (I.EQ.NP+1) THEN
          APOLE(NPOLES) = DCMPLX(-XI, 0D0)
          SPOLE(NPOLES) = -1
          GRPFULL(NPOLES) = .TRUE.
        ELSE
          APOLE(NPOLES) = DCMPLX(XI, 0D0)
          SPOLE(NPOLES) = 1
          GRPFULL(NPOLES) = .TRUE.
        ENDIF
      ENDDO
C     residues and double-pole coefficients, from (x-a)B and (x-a)^2 B at
C     a+-e, a+-e/2, a+-e/4: r(e) = R + c2 e^2 + c4 e^4 is extrapolated by
C     R = [64 r(e/4) - 20 r(e/2) + r(e)]/45 (error O(e^6)); same for D.
      DO I = 1, NPOLES
        AR = DBLE(APOLE(I))
        SUBPOLE(I) = .TRUE.
C       the step: a fraction of the distance to the nearest pole or end
        DMIN = MIN(1D0-AR, 1D0+AR)
        DO J = 1, NPOLES
          IF (J.EQ.I) CYCLE
          IF (ABS(DBLE(APOLE(J))-AR).LT.1D-10) CYCLE
          DMIN = MIN(DMIN, ABS(DBLE(APOLE(J))-AR))
        ENDDO
        IF (DMIN.LE.0D0) THEN
          WRITE(*,*) 'GPD: two poles coincide at x =', AR
          IERR = 2
          DMIN = 1D-6
        ENDIF
        EPS1 = MIN(GPD_EPSRES*DMIN, 1D-2)
        RV3 = (0D0,0D0)
        RA3 = (0D0,0D0)
        DV3 = (0D0,0D0)
        DA3 = (0D0,0D0)
        IF (.NOT.GRPFULL(I)) THEN
          GPD_NMASK = GRPN(I)
          GPD_MASKIDS(1:GRPN(I)) = GRPM(1:GRPN(I),I)
        ENDIF
        DO L = 1, 3
          EPS2 = EPS1/2D0**(L-1)
          CALL GPD_XKERNEL(KERNEL, AR+EPS2, N, VP1, AP1)
          CALL GPD_XKERNEL(KERNEL, AR-EPS2, N, VM1, AM1)
          RV3(:,L) = 0.5D0*((AR+EPS2-APOLE(I))*VP1 + (AR-EPS2-APOLE(I))*VM1)
          RA3(:,L) = 0.5D0*((AR+EPS2-APOLE(I))*AP1 + (AR-EPS2-APOLE(I))*AM1)
          DV3(:,L) = 0.5D0*EPS2**2*(VP1 + VM1)
          DA3(:,L) = 0.5D0*EPS2**2*(AP1 + AM1)
        ENDDO
        GPD_NMASK = 0
        DDMAX = 0D0
        SCALE = 1D-300
        DO K = 1, N
          RESV(K,I) = (64D0*RV3(K,3) - 20D0*RV3(K,2) + RV3(K,1))/45D0
          RESA(K,I) = (64D0*RA3(K,3) - 20D0*RA3(K,2) + RA3(K,1))/45D0
C         double pole: d(e) = e^2/2 [B(a+e) + B(a-e)] -> D + c e^2 + ...,
C         compared with the scale of all the amplitudes at this pole (a
C         component that vanishes is rounding noise, not a scale)
          IF (DIMAG(APOLE(I)).EQ.0D0) THEN
            DD = (64D0*DV3(K,3) - 20D0*DV3(K,2) + DV3(K,1))/45D0
            DDMAX = MAX(DDMAX, ABS(DD))
            DD = (64D0*DA3(K,3) - 20D0*DA3(K,2) + DA3(K,1))/45D0
            DDMAX = MAX(DDMAX, ABS(DD))
            SCALE = MAX(SCALE, ABS(DV3(K,1)), ABS(RESV(K,I))*EPS1,
     &                  ABS(DA3(K,1)), ABS(RESA(K,I))*EPS1)
          ENDIF
        ENDDO
        IF (DIMAG(APOLE(I)).EQ.0D0) THEN
          DREL = DDMAX/SCALE
          IF (DREL.GT.DPWORST) DPWORST = DREL
          IF (DREL.GT.GPD_DPTOL) THEN
            IF (IERR.NE.1) WRITE(*,'(A,F14.10,A,F12.8,A,ES9.2)')
     &        ' GPD: double pole at x =', AR, ' (xi =', XI,
     &        '), relative size', DREL
            IERR = 1
          ENDIF
        ENDIF
        CALL GPD_EVAL(IPART, AR, XI, T, MU, G4)
        GPOLE(:,I) = G4
      ENDDO
C     breakpoints: -1, -xi, xi, real parts of the poles, 1
      NBREAK = 0
      CALL GPD_ADDBREAK(-1D0)
      CALL GPD_ADDBREAK(1D0)
      CALL GPD_ADDBREAK(-XI)
      CALL GPD_ADDBREAK(XI)
      DO I = 1, NPOLES
        CALL GPD_ADDBREAK(DBLE(APOLE(I)))
      ENDDO
      DEALLOCATE(VP1,AP1,VM1,AM1,VP2,AP2,VM2,AM2)
      DEALLOCATE(RV3,RA3,DV3,DA3)
      END

      SUBROUTINE GPD_ADDBREAK(X)
      USE GPD_XINT_MOD
      IMPLICIT NONE
      DOUBLE PRECISION X, TMP
      INTEGER I, J
      IF (X.LT.-1D0 .OR. X.GT.1D0) RETURN
      DO I = 1, NBREAK
        IF (ABS(BREAKS(I)-X).LT.1D-13) RETURN
      ENDDO
      NBREAK = NBREAK + 1
      BREAKS(NBREAK) = X
      DO I = NBREAK, 2, -1
        IF (BREAKS(I).LT.BREAKS(I-1)) THEN
          TMP = BREAKS(I)
          BREAKS(I) = BREAKS(I-1)
          BREAKS(I-1) = TMP
        ENDIF
      ENDDO
      END

C ----------------------------------------------------------------------
      SUBROUTINE GPD_XINTEGRAND(KERNEL, X, N, F)
C     The subtracted integrand F(N,4) at X (H, E, Ht, Et).
      USE GPD_XINT_MOD
      IMPLICIT NONE
      EXTERNAL KERNEL
      INTEGER N
      DOUBLE PRECISION X
      COMPLEX*16 F(N,4)
      COMPLEX*16 BV(N), BA(N), DEN
      DOUBLE PRECISION G4(4)
      INTEGER I
      CALL GPD_XKERNEL(KERNEL, X, N, BV, BA)
      CALL GPD_EVAL(IPARTC, X, XIC, TC, MUC, G4)
      F(:,1) = G4(1)*BV
      F(:,2) = G4(2)*BV
      F(:,3) = G4(3)*BA
      F(:,4) = G4(4)*BA
      DO I = 1, NPOLES
        IF (.NOT.SUBPOLE(I)) CYCLE
        DEN = 1D0/(X - APOLE(I))
        F(:,1) = F(:,1) - GPOLE(1,I)*RESV(:,I)*DEN
        F(:,2) = F(:,2) - GPOLE(2,I)*RESV(:,I)*DEN
        F(:,3) = F(:,3) - GPOLE(3,I)*RESA(:,I)*DEN
        F(:,4) = F(:,4) - GPOLE(4,I)*RESA(:,I)*DEN
      ENDDO
      END

      SUBROUTINE GPD_XINTEGRAND2(KERNEL, U, N, F, ISYM, XC)
C     f(U) (ISYM=0) or f(XC+U) + f(XC-U) (ISYM=1).
      IMPLICIT NONE
      EXTERNAL KERNEL
      INTEGER N, ISYM
      DOUBLE PRECISION U, XC
      COMPLEX*16 F(N,4), F2(N,4)
      IF (ISYM.EQ.0) THEN
        CALL GPD_XINTEGRAND(KERNEL, U, N, F)
      ELSE
        CALL GPD_XINTEGRAND(KERNEL, XC+U, N, F)
        CALL GPD_XINTEGRAND(KERNEL, XC-U, N, F2)
        F = F + F2
      ENDIF
      END

C ----------------------------------------------------------------------
      SUBROUTINE GPD_XADDBACK(N, C)
C     C(N,4) = sum_i G(a_i) R_i int_-1^1 dx/(x - a_i + i s_i 0)
      USE GPD_XINT_MOD
      IMPLICIT NONE
      INTEGER N
      COMPLEX*16 C(N,4)
      COMPLEX*16 L
      DOUBLE PRECISION A, PI
      PARAMETER (PI=3.14159265358979323846D0)
      INTEGER I
      C = (0D0,0D0)
      DO I = 1, NPOLES
        IF (.NOT.SUBPOLE(I)) CYCLE
        IF (DIMAG(APOLE(I)).EQ.0D0) THEN
          A = DBLE(APOLE(I))
          L = DCMPLX(LOG(ABS((1D0-A)/(1D0+A))), -PI*SPOLE(I))
        ELSE
          L = LOG((APOLE(I)-1D0)/(APOLE(I)+1D0))
        ENDIF
        C(:,1) = C(:,1) + GPOLE(1,I)*RESV(:,I)*L
        C(:,2) = C(:,2) + GPOLE(2,I)*RESV(:,I)*L
        C(:,3) = C(:,3) + GPOLE(3,I)*RESA(:,I)*L
        C(:,4) = C(:,4) + GPOLE(4,I)*RESA(:,I)*L
      ENDDO
      END

C ----------------------------------------------------------------------
      SUBROUTINE GPD_XQUAD(KERNEL, N, C)
C     C(N,4): the CFFs by adaptive Gauss-Kronrod (7-15) quadrature of the
C     subtracted integrand, plus the add-back. Around every subtracted real
C     pole a the integral runs over a symmetric window, as
C     int_0^h [f(a+u) + f(a-u)] du: whatever is left of 1/(x-a) after the
C     subtraction (the error of the numerical residue) cancels exactly
C     there, as in a principal value. Elsewhere the pieces run between the
C     breakpoints (-1, -xi, xi, the poles, 1).
      USE GPD_XINT_MOD
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      EXTERNAL KERNEL
      INTEGER N
      COMPLEX*16 C(N,4)
      COMPLEX*16, ALLOCATABLE :: RK(:,:), RG(:,:), TOT(:,:), ADD(:,:)
      DOUBLE PRECISION HW(MAXPOLE+6), A, B, SIZE, U0
      LOGICAL ISPOLE(MAXPOLE+6)
      INTEGER IB, I
      ALLOCATE(RK(N,4), RG(N,4), TOT(N,4), ADD(N,4))
C     which breakpoints are real subtracted poles, and their windows
      DO IB = 1, NBREAK
        ISPOLE(IB) = .FALSE.
        HW(IB) = 0D0
        IF (IB.EQ.1 .OR. IB.EQ.NBREAK) CYCLE
        DO I = 1, NPOLES
          IF (SUBPOLE(I) .AND. DIMAG(APOLE(I)).EQ.0D0 .AND.
     &        ABS(DBLE(APOLE(I))-BREAKS(IB)).LT.1D-13) ISPOLE(IB)=.TRUE.
        ENDDO
        IF (ISPOLE(IB)) HW(IB) = 0.5D0*MIN(BREAKS(IB)-BREAKS(IB-1),
     &                                    BREAKS(IB+1)-BREAKS(IB))
      ENDDO
C     the absolute scale of the tolerance: a coarse first pass
      CALL GPD_XADDBACK(N, ADD)
      SIZE = MAXVAL(ABS(ADD))
      DO IB = 1, NBREAK-1
        CALL GPD_GK15(KERNEL, BREAKS(IB), BREAKS(IB+1), N, RK, RG, 0, 0D0)
        SIZE = SIZE + MAXVAL(ABS(RK))
      ENDDO
      SIZE = MAX(SIZE, 1D-300)
      TOT = (0D0,0D0)
      DO IB = 1, NBREAK-1
        A = BREAKS(IB) + HW(IB)
        B = BREAKS(IB+1) - HW(IB+1)
        IF (B.GT.A) THEN
          CALL GPD_ADAPT(KERNEL, A, B, N, SIZE, 0, 0D0, RK)
          TOT = TOT + RK
        ENDIF
      ENDDO
C     the windows: the inner part u < U0 = 1e-2 h by two-point Gauss (exact
C     to cubic order in u; a pole on the kink of the GPD at -+xi makes the
C     window integrand linear there; the next singularity is >= 2h away, so
C     the rule is exact to ~1e-14), so that the integrand is never taken
C     where a double pole that cancels between graphs leaves rounding noise
C     of size eps D/u^2 (eta_c gamma: 1e-9 at U0 = 1e-5 h, 2e-12 at 1e-2 h)
      DO IB = 2, NBREAK-1
        IF (.NOT.ISPOLE(IB)) CYCLE
        U0 = 1D-2*HW(IB)
        CALL GPD_ADAPT(KERNEL, U0, HW(IB), N, SIZE, 1, BREAKS(IB), RK)
        TOT = TOT + RK
        CALL GPD_XINTEGRAND2(KERNEL, U0*(0.5D0-0.5D0/SQRT(3D0)), N,
     &       RG, 1, BREAKS(IB))
        TOT = TOT + 0.5D0*U0*RG
        CALL GPD_XINTEGRAND2(KERNEL, U0*(0.5D0+0.5D0/SQRT(3D0)), N,
     &       RG, 1, BREAKS(IB))
        TOT = TOT + 0.5D0*U0*RG
      ENDDO
      C = TOT + ADD
      DEALLOCATE(RK, RG, TOT, ADD)
      END

      SUBROUTINE GPD_ADAPT(KERNEL, A0, B0, N, SIZE, ISYM, XC, RES)
C     Adaptive GK15 of the subtracted integrand on [A0,B0] (ISYM=0), or of
C     f(XC+u) + f(XC-u) on u in [A0,B0] (ISYM=1).
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      EXTERNAL KERNEL
      INTEGER N, ISYM
      DOUBLE PRECISION A0, B0, SIZE, XC
      COMPLEX*16 RES(N,4)
      INTEGER MAXSTACK, MAXCALL
      PARAMETER (MAXSTACK=4000)
C     intervals per window before the refinement stops (an integrand that
C     never meets the tolerance would otherwise take ~2^30 of them)
      PARAMETER (MAXCALL=20000)
      DOUBLE PRECISION SA(MAXSTACK), SB(MAXSTACK), A, B, ERR
      COMPLEX*16 RK(N,4), RG(N,4)
      INTEGER NS, NCALL
      LOGICAL WARNNAN, WARNCAP
      DATA WARNNAN/.FALSE./, WARNCAP/.FALSE./
      SAVE WARNNAN, WARNCAP
      RES = (0D0,0D0)
      NCALL = 0
      NS = 1
      SA(1) = A0
      SB(1) = B0
      DO WHILE (NS.GT.0)
        A = SA(NS)
        B = SB(NS)
        NS = NS - 1
        CALL GPD_GK15(KERNEL, A, B, N, RK, RG, ISYM, XC)
        NCALL = NCALL + 1
        ERR = MAXVAL(ABS(RK-RG))
        IF (ERR.NE.ERR) THEN
C         a NaN integrand: no refinement can help, return it at once
          IF (.NOT.WARNNAN) THEN
            WRITE(*,*) 'GPD: NaN in the x integrand on [', A, ',', B,
     &        ']; the x integral is NaN'
            WARNNAN = .TRUE.
          ENDIF
          RES = RK
          RETURN
        ENDIF
        IF (NCALL.EQ.MAXCALL .AND. .NOT.WARNCAP) THEN
          WRITE(*,*) 'GPD: the x quadrature stopped refining after',
     &      MAXCALL, ' intervals of a window (gpd_xtol not reached)'
          WARNCAP = .TRUE.
        ENDIF
C       (no bisection below 1e-9 of the interval: below that the error
C       estimate is rounding, and refining further only accumulates it)
        IF (ERR.LE.GPD_XTOL*SIZE*MAX((B-A)/2D0,1D-3) .OR.
     &      (B-A).LT.1D-9*(B0-A0) .OR. NS.GE.MAXSTACK-2 .OR.
     &      NCALL.GE.MAXCALL) THEN
          RES = RES + RK
        ELSE
          NS = NS + 1
          SA(NS) = A
          SB(NS) = 0.5D0*(A+B)
          NS = NS + 1
          SA(NS) = 0.5D0*(A+B)
          SB(NS) = B
        ENDIF
      ENDDO
      END

      SUBROUTINE GPD_GK15(KERNEL, A, B, N, RK, RG, ISYM, XC)
C     Gauss-Kronrod 7-15 rule on [A,B]: RK (Kronrod), RG (Gauss), of the
C     subtracted integrand (ISYM=0) or of f(XC+u) + f(XC-u) (ISYM=1).
      IMPLICIT NONE
      EXTERNAL KERNEL
      INTEGER N, ISYM
      DOUBLE PRECISION A, B, XC
      COMPLEX*16 RK(N,4), RG(N,4)
      COMPLEX*16 F(N,4)
      DOUBLE PRECISION XGK(8), WGK(8), WG(4), C, H
      INTEGER J
      DATA XGK/0.991455371120812639206854697526329D0,
     &  0.949107912342758524526189684047851D0,
     &  0.864864423359769072789712788640926D0,
     &  0.741531185599394439863864773280788D0,
     &  0.586087235467691130294144845693013D0,
     &  0.405845151377397166906606412076961D0,
     &  0.207784955007898467600689403773245D0,
     &  0.000000000000000000000000000000000D0/
      DATA WGK/0.022935322010529224963732008058970D0,
     &  0.063092092629978553290700663189204D0,
     &  0.104790010322250183839876322541518D0,
     &  0.140653259715525918745189590510238D0,
     &  0.169004726639267902826583426598550D0,
     &  0.190350578064785409913256402421014D0,
     &  0.204432940075298892414161999234649D0,
     &  0.209482141084727828012999174891714D0/
      DATA WG/0.129484966168869693270611432679082D0,
     &  0.279705391489276667901467771423780D0,
     &  0.381830050505118944950369775488975D0,
     &  0.417959183673469387755102040816327D0/
      C = 0.5D0*(A+B)
      H = 0.5D0*(B-A)
      CALL GPD_XINTEGRAND2(KERNEL, C, N, F, ISYM, XC)
      RK = WGK(8)*F
      RG = WG(4)*F
      DO J = 1, 7
        CALL GPD_XINTEGRAND2(KERNEL, C-H*XGK(J), N, F, ISYM, XC)
        RK = RK + WGK(J)*F
        IF (MOD(J,2).EQ.0) RG = RG + WG(J/2)*F
        CALL GPD_XINTEGRAND2(KERNEL, C+H*XGK(J), N, F, ISYM, XC)
        RK = RK + WGK(J)*F
        IF (MOD(J,2).EQ.0) RG = RG + WG(J/2)*F
      ENDDO
      RK = RK*H
      RG = RG*H
      END

C ----------------------------------------------------------------------
      SUBROUTINE GPD_XMAP(R, XI, X, W)
C     Monte Carlo map R in (0,1) -> x in (-1,1) with weight W = 1/density:
C     0.4 for each DGLAP region (density ~ 1/|x|), 0.2 for the ERBL region
C     (flat).
      IMPLICIT NONE
      DOUBLE PRECISION R, XI, X, W, U, P1, P2, LN
      PARAMETER (P1=0.4D0, P2=0.2D0)
      LN = LOG(1D0/XI)
      IF (R.LT.P1) THEN
        U = R/P1
        X = -XI*EXP(U*LN)
        W = -X*LN/P1
      ELSE IF (R.LT.P1+P2) THEN
        U = (R-P1)/P2
        X = -XI + 2D0*XI*U
        W = 2D0*XI/P2
      ELSE
        U = (R-P1-P2)/P1
        X = XI*EXP(U*LN)
        W = X*LN/P1
      ENDIF
      END

      SUBROUTINE GPD_XMC(KERNEL, R, N, C)
C     One Monte Carlo estimate of the CFFs: W(x) F(x) + add-back, with x
C     the image of R in (0,1). Its average over R is the integral.
      USE GPD_XINT_MOD
      IMPLICIT NONE
      EXTERNAL KERNEL
      INTEGER N
      DOUBLE PRECISION R
      COMPLEX*16 C(N,4), F(N,4)
      DOUBLE PRECISION X, W
      CALL GPD_XMAP(R, XIC, X, W)
      CALL GPD_XINTEGRAND(KERNEL, X, N, F)
      CALL GPD_XADDBACK(N, C)
      C = C + W*F
      END

C ----------------------------------------------------------------------
      SUBROUTINE GPD_POLE_ADD(K, SIG, M, W, P, NP, AP, SP)
C     The pole in x of a propagator of momentum K + SIG x P (P^2 = 0),
C     mass M and width W: (K + SIG x P)^2 - M^2 + i M W
C       = 2 SIG P.K (x - a) + i M W,   a = -(K^2 - M^2)/(2 SIG P.K),
C     i.e. the pole sits at a - i M W/(2 SIG P.K) and, for W = 0, has the
C     i0 sign s = sign(SIG P.K): 1/(x - a + i s 0).
      IMPLICIT NONE
      DOUBLE PRECISION K(0:3), SIG, M, W, P(0:3)
      INTEGER NP, SP(*)
      COMPLEX*16 AP(*)
      DOUBLE PRECISION PK, KK, A
C     light-cone forms of P.K and K^2 (exact identities): in a frame
C     boosted along z they keep the digits that P(0)K(0) - P(3)K(3) and
C     K(0)^2 - K(3)^2 cancel
      PK = SIG*(0.5D0*((P(0)+P(3))*(K(0)-K(3)) + (P(0)-P(3))*(K(0)+K(3)))
     &     - P(1)*K(1) - P(2)*K(2))
      KK = (K(0)-K(3))*(K(0)+K(3)) - K(1)**2 - K(2)**2
      IF (PK.EQ.0D0) RETURN
      A = -(KK - M*M)/(2D0*PK)
      NP = NP + 1
      IF (W.EQ.0D0) THEN
        AP(NP) = DCMPLX(A, 0D0)
      ELSE
        AP(NP) = DCMPLX(A, -M*W/(2D0*PK))
      ENDIF
      SP(NP) = NINT(SIGN(1D0, PK))
      END
