C     =================================================================
C     =====
C     Bound states (quarkonia, leptonia) in MadLoop
C     =================================================================
C     =====
C     
C     MadLoop computes the loop amplitudes of the constituent process,
C      the
C     constituents of a bound state of momentum K and mass M = m1+m2
C      at zero
C     relative momentum, p_i = (m_i/M) K. The amplitudes are projected
C      onto
C     the bound states as at tree level (arXiv:2510.26773): the colour
C     projector is in the colour factors, the spin projector is the
C      sum over
C     the constituent helicities h weighted by PROJ(h,J_z)
C      (ONIA_LOOP_PROJ).
C     The projected interference with the Born is MadLoop's ordinary
C      one
C     with the Born amplitudes A(h) replaced by the effective Born
C     W(h) = F sum_{J_z} conj(PROJ(h,J_z)) B(J_z),
C     B(J_z) = sum_h' PROJ(h',J_z) A(h'),
C     where h' runs over the configurations with the helicities of h
C      for
C     the other legs and F is the colour and LDME normalisation of the
C     tree-level output. The users of MadLoop see the physical process:
C     SLOOPMATRIX takes the momenta of the bound states, not of their
C     constituents.
C     
C     Process: a a > etac(1|1S01) [ virt = QCD ]

      SUBROUTINE ML5_0_SLOOPMATRIX(P_PHYS,ANS)
C     The loop matrix element for the physical momenta P_PHYS (a bound
C     state with mass m1+m2), computed from its constituent legs.
C     After SLOOPMATRIXHEL(P_PHYS,HEL,ANS) (USERHEL=HEL) it is that of
C      the
C     physical helicity configuration HEL (1 to ONIA_PHYS_NCOMB): the
C     helicity of each elementary leg and the J_z of each bound state,
C      in
C     the order of the helicity matrix of the physical process (as the
C     tree-level output and the FKS Born). The constituent helicities
C      are
C     summed over; as in MadLoop, a configuration is not averaged over
C      the
C     initial helicities, so the configurations add up to ONIA_HELAVG
C     times the unselected result.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=4, NEXTERNAL_PHYS=3)
      REAL*8 P_PHYS(0:3,NEXTERNAL_PHYS), ANS(0:3,*)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER USERHEL
      COMMON/ML5_0_USERCHOICE/USERHEL
      INCLUDE 'nsquaredSO.inc'
      INTEGER RET_CODE_H,RET_CODE_T,RET_CODE_U
      REAL*8 ACCURACY(0:NSQUAREDSO)
      COMMON/ML5_0_ACC/ACCURACY,RET_CODE_H,RET_CODE_T,RET_CODE_U
      INTEGER ML_GET_RET_CODE_LIBRARY, SET_RET_CODE_U
      INTEGER LIB
      LOGICAL WARNED
      DATA WARNED/.FALSE./
      SAVE WARNED
      REAL*8 QREST(3)
      COMMON/ML5_0_ONIA_RELMOM/QREST
      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     &  FPE_IN_QP_REDUCTION
      LOGICAL FAILED_WARNED
      DATA FAILED_WARNED/.FALSE./
      SAVE FAILED_WARNED
      INTEGER I
      INCLUDE 'onia_loop_data.inc'
C     A physical helicity configuration (SLOOPMATRIXHEL): the J_z of
C      each
C     bound state (ONIA_SELECT_JZ) and the helicities of the other legs
C     (ONIA_OTHERS_SEL); a J_z selection of the user is kept aside
      INTEGER HEL_PHYS, IL, JZSEL(1), JZMASK_USER(ONIA_NONIA), NDIM_HEL
      HEL_PHYS=USERHEL
      IF (HEL_PHYS.NE.-1) THEN
        IF (HEL_PHYS.LT.1.OR.HEL_PHYS.GT.ONIA_PHYS_NCOMB) THEN
          WRITE(*,*) 'ERROR:: SLOOPMATRIXHEL: helicity configuration',
     &     HEL_PHYS,' out of range (1 to',ONIA_PHYS_NCOMB,')'
          STOP 1
        ENDIF
        CALL ML5_0_ONIA_JZ_MASK(.FALSE.,JZMASK_USER)
        DO IL=1,ONIA_NPHYS
          IF (ONIA_PHYS_LEG(IL).LT.0) THEN
            JZSEL(1)=ONIA_PHYS_HEL(IL,HEL_PHYS)
            CALL ML5_0_ONIA_SELECT_JZ(-ONIA_PHYS_LEG(IL),1,JZSEL)
          ENDIF
        ENDDO
        CALL ML5_0_ONIA_OTHERS_SEL(.TRUE.,
     &   ONIA_PHYS_OTHERS(HEL_PHYS))
C       MadLoop sums over the selected constituent configurations
        USERHEL=-1
      ENDIF
      CALL ML5_0_ONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
      CALL ML5_0_SLOOPMATRIX_CONSTITUENTS(P,ANS)
C     At threshold only TIRust (MLReductionLib=9) reduces the integrals
C     reliably. The other libraries can return wrong values that pass
C     MadLoop's stability tests (the Gram determinants vanish and the
C     Coulomb-exchange propagators are linearly dependent), so a result
C     they give is reported unstable (H=4, U=0) whatever its accuracy.
C     With a relative momentum (diagnostic) the integrals are ordinary
C      ones,
C     and without coloured constituents (leptonia) no QCD loop holds
C      both
C     constituents of a bound state.
      IF ((RET_CODE_H.EQ.2.OR.RET_CODE_H.EQ.3).AND.ALL(QREST.EQ.0D0)
     &  .AND.ONIA_THRESHOLD_LOOPS) THEN
      LIB=ML_GET_RET_CODE_LIBRARY()
      IF (LIB.NE.9) THEN
        IF (.NOT.WARNED) THEN
          WRITE(*,*) 'WARNING:: bound states at threshold: only',
     &       ' TIRust (MLReductionLib=9) is reliable; the results of',
     &       ' library',LIB,' are reported unstable.'
          WARNED=.TRUE.
        ENDIF
        RET_CODE_H=4
        RET_CODE_U=SET_RET_CODE_U(LIB,.FALSE.,.FALSE.)
      ENDIF
      ENDIF
C     A failed reduction (an integral TIRust refuses, or a
C     floating-point exception) leaves its integrals out of the
C     result. MadLoop's stability tests (CTModeRun=-1) then try the
C     next library or quadruple precision, and report the point
C     unstable if none succeeds. The forced modes (CTModeRun=1 or 4)
C     and the initialisation points run no test: the incomplete result
C     is reported unstable here. The flags are those of the evaluation
C     returned (MadLoop clears them when it changes library or
C     precision).
      IF ((FPE_IN_DP_REDUCTION.OR.FPE_IN_QP_REDUCTION)
     &  .AND.RET_CODE_H.NE.4) THEN
      IF (.NOT.FAILED_WARNED) THEN
        WRITE(*,*) 'WARNING:: bound states: the reduction of',
     &     ' some loop integrals failed (TIRust refuses some at',
     &     ' this point). Their contribution is missing; the result',
     &     ' is reported unstable.'
        FAILED_WARNED=.TRUE.
      ENDIF
      RET_CODE_H=4
      RET_CODE_U=SET_RET_CODE_U(ML_GET_RET_CODE_LIBRARY(),.FALSE.,
     &   .FALSE.)
      DO I=0,NSQUAREDSO
        ACCURACY(I)=1D0
      ENDDO
      ENDIF
      IF (HEL_PHYS.NE.-1) THEN
C       As MadLoop's SLOOPMATRIXHEL and the tree-level SMATRIXHEL: one
C       configuration is not averaged over the initial helicities (the
C       configurations add up to ONIA_HELAVG times the unselected
C        result)
        CALL ML5_0_GET_ANSWER_DIMENSION(NDIM_HEL)
        DO I=0,NDIM_HEL
          DO IL=0,3
            ANS(IL,I)=ANS(IL,I)*ONIA_HELAVG
          ENDDO
        ENDDO
        CALL ML5_0_ONIA_JZ_MASK(.TRUE.,JZMASK_USER)
        CALL ML5_0_ONIA_OTHERS_SEL(.TRUE.,0)
C       as MadLoop's SLOOPMATRIXHEL, which leaves USERHEL set
        USERHEL=HEL_PHYS
      ENDIF
      END

      SUBROUTINE ML5_0_ONIA_MASSES(CMASS)
C     The masses of the constituent legs
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
      REAL*8 CMASS(NEXTERNAL)
      INCLUDE 'coupl.inc'
      CMASS(1)=ZERO
      CMASS(2)=ZERO
      CMASS(3)=ABS(MDL_MC)
      CMASS(4)=ABS(MDL_MC)
      END

      SUBROUTINE ML5_0_MP_ONIA_MASSES(CMASS)
C     The masses of the constituent legs, in quadruple precision
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E0_16)
      REAL*16 CMASS(NEXTERNAL)
      INCLUDE 'mp_coupl_same_name.inc'
      CMASS(1)=ZERO
      CMASS(2)=ZERO
      CMASS(3)=SQRT(REAL(REAL(ABS(MDL_MC),KIND=8)**2,KIND=16))
      CMASS(4)=SQRT(REAL(REAL(ABS(MDL_MC),KIND=8)**2,KIND=16))
      END

      SUBROUTINE ML5_0_ONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
C     The momenta of the constituent legs: r_i K for the constituents
C      of
C     a bound state of momentum K (r_i = m_i/(m1+m2), exactly 1/2 for
C     equal masses, so both constituents carry the same momentum bit
C      for
C     bit), the other momenta copied.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=4, NEXTERNAL_PHYS=3)
      REAL*8 P_PHYS(0:3,NEXTERNAL_PHYS), P(0:3,NEXTERNAL)
      REAL*8 CMASS(NEXTERNAL), FRAC, QLAB(0:3)
      INTEGER I, PARTNER, K
      INCLUDE 'onia_loop_data.inc'
      REAL*8 QREST(3)
      COMMON/ML5_0_ONIA_RELMOM/QREST
      CALL ML5_0_ONIA_MASSES(CMASS)
      DO I=1,NEXTERNAL
        PARTNER=ONIA_PARTNER(I)
        IF (PARTNER.EQ.0) THEN
          FRAC=1D0
        ELSE
          FRAC=CMASS(I)/(CMASS(I)+CMASS(PARTNER))
        ENDIF
        P(0:3,I)=FRAC*P_PHYS(0:3,ONIA_MAPPING(I))
      ENDDO
C     A relative momentum of the constituents (diagnostic, see
C     ONIA_SET_RELATIVE_MOMENTUM): +-q, q the rest-frame vector QREST
C     boosted to the frame of the bound state
      IF (ANY(QREST.NE.0D0)) THEN
        DO K=1,ONIA_NONIA
          CALL ML5_0_ONIA_BOOST_REST_VECTOR(
     &     P_PHYS(0,ONIA_MAPPING(ONIA_Q1(K))),QREST,QLAB)
          P(0:3,ONIA_Q1(K))=P(0:3,ONIA_Q1(K))+QLAB(0:3)
          P(0:3,ONIA_Q2(K))=P(0:3,ONIA_Q2(K))-QLAB(0:3)
        ENDDO
      ENDIF
      END

      SUBROUTINE ML5_0_ONIA_SET_RELATIVE_MOMENTUM(Q)
C     Diagnostic: give the constituents of every bound state the
C      relative
C     momentum +-Q (a 3-vector in the rest frame of the bound state,
C      which
C     must then have the mass 2 sqrt(m^2+Q^2), equal constituent
C      masses).
C     The default, Q = 0, is the zero relative momentum of the
C      projection.
C     Used to check the threshold integrals by extrapolating v -> 0
C      with
C     the Coulomb singularity separated (tools/onia_loop_check.py).
      IMPLICIT NONE
      REAL*8 Q(3)
      REAL*8 QREST(3)
      COMMON/ML5_0_ONIA_RELMOM/QREST
      QREST(1:3)=Q(1:3)
      END

      BLOCK DATA ML5_0_ONIA_RELMOM_DEFAULT
      IMPLICIT NONE
      REAL*8 QREST(3)
      COMMON/ML5_0_ONIA_RELMOM/QREST
      DATA QREST/3*0D0/
      END

      SUBROUTINE ML5_0_ONIA_BOOST_REST_VECTOR(K,Q,QLAB)
C     The 4-vector (0,Q) of the rest frame of K boosted to the frame
C      of K
      IMPLICIT NONE
      REAL*8 K(0:3), Q(3), QLAB(0:3), M, BQ, G
      M=SQRT(K(0)**2-K(1)**2-K(2)**2-K(3)**2)
      BQ=(K(1)*Q(1)+K(2)*Q(2)+K(3)*Q(3))/M
      G=BQ/(K(0)+M)
      QLAB(0)=BQ
      QLAB(1:3)=Q(1:3)+G*K(1:3)
      END

      SUBROUTINE ML5_0_MP_ONIA_BOOST_REST_VECTOR(K,Q,QLAB)
C     The 4-vector (0,Q) of the rest frame of K boosted to the frame
C      of K
      IMPLICIT NONE
      REAL*16 K(0:3), Q(3), QLAB(0:3), M, BQ, G
      M=SQRT(K(0)**2-K(1)**2-K(2)**2-K(3)**2)
      BQ=(K(1)*Q(1)+K(2)*Q(2)+K(3)*Q(3))/M
      G=BQ/(K(0)+M)
      QLAB(0)=BQ
      QLAB(1:3)=Q(1:3)+G*K(1:3)
      END

      SUBROUTINE ML5_0_MP_ONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
C     The momenta of the constituent legs, in quadruple precision
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=4, NEXTERNAL_PHYS=3)
      REAL*16 P_PHYS(0:3,NEXTERNAL_PHYS), P(0:3,NEXTERNAL)
      REAL*16 CMASS(NEXTERNAL), FRAC, QLAB(0:3), QMP(3)
      INTEGER I, PARTNER, K
      INCLUDE 'onia_loop_data.inc'
      REAL*8 QREST(3)
      COMMON/ML5_0_ONIA_RELMOM/QREST
      CALL ML5_0_MP_ONIA_MASSES(CMASS)
      DO I=1,NEXTERNAL
        PARTNER=ONIA_PARTNER(I)
        IF (PARTNER.EQ.0) THEN
          FRAC=1.0E0_16
        ELSE
          FRAC=CMASS(I)/(CMASS(I)+CMASS(PARTNER))
        ENDIF
        P(0:3,I)=FRAC*P_PHYS(0:3,ONIA_MAPPING(I))
      ENDDO
      IF (ANY(QREST.NE.0D0)) THEN
        QMP(1:3)=QREST(1:3)
        DO K=1,ONIA_NONIA
          CALL ML5_0_MP_ONIA_BOOST_REST_VECTOR(
     &     P_PHYS(0,ONIA_MAPPING(ONIA_Q1(K))),QMP,QLAB)
          P(0:3,ONIA_Q1(K))=P(0:3,ONIA_Q1(K))+QLAB(0:3)
          P(0:3,ONIA_Q2(K))=P(0:3,ONIA_Q2(K))-QLAB(0:3)
        ENDDO
      ENDIF
      END

      SUBROUTINE ML5_0_MP_ONIA_PHYSICAL_MOMENTA(P,P_PHYS)
C     The physical momenta (a bound state carries the sum of the
C      momenta
C     of its constituents), in quadruple precision
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=4, NEXTERNAL_PHYS=3)
      REAL*16 P_PHYS(0:3,NEXTERNAL_PHYS), P(0:3,NEXTERNAL)
      INTEGER I
      INCLUDE 'onia_loop_data.inc'
      P_PHYS(0:3,1:NEXTERNAL_PHYS)=0.0E0_16
      DO I=1,NEXTERNAL
        P_PHYS(0:3,ONIA_MAPPING(I))=P_PHYS(0:3,ONIA_MAPPING(I))+P(0:3
     $   ,I)
      ENDDO
      END

      SUBROUTINE ML5_0_MP_IMPROVE_PS_POINT_PRECISION(KEEP_OFFSHELL_MASS
     $ , P)
C     The precision improvement of a phase-space point with bound
C      states,
C     in place of MadLoop's generic one (which can fail for a bound
C      state at
C     rest). TIRust recognises the degenerate Coulomb-exchange
C      integrals at
C     the precision of its inputs, so the momenta must be exactly on
C      shell
C     with the masses the loops use (the double-precision squared
C      masses,
C     MP_ONIA_MASSES) and conserve momentum: the physical final-state
C     particles are put on their mass shell (keeping their 3-momenta; a
C     bound state of mass m1+m2), the initial momenta are rebuilt from
C      the
C     final ones along their original directions, and the bound states
C      are
C     split again into their constituents at zero relative momentum.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS, NINCOMING
      PARAMETER (NEXTERNAL=4, NEXTERNAL_PHYS=3)
      PARAMETER (NINCOMING=2)
      REAL*16 P(0:3,NEXTERNAL)
      LOGICAL KEEP_OFFSHELL_MASS(NEXTERNAL)
      REAL*16 P_PHYS(0:3,NEXTERNAL_PHYS)
      LOGICAL KEEP_PHYS(NEXTERNAL_PHYS)
      REAL*16 CMASS(NEXTERNAL), PMASS(NEXTERNAL_PHYS)
      REAL*16 PF(0:3), NDIR(3), A, C, E0, DEN, DISC, KMOM
      INTEGER I
      INCLUDE 'onia_loop_data.inc'
      REAL*8 QREST(3)
      COMMON/ML5_0_ONIA_RELMOM/QREST
      DO I=1,NEXTERNAL
        KEEP_PHYS(ONIA_MAPPING(I))=KEEP_OFFSHELL_MASS(I)
      ENDDO
      CALL ML5_0_MP_ONIA_PHYSICAL_MOMENTA(P,P_PHYS)
      CALL ML5_0_MP_ONIA_MASSES(CMASS)
      PMASS(1:NEXTERNAL_PHYS)=0.0E0_16
      DO I=1,NEXTERNAL
        IF (ONIA_PARTNER(I).NE.0.AND.ANY(QREST.NE.0D0)) THEN
C         the diagnostic relative momentum: 2 sqrt(m^2+q^2)
          PMASS(ONIA_MAPPING(I))=PMASS(ONIA_MAPPING(I))+SQRT(CMASS(I)*
     $     *2
     &     +REAL(QREST(1),16)**2+REAL(QREST(2),16)**2+REAL(QREST(3),16)**2)
        ELSE
          PMASS(ONIA_MAPPING(I))=PMASS(ONIA_MAPPING(I))+CMASS(I)
        ENDIF
      ENDDO
      PF(0:3)=0.0E0_16
      DO I=NINCOMING+1,NEXTERNAL_PHYS
        IF (.NOT.KEEP_PHYS(I)) THEN
          P_PHYS(0,I)=SQRT(PMASS(I)**2+P_PHYS(1,I)**2+P_PHYS(2,I)**2
     &     +P_PHYS(3,I)**2)
        ENDIF
        PF(0:3)=PF(0:3)+P_PHYS(0:3,I)
      ENDDO
      IF (NINCOMING.EQ.1) THEN
        P_PHYS(0:3,1)=PF(0:3)
      ELSE
        KMOM=SQRT(P_PHYS(1,1)**2+P_PHYS(2,1)**2+P_PHYS(3,1)**2)
        IF (KMOM.GT.0.0E0_16) THEN
          NDIR(1:3)=P_PHYS(1:3,1)/KMOM
C         p1 = (sqrt(k^2+m1^2), k n) with (PF-p1)^2 = m2^2
          E0=PF(0)
          C=PF(1)*NDIR(1)+PF(2)*NDIR(2)+PF(3)*NDIR(3)
          A=(E0**2-PF(1)**2-PF(2)**2-PF(3)**2+PMASS(1)**2-PMASS(2)**2)
     $     /2.0E0_16
          DEN=E0**2-C**2
          DISC=A**2-DEN*PMASS(1)**2
          IF (DEN.GT.0.0E0_16.AND.DISC.GE.0.0E0_16) THEN
            KMOM=(A*C+E0*SQRT(DISC))/DEN
            P_PHYS(0,1)=SQRT(KMOM**2+PMASS(1)**2)
            P_PHYS(1:3,1)=KMOM*NDIR(1:3)
            P_PHYS(0:3,2)=PF(0:3)-P_PHYS(0:3,1)
          ENDIF
        ENDIF
      ENDIF
      CALL ML5_0_MP_ONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
      END

      SUBROUTINE ML5_0_ONIA_LOOP_PROJ(P1,M1,NHEL1,P2,M2,NHEL2,P3,M3
     $ ,NHEL3,SPIN,PROJ)
C     The spin projector of a bound state (as ONIA_PROJ of the
C      tree-level
C     output): the weight of the constituent helicities NHEL1
C      (particle,
C     momentum P1) and NHEL2 (antiparticle, P2) in the state of spin
C      SPIN
C     and polarisation NHEL3 (momentum P3, mass M3).
      USE ALOHA_OBJECT
      IMPLICIT NONE
      REAL*8 P1(0:3), P2(0:3), P3(0:3), M1, M2, M3
      INTEGER NHEL1, NHEL2, NHEL3, SPIN
      COMPLEX*16 PROJ
      TYPE(ALOHA) FI, FO, VC
      COMPLEX*16 CI, TMP
      PARAMETER (CI=(0D0,1D0))
      CALL IXXXXX(P1,M1,NHEL1,+1,1,FI)
      CALL OXXXXX(P2,M2,NHEL2,-1,1,FO)
      IF (SPIN.EQ.0) THEN
        TMP = -FI%W(1)*FO%W(1)-FI%W(2)*FO%W(2)+FI%W(3)*FO%W(3)
     &        +FI%W(4)*FO%W(4)
      ELSEIF (SPIN.EQ.1) THEN
        CALL VXXXXX(P3,M3,NHEL3,+1,VC)
        TMP = (FI%W(1)*FO%W(4)-FI%W(3)*FO%W(2))*(VC%W(2)+CI*VC%W(3))
     &      + (FI%W(2)*FO%W(3)-FI%W(4)*FO%W(1))*(VC%W(2)-CI*VC%W(3))
     &      + (FI%W(1)*FO%W(3)+FI%W(4)*FO%W(2))*(VC%W(1)+VC%W(4))
     &      + (FI%W(2)*FO%W(4)+FI%W(3)*FO%W(1))*(VC%W(1)-VC%W(4))
      ELSE
        WRITE(*,*) 'ERROR:: spin projector not implemented for spin'
     $   ,SPIN
        STOP 1
      ENDIF
      PROJ = 0.5D0/SQRT(2D0*M1*M2)*TMP
      END

      SUBROUTINE ML5_0_MP_ONIA_LOOP_PROJ(P1,M1,NHEL1,P2,M2,NHEL2,P3,M3
     $ ,NHEL3,SPIN,PROJ)
C     The spin projector of a bound state, in quadruple precision
      USE ALOHA_OBJECT
      IMPLICIT NONE
      REAL*16 P1(0:3), P2(0:3), P3(0:3), M1, M2, M3
      INTEGER NHEL1, NHEL2, NHEL3, SPIN
      COMPLEX*32 PROJ
      TYPE(MP_ALOHA) FI, FO, VC
      COMPLEX*32 CI, TMP
      PARAMETER (CI=(0.0E0_16,1.0E0_16))
      CALL MP_IXXXXX(P1,M1,NHEL1,+1,1,FI)
      CALL MP_OXXXXX(P2,M2,NHEL2,-1,1,FO)
      IF (SPIN.EQ.0) THEN
        TMP = -FI%W(1)*FO%W(1)-FI%W(2)*FO%W(2)+FI%W(3)*FO%W(3)
     &        +FI%W(4)*FO%W(4)
      ELSEIF (SPIN.EQ.1) THEN
        CALL MP_VXXXXX(P3,M3,NHEL3,+1,VC)
        TMP = (FI%W(1)*FO%W(4)-FI%W(3)*FO%W(2))*(VC%W(2)+CI*VC%W(3))
     &      + (FI%W(2)*FO%W(3)-FI%W(4)*FO%W(1))*(VC%W(2)-CI*VC%W(3))
     &      + (FI%W(1)*FO%W(3)+FI%W(4)*FO%W(2))*(VC%W(1)+VC%W(4))
     &      + (FI%W(2)*FO%W(4)+FI%W(3)*FO%W(1))*(VC%W(1)-VC%W(4))
      ELSE
        WRITE(*,*) 'ERROR:: spin projector not implemented for spin'
     $   ,SPIN
        STOP 1
      ENDIF
      PROJ = 0.5E0_16/SQRT(2.0E0_16*M1*M2)*TMP
      END

      SUBROUTINE ML5_0_ONIA_BORN_AMPS(P,NHEL,H,IC)
C     The Born amplitudes (AMP) of the constituent helicity
C      configuration
C     NHEL, without the counterterms
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NBORNAMPS, NWAVEFUNCS
      PARAMETER (NEXTERNAL=4, NBORNAMPS=2)
      PARAMETER (NWAVEFUNCS=8)
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL), H
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      INCLUDE 'coupl.inc'
      COMPLEX*16 AMP(NBORNAMPS)
      COMMON/ML5_0_AMPS/AMP
      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1,W(1))
      CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1,W(2))
      CALL OXXXXX(P(0,3),MDL_MC,NHEL(3),+1, FLAVOR(3),W(3))
      CALL IXXXXX(P(0,4),MDL_MC,NHEL(4),-1, FLAVOR(4),W(4))
      CALL FFV1_1(W(3),W(1),GC_2,MDL_MC,ZERO,W(5))
      CALL FFV1_0(W(4),W(5),W(2),GC_2,AMP(1))
      CALL FFV1_2(W(4),W(1),GC_2,MDL_MC,ZERO,W(6))
      CALL FFV1_0(W(6),W(3),W(2),GC_2,AMP(2))
      END

      SUBROUTINE ML5_0_MP_ONIA_BORN_AMPS(P,NHEL,H,IC)
C     The Born amplitudes, in quadruple precision
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NBORNAMPS, NWAVEFUNCS
      PARAMETER (NEXTERNAL=4, NBORNAMPS=2)
      PARAMETER (NWAVEFUNCS=8)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E0_16)
      REAL*16 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL), H
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      INCLUDE 'mp_coupl_same_name.inc'
      COMPLEX*32 AMP(NBORNAMPS)
      COMMON/ML5_0_MP_AMPS/AMP
      TYPE(MP_ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/W
      CALL MP_VXXXXX(P(0,1),ZERO,NHEL(1),-1,W(1))
      CALL MP_VXXXXX(P(0,2),ZERO,NHEL(2),-1,W(2))
      CALL MP_OXXXXX(P(0,3),MDL_MC,NHEL(3),+1, FLAVOR(3),W(3))
      CALL MP_IXXXXX(P(0,4),MDL_MC,NHEL(4),-1, FLAVOR(4),W(4))
      CALL MP_FFV1_1(W(3),W(1),GC_2,MDL_MC,ZERO,W(5))
      CALL MP_FFV1_0(W(4),W(5),W(2),GC_2,AMP(1))
      CALL MP_FFV1_2(W(4),W(1),GC_2,MDL_MC,ZERO,W(6))
      CALL MP_FFV1_0(W(6),W(3),W(2),GC_2,AMP(2))
      END

      SUBROUTINE ML5_0_ONIA_PREPARE(P)
C     The spin projectors, the projected Born amplitudes, the effective
C     Born W(h) and the projected Born matrix element for the
C      constituent
C     momenta P.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=4, NCOMB=16, NBORNAMPS=2)
      REAL*8 P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLJZ
      INTEGER IOSEL, IOTH
      COMPLEX*16 AMP(NBORNAMPS)
      COMMON/ML5_0_AMPS/AMP
      COMPLEX*16 WEFF(NBORNAMPS,NCOMB)
      REAL*8 BORN2
      COMMON/ML5_0_ONIA_EFFECTIVE/WEFF,BORN2
      LOGICAL LIVE(NCOMB)
      COMMON/ML5_0_ONIA_LIVE/LIVE
      COMPLEX*16 PROJW(ONIA_NJZT,NCOMB), PR, PRK
      COMPLEX*16 AMPH(NBORNAMPS,NCOMB)
      COMPLEX*16 BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      COMPLEX*16 TMP
      COMPLEX*16 JFLOW(ONIA_NFLOW), ML5_0_ONIA_SANDWICH
      REAL*8 CMASS(NEXTERNAL), KQ(0:3), FACT
      COMMON/ML5_0_ONIA_PROJECTED/BPROJ,FACT
      LOGICAL ACTIVE(NCOMB)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      INTEGER H, T, K, I, J, O, Q1, Q2
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL ML5_0_ONIA_MASSES(CMASS)
C     The normalisation of the colour projectors and the LDMEs
      FACT=3.33333333333333315D-01
      DO K=1,ONIA_NONIA
        Q1=ONIA_Q1(K)
        Q2=ONIA_Q2(K)
        FACT=FACT*(CMASS(Q1)+CMASS(Q2))/(2D0*CMASS(Q1)*CMASS(Q2))
        IF (ONIA_C(K).EQ.1) THEN
          FACT=FACT/6D0/(2D0*ONIA_J(K)+1D0)
        ELSE
          FACT=FACT/8D0/(2D0*ONIA_J(K)+1D0)
        ENDIF
      ENDDO

      FACT=FACT*LDME_441
C     The spin projectors of the constituent helicity configurations
      DO H=1,NCOMB
        ACTIVE(H)=.FALSE.
        DO T=1,ONIA_NJZT
          PR=(1D0,0D0)
          DO K=1,ONIA_NONIA
            Q1=ONIA_Q1(K)
            Q2=ONIA_Q2(K)
            KQ(0:3)=P(0:3,Q1)+P(0:3,Q2)
            CALL ML5_0_ONIA_LOOP_PROJ(P(0,Q1),CMASS(Q1),ONIA_HELC(Q1,H)
     $       ,
     &       P(0,Q2),CMASS(Q2),ONIA_HELC(Q2,H),KQ,
     &       SQRT(MAX(KQ(0)**2-KQ(1)**2-KQ(2)**2-KQ(3)**2,0D0)),
     &       ONIA_JZ(K,T),ONIA_S(K),PRK)
            PR=PR*PRK
          ENDDO
          PROJW(T,H)=PR
          IF (PR.NE.(0D0,0D0)) ACTIVE(H)=.TRUE.
        ENDDO
      ENDDO
C     The Born amplitudes of the constituent helicity configurations
      DO H=1,NCOMB
        IF (ACTIVE(H)) THEN
          DO I=1,NEXTERNAL
            NHEL(I)=ONIA_HELC(I,H)
          ENDDO
          CALL ML5_0_ONIA_BORN_AMPS(P,NHEL,H,IC)
          AMPH(1:NBORNAMPS,H)=AMP(1:NBORNAMPS)
        ELSE
          AMPH(1:NBORNAMPS,H)=(0D0,0D0)
        ENDIF
      ENDDO
C     The projected Born amplitudes
      BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,1:ONIA_NJZT)=(0D0,0D0)
      DO H=1,NCOMB
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          BPROJ(1:NBORNAMPS,O,T)=BPROJ(1:NBORNAMPS,O,T)
     &     +PROJW(T,H)*AMPH(1:NBORNAMPS,H)
        ENDDO
      ENDDO
C     The J_z selected with ONIA_SELECT_JZ (all by default)
      CALL ML5_0_ONIA_JZ_TSEL(TSEL,ALLJZ)
      IF (.NOT.ALLJZ) THEN
        DO T=1,ONIA_NJZT
          IF (TSEL(T)) CYCLE
          BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,T)=(0D0,0D0)
        ENDDO
      ENDIF
C     The helicities of the other legs selected by SLOOPMATRIXHEL, as
C     their signature ONIA_OTHERS (all by default)
      CALL ML5_0_ONIA_OTHERS_SEL(.FALSE.,IOSEL)
      IF (IOSEL.NE.0) THEN
        DO IOTH=1,ONIA_NOTHERS
          IF (IOTH.EQ.IOSEL) CYCLE
          BPROJ(1:NBORNAMPS,IOTH,1:ONIA_NJZT)=(0D0,0D0)
        ENDDO
      ENDIF
C     The effective Born of each constituent helicity configuration
      DO H=1,NCOMB
        WEFF(1:NBORNAMPS,H)=(0D0,0D0)
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          WEFF(1:NBORNAMPS,H)=WEFF(1:NBORNAMPS,H)
     &     +DCONJG(PROJW(T,H))*BPROJ(1:NBORNAMPS,O,T)
        ENDDO
        WEFF(1:NBORNAMPS,H)=FACT*WEFF(1:NBORNAMPS,H)
      ENDDO
      DO H=1,NCOMB
        LIVE(H)=ANY(WEFF(1:NBORNAMPS,H).NE.(0D0,0D0))
      ENDDO
C     The projected Born matrix element, from the colour flows of the
C     projected Born amplitudes
      BORN2=0D0
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          CALL ML5_0_ONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW)
          BORN2=BORN2+DBLE(ML5_0_ONIA_SANDWICH(ONIA_NFLOW,JFLOW,
     &     ONIA_NFLOW,JFLOW,ONIA_NFLOW,ONIA_FCM_RE,ONIA_FCM_IM))
        ENDDO
      ENDDO
      BORN2=BORN2*FACT/DBLE(ONIA_IDEN)
      END

      SUBROUTINE ML5_0_MP_ONIA_PREPARE(P)
C     The effective Born W(h) for the constituent momenta P, in
C      quadruple
C     precision
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=4, NCOMB=16, NBORNAMPS=2)
      REAL*16 P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLJZ
      INTEGER IOSEL, IOTH
      COMPLEX*32 AMP(NBORNAMPS)
      COMMON/ML5_0_MP_AMPS/AMP
      COMPLEX*32 WEFF(NBORNAMPS,NCOMB)
      COMMON/ML5_0_MP_ONIA_EFFECTIVE/WEFF
      LOGICAL LIVE(NCOMB)
      COMMON/ML5_0_ONIA_LIVE/LIVE
      COMPLEX*32 PROJW(ONIA_NJZT,NCOMB), PR, PRK, CZERO
      PARAMETER (CZERO=(0.0E0_16,0.0E0_16))
      COMPLEX*32 AMPH(NBORNAMPS,NCOMB)
      COMPLEX*32 BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      REAL*16 CMASS(NEXTERNAL), KQ(0:3), FACT
      LOGICAL ACTIVE(NCOMB)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      INTEGER H, T, K, I, O, Q1, Q2
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL ML5_0_MP_ONIA_MASSES(CMASS)
      FACT=(1.0E0_16/3.0E0_16)
      DO K=1,ONIA_NONIA
        Q1=ONIA_Q1(K)
        Q2=ONIA_Q2(K)
        FACT=FACT*(CMASS(Q1)+CMASS(Q2))/(2.0E0_16*CMASS(Q1)*CMASS(Q2))
        IF (ONIA_C(K).EQ.1) THEN
          FACT=FACT/6.0E0_16/(2.0E0_16*ONIA_J(K)+1.0E0_16)
        ELSE
          FACT=FACT/8.0E0_16/(2.0E0_16*ONIA_J(K)+1.0E0_16)
        ENDIF
      ENDDO

      FACT=FACT*LDME_441
      DO H=1,NCOMB
        ACTIVE(H)=.FALSE.
        DO T=1,ONIA_NJZT
          PR=(1.0E0_16,0.0E0_16)
          DO K=1,ONIA_NONIA
            Q1=ONIA_Q1(K)
            Q2=ONIA_Q2(K)
            KQ(0:3)=P(0:3,Q1)+P(0:3,Q2)
            CALL ML5_0_MP_ONIA_LOOP_PROJ(P(0,Q1),CMASS(Q1)
     $       ,ONIA_HELC(Q1,H),
     &       P(0,Q2),CMASS(Q2),ONIA_HELC(Q2,H),KQ,
     &       SQRT(MAX(KQ(0)**2-KQ(1)**2-KQ(2)**2-KQ(3)**2,0.0E0_16)),
     &       ONIA_JZ(K,T),ONIA_S(K),PRK)
            PR=PR*PRK
          ENDDO
          PROJW(T,H)=PR
          IF (PR.NE.CZERO) ACTIVE(H)=.TRUE.
        ENDDO
      ENDDO
      DO H=1,NCOMB
        IF (ACTIVE(H)) THEN
          DO I=1,NEXTERNAL
            NHEL(I)=ONIA_HELC(I,H)
          ENDDO
          CALL ML5_0_MP_ONIA_BORN_AMPS(P,NHEL,H,IC)
          AMPH(1:NBORNAMPS,H)=AMP(1:NBORNAMPS)
        ELSE
          AMPH(1:NBORNAMPS,H)=CZERO
        ENDIF
      ENDDO
      BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,1:ONIA_NJZT)=CZERO
      DO H=1,NCOMB
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          BPROJ(1:NBORNAMPS,O,T)=BPROJ(1:NBORNAMPS,O,T)
     &     +PROJW(T,H)*AMPH(1:NBORNAMPS,H)
        ENDDO
      ENDDO
C     The J_z selected with ONIA_SELECT_JZ (all by default)
      CALL ML5_0_ONIA_JZ_TSEL(TSEL,ALLJZ)
      IF (.NOT.ALLJZ) THEN
        DO T=1,ONIA_NJZT
          IF (TSEL(T)) CYCLE
          BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,T)=CZERO
        ENDDO
      ENDIF
C     The helicities of the other legs selected by SLOOPMATRIXHEL, as
C     their signature ONIA_OTHERS (all by default)
      CALL ML5_0_ONIA_OTHERS_SEL(.FALSE.,IOSEL)
      IF (IOSEL.NE.0) THEN
        DO IOTH=1,ONIA_NOTHERS
          IF (IOTH.EQ.IOSEL) CYCLE
          BPROJ(1:NBORNAMPS,IOTH,1:ONIA_NJZT)=CZERO
        ENDDO
      ENDIF
      DO H=1,NCOMB
        WEFF(1:NBORNAMPS,H)=CZERO
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          WEFF(1:NBORNAMPS,H)=WEFF(1:NBORNAMPS,H)
     &     +CONJG(PROJW(T,H))*BPROJ(1:NBORNAMPS,O,T)
        ENDDO
        WEFF(1:NBORNAMPS,H)=FACT*WEFF(1:NBORNAMPS,H)
      ENDDO
      DO H=1,NCOMB
        LIVE(H)=ANY(WEFF(1:NBORNAMPS,H).NE.CZERO)
      ENDDO
      END

      LOGICAL FUNCTION ML5_0_ONIA_HEL_LIVE(H)
C     .TRUE. if the constituent helicity configuration H has a non-zero
C     effective Born at the last (MP_)ONIA_PREPARE point. Which
C      constituent
C     helicities contribute depends on the frame (the constituents of a
C     bound state at rest have no helicity axis of their own), so'
C     //' MadLoop's
C     helicity filter, learnt on its first points, is not used for
C      bound
C     states (HelicityFilterLevel=0) and the loops skip the others
C      point by
C     point.
      IMPLICIT NONE
      INTEGER NCOMB
      PARAMETER (NCOMB=16)
      INTEGER H
      LOGICAL LIVE(NCOMB)
      COMMON/ML5_0_ONIA_LIVE/LIVE
      ML5_0_ONIA_HEL_LIVE=LIVE(H)
      END

      SUBROUTINE ML5_0_ONIA_EFFECTIVE_BORN(H,AMP)
C     AMP = the effective Born W(H) of the last ONIA_PREPARE
      IMPLICIT NONE
      INTEGER NCOMB, NBORNAMPS
      PARAMETER (NCOMB=16, NBORNAMPS=2)
      INTEGER H
      COMPLEX*16 AMP(NBORNAMPS)
      COMPLEX*16 WEFF(NBORNAMPS,NCOMB)
      REAL*8 BORN2
      COMMON/ML5_0_ONIA_EFFECTIVE/WEFF,BORN2
      AMP(1:NBORNAMPS)=WEFF(1:NBORNAMPS,H)
      END

      SUBROUTINE ML5_0_MP_ONIA_EFFECTIVE_BORN(H,AMP)
C     AMP = the effective Born W(H) of the last MP_ONIA_PREPARE
      IMPLICIT NONE
      INTEGER NCOMB, NBORNAMPS
      PARAMETER (NCOMB=16, NBORNAMPS=2)
      INTEGER H
      COMPLEX*32 AMP(NBORNAMPS)
      COMPLEX*32 WEFF(NBORNAMPS,NCOMB)
      COMMON/ML5_0_MP_ONIA_EFFECTIVE/WEFF
      AMP(1:NBORNAMPS)=WEFF(1:NBORNAMPS,H)
      END


      SUBROUTINE ML5_0_ONIA_SELECT_JZ(K,NJZ,JZS)
C     Restricts the polarisation of bound state K (1 to the number of
C      bound
C     states, in the order of the process) to the NJZ values
C      JZS(1:NJZ) of
C     J_z; NJZ = 0 selects every J_z again (the default). J_z is the
C      helicity
C     of the bound state in the frame of the momenta given to
C      SLOOPMATRIX (the
C     polarisation vectors of VXXXXX for its momentum there): with
C      momenta in
C     the partonic centre-of-mass frame, J_z = +-1 and 0 of a J = 1
C      state are
C     its transverse and longitudinal polarisations in the helicity
C      frame.
C     The Born, the virtual and the predicted poles are those of the
C      selected
C     J_z, summed over them.
      IMPLICIT NONE
      INTEGER K, NJZ, JZS(*)
      INCLUDE 'onia_loop_data.inc'
      INTEGER MASK(ONIA_NONIA), I
      IF (K.LT.1.OR.K.GT.ONIA_NONIA) THEN
        WRITE(*,*) 'ERROR:: ONIA_SELECT_JZ: no bound state',K
        STOP 1
      ENDIF
      CALL ML5_0_ONIA_JZ_MASK(.FALSE.,MASK)
      MASK(K)=0
      DO I=1,NJZ
        IF (ABS(JZS(I)).GT.ONIA_J(K)) THEN
          WRITE(*,*) 'ERROR:: ONIA_SELECT_JZ: J_z',JZS(I),
     &     'out of range for J =',ONIA_J(K)
          STOP 1
        ENDIF
        MASK(K)=IBSET(MASK(K),JZS(I)+ONIA_J(K))
      ENDDO
      CALL ML5_0_ONIA_JZ_MASK(.TRUE.,MASK)
      END

      SUBROUTINE ML5_0_ONIA_JZ_MASK(SET,MASK)
C     The J_z selected for each bound state, bit J_z+J of MASK(K) (0:
C      all):
C     stored (SET) or returned
      IMPLICIT NONE
      LOGICAL SET
      INCLUDE 'onia_loop_data.inc'
      INTEGER MASK(ONIA_NONIA), SAVED(ONIA_NONIA)
      DATA SAVED/ONIA_NONIA*0/
      SAVE SAVED
      IF (SET) THEN
        SAVED(1:ONIA_NONIA)=MASK(1:ONIA_NONIA)
      ELSE
        MASK(1:ONIA_NONIA)=SAVED(1:ONIA_NONIA)
      ENDIF
      END

      SUBROUTINE ML5_0_ONIA_OTHERS_SEL(SET,OSEL)
C     The signature (ONIA_OTHERS) of the helicities of the legs other
C     than the constituents that SLOOPMATRIXHEL selects (0: all),
C      stored
C     (SET) or returned
      IMPLICIT NONE
      LOGICAL SET
      INTEGER OSEL, SAVED
      DATA SAVED/0/
      SAVE SAVED
      IF (SET) THEN
        SAVED=OSEL
      ELSE
        OSEL=SAVED
      ENDIF
      END

      SUBROUTINE ML5_0_ONIA_JZ_TSEL(TSEL,ALLSEL)
C     TSEL(T): whether the J_z combination T of the bound states is
C     selected (ONIA_SELECT_JZ); ALLSEL: every one is
      IMPLICIT NONE
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLSEL
      INTEGER MASK(ONIA_NONIA), T, K
      CALL ML5_0_ONIA_JZ_MASK(.FALSE.,MASK)
      ALLSEL=.TRUE.
      DO K=1,ONIA_NONIA
        IF (MASK(K).NE.0) ALLSEL=.FALSE.
      ENDDO
      DO T=1,ONIA_NJZT
        TSEL(T)=.TRUE.
        DO K=1,ONIA_NONIA
          IF (MASK(K).EQ.0) CYCLE
          IF (.NOT.BTEST(MASK(K),ONIA_JZ(K,T)+ONIA_J(K))) TSEL(T)
     $     =.FALSE.
        ENDDO
      ENDDO
      END

      SUBROUTINE ML5_0_ONIA_COLOUR_FLOWS(N0,N1,MA,MF,MRE,MIM,NF,V,J)
C     The colour flows J(1:NF) of the Born amplitudes V in a colour
C     basis whose amplitude-to-flow map is entries N0 to N1 of MA
C     (amplitude), MF (flow) and MRE + i MIM (coefficient). The colour
C     matrices of the onia layer are matrices between flows
C     (loop_onia.py, "Colour in the basis of colour flows"):
C     sum_ab conj(U_a) C_ab V_b = ONIA_SANDWICH of the flows of U
C     (Born basis) and V.
      IMPLICIT NONE
      INTEGER N0, N1, NF, MA(*), MF(*)
      REAL*8 MRE(*), MIM(*)
      COMPLEX*16 V(*), J(*)
      INTEGER N
      J(1:NF)=(0D0,0D0)
      DO N=N0,N1
        J(MF(N))=J(MF(N))+DCMPLX(MRE(N),MIM(N))*V(MA(N))
      ENDDO
      END

      SUBROUTINE ML5_0_ONIA_BORN_FLOWS(V,J)
C     The colour flows J(1:ONIA_NFLOW) of the Born amplitudes V in the
C     Born colour basis
      IMPLICIT NONE
      INCLUDE 'onia_loop_data.inc'
      COMPLEX*16 V(*), J(*)
      CALL ML5_0_ONIA_COLOUR_FLOWS(ONIA_BF_MPTR(1),ONIA_BF_MPTR(2)-1,
     & ONIA_BF_MA,ONIA_BF_MF,ONIA_BF_MC_RE,ONIA_BF_MC_IM,ONIA_NFLOW,V,J)
      END

      COMPLEX*16 FUNCTION ML5_0_ONIA_SANDWICH(NL,JL,NR,JR,LD,XRE,XIM)
C     sum_cd conj(JL(c)) X(c,d) JR(d) over c <= NL, d <= NR, for the
C     flow matrix X = XRE + i XIM of leading dimension LD
      IMPLICIT NONE
      INTEGER NL, NR, LD
      COMPLEX*16 JL(*), JR(*)
      REAL*8 XRE(LD,*), XIM(LD,*)
      COMPLEX*16 TMP, RES
      INTEGER C, D
      RES=(0D0,0D0)
      DO C=1,NL
        IF (JL(C).EQ.(0D0,0D0)) CYCLE
        TMP=(0D0,0D0)
        DO D=1,NR
          TMP=TMP+DCMPLX(XRE(C,D),XIM(C,D))*JR(D)
        ENDDO
        RES=RES+DCONJG(JL(C))*TMP
      ENDDO
      ML5_0_ONIA_SANDWICH=RES
      END

      SUBROUTINE ML5_0_ONIA_BORN(P,ANSB)
C     The projected Born matrix element (ANSB(0), and ANSB(1) for its
C     single squared split order) for the constituent momenta P. It
C      also
C     prepares the effective Born.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=4, NCOMB=16, NBORNAMPS=2)
      REAL*8 P(0:3,NEXTERNAL), ANSB(0:*)
      COMPLEX*16 WEFF(NBORNAMPS,NCOMB)
      REAL*8 BORN2
      COMMON/ML5_0_ONIA_EFFECTIVE/WEFF,BORN2
      CALL ML5_0_ONIA_PREPARE(P)
      ANSB(0)=BORN2
      ANSB(1)=BORN2
      END

      SUBROUTINE ML5_0_ONIA_IR_POLES(P_PHYS,POLES)
C     The IR poles of the renormalised virtual predicted by the
C      universal
C     structure of the one-loop amplitudes (Catani, Dittmaier,
C      Trocsanyi,
C     hep-ph/0011222; arXiv:2402.19221 eqs. (216)-(217) for S waves): a
C     colour-singlet bound state is a colourless particle, a colour
C      octet a
C     massive adjoint particle of mass m1+m2, with the colour charge
C      T_Q +
C     T_Qbar of its constituents. POLES(1) and POLES(2) are the
C     coefficients of 1/eps and 1/eps^2 in the normalisation of
C      ANS(2:3) of
C     SLOOPMATRIX (alpha_s/(2 pi) (4 pi)^eps/Gamma(1-eps) at the scale
C      MU_R).
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS, NBORNAMPS, NINCOMING
      PARAMETER (NEXTERNAL=4, NEXTERNAL_PHYS=3)
      PARAMETER (NBORNAMPS=2, NINCOMING=2)
      REAL*8 P_PHYS(0:3,NEXTERNAL_PHYS), POLES(2)
      INCLUDE 'onia_loop_data.inc'
      INCLUDE 'onia_loop_links.inc'
      INCLUDE 'coupl.inc'
      COMPLEX*16 BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      REAL*8 FACT
      COMMON/ML5_0_ONIA_PROJECTED/BPROJ,FACT
      COMPLEX*16 WEFF(NBORNAMPS,16)
      REAL*8 BORN2
      COMMON/ML5_0_ONIA_EFFECTIVE/WEFF,BORN2
      REAL*8 P(0:3,NEXTERNAL), CMASS(NEXTERNAL)
      REAL*8 PMASS(NEXTERNAL_PHYS), PJ(0:3), PK(0:3)
      REAL*8 LINKED, PJK, SJK, V, MU2, AO2PI, PI
      COMPLEX*16 JFLOW(ONIA_NFLOW,ONIA_NOTHERS,ONIA_NJZT)
      COMPLEX*16 ML5_0_ONIA_SANDWICH
      INTEGER L, I, J, O, T, JL, KL
      PI=4D0*DATAN(1D0)
      MU2=MU_R**2
      AO2PI=G**2/(8D0*PI**2)
      CALL ML5_0_ONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
      CALL ML5_0_ONIA_PREPARE(P)
      CALL ML5_0_ONIA_MASSES(CMASS)
      PMASS(1:NEXTERNAL_PHYS)=0D0
      DO I=1,NEXTERNAL
        PMASS(ONIA_MAPPING(I))=PMASS(ONIA_MAPPING(I))+CMASS(I)
      ENDDO
      POLES(1)=0D0
      POLES(2)=0D0
C     the colour flows of the projected Born amplitudes
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          CALL ML5_0_ONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW(1,O,T))
        ENDDO
      ENDDO
C     the colour-linked Borns <T_j.T_k>, j != k
      DO L=1,ONIA_NLINKS
        LINKED=0D0
        DO O=1,ONIA_NOTHERS
          DO T=1,ONIA_NJZT
            LINKED=LINKED+DBLE(ML5_0_ONIA_SANDWICH(ONIA_NFLOW,
     &       JFLOW(1,O,T),ONIA_NFLOW,JFLOW(1,O,T),ONIA_NFLOW,
     &       ONIA_FCL_RE(1,1,L),ONIA_FCL_IM(1,1,L)))
          ENDDO
        ENDDO
        LINKED=LINKED*FACT/DBLE(ONIA_IDEN)
        JL=ONIA_LINK_J(L)
        KL=ONIA_LINK_K(L)
C       all momenta outgoing
        PJ(0:3)=P_PHYS(0:3,JL)
        PK(0:3)=P_PHYS(0:3,KL)
        IF (JL.LE.NINCOMING) PJ(0:3)=-PJ(0:3)
        IF (KL.LE.NINCOMING) PK(0:3)=-PK(0:3)
        PJK=PJ(0)*PK(0)-PJ(1)*PK(1)-PJ(2)*PK(2)-PJ(3)*PK(3)
        SJK=ABS(2D0*PJK)
        IF (PMASS(JL).EQ.0D0.AND.PMASS(KL).EQ.0D0) THEN
          POLES(2)=POLES(2)+LINKED
          POLES(1)=POLES(1)+LINKED*LOG(MU2/SJK)
        ELSEIF (PMASS(JL).EQ.0D0.OR.PMASS(KL).EQ.0D0) THEN
          POLES(2)=POLES(2)+LINKED/2D0
          POLES(1)=POLES(1)+LINKED/2D0*(LOG(MU2/SJK)
     &     +LOG(MAX(PMASS(JL),PMASS(KL))**2/SJK))
        ELSE
          V=SQRT(1D0-(PMASS(JL)*PMASS(KL)/PJK)**2)
          POLES(1)=POLES(1)+LINKED/(2D0*V)*LOG((1D0-V)/(1D0+V))
        ENDIF
      ENDDO
C     the Gamma_j: gamma_g, gamma_q (massless), C_F (massive quark),
C      C_A
C     (colour-octet bound state)
      DO J=1,NEXTERNAL_PHYS
        POLES(1)=POLES(1)-ONIA_LEG_GAMMA(J)*BORN2
      ENDDO
      POLES(1)=AO2PI*POLES(1)
      POLES(2)=AO2PI*POLES(2)
      END

