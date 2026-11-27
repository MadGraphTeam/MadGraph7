C ======================================================================
C  Bound states (quarkonia, leptonia) in MadLoop
C ======================================================================
C
C  MadLoop computes the loop amplitudes of the constituent process, the
C  constituents of a bound state of momentum K and mass M = m1+m2 at zero
C  relative momentum, p_i = (m_i/M) K. The amplitudes are projected onto
C  the bound states as at tree level (arXiv:2510.26773): the colour
C  projector is in the colour factors, the spin projector is the sum over
C  the constituent helicities h weighted by PROJ(h,J_z) (ONIA_PROJ).
C  The projected interference with the Born is MadLoop's ordinary one
C  with the Born amplitudes A(h) replaced by the effective Born
C     W(h) = F sum_{J_z} conj(PROJ(h,J_z)) B(J_z),
C     B(J_z) = sum_h' PROJ(h',J_z) A(h'),
C  where h' runs over the configurations with the helicities of h for
C  the other legs and F is the colour and LDME normalisation of the
C  tree-level output. The users of MadLoop see the physical process:
C  SLOOPMATRIX takes the momenta of the bound states, not of their
C  constituents.
C
%(process_lines)s

      SUBROUTINE %(proc_prefix)sSLOOPMATRIX(P_PHYS,ANS)
C     The loop matrix element for the physical momenta P_PHYS (a bound
C     state with mass m1+m2), computed from its constituent legs.
C     After SLOOPMATRIXHEL(P_PHYS,HEL,ANS) (USERHEL=HEL) it is that of the
C     physical helicity configuration HEL (1 to ONIA_PHYS_NCOMB): the
C     helicity of each elementary leg and the J_z of each bound state, in
C     the order of the helicity matrix of the physical process (as the
C     tree-level output and the FKS Born). The constituent helicities are
C     summed over; as in MadLoop, a configuration is not averaged over the
C     initial helicities, so the configurations add up to ONIA_HELAVG
C     times the unselected result.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      %(real_dp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), ANS(0:3,0:*)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INTEGER USERHEL
      COMMON/%(proc_prefix)sUSERCHOICE/USERHEL
      INCLUDE 'nsquaredSO.inc'
      INTEGER RET_CODE_H,RET_CODE_T,RET_CODE_U
      %(real_dp_format)s ACCURACY(0:NSQUAREDSO)
      COMMON/%(proc_prefix)sACC/ACCURACY,RET_CODE_H,RET_CODE_T,RET_CODE_U
      INTEGER ML_GET_RET_CODE_LIBRARY, SET_RET_CODE_U
      INTEGER LIB
      LOGICAL WARNED
      DATA WARNED/.FALSE./
      SAVE WARNED
      %(real_dp_format)s QREST(3)
      COMMON/%(proc_prefix)sONIA_RELMOM/QREST
      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
      COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     &  FPE_IN_QP_REDUCTION
      LOGICAL FAILED_WARNED
      DATA FAILED_WARNED/.FALSE./
      SAVE FAILED_WARNED
      INTEGER I
## if(OniaPwave){
      LOGICAL PW_REFUSED, PW_WARNED
      COMMON/%(proc_prefix)sPW_REFUSED/PW_REFUSED
      DATA PW_WARNED/.FALSE./
      SAVE PW_WARNED
## }
      INCLUDE 'onia_loop_data.inc'
C     A physical helicity configuration (SLOOPMATRIXHEL): the J_z of each
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
        CALL %(proc_prefix)sONIA_JZ_MASK(.FALSE.,JZMASK_USER)
        DO IL=1,ONIA_NPHYS
          IF (ONIA_PHYS_LEG(IL).LT.0) THEN
            JZSEL(1)=ONIA_PHYS_HEL(IL,HEL_PHYS)
            CALL %(proc_prefix)sONIA_SELECT_JZ(-ONIA_PHYS_LEG(IL),1,JZSEL)
          ENDIF
        ENDDO
        CALL %(proc_prefix)sONIA_OTHERS_SEL(.TRUE.,
     &   ONIA_PHYS_OTHERS(HEL_PHYS))
C       MadLoop sums over the selected constituent configurations
        USERHEL=-1
      ENDIF
      CALL %(proc_prefix)sONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
## if(OniaPwave){
      PW_REFUSED=.FALSE.
## }
      CALL %(proc_prefix)sSLOOPMATRIX_CONSTITUENTS(P,ANS)
C     At threshold only TIRust (MLReductionLib=9) reduces the integrals
C     reliably. The other libraries can return wrong values that pass
C     MadLoop's stability tests (the Gram determinants vanish and the
C     Coulomb-exchange propagators are linearly dependent), so a result
C     they give is reported unstable (H=4, U=0) whatever its accuracy.
C     With a relative momentum (diagnostic) the integrals are ordinary ones,
C     and without coloured constituents (leptonia) no QCD loop holds both
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
## if(OniaPwave){
C     An integral with a doubled line that TIRust does not provide (at
C     threshold: those with a Coulomb exchange) is missing from the result,
C     which is then reported unstable whatever the mode (CTModeRun).
      IF (PW_REFUSED) THEN
        IF (.NOT.PW_WARNED) THEN
          WRITE(*,*) 'WARNING:: P-wave bound state: TIRust does not',
     &     ' provide some integrals with a doubled line of this',
     &     ' process (at threshold: with a Coulomb exchange). Their',
     &     ' contribution is missing; the result is reported unstable.'
          PW_WARNED=.TRUE.
        ENDIF
        RET_CODE_H=4
        DO I=0,NSQUAREDSO
          ACCURACY(I)=1D0
        ENDDO
      ENDIF
## }
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
C       configurations add up to ONIA_HELAVG times the unselected result)
        CALL %(proc_prefix)sGET_ANSWER_DIMENSION(NDIM_HEL)
        DO I=0,NDIM_HEL
          DO IL=0,3
            ANS(IL,I)=ANS(IL,I)*ONIA_HELAVG
          ENDDO
        ENDDO
        CALL %(proc_prefix)sONIA_JZ_MASK(.TRUE.,JZMASK_USER)
        CALL %(proc_prefix)sONIA_OTHERS_SEL(.TRUE.,0)
C       as MadLoop's SLOOPMATRIXHEL, which leaves USERHEL set
        USERHEL=HEL_PHYS
      ENDIF
      END

      SUBROUTINE %(proc_prefix)sONIA_MASSES(CMASS)
C     The masses of the constituent legs
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0D0)
      %(real_dp_format)s CMASS(NEXTERNAL)
      INCLUDE 'coupl.inc'
%(onia_dp_masses)s
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_MASSES(CMASS)
C     The masses of the constituent legs, in quadruple precision
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_mp_format)s ZERO
      PARAMETER (ZERO=0.0E0_16)
      %(real_mp_format)s CMASS(NEXTERNAL)
      INCLUDE 'mp_coupl_same_name.inc'
%(onia_mp_masses)s
      END

      SUBROUTINE %(proc_prefix)sONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
C     The momenta of the constituent legs: r_i K for the constituents of
C     a bound state of momentum K (r_i = m_i/(m1+m2), exactly 1/2 for
C     equal masses, so both constituents carry the same momentum bit for
C     bit), the other momenta copied.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      %(real_dp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), P(0:3,NEXTERNAL)
      %(real_dp_format)s CMASS(NEXTERNAL), FRAC, QLAB(0:3)
      INTEGER I, PARTNER, K
      INCLUDE 'onia_loop_data.inc'
      %(real_dp_format)s QREST(3)
      COMMON/%(proc_prefix)sONIA_RELMOM/QREST
      CALL %(proc_prefix)sONIA_MASSES(CMASS)
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
          CALL %(proc_prefix)sONIA_BOOST_REST_VECTOR(
     &     P_PHYS(0,ONIA_MAPPING(ONIA_Q1(K))),QREST,QLAB)
          P(0:3,ONIA_Q1(K))=P(0:3,ONIA_Q1(K))+QLAB(0:3)
          P(0:3,ONIA_Q2(K))=P(0:3,ONIA_Q2(K))-QLAB(0:3)
        ENDDO
      ENDIF
      END

      SUBROUTINE %(proc_prefix)sONIA_SET_RELATIVE_MOMENTUM(Q)
C     Diagnostic: give the constituents of every bound state the relative
C     momentum +-Q (a 3-vector in the rest frame of the bound state, which
C     must then have the mass 2 sqrt(m^2+Q^2), equal constituent masses).
C     The default, Q = 0, is the zero relative momentum of the projection.
C     Used to check the threshold integrals by extrapolating v -> 0 with
C     the Coulomb singularity separated (tools/onia_loop_check.py).
      IMPLICIT NONE
      %(real_dp_format)s Q(3)
      %(real_dp_format)s QREST(3)
      COMMON/%(proc_prefix)sONIA_RELMOM/QREST
      QREST(1:3)=Q(1:3)
      END

      BLOCK DATA %(proc_prefix)sONIA_RELMOM_DEFAULT
      IMPLICIT NONE
      %(real_dp_format)s QREST(3)
      COMMON/%(proc_prefix)sONIA_RELMOM/QREST
      DATA QREST/3*0D0/
      END

      SUBROUTINE %(proc_prefix)sONIA_BOOST_REST_VECTOR(K,Q,QLAB)
C     The 4-vector (0,Q) of the rest frame of K boosted to the frame of K
      IMPLICIT NONE
      %(real_dp_format)s K(0:3), Q(3), QLAB(0:3), M, BQ, G
      M=SQRT(K(0)**2-K(1)**2-K(2)**2-K(3)**2)
      BQ=(K(1)*Q(1)+K(2)*Q(2)+K(3)*Q(3))/M
      G=BQ/(K(0)+M)
      QLAB(0)=BQ
      QLAB(1:3)=Q(1:3)+G*K(1:3)
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_BOOST_REST_VECTOR(K,Q,QLAB)
C     The 4-vector (0,Q) of the rest frame of K boosted to the frame of K
      IMPLICIT NONE
      %(real_mp_format)s K(0:3), Q(3), QLAB(0:3), M, BQ, G
      M=SQRT(K(0)**2-K(1)**2-K(2)**2-K(3)**2)
      BQ=(K(1)*Q(1)+K(2)*Q(2)+K(3)*Q(3))/M
      G=BQ/(K(0)+M)
      QLAB(0)=BQ
      QLAB(1:3)=Q(1:3)+G*K(1:3)
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
C     The momenta of the constituent legs, in quadruple precision
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      %(real_mp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), P(0:3,NEXTERNAL)
      %(real_mp_format)s CMASS(NEXTERNAL), FRAC, QLAB(0:3), QMP(3)
      INTEGER I, PARTNER, K
      INCLUDE 'onia_loop_data.inc'
      %(real_dp_format)s QREST(3)
      COMMON/%(proc_prefix)sONIA_RELMOM/QREST
      CALL %(proc_prefix)sMP_ONIA_MASSES(CMASS)
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
          CALL %(proc_prefix)sMP_ONIA_BOOST_REST_VECTOR(
     &     P_PHYS(0,ONIA_MAPPING(ONIA_Q1(K))),QMP,QLAB)
          P(0:3,ONIA_Q1(K))=P(0:3,ONIA_Q1(K))+QLAB(0:3)
          P(0:3,ONIA_Q2(K))=P(0:3,ONIA_Q2(K))-QLAB(0:3)
        ENDDO
      ENDIF
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PHYSICAL_MOMENTA(P,P_PHYS)
C     The physical momenta (a bound state carries the sum of the momenta
C     of its constituents), in quadruple precision
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      %(real_mp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), P(0:3,NEXTERNAL)
      INTEGER I
      INCLUDE 'onia_loop_data.inc'
      P_PHYS(0:3,1:NEXTERNAL_PHYS)=0.0E0_16
      DO I=1,NEXTERNAL
        P_PHYS(0:3,ONIA_MAPPING(I))=P_PHYS(0:3,ONIA_MAPPING(I))+P(0:3,I)
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sMP_IMPROVE_PS_POINT_PRECISION(KEEP_OFFSHELL_MASS, P)
C     The precision improvement of a phase-space point with bound states,
C     in place of MadLoop's generic one (which can fail for a bound state at
C     rest). TIRust recognises the degenerate Coulomb-exchange integrals at
C     the precision of its inputs, so the momenta must be exactly on shell
C     with the masses the loops use (the double-precision squared masses,
C     MP_ONIA_MASSES) and conserve momentum: the physical final-state
C     particles are put on their mass shell (keeping their 3-momenta; a
C     bound state of mass m1+m2), the initial momenta are rebuilt from the
C     final ones along their original directions, and the bound states are
C     split again into their constituents at zero relative momentum.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS, NINCOMING
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      PARAMETER (NINCOMING=%(nincoming)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      LOGICAL KEEP_OFFSHELL_MASS(NEXTERNAL)
      %(real_mp_format)s P_PHYS(0:3,NEXTERNAL_PHYS)
      LOGICAL KEEP_PHYS(NEXTERNAL_PHYS)
      %(real_mp_format)s CMASS(NEXTERNAL), PMASS(NEXTERNAL_PHYS)
      %(real_mp_format)s PF(0:3), NDIR(3), A, C, E0, DEN, DISC, KMOM
      INTEGER I
      INCLUDE 'onia_loop_data.inc'
      %(real_dp_format)s QREST(3)
      COMMON/%(proc_prefix)sONIA_RELMOM/QREST
      DO I=1,NEXTERNAL
        KEEP_PHYS(ONIA_MAPPING(I))=KEEP_OFFSHELL_MASS(I)
      ENDDO
      CALL %(proc_prefix)sMP_ONIA_PHYSICAL_MOMENTA(P,P_PHYS)
      CALL %(proc_prefix)sMP_ONIA_MASSES(CMASS)
      PMASS(1:NEXTERNAL_PHYS)=0.0E0_16
      DO I=1,NEXTERNAL
        IF (ONIA_PARTNER(I).NE.0.AND.ANY(QREST.NE.0D0)) THEN
C         the diagnostic relative momentum: 2 sqrt(m^2+q^2)
          PMASS(ONIA_MAPPING(I))=PMASS(ONIA_MAPPING(I))+SQRT(CMASS(I)**2
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
          A=(E0**2-PF(1)**2-PF(2)**2-PF(3)**2+PMASS(1)**2-PMASS(2)**2)/2.0E0_16
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
      CALL %(proc_prefix)sMP_ONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
      END

## if(not OniaPwave){
      SUBROUTINE %(proc_prefix)sONIA_PROJ(P1,M1,NHEL1,P2,M2,NHEL2,P3,M3,NHEL3,SPIN,PROJ)
C     The spin projector of a bound state (as ONIA_PROJ of the tree-level
C     output): the weight of the constituent helicities NHEL1 (particle,
C     momentum P1) and NHEL2 (antiparticle, P2) in the state of spin SPIN
C     and polarisation NHEL3 (momentum P3, mass M3).
      USE ALOHA_OBJECT
      IMPLICIT NONE
      %(real_dp_format)s P1(0:3), P2(0:3), P3(0:3), M1, M2, M3
      INTEGER NHEL1, NHEL2, NHEL3, SPIN
      %(complex_dp_format)s PROJ
      TYPE(ALOHA) FI, FO, VC
      %(complex_dp_format)s CI, TMP
      PARAMETER (CI=(0D0,1D0))
      CALL IXXXXX(P1,M1,NHEL1,+1,1,FI)
      CALL OXXXXX(P2,M2,NHEL2,-1,1,FO)
      IF (SPIN.EQ.0) THEN
        TMP = -FI%%W(1)*FO%%W(1)-FI%%W(2)*FO%%W(2)+FI%%W(3)*FO%%W(3)
     &        +FI%%W(4)*FO%%W(4)
      ELSEIF (SPIN.EQ.1) THEN
        CALL VXXXXX(P3,M3,NHEL3,+1,VC)
        TMP = (FI%%W(1)*FO%%W(4)-FI%%W(3)*FO%%W(2))*(VC%%W(2)+CI*VC%%W(3))
     &      + (FI%%W(2)*FO%%W(3)-FI%%W(4)*FO%%W(1))*(VC%%W(2)-CI*VC%%W(3))
     &      + (FI%%W(1)*FO%%W(3)+FI%%W(4)*FO%%W(2))*(VC%%W(1)+VC%%W(4))
     &      + (FI%%W(2)*FO%%W(4)+FI%%W(3)*FO%%W(1))*(VC%%W(1)-VC%%W(4))
      ELSE
        WRITE(*,*) 'ERROR:: spin projector not implemented for spin',SPIN
        STOP 1
      ENDIF
      PROJ = 0.5D0/SQRT(2D0*M1*M2)*TMP
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PROJ(P1,M1,NHEL1,P2,M2,NHEL2,P3,M3,NHEL3,SPIN,PROJ)
C     The spin projector of a bound state, in quadruple precision
      USE ALOHA_OBJECT
      IMPLICIT NONE
      %(real_mp_format)s P1(0:3), P2(0:3), P3(0:3), M1, M2, M3
      INTEGER NHEL1, NHEL2, NHEL3, SPIN
      %(complex_mp_format)s PROJ
      TYPE(MP_ALOHA) FI, FO, VC
      %(complex_mp_format)s CI, TMP
      PARAMETER (CI=(0.0E0_16,1.0E0_16))
      CALL MP_IXXXXX(P1,M1,NHEL1,+1,1,FI)
      CALL MP_OXXXXX(P2,M2,NHEL2,-1,1,FO)
      IF (SPIN.EQ.0) THEN
        TMP = -FI%%W(1)*FO%%W(1)-FI%%W(2)*FO%%W(2)+FI%%W(3)*FO%%W(3)
     &        +FI%%W(4)*FO%%W(4)
      ELSEIF (SPIN.EQ.1) THEN
        CALL MP_VXXXXX(P3,M3,NHEL3,+1,VC)
        TMP = (FI%%W(1)*FO%%W(4)-FI%%W(3)*FO%%W(2))*(VC%%W(2)+CI*VC%%W(3))
     &      + (FI%%W(2)*FO%%W(3)-FI%%W(4)*FO%%W(1))*(VC%%W(2)-CI*VC%%W(3))
     &      + (FI%%W(1)*FO%%W(3)+FI%%W(4)*FO%%W(2))*(VC%%W(1)+VC%%W(4))
     &      + (FI%%W(2)*FO%%W(4)+FI%%W(3)*FO%%W(1))*(VC%%W(1)-VC%%W(4))
      ELSE
        WRITE(*,*) 'ERROR:: spin projector not implemented for spin',SPIN
        STOP 1
      ENDIF
      PROJ = 0.5E0_16/SQRT(2.0E0_16*M1*M2)*TMP
      END

      SUBROUTINE %(proc_prefix)sONIA_BORN_AMPS(P,NHEL,H,IC)
C     The Born amplitudes (AMP) of the constituent helicity configuration
C     NHEL, without the counterterms
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NBORNAMPS, NWAVEFUNCS
      PARAMETER (NEXTERNAL=%(nexternal)d, NBORNAMPS=%(nbornamps)d)
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0D0)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL), H
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      INCLUDE 'coupl.inc'
      %(complex_dp_format)s AMP(NBORNAMPS)
      COMMON/%(proc_prefix)sAMPS/AMP
      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/%(proc_prefix)sW/W
%(onia_born_calls)s
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_BORN_AMPS(P,NHEL,H,IC)
C     The Born amplitudes, in quadruple precision
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NBORNAMPS, NWAVEFUNCS
      PARAMETER (NEXTERNAL=%(nexternal)d, NBORNAMPS=%(nbornamps)d)
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      %(real_mp_format)s ZERO
      PARAMETER (ZERO=0.0E0_16)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL), H
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      INCLUDE 'mp_coupl_same_name.inc'
      %(complex_mp_format)s AMP(NBORNAMPS)
      COMMON/%(proc_prefix)sMP_AMPS/AMP
      TYPE(MP_ALOHA) W(NWAVEFUNCS)
      COMMON/%(proc_prefix)sMP_W/W
%(onia_mp_born_calls)s
      END

      SUBROUTINE %(proc_prefix)sONIA_PREPARE(P)
C     The spin projectors, the projected Born amplitudes, the effective
C     Born W(h) and the projected Born matrix element for the constituent
C     momenta P.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLJZ
      INTEGER IOSEL, IOTH
      %(complex_dp_format)s AMP(NBORNAMPS)
      COMMON/%(proc_prefix)sAMPS/AMP
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(complex_dp_format)s PROJW(ONIA_NJZT,NCOMB), PR, PRK
      %(complex_dp_format)s AMPH(NBORNAMPS,NCOMB)
      %(complex_dp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s TMP
      %(complex_dp_format)s JFLOW(ONIA_NFLOW), %(proc_prefix)sONIA_SANDWICH
      %(real_dp_format)s CMASS(NEXTERNAL), KQ(0:3), FACT
      COMMON/%(proc_prefix)sONIA_PROJECTED/BPROJ,FACT
      LOGICAL ACTIVE(NCOMB)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      INTEGER H, T, K, I, J, O, Q1, Q2
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL %(proc_prefix)sONIA_MASSES(CMASS)
C     The normalisation of the colour projectors and the LDMEs
      FACT=%(onia_dp_norm)s
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
%(onia_ldme_perturbative)s
      FACT=FACT*%(onia_ldme_product)s
C     The spin projectors of the constituent helicity configurations
      DO H=1,NCOMB
        ACTIVE(H)=.FALSE.
        DO T=1,ONIA_NJZT
          PR=(1D0,0D0)
          DO K=1,ONIA_NONIA
            Q1=ONIA_Q1(K)
            Q2=ONIA_Q2(K)
            KQ(0:3)=P(0:3,Q1)+P(0:3,Q2)
            CALL %(proc_prefix)sONIA_PROJ(P(0,Q1),CMASS(Q1),ONIA_HELC(Q1,H),
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
          CALL %(proc_prefix)sONIA_BORN_AMPS(P,NHEL,H,IC)
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
      CALL %(proc_prefix)sONIA_JZ_TSEL(TSEL,ALLJZ)
      IF (.NOT.ALLJZ) THEN
        DO T=1,ONIA_NJZT
          IF (TSEL(T)) CYCLE
          BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,T)=(0D0,0D0)
        ENDDO
      ENDIF
C     The helicities of the other legs selected by SLOOPMATRIXHEL, as
C     their signature ONIA_OTHERS (all by default)
      CALL %(proc_prefix)sONIA_OTHERS_SEL(.FALSE.,IOSEL)
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
          CALL %(proc_prefix)sONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW)
          BORN2=BORN2+DBLE(%(proc_prefix)sONIA_SANDWICH(ONIA_NFLOW,JFLOW,
     &     ONIA_NFLOW,JFLOW,ONIA_NFLOW,ONIA_FCM_RE,ONIA_FCM_IM))
        ENDDO
      ENDDO
      BORN2=BORN2*FACT/DBLE(ONIA_IDEN)
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PREPARE(P)
C     The effective Born W(h) for the constituent momenta P, in quadruple
C     precision
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLJZ
      INTEGER IOSEL, IOTH
      %(complex_mp_format)s AMP(NBORNAMPS)
      COMMON/%(proc_prefix)sMP_AMPS/AMP
      %(complex_mp_format)s WEFF(NBORNAMPS,NCOMB)
      COMMON/%(proc_prefix)sMP_ONIA_EFFECTIVE/WEFF
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(complex_mp_format)s PROJW(ONIA_NJZT,NCOMB), PR, PRK, CZERO
      PARAMETER (CZERO=(0.0E0_16,0.0E0_16))
      %(complex_mp_format)s AMPH(NBORNAMPS,NCOMB)
      %(complex_mp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_mp_format)s CMASS(NEXTERNAL), KQ(0:3), FACT
      LOGICAL ACTIVE(NCOMB)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      INTEGER H, T, K, I, O, Q1, Q2
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL %(proc_prefix)sMP_ONIA_MASSES(CMASS)
      FACT=%(onia_mp_norm)s
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
%(onia_ldme_perturbative)s
      FACT=FACT*%(onia_ldme_product)s
      DO H=1,NCOMB
        ACTIVE(H)=.FALSE.
        DO T=1,ONIA_NJZT
          PR=(1.0E0_16,0.0E0_16)
          DO K=1,ONIA_NONIA
            Q1=ONIA_Q1(K)
            Q2=ONIA_Q2(K)
            KQ(0:3)=P(0:3,Q1)+P(0:3,Q2)
            CALL %(proc_prefix)sMP_ONIA_PROJ(P(0,Q1),CMASS(Q1),ONIA_HELC(Q1,H),
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
          CALL %(proc_prefix)sMP_ONIA_BORN_AMPS(P,NHEL,H,IC)
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
      CALL %(proc_prefix)sONIA_JZ_TSEL(TSEL,ALLJZ)
      IF (.NOT.ALLJZ) THEN
        DO T=1,ONIA_NJZT
          IF (TSEL(T)) CYCLE
          BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,T)=CZERO
        ENDDO
      ENDIF
C     The helicities of the other legs selected by SLOOPMATRIXHEL, as
C     their signature ONIA_OTHERS (all by default)
      CALL %(proc_prefix)sONIA_OTHERS_SEL(.FALSE.,IOSEL)
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

## }
      LOGICAL FUNCTION %(proc_prefix)sONIA_HEL_LIVE(H)
C     .TRUE. if the constituent helicity configuration H has a non-zero
C     effective Born at the last (MP_)ONIA_PREPARE point. Which constituent
C     helicities contribute depends on the frame (the constituents of a
C     bound state at rest have no helicity axis of their own), so MadLoop's
C     helicity filter, learnt on its first points, is not used for bound
C     states (HelicityFilterLevel=0) and the loops skip the others point by
C     point.
      IMPLICIT NONE
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER H
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(proc_prefix)sONIA_HEL_LIVE=LIVE(H)
      END

## if(not OniaPwave){
      SUBROUTINE %(proc_prefix)sONIA_EFFECTIVE_BORN(H,AMP)
C     AMP = the effective Born W(H) of the last ONIA_PREPARE
      IMPLICIT NONE
      INTEGER NCOMB, NBORNAMPS
      PARAMETER (NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      INTEGER H
      %(complex_dp_format)s AMP(NBORNAMPS)
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      AMP(1:NBORNAMPS)=WEFF(1:NBORNAMPS,H)
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_EFFECTIVE_BORN(H,AMP)
C     AMP = the effective Born W(H) of the last MP_ONIA_PREPARE
      IMPLICIT NONE
      INTEGER NCOMB, NBORNAMPS
      PARAMETER (NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      INTEGER H
      %(complex_mp_format)s AMP(NBORNAMPS)
      %(complex_mp_format)s WEFF(NBORNAMPS,NCOMB)
      COMMON/%(proc_prefix)sMP_ONIA_EFFECTIVE/WEFF
      AMP(1:NBORNAMPS)=WEFF(1:NBORNAMPS,H)
      END

## }
## if(OniaPwave){
C ======================================================================
C  A P-wave bound state (one, ONIA_IPW)
C ======================================================================
C
C  Its amplitude is the derivative of the constituent amplitude in the
C  relative momentum q (p_Q = K/2+q, p_Qbar = K/2-q) along the polarisation
C  vectors eps(L_z) of the orbital angular momentum, combined with the spin
C  projectors as at tree level (matrix_standalone_v4_onia_pwave.inc):
C     A(J_z) = sum_{L_z} CG(1 L_z; S S_z|J J_z)
C              sum_h [PROJ0(h;S_z) M1(h;L_z) + PROJ1(h;S_z,L_z) M0(h)],
C  M0 the constituent amplitude at q=0, M1 its derivative along eps(L_z)
C  (direction ILZ = L_z+2), PROJ0 and PROJ1 the spin projector and its
C  derivative. All of them are computed in dual numbers (value, derivative
C  along the current direction). The projected interference is
C     sum_h conj(W1(h)) M0(h) + sum_{h,L_z} conj(W2(h,L_z)) M1(h;L_z),
C     W1(h) = F sum_{J_z} B(J_z) sum_{L_z} CG conj(PROJ1),
C     W2(h,L_z) = F sum_{J_z} B(J_z) CG conj(PROJ0),
C  with B the projected Born. The derivative of a loop amplitude is that of
C  its numerator (the dual coefficients, DWL) plus that of its propagators,
C  -2 (S_i-l).dS_i/D_i^2 with dS_i = c_i eps(L_z) the derivative of the shift
C  S_i of line i: these are accumulated as the polynomials
C     PCOEFS(mu) = sum_h conj(V(h,mu)) N0(h),
C     conj(V(h,mu)) = sum_{L_z} conj(W2(h,L_z)) eps(mu;L_z),
C  and reduced with line i doubled (ONIA_PW_DOUBLED).

      SUBROUTINE %(proc_prefix)sONIA_PW_EPSILONS(P)
C     The directions eps(L_z) of the relative momentum of the P-wave bound
C     state of momentum K: the vectors of VXXXXX(K,M,L_z,+1), as at tree
C     level, for ILZ = L_z+2
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      %(complex_dp_format)s EPSLZ(0:3,3)
      COMMON/%(proc_prefix)sPW_EPS/EPSLZ
      %(real_dp_format)s CMASS(NEXTERNAL)
      TYPE(DUAL) KD(0:3)
      TYPE(ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2
      CALL %(proc_prefix)sONIA_MASSES(CMASS)
      Q1=ONIA_Q1(ONIA_IPW)
      Q2=ONIA_Q2(ONIA_IPW)
      DO I=0,3
        KD(I)=P(I,Q1)+P(I,Q2)
      ENDDO
      DO ILZ=1,3
        CALL VXXXXX(KD,CMASS(Q1)+CMASS(Q2),ILZ-2,+1,EPSW)
        DO I=0,3
          EPSLZ(I,ILZ)=EPSW%%W(I+1)%%COMP(0)
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sONIA_PW_SET_DUAL(P,ILZ,PD)
C     The dual constituent momenta for the direction ILZ of the last
C     ONIA_PW_EPSILONS: p_Q = K/2 + d eps, p_Qbar = K/2 - d eps
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INTEGER ILZ
      TYPE(DUAL) PD(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      %(complex_dp_format)s EPSLZ(0:3,3)
      COMMON/%(proc_prefix)sPW_EPS/EPSLZ
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_dp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sPW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, J
      DO J=1,NEXTERNAL
        DO I=0,3
          PD(I,J)=P(I,J)
        ENDDO
      ENDDO
      DO I=0,3
        PD(I,ONIA_Q1(ONIA_IPW))%%COMP(1)=EPSLZ(I,ILZ)
        PD(I,ONIA_Q2(ONIA_IPW))%%COMP(1)=-EPSLZ(I,ILZ)
        PW_EPS(I)=EPSLZ(I,ILZ)
      ENDDO
      PW_ILZ=ILZ
      END

      SUBROUTINE %(proc_prefix)sONIA_PW_PROJ(P1,M1,NHEL1,P2,M2,NHEL2,P3,M3,NHEL3,SPIN,PROJ)
C     The spin projector of a bound state, for dual constituent momenta (the
C     formula of ONIA_PROJ_DUAL of the tree-level output)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      TYPE(DUAL) P1(0:3), P2(0:3), P3(0:3), PROJ, TMP
      %(real_dp_format)s M1, M2, M3
      INTEGER NHEL1, NHEL2, NHEL3, SPIN
      TYPE(ALOHA) FI, FO, VC
      %(complex_dp_format)s CI
      PARAMETER (CI=(0D0,1D0))
      CALL IXXXXX(P1,M1,NHEL1,+1,1,FI)
      CALL OXXXXX(P2,M2,NHEL2,-1,1,FO)
      IF (SPIN.EQ.0) THEN
        TMP = -FI%%W(1)*FO%%W(1)-FI%%W(2)*FO%%W(2)+FI%%W(3)*FO%%W(3)
     &        +FI%%W(4)*FO%%W(4)
      ELSEIF (SPIN.EQ.1) THEN
        CALL VXXXXX(P3,M3,NHEL3,+1,VC)
        TMP = (FI%%W(1)*FO%%W(4)-FI%%W(3)*FO%%W(2))*(VC%%W(2)+CI*VC%%W(3))
     &      + (FI%%W(2)*FO%%W(3)-FI%%W(4)*FO%%W(1))*(VC%%W(2)-CI*VC%%W(3))
     &      + (FI%%W(1)*FO%%W(3)+FI%%W(4)*FO%%W(2))*(VC%%W(1)+VC%%W(4))
     &      + (FI%%W(2)*FO%%W(4)+FI%%W(3)*FO%%W(1))*(VC%%W(1)-VC%%W(4))
      ELSE
        WRITE(*,*) 'ERROR:: spin projector not implemented for spin',SPIN
        STOP 1
      ENDIF
      PROJ = 0.5D0/SQRT(2D0*M1*M2)*TMP
      END

      SUBROUTINE %(proc_prefix)sONIA_BORN_AMPS(P,NHEL,H,IC)
C     The (dual) Born amplitudes of the constituent helicity configuration
C     NHEL for the dual momenta P, without the counterterms
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: PWW, PWAMP
      IMPLICIT NONE
      INTEGER NEXTERNAL, NBORNAMPS, NWAVEFUNCS
      PARAMETER (NEXTERNAL=%(nexternal)d, NBORNAMPS=%(nbornamps)d)
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0D0)
      TYPE(DUAL) P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL), H
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      INCLUDE 'coupl.inc'
%(onia_born_calls)s
      END

      SUBROUTINE %(proc_prefix)sONIA_PWS_FLOWS(B,V,J)
C     The colour flows J(1:PWS_SF_NF(B)) of the Born amplitudes V
C     in colour basis B of the soft currents (onia_loop_pwsoft.inc)
      IMPLICIT NONE
      INTEGER B
      INCLUDE 'onia_loop_data.inc'
      INCLUDE 'onia_loop_pwsoft.inc'
      %(complex_dp_format)s V(*), J(*)
      CALL %(proc_prefix)sONIA_COLOUR_FLOWS(PWS_SF_MPTR(B),PWS_SF_MPTR(B+1)-1,
     & PWS_SF_MA,PWS_SF_MF,PWS_SF_MC_RE,PWS_SF_MC_IM,PWS_SF_NF(B),V,J)
      END

## if(not OniaPwave2){
      SUBROUTINE %(proc_prefix)sONIA_PREPARE(P)
C     The spin projectors and their derivatives, the projected Born
C     amplitudes, the weights W1 (WEFF), W2 and V of each constituent
C     helicity configuration and the projected Born matrix element, for the
C     constituent momenta P (at zero relative momentum).
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: PWAMP, %(proc_prefix)sPW_DUAL_INIT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLJZ
      INTEGER IOSEL, IOTH
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      %(complex_dp_format)s W2(NBORNAMPS,NCOMB,3), VW(NBORNAMPS,0:3,NCOMB)
      INTEGER FIRSTLZ(NCOMB)
      LOGICAL LZLIVE(NCOMB,3)
      COMMON/%(proc_prefix)sPW_WEIGHTS/W2,VW,FIRSTLZ,LZLIVE
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(complex_dp_format)s EPSLZ(0:3,3)
      COMMON/%(proc_prefix)sPW_EPS/EPSLZ
      %(complex_dp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s FACT
      COMMON/%(proc_prefix)sONIA_PROJECTED/BPROJ,FACT
      %(complex_dp_format)s VS(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      COMMON/%(proc_prefix)sPW_SOFT/VS
      %(complex_dp_format)s PR0(ONIA_NJZT,NCOMB,3), PR1(ONIA_NJZT,NCOMB,3)
      %(complex_dp_format)s A0(NBORNAMPS,NCOMB), A1(NBORNAMPS,NCOMB,3)
      %(complex_dp_format)s TMP
      %(complex_dp_format)s JFLOW(ONIA_NFLOW), %(proc_prefix)sONIA_SANDWICH
      TYPE(DUAL) PD(0:3,NEXTERNAL), KD(0:3), PR, PRK
      %(real_dp_format)s CMASS(NEXTERNAL), CG
      LOGICAL ACTIVE(NCOMB)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      INTEGER H, T, K, I, J, O, Q1, Q2, ILZ, LZ, SZ, MU
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL %(proc_prefix)sPW_DUAL_INIT()
      CALL %(proc_prefix)sONIA_MASSES(CMASS)
C     The normalisation of the colour projectors and the LDMEs
      FACT=%(onia_dp_norm)s
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
%(onia_ldme_perturbative)s
      FACT=FACT*%(onia_ldme_product)s
C     The directions of the relative momentum
      CALL %(proc_prefix)sONIA_PW_EPSILONS(P)
C     The spin projectors (with the Clebsch-Gordan coefficients) and their
C     derivatives along each direction
      DO H=1,NCOMB
        ACTIVE(H)=.FALSE.
        DO ILZ=1,3
          LZ=ILZ-2
          CALL %(proc_prefix)sONIA_PW_SET_DUAL(P,ILZ,PD)
          DO T=1,ONIA_NJZT
            PR=(1D0,0D0)
            DO K=1,ONIA_NONIA
              Q1=ONIA_Q1(K)
              Q2=ONIA_Q2(K)
              DO I=0,3
                KD(I)=PD(I,Q1)+PD(I,Q2)
              ENDDO
              IF (K.EQ.ONIA_IPW) THEN
                SZ=ONIA_JZ(K,T)-LZ
                IF (ABS(SZ).GT.ONIA_S(K)) THEN
                  PR=(0D0,0D0)
                  EXIT
                ENDIF
                CALL CLEBSCH_GORDAN(1,LZ,ONIA_S(K),SZ,ONIA_J(K),ONIA_JZ(K,T),CG)
              ELSE
                SZ=ONIA_JZ(K,T)
                CG=1D0
              ENDIF
              CALL %(proc_prefix)sONIA_PW_PROJ(PD(0,Q1),CMASS(Q1),ONIA_HELC(Q1,H),
     &         PD(0,Q2),CMASS(Q2),ONIA_HELC(Q2,H),KD,CMASS(Q1)+CMASS(Q2),
     &         SZ,ONIA_S(K),PRK)
              PR=PR*PRK*CG
            ENDDO
            PR0(T,H,ILZ)=PR%%COMP(0)
            PR1(T,H,ILZ)=PR%%COMP(1)
            IF (PR0(T,H,ILZ).NE.(0D0,0D0).OR.PR1(T,H,ILZ).NE.(0D0,0D0))
     &        ACTIVE(H)=.TRUE.
          ENDDO
        ENDDO
      ENDDO
C     The Born amplitudes and their derivatives
      DO H=1,NCOMB
        A0(1:NBORNAMPS,H)=(0D0,0D0)
        A1(1:NBORNAMPS,H,1:3)=(0D0,0D0)
        IF (.NOT.ACTIVE(H)) CYCLE
        DO I=1,NEXTERNAL
          NHEL(I)=ONIA_HELC(I,H)
        ENDDO
        DO ILZ=1,3
          CALL %(proc_prefix)sONIA_PW_SET_DUAL(P,ILZ,PD)
          CALL %(proc_prefix)sONIA_BORN_AMPS(PD,NHEL,H,IC)
          DO I=1,NBORNAMPS
            A0(I,H)=PWAMP(I)%%COMP(0)
            A1(I,H,ILZ)=PWAMP(I)%%COMP(1)
          ENDDO
        ENDDO
      ENDDO
C     The projected Born amplitudes
      BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,1:ONIA_NJZT)=(0D0,0D0)
      DO H=1,NCOMB
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO ILZ=1,3
            BPROJ(1:NBORNAMPS,O,T)=BPROJ(1:NBORNAMPS,O,T)
     &       +PR0(T,H,ILZ)*A1(1:NBORNAMPS,H,ILZ)+PR1(T,H,ILZ)*A0(1:NBORNAMPS,H)
          ENDDO
        ENDDO
      ENDDO
C     The S-wave amplitudes of the soft current (ONIA_PW_SOFT_POLE): the
C     values of the Born amplitudes with the spin projectors of
C     S_z = J_z-L_z, VS = sum_{L_z} CG eps(L_z) A(S_z)
      VS(1:NBORNAMPS,0:3,1:ONIA_NOTHERS,1:ONIA_NJZT)=(0D0,0D0)
      DO H=1,NCOMB
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO ILZ=1,3
            IF (PR0(T,H,ILZ).EQ.(0D0,0D0)) CYCLE
            DO MU=0,3
              VS(1:NBORNAMPS,MU,O,T)=VS(1:NBORNAMPS,MU,O,T)
     &         +PR0(T,H,ILZ)*EPSLZ(MU,ILZ)*A0(1:NBORNAMPS,H)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
C     The J_z selected with ONIA_SELECT_JZ (all by default)
      CALL %(proc_prefix)sONIA_JZ_TSEL(TSEL,ALLJZ)
      IF (.NOT.ALLJZ) THEN
        DO T=1,ONIA_NJZT
          IF (TSEL(T)) CYCLE
          BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,T)=(0D0,0D0)
          VS(1:NBORNAMPS,0:3,1:ONIA_NOTHERS,T)=(0D0,0D0)
        ENDDO
      ENDIF
C     The helicities of the other legs selected by SLOOPMATRIXHEL, as
C     their signature ONIA_OTHERS (all by default)
      CALL %(proc_prefix)sONIA_OTHERS_SEL(.FALSE.,IOSEL)
      IF (IOSEL.NE.0) THEN
        DO IOTH=1,ONIA_NOTHERS
          IF (IOTH.EQ.IOSEL) CYCLE
          BPROJ(1:NBORNAMPS,IOTH,1:ONIA_NJZT)=(0D0,0D0)
          VS(1:NBORNAMPS,0:3,IOTH,1:ONIA_NJZT)=(0D0,0D0)
        ENDDO
      ENDIF
C     The weights of each constituent helicity configuration
      DO H=1,NCOMB
        WEFF(1:NBORNAMPS,H)=(0D0,0D0)
        W2(1:NBORNAMPS,H,1:3)=(0D0,0D0)
        VW(1:NBORNAMPS,0:3,H)=(0D0,0D0)
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO ILZ=1,3
            WEFF(1:NBORNAMPS,H)=WEFF(1:NBORNAMPS,H)
     &       +DCONJG(PR1(T,H,ILZ))*BPROJ(1:NBORNAMPS,O,T)
            W2(1:NBORNAMPS,H,ILZ)=W2(1:NBORNAMPS,H,ILZ)
     &       +DCONJG(PR0(T,H,ILZ))*BPROJ(1:NBORNAMPS,O,T)
          ENDDO
        ENDDO
        WEFF(1:NBORNAMPS,H)=FACT*WEFF(1:NBORNAMPS,H)
        W2(1:NBORNAMPS,H,1:3)=FACT*W2(1:NBORNAMPS,H,1:3)
        DO ILZ=1,3
          DO MU=0,3
            VW(1:NBORNAMPS,MU,H)=VW(1:NBORNAMPS,MU,H)
     &       +W2(1:NBORNAMPS,H,ILZ)*DCONJG(EPSLZ(MU,ILZ))
          ENDDO
        ENDDO
      ENDDO
C     The directions to compute for each configuration: those with a weight
C     W2, the first of which also takes the weights W1 and V
      DO H=1,NCOMB
        FIRSTLZ(H)=0
        DO ILZ=1,3
          LZLIVE(H,ILZ)=ANY(W2(1:NBORNAMPS,H,ILZ).NE.(0D0,0D0))
          IF (LZLIVE(H,ILZ).AND.FIRSTLZ(H).EQ.0) FIRSTLZ(H)=ILZ
        ENDDO
        IF (FIRSTLZ(H).EQ.0.AND.ANY(WEFF(1:NBORNAMPS,H).NE.(0D0,0D0))) THEN
          FIRSTLZ(H)=1
          LZLIVE(H,1)=.TRUE.
        ENDIF
        LIVE(H)=FIRSTLZ(H).NE.0
      ENDDO
C     The projected Born matrix element, from the colour flows of the
C     projected Born amplitudes
      BORN2=0D0
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          CALL %(proc_prefix)sONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW)
          BORN2=BORN2+DBLE(%(proc_prefix)sONIA_SANDWICH(ONIA_NFLOW,JFLOW,
     &     ONIA_NFLOW,JFLOW,ONIA_NFLOW,ONIA_FCM_RE,ONIA_FCM_IM))
        ENDDO
      ENDDO
      BORN2=BORN2*FACT/DBLE(ONIA_IDEN)
      END

      SUBROUTINE %(proc_prefix)sONIA_PW_SOFT_POLE(P_PHYS,PMASS,CMASS,CONTR)
C     The single pole of the virtual of the P-wave bound state beyond its
C     colour-linked Borns, in units of alpha_s/(2 pi) (arXiv:2402.19221,
C     appendix B; getpoles_onia of the onia FKS layer), at the point of the
C     last ONIA_PREPARE:
C       sum_k k.VEC(k)/(P.k)                                  (k massless)
C     + sum_k k.VEC(k)/(P.k) m_k^2 P^2/(2 v_k^3 (P.k)^2)
C             [2 v_k/(1-v_k^2) - log((1+v_k)/(1-v_k))]       (k massive)
C     with v_k = sqrt(1 - m_k^2 P^2/(P.k)^2), k the coloured legs but the
C     P-wave state P. A soft gluon turns P into the S-wave states S' of
C     colour C' (P[1] -> S[8]; P[8] -> S[1], S[8]) through Q_eff(CC'):
C       VEC^mu(k) = -2 Re sum <P| T_k.Q_eff |S'^mu>,
C     with |S'^mu> the amplitudes VS of ONIA_PREPARE in the colour of S'
C     (onia_loop_pwsoft.inc, loop_onia.pwave_soft_current), normalised as
C     the projected Born, with sqrt(n_C'/n_C) for the colour projector.
C     These are minus the poles of the cross terms of the integrated soft
C     current of the FKS subtraction. The virtual has no self-like pole of
C     P for any constituent masses (the intra-pair terms of the IR operator
C     are O(q^2); eq. (B.2) of the two-P-wave manuscript): the self term of
C     the soft current is cancelled by the LDME counterterm. (The self and
C     LDME terms of arXiv:2402.19221 app. B, with 8/(m_Q m_Qbar), cancel
C     only for equal masses; e+ e- > B_c b c~ in docs/benchmarkfolder,
C     sect. 5.2.1.)
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      PARAMETER (NBORNAMPS=%(nbornamps)d)
      %(real_dp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), PMASS(NEXTERNAL_PHYS)
      %(real_dp_format)s CMASS(NEXTERNAL), CONTR
      INCLUDE 'onia_loop_data.inc'
      INCLUDE 'onia_loop_pwsoft.inc'
      %(complex_dp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s FACT
      COMMON/%(proc_prefix)sONIA_PROJECTED/BPROJ,FACT
      %(complex_dp_format)s VS(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      COMMON/%(proc_prefix)sPW_SOFT/VS
      %(real_dp_format)s NC
      PARAMETER (NC=3D0)
      %(real_dp_format)s K(0:3), PK(0:3), KTOT2, PKK, MK2, V, FAC, F
      %(real_dp_format)s MQ, MQB, R, MSUM, VK
      %(complex_dp_format)s Z, KV(NBORNAMPS)
      %(complex_dp_format)s JFLOW(ONIA_NFLOW,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s JS(PWS_NFMAX), %(proc_prefix)sONIA_SANDWICH
      INTEGER L, A, I, J, O, T, KL, IPHYS
      CONTR=0D0
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          CALL %(proc_prefix)sONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW(1,O,T))
        ENDDO
      ENDDO
      IPHYS=ONIA_MAPPING(ONIA_Q1(ONIA_IPW))
      K(0:3)=P_PHYS(0:3,IPHYS)
      KTOT2=K(0)**2-K(1)**2-K(2)**2-K(3)**2
      MQ=CMASS(ONIA_Q1(ONIA_IPW))
      MQB=CMASS(ONIA_Q2(ONIA_IPW))
      R=MQ/MQB
      MSUM=2D0+R+1D0/R
      FAC=FACT/DBLE(ONIA_IDEN)
C     the cross vectors k.VEC(k)
      DO L=1,PWS_NLINKS
        A=PWS_AUX(L)
        IF (PWS_KIND(L).EQ.1) THEN
          IF (PWS_LABEL(A).EQ.'88') THEN
            F=MSUM/2D0
          ELSE
            F=MSUM/SQRT(2D0*NC)
          ENDIF
        ELSE
          F=(1D0/R-R)/2D0
        ENDIF
        IF (F.EQ.0D0) CYCLE
        KL=PWS_LEG(L)
        PK(0:3)=P_PHYS(0:3,KL)
        PKK=K(0)*PK(0)-K(1)*PK(1)-K(2)*PK(2)-K(3)*PK(3)
        Z=(0D0,0D0)
        DO O=1,ONIA_NOTHERS
          DO T=1,ONIA_NJZT
            DO J=1,NBORNAMPS
              KV(J)=PK(0)*VS(J,0,O,T)-PK(1)*VS(J,1,O,T)
     &         -PK(2)*VS(J,2,O,T)-PK(3)*VS(J,3,O,T)
            ENDDO
            CALL %(proc_prefix)sONIA_PWS_FLOWS(A,KV,JS)
            Z=Z+%(proc_prefix)sONIA_SANDWICH(ONIA_NFLOW,JFLOW(1,O,T),
     &       PWS_SF_NF(A),JS,ONIA_NFLOW,PWS_FX_RE(1,1,L),PWS_FX_IM(1,1,L))
          ENDDO
        ENDDO
        VK=-2D0*FAC*F*SQRT(PWS_CNORM(A))*DBLE(Z)
        MK2=PMASS(KL)**2
        IF (MK2.EQ.0D0) THEN
          CONTR=CONTR+VK/PKK
        ELSE
          V=SQRT(1D0-MK2*KTOT2/PKK**2)
          CONTR=CONTR+VK/PKK*MK2*KTOT2/(2D0*V**3*PKK**2)
     &     *(2D0*V/(1D0-V**2)-LOG((1D0+V)/(1D0-V)))
        ENDIF
      ENDDO
      END

      LOGICAL FUNCTION %(proc_prefix)sONIA_PW_DIRECTION(P,H,ILZ,PD)
C     .TRUE. if the direction ILZ is computed for the constituent helicity
C     configuration H at the last ONIA_PREPARE point; PD are then its dual
C     momenta
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INTEGER H, ILZ
      TYPE(DUAL) PD(0:3,NEXTERNAL)
      %(complex_dp_format)s W2(NBORNAMPS,NCOMB,3), VW(NBORNAMPS,0:3,NCOMB)
      INTEGER FIRSTLZ(NCOMB)
      LOGICAL LZLIVE(NCOMB,3)
      COMMON/%(proc_prefix)sPW_WEIGHTS/W2,VW,FIRSTLZ,LZLIVE
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_dp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sPW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      %(proc_prefix)sONIA_PW_DIRECTION=LZLIVE(H,ILZ)
      IF (.NOT.LZLIVE(H,ILZ)) RETURN
      CALL %(proc_prefix)sONIA_PW_SET_DUAL(P,ILZ,PD)
      PW_FIRST=ILZ.EQ.FIRSTLZ(H)
      END

      LOGICAL FUNCTION %(proc_prefix)sONIA_PW_CT_INPUT(H,PASS,AMP,AMPL)
C     The counterterm amplitudes AMPL of the current direction and their
C     effective Born AMP: the derivative with W2 (PASS=1), and on the first
C     direction of H the value with W1 (PASS=2). .FALSE. if this pass has
C     no weight.
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: PWAMPL
      IMPLICIT NONE
      INTEGER NCOMB, NBORNAMPS, NCTAMPS
      PARAMETER (NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d, NCTAMPS=%(nctamps)d)
      INTEGER H, PASS
      %(complex_dp_format)s AMP(NBORNAMPS), AMPL(3,NCTAMPS)
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      %(complex_dp_format)s W2(NBORNAMPS,NCOMB,3), VW(NBORNAMPS,0:3,NCOMB)
      INTEGER FIRSTLZ(NCOMB)
      LOGICAL LZLIVE(NCOMB,3)
      COMMON/%(proc_prefix)sPW_WEIGHTS/W2,VW,FIRSTLZ,LZLIVE
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_dp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sPW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, K, COMP
      IF (PASS.EQ.1) THEN
        AMP(1:NBORNAMPS)=W2(1:NBORNAMPS,H,PW_ILZ)
        COMP=1
      ELSE
        IF (.NOT.PW_FIRST) THEN
          %(proc_prefix)sONIA_PW_CT_INPUT=.FALSE.
          RETURN
        ENDIF
        AMP(1:NBORNAMPS)=WEFF(1:NBORNAMPS,H)
        COMP=0
      ENDIF
      %(proc_prefix)sONIA_PW_CT_INPUT=ANY(AMP(1:NBORNAMPS).NE.(0D0,0D0))
      DO I=1,NCTAMPS
        DO K=1,3
          AMPL(K,I)=PWAMPL(K,I)%%COMP(COMP)
        ENDDO
      ENDDO
      END

## }
      SUBROUTINE %(proc_prefix)sONIA_EFFECTIVE_BORN(H,AMP)
C     (S-wave interface, unused with a P-wave bound state)
      IMPLICIT NONE
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
      INTEGER H
      %(complex_dp_format)s AMP(NBORNAMPS)
      AMP(1:NBORNAMPS)=(0D0,0D0)
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_EFFECTIVE_BORN(H,AMP)
C     (S-wave interface, unused with a P-wave bound state)
      IMPLICIT NONE
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
      INTEGER H
      %(complex_mp_format)s AMP(NBORNAMPS)
      AMP(1:NBORNAMPS)=(0.0E0_16,0.0E0_16)
      END

      SUBROUTINE %(proc_prefix)sPW_SPLIT_COEFS(COEFS,C0,C1)
C     The value and the derivative of the dual coefficients of a loop vertex
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
      IMPLICIT NONE
      TYPE(DUAL) COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s C0(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s C1(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      INTEGER I, J, K
      DO K=1,MAXLWFSIZE
        DO J=0,VERTEXMAXCOEFS-1
          DO I=1,MAXLWFSIZE
            C0(I,J,K)=COEFS(I,J,K)%%COMP(0)
            C1(I,J,K)=COEFS(I,J,K)%%COMP(1)
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sPW_ADD_WL(A,RANK,LCUT_SIZE,OUT_SIZE,OUT)
C     OUT = OUT + A for the loop wavefunctions of rank RANK
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      %(complex_dp_format)s A(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s OUT(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER RANK, LCUT_SIZE, OUT_SIZE, I, J, K
      DO I=1,LCUT_SIZE
        DO K=0,NCOEF_R(RANK)-1
          DO J=1,OUT_SIZE
            OUT(J,K,I)=OUT(J,K,I)+A(J,K,I)
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sPW_MERGE_LOOP(LOOP_WF,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,
     & SYMFACT,MULTIPLIER,COLOR_ID,HELCONFIG,WEIGHT,TARGET)
C     CREATE_LOOP_COEFS with the effective Born WEIGHT and the polynomials
C     TARGET in place of AMP and LOOPCOEFS
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
      %(real_dp_format)s ZERO,ONE
      PARAMETER (ZERO=0.0D0,ONE=1.0D0)
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(ZERO,ONE))
      %(complex_dp_format)s CMPLX_ZERO
      PARAMETER (CMPLX_ZERO=(ZERO,ZERO))
      INTEGER    NCOLORROWS
      PARAMETER (NCOLORROWS=%(nloopamps)d)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
      %(complex_dp_format)s LOOP_WF(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER RANK, COLOR_ID, SYMFACT, MULTIPLIER, LCUT_SIZE, HELCONFIG, LOOP_GROUP_NUMBER
      %(complex_dp_format)s WEIGHT(NBORNAMPS)
      %(complex_dp_format)s TARGET(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      %(complex_dp_format)s CFTOT
      %(complex_dp_format)s CONST(NAMPSO)
      INTEGER I
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP, %(proc_prefix)sML5SQSOINDEX
      INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
      INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
      COMMON/%(proc_prefix)sCF/CF_D,CF_N
      LOGICAL CHECKPHASE
      LOGICAL HELDOUBLECHECKED
      COMMON/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED
      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/%(proc_prefix)sFILTERS/GOODAMP,GOODHEL,HELOFFSET
      INTEGER HELPICKED
      COMMON/%(proc_prefix)sHELCHOICE/HELPICKED
      INTEGER BORNAMP_SO(NBORNAMPS)
      INTEGER LOOPAMP_SQSO(NAMPSO,NCOLORROWS)
      COMMON/%(proc_prefix)sSOTABLES/BORNAMP_SO,LOOPAMP_SQSO
      DO I=1,NAMPSO
        CONST(I)=CMPLX_ZERO
      ENDDO
      DO I=1,NBORNAMPS
        IF (WEIGHT(I).EQ.CMPLX_ZERO) CYCLE
        CFTOT=CMPLX(CF_N(COLOR_ID,I)/(ONE*ABS(CF_D(COLOR_ID,I))),ZERO,KIND=8)
        IF(CF_D(COLOR_ID,I).LT.0) CFTOT=CFTOT*IMAG1
        CONST(BORNAMP_SO(I))=CONST(BORNAMP_SO(I))+CFTOT*CONJG(WEIGHT(I))
      ENDDO
      DO I=1,NAMPSO
        IF (CONST(I).NE.CMPLX_ZERO) THEN
          CONST(I)=(CONST(I)*MULTIPLIER)/SYMFACT
          IF (.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.HELPICKED.EQ.-1) THEN
            CONST(I)=CONST(I)*GOODHEL(HELCONFIG)
          ENDIF
          IF (LOOPAMP_SQSO(I,COLOR_ID).EQ.0) LOOPAMP_SQSO(I,COLOR_ID)=
     &     %(proc_prefix)sML5SQSOINDEX(I,%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(COLOR_ID))
          CALL %(proc_prefix)sMERGE_WL(LOOP_WF,RANK,LCUT_SIZE,CONST(I),
     &     TARGET(0,LOOPAMP_SQSO(I,COLOR_ID),LOOP_GROUP_NUMBER))
        ENDIF
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sPW_CREATE_LOOP_COEFS(WL,DWL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,
     & SYMFACT,MULTIPLIER,COLOR_ID,HELCONFIG)
C     The closed loop of the current direction: its derivative DWL with W2
C     into the loop coefficients, and on the first direction its value WL
C     with W1 and with V (the propagator derivatives, PCOEFS)
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NBORNAMPS, NCOMB, NLOOPGROUPS, NSQUAREDSO
      PARAMETER (NBORNAMPS=%(nbornamps)d, NCOMB=%(ncomb)d)
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d, NSQUAREDSO=%(nSquaredSO)d)
      %(complex_dp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s DWL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER RANK, LCUT_SIZE, LOOP_GROUP_NUMBER, SYMFACT, MULTIPLIER, COLOR_ID, HELCONFIG
      %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/%(proc_prefix)sLCOEFS/LOOPCOEFS
      %(complex_dp_format)s PCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      COMMON/%(proc_prefix)sPW_PCOEFS/PCOEFS
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      %(complex_dp_format)s W2(NBORNAMPS,NCOMB,3), VW(NBORNAMPS,0:3,NCOMB)
      INTEGER FIRSTLZ(NCOMB)
      LOGICAL LZLIVE(NCOMB,3)
      COMMON/%(proc_prefix)sPW_WEIGHTS/W2,VW,FIRSTLZ,LZLIVE
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_dp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sPW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER MU
      CALL %(proc_prefix)sPW_MERGE_LOOP(DWL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,W2(1,HELCONFIG,PW_ILZ),LOOPCOEFS)
      IF (.NOT.PW_FIRST) RETURN
      CALL %(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,WEFF(1,HELCONFIG),LOOPCOEFS)
      DO MU=0,3
        CALL %(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,VW(1,MU,HELCONFIG),PCOEFS(0,1,1,MU))
      ENDDO
      END

## if(not OniaPwave2){
      SUBROUTINE %(proc_prefix)sONIA_PW_DOUBLED(I_SQSO,I_LOOPGROUP,CTMODE,NLOOPLINE,RANK,PL,M2L,RES,STABLE)
C     RES = RES + the derivatives of the propagators of the loop group: for
C     each line J whose shift S_J depends on the relative momentum,
C     dS_J = C_J eps, the integral of rank RANK+1 with line J doubled of
C        -2 C_J sum_mu g_mu (S_J - l)^mu PCOEFS(l;mu),
C     PL and M2L the lines and masses of the loop (TIR_interface.f), the
C     derivatives of PL those of the last LOOP_n (PW_DPL), along the last
C     direction PW_EPS. The shifts are those of TIRust: S_1 the sum of PL,
C     S_J the sum of the first J-1.
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NEXTERNAL, NLOOPGROUPS, NSQUAREDSO
      PARAMETER (NEXTERNAL=%(nexternal)d)
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d, NSQUAREDSO=%(nSquaredSO)d)
      INTEGER I_SQSO, I_LOOPGROUP, CTMODE, NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s RES(3)
      LOGICAL STABLE
      %(complex_dp_format)s PCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      COMMON/%(proc_prefix)sPW_PCOEFS/PCOEFS
      %(complex_dp_format)s PW_DPL(0:3,NEXTERNAL)
      COMMON/%(proc_prefix)sPW_DPL/PW_DPL
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_dp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sPW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      %(real_dp_format)s SHIFT(0:3,NLOOPLINE), PLC(0:3,NLOOPLINE), PREFIX, G(0:3), C
      %(complex_dp_format)s DSHIFT(0:3,NLOOPLINE), DPREFIX, EE, EN, PC
      %(mass_dp_format)s M2LC(NLOOPLINE)
      %(complex_dp_format)s QC(0:LOOPMAXCOEFS-1), TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER POWERS(NLOOPLINE), I, J, K, MU, NC, NC1, KK
      LOGICAL OK
      LOGICAL PW_REFUSED
      COMMON/%(proc_prefix)sPW_REFUSED/PW_REFUSED
      DATA G/1D0,-1D0,-1D0,-1D0/
      IF (RANK+1.GT.%(maxrank)d) THEN
        WRITE(*,*) 'ERROR:: P-wave: rank',RANK+1,'above the maximal rank'
        STOP 1
      ENDIF
      NC=NCOEF_R(RANK)
      NC1=NCOEF_R(RANK+1)
      IF (ALL(PCOEFS(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.(0D0,0D0))) RETURN
      DO I=0,3
        SHIFT(I,1)=0D0
        DSHIFT(I,1)=(0D0,0D0)
        DO J=1,NLOOPLINE
          SHIFT(I,1)=SHIFT(I,1)+PL(I,J)
          DSHIFT(I,1)=DSHIFT(I,1)+PW_DPL(I,J)
        ENDDO
        PREFIX=0D0
        DPREFIX=(0D0,0D0)
        DO J=2,NLOOPLINE
          PREFIX=PREFIX+PL(I,J-1)
          DPREFIX=DPREFIX+PW_DPL(I,J-1)
          SHIFT(I,J)=PREFIX
          DSHIFT(I,J)=DPREFIX
        ENDDO
      ENDDO
      EN=(0D0,0D0)
      DO MU=0,3
        EN=EN+G(MU)*PW_EPS(MU)*DCONJG(PW_EPS(MU))
      ENDDO
      DO J=1,NLOOPLINE
C       C_J = dS_J.eps*/(eps.eps*)
        EE=(0D0,0D0)
        DO MU=0,3
          EE=EE+G(MU)*DSHIFT(MU,J)*DCONJG(PW_EPS(MU))
        ENDDO
        EE=EE/EN
        C=DBLE(EE)
        IF (ABS(C).LT.0.5D0.AND.ABS(EE).LT.1D-8) CYCLE
        IF (ABS(C-NINT(C)).GT.1D-8.OR.ABS(DIMAG(EE)).GT.1D-8.OR.J.EQ.1) THEN
          WRITE(*,*) 'ERROR:: P-wave: unexpected derivative of a loop line',J,EE
          STABLE=.FALSE.
          RETURN
        ENDIF
        C=DBLE(NINT(C))
        DO K=0,NC1-1
          QC(K)=(0D0,0D0)
        ENDDO
        DO MU=0,3
          DO K=0,NC-1
            PC=PCOEFS(K,I_SQSO,I_LOOPGROUP,MU)
            IF (PC.EQ.(0D0,0D0)) CYCLE
            QC(K)=QC(K)-2D0*C*G(MU)*SHIFT(MU,J)*PC
            KK=COMB_COEF_POS(K,MU+1)
            QC(KK)=QC(KK)+2D0*C*G(MU)*PC
          ENDDO
        ENDDO
        DO I=1,NLOOPLINE
          POWERS(I)=1
        ENDDO
        POWERS(J)=2
        PLC=PL
        M2LC=M2L
        CALL %(proc_prefix)sTIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK+1,POWERS,PLC,M2LC,
     &   TIRCOEFS,OK)
        IF (.NOT.OK) THEN
          STABLE=.FALSE.
          PW_REFUSED=.TRUE.
          RETURN
        ENDIF
        DO I=1,3
          DO K=0,NC1-1
            RES(I)=RES(I)+QC(K)*TIRCOEFS(K,I)
          ENDDO
        ENDDO
      ENDDO
      END
## }

## if(OniaPwave2){
C ======================================================================
C  Two P-wave bound states (ONIA_IPW and ONIA_IPW2)
C ======================================================================
C
C  The amplitude is the mixed derivative of the constituent amplitude in
C  the relative momenta q1, q2 of the two states, along the directions
C  eps1(L_z1) and eps2(L_z2) of the pair IDIR = ILZ1 + 3 (ILZ2-1), ILZ =
C  L_z+2 (9 pairs). The dual numbers have the components 0 (value), 1
C  (d/d eps1), 2 (d/d eps2) and 3 (d2/d eps1 d eps2), as at tree level
C  (matrix_standalone_v4_onia_pwave.inc). With PR_c the components of the
C  product of the spin projectors (with the Clebsch-Gordan coefficients of
C  both states) and L_c those of a loop amplitude, the projected loop is
C  sum_{h,IDIR} sum_c PR_(3-c) L_c, so that its interference is
C     sum_{h,IDIR} sum_c conj(W_c) L_c,   W_c = F sum_T conj(PR_(3-c)) B(T),
C  W0 (WEFF) independent of the direction, W1 of ILZ1, W2 of ILZ2, W3 of
C  the pair. A propagator of line i with shift S_i(q1,q2) contributes
C  through 1/D_i = 1/D - (D_1 e1 + D_2 e2 + D_12 e1 e2)/D^2
C  + 2 D_1 D_2 e1 e2/D^3, with D_a = 2 (S_i-l).dS_ia, dS_ia = c_ia eps_a
C  and D_12 = 2 dS_i1.dS_i2: the polynomials merged with the direction
C  sums of the weights (PW2_PCOEFS)
C     P1(mu) = N0 x VW1 + N2 x V3B,   P2(mu) = N0 x VW2 + N1 x V3A,
C     PX = N0 x XW,                   PT(mu,nu) = N0 x TW
C  (N_c the loop numerators) are reduced with raised propagator powers
C  (ONIA_PW_DOUBLED): line i doubled at rank R+1, line i tripled and two
C  lines doubled at rank R+2.

      SUBROUTINE %(proc_prefix)sONIA_PW2_EPSILONS(P)
C     The directions eps1(L_z1) and eps2(L_z2) of the relative momenta of
C     the two P-wave bound states (the vectors of VXXXXX, as at tree level)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      %(complex_dp_format)s EPS1(0:3,3), EPS2(0:3,3)
      COMMON/%(proc_prefix)sPW2_EPS/EPS1,EPS2
      %(real_dp_format)s CMASS(NEXTERNAL)
      TYPE(DUAL) KD(0:3)
      TYPE(ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2, S, K
      CALL %(proc_prefix)sONIA_MASSES(CMASS)
      DO S=1,2
        IF (S.EQ.1) THEN
          K=ONIA_IPW
        ELSE
          K=ONIA_IPW2
        ENDIF
        Q1=ONIA_Q1(K)
        Q2=ONIA_Q2(K)
        DO I=0,3
          KD(I)=P(I,Q1)+P(I,Q2)
        ENDDO
        DO ILZ=1,3
          CALL VXXXXX(KD,CMASS(Q1)+CMASS(Q2),ILZ-2,+1,EPSW)
          DO I=0,3
            IF (S.EQ.1) THEN
              EPS1(I,ILZ)=EPSW%%W(I+1)%%COMP(0)
            ELSE
              EPS2(I,ILZ)=EPSW%%W(I+1)%%COMP(0)
            ENDIF
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sONIA_PW2_SET_DUAL(P,IDIR,PD)
C     The dual constituent momenta of the direction pair IDIR: component 1
C     +-eps1(ILZ1) on the constituents of the first P-wave state, component
C     2 +-eps2(ILZ2) on those of the second
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INTEGER IDIR
      TYPE(DUAL) PD(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      %(complex_dp_format)s EPS1(0:3,3), EPS2(0:3,3)
      COMMON/%(proc_prefix)sPW2_EPS/EPS1,EPS2
      INTEGER PW_IDIR, PW_ILZ1, PW_ILZ2
      %(complex_dp_format)s PW_EPS1(0:3), PW_EPS2(0:3)
      COMMON/%(proc_prefix)sPW2_DIR/PW_EPS1,PW_EPS2,PW_IDIR,PW_ILZ1,PW_ILZ2
      INTEGER I, J
      PW_IDIR=IDIR
      PW_ILZ1=MOD(IDIR-1,3)+1
      PW_ILZ2=(IDIR-1)/3+1
      DO J=1,NEXTERNAL
        DO I=0,3
          PD(I,J)=P(I,J)
        ENDDO
      ENDDO
      DO I=0,3
        PD(I,ONIA_Q1(ONIA_IPW))%%COMP(1)=EPS1(I,PW_ILZ1)
        PD(I,ONIA_Q2(ONIA_IPW))%%COMP(1)=-EPS1(I,PW_ILZ1)
        PD(I,ONIA_Q1(ONIA_IPW2))%%COMP(2)=EPS2(I,PW_ILZ2)
        PD(I,ONIA_Q2(ONIA_IPW2))%%COMP(2)=-EPS2(I,PW_ILZ2)
        PW_EPS1(I)=EPS1(I,PW_ILZ1)
        PW_EPS2(I)=EPS2(I,PW_ILZ2)
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sONIA_PREPARE(P)
C     Two P-wave bound states: the dual components of the product of the
C     spin projectors for each direction pair, the projected Born
C     amplitudes, the weights W0 (WEFF), W1, W2, W3 and their direction sums
C     of each constituent helicity configuration, and the projected Born
C     matrix element, for the constituent momenta P (at zero relative
C     momenta)
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: PWAMP, %(proc_prefix)sPW_DUAL_INIT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLJZ
      INTEGER IOSEL, IOTH
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      %(complex_dp_format)s W1(NBORNAMPS,NCOMB,3), W2(NBORNAMPS,NCOMB,3)
      %(complex_dp_format)s W3(NBORNAMPS,NCOMB,9)
      %(complex_dp_format)s VW1(NBORNAMPS,0:3,NCOMB), VW2(NBORNAMPS,0:3,NCOMB)
      %(complex_dp_format)s V3A(NBORNAMPS,0:3,NCOMB,3), V3B(NBORNAMPS,0:3,NCOMB,3)
      %(complex_dp_format)s XW(NBORNAMPS,NCOMB), TW(NBORNAMPS,0:3,0:3,NCOMB)
      COMMON/%(proc_prefix)sPW2_WEIGHTS/W1,W2,W3,VW1,VW2,V3A,V3B,XW,TW
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(complex_dp_format)s EPS1(0:3,3), EPS2(0:3,3)
      COMMON/%(proc_prefix)sPW2_EPS/EPS1,EPS2
      %(complex_dp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s FACT
      COMMON/%(proc_prefix)sONIA_PROJECTED/BPROJ,FACT
      %(complex_dp_format)s PRC(ONIA_NJZT,NCOMB,9,0:3)
      %(complex_dp_format)s A(NBORNAMPS,NCOMB,9,0:3)
      SAVE PRC, A
      %(complex_dp_format)s VS1(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s VS2(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s GVSS(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s VVSS(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s PWK1(0:3), PWK2(0:3), PWM1, PWM2
      COMMON/%(proc_prefix)sPW2_SOFT/VS1,VS2,GVSS,VVSS,PWK1,PWK2,PWM1,PWM2
      %(complex_dp_format)s TMP, E12, E1V2, E2V1
      %(complex_dp_format)s JFLOW(ONIA_NFLOW), %(proc_prefix)sONIA_SANDWICH
      TYPE(DUAL) PD(0:3,NEXTERNAL), KD(0:3), PR, PRK
      %(real_dp_format)s CMASS(NEXTERNAL), CG, GMET(0:3)
      LOGICAL ACTIVE(NCOMB)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      DATA GMET/1D0,-1D0,-1D0,-1D0/
      INTEGER H, T, K, I, J, O, Q1, Q2, IDIR, ILZ1, ILZ2, LZ, SZ, MU, NU, C
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL %(proc_prefix)sPW_DUAL_INIT()
      CALL %(proc_prefix)sONIA_MASSES(CMASS)
C     The normalisation of the colour projectors and the LDMEs
      FACT=%(onia_dp_norm)s
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
%(onia_ldme_perturbative)s
      FACT=FACT*%(onia_ldme_product)s
C     The directions of the relative momenta
      CALL %(proc_prefix)sONIA_PW2_EPSILONS(P)
C     The products of the spin projectors (with the Clebsch-Gordan
C     coefficients of both P-wave states) and their dual components
      DO H=1,NCOMB
        ACTIVE(H)=.FALSE.
        DO IDIR=1,9
          ILZ1=MOD(IDIR-1,3)+1
          ILZ2=(IDIR-1)/3+1
          CALL %(proc_prefix)sONIA_PW2_SET_DUAL(P,IDIR,PD)
          DO T=1,ONIA_NJZT
            PR=(1D0,0D0)
            DO K=1,ONIA_NONIA
              Q1=ONIA_Q1(K)
              Q2=ONIA_Q2(K)
              DO I=0,3
                KD(I)=PD(I,Q1)+PD(I,Q2)
              ENDDO
              IF (K.EQ.ONIA_IPW.OR.K.EQ.ONIA_IPW2) THEN
                IF (K.EQ.ONIA_IPW) THEN
                  LZ=ILZ1-2
                ELSE
                  LZ=ILZ2-2
                ENDIF
                SZ=ONIA_JZ(K,T)-LZ
                IF (ABS(SZ).GT.ONIA_S(K)) THEN
                  PR=(0D0,0D0)
                  EXIT
                ENDIF
                CALL CLEBSCH_GORDAN(1,LZ,ONIA_S(K),SZ,ONIA_J(K),ONIA_JZ(K,T),CG)
              ELSE
                SZ=ONIA_JZ(K,T)
                CG=1D0
              ENDIF
              CALL %(proc_prefix)sONIA_PW_PROJ(PD(0,Q1),CMASS(Q1),ONIA_HELC(Q1,H),
     &         PD(0,Q2),CMASS(Q2),ONIA_HELC(Q2,H),KD,CMASS(Q1)+CMASS(Q2),
     &         SZ,ONIA_S(K),PRK)
              PR=PR*PRK*CG
            ENDDO
            DO C=0,3
              PRC(T,H,IDIR,C)=PR%%COMP(C)
              IF (PRC(T,H,IDIR,C).NE.(0D0,0D0)) ACTIVE(H)=.TRUE.
            ENDDO
          ENDDO
        ENDDO
      ENDDO
C     The Born amplitudes and their derivatives
      DO H=1,NCOMB
        A(1:NBORNAMPS,H,1:9,0:3)=(0D0,0D0)
        IF (.NOT.ACTIVE(H)) CYCLE
        DO I=1,NEXTERNAL
          NHEL(I)=ONIA_HELC(I,H)
        ENDDO
        DO IDIR=1,9
          CALL %(proc_prefix)sONIA_PW2_SET_DUAL(P,IDIR,PD)
          CALL %(proc_prefix)sONIA_BORN_AMPS(PD,NHEL,H,IC)
          DO C=0,3
            DO I=1,NBORNAMPS
              A(I,H,IDIR,C)=PWAMP(I)%%COMP(C)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
C     The projected Born amplitudes: the mixed component of PR x A
      BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,1:ONIA_NJZT)=(0D0,0D0)
      DO H=1,NCOMB
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO IDIR=1,9
            DO C=0,3
              BPROJ(1:NBORNAMPS,O,T)=BPROJ(1:NBORNAMPS,O,T)
     &         +PRC(T,H,IDIR,3-C)*A(1:NBORNAMPS,H,IDIR,C)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
C     The J_z selected with ONIA_SELECT_JZ (all by default)
      CALL %(proc_prefix)sONIA_JZ_TSEL(TSEL,ALLJZ)
      IF (.NOT.ALLJZ) THEN
        DO T=1,ONIA_NJZT
          IF (TSEL(T)) CYCLE
          BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,T)=(0D0,0D0)
        ENDDO
      ENDIF
C     The helicities of the other legs selected by SLOOPMATRIXHEL, as
C     their signature ONIA_OTHERS (all by default)
      CALL %(proc_prefix)sONIA_OTHERS_SEL(.FALSE.,IOSEL)
      IF (IOSEL.NE.0) THEN
        DO IOTH=1,ONIA_NOTHERS
          IF (IOTH.EQ.IOSEL) CYCLE
          BPROJ(1:NBORNAMPS,IOTH,1:ONIA_NJZT)=(0D0,0D0)
        ENDDO
      ENDIF
C     The amplitudes of the soft current (ONIA_PW_SOFT_POLE): pair 1 in
C     its S-wave states (value in q1, the directions eps1 and the spin
C     projectors of S_z1 = J_z1-L_z1) with pair 2 differentiated (VS1), the
C     same with 1 <-> 2 (VS2), and both pairs in their S-wave states
C     contracted with eps1.eps2 (GVSS) and (v2.eps1)(v1.eps2) (VVSS), v_i
C     = K_i/M_i (research/two_pwave/README.md sect. 4.1)
      DO K=1,2
        IF (K.EQ.1) THEN
          Q1=ONIA_Q1(ONIA_IPW)
          Q2=ONIA_Q2(ONIA_IPW)
          PWK1(0:3)=P(0:3,Q1)+P(0:3,Q2)
          PWM1=CMASS(Q1)+CMASS(Q2)
        ELSE
          Q1=ONIA_Q1(ONIA_IPW2)
          Q2=ONIA_Q2(ONIA_IPW2)
          PWK2(0:3)=P(0:3,Q1)+P(0:3,Q2)
          PWM2=CMASS(Q1)+CMASS(Q2)
        ENDIF
      ENDDO
      VS1(1:NBORNAMPS,0:3,1:ONIA_NOTHERS,1:ONIA_NJZT)=(0D0,0D0)
      VS2(1:NBORNAMPS,0:3,1:ONIA_NOTHERS,1:ONIA_NJZT)=(0D0,0D0)
      GVSS(1:NBORNAMPS,1:ONIA_NOTHERS,1:ONIA_NJZT)=(0D0,0D0)
      VVSS(1:NBORNAMPS,1:ONIA_NOTHERS,1:ONIA_NJZT)=(0D0,0D0)
      DO H=1,NCOMB
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          IF (.NOT.(ALLJZ.OR.TSEL(T))) CYCLE
          DO IDIR=1,9
            ILZ1=MOD(IDIR-1,3)+1
            ILZ2=(IDIR-1)/3+1
            DO MU=0,3
              VS1(1:NBORNAMPS,MU,O,T)=VS1(1:NBORNAMPS,MU,O,T)+EPS1(MU,ILZ1)
     &         *(PRC(T,H,IDIR,0)*A(1:NBORNAMPS,H,IDIR,2)
     &         +PRC(T,H,IDIR,2)*A(1:NBORNAMPS,H,IDIR,0))
              VS2(1:NBORNAMPS,MU,O,T)=VS2(1:NBORNAMPS,MU,O,T)+EPS2(MU,ILZ2)
     &         *(PRC(T,H,IDIR,0)*A(1:NBORNAMPS,H,IDIR,1)
     &         +PRC(T,H,IDIR,1)*A(1:NBORNAMPS,H,IDIR,0))
            ENDDO
            IF (PRC(T,H,IDIR,0).EQ.(0D0,0D0)) CYCLE
            E12=(0D0,0D0)
            E1V2=(0D0,0D0)
            E2V1=(0D0,0D0)
            DO MU=0,3
              E12=E12+GMET(MU)*EPS1(MU,ILZ1)*EPS2(MU,ILZ2)
              E1V2=E1V2+GMET(MU)*EPS1(MU,ILZ1)*PWK2(MU)/PWM2
              E2V1=E2V1+GMET(MU)*EPS2(MU,ILZ2)*PWK1(MU)/PWM1
            ENDDO
            GVSS(1:NBORNAMPS,O,T)=GVSS(1:NBORNAMPS,O,T)
     &       +E12*PRC(T,H,IDIR,0)*A(1:NBORNAMPS,H,IDIR,0)
            VVSS(1:NBORNAMPS,O,T)=VVSS(1:NBORNAMPS,O,T)
     &       +E1V2*E2V1*PRC(T,H,IDIR,0)*A(1:NBORNAMPS,H,IDIR,0)
          ENDDO
        ENDDO
      ENDDO
C     The weights of each constituent helicity configuration and their
C     direction sums
      DO H=1,NCOMB
        WEFF(1:NBORNAMPS,H)=(0D0,0D0)
        W1(1:NBORNAMPS,H,1:3)=(0D0,0D0)
        W2(1:NBORNAMPS,H,1:3)=(0D0,0D0)
        W3(1:NBORNAMPS,H,1:9)=(0D0,0D0)
        VW1(1:NBORNAMPS,0:3,H)=(0D0,0D0)
        VW2(1:NBORNAMPS,0:3,H)=(0D0,0D0)
        V3A(1:NBORNAMPS,0:3,H,1:3)=(0D0,0D0)
        V3B(1:NBORNAMPS,0:3,H,1:3)=(0D0,0D0)
        XW(1:NBORNAMPS,H)=(0D0,0D0)
        TW(1:NBORNAMPS,0:3,0:3,H)=(0D0,0D0)
        LIVE(H)=.FALSE.
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO IDIR=1,9
            ILZ1=MOD(IDIR-1,3)+1
            ILZ2=(IDIR-1)/3+1
            WEFF(1:NBORNAMPS,H)=WEFF(1:NBORNAMPS,H)
     &       +DCONJG(PRC(T,H,IDIR,3))*BPROJ(1:NBORNAMPS,O,T)
            W1(1:NBORNAMPS,H,ILZ1)=W1(1:NBORNAMPS,H,ILZ1)
     &       +DCONJG(PRC(T,H,IDIR,2))*BPROJ(1:NBORNAMPS,O,T)
            W2(1:NBORNAMPS,H,ILZ2)=W2(1:NBORNAMPS,H,ILZ2)
     &       +DCONJG(PRC(T,H,IDIR,1))*BPROJ(1:NBORNAMPS,O,T)
            W3(1:NBORNAMPS,H,IDIR)=W3(1:NBORNAMPS,H,IDIR)
     &       +DCONJG(PRC(T,H,IDIR,0))*BPROJ(1:NBORNAMPS,O,T)
          ENDDO
        ENDDO
        WEFF(1:NBORNAMPS,H)=FACT*WEFF(1:NBORNAMPS,H)
        W1(1:NBORNAMPS,H,1:3)=FACT*W1(1:NBORNAMPS,H,1:3)
        W2(1:NBORNAMPS,H,1:3)=FACT*W2(1:NBORNAMPS,H,1:3)
        W3(1:NBORNAMPS,H,1:9)=FACT*W3(1:NBORNAMPS,H,1:9)
C       (the weights of the polynomials of the propagator derivatives carry
C       the complex conjugate of the directions: PW_MERGE_LOOP conjugates
C       them back)
        DO IDIR=1,9
          ILZ1=MOD(IDIR-1,3)+1
          ILZ2=(IDIR-1)/3+1
          E12=(0D0,0D0)
          DO MU=0,3
            E12=E12+GMET(MU)*EPS1(MU,ILZ1)*EPS2(MU,ILZ2)
          ENDDO
          XW(1:NBORNAMPS,H)=XW(1:NBORNAMPS,H)+W3(1:NBORNAMPS,H,IDIR)*DCONJG(E12)
          DO MU=0,3
            V3A(1:NBORNAMPS,MU,H,ILZ1)=V3A(1:NBORNAMPS,MU,H,ILZ1)
     &       +W3(1:NBORNAMPS,H,IDIR)*DCONJG(EPS2(MU,ILZ2))
            V3B(1:NBORNAMPS,MU,H,ILZ2)=V3B(1:NBORNAMPS,MU,H,ILZ2)
     &       +W3(1:NBORNAMPS,H,IDIR)*DCONJG(EPS1(MU,ILZ1))
            DO NU=0,3
              TW(1:NBORNAMPS,MU,NU,H)=TW(1:NBORNAMPS,MU,NU,H)
     &         +W3(1:NBORNAMPS,H,IDIR)*DCONJG(EPS1(MU,ILZ1)*EPS2(NU,ILZ2))
            ENDDO
          ENDDO
        ENDDO
        DO ILZ1=1,3
          DO MU=0,3
            VW1(1:NBORNAMPS,MU,H)=VW1(1:NBORNAMPS,MU,H)
     &       +W1(1:NBORNAMPS,H,ILZ1)*DCONJG(EPS1(MU,ILZ1))
            VW2(1:NBORNAMPS,MU,H)=VW2(1:NBORNAMPS,MU,H)
     &       +W2(1:NBORNAMPS,H,ILZ1)*DCONJG(EPS2(MU,ILZ1))
          ENDDO
        ENDDO
        LIVE(H)=ANY(WEFF(1:NBORNAMPS,H).NE.(0D0,0D0))
     &   .OR.ANY(W1(1:NBORNAMPS,H,1:3).NE.(0D0,0D0))
     &   .OR.ANY(W2(1:NBORNAMPS,H,1:3).NE.(0D0,0D0))
     &   .OR.ANY(W3(1:NBORNAMPS,H,1:9).NE.(0D0,0D0))
      ENDDO
C     The projected Born matrix element, from the colour flows of the
C     projected Born amplitudes
      BORN2=0D0
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          CALL %(proc_prefix)sONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW)
          BORN2=BORN2+DBLE(%(proc_prefix)sONIA_SANDWICH(ONIA_NFLOW,JFLOW,
     &     ONIA_NFLOW,JFLOW,ONIA_NFLOW,ONIA_FCM_RE,ONIA_FCM_IM))
        ENDDO
      ENDDO
      BORN2=BORN2*FACT/DBLE(ONIA_IDEN)
      END

      SUBROUTINE %(proc_prefix)sONIA_PW_SOFT_POLE(P_PHYS,PMASS,CMASS,CONTR)
C     Two P-wave bound states: the single pole of the virtual beyond the
C     colour-linked Borns, in units of alpha_s/(2 pi), at the point of the
C     last ONIA_PREPARE (research/two_pwave/README.md): the P -> S
C     transitions of each state with the other one differentiated (term 2,
C     ONIA_PW2_SOFT_TERM) and the dipole-dipole exchange between the two
C     states (term 3, ONIA_PW2_DIPOLE_DIPOLE; He, Kniehl, Wang,
C     arXiv:1809.07993)
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      %(real_dp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), PMASS(NEXTERNAL_PHYS)
      %(real_dp_format)s CMASS(NEXTERNAL), CONTR
      %(real_dp_format)s C1, C2, PDD
      CALL %(proc_prefix)sONIA_PW2_SOFT_TERM(1,P_PHYS,PMASS,CMASS,C1)
      CALL %(proc_prefix)sONIA_PW2_SOFT_TERM(2,P_PHYS,PMASS,CMASS,C2)
      CALL %(proc_prefix)sONIA_PW2_DIPOLE_DIPOLE(CMASS,PDD)
      CONTR=C1+C2+PDD
      END

      SUBROUTINE %(proc_prefix)sONIA_PW2_SOFT_TERM(S,P_PHYS,PMASS,CMASS,CONTR)
C     The P -> S term of P-wave state S (1: ONIA_IPW, 2: ONIA_IPW2) with
C     the other state differentiated: ONIA_PW_SOFT_POLE of one P-wave state
C     with its amplitudes VS -> VS_S and its colour data
C     (onia_loop_pwsoft.inc, loop_onia.pwave_soft_current of state S; the
C     other state is a coloured leg if it is an octet)
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      PARAMETER (NBORNAMPS=%(nbornamps)d)
      INTEGER S
      %(real_dp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), PMASS(NEXTERNAL_PHYS)
      %(real_dp_format)s CMASS(NEXTERNAL), CONTR
      INCLUDE 'onia_loop_data.inc'
      INCLUDE 'onia_loop_pwsoft.inc'
      %(complex_dp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s FACT
      COMMON/%(proc_prefix)sONIA_PROJECTED/BPROJ,FACT
      %(complex_dp_format)s VS1(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s VS2(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s GVSS(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s VVSS(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s PWK1(0:3), PWK2(0:3), PWM1, PWM2
      COMMON/%(proc_prefix)sPW2_SOFT/VS1,VS2,GVSS,VVSS,PWK1,PWK2,PWM1,PWM2
      %(complex_dp_format)s VS(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s NC
      PARAMETER (NC=3D0)
      %(real_dp_format)s K(0:3), PK(0:3), KTOT2, PKK, MK2, V, FAC, F
      %(real_dp_format)s MQ, MQB, R, MSUM, VK
      %(complex_dp_format)s Z, KV(NBORNAMPS)
      %(complex_dp_format)s JFLOW(ONIA_NFLOW,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s JS(PWS_NFMAX), %(proc_prefix)sONIA_SANDWICH
      INTEGER L, A, I, J, O, T, KL, IPHYS, KS, B
      CONTR=0D0
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          CALL %(proc_prefix)sONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW(1,O,T))
        ENDDO
      ENDDO
      IF (S.EQ.1) THEN
        KS=ONIA_IPW
        VS=VS1
      ELSE
        KS=ONIA_IPW2
        VS=VS2
      ENDIF
      IPHYS=ONIA_MAPPING(ONIA_Q1(KS))
      K(0:3)=P_PHYS(0:3,IPHYS)
      KTOT2=K(0)**2-K(1)**2-K(2)**2-K(3)**2
      MQ=CMASS(ONIA_Q1(KS))
      MQB=CMASS(ONIA_Q2(KS))
      R=MQ/MQB
      MSUM=2D0+R+1D0/R
      FAC=FACT/DBLE(ONIA_IDEN)
C     the cross vectors k.VEC(k)
      DO L=1,PWS_NLINKS(S)
        A=PWS_AUX(L,S)
        IF (PWS_KIND(L,S).EQ.1) THEN
          IF (PWS_LABEL(A,S).EQ.'88') THEN
            F=MSUM/2D0
          ELSE
            F=MSUM/SQRT(2D0*NC)
          ENDIF
        ELSE
          F=(1D0/R-R)/2D0
        ENDIF
        IF (F.EQ.0D0) CYCLE
        KL=PWS_LEG(L,S)
        PK(0:3)=P_PHYS(0:3,KL)
        PKK=K(0)*PK(0)-K(1)*PK(1)-K(2)*PK(2)-K(3)*PK(3)
C       the colour basis of S-wave state A of state S
        B=A+(S-1)*PWS_NAUXMAX
        Z=(0D0,0D0)
        DO O=1,ONIA_NOTHERS
          DO T=1,ONIA_NJZT
            DO J=1,NBORNAMPS
              KV(J)=PK(0)*VS(J,0,O,T)-PK(1)*VS(J,1,O,T)
     &         -PK(2)*VS(J,2,O,T)-PK(3)*VS(J,3,O,T)
            ENDDO
            CALL %(proc_prefix)sONIA_PWS_FLOWS(B,KV,JS)
            Z=Z+%(proc_prefix)sONIA_SANDWICH(ONIA_NFLOW,JFLOW(1,O,T),
     &       PWS_SF_NF(B),JS,ONIA_NFLOW,PWS_FX_RE(1,1,L,S),
     &       PWS_FX_IM(1,1,L,S))
          ENDDO
        ENDDO
        VK=-2D0*FAC*F*SQRT(PWS_CNORM(A,S))*DBLE(Z)
        MK2=PMASS(KL)**2
        IF (MK2.EQ.0D0) THEN
          CONTR=CONTR+VK/PKK
        ELSE
          V=SQRT(1D0-MK2*KTOT2/PKK**2)
          CONTR=CONTR+VK/PKK*MK2*KTOT2/(2D0*V**3*PKK**2)
     &     *(2D0*V/(1D0-V**2)-LOG((1D0+V)/(1D0-V)))
        ENDIF
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sONIA_PW2_KERNEL(W,F1,F2)
C     The dipole-dipole kernel of He, Kniehl, Wang (arXiv:1809.07993, eq.
C     (8)), w = v1.v2 >= 1: f1 = -W'(w)/2 and f2 = -W''(w)/2 of the
C     massive soft function W(w) = w ln(w+s)/s, s = sqrt(w^2-1); below
C     t = w^2-1 = 0.1 their series in t (f1(1) = -1/3, f2(1) = 1/5)
      IMPLICIT NONE
      %(real_dp_format)s W, F1, F2
      %(real_dp_format)s T, S, AK
      INTEGER K
      T=W**2-1D0
      IF (T.LT.0.1D0) THEN
        AK=1D0
        F1=-1D0/3D0
        F2=0D0
        DO K=1,18
          AK=AK*DBLE(2*K-1)/DBLE(2*K)
          F1=F1-(-T)**K*AK/DBLE(2*K+3)
          F2=F2+2D0*W*DBLE(K)*(-1D0)**(K+1)*AK*T**(K-1)/DBLE(2*K+3)
        ENDDO
      ELSE
        S=SQRT(T)
        F1=(LOG(W+S)-W*S)/(2D0*S**3)
        F2=((W**2+2D0)*S-3D0*W*LOG(W+S))/(2D0*S**5)
      ENDIF
      END

      SUBROUTINE %(proc_prefix)sONIA_PW2_DIPOLE_DIPOLE(CMASS,PDD)
C     The dipole-dipole exchange between the two P-wave bound states (term
C     3 of research/two_pwave/README.md, sect. 4.3), at the point of the
C     last ONIA_PREPARE, in units of alpha_s/(2 pi):
C       PDD = 4 [f1 Re Z_G + f2 Re Z_V] + 2 pi [Im Z_G/s^3 - 3 w Im Z_V/s^5]
C       Z_G = FAC sum conj(BPROJ) XDD GVSS,  Z_V = the same with VVSS,
C       XDD = sum_(S1'k1,S2'k2) sqrt(CN1 CN2) F1(k1) F2(k2)/(M1 M2) PDD_X
C     (PDD_X: the flow matrix PDD_FX between the Born colour flows and
C     those of the term, colour basis 2*PWS_NAUXMAX+N)
C     with w = K1.K2/(M1 M2), s = sqrt(w^2-1) and F the factors of Q_eff of
C     ONIA_PW_SOFT_POLE (Delta: MSUM/sqrt(2 N_c), D: MSUM/2, T: (1/r-r)/2)
      IMPLICIT NONE
      INTEGER NEXTERNAL, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NBORNAMPS=%(nbornamps)d)
      %(real_dp_format)s CMASS(NEXTERNAL), PDD
      INCLUDE 'onia_loop_data.inc'
      INCLUDE 'onia_loop_pwsoft.inc'
      %(complex_dp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s FACT
      COMMON/%(proc_prefix)sONIA_PROJECTED/BPROJ,FACT
      %(complex_dp_format)s VS1(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s VS2(NBORNAMPS,0:3,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s GVSS(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s VVSS(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s PWK1(0:3), PWK2(0:3), PWM1, PWM2
      COMMON/%(proc_prefix)sPW2_SOFT/VS1,VS2,GVSS,VVSS,PWK1,PWK2,PWM1,PWM2
      %(real_dp_format)s NC, PI
      PARAMETER (NC=3D0)
      %(complex_dp_format)s ZG, ZV
      %(complex_dp_format)s JFLOW(ONIA_NFLOW,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s JG(PWS_NFMAX), JV(PWS_NFMAX)
      %(complex_dp_format)s %(proc_prefix)sONIA_SANDWICH
      %(real_dp_format)s W, F1, F2, S, FK(2), R, MSUM, FAC, CN
      INTEGER N, I, J, O, T, ST, KS, B
      CHARACTER*2 LAB
      INTEGER KK
      PDD=0D0
      IF (PDD_N.EQ.0) RETURN
      PI=4D0*DATAN(1D0)
      FAC=FACT/DBLE(ONIA_IDEN)
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          CALL %(proc_prefix)sONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW(1,O,T))
        ENDDO
      ENDDO
      ZG=(0D0,0D0)
      ZV=(0D0,0D0)
      DO N=1,PDD_N
        DO ST=1,2
          IF (ST.EQ.1) THEN
            KS=ONIA_IPW
            LAB=PDD_LABEL1(N)
            KK=PDD_KIND1(N)
          ELSE
            KS=ONIA_IPW2
            LAB=PDD_LABEL2(N)
            KK=PDD_KIND2(N)
          ENDIF
          R=CMASS(ONIA_Q1(KS))/CMASS(ONIA_Q2(KS))
          MSUM=2D0+R+1D0/R
          IF (KK.EQ.2) THEN
            FK(ST)=(1D0/R-R)/2D0
          ELSEIF (LAB.EQ.'88') THEN
            FK(ST)=MSUM/2D0
          ELSE
            FK(ST)=MSUM/SQRT(2D0*NC)
          ENDIF
        ENDDO
        IF (FK(1)*FK(2).EQ.0D0) CYCLE
        CN=SQRT(PDD_CNORM1(N)*PDD_CNORM2(N))*FK(1)*FK(2)/(PWM1*PWM2)
        B=2*PWS_NAUXMAX+N
        DO O=1,ONIA_NOTHERS
          DO T=1,ONIA_NJZT
            CALL %(proc_prefix)sONIA_PWS_FLOWS(B,GVSS(1,O,T),JG)
            CALL %(proc_prefix)sONIA_PWS_FLOWS(B,VVSS(1,O,T),JV)
            ZG=ZG+CN*%(proc_prefix)sONIA_SANDWICH(ONIA_NFLOW,JFLOW(1,O,T),
     &       PWS_SF_NF(B),JG,ONIA_NFLOW,PDD_FX_RE(1,1,N),PDD_FX_IM(1,1,N))
            ZV=ZV+CN*%(proc_prefix)sONIA_SANDWICH(ONIA_NFLOW,JFLOW(1,O,T),
     &       PWS_SF_NF(B),JV,ONIA_NFLOW,PDD_FX_RE(1,1,N),PDD_FX_IM(1,1,N))
          ENDDO
        ENDDO
      ENDDO
      ZG=FAC*ZG
      ZV=FAC*ZV
      W=(PWK1(0)*PWK2(0)-PWK1(1)*PWK2(1)-PWK1(2)*PWK2(2)-PWK1(3)*PWK2(3))
     & /(PWM1*PWM2)
      CALL %(proc_prefix)sONIA_PW2_KERNEL(W,F1,F2)
      PDD=4D0*(F1*DBLE(ZG)+F2*DBLE(ZV))
C     the i pi part of the soft function of two outgoing pairs (zero for
C     2 -> 2 processes, where Z_G and Z_V are real)
      S=SQRT(MAX(W**2-1D0,0D0))
      IF (S.GT.0D0) PDD=PDD+2D0*PI*(DIMAG(ZG)/S**3-3D0*W*DIMAG(ZV)/S**5)
      END

      SUBROUTINE %(proc_prefix)sONIA_NRQCD_CT(P_PHYS,MU_L,CT)
C     Two P-wave bound states: the MS-bar NRQCD counterterm of the
C     dipole-dipole exchange (the eight-quark operators of He, Kniehl,
C     Wang, arXiv:1809.07993; research/two_pwave/README.md sect. 5), for
C     the physical momenta P_PHYS and the NRQCD scale MU_L, in the units of
C     ANS(1:2) of SLOOPMATRIX at the scale MU_R (alpha_s/(2 pi)
C     (4 pi)^eps/Gamma(1-eps)): CT(1) the finite part, CT(2) the
C     coefficient of 1/eps,
C       CT = -PDD [1/eps + ln(mu_R^2/mu_L^2)].
C     The virtual V of SLOOPMATRIX is the plain QCD one: V + CT has no
C     dipole-dipole pole. It is not added to ANS.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      %(real_dp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), MU_L, CT(2)
      INCLUDE 'coupl.inc'
      %(real_dp_format)s P(0:3,NEXTERNAL), CMASS(NEXTERNAL), PDD, AO2PI, PI
      PI=4D0*DATAN(1D0)
      AO2PI=G**2/(8D0*PI**2)
      CALL %(proc_prefix)sONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
      CALL %(proc_prefix)sONIA_PREPARE(P)
      CALL %(proc_prefix)sONIA_MASSES(CMASS)
      CALL %(proc_prefix)sONIA_PW2_DIPOLE_DIPOLE(CMASS,PDD)
      CT(2)=-AO2PI*PDD
      CT(1)=-AO2PI*PDD*LOG(MU_R**2/MU_L**2)
      END

      LOGICAL FUNCTION %(proc_prefix)sONIA_PW_DIRECTION(P,H,IDIR,PD)
C     Two P-wave bound states: .TRUE. if the direction pair IDIR is computed
C     for the constituent helicity configuration H (all pairs of a live
C     configuration); PD are then its dual momenta
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d)
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INTEGER H, IDIR
      TYPE(DUAL) PD(0:3,NEXTERNAL)
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(proc_prefix)sONIA_PW_DIRECTION=LIVE(H)
      IF (.NOT.LIVE(H)) RETURN
      CALL %(proc_prefix)sONIA_PW2_SET_DUAL(P,IDIR,PD)
      END

      LOGICAL FUNCTION %(proc_prefix)sONIA_PW_CT_INPUT(H,PASS,AMP,AMPL)
C     Two P-wave bound states: the counterterm amplitudes AMPL of the current
C     direction pair and their effective Born AMP: the mixed component with
C     W3 (PASS=1, every pair), the d/d eps1 component with W1 (PASS=2, once
C     per ILZ1), the d/d eps2 component with W2 (PASS=3, once per ILZ2) and
C     the value with W0 (PASS=4, on the first pair). .FALSE. if this pass
C     has no weight.
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: PWAMPL
      IMPLICIT NONE
      INTEGER NCOMB, NBORNAMPS, NCTAMPS
      PARAMETER (NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d, NCTAMPS=%(nctamps)d)
      INTEGER H, PASS
      %(complex_dp_format)s AMP(NBORNAMPS), AMPL(3,NCTAMPS)
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      %(complex_dp_format)s W1(NBORNAMPS,NCOMB,3), W2(NBORNAMPS,NCOMB,3)
      %(complex_dp_format)s W3(NBORNAMPS,NCOMB,9)
      %(complex_dp_format)s VW1(NBORNAMPS,0:3,NCOMB), VW2(NBORNAMPS,0:3,NCOMB)
      %(complex_dp_format)s V3A(NBORNAMPS,0:3,NCOMB,3), V3B(NBORNAMPS,0:3,NCOMB,3)
      %(complex_dp_format)s XW(NBORNAMPS,NCOMB), TW(NBORNAMPS,0:3,0:3,NCOMB)
      COMMON/%(proc_prefix)sPW2_WEIGHTS/W1,W2,W3,VW1,VW2,V3A,V3B,XW,TW
      INTEGER PW_IDIR, PW_ILZ1, PW_ILZ2
      %(complex_dp_format)s PW_EPS1(0:3), PW_EPS2(0:3)
      COMMON/%(proc_prefix)sPW2_DIR/PW_EPS1,PW_EPS2,PW_IDIR,PW_ILZ1,PW_ILZ2
      INTEGER I, K, COMP
      %(proc_prefix)sONIA_PW_CT_INPUT=.FALSE.
      IF (PASS.EQ.1) THEN
        AMP(1:NBORNAMPS)=W3(1:NBORNAMPS,H,PW_IDIR)
        COMP=3
      ELSEIF (PASS.EQ.2) THEN
        IF (PW_ILZ2.NE.1) RETURN
        AMP(1:NBORNAMPS)=W1(1:NBORNAMPS,H,PW_ILZ1)
        COMP=1
      ELSEIF (PASS.EQ.3) THEN
        IF (PW_ILZ1.NE.1) RETURN
        AMP(1:NBORNAMPS)=W2(1:NBORNAMPS,H,PW_ILZ2)
        COMP=2
      ELSE
        IF (PW_IDIR.NE.1) RETURN
        AMP(1:NBORNAMPS)=WEFF(1:NBORNAMPS,H)
        COMP=0
      ENDIF
      %(proc_prefix)sONIA_PW_CT_INPUT=ANY(AMP(1:NBORNAMPS).NE.(0D0,0D0))
      DO I=1,NCTAMPS
        DO K=1,3
          AMPL(K,I)=PWAMPL(K,I)%%COMP(COMP)
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sPW_SPLIT_COEFS2(COEFS,C0,C1,C2,C3)
C     The four components of the dual coefficients of a loop vertex (two
C     P-wave bound states)
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
      IMPLICIT NONE
      TYPE(DUAL) COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s C0(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s C1(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s C2(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s C3(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      INTEGER I, J, K
      DO K=1,MAXLWFSIZE
        DO J=0,VERTEXMAXCOEFS-1
          DO I=1,MAXLWFSIZE
            C0(I,J,K)=COEFS(I,J,K)%%COMP(0)
            C1(I,J,K)=COEFS(I,J,K)%%COMP(1)
            C2(I,J,K)=COEFS(I,J,K)%%COMP(2)
            C3(I,J,K)=COEFS(I,J,K)%%COMP(3)
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sPW_CREATE_LOOP_COEFS2(WL,DWL,DWL2,D2WL,RANK,LCUT_SIZE,
     & LOOP_GROUP_NUMBER,SYMFACT,MULTIPLIER,COLOR_ID,HELCONFIG)
C     Two P-wave bound states: the closed loop of the current direction pair
C     merged with its weights: its mixed component D2WL with W3 (every
C     pair); its d/d eps1 component DWL with W1 into the loop coefficients
C     and with V3A into P2 (once per ILZ1); its d/d eps2 component DWL2 with
C     W2 and with V3B into P1 (once per ILZ2); its value WL with W0, with
C     VW1 into P1, VW2 into P2, XW into PX and TW into PT (first pair)
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NBORNAMPS, NCOMB, NLOOPGROUPS, NSQUAREDSO
      PARAMETER (NBORNAMPS=%(nbornamps)d, NCOMB=%(ncomb)d)
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d, NSQUAREDSO=%(nSquaredSO)d)
      %(complex_dp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s DWL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s DWL2(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s D2WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER RANK, LCUT_SIZE, LOOP_GROUP_NUMBER, SYMFACT, MULTIPLIER, COLOR_ID, HELCONFIG
      %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/%(proc_prefix)sLCOEFS/LOOPCOEFS
      %(complex_dp_format)s P1(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      %(complex_dp_format)s P2(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      %(complex_dp_format)s PX(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      %(complex_dp_format)s PT(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3,0:3)
      COMMON/%(proc_prefix)sPW2_PCOEFS/P1,P2,PX,PT
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      %(complex_dp_format)s W1(NBORNAMPS,NCOMB,3), W2(NBORNAMPS,NCOMB,3)
      %(complex_dp_format)s W3(NBORNAMPS,NCOMB,9)
      %(complex_dp_format)s VW1(NBORNAMPS,0:3,NCOMB), VW2(NBORNAMPS,0:3,NCOMB)
      %(complex_dp_format)s V3A(NBORNAMPS,0:3,NCOMB,3), V3B(NBORNAMPS,0:3,NCOMB,3)
      %(complex_dp_format)s XW(NBORNAMPS,NCOMB), TW(NBORNAMPS,0:3,0:3,NCOMB)
      COMMON/%(proc_prefix)sPW2_WEIGHTS/W1,W2,W3,VW1,VW2,V3A,V3B,XW,TW
      INTEGER PW_IDIR, PW_ILZ1, PW_ILZ2
      %(complex_dp_format)s PW_EPS1(0:3), PW_EPS2(0:3)
      COMMON/%(proc_prefix)sPW2_DIR/PW_EPS1,PW_EPS2,PW_IDIR,PW_ILZ1,PW_ILZ2
      INTEGER MU, NU
      CALL %(proc_prefix)sPW_MERGE_LOOP(D2WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,W3(1,HELCONFIG,PW_IDIR),LOOPCOEFS)
      IF (PW_ILZ2.EQ.1) THEN
        CALL %(proc_prefix)sPW_MERGE_LOOP(DWL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,W1(1,HELCONFIG,PW_ILZ1),LOOPCOEFS)
        DO MU=0,3
          CALL %(proc_prefix)sPW_MERGE_LOOP(DWL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &     MULTIPLIER,COLOR_ID,HELCONFIG,V3A(1,MU,HELCONFIG,PW_ILZ1),P2(0,1,1,MU))
        ENDDO
      ENDIF
      IF (PW_ILZ1.EQ.1) THEN
        CALL %(proc_prefix)sPW_MERGE_LOOP(DWL2,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,W2(1,HELCONFIG,PW_ILZ2),LOOPCOEFS)
        DO MU=0,3
          CALL %(proc_prefix)sPW_MERGE_LOOP(DWL2,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &     MULTIPLIER,COLOR_ID,HELCONFIG,V3B(1,MU,HELCONFIG,PW_ILZ2),P1(0,1,1,MU))
        ENDDO
      ENDIF
      IF (PW_IDIR.NE.1) RETURN
      CALL %(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,WEFF(1,HELCONFIG),LOOPCOEFS)
      CALL %(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,XW(1,HELCONFIG),PX)
      DO MU=0,3
        CALL %(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,VW1(1,MU,HELCONFIG),P1(0,1,1,MU))
        CALL %(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,VW2(1,MU,HELCONFIG),P2(0,1,1,MU))
        DO NU=0,3
          CALL %(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &     MULTIPLIER,COLOR_ID,HELCONFIG,TW(1,MU,NU,HELCONFIG),PT(0,1,1,MU,NU))
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sONIA_PW_DOUBLED(I_SQSO,I_LOOPGROUP,CTMODE,NLOOPLINE,RANK,PL,M2L,RES,STABLE)
C     Two P-wave bound states: RES = RES + the propagator-derivative terms
C     of the loop group, for the lines J with the derivatives dS_J1 =
C     C1_J eps1 and dS_J2 = C2_J eps2 of their shifts (TIRust's shifts: S_1
C     the sum of PL, S_J that of the first J-1):
C     - line J doubled (rank RANK+1): the numerator
C         -2 sum_mu g_mu (S_J-l)^mu [C1_J P1(l;mu) + C2_J P2(l;mu)]
C         -2 C1_J C2_J PX(l)
C     - line J tripled (rank RANK+2): 8 C1_J C2_J T_JJ,
C     - lines J and JJ doubled, J /= JJ (rank RANK+2): 4 C1_J C2_JJ T_JJJ,
C     T_JK = sum_mu,nu g_mu g_nu (S_J-l)^mu (S_K-l)^nu PT(l;mu,nu)
C     (onia_loop.f header). PL and M2L are the lines and masses of the loop
C     (TIR_interface.f), the derivatives of PL those of the last LOOP_n
C     (PW_DPL along PW_EPS1, PW_DPL2 along PW_EPS2).
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NEXTERNAL, NLOOPGROUPS, NSQUAREDSO
      PARAMETER (NEXTERNAL=%(nexternal)d)
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d, NSQUAREDSO=%(nSquaredSO)d)
      INTEGER I_SQSO, I_LOOPGROUP, CTMODE, NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(complex_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s RES(3)
      LOGICAL STABLE
      %(complex_dp_format)s P1(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      %(complex_dp_format)s P2(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      %(complex_dp_format)s PX(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      %(complex_dp_format)s PT(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3,0:3)
      COMMON/%(proc_prefix)sPW2_PCOEFS/P1,P2,PX,PT
      %(complex_dp_format)s PW_DPL(0:3,NEXTERNAL)
      COMMON/%(proc_prefix)sPW_DPL/PW_DPL
      %(complex_dp_format)s PW_DPL2(0:3,NEXTERNAL)
      COMMON/%(proc_prefix)sPW_DPL2/PW_DPL2
      INTEGER PW_IDIR, PW_ILZ1, PW_ILZ2
      %(complex_dp_format)s PW_EPS1(0:3), PW_EPS2(0:3)
      COMMON/%(proc_prefix)sPW2_DIR/PW_EPS1,PW_EPS2,PW_IDIR,PW_ILZ1,PW_ILZ2
      %(real_dp_format)s SHIFT(0:3,NLOOPLINE), PLC(0:3,NLOOPLINE), PREFIX, G(0:3)
      %(real_dp_format)s C1(NLOOPLINE), C2(NLOOPLINE), F
      %(complex_dp_format)s DS1(0:3,NLOOPLINE), DS2(0:3,NLOOPLINE), DP1, DP2
      %(complex_dp_format)s EE, EN1, EN2, PC
      %(complex_dp_format)s M2LC(NLOOPLINE)
      %(complex_dp_format)s QC(0:LOOPMAXCOEFS-1), TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER POWERS(NLOOPLINE), I, J, JJ, K, MU, NU, NC, NC1, NC2, KK, S
      LOGICAL OK
      LOGICAL PW_REFUSED
      COMMON/%(proc_prefix)sPW_REFUSED/PW_REFUSED
      DATA G/1D0,-1D0,-1D0,-1D0/
      IF (RANK+2.GT.%(maxrank)d) THEN
        WRITE(*,*) 'ERROR:: two P-wave states: rank',RANK+2,'above the maximal rank'
        STOP 1
      ENDIF
      NC=NCOEF_R(RANK)
      NC1=NCOEF_R(RANK+1)
      NC2=NCOEF_R(RANK+2)
      IF (ALL(P1(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.(0D0,0D0)).AND.
     & ALL(P2(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.(0D0,0D0)).AND.
     & ALL(PX(0:NC-1,I_SQSO,I_LOOPGROUP).EQ.(0D0,0D0)).AND.
     & ALL(PT(0:NC-1,I_SQSO,I_LOOPGROUP,0:3,0:3).EQ.(0D0,0D0))) RETURN
      DO I=0,3
        SHIFT(I,1)=0D0
        DS1(I,1)=(0D0,0D0)
        DS2(I,1)=(0D0,0D0)
        DO J=1,NLOOPLINE
          SHIFT(I,1)=SHIFT(I,1)+PL(I,J)
          DS1(I,1)=DS1(I,1)+PW_DPL(I,J)
          DS2(I,1)=DS2(I,1)+PW_DPL2(I,J)
        ENDDO
        PREFIX=0D0
        DP1=(0D0,0D0)
        DP2=(0D0,0D0)
        DO J=2,NLOOPLINE
          PREFIX=PREFIX+PL(I,J-1)
          DP1=DP1+PW_DPL(I,J-1)
          DP2=DP2+PW_DPL2(I,J-1)
          SHIFT(I,J)=PREFIX
          DS1(I,J)=DP1
          DS2(I,J)=DP2
        ENDDO
      ENDDO
      EN1=(0D0,0D0)
      EN2=(0D0,0D0)
      DO MU=0,3
        EN1=EN1+G(MU)*PW_EPS1(MU)*DCONJG(PW_EPS1(MU))
        EN2=EN2+G(MU)*PW_EPS2(MU)*DCONJG(PW_EPS2(MU))
      ENDDO
C     C1_J = dS_J1.eps1*/(eps1.eps1*), C2_J the same for the second state:
C     integers, zero for J=1
      DO J=1,NLOOPLINE
        DO S=1,2
          EE=(0D0,0D0)
          DO MU=0,3
            IF (S.EQ.1) THEN
              EE=EE+G(MU)*DS1(MU,J)*DCONJG(PW_EPS1(MU))
            ELSE
              EE=EE+G(MU)*DS2(MU,J)*DCONJG(PW_EPS2(MU))
            ENDIF
          ENDDO
          IF (S.EQ.1) THEN
            EE=EE/EN1
          ELSE
            EE=EE/EN2
          ENDIF
          F=DBLE(EE)
          IF (ABS(F).LT.0.5D0.AND.ABS(EE).LT.1D-8) THEN
            F=0D0
          ELSEIF (ABS(F-NINT(F)).GT.1D-8.OR.ABS(DIMAG(EE)).GT.1D-8.OR.J.EQ.1) THEN
            WRITE(*,*) 'ERROR:: P-wave: unexpected derivative of a loop line',J,S,EE
            STABLE=.FALSE.
            RETURN
          ELSE
            F=DBLE(NINT(F))
          ENDIF
          IF (S.EQ.1) THEN
            C1(J)=F
          ELSE
            C2(J)=F
          ENDIF
        ENDDO
      ENDDO
C     Line J doubled
      DO J=1,NLOOPLINE
        IF (C1(J).EQ.0D0.AND.C2(J).EQ.0D0) CYCLE
        DO K=0,NC1-1
          QC(K)=(0D0,0D0)
        ENDDO
        DO MU=0,3
          DO K=0,NC-1
            PC=C1(J)*P1(K,I_SQSO,I_LOOPGROUP,MU)+C2(J)*P2(K,I_SQSO,I_LOOPGROUP,MU)
            IF (PC.EQ.(0D0,0D0)) CYCLE
            QC(K)=QC(K)-2D0*G(MU)*SHIFT(MU,J)*PC
            KK=COMB_COEF_POS(K,MU+1)
            QC(KK)=QC(KK)+2D0*G(MU)*PC
          ENDDO
        ENDDO
        IF (C1(J).NE.0D0.AND.C2(J).NE.0D0) THEN
          DO K=0,NC-1
            QC(K)=QC(K)-2D0*C1(J)*C2(J)*PX(K,I_SQSO,I_LOOPGROUP)
          ENDDO
        ENDIF
        IF (ALL(QC(0:NC1-1).EQ.(0D0,0D0))) CYCLE
        DO I=1,NLOOPLINE
          POWERS(I)=1
        ENDDO
        POWERS(J)=2
        PLC=PL
        M2LC=M2L
        CALL %(proc_prefix)sTIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK+1,POWERS,PLC,M2LC,
     &   TIRCOEFS,OK)
        IF (.NOT.OK) THEN
          STABLE=.FALSE.
          PW_REFUSED=.TRUE.
          RETURN
        ENDIF
        DO I=1,3
          DO K=0,NC1-1
            RES(I)=RES(I)+QC(K)*TIRCOEFS(K,I)
          ENDDO
        ENDDO
      ENDDO
C     Line J tripled (JJ=J) and lines J and JJ doubled (JJ /= J)
      DO J=1,NLOOPLINE
        IF (C1(J).EQ.0D0) CYCLE
        DO JJ=1,NLOOPLINE
          IF (C2(JJ).EQ.0D0) CYCLE
          IF (J.EQ.JJ) THEN
            F=8D0*C1(J)*C2(JJ)
          ELSE
            F=4D0*C1(J)*C2(JJ)
          ENDIF
          DO K=0,NC2-1
            QC(K)=(0D0,0D0)
          ENDDO
          DO MU=0,3
            DO NU=0,3
              DO K=0,NC-1
                PC=PT(K,I_SQSO,I_LOOPGROUP,MU,NU)
                IF (PC.EQ.(0D0,0D0)) CYCLE
                PC=F*G(MU)*G(NU)*PC
C               (S_J-l)^mu (S_JJ-l)^nu
                QC(K)=QC(K)+SHIFT(MU,J)*SHIFT(NU,JJ)*PC
                KK=COMB_COEF_POS(K,MU+1)
                QC(KK)=QC(KK)-SHIFT(NU,JJ)*PC
                QC(COMB_COEF_POS(KK,NU+1))=QC(COMB_COEF_POS(KK,NU+1))+PC
                KK=COMB_COEF_POS(K,NU+1)
                QC(KK)=QC(KK)-SHIFT(MU,J)*PC
              ENDDO
            ENDDO
          ENDDO
          IF (ALL(QC(0:NC2-1).EQ.(0D0,0D0))) CYCLE
          DO I=1,NLOOPLINE
            POWERS(I)=1
          ENDDO
          IF (J.EQ.JJ) THEN
            POWERS(J)=3
          ELSE
            POWERS(J)=2
            POWERS(JJ)=2
          ENDIF
          PLC=PL
          M2LC=M2L
          CALL %(proc_prefix)sTIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK+2,POWERS,PLC,M2LC,
     &     TIRCOEFS,OK)
          IF (.NOT.OK) THEN
            STABLE=.FALSE.
            PW_REFUSED=.TRUE.
            RETURN
          ENDIF
          DO I=1,3
            DO K=0,NC2-1
              RES(I)=RES(I)+QC(K)*TIRCOEFS(K,I)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      END
## }

C ----------------------------------------------------------------------
C  The P-wave bound state in quadruple precision (the routines above in
C  real*16, complex*32 and MP_Dual)
C ----------------------------------------------------------------------

      SUBROUTINE %(proc_prefix)sMP_ONIA_CG(L,LZ,S,SZ,J,JZ,CG)
C     The Clebsch-Gordan coefficient <L LZ; S SZ|J JZ> in quadruple
C     precision: the square of the double-precision one (CLEBSCH_GORDAN) is
C     a rational of small denominator for these spins
      IMPLICIT NONE
      INTEGER L, LZ, S, SZ, J, JZ, Q, N
      %(real_mp_format)s CG
      REAL*8 CGD, C2
      CALL CLEBSCH_GORDAN(L,LZ,S,SZ,J,JZ,CGD)
      C2=CGD**2
      DO Q=1,120
        N=NINT(C2*Q)
        IF (ABS(C2-DBLE(N)/DBLE(Q)).LT.1D-12) THEN
          CG=SIGN(SQRT(REAL(N,KIND=16)/REAL(Q,KIND=16)),REAL(CGD,KIND=16))
          RETURN
        ENDIF
      ENDDO
      WRITE(*,*) 'ERROR:: no rational Clebsch-Gordan coefficient',CGD
      STOP 1
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PW_EPSILONS(P)
C     The directions eps(L_z) of the relative momentum, ILZ = L_z+2
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      %(complex_mp_format)s EPSLZ(0:3,3)
      COMMON/%(proc_prefix)sMP_PW_EPS/EPSLZ
      %(real_mp_format)s CMASS(NEXTERNAL)
      TYPE(MP_DUAL) KD(0:3)
      TYPE(MP_ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2
      CALL %(proc_prefix)sMP_ONIA_MASSES(CMASS)
      Q1=ONIA_Q1(ONIA_IPW)
      Q2=ONIA_Q2(ONIA_IPW)
      DO I=0,3
        KD(I)=P(I,Q1)+P(I,Q2)
      ENDDO
      DO ILZ=1,3
        CALL MP_VXXXXX(KD,CMASS(Q1)+CMASS(Q2),ILZ-2,+1,EPSW)
        DO I=0,3
          EPSLZ(I,ILZ)=EPSW%%W(I+1)%%COMP(0)
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PW_SET_DUAL(P,ILZ,PD)
C     The dual constituent momenta for the direction ILZ
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INTEGER ILZ
      TYPE(MP_DUAL) PD(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      %(complex_mp_format)s EPSLZ(0:3,3)
      COMMON/%(proc_prefix)sMP_PW_EPS/EPSLZ
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_mp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sMP_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, J
      DO J=1,NEXTERNAL
        DO I=0,3
          PD(I,J)=P(I,J)
        ENDDO
      ENDDO
      DO I=0,3
        PD(I,ONIA_Q1(ONIA_IPW))%%COMP(1)=EPSLZ(I,ILZ)
        PD(I,ONIA_Q2(ONIA_IPW))%%COMP(1)=-EPSLZ(I,ILZ)
        PW_EPS(I)=EPSLZ(I,ILZ)
      ENDDO
      PW_ILZ=ILZ
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PW_PROJ(P1,M1,NHEL1,P2,M2,NHEL2,P3,M3,NHEL3,SPIN,PROJ)
C     The spin projector of a bound state for dual constituent momenta
      USE ALOHA_OBJECT
      IMPLICIT NONE
      TYPE(MP_DUAL) P1(0:3), P2(0:3), P3(0:3), PROJ, TMP
      %(real_mp_format)s M1, M2, M3
      INTEGER NHEL1, NHEL2, NHEL3, SPIN
      TYPE(MP_ALOHA) FI, FO, VC
      %(complex_mp_format)s CI
      PARAMETER (CI=(0.0E0_16,1.0E0_16))
      CALL MP_IXXXXX(P1,M1,NHEL1,+1,1,FI)
      CALL MP_OXXXXX(P2,M2,NHEL2,-1,1,FO)
      IF (SPIN.EQ.0) THEN
        TMP = -FI%%W(1)*FO%%W(1)-FI%%W(2)*FO%%W(2)+FI%%W(3)*FO%%W(3)
     &        +FI%%W(4)*FO%%W(4)
      ELSEIF (SPIN.EQ.1) THEN
        CALL MP_VXXXXX(P3,M3,NHEL3,+1,VC)
        TMP = (FI%%W(1)*FO%%W(4)-FI%%W(3)*FO%%W(2))*(VC%%W(2)+CI*VC%%W(3))
     &      + (FI%%W(2)*FO%%W(3)-FI%%W(4)*FO%%W(1))*(VC%%W(2)-CI*VC%%W(3))
     &      + (FI%%W(1)*FO%%W(3)+FI%%W(4)*FO%%W(2))*(VC%%W(1)+VC%%W(4))
     &      + (FI%%W(2)*FO%%W(4)+FI%%W(3)*FO%%W(1))*(VC%%W(1)-VC%%W(4))
      ELSE
        WRITE(*,*) 'ERROR:: spin projector not implemented for spin',SPIN
        STOP 1
      ENDIF
      PROJ = 0.5E0_16/SQRT(2.0E0_16*M1*M2)*TMP
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_BORN_AMPS(P,NHEL,H,IC)
C     The (dual) Born amplitudes for the dual momenta P, quadruple precision
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: MPPWW, MPPWAMP
      IMPLICIT NONE
      INTEGER NEXTERNAL, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NBORNAMPS=%(nbornamps)d)
      %(real_mp_format)s ZERO
      PARAMETER (ZERO=0.0E0_16)
      TYPE(MP_DUAL) P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL), H
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      INCLUDE 'mp_coupl_same_name.inc'
%(onia_mp_born_calls)s
      END

## if(not OniaPwave2){
      SUBROUTINE %(proc_prefix)sMP_ONIA_PREPARE(P)
C     The weights of the P-wave bound state in quadruple precision (as
C     ONIA_PREPARE, without the Born matrix element)
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: MPPWAMP, %(proc_prefix)sPW_DUAL_INIT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLJZ
      INTEGER IOSEL, IOTH
      %(complex_mp_format)s WEFF(NBORNAMPS,NCOMB)
      COMMON/%(proc_prefix)sMP_ONIA_EFFECTIVE/WEFF
      %(complex_mp_format)s W2(NBORNAMPS,NCOMB,3), VW(NBORNAMPS,0:3,NCOMB)
      INTEGER FIRSTLZ(NCOMB)
      LOGICAL LZLIVE(NCOMB,3)
      COMMON/%(proc_prefix)sMP_PW_WEIGHTS/W2,VW,FIRSTLZ,LZLIVE
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(complex_mp_format)s EPSLZ(0:3,3)
      COMMON/%(proc_prefix)sMP_PW_EPS/EPSLZ
      %(complex_mp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_mp_format)s PR0(ONIA_NJZT,NCOMB,3), PR1(ONIA_NJZT,NCOMB,3)
      %(complex_mp_format)s A0(NBORNAMPS,NCOMB), A1(NBORNAMPS,NCOMB,3), CZERO
      PARAMETER (CZERO=(0.0E0_16,0.0E0_16))
      TYPE(MP_DUAL) PD(0:3,NEXTERNAL), KD(0:3), PR, PRK
      %(real_mp_format)s CMASS(NEXTERNAL), CG, FACT
      LOGICAL ACTIVE(NCOMB)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      INTEGER H, T, K, I, O, Q1, Q2, ILZ, LZ, SZ, MU
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL %(proc_prefix)sPW_DUAL_INIT()
      CALL %(proc_prefix)sMP_ONIA_MASSES(CMASS)
      FACT=%(onia_mp_norm)s
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
%(onia_ldme_perturbative)s
      FACT=FACT*%(onia_ldme_product)s
      CALL %(proc_prefix)sMP_ONIA_PW_EPSILONS(P)
      DO H=1,NCOMB
        ACTIVE(H)=.FALSE.
        DO ILZ=1,3
          LZ=ILZ-2
          CALL %(proc_prefix)sMP_ONIA_PW_SET_DUAL(P,ILZ,PD)
          DO T=1,ONIA_NJZT
            PR=(1.0E0_16,0.0E0_16)
            DO K=1,ONIA_NONIA
              Q1=ONIA_Q1(K)
              Q2=ONIA_Q2(K)
              DO I=0,3
                KD(I)=PD(I,Q1)+PD(I,Q2)
              ENDDO
              IF (K.EQ.ONIA_IPW) THEN
                SZ=ONIA_JZ(K,T)-LZ
                IF (ABS(SZ).GT.ONIA_S(K)) THEN
                  PR=CZERO
                  EXIT
                ENDIF
                CALL %(proc_prefix)sMP_ONIA_CG(1,LZ,ONIA_S(K),SZ,ONIA_J(K),ONIA_JZ(K,T),CG)
              ELSE
                SZ=ONIA_JZ(K,T)
                CG=1.0E0_16
              ENDIF
              CALL %(proc_prefix)sMP_ONIA_PW_PROJ(PD(0,Q1),CMASS(Q1),ONIA_HELC(Q1,H),
     &         PD(0,Q2),CMASS(Q2),ONIA_HELC(Q2,H),KD,CMASS(Q1)+CMASS(Q2),
     &         SZ,ONIA_S(K),PRK)
              PR=PR*PRK*CG
            ENDDO
            PR0(T,H,ILZ)=PR%%COMP(0)
            PR1(T,H,ILZ)=PR%%COMP(1)
            IF (PR0(T,H,ILZ).NE.CZERO.OR.PR1(T,H,ILZ).NE.CZERO) ACTIVE(H)=.TRUE.
          ENDDO
        ENDDO
      ENDDO
      DO H=1,NCOMB
        A0(1:NBORNAMPS,H)=CZERO
        A1(1:NBORNAMPS,H,1:3)=CZERO
        IF (.NOT.ACTIVE(H)) CYCLE
        DO I=1,NEXTERNAL
          NHEL(I)=ONIA_HELC(I,H)
        ENDDO
        DO ILZ=1,3
          CALL %(proc_prefix)sMP_ONIA_PW_SET_DUAL(P,ILZ,PD)
          CALL %(proc_prefix)sMP_ONIA_BORN_AMPS(PD,NHEL,H,IC)
          DO I=1,NBORNAMPS
            A0(I,H)=MPPWAMP(I)%%COMP(0)
            A1(I,H,ILZ)=MPPWAMP(I)%%COMP(1)
          ENDDO
        ENDDO
      ENDDO
      BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,1:ONIA_NJZT)=CZERO
      DO H=1,NCOMB
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO ILZ=1,3
            BPROJ(1:NBORNAMPS,O,T)=BPROJ(1:NBORNAMPS,O,T)
     &       +PR0(T,H,ILZ)*A1(1:NBORNAMPS,H,ILZ)+PR1(T,H,ILZ)*A0(1:NBORNAMPS,H)
          ENDDO
        ENDDO
      ENDDO
C     The J_z selected with ONIA_SELECT_JZ (all by default)
      CALL %(proc_prefix)sONIA_JZ_TSEL(TSEL,ALLJZ)
      IF (.NOT.ALLJZ) THEN
        DO T=1,ONIA_NJZT
          IF (TSEL(T)) CYCLE
          BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,T)=CZERO
        ENDDO
      ENDIF
C     The helicities of the other legs selected by SLOOPMATRIXHEL, as
C     their signature ONIA_OTHERS (all by default)
      CALL %(proc_prefix)sONIA_OTHERS_SEL(.FALSE.,IOSEL)
      IF (IOSEL.NE.0) THEN
        DO IOTH=1,ONIA_NOTHERS
          IF (IOTH.EQ.IOSEL) CYCLE
          BPROJ(1:NBORNAMPS,IOTH,1:ONIA_NJZT)=CZERO
        ENDDO
      ENDIF
      DO H=1,NCOMB
        WEFF(1:NBORNAMPS,H)=CZERO
        W2(1:NBORNAMPS,H,1:3)=CZERO
        VW(1:NBORNAMPS,0:3,H)=CZERO
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO ILZ=1,3
            WEFF(1:NBORNAMPS,H)=WEFF(1:NBORNAMPS,H)
     &       +CONJG(PR1(T,H,ILZ))*BPROJ(1:NBORNAMPS,O,T)
            W2(1:NBORNAMPS,H,ILZ)=W2(1:NBORNAMPS,H,ILZ)
     &       +CONJG(PR0(T,H,ILZ))*BPROJ(1:NBORNAMPS,O,T)
          ENDDO
        ENDDO
        WEFF(1:NBORNAMPS,H)=FACT*WEFF(1:NBORNAMPS,H)
        W2(1:NBORNAMPS,H,1:3)=FACT*W2(1:NBORNAMPS,H,1:3)
        DO ILZ=1,3
          DO MU=0,3
            VW(1:NBORNAMPS,MU,H)=VW(1:NBORNAMPS,MU,H)
     &       +W2(1:NBORNAMPS,H,ILZ)*CONJG(EPSLZ(MU,ILZ))
          ENDDO
        ENDDO
      ENDDO
      DO H=1,NCOMB
        FIRSTLZ(H)=0
        DO ILZ=1,3
          LZLIVE(H,ILZ)=ANY(W2(1:NBORNAMPS,H,ILZ).NE.CZERO)
          IF (LZLIVE(H,ILZ).AND.FIRSTLZ(H).EQ.0) FIRSTLZ(H)=ILZ
        ENDDO
        IF (FIRSTLZ(H).EQ.0.AND.ANY(WEFF(1:NBORNAMPS,H).NE.CZERO)) THEN
          FIRSTLZ(H)=1
          LZLIVE(H,1)=.TRUE.
        ENDIF
        LIVE(H)=FIRSTLZ(H).NE.0
      ENDDO
      END

      LOGICAL FUNCTION %(proc_prefix)sMP_ONIA_PW_DIRECTION(P,H,ILZ,PD)
C     As ONIA_PW_DIRECTION, in quadruple precision
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INTEGER H, ILZ
      TYPE(MP_DUAL) PD(0:3,NEXTERNAL)
      %(complex_mp_format)s W2(NBORNAMPS,NCOMB,3), VW(NBORNAMPS,0:3,NCOMB)
      INTEGER FIRSTLZ(NCOMB)
      LOGICAL LZLIVE(NCOMB,3)
      COMMON/%(proc_prefix)sMP_PW_WEIGHTS/W2,VW,FIRSTLZ,LZLIVE
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_mp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sMP_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      %(proc_prefix)sMP_ONIA_PW_DIRECTION=LZLIVE(H,ILZ)
      IF (.NOT.LZLIVE(H,ILZ)) RETURN
      CALL %(proc_prefix)sMP_ONIA_PW_SET_DUAL(P,ILZ,PD)
      PW_FIRST=ILZ.EQ.FIRSTLZ(H)
      END

      LOGICAL FUNCTION %(proc_prefix)sMP_ONIA_PW_CT_INPUT(H,PASS,AMP,AMPL)
C     As ONIA_PW_CT_INPUT, in quadruple precision
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: MPPWAMPL
      IMPLICIT NONE
      INTEGER NCOMB, NBORNAMPS, NCTAMPS
      PARAMETER (NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d, NCTAMPS=%(nctamps)d)
      INTEGER H, PASS
      %(complex_mp_format)s AMP(NBORNAMPS), AMPL(3,NCTAMPS), CZERO
      PARAMETER (CZERO=(0.0E0_16,0.0E0_16))
      %(complex_mp_format)s WEFF(NBORNAMPS,NCOMB)
      COMMON/%(proc_prefix)sMP_ONIA_EFFECTIVE/WEFF
      %(complex_mp_format)s W2(NBORNAMPS,NCOMB,3), VW(NBORNAMPS,0:3,NCOMB)
      INTEGER FIRSTLZ(NCOMB)
      LOGICAL LZLIVE(NCOMB,3)
      COMMON/%(proc_prefix)sMP_PW_WEIGHTS/W2,VW,FIRSTLZ,LZLIVE
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_mp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sMP_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER I, K, COMP
      IF (PASS.EQ.1) THEN
        AMP(1:NBORNAMPS)=W2(1:NBORNAMPS,H,PW_ILZ)
        COMP=1
      ELSE
        IF (.NOT.PW_FIRST) THEN
          %(proc_prefix)sMP_ONIA_PW_CT_INPUT=.FALSE.
          RETURN
        ENDIF
        AMP(1:NBORNAMPS)=WEFF(1:NBORNAMPS,H)
        COMP=0
      ENDIF
      %(proc_prefix)sMP_ONIA_PW_CT_INPUT=ANY(AMP(1:NBORNAMPS).NE.CZERO)
      DO I=1,NCTAMPS
        DO K=1,3
          AMPL(K,I)=MPPWAMPL(K,I)%%COMP(COMP)
        ENDDO
      ENDDO
      END
## }

      SUBROUTINE %(proc_prefix)sMP_ONIA_PW_CT(H,HEL_MULT,ANS)
C     ANS = ANS + the interference of the counterterm amplitudes of the
C     current direction with their effective Born (MP_ONIA_PW_CT_INPUT: the
C     derivative with W2, on the first direction the value with W1), in
C     quadruple precision
      IMPLICIT NONE
      INTEGER NBORNAMPS, NCTAMPS, NCOLORROWS
      PARAMETER (NBORNAMPS=%(nbornamps)d, NCTAMPS=%(nctamps)d, NCOLORROWS=%(nloopamps)d)
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
      %(complex_mp_format)s IMAG1
      PARAMETER (IMAG1=(0.0E0_16,1.0E0_16))
      INTEGER H, HEL_MULT
      %(real_mp_format)s ANS(3,0:NSQUAREDSO), TEMP2
      %(complex_mp_format)s AMP(NBORNAMPS), AMPL(3,NCTAMPS), CTEMP, CFTOT
      INTEGER PASS, I, J, K, ITEMP
      LOGICAL %(proc_prefix)sMP_ONIA_PW_CT_INPUT
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP, %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP
      INTEGER %(proc_prefix)sML5SQSOINDEX
      INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
      INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
      COMMON/%(proc_prefix)sCF/CF_D,CF_N
      INTEGER SQSO_TARGET
      COMMON/%(proc_prefix)sSOCHOICE/SQSO_TARGET
      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE
      LOGICAL LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/%(proc_prefix)sSO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,
     & MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
## if(OniaPwave2){
C     (two P-wave bound states: the four components of each direction pair)
      DO PASS=1,4
## } else {
      DO PASS=1,2
## }
        IF (.NOT.%(proc_prefix)sMP_ONIA_PW_CT_INPUT(H,PASS,AMP,AMPL)) CYCLE
        DO J=1,NBORNAMPS
          CTEMP=HEL_MULT*2.0E0_16*CONJG(AMP(J))
          DO I=1,NCTAMPS
            CFTOT=CMPLX(CF_N(I,J)/REAL(ABS(CF_D(I,J)),KIND=16),0.0E0_16,KIND=16)
            IF (CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
            ITEMP=%(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),
     &       %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(J))
            IF (.NOT.FILTER_SO.OR.SQSO_TARGET.EQ.ITEMP) THEN
              DO K=1,3
                TEMP2=REAL(CFTOT*AMPL(K,I)*CTEMP,KIND=16)
                ANS(K,ITEMP)=ANS(K,ITEMP)+TEMP2
                ANS(K,0)=ANS(K,0)+TEMP2
              ENDDO
            ENDIF
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE MP_%(proc_prefix)sPW_SPLIT_COEFS(COEFS,C0,C1)
C     The value and the derivative of the dual coefficients of a loop vertex
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
      IMPLICIT NONE
      TYPE(MP_DUAL) COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s C0(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s C1(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      INTEGER I, J, K
      DO K=1,MAXLWFSIZE
        DO J=0,VERTEXMAXCOEFS-1
          DO I=1,MAXLWFSIZE
            C0(I,J,K)=COEFS(I,J,K)%%COMP(0)
            C1(I,J,K)=COEFS(I,J,K)%%COMP(1)
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE MP_%(proc_prefix)sPW_ADD_WL(A,RANK,LCUT_SIZE,OUT_SIZE,OUT)
C     OUT = OUT + A for the loop wavefunctions of rank RANK
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      %(complex_mp_format)s A(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s OUT(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER RANK, LCUT_SIZE, OUT_SIZE, I, J, K
      DO I=1,LCUT_SIZE
        DO K=0,NCOEF_R(RANK)-1
          DO J=1,OUT_SIZE
            OUT(J,K,I)=OUT(J,K,I)+A(J,K,I)
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE MP_%(proc_prefix)sPW_MERGE_LOOP(LOOP_WF,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,
     & SYMFACT,MULTIPLIER,COLOR_ID,HELCONFIG,WEIGHT,TARGET)
C     MP_CREATE_LOOP_COEFS with the effective Born WEIGHT and the polynomials
C     TARGET in place of AMP and LOOPCOEFS
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
      %(real_mp_format)s ZERO,ONE
      PARAMETER (ZERO=0.0E0_16,ONE=1.0E0_16)
      %(complex_mp_format)s IMAG1
      PARAMETER (IMAG1=(ZERO,ONE))
      %(complex_mp_format)s CMPLX_ZERO
      PARAMETER (CMPLX_ZERO=(ZERO,ZERO))
      INTEGER    NCOLORROWS
      PARAMETER (NCOLORROWS=%(nloopamps)d)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
      %(complex_mp_format)s LOOP_WF(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER RANK, COLOR_ID, SYMFACT, MULTIPLIER, LCUT_SIZE, HELCONFIG, LOOP_GROUP_NUMBER
      %(complex_mp_format)s WEIGHT(NBORNAMPS)
      %(complex_mp_format)s TARGET(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      %(complex_mp_format)s CFTOT
      %(complex_mp_format)s CONST(NAMPSO)
      INTEGER I
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP, %(proc_prefix)sML5SQSOINDEX
      INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
      INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
      COMMON/%(proc_prefix)sCF/CF_D,CF_N
      LOGICAL CHECKPHASE
      LOGICAL HELDOUBLECHECKED
      COMMON/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED
      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/%(proc_prefix)sFILTERS/GOODAMP,GOODHEL,HELOFFSET
      INTEGER HELPICKED
      COMMON/%(proc_prefix)sHELCHOICE/HELPICKED
      INTEGER BORNAMP_SO(NBORNAMPS)
      INTEGER LOOPAMP_SQSO(NAMPSO,NCOLORROWS)
      COMMON/%(proc_prefix)sSOTABLES/BORNAMP_SO,LOOPAMP_SQSO
      DO I=1,NAMPSO
        CONST(I)=CMPLX_ZERO
      ENDDO
      DO I=1,NBORNAMPS
        IF (WEIGHT(I).EQ.CMPLX_ZERO) CYCLE
        CFTOT=CMPLX(CF_N(COLOR_ID,I)/(ONE*ABS(CF_D(COLOR_ID,I))),ZERO,KIND=16)
        IF(CF_D(COLOR_ID,I).LT.0) CFTOT=CFTOT*IMAG1
        CONST(BORNAMP_SO(I))=CONST(BORNAMP_SO(I))+CFTOT*CONJG(WEIGHT(I))
      ENDDO
      DO I=1,NAMPSO
        IF (CONST(I).NE.CMPLX_ZERO) THEN
          CONST(I)=(CONST(I)*MULTIPLIER)/SYMFACT
          IF (.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.HELPICKED.EQ.-1) THEN
            CONST(I)=CONST(I)*GOODHEL(HELCONFIG)
          ENDIF
          IF (LOOPAMP_SQSO(I,COLOR_ID).EQ.0) LOOPAMP_SQSO(I,COLOR_ID)=
     &     %(proc_prefix)sML5SQSOINDEX(I,%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(COLOR_ID))
          CALL MP_%(proc_prefix)sMERGE_WL(LOOP_WF,RANK,LCUT_SIZE,CONST(I),
     &     TARGET(0,LOOPAMP_SQSO(I,COLOR_ID),LOOP_GROUP_NUMBER))
        ENDIF
      ENDDO
      END

      SUBROUTINE MP_%(proc_prefix)sPW_CREATE_LOOP_COEFS(WL,DWL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,
     & SYMFACT,MULTIPLIER,COLOR_ID,HELCONFIG)
C     As PW_CREATE_LOOP_COEFS, in quadruple precision
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NBORNAMPS, NCOMB, NLOOPGROUPS, NSQUAREDSO
      PARAMETER (NBORNAMPS=%(nbornamps)d, NCOMB=%(ncomb)d)
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d, NSQUAREDSO=%(nSquaredSO)d)
      %(complex_mp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s DWL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER RANK, LCUT_SIZE, LOOP_GROUP_NUMBER, SYMFACT, MULTIPLIER, COLOR_ID, HELCONFIG
      %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/%(proc_prefix)sMP_LCOEFS/LOOPCOEFS
      %(complex_mp_format)s PCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      COMMON/%(proc_prefix)sMP_PW_PCOEFS/PCOEFS
      %(complex_mp_format)s WEFF(NBORNAMPS,NCOMB)
      COMMON/%(proc_prefix)sMP_ONIA_EFFECTIVE/WEFF
      %(complex_mp_format)s W2(NBORNAMPS,NCOMB,3), VW(NBORNAMPS,0:3,NCOMB)
      INTEGER FIRSTLZ(NCOMB)
      LOGICAL LZLIVE(NCOMB,3)
      COMMON/%(proc_prefix)sMP_PW_WEIGHTS/W2,VW,FIRSTLZ,LZLIVE
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_mp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sMP_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      INTEGER MU
      CALL MP_%(proc_prefix)sPW_MERGE_LOOP(DWL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,W2(1,HELCONFIG,PW_ILZ),LOOPCOEFS)
      IF (.NOT.PW_FIRST) RETURN
      CALL MP_%(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,WEFF(1,HELCONFIG),LOOPCOEFS)
      DO MU=0,3
        CALL MP_%(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,VW(1,MU,HELCONFIG),PCOEFS(0,1,1,MU))
      ENDDO
      END

## if(not OniaPwave2){
      SUBROUTINE %(proc_prefix)sMP_ONIA_PW_DOUBLED(I_SQSO,I_LOOPGROUP,CTMODE,NLOOPLINE,RANK,PL,M2L,RES,STABLE)
C     As ONIA_PW_DOUBLED, in quadruple precision
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NEXTERNAL, NLOOPGROUPS, NSQUAREDSO
      PARAMETER (NEXTERNAL=%(nexternal)d)
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d, NSQUAREDSO=%(nSquaredSO)d)
      INTEGER I_SQSO, I_LOOPGROUP, CTMODE, NLOOPLINE, RANK
      %(real_mp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_mp_format)s RES(3)
      LOGICAL STABLE
      %(complex_mp_format)s PCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      COMMON/%(proc_prefix)sMP_PW_PCOEFS/PCOEFS
      %(complex_mp_format)s MP_PW_DPL(0:3,NEXTERNAL)
      COMMON/%(proc_prefix)sMP_PW_DPL/MP_PW_DPL
      INTEGER PW_ILZ
      LOGICAL PW_FIRST
      %(complex_mp_format)s PW_EPS(0:3)
      COMMON/%(proc_prefix)sMP_PW_DIR/PW_EPS,PW_ILZ,PW_FIRST
      %(real_mp_format)s SHIFT(0:3,NLOOPLINE), PLC(0:3,NLOOPLINE), PREFIX, G(0:3), C
      %(complex_mp_format)s DSHIFT(0:3,NLOOPLINE), DPREFIX, EE, EN, PC, CZERO
      PARAMETER (CZERO=(0.0E0_16,0.0E0_16))
      %(mass_dp_format)s M2LC(NLOOPLINE)
      %(complex_mp_format)s QC(0:LOOPMAXCOEFS-1), TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER POWERS(NLOOPLINE), I, J, K, MU, NC, NC1, KK
      LOGICAL OK
      LOGICAL PW_REFUSED
      COMMON/%(proc_prefix)sPW_REFUSED/PW_REFUSED
      DATA G/1.0E0_16,-1.0E0_16,-1.0E0_16,-1.0E0_16/
      IF (RANK+1.GT.%(maxrank)d) THEN
        WRITE(*,*) 'ERROR:: P-wave: rank',RANK+1,'above the maximal rank'
        STOP 1
      ENDIF
      NC=NCOEF_R(RANK)
      NC1=NCOEF_R(RANK+1)
      IF (ALL(PCOEFS(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.CZERO)) RETURN
      DO I=0,3
        SHIFT(I,1)=0.0E0_16
        DSHIFT(I,1)=CZERO
        DO J=1,NLOOPLINE
          SHIFT(I,1)=SHIFT(I,1)+PL(I,J)
          DSHIFT(I,1)=DSHIFT(I,1)+MP_PW_DPL(I,J)
        ENDDO
        PREFIX=0.0E0_16
        DPREFIX=CZERO
        DO J=2,NLOOPLINE
          PREFIX=PREFIX+PL(I,J-1)
          DPREFIX=DPREFIX+MP_PW_DPL(I,J-1)
          SHIFT(I,J)=PREFIX
          DSHIFT(I,J)=DPREFIX
        ENDDO
      ENDDO
      EN=CZERO
      DO MU=0,3
        EN=EN+G(MU)*PW_EPS(MU)*CONJG(PW_EPS(MU))
      ENDDO
      DO J=1,NLOOPLINE
        EE=CZERO
        DO MU=0,3
          EE=EE+G(MU)*DSHIFT(MU,J)*CONJG(PW_EPS(MU))
        ENDDO
        EE=EE/EN
        C=REAL(EE,KIND=16)
        IF (ABS(C).LT.0.5E0_16.AND.ABS(EE).LT.1.0E-8_16) CYCLE
        IF (ABS(C-NINT(C)).GT.1.0E-8_16.OR.ABS(AIMAG(EE)).GT.1.0E-8_16.OR.J.EQ.1) THEN
          WRITE(*,*) 'ERROR:: P-wave: unexpected derivative of a loop line',J
          STABLE=.FALSE.
          RETURN
        ENDIF
        C=REAL(NINT(C),KIND=16)
        DO K=0,NC1-1
          QC(K)=CZERO
        ENDDO
        DO MU=0,3
          DO K=0,NC-1
            PC=PCOEFS(K,I_SQSO,I_LOOPGROUP,MU)
            IF (PC.EQ.CZERO) CYCLE
            QC(K)=QC(K)-2.0E0_16*C*G(MU)*SHIFT(MU,J)*PC
            KK=COMB_COEF_POS(K,MU+1)
            QC(KK)=QC(KK)+2.0E0_16*C*G(MU)*PC
          ENDDO
        ENDDO
        DO I=1,NLOOPLINE
          POWERS(I)=1
        ENDDO
        POWERS(J)=2
        PLC=PL
        M2LC=M2L
        CALL %(proc_prefix)sMP_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK+1,POWERS,PLC,M2LC,
     &   TIRCOEFS,OK)
        IF (.NOT.OK) THEN
          STABLE=.FALSE.
          PW_REFUSED=.TRUE.
          RETURN
        ENDIF
        DO I=1,3
          DO K=0,NC1-1
            RES(I)=RES(I)+QC(K)*TIRCOEFS(K,I)
          ENDDO
        ENDDO
      ENDDO
      END
## } else {
C ----------------------------------------------------------------------
C  Two P-wave bound states in quadruple precision: the routines of the
C  two-state block above in real*16, complex*32 and MP_Dual (the dual
C  numbers of mp_dual2_variables.f). The weights are those of
C  ONIA_PREPARE without the soft-current amplitudes and the Born matrix
C  element, which only the pole prediction (double precision) uses.
C ----------------------------------------------------------------------

      SUBROUTINE %(proc_prefix)sMP_ONIA_PW2_EPSILONS(P)
C     As ONIA_PW2_EPSILONS, in quadruple precision
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      %(complex_mp_format)s EPS1(0:3,3), EPS2(0:3,3)
      COMMON/%(proc_prefix)sMP_PW2_EPS/EPS1,EPS2
      %(real_mp_format)s CMASS(NEXTERNAL)
      TYPE(MP_DUAL) KD(0:3)
      TYPE(MP_ALOHA) EPSW
      INTEGER I, ILZ, Q1, Q2, S, K
      CALL %(proc_prefix)sMP_ONIA_MASSES(CMASS)
      DO S=1,2
        IF (S.EQ.1) THEN
          K=ONIA_IPW
        ELSE
          K=ONIA_IPW2
        ENDIF
        Q1=ONIA_Q1(K)
        Q2=ONIA_Q2(K)
        DO I=0,3
          KD(I)=P(I,Q1)+P(I,Q2)
        ENDDO
        DO ILZ=1,3
          CALL MP_VXXXXX(KD,CMASS(Q1)+CMASS(Q2),ILZ-2,+1,EPSW)
          DO I=0,3
            IF (S.EQ.1) THEN
              EPS1(I,ILZ)=EPSW%%W(I+1)%%COMP(0)
            ELSE
              EPS2(I,ILZ)=EPSW%%W(I+1)%%COMP(0)
            ENDIF
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PW2_SET_DUAL(P,IDIR,PD)
C     As ONIA_PW2_SET_DUAL, in quadruple precision
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INTEGER IDIR
      TYPE(MP_DUAL) PD(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      %(complex_mp_format)s EPS1(0:3,3), EPS2(0:3,3)
      COMMON/%(proc_prefix)sMP_PW2_EPS/EPS1,EPS2
      INTEGER PW_IDIR, PW_ILZ1, PW_ILZ2
      %(complex_mp_format)s PW_EPS1(0:3), PW_EPS2(0:3)
      COMMON/%(proc_prefix)sMP_PW2_DIR/PW_EPS1,PW_EPS2,PW_IDIR,PW_ILZ1,PW_ILZ2
      INTEGER I, J
      PW_IDIR=IDIR
      PW_ILZ1=MOD(IDIR-1,3)+1
      PW_ILZ2=(IDIR-1)/3+1
      DO J=1,NEXTERNAL
        DO I=0,3
          PD(I,J)=P(I,J)
        ENDDO
      ENDDO
      DO I=0,3
        PD(I,ONIA_Q1(ONIA_IPW))%%COMP(1)=EPS1(I,PW_ILZ1)
        PD(I,ONIA_Q2(ONIA_IPW))%%COMP(1)=-EPS1(I,PW_ILZ1)
        PD(I,ONIA_Q1(ONIA_IPW2))%%COMP(2)=EPS2(I,PW_ILZ2)
        PD(I,ONIA_Q2(ONIA_IPW2))%%COMP(2)=-EPS2(I,PW_ILZ2)
        PW_EPS1(I)=EPS1(I,PW_ILZ1)
        PW_EPS2(I)=EPS2(I,PW_ILZ2)
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PREPARE(P)
C     Two P-wave bound states: the weights W0 (WEFF), W1, W2, W3 and their
C     direction sums in quadruple precision (as ONIA_PREPARE)
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: MPPWAMP, %(proc_prefix)sPW_DUAL_INIT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLJZ
      INTEGER IOSEL, IOTH
      %(complex_mp_format)s WEFF(NBORNAMPS,NCOMB)
      COMMON/%(proc_prefix)sMP_ONIA_EFFECTIVE/WEFF
      %(complex_mp_format)s W1(NBORNAMPS,NCOMB,3), W2(NBORNAMPS,NCOMB,3)
      %(complex_mp_format)s W3(NBORNAMPS,NCOMB,9)
      %(complex_mp_format)s VW1(NBORNAMPS,0:3,NCOMB), VW2(NBORNAMPS,0:3,NCOMB)
      %(complex_mp_format)s V3A(NBORNAMPS,0:3,NCOMB,3), V3B(NBORNAMPS,0:3,NCOMB,3)
      %(complex_mp_format)s XW(NBORNAMPS,NCOMB), TW(NBORNAMPS,0:3,0:3,NCOMB)
      COMMON/%(proc_prefix)sMP_PW2_WEIGHTS/W1,W2,W3,VW1,VW2,V3A,V3B,XW,TW
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(complex_mp_format)s EPS1(0:3,3), EPS2(0:3,3)
      COMMON/%(proc_prefix)sMP_PW2_EPS/EPS1,EPS2
      %(complex_mp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_mp_format)s PRC(ONIA_NJZT,NCOMB,9,0:3)
      %(complex_mp_format)s A(NBORNAMPS,NCOMB,9,0:3)
      SAVE PRC, A
      %(complex_mp_format)s E12, CZERO
      PARAMETER (CZERO=(0.0E0_16,0.0E0_16))
      TYPE(MP_DUAL) PD(0:3,NEXTERNAL), KD(0:3), PR, PRK
      %(real_mp_format)s CMASS(NEXTERNAL), CG, FACT, GMET(0:3)
      LOGICAL ACTIVE(NCOMB)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      DATA GMET/1.0E0_16,-1.0E0_16,-1.0E0_16,-1.0E0_16/
      INTEGER H, T, K, I, O, Q1, Q2, IDIR, ILZ1, ILZ2, LZ, SZ, MU, NU, C
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      CALL %(proc_prefix)sPW_DUAL_INIT()
      CALL %(proc_prefix)sMP_ONIA_MASSES(CMASS)
C     The normalisation of the colour projectors and the LDMEs
      FACT=%(onia_mp_norm)s
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
%(onia_ldme_perturbative)s
      FACT=FACT*%(onia_ldme_product)s
C     The directions of the relative momenta
      CALL %(proc_prefix)sMP_ONIA_PW2_EPSILONS(P)
C     The products of the spin projectors (with the Clebsch-Gordan
C     coefficients of both P-wave states) and their dual components
      DO H=1,NCOMB
        ACTIVE(H)=.FALSE.
        DO IDIR=1,9
          ILZ1=MOD(IDIR-1,3)+1
          ILZ2=(IDIR-1)/3+1
          CALL %(proc_prefix)sMP_ONIA_PW2_SET_DUAL(P,IDIR,PD)
          DO T=1,ONIA_NJZT
            PR=(1.0E0_16,0.0E0_16)
            DO K=1,ONIA_NONIA
              Q1=ONIA_Q1(K)
              Q2=ONIA_Q2(K)
              DO I=0,3
                KD(I)=PD(I,Q1)+PD(I,Q2)
              ENDDO
              IF (K.EQ.ONIA_IPW.OR.K.EQ.ONIA_IPW2) THEN
                IF (K.EQ.ONIA_IPW) THEN
                  LZ=ILZ1-2
                ELSE
                  LZ=ILZ2-2
                ENDIF
                SZ=ONIA_JZ(K,T)-LZ
                IF (ABS(SZ).GT.ONIA_S(K)) THEN
                  PR=CZERO
                  EXIT
                ENDIF
                CALL %(proc_prefix)sMP_ONIA_CG(1,LZ,ONIA_S(K),SZ,ONIA_J(K),ONIA_JZ(K,T),CG)
              ELSE
                SZ=ONIA_JZ(K,T)
                CG=1.0E0_16
              ENDIF
              CALL %(proc_prefix)sMP_ONIA_PW_PROJ(PD(0,Q1),CMASS(Q1),ONIA_HELC(Q1,H),
     &         PD(0,Q2),CMASS(Q2),ONIA_HELC(Q2,H),KD,CMASS(Q1)+CMASS(Q2),
     &         SZ,ONIA_S(K),PRK)
              PR=PR*PRK*CG
            ENDDO
            DO C=0,3
              PRC(T,H,IDIR,C)=PR%%COMP(C)
              IF (PRC(T,H,IDIR,C).NE.CZERO) ACTIVE(H)=.TRUE.
            ENDDO
          ENDDO
        ENDDO
      ENDDO
C     The Born amplitudes and their derivatives
      DO H=1,NCOMB
        A(1:NBORNAMPS,H,1:9,0:3)=CZERO
        IF (.NOT.ACTIVE(H)) CYCLE
        DO I=1,NEXTERNAL
          NHEL(I)=ONIA_HELC(I,H)
        ENDDO
        DO IDIR=1,9
          CALL %(proc_prefix)sMP_ONIA_PW2_SET_DUAL(P,IDIR,PD)
          CALL %(proc_prefix)sMP_ONIA_BORN_AMPS(PD,NHEL,H,IC)
          DO C=0,3
            DO I=1,NBORNAMPS
              A(I,H,IDIR,C)=MPPWAMP(I)%%COMP(C)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
C     The projected Born amplitudes: the mixed component of PR x A
      BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,1:ONIA_NJZT)=CZERO
      DO H=1,NCOMB
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO IDIR=1,9
            DO C=0,3
              BPROJ(1:NBORNAMPS,O,T)=BPROJ(1:NBORNAMPS,O,T)
     &         +PRC(T,H,IDIR,3-C)*A(1:NBORNAMPS,H,IDIR,C)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
C     The J_z selected with ONIA_SELECT_JZ (all by default)
      CALL %(proc_prefix)sONIA_JZ_TSEL(TSEL,ALLJZ)
      IF (.NOT.ALLJZ) THEN
        DO T=1,ONIA_NJZT
          IF (TSEL(T)) CYCLE
          BPROJ(1:NBORNAMPS,1:ONIA_NOTHERS,T)=CZERO
        ENDDO
      ENDIF
C     The helicities of the other legs selected by SLOOPMATRIXHEL, as
C     their signature ONIA_OTHERS (all by default)
      CALL %(proc_prefix)sONIA_OTHERS_SEL(.FALSE.,IOSEL)
      IF (IOSEL.NE.0) THEN
        DO IOTH=1,ONIA_NOTHERS
          IF (IOTH.EQ.IOSEL) CYCLE
          BPROJ(1:NBORNAMPS,IOTH,1:ONIA_NJZT)=CZERO
        ENDDO
      ENDIF
C     The weights of each constituent helicity configuration and their
C     direction sums
      DO H=1,NCOMB
        WEFF(1:NBORNAMPS,H)=CZERO
        W1(1:NBORNAMPS,H,1:3)=CZERO
        W2(1:NBORNAMPS,H,1:3)=CZERO
        W3(1:NBORNAMPS,H,1:9)=CZERO
        VW1(1:NBORNAMPS,0:3,H)=CZERO
        VW2(1:NBORNAMPS,0:3,H)=CZERO
        V3A(1:NBORNAMPS,0:3,H,1:3)=CZERO
        V3B(1:NBORNAMPS,0:3,H,1:3)=CZERO
        XW(1:NBORNAMPS,H)=CZERO
        TW(1:NBORNAMPS,0:3,0:3,H)=CZERO
        LIVE(H)=.FALSE.
        IF (.NOT.ACTIVE(H)) CYCLE
        O=ONIA_OTHERS(H)
        DO T=1,ONIA_NJZT
          DO IDIR=1,9
            ILZ1=MOD(IDIR-1,3)+1
            ILZ2=(IDIR-1)/3+1
            WEFF(1:NBORNAMPS,H)=WEFF(1:NBORNAMPS,H)
     &       +CONJG(PRC(T,H,IDIR,3))*BPROJ(1:NBORNAMPS,O,T)
            W1(1:NBORNAMPS,H,ILZ1)=W1(1:NBORNAMPS,H,ILZ1)
     &       +CONJG(PRC(T,H,IDIR,2))*BPROJ(1:NBORNAMPS,O,T)
            W2(1:NBORNAMPS,H,ILZ2)=W2(1:NBORNAMPS,H,ILZ2)
     &       +CONJG(PRC(T,H,IDIR,1))*BPROJ(1:NBORNAMPS,O,T)
            W3(1:NBORNAMPS,H,IDIR)=W3(1:NBORNAMPS,H,IDIR)
     &       +CONJG(PRC(T,H,IDIR,0))*BPROJ(1:NBORNAMPS,O,T)
          ENDDO
        ENDDO
        WEFF(1:NBORNAMPS,H)=FACT*WEFF(1:NBORNAMPS,H)
        W1(1:NBORNAMPS,H,1:3)=FACT*W1(1:NBORNAMPS,H,1:3)
        W2(1:NBORNAMPS,H,1:3)=FACT*W2(1:NBORNAMPS,H,1:3)
        W3(1:NBORNAMPS,H,1:9)=FACT*W3(1:NBORNAMPS,H,1:9)
C       (the weights of the polynomials of the propagator derivatives carry
C       the complex conjugate of the directions: MP_PW_MERGE_LOOP conjugates
C       them back)
        DO IDIR=1,9
          ILZ1=MOD(IDIR-1,3)+1
          ILZ2=(IDIR-1)/3+1
          E12=CZERO
          DO MU=0,3
            E12=E12+GMET(MU)*EPS1(MU,ILZ1)*EPS2(MU,ILZ2)
          ENDDO
          XW(1:NBORNAMPS,H)=XW(1:NBORNAMPS,H)+W3(1:NBORNAMPS,H,IDIR)*CONJG(E12)
          DO MU=0,3
            V3A(1:NBORNAMPS,MU,H,ILZ1)=V3A(1:NBORNAMPS,MU,H,ILZ1)
     &       +W3(1:NBORNAMPS,H,IDIR)*CONJG(EPS2(MU,ILZ2))
            V3B(1:NBORNAMPS,MU,H,ILZ2)=V3B(1:NBORNAMPS,MU,H,ILZ2)
     &       +W3(1:NBORNAMPS,H,IDIR)*CONJG(EPS1(MU,ILZ1))
            DO NU=0,3
              TW(1:NBORNAMPS,MU,NU,H)=TW(1:NBORNAMPS,MU,NU,H)
     &         +W3(1:NBORNAMPS,H,IDIR)*CONJG(EPS1(MU,ILZ1)*EPS2(NU,ILZ2))
            ENDDO
          ENDDO
        ENDDO
        DO ILZ1=1,3
          DO MU=0,3
            VW1(1:NBORNAMPS,MU,H)=VW1(1:NBORNAMPS,MU,H)
     &       +W1(1:NBORNAMPS,H,ILZ1)*CONJG(EPS1(MU,ILZ1))
            VW2(1:NBORNAMPS,MU,H)=VW2(1:NBORNAMPS,MU,H)
     &       +W2(1:NBORNAMPS,H,ILZ1)*CONJG(EPS2(MU,ILZ1))
          ENDDO
        ENDDO
        LIVE(H)=ANY(WEFF(1:NBORNAMPS,H).NE.CZERO)
     &   .OR.ANY(W1(1:NBORNAMPS,H,1:3).NE.CZERO)
     &   .OR.ANY(W2(1:NBORNAMPS,H,1:3).NE.CZERO)
     &   .OR.ANY(W3(1:NBORNAMPS,H,1:9).NE.CZERO)
      ENDDO
      END

      LOGICAL FUNCTION %(proc_prefix)sMP_ONIA_PW_DIRECTION(P,H,IDIR,PD)
C     As ONIA_PW_DIRECTION (two P-wave states), in quadruple precision
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d)
      %(real_mp_format)s P(0:3,NEXTERNAL)
      INTEGER H, IDIR
      TYPE(MP_DUAL) PD(0:3,NEXTERNAL)
      LOGICAL LIVE(NCOMB)
      COMMON/%(proc_prefix)sONIA_LIVE/LIVE
      %(proc_prefix)sMP_ONIA_PW_DIRECTION=LIVE(H)
      IF (.NOT.LIVE(H)) RETURN
      CALL %(proc_prefix)sMP_ONIA_PW2_SET_DUAL(P,IDIR,PD)
      END

      LOGICAL FUNCTION %(proc_prefix)sMP_ONIA_PW_CT_INPUT(H,PASS,AMP,AMPL)
C     As ONIA_PW_CT_INPUT (two P-wave states), in quadruple precision
      USE ALOHA_OBJECT
      USE %(proc_prefix)sPW_DUAL, ONLY: MPPWAMPL
      IMPLICIT NONE
      INTEGER NCOMB, NBORNAMPS, NCTAMPS
      PARAMETER (NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d, NCTAMPS=%(nctamps)d)
      INTEGER H, PASS
      %(complex_mp_format)s AMP(NBORNAMPS), AMPL(3,NCTAMPS), CZERO
      PARAMETER (CZERO=(0.0E0_16,0.0E0_16))
      %(complex_mp_format)s WEFF(NBORNAMPS,NCOMB)
      COMMON/%(proc_prefix)sMP_ONIA_EFFECTIVE/WEFF
      %(complex_mp_format)s W1(NBORNAMPS,NCOMB,3), W2(NBORNAMPS,NCOMB,3)
      %(complex_mp_format)s W3(NBORNAMPS,NCOMB,9)
      %(complex_mp_format)s VW1(NBORNAMPS,0:3,NCOMB), VW2(NBORNAMPS,0:3,NCOMB)
      %(complex_mp_format)s V3A(NBORNAMPS,0:3,NCOMB,3), V3B(NBORNAMPS,0:3,NCOMB,3)
      %(complex_mp_format)s XW(NBORNAMPS,NCOMB), TW(NBORNAMPS,0:3,0:3,NCOMB)
      COMMON/%(proc_prefix)sMP_PW2_WEIGHTS/W1,W2,W3,VW1,VW2,V3A,V3B,XW,TW
      INTEGER PW_IDIR, PW_ILZ1, PW_ILZ2
      %(complex_mp_format)s PW_EPS1(0:3), PW_EPS2(0:3)
      COMMON/%(proc_prefix)sMP_PW2_DIR/PW_EPS1,PW_EPS2,PW_IDIR,PW_ILZ1,PW_ILZ2
      INTEGER I, K, COMP
      %(proc_prefix)sMP_ONIA_PW_CT_INPUT=.FALSE.
      IF (PASS.EQ.1) THEN
        AMP(1:NBORNAMPS)=W3(1:NBORNAMPS,H,PW_IDIR)
        COMP=3
      ELSEIF (PASS.EQ.2) THEN
        IF (PW_ILZ2.NE.1) RETURN
        AMP(1:NBORNAMPS)=W1(1:NBORNAMPS,H,PW_ILZ1)
        COMP=1
      ELSEIF (PASS.EQ.3) THEN
        IF (PW_ILZ1.NE.1) RETURN
        AMP(1:NBORNAMPS)=W2(1:NBORNAMPS,H,PW_ILZ2)
        COMP=2
      ELSE
        IF (PW_IDIR.NE.1) RETURN
        AMP(1:NBORNAMPS)=WEFF(1:NBORNAMPS,H)
        COMP=0
      ENDIF
      %(proc_prefix)sMP_ONIA_PW_CT_INPUT=ANY(AMP(1:NBORNAMPS).NE.CZERO)
      DO I=1,NCTAMPS
        DO K=1,3
          AMPL(K,I)=MPPWAMPL(K,I)%%COMP(COMP)
        ENDDO
      ENDDO
      END

      SUBROUTINE MP_%(proc_prefix)sPW_SPLIT_COEFS2(COEFS,C0,C1,C2,C3)
C     As PW_SPLIT_COEFS2, in quadruple precision
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
      IMPLICIT NONE
      TYPE(MP_DUAL) COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s C0(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s C1(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s C2(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s C3(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      INTEGER I, J, K
      DO K=1,MAXLWFSIZE
        DO J=0,VERTEXMAXCOEFS-1
          DO I=1,MAXLWFSIZE
            C0(I,J,K)=COEFS(I,J,K)%%COMP(0)
            C1(I,J,K)=COEFS(I,J,K)%%COMP(1)
            C2(I,J,K)=COEFS(I,J,K)%%COMP(2)
            C3(I,J,K)=COEFS(I,J,K)%%COMP(3)
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE MP_%(proc_prefix)sPW_CREATE_LOOP_COEFS2(WL,DWL,DWL2,D2WL,RANK,LCUT_SIZE,
     & LOOP_GROUP_NUMBER,SYMFACT,MULTIPLIER,COLOR_ID,HELCONFIG)
C     As PW_CREATE_LOOP_COEFS2, in quadruple precision
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NBORNAMPS, NCOMB, NLOOPGROUPS, NSQUAREDSO
      PARAMETER (NBORNAMPS=%(nbornamps)d, NCOMB=%(ncomb)d)
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d, NSQUAREDSO=%(nSquaredSO)d)
      %(complex_mp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s DWL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s DWL2(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s D2WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER RANK, LCUT_SIZE, LOOP_GROUP_NUMBER, SYMFACT, MULTIPLIER, COLOR_ID, HELCONFIG
      %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/%(proc_prefix)sMP_LCOEFS/LOOPCOEFS
      %(complex_mp_format)s P1(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      %(complex_mp_format)s P2(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      %(complex_mp_format)s PX(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      %(complex_mp_format)s PT(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3,0:3)
      COMMON/%(proc_prefix)sMP_PW2_PCOEFS/P1,P2,PX,PT
      %(complex_mp_format)s WEFF(NBORNAMPS,NCOMB)
      COMMON/%(proc_prefix)sMP_ONIA_EFFECTIVE/WEFF
      %(complex_mp_format)s W1(NBORNAMPS,NCOMB,3), W2(NBORNAMPS,NCOMB,3)
      %(complex_mp_format)s W3(NBORNAMPS,NCOMB,9)
      %(complex_mp_format)s VW1(NBORNAMPS,0:3,NCOMB), VW2(NBORNAMPS,0:3,NCOMB)
      %(complex_mp_format)s V3A(NBORNAMPS,0:3,NCOMB,3), V3B(NBORNAMPS,0:3,NCOMB,3)
      %(complex_mp_format)s XW(NBORNAMPS,NCOMB), TW(NBORNAMPS,0:3,0:3,NCOMB)
      COMMON/%(proc_prefix)sMP_PW2_WEIGHTS/W1,W2,W3,VW1,VW2,V3A,V3B,XW,TW
      INTEGER PW_IDIR, PW_ILZ1, PW_ILZ2
      %(complex_mp_format)s PW_EPS1(0:3), PW_EPS2(0:3)
      COMMON/%(proc_prefix)sMP_PW2_DIR/PW_EPS1,PW_EPS2,PW_IDIR,PW_ILZ1,PW_ILZ2
      INTEGER MU, NU
      CALL MP_%(proc_prefix)sPW_MERGE_LOOP(D2WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,W3(1,HELCONFIG,PW_IDIR),LOOPCOEFS)
      IF (PW_ILZ2.EQ.1) THEN
        CALL MP_%(proc_prefix)sPW_MERGE_LOOP(DWL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,W1(1,HELCONFIG,PW_ILZ1),LOOPCOEFS)
        DO MU=0,3
          CALL MP_%(proc_prefix)sPW_MERGE_LOOP(DWL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &     MULTIPLIER,COLOR_ID,HELCONFIG,V3A(1,MU,HELCONFIG,PW_ILZ1),P2(0,1,1,MU))
        ENDDO
      ENDIF
      IF (PW_ILZ1.EQ.1) THEN
        CALL MP_%(proc_prefix)sPW_MERGE_LOOP(DWL2,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,W2(1,HELCONFIG,PW_ILZ2),LOOPCOEFS)
        DO MU=0,3
          CALL MP_%(proc_prefix)sPW_MERGE_LOOP(DWL2,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &     MULTIPLIER,COLOR_ID,HELCONFIG,V3B(1,MU,HELCONFIG,PW_ILZ2),P1(0,1,1,MU))
        ENDDO
      ENDIF
      IF (PW_IDIR.NE.1) RETURN
      CALL MP_%(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,WEFF(1,HELCONFIG),LOOPCOEFS)
      CALL MP_%(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     & MULTIPLIER,COLOR_ID,HELCONFIG,XW(1,HELCONFIG),PX)
      DO MU=0,3
        CALL MP_%(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,VW1(1,MU,HELCONFIG),P1(0,1,1,MU))
        CALL MP_%(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &   MULTIPLIER,COLOR_ID,HELCONFIG,VW2(1,MU,HELCONFIG),P2(0,1,1,MU))
        DO NU=0,3
          CALL MP_%(proc_prefix)sPW_MERGE_LOOP(WL,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,
     &     MULTIPLIER,COLOR_ID,HELCONFIG,TW(1,MU,NU,HELCONFIG),PT(0,1,1,MU,NU))
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sMP_ONIA_PW_DOUBLED(I_SQSO,I_LOOPGROUP,CTMODE,NLOOPLINE,RANK,PL,M2L,RES,STABLE)
C     As ONIA_PW_DOUBLED (two P-wave states), in quadruple precision
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER NEXTERNAL, NLOOPGROUPS, NSQUAREDSO
      PARAMETER (NEXTERNAL=%(nexternal)d)
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d, NSQUAREDSO=%(nSquaredSO)d)
      INTEGER I_SQSO, I_LOOPGROUP, CTMODE, NLOOPLINE, RANK
      %(real_mp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_mp_format)s RES(3)
      LOGICAL STABLE
      %(complex_mp_format)s P1(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      %(complex_mp_format)s P2(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3)
      %(complex_mp_format)s PX(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      %(complex_mp_format)s PT(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS,0:3,0:3)
      COMMON/%(proc_prefix)sMP_PW2_PCOEFS/P1,P2,PX,PT
      %(complex_mp_format)s MP_PW_DPL(0:3,NEXTERNAL)
      COMMON/%(proc_prefix)sMP_PW_DPL/MP_PW_DPL
      %(complex_mp_format)s MP_PW_DPL2(0:3,NEXTERNAL)
      COMMON/%(proc_prefix)sMP_PW_DPL2/MP_PW_DPL2
      INTEGER PW_IDIR, PW_ILZ1, PW_ILZ2
      %(complex_mp_format)s PW_EPS1(0:3), PW_EPS2(0:3)
      COMMON/%(proc_prefix)sMP_PW2_DIR/PW_EPS1,PW_EPS2,PW_IDIR,PW_ILZ1,PW_ILZ2
      %(real_mp_format)s SHIFT(0:3,NLOOPLINE), PLC(0:3,NLOOPLINE), PREFIX, G(0:3)
      %(real_mp_format)s C1(NLOOPLINE), C2(NLOOPLINE), F
      %(complex_mp_format)s DS1(0:3,NLOOPLINE), DS2(0:3,NLOOPLINE), DP1, DP2
      %(complex_mp_format)s EE, EN1, EN2, PC, CZERO
      PARAMETER (CZERO=(0.0E0_16,0.0E0_16))
      %(mass_dp_format)s M2LC(NLOOPLINE)
      %(complex_mp_format)s QC(0:LOOPMAXCOEFS-1), TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER POWERS(NLOOPLINE), I, J, JJ, K, MU, NU, NC, NC1, NC2, KK, S
      LOGICAL OK
      LOGICAL PW_REFUSED
      COMMON/%(proc_prefix)sPW_REFUSED/PW_REFUSED
      DATA G/1.0E0_16,-1.0E0_16,-1.0E0_16,-1.0E0_16/
      IF (RANK+2.GT.%(maxrank)d) THEN
        WRITE(*,*) 'ERROR:: two P-wave states: rank',RANK+2,'above the maximal rank'
        STOP 1
      ENDIF
      NC=NCOEF_R(RANK)
      NC1=NCOEF_R(RANK+1)
      NC2=NCOEF_R(RANK+2)
      IF (ALL(P1(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.CZERO).AND.
     & ALL(P2(0:NC-1,I_SQSO,I_LOOPGROUP,0:3).EQ.CZERO).AND.
     & ALL(PX(0:NC-1,I_SQSO,I_LOOPGROUP).EQ.CZERO).AND.
     & ALL(PT(0:NC-1,I_SQSO,I_LOOPGROUP,0:3,0:3).EQ.CZERO)) RETURN
      DO I=0,3
        SHIFT(I,1)=0.0E0_16
        DS1(I,1)=CZERO
        DS2(I,1)=CZERO
        DO J=1,NLOOPLINE
          SHIFT(I,1)=SHIFT(I,1)+PL(I,J)
          DS1(I,1)=DS1(I,1)+MP_PW_DPL(I,J)
          DS2(I,1)=DS2(I,1)+MP_PW_DPL2(I,J)
        ENDDO
        PREFIX=0.0E0_16
        DP1=CZERO
        DP2=CZERO
        DO J=2,NLOOPLINE
          PREFIX=PREFIX+PL(I,J-1)
          DP1=DP1+MP_PW_DPL(I,J-1)
          DP2=DP2+MP_PW_DPL2(I,J-1)
          SHIFT(I,J)=PREFIX
          DS1(I,J)=DP1
          DS2(I,J)=DP2
        ENDDO
      ENDDO
      EN1=CZERO
      EN2=CZERO
      DO MU=0,3
        EN1=EN1+G(MU)*PW_EPS1(MU)*CONJG(PW_EPS1(MU))
        EN2=EN2+G(MU)*PW_EPS2(MU)*CONJG(PW_EPS2(MU))
      ENDDO
C     C1_J = dS_J1.eps1*/(eps1.eps1*), C2_J the same for the second state:
C     integers, zero for J=1
      DO J=1,NLOOPLINE
        DO S=1,2
          EE=CZERO
          DO MU=0,3
            IF (S.EQ.1) THEN
              EE=EE+G(MU)*DS1(MU,J)*CONJG(PW_EPS1(MU))
            ELSE
              EE=EE+G(MU)*DS2(MU,J)*CONJG(PW_EPS2(MU))
            ENDIF
          ENDDO
          IF (S.EQ.1) THEN
            EE=EE/EN1
          ELSE
            EE=EE/EN2
          ENDIF
          F=REAL(EE,KIND=16)
          IF (ABS(F).LT.0.5E0_16.AND.ABS(EE).LT.1.0E-8_16) THEN
            F=0.0E0_16
          ELSEIF (ABS(F-NINT(F)).GT.1.0E-8_16.OR.ABS(AIMAG(EE)).GT.1.0E-8_16.OR.J.EQ.1) THEN
            WRITE(*,*) 'ERROR:: P-wave: unexpected derivative of a loop line',J,S
            STABLE=.FALSE.
            RETURN
          ELSE
            F=REAL(NINT(F),KIND=16)
          ENDIF
          IF (S.EQ.1) THEN
            C1(J)=F
          ELSE
            C2(J)=F
          ENDIF
        ENDDO
      ENDDO
C     Line J doubled
      DO J=1,NLOOPLINE
        IF (C1(J).EQ.0.0E0_16.AND.C2(J).EQ.0.0E0_16) CYCLE
        DO K=0,NC1-1
          QC(K)=CZERO
        ENDDO
        DO MU=0,3
          DO K=0,NC-1
            PC=C1(J)*P1(K,I_SQSO,I_LOOPGROUP,MU)+C2(J)*P2(K,I_SQSO,I_LOOPGROUP,MU)
            IF (PC.EQ.CZERO) CYCLE
            QC(K)=QC(K)-2.0E0_16*G(MU)*SHIFT(MU,J)*PC
            KK=COMB_COEF_POS(K,MU+1)
            QC(KK)=QC(KK)+2.0E0_16*G(MU)*PC
          ENDDO
        ENDDO
        IF (C1(J).NE.0.0E0_16.AND.C2(J).NE.0.0E0_16) THEN
          DO K=0,NC-1
            QC(K)=QC(K)-2.0E0_16*C1(J)*C2(J)*PX(K,I_SQSO,I_LOOPGROUP)
          ENDDO
        ENDIF
        IF (ALL(QC(0:NC1-1).EQ.CZERO)) CYCLE
        DO I=1,NLOOPLINE
          POWERS(I)=1
        ENDDO
        POWERS(J)=2
        PLC=PL
        M2LC=M2L
        CALL %(proc_prefix)sMP_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK+1,POWERS,PLC,M2LC,
     &   TIRCOEFS,OK)
        IF (.NOT.OK) THEN
          STABLE=.FALSE.
          PW_REFUSED=.TRUE.
          RETURN
        ENDIF
        DO I=1,3
          DO K=0,NC1-1
            RES(I)=RES(I)+QC(K)*TIRCOEFS(K,I)
          ENDDO
        ENDDO
      ENDDO
C     Line J tripled (JJ=J) and lines J and JJ doubled (JJ /= J)
      DO J=1,NLOOPLINE
        IF (C1(J).EQ.0.0E0_16) CYCLE
        DO JJ=1,NLOOPLINE
          IF (C2(JJ).EQ.0.0E0_16) CYCLE
          IF (J.EQ.JJ) THEN
            F=8.0E0_16*C1(J)*C2(JJ)
          ELSE
            F=4.0E0_16*C1(J)*C2(JJ)
          ENDIF
          DO K=0,NC2-1
            QC(K)=CZERO
          ENDDO
          DO MU=0,3
            DO NU=0,3
              DO K=0,NC-1
                PC=PT(K,I_SQSO,I_LOOPGROUP,MU,NU)
                IF (PC.EQ.CZERO) CYCLE
                PC=F*G(MU)*G(NU)*PC
C               (S_J-l)^mu (S_JJ-l)^nu
                QC(K)=QC(K)+SHIFT(MU,J)*SHIFT(NU,JJ)*PC
                KK=COMB_COEF_POS(K,MU+1)
                QC(KK)=QC(KK)-SHIFT(NU,JJ)*PC
                QC(COMB_COEF_POS(KK,NU+1))=QC(COMB_COEF_POS(KK,NU+1))+PC
                KK=COMB_COEF_POS(K,NU+1)
                QC(KK)=QC(KK)-SHIFT(MU,J)*PC
              ENDDO
            ENDDO
          ENDDO
          IF (ALL(QC(0:NC2-1).EQ.CZERO)) CYCLE
          DO I=1,NLOOPLINE
            POWERS(I)=1
          ENDDO
          IF (J.EQ.JJ) THEN
            POWERS(J)=3
          ELSE
            POWERS(J)=2
            POWERS(JJ)=2
          ENDIF
          PLC=PL
          M2LC=M2L
          CALL %(proc_prefix)sMP_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK+2,POWERS,PLC,M2LC,
     &     TIRCOEFS,OK)
          IF (.NOT.OK) THEN
            STABLE=.FALSE.
            PW_REFUSED=.TRUE.
            RETURN
          ENDIF
          DO I=1,3
            DO K=0,NC2-1
              RES(I)=RES(I)+QC(K)*TIRCOEFS(K,I)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      END
## }
## }

      SUBROUTINE %(proc_prefix)sONIA_SELECT_JZ(K,NJZ,JZS)
C     Restricts the polarisation of bound state K (1 to the number of bound
C     states, in the order of the process) to the NJZ values JZS(1:NJZ) of
C     J_z; NJZ = 0 selects every J_z again (the default). J_z is the helicity
C     of the bound state in the frame of the momenta given to SLOOPMATRIX (the
C     polarisation vectors of VXXXXX for its momentum there): with momenta in
C     the partonic centre-of-mass frame, J_z = +-1 and 0 of a J = 1 state are
C     its transverse and longitudinal polarisations in the helicity frame.
C     The Born, the virtual and the predicted poles are those of the selected
C     J_z, summed over them.
      IMPLICIT NONE
      INTEGER K, NJZ, JZS(*)
      INCLUDE 'onia_loop_data.inc'
      INTEGER MASK(ONIA_NONIA), I
      IF (K.LT.1.OR.K.GT.ONIA_NONIA) THEN
        WRITE(*,*) 'ERROR:: ONIA_SELECT_JZ: no bound state',K
        STOP 1
      ENDIF
      CALL %(proc_prefix)sONIA_JZ_MASK(.FALSE.,MASK)
      MASK(K)=0
      DO I=1,NJZ
        IF (ABS(JZS(I)).GT.ONIA_J(K)) THEN
          WRITE(*,*) 'ERROR:: ONIA_SELECT_JZ: J_z',JZS(I),
     &     'out of range for J =',ONIA_J(K)
          STOP 1
        ENDIF
        MASK(K)=IBSET(MASK(K),JZS(I)+ONIA_J(K))
      ENDDO
      CALL %(proc_prefix)sONIA_JZ_MASK(.TRUE.,MASK)
      END

      SUBROUTINE %(proc_prefix)sONIA_JZ_MASK(SET,MASK)
C     The J_z selected for each bound state, bit J_z+J of MASK(K) (0: all):
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

      SUBROUTINE %(proc_prefix)sONIA_OTHERS_SEL(SET,OSEL)
C     The signature (ONIA_OTHERS) of the helicities of the legs other
C     than the constituents that SLOOPMATRIXHEL selects (0: all), stored
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

      SUBROUTINE %(proc_prefix)sONIA_JZ_TSEL(TSEL,ALLSEL)
C     TSEL(T): whether the J_z combination T of the bound states is
C     selected (ONIA_SELECT_JZ); ALLSEL: every one is
      IMPLICIT NONE
      INCLUDE 'onia_loop_data.inc'
      LOGICAL TSEL(ONIA_NJZT), ALLSEL
      INTEGER MASK(ONIA_NONIA), T, K
      CALL %(proc_prefix)sONIA_JZ_MASK(.FALSE.,MASK)
      ALLSEL=.TRUE.
      DO K=1,ONIA_NONIA
        IF (MASK(K).NE.0) ALLSEL=.FALSE.
      ENDDO
      DO T=1,ONIA_NJZT
        TSEL(T)=.TRUE.
        DO K=1,ONIA_NONIA
          IF (MASK(K).EQ.0) CYCLE
          IF (.NOT.BTEST(MASK(K),ONIA_JZ(K,T)+ONIA_J(K))) TSEL(T)=.FALSE.
        ENDDO
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sONIA_COLOUR_FLOWS(N0,N1,MA,MF,MRE,MIM,NF,V,J)
C     The colour flows J(1:NF) of the Born amplitudes V in a colour
C     basis whose amplitude-to-flow map is entries N0 to N1 of MA
C     (amplitude), MF (flow) and MRE + i MIM (coefficient). The colour
C     matrices of the onia layer are matrices between flows
C     (loop_onia.py, "Colour in the basis of colour flows"):
C     sum_ab conj(U_a) C_ab V_b = ONIA_SANDWICH of the flows of U
C     (Born basis) and V.
      IMPLICIT NONE
      INTEGER N0, N1, NF, MA(*), MF(*)
      %(real_dp_format)s MRE(*), MIM(*)
      %(complex_dp_format)s V(*), J(*)
      INTEGER N
      J(1:NF)=(0D0,0D0)
      DO N=N0,N1
        J(MF(N))=J(MF(N))+DCMPLX(MRE(N),MIM(N))*V(MA(N))
      ENDDO
      END

      SUBROUTINE %(proc_prefix)sONIA_BORN_FLOWS(V,J)
C     The colour flows J(1:ONIA_NFLOW) of the Born amplitudes V in the
C     Born colour basis
      IMPLICIT NONE
      INCLUDE 'onia_loop_data.inc'
      %(complex_dp_format)s V(*), J(*)
      CALL %(proc_prefix)sONIA_COLOUR_FLOWS(ONIA_BF_MPTR(1),ONIA_BF_MPTR(2)-1,
     & ONIA_BF_MA,ONIA_BF_MF,ONIA_BF_MC_RE,ONIA_BF_MC_IM,ONIA_NFLOW,V,J)
      END

      %(complex_dp_format)s FUNCTION %(proc_prefix)sONIA_SANDWICH(NL,JL,NR,JR,LD,XRE,XIM)
C     sum_cd conj(JL(c)) X(c,d) JR(d) over c <= NL, d <= NR, for the
C     flow matrix X = XRE + i XIM of leading dimension LD
      IMPLICIT NONE
      INTEGER NL, NR, LD
      %(complex_dp_format)s JL(*), JR(*)
      %(real_dp_format)s XRE(LD,*), XIM(LD,*)
      %(complex_dp_format)s TMP, RES
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
      %(proc_prefix)sONIA_SANDWICH=RES
      END

      SUBROUTINE %(proc_prefix)sONIA_BORN(P,ANSB)
C     The projected Born matrix element (ANSB(0), and ANSB(1) for its
C     single squared split order) for the constituent momenta P. It also
C     prepares the effective Born.
      IMPLICIT NONE
      INTEGER NEXTERNAL, NCOMB, NBORNAMPS
      PARAMETER (NEXTERNAL=%(nexternal)d, NCOMB=%(ncomb)d, NBORNAMPS=%(nbornamps)d)
      %(real_dp_format)s P(0:3,NEXTERNAL), ANSB(0:*)
      %(complex_dp_format)s WEFF(NBORNAMPS,NCOMB)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      CALL %(proc_prefix)sONIA_PREPARE(P)
      ANSB(0)=BORN2
      ANSB(1)=BORN2
      END

      SUBROUTINE %(proc_prefix)sONIA_IR_POLES(P_PHYS,POLES)
C     The IR poles of the renormalised virtual predicted by the universal
C     structure of the one-loop amplitudes (Catani, Dittmaier, Trocsanyi,
C     hep-ph/0011222; arXiv:2402.19221 eqs. (216)-(217) for S waves): a
C     colour-singlet bound state is a colourless particle, a colour octet a
C     massive adjoint particle of mass m1+m2, with the colour charge T_Q +
C     T_Qbar of its constituents. POLES(1) and POLES(2) are the
C     coefficients of 1/eps and 1/eps^2 in the normalisation of ANS(2:3) of
C     SLOOPMATRIX (alpha_s/(2 pi) (4 pi)^eps/Gamma(1-eps) at the scale MU_R).
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS, NBORNAMPS, NINCOMING
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      PARAMETER (NBORNAMPS=%(nbornamps)d, NINCOMING=%(nincoming)d)
      %(real_dp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), POLES(2)
      INCLUDE 'onia_loop_data.inc'
      INCLUDE 'onia_loop_links.inc'
      INCLUDE 'coupl.inc'
      %(complex_dp_format)s BPROJ(NBORNAMPS,ONIA_NOTHERS,ONIA_NJZT)
      %(real_dp_format)s FACT
      COMMON/%(proc_prefix)sONIA_PROJECTED/BPROJ,FACT
      %(complex_dp_format)s WEFF(NBORNAMPS,%(ncomb)d)
      %(real_dp_format)s BORN2
      COMMON/%(proc_prefix)sONIA_EFFECTIVE/WEFF,BORN2
      %(real_dp_format)s P(0:3,NEXTERNAL), CMASS(NEXTERNAL)
      %(real_dp_format)s PMASS(NEXTERNAL_PHYS), PJ(0:3), PK(0:3)
      %(real_dp_format)s LINKED, PJK, SJK, V, MU2, AO2PI, PI
      %(complex_dp_format)s JFLOW(ONIA_NFLOW,ONIA_NOTHERS,ONIA_NJZT)
      %(complex_dp_format)s %(proc_prefix)sONIA_SANDWICH
      INTEGER L, I, J, O, T, JL, KL
## if(OniaPwave){
      %(real_dp_format)s PSPOLE
## }
      PI=4D0*DATAN(1D0)
      MU2=MU_R**2
      AO2PI=G**2/(8D0*PI**2)
      CALL %(proc_prefix)sONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
      CALL %(proc_prefix)sONIA_PREPARE(P)
      CALL %(proc_prefix)sONIA_MASSES(CMASS)
      PMASS(1:NEXTERNAL_PHYS)=0D0
      DO I=1,NEXTERNAL
        PMASS(ONIA_MAPPING(I))=PMASS(ONIA_MAPPING(I))+CMASS(I)
      ENDDO
      POLES(1)=0D0
      POLES(2)=0D0
C     the colour flows of the projected Born amplitudes
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          CALL %(proc_prefix)sONIA_BORN_FLOWS(BPROJ(1,O,T),JFLOW(1,O,T))
        ENDDO
      ENDDO
C     the colour-linked Borns <T_j.T_k>, j != k
      DO L=1,ONIA_NLINKS
        LINKED=0D0
        DO O=1,ONIA_NOTHERS
          DO T=1,ONIA_NJZT
            LINKED=LINKED+DBLE(%(proc_prefix)sONIA_SANDWICH(ONIA_NFLOW,
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
C     the Gamma_j: gamma_g, gamma_q (massless), C_F (massive quark), C_A
C     (colour-octet bound state)
      DO J=1,NEXTERNAL_PHYS
        POLES(1)=POLES(1)-ONIA_LEG_GAMMA(J)*BORN2
      ENDDO
## if(OniaPwave){
C     the P -> S transitions of the soft current of the P-wave state
      CALL %(proc_prefix)sONIA_PW_SOFT_POLE(P_PHYS,PMASS,CMASS,PSPOLE)
      POLES(1)=POLES(1)+PSPOLE
## }
      POLES(1)=AO2PI*POLES(1)
      POLES(2)=AO2PI*POLES(2)
      END
