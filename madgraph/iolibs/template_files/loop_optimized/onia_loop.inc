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
      IMPLICIT NONE
      INTEGER NEXTERNAL, NEXTERNAL_PHYS
      PARAMETER (NEXTERNAL=%(nexternal)d, NEXTERNAL_PHYS=%(onia_nexternal_phys)d)
      %(real_dp_format)s P_PHYS(0:3,NEXTERNAL_PHYS), ANS(0:3,*)
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
      INCLUDE 'onia_loop_data.inc'
      IF (USERHEL.NE.-1) THEN
        WRITE(*,*) 'ERROR:: MadLoop sums over the polarisations of the',
     &   ' bound states: the selection of a helicity configuration',
     &   ' (SLOOPMATRIXHEL) is not supported for this process.'
        STOP 1
      ENDIF
      CALL %(proc_prefix)sONIA_CONSTITUENT_MOMENTA(P_PHYS,P)
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
C     The projected Born matrix element
      BORN2=0D0
      DO O=1,ONIA_NOTHERS
        DO T=1,ONIA_NJZT
          DO I=1,NBORNAMPS
            IF (BPROJ(I,O,T).EQ.(0D0,0D0)) CYCLE
            TMP=(0D0,0D0)
            DO J=1,NBORNAMPS
              TMP=TMP+DCMPLX(ONIA_CFB_RE(I,J),ONIA_CFB_IM(I,J))*BPROJ(J,O,T)
            ENDDO
            BORN2=BORN2+DBLE(DCONJG(BPROJ(I,O,T))*TMP)
          ENDDO
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
      %(complex_dp_format)s TMP
      INTEGER L, I, J, O, T, JL, KL
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
C     the colour-linked Borns <T_j.T_k>, j != k
      DO L=1,ONIA_NLINKS
        LINKED=0D0
        DO O=1,ONIA_NOTHERS
          DO T=1,ONIA_NJZT
            DO I=1,NBORNAMPS
              IF (BPROJ(I,O,T).EQ.(0D0,0D0)) CYCLE
              TMP=(0D0,0D0)
              DO J=1,NBORNAMPS
                TMP=TMP+DCMPLX(ONIA_CL_RE(I,J,L),ONIA_CL_IM(I,J,L))*BPROJ(J,O,T)
              ENDDO
              LINKED=LINKED+DBLE(DCONJG(BPROJ(I,O,T))*TMP)
            ENDDO
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
      POLES(1)=AO2PI*POLES(1)
      POLES(2)=AO2PI*POLES(2)
      END
