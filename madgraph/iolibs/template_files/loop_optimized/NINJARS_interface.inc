## if(ninjars_available){
C     NinjaRS tensor reduction using oneloopmaster scalar integrals.
      SUBROUTINE %(proc_prefix)sNINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d,NSQUAREDSO=%(nSquaredSO)d)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s RES(3)
      LOGICAL STABLE
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION=1.D0/(16.D0*3.14159265358979323846D0**2))
      %(real_dp_format)s P_TMP(0:3,0:NLOOPLINE-1),ABSP_TMP(0:3),REF_P
      %(real_dp_format)s P_NRS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      %(real_dp_format)s PDEN_DUMMY(0:3,NLOOPLINE-1),REAL_S_MAT(NLOOPLINE,NLOOPLINE)
      %(real_dp_format)s NRS_SCALE
      %(complex_dp_format)s MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      %(complex_dp_format)s TENSORCOEFS(0:LOOPMAXCOEFS-1),NRS_RES(0:2),R1
      INTEGER I,J,K,CURR_MAXCOEF,NRS_STATUS
      %(include_vector)s
      INCLUDE 'coupl.inc'
      %(real_dp_format)s LSCALE
      INTEGER CTMODE
      COMMON/%(proc_prefix)sCT/LSCALE,CTMODE
## if(not AmplitudeReduction){
      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R
      %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## }else{
      INTEGER ID,R
      COMMON/%(proc_prefix)sLOOP/ID,R
      %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
      COMMON/%(proc_prefix)sLCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_NRS)
C     Preserve MadLoop's loop-direction stability test and mass ordering.
      CALL %(proc_prefix)sSWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)
      MASSES=M2L
C     Match the existing Ninja loop momentum convention, including its
C     accumulation order and the residual momentum on the first line.
      P_TMP=0
      ABSP_TMP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I)=ABSP_TMP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP_TMP)
      IF (REF_P.GT.1.D-8) THEN
        DO I=0,3
          IF (ABS(P_TMP(I,0))/MAX(REF_P*1.D-6,ABSP_TMP(I)).GT.1.D-6) THEN
            FPE_IN_DP_REDUCTION=.TRUE.
            RETURN
          ENDIF
        ENDDO
      ENDIF
      DO I=0,3
        DO J=1,NLOOPLINE-1
          DO K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          P_NRS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=P_NRS(I,J)
        ENDDO
      ENDDO
C     Use MadLoop's thresholded invariants, including complex on-shell masses.
      CALL %(proc_prefix)sBUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          REAL_S_MAT(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J),KIND=KI_NRS)
        ENDDO
      ENDDO
      CURR_MAXCOEF=NCOEF_R(RANK)
      DO I=0,CURR_MAXCOEF-1
## if(not AmplitudeReduction){
        TENSORCOEFS(I)=LOOPCOEFS(I,SQSOINDEX,ID)
## }else{
        TENSORCOEFS(I)=LOOPCOEFS(I,ID)
## }
C       q_loop -> -q_loop in NinjaRS's denominator convention.
        IF (MOD(COEFTORANK_MAP(I),2).EQ.1) TENSORCOEFS(I)=-TENSORCOEFS(I)
      ENDDO
      NRS_SCALE=REAL(MU_R**2,KIND=KI_NRS)
      CALL NINJARS_TENSOR_EVALUATE(TENSORCOEFS(0:CURR_MAXCOEF-1),NLOOPLINE,RANK,REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,NRS_RES,R1,NRS_STATUS)
      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS
      DO I=0,2
        STABLE=STABLE.AND.IEEE_IS_FINITE(REAL(NRS_RES(I),KIND=KI_NRS))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(NRS_RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_DP_REDUCTION=.TRUE.
        RETURN
      ENDIF
      DO I=0,2
## if(AmplitudeReduction){
        RES(I+1)=NORMALIZATION*CMPLX(NRS_RES(I),KIND=KI_NRS)
## }else{
        RES(I+1)=NORMALIZATION*2.D0*DBLE(NRS_RES(I))
## }
      ENDDO
      DO I=1,3
        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I)))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_DP_REDUCTION=.TRUE.
        RES=(0.D0,0.D0)
      ENDIF
      END
## if(ninjars_supports_quad_prec){
      SUBROUTINE %(proc_prefix)sMP_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d,NSQUAREDSO=%(nSquaredSO)d)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK
      %(real_mp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s RES(3)
      LOGICAL STABLE
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION=1.D0/(16.D0*3.14159265358979323846D0**2))
      %(real_mp_format)s P_TMP(0:3,0:NLOOPLINE-1),ABSP_TMP(0:3),REF_P
      %(real_mp_format)s P_NRS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      %(real_mp_format)s PDEN_DUMMY(0:3,NLOOPLINE-1),REAL_S_MAT(NLOOPLINE,NLOOPLINE)
      %(real_mp_format)s NRS_SCALE
      %(complex_mp_format)s MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      %(complex_mp_format)s TENSORCOEFS(0:LOOPMAXCOEFS-1),NRS_RES(0:2),R1
      INTEGER I,J,K,CURR_MAXCOEF,NRS_STATUS
      %(include_vector)s
      INCLUDE 'coupl.inc'
      %(real_dp_format)s LSCALE
      INTEGER CTMODE
      COMMON/%(proc_prefix)sCT/LSCALE,CTMODE
## if(not AmplitudeReduction){
      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R
      %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## }else{
      INTEGER ID,R
      COMMON/%(proc_prefix)sLOOP/ID,R
      %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
      COMMON/%(proc_prefix)sMP_LCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QNRS)
C     Preserve MadLoop's loop-direction stability test and mass ordering.
      CALL %(proc_prefix)sMP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)

C     Match the existing Ninja loop momentum convention, including its
C     accumulation order and the residual momentum on the first line.
      P_TMP=0
      ABSP_TMP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I)=ABSP_TMP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP_TMP)
      IF (REF_P.GT.1.D-8) THEN
        DO I=0,3
          IF (ABS(P_TMP(I,0))/MAX(REF_P*1.D-6,ABSP_TMP(I)).GT.1.D-6) THEN
            FPE_IN_QP_REDUCTION=.TRUE.
            RETURN
          ENDIF
        ENDDO
      ENDIF
      DO I=0,3
        DO J=1,NLOOPLINE-1
          DO K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          P_NRS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=P_NRS(I,J)
        ENDDO
      ENDDO
C     Use MadLoop's thresholded invariants, including complex on-shell masses.
      CALL %(proc_prefix)sMP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          REAL_S_MAT(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J),KIND=KI_QNRS)
        ENDDO
      ENDDO
      CURR_MAXCOEF=NCOEF_R(RANK)
      DO I=0,CURR_MAXCOEF-1
## if(not AmplitudeReduction){
        TENSORCOEFS(I)=LOOPCOEFS(I,SQSOINDEX,ID)
## }else{
        TENSORCOEFS(I)=LOOPCOEFS(I,ID)
## }
C       q_loop -> -q_loop in NinjaRS's denominator convention.
        IF (MOD(COEFTORANK_MAP(I),2).EQ.1) TENSORCOEFS(I)=-TENSORCOEFS(I)
      ENDDO
      NRS_SCALE=REAL(MU_R**2,KIND=KI_QNRS)
      CALL NINJARS_TENSOR_EVALUATE(TENSORCOEFS(0:CURR_MAXCOEF-1),NLOOPLINE,RANK,REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,NRS_RES,R1,NRS_STATUS)
      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS
      DO I=0,2
        STABLE=STABLE.AND.IEEE_IS_FINITE(REAL(NRS_RES(I),KIND=KI_QNRS))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(NRS_RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RETURN
      ENDIF
      DO I=0,2
## if(AmplitudeReduction){
        RES(I+1)=NORMALIZATION*CMPLX(NRS_RES(I),KIND=KI_NRS)
## }else{
        RES(I+1)=NORMALIZATION*2.D0*DBLE(NRS_RES(I))
## }
      ENDDO
      DO I=1,3
        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I)))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RES=(0.D0,0.D0)
      ENDIF
      END
## }
## }
