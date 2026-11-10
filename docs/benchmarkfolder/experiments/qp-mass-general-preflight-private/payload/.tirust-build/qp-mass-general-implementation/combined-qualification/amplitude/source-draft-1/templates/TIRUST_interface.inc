## if(tirust_available){
C     TIRust tensor reduction. The public diagnostic returns MadLoop-ordered
C     moments, marks an unevaluated finite coefficient with NaN, and performs
C     no numerator contraction or counterterm addition.
      SUBROUTINE %(proc_prefix)sTIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      USE MTIRUST
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE,IEEE_VALUE,IEEE_QUIET_NAN
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS
      %(real_dp_format)s P_TMP(0:3,0:NLOOPLINE-1),ABSP(0:3),REF_P,PREFIX_SUM
      %(real_dp_format)s SHIFTS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      %(real_dp_format)s PDEN_DUMMY(0:3,NLOOPLINE-1),INVARIANTS(NLOOPLINE,NLOOPLINE),SCALE
      %(complex_dp_format)s MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      %(complex_dp_format)s RAW(LOOPMAXCOEFS,4)
      INTEGER TIRUST_MAP(0:LOOPMAXCOEFS-1)
      %(tirust_coefmap)s
      %(include_vector)s
      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      STABLE=.FALSE.
      TIRCOEFS=0
      IF(RANK.LT.0.OR.RANK.GT.%(maxrank)d) RETURN
      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRustComputeFinite)) RETURN
      CALL %(proc_prefix)sINITIALIZE_TIRUST(STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      CALL TIRUST_SET_CACHE_BUDGET(INT(TIRustCacheMaxBytes,C_INT64_T),STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_TIR)
      CALL %(proc_prefix)sSWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)
      MASSES=M2L
C     Preserve the summation and threshold conventions of the Ninja adapter.
      P_TMP=0
      ABSP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP(I)=ABSP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP)
      IF(REF_P.GT.1D-8) THEN
        DO I=0,3
          IF(ABS(P_TMP(I,0))/MAX(REF_P*1D-6,ABSP(I)).GT.1D-6) RETURN
        ENDDO
      ENDIF
C     Accumulate ordered prefixes once, preserving the zero-seeded sum.
      DO I=0,3
        PREFIX_SUM=REAL(0,KIND=KI_TIR)
        DO J=1,NLOOPLINE-1
          PREFIX_SUM=PREFIX_SUM+PL(I,J)
          P_TMP(I,J)=PREFIX_SUM
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          SHIFTS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=SHIFTS(I,J)
        ENDDO
      ENDDO
      CALL %(proc_prefix)sBUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          INVARIANTS(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J),KIND=KI_TIR)
        ENDDO
      ENDDO
      SCALE=MU_R**2
      NC=NCOEF_R(RANK)
      CACHE_ENABLED=TIRustGlobalCache.EQ.-1.OR.NLOOPLINE.LE.TIRustGlobalCache
      CHECK_PASS=CTMODE.EQ.2.OR.CTMODE.EQ.5
      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS,MASSES,SCALE,RAW(1:NC,1:4),STATUS,MAX(0,LoopPoleMode),TIRustComputeFinite,CHECK_PASS,CACHE_ENABLED)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      DO I=0,NC-1
        K=TIRUST_MAP(I)
        DO J=1,4
          IF(J.EQ.4.AND..NOT.TIRustComputeFinite) CYCLE
          IF(.NOT.IEEE_IS_FINITE(REAL(RAW(K,J),KIND=KI_TIR)).OR..NOT.IEEE_IS_FINITE(AIMAG(RAW(K,J)))) RETURN
        ENDDO
        IF(TIRustComputeFinite) THEN
          TIRCOEFS(I,1)=RAW(K,4)
        ELSE
          TIRCOEFS(I,1)=CMPLX(IEEE_VALUE(REAL(0,KIND=KI_TIR),IEEE_QUIET_NAN),REAL(0,KIND=KI_TIR),KIND=KI_TIR)
        ENDIF
        TIRCOEFS(I,2)=RAW(K,1)+RAW(K,2)
        TIRCOEFS(I,3)=RAW(K,3)
      ENDDO
      STABLE=.TRUE.
      END

%(tirust_qp_tensor_interfaces)s      SUBROUTINE %(proc_prefix)sTIRUSTLOOP(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE
      CALL %(proc_prefix)sTIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      END

%(tirust_qp_loop_interfaces)s      SUBROUTINE %(proc_prefix)sINITIALIZE_TIRUST(STATUS)
      USE MTIRUST
      IMPLICIT NONE
      INTEGER STATUS
      INTEGER LAST_GENERATION
      COMMON/%(proc_prefix)sTIRUST_INIT/LAST_GENERATION
      STATUS=TIRUST_SUCCESS
      IF(LAST_GENERATION.EQ.TIRUST_CONTEXT_GENERATION()) RETURN
      %(tirust_plan_registration)s
      LAST_GENERATION=TIRUST_CONTEXT_GENERATION()
      END

      BLOCK DATA %(proc_prefix)sTIRUST_INIT_DATA
      IMPLICIT NONE
      INTEGER LAST_GENERATION
      COMMON/%(proc_prefix)sTIRUST_INIT/LAST_GENERATION
      DATA LAST_GENERATION/-2/
      END

      SUBROUTINE %(proc_prefix)sCLEAR_TIRUST_CACHE()
      USE MTIRUST, ONLY: TIRUST_CLEAR_CACHE
      CALL TIRUST_CLEAR_CACHE()
      END

      SUBROUTINE %(proc_prefix)sFINALIZE_TIRUST()
      USE MTIRUST, ONLY: TIRUST_FINALIZE
      IMPLICIT NONE
      INTEGER LAST_GENERATION
      COMMON/%(proc_prefix)sTIRUST_INIT/LAST_GENERATION
      CALL TIRUST_FINALIZE()
      LAST_GENERATION=-2
      END
## }
