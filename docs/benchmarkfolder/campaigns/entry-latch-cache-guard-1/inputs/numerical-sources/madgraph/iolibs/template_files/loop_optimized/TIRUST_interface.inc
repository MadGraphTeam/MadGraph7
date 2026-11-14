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
C     integer exponent classification does not quiet sNaN.
      INTEGER,PARAMETER :: TIRUST_INVARIANT_KIND=C_INT64_T
      INTEGER(TIRUST_INVARIANT_KIND),PARAMETER ::
     $ TIRUST_EXPONENT_MASK=INT(Z'7FF0000000000000',TIRUST_INVARIANT_KIND)
C     Restrict this source to the qualified GNU IEEE formats.
      INTEGER,PARAMETER :: TIRUST_FORMAT_ASSERT=1/MERGE(1,0,
     $ STORAGE_SIZE(REAL(0,KIND=KI_TIR)).EQ.64.AND.
     $ STORAGE_SIZE(INT(0,KIND=TIRUST_INVARIANT_KIND)).EQ.64.AND.
     $ BIT_SIZE(INT(0,KIND=TIRUST_INVARIANT_KIND)).EQ.64.AND.
     $ RADIX(REAL(0,KIND=KI_TIR)).EQ.2.AND.
     $ DIGITS(REAL(0,KIND=KI_TIR)).EQ.53.AND.
     $ MAXEXPONENT(REAL(0,KIND=KI_TIR)).EQ.1024.AND.
     $ MINEXPONENT(REAL(0,KIND=KI_TIR)).EQ.-1021)
      INTEGER(TIRUST_INVARIANT_KIND)
     $ TIRUST_INVARIANT_BITS(NLOOPLINE*NLOOPLINE)
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
C     the ABI consumes the upper triangle (I<J).
C     Retain all original arithmetic and each already-equal pair's bits.
C     Classify every entry through integer bits before any FP comparison.
      TIRUST_INVARIANT_BITS=TRANSFER(INVARIANTS,
     $ TIRUST_INVARIANT_BITS,NLOOPLINE*NLOOPLINE)
      IF(ALL(IAND(TIRUST_INVARIANT_BITS,TIRUST_EXPONENT_MASK)
     $ .NE.TIRUST_EXPONENT_MASK)) THEN
        DO J=2,NLOOPLINE
          DO I=1,J-1
            IF(INVARIANTS(J,I).NE.INVARIANTS(I,J)) THEN
              INVARIANTS(J,I)=INVARIANTS(I,J)
            ENDIF
          ENDDO
        ENDDO
      ENDIF
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

## if(tirust_supports_quad_prec){
C     TIRust tensor reduction. The public diagnostic returns MadLoop-ordered
C     moments, marks an unevaluated finite coefficient with NaN, and performs
C     no numerator contraction or counterterm addition.
      SUBROUTINE %(proc_prefix)sMP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      USE MTIRUST
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE,IEEE_VALUE,IEEE_QUIET_NAN
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS
C     integer exponent classification does not quiet sNaN.
      INTEGER,PARAMETER :: TIRUST_INVARIANT_KIND=SELECTED_INT_KIND(33)
      INTEGER(TIRUST_INVARIANT_KIND),PARAMETER ::
     $ TIRUST_EXPONENT_MASK=INT(Z'7FFF0000000000000000000000000000',
     $ TIRUST_INVARIANT_KIND)
C     Restrict this source to the qualified GNU IEEE formats.
      INTEGER,PARAMETER :: TIRUST_FORMAT_ASSERT=1/MERGE(1,0,
     $ STORAGE_SIZE(REAL(0,KIND=KI_QTIR)).EQ.128.AND.
     $ STORAGE_SIZE(INT(0,KIND=TIRUST_INVARIANT_KIND)).EQ.128.AND.
     $ BIT_SIZE(INT(0,KIND=TIRUST_INVARIANT_KIND)).EQ.128.AND.
     $ RADIX(REAL(0,KIND=KI_QTIR)).EQ.2.AND.
     $ DIGITS(REAL(0,KIND=KI_QTIR)).EQ.113.AND.
     $ MAXEXPONENT(REAL(0,KIND=KI_QTIR)).EQ.16384.AND.
     $ MINEXPONENT(REAL(0,KIND=KI_QTIR)).EQ.-16381)
      INTEGER(TIRUST_INVARIANT_KIND)
     $ TIRUST_INVARIANT_BITS(NLOOPLINE*NLOOPLINE)
      %(real_mp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_mp_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS
      %(real_mp_format)s P_TMP(0:3,0:NLOOPLINE-1),ABSP(0:3),REF_P,PREFIX_SUM
      %(real_mp_format)s SHIFTS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      %(real_mp_format)s PDEN_DUMMY(0:3,NLOOPLINE-1),INVARIANTS(NLOOPLINE,NLOOPLINE),SCALE
      %(complex_mp_format)s MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      %(complex_mp_format)s RAW(LOOPMAXCOEFS,4)
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
      MASSES=CMPLX(M2L,KIND=KI_QTIR)
      CALL %(proc_prefix)sMP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)
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
        PREFIX_SUM=REAL(0,KIND=KI_QTIR)
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
      CALL %(proc_prefix)sMP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          INVARIANTS(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J),KIND=KI_QTIR)
        ENDDO
      ENDDO
C     the ABI consumes the upper triangle (I<J).
C     Retain all original arithmetic and each already-equal pair's bits.
C     Classify every entry through integer bits before any FP comparison.
      TIRUST_INVARIANT_BITS=TRANSFER(INVARIANTS,
     $ TIRUST_INVARIANT_BITS,NLOOPLINE*NLOOPLINE)
      IF(ALL(IAND(TIRUST_INVARIANT_BITS,TIRUST_EXPONENT_MASK)
     $ .NE.TIRUST_EXPONENT_MASK)) THEN
        DO J=2,NLOOPLINE
          DO I=1,J-1
            IF(INVARIANTS(J,I).NE.INVARIANTS(I,J)) THEN
              INVARIANTS(J,I)=INVARIANTS(I,J)
            ENDIF
          ENDDO
        ENDDO
      ENDIF
      SCALE=REAL(MU_R,KIND=KI_QTIR)**2
      NC=NCOEF_R(RANK)
      CACHE_ENABLED=TIRustGlobalCache.EQ.-1.OR.NLOOPLINE.LE.TIRustGlobalCache
      CHECK_PASS=CTMODE.EQ.2.OR.CTMODE.EQ.5
      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS,MASSES,SCALE,RAW(1:NC,1:4),STATUS,MAX(0,LoopPoleMode),TIRustComputeFinite,CHECK_PASS,CACHE_ENABLED)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      DO I=0,NC-1
        K=TIRUST_MAP(I)
        DO J=1,4
          IF(J.EQ.4.AND..NOT.TIRustComputeFinite) CYCLE
          IF(.NOT.IEEE_IS_FINITE(REAL(RAW(K,J),KIND=KI_QTIR)).OR..NOT.IEEE_IS_FINITE(AIMAG(RAW(K,J)))) RETURN
        ENDDO
        IF(TIRustComputeFinite) THEN
          TIRCOEFS(I,1)=RAW(K,4)
        ELSE
          TIRCOEFS(I,1)=CMPLX(IEEE_VALUE(REAL(0,KIND=KI_QTIR),IEEE_QUIET_NAN),REAL(0,KIND=KI_QTIR),KIND=KI_QTIR)
        ENDIF
        TIRCOEFS(I,2)=RAW(K,1)+RAW(K,2)
        TIRCOEFS(I,3)=RAW(K,3)
      ENDDO
      STABLE=.TRUE.
      END

## }
      SUBROUTINE %(proc_prefix)sTIRUSTLOOP(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE
      CALL %(proc_prefix)sTIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      END

## if(tirust_supports_quad_prec){
      SUBROUTINE %(proc_prefix)sMP_TIRUST_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MTIRUST, ONLY: KI_QTIR
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER(NLOOPGROUPS=%(nloop_groups)d,NSQUAREDSO=%(nSquaredSO)d)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK,CTMODE,I,J
      %(real_mp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s RES(3)
      %(complex_mp_format)s COEFS(0:LOOPMAXCOEFS-1,3),RESULT(3)
      %(real_dp_format)s LSCALE
      COMMON/%(proc_prefix)sCT/LSCALE,CTMODE
      %(real_mp_format)s NORMALIZATION
      PARAMETER(NORMALIZATION=1.0E0_16/(16.0E0_16*3.141592653589793238462643383279502884E0_16**2))
      LOGICAL STABLE,FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
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
      RES=0
      CALL %(proc_prefix)sMP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,PL,M2L,COEFS,STABLE)
      IF(.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RETURN
      ENDIF
      RESULT=0
      DO I=1,3
        DO J=0,NCOEF_R(RANK)-1
## if(not AmplitudeReduction){
          RESULT(I)=RESULT(I)+LOOPCOEFS(J,SQSOINDEX,ID)*COEFS(J,I)
## }else{
          RESULT(I)=RESULT(I)+LOOPCOEFS(J,ID)*COEFS(J,I)
## }
        ENDDO
## if(not AmplitudeReduction){
        RESULT(I)=2.0E0_16*NORMALIZATION*REAL(RESULT(I),KIND=KI_QTIR)
## }else{
        RESULT(I)=NORMALIZATION*RESULT(I)
## }
        RES(I)=CMPLX(RESULT(I),KIND=KIND(0D0))
        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I))).AND.IEEE_IS_FINITE(AIMAG(RES(I)))
      ENDDO
      IF(.NOT.STABLE) THEN
        RES=0
        FPE_IN_QP_REDUCTION=.TRUE.
      ENDIF
      END
## }

      SUBROUTINE %(proc_prefix)sINITIALIZE_TIRUST(STATUS)
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
      USE MTIRUST, ONLY: TIRUST_CLEAR_CACHE,
     $                  TIRUST_CONTEXT_GENERATION
C     The shared module context may belong to another subprocess.
C     Event clearing must not create it before any TIRust use.
      IF(TIRUST_CONTEXT_GENERATION().GE.0) THEN
        CALL TIRUST_CLEAR_CACHE()
      ENDIF
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
