## if(tirust_supports_quad_prec){
      SUBROUTINE %(proc_prefix)sMP_TIRUST_LOOP%(qp_mass_suffix)s(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MTIRUST, ONLY: KI_QTIR
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER(NLOOPGROUPS=%(nloop_groups)d,NSQUAREDSO=%(nSquaredSO)d)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK,CTMODE,I,J
      %(real_mp_format)s PL(0:3,NLOOPLINE)
      %(qp_mass_format)s M2L(NLOOPLINE)
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
      CALL %(proc_prefix)sMP_TIRUST_TENSOR_DIAGNOSTIC%(qp_mass_suffix)s(CTMODE,NLOOPLINE,RANK,PL,M2L,COEFS,STABLE)
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

