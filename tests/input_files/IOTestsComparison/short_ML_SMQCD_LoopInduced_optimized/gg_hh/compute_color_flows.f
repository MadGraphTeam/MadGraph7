C     Set of subroutines to project the Feynman diagrama amplitudes
C      AMPL onto the color flow space.

      SUBROUTINE ML5_0_COMPUTE_COLOR_FLOWS(HEL_MULT,DO_CUMULATIVE)
      IMPLICIT NONE
      LOGICAL DO_CUMULATIVE
      INTEGER HEL_MULT
      CALL ML5_0_REINITIALIZE_JAMPS()
      CALL ML5_0_DO_COMPUTE_COLOR_FLOWS(.FALSE.)
      END SUBROUTINE

      SUBROUTINE ML5_0_COMPUTE_COLOR_FLOWS_DERIVED_QUANTITIES(HEL_MULT)
      IMPLICIT NONE
      INTEGER HEL_MULT
      CONTINUE
      END SUBROUTINE

      SUBROUTINE ML5_0_DEALLOCATE_COLOR_FLOWS()
      IMPLICIT NONE
      CALL ML5_0_DO_COMPUTE_COLOR_FLOWS(.TRUE.)
      END SUBROUTINE

      SUBROUTINE ML5_0_DO_COMPUTE_COLOR_FLOWS(CLEANUP)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      CHARACTER*512 PROC_PREFIX
      PARAMETER ( PROC_PREFIX='ML5_0_')
      CHARACTER*512 LOOPCOLORFLOWCOEFSNAME
      PARAMETER ( LOOPCOLORFLOWCOEFSNAME='LoopColorFlowCoefs.dat')
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0.0D0,1.0D0))

      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=20)
      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=0, NLOOPAMPSO=0)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=1)
C     
C     LOCAL VARIABLES 
C     
C     When this subroutine is called with CLEANUP=True, it deallocates
C      all its
C     arrays
      LOGICAL CLEANUP
C     Storing concatenated filenames
      CHARACTER*512 TMP
      CHARACTER*512 LOOPCOLORFLOWCOEFSN

      INTEGER I, J, K, SOINDEX, ARRAY_SIZE
      COMPLEX*16 PROJ_COEF


C     
C     FUNCTIONS
C     
      INTEGER ML5_0_ML5SOINDEX_FOR_BORN_AMP
      INTEGER ML5_0_ML5SOINDEX_FOR_LOOP_AMP
C     
C     GLOBAL VARIABLES
C     
      CHARACTER(512) MLPATH
      COMMON/MLPATH/MLPATH

      COMPLEX*16 AMPL(3,NLOOPAMPS)
      COMMON/ML5_0_AMPL/AMPL
      COMPLEX*16 JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      COMMON/ML5_0_JAMPL/JAMPL


C     Now a more advanced data structure for storing the projection
C      coefficient
C     This is of course not F77 standard but widely supported by now.
      TYPE PROJCOEFFS
        INTEGER, DIMENSION(:), ALLOCATABLE :: NUM
        INTEGER, DIMENSION(:), ALLOCATABLE :: DENOM
        INTEGER, DIMENSION(:), ALLOCATABLE :: AMPID
      ENDTYPE PROJCOEFFS
      TYPE(PROJCOEFFS), DIMENSION(NLOOPFLOWS), SAVE ::
     $  LOOPCOLORPROJECTOR

C     ----------
C     BEGIN CODE
C     ----------

C     CleanUp duties, i.e. array deallocation
      IF (CLEANUP) THEN
        DO I=1,NLOOPFLOWS
          IF(ALLOCATED(LOOPCOLORPROJECTOR(I)%NUM))
     $      DEALLOCATE(LOOPCOLORPROJECTOR(I)%NUM)
          IF(ALLOCATED(LOOPCOLORPROJECTOR(I)%DENOM))
     $      DEALLOCATE(LOOPCOLORPROJECTOR(I)%DENOM)
          IF(ALLOCATED(LOOPCOLORPROJECTOR(I)%AMPID))
     $      DEALLOCATE(LOOPCOLORPROJECTOR(I)%AMPID)
        ENDDO
        RETURN
      ENDIF

C     Initialization; must allocate array from data files. All arrays
C      are allocated at once, so we only need to check if
C      LoopColorProjector(0)%Num is allocated and all other arrays
C      must share the same status.
      IF(.NOT.ALLOCATED(LOOPCOLORPROJECTOR(1)%NUM)) THEN
        CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
        CALL JOINPATH(TMP,LOOPCOLORFLOWCOEFSNAME,LOOPCOLORFLOWCOEFSN)

C       Initialize the LoopColorProjector
        OPEN(1, FILE=LOOPCOLORFLOWCOEFSN, ERR=201, STATUS='OLD',      
     $        ACTION='READ')
        DO I=1,NLOOPFLOWS
          READ(1,*,END=998) ARRAY_SIZE
          ALLOCATE(LOOPCOLORPROJECTOR(I)%NUM(ARRAY_SIZE))
          ALLOCATE(LOOPCOLORPROJECTOR(I)%DENOM(ARRAY_SIZE))
          ALLOCATE(LOOPCOLORPROJECTOR(I)%AMPID(ARRAY_SIZE))
          READ(1,*,END=998) (LOOPCOLORPROJECTOR(I)%NUM(J),J=1
     $     ,ARRAY_SIZE)
          READ(1,*,END=998) (LOOPCOLORPROJECTOR(I)%DENOM(J),J=1
     $     ,ARRAY_SIZE)
          READ(1,*,END=998) (LOOPCOLORPROJECTOR(I)%AMPID(J),J=1
     $     ,ARRAY_SIZE)
        ENDDO
        GOTO 203
 201    CONTINUE
        STOP 'Color projection coefficients could not be initialized'
     $   //' from file ML5_0_LoopColorFlowCoefs.dat.'
 203    CONTINUE
        CLOSE(1)

        GOTO 999
 998    CONTINUE
        STOP 'End of file reached. Should not have happened.'
 999    CONTINUE

      ENDIF

C     Here we calculate JAMPB
C     Projection of the loop amplitudes
      DO I=1,NLOOPFLOWS
        DO J=1,SIZE(LOOPCOLORPROJECTOR(I)%AMPID)
          SOINDEX = ML5_0_ML5SOINDEX_FOR_LOOP_AMP(LOOPCOLORPROJECTOR(I)
     $     %AMPID(J))
          PROJ_COEF=DCMPLX(LOOPCOLORPROJECTOR(I)%NUM(J)
     $     /DBLE(ABS(LOOPCOLORPROJECTOR(I)%DENOM(J))),0.0D0)
          IF(LOOPCOLORPROJECTOR(I)%DENOM(J).LT.0) PROJ_COEF=PROJ_COEF
     $     *IMAG1
          DO K=1,3
            JAMPL(K,I,SOINDEX) = JAMPL(K,I,SOINDEX) + PROJ_COEF*AMPL(K
     $       ,LOOPCOLORPROJECTOR(I)%AMPID(J))
          ENDDO
        ENDDO
      ENDDO

      END SUBROUTINE

C     This subroutine initializes the color flow matrix.
      SUBROUTINE ML5_0_INITIALIZE_FLOW_COLORMATRIX()
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      CHARACTER*512 PROC_PREFIX
      PARAMETER ( PROC_PREFIX='ML5_0_')
      CHARACTER*512 LOOPCOLORFLOWMATRIXNAME
      PARAMETER ( LOOPCOLORFLOWMATRIXNAME='LoopColorFlowMatrix.dat')

      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=1)
C     
C     LOCAL VARIABLES 
C     
C     Storing concatenated filenames
      CHARACTER*512 TMP
      CHARACTER*512 LOOPCOLORFLOWMATRIXN
      INTEGER I, J
C     
C     GLOBAL VARIABLES
C     
      CHARACTER(512) MLPATH
      COMMON/MLPATH/MLPATH

C     Now a more advanced data structure for storing the projection
C      coefficient
C     This is of course not F77 standard but widely supported by now.
      TYPE COLORCOEFF
        SEQUENCE
        INTEGER :: NUM
        INTEGER :: DENOM
      ENDTYPE COLORCOEFF
      TYPE(COLORCOEFF), DIMENSION(NLOOPFLOWS,NLOOPFLOWS) ::
     $  LOOPCOLORFLOWMATRIX
      LOGICAL CMINITIALIZED
      DATA CMINITIALIZED/.FALSE./
      COMMON/ML5_0_FLOW_COLOR_MATRIX/LOOPCOLORFLOWMATRIX, CMINITIALIZED

C     ----------
C     BEGIN CODE
C     ----------

C     Initialization
      IF(.NOT.CMINITIALIZED) THEN
        CMINITIALIZED = .TRUE.
        CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
        CALL JOINPATH(TMP,LOOPCOLORFLOWMATRIXNAME,LOOPCOLORFLOWMATRIXN)
C       Initialize the LoopColorFlowMatrix
        OPEN(1, FILE=LOOPCOLORFLOWMATRIXN, ERR=701, STATUS='OLD',     
     $         ACTION='READ')
        DO I=1,NLOOPFLOWS
          READ(1,*,END=898) (LOOPCOLORFLOWMATRIX(I,J)%NUM,J=1
     $     ,NLOOPFLOWS)
          READ(1,*,END=898) (LOOPCOLORFLOWMATRIX(I,J)%DENOM,J=1
     $     ,NLOOPFLOWS)
        ENDDO
        GOTO 703
 701    CONTINUE
        STOP 'Color factors could not be initialized from file'
     $   //' ML5_0_BornColorFlowMatrix.dat.'
 703    CONTINUE
        CLOSE(1)

        GOTO 899
 898    CONTINUE
        STOP 'End of file reached. Should not have happened.'
 899    CONTINUE

      ENDIF
      END SUBROUTINE


C     This routine is used as a crosscheck only to make sure the loop
C      ME computed
C     from the JAMP is equal to the amplitude computed directly. It is
C      a consistency
C     check of the color projection and computation.

      SUBROUTINE ML5_0_COMPUTE_RES_FROM_JAMP(RES,HEL_MULT)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=0, NLOOPAMPSO=0)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=1)
C     
C     ARGUMENT 
C     
      DOUBLE COMPLEX RES_CPLX(0:3,0:NSQUAREDSO)
      REAL*8 RES(0:3,0:NSQUAREDSO)
C     HEL_MULT is the helicity multiplier which can be more than one
C     if several helicity configuration are mapped onto the one being
C     currently computed.
      INTEGER HEL_MULT
      DOUBLE COMPLEX JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      COMMON/ML5_0_JAMPL/JAMPL
C     
C     LOCAL VARIABLES 
C     
      INTEGER I, J
C     ----------
C     BEGIN CODE
C     ----------
      CALL ML5_0_DO_COMPUTE_INTER_JAMP(RES_CPLX, HEL_MULT, JAMPL,
     $  JAMPL)
      DO I=0,3
        DO J=0, NSQUAREDSO
          RES(I,J) = DREAL(RES_CPLX(I,J))  !RES is real for matrix element computation and complex for density matrix computation. A conversion is necessary
        ENDDO
      ENDDO
      END SUBROUTINE





      INTEGER FUNCTION ML5_0_GET_HEL_INDEX(NHEL)
C     returned th 
C     
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER NCOMB
      PARAMETER (NCOMB=4)
      LOGICAL FOUND
      INTEGER I
      INTEGER NHEL(NEXTERNAL)
      INTEGER HELC(NEXTERNAL,NCOMB)
      COMMON/ML5_0_HELCONFIGS/HELC
C     
C     START CODE
C     
      DO ML5_0_GET_HEL_INDEX=1,NCOMB
        DO I=1,NEXTERNAL
          IF (HELC(I, ML5_0_GET_HEL_INDEX).NE.NHEL(I))THEN
            EXIT
          ELSE IF (I.EQ.NEXTERNAL) THEN
            RETURN
          ENDIF
        ENDDO
      ENDDO
      END




      SUBROUTINE ML5_0_DO_COMPUTE_INTER_JAMP(INTER,HEL_MULT,JAMPL1
     $ ,JAMPL2)
C     Computation of the interference between JAMPL1 and JAMPL2
C     This subroutine is used to compute the matrix element JAMPL1 =
C      JAMPL2 or the density matrix JAMPL1 != JAMPL2
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0.0D0,1.0D0))

      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=0, NLOOPAMPSO=0, NSO=0)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=1)
C     
C     ARGUMENT 
C     
      COMPLEX*16 INTER(0:3,0:NSQUAREDSO)
C     HEL_MULT is the helicity multiplier which can be more than one
C     if several helicity configuration are mapped onto the one being
C     currently computed.
      INTEGER HEL_MULT
C     
C     LOCAL VARIABLES 
C     
      INTEGER I, J, K, M, N, ISQSO, DOUBLEFACT
      INTEGER LOWERBOUND
      INTEGER ORDERS_A(NSO), ORDERS_B(NSO)
      COMPLEX*16 COLOR_COEF
      COMPLEX*16 TEMP(3)
C     
C     FUNCTIONS
C     
C     This function belongs to the loop ME computation (loop_matrix.f)
C      and is prefixed with ML5 
      INTEGER ML5_0_ML5SQSOINDEX
C     This function belongs to the Born ME computation (born_matrix.f)
C      and is not prefixed at all
      INTEGER ML5_0_SQSOINDEX, ML5_0_SOINDEX_FOR_AMPORDERS
C     
C     GLOBAL VARIABLES
C     
      CHARACTER(512) MLPATH
      COMMON/MLPATH/MLPATH
      INTEGER SQSO_TARGET
      COMMON/ML5_0_SOCHOICE/SQSO_TARGET


      COMPLEX*16 JAMPL1(3,NLOOPFLOWS,NLOOPAMPSO)
      COMPLEX*16 JAMPL2(3,NLOOPFLOWS,NLOOPAMPSO)


      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
     $ ,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
     $ ,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/ML5_0_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE
     $ ,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE
     $ ,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

C     Now a more advanced data structure for storing the projection
C      coefficient
C     This is of course not F77 standard but widely supported by now.
      TYPE COLORCOEFF
        SEQUENCE
        INTEGER :: NUM
        INTEGER :: DENOM
      ENDTYPE COLORCOEFF
      TYPE(COLORCOEFF), DIMENSION(NLOOPFLOWS,NLOOPFLOWS) ::
     $  LOOPCOLORFLOWMATRIX
      LOGICAL CMINITIALIZED
      COMMON/ML5_0_FLOW_COLOR_MATRIX/LOOPCOLORFLOWMATRIX, CMINITIALIZED

C     ----------
C     BEGIN CODE
C     ----------

C     Initialization
      IF(.NOT.CMINITIALIZED) THEN
        CALL ML5_0_INITIALIZE_FLOW_COLORMATRIX()
      ENDIF

      DO I=0,NSQUAREDSO
        DO K=0,3
          INTER(K,I)=0.0D0
        ENDDO
      ENDDO


C     Compute the Loop ME from the loop color flow amplitudes (JAMPL)
      DO I=1,NLOOPFLOWS
        DO J=I,NLOOPFLOWS
          COLOR_COEF=DCMPLX(LOOPCOLORFLOWMATRIX(I,J)%NUM
     $     /DBLE(ABS(LOOPCOLORFLOWMATRIX(I,J)%DENOM)),0.0D0)
          IF (LOOPCOLORFLOWMATRIX(I,J)%DENOM.LT.0)
     $      COLOR_COEF=COLOR_COEF*IMAG1
          DO M=1,NLOOPAMPSO
            IF (I.EQ.J) THEN
              LOWERBOUND = M
            ELSE
              LOWERBOUND = 1
            ENDIF
            DO N=LOWERBOUND,NLOOPAMPSO
              ISQSO = ML5_0_ML5SQSOINDEX(M,N)
              IF(J.NE.I) THEN
                DOUBLEFACT=2
              ELSEIF (M.NE.N) THEN
                DOUBLEFACT=2
              ELSE
                DOUBLEFACT=1
              ENDIF
C             I have removed the conversion to real because JAMPL1 and
C              JAMPL2 are now different. The conversion to dble is
C              done later in COMPUTE_RES_FROM_JAMP if we only want the
C              matrix element
              TEMP(1) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(1,I,M)
     $         *DCONJG(JAMPL2(1,J,N))))
C             Computing the quantities below is not strictly necessary
C              since the result should be finite
C             It is however a good cross-check.
              TEMP(2) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(2,I,M)
     $         *DCONJG(JAMPL2(1,J,N)) + JAMPL1(1,I,M)*DCONJG(JAMPL2(2
     $         ,J,N))))
              TEMP(3) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(3,I,M)
     $         *DCONJG(JAMPL2(1,J,N)) + JAMPL1(1,I,M)*DCONJG(JAMPL2(3
     $         ,J,N))+JAMPL1(2,I,M)*DCONJG(JAMPL2(2,J,N))))
              DO K=1,3
                INTER(K,ISQSO) = INTER(K,ISQSO) + TEMP(K)
              ENDDO
              IF((.NOT.FILTER_SO).OR.SQSO_TARGET.EQ.
     $         -1.OR.SQSO_TARGET.EQ.ISQSO) THEN
                DO K=1,3
                  INTER(K,0) = INTER(K,0) + TEMP(K)
                ENDDO
              ENDIF
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      END SUBROUTINE


      SUBROUTINE ML5_0_SMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=4)
      INTEGER NSQSO_BORN
      PARAMETER (NSQSO_BORN=0)

      INTEGER NSQUAREDSO
      PARAMETER (NSQUAREDSO=0)
      INTEGER ANS_DIMENSION
      PARAMETER(ANS_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) ::HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)

C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL)
      REAL*8 ANS(0:3,0:ANS_DIMENSION)
      INTEGER HEL
C     
C     GLOBAL VARIABLES
C     
      INTEGER USERHEL
      COMMON/ML5_0_HELUSERCHOICE/USERHEL
C     ----------
C     BEGIN CODE
C     ----------
      USERHEL=HEL
      CALL ML5_0_SLOOPMATRIX(P,ANS)
      USERHEL=-1

      END


C     This subroutine resets to 0 the common arrays JAMPL, JAMPB and
C      possibly JAMPL_FOR_AMP2 if used
C     The common array JAMPL_ALL is reseted to 0 in the previous
C      subroutine
      SUBROUTINE ML5_0_REINITIALIZE_JAMPS()
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      COMPLEX*16 CMPLXZERO
      PARAMETER (CMPLXZERO=(0.0D0,0.0D0))
      REAL*8 ZERO
      PARAMETER (ZERO=0.0D0)
      INTEGER    NLOOPAMPSO
      PARAMETER (NLOOPAMPSO=0)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=1)
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K,L
C     
C     GLOBAL VARIABLES
C     
      COMPLEX*16 JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      COMMON/ML5_0_JAMPL/JAMPL

C     ----------
C     BEGIN CODE
C     ----------

      DO I=1,NLOOPAMPSO
        DO J=1,NLOOPFLOWS
          DO K=1,3
            JAMPL(K,J,I)=CMPLXZERO
          ENDDO
        ENDDO
      ENDDO

      END SUBROUTINE


C     This subroutine resets all cumulative (for each helicity) arrays
C      in common block and deriving from the color flow computation.
C     This subroutine is called by loop_matrix.f at appropriate time
C      when these arrays must be reset.
C     An example of this is the JAMP2 which must be cumulatively
C      computed as loop_matrix.f loops over helicity configs but must
C      be resets when loop_matrix.f starts over with helicity one (for
C      the stability test for example).
      SUBROUTINE ML5_0_REINITIALIZE_CUMULATIVE_ARRAYS()
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      COMPLEX*16 CMPLXZERO
      PARAMETER (CMPLXZERO=(0.0D0,0.0D0))
      REAL*8 ZERO
      PARAMETER (ZERO=0.0D0)
      INTEGER    NLOOPDIAGRAMS
      PARAMETER (NLOOPDIAGRAMS=16)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=1)
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
C     
C     GLOBAL VARIABLES
C     

C     ----------
C     BEGIN CODE
C     ----------


      CONTINUE

      END SUBROUTINE


