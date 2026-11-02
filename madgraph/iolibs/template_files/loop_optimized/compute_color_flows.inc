C Set of subroutines to project the Feynman diagram amplitudes AMPL onto the color flow space.

      SUBROUTINE %(proc_prefix)sCOMPUTE_COLOR_FLOWS(HEL_MULT,DO_CUMULATIVE)
      IMPLICIT NONE
      LOGICAL DO_CUMULATIVE
      INTEGER HEL_MULT
        CALL %(proc_prefix)sREINITIALIZE_JAMPS()
        CALL %(proc_prefix)sDO_COMPUTE_COLOR_FLOWS(.FALSE.)
      END SUBROUTINE

      SUBROUTINE %(proc_prefix)sCOMPUTE_COLOR_FLOWS_DERIVED_QUANTITIES(HEL_MULT)
      IMPLICIT NONE
      INTEGER HEL_MULT      
## if(MadEventOutput) {
        CALL %(proc_prefix)sCOMPUTE_AMPL2(HEL_MULT)
        CALL %(proc_prefix)sCOMPUTE_JAMP2(HEL_MULT)  
## }
        CONTINUE
      END SUBROUTINE
      
      SUBROUTINE %(proc_prefix)sDEALLOCATE_COLOR_FLOWS()
      IMPLICIT NONE
        CALL %(proc_prefix)sDO_COMPUTE_COLOR_FLOWS(.TRUE.)
      END SUBROUTINE
            
      SUBROUTINE %(proc_prefix)sDO_COMPUTE_COLOR_FLOWS(CLEANUP)
      implicit none
C  
C CONSTANTS 
C 
      CHARACTER*512 Proc_Prefix
      PARAMETER ( Proc_Prefix='%(proc_prefix)s')
      CHARACTER*512 LoopColorFlowCoefsName
      PARAMETER ( LoopColorFlowCoefsName='LoopColorFlowCoefs.dat')
## if(not LoopInduced){
      CHARACTER*512 BornColorFlowCoefsName
      PARAMETER ( BornColorFlowCoefsName='BornColorFlowCoefs.dat')
## }
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0.0d0,1.0d0))
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }

## if (not LoopInduced) {
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
## }
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)   
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d) 
      INTEGER    NBORNAMPSO
      PARAMETER (NBORNAMPSO=NLOOPAMPSO)

## }
C  
C LOCAL VARIABLES 
C
C When this subroutine is called with CLEANUP=True, it deallocates all its
C arrays
      LOGICAL CLEANUP
C Storing concatenated filenames
      CHARACTER*512 TMP
      CHARACTER*512 LoopColorFlowCoefsN
## if(not LoopInduced){
      CHARACTER*512 BornColorFlowCoefsN
## }

      INTEGER I, J, K, SOINDEX, ARRAY_SIZE
      %(complex_dp_format)s PROJ_COEF

## if(LoopInduced and MadEventOutput){
      integer config_index
      integer CONFIG_MAP(NLOOPAMPS)
      %(config_map_definition)s
## }

C  
C FUNCTIONS
C
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP
C
C GLOBAL VARIABLES
C
      character(512) MLPath
      common/MLPATH/MLPath

## if(not LoopInduced) {
      %(complex_dp_format)s AMP(NBORNAMPS)
      common/%(proc_prefix)sAMPS/AMP
## }
      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
      common/%(proc_prefix)sAMPL/AMPL
      %(complex_dp_format)s JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      common/%(proc_prefix)sJAMPL/JAMPL
## if(LoopInduced and MadEventOutput){
      %(complex_dp_format)s JAMPL_FOR_AMP2(3,NLOOPFLOWS,NLOOPAMPSO,NCONFIGS)
      common/%(proc_prefix)sJAMPL_FOR_AMP2/JAMPL_FOR_AMP2
## }

## if(not LoopInduced){
      %(complex_dp_format)s JAMPB(NBORNFLOWS,NBORNAMPSO)
      common/%(proc_prefix)sJAMPB/JAMPB
## }

C Now a more advanced data structure for storing the projection coefficient
C This is of course not F77 standard but widely supported by now.
      TYPE ProjCoeffs
        integer, dimension(:), allocatable :: Num
        integer, dimension(:), allocatable :: Denom
        integer, dimension(:), allocatable :: AmpID           
      ENDTYPE ProjCoeffs
      TYPE(ProjCoeffs), dimension(NLOOPFLOWS), save :: LoopColorProjector
## if(not LoopInduced){
      TYPE(ProjCoeffs), dimension(NBORNFLOWS), save :: BornColorProjector
## }

C ----------
C BEGIN CODE
C ----------

C CleanUp duties, i.e. array deallocation
if (CleanUp) then
  DO I=1,NLOOPFLOWS
    if(ALLOCATED(LoopColorProjector(I)%Num)) deallocate(LoopColorProjector(I)%Num)
    if(ALLOCATED(LoopColorProjector(I)%Denom)) deallocate(LoopColorProjector(I)%Denom)
    if(ALLOCATED(LoopColorProjector(I)%AmpID)) deallocate(LoopColorProjector(I)%AmpID)
  ENDDO
## if(not LoopInduced){
  DO I=1,NBORNFLOWS
    if(ALLOCATED(BornColorProjector(I)%Num)) deallocate(BornColorProjector(I)%Num)
    if(ALLOCATED(BornColorProjector(I)%Denom)) deallocate(BornColorProjector(I)%Denom)
    if(ALLOCATED(BornColorProjector(I)%AmpID)) deallocate(BornColorProjector(I)%AmpID)
  ENDDO
## }  
  return
endif

C Initialization; must allocate array from data files. All arrays are allocated at once, so we only need to check if LoopColorProjector(0)%Num is allocated and all other arrays must share the same status.
if(.not.allocated(LoopColorProjector(1)%Num)) then
  CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
  CALL JOINPATH(TMP,LoopColorFlowCoefsName,LoopColorFlowCoefsN)
## if(not LoopInduced){
  CALL JOINPATH(TMP,BornColorFlowCoefsName,BornColorFlowCoefsN)
## }

C Initialize the LoopColorProjector
  OPEN(1, FILE=LoopColorFlowCoefsN, err=201, status='OLD',           action='READ')
  DO I=1,NLOOPFLOWS
    READ(1,*,END=998) ARRAY_SIZE
    allocate(LoopColorProjector(I)%Num(ARRAY_SIZE))
    allocate(LoopColorProjector(I)%Denom(ARRAY_SIZE))
    allocate(LoopColorProjector(I)%AmpID(ARRAY_SIZE))
    READ(1,*,END=998) (LoopColorProjector(I)%Num(J),J=1,ARRAY_SIZE)
    READ(1,*,END=998) (LoopColorProjector(I)%Denom(J),J=1,ARRAY_SIZE)
    READ(1,*,END=998) (LoopColorProjector(I)%AmpID(J),J=1,ARRAY_SIZE)
  ENDDO
  GOTO 203
201  CONTINUE
  STOP 'Color projection coefficients could not be initialized from file %(proc_prefix)sLoopColorFlowCoefs.dat.' 
203  CONTINUE
  CLOSE(1)

## if (not LoopInduced) {
C Initialize the BornColorProjector
  OPEN(1, FILE=BornColorFlowCoefsN, err=301, status='OLD',           action='READ')
  DO I=1,NBORNFLOWS
    READ(1,*,END=998) ARRAY_SIZE
    allocate(BornColorProjector(I)%Num(ARRAY_SIZE))
    allocate(BornColorProjector(I)%Denom(ARRAY_SIZE))
    allocate(BornColorProjector(I)%AmpID(ARRAY_SIZE))
    READ(1,*,END=998) (BornColorProjector(I)%Num(J),J=1,ARRAY_SIZE)
    READ(1,*,END=998) (BornColorProjector(I)%Denom(J),J=1,ARRAY_SIZE)
    READ(1,*,END=998) (BornColorProjector(I)%AmpID(J),J=1,ARRAY_SIZE)
  ENDDO
  GOTO 303
301  CONTINUE
  STOP 'Color projection coefficients could not be initialized from file %(proc_prefix)sBornColorFlowCoefs.dat.'  
303  CONTINUE
  CLOSE(1)

## }
  GOTO 999
998 CONTINUE
  STOP 'End of file reached. Should not have happened.' 
999 CONTINUE

ENDIF

## if(not LoopInduced){
C Start by projecting the Born amplitudes
DO I=1,NBORNFLOWS
  DO J=1,SIZE(BornColorProjector(I)%AmpID)
    SOINDEX = %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(BornColorProjector(I)%AmpID(J))
    PROJ_COEF=DCMPLX(BornColorProjector(I)%Num(J)/DBLE(ABS(BornColorProjector(I)%Denom(J))),0.0d0)
    IF(BornColorProjector(I)%Denom(J).LT.0) PROJ_COEF=PROJ_COEF*IMAG1
    JAMPB(I,SOINDEX) = JAMPB(I,SOINDEX) + PROJ_COEF*AMP(BornColorProjector(I)%AmpID(J))
  ENDDO
ENDDO

## }
C Projection of the loop amplitudes
DO I=1,NLOOPFLOWS
  DO J=1,SIZE(LoopColorProjector(I)%AmpID)
    SOINDEX = %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(LoopColorProjector(I)%AmpID(J))
    PROJ_COEF=DCMPLX(LoopColorProjector(I)%Num(J)/DBLE(ABS(LoopColorProjector(I)%Denom(J))),0.0d0)
    IF(LoopColorProjector(I)%Denom(J).LT.0) PROJ_COEF=PROJ_COEF*IMAG1
    DO K=1,3
      JAMPL(K,I,SOINDEX) = JAMPL(K,I,SOINDEX) + PROJ_COEF*AMPL(K,LoopColorProjector(I)%AmpID(J))
## if(LoopInduced and MadEventOutput){
      config_index = CONFIG_MAP(LoopColorProjector(I)%AmpID(J))
      IF (config_index.ne.0) THEN
        JAMPL_FOR_AMP2(K,I,SOINDEX,config_index) = JAMPL_FOR_AMP2(K,I,SOINDEX,config_index) + PROJ_COEF*AMPL(K,LoopColorProjector(I)%AmpID(J))
      ENDIF
## }
    ENDDO
  ENDDO
ENDDO

      END SUBROUTINE

c This subroutine initializes the color flow matrix.
      SUBROUTINE %(proc_prefix)sINITIALIZE_FLOW_COLORMATRIX()
      implicit none
C  
C CONSTANTS 
C 
      CHARACTER*512 Proc_Prefix
      PARAMETER ( Proc_Prefix='%(proc_prefix)s')
      CHARACTER*512 LoopColorFlowMatrixName
      PARAMETER ( LoopColorFlowMatrixName='LoopColorFlowMatrix.dat')
## if(not LoopInduced){
      CHARACTER*512 BornColorFlowMatrixName
      PARAMETER ( BornColorFlowMatrixName='BornColorFlowMatrix.dat')
## }

      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)   
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
## }
C  
C LOCAL VARIABLES 
C
C Storing concatenated filenames
      CHARACTER*512 TMP
      CHARACTER*512 LoopColorFlowMatrixN
## if(not LoopInduced){
      CHARACTER*512 BornColorFlowMatrixN
## }
      INTEGER I, J
C
C GLOBAL VARIABLES
C
      character(512) MLPath
      common/MLPATH/MLPath

C Now a more advanced data structure for storing the projection coefficient
C This is of course not F77 standard but widely supported by now.
      TYPE ColorCoeff
        sequence
        integer :: Num
        integer :: Denom          
      ENDTYPE ColorCoeff
## if(not LoopInduced){
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NBORNFLOWS) :: LoopColorFlowMatrix
      TYPE(ColorCoeff), dimension(NBORNFLOWS,NBORNFLOWS) :: BornColorFlowMatrix
## } else {
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NLOOPFLOWS) :: LoopColorFlowMatrix
## }
      LOGICAL CMInitialized
      DATA CMInitialized/.FALSE./
## if(not LoopInduced){
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, BornColorFlowMatrix, CMInitialized
## } else {
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, CMInitialized
## }

C ----------
C BEGIN CODE
C ----------

C Initialization
if(.NOT.CMInitialized) then
  CMInitialized = .TRUE.
  CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
  CALL JOINPATH(TMP,LoopColorFlowMatrixName,LoopColorFlowMatrixN)
## if(not LoopInduced){
  CALL JOINPATH(TMP,BornColorFlowMatrixName,BornColorFlowMatrixN)
C Initialize the BornColorFlowMatrix
  OPEN(1, FILE=BornColorFlowMatrixN, err=601, status='OLD',           action='READ')
  DO I=1,NBORNFLOWS
    READ(1,*,END=898) (BornColorFlowMatrix(I,J)%Num,J=1,NBORNFLOWS)
    READ(1,*,END=898) (BornColorFlowMatrix(I,J)%Denom,J=1,NBORNFLOWS)
  ENDDO
  GOTO 603
601  CONTINUE
  STOP 'Color factors could not be initialized from file %(proc_prefix)sBornColorFlowMatrix.dat.' 
603  CONTINUE
  CLOSE(1)
## }
C Initialize the LoopColorFlowMatrix
  OPEN(1, FILE=LoopColorFlowMatrixN, err=701, status='OLD',           action='READ')
  DO I=1,NLOOPFLOWS
## if(not LoopInduced){
    READ(1,*,END=898) (LoopColorFlowMatrix(I,J)%Num,J=1,NBORNFLOWS)
    READ(1,*,END=898) (LoopColorFlowMatrix(I,J)%Denom,J=1,NBORNFLOWS)
## } else {
    READ(1,*,END=898) (LoopColorFlowMatrix(I,J)%Num,J=1,NLOOPFLOWS)
    READ(1,*,END=898) (LoopColorFlowMatrix(I,J)%Denom,J=1,NLOOPFLOWS)
## }
  ENDDO
  GOTO 703
701  CONTINUE
  STOP 'Color factors could not be initialized from file %(proc_prefix)sBornColorFlowMatrix.dat.' 
703  CONTINUE
  CLOSE(1)

  GOTO 899
898 CONTINUE
  STOP 'End of file reached. Should not have happened.' 
899 CONTINUE

ENDIF
      END SUBROUTINE

## if(LoopInduced and MadEventOutput){
C For the loop-inducedd MadEvent this subroutine computes the squared of 
C each individual JAMPL to created the usual JAMP2 array used for the
C color assignation of the events.
      SUBROUTINE %(proc_prefix)sCOMPUTE_JAMP2(HEL_MULT)
      implicit none
C  
C CONSTANTS 
C 
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0.0d0,1.0d0))

      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)   
C  
C ARGUMENT 
C
C HEL_MULT is the helicity multiplier which can be more than one if several helicity configuration are mapped onto the one being currently computed.
      integer HEL_MULT
C  
C LOCAL VARIABLES 
C
      INTEGER I, J, M, N, K, ISQSO, DOUBLEFACT
      %(complex_dp_format)s COLOR_COEF
C  
C FUNCTIONS
C
      INTEGER %(proc_prefix)sML5SQSOINDEX
C
C GLOBAL VARIABLES
C
      LOGICAL CHOSEN_SO_CONFIGS(NSQUAREDSO)
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS

      %(complex_dp_format)s JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      common/%(proc_prefix)sJAMPL/JAMPL

      %(real_dp_format)s JAMP2(NLOOPFLOWS)
      common/%(proc_prefix)sJAMP2/JAMP2

C Now a more advanced data structure for storing the projection coefficient
C This is of course not F77 standard but widely supported by now.
      TYPE ColorCoeff
        sequence
        integer :: Num
        integer :: Denom          
      ENDTYPE ColorCoeff
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NLOOPFLOWS) :: LoopColorFlowMatrix
      LOGICAL CMInitialized
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, CMInitialized

C ----------
C BEGIN CODE
C ----------

C Initialization
if(.NOT.CMInitialized) then
  CALL %(proc_prefix)sINITIALIZE_FLOW_COLORMATRIX()
ENDIF

C Compute the Loop ME from the loop color flow amplitudes (JAMPL)
DO I=1,NLOOPFLOWS
  COLOR_COEF=DCMPLX(LoopColorFlowMatrix(I,I)%Num/DBLE(ABS(LoopColorFlowMatrix(I,I)%Denom)),0.0d0)
  IF (LoopColorFlowMatrix(I,I)%Denom.LT.0) COLOR_COEF=COLOR_COEF*IMAG1
  DO M=1,NLOOPAMPSO
    DO N=M,NLOOPAMPSO
      ISQSO = %(proc_prefix)sML5SQSOINDEX(M,N)
      IF(CHOSEN_SO_CONFIGS(ISQSO)) THEN
        IF(M.ne.N) THEN
          DOUBLEFACT=2 
        ELSE 
          DOUBLEFACT=1
        ENDIF
        JAMP2(I)=JAMP2(I)+DABS(DOUBLEFACT*HEL_MULT*DBLE(COLOR_COEF*(JAMPL(1,I,M)*DCONJG(JAMPL(1,I,N)))))
      ENDIF
    ENDDO
  ENDDO
ENDDO

      END SUBROUTINE
## }

c This routine is used as a crosscheck only to make sure the loop ME computed
c from the JAMP is equal to the amplitude computed directly. It is a consistency
c check of the color projection and computation.

      SUBROUTINE %(proc_prefix)sCOMPUTE_RES_FROM_JAMP(RES,HEL_MULT)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
C     
C     ARGUMENT 
C     
      DOUBLE COMPLEX RES_CPLX(0:3,0:NSQUAREDSO)
      REAL*8 RES(0:3,0:NSQUAREDSO)
C     HEL_MULT is the helicity multiplier which can be more than one
C      if several helicity configuration are mapped onto the one being
C      currently computed.
      INTEGER HEL_MULT
      DOUBLE COMPLEX JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      COMMON/%(proc_prefix)sJAMPL/JAMPL
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
      DOUBLE COMPLEX JAMPB(NBORNFLOWS,NLOOPAMPSO)
      COMMON/%(proc_prefix)sJAMPB/JAMPB
## }
C     
C     LOCAL VARIABLES 
C     
      INTEGER I, J
C     ----------
C     BEGIN CODE
C     ----------
## if(LoopInduced){
      CALL %(proc_prefix)sDO_COMPUTE_INTER_JAMP(RES_CPLX, HEL_MULT, JAMPL, JAMPL)
## } else {
      CALL %(proc_prefix)sDO_COMPUTE_INTER_JAMP(RES_CPLX, HEL_MULT, JAMPL, JAMPL, JAMPB, JAMPB)
## }
      DO I=0,3
        DO J=0, NSQUAREDSO
          RES(I,J) = DREAL(RES_CPLX(I,J)) !RES is real for matrix element computation and complex for density matrix computation. A conversion is necessary
        ENDDO
      ENDDO
      END SUBROUTINE





INTEGER FUNCTION %(proc_prefix)sGET_HEL_INDEX(NHEL)
c returned th 
c
    implicit none
    integer nexternal
    parameter (nexternal=%(nexternal)i)
    integer NCOMB
    parameter (ncomb=%(ncomb)i)
    LOGICAL FOUND
    INTEGER I
    INTEGER NHEL(NEXTERNAL)
    INTEGER HELC(NEXTERNAL,NCOMB)
    COMMON/%(proc_prefix)sHELCONFIGS/HELC
c    
c   START CODE
c
    DO %(proc_prefix)sGET_HEL_INDEX=1,NCOMB
        DO I=1,NEXTERNAL
           IF (HELC(I, %(proc_prefix)sGET_HEL_INDEX).ne.NHEL(I))then 
              exit
           ELSE IF (I.eq.nexternal) then
              return
           endif   
        ENDDO
    ENDDO
END 


## if(LoopInduced and MadEventOutput){
      SUBROUTINE %(proc_prefix)sCOMPUTE_AMPL2(HEL_MULT)
C  
C CONSTANTS 
C 
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
      INTEGER    NLOOPDIAGRAMS
      PARAMETER (NLOOPDIAGRAMS=%(nloopdiags)d)
      INTEGER    NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d, NSO=%(nSO)d)
C  
C LOCAL VARIABLES 
C
      INTEGER I,J, SOINDEX
      integer config_index_map(NCONFIGS)
      %(config_index_map_definition)s
C
C ARGUMENTS
C 
      INTEGER HEL_MULT
C
C GLOBAL VARIABLES
C
      %(real_dp_format)s AMP2(NLOOPDIAGRAMS)
      common/%(proc_prefix)sAMP2/AMP2
      %(real_dp_format)s  AMP2_ALL(3,NCONFIGS,0:NSQUAREDSO)   
      common/%(proc_prefix)sAMP2_ALL/AMP2_ALL
      LOGICAL CHOSEN_SO_CONFIGS(NSQUAREDSO)
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS
C ----------
C BEGIN CODE
C ----------

      do i=1,nconfigs
C The configurations are not necessarily consecutive when grouping subprocesses. So, here, we simply skip those which have no diagrams for this process.
        if (config_index_map(i).eq.0) then
          cycle
        endif
        DO J=1,NSQUAREDSO
C For the multi-channeling, it is in general better to ignore the squared order constraints (which, if they are any, would restrict the AMP2 quantities to be coming from the interference terms only).  
         IF (.TRUE.) THEN
C Choosing the if statement below instead would as a result to apply the squared order constraints to the AMP2 quantities given to MadEvent for the multichanneling.
c         IF (CHOSEN_SO_CONFIGS(I)) THEN
            AMP2(config_index_map(i))=AMP2(config_index_map(i))+DABS(AMP2_ALL(1,i,J))
          ENDIF
        ENDDO
      enddo

      END SUBROUTINE
## }


## if(LoopInduced) {
      SUBROUTINE %(proc_prefix)sDO_COMPUTE_INTER_JAMP(INTER,HEL_MULT,JAMPL1,JAMPL2)
## } else {
      SUBROUTINE %(proc_prefix)sDO_COMPUTE_INTER_JAMP(INTER,HEL_MULT,JAMPL1,JAMPL2,JAMPB1,JAMPB2)
## }
C     Computation of the interference between the colour flows of two
C     helicity configurations, 1 and 2: the matrix element when they are the
C     same, an element rho(1,2) of the density matrix otherwise. INTER(0,*) is
C     tree x tree, INTER(1:3,*) the finite part, single and double pole of
C     loop x tree (loop x loop for loop-induced processes). Element (1,2) is
C     sum_ab C_ab A_a(1) conjg(A_b(2)), complex when 1 != 2.
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0.0d0,1.0d0))

      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d, NSO=%(nSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
C     We use the same list of amplitude split order for Born and Loop, so that NBORNAMPSO is always equal to NLOOPAMPSO  
      INTEGER    NBORNAMPSO
      PARAMETER (NBORNAMPSO=NLOOPAMPSO)
## }
C     
C     ARGUMENT 
C     
      %(complex_dp_format)s INTER(0:3,0:NSQUAREDSO)
C     HEL_MULT is the helicity multiplier which can be more than one
C      if several helicity configuration are mapped onto the one being
C      currently computed.
      INTEGER HEL_MULT      
C     
C     LOCAL VARIABLES 
C     
      INTEGER I, J, K, M, N, ISQSO, DOUBLEFACT
## if(LoopInduced) {
      INTEGER LOWERBOUND
## }
## if(LoopInduced and MadEventOutput){
      integer config_i, config_j
## }
      INTEGER ORDERS_A(NSO), ORDERS_B(NSO)
      %(complex_dp_format)s COLOR_COEF
      %(complex_dp_format)s TEMP(3)
## if(LoopInduced and MadEventOutput){
      %(complex_dp_format)s TEMP_AMP2(3,NCONFIGS)
## }
C  
C FUNCTIONS
C
C This function belongs to the loop ME computation (loop_matrix.f) and is prefixed with ML5 
      INTEGER %(proc_prefix)sML5SQSOINDEX
C This function belongs to the Born ME computation (born_matrix.f) and is not prefixed at all
      INTEGER %(proc_prefix)sSQSOINDEX, %(proc_prefix)sSOINDEX_FOR_AMPORDERS
C
C GLOBAL VARIABLES
C
      character(512) MLPath
      common/MLPATH/MLPath
      INTEGER SQSO_TARGET
      common/%(proc_prefix)sSOCHOICE/SQSO_TARGET


      %(complex_dp_format)s JAMPL1(3,NLOOPFLOWS,NLOOPAMPSO)
      %(complex_dp_format)s JAMPL2(3,NLOOPFLOWS,NLOOPAMPSO)

## if(not LoopInduced){
      %(complex_dp_format)s JAMPB1(NBORNFLOWS,NBORNAMPSO)
      %(complex_dp_format)s JAMPB2(NBORNFLOWS,NBORNAMPSO)
## }
## if(LoopInduced and MadEventOutput){
      %(real_dp_format)s AMP2_ALL(3,NCONFIGS,0:NSQUAREDSO)    
      common/%(proc_prefix)sAMP2_ALL/AMP2_ALL
      %(complex_dp_format)s JAMPL_FOR_AMP2(3,NLOOPFLOWS,NLOOPAMPSO,NCONFIGS)
      common/%(proc_prefix)sJAMPL_FOR_AMP2/JAMPL_FOR_AMP2
## }

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
      common/%(proc_prefix)sSO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

C Now a more advanced data structure for storing the projection coefficient
C This is of course not F77 standard but widely supported by now.
      TYPE ColorCoeff
        sequence
        integer :: Num
        integer :: Denom          
      ENDTYPE ColorCoeff
## if(not LoopInduced){
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NBORNFLOWS) :: LoopColorFlowMatrix
      TYPE(ColorCoeff), dimension(NBORNFLOWS,NBORNFLOWS) :: BornColorFlowMatrix
## } else {
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NLOOPFLOWS) :: LoopColorFlowMatrix
## }
      LOGICAL CMInitialized
## if(not LoopInduced){
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, BornColorFlowMatrix, CMInitialized
## } else {
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, CMInitialized
## }

C ----------
C BEGIN CODE
C ----------
## if(LoopInduced and MadEventOutput){
C Always reinitialize AMP2_ALL because it is not a cumulative quantity
CALL %(proc_prefix)sREINITIALIZE_AMP2_ALL()
## }

C Initialization
if(.NOT.CMInitialized) then
  CALL %(proc_prefix)sINITIALIZE_FLOW_COLORMATRIX()
ENDIF

DO I=0,NSQUAREDSO
  DO K=0,3
    INTER(K,I)=0.0d0
  ENDDO
ENDDO

## if(not LoopInduced){
c Compute the Born ME from the Born color flow amplitudes (JAMP)
DO I=1, NBORNFLOWS
  DO J=I, NBORNFLOWS
    COLOR_COEF=DCMPLX(BornColorFlowMatrix(I,J)%Num/DBLE(ABS(BornColorFlowMatrix(I,J)%Denom)),0.0d0)
    IF (BornColorFlowMatrix(I,J)%Denom.LT.0) COLOR_COEF=COLOR_COEF*IMAG1
    DO M=1,NBORNAMPSO
C It may be that this AmpSO index does not receive contribution by the Born amps (because we put the loop and Born amplitude split orders in a common list)
      IF ((ABS(JAMPB1(I,M))+ABS(JAMPB1(J,M))).eq.0.0d0) CYCLE
      DO N=1,NBORNAMPSO
        IF ((ABS(JAMPB2(I,N))+ABS(JAMPB2(J,N))).eq.0.0d0) CYCLE
C First fetch what orders the split order indices M, N correspond to
        CALL %(proc_prefix)sML5GET_ORDERS_FOR_AMPSOINDEX(M,ORDERS_A)
        CALL %(proc_prefix)sML5GET_ORDERS_FOR_AMPSOINDEX(N,ORDERS_B)
C Now figure out to which SQSOINDEX these orders together correspond to *in the Born subroutine* (Notice the absence of the ML5 prefix in the functions used then)
        ISQSO = %(proc_prefix)sSQSOINDEX(%(proc_prefix)sSOINDEX_FOR_AMPORDERS(ORDERS_A),%(proc_prefix)sSOINDEX_FOR_AMPORDERS(ORDERS_B))
        IF(J.ne.I) THEN
          DOUBLEFACT=2 
        ELSE 
          DOUBLEFACT=1
        ENDIF
C The (I,J) and (J,I) terms are both written out, so that the result is the
C density-matrix element when JAMPB1 and JAMPB2 differ. When they are the same
C it is real, and equal to the former DOUBLEFACT*DBLE(C*JAMPB(I,M)*DCONJG(JAMPB(J,N))).
        TEMP(1) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPB1(I,M)*DCONJG(JAMPB2(J,N))+JAMPB1(J,M)*DCONJG(JAMPB2(I,N))))/2D0
        INTER(0,ISQSO) = INTER(0,ISQSO) + TEMP(1)
        IF((.not.FILTER_SO).or.SQSO_TARGET.eq.-1.or.SQSO_TARGET.eq.ISQSO) THEN
          INTER(0,0) = INTER(0,0) + TEMP(1)
        ENDIF
      ENDDO
    ENDDO
  ENDDO
ENDDO
## }

C Compute the Loop ME from the loop color flow amplitudes (JAMPL)
DO I=1,NLOOPFLOWS
## if(not LoopInduced){
  DO J=1,NBORNFLOWS
## } else {
  DO J=I,NLOOPFLOWS
## }
    COLOR_COEF=DCMPLX(LoopColorFlowMatrix(I,J)%Num/DBLE(ABS(LoopColorFlowMatrix(I,J)%Denom)),0.0d0)
    IF (LoopColorFlowMatrix(I,J)%Denom.LT.0) COLOR_COEF=COLOR_COEF*IMAG1
    DO M=1,NLOOPAMPSO
## if(not LoopInduced){
C It may be that this AmpSO index does not receive contribution by the Loop amps (because we put the loop and Born amplitude split orders in a common list)
      IF((ABS(JAMPL1(1,I,M))+ABS(JAMPL1(2,I,M))+ABS(JAMPL1(3,I,M))+ABS(JAMPL2(1,I,M))+ABS(JAMPL2(2,I,M))+ABS(JAMPL2(3,I,M))).eq.0.0d0) CYCLE
      DO N=1,NBORNAMPSO
C Same for contributions of split order index N by the Born amps
      IF((ABS(JAMPB1(J,N))+ABS(JAMPB2(J,N))).eq.0.0d0) CYCLE
## } else {
      IF (I.eq.J) THEN
        LOWERBOUND = M
      ELSE
        LOWERBOUND = 1
      ENDIF
      DO N=LOWERBOUND,NLOOPAMPSO
## }
         ISQSO = %(proc_prefix)sML5SQSOINDEX(M,N)
## if(LoopInduced){
        IF(J.ne.I) THEN
          DOUBLEFACT=2
        ELSEIF (M.ne.N) THEN
          DOUBLEFACT=2
        ELSE
          DOUBLEFACT=1
        ENDIF
C The conversion to real has been removed because JAMPL1 and JAMPL2 are now different. The conversion to dble is done later in COMPUTE_RES_FROM_JAMP if we only want the matrix element
C The result is divided by 2 because in the case I=J, we compute the interference 2 times and in the case J>I, the value of DOUBLEFACT introduces a factor 2 that is not needed
        TEMP(1) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(1,I,M)*DCONJG(JAMPL2(1,J,N)) + JAMPL1(1,J,M)*DCONJG(JAMPL2(1,I,N))))/2D0
C Computing the quantities below is not strictly necessary since the result should be finite
C It is however a good cross-check.
        TEMP(2) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(2,I,M)*DCONJG(JAMPL2(1,J,N))+JAMPL1(2,J,M)*DCONJG(JAMPL2(1,I,N)) + JAMPL1(1,I,M)*DCONJG(JAMPL2(2,J,N))+JAMPL1(1,J,M)*DCONJG(JAMPL2(2,I,N))))/2D0
        TEMP(3) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(3,I,M)*DCONJG(JAMPL2(1,J,N))+JAMPL1(3,J,M)*DCONJG(JAMPL2(1,I,N)) + JAMPL1(1,I,M)*DCONJG(JAMPL2(3,J,N))+JAMPL1(1,J,M)*DCONJG(JAMPL2(3,I,N)) + JAMPL1(2,I,M)*DCONJG(JAMPL2(2,J,N))+JAMPL1(2,J,M)*DCONJG(JAMPL2(2,I,N))))/2D0
## if(MadEventOutput) {
        DO config_i=1,nconfigs
          DO K=1,3
            TEMP_AMP2(K,config_i) = 0.0d0
          ENDDO
C For the multi-channeling, to choose to square against only among a given configuration, use the do loop below
          DO config_j=config_i,config_i
c For the multi-channeling, to choose to square against all other configuration, use the do loop below
c          DO config_j=1,nconfigs
            TEMP_AMP2(1,config_i) = TEMP_AMP2(1,config_i) + DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL_FOR_AMP2(1,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(1,J,N,config_j))))
            TEMP_AMP2(2,config_i) = TEMP_AMP2(2,config_i) + DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL_FOR_AMP2(2,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(1,J,N,config_j)) + JAMPL_FOR_AMP2(1,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(2,J,N,config_j))))
            TEMP_AMP2(3,config_i) = TEMP_AMP2(3,config_i) + DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL_FOR_AMP2(3,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(1,J,N,config_j)) + JAMPL_FOR_AMP2(1,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(3,J,N,config_j))+JAMPL_FOR_AMP2(2,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(2,J,N,config_j))))

          ENDDO
        ENDDO
## }
## }
        DO K=1,3
## if(not LoopInduced){      
C Loop(1) x conjg(Born(2)) plus Born(1) x conjg(Loop(2)). For the same
C helicity this is the former 2*DBLE(C*JAMPL(K,I,M)*DCONJG(JAMPB(J,N))).
          TEMP(K) = HEL_MULT*(COLOR_COEF*JAMPL1(K,I,M)*DCONJG(JAMPB2(J,N)) + DCONJG(COLOR_COEF)*JAMPB1(J,N)*DCONJG(JAMPL2(K,I,M)))
## }
          INTER(K,ISQSO) = INTER(K,ISQSO) + TEMP(K)
## if(LoopInduced and MadEventOutput){
          DO config_i=1,nconfigs
            AMP2_ALL(K,config_i,ISQSO) = AMP2_ALL(K,config_i,ISQSO) + TEMP_AMP2(K,config_i)
          ENDDO
## }
        ENDDO
        IF((.NOT.FILTER_SO).OR.SQSO_TARGET.eq.-1.or.SQSO_TARGET.eq.ISQSO) THEN
          DO K=1,3
            INTER(K,0) = INTER(K,0) + TEMP(K)
## if(LoopInduced and MadEventOutput){
C           Notice here that the squared order constraints will apply to the determination of the AMP2 for multichanneling. Place it outside the corresponding statement if this is not desirable.
            do config_i=1,nconfigs
              AMP2_ALL(K,config_i,0) = AMP2_ALL(K,config_i,0) + TEMP_AMP2(K,config_i)
            enddo
## }
          ENDDO
        ENDIF
      ENDDO
    ENDDO
  ENDDO
ENDDO
      END SUBROUTINE


## if(LoopInduced){
C A process with a Born gets SMATRIXHEL (of the Born) from born_matrix.f
     SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
      %(get_nsqso_born)s
      INTEGER NSQUAREDSO
	PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
      INTEGER ANS_DIMENSION
	PARAMETER(ANS_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) ::HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)  

C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(0:3,0:ANS_DIMENSION)
      INTEGER HEL
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL %(proc_prefix)sSLOOPMATRIX(P,ANS)
      USERHEL=-1

      END
## }

## if(UseDensity){

C This subroutine resets to 0 the colour flows stored per helicity for the
C density matrix (JAMPL_ALL, and JAMPB_ALL for a process with a Born). It is
C called before the reference evaluation of each phase-space point, so that a
C helicity skipped by the helicity filter has zero colour flows rather than
C those of the previous point.
SUBROUTINE %(proc_prefix)sREINITIALIZE_JAMP_ALL()
implicit none
C  
C CONSTANTS 
C 
      %(complex_dp_format)s CMPLXZERO
      PARAMETER (CMPLXZERO=(0.0d0,0.0d0))
      INTEGER    NLOOPAMPSO
      PARAMETER (NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
## }
C
C GLOBAL VARIABLES
C
      DOUBLE COMPLEX JAMPL_ALL(3, NLOOPFLOWS,NLOOPAMPSO, NCOMB)
      COMMON/%(proc_prefix)sDENSITY_MATRIX/JAMPL_ALL
## if(not LoopInduced){
      DOUBLE COMPLEX JAMPB_ALL(NBORNFLOWS,NLOOPAMPSO, NCOMB)
      COMMON/%(proc_prefix)sDENSITY_MATRIX_B/JAMPB_ALL
## }

C ----------
C BEGIN CODE
C ----------

      JAMPL_ALL(:,:,:,:)=CMPLXZERO
## if(not LoopInduced){
      JAMPB_ALL(:,:,:)=CMPLXZERO
## }
      END SUBROUTINE
## }

C This subroutine resets to 0 the common arrays JAMPL, JAMPB and possibly JAMPL_FOR_AMP2 if used
c The common array JAMPL_ALL is reset to 0 in the previous subroutine
SUBROUTINE %(proc_prefix)sREINITIALIZE_JAMPS()
implicit none
C  
C CONSTANTS 
C 
      %(complex_dp_format)s CMPLXZERO
      PARAMETER (CMPLXZERO=(0.0d0,0.0d0))
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0.0d0)
      INTEGER    NLOOPAMPSO
      PARAMETER (NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
C     We use the same list of amplitude split order for Born and Loop, so that NBORNAMPSO is always equal to NLOOPAMPSO
      INTEGER    NBORNAMPSO
      PARAMETER (NBORNAMPSO=NLOOPAMPSO)
## }
C
C LOCAL VARIABLES
C
      INTEGER I,J,K,L
C
C GLOBAL VARIABLES
C
      %(complex_dp_format)s JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      common/%(proc_prefix)sJAMPL/JAMPL
## if(not LoopInduced){
      %(complex_dp_format)s JAMPB(NBORNFLOWS,NBORNAMPSO)
      common/%(proc_prefix)sJAMPB/JAMPB
## }
## if(LoopInduced and MadEventOutput){
      %(complex_dp_format)s JAMPL_FOR_AMP2(3,NLOOPFLOWS,NLOOPAMPSO,NCONFIGS)
      common/%(proc_prefix)sJAMPL_FOR_AMP2/JAMPL_FOR_AMP2
## }

C ----------
C BEGIN CODE
C ----------

DO I=1,NLOOPAMPSO
  DO J=1,NLOOPFLOWS
    DO K=1,3
      JAMPL(K,J,I)=CMPLXZERO
    ENDDO
  ENDDO
ENDDO
## if(LoopInduced and MadEventOutput){
DO L=1,NCONFIGS
  DO I=1,NLOOPAMPSO
    DO J=1,NLOOPFLOWS
      DO K=1,3
        JAMPL_FOR_AMP2(K,J,I,L)=CMPLXZERO
      ENDDO
    ENDDO
  ENDDO
ENDDO
## }
## if(not LoopInduced){
DO I=1,NBORNAMPSO
  DO J=1,NBORNFLOWS
    JAMPB(J,I)=CMPLXZERO
  ENDDO
ENDDO
## }

END SUBROUTINE

## if(LoopInduced and MadEventOutput){
C This subroutine resets to 0 the common arrays AMP2_ALL
SUBROUTINE %(proc_prefix)sREINITIALIZE_AMP2_ALL()
implicit none
C  
C CONSTANTS 
C 
      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d, NSO=%(nSO)d)
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0.0d0)
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
C
C LOCAL VARIABLES
C
      INTEGER I,K,L
C
C GLOBAL VARIABLES
C
      %(real_dp_format)s AMP2_ALL(3,NCONFIGS,0:NSQUAREDSO)
      common/%(proc_prefix)sAMP2_ALL/AMP2_ALL
C ----------
C BEGIN CODE
C ----------

DO I=1,NCONFIGS
  DO K=1,3
    DO L=0,NSQUAREDSO
      AMP2_ALL(K,I,L)=ZERO
    ENDDO
  ENDDO
ENDDO

END SUBROUTINE
## }

C This subroutine resets all cumulative (for each helicity) arrays in common block and deriving from the color flow computation.
C This subroutine is called by loop_matrix.f at appropriate time when these arrays must be reset.
C An example of this is the JAMP2 which must be cumulatively computed as loop_matrix.f loops over helicity configs but must be resets when loop_matrix.f starts over with helicity one (for the stability test for example).
SUBROUTINE %(proc_prefix)sREINITIALIZE_CUMULATIVE_ARRAYS()
implicit none
C  
C CONSTANTS 
C 
      %(complex_dp_format)s CMPLXZERO
      PARAMETER (CMPLXZERO=(0.0d0,0.0d0))
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0.0d0)
      INTEGER    NLOOPDIAGRAMS
      PARAMETER (NLOOPDIAGRAMS=%(nloopdiags)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }
C
C LOCAL VARIABLES
C
      INTEGER I,J,K
C
C GLOBAL VARIABLES
C
## if(LoopInduced and MadEventOutput){
      %(real_dp_format)s AMP2(NLOOPDIAGRAMS)
      common/%(proc_prefix)sAMP2/AMP2
      %(real_dp_format)s JAMP2(NLOOPFLOWS)
      common/%(proc_prefix)sJAMP2/JAMP2
## }

C ----------
C BEGIN CODE
C ----------

## if(LoopInduced and MadEventOutput){
  DO I=1,NLOOPDIAGRAMS
    AMP2(I)=ZERO
  ENDDO
  DO I=1,NLOOPFLOWS
    JAMP2(I)=ZERO
  ENDDO
## }

CONTINUE

END SUBROUTINE

## if(UseDensity) {
C ------------------------------------------------------------------------
C Spin-density matrix of the legs POS(1..N_CHANGING), summed over the
C helicities of every other leg. ALLOW_HEL((I-1)*N_CHANGING+K) is the helicity
C of leg POS(K) in combination I, and an element (I,J) is
C     sum_{other hels} sum_ab C_ab A_a(I) conjg(A_b(J)),
C packed as the upper triangle, row by row: (1,1),(1,2),...,(1,N),(2,2),...
C The colour flows of every helicity are stored by STORE_JAMP during the
C reference evaluation of SLOOPMATRIX (never a rotated one of the stability
C test, which would change the phases of the off-diagonal elements).
C ------------------------------------------------------------------------

      SUBROUTINE %(proc_prefix)sGET_ALL_INTER(NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB_RHO, INTER)
C     Density matrix for one helicity configuration NHEL of the legs that are
C     not in POS. INTER(:,0) is tree x tree, INTER(:,1:3) the finite part,
C     single and double pole (summed over the selected squared split orders).
      IMPLICIT NONE
C     
C     CONSTANT
C     
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)i)
C     
C     ARGUMENTS
C     
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB_RHO
      INTEGER POS(N_CHANGING)
      INTEGER ALLOW_HEL(N_CHANGING * N_COMB_RHO)
      DOUBLE COMPLEX INTER(N_COMB_RHO * (N_COMB_RHO + 1)/2, 0:3)
C     
C     LOCAL VARIABLES
C     
      DOUBLE COMPLEX INTER_TMP(0:3,0:NSQUAREDSO)
      INTEGER ORDER_HEL(N_COMB_RHO)
      INTEGER I, J, K, N, SOL

      DO I = 1, N_COMB_RHO
        DO N = 1, N_CHANGING
          NHEL(POS(N)) = ALLOW_HEL((I-1)*N_CHANGING+N)
        ENDDO
        CALL %(proc_prefix)sFIND_H(NHEL, ORDER_HEL(I))
        IF (ORDER_HEL(I).EQ.0) THEN
          WRITE(*,*) 'Helicity configuration not found:', NHEL
          STOP 'Density matrix: invalid helicity for a changing leg.'
        ENDIF
      ENDDO

      SOL = 0
      DO I = 1, N_COMB_RHO
        DO J = I, N_COMB_RHO
          SOL = SOL + 1
          CALL %(proc_prefix)sCOMPUTE_INTER_JAMP(INTER_TMP, ORDER_HEL(I), ORDER_HEL(J))
          DO K = 0, 3
            INTER(SOL,K) = INTER_TMP(K,0)
          ENDDO
        ENDDO
      ENDDO

      RETURN
      END SUBROUTINE


      SUBROUTINE %(proc_prefix)sGET_DENSITY_ALL(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, ALPHAS, SCALE2, INTER)
C     Full density matrix at the phase-space point P: INTER(:,0) is the Born
C     (zero for a loop-induced process), INTER(:,1:3) the finite part, single
C     and double pole of the virtual, normalised as ANS(0:3,0) of SLOOPMATRIX:
C     the trace of INTER(:,K) is ANS(K,0), to the accuracy of the stability test.
C     ALPHAS and SCALE2 (mu_R^2) are used when both are non-zero.
      IMPLICIT NONE
C     
C     CONSTANT
C     
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
## if(LoopInduced){
      INTEGER NSQSO_BORN
      PARAMETER (NSQSO_BORN=0)
## } else {
      %(get_nsqso_born)s
## }
      INTEGER AUX_DIMENSION
      PARAMETER(AUX_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)i)
      INCLUDE 'MadLoopParams.inc'
C     
C     ARGUMENTS
C     
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(N_CHANGING)
      INTEGER ALLOW_HEL(N_CHANGING*N_COMB)
      DOUBLE COMPLEX INTER(N_COMB * (N_COMB + 1)/2, 0:3)
      DOUBLE PRECISION P(0:3, NEXTERNAL)
      DOUBLE PRECISION ALPHAS, SCALE2
C     
C     LOCAL VARIABLES
C     
      DOUBLE COMPLEX TMP_INTER(N_COMB * (N_COMB + 1)/2, 0:3)
      INTEGER THISNHEL(NEXTERNAL)
      REAL*8 AUX(0:3,0:AUX_DIMENSION)
      INTEGER I, K, IHEL, IPART, JJ
      DOUBLE PRECISION POLFACT
C     
C     GLOBAL VARIABLES
C     
      INTEGER HELC(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sHELCONFIGS/HELC
C     beam polarisation, filled from python through PY_SET_BEAMPOL.
C     Same common block and convention as the v1 MadSpin path
C     (matrix_standalone_msP_v4.inc / msF).
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(nincoming)d)
      DOUBLE PRECISION BEAMPOL(2)
      COMMON/TO_BEAMPOL/BEAMPOL

      IF ((ALPHAS.ne.0D0).and.(SCALE2.ne.0D0)) THEN
        call UPDATE_AS_PARAM2(SCALE2, ALPHAS)
      ENDIF

      CALL %(proc_prefix)sSLOOPMATRIX(P, AUX)

C     HelicityFilterLevel=2 evaluates one helicity for all those equal to it
C     up to a sign, which is right for the summed matrix element only: the
C     others never get their colour flows stored. (MadLoopParams is read
C     during the first SLOOPMATRIX call, hence the check after it.)
      IF (HelicityFilterLevel.GT.1) THEN
        WRITE(*,*) 'ERROR: the density matrix needs HelicityFilterLevel'
        WRITE(*,*) '0 or 1 in MadLoopParams.dat, not', HelicityFilterLevel
        STOP 1
      ENDIF

      INTER(:,:) = (0D0,0D0)
C     One pass per helicity configuration of the legs not in POS: those whose
C     changing legs are in the first combination of ALLOW_HEL.
      DO IHEL =1, NCOMB
        THISNHEL(:) = HELC(:, IHEL)
        DO IPART=1,N_CHANGING
          IF(THISNHEL(POS(IPART)).NE.ALLOW_HEL(IPART)) GOTO 10
        ENDDO
        CALL %(proc_prefix)sGET_ALL_INTER(THISNHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, TMP_INTER)
C       Support for polarised beam: reweight the initial-state helicity
C       sum exactly as SMATRIX_PROD does in the v1 path. Skipped unless
C       the process has two incoming legs -- the same shared library
C       also holds the 1 -> N decay matrix elements, whose leg 1 is the
C       decaying resonance and not a beam.
        POLFACT = 1D0
        IF (NINITIAL.EQ.2) THEN
          DO JJ=1,NINITIAL
C           |BEAMPOL| runs from 1 (unpolarised) to 2 (fully
C           polarised), so anything at or below 1 means "no
C           polarisation" -- including the zero-filled common
C           block of an output whose link line does not pull
C           in BLOCK DATA BEAMPOL_DEFAULT.
            IF (ABS(BEAMPOL(JJ)).LE.1D0) CYCLE
            IF (THISNHEL(JJ).EQ.INT(SIGN(1D0,BEAMPOL(JJ)))) THEN
              POLFACT = POLFACT*ABS(BEAMPOL(JJ))
            ELSE
              POLFACT = POLFACT*(2D0-ABS(BEAMPOL(JJ)))
            ENDIF
          ENDDO
        ENDIF
        DO K = 0, 3
          DO I = 1, N_COMB*(N_COMB+1)/2
            INTER(I,K) = INTER(I,K) + POLFACT*TMP_INTER(I,K)
          ENDDO
        ENDDO
 10   ENDDO

      RETURN
      END


      SUBROUTINE %(proc_prefix)sDENSITY_PROJECT_ANS(ANS)
C     The projector W of the density matrix (the FKS standalone output with
C     --density): if the file density_w.dat exists in the working
C     directory (one line 'C1 C2 Re Im' per element W(C1,C2), read once),
C     the result ANS(0:3,0:NSQUAREDSO) of SLOOPMATRIX is replaced by
C     Re Tr(W rho), rho the density matrix of the output's density legs
C     built from the colour flows stored for this point. The Born
C     (ANS(0,*)) and the colour-linked Borns of MadFKS read the same file,
C     so that check_poles can test each element of the density matrix.
      IMPLICIT NONE
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
## if(LoopInduced){
      INTEGER NSQSO_BORN
      PARAMETER (NSQSO_BORN=0)
## } else {
      %(get_nsqso_born)s
## }
      INTEGER ANS_DIMENSION
      PARAMETER(ANS_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)i)
      INTEGER N_CHANGING
      PARAMETER (N_CHANGING=%(dens_nchanging)i)
      INTEGER N_COMB
      PARAMETER (N_COMB=%(dens_ncomb)i)
      REAL*8 ANS(0:3,0:ANS_DIMENSION)
C
      INTEGER POS(N_CHANGING), ALLOW_HEL(N_CHANGING*N_COMB)
      INTEGER ORDER_HEL(N_COMB), THISNHEL(NEXTERNAL)
      DOUBLE COMPLEX W(N_COMB,N_COMB), RES(0:3,0:NSQUAREDSO)
      DOUBLE COMPLEX INTER_TMP(0:3,0:NSQUAREDSO)
      LOGICAL W_ON, W_READ
      SAVE W, W_ON, W_READ
      DATA W_ON /.FALSE./, W_READ /.FALSE./
      INTEGER I, K, IHEL, IPART, C1, C2, N
      DOUBLE PRECISION WR, WI
      INTEGER HELC(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sHELCONFIGS/HELC
      INTEGER USERHEL
      COMMON/%(proc_prefix)sUSERCHOICE/USERHEL
C     the projector set by the caller (DENSITY_SET_W), which takes
C     precedence over density_w.dat
      LOGICAL API_SET, API_ON
      DOUBLE COMPLEX API_W(N_COMB,N_COMB)
      COMMON/%(proc_prefix)sDENSITY_W_API/API_W, API_ON, API_SET

      IF (API_SET) THEN
        IF (.NOT.API_ON) RETURN
        W(:,:)=API_W(:,:)
        GOTO 25
      ENDIF
      IF (.NOT.W_READ) THEN
        W_READ=.TRUE.
        W(:,:)=(0D0,0D0)
        OPEN(UNIT=82,FILE='density_w.dat',STATUS='OLD',ERR=20)
 10     READ(82,*,END=15,ERR=15) C1, C2, WR, WI
        W(C1,C2)=DCMPLX(WR,WI)
        GOTO 10
 15     CLOSE(82)
        W_ON=.TRUE.
        WRITE(*,*) 'MadLoop: density-matrix projector W read from density_w.dat'
 20     CONTINUE
      ENDIF
      IF (.NOT.W_ON) RETURN
 25   CONTINUE
      IF (USERHEL.NE.-1) THEN
        WRITE(*,*) 'ERROR: the density-matrix projector W needs the sum'
        WRITE(*,*) 'over all helicities, not a single helicity.'
        STOP 1
      ENDIF

      %(dens_pos)s
      %(dens_allow_hel)s

      RES(:,:)=(0D0,0D0)
C     one pass per helicity configuration of the legs not in POS
      DO IHEL=1,NCOMB
        THISNHEL(:)=HELC(:,IHEL)
        DO IPART=1,N_CHANGING
          IF (THISNHEL(POS(IPART)).NE.ALLOW_HEL(IPART)) GOTO 30
        ENDDO
        DO I=1,N_COMB
          DO N=1,N_CHANGING
            THISNHEL(POS(N))=ALLOW_HEL((I-1)*N_CHANGING+N)
          ENDDO
          CALL %(proc_prefix)sFIND_H(THISNHEL, ORDER_HEL(I))
        ENDDO
        DO C1=1,N_COMB
          DO C2=1,N_COMB
            IF (W(C2,C1).EQ.(0D0,0D0)) CYCLE
            CALL %(proc_prefix)sCOMPUTE_INTER_JAMP(INTER_TMP, ORDER_HEL(C1), ORDER_HEL(C2))
            RES(:,:)=RES(:,:)+W(C2,C1)*INTER_TMP(:,:)
          ENDDO
        ENDDO
 30   CONTINUE
      ENDDO
      DO K=0,3
        DO I=0,NSQUAREDSO
          ANS(K,I)=DBLE(RES(K,I))
        ENDDO
      ENDDO
      END


      SUBROUTINE %(proc_prefix)sDENSITY_SET_W(WIN, N, ON)
C     Sets the projector W of the density matrix (N x N, the combinations
C     of DENSITY_PROJECT_ANS) and switches it on or off: with it on, the
C     result of SLOOPMATRIX is Re Tr(W rho). Used by the aMC@NLO
C     density-matrix weights (density_weights.f); it takes precedence over
C     the file density_w.dat.
      IMPLICIT NONE
      INTEGER N_COMB
      PARAMETER (N_COMB=%(dens_ncomb)i)
      INTEGER N
      DOUBLE COMPLEX WIN(N,N)
      LOGICAL ON
      LOGICAL API_SET, API_ON
      DOUBLE COMPLEX API_W(N_COMB,N_COMB)
      COMMON/%(proc_prefix)sDENSITY_W_API/API_W, API_ON, API_SET
      IF (N.NE.N_COMB) THEN
        WRITE(*,*) 'ERROR: density projector of size', N, 'for',
     &    N_COMB, 'helicity combinations'
        STOP 1
      ENDIF
      API_SET=.TRUE.
      API_ON=ON
      IF (ON) API_W(:,:)=WIN(:,:)
      END


      LOGICAL FUNCTION %(proc_prefix)sDENSITY_W_ACTIVE()
C     True when the projector set by DENSITY_SET_W is on
      IMPLICIT NONE
      INTEGER N_COMB
      PARAMETER (N_COMB=%(dens_ncomb)i)
      LOGICAL API_SET, API_ON
      DOUBLE COMPLEX API_W(N_COMB,N_COMB)
      COMMON/%(proc_prefix)sDENSITY_W_API/API_W, API_ON, API_SET
      %(proc_prefix)sDENSITY_W_ACTIVE=API_SET.AND.API_ON
      END


      SUBROUTINE %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, ALPHAS, SCALE2, INTER)
C     Density matrix of the one-loop result: the finite part of the virtual
C     for a process with a Born, the (finite) loop-induced amplitude squared
C     otherwise. GET_DENSITY_ALL also returns the Born and the poles.
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)i)
C     
C     ARGUMENTS
C     
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(N_CHANGING)
      INTEGER ALLOW_HEL(N_CHANGING*N_COMB)
      DOUBLE COMPLEX INTER(N_COMB * (N_COMB + 1)/2)
      DOUBLE PRECISION P(0:3, NEXTERNAL)
      DOUBLE PRECISION ALPHAS, SCALE2
C     
C     LOCAL VARIABLES
C     
      DOUBLE COMPLEX TMP_INTER(N_COMB * (N_COMB + 1)/2, 0:3)
      INTEGER I

      CALL %(proc_prefix)sGET_DENSITY_ALL(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, ALPHAS, SCALE2, TMP_INTER)
      DO I = 1, N_COMB*(N_COMB+1)/2
## if(LoopInduced){
C       the pole slots of a loop-induced amplitude are zero up to rounding
        INTER(I) = TMP_INTER(I,1) + TMP_INTER(I,2) + TMP_INTER(I,3)
## } else {
        INTER(I) = TMP_INTER(I,1)
## }
      ENDDO
      RETURN
      END

     SUBROUTINE %(proc_prefix)sFIND_H(NHEL, H)
C     This subroutine parses HELC to find NHEL in it. It returns H, the position of this helicity state.
C     If it does not find the state in HELC, it returns 0
      IMPLICIT NONE
      INTEGER NLOOPFLOWS, NEXTERNAL
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d, NEXTERNAL=%(nexternal)i)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      LOGICAL equal
      INTEGER I, H, J
      INTEGER HELC(NEXTERNAL,NCOMB), NHEL(NEXTERNAL)
      COMMON/%(proc_prefix)sHELCONFIGS/HELC

      H = 0
      DO I = 1, NCOMB
        DO J = 1, NEXTERNAL
          equal = (HELC(J,I) == NHEL(J))
          if (.not.equal) exit
        ENDDO
        if (equal) then
        H = I
        exit
        endif
      ENDDO
      RETURN
      END SUBROUTINE

      SUBROUTINE %(proc_prefix)sSTORE_JAMP(H)
C     This subroutine stores the colour flows of the helicity configuration
C     HELC(:,H) in JAMPL_ALL (and the Born ones in JAMPB_ALL)
      IMPLICIT NONE
      INTEGER NLOOPFLOWS, NEXTERNAL
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d, NEXTERNAL=%(nexternal)i)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER    NLOOPAMPSO
      PARAMETER (NLOOPAMPSO=%(nAmpSO)d)
      INTEGER H

      DOUBLE COMPLEX JAMPL_ALL(3,NLOOPFLOWS,NLOOPAMPSO,NCOMB)
      COMMON/%(proc_prefix)sDENSITY_MATRIX/JAMPL_ALL
      COMPLEX*16 JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      COMMON/%(proc_prefix)sJAMPL/JAMPL
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
      DOUBLE COMPLEX JAMPB_ALL(NBORNFLOWS,NLOOPAMPSO,NCOMB)
      COMMON/%(proc_prefix)sDENSITY_MATRIX_B/JAMPB_ALL
      COMPLEX*16 JAMPB(NBORNFLOWS,NLOOPAMPSO)
      COMMON/%(proc_prefix)sJAMPB/JAMPB
## }

      JAMPL_ALL(:,:,:,H) = JAMPL(:,:,:)
## if(not LoopInduced){
      JAMPB_ALL(:,:,H) = JAMPB(:,:)
## }
      RETURN
      END SUBROUTINE


      SUBROUTINE %(proc_prefix)sNORMALISE_RHO(RHO, N_COMB_RHO, RHO_NORM)
C     This subroutine stores the JAMP with helicity state HELC(:,H) in JAMPL_ALL
      IMPLICIT NONE
      INTEGER NLOOPFLOWS, NEXTERNAL
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d, NEXTERNAL=%(nexternal)i)
      INTEGER N_COMB_RHO
      INTEGER    NLOOPAMPSO, NSQUAREDSO
      PARAMETER (NLOOPAMPSO=%(nAmpSO)d, NSQUAREDSO=%(nSquaredSO)d)

      DOUBLE COMPLEX RHO(N_COMB_RHO * (N_COMB_RHO + 1)/2)
      DOUBLE COMPLEX RHO_NORM(N_COMB_RHO * (N_COMB_RHO + 1)/2)
      DOUBLE PRECISION TRACE
      INTEGER DIAG_INDICES(N_COMB_RHO), COUNTER, IDIAG, I, SOINDEX, J

      COUNTER = 0
      DO IDIAG = 1, N_COMB_RHO
        DIAG_INDICES(IDIAG) = (IDIAG-1) * N_COMB_RHO - COUNTER + 1
        COUNTER = COUNTER + (IDIAG-1)
      ENDDO

      TRACE = 0
      DO I = 1, N_COMB_RHO
        TRACE = TRACE + DREAL(RHO(DIAG_INDICES(I)))
      ENDDO

      DO J = 1, N_COMB_RHO * (N_COMB_RHO + 1)/2
        RHO_NORM(J) = RHO(J) / TRACE
      ENDDO


      RETURN
      END SUBROUTINE



      SUBROUTINE %(proc_prefix)sCOMPUTE_INTER_JAMP(INTER,H1,H2)
C     Density-matrix element between the helicity configurations H1 and H2,
C     from the colour flows stored by STORE_JAMP, divided by the averaging
C     and symmetry factor IDEN as ANS of SLOOPMATRIX.
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER HEL_MULT
      PARAMETER (HEL_MULT=1)
      INTEGER IDEN
      %(den_factor_line)s
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
## }
C     
C     ARGUMENTS
C     
      DOUBLE COMPLEX INTER(0:3,0:NSQUAREDSO)
      INTEGER H1, H2
C     
C     GLOBAL VARIABLES
C     
      DOUBLE COMPLEX JAMPL_ALL(3,NLOOPFLOWS,NLOOPAMPSO,NCOMB)
      COMMON/%(proc_prefix)sDENSITY_MATRIX/JAMPL_ALL
## if(not LoopInduced){
      DOUBLE COMPLEX JAMPB_ALL(NBORNFLOWS,NLOOPAMPSO,NCOMB)
      COMMON/%(proc_prefix)sDENSITY_MATRIX_B/JAMPB_ALL
## }
C     
C     LOCAL VARIABLES
C     
      INTEGER I, SOINDEX

## if(LoopInduced){
      CALL %(proc_prefix)sDO_COMPUTE_INTER_JAMP(INTER,HEL_MULT,JAMPL_ALL(1,1,1,H1),JAMPL_ALL(1,1,1,H2))
## } else {
      CALL %(proc_prefix)sDO_COMPUTE_INTER_JAMP(INTER,HEL_MULT,JAMPL_ALL(1,1,1,H1),JAMPL_ALL(1,1,1,H2),JAMPB_ALL(1,1,H1),JAMPB_ALL(1,1,H2))
## }

      DO SOINDEX = 0, NSQUAREDSO
        DO I=0, 3
          INTER(I, SOINDEX) = INTER(I, SOINDEX)/IDEN
        ENDDO
      ENDDO
      
      RETURN
      END SUBROUTINE

## }
