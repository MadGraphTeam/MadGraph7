C     ===========================================
C     ===== Beginning of CutTools Interface =====
C     ===========================================
      SUBROUTINE FX_CTLOOP_CHECKED(NLOOPLINE,PL,M2L,RANK,RES,STABLE
     $ ,CT_STATUS)
      USE MOCK_STATE, ONLY: NORMALIZATION_COUNT
      USE CTS_CHECKED_REDUCTION, ONLY: CTSXCUT_CHECKED
      USE CTS_CHECKED_PROTOCOL, ONLY: CTS_CHECKED_REPORT
      USE CMS_SCALAR_OBSERVER, ONLY: CMS_OK

C     

C     
C     Interface between MG5 and CutTools.
C     

C     
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      LOGICAL CHECKPCONSERVATION
      PARAMETER (CHECKPCONSERVATION=.TRUE.)
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0*
     $ *2))
C     
C     ARGUMENTS 
C     
      INTEGER NLOOPLINE, RANK
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)
      REAL*8 REF_P
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 M2LCT(0:NLOOPLINE-1)
      COMPLEX*16 RES(3)
      LOGICAL STABLE
C     
C     LOCAL VARIABLES 
C     
      COMPLEX*16 R1, ACC
      INTEGER I, J, K
      TYPE(CTS_CHECKED_REPORT) CT_STATUS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/FX_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION

      LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
     $ ,COLLIERINIT
      COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT
     $ ,NINJAINIT,COLLIERINIT
C     
C     EXTERNAL FUNCTIONS
C     
      EXTERNAL FX_LOOPNUM
      EXTERNAL FX_MPLOOPNUM
C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'coupl.inc'
      INTEGER CTMODE
      REAL*8 LSCALE
      COMMON/FX_CT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/FX_LOOP/ID,SQSOINDEX,R

C     ----------
C     BEGIN CODE
C     ----------

C     INITIALIZE CUTTOOLS IF NEEDED
      IF (CTINIT) THEN
        CTINIT=.FALSE.
        CALL FX_INITCT()
        CALL CITE('Ossola:2007ax','one-loop reduction with CutTools')
      ENDIF

C     YOU CAN FIND THE DETAILS ABOUT THE DIFFERENT CTMODE AT THE
C      BEGINNING OF THE FILE CTS_CUTS.F90 IN THE CUTTOOLS DISTRIBUTION

C     CONVERT THE MASSES TO BE COMPLEX
      DO I=1,NLOOPLINE
        M2LCT(I-1)=M2L(I)
      ENDDO

C     CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO CT CONVENTIONS
      DO I=0,3
        ABSPCT(I)=0.D0
        DO J=0,(NLOOPLINE-1)
          PCT(I,J)=0.D0
        ENDDO
      ENDDO
      DO I=0,3
        DO J=1,NLOOPLINE
          PCT(I,0)=PCT(I,0)+PL(I,J)
          ABSPCT(I)=ABSPCT(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P = MAX(ABSPCT(0), ABSPCT(1),ABSPCT(2),ABSPCT(3))
      DO I=0,3
        ABSPCT(I) = MAX(REF_P*1E-6, ABSPCT(I))
      ENDDO
      IF (CHECKPCONSERVATION.AND.REF_P.GT.1D-8) THEN
        IF ((PCT(0,0)/ABSPCT(0)).GT.1.D-6) THEN
          WRITE(*,*) 'energy is not conserved (flag:CT95)',PCT(0,0)
          STOP 'energy is not conserved (flag:CT95)'
        ELSEIF ((PCT(1,0)/ABSPCT(1)).GT.1.D-6) THEN
          WRITE(*,*) 'px is not conserved (flag:CT95)',PCT(1,0)
          STOP 'px is not conserved (flag:CT95)'
        ELSEIF ((PCT(2,0)/ABSPCT(2)).GT.1.D-6) THEN
          WRITE(*,*) 'py is not conserved (flag:CT95)',PCT(2,0)
          STOP 'py is not conserved (flag:CT95)'
        ELSEIF ((PCT(3,0)/ABSPCT(3)).GT.1.D-6) THEN
          WRITE(*,*) 'pz is not conserved (flag:CT95)',PCT(3,0)
          STOP 'pz is not conserved (flag:CT95)'
        ENDIF
      ENDIF
      DO I=0,3
        DO J=1,(NLOOPLINE-1)
          DO K=1,J
            PCT(I,J)=PCT(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO

      CALL CTSXCUT_CHECKED(CTMODE,LSCALE,MU_R,NLOOPLINE,FX_LOOPNUM
     $ ,FX_MPLOOPNUM,RANK,PCT,M2LCT,RES,ACC,R1,STABLE,CT_STATUS)
C     The existing FPE flag is a hard-refusal latch, not an invented
C      IEEE exception.
C     Select the active MadLoop pass; CT_STATUS records actual
C      internal precision.
      IF(CT_STATUS%STATUS.NE.CMS_OK.OR..NOT.CT_STATUS%RESULT_VALID)
     $  THEN
        STABLE=.FALSE.
        IF(CTMODE.GE.4) THEN
          FPE_IN_QP_REDUCTION=.TRUE.
        ELSE
          FPE_IN_DP_REDUCTION=.TRUE.
        ENDIF
        RETURN
      ENDIF

      NORMALIZATION_COUNT=NORMALIZATION_COUNT+1
      RES(1)=NORMALIZATION*2.0D0*DBLE(RES(1))
      RES(2)=NORMALIZATION*2.0D0*DBLE(RES(2))
      RES(3)=NORMALIZATION*2.0D0*DBLE(RES(3))
C     WRITE(*,*) 'CutTools: Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END


