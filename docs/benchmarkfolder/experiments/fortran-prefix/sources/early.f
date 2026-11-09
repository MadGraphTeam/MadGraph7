      SUBROUTINE EARLY_TIRLOOP(I_SQSO,I_LOOPGROUP,I_LIB,NLOOPLINE,PL
     $ ,M2L,RANK,RES,STABLE)
C     

C     
C     Interface between MG5 and TIR.
C     

C     
C     
C     CONSTANTS 
C     
      INTEGER NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=1)
C     These are constants related to the split orders
      INTEGER NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=6)
      LOGICAL CHECKPCONSERVATION
      PARAMETER (CHECKPCONSERVATION=.TRUE.)
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0*
     $ *2))
      INTEGER TIR_CACHE_SIZE
      INCLUDE 'tir_cache_size.inc'
C     
C     ARGUMENTS 
C     
      INTEGER I_SQSO,I_LOOPGROUP,I_LIB
      INTEGER NLOOPLINE, RANK
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)
      REAL*8 REF_P
      REAL*8 PDEN(0:3,NLOOPLINE-1)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 M2LCT(0:NLOOPLINE-1)
      COMPLEX*16 RES(3)
      LOGICAL STABLE
C     
C     LOCAL VARIABLES 
C     
      INTEGER I, J, K
      INTEGER NLOOPCOEFS
      LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
     $ ,COLLIERINIT
      COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT
     $ ,NINJAINIT,COLLIERINIT

C     This variable will be used to detect changes in the TIR library
C      used so as to force the reset of the TIR filter.
      INTEGER LAST_LIB_USED
      DATA LAST_LIB_USED/-1/
      INTEGER LAST_POLE_MODE
      LOGICAL LAST_UV_POLES,LAST_IR_POLES,UV_POLES,IR_POLES
      DATA LAST_POLE_MODE/-2/
      DATA LAST_UV_POLES,LAST_IR_POLES/.TRUE.,.TRUE./

      COMPLEX*16 TIRCOEFS(0:LOOPMAXCOEFS-1,3)
     $ ,TIRCOEFSERRORS(0:LOOPMAXCOEFS-1,3)
      COMPLEX*16 PJCOEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER TIR_CACHE_INDEX
      COMPLEX*16 TIRCOEFS_CACHING(3,0:LOOPMAXCOEFS-1,NLOOPGROUPS
     $ ,TIR_CACHE_SIZE)
      SAVE TIRCOEFS_CACHING
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/EARLY_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION
C     
C     EXTERNAL FUNCTIONS
C     
      INTEGER EARLY_TIRCACHE_INDEX
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'

      INCLUDE 'coupl.inc'
      INTEGER CTMODE
      REAL*8 LSCALE
      COMMON/EARLY_CT/LSCALE,CTMODE

      INTEGER ID,R
      COMMON/EARLY_LOOP/ID,R

C     The argument ILIB is the TIR library to be used for that
C      specific library.
      INTEGER LIBINDEX
      COMMON/EARLY_I_LIB/LIBINDEX

      COMPLEX*16 LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
      COMMON/EARLY_LCOEFS/LOOPCOEFS
C     The index 0 of the second dimensions is filled with .false. as
C      it makes the implementation simpler this way when the user sets
C      TIR_CACHE_SIZE to be 0.
      LOGICAL TIR_DONE(NLOOPGROUPS,0:TIR_CACHE_SIZE)
      COMMON/EARLY_TIRCACHING/TIR_DONE
C     ----------
C     BEGIN CODE
C     ----------

C     Reject unavailable OPP backends which reach the TIR fallback.
C      Check before
C     consulting the cache so an unsupported ID can never reuse old
C      coefficients.
      SELECT CASE(MLREDUCTIONLIB(I_LIB))
      CASE(2,3,4,7)
        CONTINUE
      CASE(9)
        WRITE(*,*) 'TIRust unavailable: install tirust, regenerate and'
     $   //' rebuild.'
        STOP 1
      CASE DEFAULT
        IF(MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          WRITE(*,*) 'NinjaRS (MLReductionLib=8) is unavailable in'
     $     //' this generated process.'
          WRITE(*,*) 'Run install ninjars in MadGraph, then regenerate'
     $     //' and rebuild this process.'
        ELSE
          WRITE(*,*) 'ERROR:: Unsupported TIR reduction library:'
     $     ,MLREDUCTIONLIB(I_LIB)
          WRITE(*,*) 'Select an available MLReductionLib and'
     $     //' regenerate and rebuild this process.'
        ENDIF
        STOP 1
      END SELECT

C     Initialize for the very first time ML is called LAST_ILIB with
C      the first ILIB used.
      CALL ML_LOOP_POLE_FLAGS(UV_POLES,IR_POLES)
      IF(LAST_POLE_MODE.NE.LOOPPOLEMODE.OR.(LAST_UV_POLES.NEQV.UV_POLES)
     $ .OR.(LAST_IR_POLES.NEQV.IR_POLES)) THEN
        IF(LAST_POLE_MODE.NE.-2) CALL EARLY_CLEAR_TIR_CACHE()
        LAST_POLE_MODE=LOOPPOLEMODE
        LAST_UV_POLES=UV_POLES
        LAST_IR_POLES=IR_POLES
      ENDIF
      IF(LAST_LIB_USED.EQ.-1) THEN
        LAST_LIB_USED = MLREDUCTIONLIB(LIBINDEX)
      ELSE
C       We changed the TIR library so we must refresh the cache.
        IF(MLREDUCTIONLIB(LIBINDEX).NE.LAST_LIB_USED) THEN
          LAST_LIB_USED = MLREDUCTIONLIB(LIBINDEX)
          CALL EARLY_CLEAR_TIR_CACHE()
        ENDIF
      ENDIF

      IF (MLREDUCTIONLIB(I_LIB).EQ.4) THEN
C       Golem95 not available
        WRITE(*,*) 'ERROR:: Golem95 is not interfaced.'
        STOP
      ENDIF

      IF(MLREDUCTIONLIB(I_LIB).NE.9) THEN
C       INITIALIZE TIR IF NEEDED
        IF (TIRINIT) THEN
          TIRINIT=.FALSE.
          CALL EARLY_INITTIR()
        ENDIF

C       CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO CT CONVENTIONS
        DO I=0,3
          ABSPCT(I) = 0.D0
          PCT(I,0)=0.D0
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
            WRITE(*,*) 'energy is not conserved (flag: TIR)',PCT(0,0)
            STOP 'energy is not conserved (flag: TIR)'
          ELSEIF ((PCT(1,0)/ABSPCT(1)).GT.1.D-6) THEN
            WRITE(*,*) 'px is not conserved (flag: TIR)',PCT(1,0)
            STOP 'px is not conserved (flag: TIR)'
          ELSEIF ((PCT(2,0)/ABSPCT(2)).GT.1.D-6) THEN
            WRITE(*,*) 'py is not conserved (flag: TIR)',PCT(2,0)
            STOP 'py is not conserved (flag: TIR)'
          ELSEIF ((PCT(3,0)/ABSPCT(3)).GT.1.D-6) THEN
            WRITE(*,*) 'pz is not conserved (flag: TIR)',PCT(3,0)
            STOP 'pz is not conserved (flag: TIR)'
          ENDIF
        ENDIF
      ENDIF
C     NUMBER OF INDEPENDENT LOOPCOEFS FOR RANK=RANK
      NLOOPCOEFS=0
      DO I=0,RANK
        NLOOPCOEFS=NLOOPCOEFS+(3+I)*(2+I)*(1+I)/6
      ENDDO
      IF (TIR_CACHE_SIZE.EQ.0) THEN
        TIR_CACHE_INDEX = 0
      ELSE
        TIR_CACHE_INDEX = EARLY_TIRCACHE_INDEX(CTMODE)
C       This way whenever the CTMode exceeds the cache size it rolls
C        back to the beginning and MadLoop will have taken care of
C        having re-initialized TIR_DONE between labels 200 and 205 in
C        loop_matrix.f
        TIR_CACHE_INDEX = MOD(TIR_CACHE_INDEX-1,TIR_CACHE_SIZE)+1
      ENDIF

      IF(.NOT.TIR_DONE(I_LOOPGROUP,TIR_CACHE_INDEX))THEN
        IF(MLREDUCTIONLIB(I_LIB).NE.9) THEN
C         CONVERT THE MASSES TO BE COMPLEX
          DO I=1,NLOOPLINE
            M2LCT(I-1)=M2L(I)
          ENDDO

          PCT(:,1:NLOOPLINE-1)=0.D0
          DO I=0,3
            DO J=1,(NLOOPLINE-1)
              DO K=1,J
                PCT(I,J)=PCT(I,J)+PL(I,K)
              ENDDO
            ENDDO
          ENDDO

          DO I=0,3
            DO J=1,(NLOOPLINE-1)
              PDEN(I,J)=PCT(I,J)
            ENDDO
          ENDDO
        ENDIF
        SELECT CASE(MLREDUCTIONLIB(I_LIB))
        CASE(2)
C         PJFry++
          WRITE(*,*) 'ERROR:: PJFRY++ is not interfaced.'
          STOP
        CASE(3)
C         IREGI
          WRITE(*,*) 'ERROR:: IREGI is not interfaced.'
          STOP
        CASE(7)
C         COLLIER
          CALL EARLY_COLLIERLOOP(CTMODE,NLOOPLINE,RANK,PL,PDEN,M2L
     $     ,TIRCOEFS,TIRCOEFSERRORS)
C         Shift the TIR coefficients by the corresponding COLLIER
C          error if in CTMODE 2.
          IF (COLLIERUSEINTERNALSTABILITYTEST.AND.CTMODE.EQ.2) THEN
C           We add here the numerical inaccuracies linearly to be
C            conservative 
            DO I=1,3
              DO J=0,NLOOPCOEFS-1
                TIRCOEFS(J,I)=TIRCOEFS(J,I)+TIRCOEFSERRORS(J,I)
              ENDDO
            ENDDO
          ENDIF
        CASE(9)
          WRITE(*,*) 'TIRust unavailable: install tirust, regenerate'
     $     //' and rebuild.'
          STOP 1
        END SELECT
C       The zero index is dummy and means no caching at all since
C        TIR_CACHE_SIZE=0.
        IF(TIR_CACHE_INDEX.NE.0) THEN
          DO I=1,3
            DO J=0,NLOOPCOEFS-1
              TIRCOEFS_CACHING(I,J,I_LOOPGROUP,TIR_CACHE_INDEX)
     $         =TIRCOEFS(J,I)
            ENDDO
          ENDDO
          TIR_DONE(I_LOOPGROUP,TIR_CACHE_INDEX)=.TRUE.
        ENDIF
      ELSE
        DO I=1,3
          DO J=0,NLOOPCOEFS-1
            TIRCOEFS(J,I)=TIRCOEFS_CACHING(I,J,I_LOOPGROUP
     $       ,TIR_CACHE_INDEX)
          ENDDO
        ENDDO
      ENDIF
      DO I=1,3
        RES(I)=(0.0D0,0.0D0)
        DO J=0,NLOOPCOEFS-1
          RES(I)=RES(I)+LOOPCOEFS(J,I_LOOPGROUP)*TIRCOEFS(J,I)
        ENDDO
      ENDDO
      RES(1)=NORMALIZATION*RES(1)
      RES(2)=NORMALIZATION*RES(2)
      RES(3)=NORMALIZATION*RES(3)
C     IF(MLReductionLib(I_LIB).EQ.2) THEN
C     WRITE(*,*) 'PJFry: Loop ID',ID,' =',RES(1),RES(2),RES(3)
C     ELSEIF(MLReductionLib(I_LIB).EQ.3) THEN
C     WRITE(*,*) 'Iregi: Loop ID',ID,' =',RES(1),RES(2),RES(3)
C     ELSEIF(MLReductionLib(I_LIB).EQ.7) THEN
C     WRITE(*,*) 'COLLIER: Loop ID',ID,' =',RES(1),RES(2),RES(3)
C     ENDIF
      END


