C     Execute the real host automatic-rescue branch with a synthetic
C      callback.
C     PL_FIXTURE and NUM_FIXTURE have original, rotation1, rotation2
C      columns.
      INTEGER(C_INT) FUNCTION TIRUST_HOST_RESCUE(THRESHOLD,ROTATIONS,
     & PROJECTION,PL_FIXTURE,MASSES,MU2,NUM_FIXTURE,RESULT,OUT_ACCURACY,
     & RET_CODE,N_CALLS,MODES,ROTATION_LOG,STATUSES,EVALUATIONS,
     & DP_ACCURACY,QP_ACCURACY) BIND(C,NAME='tirust_host_rescue')
      USE, INTRINSIC :: ISO_C_BINDING
      USE, INTRINSIC :: IEEE_ARITHMETIC
      IMPLICIT NONE
      REAL(C_DOUBLE), VALUE :: THRESHOLD,MU2
      INTEGER(C_INT), VALUE :: ROTATIONS,PROJECTION
      REAL(C_DOUBLE), INTENT(IN) :: PL_FIXTURE(0:3,4,0:2)
      COMPLEX(C_DOUBLE_COMPLEX), INTENT(IN) :: MASSES(4)
      COMPLEX(C_DOUBLE_COMPLEX), INTENT(IN) :: NUM_FIXTURE(126,0:2)
      REAL(C_DOUBLE), INTENT(OUT) :: RESULT(3),OUT_ACCURACY
      REAL(C_DOUBLE), INTENT(OUT) :: DP_ACCURACY,QP_ACCURACY
      INTEGER(C_INT), INTENT(OUT) :: RET_CODE,N_CALLS,MODES(8)
      INTEGER(C_INT), INTENT(OUT) :: ROTATION_LOG(8),STATUSES(8)
      REAL(C_DOUBLE), INTENT(OUT) :: EVALUATIONS(3,8)
      INTEGER NSQUAREDSO,NEXTERNAL
      PARAMETER(NSQUAREDSO=1,NEXTERNAL=4)
      INTEGER NLOOPLIB
      PARAMETER (NLOOPLIB=9)
C     CutTools, Ninja and NinjaRS can provide QP when installed with
C     support
      INTEGER QP_NLOOPLIB,TIRUST_QP_COUNT
      PARAMETER (TIRUST_QP_COUNT=1)
      PARAMETER (QP_NLOOPLIB=3+TIRUST_QP_COUNT)


      INTEGER MAXSTABILITYLENGTH,I_LIB,I_QP_LIB
      INTEGER INDEX_QP_TOOLS(QP_NLOOPLIB+1)
      INTEGER CTMODE,STAB_INDEX,BASIC_CT_MODE,N_DP_EVAL,N_QP_EVAL
      INTEGER I,J,K,NEPS,RET_CODE_H,RET_CODE_T,RET_CODE_U
      INTEGER SET_RET_CODE_U,CURRENT_ROTATION,STATUS,NDP,NQP
      LOGICAL EVAL_DONE(4),DOING_QP_EVALS,QP_TOOLS_AVAILABLE
      LOGICAL LOOPLIBS_AVAILABLE(NLOOPLIB)
      DATA LOOPLIBS_AVAILABLE/.TRUE.,.FALSE.,.TRUE.,.FALSE.,.FALSE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE./
      COMMON/ML5_0_LOOPLIBS_AV/ LOOPLIBS_AVAILABLE
C     A FLAG TO DENOTE WHETHER THE CORRESPONDING DIRECTION TESTS
C     AVAILABLE OR NOT IN THE LOOPLIBS
      LOGICAL LOOPLIBS_DIRECTEST(NLOOPLIB)
      DATA LOOPLIBS_DIRECTEST /9*.TRUE./
C     Specifying for which reduction tool quadruple precision is
C     available.
C     The index 0 is dummy and simply means that the corresponding
C     loop_library is not available
C     in which case neither is its quadruple precision version.
      LOGICAL LOOPLIBS_QPAVAILABLE(0:NLOOPLIB)
      DATA LOOPLIBS_QPAVAILABLE(0:5)/.FALSE.,.TRUE.,4*.FALSE./
      DATA LOOPLIBS_QPAVAILABLE(6)/.TRUE./
      DATA LOOPLIBS_QPAVAILABLE(7)/.FALSE./
      DATA LOOPLIBS_QPAVAILABLE(8)/.TRUE./
      DATA LOOPLIBS_QPAVAILABLE(9)/.TRUE./

      LOGICAL CHECKPHASE,HELDOUBLECHECKED,MP_DONE_ONCE
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED
      INTEGER NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL
      INTEGER NRS_RETRY_HARD_SKIP
      REAL*8 NRS_NATIVE_ACC,NRS_COMP_ACC
      REAL*8 ANS(0:3,0:1),ACC(0:1),ACCURACY(0:1)
      REAL*8 DP_RES(3,0:1,4),QP_RES(3,0:1,4)
      REAL*8 TEMP(0:3,0:1),TEMP1(0:1),MU_R
      REAL*8 PS(0:3,4),P(0:3,4)
      REAL*16 MP_PS(0:3,4),MP_P(0:3,4)
      COMPLEX(C_DOUBLE_COMPLEX) COMPLEX_RESULT(3)
      LOGICAL CHOSEN_SO_CONFIGS(1)
      COMMON/AUDIT_STABILITY_TESTS/MAXSTABILITYLENGTH
      COMMON/AUDIT_CHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS
      COMMON/AUDIT_I_LIB/I_LIB
      COMMON/AUDIT_COUPL/MU_R
      COMMON/AUDIT_ROTATION_STATE/CURRENT_ROTATION
      INCLUDE 'MadLoopParams.inc'
      INTERFACE
      INTEGER(C_INT) FUNCTION NATIVE_CONTRACT(MODE,PL,M,SCALE,
     &   COEFS,OUT) BIND(C,NAME='tirust_stability_contract')
      IMPORT C_INT,C_DOUBLE,C_DOUBLE_COMPLEX
      INTEGER(C_INT),VALUE :: MODE
      REAL(C_DOUBLE),INTENT(IN) :: PL(0:3,4)
      COMPLEX(C_DOUBLE_COMPLEX),INTENT(IN) :: M(4),COEFS(126)
      REAL(C_DOUBLE),VALUE :: SCALE
      COMPLEX(C_DOUBLE_COMPLEX),INTENT(OUT) :: OUT(3)
      END FUNCTION
      SUBROUTINE NATIVE_CLEAR() BIND(C,NAME='tirust_stability_clear')
      END SUBROUTINE
      END INTERFACE
      TIRUST_HOST_RESCUE=2_C_INT
      RESULT=0D0
      OUT_ACCURACY=-1D0
      DP_ACCURACY=-1D0
      QP_ACCURACY=-1D0
      RET_CODE=0
      N_CALLS=0
      MODES=0
      ROTATION_LOG=0
      STATUSES=0
      EVALUATIONS=0D0
      IF(ROTATIONS.LT.0.OR.ROTATIONS.GT.2) RETURN
      IF(PROJECTION.LT.0.OR.PROJECTION.GT.1) RETURN
      IF(.NOT.IEEE_IS_FINITE(THRESHOLD).OR.THRESHOLD.LE.0D0) RETURN
      CALL DEFAULTPARAM()
      MLREDUCTIONLIB=0
      MLREDUCTIONLIB(1)=9
      CTMODERUN=-1
      MLSTABTHRES=THRESHOLD
      NROTATIONS_DP=ROTATIONS
      NROTATIONS_QP=ROTATIONS
      MAXSTABILITYLENGTH=4
      CHOSEN_SO_CONFIGS=.TRUE.
      I_LIB=1
      I_QP_LIB=1
      QP_TOOLS_AVAILABLE=.FALSE.
      INDEX_QP_TOOLS(1:QP_NLOOPLIB+1)=0
C     Explain an explicit NinjaRS-only request before dropping
C     unavailable tools.
      IF(MLREDUCTIONLIB(1).EQ.8.AND.MLREDUCTIONLIB(2)
     $   .EQ.0.AND..NOT.LOOPLIBS_AVAILABLE(8))THEN
      WRITE(*,*) 'NinjaRS (MLReductionLib=8) is unavailable in'
     $     //' this generated process.'
      WRITE(*,*) 'Run install ninjars in MadGraph, then regenerate'
     $     //' and rebuild this process.'
      STOP 'Requested NinjaRS backend is unavailable.'
      ENDIF
      IF(MLREDUCTIONLIB(1).EQ.9.AND.MLREDUCTIONLIB(2)
     $   .EQ.0.AND..NOT.LOOPLIBS_AVAILABLE(9))THEN
      WRITE(*,*) 'TIRust (MLReductionLib=9) is unavailable in this'
     $     //' generated process.'
      WRITE(*,*) 'Run install tirust, then regenerate and rebuild'
     $     //' this process.'
      WRITE(*,*) 'Requested TIRust backend is unavailable.'
      STOP 1
      ENDIF
C     SKIP THE ONES THAT NOT AVAILABLE
      J=1
      DO I=1,NLOOPLIB
        IF(MLREDUCTIONLIB(J).EQ.0)EXIT
        IF(.NOT.LOOPLIBS_AVAILABLE(MLREDUCTIONLIB(J)))THEN
          MLREDUCTIONLIB(J:NLOOPLIB-1)=MLREDUCTIONLIB(J+1:NLOOPLIB)
          MLREDUCTIONLIB(NLOOPLIB)=0
        ELSE
          J=J+1
        ENDIF
      ENDDO
      IF(MLREDUCTIONLIB(1).EQ.0)THEN
        STOP 'No available loop reduction lib is provided. Make sure'
     $     //' MLReductionLib is correct.'
      ENDIF
      J=0
      DO I=1,NLOOPLIB
        IF(LOOPLIBS_QPAVAILABLE(MLREDUCTIONLIB(I)))THEN
          J=J+1
          IF(.NOT.QP_TOOLS_AVAILABLE) THEN
            QP_TOOLS_AVAILABLE=.TRUE.
          ENDIF
          INDEX_QP_TOOLS(J)=I
        ENDIF
      ENDDO


      CHECKPHASE=.FALSE.
      HELDOUBLECHECKED=.TRUE.
      MP_DONE_ONCE=.FALSE.
      DOING_QP_EVALS=.FALSE.
      FPE_IN_DP_REDUCTION=.FALSE.
      FPE_IN_QP_REDUCTION=.FALSE.
      EVAL_DONE=.FALSE.
      EVAL_DONE(1)=.TRUE.
      STAB_INDEX=0
      CTMODE=1
      CURRENT_ROTATION=0
      CALL AUDIT_SET_N_EVALS(N_DP_EVAL,N_QP_EVAL)
      NEPS=0
      NDP=0
      NQP=0
      DP_RES=0D0
      QP_RES=0D0
      ACC=0D0
      ACCURACY=-1D0
      ANS=0D0
      TEMP=0D0
      TEMP1=0D0
      NRS_RETRY_MODE=.FALSE.
      NRS_RETRY_USED=.FALSE.
      NRS_RETRY_COUNT=0
      NRS_RETRY_PASS=0
      NRS_RETRY_FAIL=0
      NRS_RETRY_HARD_SKIP=0
      NRS_NATIVE_ACC=0D0
      NRS_COMP_ACC=0D0
      PS=PL_FIXTURE(:,:,0)
      P=PS
      MP_PS=REAL(PS,KIND=16)
      MP_P=MP_PS
      CALL NATIVE_CLEAR()
C     These two labels are the physics-evaluation entry points
C      targeted by
C     the untouched production control block below.
 200  CONTINUE
 300  CONTINUE
      IF(N_CALLS.GE.8) RETURN
      STATUS=NATIVE_CONTRACT(INT(CTMODE,C_INT),
     & PL_FIXTURE(:,:,CURRENT_ROTATION),MASSES,MU2,
     & NUM_FIXTURE(:,CURRENT_ROTATION),COMPLEX_RESULT)
C     The standalone diagnostic resets DEFAULTPARAM; restore caller
C      policy.
      MLREDUCTIONLIB=0
      MLREDUCTIONLIB(1)=9
      MLSTABTHRES=THRESHOLD
      CTMODERUN=-1
      NROTATIONS_DP=ROTATIONS
      NROTATIONS_QP=ROTATIONS
      N_CALLS=N_CALLS+1
      MODES(N_CALLS)=CTMODE
      ROTATION_LOG(N_CALLS)=CURRENT_ROTATION
      STATUSES(N_CALLS)=STATUS
      IF(CTMODE.GE.4) THEN
        MP_DONE_ONCE=.TRUE.
        NQP=NQP+1
        FPE_IN_QP_REDUCTION=FPE_IN_QP_REDUCTION.OR.STATUS.NE.0
      ELSE
        NDP=NDP+1
        FPE_IN_DP_REDUCTION=FPE_IN_DP_REDUCTION.OR.STATUS.NE.0
      ENDIF
      IF(PROJECTION.EQ.0) THEN
        ANS(1:3,0)=REAL(COMPLEX_RESULT,KIND=8)
      ELSE
        ANS(1:3,0)=AIMAG(COMPLEX_RESULT)
      ENDIF
      ANS(1:3,1)=ANS(1:3,0)
      EVALUATIONS(:,N_CALLS)=ANS(1:3,0)
      IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1))
     $  THEN
        STAB_INDEX=STAB_INDEX+1
        IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLREDUCTIONLIB(I_LIB)
     $   )) THEN
C         Only run over the reduction algorithms which support
C          quadruple precision
          DO I=0,NSQUAREDSO
            DO K=1,3
              QP_RES(K,I,STAB_INDEX)=ANS(K,I)
            ENDDO
          ENDDO
        ELSE
          DO I=0,NSQUAREDSO
            DO K=1,3
              DP_RES(K,I,STAB_INDEX)=ANS(K,I)
            ENDDO
          ENDDO
        ENDIF

        IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLREDUCTIONLIB(I_LIB)
     $   )) THEN
          BASIC_CT_MODE=4
        ELSE
          BASIC_CT_MODE=1
        ENDIF

C       BEGINNING OF THE DEFINITIONS OF THE DIFFERENT EVALUATION
C        METHODS

        IF(.NOT.EVAL_DONE(2)) THEN
          EVAL_DONE(2)=.TRUE.
          IF(LOOPLIBS_DIRECTEST(MLREDUCTIONLIB(I_LIB)))THEN
            CTMODE=BASIC_CT_MODE+1
            GOTO 300
          ELSE
C           If some TIR library would not support the loop direction
C            test (they all do for now), then we would just copy the
C            answer from mode 1 and carry on.
            STAB_INDEX=STAB_INDEX+1
            IF(DOING_QP_EVALS)THEN
              DO I=0,NSQUAREDSO
                DO K=1,3
                  QP_RES(K,I,STAB_INDEX)=ANS(K,I)
                ENDDO
              ENDDO
            ELSE
              DO I=0,NSQUAREDSO
                DO K=1,3
                  DP_RES(K,I,STAB_INDEX)=ANS(K,I)
                ENDDO
              ENDDO
            ENDIF
          ENDIF
        ENDIF

        CTMODE=BASIC_CT_MODE

        IF(.NOT.EVAL_DONE(3).AND.
     $    ((DOING_QP_EVALS.AND.NROTATIONS_QP.GE.1)
     $   .OR.((.NOT.DOING_QP_EVALS).AND.NROTATIONS_DP.GE.1)) ) THEN
          EVAL_DONE(3)=.TRUE.
          CALL AUDIT_ROTATE_PS(PS,P,1)
          IF (DOING_QP_EVALS) CALL AUDIT_MP_ROTATE_PS(MP_PS,MP_P,1)
          GOTO 200
        ENDIF

        IF(.NOT.EVAL_DONE(4).AND.
     $    ((DOING_QP_EVALS.AND.NROTATIONS_QP.GE.2)
     $   .OR.((.NOT.DOING_QP_EVALS).AND.NROTATIONS_DP.GE.2)) ) THEN
          EVAL_DONE(4)=.TRUE.
          CALL AUDIT_ROTATE_PS(PS,P,2)
          IF (DOING_QP_EVALS) CALL AUDIT_MP_ROTATE_PS(MP_PS,MP_P,2)
          GOTO 200
        ENDIF

        CALL AUDIT_ROTATE_PS(PS,P,0)
        IF (DOING_QP_EVALS) CALL AUDIT_MP_ROTATE_PS(MP_PS,MP_P,0)

C       END OF THE DEFINITIONS OF THE DIFFERENT EVALUATION METHODS

        IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLREDUCTIONLIB(I_LIB)
     $   )) THEN
          CALL AUDIT_COMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)
C         If a floating point exception was encountered during the
C          reduction,
C         the result cannot be trusted at all and we hardset all
C          accuracies to 1.0
          IF(FPE_IN_QP_REDUCTION) THEN
            DO I=0,NSQUAREDSO
              ACC(I)=1.0D0
            ENDDO
          ENDIF
          DO I=0,NSQUAREDSO
            ACCURACY(I)=ACC(I)
          ENDDO
          RET_CODE_H=3
          RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.TRUE.
     $     ,.TRUE.)
          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
            I_QP_LIB=I_QP_LIB+1
            IF(I_QP_LIB.GT.QP_NLOOPLIB.OR.INDEX_QP_TOOLS(I_QP_LIB)
     $       .EQ.0)THEN
              RET_CODE_H=4
              RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.TRUE.
     $         ,.FALSE.)
              NEPS=NEPS+1
              CALL AUDIT_COMPUTE_ACCURACY(DP_RES,N_DP_EVAL,TEMP1,TEMP)
              CALL AUDIT_COMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)
              IF(NEPS.LE.10) THEN
                WRITE(*,*) '##W03 WARNING An unstable PS point was',  
     $                ' detected.'
                IF(FPE_IN_QP_REDUCTION) THEN
                  WRITE(*,*) '## The last QP reduction was deemed'
     $             //' unstable because a floating point exception was'
     $             //' encountered.'
                ENDIF
                IF (NSQUAREDSO.NE.1) THEN
                  WRITE(*,*) '##Accuracies for each split order,'
     $             //' starting with the summed case'
                  WRITE(*,*) '##DP accuracies (for each split order):'
     $             //' ',(TEMP1(I),I=0,NSQUAREDSO)
                  WRITE(*,*) '##QP accuracies (for each split order):'
     $             //' ',(ACC(I),I=0,NSQUAREDSO)
                ELSE
                  WRITE(*,*) '##DP accuracy: ',TEMP1(1)
                  WRITE(*,*) '##QP accuracy: ',ACC(1)
                ENDIF
                DO J=0,NSQUAREDSO
                  IF (NSQUAREDSO.NE.1.OR.J.NE.0) THEN
                    IF (J.EQ.0) THEN
                      WRITE(*,*) 'Details for all split orders summed'
     $                 //' :'
                    ELSE
                      WRITE(*,*) 'Details for split order index : ',J
                    ENDIF
                    WRITE(*,*) 'Best estimate (fin,1eps,2eps):',(ANS(I
     $               ,J),I=1,3)
                    WRITE(*,*) 'Finite double precision evaluations :'
     $               ,(DP_RES(1,J,I),I=1,N_DP_EVAL)
                    WRITE(*,*) 'Finite quad   precision evaluations :'
     $               ,(QP_RES(1,J,I),I=1,N_QP_EVAL)
                  ENDIF
                ENDDO
                WRITE(*,*) 'PS point specification :'
                WRITE(*,*) 'Renormalization scale MU_R=',MU_R
                DO I=1,NEXTERNAL
                  WRITE (*,'(i2,1x,4e27.17)') I, P(0,I),P(1,I),P(2,I)
     $             ,P(3,I)
                ENDDO
              ENDIF
              IF(NEPS.EQ.10) THEN
                WRITE(*,*) 'Further output of the details of these'
     $           //' unstable PS points will now be suppressed.'
              ENDIF
            ELSE
C             A new reduction tool will be used. Reinitialize the FPE
C              flags.
              FPE_IN_DP_REDUCTION=.FALSE.
              FPE_IN_QP_REDUCTION=.FALSE.
              I_LIB=INDEX_QP_TOOLS(I_QP_LIB)
              EVAL_DONE(1)=.TRUE.
              DO I=2,MAXSTABILITYLENGTH
                EVAL_DONE(I)=.FALSE.
              ENDDO
              STAB_INDEX=0
              IF(NROTATIONS_QP.GE.1)THEN
                GOTO 200
              ELSE
                GOTO 300
              ENDIF
            ENDIF
          ENDIF
        ELSEIF(.NOT.DOING_QP_EVALS)THEN
          CALL AUDIT_COMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)
C         If a floating point exception was encountered during the
C          reduction,
C         the result cannot be trusted at all and we hardset all
C          accuracies to 1.0
          IF(FPE_IN_DP_REDUCTION) THEN
            DO I=0,NSQUAREDSO
              ACC(I)=1.0D0
            ENDDO
          ENDIF
C         One explicit compensated DP retry with the same stability
C          checks.
          IF(NRS_RETRY_MODE) THEN
            NRS_COMP_ACC=MAXVAL(ACC)
            NRS_RETRY_MODE=.FALSE.
            IF(MAXVAL(ACC).LT.MLSTABTHRES.AND..NOT.FPE_IN_DP_REDUCTION)
     $        THEN
              NRS_RETRY_PASS=NRS_RETRY_PASS+1
            ELSE
              NRS_RETRY_FAIL=NRS_RETRY_FAIL+1
            ENDIF
          ENDIF
          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
            IF(MLREDUCTIONLIB(I_LIB).EQ.8.AND..NOT.NRS_RETRY_USED) THEN
              NRS_NATIVE_ACC=MAXVAL(ACC)
              IF(.NOT.FPE_IN_DP_REDUCTION.AND.ALL(IEEE_IS_FINITE(ACC))
     $         .AND.ALL(IEEE_IS_FINITE(DP_RES(:,:,1:N_DP_EVAL)))) THEN
                NRS_RETRY_USED=.TRUE.
                NRS_RETRY_MODE=.TRUE.
                NRS_RETRY_COUNT=NRS_RETRY_COUNT+1
                EVAL_DONE(1)=.TRUE.
                DO I=2,MAXSTABILITYLENGTH
                  EVAL_DONE(I)=.FALSE.
                ENDDO
                STAB_INDEX=0
                CTMODE=1
                DP_RES=0D0
                IF(NROTATIONS_DP.GE.1) THEN
                  GOTO 200
                ELSE
                  GOTO 300
                ENDIF
              ELSE
                NRS_RETRY_HARD_SKIP=NRS_RETRY_HARD_SKIP+1
              ENDIF
            ENDIF
C           Mode is off before advancing a backend or entering genuine
C            QP.
            NRS_RETRY_MODE=.FALSE.
            I_LIB=I_LIB+1
            IF((I_LIB.GT.NLOOPLIB.OR.MLREDUCTIONLIB(I_LIB).EQ.0)
     $       .AND.QP_TOOLS_AVAILABLE)THEN
              I_LIB=INDEX_QP_TOOLS(1)
C             A new reduction tool will be used. Reinitialize the FPE
C              flags.
              FPE_IN_DP_REDUCTION=.FALSE.
              FPE_IN_QP_REDUCTION=.FALSE.
              I_QP_LIB=1
              DOING_QP_EVALS=.TRUE.
              EVAL_DONE(1)=.TRUE.
              DO I=2,MAXSTABILITYLENGTH
                EVAL_DONE(I)=.FALSE.
              ENDDO
              STAB_INDEX=0
              CTMODE=4
              GOTO 200
            ELSEIF(I_LIB.LE.NLOOPLIB.AND.MLREDUCTIONLIB(I_LIB).GT.0)
     $       THEN
C             A new reduction tool will be used. Reinitialize the FPE
C              flags.
              FPE_IN_DP_REDUCTION=.FALSE.
              FPE_IN_QP_REDUCTION=.FALSE.
              EVAL_DONE(1)=.TRUE.
              DO I=2,MAXSTABILITYLENGTH
                EVAL_DONE(I)=.FALSE.
              ENDDO
              STAB_INDEX=0
              IF(NROTATIONS_DP.GE.1)THEN
                GOTO 200
              ELSE
                GOTO 300
              ENDIF
            ELSE
              DO I=0,NSQUAREDSO
                ACCURACY(I)=ACC(I)
              ENDDO
              RET_CODE_H=4
              RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.FALSE.
     $         ,.FALSE.)
              NEPS=NEPS+1
              IF(NEPS.LE.10) THEN
                WRITE(*,*) '##W03 WARNING An unstable PS point was',  
     $                ' detected.'
                WRITE(*,*) '##W03 WARNING No quadruple precision will'
     $           //' be used.'
                IF(FPE_IN_DP_REDUCTION) THEN
                  WRITE(*,*) '## The last DP reduction was deemed'
     $             //' unstable because a floating point exception was'
     $             //' encountered.'
                ENDIF
                CALL AUDIT_COMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)
                IF (NSQUAREDSO.NE.1) THEN
                  WRITE(*,*) 'Accuracies for each split order,'
     $             //' starting with the summed case'
                  WRITE(*,*) 'DP accuracies (for each split order): '
     $             ,(ACC(I),I=0,NSQUAREDSO)
                ELSE
                  WRITE(*,*) 'DP accuracy: ',ACC(1)
                ENDIF
                DO J=0,NSQUAREDSO
                  IF (NSQUAREDSO.NE.1.OR.J.NE.0) THEN
                    IF (J.EQ.0) THEN
                      WRITE(*,*) 'Details for all split orders summed'
     $                 //' :'
                    ELSE
                      WRITE(*,*) 'Details for split order index : ',J
                    ENDIF
                    WRITE(*,*) 'Best estimate (fin,1eps,2eps):',(ANS(I
     $               ,J),I=1,3)
                    WRITE(*,*) 'Finite double precision evaluations :'
     $               ,(DP_RES(1,J,I),I=1,N_DP_EVAL)
                  ENDIF
                ENDDO
                WRITE(*,*) 'PS point specification :'
                WRITE(*,*) 'Renormalization scale MU_R=',MU_R
                DO I=1,NEXTERNAL
                  WRITE (*,'(i2,1x,4e27.17)') I, P(0,I),P(1,I),P(2,I)
     $             ,P(3,I)
                ENDDO
              ENDIF
              IF(NEPS.EQ.10) THEN
                WRITE(*,*) 'Further output of the details of these'
     $           //' unstable PS points will now be suppressed.'
              ENDIF
            ENDIF
          ELSE
            DO I=0,NSQUAREDSO
              ACCURACY(I)=ACC(I)
            ENDDO
            RET_CODE_H=2
            RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.FALSE.
     $       ,.TRUE.)
          ENDIF
        ENDIF
      ELSE
        RET_CODE_H=1
        DO I=0,NSQUAREDSO
          ACCURACY(I)=-1.0D0
        ENDDO
        RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.FALSE.
     $   ,.FALSE.)
      ENDIF

 9999 CONTINUE

C     Finalize the return code
      IF (MP_DONE_ONCE) THEN
        RET_CODE_T=2
      ELSE
        RET_CODE_T=1
      ENDIF
      IF(CHECKPHASE.OR..NOT.HELDOUBLECHECKED) THEN
        RET_CODE_H=1
        RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.FALSE.
     $   ,.FALSE.)
        RET_CODE_T=RET_CODE_T+2
        DO I=0,NSQUAREDSO
          ACCURACY(I)=-1.0D0
        ENDDO
      ENDIF


C     Report the branch's saved accuracy and actual H/T/U finalization.
      RESULT=ANS(1:3,1)
      OUT_ACCURACY=ACCURACY(1)
      RET_CODE=100*RET_CODE_H+10*RET_CODE_T+RET_CODE_U
      IF(NDP.GE.2) THEN
        CALL AUDIT_COMPUTE_ACCURACY(DP_RES,NDP,TEMP1,TEMP)
        DP_ACCURACY=TEMP1(1)
      ENDIF
      IF(NQP.GE.2) THEN
        CALL AUDIT_COMPUTE_ACCURACY(QP_RES,NQP,TEMP1,TEMP)
        QP_ACCURACY=TEMP1(1)
      ENDIF
      CALL NATIVE_CLEAR()
      TIRUST_HOST_RESCUE=0_C_INT
      END

C     Geometry is supplied explicitly; these callbacks only select its
C      index.
      SUBROUTINE AUDIT_ROTATE_PS(PS,P,NUMBER)
      IMPLICIT NONE
      REAL*8 PS(0:3,4),P(0:3,4)
      INTEGER NUMBER,CURRENT_ROTATION
      COMMON/AUDIT_ROTATION_STATE/CURRENT_ROTATION
      CURRENT_ROTATION=NUMBER
      P=PS
      END

      SUBROUTINE AUDIT_MP_ROTATE_PS(PS,P,NUMBER)
      IMPLICIT NONE
      REAL*16 PS(0:3,4),P(0:3,4)
      INTEGER NUMBER,CURRENT_ROTATION
      COMMON/AUDIT_ROTATION_STATE/CURRENT_ROTATION
      CURRENT_ROTATION=NUMBER
      P=PS
      END
      INTEGER FUNCTION SET_RET_CODE_U(MLRED,DOING_QP,STABLE)
C     
C     This functions returns the value of U
C     
C     
C     U == 0
C     Not stable.
C     U == 1
C     Stable with CutTools in double precision.
C     U == 2
C     Stable with PJFry++ (H=2), or NinjaRS in quadruple precision
C      (H=3).
C     U == 3
C     Stable with IREGI (H=2), or TIRust in quadruple precision (H=3).
C     U == 4
C     Stable with Golem95.
C     U == 5
C     Stable with Samurai.
C     U == 6
C     Stable with Ninja in double precision.
C     U == 7
C     Stable with COLLIER.
C     U == 8
C     Stable with Ninja in quadruple precision (H=3), or NinjaRS in
C      double precision (H=2).
C     U == 9
C     Stable with CutTools in quadruple precision (H=3), or TIRust in
C      double precision (H=2).
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
C     
C     ARGUMENTS
C     
      INTEGER MLRED
      LOGICAL DOING_QP,STABLE
C     
C     LOCAL VARIABLES
C     
C     
C     FUNCTION
C     
C     
C     BEGIN CODE
C     
      IF(.NOT.STABLE)THEN
        SET_RET_CODE_U=0
        RETURN
      ENDIF
      IF(DOING_QP)THEN
        IF(MLRED.EQ.1)THEN
          SET_RET_CODE_U=9
          RETURN
        ELSEIF(MLRED.EQ.6)THEN
          SET_RET_CODE_U=8
          RETURN
        ELSEIF(MLRED.EQ.8)THEN
          SET_RET_CODE_U=2
          RETURN
        ELSEIF(MLRED.EQ.9)THEN
          SET_RET_CODE_U=3
          RETURN
        ELSE
          STOP 'Requested library has no quadruple precision adapter'
        ENDIF
      ENDIF
      IF(MLRED.GE.1.AND.MLRED.LE.9)THEN
        SET_RET_CODE_U=MLRED
      ELSE
        STOP 'Unknown MadLoop reduction library; supported IDs are 1'
     $   //' through 9'
      ENDIF
      END

      SUBROUTINE AUDIT_SET_N_EVALS(N_DP_EVALS,N_QP_EVALS)

      IMPLICIT NONE
      INTEGER N_DP_EVALS, N_QP_EVALS

      INCLUDE 'MadLoopParams.inc'

      IF(CTMODERUN.LE.-1) THEN
        N_DP_EVALS=2+NROTATIONS_DP
        N_QP_EVALS=2+NROTATIONS_QP
      ELSE
        N_DP_EVALS=1
        N_QP_EVALS=1
      ENDIF

      IF(N_DP_EVALS.GT.20.OR.N_QP_EVALS.GT.20) THEN
        WRITE(*,*) 'ERROR:: Increase hardcoded maxstabilitylength.'
        STOP 1
      ENDIF

      END


