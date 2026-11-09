IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1)) THEN
  STAB_INDEX=STAB_INDEX+1  
  IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLReductionLib(I_LIB))) THEN
C   Only run over the reduction algorithms which support quadruple precision
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

  IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLReductionLib(I_LIB))) THEN	
      BASIC_CT_MODE=4
  ELSE
      BASIC_CT_MODE=1
  ENDIF

C BEGINNING OF THE DEFINITIONS OF THE DIFFERENT EVALUATION METHODS

  IF(.NOT.EVAL_DONE(2)) THEN
	EVAL_DONE(2)=.TRUE.
	IF(LOOPLIBS_DIRECTEST(MLReductionLib(I_LIB)))THEN 
	  CTMODE=BASIC_CT_MODE+1
	  goto 300
    ELSE
C     If some TIR library would not support the loop direction test (they all do for now), then we would just copy the answer from mode 1 and carry on.
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
  
  IF(.NOT.EVAL_DONE(3).AND. ((DOING_QP_EVALS.AND.NRotations_QP.GE.1).OR.((.NOT.DOING_QP_EVALS).AND.NRotations_DP.GE.1)) ) THEN
	EVAL_DONE(3)=.TRUE.
	CALL %(proc_prefix)sROTATE_PS(PS,P,1)
	IF (DOING_QP_EVALS) CALL %(proc_prefix)sMP_ROTATE_PS(MP_PS,MP_P,1)
	goto 200
  ENDIF

  IF(.NOT.EVAL_DONE(4).AND. ((DOING_QP_EVALS.AND.NRotations_QP.GE.2).OR.((.NOT.DOING_QP_EVALS).AND.NRotations_DP.GE.2)) ) THEN
	EVAL_DONE(4)=.TRUE.
	CALL %(proc_prefix)sROTATE_PS(PS,P,2)
	IF (DOING_QP_EVALS) CALL %(proc_prefix)sMP_ROTATE_PS(MP_PS,MP_P,2)
	goto 200
  ENDIF

  CALL %(proc_prefix)sROTATE_PS(PS,P,0)
  IF (DOING_QP_EVALS) CALL %(proc_prefix)sMP_ROTATE_PS(MP_PS,MP_P,0)

C END OF THE DEFINITIONS OF THE DIFFERENT EVALUATION METHODS

  IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLReductionLib(I_LIB))) THEN
    CALL %(proc_prefix)sCOMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)
C   If a floating point exception was encountered during the reduction,
C   the result cannot be trusted at all and we hardset all accuracies to 1.0
    IF(FPE_IN_QP_REDUCTION) THEN
      DO I=0,NSQUAREDSO
        ACC(I)=1.0d0
      ENDDO
    ENDIF
	DO I=0,NSQUAREDSO
	  ACCURACY(I)=ACC(I)
	ENDDO
	RET_CODE_H=3
	RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.TRUE.,.TRUE.)	
	IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
	  I_QP_LIB=I_QP_LIB+1
	  IF(I_QP_LIB.GT.QP_NLOOPLIB.OR.INDEX_QP_TOOLS(I_QP_LIB).EQ.0)THEN
	  RET_CODE_H=4
	  RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.TRUE.,.FALSE.)
	  NEPS=NEPS+1
      CALL %(proc_prefix)sCOMPUTE_ACCURACY(DP_RES,N_DP_EVAL,TEMP1,TEMP)
      CALL %(proc_prefix)sCOMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)
	  IF(NEPS.LE.10) THEN	  
        WRITE(*,*) '##W03 WARNING An unstable PS point was',       ' detected.'
        IF(FPE_IN_QP_REDUCTION) THEN
          WRITE(*,*) '## The last QP reduction was deemed unstable because a floating point exception was encountered.'        
        ENDIF
		IF (NSQUAREDSO.NE.1) THEN
		  WRITE(*,*) '##Accuracies for each split order, starting with the summed case'
	      WRITE(*,*) '##DP accuracies (for each split order): ',(TEMP1(I),I=0,NSQUAREDSO)
	      WRITE(*,*) '##QP accuracies (for each split order): ',(ACC(I),I=0,NSQUAREDSO)
		ELSE
	      WRITE(*,*) '##DP accuracy: ',TEMP1(1)
	      WRITE(*,*) '##QP accuracy: ',ACC(1)
		ENDIF
	    DO J=0,NSQUAREDSO
		  IF (NSQUAREDSO.NE.1.OR.J.NE.0) THEN
		    IF (J.EQ.0) THEN
		      WRITE(*,*) 'Details for all split orders summed :'
	        ELSE
		      WRITE(*,*) 'Details for split order index : ',J
		    ENDIF
	        WRITE(*,*) 'Best estimate (fin,1eps,2eps):',(ANS(I,J),I=1,3)
	        WRITE(*,*) 'Finite double precision evaluations :',(DP_RES(1,J,I),I=1,N_DP_EVAL)
	        WRITE(*,*) 'Finite quad   precision evaluations :',(QP_RES(1,J,I),I=1,N_QP_EVAL)
		  ENDIF
		ENDDO
	    WRITE(*,*) 'PS point specification :'
	    WRITE(*,*) 'Renormalization scale MU_R=',MU_R	
	    DO I=1,NEXTERNAL
          WRITE (*,'(i2,1x,4e27.17)') i, P(0,i),P(1,i),P(2,i),P(3,i) 
        ENDDO
	  ENDIF
	  IF(NEPS.EQ.10) THEN
	    WRITE(*,*) 'Further output of the details of these unstable PS points will now be suppressed.'
	  ENDIF
    ELSE
C    A new reduction tool will be used. Reinitialize the FPE flags.
	 FPE_IN_DP_REDUCTION=.False.
	 FPE_IN_QP_REDUCTION=.False.
     I_LIB=INDEX_QP_TOOLS(I_QP_LIB)
     EVAL_DONE(1)=.TRUE.
     DO I=2,MAXSTABILITYLENGTH
     EVAL_DONE(I)=.FALSE.
     ENDDO
     STAB_INDEX=0
     IF(NRotations_QP.GE.1)THEN
	goto 200
     ELSE
	goto 300
     ENDIF
    ENDIF
    ENDIF
  ELSEIF(.NOT.DOING_QP_EVALS)THEN
    CALL %(proc_prefix)sCOMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)
C   If a floating point exception was encountered during the reduction,
C   the result cannot be trusted at all and we hardset all accuracies to 1.0
    IF(FPE_IN_DP_REDUCTION) THEN
      DO I=0,NSQUAREDSO
        ACC(I)=1.0d0
      ENDDO
    ENDIF
## if(ninjars_available) {
C         One explicit compensated DP retry with the same stability checks.
          IF(NRS_RETRY_MODE) THEN
            NRS_COMP_ACC=MAXVAL(ACC)
            NRS_RETRY_MODE=.FALSE.
            IF(MAXVAL(ACC).LT.MLSTABTHRES.AND..NOT.FPE_IN_DP_REDUCTION) THEN
              NRS_RETRY_PASS=NRS_RETRY_PASS+1
            ELSE
              NRS_RETRY_FAIL=NRS_RETRY_FAIL+1
            ENDIF
          ENDIF
          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
            IF(MLREDUCTIONLIB(I_LIB).EQ.8.AND..NOT.NRS_RETRY_USED) THEN
              NRS_NATIVE_ACC=MAXVAL(ACC)
              IF(.NOT.FPE_IN_DP_REDUCTION.AND.ALL(IEEE_IS_FINITE(ACC)).AND.ALL(IEEE_IS_FINITE(DP_RES(:,:,1:N_DP_EVAL)))) THEN
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
C           Mode is off before advancing a backend or entering genuine QP.
            NRS_RETRY_MODE=.FALSE.
## } else {
	IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
## }
	  I_LIB=I_LIB+1
	  IF((I_LIB.GT.NLOOPLIB.OR.MLReductionLib(I_LIB).EQ.0).AND.QP_TOOLS_AVAILABLE)THEN
	        I_LIB=INDEX_QP_TOOLS(1)
C       A new reduction tool will be used. Reinitialize the FPE flags.
	    FPE_IN_DP_REDUCTION=.False.
	    FPE_IN_QP_REDUCTION=.False.	    
		I_QP_LIB=1
		DOING_QP_EVALS=.TRUE.
	  	EVAL_DONE(1)=.TRUE.
	  	DO I=2,MAXSTABILITYLENGTH
        	EVAL_DONE(I)=.FALSE.
      		ENDDO
	  	STAB_INDEX=0
		CTMODE=4
	  	goto 200
	  ELSEIF(I_LIB.LE.NLOOPLIB.AND.MLReductionLib(I_LIB).GT.0)THEN
C       A new reduction tool will be used. Reinitialize the FPE flags.
	    FPE_IN_DP_REDUCTION=.False.
	    FPE_IN_QP_REDUCTION=.False.
	    EVAL_DONE(1)=.TRUE.
		DO I=2,MAXSTABILITYLENGTH
		EVAL_DONE(I)=.FALSE.
		ENDDO
		STAB_INDEX=0
		IF(NRotations_DP.GE.1)THEN
		   goto 200
		ELSE
		   goto 300
		ENDIF
	  ELSE
	     DO I=0,NSQUAREDSO
	       ACCURACY(I)=ACC(I)
	     ENDDO
	     RET_CODE_H=4
	     RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.FALSE.,.FALSE.)
	     NEPS=NEPS+1
             IF(NEPS.LE.10) THEN
               WRITE(*,*) '##W03 WARNING An unstable PS point was',       ' detected.'
	           WRITE(*,*) '##W03 WARNING No quadruple precision will be used.'
               IF(FPE_IN_DP_REDUCTION) THEN
                 WRITE(*,*) '## The last DP reduction was deemed unstable because a floating point exception was encountered.'        
               ENDIF
               CALL %(proc_prefix)sCOMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)
               IF (NSQUAREDSO.NE.1) THEN
                  WRITE(*,*) 'Accuracies for each split order, starting with the summed case'
              	  WRITE(*,*) 'DP accuracies (for each split order): ',(ACC(I),I=0,NSQUAREDSO)
               ELSE
              WRITE(*,*) 'DP accuracy: ',ACC(1)
	        ENDIF
              DO J=0,NSQUAREDSO
                  IF (NSQUAREDSO.NE.1.OR.J.NE.0) THEN
                    IF (J.EQ.0) THEN
                      WRITE(*,*) 'Details for all split orders summed :'
                    ELSE
                      WRITE(*,*) 'Details for split order index : ',J
                    ENDIF
                    WRITE(*,*) 'Best estimate (fin,1eps,2eps):',(ANS(I,J),I=1,3)
               	     WRITE(*,*) 'Finite double precision evaluations :',(DP_RES(1,J,I),I=1,N_DP_EVAL)
                  ENDIF
                ENDDO
            WRITE(*,*) 'PS point specification :'
            WRITE(*,*) 'Renormalization scale MU_R=',MU_R
            DO I=1,NEXTERNAL
            WRITE (*,'(i2,1x,4e27.17)') i, P(0,i),P(1,i),P(2,i),P(3,i)
            ENDDO
            ENDIF
            IF(NEPS.EQ.10) THEN
             WRITE(*,*) 'Further output of the details of these unstable PS points will now be suppressed.'
            ENDIF
	  ENDIF
    ELSE
	  DO I=0,NSQUAREDSO
	    ACCURACY(I)=ACC(I)
	  ENDDO
      	  RET_CODE_H=2
	  RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.FALSE.,.TRUE.)	
	  ENDIF	
  ENDIF
ELSE
  RET_CODE_H=1
  DO I=0,NSQUAREDSO
    ACCURACY(I)=-1.0d0
  ENDDO
  RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.FALSE.,.FALSE.)
ENDIF

 9999 CONTINUE

C Finalize the return code
IF (MP_DONE_ONCE) THEN
  RET_CODE_T=2
ELSE
  RET_CODE_T=1
ENDIF
IF(CHECKPHASE.OR..NOT.HELDOUBLECHECKED) THEN
  RET_CODE_H=1
  RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.FALSE.,.FALSE.)
  RET_CODE_T=RET_CODE_T+2
  DO I=0,NSQUAREDSO
    ACCURACY(I)=-1.0d0
  ENDDO
ENDIF

