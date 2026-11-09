C     Source-derived host policy fixture. See run_policy.py for slices.
      SUBROUTINE AUDIT_HOST()
      USE AUDIT_STATE
      USE, INTRINSIC :: IEEE_ARITHMETIC
      IMPLICIT NONE
      INTEGER I,J,SET_RET_CODE_U
      NROTATIONS_DP_BU = NROTATIONS_DP
      NROTATIONS_QP_BU = NROTATIONS_QP
      IF(MAX_SPIN_EXTERNAL_PARTICLE.GT.3.AND.USERHEL.NE.-1) THEN
        IF(.NOT.WARNED_LORENTZ_STAB_TEST_OFF) THEN
          WRITE(*,*) '##WARNING: Evaluation of a specific helicity was'
     $     //' asked for this PS point, and there is a spin-2 (or'
     $     //' higher) particle in the external states.'
          WRITE(*,*) '##WARNING: As a result, MadLoop disabled the'
     $     //' Lorentz rotation test for this phase-space point only.'
          WRITE(*,*) '##WARNING: Further warning of that type'
     $     //' suppressed.'
          WARNED_LORENTZ_STAB_TEST_OFF = .TRUE.
        ENDIF
        NROTATIONS_QP=0
        NROTATIONS_DP=0
        CALL ML5_0_SET_N_EVALS(N_DP_EVAL,N_QP_EVAL)
      ENDIF

      IF(NTRY.EQ.0) THEN
        HELDOUBLECHECKED=(.NOT.DOUBLECHECKHELICITYFILTER)
     $   .OR.(HELICITYFILTERLEVEL.EQ.0)
        OPEN(1, FILE=LOOPFILTERFN, ERR=100, STATUS='OLD',          
     $    ACTION='READ')
        DO J=1,NLOOPGROUPS
          READ(1,*,END=101) (GOODAMP(I,J),I=1,NSQUAREDSO)
        ENDDO
        GOTO 101
 100    CONTINUE
        FOUNDLOOPFILTER=.FALSE.
        DO J=1,NLOOPGROUPS
          DO I=1,NSQUAREDSO
            GOODAMP(I,J)=(.NOT.USELOOPFILTER)
          ENDDO
        ENDDO
 101    CONTINUE
        CLOSE(1)

        IF (.NOT.USELOOPFILTER) THEN
          DO J=1,NLOOPGROUPS
            DO I=1,NSQUAREDSO
              GOODAMP(I,J)=.TRUE.
            ENDDO
          ENDDO
        ENDIF

        IF (HELICITYFILTERLEVEL.EQ.0) THEN
          FOUNDHELFILTER=.TRUE.
          DO J=1,NCOMB
            GOODHEL(J)=1
          ENDDO
          GOTO 122
        ENDIF
        OPEN(1, FILE=HELFILTERFN, ERR=102, STATUS='OLD',          
     $    ACTION='READ')
        DO I=1,NCOMB
          READ(1,*,END=103) GOODHEL(I)
        ENDDO
        GOTO 103
 102    CONTINUE
        FOUNDHELFILTER=.FALSE.
        DO J=1,NCOMB
          GOODHEL(J)=1
        ENDDO
 103    CONTINUE
        CLOSE(1)
        IF (HELICITYFILTERLEVEL.EQ.1) THEN
C         We must make sure to remove the matching-helicity
C          optimisation, as requested by the user.
          DO J=1,NCOMB
            IF ((GOODHEL(J).GT.1).OR.(GOODHEL(J).LT.-HELOFFSET)) THEN
              GOODHEL(J)=1
            ENDIF
          ENDDO
        ENDIF
 122    CONTINUE
      ENDIF

C     First compute the borns, it will store them in ANS(0,I)
C     It is left untouched for the rest of MadLoop evaluation.
C     Notice that the squared split order index I does NOT
C     correspond to the same ordering of J for the loop ME 
C     results stored in ANS(K,J), with K in [1-3].The ordering 
C     of each can be obtained with ML5SOINDEX_FOR_SQUARED_ORDERS 
C     and SQSOINDEX_FROM_ORDERS for the loop ME and born ME 
C     respectively. For this to work, we assume that there is 
C     always more squared split orders in the loop ME than in the
C     born ME, which is practically always true. In any case, only
C     the split_order summed value I=0 is used in ML5 code.
      DO I=0,NSQSO_BORN
        BORNBUFF(I)=0.0D0
      ENDDO
      CALL ML5_0_BORN_WITH_RESCUE(P_USER,USERHEL,IC,BORNBUFF,
     $ BORN_RESCUED,BORN_OK)
      BORN_QP_CHECKED=.FALSE.
      IF (.NOT.BORN_OK) THEN
        NROTATIONS_DP=NROTATIONS_DP_BU
        NROTATIONS_QP=NROTATIONS_QP_BU
        MP_DONE=.FALSE.
        MP_DONE_ONCE=.TRUE.
        MP_PS_SET=.FALSE.
        RET_CODE_H=4
        RET_CODE_T=2
        RET_CODE_U=0
        ANS=IEEE_VALUE(0D0,IEEE_QUIET_NAN)
        ACCURACY=1D0
        IF(AUTOMATIC_CACHE_CLEARING) CALL ML5_0_CLEAR_CACHES()
        CALL ML5_0_SET_COLLIER_GLOBAL_CACHE(.FALSE.)
        RETURN
      ENDIF

      DO I=0,NSQSO_BORN
        ANS(0,I)=BORNBUFF(I)
      ENDDO

C     We set here the reference to the born summed over all split
C      orders
      REF=0.0D0
      DO I=1,NSQSO_BORN
        REF=REF+ANS(0,I)
      ENDDO

      MP_DONE=.FALSE.
      MP_DONE_ONCE=.FALSE.
      MP_PS_SET=.FALSE.
      STAB_INDEX=0
      DOING_QP_EVALS=.FALSE.
      EVAL_DONE(1)=.TRUE.
      DO I=2,MAXSTABILITYLENGTH
        EVAL_DONE(I)=.FALSE.
      ENDDO

      IF(.NOT.BYPASS_FILTERS) THEN
        NTRY=NTRY+1
      ENDIF

      IF (USER_STAB_PREC.GT.0.0D0) THEN
        MLSTABTHRES_BU=MLSTABTHRES
        MLSTABTHRES=USER_STAB_PREC
C       In the initialization, I cannot perform stability test and
C        therefore guarantee any precision
        CTMODEINIT_BU=CTMODEINIT
C       So either one choses quad precision directly
C       CTMODEINIT=4
C       Or, because this is very slow, we keep the original value. The
C        accuracy returned is -1 and tells the MC that he should not
C        trust the evaluation for checks.
        CTMODEINIT=CTMODEINIT_BU
      ENDIF

      IF(DONEHELDOUBLECHECK.AND.(.NOT.HELDOUBLECHECKED)) THEN
        HELDOUBLECHECKED=.TRUE.
        DONEHELDOUBLECHECK=.FALSE.
      ENDIF

      CHECKPHASE=(NTRY.LE.CHECKCYCLE).AND.(((.NOT.FOUNDLOOPFILTER)
     $ .AND.USELOOPFILTER).OR.(.NOT.FOUNDHELFILTER))

      IF (WRITEOUTFILTERS.AND.LOOPPOLEMODE.LT.0) THEN
        IF ((HELICITYFILTERLEVEL.NE.0).AND.(.NOT. CHECKPHASE)
     $   .AND.(.NOT.FOUNDHELFILTER)) THEN
          OPEN(1, FILE=HELFILTERFN, ERR=110, STATUS='NEW'
     $     ,ACTION='WRITE')
          DO I=1,NCOMB
            WRITE(1,*) GOODHEL(I)
          ENDDO
 110      CONTINUE
          CLOSE(1)
          FOUNDHELFILTER=.TRUE.
        ENDIF

        IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDLOOPFILTER)
     $   .AND.USELOOPFILTER) THEN
          OPEN(1, FILE=LOOPFILTERFN, ERR=111, STATUS='NEW'
     $     ,ACTION='WRITE')
          DO J=1,NLOOPGROUPS
            WRITE(1,*) (GOODAMP(I,J),I=1,NSQUAREDSO)
          ENDDO
 111      CONTINUE
          CLOSE(1)
          FOUNDLOOPFILTER=.TRUE.
        ENDIF
      ENDIF

      IF (BYPASS_FILTERS) THEN
        OLD_CHECKPHASE = CHECKPHASE
        OLD_HELDOUBLECHECKED = HELDOUBLECHECKED
        CHECKPHASE = .FALSE.
        HELDOUBLECHECKED = .TRUE.
        DO I=1,NCOMB
          OLD_GOODHEL(I)=GOODHEL(I)
          GOODHEL(I)=1
        ENDDO
        DO I=1,NSQUAREDSO
          DO J=1,NLOOPGROUPS
            OLD_GOODAMP(I,J)=GOODAMP(I,J)
            GOODAMP(I,J)=.TRUE.
          ENDDO
        ENDDO
      ENDIF

      IF(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
        HELPICKED=1
        CTMODE=CTMODEINIT
      ELSE
        IF (USERHEL.NE.-1) THEN
          IF(GOODHEL(USERHEL).EQ.-HELOFFSET) THEN
            DO I=0,NSQUAREDSO
              ANS(1,I)=0.0D0
              ANS(2,I)=0.0D0
              ANS(3,I)=0.0D0
            ENDDO
            GOTO 9999
          ENDIF
        ENDIF
        HELPICKED=USERHEL
        IF (CTMODERUN.NE.-1) THEN
          CTMODE=CTMODERUN
        ELSE
          CTMODE=1
        ENDIF
      ENDIF

C     TEST HOOK: no loop amplitude or native reducer is evaluated.
      CALL SIMULATED_LOOP_COMPLETION()
      IF (LOOP_MODE.NE.0) THEN
          IF(FPE_IN_QP_REDUCTION) THEN
            DO I=0,NSQUAREDSO
              ACC(I)=1.0D0
            ENDDO
          ENDIF
          DO I=0,NSQUAREDSO
            ACCURACY(I)=ACC(I)
          ENDDO
C         A current-call certificate, never inherited from COMMON status.
          IF(BORN_RESCUED) THEN
            BORN_QP_CHECKED=ALL(IEEE_IS_FINITE(ACC)).AND.
     $        ALL(IEEE_IS_FINITE(ANS)).AND.
     $        MAXVAL(ACC).LT.MLSTABTHRES.AND..NOT.FPE_IN_QP_REDUCTION
          ENDIF
          RET_CODE_H=3
          RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.TRUE.
     $     ,.TRUE.)
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

C     Finally for the summed result in ANS(1:3,0), make sure to only
C     consider the squared order asked for by the user.
C     Notice that this filtering using CHOSEN_SO_CONFIGS happens
C     here only while everywhere else one always considers the sum.
      DO J=1,3
        ANS(J,0)=0.0D0
      ENDDO
      DO I=1,NSQUAREDSO
        IF (CHOSEN_SO_CONFIGS(I)) THEN
          DO J=1,3
            ANS(J,0)=ANS(J,0)+ANS(J,I)
          ENDDO
        ENDIF
      ENDDO

C     Reinitialize the default threshold if it was specified by the
C      user
      IF (USER_STAB_PREC.GT.0.0D0) THEN
        MLSTABTHRES=MLSTABTHRES_BU
        CTMODEINIT=CTMODEINIT_BU
      ENDIF

C     Reinitialize the Lorentz test if it had been disabled because
C      spin-2 particles are in the external states.
      NROTATIONS_DP = NROTATIONS_DP_BU
      NROTATIONS_QP = NROTATIONS_QP_BU

C     Reinitialize the check phase logicals and the filters if check
C      bypassed
      IF (BYPASS_FILTERS) THEN
        CHECKPHASE = OLD_CHECKPHASE
        HELDOUBLECHECKED = OLD_HELDOUBLECHECKED
        DO I=1,NCOMB
          GOODHEL(I)=OLD_GOODHEL(I)
        ENDDO
        DO I=1,NSQUAREDSO
          DO J=1,NLOOPGROUPS
            GOODAMP(I,J)=OLD_GOODAMP(I,J)
          ENDDO
        ENDDO
      ENDIF

C     Make sure that we finish by emptying caches
      IF (AUTOMATIC_CACHE_CLEARING) THEN
        CALL ML5_0_CLEAR_CACHES()
      ENDIF

C     Now make sure to turn off the global COLLIER cache if applicable
      CALL ML5_0_SET_COLLIER_GLOBAL_CACHE(.FALSE.)

C     A recovered Born requires a genuine checked QP loop in this call.
C     Keep intentionally unchecked H1 behavior; reject stale checked codes.
      IF(BORN_RESCUED.AND.(RET_CODE_H.EQ.2.OR.RET_CODE_H.EQ.3)
     $   .AND..NOT.BORN_QP_CHECKED) THEN
        RET_CODE_H=4
        RET_CODE_U=0
        ACCURACY=1D0
      ENDIF

      END

