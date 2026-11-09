C     Standalone tensor diagnostic, never an incomplete physical amplitude.
      PROGRAM TIRUST_POLE_DIAGNOSTIC
      USE, INTRINSIC :: IEEE_ARITHMETIC
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NF,ID,N,R,NC,I,J,L,MODE,CTMODE,FINITE,LOOPS
      INTEGER*8 T0,T1,RATE
      REAL*8 SHIFTS(0:3,6),PL0(0:3,6),PL(0:3,6),MU2,MR,MI
      REAL*16 QPL(0:3,6),RESULT(6),SECONDS
      COMPLEX*16 MASSES0(6),MASSES(6),COEFS(0:LOOPMAXCOEFS-1,3)
      COMPLEX*32 QCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE,NANFINITE
      CHARACTER*1024 ARG
      CALL SETPARA('param_card.dat')
      CALL DEFAULTPARAM()
      MLReductionLib=0
      MLReductionLib(1)=9
      TIRustGlobalCache=0
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      OPEN(71,FILE=TRIM(ARG),STATUS='OLD')
      OPEN(72,FILE='pole-values.dat',STATUS='REPLACE')
      READ(71,*) NF
      DO ID=1,NF
        READ(71,*) N,R,MU2,LOOPS,MODE,CTMODE
        IF(N.LT.1.OR.N.GT.6.OR.R.LT.0.OR.R.GT.6) STOP 2
        NC=(R+1)*(R+2)*(R+3)*(R+4)/24
        IF(NC.GT.LOOPMAXCOEFS.OR.LOOPS.LT.1) STOP 3
        DO I=1,N
          READ(71,*) SHIFTS(:,I)
        ENDDO
        DO I=1,N
          READ(71,*) MR,MI
          MASSES0(I)=CMPLX(MR,MI,KIND=8)
        ENDDO
        PL0=0D0
        DO I=1,N-1
          PL0(:,I)=SHIFTS(:,I+1)-SHIFTS(:,I)
        ENDDO
        PL0(:,N)=SHIFTS(:,1)-SHIFTS(:,N)
        MU_R=SQRT(MU2)
        CALL UPDATE_AS_PARAM()
        LoopPoleMode=MODE
        DO FINITE=1,0,-1
          TIRustComputeFinite=FINITE.EQ.1
C         One untimed warm call; plan initialization is excluded from timing.
          DO L=0,LOOPS
            PL=PL0
            QPL=REAL(PL0,KIND=16)
            MASSES=MASSES0
            COEFS=0
            QCOEFS=0
            CALL @PREFIX@CLEAR_TIRUST_CACHE()
            IF(L.EQ.1) CALL SYSTEM_CLOCK(T0,RATE)
            IF(CTMODE.EQ.1) THEN
              CALL @PREFIX@TIRUST_TENSOR_DIAGNOSTIC(CTMODE,N,R,PL,MASSES,COEFS,STABLE)
              QCOEFS=COEFS
            ELSE IF(CTMODE.EQ.4) THEN
              CALL @PREFIX@MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,N,R,QPL,MASSES,QCOEFS,STABLE)
            ELSE
              STOP 4
            ENDIF
            IF(.NOT.STABLE) EXIT
          ENDDO
          CALL SYSTEM_CLOCK(T1)
          SECONDS=0
          IF(STABLE) SECONDS=REAL(T1-T0,KIND=16)/REAL(RATE,KIND=16)
          NANFINITE=ALL(IEEE_IS_NAN(REAL(QCOEFS(0:NC-1,1),KIND=16)))
          WRITE(72,'(A,5I8,2L3,I8,ES46.35E4)') 'F ',ID,CTMODE,MODE,FINITE,LOOPS,STABLE,NANFINITE,NC,SECONDS
          IF(STABLE) THEN
            DO I=0,NC-1
              DO J=1,3
                RESULT(2*J-1)=REAL(QCOEFS(I,J),KIND=16)
                RESULT(2*J)=AIMAG(QCOEFS(I,J))
              ENDDO
              WRITE(72,'(A,5I8,6ES46.35E4)') 'C ',ID,CTMODE,MODE,FINITE,I,RESULT
            ENDDO
          ENDIF
          FLUSH(72)
        ENDDO
      ENDDO
      CLOSE(71)
      CLOSE(72)
      CALL @PREFIX@FINALIZE_TIRUST()
      END
