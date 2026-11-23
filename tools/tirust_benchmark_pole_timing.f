C     Bounded paired tensor timing; no numerator contraction or amplitude.
C     Scope 0: generated MadLoop adapter, MadLoop coefficient ordering.
C     Scope 1: direct public bridge, IREGI coefficient ordering. The omitted
C     bridge finite slot stays zero; only the adapter supplies NaN + 0i.
      PROGRAM TIRUST_POLE_TIMING
      USE MTIRUST, ONLY: TIRUST_TENSOR_EVALUATE,TIRUST_REGISTER_PLAN,
     $ TIRUST_CLEAR_CACHE,TIRUST_SUCCESS
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      INCLUDE 'loop_max_coefs.inc'
      INTEGER SCOPE,N,R,NC,CTMODE,NPAIRS,MAXCALIB,MAXLOOPS
      INTEGER I,J,FINITE,ITER,PAIR,POSITION,COMPLETED,REGSTATUS
      INTEGER REQUESTED,NEXTLOOPS,CHOSEN(0:1)
      REAL*8 MU2,MR,MI,TARGET,WALL,CPU,FACTOR
      REAL*8 SHIFTS(0:3,6),PL0(0:3,6),PL(0:3,6)
      REAL*8 INVARIANTS(6,6),DELTA(0:3)
      REAL*16 QPL(0:3,6),QSHIFTS(0:3,6),QINV(6,6),QD(0:3)
      REAL*16 VALUES(6)
      COMPLEX*16 MASSES0(6),MASSES(6)
      COMPLEX*16 COEFS(0:LOOPMAXCOEFS-1,3),RAW(210,4)
      COMPLEX*32 QMASSES(6),QCOEFS(0:LOOPMAXCOEFS-1,3)
      COMPLEX*32 QRAW(210,4),DISPLAY(0:209,3)
      LOGICAL STABLE,WARMOK,CALIBRATED(0:1)
      CHARACTER*1024 ARG

      CALL GET_COMMAND_ARGUMENT(1,ARG)
      OPEN(71,FILE=TRIM(ARG),STATUS='OLD')
      READ(71,*) SCOPE,N,R,MU2,CTMODE,NPAIRS,TARGET,
     $ MAXCALIB,MAXLOOPS
      IF(SCOPE.LT.0.OR.SCOPE.GT.1.OR.N.LT.1.OR.N.GT.6.OR.
     $ R.LT.0.OR.R.GT.6.OR.MU2.LE.0D0) STOP 2
      IF(CTMODE.NE.1.AND.CTMODE.NE.4) STOP 3
      IF(NPAIRS.LT.1.OR.TARGET.LE.0D0.OR.MAXCALIB.LT.1.OR.
     $ MAXLOOPS.LT.1) STOP 4
      NC=(R+1)*(R+2)*(R+3)*(R+4)/24
      IF(SCOPE.EQ.0.AND.NC.GT.LOOPMAXCOEFS) STOP 5
      SHIFTS=0D0
      MASSES0=CMPLX(0,0,KIND=8)
      DO I=1,N
        READ(71,*) SHIFTS(:,I)
      ENDDO
      DO I=1,N
        READ(71,*) MR,MI
        MASSES0(I)=CMPLX(MR,MI,KIND=8)
      ENDDO
      CLOSE(71)
      IF(ANY(SHIFTS(:,1).NE.0D0)) STOP 6
      CALL SETPARA('param_card.dat')
      CALL DEFAULTPARAM()
      MLReductionLib=0
      MLReductionLib(1)=9
      TIRustGlobalCache=0
      LoopPoleMode=0
      MU_R=SQRT(MU2)
      CALL UPDATE_AS_PARAM()
      PL0=0D0
      DO I=1,N-1
        PL0(:,I)=SHIFTS(:,I+1)-SHIFTS(:,I)
      ENDDO
      PL0(:,N)=SHIFTS(:,1)-SHIFTS(:,N)
      QSHIFTS=REAL(SHIFTS,KIND=16)
      INVARIANTS=0D0
      QINV=0.0_16
      DO I=1,N
        DO J=1,N
          DELTA=SHIFTS(:,I)-SHIFTS(:,J)
          INVARIANTS(I,J)=DELTA(0)**2-SUM(DELTA(1:3)**2)
          QD=QSHIFTS(:,I)-QSHIFTS(:,J)
          QINV(I,J)=QD(0)**2-SUM(QD(1:3)**2)
        ENDDO
      ENDDO
      REGSTATUS=TIRUST_SUCCESS
      IF(SCOPE.EQ.1) CALL TIRUST_REGISTER_PLAN(N,R,REGSTATUS)
      OPEN(72,FILE='pole-timing-values.dat',STATUS='REPLACE')
      WRITE(72,'(A,5I10,ES26.17E3,2I12)') 'H ',SCOPE,CTMODE,
     $ N,R,NC,TARGET,NPAIRS,MAXLOOPS
      FLUSH(72)
      COMPLETED=0
      WARMOK=.TRUE.
C     Both modes get their own untimed preflight even if one refuses.
      DO FINITE=1,0,-1
        CALL BATCH(0,1,FINITE,1,STABLE,WALL,CPU)
        WARMOK=WARMOK.AND.STABLE
      ENDDO
      IF(.NOT.WARMOK) GOTO 900
      CALIBRATED=.FALSE.
      CHOSEN=0
      DO FINITE=1,0,-1
        REQUESTED=1
        DO ITER=1,MAXCALIB
          CALL BATCH(1,ITER,FINITE,REQUESTED,STABLE,WALL,CPU)
          IF(.NOT.STABLE) GOTO 900
          IF(WALL.GE.0.75D0*TARGET.AND.
     $       WALL.LE.1.75D0*TARGET) THEN
            CHOSEN(FINITE)=REQUESTED
            CALIBRATED(FINITE)=.TRUE.
            EXIT
          ENDIF
C         Limit the real-valued scaling before integer conversion, avoiding
C         overflow on a timer-resolution zero or an extremely short call.
          IF(WALL.LE.0D0) THEN
            NEXTLOOPS=MAXLOOPS
          ELSE
            FACTOR=MIN(DBLE(MAXLOOPS),
     $        DBLE(REQUESTED)*TARGET/WALL)
            NEXTLOOPS=MAX(1,CEILING(FACTOR))
          ENDIF
          REQUESTED=NEXTLOOPS
        ENDDO
      ENDDO
      IF(.NOT.ALL(CALIBRATED)) GOTO 900
C     Adjacent pairs alternate full/poles, then poles/full. Counts are frozen
C     after calibration, independently for each finite-evaluation mode.
      DO PAIR=1,NPAIRS
        DO POSITION=1,2
          FINITE=MOD(PAIR+POSITION+1,2)
          CALL BATCH(2,PAIR,FINITE,CHOSEN(FINITE),STABLE,WALL,CPU)
          IF(.NOT.STABLE) GOTO 900
        ENDDO
        COMPLETED=PAIR
      ENDDO
 900  CONTINUE
      WRITE(72,'(A,I10)') 'E ',COMPLETED
      CLOSE(72)
      CALL @PREFIX@FINALIZE_TIRUST()

      CONTAINS

      SUBROUTINE EVALUATE(WITHFINITE,OK)
      INTEGER WITHFINITE,STATUS
      LOGICAL OK
      OK=.FALSE.
      IF(REGSTATUS.NE.TIRUST_SUCCESS) RETURN
      TIRustComputeFinite=WITHFINITE.EQ.1
C     The generated routines may reorder their momentum/mass inputs.
      PL=PL0
      QPL=REAL(PL0,KIND=16)
      MASSES=MASSES0
      QMASSES=CMPLX(MASSES0,KIND=16)
      IF(SCOPE.EQ.0) THEN
        IF(CTMODE.EQ.1) THEN
          CALL @PREFIX@TIRUST_TENSOR_DIAGNOSTIC(CTMODE,N,R,PL,
     $      MASSES,COEFS,OK)
        ELSE
          CALL @PREFIX@MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,N,R,
     $      QPL,MASSES,QCOEFS,OK)
        ENDIF
      ELSE
        IF(CTMODE.EQ.1) THEN
          CALL TIRUST_TENSOR_EVALUATE(N,R,SHIFTS(:,1:N),
     $      INVARIANTS(1:N,1:N),MASSES(1:N),MU2,RAW(1:NC,1:4),
     $      STATUS,0,TIRustComputeFinite,.FALSE.,.FALSE.)
        ELSE
          CALL TIRUST_TENSOR_EVALUATE(N,R,QSHIFTS(:,1:N),
     $      QINV(1:N,1:N),QMASSES(1:N),REAL(MU2,KIND=16),
     $      QRAW(1:NC,1:4),STATUS,0,TIRustComputeFinite,
     $      .FALSE.,.FALSE.)
        ENDIF
        OK=STATUS.EQ.TIRUST_SUCCESS
      ENDIF
      END SUBROUTINE EVALUATE

      SUBROUTINE BATCH(PHASE,REPETITION,WITHFINITE,LOOPS,OK,
     $ WALLSECONDS,CPUSECONDS)
      INTEGER PHASE,REPETITION,WITHFINITE,LOOPS,L,DONE,K,S
      INTEGER STATUS
      INTEGER*8 T0,T1,RATE
      REAL*8 WALLSECONDS,CPUSECONDS,C0,C1
      LOGICAL OK
C     Clearing is outside the timer; cache reuse stays disabled on every call.
      CALL TIRUST_CLEAR_CACHE(STATUS)
      OK=STATUS.EQ.TIRUST_SUCCESS
      DONE=0
      CALL CPU_TIME(C0)
      CALL SYSTEM_CLOCK(T0,RATE)
      IF(RATE.LE.0) STOP 7
      IF(OK) THEN
        DO L=1,LOOPS
          CALL EVALUATE(WITHFINITE,OK)
          IF(.NOT.OK) EXIT
          DONE=DONE+1
        ENDDO
      ENDIF
      CALL SYSTEM_CLOCK(T1)
      CALL CPU_TIME(C1)
      WALLSECONDS=DBLE(T1-T0)/DBLE(RATE)
      CPUSECONDS=C1-C0
      WRITE(72,'(A,4I12,L3,I12,2ES26.17E3)') 'F ',PHASE,
     $ REPETITION,WITHFINITE,DONE,OK,NC,WALLSECONDS,CPUSECONDS
C     Promotion, pole packing and output all happen after both timers stop.
      IF(OK) THEN
        IF(SCOPE.EQ.0) THEN
          IF(CTMODE.EQ.1) THEN
            DISPLAY(0:NC-1,:)=COEFS(0:NC-1,:)
          ELSE
            DISPLAY(0:NC-1,:)=QCOEFS(0:NC-1,:)
          ENDIF
        ELSE
          IF(CTMODE.EQ.1) THEN
            DISPLAY(0:NC-1,1)=RAW(1:NC,4)
            DISPLAY(0:NC-1,2)=RAW(1:NC,1)+RAW(1:NC,2)
            DISPLAY(0:NC-1,3)=RAW(1:NC,3)
          ELSE
            DISPLAY(0:NC-1,1)=QRAW(1:NC,4)
            DISPLAY(0:NC-1,2)=QRAW(1:NC,1)+QRAW(1:NC,2)
            DISPLAY(0:NC-1,3)=QRAW(1:NC,3)
          ENDIF
        ENDIF
        DO K=0,NC-1
          DO S=1,3
            VALUES(2*S-1)=REAL(DISPLAY(K,S),KIND=16)
            VALUES(2*S)=AIMAG(DISPLAY(K,S))
          ENDDO
          WRITE(72,'(A,4I10,6ES46.35E4)') 'C ',PHASE,
     $      REPETITION,WITHFINITE,K,VALUES
        ENDDO
      ENDIF
      FLUSH(72)
      END SUBROUTINE BATCH
      END PROGRAM TIRUST_POLE_TIMING
