C     Persistent benchmark template. Expand ML5_0_, 5, 1.
C     Usage: check --masses, or check points.dat protocol.fifo.
      PROGRAM BENCHMARK_MADLOOP_WORKER
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER(NEXTERNAL=5)
      INCLUDE 'coupl.inc'
      INTEGER NSQSO_BORN
      PARAMETER(NSQSO_BORN=1)
      REAL*8 ZERO
      PARAMETER(ZERO=0D0)
      REAL*8 PMASS(NEXTERNAL)
      REAL*8, ALLOCATABLE :: POINTS(:,:,:),ANS(:,:),ACC(:)
      REAL*8, ALLOCATABLE :: VALUES(:,:,:),ACCURACIES(:,:)
      INTEGER, ALLOCATABLE :: CODES(:)
      REAL*8 P(0:3,NEXTERNAL),TOTAL(0:3),COLD,ELAPSED
      INTEGER NPOINTS,NDIM,NSO,I,J,K,L,NLOOPS,CODE
      INTEGER TENS,WARM_POINTS,WARM_POINT,IOS,SEQUENCE
      INTEGER PREVIOUS_SEQUENCE
      LOGICAL BENCHMARK_WARM_STATUS
      INTEGER*8 TIMED_CODES(0:499),T0,T1,RATE
      CHARACTER*1024 ARG
      CHARACTER*16 COMMAND

      CALL SETPARA('param_card.dat')
      INCLUDE 'pmass.inc'
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      IF(TRIM(ARG).EQ.'--masses') THEN
         OPEN(73,FILE='masses.dat',STATUS='REPLACE')
         WRITE(*,'(A,I8)') 'MASSES ',NEXTERNAL
         DO I=1,NEXTERNAL
            WRITE(*,'(I8,ES26.17E3)') I,PMASS(I)
            WRITE(73,'(ES26.17E3)') PMASS(I)
         ENDDO
         CLOSE(73)
         STOP
      ENDIF
C     A dedicated FIFO carries protocol; stdout is diagnostic only.
      CALL GET_COMMAND_ARGUMENT(2,ARG)
      OPEN(74,FILE=TRIM(ARG),STATUS='OLD',ACTION='WRITE')
      CALL ML5_0_GET_ANSWER_DIMENSION(NDIM)
      CALL ML5_0_GET_NSQSO_LOOP(NSO)
      IF(NDIM.NE.MAX(NSO,NSQSO_BORN)) STOP 2
      OPEN(73,FILE='dimensions.dat',STATUS='REPLACE')
      WRITE(73,*) NDIM,NSO,NSQSO_BORN
      CLOSE(73)
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      OPEN(73,FILE=TRIM(ARG),STATUS='OLD')
      READ(73,*) NPOINTS
      IF(NPOINTS.LE.0) STOP 1
      ALLOCATE(POINTS(0:3,NEXTERNAL,NPOINTS))
      DO K=1,NPOINTS
         DO I=1,NEXTERNAL
            READ(73,*) POINTS(:,I,K)
         ENDDO
      ENDDO
      CLOSE(73)
      ALLOCATE(ANS(0:3,0:NDIM),ACC(0:NSO))
      ALLOCATE(VALUES(0:3,0:NDIM,NPOINTS))
      ALLOCATE(ACCURACIES(0:NSO,NPOINTS),CODES(NPOINTS))
      ANS=0D0
      ACC=-1D0
      VALUES=0D0
      OPEN(75,FILE='warmup.dat',STATUS='REPLACE')
      WRITE(75,'(A)')
     $ '# call point order status Born finite single double accuracy'
      CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)
      P=POINTS(:,:,1)
      TOTAL=P(:,1)+P(:,2)
      MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
      CALL UPDATE_AS_PARAM()
      CALL SYSTEM_CLOCK(T0,RATE)
      IF(RATE.LE.0) STOP 5
      CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
C     The first call bootstraps MadLoop and may have an unknown status.
C     Retain it, but require a recognized status on every later call.
      CALL BENCHMARK_TRACE_WARMUP(75,0,1,NDIM,NSO,CODE,ANS,ACC)
      WARM_POINTS=0
      DO K=1,1000+NPOINTS
         WARM_POINT=1+MOD(K-1,NPOINTS)
         P=POINTS(:,:,WARM_POINT)
         TOTAL=P(:,1)+P(:,2)
         MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
         CALL UPDATE_AS_PARAM()
         ANS=0D0
         ACC=-1D0
         CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
         CALL BENCHMARK_TRACE_WARMUP(75,K,WARM_POINT,NDIM,NSO,
     $        CODE,ANS,ACC)
         IF(.NOT.BENCHMARK_WARM_STATUS(CODE)) THEN
            WRITE(74,'(A,2I12)') 'ERROR warmup-status ',
     $           WARM_POINT,CODE
            FLUSH(74)
            CLOSE(75)
            CALL ML5_0_EXIT_MADLOOP()
            STOP 3
         ENDIF
         TENS=MOD(CODE,100)/10
         IF(TENS.EQ.1.OR.TENS.EQ.2) THEN
            WARM_POINTS=WARM_POINTS+1
         ELSE
            WARM_POINTS=0
         ENDIF
         IF(WARM_POINTS.GE.NPOINTS) EXIT
      ENDDO
      IF(WARM_POINTS.LT.NPOINTS) THEN
         WRITE(74,'(A,2I12)') 'ERROR warmup-limit ',WARM_POINT,CODE
         FLUSH(74)
         CLOSE(75)
         CALL ML5_0_EXIT_MADLOOP()
         STOP 3
      ENDIF
      CLOSE(75)
      CALL SYSTEM_CLOCK(T1)
      COLD=DBLE(T1-T0)/DBLE(RATE)
C     Keep exactly the same diagnostic policy for every measured backend.
      OPEN(6,FILE='/dev/null',STATUS='OLD')
      WRITE(74,'(A,5I12,ES26.17E3)') 'READY ',1,NPOINTS,
     $ NDIM,NSO,NSQSO_BORN,COLD
      FLUSH(74)
      PREVIOUS_SEQUENCE=0

 100  CONTINUE
      READ(5,*,IOSTAT=IOS) COMMAND,SEQUENCE,NLOOPS
      IF(IOS.LT.0) GOTO 900
      IF(IOS.NE.0) GOTO 800
      IF(TRIM(COMMAND).EQ.'STOP') GOTO 900
      IF(TRIM(COMMAND).NE.'RUN') GOTO 800
      IF(SEQUENCE.NE.PREVIOUS_SEQUENCE+1.OR.NLOOPS.LE.0) GOTO 800
      PREVIOUS_SEQUENCE=SEQUENCE
      TIMED_CODES=0
      CALL SYSTEM_CLOCK(T0)
      DO L=1,NLOOPS
         DO K=1,NPOINTS
            P=POINTS(:,:,K)
            TOTAL=P(:,1)+P(:,2)
            MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
            CALL UPDATE_AS_PARAM()
            ANS=0D0
            CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
            IF(CODE.LT.0.OR.CODE.GT.499) STOP 4
            TIMED_CODES(CODE)=TIMED_CODES(CODE)+1
C           The final timed traversal supplies numerical evidence.
            IF(L.EQ.NLOOPS) THEN
               VALUES(:,:,K)=ANS
               ACCURACIES(:,K)=ACC
               CODES(K)=CODE
            ENDIF
         ENDDO
      ENDDO
      CALL SYSTEM_CLOCK(T1)
      ELAPSED=DBLE(T1-T0)/DBLE(RATE)
      OPEN(73,FILE='timing.dat',STATUS='REPLACE')
      WRITE(73,'(2ES26.17E3)') COLD,ELAPSED
      CLOSE(73)
      OPEN(73,FILE='values.dat',STATUS='REPLACE')
      DO K=1,NPOINTS
         DO J=0,NDIM
            IF(J.LE.NSO) THEN
               WRITE(73,'(3I8,5ES26.17E3)') K,J,CODES(K),
     $              VALUES(:,J,K),ACCURACIES(J,K)
            ELSE
C              Born-only columns have no per-order loop accuracy.
               WRITE(73,'(3I8,5ES26.17E3)') K,J,CODES(K),
     $              VALUES(:,J,K),ACCURACIES(0,K)
            ENDIF
         ENDDO
      ENDDO
      CLOSE(73)
      OPEN(73,FILE='timed_status.dat',STATUS='REPLACE')
      DO I=0,499
         IF(TIMED_CODES(I).GT.0) WRITE(73,'(I5,1X,I20)')
     $        I,TIMED_CODES(I)
      ENDDO
      CLOSE(73)
      WRITE(74,'(A,2I12)') 'DONE ',SEQUENCE,NLOOPS
      FLUSH(74)
      GOTO 100

 800  CONTINUE
      WRITE(74,'(A)') 'ERROR invalid-command'
      FLUSH(74)
      CALL ML5_0_EXIT_MADLOOP()
      STOP 8
 900  CONTINUE
      CALL ML5_0_EXIT_MADLOOP()
      WRITE(74,'(A)') 'BYE'
      FLUSH(74)
      CLOSE(74)
      END

      LOGICAL FUNCTION BENCHMARK_WARM_STATUS(CODE)
C     T=3 is the only allowed continuing initialization state for this
C     CTModeInit=1 campaign. T=4, unknown/failing reductions, and backend
C     aliases outside the five qualified libraries fail immediately.
      IMPLICIT NONE
      INTEGER CODE,H,T,U
      LOGICAL STABLE_ALIAS
      BENCHMARK_WARM_STATUS=.FALSE.
      IF(CODE.LT.0.OR.CODE.GT.499) RETURN
      H=CODE/100
      T=MOD(CODE,100)/10
      U=MOD(CODE,10)
      IF(T.NE.1.AND.T.NE.2.AND.T.NE.3) RETURN
      IF(U.EQ.0.OR.H.EQ.4) RETURN
      STABLE_ALIAS=(H.EQ.2.AND.(U.EQ.1.OR.U.EQ.6.OR.U.EQ.7
     $ .OR.U.EQ.8.OR.U.EQ.9)).OR.(H.EQ.3.AND.(U.EQ.9
     $ .OR.U.EQ.8.OR.U.EQ.2.OR.U.EQ.3))
      BENCHMARK_WARM_STATUS=STABLE_ALIAS.OR.(T.EQ.3.AND.H.EQ.1
     $ .AND.(U.EQ.1.OR.U.EQ.6.OR.U.EQ.7.OR.U.EQ.8.OR.U.EQ.9))
      END

      SUBROUTINE BENCHMARK_TRACE_WARMUP(LUN,CALL_INDEX,POINT_INDEX,
     $ NDIM,NSO,CODE,ANS,ACC)
C     Every answer column and accuracy is retained outside timed runs.
      IMPLICIT NONE
      INTEGER LUN,CALL_INDEX,POINT_INDEX,NDIM,NSO,CODE,J
      REAL*8 ANS(0:3,0:NDIM),ACC(0:NSO),ESTIMATE
      DO J=0,NDIM
         ESTIMATE=ACC(0)
         IF(J.LE.NSO) ESTIMATE=ACC(J)
         WRITE(LUN,'(4I12,5ES26.17E3)') CALL_INDEX,POINT_INDEX,J,
     $        CODE,ANS(:,J),ESTIMATE
      ENDDO
      FLUSH(LUN)
      END
