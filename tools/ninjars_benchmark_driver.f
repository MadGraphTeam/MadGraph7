      PROGRAM BENCHMARK_MADLOOP
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER(NEXTERNAL=@NEXTERNAL@)
      INCLUDE 'coupl.inc'
      REAL*8 ZERO
      PARAMETER(ZERO=0D0)
      REAL*8 PMASS(NEXTERNAL)
      REAL*8, ALLOCATABLE :: POINTS(:,:,:),ANS(:,:),ACC(:)
      REAL*8, ALLOCATABLE :: VALUES(:,:,:),ACCURACIES(:,:)
      INTEGER, ALLOCATABLE :: CODES(:)
      REAL*8 P(0:3,NEXTERNAL),TOTAL(0:3),COLD,ELAPSED
      INTEGER NPOINTS,NDIM,NSO,I,J,K,L,NLOOPS,CODE
      INTEGER TENS,WARM_POINTS
      INTEGER*8 TIMED_CODES(0:499)
      INTEGER*8 T0,T1,RATE
      CHARACTER*1024 ARG

      CALL SETPARA('param_card.dat')
      INCLUDE 'pmass.inc'
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      IF(TRIM(ARG).EQ.'--masses') THEN
         OPEN(73,FILE='masses.dat',STATUS='REPLACE')
         WRITE(73,'(ES26.17E3)') PMASS
         CLOSE(73)
         STOP
      ENDIF
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
      CALL GET_COMMAND_ARGUMENT(2,ARG)
      READ(ARG,*) NLOOPS
      IF(NLOOPS.LE.0) STOP 2
      CALL @PREFIX@GET_ANSWER_DIMENSION(NDIM)
      CALL @PREFIX@GET_NSQSO_LOOP(NSO)
      ALLOCATE(ANS(0:3,0:NDIM),ACC(0:NSO))
      ALLOCATE(VALUES(0:3,0:NDIM,NPOINTS))
      ALLOCATE(ACCURACIES(0:NSO,NPOINTS),CODES(NPOINTS))
      ANS=0D0
      VALUES=0D0
      TIMED_CODES=0
      CALL @PREFIX@FORCE_STABILITY_CHECK(.TRUE.)
      P=POINTS(:,:,1)
      TOTAL=P(:,1)+P(:,2)
      MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
      CALL UPDATE_AS_PARAM()
      CALL SYSTEM_CLOCK(T0,RATE)
      CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
C     Wait for a full warm pass after helicity initialization finishes.
      WARM_POINTS=0
      DO K=1,1000+NPOINTS
         P=POINTS(:,:,1+MOD(K-1,NPOINTS))
         TOTAL=P(:,1)+P(:,2)
         MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
         CALL UPDATE_AS_PARAM()
         CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
         TENS=MOD(CODE,100)/10
         IF(TENS.EQ.1.OR.TENS.EQ.2) THEN
            WARM_POINTS=WARM_POINTS+1
         ELSE
            WARM_POINTS=0
         ENDIF
         IF(WARM_POINTS.GE.NPOINTS) EXIT
      ENDDO
      IF(WARM_POINTS.LT.NPOINTS) STOP 3
      CALL SYSTEM_CLOCK(T1)
      COLD=DBLE(T1-T0)/DBLE(RATE)
C     Save numerical outputs outside the timing interval.
      DO K=1,NPOINTS
         P=POINTS(:,:,K)
         TOTAL=P(:,1)+P(:,2)
         MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
         CALL UPDATE_AS_PARAM()
         CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
         VALUES(:,:,K)=ANS
         ACCURACIES(:,K)=ACC
         CODES(K)=CODE
      ENDDO
C     Suppress any repeated Fortran diagnostics inside the timed region.
C     Validate every timed return code in the retained histogram afterward.
      OPEN(6,FILE='/dev/null',STATUS='OLD')
      CALL SYSTEM_CLOCK(T0)
      DO L=1,NLOOPS
         DO K=1,NPOINTS
            P=POINTS(:,:,K)
            TOTAL=P(:,1)+P(:,2)
            MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
            CALL UPDATE_AS_PARAM()
            CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
            IF(CODE.LT.0.OR.CODE.GT.499) STOP 4
            TIMED_CODES(CODE)=TIMED_CODES(CODE)+1
         ENDDO
      ENDDO
      CALL SYSTEM_CLOCK(T1)
      ELAPSED=DBLE(T1-T0)/DBLE(RATE)
      OPEN(73,FILE='timing.dat',STATUS='REPLACE')
      WRITE(73,'(2ES26.17E3)') COLD,ELAPSED
      CLOSE(73)
      OPEN(73,FILE='values.dat',STATUS='REPLACE')
      DO K=1,NPOINTS
         DO J=0,NSO
            WRITE(73,'(3I8,5ES26.17E3)') K,J,CODES(K),
     $           VALUES(:,J,K),ACCURACIES(J,K)
         ENDDO
      ENDDO
      CLOSE(73)
      OPEN(73,FILE='timed_status.dat',STATUS='REPLACE')
      DO I=0,499
         IF(TIMED_CODES(I).GT.0) WRITE(73,'(I5,1X,I20)')
     $        I,TIMED_CODES(I)
      ENDDO
      CLOSE(73)
      END
