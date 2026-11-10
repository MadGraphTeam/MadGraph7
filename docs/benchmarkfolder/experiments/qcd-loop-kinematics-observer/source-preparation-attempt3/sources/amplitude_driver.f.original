      PROGRAM CHECK_BORN_FALLBACK_AMPLITUDE
      USE, INTRINSIC :: IEEE_EXCEPTIONS
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      REAL*8 POINTS(0:3,5,2),P(0:3,5),ANS(0:3,0:1),ACC(0:1)
      REAL*8 TOTAL(0:3)
      INTEGER I,J,K,N,CODE,TENS,RESCUES,FAILS
      CHARACTER*1024 CARD,POINTFILE,RESOURCES
      LOGICAL INVALID,DIVZERO,OVERFLOW
      COMMON/AUDIT_BORN_RESCUE_COUNTS/RESCUES,FAILS
      CALL GET_COMMAND_ARGUMENT(1,CARD)
      CALL GET_COMMAND_ARGUMENT(2,POINTFILE)
      CALL GET_COMMAND_ARGUMENT(3,RESOURCES)
      CALL SETPARA(TRIM(CARD))
      CALL SETMADLOOPPATH(RESOURCES)
      CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)
      OPEN(17,FILE=TRIM(POINTFILE),STATUS='OLD')
      READ(17,*) N
      IF(N.NE.2) STOP 11
      DO K=1,2
        DO I=1,5
          READ(17,*) POINTS(:,I,K)
        ENDDO
      ENDDO
      CLOSE(17)
C     Initialize exclusively on the original ordinary anchor.  Every call
C     is retained; this diagnostic has a strict controller wall-clock bound.
      DO I=1,20
        P=POINTS(:,:,1)
        TOTAL=P(:,1)+P(:,2)
        MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
        CALL UPDATE_AS_PARAM()
        CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
        ANS=0D0
        WRITE(*,'(A,1X,I4)') 'AMPLITUDE_WARMUP_BEGIN',I
        FLUSH(6)
        CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
        WRITE(*,'(A,2(1X,I4),5(1X,ES26.17E3))')
     $    'AMPLITUDE_WARMUP',I,CODE,ANS(:,0),ACC(0)
        FLUSH(6)
        TENS=MOD(CODE,100)/10
        IF(TENS.EQ.1.OR.TENS.EQ.2) EXIT
      ENDDO
      IF(I.GT.20) STOP 12
C     Two ordinary calls bracket the exact stress point.  No failure code
C     is silently promoted; the analyzer inspects actual H/T/U and outputs.
      DO I=1,3
        K=1
        IF(I.EQ.2) K=2
        P=POINTS(:,:,K)
        TOTAL=P(:,1)+P(:,2)
        MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
        CALL UPDATE_AS_PARAM()
        CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
        ANS=0D0
        WRITE(*,'(A,2(1X,I4))') 'AMPLITUDE_BEGIN',I,K
        FLUSH(6)
        CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
        CALL IEEE_GET_FLAG(IEEE_INVALID,INVALID)
        CALL IEEE_GET_FLAG(IEEE_DIVIDE_BY_ZERO,DIVZERO)
        CALL IEEE_GET_FLAG(IEEE_OVERFLOW,OVERFLOW)
        DO J=0,1
          WRITE(*,'(A,4(1X,I4),5(1X,ES26.17E3),3(1X,L1))')
     $      'AMPLITUDE_VALUE',I,K,J,CODE,ANS(:,J),ACC(J),
     $      INVALID,DIVZERO,OVERFLOW
        ENDDO
        WRITE(*,'(A,3(1X,I5))') 'BORN_RESCUE_COUNTS',I,RESCUES,FAILS
        WRITE(*,'(A,1X,I4,2(1X,ES26.17E3))')
     $    'LOOP_NORMALIZATION',I,MU_R,G
        FLUSH(6)
      ENDDO
      CALL ML5_0_EXIT_MADLOOP()
      WRITE(*,*) 'AMPLITUDE_DIAGNOSTIC_COMPLETE'
      END
