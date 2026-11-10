      SUBROUTINE AUDIT_COMPUTE_ACCURACY(FULLLIST, LENGTH, ACC,
     $  ESTIMATE)
      IMPLICIT NONE
C     
C     PARAMETERS 
C     
      INTEGER MAXSTABILITYLENGTH
      COMMON/AUDIT_STABILITY_TESTS/MAXSTABILITYLENGTH
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      REAL*8 FULLLIST(3,0:NSQUAREDSO,MAXSTABILITYLENGTH)
      INTEGER LENGTH
      REAL*8 ACC(0:NSQUAREDSO), ESTIMATE(0:3,0:NSQUAREDSO)
C     
C     LOCAL VARIABLES 
C     
      LOGICAL MASK(MAXSTABILITYLENGTH)
      LOGICAL MASK3(3)
      DATA MASK3/.TRUE.,.TRUE.,.TRUE./
      INTEGER I,J,K
      REAL*8 AVG
      REAL*8 DIFF
      REAL*8 ACCURACIES(3)
      REAL*8 LIST(MAXSTABILITYLENGTH)

C     
C     GLOBAL VARIABLES
C     
      LOGICAL CHOSEN_SO_CONFIGS(NSQUAREDSO)
      COMMON/AUDIT_CHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS
      INTEGER I_LIB
      COMMON/AUDIT_I_LIB/I_LIB
      INCLUDE 'MadLoopParams.inc'

C     ----------
C     BEGIN CODE
C     ----------
      DO I=1,LENGTH
        MASK(I)=.TRUE.
      ENDDO
      DO I=LENGTH+1,MAXSTABILITYLENGTH
        MASK(I)=.FALSE.
C       For some architectures, it is necessary to initialize all the
C        elements of fulllist(i,j)
C       Beware that if the length provided is incorrect, then this can
C        corrup the fulllist given in argument.
        DO J=0,NSQUAREDSO
          DO K=1,3
            FULLLIST(K,J,I)=0.0D0
          ENDDO
        ENDDO
      ENDDO

      DO K=0,NSQUAREDSO

        DO I=1,3
          DO J=1,MAXSTABILITYLENGTH
            LIST(J)=FULLLIST(I,K,J)
          ENDDO
          DIFF=MAXVAL(LIST,1,MASK)-MINVAL(LIST,1,MASK)
          AVG=(MAXVAL(LIST,1,MASK)+MINVAL(LIST,1,MASK))/2.0D0
          ESTIMATE(I,K)=AVG
          IF (AVG.EQ.0.0D0) THEN
            ACCURACIES(I)=DIFF
          ELSE
            ACCURACIES(I)=DIFF/ABS(AVG)
          ENDIF
        ENDDO

C       The technique below is too sensitive, typically to
C       unstablities in very small poles
C       acc(k)=MAXVAL(ACCURACIES,1,MASK3)
C       The following is used instead
        ACC(K) = 0.0D0
        AVG = 0.0D0
        DO I=1,3
          ACC(K) = ACC(K) + ACCURACIES(I)*ABS(ESTIMATE(I,K))
          AVG = AVG + ESTIMATE(I,K)
        ENDDO
        IF (AVG.NE.0.0D0) THEN
          ACC(K)  = ACC(K) / ( ABS(AVG) / 3.0D0)
        ENDIF

C       When using COLLIER with the internal stability test, the first
C        evaluation is typically more reliable so we do not want to
C        use the average but rather the first evaluation.
        IF (MLREDUCTIONLIB(I_LIB)
     $   .EQ.7.AND.COLLIERUSEINTERNALSTABILITYTEST) THEN
          DO I=1,3
            ESTIMATE(I,K) = FULLLIST(I,K,1)
          ENDDO
        ENDIF

C       Make sure to hard-set to zero accuracies of coupling orders
C        not included
        IF (K.NE.0) THEN
          IF (.NOT.CHOSEN_SO_CONFIGS(K)) THEN
            ACC(K) = 0.0D0
          ENDIF
        ENDIF

C       If NaN are present in the evaluation, automatically set the
C        accuracy to 1.0d99.
        DO I=1,3
          DO J=1,MAXSTABILITYLENGTH
            IF (ISNAN(FULLLIST(I,K,J))) THEN
              ACC(K) = 1.0D99
            ENDIF
          ENDDO
        ENDDO

      ENDDO

      END


