      SUBROUTINE TEST_UPDATE_WL_2_1_GATHER(A,LCUT_SIZE,B,IN_SIZE
     $ ,OUT_SIZE,OUT)
      USE TEST_POLYNOMIAL_CONSTANTS
      USE, INTRINSIC :: IEEE_EXCEPTIONS
      IMPLICIT NONE
      INTEGER I,J,K,L,M
      INTEGER N
      INTEGER, PARAMETER :: PREDECESSOR(0:4,0:34)=RESHAPE((/0,-1,-1,-1
     $ ,-1,1,0,-1,-1,-1,2,-1,0,-1,-1,3,-1,-1,0,-1,4,-1,-1,-1,0,5,1,-1,
     $ -1,-1,6,2,1,-1,-1,7,-1,2,-1,-1,8,3,-1,1,-1,9,-1,3,2,-1,10,-1,-1
     $ ,3,-1,11,4,-1,-1,1,12,-1,4,-1,2,13,-1,-1,4,3,14,-1,-1,-1,4,-1,5
     $ ,-1,-1,-1,-1,6,5,-1,-1,-1,7,6,-1,-1,-1,-1,7,-1,-1,-1,8,-1,5,-1,
     $ -1,9,8,6,-1,-1,-1,9,7,-1,-1,10,-1,8,-1,-1,-1,10,9,-1,-1,-1,-1
     $ ,10,-1,-1,11,-1,-1,5,-1,12,11,-1,6,-1,-1,12,-1,7,-1,13,-1,11,8,
     $ -1,-1,13,12,9,-1,-1,-1,13,10,-1,14,-1,-1,11,-1,-1,14,-1,12,-1,
     $ -1,-1,14,13,-1,-1,-1,-1,14/),(/5,35/))
      LOGICAL ACTIVE(0:4,MAXLWFSIZE),SKIP_A(0:4,MAXLWFSIZE)
      LOGICAL HALT_FLAGS(SIZE(IEEE_ALL))
      COMPLEX*16 ACC
      COMPLEX*16 A(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      COMPLEX*16 B(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      COMPLEX*16 OUT(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE
      INTEGER NEW_POSITION
      COMPLEX*16 UPDATER_COEF

C     Experimental coefficient-major gather
      CALL IEEE_GET_HALTING_MODE(IEEE_ALL,HALT_FLAGS)
      IF(ANY(HALT_FLAGS)) THEN
        CALL TEST_UPDATE_WL_2_1(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
        RETURN
      ENDIF
      IF(.NOT.ALL(GATHER_BOUNDED(REAL(A(1:IN_SIZE,0:14,1:LCUT_SIZE)
     $ ,KIND=8))).OR..NOT.ALL(GATHER_BOUNDED(AIMAG(A(1:IN_SIZE,0:14
     $ ,1:LCUT_SIZE)))).OR..NOT.ALL(GATHER_BOUNDED(REAL(B(1:OUT_SIZE
     $ ,0:4,1:IN_SIZE),KIND=8)))
     $ .OR..NOT.ALL(GATHER_BOUNDED(AIMAG(B(1:OUT_SIZE,0:4,1:IN_SIZE))))
     $ ) THEN
        CALL TEST_UPDATE_WL_2_1(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
        RETURN
      ENDIF

      OUT(:,:,:)=(0.0D0,0.0D0)
      DO J=1,OUT_SIZE
        DO M=0,4
          DO K=1,IN_SIZE
            ACTIVE(M,K)=B(J,M,K).NE.(0.0D0,0.0D0)
            SKIP_A(M,K)=.FALSE.

          ENDDO
        ENDDO
        DO I=1,LCUT_SIZE
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,0,K)
          ENDDO
          OUT(J,0,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,1,K)
          ENDDO
          OUT(J,1,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,2,K)
          ENDDO
          OUT(J,2,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,3,K)
          ENDDO
          OUT(J,3,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,4,K)
          ENDDO
          OUT(J,4,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,1,K)
          ENDDO
          OUT(J,5,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,2,K)
          ENDDO
          OUT(J,6,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,2,K)
          ENDDO
          OUT(J,7,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,3,K)
          ENDDO
          OUT(J,8,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,3,K)
          ENDDO
          OUT(J,9,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,3,K)
          ENDDO
          OUT(J,10,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,4,K)
          ENDDO
          OUT(J,11,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,4,K)
          ENDDO
          OUT(J,12,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,4,K)
          ENDDO
          OUT(J,13,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,4,K)
          ENDDO
          OUT(J,14,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,1,K)
          ENDDO
          OUT(J,15,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,2,K)
          ENDDO
          OUT(J,16,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,2,K)
          ENDDO
          OUT(J,17,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,2,K)
          ENDDO
          OUT(J,18,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,3,K)
          ENDDO
          OUT(J,19,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,3,K)
          ENDDO
          OUT(J,20,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,3,K)
          ENDDO
          OUT(J,21,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,3,K)
          ENDDO
          OUT(J,22,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,3,K)
          ENDDO
          OUT(J,23,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,3,K)
          ENDDO
          OUT(J,24,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,4,K)
          ENDDO
          OUT(J,25,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,4,K)
          ENDDO
          OUT(J,26,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,4,K)
          ENDDO
          OUT(J,27,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,4,K)
          ENDDO
          OUT(J,28,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,4,K)
          ENDDO
          OUT(J,29,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,4,K)
          ENDDO
          OUT(J,30,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,4,K)
          ENDDO
          OUT(J,31,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,4,K)
          ENDDO
          OUT(J,32,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,4,K)
          ENDDO
          OUT(J,33,I)=ACC
          ACC=(0.0D0,0.0D0)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,4,K)
          ENDDO
          OUT(J,34,I)=ACC
        ENDDO
      ENDDO

      CONTAINS
      ELEMENTAL LOGICAL FUNCTION GATHER_BOUNDED(X)
      REAL*8, INTENT(IN) :: X
      INTEGER*8, PARAMETER :: ONE(1)=TRANSFER(1.0_8,(/0_8/))
      GATHER_BOUNDED=.FALSE.
      IF(STORAGE_SIZE(X).NE.64.OR.DIGITS(X).NE.53.OR.RADIX(X)
     $ .NE.2.OR.MAXEXPONENT(X).NE.1024) RETURN
      IF(ONE(1).NE.ISHFT(1023_8,52)) RETURN
      ASSOCIATE(WORDS=>TRANSFER(X,ONE))
      GATHER_BOUNDED=IBITS(WORDS(1),52,11).GE.923_8.AND.IBITS(WORDS(1)
     $ ,52,11).LE.1123_8
      IF(IBCLR(WORDS(1),63).EQ.0_8) GATHER_BOUNDED=.TRUE.
      END ASSOCIATE
      END FUNCTION

      END
