      SUBROUTINE TEST_UPDATE_WL_2_1_GATHER(A,LCUT_SIZE,B,IN_SIZE
     $ ,OUT_SIZE,OUT)
      USE TEST_POLYNOMIAL_CONSTANTS
      USE, INTRINSIC :: IEEE_ARITHMETIC
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
      COMPLEX*32 ACC
      COMPLEX*32 A(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      COMPLEX*32 B(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      COMPLEX*32 OUT(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE
      TYPE(IEEE_ROUND_TYPE) WL_ROUNDING
      LOGICAL WL_NEAREST
      LOGICAL WL_SKIP_ZERO
      LOGICAL WL_AZERO(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER WL_I,WL_K,WL_L
      INTEGER NEW_POSITION
      COMPLEX*32 UPDATER_COEF

C     Experimental coefficient-major gather
      CALL IEEE_GET_HALTING_MODE(IEEE_ALL,HALT_FLAGS)
      IF(ANY(HALT_FLAGS)) THEN
        CALL TEST_UPDATE_WL_2_1(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
        RETURN
      ENDIF
      IF(.NOT.ALL(GATHER_BOUNDED(REAL(A(1:IN_SIZE,0:14,1:LCUT_SIZE)
     $ ,KIND=16))).OR..NOT.ALL(GATHER_BOUNDED(AIMAG(A(1:IN_SIZE,0:14
     $ ,1:LCUT_SIZE)))).OR..NOT.ALL(GATHER_BOUNDED(REAL(B(1:OUT_SIZE
     $ ,0:4,1:IN_SIZE),KIND=16)))
     $ .OR..NOT.ALL(GATHER_BOUNDED(AIMAG(B(1:OUT_SIZE,0:4,1:IN_SIZE))))
     $ ) THEN
        CALL TEST_UPDATE_WL_2_1(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
        RETURN
      ENDIF

      CALL IEEE_GET_ROUNDING_MODE(WL_ROUNDING)
      WL_NEAREST=WL_ROUNDING.EQ.IEEE_NEAREST
      IF(.NOT.WL_NEAREST.OR..NOT.ALL(WL_IS_FINITE(REAL(A(1:IN_SIZE
     $ ,0:14,1:LCUT_SIZE),KIND=16)))
     $ .OR..NOT.ALL(WL_IS_FINITE(AIMAG(A(1:IN_SIZE,0:14,1:LCUT_SIZE))))
     $ .OR..NOT.ALL(WL_IS_FINITE(REAL(B(1:OUT_SIZE,0:4,1:IN_SIZE)
     $ ,KIND=16))).OR..NOT.ALL(WL_IS_FINITE(AIMAG(B(1:OUT_SIZE,0:4
     $ ,1:IN_SIZE))))) THEN
        CALL TEST_UPDATE_WL_2_1_DENSE(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE
     $   ,OUT)
        RETURN
      ENDIF
      WL_AZERO(1:IN_SIZE,0:14,1:LCUT_SIZE)=.FALSE.
      IF(WL_NEAREST) THEN
        DO WL_K=1,IN_SIZE
          DO WL_I=1,LCUT_SIZE
            DO WL_L=0,14
              IF(WL_IS_FINITE(REAL(A(WL_K,WL_L,WL_I),KIND=16))
     $         .AND.WL_IS_FINITE(AIMAG(A(WL_K,WL_L,WL_I))))
     $          WL_AZERO(WL_K,WL_L,WL_I)=A(WL_K,WL_L,WL_I).EQ.0
            ENDDO
          ENDDO
        ENDDO
      ENDIF

      OUT(:,:,:)=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
      DO J=1,OUT_SIZE
        DO M=0,4
          DO K=1,IN_SIZE
            ACTIVE(M,K)=B(J,M,K).NE.CMPLX(0.0E0_16,0.0E0_16,KIND=16)
            SKIP_A(M,K)=WL_NEAREST

          ENDDO
        ENDDO
        DO I=1,LCUT_SIZE
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,0,I)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,0,K)
          ENDDO
          OUT(J,0,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,1,I)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,0,I)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,1,K)
          ENDDO
          OUT(J,1,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,2,I)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,0,I)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,2,K)
          ENDDO
          OUT(J,2,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,3,I)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,0,I)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,3,K)
          ENDDO
          OUT(J,3,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,4,I)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,0,I)) CYCLE
            ACC=ACC+A(K,0,I)*B(J,4,K)
          ENDDO
          OUT(J,4,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,5,I)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,1,I)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,1,K)
          ENDDO
          OUT(J,5,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,6,I)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,2,I)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,1,I)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,2,K)
          ENDDO
          OUT(J,6,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,7,I)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,2,I)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,2,K)
          ENDDO
          OUT(J,7,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,8,I)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,3,I)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,1,I)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,3,K)
          ENDDO
          OUT(J,8,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,9,I)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,3,I)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,2,I)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,3,K)
          ENDDO
          OUT(J,9,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,10,I)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,3,I)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,3,K)
          ENDDO
          OUT(J,10,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,11,I)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,4,I)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,1,I)) CYCLE
            ACC=ACC+A(K,1,I)*B(J,4,K)
          ENDDO
          OUT(J,11,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,12,I)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,4,I)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,2,I)) CYCLE
            ACC=ACC+A(K,2,I)*B(J,4,K)
          ENDDO
          OUT(J,12,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,13,I)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,4,I)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,3,I)) CYCLE
            ACC=ACC+A(K,3,I)*B(J,4,K)
          ENDDO
          OUT(J,13,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(0,K)) CYCLE
            IF(SKIP_A(0,K).AND.WL_AZERO(K,14,I)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,0,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,4,I)) CYCLE
            ACC=ACC+A(K,4,I)*B(J,4,K)
          ENDDO
          OUT(J,14,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,5,I)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,1,K)
          ENDDO
          OUT(J,15,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,6,I)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,5,I)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,2,K)
          ENDDO
          OUT(J,16,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,7,I)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,6,I)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,2,K)
          ENDDO
          OUT(J,17,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,7,I)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,2,K)
          ENDDO
          OUT(J,18,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,8,I)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,5,I)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,3,K)
          ENDDO
          OUT(J,19,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,9,I)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,8,I)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,6,I)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,3,K)
          ENDDO
          OUT(J,20,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,9,I)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,7,I)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,3,K)
          ENDDO
          OUT(J,21,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,10,I)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,8,I)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,3,K)
          ENDDO
          OUT(J,22,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,10,I)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,9,I)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,3,K)
          ENDDO
          OUT(J,23,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,10,I)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,3,K)
          ENDDO
          OUT(J,24,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,11,I)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,5,I)) CYCLE
            ACC=ACC+A(K,5,I)*B(J,4,K)
          ENDDO
          OUT(J,25,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,12,I)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,11,I)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,6,I)) CYCLE
            ACC=ACC+A(K,6,I)*B(J,4,K)
          ENDDO
          OUT(J,26,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,12,I)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,7,I)) CYCLE
            ACC=ACC+A(K,7,I)*B(J,4,K)
          ENDDO
          OUT(J,27,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,13,I)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,11,I)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,8,I)) CYCLE
            ACC=ACC+A(K,8,I)*B(J,4,K)
          ENDDO
          OUT(J,28,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,13,I)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,12,I)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,9,I)) CYCLE
            ACC=ACC+A(K,9,I)*B(J,4,K)
          ENDDO
          OUT(J,29,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,13,I)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,10,I)) CYCLE
            ACC=ACC+A(K,10,I)*B(J,4,K)
          ENDDO
          OUT(J,30,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(1,K)) CYCLE
            IF(SKIP_A(1,K).AND.WL_AZERO(K,14,I)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,1,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,11,I)) CYCLE
            ACC=ACC+A(K,11,I)*B(J,4,K)
          ENDDO
          OUT(J,31,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(2,K)) CYCLE
            IF(SKIP_A(2,K).AND.WL_AZERO(K,14,I)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,2,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,12,I)) CYCLE
            ACC=ACC+A(K,12,I)*B(J,4,K)
          ENDDO
          OUT(J,32,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(3,K)) CYCLE
            IF(SKIP_A(3,K).AND.WL_AZERO(K,14,I)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,3,K)
          ENDDO
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,13,I)) CYCLE
            ACC=ACC+A(K,13,I)*B(J,4,K)
          ENDDO
          OUT(J,33,I)=ACC
          ACC=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          DO K=1,IN_SIZE
            IF(.NOT.ACTIVE(4,K)) CYCLE
            IF(SKIP_A(4,K).AND.WL_AZERO(K,14,I)) CYCLE
            ACC=ACC+A(K,14,I)*B(J,4,K)
          ENDDO
          OUT(J,34,I)=ACC
        ENDDO
      ENDDO

      CONTAINS
      ELEMENTAL LOGICAL FUNCTION WL_IS_FINITE(X)
      REAL*16, INTENT(IN) :: X
      INTEGER*8, PARAMETER :: ONE_WORDS(2)=TRANSFER(1.0_16,(/0_8,0_8/))
      INTEGER, PARAMETER :: HI=MERGE(1,2,ONE_WORDS(1).NE.0_8)
      WL_IS_FINITE=.FALSE.
      IF(STORAGE_SIZE(X).NE.128.OR.DIGITS(X).NE.113.OR.RADIX(X)
     $ .NE.2.OR.MAXEXPONENT(X).NE.16384) RETURN
      IF(ONE_WORDS(HI).NE.ISHFT(16383_8,48).OR.ONE_WORDS(3-HI).NE.0_8)
     $  RETURN
      ASSOCIATE(WORDS=>TRANSFER(X,ONE_WORDS))
      WL_IS_FINITE=IBITS(WORDS(HI),48,15).NE.32767_8
      END ASSOCIATE
      END FUNCTION

      ELEMENTAL LOGICAL FUNCTION GATHER_BOUNDED(X)
      REAL*16, INTENT(IN) :: X
      INTEGER*8, PARAMETER :: ONE(2)=TRANSFER(1.0_16,(/0_8,0_8/))
      INTEGER, PARAMETER :: HI=MERGE(1,2,ONE(1).NE.0_8)
      GATHER_BOUNDED=.FALSE.
      IF(STORAGE_SIZE(X).NE.128.OR.DIGITS(X).NE.113.OR.RADIX(X)
     $ .NE.2.OR.MAXEXPONENT(X).NE.16384) RETURN
      IF(ONE(HI).NE.ISHFT(16383_8,48).OR.ONE(3-HI).NE.0_8) RETURN
      ASSOCIATE(WORDS=>TRANSFER(X,ONE))
      GATHER_BOUNDED=IBITS(WORDS(HI),48,15)
     $ .GE.15383_8.AND.IBITS(WORDS(HI),48,15).LE.17383_8
      IF(IBCLR(WORDS(HI),63).EQ.0_8.AND.WORDS(3-HI).EQ.0_8)
     $  GATHER_BOUNDED=.TRUE.
      END ASSOCIATE
      END FUNCTION

      END
