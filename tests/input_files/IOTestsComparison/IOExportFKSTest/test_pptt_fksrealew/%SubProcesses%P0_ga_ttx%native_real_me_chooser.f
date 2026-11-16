      SUBROUTINE SMATRIX_REAL_NATIVE(P,WGT)
      USE MC_NATIVE_CONTEXT, ONLY: SHARED_REAL_ACTIVE
     $ ,SHARED_REAL_EPOCH,SHARED_REAL_POINT,ACTIVE_HISTORY
     $ ,HISTORY_PERMUTATIONS
      USE MC_BORN_TYPES, ONLY: BORNMODELSTATE,BORN_MODEL_STATE_EQUAL
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'real_matrix_weights.inc'
      INTEGER,PARAMETER::CACHE_SIZE=16
      LOGICAL,PARAMETER::SHARE_REAL_FRAME(27)=[.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.]
      TYPE REALCACHEENTRY
        INTEGER::EVALUATOR=0,SECTOR=0
        REAL(8) MOMENTA(0:3,NEXTERNAL),WEIGHT
     $   ,AMPLITUDES(AMP_SPLIT_SIZE)
        REAL(8) DIAGRAM_WEIGHTS(SIZE(REAL_AMP2))
        LOGICAL::HAS_DIAGRAM_WEIGHTS=.FALSE.
        TYPE(BORNMODELSTATE) MODEL
        END TYPE
        TYPE(REALCACHEENTRY),SAVE::CACHE(CACHE_SIZE)
        TYPE(BORNMODELSTATE),SAVE::STATE
        INTEGER,SAVE::EPOCH=-1,NEXT_ENTRY=1
        DOUBLE PRECISION P(0:3,NEXTERNAL),Q(0:3,NEXTERNAL),R(0:3
     $   ,NEXTERNAL),WGT
        DOUBLE PRECISION WGT_ME_BORN,WGT_ME_REAL
        COMMON/C_WGT_ME_TREE/WGT_ME_BORN,WGT_ME_REAL
        INTEGER NFKSPROCESS,EVALUATOR,EVALUATOR_SECTOR,REQUESTED_ALIAS
     $   ,I,J
        COMMON/C_NFKSPROCESS/NFKSPROCESS
        IF(NFKSPROCESS.LT.1.OR.NFKSPROCESS.GT.27)STOP 'Invalid native'
     $   //' real sector'
        REQUESTED_ALIAS=NFKSPROCESS
        R=P
        IF(SHARED_REAL_ACTIVE.AND.SHARE_REAL_FRAME(NFKSPROCESS))THEN
          R=SHARED_REAL_POINT
          IF(ACTIVE_HISTORY.GT.0)THEN
            DO I=1,NEXTERNAL
              R(:,I)=SHARED_REAL_POINT(:,HISTORY_PERMUTATIONS(I
     $         ,ACTIVE_HISTORY))
            ENDDO
          ENDIF
        ENDIF
        SELECT CASE(NFKSPROCESS)
        CASE(1)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=1
        CASE(2)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=2
        CASE(3)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=3
        CASE(4)
          Q=R
          EVALUATOR=2
          EVALUATOR_SECTOR=4
        CASE(5)
          Q=R
          EVALUATOR=2
          EVALUATOR_SECTOR=5
        CASE(6)
          Q=R
          EVALUATOR=2
          EVALUATOR_SECTOR=6
        CASE(7)
          Q=R
          EVALUATOR=2
          EVALUATOR_SECTOR=7
        CASE(8)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=8
        CASE(9)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=9
        CASE(10)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=10
        CASE(11)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=11
        CASE(12)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=3
          EVALUATOR_SECTOR=8
        CASE(13)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=3
          EVALUATOR_SECTOR=9
        CASE(14)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=3
          EVALUATOR_SECTOR=10
        CASE(15)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=3
          EVALUATOR_SECTOR=11
        CASE(16)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=2
          EVALUATOR_SECTOR=4
        CASE(17)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=2
          EVALUATOR_SECTOR=5
        CASE(18)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=2
          EVALUATOR_SECTOR=6
        CASE(19)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=2
          EVALUATOR_SECTOR=7
        CASE(20)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(21)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(22)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(23)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(24)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(25)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(26)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(27)
          STOP 'Born-only native alias has no local real evaluator'
        CASE DEFAULT
          STOP 'Invalid native real sector'
        END SELECT
        IF(SHARED_REAL_ACTIVE)THEN
          IF(EPOCH.NE.SHARED_REAL_EPOCH)THEN
            CACHE%EVALUATOR=0
            NEXT_ENTRY=1
            EPOCH=SHARED_REAL_EPOCH
          ENDIF
          CALL MC_CAPTURE_MODEL_STATE(STATE)
          DO J=1,CACHE_SIZE
            IF(CACHE(J)%EVALUATOR.NE.EVALUATOR)CYCLE
            IF(CACHE(J)%SECTOR.NE.REQUESTED_ALIAS)CYCLE
            IF(ANY(CACHE(J)%MOMENTA.NE.Q))CYCLE
            IF(.NOT.BORN_MODEL_STATE_EQUAL(CACHE(J)%MODEL,STATE))CYCLE
            IF(GET_REAL_DIAGRAM_WEIGHTS.AND..NOT.CACHE(J)
     $       %HAS_DIAGRAM_WEIGHTS)CYCLE
            WGT=CACHE(J)%WEIGHT
            AMP_SPLIT=CACHE(J)%AMPLITUDES
            IF(GET_REAL_DIAGRAM_WEIGHTS)REAL_AMP2=CACHE(J)
     $       %DIAGRAM_WEIGHTS
            WGT_ME_REAL=WGT
            RETURN
          ENDDO
        ENDIF
        NFKSPROCESS=EVALUATOR_SECTOR
        SELECT CASE(EVALUATOR)
        CASE(1)
          CALL SMATRIX_REAL_LOCAL(Q,AMP_SPLIT,WGT)
        CASE(2)
          CALL SMATRIX_REAL_LOCAL(Q,AMP_SPLIT,WGT)
        CASE(3)
          CALL SMATRIX_REAL_LOCAL(Q,AMP_SPLIT,WGT)
        CASE DEFAULT
          STOP 'Invalid local real evaluator'
        END SELECT
        NFKSPROCESS=REQUESTED_ALIAS
        WGT_ME_REAL=WGT
        IF(SHARED_REAL_ACTIVE)THEN
          J=NEXT_ENTRY
          CACHE(J)%EVALUATOR=EVALUATOR
          CACHE(J)%SECTOR=REQUESTED_ALIAS
          CACHE(J)%MOMENTA=Q
          CACHE(J)%MODEL%REAL_VALUES=STATE%REAL_VALUES
          CACHE(J)%MODEL%CONFIGURATION_EPOCH=STATE%CONFIGURATION_EPOCH
          CACHE(J)%MODEL%COMPLEX_VALUES=STATE%COMPLEX_VALUES
          CACHE(J)%WEIGHT=WGT
          CACHE(J)%AMPLITUDES=AMP_SPLIT
          CACHE(J)%HAS_DIAGRAM_WEIGHTS=GET_REAL_DIAGRAM_WEIGHTS
          IF(GET_REAL_DIAGRAM_WEIGHTS)CACHE(J)
     $     %DIAGRAM_WEIGHTS=REAL_AMP2
          NEXT_ENTRY=MOD(NEXT_ENTRY,CACHE_SIZE)+1
        ENDIF
        END
        SUBROUTINE SMATRIX_REAL(P,RET_AMP_SPLIT,WGT)
        IMPLICIT NONE
        INCLUDE 'nexternal.inc'
        INCLUDE 'orders.inc'
        DOUBLE PRECISION P(0:3,NEXTERNAL),RET_AMP_SPLIT(AMP_SPLIT_SIZE)
     $   ,WGT
        CALL SMATRIX_REAL_NATIVE(P,WGT)
        RET_AMP_SPLIT=AMP_SPLIT
        END
