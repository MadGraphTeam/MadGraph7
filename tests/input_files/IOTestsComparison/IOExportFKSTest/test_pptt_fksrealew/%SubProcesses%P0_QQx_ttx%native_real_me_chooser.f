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
      LOGICAL,PARAMETER::SHARE_REAL_FRAME(67)=[.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.
     $ ,.TRUE.]
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
        IF(NFKSPROCESS.LT.1.OR.NFKSPROCESS.GT.67)STOP 'Invalid native'
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
          EVALUATOR=1
          EVALUATOR_SECTOR=4
        CASE(5)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=5
        CASE(6)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=6
        CASE(7)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=7
        CASE(8)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=8
        CASE(9)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=9
        CASE(10)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=10
        CASE(11)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=11
        CASE(12)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=12
        CASE(13)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=13
        CASE(14)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=14
        CASE(15)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=15
        CASE(16)
          Q=R
          EVALUATOR=1
          EVALUATOR_SECTOR=16
        CASE(17)
          Q=R
          EVALUATOR=2
          EVALUATOR_SECTOR=17
        CASE(18)
          Q=R
          EVALUATOR=2
          EVALUATOR_SECTOR=18
        CASE(19)
          Q=R
          EVALUATOR=2
          EVALUATOR_SECTOR=19
        CASE(20)
          Q=R
          EVALUATOR=2
          EVALUATOR_SECTOR=20
        CASE(21)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=21
        CASE(22)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=22
        CASE(23)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=23
        CASE(24)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=24
        CASE(25)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=25
        CASE(26)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=26
        CASE(27)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=27
        CASE(28)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=28
        CASE(29)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=29
        CASE(30)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=30
        CASE(31)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=31
        CASE(32)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=32
        CASE(33)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=33
        CASE(34)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=34
        CASE(35)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=35
        CASE(36)
          Q=R
          EVALUATOR=3
          EVALUATOR_SECTOR=36
        CASE(37)
          Q=R
          EVALUATOR=4
          EVALUATOR_SECTOR=37
        CASE(38)
          Q=R
          EVALUATOR=4
          EVALUATOR_SECTOR=38
        CASE(39)
          Q=R
          EVALUATOR=4
          EVALUATOR_SECTOR=39
        CASE(40)
          Q=R
          EVALUATOR=4
          EVALUATOR_SECTOR=40
        CASE(41)
          Q=R
          EVALUATOR=5
          EVALUATOR_SECTOR=41
        CASE(42)
          Q=R
          EVALUATOR=5
          EVALUATOR_SECTOR=42
        CASE(43)
          Q=R
          EVALUATOR=5
          EVALUATOR_SECTOR=43
        CASE(44)
          Q=R
          EVALUATOR=5
          EVALUATOR_SECTOR=44
        CASE(45)
          Q=R
          EVALUATOR=6
          EVALUATOR_SECTOR=45
        CASE(46)
          Q=R
          EVALUATOR=6
          EVALUATOR_SECTOR=46
        CASE(47)
          Q=R
          EVALUATOR=6
          EVALUATOR_SECTOR=47
        CASE(48)
          Q=R
          EVALUATOR=6
          EVALUATOR_SECTOR=48
        CASE(49)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=2
          EVALUATOR_SECTOR=17
        CASE(50)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=2
          EVALUATOR_SECTOR=18
        CASE(51)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=2
          EVALUATOR_SECTOR=19
        CASE(52)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=2
          EVALUATOR_SECTOR=20
        CASE(53)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=5
          EVALUATOR_SECTOR=41
        CASE(54)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=5
          EVALUATOR_SECTOR=42
        CASE(55)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=5
          EVALUATOR_SECTOR=43
        CASE(56)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=5
          EVALUATOR_SECTOR=44
        CASE(57)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=6
          EVALUATOR_SECTOR=45
        CASE(58)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=6
          EVALUATOR_SECTOR=46
        CASE(59)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=6
          EVALUATOR_SECTOR=47
        CASE(60)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=6
          EVALUATOR_SECTOR=48
        CASE(61)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=4
          EVALUATOR_SECTOR=37
        CASE(62)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=4
          EVALUATOR_SECTOR=38
        CASE(63)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=4
          EVALUATOR_SECTOR=39
        CASE(64)
          Q(:,1)=R(:,1)
          Q(:,2)=R(:,2)
          Q(:,3)=R(:,3)
          Q(:,4)=R(:,4)
          Q(:,5)=R(:,5)
          EVALUATOR=4
          EVALUATOR_SECTOR=40
        CASE(65)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(66)
          STOP 'Born-only native alias has no local real evaluator'
        CASE(67)
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
        CASE(4)
          CALL SMATRIX_REAL_LOCAL(Q,AMP_SPLIT,WGT)
        CASE(5)
          CALL SMATRIX_REAL_LOCAL(Q,AMP_SPLIT,WGT)
        CASE(6)
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
