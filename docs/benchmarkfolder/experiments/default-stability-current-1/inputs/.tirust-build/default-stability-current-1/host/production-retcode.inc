      INTEGER FUNCTION SET_RET_CODE_U(MLRed,DOING_QP,STABLE)
C
C This functions returns the value of U
C
C
C     U == 0
C         Not stable.
C     U == 1
C         Stable with CutTools in double precision.
C     U == 2
C         Stable with PJFry++ (H=2), or NinjaRS in quadruple precision (H=3).
C     U == 3
C         Stable with IREGI (H=2), or TIRust in quadruple precision (H=3).
C     U == 4
C         Stable with Golem95.
C     U == 5
C         Stable with Samurai.
C     U == 6
C         Stable with Ninja in double precision.
C     U == 7
C         Stable with COLLIER.
C     U == 8
C         Stable with Ninja in quadruple precision (H=3), or NinjaRS in double precision (H=2).
C     U == 9
C         Stable with CutTools in quadruple precision (H=3), or TIRust in double precision (H=2).
C
      IMPLICIT NONE
C     
C CONSTANTS
C
C
C ARGUMENTS
C
      INTEGER MLRed
      LOGICAL DOING_QP,STABLE
C
C LOCAL VARIABLES
C
C
C FUNCTION
C
C
C BEGIN CODE
C
      IF(.NOT.STABLE)THEN
         SET_RET_CODE_U=0
         RETURN
      ENDIF
      IF(DOING_QP)THEN
         IF(MLRed.EQ.1)THEN
            SET_RET_CODE_U=9
            RETURN
         ELSEIF(MLRed.EQ.6)THEN
            SET_RET_CODE_U=8
            RETURN
         ELSEIF(MLRed.EQ.8)THEN
            SET_RET_CODE_U=2
            RETURN
         ELSEIF(MLRed.EQ.9)THEN
            SET_RET_CODE_U=3
            RETURN
         ELSE
            STOP 'Requested library has no quadruple precision adapter'
         ENDIF
      ENDIF
      IF(MLRed.GE.1.AND.MLRed.LE.9)THEN
         SET_RET_CODE_U=MLRed
      ELSE
         STOP 'Unknown MadLoop reduction library; supported IDs are 1 through 9'
      ENDIF
      END

