      SUBROUTINE MC_NATIVE_PROPS(PMASS,PWIDTH,POW)
      USE MC_NATIVE_CONTEXT, ONLY: ACTIVE_CONTEXT,ENSURE_NATIVE_CONTEXT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxconfigs.inc'
      INCLUDE 'coupl.inc'
      DOUBLE PRECISION ZERO
      PARAMETER(ZERO=0D0)
      DOUBLE PRECISION PMASS(-NEXTERNAL:0,LMAXCONFIGS),PWIDTH(
     $ -NEXTERNAL:0,LMAXCONFIGS)
      INTEGER POW(-NEXTERNAL:0,LMAXCONFIGS)
      CALL ENSURE_NATIVE_CONTEXT()
      PMASS=0D0
      PWIDTH=0D0
      POW=0
      SELECT CASE(ACTIVE_CONTEXT)
      CASE(1)
        PMASS( -1,   1)  = ZERO
        PWIDTH( -1,   1) = ZERO
        POW( -1,   1) = 2
        PMASS( -1,   2)  = ZERO
        PWIDTH( -1,   2) = ZERO
        POW( -1,   2) = 2
        PMASS( -1,   3)  = ZERO
        PWIDTH( -1,   3) = ZERO
        POW( -1,   3) = 2
        PMASS( -1,   4)  = ZERO
        PWIDTH( -1,   4) = ZERO
        POW( -1,   4) = 2
        PMASS( -1,   5)  = ZERO
        PWIDTH( -1,   5) = ZERO
        POW( -1,   5) = 2
        PMASS( -1,   6)  = ZERO
        PWIDTH( -1,   6) = ZERO
        POW( -1,   6) = 2
        PMASS( -1,   7)  = ZERO
        PWIDTH( -1,   7) = ZERO
        POW( -1,   7) = 2
        PMASS( -1,   8)  = ZERO
        PWIDTH( -1,   8) = ZERO
        POW( -1,   8) = 2
        PMASS( -1,   9)  = ZERO
        PWIDTH( -1,   9) = ZERO
        POW( -1,   9) = 2
        PMASS( -1,  10)  = ZERO
        PWIDTH( -1,  10) = ZERO
        POW( -1,  10) = 2
        PMASS( -1,  11)  = ZERO
        PWIDTH( -1,  11) = ZERO
        POW( -1,  11) = 2
        PMASS( -1,  12)  = ZERO
        PWIDTH( -1,  12) = ZERO
        POW( -1,  12) = 2
        PMASS( -1,  13)  = ZERO
        PWIDTH( -1,  13) = ZERO
        POW( -1,  13) = 2
        PMASS( -1,  14)  = ZERO
        PWIDTH( -1,  14) = ZERO
        POW( -1,  14) = 2
        PMASS( -1,  15)  = ZERO
        PWIDTH( -1,  15) = ZERO
        POW( -1,  15) = 2
      END SELECT
      END
