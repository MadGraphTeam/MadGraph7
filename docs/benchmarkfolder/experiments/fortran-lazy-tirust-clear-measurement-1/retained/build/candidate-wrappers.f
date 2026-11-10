      SUBROUTINE A_CLEAR()
      USE MTIRUST, ONLY: TIRUST_CLEAR_CACHE,
     $                  TIRUST_CONTEXT_GENERATION
C     The shared module context may belong to another subprocess.
C     Event clearing must not create it before any TIRust use.
      IF(TIRUST_CONTEXT_GENERATION().GE.0) THEN
        CALL TIRUST_CLEAR_CACHE()
      ENDIF
      END

      SUBROUTINE B_CLEAR()
      USE MTIRUST, ONLY: TIRUST_CLEAR_CACHE,
     $                  TIRUST_CONTEXT_GENERATION
C     The shared module context may belong to another subprocess.
C     Event clearing must not create it before any TIRust use.
      IF(TIRUST_CONTEXT_GENERATION().GE.0) THEN
        CALL TIRUST_CLEAR_CACHE()
      ENDIF
      END
