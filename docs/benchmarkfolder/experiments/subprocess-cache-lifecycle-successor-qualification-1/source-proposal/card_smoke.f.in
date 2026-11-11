C     Bounded actual SETPARA smoke; no tensor or amplitude entry points.
C     Link the same generated numerical objects, real model reader, and
C     observed forwarding wrapper as the physical lifecycle executable.
C     Each card directory also contains the frozen export ident_card.dat.
C
C     records.txt schema (ASCII whitespace; exact order, no extras):
C       HEADER 1
C       INITIAL module_generation observer_generation health
C       CARD ordinal path MZ_hex g_before g_after og_before og_after health
C       CARD ... [exactly three: nominal, alternate, nominal]
C       DONE 3 module_generation observer_generation health
C       CLOSED module_generation observer_generation health
C     MZ_hex is the binary64 representation as 16 uppercase hex digits.
C     All generations must be -1 and every health value must be zero.
C     Ordinals 1/2/3 are LC_BEGIN actions with process tag zero. Require
C     observed trace with exactly these three BEGINs, one per flush, and
C     a final empty close flush. Any native-wrapper record fails smoke.
C     Expected MZ: 91.188D0, 92D0, exact first-value bits restored.
C     A reader STOP (including STOP 0) cannot pass without DONE/CLOSED.
      PROGRAM SUBPROCESS_CARD_SMOKE
      USE MTIRUST, ONLY: TIRUST_CONTEXT_GENERATION
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT,C_INT64_T
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INTEGER G0,G1
      INTEGER(C_INT64_T) OG0,OG1,NOMINAL_BITS,MZ_BITS
      INTERFACE
        SUBROUTINE LC_OPEN() BIND(C,NAME='lc_open')
        END SUBROUTINE
        SUBROUTINE LC_BEGIN(A,P) BIND(C,NAME='lc_begin')
          IMPORT C_INT
          INTEGER(C_INT),VALUE :: A,P
        END SUBROUTINE
        SUBROUTINE LC_FLUSH() BIND(C,NAME='lc_flush')
        END SUBROUTINE
        SUBROUTINE LC_CLOSE() BIND(C,NAME='lc_close')
        END SUBROUTINE
        FUNCTION LC_HEALTH() BIND(C,NAME='lc_health') RESULT(H)
          IMPORT C_INT
          INTEGER(C_INT) H
        END FUNCTION
        FUNCTION LC_GENERATION() BIND(C,NAME='lc_generation')
     $      RESULT(G)
          IMPORT C_INT64_T
          INTEGER(C_INT64_T) G
        END FUNCTION
      END INTERFACE
      OPEN(73,FILE='records.txt',STATUS='NEW')
      WRITE(73,*) 'HEADER',1
      CALL LC_OPEN()
      G0=TIRUST_CONTEXT_GENERATION()
      OG0=LC_GENERATION()
      WRITE(73,*) 'INITIAL',G0,OG0,LC_HEALTH()
      FLUSH(73)
      IF(G0.NE.-1.OR.OG0.NE.-1.OR.LC_HEALTH().NE.0) STOP 41
      CALL READ_CARD(1,'nominal/param_card.dat')
      CALL READ_CARD(2,'alternate/param_card.dat')
      CALL READ_CARD(3,'nominal/param_card.dat')
      G0=TIRUST_CONTEXT_GENERATION()
      OG0=LC_GENERATION()
      WRITE(73,*) 'DONE',3,G0,OG0,LC_HEALTH()
      FLUSH(73)
      IF(G0.NE.-1.OR.OG0.NE.-1.OR.LC_HEALTH().NE.0) STOP 42
      CALL LC_CLOSE()
      G0=TIRUST_CONTEXT_GENERATION()
      OG0=LC_GENERATION()
      WRITE(73,*) 'CLOSED',G0,OG0,LC_HEALTH()
      CLOSE(73)
      IF(G0.NE.-1.OR.OG0.NE.-1.OR.LC_HEALTH().NE.0) STOP 43
      CONTAINS

      SUBROUTINE READ_CARD(ORDINAL,CARD_PATH)
      INTEGER ORDINAL
      CHARACTER*(*) CARD_PATH
      G0=TIRUST_CONTEXT_GENERATION()
      OG0=LC_GENERATION()
      IF(G0.NE.-1.OR.OG0.NE.-1.OR.LC_HEALTH().NE.0) STOP 44
      CALL LC_BEGIN(INT(ORDINAL,C_INT),0_C_INT)
      CALL SETPARA(CARD_PATH)
      G1=TIRUST_CONTEXT_GENERATION()
      OG1=LC_GENERATION()
      MZ_BITS=TRANSFER(MDL_MZ,0_C_INT64_T)
      WRITE(73,'(A,1X,I1,1X,A,1X,Z16.16,5(1X,I12))')
     $ 'CARD',ORDINAL,CARD_PATH,MZ_BITS,G0,G1,OG0,OG1,LC_HEALTH()
      CALL LC_FLUSH()
      FLUSH(73)
      IF(G1.NE.-1.OR.OG1.NE.-1.OR.LC_HEALTH().NE.0) STOP 45
      IF(ORDINAL.EQ.1) THEN
        NOMINAL_BITS=MZ_BITS
        IF(MZ_BITS.NE.TRANSFER(91.188D0,0_C_INT64_T)) STOP 46
      ELSEIF(ORDINAL.EQ.2) THEN
        IF(MZ_BITS.NE.TRANSFER(92D0,0_C_INT64_T)) STOP 47
      ELSEIF(ORDINAL.EQ.3) THEN
        IF(MZ_BITS.NE.NOMINAL_BITS) STOP 48
      ELSE
        STOP 49
      ENDIF
      END SUBROUTINE
      END PROGRAM
