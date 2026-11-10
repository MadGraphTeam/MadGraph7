      SUBROUTINE TX_SMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
C

C
C MadGraph StandAlone Version
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C
C Pure source/control fixture, not a model.
C
C BORN-TRANSACTION-BEGIN IEEE_IMPORT
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE,
     $ IEEE_VALUE,IEEE_QUIET_NAN
C BORN-TRANSACTION-END IEEE_IMPORT
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=4)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=3)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=4)
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      DATA CHOSEN_SO_CONFIGS/.TRUE.,.FALSE.,.TRUE./
      COMMON/TX_CHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
C     FLAV_IDX (the dummy argument) is declared with NFLAV below.
C
C LOCAL VARIABLES
C
      INTEGER FLAVOR(NEXTERNAL)
      REAL*8 T(NSQAMPSO), BUFF
      INTEGER IHEL,IDEN, I, J, JJ
C     beam polarisation, see the same block in matrix_standalone_v4.inc
      DOUBLE PRECISION BEAMPOL(2)
      COMMON/TO_BEAMPOL/BEAMPOL
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/2,2/
      INTEGER JC(NEXTERNAL)
      INTEGER NFLAV
      PARAMETER (NFLAV=2)
      INTEGER NNTRY_FLAV, NGOODHEL_FLAV
      PARAMETER (NNTRY_FLAV=NFLAV)
      PARAMETER (NGOODHEL_FLAV=NCOMB*NFLAV)
      INTEGER FLAV_IDX
      INTEGER TX_GET_FLAVOR_INDEX
      INTEGER NTRY(NFLAV)
      LOGICAL GOODHEL(NCOMB,NFLAV)
      DATA NTRY/NNTRY_FLAV*0/
      DATA GOODHEL/NGOODHEL_FLAV*.FALSE./
      DATA IDEN/4/
C
C GLOBAL VARIABLES
C
      INTEGER NHEL(NEXTERNAL,NCOMB)

      COMMON/TX_PROCESS_NHEL/NHEL

      INTEGER USERHEL
      DATA USERHEL/-1/
      COMMON/TX_HELUSERCHOICE/USERHEL
      LOGICAL HELRESET
      common/TX_helreset/HELRESET
      data HELRESET/.true./

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/TX_BORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/

C
C     FUNCTIONS
C
      LOGICAL TX_IS_BORN_HEL_SELECTED
      INTEGER TX_BROKEN_SYM
C RUNTIME-OBSERVER-BEGIN DECLARATION
      INTEGER FX_OBS_NTRY(NFLAV)
      LOGICAL FX_OBS_GOODHEL(NCOMB,NFLAV),FX_OBS_HELRESET
C RUNTIME-OBSERVER-END DECLARATION

C BORN-TRANSACTION-BEGIN ENTRY_AND_SNAPSHOT
      LOGICAL BT_CHECKED_CALL,BT_OK,BT_HELRESET_SAVED
      INTEGER BT_NTRY_SAVED(NFLAV)
      LOGICAL BT_GOODHEL_SAVED(NCOMB,NFLAV)
      BT_CHECKED_CALL=.FALSE.
      GOTO 9001
      ENTRY TX_SMATRIX_SPLITORDERS_CHECKED(P,FLAV_IDX,ANS,BT_OK)
      BT_CHECKED_CALL=.TRUE.
      BT_OK=.FALSE.
      BT_NTRY_SAVED=NTRY
      BT_GOODHEL_SAVED=GOODHEL
      BT_HELRESET_SAVED=HELRESET
9001 CONTINUE
C BORN-TRANSACTION-END ENTRY_AND_SNAPSHOT
c----------
c   Check if helreset mode is on
c---------
 if (HELRESET) then
   do i=1,NFLAV
      NTRY(i) = 0
   enddo
   do i=1,NCOMB
      do j=1,NFLAV
         GOODHEL(I,j) = .false.
      enddo
   enddo
    HELRESET = .false.
 endif


C ----------
C BEGIN CODE
C ----------
C     FLAV_IDX=0 (or out of range): GET_FLAVOR_INDEX could not resolve the
C     requested flavor -- it is not an allowed combination, so its matrix
C     element is identically zero. Short-circuit before touching the
C     1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS(:) = 0d0
C BORN-TRANSACTION-BEGIN EARLY_RETURN
      IF (BT_CHECKED_CALL) GOTO 9010
C BORN-TRANSACTION-END EARLY_RETURN
        RETURN
      ENDIF
      CALL TX_GET_FLAVOR(FLAV_IDX, FLAVOR)
      IF(USERHEL.EQ.-1) NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
      ANS(:) = 0d0
C     When spin-2 particles are involved, the Helicity filtering is dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero. But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the helicity sum is).
C     For this reason, we simply remove the filtering when there is only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        GOODHEL(:,:) = .True.
      ENDIF
      DO IHEL=1,NCOMB
         IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
           IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 20 .OR.USERHEL.NE.-1) THEN
              IF(NTRY(FLAV_IDX).GE.2.AND.POLARIZATIONS(0,0).ne.-1.and.(.not.TX_IS_BORN_HEL_SELECTED(IHEL))) THEN
                CYCLE
              ENDIF
              CALL TX_MATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX, T)
C BORN-TRANSACTION-BEGIN RAW_T
      IF (BT_CHECKED_CALL) THEN
        IF (.NOT.(ALL(IEEE_IS_FINITE(T)))) GOTO 9020
      ENDIF
C BORN-TRANSACTION-END RAW_T
C Support for polarised beam, see matrix_standalone_v4.inc
              IF (NINITIAL.EQ.2) THEN
                DO JJ=1,NINITIAL
                  IF (ABS(BEAMPOL(JJ)).LE.1D0) CYCLE
                  IF (NHEL(JJ,IHEL).EQ.INT(SIGN(1D0,BEAMPOL(JJ)))) THEN
                    DO I=1,NSQAMPSO
                      T(I)=T(I)*ABS(BEAMPOL(JJ))
                    ENDDO
                  ELSE
                    DO I=1,NSQAMPSO
                      T(I)=T(I)*(2D0-ABS(BEAMPOL(JJ)))
                    ENDDO
                  ENDIF
                ENDDO
              ENDIF
C BORN-TRANSACTION-BEGIN WEIGHTED_T
      IF (BT_CHECKED_CALL) THEN
        IF (.NOT.(ALL(IEEE_IS_FINITE(T)))) GOTO 9020
      ENDIF
C BORN-TRANSACTION-END WEIGHTED_T
              BUFF=0D0
              DO I=1,NSQAMPSO
                IF(POLARIZATIONS(0,0).eq.-1.or.TX_IS_BORN_HEL_SELECTED(IHEL)) THEN
                  ANS(I)=ANS(I)+T(I)
                ENDIF
                BUFF=BUFF+T(I)
              ENDDO
C BORN-TRANSACTION-BEGIN LEARNING_BUFFER
      IF (BT_CHECKED_CALL) THEN
        IF (.NOT.(IEEE_IS_FINITE(BUFF))) GOTO 9020
      ENDIF
C BORN-TRANSACTION-END LEARNING_BUFFER
C BORN-TRANSACTION-BEGIN PARTIAL_SUM
      IF (BT_CHECKED_CALL) THEN
        IF (.NOT.(ALL(IEEE_IS_FINITE(ANS)))) GOTO 9020
      ENDIF
C BORN-TRANSACTION-END PARTIAL_SUM
              IF (BUFF .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
                GOODHEL(IHEL,FLAV_IDX)=.TRUE.
              ENDIF
            ENDIF
          ENDIF
       ENDDO
      ANS(0)=0.0d0
      DO I=1,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)*TX_BROKEN_SYM(FLAVOR)
        IF (CHOSEN_SO_CONFIGS(I)) THEN
          ANS(0)=ANS(0)+ANS(I)
        ENDIF
      ENDDO
      IF(USERHEL.NE.-1) THEN
        DO I=0,NSQAMPSO
          ANS(I)=ANS(I)*HELAVGFACTOR
        ENDDO
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).ne.-1) THEN
            DO I=0,NSQAMPSO
              ANS(I)=ANS(I)*BEAMS_HELAVGFACTOR(J)
              ANS(I)=ANS(I)/POLARIZATIONS(J,0)
            ENDDO
          ENDIF
        ENDDO
      ENDIF
C BORN-TRANSACTION-BEGIN FINALIZE
9010 CONTINUE
      IF (BT_CHECKED_CALL) THEN
        IF (.NOT.(ALL(IEEE_IS_FINITE(ANS)))) GOTO 9020
        BT_OK=.TRUE.
      ENDIF
      RETURN
9020 CONTINUE
      NTRY=BT_NTRY_SAVED
      GOODHEL=BT_GOODHEL_SAVED
      HELRESET=BT_HELRESET_SAVED
      BT_OK=.FALSE.
      ANS=IEEE_VALUE(0D0,IEEE_QUIET_NAN)
      RETURN
C BORN-TRANSACTION-END FINALIZE
C RUNTIME-OBSERVER-BEGIN ENTRY
      RETURN
      ENTRY TX_OBSERVE_LEARNING(FX_OBS_NTRY,
     $ FX_OBS_GOODHEL,FX_OBS_HELRESET)
      FX_OBS_NTRY=NTRY
      FX_OBS_GOODHEL=GOODHEL
      FX_OBS_HELRESET=HELRESET
      RETURN
C RUNTIME-OBSERVER-END ENTRY
      END



      SUBROUTINE TX_SMATRIXHEL_SPLITORDERS_CHECKED(P,HEL,FLAV_IDX,ANS,OK)
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=3)
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
      INTEGER HEL,FLAV_IDX,USERHEL,BT_USERHEL_SAVED
      LOGICAL OK
      COMMON/TX_HELUSERCHOICE/USERHEL
      BT_USERHEL_SAVED=USERHEL
      USERHEL=HEL
      CALL TX_SMATRIX_SPLITORDERS_CHECKED(P,FLAV_IDX,ANS,OK)
      IF (OK) THEN
        USERHEL=-1
      ELSE
        USERHEL=BT_USERHEL_SAVED
      ENDIF
      END

      SUBROUTINE TX_SMATRIXHEL_CHECKED(P,HEL,FLAV_IDX,ANS,OK)
      IMPLICIT NONE
      INTEGER NEXTERNAL,NSQAMPSO
      PARAMETER (NEXTERNAL=5)
      PARAMETER (NSQAMPSO=3)
      REAL*8 P(0:3,NEXTERNAL),ANS,ANS_SPLIT(0:NSQAMPSO)
      INTEGER HEL,FLAV_IDX
      LOGICAL OK
      CALL TX_SMATRIXHEL_SPLITORDERS_CHECKED(
     $ P,HEL,FLAV_IDX,ANS_SPLIT,OK)
      ANS=ANS_SPLIT(0)
      END
      SUBROUTINE LEG_SMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
C

C
C MadGraph StandAlone Version
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C
C Pure source/control fixture, not a model.
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=4)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=3)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=4)
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      DATA CHOSEN_SO_CONFIGS/.TRUE.,.FALSE.,.TRUE./
      COMMON/LEG_CHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
C     FLAV_IDX (the dummy argument) is declared with NFLAV below.
C
C LOCAL VARIABLES
C
      INTEGER FLAVOR(NEXTERNAL)
      REAL*8 T(NSQAMPSO), BUFF
      INTEGER IHEL,IDEN, I, J, JJ
C     beam polarisation, see the same block in matrix_standalone_v4.inc
      DOUBLE PRECISION BEAMPOL(2)
      COMMON/TO_BEAMPOL/BEAMPOL
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/2,2/
      INTEGER JC(NEXTERNAL)
      INTEGER NFLAV
      PARAMETER (NFLAV=2)
      INTEGER NNTRY_FLAV, NGOODHEL_FLAV
      PARAMETER (NNTRY_FLAV=NFLAV)
      PARAMETER (NGOODHEL_FLAV=NCOMB*NFLAV)
      INTEGER FLAV_IDX
      INTEGER LEG_GET_FLAVOR_INDEX
      INTEGER NTRY(NFLAV)
      LOGICAL GOODHEL(NCOMB,NFLAV)
      DATA NTRY/NNTRY_FLAV*0/
      DATA GOODHEL/NGOODHEL_FLAV*.FALSE./
      DATA IDEN/4/
C
C GLOBAL VARIABLES
C
      INTEGER NHEL(NEXTERNAL,NCOMB)

      COMMON/LEG_PROCESS_NHEL/NHEL

      INTEGER USERHEL
      DATA USERHEL/-1/
      COMMON/LEG_HELUSERCHOICE/USERHEL
      LOGICAL HELRESET
      common/LEG_helreset/HELRESET
      data HELRESET/.true./

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/LEG_BORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/

C
C     FUNCTIONS
C
      LOGICAL LEG_IS_BORN_HEL_SELECTED
      INTEGER LEG_BROKEN_SYM
C RUNTIME-OBSERVER-BEGIN DECLARATION
      INTEGER FX_OBS_NTRY(NFLAV)
      LOGICAL FX_OBS_GOODHEL(NCOMB,NFLAV),FX_OBS_HELRESET
C RUNTIME-OBSERVER-END DECLARATION

c----------
c   Check if helreset mode is on
c---------
 if (HELRESET) then
   do i=1,NFLAV
      NTRY(i) = 0
   enddo
   do i=1,NCOMB
      do j=1,NFLAV
         GOODHEL(I,j) = .false.
      enddo
   enddo
    HELRESET = .false.
 endif


C ----------
C BEGIN CODE
C ----------
C     FLAV_IDX=0 (or out of range): GET_FLAVOR_INDEX could not resolve the
C     requested flavor -- it is not an allowed combination, so its matrix
C     element is identically zero. Short-circuit before touching the
C     1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS(:) = 0d0
        RETURN
      ENDIF
      CALL LEG_GET_FLAVOR(FLAV_IDX, FLAVOR)
      IF(USERHEL.EQ.-1) NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
      ANS(:) = 0d0
C     When spin-2 particles are involved, the Helicity filtering is dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero. But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the helicity sum is).
C     For this reason, we simply remove the filtering when there is only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        GOODHEL(:,:) = .True.
      ENDIF
      DO IHEL=1,NCOMB
         IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
           IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 20 .OR.USERHEL.NE.-1) THEN
              IF(NTRY(FLAV_IDX).GE.2.AND.POLARIZATIONS(0,0).ne.-1.and.(.not.LEG_IS_BORN_HEL_SELECTED(IHEL))) THEN
                CYCLE
              ENDIF
              CALL LEG_MATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX, T)
C Support for polarised beam, see matrix_standalone_v4.inc
              IF (NINITIAL.EQ.2) THEN
                DO JJ=1,NINITIAL
                  IF (ABS(BEAMPOL(JJ)).LE.1D0) CYCLE
                  IF (NHEL(JJ,IHEL).EQ.INT(SIGN(1D0,BEAMPOL(JJ)))) THEN
                    DO I=1,NSQAMPSO
                      T(I)=T(I)*ABS(BEAMPOL(JJ))
                    ENDDO
                  ELSE
                    DO I=1,NSQAMPSO
                      T(I)=T(I)*(2D0-ABS(BEAMPOL(JJ)))
                    ENDDO
                  ENDIF
                ENDDO
              ENDIF
              BUFF=0D0
              DO I=1,NSQAMPSO
                IF(POLARIZATIONS(0,0).eq.-1.or.LEG_IS_BORN_HEL_SELECTED(IHEL)) THEN
                  ANS(I)=ANS(I)+T(I)
                ENDIF
                BUFF=BUFF+T(I)
              ENDDO
              IF (BUFF .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
                GOODHEL(IHEL,FLAV_IDX)=.TRUE.
              ENDIF
            ENDIF
          ENDIF
       ENDDO
      ANS(0)=0.0d0
      DO I=1,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)*LEG_BROKEN_SYM(FLAVOR)
        IF (CHOSEN_SO_CONFIGS(I)) THEN
          ANS(0)=ANS(0)+ANS(I)
        ENDIF
      ENDDO
      IF(USERHEL.NE.-1) THEN
        DO I=0,NSQAMPSO
          ANS(I)=ANS(I)*HELAVGFACTOR
        ENDDO
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).ne.-1) THEN
            DO I=0,NSQAMPSO
              ANS(I)=ANS(I)*BEAMS_HELAVGFACTOR(J)
              ANS(I)=ANS(I)/POLARIZATIONS(J,0)
            ENDDO
          ENDIF
        ENDDO
      ENDIF
C RUNTIME-OBSERVER-BEGIN ENTRY
      RETURN
      ENTRY LEG_OBSERVE_LEARNING(FX_OBS_NTRY,
     $ FX_OBS_GOODHEL,FX_OBS_HELRESET)
      FX_OBS_NTRY=NTRY
      FX_OBS_GOODHEL=GOODHEL
      FX_OBS_HELRESET=HELRESET
      RETURN
C RUNTIME-OBSERVER-END ENTRY
      END



      SUBROUTINE LEG_SMATRIX(P, FLAV_IDX, ANS_SUMMED)
C
C     Simple routine wrapper to provide the same interface for
C     backward compatibility for usage without split orders.
C
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=3)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
      INTEGER FLAV_IDX
C
C VARIABLES
C
      INTEGER I
      REAL*8 ANS(0:NSQAMPSO)
C
C BEGIN CODE
C
      CALL LEG_SMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
      ANS_SUMMED=ANS(0)

      END

      SUBROUTINE LEG_SMATRIXHEL(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=4)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
      INTEGER FLAV_IDX
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/LEG_HELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL LEG_SMATRIX(P, FLAV_IDX, ANS)
      USERHEL=-1

      END

      SUBROUTINE LEG_SMATRIXHEL_SPLITORDERS(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=4)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=3)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
      INTEGER HEL
      INTEGER FLAV_IDX
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/LEG_HELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL LEG_SMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
      USERHEL=-1

      END

