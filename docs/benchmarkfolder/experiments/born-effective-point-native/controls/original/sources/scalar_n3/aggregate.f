      SUBROUTINE TX_SMATRIX(P, FLAV_IDX, ANS)
C     

C     
C     MadGraph7 StandAlone Version
C     
C     Returns amplitude squared summed/avg over colors
C     and helicities
C     for the point in phase space P(0:3,NEXTERNAL)
C     
C     Pure source/control fixture, not a model.
C     
C     BORN-TRANSACTION-BEGIN IEEE_IMPORT
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE,
     $ IEEE_VALUE,IEEE_QUIET_NAN
C     BORN-TRANSACTION-END IEEE_IMPORT
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=4)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=4)
C     
C     ARGUMENTS
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
C     
C     LOCAL VARIABLES
C     
      INTEGER NHEL(NEXTERNAL,NCOMB)
C     put in common block to expose this variable to python interface
      COMMON/TX_PROCESS_NHEL/NHEL
      REAL*8 T
      REAL*8 TX_MATRIX
      INTEGER IHEL,IDEN, I, J, JJ
C     beam polarisation. Unpolarised unless something fills
C     /to_beampol/: PY_SET_BEAMPOL for the MadSpin density
C     modes, the driver's stdin for the v1 path. Convention as
C     in madevent's /to_polarization/: 1 is unpolarised,
C     |BEAMPOL| grows to 2 for a fully polarised beam and its
C     sign gives the favoured helicity.
      DOUBLE PRECISION BEAMPOL(2)
      COMMON/TO_BEAMPOL/BEAMPOL
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/2,2/
      INTEGER FLAVOR(NEXTERNAL)
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

C     
C     GLOBAL VARIABLES
C     
      INTEGER USERHEL
      COMMON/TX_HELUSERCHOICE/USERHEL
      DATA USERHEL/-1/
      LOGICAL HELRESET
      COMMON/TX_HELRESET/HELRESET
      DATA HELRESET/.TRUE./


      DATA IDEN/4/

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/TX_BORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/

C     
C     FUNCTIONS
C     
      LOGICAL TX_IS_BORN_HEL_SELECTED
      INTEGER TX_BROKEN_SYM
C     RUNTIME-OBSERVER-BEGIN DECLARATION
      INTEGER FX_OBS_NTRY(NFLAV)
      LOGICAL FX_OBS_GOODHEL(NCOMB,NFLAV),FX_OBS_HELRESET
C     RUNTIME-OBSERVER-END DECLARATION
C     BORN-TRANSACTION-BEGIN ENTRY_AND_SNAPSHOT
      LOGICAL BT_CHECKED_CALL,BT_OK,BT_HELRESET_SAVED
      INTEGER BT_NTRY_SAVED(NFLAV)
      LOGICAL BT_GOODHEL_SAVED(NCOMB,NFLAV)
      BT_CHECKED_CALL=.FALSE.
      GOTO 9001
      ENTRY TX_SMATRIX_CHECKED(P,FLAV_IDX,ANS,BT_OK)
      BT_CHECKED_CALL=.TRUE.
      BT_OK=.FALSE.
      BT_NTRY_SAVED=NTRY
      BT_GOODHEL_SAVED=GOODHEL
      BT_HELRESET_SAVED=HELRESET
 9001 CONTINUE
C     BORN-TRANSACTION-END ENTRY_AND_SNAPSHOT
C     ----------
C     Check if helreset mode is on
C     ---------
      IF (HELRESET) THEN
        DO I=1,NFLAV
          NTRY(I) = 0
        ENDDO
        DO I=1,NCOMB
          DO J=1,NFLAV
            GOODHEL(I,J) = .FALSE.
          ENDDO
        ENDDO
        HELRESET = .FALSE.
      ENDIF

C     ----------
C     BEGIN CODE
C     ----------
C     FLAV_IDX=0 (or out of range) means GET_FLAVOR_INDEX could not
C      resolve
C     the requested flavor: it is not an allowed combination, so its
C      matrix
C     element is identically zero. Short-circuit before touching the
C     1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS = 0D0
C       BORN-TRANSACTION-BEGIN EARLY_RETURN
        IF (BT_CHECKED_CALL) GOTO 9010
C       BORN-TRANSACTION-END EARLY_RETURN
        RETURN
      ENDIF
      CALL TX_GET_FLAVOR(FLAV_IDX, FLAVOR)
      IF(USERHEL.EQ.-1) NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      DO IHEL=1,NEXTERNAL
        JC(IHEL) = +1
      ENDDO
C     When spin-2 particles are involved, the Helicity filtering is
C      dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points
C      have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero.
C      But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS
C      helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the
C      helicity sum is).
C     For this reason, we simply remove the filtering when there is
C      only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        DO IHEL=1,NCOMB
          DO J=1,NFLAV
            GOODHEL(IHEL,J)=.TRUE.
          ENDDO
        ENDDO
      ENDIF
      ANS = 0D0
      DO IHEL=1,NCOMB
        IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
          IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT.
     $      20.OR.USERHEL.NE.-1) THEN
            IF(NTRY(FLAV_IDX).GE.2.AND.POLARIZATIONS(0,0).NE.
     $       -1.AND.(.NOT.TX_IS_BORN_HEL_SELECTED(IHEL))) THEN
              CYCLE
            ENDIF
            T=TX_MATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX)
C           BORN-TRANSACTION-BEGIN RAW_T
            IF (BT_CHECKED_CALL) THEN
              IF (.NOT.(IEEE_IS_FINITE(T))) GOTO 9020
            ENDIF
C           BORN-TRANSACTION-END RAW_T
C           Support for polarised beam. Same reweighting of the
C            initial-state
C           helicity sum as madevent's matrix.f and as the v1 MadSpin
C            msP/msF
C           templates. Inert (and skipped) unless /to_beampol/ has
C            been filled:
C           |BEAMPOL| runs from 1 (unpolarised) to 2 (fully
C            polarised), so
C           anything at or below 1 -- including a zero-filled common
C            block --
C           means "no polarisation". 1 -> N matrix elements are left
C            alone: their
C           leg 1 is a decaying resonance, not a beam.
            IF (NINITIAL.EQ.2) THEN
              DO JJ=1,NINITIAL
                IF (ABS(BEAMPOL(JJ)).LE.1D0) CYCLE
                IF (NHEL(JJ,IHEL).EQ.INT(SIGN(1D0,BEAMPOL(JJ)))) THEN
                  T=T*ABS(BEAMPOL(JJ))
                ELSE
                  T=T*(2D0-ABS(BEAMPOL(JJ)))
                ENDIF
              ENDDO
            ENDIF
C           BORN-TRANSACTION-BEGIN WEIGHTED_T
            IF (BT_CHECKED_CALL) THEN
              IF (.NOT.(IEEE_IS_FINITE(T))) GOTO 9020
            ENDIF
C           BORN-TRANSACTION-END WEIGHTED_T
            IF(POLARIZATIONS(0,0).EQ.
     $       -1.OR.TX_IS_BORN_HEL_SELECTED(IHEL)) THEN
              ANS=ANS+T
            ENDIF
C           BORN-TRANSACTION-BEGIN PARTIAL_SUM
            IF (BT_CHECKED_CALL) THEN
              IF (.NOT.(IEEE_IS_FINITE(ANS))) GOTO 9020
            ENDIF
C           BORN-TRANSACTION-END PARTIAL_SUM
            IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
              GOODHEL(IHEL,FLAV_IDX)=.TRUE.
            ENDIF
          ENDIF
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)*TX_BROKEN_SYM(FLAVOR)
      IF(USERHEL.NE.-1) THEN
        ANS=ANS*HELAVGFACTOR
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).NE.-1) THEN
            ANS=ANS*BEAMS_HELAVGFACTOR(J)
            ANS=ANS/POLARIZATIONS(J,0)
          ENDIF
        ENDDO
      ENDIF
C     BORN-TRANSACTION-BEGIN FINALIZE
 9010 CONTINUE
      IF (BT_CHECKED_CALL) THEN
        IF (.NOT.(IEEE_IS_FINITE(ANS))) GOTO 9020
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
C     BORN-TRANSACTION-END FINALIZE
C     RUNTIME-OBSERVER-BEGIN ENTRY
      RETURN
      ENTRY TX_OBSERVE_LEARNING(FX_OBS_NTRY,
     $ FX_OBS_GOODHEL,FX_OBS_HELRESET)
      FX_OBS_NTRY=NTRY
      FX_OBS_GOODHEL=GOODHEL
      FX_OBS_HELRESET=HELRESET
      RETURN
C     RUNTIME-OBSERVER-END ENTRY
      END



      SUBROUTINE TX_SMATRIXHEL_CHECKED(P,HEL,FLAV_IDX,ANS,OK)
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL,FLAV_IDX,USERHEL,BT_USERHEL_SAVED
      LOGICAL OK
      COMMON/TX_HELUSERCHOICE/USERHEL
      BT_USERHEL_SAVED=USERHEL
      USERHEL=HEL
      CALL TX_SMATRIX_CHECKED(P,FLAV_IDX,ANS,OK)
      IF (OK) THEN
        USERHEL=-1
      ELSE
        USERHEL=BT_USERHEL_SAVED
      ENDIF
      END

      SUBROUTINE TX_SMATRIXHEL_SPLITORDERS_CHECKED(P,HEL,FLAV_IDX,ANS
     $ ,OK)
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      REAL*8 P(0:3,NEXTERNAL),ANS(0:1),ANS_SCALAR
      INTEGER HEL,FLAV_IDX
      LOGICAL OK
      CALL TX_SMATRIXHEL_CHECKED(
     $ P,HEL,FLAV_IDX,ANS_SCALAR,OK)
      ANS(0)=ANS_SCALAR
      ANS(1)=ANS_SCALAR
      END
      SUBROUTINE LEG_SMATRIX(P, FLAV_IDX, ANS)
C     

C     
C     MadGraph7 StandAlone Version
C     
C     Returns amplitude squared summed/avg over colors
C     and helicities
C     for the point in phase space P(0:3,NEXTERNAL)
C     
C     Pure source/control fixture, not a model.
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=4)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=4)
C     
C     ARGUMENTS
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
C     
C     LOCAL VARIABLES
C     
      INTEGER NHEL(NEXTERNAL,NCOMB)
C     put in common block to expose this variable to python interface
      COMMON/LEG_PROCESS_NHEL/NHEL
      REAL*8 T
      REAL*8 LEG_MATRIX
      INTEGER IHEL,IDEN, I, J, JJ
C     beam polarisation. Unpolarised unless something fills
C     /to_beampol/: PY_SET_BEAMPOL for the MadSpin density
C     modes, the driver's stdin for the v1 path. Convention as
C     in madevent's /to_polarization/: 1 is unpolarised,
C     |BEAMPOL| grows to 2 for a fully polarised beam and its
C     sign gives the favoured helicity.
      DOUBLE PRECISION BEAMPOL(2)
      COMMON/TO_BEAMPOL/BEAMPOL
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/2,2/
      INTEGER FLAVOR(NEXTERNAL)
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

C     
C     GLOBAL VARIABLES
C     
      INTEGER USERHEL
      COMMON/LEG_HELUSERCHOICE/USERHEL
      DATA USERHEL/-1/
      LOGICAL HELRESET
      COMMON/LEG_HELRESET/HELRESET
      DATA HELRESET/.TRUE./


      DATA IDEN/4/

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/LEG_BORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/

C     
C     FUNCTIONS
C     
      LOGICAL LEG_IS_BORN_HEL_SELECTED
      INTEGER LEG_BROKEN_SYM
C     RUNTIME-OBSERVER-BEGIN DECLARATION
      INTEGER FX_OBS_NTRY(NFLAV)
      LOGICAL FX_OBS_GOODHEL(NCOMB,NFLAV),FX_OBS_HELRESET
C     RUNTIME-OBSERVER-END DECLARATION
C     ----------
C     Check if helreset mode is on
C     ---------
      IF (HELRESET) THEN
        DO I=1,NFLAV
          NTRY(I) = 0
        ENDDO
        DO I=1,NCOMB
          DO J=1,NFLAV
            GOODHEL(I,J) = .FALSE.
          ENDDO
        ENDDO
        HELRESET = .FALSE.
      ENDIF

C     ----------
C     BEGIN CODE
C     ----------
C     FLAV_IDX=0 (or out of range) means GET_FLAVOR_INDEX could not
C      resolve
C     the requested flavor: it is not an allowed combination, so its
C      matrix
C     element is identically zero. Short-circuit before touching the
C     1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS = 0D0
        RETURN
      ENDIF
      CALL LEG_GET_FLAVOR(FLAV_IDX, FLAVOR)
      IF(USERHEL.EQ.-1) NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      DO IHEL=1,NEXTERNAL
        JC(IHEL) = +1
      ENDDO
C     When spin-2 particles are involved, the Helicity filtering is
C      dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points
C      have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero.
C      But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS
C      helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the
C      helicity sum is).
C     For this reason, we simply remove the filtering when there is
C      only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        DO IHEL=1,NCOMB
          DO J=1,NFLAV
            GOODHEL(IHEL,J)=.TRUE.
          ENDDO
        ENDDO
      ENDIF
      ANS = 0D0
      DO IHEL=1,NCOMB
        IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
          IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT.
     $      20.OR.USERHEL.NE.-1) THEN
            IF(NTRY(FLAV_IDX).GE.2.AND.POLARIZATIONS(0,0).NE.
     $       -1.AND.(.NOT.LEG_IS_BORN_HEL_SELECTED(IHEL))) THEN
              CYCLE
            ENDIF
            T=LEG_MATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX)
C           Support for polarised beam. Same reweighting of the
C            initial-state
C           helicity sum as madevent's matrix.f and as the v1 MadSpin
C            msP/msF
C           templates. Inert (and skipped) unless /to_beampol/ has
C            been filled:
C           |BEAMPOL| runs from 1 (unpolarised) to 2 (fully
C            polarised), so
C           anything at or below 1 -- including a zero-filled common
C            block --
C           means "no polarisation". 1 -> N matrix elements are left
C            alone: their
C           leg 1 is a decaying resonance, not a beam.
            IF (NINITIAL.EQ.2) THEN
              DO JJ=1,NINITIAL
                IF (ABS(BEAMPOL(JJ)).LE.1D0) CYCLE
                IF (NHEL(JJ,IHEL).EQ.INT(SIGN(1D0,BEAMPOL(JJ)))) THEN
                  T=T*ABS(BEAMPOL(JJ))
                ELSE
                  T=T*(2D0-ABS(BEAMPOL(JJ)))
                ENDIF
              ENDDO
            ENDIF
            IF(POLARIZATIONS(0,0).EQ.
     $       -1.OR.LEG_IS_BORN_HEL_SELECTED(IHEL)) THEN
              ANS=ANS+T
            ENDIF
            IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
              GOODHEL(IHEL,FLAV_IDX)=.TRUE.
            ENDIF
          ENDIF
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)*LEG_BROKEN_SYM(FLAVOR)
      IF(USERHEL.NE.-1) THEN
        ANS=ANS*HELAVGFACTOR
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).NE.-1) THEN
            ANS=ANS*BEAMS_HELAVGFACTOR(J)
            ANS=ANS/POLARIZATIONS(J,0)
          ENDIF
        ENDDO
      ENDIF
C     RUNTIME-OBSERVER-BEGIN ENTRY
      RETURN
      ENTRY LEG_OBSERVE_LEARNING(FX_OBS_NTRY,
     $ FX_OBS_GOODHEL,FX_OBS_HELRESET)
      FX_OBS_NTRY=NTRY
      FX_OBS_GOODHEL=GOODHEL
      FX_OBS_HELRESET=HELRESET
      RETURN
C     RUNTIME-OBSERVER-END ENTRY
      END



      SUBROUTINE LEG_SMATRIXHEL(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=4)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX

C     
C     ARGUMENTS
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
      INTEGER FLAV_IDX
C     
C     GLOBAL VARIABLES
C     
      INTEGER USERHEL
      COMMON/LEG_HELUSERCHOICE/USERHEL
C     ----------
C     BEGIN CODE
C     ----------
      USERHEL=HEL
      CALL LEG_SMATRIX(P,FLAV_IDX,ANS)
      USERHEL=-1

      END


