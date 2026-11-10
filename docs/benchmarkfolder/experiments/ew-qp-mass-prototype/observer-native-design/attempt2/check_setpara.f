C     Actual model/card startup only: no amplitude or reducer link inputs.
C     Model definitions and archive are the same frozen generated model.
C     Integer storage-word output; existing MP reader promotes REAL*8.
      PROGRAM EW_CARD_SMOKE
      USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT64
      USE, INTRINSIC :: IEEE_EXCEPTIONS
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE 'mp_coupl.inc'
      REAL*8 VALUES(9),EXPECTED(9)
      REAL*16 MPVALUES(9),PROMOTED
      INTEGER(INT64) WORD8,WORDS16(2),WANT16(2)
      INTEGER I
      LOGICAL FLAGS(5)
      CHARACTER*2 NAMES(9)
      DATA NAMES/'MT','MZ','MW','MH','WT','WZ','WW','WH','MU'/
      DATA EXPECTED/173.3D0,91.188D0,80.419D0,125D0,
     $ 0D0,0D0,0D0,0D0,1000D0/
      OPEN(918,FILE='card-smoke.hex',STATUS='NEW')
      WRITE(918,'(A)') 'EW_CARD_SMOKE_V1'
      FLUSH(918)
      CALL SETPARA('MadLoop5_resources/param_card.dat')
      CALL IEEE_GET_FLAG(IEEE_ALL,FLAGS)
      VALUES=(/MDL_MT,MDL_MZ,MDL_MW,MDL_MH,
     $ MDL_WT,MDL_WZ,MDL_WW,MDL_WH,MU_R/)
      MPVALUES=(/MP__MDL_MT,MP__MDL_MZ,MP__MDL_MW,MP__MDL_MH,
     $ MP__MDL_WT,MP__MDL_WZ,MP__MDL_WW,MP__MDL_WH,
     $ REAL(MP__MU_R,KIND=16)/)
      DO I=1,9
        WORD8=TRANSFER(VALUES(I),WORD8)
        WORDS16=TRANSFER(MPVALUES(I),WORDS16)
        PROMOTED=REAL(EXPECTED(I),KIND=16)
        WANT16=TRANSFER(PROMOTED,WANT16)
        WRITE(918,'(A,1X,A,3(1X,Z16.16))')
     $    'MODEL',NAMES(I),WORD8,WORDS16
        FLUSH(918)
        IF(WORD8.NE.TRANSFER(EXPECTED(I),WORD8)) STOP 91
        IF(ANY(WORDS16.NE.WANT16)) STOP 92
      ENDDO
      IF(AIMAG(MP__MU_R).NE.0E0_16) STOP 93
      WRITE(918,'(A,5(1X,L1))') 'MODEL_IEEE_FLAGS',FLAGS
      WRITE(918,'(A)') 'COMPLETE_CARD_ONLY'
      CLOSE(918)
      END
