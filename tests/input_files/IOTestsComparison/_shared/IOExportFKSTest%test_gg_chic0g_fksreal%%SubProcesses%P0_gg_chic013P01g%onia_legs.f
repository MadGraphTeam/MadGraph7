      LOGICAL FUNCTION FKS_ONIA_LEG(I)
C     True if leg I (physical numbering) of the real emission of
C     the current FKS configuration is a bound state (written by
C     export_fks.write_onia_legs_file)
      IMPLICIT NONE
      INTEGER I, IPOS
      INTEGER NCONF, NEXT
      PARAMETER (NCONF=9, NEXT=5)
      LOGICAL ONIUM(NCONF,NEXT)
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      DATA (ONIUM(1,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (ONIUM(2,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (ONIUM(3,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (ONIUM(4,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (ONIUM(5,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (ONIUM(6,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (ONIUM(7,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (ONIUM(8,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (ONIUM(9,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      FKS_ONIA_LEG = .FALSE.
      IF (NFKSPROCESS.LT.1 .OR. NFKSPROCESS.GT.NCONF) RETURN
      IF (I.LT.1 .OR. I.GT.NEXT) RETURN
      FKS_ONIA_LEG = ONIUM(NFKSPROCESS,I)
      END


      LOGICAL FUNCTION FKS_ONIA_NOSECTOR(I)
C     True if leg I (physical numbering) of the real emission of
C     the current FKS configuration is a bound state that owns no
C     FKS sector of that real: it is left out of the partition
C     (fks_Sij). Onia that are the FKS sisters of the gluon
C     (soft partners, onium_emission_split) are part of it.
      IMPLICIT NONE
      INTEGER I, IPOS
      INTEGER NCONF, NEXT
      PARAMETER (NCONF=9, NEXT=5)
      LOGICAL NOSEC(NCONF,NEXT)
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      DATA (NOSEC(1,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .FALSE.,
     $  .FALSE., .FALSE. /
      DATA (NOSEC(2,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .FALSE.,
     $  .FALSE., .FALSE. /
      DATA (NOSEC(3,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .FALSE.,
     $  .FALSE., .FALSE. /
      DATA (NOSEC(4,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .FALSE.,
     $  .FALSE., .FALSE. /
      DATA (NOSEC(5,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (NOSEC(6,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (NOSEC(7,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (NOSEC(8,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      DATA (NOSEC(9,IPOS),IPOS=1,NEXT) / .FALSE., .FALSE., .TRUE.,
     $  .FALSE., .FALSE. /
      FKS_ONIA_NOSECTOR = .FALSE.
      IF (NFKSPROCESS.LT.1 .OR. NFKSPROCESS.GT.NCONF) RETURN
      IF (I.LT.1 .OR. I.GT.NEXT) RETURN
      FKS_ONIA_NOSECTOR = NOSEC(NFKSPROCESS,I)
      END


      SUBROUTINE ONIA_SET_LDME_COMMON()
C     The LDMEs of the onia card in COMMON/LDME for MadLoop's
C      bound-state
C     layer (onia_loop.f), once (export_fks.ONIA_SET_LDME_COMMON)
      IMPLICIT NONE
      CHARACTER*40 FILES(4)
      DATA FILES /'onia_card.dat', '../onia_card.dat',
     &  '../../Cards/onia_card.dat', '../../../Cards/onia_card.dat'/
      INTEGER K
      LOGICAL FOUND, DONE
      DATA DONE/.FALSE./
      SAVE DONE
      IF (DONE) RETURN
      DONE = .TRUE.
      DO K = 1, 4
        INQUIRE(FILE=FILES(K), EXIST=FOUND)
        IF (FOUND) THEN
          CALL SETONIA(TRIM(FILES(K)))
          RETURN
        ENDIF
      ENDDO
      WRITE(*,*) 'Error: the onia card onia_card.dat was not found'
      STOP 1
      END


      LOGICAL FUNCTION ONIA_BORN_VANISHES()
C     True if the Born vanishes identically and has no P-wave soft
C     current: its finite real is integrated in the hard
C     real-emission channels only (export_fks.write_onia_legs_file)
      IMPLICIT NONE
      ONIA_BORN_VANISHES = .FALSE.
      END

