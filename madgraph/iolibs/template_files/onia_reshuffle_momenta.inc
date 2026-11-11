      IMPLICIT NONE
C
C     Maps the phase-space momenta P, generated with the physical onium
C     masses, onto momenta P_RESH with the masses PMASS (for an onium: the
C     sum of its constituent masses), arXiv:2607.26739.
C     2 -> N: initial-state (MOM_RESH_TYPE=0) or final-state (1, 2, 3)
C     reshuffling.
C     1 -> N (decays): the mass of the decaying particle is fixed, so only
C     the final-state reshuffling applies, at the parent's invariant mass
C     (type 0 falls back to type 1). If the parent itself changes mass (a
C     decaying onium at its physical mass), all decay products are instead
C     rescaled by a common factor in the parent's rest frame.
C     When no momenta P_RESH exist (not enough energy for the target
C     masses), RESH_FAILED is set and the caller returns a zero matrix element.
C
C     CONSTANT
C
      INCLUDE 'nexternal.inc'
      INCLUDE 'onia.inc'
C
C     ARGUMENTS
C
      REAL*8    P(0:3,NEXTERNAL), P_RESH(0:3,NEXTERNAL)
      REAL*8    PMASS(NEXTERNAL)
C
C     GLOBAL
C
      INTEGER MOM_RESH_TYPE
      COMMON/TO_MOM_RESH_TYPE/MOM_RESH_TYPE
      LOGICAL RESH_FAILED
      COMMON/TO_RESH_FAILED/RESH_FAILED
C
C     LOCALS
C
      LOGICAL RESHUFFLE, ISONIUM(NEXTERNAL), WARNED
      DATA WARNED/.FALSE./
      SAVE WARNED
      INTEGER I, J, CNT_ONIA, CNT_ELEM, NELEM, NFONIA, NIONIA, RESH_TYPE
      INTEGER IELEM
      INTEGER MAP(NEXTERNAL), ITER

      REAL*8 TINY
      PARAMETER (TINY=1D-99)

      ! HADRONIC SYSTEM
      REAL*8 HAD_P(0:3,NEXTERNAL), HAD_M(NEXTERNAL), HAD_K(0:3,NEXTERNAL)
      REAL*8 HAD_PIN(0:3,NEXTERNAL), HAD_MIN(NEXTERNAL), HAD_KX(0:3)
      REAL*8 HAD_SQRTS, MREC
C     the direction of the first incoming leg along the beam (+1 or -1): a
C     standalone call may have it along -z (the mirror of a subprocess)
      REAL*8 S1

      ! PARTONIC SYSTEM
      REAL*8 PAR_P(0:3,NEXTERNAL), PAR_M(NEXTERNAL), PAR_K(0:3,NEXTERNAL)
      REAL*8 PAR_PIN(0:3,NEXTERNAL), PAR_MIN(NEXTERNAL), PAR_KX(0:3)
      REAL*8 PAR_SQRTS, PAR_E1, PAR_E2

      ! AUXILIARY
      REAL*8 PCOM(0:3,NEXTERNAL), PAUX(0:3), PIN(0:3), UNIT_VECTOR(1:3)
      REAL*8 XI, F, DF, EI, MPARENT

C     ----------
C     BEGIN CODE
C     ----------

      RESH_FAILED = .FALSE.
      RESH_TYPE = MOM_RESH_TYPE
      ISONIUM = .FALSE.
      NIONIA = 0
      NFONIA = 0
      DO I = 1, NONIA
        ISONIUM(ONIA(I)) = .TRUE.
        IF (ONIA(I).LE.NINCOMING) THEN
          NIONIA = NIONIA + 1
        ELSE
          NFONIA = NFONIA + 1
        ENDIF
      ENDDO
      NELEM = NEXTERNAL-NINCOMING-NFONIA

C     Only an onium can differ from its target mass
      RESHUFFLE = .FALSE.
      DO I = 1, NEXTERNAL
        IF (ISONIUM(I)) THEN
          IF (ABS(PMASS(I)-MASS(P(:,I)))/PMASS(I).GT.1D-11) THEN
            RESHUFFLE = .TRUE.
          ENDIF
        ENDIF
      ENDDO
      IF (.NOT.RESHUFFLE) THEN
        P_RESH = P
        RETURN
      ENDIF

      IF (NINCOMING.EQ.1) THEN
        MPARENT = MASS(P(:,1))
        IF (ABS(PMASS(1)-MPARENT)/PMASS(1).GT.1D-11) THEN
C     DECAYING PARTICLE AT ANOTHER MASS: RESCALE IN ITS REST FRAME
          DO I = 2, NEXTERNAL
            CALL BOOST(P(0:3,1),+1D0,P(0:3,I),PCOM(0:3,I))
          ENDDO
          IF (SUM(PMASS(2:NEXTERNAL)).GE.PMASS(1)) THEN
            RESH_FAILED = .TRUE.
            P_RESH = P
            RETURN
          ENDIF
C         Newton iteration for XI: SUM_I SQRT(M_I^2+XI^2*|P_I|^2) = PMASS(1)
          XI = PMASS(1)/MPARENT
          DO ITER = 1, 100
            F = -PMASS(1)
            DF = 0D0
            DO I = 2, NEXTERNAL
              EI = SQRT(PMASS(I)**2+XI**2*PABS2(PCOM(:,I)))
              F = F + EI
              IF (EI.GT.0D0) DF = DF + XI*PABS2(PCOM(:,I))/EI
            ENDDO
            IF (DF.LE.0D0) EXIT
            XI = XI - F/DF
            IF (ABS(F).LT.1D-14*PMASS(1)) EXIT
          ENDDO
          P_RESH(0:3,1) = P(0:3,1)*PMASS(1)/MPARENT
          DO I = 2, NEXTERNAL
            PAUX(1:3) = XI*PCOM(1:3,I)
            PAUX(0) = SQRT(PMASS(I)**2+PABS2(PAUX))
            CALL BOOST(P_RESH(0:3,1),-1D0,PAUX,P_RESH(0:3,I))
          ENDDO
          RETURN
        ENDIF
        IF (RESH_TYPE.EQ.0) THEN
          IF (.NOT.WARNED) THEN
            PRINT*, 'Initial-state momentum reshuffling is not possible',
     &              ' for a decay: final-state reshuffling (1) is used.'
            WARNED = .TRUE.
          ENDIF
          RESH_TYPE = 1
        ENDIF
      ELSE IF (NIONIA.GT.0) THEN
        PRINT*, 'Momentum reshuffling with an incoming onium is only',
     &          ' possible for a 1->N decay.'
        STOP
      ENDIF

C     TOTAL INCOMING MOMENTUM
      PIN = 0D0
      DO I = 1, NINCOMING
        PIN = PIN + P(0:3,I)
      ENDDO

      IF (RESH_TYPE > 0) THEN
C     FINAL-STATE MOMENTUM RESHUFFLING
        DO I = 1, NEXTERNAL
          CALL BOOST(PIN,+1D0,P(0:3,I),PCOM(0:3,I))
        ENDDO
        S1 = 1D0
        IF (NINCOMING.EQ.2) S1 = SIGN(1D0,PCOM(3,1))

        CNT_ONIA = 0
        CNT_ELEM = 0
        HAD_KX(0:3) = 0D0
        DO I = NINCOMING+1, NEXTERNAL
          IF (ISONIUM(I)) THEN
            CNT_ONIA = CNT_ONIA + 1
            MAP(I) = CNT_ONIA
            HAD_P(0:3,CNT_ONIA) = PCOM(0:3,I)
            HAD_M(CNT_ONIA) = MASS(HAD_P(0:3,CNT_ONIA))
            PAR_M(CNT_ONIA) = PMASS(I)
          ELSE
            CNT_ELEM = CNT_ELEM + 1
            MAP(I) = -CNT_ELEM
            IELEM = I
            HAD_K(0:3,CNT_ELEM) = PCOM(0:3,I)
            HAD_KX(0:3) = HAD_KX(0:3) + HAD_K(0:3,CNT_ELEM)
          ENDIF
        ENDDO
        MREC = MASS(HAD_KX)
C       A single recoiling particle is put on its own mass shell: the
C       invariant mass of its momentum in an event record (11 digits)
C       is not its mass (0.016 GeV for a massless quark of 1.8 TeV), and
C       an event and its mirror in z gave it differently
        IF (NELEM.EQ.1) MREC = PMASS(IELEM)

        HAD_PIN(0:3,NFONIA) = HAD_P(0:3,NFONIA)
        HAD_MIN(NFONIA) = HAD_M(NFONIA)
        PAR_MIN(NFONIA) = PAR_M(NFONIA)
        DO I = NFONIA-1, 1, -1
          HAD_PIN(0:3,I) = HAD_PIN(0:3,I+1) + HAD_P(0:3,I)
          HAD_MIN(I) = MASS(HAD_PIN(0:3,I))
          PAR_MIN(I) = PHI(HAD_MIN(I),PAR_M(I),PAR_MIN(I+1),HAD_M(I),HAD_MIN(I+1),RESH_TYPE)
        ENDDO

        HAD_SQRTS = 0D0
        DO I = 1, NINCOMING
          HAD_SQRTS = HAD_SQRTS + PCOM(0,I)
        ENDDO
        IF (NINCOMING.EQ.1) THEN
C         The mass of the decaying particle is fixed; without a recoil
C         system the onia carry all of it.
          PAR_SQRTS = HAD_SQRTS
          IF (NELEM.EQ.0) PAR_MIN(1) = PAR_SQRTS
          IF (PAR_MIN(1)+MREC.GT.PAR_SQRTS*(1D0+1D-12)) THEN
            RESH_FAILED = .TRUE.
            P_RESH = P
            RETURN
          ENDIF
        ELSE IF (NELEM.EQ.0) THEN
C         Without a recoil system (e.g. a 2 -> 1 process) the onia carry
C         all of the energy: the partonic energy is their reshuffled mass.
C         PHI would keep it near the physical one, leaving the onia off
C         their mass shell and the event without momentum conservation.
          PAR_SQRTS = PAR_MIN(1)
        ELSE
          PAR_SQRTS = PHI(HAD_SQRTS,PAR_MIN(1),MREC,HAD_MIN(1),MREC,RESH_TYPE)
        ENDIF

        PAR_PIN(0,1) = EPSILON(PAR_SQRTS,PAR_MIN(1),MREC)
        IF (SQRT(PABS2(HAD_PIN(0:3,1))).GT.TINY) THEN
          PAR_PIN(1:3,1) = PI(PAR_SQRTS,PAR_MIN(1),MREC)*HAD_PIN(1:3,1)/SQRT(PABS2(HAD_PIN(0:3,1)))
        ELSE
          PAR_PIN(1:3,1) = 0D0
        ENDIF

        IF (NELEM.GT.0) THEN
          PAR_KX(0) = EPSILON(PAR_SQRTS,MREC,PAR_MIN(1))
          IF (SQRT(PABS2(HAD_KX)).GT.TINY) THEN
            PAR_KX(1:3) = PI(PAR_SQRTS,PAR_MIN(1),MREC)*HAD_KX(1:3)/SQRT(PABS2(HAD_KX))
          ELSE
            PAR_KX(1:3) = 0D0
          ENDIF
        ENDIF

        IF (NELEM.GT.1) THEN
          DO I = 1, NELEM
            CALL BOOST(HAD_KX,+1D0,HAD_K(0:3,I),PAUX)
            CALL BOOST(PAR_KX,-1D0,PAUX,PAR_K(0:3,I))
          ENDDO
        ELSE IF (NELEM.EQ.1) THEN
          PAR_K(0:3,1) = PAR_KX(0:3)
        ENDIF

        DO I = 1, NFONIA-1
          CALL BOOST(HAD_PIN(0:3,I),+1D0,HAD_P(0:3,I),PAUX)
          UNIT_VECTOR = PAUX(1:3)/PI(HAD_MIN(I),HAD_M(I),HAD_MIN(I+1))
          PAUX(0) = EPSILON(PAR_MIN(I),PAR_M(I),PAR_MIN(I+1))
          PAUX(1:3) = PI(PAR_MIN(I),PAR_M(I),PAR_MIN(I+1))*UNIT_VECTOR
          CALL BOOST(PAR_PIN(0:3,I),-1D0,PAUX,PAR_P(0:3,I))
          PAUX(0) = EPSILON(PAR_MIN(I),PAR_MIN(I+1),PAR_M(I))
          PAUX(1:3) = -PAUX(1:3)
          CALL BOOST(PAR_PIN(0:3,I),-1D0,PAUX,PAR_PIN(0:3,I+1))
        ENDDO
        PAR_P(0:3,NFONIA) = PAR_PIN(0:3,NFONIA)

        DO I = NINCOMING+1, NEXTERNAL
          J = MAP(I)
          IF (J.GT.0) THEN
            P_RESH(0:3,I) = PAR_P(0:3,J)
          ELSE
            P_RESH(0:3,I) = PAR_K(0:3,-J)
          ENDIF
        ENDDO
        IF (NINCOMING.EQ.1) THEN
          P_RESH(0:3,1) = (/PAR_SQRTS,0D0,0D0,0D0/)
        ELSE
          P_RESH(0:3,1) = 0.5D0*PAR_SQRTS*(/1D0,0D0,0D0,S1/)
          P_RESH(0:3,2) = 0.5D0*PAR_SQRTS*(/1D0,0D0,0D0,-S1/)
        ENDIF

        PCOM = P_RESH
        DO I = 1, NEXTERNAL
          CALL BOOST(PIN,-1D0,PCOM(0:3,I),P_RESH(0:3,I))
        ENDDO
      ELSE
C     INITIAL-STATE RESHUFFLING
        DO I = NINCOMING+1, NEXTERNAL
          IF (MASS(P(0:3,I)).NE.PMASS(I)) THEN
            P_RESH(1:3,I) = P(1:3,I)
            P_RESH(0,I) = SQRT(PMASS(I)**2+PABS2(P(0:3,I)))
          ELSE
            P_RESH(0:3,I) = P(0:3,I)
          ENDIF
        ENDDO
        PAR_SQRTS = SUM(P_RESH(0,NINCOMING+1:))
        PAR_E1 = 0.5D0*(PAR_SQRTS+P(0,1)-P(0,2))
        PAR_E2 = 0.5D0*(PAR_SQRTS-P(0,1)+P(0,2))
        S1 = SIGN(1D0,P(3,1))
        P_RESH(0:3,1) = PAR_E1*(/ 1D0, 0D0, 0D0, S1/)
        P_RESH(0:3,2) = PAR_E2*(/ 1D0, 0D0, 0D0,-S1/)
      ENDIF

      CONTAINS

        DOUBLE PRECISION FUNCTION MASS(P)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8    P(0:3)

C         ----------
C         BEGIN CODE
C         ----------
C         A mass below the rounding of E^2-|p|^2 is zero: relative to the
C         energy (an absolute cut-off would zero the invariant mass of a
C         nearly collinear recoil system, whose rest frame then has an
C         infinite boost)
          MASS = P(0)*P(0)-P(1)*P(1)-P(2)*P(2)-P(3)*P(3)
          IF (MASS.LT.1D-14*P(0)*P(0)) THEN
            MASS = 0D0
          ELSE
            MASS = SQRT(MASS)
          ENDIF
          RETURN
        END

        DOUBLE PRECISION FUNCTION PABS2(P)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8    P(0:3)

C         ----------
C         BEGIN CODE
C         ----------
C         Exact: a cut-off would give a soft massless particle zero energy
C         and a non-zero momentum (off shell) in the rescale of a decay
          PABS2 = P(1)*P(1)+P(2)*P(2)+P(3)*P(3)
          RETURN
        END

        DOUBLE PRECISION FUNCTION PI(M,M1,M2)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8 M,M1,M2

C         ----------
C         BEGIN CODE
C         ----------
          PI = 0.5D0*M*SQRT(1D0-(M1+M2)**2/M**2)*SQRT(1D0-(M1-M2)**2/M**2)
          RETURN
        END

        DOUBLE PRECISION FUNCTION EPSILON(M,M1,M2)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8 M,M1,M2

C         ----------
C         BEGIN CODE
C         ----------
          EPSILON = 0.5D0*M*(1D0+(M1**2-M2**2)/M**2)
          RETURN
        END

        DOUBLE PRECISION FUNCTION PHI(M,MP1,MP2,M1,M2,MOM_RESH_TYPE)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          INTEGER    MOM_RESH_TYPE
          REAL*8     M,MP1,MP2,M1,M2
C
C         LOCALS
C
          REAL*8     N
          PARAMETER (N=100D0)

C         ----------
C         BEGIN CODE
C         ----------
          IF (MOM_RESH_TYPE == 1) THEN
            PHI = 0.5D0*(M+MP1+MP2+SQRT((M-MP1-MP2)**2+(M1+M2-MP1-MP2)**2/N))
          ELSE IF (MOM_RESH_TYPE == 2) THEN
            IF (M.GE.(MP1+MP2)) THEN
              PHI = M
            ELSE
              PHI = (MP1+MP2)
            ENDIF
          ELSE IF (MOM_RESH_TYPE == 3) THEN
            IF (M.GE.(MP1+MP2)) THEN
              PHI = M
            ELSE
              PHI = MAX((MP1+MP2)/(M1+M2)*M, MP1+MP2)
            ENDIF
          ELSE
            STOP 1
          END IF
        END

        SUBROUTINE BOOST(PREF, DIRECTION, P, PBOOST)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8 PREF(0:3), DIRECTION, P(0:3), PBOOST(0:3)
C
C         LOCALS
C
          REAL*8 BETA(1:3), BETA2, GAMMA, LAMBDA(0:3,0:3)
          REAL*8 TINY
          PARAMETER (TINY=1D-99)

C         ----------
C         BEGIN CODE
C         ----------
          BETA(1:3) = DIRECTION*PREF(1:3)/PREF(0)

          BETA2 = BETA(1)*BETA(1)+BETA(2)*BETA(2)+BETA(3)*BETA(3)
          GAMMA = 1D0/SQRT(1D0-BETA2)
          BETA2 = MAX(BETA2,TINY)

C         CONSTRUCT LORENTZ TRANSFORMATION MATRIX
          LAMBDA(0,0) = GAMMA
          LAMBDA(0,1) = -GAMMA*BETA(1)
          LAMBDA(0,2) = -GAMMA*BETA(2)
          LAMBDA(0,3) = -GAMMA*BETA(3)

          LAMBDA(1,0) = LAMBDA(0,1)
          LAMBDA(1,1) = 1D0+(GAMMA-1D0)*BETA(1)*BETA(1)/BETA2
          LAMBDA(1,2) = (GAMMA-1D0)*BETA(1)*BETA(2)/BETA2
          LAMBDA(1,3) = (GAMMA-1D0)*BETA(1)*BETA(3)/BETA2

          LAMBDA(2,0) = LAMBDA(0,2)
          LAMBDA(2,1) = LAMBDA(1,2)
          LAMBDA(2,2) = 1D0+(GAMMA-1D0)*BETA(2)*BETA(2)/BETA2
          LAMBDA(2,3) = (GAMMA-1D0)*BETA(2)*BETA(3)/BETA2

          LAMBDA(3,0) = LAMBDA(0,3)
          LAMBDA(3,1) = LAMBDA(1,3)
          LAMBDA(3,2) = LAMBDA(2,3)
          LAMBDA(3,3) = 1D0+(GAMMA-1D0)*BETA(3)*BETA(3)/BETA2

C         APPLY TRANSFORMATION
          PBOOST = MATMUL(LAMBDA,P)
        END
      END
