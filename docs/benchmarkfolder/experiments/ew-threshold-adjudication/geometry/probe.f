C     Private geometry diagnostic. No reducer or amplitude is evaluated.
      PROGRAM EW_THRESHOLD_GEOMETRY
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER(NEXTERNAL=5)
      INCLUDE 'coupl.inc'
      INCLUDE 'mp_coupl.inc'
      INCLUDE 'input.inc'
      INCLUDE 'mp_input.inc'
      INCLUDE 'MadLoopParams.inc'
      REAL*8 P(0:3,5),TOTAL(0:3),M2_DP
      REAL*16 RAW(0:3,5),MP_PS(0:3,5),MP_P(0:3,5)
      COMMON/ML5_0_MP_PSPOINT/MP_PS,MP_P
      INTEGER I,J,K,NP
      CHARACTER(512) PARAMFN
      CALL SETPARA('param_card.dat')
      PARAMFN='MadLoopParams.dat'
      CALL MADLOOPPARAMREADER(PARAMFN,.FALSE.)
      WRITE(*,'(A,I6)') 'IMPROVEPSPOINT ',IMPROVEPSPOINT
      WRITE(*,'(A,ES45.35E4)') 'INITIAL_OSTHRES ',REAL(OSTHRES,16)
      WRITE(*,'(A,ES45.35E4)') 'DP_TOP_PROMOTED ',REAL(MDL_MT,16)
      WRITE(*,'(A,ES45.35E4)') 'MP_TOP ',MP__MDL_MT
      WRITE(*,'(A,ES45.35E4)') 'EXACT_DECIMAL_TOP ',173.3E0_16
      M2_DP=MDL_MT**2
      WRITE(*,'(A,ES45.35E4)') 'DP_SQUARE_PROMOTED ',REAL(M2_DP,16)
      WRITE(*,'(A,ES45.35E4)') 'MP_SQUARE ',MP__MDL_MT**2
      WRITE(*,'(A,ES45.35E4)') 'MP_MINUS_DP_SQUARE ',
     $ MP__MDL_MT**2-REAL(M2_DP,16)
      WRITE(*,'(A,2ES45.35E4)') 'DP_AEW ',REAL(MDL_AEW,16),
     $ 0E0_16
      WRITE(*,'(A,2ES45.35E4)') 'MP_AEW ',REAL(MP__MDL_AEW,16),
     $ 0E0_16
      WRITE(*,'(A,ES45.35E4)') 'DP_GAL_PROMOTED ',REAL(GAL(1),16)
      WRITE(*,'(A,2ES45.35E4)') 'MP_GAL ',REAL(MP__GAL(1),16),
     $ AIMAG(MP__GAL(1))
      OPEN(73,FILE='selected.dat',STATUS='OLD')
      READ(73,*) NP
      IF(NP.NE.2) STOP 1
      DO K=1,NP
        DO I=1,5
          READ(73,*) P(:,I)
        ENDDO
        RAW=REAL(P,16)
        TOTAL=P(:,1)+P(:,2)
        MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
        CALL UPDATE_AS_PARAM()
        WRITE(*,'(A,I4,3ES45.35E4)') 'SCALE ',K,REAL(MU_R,16),
     $   REAL(MU_R**2,16),REAL(MU_R,16)**2
        CALL ML5_0_SET_MP_PS(P)
        DO I=1,5
          WRITE(*,'(A,2I4,4ES45.35E4)') 'RAW ',K,I,RAW(:,I)
          WRITE(*,'(A,2I4,4ES45.35E4)') 'MP_PS ',K,I,MP_PS(:,I)
          WRITE(*,'(A,2I4,4ES45.35E4)') 'MP_P ',K,I,MP_P(:,I)
        ENDDO
        CALL EW_GEOMETRY_SUMMARY(K,0,RAW,MP__MDL_MT)
        CALL EW_GEOMETRY_SUMMARY(K,1,MP_P,MP__MDL_MT)
        CALL EW_GEOMETRY_MATRICES(K,0,RAW,REAL(M2_DP,16))
        CALL EW_GEOMETRY_MATRICES(K,1,MP_P,REAL(M2_DP,16))
      ENDDO
      CLOSE(73)
      END

      SUBROUTINE EW_GEOMETRY_SUMMARY(K,STAGE,P,MASS)
      IMPLICIT NONE
      INTEGER K,STAGE,I
      REAL*16 P(0:3,5),MASS,V(0:3),S,DELTA,BETA,QDOT,RES
      DO I=1,5
        RES=QDOT(P(:,I))
        IF(I.EQ.3.OR.I.EQ.4) RES=RES-MASS**2
        WRITE(*,'(A,3I4,ES45.35E4)') 'SHELL ',K,STAGE,I,RES
      ENDDO
      V=P(:,1)+P(:,2)-P(:,3)-P(:,4)-P(:,5)
      WRITE(*,'(A,2I4,4ES45.35E4)') 'CONSERVATION ',K,STAGE,V
      V=P(:,3)+P(:,4)
      S=QDOT(V)
      DELTA=S-4E0_16*MASS**2
      BETA=SIGN(SQRT(ABS(DELTA/S)),DELTA)
      WRITE(*,'(A,2I4,4ES45.35E4)') 'PAIR ',K,STAGE,S,DELTA,
     $ BETA,SQRT(S)-2E0_16*MASS
      END

      REAL*16 FUNCTION QDOT(P)
      IMPLICIT NONE
      REAL*16 P(0:3)
      QDOT=P(0)**2-P(1)**2-P(2)**2-P(3)**2
      END

      SUBROUTINE EW_GEOMETRY_MATRICES(K,STAGE,P,MASS2)
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER K,STAGE,I,J,H,PAT
      REAL*16 P(0:3,5),MASS2,PL(0:3,5),SHIFTS(5,0:3),V(0:3)
      REAL*16 QDOT,RAW_INV,ADJUSTED,REF
      REAL*8 THRESHOLDS(3)
      COMPLEX*32 MASSES(5),SMAT(5,5)
      DATA THRESHOLDS/1D-8,1D-14,1D-24/
C     Canonical five external-leg order, not a captured diagram order:
C     PL=(-p1,-p2,p3,p4,p5), qi=sum(PL(1:i-1)), q1=0.
      PL=P
      PL(:,1)=-P(:,1)
      PL(:,2)=-P(:,2)
      SHIFTS=0E0_16
      DO I=2,5
        SHIFTS(I,:)=SHIFTS(I-1,:)+PL(:,I-1)
      ENDDO
      DO I=1,5
        WRITE(*,'(A,3I4,4ES45.35E4)') 'SHIFT ',K,STAGE,I,
     $   SHIFTS(I,:)
      ENDDO
      DO PAT=0,2
        MASSES=CMPLX(0E0_16,0E0_16,KIND=16)
        IF(PAT.EQ.1) MASSES=CMPLX(MASS2,0E0_16,KIND=16)
        IF(PAT.EQ.2) THEN
          MASSES(2)=CMPLX(MASS2,0E0_16,KIND=16)
          MASSES(4)=CMPLX(MASS2,0E0_16,KIND=16)
        ENDIF
        DO I=1,5
          WRITE(*,'(A,4I4,2ES45.35E4)') 'MASS2 ',K,STAGE,PAT,I,
     $     REAL(MASSES(I),16),AIMAG(MASSES(I))
        ENDDO
        DO H=1,3
          OSTHRES=THRESHOLDS(H)
          CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(5,SHIFTS,MASSES,SMAT)
          DO I=1,5
            DO J=1,5
              V=SHIFTS(I,:)-SHIFTS(J,:)
              RAW_INV=QDOT(V)
              ADJUSTED=REAL(SMAT(I,J)+MASSES(I)+MASSES(J),16)
              REF=((ABS(SHIFTS(I,0))+ABS(SHIFTS(J,0)))/2E0_16)**2
              WRITE(*,'(A,6I4,6ES45.35E4)') 'MATRIX ',K,STAGE,PAT,
     $         H,I,J,REAL(OSTHRES,16),RAW_INV,ADJUSTED,
     $         REAL(SMAT(I,J),16),AIMAG(SMAT(I,J)),REF
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      END
