      PROGRAM PW2_FD_DRIVER
C     Scratch: two P-wave bound states (e+ e- > onium1 onium2, an output
C     patched by fd2_patch.py). The mixed derivative of the loop
C     amplitudes with frozen weights (test mode 1) against mixed central
C     differences of their values (test mode 2) at the base relative
C     momenta Q01, Q02 of the two states (Richardson with steps H, H/2).
C     The bound states have the masses 2 sqrt(m^2+q_i^2) at every point,
C     so that every external leg stays on its mass shell; sqrt(s) and the
C     direction of the first bound state are fixed.
C     Input: SQRTS Q01(1:3) Q02(1:3) H MUR
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
      INCLUDE 'coupl.inc'
      REAL*8 PMASS(NEXTERNAL), E1(3,3), E2(3,3)
      REAL*8 P(0:3,NEXTERNAL), Q01(3), Q02(3), Q1(3), Q2(3), H, HS
      REAL*8 M, SQRTS, MUR
      REAL*8, ALLOCATABLE :: ANS(:,:), PREC(:)
      REAL*8 RDUAL(3), RV(3), FDK(3,2), FD(3)
      INTEGER NDIM, NSO, RET, I, D1, D2, IDIR, K, S1, S2
      CALL SETPARA('param_card.dat')
      CALL SETONIA('onia_card.dat')
      INCLUDE 'pmass.inc'
      CALL ML5_0_GET_ANSWER_DIMENSION(NDIM)
      CALL ML5_0_GET_NSQSO_LOOP(NSO)
      ALLOCATE(ANS(0:3,0:NDIM),PREC(0:NSO))
      READ(*,*) SQRTS,Q01(1),Q01(2),Q01(3),Q02(1),Q02(2),Q02(3),H,MUR
C     the constituent mass (equal masses, a bound state of mass 2 m)
      M=PMASS(3)/2D0
      MU_R=MUR
      CALL UPDATE_AS_PARAM()
      CALL ML5_0_PW2_SET_BASE(Q01,Q02)
      CALL ML5_0_PW2_TEST_DIRS(1,E1)
      CALL ML5_0_PW2_TEST_DIRS(2,E2)
C     mode 1: the mixed dual derivative
      CALL ML5_0_PW_SET_TEST(1,0)
      CALL ML5_0_PW2_SET_RELMOM(Q01,Q02)
      CALL SETPS(Q01,Q02)
      CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
      WRITE(*,'(A,3ES26.16,I5)') 'DUAL',ANS(1:3,0),RET
C     mode 2: mixed central differences, steps H and H/2, summed over
C     the four pairs of test directions perpendicular to Q01 and Q02
      FDK=0D0
      DO K=1,2
        HS=H/DBLE(K)
        DO D2=1,2
          DO D1=1,2
            IDIR=D1+3*(D2-1)
            CALL ML5_0_PW_SET_TEST(2,IDIR)
            DO S2=-1,1,2
              DO S1=-1,1,2
                Q1=Q01+S1*HS*E1(1:3,D1)
                Q2=Q02+S2*HS*E2(1:3,D2)
                CALL ML5_0_PW2_SET_RELMOM(Q1,Q2)
                CALL SETPS(Q1,Q2)
                CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
                RV(1:3)=ANS(1:3,0)
                FDK(1:3,K)=FDK(1:3,K)+S1*S2*RV(1:3)/(4D0*HS*HS)
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      FD=(4D0*FDK(1:3,2)-FDK(1:3,1))/3D0
C     the dual derivative again, after the initialisation of MadLoop
      CALL ML5_0_PW_SET_TEST(1,0)
      CALL ML5_0_PW2_SET_RELMOM(Q01,Q02)
      CALL SETPS(Q01,Q02)
      CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
      RDUAL(1:3)=ANS(1:3,0)
      WRITE(*,'(A,3ES26.16,I5)') 'DUAL2',RDUAL,RET
      WRITE(*,'(A,3ES26.16)') 'FD_H',FDK(1:3,1)
      WRITE(*,'(A,3ES26.16)') 'FD_H2',FDK(1:3,2)
      WRITE(*,'(A,3ES26.16)') 'FD',FD
      DO I=1,3
        WRITE(*,'(A,I2,ES12.3)') 'RELDIFF',I,
     &   ABS(FD(I)-RDUAL(I))/MAX(ABS(RDUAL(I)),1D-300)
      ENDDO
            CONTAINS
      SUBROUTINE SETPS(QA,QB)
      REAL*8 QA(3), QB(3), MA, MB, E3, K3, N(3), TH, PH
      MA=2D0*SQRT(M**2+QA(1)**2+QA(2)**2+QA(3)**2)
      MB=2D0*SQRT(M**2+QB(1)**2+QB(2)**2+QB(3)**2)
      TH=0.9D0
      PH=0.4D0
      N(1)=SIN(TH)*COS(PH)
      N(2)=SIN(TH)*SIN(PH)
      N(3)=COS(TH)
      P=0D0
      P(0,1)=SQRTS/2D0
      P(3,1)=SQRTS/2D0
      P(0,2)=SQRTS/2D0
      P(3,2)=-SQRTS/2D0
      E3=(SQRTS**2+MA**2-MB**2)/(2D0*SQRTS)
      K3=SQRT(E3**2-MA**2)
      P(0,3)=E3
      P(1:3,3)=K3*N
      P(0,4)=SQRTS-E3
      P(1:3,4)=-K3*N
      END SUBROUTINE
      END
