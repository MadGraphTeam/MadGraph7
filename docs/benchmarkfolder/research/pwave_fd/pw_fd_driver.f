      PROGRAM PW_FD_DRIVER
C     Scratch: the derivative of the loop amplitudes of a P-wave output with
C     frozen weights (test mode 1) against central differences of their
C     values (test mode 2) at the base relative momentum Q0 (2 -> 1, the
C     bound state at rest; Richardson with steps H and H/2)
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
      INCLUDE 'coupl.inc'
      REAL*8 PMASS(NEXTERNAL), E(3,3)
      REAL*8 P(0:3,NEXTERNAL), Q0(3), Q(3), H, MQ, M, PI
      REAL*8, ALLOCATABLE :: ANS(:,:), PREC(:)
      REAL*8 RDUAL(3), RP(3), RM(3), FD1(3), FD2(3), FD(3), HS
      INTEGER NDIM, NSO, RET, I, D, K, S
      PI = 4D0*DATAN(1D0)
      CALL SETPARA('param_card.dat')
      CALL SETONIA('onia_card.dat')
      INCLUDE 'pmass.inc'
      CALL ML5_0_GET_ANSWER_DIMENSION(NDIM)
      CALL ML5_0_GET_NSQSO_LOOP(NSO)
      ALLOCATE(ANS(0:3,0:NDIM),PREC(0:NSO))
      READ(*,*) Q0(1),Q0(2),Q0(3),H,K
      CALL ML5_0_PW_SET_PART(K)
C     the bound state at rest, mass 2 sqrt(m^2+q0^2)
      M=PMASS(3)/2D0
      CALL ML5_0_PW_SET_BASE(Q0)
      CALL ML5_0_PW_TEST_DIRS(E)
      CALL SETPS(Q0)
      MU_R=2D0*M
      CALL UPDATE_AS_PARAM()
C     mode 1: the dual derivative
      CALL ML5_0_PW_SET_TEST(1,0)
      CALL ML5_0_ONIA_SET_RELATIVE_MOMENTUM(Q0)
      CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
      RDUAL(1:3)=ANS(1:3,0)
      WRITE(*,'(A,3ES26.16,I5)') 'DUAL',RDUAL,RET
      CALL ML5_0_PW_SET_NODOUBLE(.TRUE.)
      CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
      WRITE(*,'(A,3ES26.16,I5)') 'DUAL_NODOUBLE',ANS(1:3,0),RET
      CALL ML5_0_PW_SET_NODOUBLE(.FALSE.)
C     mode 2: central differences, steps H and H/2
      FD1=0D0
      FD2=0D0
      DO K=1,2
        HS=H/DBLE(K)
        DO D=1,2
          CALL ML5_0_PW_SET_TEST(2,D)
          Q=Q0+HS*E(1:3,D)
          CALL SETPS(Q)
          CALL ML5_0_ONIA_SET_RELATIVE_MOMENTUM(Q)
          CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
          RP(1:3)=ANS(1:3,0)
          Q=Q0-HS*E(1:3,D)
          CALL SETPS(Q)
          CALL ML5_0_ONIA_SET_RELATIVE_MOMENTUM(Q)
          CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
          RM(1:3)=ANS(1:3,0)
          IF (K.EQ.1) FD1=FD1+(RP-RM)/(2D0*HS)
          IF (K.EQ.2) FD2=FD2+(RP-RM)/(2D0*HS)
        ENDDO
      ENDDO
      FD=(4D0*FD2-FD1)/3D0
C     the dual derivative again, after the initialisation of MadLoop
      CALL SETPS(Q0)
      CALL ML5_0_PW_SET_TEST(1,0)
      CALL ML5_0_ONIA_SET_RELATIVE_MOMENTUM(Q0)
      CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
      WRITE(*,'(A,3ES26.16,I5)') 'DUAL2',ANS(1:3,0),RET
      RDUAL(1:3)=ANS(1:3,0)
      WRITE(*,'(A,3ES26.16)') 'FD_H',FD1
      WRITE(*,'(A,3ES26.16)') 'FD_H2',FD2
      WRITE(*,'(A,3ES26.16)') 'FD',FD
      DO I=1,3
        WRITE(*,'(A,I2,ES12.3)') 'RELDIFF',I,
     &   ABS(FD(I)-RDUAL(I))/MAX(ABS(RDUAL(I)),1D-300)
      ENDDO
            CONTAINS
      SUBROUTINE SETPS(QQ)
      REAL*8 QQ(3)
      MQ=2D0*SQRT(M**2+QQ(1)**2+QQ(2)**2+QQ(3)**2)
      P=0D0
      P(0,1)=MQ/2D0
      P(3,1)=MQ/2D0
      P(0,2)=MQ/2D0
      P(3,2)=-MQ/2D0
      P(0,3)=MQ
      END SUBROUTINE
      END
