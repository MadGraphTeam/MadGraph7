      PROGRAM CMS_ONSHELL_PROBE
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      REAL*8 P(2,0:3),RAW
      REAL*16 QP(2,0:3),QRAW
      COMPLEX*16 M(2),S(2,2)
      COMPLEX*32 QM(2),QS(2,2)
      OSTHRES=1D-8
      M=DCMPLX(0D0,0D0)
      QM=CMPLX(0.0_16,0.0_16,KIND=16)
      OPEN(73,FILE='cms_onshell_probe.dat',STATUS='REPLACE')
      P(1,0)=5.00000000000000000D+02
      P(1,1)=0.00000000000000000D+00
      P(1,2)=0.00000000000000000D+00
      P(1,3)=5.00000000000000000D+02
      P(2,0)=5.00000000000000000D+02
      P(2,1)=4.99999916666670818D-01
      P(2,2)=0.00000000000000000D+00
      P(2,3)=4.99999750000020811D+02
      QP=REAL(P,KIND=16)
      RAW=(P(1,0)-P(2,0))**2-SUM((P(1,1:3)-P(2,1:3))**2)
      QRAW=(QP(1,0)-QP(2,0))**2-
     $ SUM((QP(1,1:3)-QP(2,1:3))**2)
      CALL ML5_0_BUILD_KINEMATIC_MATRIX(2,P,M,S)
      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(2,QP,QM,QS)
      WRITE(73,'(A,4ES45.32E4)') 'forward-0.001 ',RAW,
     $ DBLE(S(1,2)),QRAW,REAL(QS(1,2),KIND=16)
      P(1,0)=5.00000000000000000D+02
      P(1,1)=0.00000000000000000D+00
      P(1,2)=0.00000000000000000D+00
      P(1,3)=5.00000000000000000D+02
      P(2,0)=5.00000000000000000D+02
      P(2,1)=4.99999999999916635D-04
      P(2,2)=0.00000000000000000D+00
      P(2,3)=4.99999999999750003D+02
      QP=REAL(P,KIND=16)
      RAW=(P(1,0)-P(2,0))**2-SUM((P(1,1:3)-P(2,1:3))**2)
      QRAW=(QP(1,0)-QP(2,0))**2-
     $ SUM((QP(1,1:3)-QP(2,1:3))**2)
      CALL ML5_0_BUILD_KINEMATIC_MATRIX(2,P,M,S)
      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(2,QP,QM,QS)
      WRITE(73,'(A,4ES45.32E4)') 'forward-1e-06 ',RAW,
     $ DBLE(S(1,2)),QRAW,REAL(QS(1,2),KIND=16)
      P(1,0)=5.00000000000000000D+02
      P(1,1)=0.00000000000000000D+00
      P(1,2)=0.00000000000000000D+00
      P(1,3)=5.00000000000000000D+02
      P(2,0)=5.00000000000000000D+02
      P(2,1)=5.00000000000000041D-06
      P(2,2)=0.00000000000000000D+00
      P(2,3)=5.00000000000000000D+02
      QP=REAL(P,KIND=16)
      RAW=(P(1,0)-P(2,0))**2-SUM((P(1,1:3)-P(2,1:3))**2)
      QRAW=(QP(1,0)-QP(2,0))**2-
     $ SUM((QP(1,1:3)-QP(2,1:3))**2)
      CALL ML5_0_BUILD_KINEMATIC_MATRIX(2,P,M,S)
      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(2,QP,QM,QS)
      WRITE(73,'(A,4ES45.32E4)') 'forward-1e-08 ',RAW,
     $ DBLE(S(1,2)),QRAW,REAL(QS(1,2),KIND=16)
      CLOSE(73)
      END
      SUBROUTINE ML5_0_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L
     $ ,S_MAT)
C     
C     Helper function that compute the loop kinematic matrix with
C      proper thresholds
C     NLOOPLINE : Number of loop lines
C     P_LOOP    : List of external momenta running in the loop, i.e.
C      q_i in the denominator (l_i+q_i)**2-m_i**2
C     M2L       : List of complex-valued masses running in the loop.
C     S_MAT(N,N): Kinematic matrix output.
C     
C     ARGUMENTS
C     
      INTEGER NLOOPLINE
      REAL*8 P_LOOP(NLOOPLINE,0:3)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 S_MAT(NLOOPLINE,NLOOPLINE)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      COMPLEX*16 DIFFSQ
      REAL*8 REF_NORMALIZATION

C     ----------
C     BEGIN CODE
C     ----------

      DO I=1,NLOOPLINE
        DO J=1,NLOOPLINE

          IF(I.EQ.J)THEN
            S_MAT(I,J)=-(M2L(I)+M2L(J))
          ELSE
            DIFFSQ = (DCMPLX(P_LOOP(I,0),0.0D0)-DCMPLX(P_LOOP(J,0)
     $       ,0.0D0))**2
            DO K=1,3
              DIFFSQ = DIFFSQ - (DCMPLX(P_LOOP(I,K),0.0D0)
     $         -DCMPLX(P_LOOP(J,K),0.0D0))**2
            ENDDO
C           Default value of the kinematic matrix
            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)
C           And we now test various thresholds. Normally, at most one
C            applies.
            IF(ABS(M2L(I)).NE.0.0D0)THEN
              IF(ABS((DIFFSQ-M2L(I))/M2L(I)).LT.OSTHRES)THEN
                S_MAT(I,J)=-M2L(J)
              ENDIF
            ENDIF
            IF(ABS(M2L(J)).NE.0.0D0)THEN
              IF(ABS((DIFFSQ-M2L(J))/M2L(J)).LT.OSTHRES)THEN
                S_MAT(I,J)=-M2L(I)
              ENDIF
            ENDIF
C           Choose what seems the most appropriate way to compare
C           massless onshellness.
            REF_NORMALIZATION=0.0D0
C           Here, we chose to base the threshold only on the energy
C            component
            DO K=0,0
              REF_NORMALIZATION = REF_NORMALIZATION + ABS(P_LOOP(I,K))
     $          + ABS(P_LOOP(J,K))
            ENDDO
            REF_NORMALIZATION = (REF_NORMALIZATION/2.0D0)**2
            IF(REF_NORMALIZATION.NE.0.0D0)THEN
              IF(ABS(DIFFSQ/REF_NORMALIZATION).LT.OSTHRES)THEN
                S_MAT(I,J)=-(M2L(I)+M2L(J))
              ENDIF
            ENDIF
          ENDIF

        ENDDO
      ENDDO

      END
      SUBROUTINE ML5_0_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L
     $ ,S_MAT)
C     
C     Helper function that compute the loop kinematic matrix with
C      proper thresholds
C     NLOOPLINE : Number of loop lines
C     P_LOOP    : List of external momenta running in the loop, i.e.
C      q_i in the denominator (l_i+q_i)**2-m_i**2
C     M2L       : List of complex-valued masses running in the loop.
C     S_MAT(N,N): Kinematic matrix output.
C     
C     ARGUMENTS
C     
      INTEGER NLOOPLINE
      REAL*16 P_LOOP(NLOOPLINE,0:3)
      COMPLEX*32 M2L(NLOOPLINE)
      COMPLEX*32 S_MAT(NLOOPLINE,NLOOPLINE)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      COMPLEX*32 DIFFSQ
      REAL*16 REF_NORMALIZATION

C     ----------
C     BEGIN CODE
C     ----------

      DO I=1,NLOOPLINE
        DO J=1,NLOOPLINE

          IF(I.EQ.J)THEN
            S_MAT(I,J)=-(M2L(I)+M2L(J))
          ELSE
            DIFFSQ = (CMPLX(P_LOOP(I,0),0.0E0_16,KIND=16)
     $       -CMPLX(P_LOOP(J,0),0.0E0_16,KIND=16))**2
            DO K=1,3
              DIFFSQ = DIFFSQ - (CMPLX(P_LOOP(I,K),0.0E0_16,KIND=16)
     $         -CMPLX(P_LOOP(J,K),0.0E0_16,KIND=16))**2
            ENDDO
C           Default value of the kinematic matrix
            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)
C           And we now test various thresholds. Normally, at most one
C            applies.
            IF(ABS(M2L(I)).NE.0.0E0_16)THEN
              IF(ABS((DIFFSQ-M2L(I))/M2L(I)).LT.OSTHRES)THEN
                S_MAT(I,J)=-M2L(J)
              ENDIF
            ENDIF
            IF(ABS(M2L(J)).NE.0.0E0_16)THEN
              IF(ABS((DIFFSQ-M2L(J))/M2L(J)).LT.OSTHRES)THEN
                S_MAT(I,J)=-M2L(I)
              ENDIF
            ENDIF
C           Choose what seems the most appropriate way to compare
C           massless onshellness.
            REF_NORMALIZATION=0.0E0_16
C           Here, we chose to base the threshold only on the energy
C            component
            DO K=0,0
              REF_NORMALIZATION = REF_NORMALIZATION + ABS(P_LOOP(I,K))
     $          + ABS(P_LOOP(J,K))
            ENDDO
            REF_NORMALIZATION = (REF_NORMALIZATION/2.0E0_16)**2
            IF(REF_NORMALIZATION.NE.0.0E0_16)THEN
              IF(ABS(DIFFSQ/REF_NORMALIZATION).LT.OSTHRES)THEN
                S_MAT(I,J)=-(M2L(I)+M2L(J))
              ENDIF
            ENDIF
          ENDIF

        ENDDO
      ENDDO

      END
