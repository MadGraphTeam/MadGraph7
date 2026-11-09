      PROGRAM CMS_MP_SHELL_PROBE
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER NPOINTS,K,I,J,H
      REAL*8 P(0:3,4),THRESHOLDS(3)
      REAL*16 MP_PS(0:3,4),MP_P(0:3,4),Q(4,0:3),RAW
      COMMON/ML5_0_MP_PSPOINT/MP_PS,MP_P
      COMPLEX*32 M(4),S(4,4)
      CHARACTER*512 CARD
      DATA THRESHOLDS/1D-8,1D-14,1D-24/
      CALL SETPARA('param_card.dat')
      CARD='MadLoop5_resources/MadLoopParams.dat'
      CALL MADLOOPPARAMREADER(CARD,.FALSE.)
      M=CMPLX(0.0_16,0.0_16,KIND=16)
      OPEN(73,FILE='mp_shell_points.dat',STATUS='OLD')
      OPEN(74,FILE='cms_mp_shell_probe.dat',STATUS='REPLACE')
      READ(73,*) NPOINTS
      DO K=1,NPOINTS
        DO I=1,4
          READ(73,*) P(:,I)
        ENDDO
        CALL ML5_0_SET_MP_PS(P)
        DO I=1,4
          WRITE(74,'(A,2I4,4ES47.34E4)') 'RAW ',K,I,
     $      REAL(P(:,I),KIND=16)
          WRITE(74,'(A,2I4,4ES47.34E4)') 'IMPROVED ',K,I,
     $      MP_PS(:,I)
          RAW=MP_PS(0,I)**2-SUM(MP_PS(1:3,I)**2)
          WRITE(74,'(A,2I4,ES47.34E4)') 'SHELL ',K,I,RAW
        ENDDO
        Q(1,:)=0.0_16
        Q(2,:)=MP_PS(:,1)
        Q(3,:)=MP_PS(:,1)+MP_PS(:,2)
        Q(4,:)=MP_PS(:,3)
        DO H=1,3
          OSTHRES=THRESHOLDS(H)
          CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(4,Q,M,S)
          DO I=1,4
            DO J=1,4
              RAW=(Q(I,0)-Q(J,0))**2-
     $          SUM((Q(I,1:3)-Q(J,1:3))**2)
              WRITE(74,'(A,4I4,3ES47.34E4)') 'KM ',K,H,I,J,
     $          REAL(OSTHRES,KIND=16),RAW,REAL(S(I,J),KIND=16)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      CLOSE(73)
      CLOSE(74)
      END
