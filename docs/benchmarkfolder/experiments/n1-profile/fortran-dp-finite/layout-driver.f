C     Native layout parity fixture, independent of amplitude timings.
      PROGRAM CHECK_ADAPTER_LAYOUT
      USE MTIRUST
      USE, INTRINSIC :: IEEE_EXCEPTIONS
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      INCLUDE 'loop_max_coefs.inc'
      REAL*8 Q(0:3,6),P(0:3,6),PL(0:3,6)
      REAL*16 QPL(0:3,6)
      COMPLEX*16 M(6),MASSES(6),RES(0:LOOPMAXCOEFS-1,3)
      COMPLEX*32 QRES(0:LOOPMAXCOEFS-1,3)
      INTEGER N,R,C,I,J,K,STATUS,CASES,SUCCEEDED
      INTEGER*8 DP_BITS(2),QP_BITS(4)
      LOGICAL STABLE,FLAGS(5)
      CALL SETPARA('param_card.dat')
      MU_R=1D0
      OSTHRES=1D-8
      LOOPPOLEMODE=0
      TIRUSTCOMPUTEFINITE=.TRUE.
      TIRUSTGLOBALCACHE=-1
      TIRUSTCACHEMAXBYTES=268435456
      Q=0D0
      Q(1,2)=1D0
      Q(2,3)=1D0
      Q(3,4)=1D0
      Q(0,5)=0.5D0
      Q(1,6)=1D0
      Q(2,6)=1D0
      M=DCMPLX(1D0,0D0)
      M(2)=2D0
      M(3)=3D0
      M(4)=4D0
      M(5)=2D0
      M(6)=3D0
      OPEN(71,FILE='layout-values.dat',STATUS='REPLACE')
      CASES=0
      SUCCEEDED=0
      DO N=1,6
        P=0D0
        DO J=1,N-1
          P(:,J)=Q(:,J+1)-Q(:,J)
        ENDDO
        P(:,N)=-Q(:,N)
        DO R=0,2
          CALL TIRUST_REGISTER_PLAN(N,R,STATUS)
          IF(STATUS.NE.TIRUST_SUCCESS) STOP 11
          DO C=1,2
            CALL TIRUST_CLEAR_CACHE()
            PL=P
            MASSES=M
            CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
            CALL ML5_0_TIRUST_TENSOR_DIAGNOSTIC(C,N,R,PL(:,1:N)
     $       ,MASSES(1:N),RES,STABLE)
            CALL IEEE_GET_FLAG(IEEE_INVALID,FLAGS(1))
            CALL IEEE_GET_FLAG(IEEE_DIVIDE_BY_ZERO,FLAGS(2))
            CALL IEEE_GET_FLAG(IEEE_OVERFLOW,FLAGS(3))
            CALL IEEE_GET_FLAG(IEEE_UNDERFLOW,FLAGS(4))
            CALL IEEE_GET_FLAG(IEEE_INEXACT,FLAGS(5))
            CASES=CASES+1
            IF(STABLE) SUCCEEDED=SUCCEEDED+1
            WRITE(71,*) 'DP',N,R,C,STABLE
            WRITE(71,'(5L1)') FLAGS
            DO J=1,3
              DO I=0,LOOPMAXCOEFS-1
                DP_BITS=TRANSFER(RES(I,J),DP_BITS)
                WRITE(71,'(2Z16.16)') DP_BITS
              ENDDO
            ENDDO
            DO J=1,N
              DO I=0,3
                WRITE(71,'(Z16.16)') TRANSFER(PL(I,J),0_8)
              ENDDO
              DP_BITS=TRANSFER(MASSES(J),DP_BITS)
              WRITE(71,'(2Z16.16)') DP_BITS
            ENDDO
            CALL TIRUST_CLEAR_CACHE()
            QPL=REAL(P,KIND=16)
            MASSES=M
            CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
            CALL ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC(C,N,R
     $       ,QPL(:,1:N),MASSES(1:N),QRES,STABLE)
            CALL IEEE_GET_FLAG(IEEE_INVALID,FLAGS(1))
            CALL IEEE_GET_FLAG(IEEE_DIVIDE_BY_ZERO,FLAGS(2))
            CALL IEEE_GET_FLAG(IEEE_OVERFLOW,FLAGS(3))
            CALL IEEE_GET_FLAG(IEEE_UNDERFLOW,FLAGS(4))
            CALL IEEE_GET_FLAG(IEEE_INEXACT,FLAGS(5))
            CASES=CASES+1
            IF(STABLE) SUCCEEDED=SUCCEEDED+1
            WRITE(71,*) 'QP',N,R,C,STABLE
            WRITE(71,'(5L1)') FLAGS
            DO J=1,3
              DO I=0,LOOPMAXCOEFS-1
                QP_BITS=TRANSFER(QRES(I,J),QP_BITS)
                WRITE(71,'(4Z16.16)') QP_BITS
              ENDDO
            ENDDO
            DO J=1,N
              DO I=0,3
                DP_BITS=TRANSFER(QPL(I,J),DP_BITS)
                WRITE(71,'(2Z16.16)') DP_BITS
              ENDDO
              DP_BITS=TRANSFER(MASSES(J),DP_BITS)
              WRITE(71,'(2Z16.16)') DP_BITS
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      CLOSE(71)
      WRITE(*,*) 'LAYOUT_CASES',CASES,'STABLE_CASES',SUCCEEDED
      CALL TIRUST_FINALIZE()
      END
