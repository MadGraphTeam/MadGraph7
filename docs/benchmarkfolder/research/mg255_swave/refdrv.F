      PROGRAM REFDRV
C     Driver for the MG5 2.5.5 onia V0 virtuals: reads
C     G MU_R MQ ONIUM_FACTOR SCHEME [CALCOPTION], then N and N points of
C     NEXTERNAL-1 momenta; prints the finite part and the poles. With
C     TWO_POL (a V0 split into transverse and longitudinal polarisations of
C     the onium, virt_wgts(3,2)) it prints their sum (V) and each (P).
      USE constants
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
#ifdef TWO_POL
      DOUBLE PRECISION VW(3,2)
#else
      DOUBLE PRECISION VW(3)
#endif
      DOUBLE PRECISION PB(0:3,NEXTERNAL-1), MQ
      INTEGER N, I, K, OPT
#ifdef WITH_MP
      CALL MPINIT(34)
#endif
      READ(*,*) G, MU_R, MQ, ONIUM_FACTOR, renormalisation_scheme, OPT
#ifdef WITH_CALCOPTION
      calcoption=OPT
#endif
#ifdef GRID_FILE
      grid_file=GRID_FILE
#endif
#ifdef HAS_MC
      MDL_MC=MQ
#endif
#ifdef HAS_MB
      MDL_MB=MQ
#endif
      READ(*,*) N
      DO K=1,N
        DO I=1,NEXTERNAL-1
          READ(*,*) PB(0:3,I)
        ENDDO
        CALL SLOOPMATRIX(PB,VW)
#ifdef TWO_POL
        WRITE(*,'(A,3ES26.16)') 'V ',VW(1:3,1)+VW(1:3,2)
        WRITE(*,'(A,6ES26.16)') 'P ',VW(1:3,1),VW(1:3,2)
#else
        WRITE(*,'(A,3ES26.16)') 'V ',VW
#endif
      ENDDO
      END
