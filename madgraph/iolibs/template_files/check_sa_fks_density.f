      SUBROUTINE CHECK_BORN_DENSITY(P)
C     ******************************************************************
C     Spin-density matrix of the Born building blocks at the point P
C     (already in the frame of the helicities), for the legs of
C     --density (density_info.inc), summed over the helicities of every
C     other leg. Prints and writes to density_born.dat:
C       L I J Re Im   the Born (L=0) and the colour-linked Borns (L =
C                     the link number of born_links.dat), I <= J
C       T I J Re Im   the spin-correlated transition density T
C       TP I J Re Im  and TP (docs/nlo_density_matrix_plan.md, 3.3)
C     and checks, for this point:
C       - the traces against SBORN, SB_SF_LINK and BORNTILDE;
C       - the projector W: with W set to each element of a hermitian
C         basis, SBORN, SB_SF_LINK and BORNTILDE return Tr(W rho).
C     ******************************************************************
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'density_info.inc'
      INTEGER NPACK
      PARAMETER (NPACK=N_COMB_B*(N_COMB_B+1)/2)
      REAL*8 P(0:3,NEXTERNAL-1)
      COMPLEX*16, ALLOCATABLE :: RHOD(:,:)
      COMPLEX*16 T(N_COMB_B,N_COMB_B), TP(N_COMB_B,N_COMB_B)
      COMPLEX*16 ANS_CNT(2,NSPLITORDERS)
      COMMON /C_BORN_CNT/ ANS_CNT
      INTEGER NLINKS, L, I, J, K, SOL, IB, KIND, C1, C2
      INTEGER PACKIDX(N_COMB_B,N_COMB_B)
      REAL*8 BORN, BW, X, SCALE, DEV, MAXDEV(0:2)
      COMPLEX*16 BTILDE, EXPECT, GOT
      CHARACTER*12 WHAT(0:2)
      DATA WHAT /'Born        ','colour links','spin-corr.  '/

      CALL GET_DENSITY_BORN_NLINKS(NLINKS)
      ALLOCATE(RHOD(NPACK,0:NLINKS))
      SOL=0
      DO I=1,N_COMB_B
        DO J=I,N_COMB_B
          SOL=SOL+1
          PACKIDX(I,J)=SOL
          PACKIDX(J,I)=SOL
        ENDDO
      ENDDO

C     the unprojected building blocks first (any density_w.dat is read
C     by DENSITY_W_INIT and then ignored here)
      CALL DENSITY_W_INIT()
      DENS_W_ON=.FALSE.
      CALL SBORN(P,BORN)
      BTILDE=(0D0,0D0)
      DO J=1,NSPLITORDERS
        BTILDE=BTILDE+ANS_CNT(2,J)
      ENDDO
      CALL GET_DENSITY_BORN(P,RHOD)
      CALL GET_DENSITY_BORNTILDE(P,T,TP)

      WRITE(*,*) '==== SPIN-DENSITY MATRIX OF LEGS',
     &  (POS_B(K),K=1,N_CHANGING_B), '===='
      DO I=1,N_COMB_B
        WRITE(*,*) 'combination', I, ':',
     &    (ALLOW_HEL_B((I-1)*N_CHANGING_B+K),K=1,N_CHANGING_B)
      ENDDO
      OPEN(UNIT=79,FILE='density_born.dat',STATUS='UNKNOWN')
      WRITE(79,*) '# legs', (POS_B(K),K=1,N_CHANGING_B)
      DO I=1,N_COMB_B
        WRITE(79,*) '# combination', I, ':',
     &    (ALLOW_HEL_B((I-1)*N_CHANGING_B+K),K=1,N_CHANGING_B)
      ENDDO
      DO L=0,NLINKS
        DO I=1,N_COMB_B
          DO J=I,N_COMB_B
            WRITE(79,'(3(1X,I4),2(1X,ES25.15E3))') L, I, J,
     &        DBLE(RHOD(PACKIDX(I,J),L)), DIMAG(RHOD(PACKIDX(I,J),L))
          ENDDO
        ENDDO
      ENDDO
      DO I=1,N_COMB_B
        DO J=1,N_COMB_B
          WRITE(79,'(A3,2(1X,I4),2(1X,ES25.15E3))') ' T ', I, J,
     &      DBLE(T(I,J)), DIMAG(T(I,J))
          WRITE(79,'(A3,2(1X,I4),2(1X,ES25.15E3))') ' TP', I, J,
     &      DBLE(TP(I,J)), DIMAG(TP(I,J))
        ENDDO
      ENDDO
      CLOSE(79)
      DO I=1,N_COMB_B
        DO J=I,N_COMB_B
          WRITE(*,*) 'RHO_B', I, J, RHOD(PACKIDX(I,J),0)
        ENDDO
      ENDDO

C     traces
      GOT=(0D0,0D0)
      DO I=1,N_COMB_B
        GOT=GOT+RHOD(PACKIDX(I,I),0)
      ENDDO
      WRITE(*,*) 'Tr RHO_B    =', DBLE(GOT), '  BORN      =', BORN
      DO L=1,NLINKS
        GOT=(0D0,0D0)
        DO I=1,N_COMB_B
          GOT=GOT+RHOD(PACKIDX(I,I),L)
        ENDDO
        CALL SB_SF_LINK(P,L,X)
        WRITE(*,*) 'Tr RHO_link', L, '=', DBLE(GOT), '  SB_SF_LINK =', X
      ENDDO
      GOT=(0D0,0D0)
      DO I=1,N_COMB_B
        GOT=GOT+T(I,I)
      ENDDO
      WRITE(*,*) 'Tr T        =', GOT, '  BORNTILDE =', BTILDE

C     the projector W, on each element of a hermitian basis:
C     KIND=0: E_II, KIND=1: E_IJ+E_JI, KIND=2: i(E_IJ-E_JI), I<J
      MAXDEV(:)=0D0
      DO I=1,N_COMB_B
        DO J=I,N_COMB_B
          DO KIND=0,2
            IF ((I.EQ.J).NEQV.(KIND.EQ.0)) CYCLE
            DENS_W(:,:)=(0D0,0D0)
            IF (KIND.EQ.0) THEN
              DENS_W(I,I)=(1D0,0D0)
            ELSEIF (KIND.EQ.1) THEN
              DENS_W(I,J)=(1D0,0D0)
              DENS_W(J,I)=(1D0,0D0)
            ELSE
              DENS_W(I,J)=(0D0,1D0)
              DENS_W(J,I)=(0D0,-1D0)
            ENDIF
            DENS_W_ON=.TRUE.
            CALL SBORN(P,BW)
            DO L=0,NLINKS
              IF (L.EQ.0) THEN
                X=BW
              ELSE
                CALL SB_SF_LINK(P,L,X)
              ENDIF
              EXPECT=(0D0,0D0)
              SCALE=0D0
              DO C1=1,N_COMB_B
                DO C2=1,N_COMB_B
                  IF (C1.LE.C2) THEN
                    GOT=RHOD(PACKIDX(C1,C2),L)
                  ELSE
                    GOT=DCONJG(RHOD(PACKIDX(C1,C2),L))
                  ENDIF
                  EXPECT=EXPECT+DENS_W(C2,C1)*GOT
                  SCALE=MAX(SCALE,ABS(GOT))
                ENDDO
              ENDDO
              IB=MIN(L,1)
              IF (SCALE.GT.0D0) THEN
                DEV=ABS(X-EXPECT)/SCALE
                MAXDEV(IB)=MAX(MAXDEV(IB),DEV)
              ENDIF
            ENDDO
            GOT=(0D0,0D0)
            DO K=1,NSPLITORDERS
              GOT=GOT+ANS_CNT(2,K)
            ENDDO
            EXPECT=(0D0,0D0)
            SCALE=0D0
            DO C1=1,N_COMB_B
              DO C2=1,N_COMB_B
                EXPECT=EXPECT+DENS_W(C2,C1)*T(C1,C2)
                SCALE=MAX(SCALE,ABS(T(C1,C2)))
              ENDDO
            ENDDO
            IF (SCALE.GT.0D0) THEN
              MAXDEV(2)=MAX(MAXDEV(2),ABS(GOT-EXPECT)/SCALE)
            ENDIF
          ENDDO
        ENDDO
      ENDDO
      DENS_W_ON=.FALSE.
      CALL SBORN(P,BORN)
      DO IB=0,2
        WRITE(*,*) 'W projector ', WHAT(IB),
     &    ': max|Tr(W rho) - projected| / max|rho| =', MAXDEV(IB)
      ENDDO
      DEALLOCATE(RHOD)
      END
