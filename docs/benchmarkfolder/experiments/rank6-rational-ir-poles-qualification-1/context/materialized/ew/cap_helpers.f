      SUBROUTINE CAP6_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L
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

      SUBROUTINE CAP6_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L
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



      SUBROUTINE CAP6_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN,M2L)
      IMPLICIT NONE

      INTEGER CTMODE,NLOOPLINE

      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 PDEN(0:3,NLOOPLINE-1)
      COMPLEX*16 M2L(NLOOPLINE)
      REAL*8 NEW_PL(0:3,NLOOPLINE)
      REAL*8 NEW_PDEN(0:3,NLOOPLINE-1)
      COMPLEX*16 NEW_M2L(NLOOPLINE)

      INTEGER I,J,K

      IF (CTMODE.NE.2.AND.CTMODE.NE.5) THEN
        RETURN
      ENDIF

      IF (NLOOPLINE.LE.2) THEN
        RETURN
      ENDIF

      DO I=1,NLOOPLINE-1
        DO J=0,3
          NEW_PDEN(J,NLOOPLINE-I) = PDEN(J,I)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE-1
        DO J=0,3
          PDEN(J,I) = NEW_PDEN(J,I)
        ENDDO
      ENDDO

      DO I=2,NLOOPLINE
        NEW_M2L(I)=M2L(NLOOPLINE-I+2)
      ENDDO
      DO I=2,NLOOPLINE
        M2L(I)=NEW_M2L(I)
      ENDDO


      DO I=1,NLOOPLINE
        DO J=0,3
          NEW_PL(J,I) = -PL(J,NLOOPLINE+1-I)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE
        DO J=0,3
          PL(J,I) = NEW_PL(J,I)
        ENDDO
      ENDDO

      END

      SUBROUTINE CAP6_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN,M2L)
      IMPLICIT NONE

      INTEGER CTMODE,NLOOPLINE

      REAL*16 PL(0:3,NLOOPLINE)
      REAL*16 PDEN(0:3,NLOOPLINE-1)
      COMPLEX*32 M2L(NLOOPLINE)
      REAL*16 NEW_PL(0:3,NLOOPLINE)
      REAL*16 NEW_PDEN(0:3,NLOOPLINE-1)
      COMPLEX*32 NEW_M2L(NLOOPLINE)

      INTEGER I,J,K

      IF (CTMODE.NE.2.AND.CTMODE.NE.5) THEN
        RETURN
      ENDIF

      IF (NLOOPLINE.LE.2) THEN
        RETURN
      ENDIF

      DO I=1,NLOOPLINE-1
        DO J=0,3
          NEW_PDEN(J,NLOOPLINE-I) = PDEN(J,I)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE-1
        DO J=0,3
          PDEN(J,I) = NEW_PDEN(J,I)
        ENDDO
      ENDDO

      DO I=2,NLOOPLINE
        NEW_M2L(I)=M2L(NLOOPLINE-I+2)
      ENDDO
      DO I=2,NLOOPLINE
        M2L(I)=NEW_M2L(I)
      ENDDO


      DO I=1,NLOOPLINE
        DO J=0,3
          NEW_PL(J,I) = -PL(J,NLOOPLINE+1-I)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE
        DO J=0,3
          PL(J,I) = NEW_PL(J,I)
        ENDDO
      ENDDO

      END

      SUBROUTINE CAP6_INITCOLLIER()
C     
C     INITIALISATION OF COLLIER
C     
C     
C     MODULE
C     
      USE COLLIER
C     
C     CONSTANTS
C     
      CHARACTER(LEN=*) NO_FOLDER
      PARAMETER (NO_FOLDER='')
      CHARACTER(LEN=*) FOLDEROUTPUT
      PARAMETER (FOLDEROUTPUT='COLLIER_output')

C     Force COLLIER to completely reset from scratch
      LOGICAL NORESET
      PARAMETER (NORESET=.FALSE.)
C     
C     LOCAL VARIABLES 
C     
      INTEGER N_CACHES
C     
C     GLOBAL VARIABLES 
C     
      INCLUDE 'MadLoopParams.inc'
C     Now obtain the overall maximum rank, maximum external lines
C     and maximal number of processes.
      INCLUDE 'global_specs.inc'

C     ----------
C     BEGIN CODE
C     ----------

C     Initialize Collier
      IF (.NOT.COLLIERCANOUTPUT) THEN
        CALL INIT_CLL(MAXNEXTERNAL,OVERALLMAXRANK,NO_FOLDER,NORESET)
      ELSE
        CALL INIT_CLL(MAXNEXTERNAL,OVERALLMAXRANK,FOLDEROUTPUT,NORESET)
      ENDIF

C     Set target accuracy, be conservative w.r.t user request
C     Keep in mind that COLLIER has been optimized for 1d-8.
      IF (COLLIERREQUIREDACCURACY.EQ.-1.0D0) THEN
        CALL SETREQACC_CLL(MLSTABTHRES*1.0D-3)
      ELSE
        CALL SETREQACC_CLL(COLLIERREQUIREDACCURACY)
      ENDIF

C     Set COLLIER mode
      CALL SETMODE_CLL(COLLIERMODE)

C     Set the global cache strategy
C     If we use the caches for the poles as well, then it is a total of
C     4 caches per process.
      IF (COLLIERUSECACHEFORPOLES) THEN
        N_CACHES =4*NPROCS
      ELSE
        N_CACHES =NPROCS
      ENDIF
      IF (COLLIERGLOBALCACHE.EQ.-1) THEN
        CALL INITCACHESYSTEM_CLL(N_CACHES,MAXNEXTERNAL)
      ELSEIF(COLLIERGLOBALCACHE.GT.0) THEN
        CALL INITCACHESYSTEM_CLL(N_CACHES,COLLIERGLOBALCACHE)
      ENDIF

C     Make sure to start by first switching off all cache
      IF (COLLIERGLOBALCACHE.NE.0) THEN
        CALL SWITCHOFFCACHESYSTEM_CLL()
      ENDIF

C     Make sure no COLLIER error can interrupt the code
      CALL SWITCHOFFERRSTOP_CLL()

C     Specify below your other custom COLLIER parameter settings 
C     [user_specific_COLLIER_settings]

      END

