C     ===========================================
C     ===== Beginning of CutTools Interface =====
C     ===========================================
      SUBROUTINE ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C     

C     
C     Interface between MG5 and CutTools.
C     

C     
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      LOGICAL CHECKPCONSERVATION
      PARAMETER (CHECKPCONSERVATION=.TRUE.)
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0*
     $ *2))
C     
C     ARGUMENTS 
C     
      INTEGER NLOOPLINE, RANK
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)
      REAL*8 REF_P
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 M2LCT(0:NLOOPLINE-1)
      COMPLEX*16 RES(3)
      LOGICAL STABLE
C     
C     LOCAL VARIABLES 
C     
      COMPLEX*16 R1, ACC
      INTEGER I, J, K
      LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
     $ ,COLLIERINIT
      COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT
     $ ,NINJAINIT,COLLIERINIT
C     
C     EXTERNAL FUNCTIONS
C     
      EXTERNAL ML5_0_LOOPNUM
      EXTERNAL ML5_0_MPLOOPNUM
C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'coupl.inc'
      INTEGER CTMODE
      REAL*8 LSCALE
      COMMON/ML5_0_CT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

C     ----------
C     BEGIN CODE
C     ----------

C     INITIALIZE CUTTOOLS IF NEEDED
      IF (CTINIT) THEN
        CTINIT=.FALSE.
        CALL ML5_0_INITCT()
        CALL CITE('Ossola:2007ax','one-loop reduction with CutTools')
      ENDIF

C     YOU CAN FIND THE DETAILS ABOUT THE DIFFERENT CTMODE AT THE
C      BEGINNING OF THE FILE CTS_CUTS.F90 IN THE CUTTOOLS DISTRIBUTION

C     CONVERT THE MASSES TO BE COMPLEX
      DO I=1,NLOOPLINE
        M2LCT(I-1)=M2L(I)
      ENDDO

C     CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO CT CONVENTIONS
      DO I=0,3
        ABSPCT(I)=0.D0
        DO J=0,(NLOOPLINE-1)
          PCT(I,J)=0.D0
        ENDDO
      ENDDO
      DO I=0,3
        DO J=1,NLOOPLINE
          PCT(I,0)=PCT(I,0)+PL(I,J)
          ABSPCT(I)=ABSPCT(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P = MAX(ABSPCT(0), ABSPCT(1),ABSPCT(2),ABSPCT(3))
      DO I=0,3
        ABSPCT(I) = MAX(REF_P*1E-6, ABSPCT(I))
      ENDDO
      IF (CHECKPCONSERVATION.AND.REF_P.GT.1D-8) THEN
        IF ((PCT(0,0)/ABSPCT(0)).GT.1.D-6) THEN
          WRITE(*,*) 'energy is not conserved (flag:CT95)',PCT(0,0)
          STOP 'energy is not conserved (flag:CT95)'
        ELSEIF ((PCT(1,0)/ABSPCT(1)).GT.1.D-6) THEN
          WRITE(*,*) 'px is not conserved (flag:CT95)',PCT(1,0)
          STOP 'px is not conserved (flag:CT95)'
        ELSEIF ((PCT(2,0)/ABSPCT(2)).GT.1.D-6) THEN
          WRITE(*,*) 'py is not conserved (flag:CT95)',PCT(2,0)
          STOP 'py is not conserved (flag:CT95)'
        ELSEIF ((PCT(3,0)/ABSPCT(3)).GT.1.D-6) THEN
          WRITE(*,*) 'pz is not conserved (flag:CT95)',PCT(3,0)
          STOP 'pz is not conserved (flag:CT95)'
        ENDIF
      ENDIF
      DO I=0,3
        DO J=1,(NLOOPLINE-1)
          DO K=1,J
            PCT(I,J)=PCT(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO

      CALL CTSXCUT(CTMODE,LSCALE,MU_R,NLOOPLINE,ML5_0_LOOPNUM
     $ ,ML5_0_MPLOOPNUM,RANK,PCT,M2LCT,RES,ACC,R1,STABLE)
      RES(1)=NORMALIZATION*2.0D0*DBLE(RES(1))
      RES(2)=NORMALIZATION*2.0D0*DBLE(RES(2))
      RES(3)=NORMALIZATION*2.0D0*DBLE(RES(3))
C     WRITE(*,*) 'CutTools: Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

      SUBROUTINE ML5_0_INITCT()
C     
C     INITIALISATION OF CUTTOOLS
C     
C     LOCAL VARIABLES 
C     
      REAL*8 THRS
      LOGICAL EXT_NUM_FOR_R1
C     
C     GLOBAL VARIABLES 
C     
      INCLUDE 'MadLoopParams.inc'
C     ----------
C     BEGIN CODE
C     ----------

C     DEFAULT PARAMETERS FOR CUTTOOLS
C     -------------------------------  
C     THRS1 IS THE PRECISION LIMIT BELOW WHICH THE MP ROUTINES
C      ACTIVATES
      THRS=CTSTABTHRES
C     LOOPLIB SET WHAT LIBRARY CT USES
C     1 -> LOOPTOOLS
C     2 -> AVH
C     3 -> QCDLOOP
      LOOPLIB=CTLOOPLIBRARY
C     MADLOOP'S NUMERATOR IN THE OPEN LOOP IS MUCH FASTER THAN THE
C      RECONSTRUCTED ONE IN CT. SO WE BETTER USE MADLOOP ONE IN THIS
C      CASE.
      EXT_NUM_FOR_R1=.TRUE.
C     -------------------------------	  

C     The initialization below is for CT v1.8.+
      CALL CTSINIT(THRS,LOOPLIB,EXT_NUM_FOR_R1)
C     The initialization below is for the older stable CT v1.7, still
C      used for now in the beta release.
C     CALL CTSINIT(THRS,LOOPLIB)

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



C     ===========================================
C     ===== Beginning of Ninja interface  =====
C     ===========================================

      SUBROUTINE ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C     
C     Module used
C     
      USE MNINJA
C     

C     
C     Interface between MG5 and Ninja.
C     

C     
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      LOGICAL CHECKPCONSERVATION
      PARAMETER (CHECKPCONSERVATION=.TRUE.)
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0*
     $ *2))
      INTEGER NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
C     These are constants related to the split orders
      INTEGER NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
C     
C     ARGUMENTS 
C     
      INTEGER NLOOPLINE, RANK
      REAL*8 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      LOGICAL STABLE
C     
C     LOCAL VARIABLES 
C     
      REAL*8 P_TMP(0:3,0:NLOOPLINE-1),ABSP_TMP(0:3)
      REAL*8 REF_P
      REAL*8 P_NINJA(0:3,NLOOPLINE)
      REAL*8 P_S_MAT(NLOOPLINE,0:3)
      COMPLEX*16 M2L_NINJA(NLOOPLINE)
      COMPLEX*16 NINJA_RES(0:2)
      COMPLEX*16 R1
      INTEGER NINJA_STATUS
      INTEGER I, J, K
      REAL*8 PDEN_DUMMY(0:3,NLOOPLINE-1)

      COMPLEX*16 S_MAT(NLOOPLINE,NLOOPLINE)
      REAL*8 REAL_S_MAT(NLOOPLINE,NLOOPLINE)

      INTEGER CURR_MAXCOEF
      COMPLEX*16, ALLOCATABLE :: TENSORCOEFS(:)

C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'coupl.inc'

      LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
     $ ,COLLIERINIT
      COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT
     $ ,NINJAINIT,COLLIERINIT

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R
      COMPLEX*16 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LCOEFS/LOOPCOEFS

      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     $  FPE_IN_QP_REDUCTION

C     ----------
C     BEGIN CODE
C     ----------

C     For the direction test, we must switch the direction in which
C      the loop is read for CTMode equal to 2 or 4.
      CALL ML5_0_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)

C     The CT initialization is also performed here if not done already
C      because it calls MPINIT of OneLOop which is necessary on some
C      system
      IF (CTINIT) THEN
        CTINIT=.FALSE.
        CALL ML5_0_INITCT()
      ENDIF

C     INITIALIZE NINJA IF NEEDED
      IF (NINJAINIT) THEN
        NINJAINIT=.FALSE.
        CALL ML5_0_INITNINJA()
        CALL CITE('Peraro:2014cba','one-loop reduction with Ninja')
        CALL CITE('Hirschi:2016mdz','one-loop reduction with Ninja'
     $   //' (MadLoop interface)')
      ENDIF

C     CONVERT THE MASSES TO BE COMPLEX
      DO I=1,NLOOPLINE
        M2L_NINJA(I)=M2L(I)
      ENDDO

C     CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO NINJA
C      CONVENTIONS
      DO I=0,3
        ABSP_TMP = 0.D0
        DO J=0,(NLOOPLINE-1)
          P_TMP(I,J)=0.D0
        ENDDO
      ENDDO
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I) = ABSP_TMP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P = MAX(ABSP_TMP(0), ABSP_TMP(1),ABSP_TMP(2),ABSP_TMP(3))
      DO I=0,3
        ABSP_TMP(I) = MAX(REF_P*1E-6, ABSP_TMP(I))
      ENDDO

      IF (CHECKPCONSERVATION.AND.REF_P.GT.1D-8) THEN
        IF ((P_TMP(0,0)/ABSP_TMP(0)).GT.1.D-6) THEN
          WRITE(*,*) 'energy is not conserved (flag:CT692)',P_TMP(0,0)
          STOP 'energy is not conserved (flag:CT692)'
        ELSEIF ((P_TMP(1,0)/ABSP_TMP(1)).GT.1.D-6) THEN
          WRITE(*,*) 'px is not conserved (flag:CT692)',P_TMP(1,0)
          STOP 'px is not conserved (flag:CT692)'
        ELSEIF ((P_TMP(2,0)/ABSP_TMP(2)).GT.1.D-6) THEN
          WRITE(*,*) 'py is not conserved (flag:CT692)',P_TMP(2,0)
          STOP 'py is not conserved (flag:CT692)'
        ELSEIF ((P_TMP(3,0)/ABSP_TMP(3)).GT.1.D-6) THEN
          WRITE(*,*) 'pz is not conserved (flag:CT692)',P_TMP(3,0)
          STOP 'pz is not conserved (flag:CT692)'
        ENDIF
      ENDIF
      DO I=0,3
        DO J=1,(NLOOPLINE-1)
          DO K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO
C     In Ninja, the loop line index starts at 1
      DO I=0,NLOOPLINE-1
        P_NINJA(0,I+1) = P_TMP(0,I)
        P_NINJA(1,I+1) = P_TMP(1,I)
        P_NINJA(2,I+1) = P_TMP(2,I)
        P_NINJA(3,I+1) = P_TMP(3,I)
      ENDDO

C     Number of coefficients for the current rank
      CURR_MAXCOEF = 0
      DO I=0,RANK
        CURR_MAXCOEF=CURR_MAXCOEF+(3+I)*(2+I)*(1+I)/6
      ENDDO
C     Now write the tensor coefficients for Ninja
C     It should never be allocated at this stage
      IF (.NOT. ALLOCATED(TENSORCOEFS)) THEN
        ALLOCATE(TENSORCOEFS(0:CURR_MAXCOEF-1))
      ENDIF
      DO I=0,CURR_MAXCOEF-1
        TENSORCOEFS(I) = LOOPCOEFS(I,SQSOINDEX,ID)
      ENDDO
C     The loop momentum is in fact q_loop -> -q_loop, so that the
C     coefficients must be changed accordingly
      CALL ML5_0_INVERT_MOMENTA_IN_POLYNOMIAL(CURR_MAXCOEF,TENSORCOEFS)

C     Compute the kinematic matrix
      DO J=1,NLOOPLINE
        DO I=0,3
          P_S_MAT(J,I)=P_NINJA(I,J)
        ENDDO
      ENDDO
      CALL ML5_0_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,M2L,S_MAT)

      DO I=1,NLOOPLINE
        DO J=1,NLOOPLINE
          REAL_S_MAT(I,J) = DBLE(S_MAT(I,J)+M2L(I)+M2L(J))
        ENDDO
      ENDDO

C     Below is the call specifying the kinematic matrix
      CALL NINJA_TENSOR_EVALUATE(TENSORCOEFS,NLOOPLINE,RANK,REAL_S_MAT
     $ ,P_NINJA,M2L,MU_R**2,NINJA_RES,R1,NINJA_STATUS)

C     Below is the call without specification of the kinematic matrix
C     call ninja_tensor_evaluate(TENSORCOEFS,NLOOPLINE,RANK,P_NINJA,M2L
C     ,MU_R**2,NINJA_RES,R1,NINJA_STATUS)

C     If a floating point exception was found in Ninja (e.g. exactly
C      zero gram. det.)
C     Then warn loop_matrix.f so that it will flag this kinematic
C      point as unstable no matter what.
      IF (NINJA_STATUS.EQ.NINJA_UNSTABLE_KINEMATICS) THEN
        FPE_IN_DP_REDUCTION = .TRUE.
      ENDIF

C     Make sure to deallocate the tensor of coefficients
      IF (ALLOCATED(TENSORCOEFS)) THEN
        DEALLOCATE(TENSORCOEFS)
      ENDIF

      RES(1)=NORMALIZATION*2.0D0*DBLE(NINJA_RES(0))
      RES(2)=NORMALIZATION*2.0D0*DBLE(NINJA_RES(1))
      RES(3)=NORMALIZATION*2.0D0*DBLE(NINJA_RES(2))
C     WRITE(*,*) 'Ninja:  Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

C     
C     Quadruple precision version of loop_ninja
C     
      SUBROUTINE ML5_0_MP_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C     
C     Module used
C     
      USE MNINJA
C     

C     
C     Interface between MG5 and Ninja.
C     

C     
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      LOGICAL CHECKPCONSERVATION
      PARAMETER (CHECKPCONSERVATION=.TRUE.)
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0*
     $ *2))
      INTEGER NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
C     These are constants related to the split orders
      INTEGER NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
C     
C     ARGUMENTS 
C     
      INTEGER NLOOPLINE, RANK
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      LOGICAL STABLE
C     
C     LOCAL VARIABLES 
C     
      REAL*16 NINJA_SCALE
      REAL*16 P_TMP(0:3,0:NLOOPLINE-1), ABSP_TMP(0:3)
      REAL*8 REF_P
      REAL(KI_QNIN) MP_P_NINJA(0:3,NLOOPLINE)
      REAL*16 MP_P(0:3,NLOOPLINE)
      REAL*16 P_S_MAT(NLOOPLINE,0:3)
      COMPLEX*32 MP_M2L(NLOOPLINE)
      COMPLEX(KI_QNIN) MP_M2L_NINJA(NLOOPLINE)
      COMPLEX(KI_QNIN) NINJA_RES(0:2)
      COMPLEX(KI_QNIN) NINJA_R1
      COMPLEX*16 R1
      COMPLEX*16 DP_RES(0:2)
      INTEGER NINJA_STATUS
      INTEGER I, J, K
      REAL*16 PDEN_DUMMY(0:3,NLOOPLINE-1)

      COMPLEX*32 MP_S_MAT(NLOOPLINE,NLOOPLINE)
      REAL*16 MP_REAL_S_MAT(NLOOPLINE,NLOOPLINE)
      REAL(KI_QNIN) MP_REAL_S_MAT_NINJA(NLOOPLINE,NLOOPLINE)

      INTEGER CURR_MAXCOEF
      COMPLEX*32, ALLOCATABLE :: MP_TENSORCOEFS(:)
      COMPLEX(KI_QNIN), ALLOCATABLE :: MP_NINJA_TENSORCOEFS(:)

C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'coupl.inc'

      LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
     $ ,COLLIERINIT
      COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT
     $ ,NINJAINIT,COLLIERINIT

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 MP_LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_MP_LCOEFS/MP_LOOPCOEFS

      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     $  FPE_IN_QP_REDUCTION

C     ----------
C     BEGIN CODE
C     ----------

C     Cast the masses in complex quadruple precision
      DO I=1,NLOOPLINE
        MP_M2L(I) = CMPLX(M2L(I),KIND=16)
      ENDDO

C     For the direction test, we must switch the direction in which
C      the loop is read for CTMode equal to 2 or 4.
      CALL ML5_0_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MP_M2L)

C     The CT initialization is also performed here if not done already
C      because it calls MPINIT of OneLOop which is necessary on some
C      system
      IF (CTINIT) THEN
        CTINIT=.FALSE.
        CALL ML5_0_INITCT()
      ENDIF

C     INITIALIZE NINJA IF NEEDED
      IF (NINJAINIT) THEN
        NINJAINIT=.FALSE.
        CALL ML5_0_INITNINJA()
        CALL CITE('Peraro:2014cba','one-loop reduction with Ninja')
        CALL CITE('Hirschi:2016mdz','one-loop reduction with Ninja'
     $   //' (MadLoop interface)')
      ENDIF

C     CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO NINJA
C      CONVENTIONS
      DO I=0,3
        ABSP_TMP(I)=0.E0+0_16
        DO J=0,(NLOOPLINE-1)
          P_TMP(I,J)=0.E0+0_16
        ENDDO
      ENDDO
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I)=ABSP_TMP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P = MAX(ABSP_TMP(0), ABSP_TMP(1),ABSP_TMP(2),ABSP_TMP(3))
      DO I=0,3
        ABSP_TMP(I) = MAX(REF_P*1E-6, ABSP_TMP(I))
      ENDDO
      IF (CHECKPCONSERVATION.AND.REF_P.GT.1.E-8_16) THEN
        IF ((P_TMP(0,0)/ABSP_TMP(0)).GT.1.E-6_16) THEN
          WRITE(*,*) 'energy is not conserved (flag:CT968)'
     $     ,DBLE(P_TMP(0,0))
          STOP 'energy is not conserved (flag:CT968)'
        ELSEIF ((P_TMP(1,0)/ABSP_TMP(1)).GT.1.E-6_16) THEN
          WRITE(*,*) 'px is not conserved (flag:CT968)',DBLE(P_TMP(1,0)
     $     )
          STOP 'px is not conserved (flag:CT968)'
        ELSEIF ((P_TMP(2,0)/ABSP_TMP(2)).GT.1.E-6_16) THEN
          WRITE(*,*) 'py is not conserved (flag:CT968)',DBLE(P_TMP(2,0)
     $     )
          STOP 'py is not conserved (flag:CT968)'
        ELSEIF ((P_TMP(3,0)/ABSP_TMP(3)).GT.1.E-6_16) THEN
          WRITE(*,*) 'pz is not conserved (flag:CT968)',DBLE(P_TMP(3,0)
     $     )
          STOP 'pz is not conserved (flag:CT968)'
        ENDIF
      ENDIF
      DO I=0,3
        DO J=1,(NLOOPLINE-1)
          DO K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO
C     In Ninja, the loop line index starts at 1
      DO I=0,NLOOPLINE-1
        MP_P(0,I+1) = P_TMP(0,I)
        MP_P(1,I+1) = P_TMP(1,I)
        MP_P(2,I+1) = P_TMP(2,I)
        MP_P(3,I+1) = P_TMP(3,I)
      ENDDO

C     Number of coefficients for the current rank
      CURR_MAXCOEF = 0
      DO I=0,RANK
        CURR_MAXCOEF=CURR_MAXCOEF+(3+I)*(2+I)*(1+I)/6
      ENDDO
C     Now write the tensor coefficients for Ninja
C     It should never be allocated at this stage
      IF (.NOT. ALLOCATED(MP_TENSORCOEFS)) THEN
        ALLOCATE(MP_TENSORCOEFS(0:CURR_MAXCOEF-1))
      ENDIF
      IF (.NOT. ALLOCATED(MP_NINJA_TENSORCOEFS)) THEN
        ALLOCATE(MP_NINJA_TENSORCOEFS(0:CURR_MAXCOEF-1))
      ENDIF
      DO I=0,CURR_MAXCOEF-1
        MP_TENSORCOEFS(I) = MP_LOOPCOEFS(I,SQSOINDEX,ID)
      ENDDO
C     The loop momentum is in fact q_loop -> -q_loop, so that the
C     coefficients must be changed accordingly
      CALL MP_ML5_0_INVERT_MOMENTA_IN_POLYNOMIAL(CURR_MAXCOEF
     $ ,MP_TENSORCOEFS)

C     Compute the kinematic matrix
      DO J=1,NLOOPLINE
        DO I=0,3
          P_S_MAT(J,I)=MP_P(I,J)
        ENDDO
      ENDDO
      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MP_M2L
     $ ,MP_S_MAT)

      DO I=1,NLOOPLINE
        DO J=1,NLOOPLINE
          MP_REAL_S_MAT(I,J) = REAL(MP_S_MAT(I,J)+MP_M2L(I)+MP_M2L(J)
     $     ,KIND=16)
        ENDDO
      ENDDO

C     Now typecast to Ninja's quadruple precision format
      DO I=0,CURR_MAXCOEF-1
        MP_NINJA_TENSORCOEFS(I)=CMPLX(MP_TENSORCOEFS(I),KIND=KI_QNIN)
      ENDDO
      DO I=1,NLOOPLINE
        DO J=1,NLOOPLINE
          MP_REAL_S_MAT_NINJA(I,J) = REAL(MP_REAL_S_MAT(I,J)
     $     ,KIND=KI_QNIN)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE
        MP_M2L_NINJA(I)=CMPLX(MP_M2L(I),KIND=KI_QNIN)
      ENDDO
      DO I=1,NLOOPLINE
        MP_P_NINJA(0,I) = REAL(MP_P(0,I),KIND=KI_QNIN)
        MP_P_NINJA(1,I) = REAL(MP_P(1,I),KIND=KI_QNIN)
        MP_P_NINJA(2,I) = REAL(MP_P(2,I),KIND=KI_QNIN)
        MP_P_NINJA(3,I) = REAL(MP_P(3,I),KIND=KI_QNIN)
      ENDDO
      NINJA_SCALE = REAL(MU_R**2,KIND=KI_QNIN)


C     Below is the call specifying the kinematic matrix
      CALL NINJA_TENSOR_EVALUATE(MP_NINJA_TENSORCOEFS,NLOOPLINE,RANK
     $ ,MP_REAL_S_MAT_NINJA,MP_P_NINJA,MP_M2L_NINJA,NINJA_SCALE
     $ ,NINJA_RES,NINJA_R1,NINJA_STATUS)
C     Below is the call without specification of the kinematic matrix
C     call ninja_tensor_evaluate(MP_NINJA_TENSORCOEFS,NLOOPLINE,RANK,MP
C     _P_NINJA,MP_M2L_NINJA,NINJA_SCALE,NINJA_RES,NINJA_R1,NINJA_STATUS)
C     

C     If a floating point exception was found in Ninja (e.g. exactly
C      zero gram. det.)
C     Then warn loop_matrix.f so that it will flag this kinematic
C      point as unstable no matter what.
      IF (NINJA_STATUS.EQ.NINJA_UNSTABLE_KINEMATICS) THEN
        FPE_IN_QP_REDUCTION = .TRUE.
      ENDIF

C     Typecast the result back
      R1 = DCMPLX(R1)
      DO I=0,2
        DP_RES(I)=DCMPLX(NINJA_RES(I))
      ENDDO

C     Make sure to deallocate the tensor of coefficients
      IF (ALLOCATED(MP_TENSORCOEFS)) THEN
        DEALLOCATE(MP_TENSORCOEFS)
      ENDIF
      IF (ALLOCATED(MP_NINJA_TENSORCOEFS)) THEN
        DEALLOCATE(MP_NINJA_TENSORCOEFS)
      ENDIF

      RES(1)=NORMALIZATION*2.0D0*DBLE(DP_RES(0))
      RES(2)=NORMALIZATION*2.0D0*DBLE(DP_RES(1))
      RES(3)=NORMALIZATION*2.0D0*DBLE(DP_RES(2))
C     WRITE(*,*) 'QP Ninja:  Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

      SUBROUTINE ML5_0_INITNINJA()
C     
C     Module used
C     
      USE MNINJA
C     
C     Initialization of Ninja 
C     
C     LOCAL VARIABLES 
C     
      INTEGER LOOPLIB
C     
C     GLOBAL VARIABLES 
C     
      INCLUDE 'MadLoopParams.inc'
C     ----------
C     BEGIN CODE
C     ----------

C     LOOPLIB SET WHAT LIBRARY NINJA USES
C     1 -> LOOPTOOLS
C     2 -> AVH
C     3 -> QCDLOOP
      IF (CTLOOPLIBRARY.EQ.1) THEN
        WRITE(*,*) 'Warning in Ninja initialization. LoopTools is not'
     $   //' supported by the Ninja interface. It will use OneLOop'
     $   //' instead.'
        LOOPLIB = 1
      ELSEIF (CTLOOPLIBRARY.EQ.3) THEN
        WRITE(*,*) 'Warning in Ninja initialization. LoopTools is not'
     $   //' supported by the Ninja interface. It will use OneLOop'
     $   //' instead.'
        LOOPLIB = 1
      ELSEIF (CTLOOPLIBRARY.EQ.2) THEN
        LOOPLIB = 1
      ELSE
        WRITE(*,*) 'Error in Ninja initialization. Loop library ID='
     $   ,CTLOOPLIBRARY,' is not supported. Change variable'
     $   //' CTLoopLibrary in MadLoopParams.dat.'
        STOP 1
      ENDIF
      CALL NINJA_SET_INTEGRAL_LIBRARY(LOOPLIB)

      END

C     NinjaRS tensor reduction using oneloopmaster scalar integrals.
      SUBROUTINE ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED
      INTEGER NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL
      INTEGER NRS_RETRY_HARD_SKIP
      REAL*8 NRS_NATIVE_ACC,NRS_COMP_ACC
      COMMON/ML5_0_NRS_RETRY_FLAGS/NRS_RETRY_MODE,NRS_RETRY_USED
      COMMON/ML5_0_NRS_RETRY_STATS/NRS_RETRY_COUNT,NRS_RETRY_PASS,
     $ NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP
      COMMON/ML5_0_NRS_RETRY_ACC/NRS_NATIVE_ACC,NRS_COMP_ACC
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=200,NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK
      REAL*8 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      LOGICAL STABLE
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION=1.D0/(16.D0*3.14159265358979323846D0**2)
     $ )
      REAL*8 P_TMP(0:3,0:NLOOPLINE-1),ABSP_TMP(0:3),REF_P
      REAL*8 P_NRS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      REAL*8 PDEN_DUMMY(0:3,NLOOPLINE-1),REAL_S_MAT(NLOOPLINE
     $ ,NLOOPLINE)
      REAL*8 NRS_SCALE
      COMPLEX*16 MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      COMPLEX*16 TENSORCOEFS(0:LOOPMAXCOEFS-1),NRS_RES(0:2),R1
      INTEGER I,J,K,CURR_MAXCOEF,NRS_STATUS

      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R
      COMPLEX*16 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_NRS)
C     Preserve MadLoop's loop-direction stability test and mass
C      ordering.
      CALL ML5_0_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)
      MASSES=M2L
C     Match the existing Ninja loop momentum convention, including its
C     accumulation order and the residual momentum on the first line.
      P_TMP=0
      ABSP_TMP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I)=ABSP_TMP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP_TMP)
      IF (REF_P.GT.1.D-8) THEN
        DO I=0,3
          IF (ABS(P_TMP(I,0))/MAX(REF_P*1.D-6,ABSP_TMP(I)).GT.1.D-6)
     $      THEN
            FPE_IN_DP_REDUCTION=.TRUE.
            RETURN
          ENDIF
        ENDDO
      ENDIF
      DO I=0,3
        DO J=1,NLOOPLINE-1
          DO K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          P_NRS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=P_NRS(I,J)
        ENDDO
      ENDDO
C     Use MadLoop's thresholded invariants, including complex on-shell
C      masses.
      CALL ML5_0_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          REAL_S_MAT(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J)
     $     ,KIND=KI_NRS)
        ENDDO
      ENDDO
      CURR_MAXCOEF=NCOEF_R(RANK)
      DO I=0,CURR_MAXCOEF-1
        TENSORCOEFS(I)=LOOPCOEFS(I,SQSOINDEX,ID)
C       q_loop -> -q_loop in NinjaRS's denominator convention.
        IF (MOD(COEFTORANK_MAP(I),2).EQ.1) TENSORCOEFS(I)=
     $   -TENSORCOEFS(I)
      ENDDO
      NRS_SCALE=REAL(MU_R**2,KIND=KI_NRS)
      IF(NRS_RETRY_MODE) THEN
        CALL NINJARS_TENSOR_EVALUATE_COMPENSATED(TENSORCOEFS(0:CURR_MAX
     $COEF-1),NLOOPLINE,RANK,REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,NRS_RES
     $   ,R1,NRS_STATUS,MAX(0,LOOPPOLEMODE))
      ELSE
        CALL NINJARS_TENSOR_EVALUATE(TENSORCOEFS(0:CURR_MAXCOEF-1)
     $   ,NLOOPLINE,RANK,REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,NRS_RES,R1
     $   ,NRS_STATUS,MAX(0,LOOPPOLEMODE))
      ENDIF
      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS
      DO I=0,2
        STABLE=STABLE.AND.IEEE_IS_FINITE(REAL(NRS_RES(I),KIND=KI_NRS))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(NRS_RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_DP_REDUCTION=.TRUE.
        RETURN
      ENDIF
      DO I=0,2
        RES(I+1)=NORMALIZATION*2.D0*DBLE(NRS_RES(I))
      ENDDO
      DO I=1,3
        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I)))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_DP_REDUCTION=.TRUE.
        RES=(0.D0,0.D0)
      ENDIF
      END
C     Initialize retry state even when a caller invokes NINJARS_LOOP
C      directly.
      BLOCK DATA ML5_0_NINJARS_RETRY_INIT
      IMPLICIT NONE
      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED
      INTEGER NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL
      INTEGER NRS_RETRY_HARD_SKIP
      REAL*8 NRS_NATIVE_ACC,NRS_COMP_ACC
      COMMON/ML5_0_NRS_RETRY_FLAGS/NRS_RETRY_MODE,NRS_RETRY_USED
      COMMON/ML5_0_NRS_RETRY_STATS/NRS_RETRY_COUNT,NRS_RETRY_PASS,
     $ NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP
      COMMON/ML5_0_NRS_RETRY_ACC/NRS_NATIVE_ACC,NRS_COMP_ACC
      DATA NRS_RETRY_MODE,NRS_RETRY_USED/.FALSE.,.FALSE./
      DATA NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL,
     $ NRS_RETRY_HARD_SKIP/0,0,0,0/
      DATA NRS_NATIVE_ACC,NRS_COMP_ACC/-1D0,-1D0/
      END
      SUBROUTINE ML5_0_MP_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,RES
     $ ,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=200,NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      LOGICAL STABLE
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION=1.D0/(16.D0*3.14159265358979323846D0**2)
     $ )
      REAL*16 P_TMP(0:3,0:NLOOPLINE-1),ABSP_TMP(0:3),REF_P
      REAL*16 P_NRS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      REAL*16 PDEN_DUMMY(0:3,NLOOPLINE-1),REAL_S_MAT(NLOOPLINE
     $ ,NLOOPLINE)
      REAL*16 NRS_SCALE
      COMPLEX*32 MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      COMPLEX*32 TENSORCOEFS(0:LOOPMAXCOEFS-1),NRS_RES(0:2),R1
      INTEGER I,J,K,CURR_MAXCOEF,NRS_STATUS

      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_MP_LCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QNRS)
C     Preserve MadLoop's loop-direction stability test and mass
C      ordering.
      CALL ML5_0_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)

C     Match the existing Ninja loop momentum convention, including its
C     accumulation order and the residual momentum on the first line.
      P_TMP=0
      ABSP_TMP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I)=ABSP_TMP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP_TMP)
      IF (REF_P.GT.1.D-8) THEN
        DO I=0,3
          IF (ABS(P_TMP(I,0))/MAX(REF_P*1.D-6,ABSP_TMP(I)).GT.1.D-6)
     $      THEN
            FPE_IN_QP_REDUCTION=.TRUE.
            RETURN
          ENDIF
        ENDDO
      ENDIF
      DO I=0,3
        DO J=1,NLOOPLINE-1
          DO K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          P_NRS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=P_NRS(I,J)
        ENDDO
      ENDDO
C     Use MadLoop's thresholded invariants, including complex on-shell
C      masses.
      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES
     $ ,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          REAL_S_MAT(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J)
     $     ,KIND=KI_QNRS)
        ENDDO
      ENDDO
      CURR_MAXCOEF=NCOEF_R(RANK)
      DO I=0,CURR_MAXCOEF-1
        TENSORCOEFS(I)=LOOPCOEFS(I,SQSOINDEX,ID)
C       q_loop -> -q_loop in NinjaRS's denominator convention.
        IF (MOD(COEFTORANK_MAP(I),2).EQ.1) TENSORCOEFS(I)=
     $   -TENSORCOEFS(I)
      ENDDO
      NRS_SCALE=REAL(MU_R**2,KIND=KI_QNRS)
      CALL NINJARS_TENSOR_EVALUATE(TENSORCOEFS(0:CURR_MAXCOEF-1)
     $ ,NLOOPLINE,RANK,REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,NRS_RES,R1
     $ ,NRS_STATUS,MAX(0,LOOPPOLEMODE))
      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS
      DO I=0,2
        STABLE=STABLE.AND.IEEE_IS_FINITE(REAL(NRS_RES(I),KIND=KI_QNRS))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(NRS_RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RETURN
      ENDIF
      DO I=0,2
        RES(I+1)=NORMALIZATION*2.D0*DBLE(NRS_RES(I))
      ENDDO
      DO I=1,3
        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I)))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RES=(0.D0,0.D0)
      ENDIF
      END
      SUBROUTINE ML5_0_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,PL,M2L,RANK
     $ ,RES,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=200,NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*32 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      LOGICAL STABLE
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION=1.D0/(16.D0*3.14159265358979323846D0**2)
     $ )
      REAL*16 P_TMP(0:3,0:NLOOPLINE-1),ABSP_TMP(0:3),REF_P
      REAL*16 P_NRS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      REAL*16 PDEN_DUMMY(0:3,NLOOPLINE-1),REAL_S_MAT(NLOOPLINE
     $ ,NLOOPLINE)
      REAL*16 NRS_SCALE
      COMPLEX*32 MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      COMPLEX*32 TENSORCOEFS(0:LOOPMAXCOEFS-1),NRS_RES(0:2),R1
      INTEGER I,J,K,CURR_MAXCOEF,NRS_STATUS

      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_MP_LCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QNRS)
C     Preserve MadLoop's loop-direction stability test and mass
C      ordering.
      CALL ML5_0_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)

C     Match the existing Ninja loop momentum convention, including its
C     accumulation order and the residual momentum on the first line.
      P_TMP=0
      ABSP_TMP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I)=ABSP_TMP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP_TMP)
      IF (REF_P.GT.1.D-8) THEN
        DO I=0,3
          IF (ABS(P_TMP(I,0))/MAX(REF_P*1.D-6,ABSP_TMP(I)).GT.1.D-6)
     $      THEN
            FPE_IN_QP_REDUCTION=.TRUE.
            RETURN
          ENDIF
        ENDDO
      ENDIF
      DO I=0,3
        DO J=1,NLOOPLINE-1
          DO K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          ENDDO
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          P_NRS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=P_NRS(I,J)
        ENDDO
      ENDDO
C     Use MadLoop's thresholded invariants, including complex on-shell
C      masses.
      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES
     $ ,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          REAL_S_MAT(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J)
     $     ,KIND=KI_QNRS)
        ENDDO
      ENDDO
      CURR_MAXCOEF=NCOEF_R(RANK)
      DO I=0,CURR_MAXCOEF-1
        TENSORCOEFS(I)=LOOPCOEFS(I,SQSOINDEX,ID)
C       q_loop -> -q_loop in NinjaRS's denominator convention.
        IF (MOD(COEFTORANK_MAP(I),2).EQ.1) TENSORCOEFS(I)=
     $   -TENSORCOEFS(I)
      ENDDO
      NRS_SCALE=REAL(MU_R**2,KIND=KI_QNRS)
      CALL NINJARS_TENSOR_EVALUATE(TENSORCOEFS(0:CURR_MAXCOEF-1)
     $ ,NLOOPLINE,RANK,REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,NRS_RES,R1
     $ ,NRS_STATUS,MAX(0,LOOPPOLEMODE))
      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS
      DO I=0,2
        STABLE=STABLE.AND.IEEE_IS_FINITE(REAL(NRS_RES(I),KIND=KI_QNRS))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(NRS_RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RETURN
      ENDIF
      DO I=0,2
        RES(I+1)=NORMALIZATION*2.D0*DBLE(NRS_RES(I))
      ENDDO
      DO I=1,3
        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I)))
        STABLE=STABLE.AND.IEEE_IS_FINITE(AIMAG(RES(I)))
      ENDDO
      IF (.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RES=(0.D0,0.D0)
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_1_2(P1, W1, W2, M1,  RANK, SQUAREDSOINDEX,
     $  LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=1)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2
      COMPLEX*16 M1
      INTEGER P1
      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(2)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        M2L(1)=M1**2
        PAIRING(1)=P1
        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_1_2_QPMASS(P1, W1, W2, M1,MP_M1,  RANK,
     $  SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=1)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2
      COMPLEX*16 M1
      COMPLEX*32 MP_M1
      INTEGER P1
      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*32 MP_M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(2)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL ML5_0_LOOP_1_2(P1, W1, W2, M1, RANK,SQUAREDSOINDEX
     $   ,LOOPNUM)
        RETURN
      ENDIF
C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        M2L(1)=M1**2
        PAIRING(1)=P1
        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M1**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L
     $       ,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L,RANK
     $     ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM)
     $     )
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_2(W1, W2, M1, M2,  RANK, SQUAREDSOINDEX,
     $  LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=2)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2
      COMPLEX*16 M1, M2

      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(2)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        M2L(1)=M2**2
        M2L(2)=M1**2
        DO I=1,NLOOPLINE
          PAIRING(I)=1
        ENDDO

        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_2_QPMASS(W1, W2, M1,MP_M1, M2,MP_M2,  RANK
     $ , SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=2)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2
      COMPLEX*16 M1, M2
      COMPLEX*32 MP_M1, MP_M2

      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*32 MP_M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(2)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL ML5_0_LOOP_2(W1, W2, M1, M2, RANK,SQUAREDSOINDEX,LOOPNUM)
        RETURN
      ENDIF
C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        M2L(1)=M2**2
        M2L(2)=M1**2
        DO I=1,NLOOPLINE
          PAIRING(I)=1
        ENDDO

        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M2**2
          MP_M2L(2)=MP_M1**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L
     $       ,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L,RANK
     $     ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM)
     $     )
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_2_3(P1, P2, W1, W2, W3, M1, M2,  RANK,
     $  SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=2)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3
      COMPLEX*16 M1, M2
      INTEGER P1, P2
      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(3)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        M2L(1)=M2**2
        M2L(2)=M1**2
        PAIRING(1)=P1
        PAIRING(2)=P2
        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_2_3_QPMASS(P1, P2, W1, W2, W3, M1,MP_M1,
     $  M2,MP_M2,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=2)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3
      COMPLEX*16 M1, M2
      COMPLEX*32 MP_M1, MP_M2
      INTEGER P1, P2
      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*32 MP_M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(3)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL ML5_0_LOOP_2_3(P1, P2, W1, W2, W3, M1, M2, RANK
     $   ,SQUAREDSOINDEX,LOOPNUM)
        RETURN
      ENDIF
C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        M2L(1)=M2**2
        M2L(2)=M1**2
        PAIRING(1)=P1
        PAIRING(2)=P2
        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M2**2
          MP_M2L(2)=MP_M1**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L
     $       ,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L,RANK
     $     ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM)
     $     )
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_3(W1, W2, W3, M1, M2, M3,  RANK,
     $  SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=3)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3
      COMPLEX*16 M1, M2, M3

      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(3)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        M2L(1)=M3**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        DO I=1,NLOOPLINE
          PAIRING(I)=1
        ENDDO

        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_3_QPMASS(W1, W2, W3, M1,MP_M1, M2,MP_M2,
     $  M3,MP_M3,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=3)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3
      COMPLEX*16 M1, M2, M3
      COMPLEX*32 MP_M1, MP_M2, MP_M3

      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*32 MP_M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(3)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL ML5_0_LOOP_3(W1, W2, W3, M1, M2, M3, RANK,SQUAREDSOINDEX
     $   ,LOOPNUM)
        RETURN
      ENDIF
C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        M2L(1)=M3**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        DO I=1,NLOOPLINE
          PAIRING(I)=1
        ENDDO

        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M3**2
          MP_M2L(2)=MP_M1**2
          MP_M2L(3)=MP_M2**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L
     $       ,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L,RANK
     $     ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM)
     $     )
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_3_4(P1, P2, P3, W1, W2, W3, W4, M1, M2, M3
     $ ,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=3)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3, W4
      COMPLEX*16 M1, M2, M3
      INTEGER P1, P2, P3
      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(4)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        WE(4)=W4
        M2L(1)=M3**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        PAIRING(1)=P1
        PAIRING(2)=P2
        PAIRING(3)=P3
        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_3_4_QPMASS(P1, P2, P3, W1, W2, W3, W4, M1
     $ ,MP_M1, M2,MP_M2, M3,MP_M3,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=3)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3, W4
      COMPLEX*16 M1, M2, M3
      COMPLEX*32 MP_M1, MP_M2, MP_M3
      INTEGER P1, P2, P3
      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*32 MP_M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(4)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL ML5_0_LOOP_3_4(P1, P2, P3, W1, W2, W3, W4, M1, M2, M3,
     $    RANK,SQUAREDSOINDEX,LOOPNUM)
        RETURN
      ENDIF
C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        WE(4)=W4
        M2L(1)=M3**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        PAIRING(1)=P1
        PAIRING(2)=P2
        PAIRING(3)=P3
        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M3**2
          MP_M2L(2)=MP_M1**2
          MP_M2L(3)=MP_M2**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L
     $       ,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L,RANK
     $     ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM)
     $     )
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_4(W1, W2, W3, W4, M1, M2, M3, M4,  RANK,
     $  SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=4)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3, W4
      COMPLEX*16 M1, M2, M3, M4

      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(4)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        WE(4)=W4
        M2L(1)=M4**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        M2L(4)=M3**2
        DO I=1,NLOOPLINE
          PAIRING(I)=1
        ENDDO

        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_4_QPMASS(W1, W2, W3, W4, M1,MP_M1, M2
     $ ,MP_M2, M3,MP_M3, M4,MP_M4,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=4)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3, W4
      COMPLEX*16 M1, M2, M3, M4
      COMPLEX*32 MP_M1, MP_M2, MP_M3, MP_M4

      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*32 MP_M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(4)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL ML5_0_LOOP_4(W1, W2, W3, W4, M1, M2, M3, M4, RANK
     $   ,SQUAREDSOINDEX,LOOPNUM)
        RETURN
      ENDIF
C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        WE(4)=W4
        M2L(1)=M4**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        M2L(4)=M3**2
        DO I=1,NLOOPLINE
          PAIRING(I)=1
        ENDDO

        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M4**2
          MP_M2L(2)=MP_M1**2
          MP_M2L(3)=MP_M2**2
          MP_M2L(4)=MP_M3**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L
     $       ,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L,RANK
     $     ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM)
     $     )
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_5(W1, W2, W3, W4, W5, M1, M2, M3, M4, M5, 
     $  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=5)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3, W4, W5
      COMPLEX*16 M1, M2, M3, M4, M5

      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(5)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        WE(4)=W4
        WE(5)=W5
        M2L(1)=M5**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        M2L(4)=M3**2
        M2L(5)=M4**2
        DO I=1,NLOOPLINE
          PAIRING(I)=1
        ENDDO

        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_LOOP_5_QPMASS(W1, W2, W3, W4, W5, M1,MP_M1, M2
     $ ,MP_M2, M3,MP_M3, M4,MP_M4, M5,MP_M5,  RANK, SQUAREDSOINDEX,
     $  LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=5)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=200)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3, W4, W5
      COMPLEX*16 M1, M2, M3, M4, M5
      COMPLEX*32 MP_M1, MP_M2, MP_M3, MP_M4, MP_M5

      INTEGER RANK, LSYMFACT
      INTEGER LOOPNUM, SQUAREDSOINDEX
C     
C     LOCAL VARIABLES 
C     
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*16 MP_PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*32 MP_M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(5)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL COMPLEX_MASS,DOING_QP
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/ML5_0_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL ML5_0_LOOP_5(W1, W2, W3, W4, W5, M1, M2, M3, M4, M5, RANK
     $   ,SQUAREDSOINDEX,LOOPNUM)
        RETURN
      ENDIF
C     Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

      IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)
     $ .OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
        WE(1)=W1
        WE(2)=W2
        WE(3)=W3
        WE(4)=W4
        WE(5)=W5
        M2L(1)=M5**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        M2L(4)=M3**2
        M2L(5)=M4**2
        DO I=1,NLOOPLINE
          PAIRING(I)=1
        ENDDO

        R=RANK
        ID=LOOPNUM
        SQSOINDEX=SQUAREDSOINDEX
        DO I=0,3
          TEMP=1
          DO J=1,NLOOPLINE
            PL(I,J)=0.D0
            IF (DOING_QP) THEN
              MP_PL(I,J)=0.0E+0_16
            ENDIF
            DO K=TEMP,(TEMP+PAIRING(J)-1)
              PL(I,J)=PL(I,J)-W(WE(K))%P(I)
              IF (DOING_QP) THEN
                MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)
              ENDIF
            ENDDO
            TEMP=TEMP+PAIRING(J)
          ENDDO
        ENDDO
C       Determine whether the integral is with complex masses or not
C       since some reduction libraries, e.g.PJFry++ and IREGI, are
C        still
C       not able to deal with complex masses
        COMPLEX_MASS=.FALSE.
        DO I=1,NLOOPLINE
          IF(DIMAG(M2L(I)).EQ.0D0)CYCLE
          IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1D-15)THEN
            COMPLEX_MASS=.TRUE.
            EXIT
          ENDIF
        ENDDO
C       Choose the correct loop library
        CALL ML5_0_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS
     $   ,ID,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M5**2
          MP_M2L(2)=MP_M1**2
          MP_M2L(3)=MP_M2**2
          MP_M2L(4)=MP_M3**2
          MP_M2L(5)=MP_M4**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK
     $       ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL ML5_0_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL ML5_0_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L
     $       ,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX
     $       ,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL ML5_0_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,MP_PL,MP_M2L,RANK
     $     ,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM)
     $     )
        ELSE
C         Tensor Integral Reduction is used 
          CALL ML5_0_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
     $     ,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ENDIF
      ELSE
        LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0D0,0.0D0)
        S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
      ENDIF
      END

C     TIRust tensor reduction. The public diagnostic returns
C      MadLoop-ordered
C     moments, marks an unevaluated finite coefficient with NaN, and
C      performs
C     no numerator contraction or counterterm addition.
      SUBROUTINE ML5_0_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK
     $ ,PL,M2L,TIRCOEFS,STABLE)
      USE MTIRUST
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
     $ ,IEEE_VALUE,IEEE_QUIET_NAN
      USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS
      REAL*8 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS
      REAL*8 P_TMP(0:3,0:NLOOPLINE-1),ABSP(0:3),REF_P,PREFIX_SUM
      REAL*8 SHIFTS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      REAL*8 PDEN_DUMMY(0:3,NLOOPLINE-1),INVARIANTS(NLOOPLINE
     $ ,NLOOPLINE),SCALE
      COMPLEX*16 MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      COMPLEX*16 RAW(LOOPMAXCOEFS,4)
      INTEGER TIRUST_MAP(0:LOOPMAXCOEFS-1)
      DATA (TIRUST_MAP(I),I=0,9)/1,2,3,4,5,6,7,10,8,11/
      DATA (TIRUST_MAP(I),I=10,19)/13,9,12,14,15,16,17,20,26,18/
      DATA (TIRUST_MAP(I),I=20,29)/21,27,23,29,32,19,22,28,24,30/
      DATA (TIRUST_MAP(I),I=30,34)/33,25,31,34,35/

      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      STABLE=.FALSE.
      TIRCOEFS=0
      IF(RANK.LT.0.OR.RANK.GT.3) RETURN
      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))
     $  RETURN
      CALL ML5_0_INITIALIZE_TIRUST(STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      CALL TIRUST_SET_CACHE_BUDGET(INT(TIRUSTCACHEMAXBYTES,C_INT64_T)
     $ ,STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_TIR)
      CALL ML5_0_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)
      MASSES=M2L
C     Preserve the summation and threshold conventions of the Ninja
C      adapter.
      P_TMP=0
      ABSP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP(I)=ABSP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP)
      IF(REF_P.GT.1D-8) THEN
        DO I=0,3
          IF(ABS(P_TMP(I,0))/MAX(REF_P*1D-6,ABSP(I)).GT.1D-6) RETURN
        ENDDO
      ENDIF
C     Accumulate ordered prefixes once, preserving the zero-seeded sum.
      DO I=0,3
        PREFIX_SUM=REAL(0,KIND=KI_TIR)
        DO J=1,NLOOPLINE-1
          PREFIX_SUM=PREFIX_SUM+PL(I,J)
          P_TMP(I,J)=PREFIX_SUM
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          SHIFTS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=SHIFTS(I,J)
        ENDDO
      ENDDO
      CALL ML5_0_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          INVARIANTS(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J)
     $     ,KIND=KI_TIR)
        ENDDO
      ENDDO
      SCALE=MU_R**2
      NC=NCOEF_R(RANK)
      CACHE_ENABLED=TIRUSTGLOBALCACHE.EQ.
     $ -1.OR.NLOOPLINE.LE.TIRUSTGLOBALCACHE
      CHECK_PASS=CTMODE.EQ.2.OR.CTMODE.EQ.5
      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS
     $ ,MASSES,SCALE,RAW(1:NC,1:4),STATUS,MAX(0,LOOPPOLEMODE)
     $ ,TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      DO I=0,NC-1
        K=TIRUST_MAP(I)
        DO J=1,4
          IF(J.EQ.4.AND..NOT.TIRUSTCOMPUTEFINITE) CYCLE
          IF(.NOT.IEEE_IS_FINITE(REAL(RAW(K,J),KIND=KI_TIR))
     $     .OR..NOT.IEEE_IS_FINITE(AIMAG(RAW(K,J)))) RETURN
        ENDDO
        IF(TIRUSTCOMPUTEFINITE) THEN
          TIRCOEFS(I,1)=RAW(K,4)
        ELSE
          TIRCOEFS(I,1)=CMPLX(IEEE_VALUE(REAL(0,KIND=KI_TIR)
     $     ,IEEE_QUIET_NAN),REAL(0,KIND=KI_TIR),KIND=KI_TIR)
        ENDIF
        TIRCOEFS(I,2)=RAW(K,1)+RAW(K,2)
        TIRCOEFS(I,3)=RAW(K,3)
      ENDDO
      STABLE=.TRUE.
      END

C     TIRust tensor reduction. The public diagnostic returns
C      MadLoop-ordered
C     moments, marks an unevaluated finite coefficient with NaN, and
C      performs
C     no numerator contraction or counterterm addition.
      SUBROUTINE ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE
     $ ,RANK,PL,M2L,TIRCOEFS,STABLE)
      USE MTIRUST
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
     $ ,IEEE_VALUE,IEEE_QUIET_NAN
      USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*32 TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS
      REAL*16 P_TMP(0:3,0:NLOOPLINE-1),ABSP(0:3),REF_P,PREFIX_SUM
      REAL*16 SHIFTS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      REAL*16 PDEN_DUMMY(0:3,NLOOPLINE-1),INVARIANTS(NLOOPLINE
     $ ,NLOOPLINE),SCALE
      COMPLEX*32 MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      COMPLEX*32 RAW(LOOPMAXCOEFS,4)
      INTEGER TIRUST_MAP(0:LOOPMAXCOEFS-1)
      DATA (TIRUST_MAP(I),I=0,9)/1,2,3,4,5,6,7,10,8,11/
      DATA (TIRUST_MAP(I),I=10,19)/13,9,12,14,15,16,17,20,26,18/
      DATA (TIRUST_MAP(I),I=20,29)/21,27,23,29,32,19,22,28,24,30/
      DATA (TIRUST_MAP(I),I=30,34)/33,25,31,34,35/

      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      STABLE=.FALSE.
      TIRCOEFS=0
      IF(RANK.LT.0.OR.RANK.GT.3) RETURN
      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))
     $  RETURN
      CALL ML5_0_INITIALIZE_TIRUST(STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      CALL TIRUST_SET_CACHE_BUDGET(INT(TIRUSTCACHEMAXBYTES,C_INT64_T)
     $ ,STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QTIR)
      CALL ML5_0_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)
C     Preserve the summation and threshold conventions of the Ninja
C      adapter.
      P_TMP=0
      ABSP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP(I)=ABSP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP)
      IF(REF_P.GT.1D-8) THEN
        DO I=0,3
          IF(ABS(P_TMP(I,0))/MAX(REF_P*1D-6,ABSP(I)).GT.1D-6) RETURN
        ENDDO
      ENDIF
C     Accumulate ordered prefixes once, preserving the zero-seeded sum.
      DO I=0,3
        PREFIX_SUM=REAL(0,KIND=KI_QTIR)
        DO J=1,NLOOPLINE-1
          PREFIX_SUM=PREFIX_SUM+PL(I,J)
          P_TMP(I,J)=PREFIX_SUM
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          SHIFTS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=SHIFTS(I,J)
        ENDDO
      ENDDO
      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES
     $ ,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          INVARIANTS(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J)
     $     ,KIND=KI_QTIR)
        ENDDO
      ENDDO
      SCALE=REAL(MU_R,KIND=KI_QTIR)**2
      NC=NCOEF_R(RANK)
      CACHE_ENABLED=TIRUSTGLOBALCACHE.EQ.
     $ -1.OR.NLOOPLINE.LE.TIRUSTGLOBALCACHE
      CHECK_PASS=CTMODE.EQ.2.OR.CTMODE.EQ.5
      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS
     $ ,MASSES,SCALE,RAW(1:NC,1:4),STATUS,MAX(0,LOOPPOLEMODE)
     $ ,TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      DO I=0,NC-1
        K=TIRUST_MAP(I)
        DO J=1,4
          IF(J.EQ.4.AND..NOT.TIRUSTCOMPUTEFINITE) CYCLE
          IF(.NOT.IEEE_IS_FINITE(REAL(RAW(K,J),KIND=KI_QTIR))
     $     .OR..NOT.IEEE_IS_FINITE(AIMAG(RAW(K,J)))) RETURN
        ENDDO
        IF(TIRUSTCOMPUTEFINITE) THEN
          TIRCOEFS(I,1)=RAW(K,4)
        ELSE
          TIRCOEFS(I,1)=CMPLX(IEEE_VALUE(REAL(0,KIND=KI_QTIR)
     $     ,IEEE_QUIET_NAN),REAL(0,KIND=KI_QTIR),KIND=KI_QTIR)
        ENDIF
        TIRCOEFS(I,2)=RAW(K,1)+RAW(K,2)
        TIRCOEFS(I,3)=RAW(K,3)
      ENDDO
      STABLE=.TRUE.
      END

C     TIRust tensor reduction. The public diagnostic returns
C      MadLoop-ordered
C     moments, marks an unevaluated finite coefficient with NaN, and
C      performs
C     no numerator contraction or counterterm addition.
      SUBROUTINE ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC_QPMASS(CTMODE
     $ ,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      USE MTIRUST
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
     $ ,IEEE_VALUE,IEEE_QUIET_NAN
      USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*32 M2L(NLOOPLINE)
      COMPLEX*32 TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS
      REAL*16 P_TMP(0:3,0:NLOOPLINE-1),ABSP(0:3),REF_P,PREFIX_SUM
      REAL*16 SHIFTS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)
      REAL*16 PDEN_DUMMY(0:3,NLOOPLINE-1),INVARIANTS(NLOOPLINE
     $ ,NLOOPLINE),SCALE
      COMPLEX*32 MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)
      COMPLEX*32 RAW(LOOPMAXCOEFS,4)
      INTEGER TIRUST_MAP(0:LOOPMAXCOEFS-1)
      DATA (TIRUST_MAP(I),I=0,9)/1,2,3,4,5,6,7,10,8,11/
      DATA (TIRUST_MAP(I),I=10,19)/13,9,12,14,15,16,17,20,26,18/
      DATA (TIRUST_MAP(I),I=20,29)/21,27,23,29,32,19,22,28,24,30/
      DATA (TIRUST_MAP(I),I=30,34)/33,25,31,34,35/

      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      STABLE=.FALSE.
      TIRCOEFS=0
      IF(RANK.LT.0.OR.RANK.GT.3) RETURN
      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))
     $  RETURN
      CALL ML5_0_INITIALIZE_TIRUST(STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      CALL TIRUST_SET_CACHE_BUDGET(INT(TIRUSTCACHEMAXBYTES,C_INT64_T)
     $ ,STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QTIR)
      CALL ML5_0_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)
C     Preserve the summation and threshold conventions of the Ninja
C      adapter.
      P_TMP=0
      ABSP=0
      DO I=0,3
        DO J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP(I)=ABSP(I)+ABS(PL(I,J))
        ENDDO
      ENDDO
      REF_P=MAXVAL(ABSP)
      IF(REF_P.GT.1D-8) THEN
        DO I=0,3
          IF(ABS(P_TMP(I,0))/MAX(REF_P*1D-6,ABSP(I)).GT.1D-6) RETURN
        ENDDO
      ENDIF
C     Accumulate ordered prefixes once, preserving the zero-seeded sum.
      DO I=0,3
        PREFIX_SUM=REAL(0,KIND=KI_QTIR)
        DO J=1,NLOOPLINE-1
          PREFIX_SUM=PREFIX_SUM+PL(I,J)
          P_TMP(I,J)=PREFIX_SUM
        ENDDO
      ENDDO
      DO J=1,NLOOPLINE
        DO I=0,3
          SHIFTS(I,J)=P_TMP(I,J-1)
          P_S_MAT(J,I)=SHIFTS(I,J)
        ENDDO
      ENDDO
      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES
     $ ,S_MAT)
      DO J=1,NLOOPLINE
        DO I=1,NLOOPLINE
          INVARIANTS(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J)
     $     ,KIND=KI_QTIR)
        ENDDO
      ENDDO
      SCALE=REAL(MU_R,KIND=KI_QTIR)**2
      NC=NCOEF_R(RANK)
      CACHE_ENABLED=TIRUSTGLOBALCACHE.EQ.
     $ -1.OR.NLOOPLINE.LE.TIRUSTGLOBALCACHE
      CHECK_PASS=CTMODE.EQ.2.OR.CTMODE.EQ.5
      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS
     $ ,MASSES,SCALE,RAW(1:NC,1:4),STATUS,MAX(0,LOOPPOLEMODE)
     $ ,TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      DO I=0,NC-1
        K=TIRUST_MAP(I)
        DO J=1,4
          IF(J.EQ.4.AND..NOT.TIRUSTCOMPUTEFINITE) CYCLE
          IF(.NOT.IEEE_IS_FINITE(REAL(RAW(K,J),KIND=KI_QTIR))
     $     .OR..NOT.IEEE_IS_FINITE(AIMAG(RAW(K,J)))) RETURN
        ENDDO
        IF(TIRUSTCOMPUTEFINITE) THEN
          TIRCOEFS(I,1)=RAW(K,4)
        ELSE
          TIRCOEFS(I,1)=CMPLX(IEEE_VALUE(REAL(0,KIND=KI_QTIR)
     $     ,IEEE_QUIET_NAN),REAL(0,KIND=KI_QTIR),KIND=KI_QTIR)
        ENDIF
        TIRCOEFS(I,2)=RAW(K,1)+RAW(K,2)
        TIRCOEFS(I,3)=RAW(K,3)
      ENDDO
      STABLE=.TRUE.
      END

      SUBROUTINE ML5_0_TIRUSTLOOP(CTMODE,NLOOPLINE,RANK,PL,M2L
     $ ,TIRCOEFS,STABLE)
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK
      REAL*8 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE
      CALL ML5_0_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,PL,M2L
     $ ,TIRCOEFS,STABLE)
      END

      SUBROUTINE ML5_0_MP_TIRUST_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MTIRUST, ONLY: KI_QTIR
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER(NLOOPGROUPS=200,NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK,CTMODE,I,J
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      COMPLEX*32 COEFS(0:LOOPMAXCOEFS-1,3),RESULT(3)
      REAL*8 LSCALE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      REAL*16 NORMALIZATION
      PARAMETER(NORMALIZATION=1.0E0_16/(16.0E0_16
     $ *3.141592653589793238462643383279502884E0_16**2))
      LOGICAL STABLE,FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_MP_LCOEFS/LOOPCOEFS
      RES=0
      CALL ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,PL
     $ ,M2L,COEFS,STABLE)
      IF(.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RETURN
      ENDIF
      RESULT=0
      DO I=1,3
        DO J=0,NCOEF_R(RANK)-1
          RESULT(I)=RESULT(I)+LOOPCOEFS(J,SQSOINDEX,ID)*COEFS(J,I)
        ENDDO
        RESULT(I)=2.0E0_16*NORMALIZATION*REAL(RESULT(I),KIND=KI_QTIR)
        RES(I)=CMPLX(RESULT(I),KIND=KIND(0D0))
        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I)))
     $   .AND.IEEE_IS_FINITE(AIMAG(RES(I)))
      ENDDO
      IF(.NOT.STABLE) THEN
        RES=0
        FPE_IN_QP_REDUCTION=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,PL,M2L,RANK,RES
     $ ,STABLE)
      USE MTIRUST, ONLY: KI_QTIR
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER(NLOOPGROUPS=200,NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK,CTMODE,I,J
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*32 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      COMPLEX*32 COEFS(0:LOOPMAXCOEFS-1,3),RESULT(3)
      REAL*8 LSCALE
      COMMON/ML5_0_CT/LSCALE,CTMODE
      REAL*16 NORMALIZATION
      PARAMETER(NORMALIZATION=1.0E0_16/(16.0E0_16
     $ *3.141592653589793238462643383279502884E0_16**2))
      LOGICAL STABLE,FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION
      INTEGER ID,SQSOINDEX,R
      COMMON/ML5_0_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_MP_LCOEFS/LOOPCOEFS
      RES=0
      CALL ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC_QPMASS(CTMODE,NLOOPLINE
     $ ,RANK,PL,M2L,COEFS,STABLE)
      IF(.NOT.STABLE) THEN
        FPE_IN_QP_REDUCTION=.TRUE.
        RETURN
      ENDIF
      RESULT=0
      DO I=1,3
        DO J=0,NCOEF_R(RANK)-1
          RESULT(I)=RESULT(I)+LOOPCOEFS(J,SQSOINDEX,ID)*COEFS(J,I)
        ENDDO
        RESULT(I)=2.0E0_16*NORMALIZATION*REAL(RESULT(I),KIND=KI_QTIR)
        RES(I)=CMPLX(RESULT(I),KIND=KIND(0D0))
        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I)))
     $   .AND.IEEE_IS_FINITE(AIMAG(RES(I)))
      ENDDO
      IF(.NOT.STABLE) THEN
        RES=0
        FPE_IN_QP_REDUCTION=.TRUE.
      ENDIF
      END

      SUBROUTINE ML5_0_INITIALIZE_TIRUST(STATUS)
      USE MTIRUST
      IMPLICIT NONE
      INTEGER STATUS
      INTEGER LAST_GENERATION
      COMMON/ML5_0_TIRUST_INIT/LAST_GENERATION
      STATUS=TIRUST_SUCCESS
      IF(LAST_GENERATION.EQ.TIRUST_CONTEXT_GENERATION()) RETURN

      LAST_GENERATION=TIRUST_CONTEXT_GENERATION()
      END

      BLOCK DATA ML5_0_TIRUST_INIT_DATA
      IMPLICIT NONE
      INTEGER LAST_GENERATION
      COMMON/ML5_0_TIRUST_INIT/LAST_GENERATION
      DATA LAST_GENERATION/-2/
      END

      SUBROUTINE ML5_0_CLEAR_TIRUST_CACHE()
      USE MTIRUST, ONLY: TIRUST_CLEAR_CACHE
      CALL TIRUST_CLEAR_CACHE()
      END

      SUBROUTINE ML5_0_FINALIZE_TIRUST()
      USE MTIRUST, ONLY: TIRUST_FINALIZE
      IMPLICIT NONE
      INTEGER LAST_GENERATION
      COMMON/ML5_0_TIRUST_INIT/LAST_GENERATION
      CALL TIRUST_FINALIZE()
      LAST_GENERATION=-2
      END

