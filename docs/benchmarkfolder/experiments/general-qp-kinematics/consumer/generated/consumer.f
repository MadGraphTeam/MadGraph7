C     ===========================================
C     ===== Beginning of CutTools Interface =====
C     ===========================================
      SUBROUTINE KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
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
      EXTERNAL KC_LOOPNUM
      EXTERNAL KC_MPLOOPNUM
C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'coupl.inc'
      INTEGER CTMODE
      REAL*8 LSCALE
      COMMON/KC_CT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R

C     ----------
C     BEGIN CODE
C     ----------

C     INITIALIZE CUTTOOLS IF NEEDED
      IF (CTINIT) THEN
        CTINIT=.FALSE.
        CALL KC_INITCT()
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

      CALL CTSXCUT(CTMODE,LSCALE,MU_R,NLOOPLINE,KC_LOOPNUM
     $ ,KC_MPLOOPNUM,RANK,PCT,M2LCT,RES,ACC,R1,STABLE)
      RES(1)=NORMALIZATION*2.0D0*DBLE(RES(1))
      RES(2)=NORMALIZATION*2.0D0*DBLE(RES(2))
      RES(3)=NORMALIZATION*2.0D0*DBLE(RES(3))
C     WRITE(*,*) 'CutTools: Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

      SUBROUTINE KC_INITCT()
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

      SUBROUTINE KC_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L,S_MAT)
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

      SUBROUTINE KC_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L
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

      SUBROUTINE KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
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
      PARAMETER (NLOOPGROUPS=116)
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
      COMMON/KC_CT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*16 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LCOEFS/LOOPCOEFS

      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     $  FPE_IN_QP_REDUCTION

C     ----------
C     BEGIN CODE
C     ----------

C     For the direction test, we must switch the direction in which
C      the loop is read for CTMode equal to 2 or 4.
      CALL KC_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)

C     The CT initialization is also performed here if not done already
C      because it calls MPINIT of OneLOop which is necessary on some
C      system
      IF (CTINIT) THEN
        CTINIT=.FALSE.
        CALL KC_INITCT()
      ENDIF

C     INITIALIZE NINJA IF NEEDED
      IF (NINJAINIT) THEN
        NINJAINIT=.FALSE.
        CALL KC_INITNINJA()
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
      CALL KC_INVERT_MOMENTA_IN_POLYNOMIAL(CURR_MAXCOEF,TENSORCOEFS)

C     Compute the kinematic matrix
      DO J=1,NLOOPLINE
        DO I=0,3
          P_S_MAT(J,I)=P_NINJA(I,J)
        ENDDO
      ENDDO
      CALL KC_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,M2L,S_MAT)

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
      SUBROUTINE KC_MP_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
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
      PARAMETER (NLOOPGROUPS=116)
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
      COMMON/KC_CT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 MP_LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_MP_LCOEFS/MP_LOOPCOEFS

      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
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
      CALL KC_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MP_M2L)

C     The CT initialization is also performed here if not done already
C      because it calls MPINIT of OneLOop which is necessary on some
C      system
      IF (CTINIT) THEN
        CTINIT=.FALSE.
        CALL KC_INITCT()
      ENDIF

C     INITIALIZE NINJA IF NEEDED
      IF (NINJAINIT) THEN
        NINJAINIT=.FALSE.
        CALL KC_INITNINJA()
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
      CALL MP_KC_INVERT_MOMENTA_IN_POLYNOMIAL(CURR_MAXCOEF
     $ ,MP_TENSORCOEFS)

C     Compute the kinematic matrix
      DO J=1,NLOOPLINE
        DO I=0,3
          P_S_MAT(J,I)=MP_P(I,J)
        ENDDO
      ENDDO
      CALL KC_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MP_M2L
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

      SUBROUTINE KC_INITNINJA()
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
      SUBROUTINE KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED
      INTEGER NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL
      INTEGER NRS_RETRY_HARD_SKIP
      REAL*8 NRS_NATIVE_ACC,NRS_COMP_ACC
      COMMON/KC_NRS_RETRY_FLAGS/NRS_RETRY_MODE,NRS_RETRY_USED
      COMMON/KC_NRS_RETRY_STATS/NRS_RETRY_COUNT,NRS_RETRY_PASS,
     $ NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP
      COMMON/KC_NRS_RETRY_ACC/NRS_NATIVE_ACC,NRS_COMP_ACC
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=116,NSQUAREDSO=1)
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
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*16 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_NRS)
C     Preserve MadLoop's loop-direction stability test and mass
C      ordering.
      CALL KC_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)
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
      CALL KC_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
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
      BLOCK DATA KC_NINJARS_RETRY_INIT
      IMPLICIT NONE
      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED
      INTEGER NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL
      INTEGER NRS_RETRY_HARD_SKIP
      REAL*8 NRS_NATIVE_ACC,NRS_COMP_ACC
      COMMON/KC_NRS_RETRY_FLAGS/NRS_RETRY_MODE,NRS_RETRY_USED
      COMMON/KC_NRS_RETRY_STATS/NRS_RETRY_COUNT,NRS_RETRY_PASS,
     $ NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP
      COMMON/KC_NRS_RETRY_ACC/NRS_NATIVE_ACC,NRS_COMP_ACC
      DATA NRS_RETRY_MODE,NRS_RETRY_USED/.FALSE.,.FALSE./
      DATA NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL,
     $ NRS_RETRY_HARD_SKIP/0,0,0,0/
      DATA NRS_NATIVE_ACC,NRS_COMP_ACC/-1D0,-1D0/
      END
      SUBROUTINE KC_MP_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=116,NSQUAREDSO=1)
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
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_MP_LCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QNRS)
C     Preserve MadLoop's loop-direction stability test and mass
C      ordering.
      CALL KC_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)

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
      CALL KC_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
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
      SUBROUTINE KC_MP_NINJARS_LOOP_QPMASS(NLOOPLINE,PL,M2L,RANK,RES
     $ ,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=116,NSQUAREDSO=1)
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
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_MP_LCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QNRS)
C     Preserve MadLoop's loop-direction stability test and mass
C      ordering.
      CALL KC_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)

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
      CALL KC_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
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
      SUBROUTINE KC_MP_NINJARS_LOOP_QPMASS_KIN(GROUP_ID,CTMODE
     $ ,KIN_POLICY,NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MNINJARS
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER (NLOOPGROUPS=116,NSQUAREDSO=1)
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
      INTEGER GROUP_ID,CTMODE,KIN_POLICY
      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_MP_LCOEFS/LOOPCOEFS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION

      STABLE=.FALSE.
      RES=(0.D0,0.D0)
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QNRS)
C     Preserve MadLoop's loop-direction stability test and mass
C      ordering.
      CALL KC_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)

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
      CALL KC_MP_BUILD_KINEMATIC_MATRIX_KIN(GROUP_ID,CTMODE,KIN_POLICY
     $ ,NLOOPLINE,P_S_MAT,MASSES,S_MAT)
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

      SUBROUTINE KC_LOOP_2(W1, W2, M1, M2,  RANK, SQUAREDSOINDEX,
     $  LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=2)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_2_QPMASS(W1, W2, M1,MP_M1, M2,MP_M2,  RANK,
     $  SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=2)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL KC_LOOP_2(W1, W2, M1, M2, RANK,SQUAREDSOINDEX,LOOPNUM)
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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M2**2
          MP_M2L(2)=MP_M1**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1
     $       ,NLOOPLINE,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX
     $       ,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,NLOOPLINE
     $     ,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_3(W1, W2, W3, M1, M2, M3,  RANK,
     $  SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=3)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_3_QPMASS(W1, W2, W3, M1,MP_M1, M2,MP_M2, M3
     $ ,MP_M3,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=3)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL KC_LOOP_3(W1, W2, W3, M1, M2, M3, RANK,SQUAREDSOINDEX
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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M3**2
          MP_M2L(2)=MP_M1**2
          MP_M2L(3)=MP_M2**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1
     $       ,NLOOPLINE,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX
     $       ,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,NLOOPLINE
     $     ,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_4(W1, W2, W3, W4, M1, M2, M3, M4,  RANK,
     $  SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=4)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_4_QPMASS(W1, W2, W3, W4, M1,MP_M1, M2,MP_M2,
     $  M3,MP_M3, M4,MP_M4,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=4)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL KC_LOOP_4(W1, W2, W3, W4, M1, M2, M3, M4, RANK
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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M4**2
          MP_M2L(2)=MP_M1**2
          MP_M2L(3)=MP_M2**2
          MP_M2L(4)=MP_M3**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1
     $       ,NLOOPLINE,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX
     $       ,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,NLOOPLINE
     $     ,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_3_4(P1, P2, P3, W1, W2, W3, W4, M1, M2, M3, 
     $  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=3)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_3_4_QPMASS(P1, P2, P3, W1, W2, W3, W4, M1
     $ ,MP_M1, M2,MP_M2, M3,MP_M3,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=3)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL KC_LOOP_3_4(P1, P2, P3, W1, W2, W3, W4, M1, M2, M3, RANK
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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M3**2
          MP_M2L(2)=MP_M1**2
          MP_M2L(3)=MP_M2**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1
     $       ,NLOOPLINE,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX
     $       ,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,NLOOPLINE
     $     ,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_2_3(P1, P2, W1, W2, W3, M1, M2,  RANK,
     $  SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=2)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_2_3_QPMASS(P1, P2, W1, W2, W3, M1,MP_M1, M2
     $ ,MP_M2,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=2)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL KC_LOOP_2_3(P1, P2, W1, W2, W3, M1, M2, RANK
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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M2**2
          MP_M2L(2)=MP_M1**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1
     $       ,NLOOPLINE,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX
     $       ,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,NLOOPLINE
     $     ,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_5(W1, W2, W3, W4, W5, M1, M2, M3, M4, M5, 
     $  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=5)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_5_QPMASS(W1, W2, W3, W4, W5, M1,MP_M1, M2
     $ ,MP_M2, M3,MP_M3, M4,MP_M4, M5,MP_M5,  RANK, SQUAREDSOINDEX,
     $  LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=5)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL KC_LOOP_5(W1, W2, W3, W4, W5, M1, M2, M3, M4, M5, RANK
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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
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
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1
     $       ,NLOOPLINE,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX
     $       ,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,NLOOPLINE
     $     ,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_4_5(P1, P2, P3, P4, W1, W2, W3, W4, W5, M1,
     $  M2, M3, M4,  RANK, SQUAREDSOINDEX, LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=4)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3, W4, W5
      COMPLEX*16 M1, M2, M3, M4
      INTEGER P1, P2, P3, P4
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

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
        M2L(1)=M4**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        M2L(4)=M3**2
        PAIRING(1)=P1
        PAIRING(2)=P2
        PAIRING(3)=P3
        PAIRING(4)=P4
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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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

      SUBROUTINE KC_LOOP_4_5_QPMASS(P1, P2, P3, P4, W1, W2, W3, W4, W5
     $ , M1,MP_M1, M2,MP_M2, M3,MP_M3, M4,MP_M4,  RANK, SQUAREDSOINDEX
     $ , LOOPNUM)
      USE ALOHA_OBJECT
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=4)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=2)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=116)
      INTEGER    NCOMB
      PARAMETER (NCOMB=2)
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=1)
C     
C     ARGUMENTS 
C     
      INTEGER W1, W2, W3, W4, W5
      COMPLEX*16 M1, M2, M3, M4
      COMPLEX*32 MP_M1, MP_M2, MP_M3, MP_M4
      INTEGER P1, P2, P3, P4
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
      COMMON/KC_LOOP/ID,SQSOINDEX,R

      LOGICAL CHECKPHASE, HELDOUBLECHECKED
      COMMON/KC_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_LOOPRES/LOOPRES,S


      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/KC_W/W
      TYPE(MP_ALOHA) MP_W(NWAVEFUNCS)
      COMMON/KC_MP_W/MP_W

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/KC_CT/LSCALE,CTMODE
      INTEGER LIBINDEX
      COMMON/KC_I_LIB/LIBINDEX

C     ----------
C     BEGIN CODE
C     ----------

      IF(CTMODE.LT.4) THEN
        CALL KC_LOOP_4_5(P1, P2, P3, P4, W1, W2, W3, W4, W5, M1, M2,
     $    M3, M4, RANK,SQUAREDSOINDEX,LOOPNUM)
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
        M2L(1)=M4**2
        M2L(2)=M1**2
        M2L(3)=M2**2
        M2L(4)=M3**2
        PAIRING(1)=P1
        PAIRING(2)=P2
        PAIRING(3)=P3
        PAIRING(4)=P4
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
        CALL KC_CHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,COMPLEX_MASS,ID
     $   ,DOING_QP,I_LIB)
        IF(DOING_QP.AND.(MLREDUCTIONLIB(I_LIB)
     $   .EQ.8.OR.MLREDUCTIONLIB(I_LIB).EQ.9)) THEN
          MP_M2L(1)=MP_M4**2
          MP_M2L(2)=MP_M1**2
          MP_M2L(3)=MP_M2**2
          MP_M2L(4)=MP_M3**2
        ENDIF
        IF(MLREDUCTIONLIB(I_LIB).EQ.1)THEN
C         CutTools is used
          CALL KC_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.6) THEN
C         Ninja is used
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF (MLREDUCTIONLIB(I_LIB).EQ.8) THEN
          IF (.NOT.DOING_QP) THEN
            CALL KC_NINJARS_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1
     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ELSE
            CALL KC_MP_NINJARS_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1
     $       ,NLOOPLINE,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX
     $       ,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
          ENDIF
        ELSEIF(MLREDUCTIONLIB(I_LIB).EQ.9.AND.DOING_QP) THEN
          CALL KC_MP_TIRUST_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,NLOOPLINE
     $     ,MP_PL,MP_M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)
     $     ,S(SQUAREDSOINDEX,LOOPNUM))
        ELSE
C         Tensor Integral Reduction is used 
          CALL KC_TIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL
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
      SUBROUTINE KC_KINEMATIC_IDENTITY(G,CT,N,I,J,
     $ CLASS,LEG,SGN,CLOSURE,DECLARED_ONSHELL,KNOWN)
      IMPLICIT NONE
      INTEGER G,CT,N,I,J,CLASS,LEG,SGN,CLOSURE
      LOGICAL DECLARED_ONSHELL,KNOWN
      INTEGER A,B,K,DIR,NP,ORI,L
      INTEGER GOFF(116),GNUM(116),GVALID(116)
      INTEGER CLASSES(954),LEGS(954),SIGNS(954)
      INTEGER CLOSURES(954),SHELLS(954)
C     Identity only: declared on-shell is not a numerical certificate.
      DATA (GOFF(L),L=1,8) /
     $ 0,2,4,10,12,14,20,26/
      DATA (GOFF(L),L=9,16) /
     $ 32,44,56,62,74,80,82,88/
      DATA (GOFF(L),L=17,24) /
     $ 90,92,94,96,98,104,106,108/
      DATA (GOFF(L),L=25,32) /
     $ 114,120,126,138,150,162,164,166/
      DATA (GOFF(L),L=33,40) /
     $ 172,174,176,182,188,194,206,218/
      DATA (GOFF(L),L=41,48) /
     $ 230,236,248,254,266,272,284,290/
      DATA (GOFF(L),L=49,56) /
     $ 296,308,314,326,338,344,356,362/
      DATA (GOFF(L),L=57,64) /
     $ 374,386,406,426,446,466,486,506/
      DATA (GOFF(L),L=65,72) /
     $ 518,530,542,562,582,602,622,642/
      DATA (GOFF(L),L=73,80) /
     $ 662,674,676,682,684,686,688,694/
      DATA (GOFF(L),L=81,88) /
     $ 700,712,724,730,742,748,754,760/
      DATA (GOFF(L),L=89,96) /
     $ 766,772,778,790,802,808,820,822/
      DATA (GOFF(L),L=97,104) /
     $ 828,834,840,842,848,854,860,866/
      DATA (GOFF(L),L=105,112) /
     $ 872,884,896,902,914,916,922,928/
      DATA (GOFF(L),L=113,116) /
     $ 934,936,942,948/
      DATA (GNUM(L),L=1,8) /
     $ 2,2,3,2,2,3,3,3/
      DATA (GNUM(L),L=9,16) /
     $ 4,4,3,4,3,2,3,2/
      DATA (GNUM(L),L=17,24) /
     $ 2,2,2,2,3,2,2,3/
      DATA (GNUM(L),L=25,32) /
     $ 3,3,4,4,4,2,2,3/
      DATA (GNUM(L),L=33,40) /
     $ 2,2,3,3,3,4,4,4/
      DATA (GNUM(L),L=41,48) /
     $ 3,4,3,4,3,4,3,3/
      DATA (GNUM(L),L=49,56) /
     $ 4,3,4,4,3,4,3,4/
      DATA (GNUM(L),L=57,64) /
     $ 4,5,5,5,5,5,5,4/
      DATA (GNUM(L),L=65,72) /
     $ 4,4,5,5,5,5,5,5/
      DATA (GNUM(L),L=73,80) /
     $ 4,2,3,2,2,2,3,3/
      DATA (GNUM(L),L=81,88) /
     $ 4,4,3,4,3,3,3,3/
      DATA (GNUM(L),L=89,96) /
     $ 3,3,4,4,3,4,2,3/
      DATA (GNUM(L),L=97,104) /
     $ 3,3,2,3,3,3,3,3/
      DATA (GNUM(L),L=105,112) /
     $ 4,4,3,4,2,3,3,3/
      DATA (GNUM(L),L=113,116) /
     $ 2,3,3,3/
      DATA (GVALID(L),L=1,8) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=9,16) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=17,24) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=25,32) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=33,40) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=41,48) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=49,56) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=57,64) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=65,72) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=73,80) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=81,88) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=89,96) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=97,104) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=105,112) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=113,116) /
     $ 1,1,1,1/
      DATA (CLASSES(L),L=1,8) /
     $ 3,3,3,3,2,3,3,3/
      DATA (CLASSES(L),L=9,16) /
     $ 1,3,3,3,3,3,2,3/
      DATA (CLASSES(L),L=17,24) /
     $ 3,3,1,3,2,3,1,3/
      DATA (CLASSES(L),L=25,32) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=33,40) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=41,48) /
     $ 1,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=49,56) /
     $ 3,1,3,3,1,1,3,1/
      DATA (CLASSES(L),L=57,64) /
     $ 2,3,3,3,1,3,2,3/
      DATA (CLASSES(L),L=65,72) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=73,80) /
     $ 3,1,2,3,3,3,1,3/
      DATA (CLASSES(L),L=81,88) /
     $ 3,3,2,3,1,3,1,1/
      DATA (CLASSES(L),L=89,96) /
     $ 2,2,2,2,3,3,3,3/
      DATA (CLASSES(L),L=97,104) /
     $ 3,3,2,3,3,3,1,3/
      DATA (CLASSES(L),L=105,112) /
     $ 3,3,3,3,2,3,3,3/
      DATA (CLASSES(L),L=113,120) /
     $ 1,3,2,3,1,3,1,1/
      DATA (CLASSES(L),L=121,128) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=129,136) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=137,144) /
     $ 3,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=145,152) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=153,160) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=161,168) /
     $ 3,1,3,3,3,3,2,3/
      DATA (CLASSES(L),L=169,176) /
     $ 3,3,1,3,3,3,3,3/
      DATA (CLASSES(L),L=177,184) /
     $ 2,3,3,3,1,3,2,3/
      DATA (CLASSES(L),L=185,192) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=193,200) /
     $ 1,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=201,208) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=209,216) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=217,224) /
     $ 3,1,2,3,1,1,3,3/
      DATA (CLASSES(L),L=225,232) /
     $ 2,3,1,3,3,1,2,3/
      DATA (CLASSES(L),L=233,240) /
     $ 1,3,1,1,2,3,3,1/
      DATA (CLASSES(L),L=241,248) /
     $ 3,1,3,3,1,1,3,1/
      DATA (CLASSES(L),L=249,256) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=257,264) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=265,272) /
     $ 3,1,2,3,3,3,1,3/
      DATA (CLASSES(L),L=273,280) /
     $ 2,3,1,1,3,3,2,3/
      DATA (CLASSES(L),L=281,288) /
     $ 1,3,3,1,2,3,3,3/
      DATA (CLASSES(L),L=289,296) /
     $ 1,3,2,3,1,3,1,1/
      DATA (CLASSES(L),L=297,304) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=305,312) /
     $ 1,1,3,1,2,3,1,3/
      DATA (CLASSES(L),L=313,320) /
     $ 1,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=321,328) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=329,336) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=337,344) /
     $ 3,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=345,352) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=353,360) /
     $ 1,1,3,1,2,3,1,3/
      DATA (CLASSES(L),L=361,368) /
     $ 1,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=369,376) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=377,384) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=385,392) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=393,400) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=401,408) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=409,416) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=417,424) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=425,432) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=433,440) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=441,448) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=449,456) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=457,464) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=465,472) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=473,480) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=481,488) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=489,496) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=497,504) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=505,512) /
     $ 3,1,2,3,1,1,3,3/
      DATA (CLASSES(L),L=513,520) /
     $ 2,3,1,3,3,1,2,3/
      DATA (CLASSES(L),L=521,528) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=529,536) /
     $ 3,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=537,544) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=545,552) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=553,560) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=561,568) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=569,576) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=577,584) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=585,592) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=593,600) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=601,608) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=609,616) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=617,624) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=625,632) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=633,640) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=641,648) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=649,656) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=657,664) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=665,672) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=673,680) /
     $ 3,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=681,688) /
     $ 1,1,2,2,2,2,3,3/
      DATA (CLASSES(L),L=689,696) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=697,704) /
     $ 1,3,1,1,2,3,3,1/
      DATA (CLASSES(L),L=705,712) /
     $ 3,1,3,3,1,1,3,1/
      DATA (CLASSES(L),L=713,720) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=721,728) /
     $ 1,1,3,1,2,3,3,3/
      DATA (CLASSES(L),L=729,736) /
     $ 1,3,2,3,1,1,3,3/
      DATA (CLASSES(L),L=737,744) /
     $ 2,3,1,3,3,1,2,3/
      DATA (CLASSES(L),L=745,752) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=753,760) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=761,768) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=769,776) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=777,784) /
     $ 1,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=785,792) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=793,800) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=801,808) /
     $ 3,1,2,3,3,3,1,3/
      DATA (CLASSES(L),L=809,816) /
     $ 2,3,1,1,3,3,2,3/
      DATA (CLASSES(L),L=817,824) /
     $ 1,3,3,1,3,3,2,3/
      DATA (CLASSES(L),L=825,832) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=833,840) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=841,848) /
     $ 3,3,2,3,1,3,1,1/
      DATA (CLASSES(L),L=849,856) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=857,864) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=865,872) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=873,880) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=881,888) /
     $ 1,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=889,896) /
     $ 3,1,3,3,1,1,3,1/
      DATA (CLASSES(L),L=897,904) /
     $ 2,3,3,3,1,3,2,3/
      DATA (CLASSES(L),L=905,912) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=913,920) /
     $ 3,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=921,928) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=929,936) /
     $ 2,3,1,3,1,1,3,3/
      DATA (CLASSES(L),L=937,944) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=945,952) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=953,954) /
     $ 1,1/
      DATA (LEGS(L),L=1,8) /
     $ 0,0,0,0,5,0,0,0/
      DATA (LEGS(L),L=9,16) /
     $ 5,0,0,0,0,0,4,0/
      DATA (LEGS(L),L=17,24) /
     $ 0,0,4,0,2,0,5,0/
      DATA (LEGS(L),L=25,32) /
     $ 2,5,2,0,4,0,2,4/
      DATA (LEGS(L),L=33,40) /
     $ 2,0,0,4,0,5,0,0/
      DATA (LEGS(L),L=41,48) /
     $ 2,5,0,4,2,0,0,5/
      DATA (LEGS(L),L=49,56) /
     $ 0,4,0,0,2,4,0,5/
      DATA (LEGS(L),L=57,64) /
     $ 2,0,0,0,2,0,2,0/
      DATA (LEGS(L),L=65,72) /
     $ 5,4,0,0,5,0,2,0/
      DATA (LEGS(L),L=73,80) /
     $ 0,4,2,0,0,0,2,0/
      DATA (LEGS(L),L=81,88) /
     $ 0,0,4,0,5,0,4,5/
      DATA (LEGS(L),L=89,96) /
     $ 4,4,5,5,0,0,0,0/
      DATA (LEGS(L),L=97,104) /
     $ 0,0,5,0,0,0,5,0/
      DATA (LEGS(L),L=105,112) /
     $ 0,0,0,0,3,0,0,0/
      DATA (LEGS(L),L=113,120) /
     $ 3,0,2,0,5,0,2,5/
      DATA (LEGS(L),L=121,128) /
     $ 2,0,3,0,2,3,2,0/
      DATA (LEGS(L),L=129,136) /
     $ 0,3,0,5,0,0,2,5/
      DATA (LEGS(L),L=137,144) /
     $ 0,3,2,0,0,5,0,3/
      DATA (LEGS(L),L=145,152) /
     $ 0,0,2,3,0,5,2,0/
      DATA (LEGS(L),L=153,160) /
     $ 5,3,0,0,5,0,2,0/
      DATA (LEGS(L),L=161,168) /
     $ 0,3,0,0,0,0,4,0/
      DATA (LEGS(L),L=169,176) /
     $ 0,0,4,0,0,0,0,0/
      DATA (LEGS(L),L=177,184) /
     $ 3,0,0,0,3,0,2,0/
      DATA (LEGS(L),L=185,192) /
     $ 4,0,2,4,2,0,3,0/
      DATA (LEGS(L),L=193,200) /
     $ 2,3,2,0,0,3,0,4/
      DATA (LEGS(L),L=201,208) /
     $ 0,0,2,4,0,3,2,0/
      DATA (LEGS(L),L=209,216) /
     $ 0,4,0,3,0,0,2,3/
      DATA (LEGS(L),L=217,224) /
     $ 0,4,2,0,4,3,0,0/
      DATA (LEGS(L),L=225,232) /
     $ 4,0,2,0,0,3,1,0/
      DATA (LEGS(L),L=233,240) /
     $ 4,0,1,4,1,0,0,4/
      DATA (LEGS(L),L=241,248) /
     $ 0,5,0,0,1,5,0,4/
      DATA (LEGS(L),L=249,256) /
     $ 1,0,5,0,1,5,1,0/
      DATA (LEGS(L),L=257,264) /
     $ 0,5,0,4,0,0,1,4/
      DATA (LEGS(L),L=265,272) /
     $ 0,5,1,0,0,0,1,0/
      DATA (LEGS(L),L=273,280) /
     $ 1,0,5,4,0,0,5,0/
      DATA (LEGS(L),L=281,288) /
     $ 1,0,0,4,1,0,0,0/
      DATA (LEGS(L),L=289,296) /
     $ 1,0,1,0,3,0,1,3/
      DATA (LEGS(L),L=297,304) /
     $ 1,0,0,3,0,5,0,0/
      DATA (LEGS(L),L=305,312) /
     $ 1,5,0,3,1,0,5,0/
      DATA (LEGS(L),L=313,320) /
     $ 1,5,1,0,0,5,0,3/
      DATA (LEGS(L),L=321,328) /
     $ 0,0,1,3,0,5,1,0/
      DATA (LEGS(L),L=329,336) /
     $ 5,3,0,0,5,0,1,0/
      DATA (LEGS(L),L=337,344) /
     $ 0,3,1,0,3,0,1,3/
      DATA (LEGS(L),L=345,352) /
     $ 1,0,0,3,0,4,0,0/
      DATA (LEGS(L),L=353,360) /
     $ 1,4,0,3,1,0,4,0/
      DATA (LEGS(L),L=361,368) /
     $ 1,4,1,0,0,4,0,3/
      DATA (LEGS(L),L=369,376) /
     $ 0,0,1,3,0,4,1,0/
      DATA (LEGS(L),L=377,384) /
     $ 4,3,0,0,4,0,1,0/
      DATA (LEGS(L),L=385,392) /
     $ 0,3,1,0,0,3,2,0/
      DATA (LEGS(L),L=393,400) /
     $ 0,4,0,5,3,0,0,1/
      DATA (LEGS(L),L=401,408) /
     $ 5,0,0,4,0,2,1,0/
      DATA (LEGS(L),L=409,416) /
     $ 0,3,2,0,0,5,0,4/
      DATA (LEGS(L),L=417,424) /
     $ 3,0,0,1,4,0,0,5/
      DATA (LEGS(L),L=425,432) /
     $ 0,2,1,0,0,4,2,0/
      DATA (LEGS(L),L=433,440) /
     $ 0,3,0,5,4,0,0,1/
      DATA (LEGS(L),L=441,448) /
     $ 5,0,0,3,0,2,1,0/
      DATA (LEGS(L),L=449,456) /
     $ 0,4,2,0,0,5,0,3/
      DATA (LEGS(L),L=457,464) /
     $ 4,0,0,1,3,0,0,5/
      DATA (LEGS(L),L=465,472) /
     $ 0,2,1,0,0,5,2,0/
      DATA (LEGS(L),L=473,480) /
     $ 0,3,0,4,5,0,0,1/
      DATA (LEGS(L),L=481,488) /
     $ 4,0,0,3,0,2,1,0/
      DATA (LEGS(L),L=489,496) /
     $ 0,5,2,0,0,4,0,3/
      DATA (LEGS(L),L=497,504) /
     $ 5,0,0,1,3,0,0,4/
      DATA (LEGS(L),L=505,512) /
     $ 0,2,1,0,3,2,0,0/
      DATA (LEGS(L),L=513,520) /
     $ 3,0,1,0,0,2,1,0/
      DATA (LEGS(L),L=521,528) /
     $ 0,2,0,3,0,0,1,3/
      DATA (LEGS(L),L=529,536) /
     $ 0,2,1,0,0,3,0,2/
      DATA (LEGS(L),L=537,544) /
     $ 0,0,1,2,0,3,1,0/
      DATA (LEGS(L),L=545,552) /
     $ 0,4,3,0,0,5,0,2/
      DATA (LEGS(L),L=553,560) /
     $ 4,0,0,1,2,0,0,5/
      DATA (LEGS(L),L=561,568) /
     $ 0,3,1,0,0,4,3,0/
      DATA (LEGS(L),L=569,576) /
     $ 0,2,0,5,4,0,0,1/
      DATA (LEGS(L),L=577,584) /
     $ 5,0,0,2,0,3,1,0/
      DATA (LEGS(L),L=585,592) /
     $ 0,5,4,0,0,2,0,3/
      DATA (LEGS(L),L=593,600) /
     $ 5,0,0,1,3,0,0,2/
      DATA (LEGS(L),L=601,608) /
     $ 0,4,1,0,0,5,3,0/
      DATA (LEGS(L),L=609,616) /
     $ 0,4,0,2,5,0,0,1/
      DATA (LEGS(L),L=617,624) /
     $ 2,0,0,4,0,3,1,0/
      DATA (LEGS(L),L=625,632) /
     $ 0,5,3,0,0,2,0,4/
      DATA (LEGS(L),L=633,640) /
     $ 5,0,0,1,4,0,0,2/
      DATA (LEGS(L),L=641,648) /
     $ 0,3,1,0,0,5,4,0/
      DATA (LEGS(L),L=649,656) /
     $ 0,3,0,2,5,0,0,1/
      DATA (LEGS(L),L=657,664) /
     $ 2,0,0,3,0,4,1,0/
      DATA (LEGS(L),L=665,672) /
     $ 0,3,0,2,0,0,1,2/
      DATA (LEGS(L),L=673,680) /
     $ 0,3,0,0,4,0,5,0/
      DATA (LEGS(L),L=681,688) /
     $ 4,5,4,4,5,5,0,0/
      DATA (LEGS(L),L=689,696) /
     $ 3,0,5,0,3,5,3,0/
      DATA (LEGS(L),L=697,704) /
     $ 4,0,3,4,3,0,0,4/
      DATA (LEGS(L),L=705,712) /
     $ 0,5,0,0,3,5,0,4/
      DATA (LEGS(L),L=713,720) /
     $ 3,0,0,5,0,4,0,0/
      DATA (LEGS(L),L=721,728) /
     $ 3,4,0,5,3,0,0,0/
      DATA (LEGS(L),L=729,736) /
     $ 3,0,3,0,5,4,0,0/
      DATA (LEGS(L),L=737,744) /
     $ 5,0,3,0,0,4,3,0/
      DATA (LEGS(L),L=745,752) /
     $ 5,0,3,5,3,0,4,0/
      DATA (LEGS(L),L=753,760) /
     $ 3,4,3,0,5,0,3,5/
      DATA (LEGS(L),L=761,768) /
     $ 3,0,4,0,3,4,3,0/
      DATA (LEGS(L),L=769,776) /
     $ 5,0,3,5,3,0,4,0/
      DATA (LEGS(L),L=777,784) /
     $ 3,4,3,0,0,4,0,5/
      DATA (LEGS(L),L=785,792) /
     $ 0,0,3,5,0,4,3,0/
      DATA (LEGS(L),L=793,800) /
     $ 0,5,0,4,0,0,3,4/
      DATA (LEGS(L),L=801,808) /
     $ 0,5,3,0,0,0,3,0/
      DATA (LEGS(L),L=809,816) /
     $ 3,0,5,4,0,0,5,0/
      DATA (LEGS(L),L=817,824) /
     $ 3,0,0,4,0,0,4,0/
      DATA (LEGS(L),L=825,832) /
     $ 5,0,4,5,3,0,5,0/
      DATA (LEGS(L),L=833,840) /
     $ 3,5,3,0,4,0,3,4/
      DATA (LEGS(L),L=841,848) /
     $ 0,0,4,0,5,0,4,5/
      DATA (LEGS(L),L=849,856) /
     $ 3,0,5,0,3,5,3,0/
      DATA (LEGS(L),L=857,864) /
     $ 4,0,3,4,3,0,5,0/
      DATA (LEGS(L),L=865,872) /
     $ 3,5,3,0,4,0,3,4/
      DATA (LEGS(L),L=873,880) /
     $ 3,0,0,4,0,5,0,0/
      DATA (LEGS(L),L=881,888) /
     $ 3,5,0,4,3,0,0,5/
      DATA (LEGS(L),L=889,896) /
     $ 0,4,0,0,3,4,0,5/
      DATA (LEGS(L),L=897,904) /
     $ 3,0,0,0,3,0,3,0/
      DATA (LEGS(L),L=905,912) /
     $ 5,4,0,0,5,0,3,0/
      DATA (LEGS(L),L=913,920) /
     $ 0,4,0,0,4,0,5,0/
      DATA (LEGS(L),L=921,928) /
     $ 4,5,3,0,5,0,3,5/
      DATA (LEGS(L),L=929,936) /
     $ 3,0,4,0,3,4,0,0/
      DATA (LEGS(L),L=937,944) /
     $ 4,0,5,0,4,5,3,0/
      DATA (LEGS(L),L=945,952) /
     $ 5,0,3,5,3,0,4,0/
      DATA (LEGS(L),L=953,954) /
     $ 3,4/
      DATA (SIGNS(L),L=1,8) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (SIGNS(L),L=9,16) /
     $ -1,0,0,0,0,0,-1,0/
      DATA (SIGNS(L),L=17,24) /
     $ 0,0,-1,0,-1,0,-1,0/
      DATA (SIGNS(L),L=25,32) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=33,40) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=41,48) /
     $ -1,1,0,1,-1,0,0,-1/
      DATA (SIGNS(L),L=49,56) /
     $ 0,-1,0,0,-1,1,0,1/
      DATA (SIGNS(L),L=57,64) /
     $ -1,0,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=65,72) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=73,80) /
     $ 0,1,-1,0,0,0,-1,0/
      DATA (SIGNS(L),L=81,88) /
     $ 0,0,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=89,96) /
     $ -1,-1,-1,-1,0,0,0,0/
      DATA (SIGNS(L),L=97,104) /
     $ 0,0,-1,0,0,0,-1,0/
      DATA (SIGNS(L),L=105,112) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (SIGNS(L),L=113,120) /
     $ -1,0,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=121,128) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=129,136) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=137,144) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=145,152) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=153,160) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=161,168) /
     $ 0,1,0,0,0,0,-1,0/
      DATA (SIGNS(L),L=169,176) /
     $ 0,0,-1,0,0,0,0,0/
      DATA (SIGNS(L),L=177,184) /
     $ -1,0,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=185,192) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=193,200) /
     $ -1,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=201,208) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=209,216) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=217,224) /
     $ 0,1,-1,0,1,-1,0,0/
      DATA (SIGNS(L),L=225,232) /
     $ 1,0,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=233,240) /
     $ -1,0,-1,1,-1,0,0,-1/
      DATA (SIGNS(L),L=241,248) /
     $ 0,-1,0,0,-1,1,0,1/
      DATA (SIGNS(L),L=249,256) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=257,264) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=265,272) /
     $ 0,1,-1,0,0,0,-1,0/
      DATA (SIGNS(L),L=273,280) /
     $ -1,0,1,-1,0,0,1,0/
      DATA (SIGNS(L),L=281,288) /
     $ -1,0,0,1,-1,0,0,0/
      DATA (SIGNS(L),L=289,296) /
     $ -1,0,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=297,304) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=305,312) /
     $ -1,1,0,1,-1,0,-1,0/
      DATA (SIGNS(L),L=313,320) /
     $ -1,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=321,328) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=329,336) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=337,344) /
     $ 0,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=345,352) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=353,360) /
     $ -1,1,0,1,-1,0,-1,0/
      DATA (SIGNS(L),L=361,368) /
     $ -1,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=369,376) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=377,384) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=385,392) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=393,400) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=401,408) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=409,416) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=417,424) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=425,432) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=433,440) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=441,448) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=449,456) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=457,464) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=465,472) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=473,480) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=481,488) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=489,496) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=497,504) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=505,512) /
     $ 0,1,-1,0,1,-1,0,0/
      DATA (SIGNS(L),L=513,520) /
     $ 1,0,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=521,528) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=529,536) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=537,544) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=545,552) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=553,560) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=561,568) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=569,576) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=577,584) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=585,592) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=593,600) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=601,608) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=609,616) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=617,624) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=625,632) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=633,640) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=641,648) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=649,656) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=657,664) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=665,672) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=673,680) /
     $ 0,1,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=681,688) /
     $ -1,1,-1,-1,-1,-1,0,0/
      DATA (SIGNS(L),L=689,696) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=697,704) /
     $ -1,0,-1,1,-1,0,0,-1/
      DATA (SIGNS(L),L=705,712) /
     $ 0,-1,0,0,-1,1,0,1/
      DATA (SIGNS(L),L=713,720) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=721,728) /
     $ -1,1,0,1,-1,0,0,0/
      DATA (SIGNS(L),L=729,736) /
     $ -1,0,-1,0,1,-1,0,0/
      DATA (SIGNS(L),L=737,744) /
     $ 1,0,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=745,752) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=753,760) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=761,768) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=769,776) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=777,784) /
     $ -1,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=785,792) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=793,800) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=801,808) /
     $ 0,1,-1,0,0,0,-1,0/
      DATA (SIGNS(L),L=809,816) /
     $ -1,0,1,-1,0,0,1,0/
      DATA (SIGNS(L),L=817,824) /
     $ -1,0,0,1,0,0,-1,0/
      DATA (SIGNS(L),L=825,832) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=833,840) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=841,848) /
     $ 0,0,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=849,856) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=857,864) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=865,872) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=873,880) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=881,888) /
     $ -1,1,0,1,-1,0,0,-1/
      DATA (SIGNS(L),L=889,896) /
     $ 0,-1,0,0,-1,1,0,1/
      DATA (SIGNS(L),L=897,904) /
     $ -1,0,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=905,912) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=913,920) /
     $ 0,1,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=921,928) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=929,936) /
     $ -1,0,-1,0,-1,1,0,0/
      DATA (SIGNS(L),L=937,944) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=945,952) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=953,954) /
     $ -1,1/
      DATA (CLOSURES(L),L=1,8) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=9,16) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=17,24) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=25,32) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=33,40) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=41,48) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=49,56) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=57,64) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=65,72) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=73,80) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=81,88) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=89,96) /
     $ 1,1,1,1,0,0,0,0/
      DATA (CLOSURES(L),L=97,104) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=105,112) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=113,120) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=121,128) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=129,136) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=137,144) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=145,152) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=153,160) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=161,168) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=169,176) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=177,184) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=185,192) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=193,200) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=201,208) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=209,216) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=217,224) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=225,232) /
     $ -1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=233,240) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=241,248) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=249,256) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=257,264) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=265,272) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=273,280) /
     $ 1,0,0,0,0,0,-1,0/
      DATA (CLOSURES(L),L=281,288) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=289,296) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=297,304) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=305,312) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=313,320) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=321,328) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=329,336) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=337,344) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=345,352) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=353,360) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=361,368) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=369,376) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=377,384) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=385,392) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=393,400) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=401,408) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=409,416) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=417,424) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=425,432) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=433,440) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=441,448) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=449,456) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=457,464) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=465,472) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=473,480) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=481,488) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=489,496) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=497,504) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=505,512) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=513,520) /
     $ -1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=521,528) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=529,536) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=537,544) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=545,552) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=553,560) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=561,568) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=569,576) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=577,584) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=585,592) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=593,600) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=601,608) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=609,616) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=617,624) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=625,632) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=633,640) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=641,648) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=649,656) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=657,664) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=665,672) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=673,680) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=681,688) /
     $ 0,0,1,1,1,1,0,0/
      DATA (CLOSURES(L),L=689,696) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=697,704) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=705,712) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=713,720) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=721,728) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=729,736) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=737,744) /
     $ -1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=745,752) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=753,760) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=761,768) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=769,776) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=777,784) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=785,792) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=793,800) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=801,808) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=809,816) /
     $ 1,0,0,0,0,0,-1,0/
      DATA (CLOSURES(L),L=817,824) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=825,832) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=833,840) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=841,848) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=849,856) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=857,864) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=865,872) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=873,880) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=881,888) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=889,896) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=897,904) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=905,912) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=913,920) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=921,928) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=929,936) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=937,944) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=945,952) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=953,954) /
     $ 0,0/
      DATA (SHELLS(L),L=1,8) /
     $ 0,0,0,0,1,0,0,0/
      DATA (SHELLS(L),L=9,16) /
     $ 1,0,0,0,0,0,1,0/
      DATA (SHELLS(L),L=17,24) /
     $ 0,0,1,0,1,0,1,0/
      DATA (SHELLS(L),L=25,32) /
     $ 1,1,1,0,1,0,1,1/
      DATA (SHELLS(L),L=33,40) /
     $ 1,0,0,1,0,1,0,0/
      DATA (SHELLS(L),L=41,48) /
     $ 1,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=49,56) /
     $ 0,1,0,0,1,1,0,1/
      DATA (SHELLS(L),L=57,64) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=65,72) /
     $ 1,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=73,80) /
     $ 0,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=81,88) /
     $ 0,0,1,0,1,0,1,1/
      DATA (SHELLS(L),L=89,96) /
     $ 1,1,1,1,0,0,0,0/
      DATA (SHELLS(L),L=97,104) /
     $ 0,0,1,0,0,0,1,0/
      DATA (SHELLS(L),L=105,112) /
     $ 0,0,0,0,1,0,0,0/
      DATA (SHELLS(L),L=113,120) /
     $ 1,0,1,0,1,0,1,1/
      DATA (SHELLS(L),L=121,128) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=129,136) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=137,144) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=145,152) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=153,160) /
     $ 1,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=161,168) /
     $ 0,1,0,0,0,0,1,0/
      DATA (SHELLS(L),L=169,176) /
     $ 0,0,1,0,0,0,0,0/
      DATA (SHELLS(L),L=177,184) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=185,192) /
     $ 1,0,1,1,1,0,1,0/
      DATA (SHELLS(L),L=193,200) /
     $ 1,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=201,208) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=209,216) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=217,224) /
     $ 0,1,1,0,1,1,0,0/
      DATA (SHELLS(L),L=225,232) /
     $ 1,0,1,0,0,1,1,0/
      DATA (SHELLS(L),L=233,240) /
     $ 1,0,1,1,1,0,0,1/
      DATA (SHELLS(L),L=241,248) /
     $ 0,1,0,0,1,1,0,1/
      DATA (SHELLS(L),L=249,256) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=257,264) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=265,272) /
     $ 0,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=273,280) /
     $ 1,0,1,1,0,0,1,0/
      DATA (SHELLS(L),L=281,288) /
     $ 1,0,0,1,1,0,0,0/
      DATA (SHELLS(L),L=289,296) /
     $ 1,0,1,0,1,0,1,1/
      DATA (SHELLS(L),L=297,304) /
     $ 1,0,0,1,0,1,0,0/
      DATA (SHELLS(L),L=305,312) /
     $ 1,1,0,1,1,0,1,0/
      DATA (SHELLS(L),L=313,320) /
     $ 1,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=321,328) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=329,336) /
     $ 1,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=337,344) /
     $ 0,1,1,0,1,0,1,1/
      DATA (SHELLS(L),L=345,352) /
     $ 1,0,0,1,0,1,0,0/
      DATA (SHELLS(L),L=353,360) /
     $ 1,1,0,1,1,0,1,0/
      DATA (SHELLS(L),L=361,368) /
     $ 1,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=369,376) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=377,384) /
     $ 1,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=385,392) /
     $ 0,1,1,0,0,1,1,0/
      DATA (SHELLS(L),L=393,400) /
     $ 0,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=401,408) /
     $ 1,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=409,416) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=417,424) /
     $ 1,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=425,432) /
     $ 0,1,1,0,0,1,1,0/
      DATA (SHELLS(L),L=433,440) /
     $ 0,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=441,448) /
     $ 1,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=449,456) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=457,464) /
     $ 1,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=465,472) /
     $ 0,1,1,0,0,1,1,0/
      DATA (SHELLS(L),L=473,480) /
     $ 0,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=481,488) /
     $ 1,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=489,496) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=497,504) /
     $ 1,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=505,512) /
     $ 0,1,1,0,1,1,0,0/
      DATA (SHELLS(L),L=513,520) /
     $ 1,0,1,0,0,1,1,0/
      DATA (SHELLS(L),L=521,528) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=529,536) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=537,544) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=545,552) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=553,560) /
     $ 1,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=561,568) /
     $ 0,1,1,0,0,1,1,0/
      DATA (SHELLS(L),L=569,576) /
     $ 0,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=577,584) /
     $ 1,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=585,592) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=593,600) /
     $ 1,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=601,608) /
     $ 0,1,1,0,0,1,1,0/
      DATA (SHELLS(L),L=609,616) /
     $ 0,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=617,624) /
     $ 1,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=625,632) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=633,640) /
     $ 1,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=641,648) /
     $ 0,1,1,0,0,1,1,0/
      DATA (SHELLS(L),L=649,656) /
     $ 0,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=657,664) /
     $ 1,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=665,672) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=673,680) /
     $ 0,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=681,688) /
     $ 1,1,1,1,1,1,0,0/
      DATA (SHELLS(L),L=689,696) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=697,704) /
     $ 1,0,1,1,1,0,0,1/
      DATA (SHELLS(L),L=705,712) /
     $ 0,1,0,0,1,1,0,1/
      DATA (SHELLS(L),L=713,720) /
     $ 1,0,0,1,0,1,0,0/
      DATA (SHELLS(L),L=721,728) /
     $ 1,1,0,1,1,0,0,0/
      DATA (SHELLS(L),L=729,736) /
     $ 1,0,1,0,1,1,0,0/
      DATA (SHELLS(L),L=737,744) /
     $ 1,0,1,0,0,1,1,0/
      DATA (SHELLS(L),L=745,752) /
     $ 1,0,1,1,1,0,1,0/
      DATA (SHELLS(L),L=753,760) /
     $ 1,1,1,0,1,0,1,1/
      DATA (SHELLS(L),L=761,768) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=769,776) /
     $ 1,0,1,1,1,0,1,0/
      DATA (SHELLS(L),L=777,784) /
     $ 1,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=785,792) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=793,800) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=801,808) /
     $ 0,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=809,816) /
     $ 1,0,1,1,0,0,1,0/
      DATA (SHELLS(L),L=817,824) /
     $ 1,0,0,1,0,0,1,0/
      DATA (SHELLS(L),L=825,832) /
     $ 1,0,1,1,1,0,1,0/
      DATA (SHELLS(L),L=833,840) /
     $ 1,1,1,0,1,0,1,1/
      DATA (SHELLS(L),L=841,848) /
     $ 0,0,1,0,1,0,1,1/
      DATA (SHELLS(L),L=849,856) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=857,864) /
     $ 1,0,1,1,1,0,1,0/
      DATA (SHELLS(L),L=865,872) /
     $ 1,1,1,0,1,0,1,1/
      DATA (SHELLS(L),L=873,880) /
     $ 1,0,0,1,0,1,0,0/
      DATA (SHELLS(L),L=881,888) /
     $ 1,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=889,896) /
     $ 0,1,0,0,1,1,0,1/
      DATA (SHELLS(L),L=897,904) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=905,912) /
     $ 1,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=913,920) /
     $ 0,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=921,928) /
     $ 1,1,1,0,1,0,1,1/
      DATA (SHELLS(L),L=929,936) /
     $ 1,0,1,0,1,1,0,0/
      DATA (SHELLS(L),L=937,944) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=945,952) /
     $ 1,0,1,1,1,0,1,0/
      DATA (SHELLS(L),L=953,954) /
     $ 1,1/
      CLASS=0
      LEG=0
      SGN=0
      CLOSURE=0
      DECLARED_ONSHELL=.FALSE.
      KNOWN=.FALSE.
      IF(G.LT.1.OR.G.GT.116) RETURN
      IF(GVALID(G).NE.1.OR.N.NE.GNUM(G)) RETURN
      IF(I.LT.1.OR.J.LT.1.OR.I.GT.N.OR.J.GT.N) RETURN
      IF(I.EQ.J) RETURN
      DIR=0
      IF(CT.EQ.2.OR.CT.EQ.5) DIR=1
      IF(CT.NE.1.AND.CT.NE.2.AND.CT.NE.4.AND.CT.NE.5)
     $ RETURN
      A=MIN(I,J)
      B=MAX(I,J)
      ORI=1
      IF(I.GT.J) ORI=-1
      NP=N*(N-1)/2
      K=GOFF(G)+DIR*NP+(A-1)*N-(A-1)*A/2+B-A
      CLASS=CLASSES(K)
      LEG=LEGS(K)
      SGN=ORI*SIGNS(K)
      CLOSURE=ORI*CLOSURES(K)
      DECLARED_ONSHELL=SHELLS(K).EQ.1
      KNOWN=.TRUE.
      END

      SUBROUTINE KC_MP_EXTERNAL_SHELL_MASS2(LEG,MASS2,VALID)
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER LEG
      REAL*16 MASS2,MASS
      LOGICAL VALID
      INCLUDE 'mp_coupl.inc'
      MASS2=0.0E0_16
      VALID=.FALSE.
      IF(LEG.EQ.1) THEN
        VALID=.TRUE.
        RETURN
      ENDIF
      IF(LEG.EQ.2) THEN
        VALID=.TRUE.
        RETURN
      ENDIF
      IF(LEG.EQ.3) THEN
        MASS=MP__MDL_MZ
        IF(.NOT.IEEE_IS_FINITE(MASS)) RETURN
        IF(MASS.LT.0.0E0_16) RETURN
        IF(MASS.GE.SQRT(HUGE(MASS))) RETURN
        MASS2=MASS**2
        VALID=IEEE_IS_FINITE(MASS2)
        RETURN
      ENDIF
      IF(LEG.EQ.4) THEN
        VALID=.TRUE.
        RETURN
      ENDIF
      IF(LEG.EQ.5) THEN
        VALID=.TRUE.
        RETURN
      ENDIF
      END

      SUBROUTINE KC_MP_BUILD_KINEMATIC_MATRIX_KIN(G,CT,POLICY,
     $ N,P_LOOP,M2L,S_MAT)
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER G,CT,POLICY,N,I,J,K,CLASS,LEG,SGN,CLOSURE
      REAL*16 P_LOOP(N,0:3),REF_NORMALIZATION,TARGET,TOL
      REAL*16 CLOSURE_SIZE,PAIR_SCALE,RESIDUAL
      COMPLEX*32 M2L(N),S_MAT(N,N),DIFFSQ
      LOGICAL KNOWN,DECLARED_ONSHELL,TARGET_VALID,TOL_OK
      LOGICAL P_FINITE
      INCLUDE 'MadLoopParams.inc'
C     Policy zero is explicit exact legacy compatibility, including its
C     internal-mass threshold guesses. No metadata lookup occurs there.
      IF(POLICY.EQ.0) THEN
        CALL KC_MP_BUILD_KINEMATIC_MATRIX(N,P_LOOP,M2L,S_MAT)
        RETURN
      ENDIF
C     No coordinate repair, projection, parameter update or flag
C      masking.
C     First shift is the literal corner sum in these adapters, not
C      zero.
      TOL=REAL(OSTHRES,KIND=16)
      TOL_OK=IEEE_IS_FINITE(TOL)
      IF(TOL_OK) TOL_OK=TOL.GT.0.0E0_16.AND.TOL.LT.1.0E0_16
      P_FINITE=ALL(IEEE_IS_FINITE(P_LOOP))
      CLOSURE_SIZE=0.0E0_16
      IF(N.GT.0.AND.TOL_OK.AND.POLICY.EQ.1.AND.P_FINITE) THEN
        CLOSURE_SIZE=MAXVAL(ABS(P_LOOP(1,0:3)))
      ENDIF
      DO I=1,N
        DO J=1,N
          IF(I.EQ.J) THEN
            S_MAT(I,J)=-(M2L(I)+M2L(J))
          ELSE
C           Preserve the original computed-invariant arithmetic order.
            DIFFSQ=(CMPLX(P_LOOP(I,0),0.0E0_16,KIND=16)-
     $               CMPLX(P_LOOP(J,0),0.0E0_16,KIND=16))**2
            DO K=1,3
              DIFFSQ=DIFFSQ-(CMPLX(P_LOOP(I,K),0.0E0_16,KIND=16)-
     $               CMPLX(P_LOOP(J,K),0.0E0_16,KIND=16))**2
            ENDDO
            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)
            IF(POLICY.NE.1.OR..NOT.TOL_OK) CYCLE
            IF(.NOT.P_FINITE) CYCLE
            CALL KC_KINEMATIC_IDENTITY(G,CT,N,I,J,CLASS,LEG,
     $           SGN,CLOSURE,DECLARED_ONSHELL,KNOWN)
            IF(.NOT.KNOWN.OR..NOT.DECLARED_ONSHELL) CYCLE
            IF(CLASS.NE.1.AND.CLASS.NE.2) CYCLE
C           Normalize to this external pair, not a hard common shift.
            PAIR_SCALE=0.0E0_16
            DO K=0,3
              PAIR_SCALE=MAX(PAIR_SCALE,
     $                      ABS(P_LOOP(I,K)-P_LOOP(J,K)))
            ENDDO
            IF(.NOT.IEEE_IS_FINITE(PAIR_SCALE)) CYCLE
C           Complements additionally need closure at the leg's scale.
C           Reject ratios above one before division, avoiding overflow.
            IF(CLOSURE.NE.0) THEN
              IF(CLOSURE_SIZE.GT.PAIR_SCALE) CYCLE
              IF(PAIR_SCALE.EQ.0.0E0_16) THEN
                IF(CLOSURE_SIZE.NE.0.0E0_16) CYCLE
              ELSE
                IF(CLOSURE_SIZE/PAIR_SCALE.GT.TOL) CYCLE
              ENDIF
            ENDIF
            CALL KC_MP_EXTERNAL_SHELL_MASS2(LEG,TARGET,TARGET_VALID)
            IF(.NOT.TARGET_VALID) CYCLE
            IF(.NOT.IEEE_IS_FINITE(REAL(DIFFSQ,KIND=16))) CYCLE
            IF(.NOT.IEEE_IS_FINITE(AIMAG(DIFFSQ))) CYCLE
            IF(TARGET.NE.0.0E0_16) THEN
              RESIDUAL=ABS((DIFFSQ-TARGET)/TARGET)
            ELSE
C             A soft external leg must pass at its own four-vector
C             scale; unrelated hard shifts cannot dilute its defect.
C             Extreme scales conservatively retain computed invariants.
              IF(PAIR_SCALE.GE.SQRT(HUGE(PAIR_SCALE))) CYCLE
              IF(PAIR_SCALE.LT.SQRT(TINY(PAIR_SCALE))) CYCLE
              REF_NORMALIZATION=PAIR_SCALE**2
              IF(.NOT.IEEE_IS_FINITE(REF_NORMALIZATION)) CYCLE
              IF(REF_NORMALIZATION.LE.0.0E0_16) CYCLE
              RESIDUAL=ABS(DIFFSQ/REF_NORMALIZATION)
            ENDIF
            IF(IEEE_IS_FINITE(RESIDUAL)) THEN
              IF(RESIDUAL.LT.TOL) THEN
                S_MAT(I,J)=CMPLX(TARGET,0.0E0_16,KIND=16)
     $                     -M2L(I)-M2L(J)
              ENDIF
            ENDIF
          ENDIF
        ENDDO
      ENDDO
      END
      SUBROUTINE KC_MP_TIRUST_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MTIRUST, ONLY: KI_QTIR
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER(NLOOPGROUPS=116,NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK,CTMODE,I,J
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      COMPLEX*32 COEFS(0:LOOPMAXCOEFS-1,3),RESULT(3)
      REAL*8 LSCALE
      COMMON/KC_CT/LSCALE,CTMODE
      REAL*16 NORMALIZATION
      PARAMETER(NORMALIZATION=1.0E0_16/(16.0E0_16
     $ *3.141592653589793238462643383279502884E0_16**2))
      LOGICAL STABLE,FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION
      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_MP_LCOEFS/LOOPCOEFS
      RES=0
      CALL KC_MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,PL,M2L
     $ ,COEFS,STABLE)
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

      SUBROUTINE KC_MP_TIRUST_LOOP_QPMASS(NLOOPLINE,PL,M2L,RANK,RES
     $ ,STABLE)
      USE MTIRUST, ONLY: KI_QTIR
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER(NLOOPGROUPS=116,NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NLOOPLINE,RANK,CTMODE,I,J
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*32 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      COMPLEX*32 COEFS(0:LOOPMAXCOEFS-1,3),RESULT(3)
      REAL*8 LSCALE
      COMMON/KC_CT/LSCALE,CTMODE
      REAL*16 NORMALIZATION
      PARAMETER(NORMALIZATION=1.0E0_16/(16.0E0_16
     $ *3.141592653589793238462643383279502884E0_16**2))
      LOGICAL STABLE,FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION
      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_MP_LCOEFS/LOOPCOEFS
      RES=0
      CALL KC_MP_TIRUST_TENSOR_DIAGNOSTIC_QPMASS(CTMODE,NLOOPLINE,RANK
     $ ,PL,M2L,COEFS,STABLE)
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

      SUBROUTINE KC_MP_TIRUST_LOOP_QPMASS_KIN(GROUP_ID,CTMODE
     $ ,KIN_POLICY,NLOOPLINE,PL,M2L,RANK,RES,STABLE)
      USE MTIRUST, ONLY: KI_QTIR
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INTEGER NLOOPGROUPS,NSQUAREDSO
      PARAMETER(NLOOPGROUPS=116,NSQUAREDSO=1)
      INCLUDE 'loop_max_coefs.inc'
      INTEGER GROUP_ID,KIN_POLICY,NLOOPLINE,RANK,CTMODE,I,J
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*32 M2L(NLOOPLINE)
      COMPLEX*16 RES(3)
      COMPLEX*32 COEFS(0:LOOPMAXCOEFS-1,3),RESULT(3)
      REAL*16 NORMALIZATION
      PARAMETER(NORMALIZATION=1.0E0_16/(16.0E0_16
     $ *3.141592653589793238462643383279502884E0_16**2))
      LOGICAL STABLE,FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/KC_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION
     $ ,FPE_IN_QP_REDUCTION
      INTEGER ID,SQSOINDEX,R
      COMMON/KC_LOOP/ID,SQSOINDEX,R
      COMPLEX*32 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
      COMMON/KC_MP_LCOEFS/LOOPCOEFS
      RES=0
      CALL KC_MP_TIRUST_TENSOR_DIAGNOSTIC_QPMASS_KIN(GROUP_ID
     $ ,KIN_POLICY,CTMODE,NLOOPLINE,RANK,PL,M2L,COEFS,STABLE)
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

C     TIRust tensor reduction. The public diagnostic returns
C      MadLoop-ordered
C     moments, marks an unevaluated finite coefficient with NaN, and
C      performs
C     no numerator contraction or counterterm addition.
      SUBROUTINE KC_MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK
     $ ,PL,M2L,TIRCOEFS,STABLE)
      USE MTIRUST
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
     $ ,IEEE_VALUE,IEEE_QUIET_NAN
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
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


      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      STABLE=.FALSE.
      TIRCOEFS=0
      IF(RANK.LT.0.OR.RANK.GT.6) RETURN
      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))
     $  RETURN
      CALL KC_INITIALIZE_TIRUST(STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      CALL TIRUST_SET_CACHE_BUDGET(INT(TIRUSTCACHEMAXBYTES,C_INT64_T)
     $ ,STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QTIR)
      CALL KC_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)
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
      CALL KC_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
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
      SUBROUTINE KC_MP_TIRUST_TENSOR_DIAGNOSTIC_QPMASS(CTMODE
     $ ,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      USE MTIRUST
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
     $ ,IEEE_VALUE,IEEE_QUIET_NAN
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
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


      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      STABLE=.FALSE.
      TIRCOEFS=0
      IF(RANK.LT.0.OR.RANK.GT.6) RETURN
      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))
     $  RETURN
      CALL KC_INITIALIZE_TIRUST(STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      CALL TIRUST_SET_CACHE_BUDGET(INT(TIRUSTCACHEMAXBYTES,C_INT64_T)
     $ ,STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QTIR)
      CALL KC_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)
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
      CALL KC_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)
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
      SUBROUTINE KC_MP_TIRUST_TENSOR_DIAGNOSTIC_QPMASS_KIN(GROUP_ID
     $ ,KIN_POLICY,CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)
      USE MTIRUST
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
     $ ,IEEE_VALUE,IEEE_QUIET_NAN
      USE KC_POLYNOMIAL_CONSTANTS, ONLY: NCOEF_R
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER GROUP_ID,KIN_POLICY,CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS
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


      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      STABLE=.FALSE.
      TIRCOEFS=0
      IF(RANK.LT.0.OR.RANK.GT.6) RETURN
      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))
     $  RETURN
      CALL KC_INITIALIZE_TIRUST(STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      CALL TIRUST_SET_CACHE_BUDGET(INT(TIRUSTCACHEMAXBYTES,C_INT64_T)
     $ ,STATUS)
      IF(STATUS.NE.TIRUST_SUCCESS) RETURN
      PDEN_DUMMY=0
      MASSES=CMPLX(M2L,KIND=KI_QTIR)
      CALL KC_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)
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
      CALL KC_MP_BUILD_KINEMATIC_MATRIX_KIN(GROUP_ID,CTMODE,KIN_POLICY
     $ ,NLOOPLINE,P_S_MAT,MASSES,S_MAT)
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


