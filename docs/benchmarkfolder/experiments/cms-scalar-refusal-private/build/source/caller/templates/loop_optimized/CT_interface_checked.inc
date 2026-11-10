C     ===========================================
C     ===== Beginning of CutTools Interface =====
C     ===========================================
      SUBROUTINE %(proc_prefix)sCTLOOP_CHECKED(NLOOPLINE,PL,M2L,RANK,RES,STABLE,CT_STATUS)
      USE CTS_CHECKED_REDUCTION, ONLY: CTSXCUT_CHECKED
      USE CTS_CHECKED_PROTOCOL, ONLY: CTS_CHECKED_REPORT
      USE CMS_SCALAR_OBSERVER, ONLY: CMS_OK

C  
%(info_lines)s
C 
C Interface between MG5 and CutTools.
C  
%(process_lines)s
C  
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)
      %(real_dp_format)s REF_P
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s M2LCT(0:NLOOPLINE-1)	  
      %(complex_dp_format)s RES(3)
	  LOGICAL STABLE	  
C  
C LOCAL VARIABLES 
C
	  %(complex_dp_format)s R1, ACC
	  INTEGER I, J, K
      TYPE(CTS_CHECKED_REPORT) CT_STATUS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION

	  LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT,COLLIERINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT,COLLIERINIT
C
C EXTERNAL FUNCTIONS
C
      EXTERNAL %(proc_prefix)sLOOPNUM
      EXTERNAL %(proc_prefix)sMPLOOPNUM
C
C GLOBAL VARIABLES
C
      %(include_vector)s
      include 'coupl.inc'
	  INTEGER CTMODE
	  %(real_dp_format)s LSCALE
      common/%(proc_prefix)sCT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R

C ----------
C BEGIN CODE
C ----------

C INITIALIZE CUTTOOLS IF NEEDED
      IF (CTINIT) THEN
	    CTINIT=.FALSE.
		CALL %(proc_prefix)sINITCT()
		CALL cite('Ossola:2007ax','one-loop reduction with CutTools')
      ENDIF

C YOU CAN FIND THE DETAILS ABOUT THE DIFFERENT CTMODE AT THE BEGINNING OF THE FILE CTS_CUTS.F90 IN THE CUTTOOLS DISTRIBUTION

C  CONVERT THE MASSES TO BE COMPLEX
      do I=1,NLOOPLINE
	    M2LCT(I-1)=M2L(I)
	  ENDDO

C  CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO CT CONVENTIONS
      do I=0,3
        ABSPCT(I)=0.D0
        do J=0,(NLOOPLINE-1)
          PCT(I,J)=0.D0
        enddo
      enddo
      do I=0,3
        do J=1,NLOOPLINE
          PCT(I,0)=PCT(I,0)+PL(I,J)
          ABSPCT(I)=ABSPCT(I)+ABS(PL(I,J))
        enddo
      enddo
      ref_p = max(ABSPCT(0), ABSPCT(1),ABSPCT(2),ABSPCT(3))
       do I=0,3
           ABSPCT(I) = MAX(ref_p*1e-6, ABSPCT(I))
       enddo
      if (checkPConservation.and.ref_p.gt.1d-8) then
        if ((PCT(0,0)/ABSPCT(0)).gt.1.d-6) then
          write(*,*) 'energy is not conserved (flag:CT95)',PCT(0,0)
          stop 'energy is not conserved (flag:CT95)'
        elseif ((PCT(1,0)/ABSPCT(1)).gt.1.d-6) then
         write(*,*) 'px is not conserved (flag:CT95)',PCT(1,0)
         stop 'px is not conserved (flag:CT95)'
        elseif ((PCT(2,0)/ABSPCT(2)).gt.1.d-6) then
         write(*,*) 'py is not conserved (flag:CT95)',PCT(2,0)
         stop 'py is not conserved (flag:CT95)'
        elseif ((PCT(3,0)/ABSPCT(3)).gt.1.d-6) then
         write(*,*) 'pz is not conserved (flag:CT95)',PCT(3,0)
         stop 'pz is not conserved (flag:CT95)'
        endif
      endif
      do I=0,3
        do J=1,(NLOOPLINE-1)
          do K=1,J
            PCT(I,J)=PCT(I,J)+PL(I,K)
          enddo
        enddo
      enddo

      CALL CTSXCUT_CHECKED(CTMODE,LSCALE,MU_R,NLOOPLINE,%(proc_prefix)sLOOPNUM,%(proc_prefix)sMPLOOPNUM,RANK,PCT,M2LCT,RES,ACC,R1,STABLE,CT_STATUS)
C     The existing FPE flag is a hard-refusal latch, not an invented IEEE exception.
C     Select the active MadLoop pass; CT_STATUS records actual internal precision.
      IF(CT_STATUS%%STATUS.NE.CMS_OK.OR..NOT.CT_STATUS%%RESULT_VALID) THEN
        STABLE=.FALSE.
        IF(CTMODE.GE.4) THEN
          FPE_IN_QP_REDUCTION=.TRUE.
        ELSE
          FPE_IN_DP_REDUCTION=.TRUE.
        ENDIF
        RETURN
      ENDIF

## if(AmplitudeReduction) {
      RES(1)=NORMALIZATION*RES(1)
      RES(2)=NORMALIZATION*RES(2)
      RES(3)=NORMALIZATION*RES(3)
## } else {
      RES(1)=NORMALIZATION*2.0d0*DBLE(RES(1))
      RES(2)=NORMALIZATION*2.0d0*DBLE(RES(2))
      RES(3)=NORMALIZATION*2.0d0*DBLE(RES(3))
## }
C     WRITE(*,*) 'CutTools: Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

