C Bound-state (onium) version of matrix_madevent_group_v4.inc
SUBROUTINE SMATRIX%(proc_id)s(P, IFLAV, RHEL, RCOL, channel, IVEC, ANS, IHEL, ICOL)
C 
%(info_lines)s
C 
C MadGraph7 for Madevent Version
C 
C Returns amplitude squared -- no average over initial state/symmetry factor
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C 
    use DiscreteSampler
    IMPLICIT NONE
C  
C CONSTANTS
C  
    Include 'genps.inc'
    Include 'maxconfigs.inc'
    Include 'nexternal.inc'
    Include 'maxamps.inc'
    include '../../Source/vector.inc' ! defines VECSIZE_MEMMAX
    include 'run.inc'
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=%(ncomb)s)
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d) 
    INTEGER    NDIAGS
    PARAMETER (NDIAGS=%(ndiags)d) 
    INTEGER    THEL
    PARAMETER (THEL=2*NCOMB)
C  
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL),ANS
    double precision RHEL ! random number for selecting helicity
    double precision RCOL ! random number for selecting helicity
    integer channel ! channel to keep for the multi-channel
    integer ivec ! for using the correct coupling
    integer iflav
c
c   output argument
c
    integer ihel, icol
c
c global (due to reading writing)
c
C       Good-helicity filter is per (helicity, flavor, subprocess): a helicity
C       that is zero for one merged flavor but non-zero for another must not be
C       dropped, so GOODHEL/NTRY carry an extra MAXFLAVPERPROC (IFLAV) dimension.
C       NTRY is also kept per beam orientation (IM = IMIRROR), so that each
C       orientation gets its own MAXTRIES scan: a helicity can vanish in one
C       orientation only (e.g. q q~ > z with the Z at rest), and in vector mode
C       the first MAXTRIES events can all share one orientation.
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC, MAXSPROC)
	    INTEGER NTRY(2, MAXFLAVPERPROC, MAXSPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL

C  
C LOCAL VARIABLES 
C
        INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
        INCLUDE 'config_subproc_map.inc'
        INTEGER NHEL(NEXTERNAL,NCOMB)
        INTEGER ISHEL
        REAL*8 T,MATRIX%(proc_id)s
        REAL*8 R,SUMHEL,TS(NCOMB)
        INTEGER I,IDEN
        INTEGER FLAVOR(NEXTERNAL)
        INTEGER FLAVOR_FOR_SYM(NEXTERNAL)
C       Per-row FLAVOR lookup used by BROKEN_SYM. The IFLAV-indexed FLAVOR
C       table above can collapse distinct same-flavor / different-flavor
C       leshouche rows that share the same coupling group; BROKEN_SYM needs
C       row-level information to apply the identical-particle correction.
        INTEGER FLAVOR_ROW(NEXTERNAL,%(max_flavor_row)s)
        %(get_flavor_row_matrix)s
        INTEGER IPSEL
        COMMON /SUBPROC/ IPSEL
        REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
        INTEGER NGOOD
        INTEGER J, JJ
	INTEGER NB_FAIL
	SAVE NB_FAIL
	DATA NB_FAIL /0/
        double precision get_channel_cut
        external get_channel_cut
C
C BOUND STATES: projectors and amplitudes are cached over the helicities
C of one phase-space point
C
      INCLUDE 'onia.inc'
      LOGICAL IS_PROJ_COMPUTED%(proc_id)s(NONIA,-1:1)
      COMMON/IS_PROJ_COMPUTED_%(proc_id)s/IS_PROJ_COMPUTED%(proc_id)s
      LOGICAL IS_AMP_COMPUTED%(proc_id)s(4**NONIA,3**(NEXTERNAL-NONIA))
      COMMON/IS_AMP_COMPUTED_%(proc_id)s/IS_AMP_COMPUTED%(proc_id)s
	
c
C       This is just to temporarily store the reference grid for helicity of the DiscreteSampler so as to obtain its number of entries with ref_helicity_grid%n_tot_entries
	    type(SampledDimension) ref_helicity_grid
C  
C GLOBAL VARIABLES
C  
    logical init_mode
    common /to_determine_zero_hel/init_mode
    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
   

    INTEGER NB_SPIN_STATE_in(2)
    common /nb_hel_state/ nb_spin_state_in
    
    INTEGER IMIRROR, IPROC
    COMMON/TO_MIRROR/ IMIRROR,IPROC
    INTEGER IM

c       included vi run.inc    
c       double precision tmin_for_channel	
c       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c      	      	      	   2 means approximation by the	denominator of the propagator
c       common/TO_CHANNEL_STRAT/tmin_for_channel,	sde_strat

    REAL*8 POL(2)
    COMMON/TO_POLARIZATION/ POL

    INTEGER          ISUM_HEL
    LOGICAL                    MULTI_CHANNEL
    COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
%(define_iconfigs_lines)s
    DATA XTRY, XREJ /0,0/
    DATA NGOOD /0/
    DATA ISHEL/0/
    SAVE YFRAC
%(helicity_lines)s
%(den_factor_line)s

C  To be able to control when the matrix<i> subroutine can add entries to the grid for the MC over helicity configuration
   LOGICAL ALLOW_HELICITY_GRID_ENTRIES
   COMMON/TO_ALLOW_HELICITY_GRID_ENTRIES/ALLOW_HELICITY_GRID_ENTRIES
C
C  FUNCTION
C
   INTEGER BROKEN_SYM%(proc_id)s
C ----------
C BEGIN CODE
C ----------
    CALL GET_FLAVOR%(proc_id)s(IFLAV, FLAVOR)
    IS_PROJ_COMPUTED%(proc_id)s=.FALSE.
    IS_AMP_COMPUTED%(proc_id)s=.FALSE.
C   beam orientation of this event (1 if the caller did not set TO_MIRROR)
    IM = IMIRROR
    IF (IM.NE.2) IM = 1
    NTRY(IM,IFLAV,%(proc_id)s)=NTRY(IM,IFLAV,%(proc_id)s)+1
     
    IF (multi_channel) THEN
        DO I=1,NDIAGS
            AMP2(I)=0D0
        ENDDO
    ENDIF
        JAMP2(0)=%(ncolor_flow)d
        DO I=1,INT(JAMP2(0))
            JAMP2(I)=0D0
        ENDDO
    ANS = 0D0
    DO I=1,NCOMB
       TS(I)=0d0
    ENDDO

!   If the helicity grid status is 0, this means that it is not yet initialized.
!   If HEL_PICKED==-1, this means that calls to other matrix<i> where in initialization mode as well for the helicity.
    IF ((ISHEL.EQ.0.and.ISUM_HEL.eq.0).or.(DS_get_dim_status('Helicity').eq.0).or.(HEL_PICKED.eq.-1)) THEN
      DO I=1,NCOMB
         IF (GOODHEL(I,IFLAV,%(proc_id)s) .OR. NTRY(IM,IFLAV,%(proc_id)s).LE.MAXTRIES.or.(ISUM_HEL.NE.0)) THEN
            T=MATRIX%(proc_id)s(P ,NHEL(1,I),IFLAV,I,AMP2, JAMP2, IVEC)
            %(beam_polarization)s
			IF (ISUM_HEL.NE.0.and.DS_get_dim_status('Helicity').eq.0.and.ALLOW_HELICITY_GRID_ENTRIES) then
			  call DS_add_entry('Helicity',I,T)
			endif
            ANS=ANS+DABS(T)
            TS(I)=T
         ENDIF
      ENDDO
      IF(NTRY(IM,IFLAV,%(proc_id)s).EQ.(MAXTRIES+1).and.DS_get_dim_status('Helicity').ne.-1) THEN
         call reset_cumulative_variable() ! avoid biais of the initialization
      ENDIF
      IF (ISUM_HEL.NE.0) then
!         We set HEL_PICKED to -1 here so that later on, the call to DS_add_point in dsample.f does not add anything to the grid since it was already done here.
		  HEL_PICKED = -1
!         For safety, hardset the helicity sampling jacobian to 0.0d0 to make sure it is not .
		  hel_jacobian   = 1.0d0
!         We don't want to re-update the helicity grid if it was already updated by another matrix<i>, so we make sure that the reference grid is empty.
          ref_helicity_grid = DS_get_dimension(ref_grid,'Helicity')
		  IF((DS_get_dim_status('Helicity').eq.1).and.(ref_helicity_grid%n_tot_entries.eq.0)) then 
!           If we finished the initialization we can update the grid so as to start sampling over it.
!           However the grid will now be filled by dsample with different kind of weights (including pdf, flux, etc...) so by setting the grid_mode of the reference grid to 'initialization' we make sure it will be overwritten (as opposed to 'combined') by the running grid at the next update.
            CALL DS_UPDATE_GRID('Helicity')
			CALL DS_SET_GRID_MODE('Helicity','init')
          endif
	  ELSE
        IF(NTRY(IM,IFLAV,%(proc_id)s).LE.MAXTRIES)THEN
           DO I=1,NCOMB
	      IF(init_mode) THEN
                  IF (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB) THEN
                     PRINT *, 'Matrix Element/Good Helicity: %(proc_id)s ', i, 'IMIRROR', IMIRROR
	          ENDIF
	      ELSE IF (.NOT.GOODHEL(I,IFLAV,%(proc_id)s) .AND. (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB)) THEN
                 GOODHEL(I,IFLAV,%(proc_id)s)=.TRUE.
                 NGOOD = NGOOD +1
                 PRINT *,'Added good helicity ',I, 'for process %(proc_id)s flavor ',IFLAV,' mirror ',IM,TS(I)*NCOMB/ANS,' in event ',NTRY(IM,IFLAV,%(proc_id)s)
              ENDIF
           ENDDO
		endif
        IF(NTRY(IM,IFLAV,%(proc_id)s).EQ.MAXTRIES)THEN
           ISHEL=MIN(ISUM_HEL,NGOOD)
        ENDIF
      ENDIF
    ELSE IF (.not.init_mode) then            ! random helicity 
C           The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
            I = HEL_PICKED

            T=MATRIX%(proc_id)s(P ,NHEL(1,I),IFLAV,I,AMP2, JAMP2, IVEC)

            %(beam_polarization)s
c           Always one helicity at a time
            ANS = T
c           Include the Jacobian from helicity sampling
            ANS = ANS * hel_jacobian
	    IHEL = HEL_PICKED
    ELSE
	ANS = 1D0
	RETURN
    ENDIF
    IF (ANS.ne.0d0.and.(ISUM_HEL .NE. 1.or.HEL_PICKED.eq.-1)) THEN
c    CALL RANMAR(R) ! rhel passed as input
    SUMHEL=0d0
    DO I=1,NCOMB
       SUMHEL=SUMHEL+DABS(TS(I))/ANS
       IF(RHEL.LT.SUMHEL)THEN
          IHEL = I
c         Set right sign for ANS, based on sign of chosen helicity
          ANS=DSIGN(ANS,TS(I))
          GOTO 10
       ENDIF
    ENDDO
 10 CONTINUE   
    ENDIF
    IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO I=1,LMAXCONFIGS
	    J = CONFSUB(%(proc_id)s, I)
	    if (J.ne.0) then
	    if(sde_strat.eq.1) then
	         AMP2(J) = AMP2(J) * GET_CHANNEL_CUT(P, I)
                 XTOT=XTOT+AMP2(J)
	    else
	    	 AMP2(J) = GET_CHANNEL_CUT(P, I)
                 XTOT=XTOT+AMP2(J)
	    endif	
            endif	
        ENDDO
        IF (XTOT.NE.0D0) THEN
		ANS=ANS*AMP2(channel)/XTOT
        ELSE IF(ANS.ne.0d0) THEN
	     IF(NB_FAIL.ge.10)then
			write(*,*) "Problem in the multi-channeling. All amp2 are zero but not the total matrix-element"
			
	        stop 1
            else
	    nb_fail = nb_fail +1
	    endif
        ENDIF
    ENDIF
    IF (IPSEL.GE.1.AND.IPSEL.LE.%(max_flavor_row)s) THEN
       FLAVOR_FOR_SYM(:) = FLAVOR_ROW(:, IPSEL)
    ELSE
       FLAVOR_FOR_SYM(:) = FLAVOR(:)
    ENDIF
    ANS=ANS/DBLE(IDEN)*BROKEN_SYM%(proc_id)s(FLAVOR_FOR_SYM)

    call select_color(rcol, jamp2, iconfig,%(proc_id)s,  icol, ivec)

    END


      SUBROUTINE GET_FLAVOR%(proc_id)s(IFLAV, FLAVOR_OUT)
C     Returns the flavor array for a given flavor index IFLAV
      IMPLICIT NONE
      include 'nexternal.inc'
      INTEGER IFLAV, I
      INTEGER FLAVOR_OUT(NEXTERNAL)
      INTEGER FLAVOR(NEXTERNAL,%(max_flavor)s)
      %(get_flavor_matrix)s
      FLAVOR_OUT(:) = FLAVOR(:, IFLAV)
      RETURN
      END


REAL*8 FUNCTION MATRIX%(proc_id)s(P,NHEL,IFLAV, IHEL,AMP2, JAMP2, IVEC)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
    use aloha_object
    use model_object
    IMPLICIT NONE
C  
C CONSTANTS
C  
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d)
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=%(ncomb)s)
    include 'genps.inc'
    include 'nexternal.inc'
    include 'maxamps.inc'
    INTEGER    NWAVEFUNCS,     NCOLOR
    PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
    INTEGER    NCOLOR_FLOW
    PARAMETER (NCOLOR_FLOW=%(ncolor_flow)d)
    REAL*8     ZERO
    PARAMETER (ZERO=0D0)
    COMPLEX*16 IMAG1
    PARAMETER (IMAG1=(0D0,1D0))
    INTEGER NAMPSO, NSQAMPSO
    PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
	LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
	DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
	SAVE CHOSEN_SO_CONFIGS
    double precision bwcutoff
    common/to_bwcutoff/ bwcutoff
C  
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL)
    INTEGER IFLAV
    INTEGER NHEL(NEXTERNAL), FLAVOR(NEXTERNAL)
    INTEGER IHEL
    INTEGER IVEC
C  
C LOCAL VARIABLES 
C  
    INTEGER I,J,K,M,N,QQ,Q1,Q2
    REAL*8 FRAC
    COMPLEX*16 ZTEMP
%(jamp_tmp_decl)s
%(jamp_decl)s
    INTEGER CF(NCOLOR*(NCOLOR+1)/2)
    INTEGER DENOM, CF_INDEX
    COMMON /%(proc_prefix)scolor_matrix%(proc_id)s/ CF,DENOM
    COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
%(jampflow_decl)s
    type(aloha) W(NWAVEFUNCS)
C   Needed for v4 models
    COMPLEX*16 DUM0,DUM1
    DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
%(flavor_mask_decl)s
C
C BOUND STATES
C
      INCLUDE 'onia.inc'
      LOGICAL RESH_FAILED
      COMMON/TO_RESH_FAILED/RESH_FAILED
      INCLUDE 'ldme.inc'
      REAL*8 NORM, LDME, NPOL
      REAL*8 P_ONIA(0:3,NEXTERNAL+NONIA)
      REAL*8 P_RESH(0:3,NEXTERNAL+NONIA), PMASS_RESH(NEXTERNAL)
      COMPLEX*16 PROJ(4**NONIA)
      INTEGER NHEL_ONIA(NEXTERNAL+NONIA), FLAVOR_ONIA(NEXTERNAL+NONIA)
      INTEGER HELQ1(2), HELQ2(2)
      DATA HELQ1/-1, 1/
      DATA HELQ2/-1, 1/
      LOGICAL IS_PROJ_COMPUTED%(proc_id)s(NONIA,-1:1)
      COMPLEX*16 PROJ_SAVED%(proc_id)s(NONIA,-1:1,-1:1,-1:1)
      COMMON/IS_PROJ_COMPUTED_%(proc_id)s/IS_PROJ_COMPUTED%(proc_id)s
      COMMON/PROJ_SAVED_%(proc_id)s/PROJ_SAVED%(proc_id)s
      LOGICAL IS_AMP_COMPUTED%(proc_id)s(4**NONIA,3**(NEXTERNAL-NONIA))
      COMPLEX*16 AMP_SAVED%(proc_id)s(NGRAPHS,4**NONIA,3**(NEXTERNAL-NONIA))
      COMMON/IS_AMP_COMPUTED_%(proc_id)s/IS_AMP_COMPUTED%(proc_id)s
      COMMON/AMP_SAVED_%(proc_id)s/AMP_SAVED%(proc_id)s
      REAL*8 PMASS(NEXTERNAL)
      REAL*8 PMASS_ONIA(NEXTERNAL+NONIA)

	%(fake_width_declaration)s
	logical first
	data first /.true./
	save first
C
C FUNCTION
C
      INTEGER SQSOINDEX%(proc_id)s
C  
C GLOBAL VARIABLES
C
   include '../../Source/vector.inc' ! defines VECSIZE_MEMMAX
    Double Precision amp2(maxamps), jamp2(0:maxflow)
    include 'coupl.inc' ! needs VECSIZE_MEMMAX (defined in vector.inc)

	double precision small_width_treatment
	common/narrow_width/small_width_treatment

	logical init_mode
	common/to_determine_zero_hel/init_mode

	logical zeroamp_%(proc_id)s(NCOMB,NGRAPHS)
	common/to_zeroamp_%(proc_id)s/zeroamp_%(proc_id)s
	double precision ampmax_zero
c	A cancelling amplitude is only zero to the rounding of the
c	cancellation that produced it, so zeroamp has to be decided on a
c	relative scale.  Exact zeros come from the diagram structure;
c	an axial gauge reference vector instead makes whole diagrams
c	cancel to about 1d-18 of the largest amplitude.
	double precision zeroamp_tol
	parameter (zeroamp_tol=1d-12)

       double precision tmin_for_channel
       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c      	      	      	   2 means approximation by the	denominator of the propagator
       common/TO_CHANNEL_STRAT/tmin_for_channel,	sde_strat
       
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
if (first) then
	first=.false.
    CALL %(proc_prefix)sINIT_CF%(proc_id)s()
	%(fake_width_definitions)s

	if(init_mode) then
	   zeroamp_%(proc_id)s(:,:) = .true.
	endif
endif

C   Rebuild the FLAVOR(NEXTERNAL) array from the threaded flavor index.
    CALL GET_FLAVOR%(proc_id)s(IFLAV, FLAVOR)
%(flavor_mask_setup)s
      INCLUDE 'pmass.inc'
      INCLUDE 'pmassonia.inc'
C     A bound state is put on the sum of the masses of its constituents
      PMASS_RESH = 0.D0
      DO I = 1, NEXTERNAL+NONIA
        PMASS_RESH(MAPPING(I)) = PMASS_RESH(MAPPING(I))+PMASS_ONIA(I)
      ENDDO
      CALL RESHUFFLE_MOMENTA%(proc_id)s(P,PMASS_RESH,P_RESH)
      IF (RESH_FAILED) THEN
        MATRIX%(proc_id)s = 0D0
        RETURN
      ENDIF
C     A bound state (almost) at rest is put exactly at rest: with a tiny
C     but non-zero momentum (e.g. the 1d-14 that genps.f gives a decaying
C     particle) the helicity spinors of its constituents depend on a
C     direction defined by rounding, and the derivative of a P wave
C     loses all precision (terms in 1/|p| cancel).
      DO I = 1, NONIA
        QQ = ONIA(I)
        IF (P_RESH(1,QQ)**2+P_RESH(2,QQ)**2+P_RESH(3,QQ)**2.LT.
     &      (1D-8*P_RESH(0,QQ))**2) THEN
          P_RESH(1:3,QQ) = 0D0
        ENDIF
      ENDDO

C     The constituents share the bound-state momentum by their masses
      DO I = 1, NEXTERNAL+NONIA
        IF (PAIRS(I).NE.0) THEN
          FRAC = PMASS_ONIA(I)/PMASS_RESH(MAPPING(I))
        ELSE
          FRAC = 1.D0
        ENDIF
        P_ONIA(:,I) = FRAC*P_RESH(:,MAPPING(I))
        NHEL_ONIA(I) = NHEL(MAPPING(I))
        FLAVOR_ONIA(I) = FLAVOR(MAPPING(I))
      ENDDO

      NORM = 1D0
      LDME = 1D0
      DO I = 1, NONIA
        QQ = ONIA(I)
        Q1 = FINDLOC(PAIRS, I, DIM=1)
        Q2 = FINDLOC(PAIRS,-I, DIM=1)
C       Computing the projectors
        IF (.NOT.IS_PROJ_COMPUTED%(proc_id)s(I,NHEL(QQ))) THEN
          DO J = 1, SIZE(HELQ1)
            DO K = 1, SIZE(HELQ2)
              NHEL_ONIA(Q1) = HELQ1(J)
              NHEL_ONIA(Q2) = HELQ2(K)
              IF (ONIA_IN(I).EQ.1) THEN
                CALL ONIA_PROJ_IN(P_ONIA(:,Q1),PMASS_ONIA(Q1),NHEL_ONIA(Q1),P_ONIA(:,Q2),PMASS_ONIA(Q2),NHEL_ONIA(Q2),P_RESH(:,QQ),PMASS_RESH(QQ),NHEL(QQ),S_ONIA(I),PROJ(1))
              ELSE
                CALL ONIA_PROJ(P_ONIA(:,Q1),PMASS_ONIA(Q1),NHEL_ONIA(Q1),P_ONIA(:,Q2),PMASS_ONIA(Q2),NHEL_ONIA(Q2),P_RESH(:,QQ),PMASS_RESH(QQ),NHEL(QQ),S_ONIA(I),PROJ(1))
              ENDIF
              PROJ_SAVED%(proc_id)s(I,NHEL(QQ),NHEL_ONIA(Q1),NHEL_ONIA(Q2)) = PROJ(1)
            ENDDO
          ENDDO
          IS_PROJ_COMPUTED%(proc_id)s(I,NHEL(QQ)) = .TRUE.
        ENDIF
C       Computing normalisation factor from color projector
        IF (C_ONIA(I).EQ.1) THEN
          NORM = NORM/3D0
        ELSE IF (C_ONIA(I).EQ.8) THEN
          NORM = NORM*2D0
        ENDIF
C       Computing normalisation factor for LDME convention
        LDME = LDME*PMASS_RESH(QQ)/(2D0*PMASS_ONIA(Q1)*PMASS_ONIA(Q2))
        IF (ONIA_IN(I).EQ.1) THEN
C         A decaying onium: the product of LDMEs below holds its decay
C         LDME <O_H> (Block ldme_decay for a quarkonium, the Coulomb
C         <O^H>/(2J_H+1) for a leptonium); the average over the 2J+1
C         states of the Fock state is in IDEN
          NPOL = 1D0
        ELSE
          NPOL = 2D0*J_ONIA(I)+1D0
        ENDIF
        IF (C_ONIA(I).EQ.1) THEN
          LDME = LDME/6D0/NPOL
        ELSE IF (C_ONIA(I).EQ.8) THEN
          LDME = LDME/8D0/NPOL
        ENDIF
      ENDDO

C     Computing perturbative LDMEs
%(ldme_perturbative)s
C     Computing product of LDMEs
      LDME = LDME*%(ldme_product)s

C     Index of the helicity configuration of the elementary particles
      K = 1
      J = 0
      DO I = 1, NEXTERNAL+NONIA
        IF (PAIRS(I).EQ.0) THEN
          K = K + 3**J*(NHEL_ONIA(I)+1)
          J = J+1
        ENDIF
      ENDDO
C     Amplitudes for every helicity of the constituents
      DO J = 1, 4**NONIA
        PROJ(J) = SQRT(NORM)
        DO I =1, NONIA
          QQ = ONIA(I)
          Q1 = FINDLOC(PAIRS, I, DIM=1)
          Q2 = FINDLOC(PAIRS,-I, DIM=1)
          NHEL_ONIA(Q1) = HELQ1(IBITS(J-1,2*I-2,1)+1)
          NHEL_ONIA(Q2) = HELQ2(IBITS(J-1,2*I-1,1)+1)
          PROJ(J) = PROJ(J) * PROJ_SAVED%(proc_id)s(I,NHEL(QQ),NHEL_ONIA(Q1),NHEL_ONIA(Q2))
        ENDDO
        IF (PROJ(J).NE.(0D0,0D0)) THEN
          IF (.NOT.IS_AMP_COMPUTED%(proc_id)s(J,K)) THEN
            AMP(:) = (0d0,0d0)
%(helas_calls)s
C           Store amplitudes for reuse
            AMP_SAVED%(proc_id)s(:,J,K) = AMP
            IS_AMP_COMPUTED%(proc_id)s(J,K) = .TRUE.
          ENDIF
        ENDIF
      ENDDO

C     The amplitudes of the bound state
      AMP = (0D0,0D0)
      DO I = 1, 4**NONIA
        AMP = AMP + AMP_SAVED%(proc_id)s(:,I,K)*PROJ(I)
      ENDDO


JAMP(:,:) = (0d0,0d0) 
%(jamp_lines)s
%(jampflow_lines)s

    if(init_mode)then
      ampmax_zero = 0d0
      DO I=1, NGRAPHS
        ampmax_zero = max(ampmax_zero, abs(AMP(I)))
      ENDDO
      DO I=1, NGRAPHS	
        if (abs(AMP(I)).gt.zeroamp_tol*ampmax_zero) then
       	  zeroamp_%(proc_id)s(ihel,i) = .false.
       endif
      ENDDO
    endif	

    MATRIX%(proc_id)s = 0.D0 
	DO M = 1, NAMPSO
      CF_INDEX = 0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX + 1
          ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
        ENDDO
		DO N = 1, NAMPSO
		  %(select_configs_if)s
             MATRIX%(proc_id)s = MATRIX%(proc_id)s + ZTEMP*DCONJG(JAMP(I,N))*LDME
	      %(select_configs_endif)s
		ENDDO
      ENDDO
	ENDDO
    MATRIX%(proc_id)s = MATRIX%(proc_id)s/DENOM
    
    if(sde_strat.eq.1)then
%(amp2_lines)s
    endif 

    Do I = 1, NCOLOR_FLOW
	  DO M = 1, NAMPSO	
		DO N = 1, NAMPSO
		  %(select_configs_if)s
            Jamp2(i)=Jamp2(i)+DABS(DBLE(%(jamp_flow)s(i,m)*dconjg(%(jamp_flow)s(i,n))))
		  %(select_configs_endif)s
		enddo
	  enddo
    Enddo

    END

    SUBROUTINE PRINT_ZERO_AMP_%(proc_id)s()

    IMPLICIT NONE
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d)

    INTEGER    NCOMB
    PARAMETER (NCOMB=%(ncomb)d)

    logical zeroamp_%(proc_id)s(NCOMB, NGRAPHS)
    common/to_zeroamp_%(proc_id)s/zeroamp_%(proc_id)s

    integer i,j
    logical all_false

    do i=1, NGRAPHS
       all_false = .true.
       do j=1,NCOMB
       	  if (.not.zeroamp_%(proc_id)s(j, i)) then
	     all_false = .false.
	     exit
	  endif
       enddo	  
       if (all_false) then
       	  write(*,*) 'Amplitude/ZEROAMP:', %(proc_id)s, i
       else
          do j=1,NCOMB
               if (zeroamp_%(proc_id)s(j, i)) then
	         write(*,*) 'HEL/ZEROAMP:', %(proc_id)s, j  , i
	      endif	    
           enddo	
       endif
    enddo   

    return
    end
    

%(color_init_routine)s

    %(broken_sym_function)s

      SUBROUTINE RESHUFFLE_MOMENTA%(proc_id)s(P,PMASS,P_RESH)
%(onia_reshuffle_body)s
