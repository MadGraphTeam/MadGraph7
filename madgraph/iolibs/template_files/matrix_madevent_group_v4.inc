SUBROUTINE SMATRIX%(proc_id)s(P, IFLAV, RHEL, RCOL, channel, IVEC, ANS, IHEL, ICOL)
C 
%(info_lines)s
C 
C MadGraph5_aMC@NLO for Madevent Version
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
c global (due to reading writting) 
c
C       Good-helicity filter is per (helicity, flavor, subprocess): a helicity
C       that is zero for one merged flavor but non-zero for another must not be
C       dropped, so GOODHEL/NTRY carry an extra MAXFLAVPERPROC (IFLAV) dimension.
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC, MAXSPROC)
	    INTEGER NTRY(MAXFLAVPERPROC, MAXSPROC)
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
%(smatrix_me_cross_decl)s
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
C       C-parity helicity de-duplication of the init full-sum loop (uncrossed
C       base process only): FLIP(I) is the helicity row with every helicity
C       negated (an involution built once). The reuse is ALL-OR-NOTHING per
C       (flavor,subprocess): CSYMBAD latches 1 as soon as ANY row fails to pair
C       up (a self-paired row, FLIP(I)=I, has no distinct partner) or ANY pair
C       shows |M(I)|^2 != |M(FLIP(I))|^2 at a scan point. Only then is exactly
C       half the loop skipped, each survivor counted twice. That uniformity is
C       what keeps the multi-channel weight exact: halving every row scales
C       AMP2(J) and XTOT by the same 1/2, so AMP2(config)/XTOT is unchanged,
C       whereas a partial (per-row) de-duplication would rescale AMP2
C       non-uniformly across diagrams. NCSCAN counts only the passes that really
C       ran the scan -- NTRY alone is not enough, since read_good_hel restores
C       it above the threshold without ever scanning. CSYMBAD/NCSCAN sit in a
C       COMMON block (zero-initialised, like NTRY) so the good-hel file can
C       carry the verdict across jobs.
        INTEGER FLIP(NCOMB), JHEL, KHEL
        REAL*8 TSMAX
        LOGICAL DEDUP, CSYM_DONE
        INTEGER CSYMBAD(MAXFLAVPERPROC,MAXSPROC)
        INTEGER NCSCAN(MAXFLAVPERPROC,MAXSPROC)
        COMMON/BLOCK_CSYM/CSYMBAD,NCSCAN
        SAVE CSYM_DONE
        DATA CSYM_DONE/.FALSE./
%(flip_data)s

c
C       This is just to temporarily store the reference grid for helicity of the DiscreteSampler so as to obtain its number of entries with ref_helicity_grid%n_tot_entries
	    type(SampledDimension) ref_helicity_grid
C  
C GLOBAL VARIABLES
C  
    logical init_mode
    common /to_determine_zero_hel/init_mode
    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
%(xg_jamp2_decl)s
   

    INTEGER NB_SPIN_STATE_in(2)
    common /nb_hel_state/ nb_spin_state_in
    
    INTEGER IMIRROR, IPROC
    COMMON/TO_MIRROR/ IMIRROR,IPROC

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
   INTEGER BROKEN_SYM%(proc_id)s%(blas_smatrix_decl)s
C ----------
C BEGIN CODE
C ----------
%(smatrix_me_cross_decode)s
    CALL GET_FLAVOR%(proc_id)s(%(me_flav_key)s, FLAVOR)
    NTRY(%(me_flav_key)s,%(proc_id)s)=NTRY(%(me_flav_key)s,%(proc_id)s)+1
    IF (.NOT.CSYM_DONE) THEN
C     A self-paired row (FLIP(I)=I, every helicity 0) has no distinct partner,
C     so the loop could not be halved uniformly: refuse the reuse outright.
      DO I=1,NCOMB
        IF (FLIP(I).EQ.I) THEN
          DO KHEL=1,MAXSPROC
            DO JHEL=1,MAXFLAVPERPROC
              CSYMBAD(JHEL,KHEL)=1
            ENDDO
          ENDDO
        ENDIF
      ENDDO
      CSYM_DONE = .TRUE.
    ENDIF
    DEDUP = NCSCAN(%(me_flav_key)s,%(proc_id)s).GE.20 .AND. CSYMBAD(%(me_flav_key)s,%(proc_id)s).EQ.0 .AND. (%(me_csym_cross_ok)s)

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
%(blas_branch)s      DO I=1,NCOMB
         IF (GOODHEL(%(me_goodhel_idx)s,%(me_flav_key)s,%(proc_id)s) .OR. NTRY(%(me_flav_key)s,%(proc_id)s).LE.MAXTRIES.or.(ISUM_HEL.NE.0)%(smatrix_me_goodhel_or)s) THEN
C           Fast phase: skip the higher-index C-parity partner (computed once at
C           the lower index, counted twice below).
            IF (DEDUP.AND.I.GT.FLIP(I)) CYCLE
            T=%(blas_matrix_call)s
            %(beam_polarization)s
			IF (ISUM_HEL.NE.0.and.DS_get_dim_status('Helicity').eq.0.and.ALLOW_HELICITY_GRID_ENTRIES) then
			  call DS_add_entry('Helicity',I,T)
			endif
            ANS=ANS+DABS(T)
            TS(I)=T
C           The representative carries its skipped partner's identical |M|^2:
C           copy TS(FLIP) (so the event-helicity CDF/DS grid pick both) and
C           count it once more in ANS.
            IF (DEDUP.AND.I.LT.FLIP(I)) THEN
              ANS=ANS+DABS(T)
              TS(FLIP(I))=T
			  IF (ISUM_HEL.NE.0.and.DS_get_dim_status('Helicity').eq.0.and.ALLOW_HELICITY_GRID_ENTRIES)
     &          call DS_add_entry('Helicity',FLIP(I),T)
            ENDIF
         ENDIF
      ENDDO
C     Scan phase: drop the C-parity pairing of any row whose fully flipped
C     partner gave a different |M|^2 (parity/C/polarization breaking). One
C     mismatch at any scan point permanently invalidates the pair.
C       A pair of rows that are BOTH numerically zero differ only by roundoff,
C       and a purely RELATIVE test on that noise fails at random -- one such
C       pair then vetoes every real pair, because the verdict is all-or-nothing.
C       Measured on g g > t t~: rows 1/16 and 4/13 sit at |M|^2 ~ 1e-30 while
C       the scan maximum is ~1.5e+02, and the de-duplication never engaged.
C       So also require the difference to be significant against TSMAX, the
C       largest |M|^2 of this scan point: a row that far down cannot bias the
C       helicity sum whichever way it is paired, while a genuine parity
C       violation still shows up at the relative level.
      IF (%(me_csym_cross_ok)s.AND.NCSCAN(%(me_flav_key)s,%(proc_id)s).LT.20) THEN
        NCSCAN(%(me_flav_key)s,%(proc_id)s)=NCSCAN(%(me_flav_key)s,%(proc_id)s)+1
        TSMAX=0D0
        DO I=1,NCOMB
          IF (DABS(TS(I)).GT.TSMAX) TSMAX=DABS(TS(I))
        ENDDO
        DO I=1,NCOMB
          IF (FLIP(I).GT.I) THEN
            IF (DABS(TS(I)-TS(FLIP(I))).GT.1D-6*(DABS(TS(I))+DABS(TS(FLIP(I))))
     &          .AND.DABS(TS(I)-TS(FLIP(I))).GT.1D-12*TSMAX) THEN
              CSYMBAD(%(me_flav_key)s,%(proc_id)s)=1
            ENDIF
          ENDIF
        ENDDO
      ENDIF
C     Report the surviving C-parity pairs to the helicity-recycling optimizer
C     (parsed by gen_ximprove): once the scan has settled (NTRY=20), each
C     representative I<FLIP(I) with a non-zero |M|^2 whose flipped partner is
C     C-symmetric. The optimizer drops the partner's amplitudes from
C     matrix<i>_optim.f and reuses TS(rep) for it.
C     NTRY (not NCSCAN, which saturates at 20) provides the one-shot trigger;
C     in init_mode the full-sum branch runs every call, so the two coincide.
      IF (init_mode.AND.%(me_csym_cross_ok)s.AND.CSYMBAD(%(me_flav_key)s,%(proc_id)s).EQ.0.AND.NCSCAN(%(me_flav_key)s,%(proc_id)s).GE.20.AND.NTRY(%(me_flav_key)s,%(proc_id)s).EQ.20) THEN
        DO I=1,NCOMB
          IF (I.LT.FLIP(I).AND.DABS(TS(I)).GT.ANS*LIMHEL/NCOMB) THEN
            PRINT *, 'CSYM PAIR: %(proc_id)s ', I, FLIP(I)
          ENDIF
        ENDDO
      ENDIF
      IF(NTRY(%(me_flav_key)s,%(proc_id)s).EQ.(MAXTRIES+1).and.DS_get_dim_status('Helicity').ne.-1) THEN
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
        IF(NTRY(%(me_flav_key)s,%(proc_id)s).LE.MAXTRIES)THEN
           DO I=1,NCOMB
	      IF(init_mode) THEN
                  IF (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB) THEN
                     PRINT *, 'Matrix Element/Good Helicity: %(proc_id)s ', i, 'IMIRROR', IMIRROR
	          ENDIF
	      ELSE IF (%(me_goodhel_train_guard)s.NOT.GOODHEL(%(me_goodhel_idx)s,%(me_flav_key)s,%(proc_id)s) .AND. (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB)) THEN
                 GOODHEL(%(me_goodhel_idx)s,%(me_flav_key)s,%(proc_id)s)=.TRUE.
                 NGOOD = NGOOD +1
                 PRINT *,'Added good helicity ',I, 'for process %(proc_id)s flavor ',IFLAV,TS(I)*NCOMB/ANS,' in event ',NTRY(%(me_flav_key)s,%(proc_id)s)
              ENDIF
           ENDDO
		endif
        IF(NTRY(%(me_flav_key)s,%(proc_id)s).EQ.MAXTRIES)THEN
           ISHEL=MIN(ISUM_HEL,NGOOD)
        ENDIF
      ENDIF
    ELSE IF (.not.init_mode) then            ! random helicity 
C           The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
            I = HEL_PICKED

            T=MATRIX%(proc_id)s(%(me_matrix_args)s%(blas_matrix_args)s)

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
	    J = %(me_confsub_j)s
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
%(smatrix_me_iden_line)s

%(xg_jamp2_pub)s
    call select_color(rcol, jamp2, iconfig,%(proc_id)s,  icol, ivec)

    END
%(crossing_routines_me)s


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


REAL*8 FUNCTION MATRIX%(proc_id)s(P,NHEL,%(me_matrix_ic_param)sIFLAV, IHEL,AMP2, JAMP2, IVEC%(blas_arg)s)
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
    INTEGER    NWAVEFUNCS,     NCOLOR, NCOLORFOLD
    PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
    PARAMETER (NCOLORFOLD=%(ncolorfold)d)
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
%(me_matrix_ic_decl)s
    INTEGER NHEL(NEXTERNAL), FLAVOR(NEXTERNAL)
    INTEGER IHEL
    INTEGER IVEC
C  
C LOCAL VARIABLES 
C  
    INTEGER I,J,M,N
    COMPLEX*16 ZTEMP
%(jamp_tmp_decl)s
%(jamp_decl)s
    INTEGER CF(NCOLORFOLD*(NCOLORFOLD+1)/2)
    INTEGER DENOM, CF_INDEX
    COMMON /%(proc_prefix)scolor_matrix%(proc_id)s/ CF,DENOM
    COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)%(blas_decl)s
%(color_fold_decl)s
%(jampflow_decl)s
    type(aloha) W(NWAVEFUNCS)
C   Needed for v4 models
    COMPLEX*16 DUM0,DUM1
    DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
%(flavor_mask_decl)s

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

       double precision tmin_for_channel
       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c      	      	      	   2 means approximation by the	denominator of the propagator
       common/TO_CHANNEL_STRAT/tmin_for_channel,	sde_strat
       
C  
C COLOR DATA
C  
%(color_data_lines)s
%(color_fold_index)s
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
AMP(:) = (0d0,0d0)
%(helas_calls)s

JAMP(:,:) = (0d0,0d0) 
%(jamp_lines)s
%(jampflow_lines)s

    if(init_mode)then
      DO I=1, NGRAPHS	
        if (AMP(I).ne.0) then
       	  zeroamp_%(proc_id)s(ihel,i) = .false.
       endif
      ENDDO
    endif	

%(color_fold_gather)s%(blas_gather)s
    MATRIX%(proc_id)s = 0.D0 
	DO M = 1, NAMPSO%(blas_gate)s
      CF_INDEX = 0
      DO I = 1, NCOLORFOLD
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLORFOLD
          CF_INDEX = CF_INDEX + 1
          ZTEMP = ZTEMP + CF(CF_INDEX)*%(color_fold_array)s(J,M)
        ENDDO
		DO N = 1, NAMPSO
		  %(select_configs_if)s
             MATRIX%(proc_id)s = MATRIX%(proc_id)s + ZTEMP*DCONJG(%(color_fold_array)s(I,N))
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
    

%(color_init_routine)s%(blas_routine)s

    %(broken_sym_function)s
