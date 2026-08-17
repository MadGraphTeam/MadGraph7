SUBROUTINE SMATRIX%(proc_id)s(P, iflav, RHEL, RCOL, DUM, IVEC, ANS, SELECTED_HEL, SELECTED_COL)
C  
%(info_lines)s
C 
C MadGraph5_aMC@NLO for Madevent Version
C 
C Returns amplitude squared summed/avg over colors
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
    Include '../../Source/vector.inc' 
    Include 'genps.inc'
    Include 'maxconfigs.inc'
    Include 'nexternal.inc'
    Include 'maxamps.inc'
    include 'run.inc'
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=%(ncomb)d)
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d) 
    INTEGER    NDIAGS
    PARAMETER (NDIAGS=%(ndiags)d) 
    INTEGER    THEL
    PARAMETER (THEL=NCOMB)
C 
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL),ANS
    REAL*8 RHEL, RCOL
    INTEGER DUM! channel information
    integer SELECTED_HEL, SELECTED_COL, IVEC
    integer iflav
c
c global (due to reading writting)
c 
C   Good-helicity filter is per (helicity, flavor): a helicity that is zero for
C   one merged flavor but non-zero for another must not be dropped, so
C   GOODHEL/NTRY carry an extra MAXFLAVPERPROC (IFLAV) dimension.
    LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC)
    INTEGER NTRY(MAXFLAVPERPROC)
    common/BLOCK_GOODHEL/NTRY,GOODHEL
    integer nb_spin_state(2)
    data  nb_spin_state /%(nb_spin_state1)i,%(nb_spin_state2)i/
    common /nb_hel_state/ nb_spin_state
C  
C LOCAL VARIABLES 
C  
    INTEGER NHEL(NEXTERNAL,NCOMB)
    REAL*8 T,MATRIX%(proc_id)s
    REAL*8 R,SUMHEL,TS(NCOMB)
    INTEGER I,IDEN
    INTEGER IPROC,JC(NEXTERNAL),II
    REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
    INTEGER IDUM, NGOOD, J, JJ
    REAL     XRAN1
    EXTERNAL XRAN1
C   C-parity helicity de-duplication of the init full-sum loop (see below):
C   FLIP(I) is the helicity row with every helicity negated (an involution
C   built once). The de-duplication is ALL-OR-NOTHING per flavor: CSYMBAD(IFLAV)
C   latches 1 as soon as ANY row fails to pair up (a self-paired row, FLIP(I)=I,
C   has no distinct partner) or ANY pair shows |M(I)|^2 != |M(FLIP(I))|^2 at a
C   scan point. Only when every row is in a genuine matched pair is the reuse
C   enabled, and then exactly half the rows are evaluated and each counted twice.
C   That uniformity is what makes the reuse safe for the multi-channel weight:
C   AMP2(J) is accumulated inside MATRIX over this same loop, so a partial
C   (per-row) de-duplication would rescale AMP2 non-uniformly across diagrams and
C   bias AMP2(config)/XTOT even though the total |M|^2 stays exact. Halving every
C   row scales AMP2(J) and XTOT by the same 1/2, leaving the ratio untouched.
C   NCSCAN counts only the passes that actually ran the scan (NTRY alone is not
C   enough: read_good_hel restores NTRY above the threshold without ever
C   scanning, and the random-helicity branch bumps NTRY without a full sum).
C   CSYMBAD/NCSCAN live in a COMMON block (zero-initialised, like NTRY above) so
C   write_good_hel/read_good_hel can persist the verdict across jobs.
    INTEGER FLIP(NCOMB)
    REAL*8 TSMAX
    LOGICAL DEDUP
    INTEGER CSYMBAD(MAXFLAVPERPROC), NCSCAN(MAXFLAVPERPROC)
    COMMON/BLOCK_CSYM/CSYMBAD,NCSCAN
%(flip_data)s
    INTEGER FLAVOR(NEXTERNAL)
    INTEGER FLAVOR_FOR_SYM(NEXTERNAL)
C   Per-row FLAVOR lookup used by BROKEN_SYM. The IFLAV-indexed FLAVOR
C   table above can collapse distinct same-flavor / different-flavor
C   leshouche rows that share the same coupling group; BROKEN_SYM needs
C   row-level information to apply the identical-particle correction.
    INTEGER FLAVOR_ROW(NEXTERNAL,%(max_flavor_row)s)
    %(get_flavor_row_matrix)s
    INTEGER IPSEL
    COMMON /SUBPROC/ IPSEL
    INTEGER BROKEN_SYM%(proc_id)s%(blas_smatrix_decl)s

C
C GLOBAL VARIABLES
C  
    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
    COMMON/TO_AMPS/  AMP2,       JAMP2
    
c    CHARACTER*101        HEL_BUFF
c    COMMON/TO_HELICITY/  HEL_BUFF
    
    REAL*8 POL(2)
    COMMON/TO_POLARIZATION/ POL
    
    INTEGER          ISUM_HEL
    LOGICAL                    MULTI_CHANNEL
    COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
%(define_iconfigs_lines)s
    DATA IDUM /-1/
    DATA XTRY, XREJ, NGOOD /0,0,0/
    SAVE YFRAC

%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
    CALL GET_FLAVOR%(proc_id)s(IFLAV, FLAVOR)
    NTRY(IFLAV)=NTRY(IFLAV)+1
C   C-parity partner of each helicity row (all helicities negated): an
C   involution, self-paired when every helicity is 0. Fixed per process, so
C   built once (FLIP starts at 0).
      ENDDO
    ENDIF
    DO I=1,NEXTERNAL
       JC(I) = +1
    ENDDO
     
    IF (multi_channel) THEN
        DO I=1,NDIAGS
            AMP2(I)=0D0
        ENDDO
    ENDIF
        JAMP2(0)=%(ncolor)d
        DO I=1,INT(JAMP2(0))
            JAMP2(I)=0D0
        ENDDO
    ANS = 0D0
c    WRITE(HEL_BUFF,'(20I5)') (0,I=1,NEXTERNAL)
    DO I=1,NCOMB
       TS(I)=0d0
    ENDDO

!   If the helicity grid status is 0, this means that it is not yet initialized.
!   C-parity de-duplication of this init full-sum loop is all-or-nothing per
!   flavor and only kicks in once this flavor has actually completed the scan
!   (NCSCAN>=20; the scan passes stay within MAXTRIES=25 so the reuse overlaps
!   the grid build). A crossing is a separate subprocess directory in madevent,
!   so every IFLAV row here is the uncrossed base process; polarized beams are
!   self-excluded because the beam_polarization scaling makes the flipped
!   partner's |M|^2 differ, so the scan latches CSYMBAD.
    DEDUP = NCSCAN(IFLAV).GE.20 .AND. CSYMBAD(IFLAV).EQ.0
    IF (ISUM_HEL.EQ.0.or.(DS_get_dim_status('Helicity').eq.0)) THEN
%(blas_branch)s        DO I=1,NCOMB
           IF (GOODHEL(I,IFLAV) .OR. NTRY(IFLAV) .LE. MAXTRIES.OR.(ISUM_HEL.NE.0)) THEN
C            Fast phase: every row is in a matched pair (all-or-nothing), so the
C            higher-index partner is skipped and the lower index counted twice.
             IF (DEDUP.AND.I.GT.FLIP(I)) CYCLE
               T=%(blas_matrix_call)s
            %(beam_polarization)s
             IF (ISUM_HEL.NE.0) then
               call DS_add_entry('Helicity',I,T)
             endif
             ANS=ANS+DABS(T)
             TS(I)=T
C            The representative carries its skipped partner's identical
C            contribution: copy |M|^2 to TS(FLIP) (so the per-helicity CDF and
C            the DS grid pick both members), and count it once more in ANS.
             IF (DEDUP.AND.I.LT.FLIP(I)) THEN
               ANS=ANS+DABS(T)
               TS(FLIP(I))=T
               IF (ISUM_HEL.NE.0) call DS_add_entry('Helicity',FLIP(I),T)
             ENDIF
           ENDIF
        ENDDO
C       Scan phase: this pass evaluated every row, so it can test the pairing.
C       A single mismatching pair (parity/C/polarization breaking) permanently
C       disables the de-duplication for the whole flavor.
C       A pair of rows that are BOTH numerically zero differ only by roundoff,
C       and a purely RELATIVE test on that noise fails at random -- one such
C       pair then vetoes every real pair, because the verdict is all-or-nothing.
C       Measured on g g > t t~: rows 1/16 and 4/13 sit at |M|^2 ~ 1e-30 while
C       the scan maximum is ~1.5e+02, and the de-duplication never engaged.
C       So also require the difference to be significant against TSMAX, the
C       largest |M|^2 of this scan point: a row that far down cannot bias the
C       helicity sum whichever way it is paired, while a genuine parity
C       violation still shows up at the relative level.
        IF (NCSCAN(IFLAV).LT.20) THEN
          NCSCAN(IFLAV)=NCSCAN(IFLAV)+1
          TSMAX=0D0
          DO I=1,NCOMB
            IF (DABS(TS(I)).GT.TSMAX) TSMAX=DABS(TS(I))
          ENDDO
          DO I=1,NCOMB
            IF (FLIP(I).GT.I) THEN
              IF (DABS(TS(I)-TS(FLIP(I))).GT.1D-6*(DABS(TS(I))+DABS(TS(FLIP(I))))
     &            .AND.DABS(TS(I)-TS(FLIP(I))).GT.1D-12*TSMAX) THEN
                CSYMBAD(IFLAV)=1
              ENDIF
            ENDIF
          ENDDO
        ENDIF
        IF(NTRY(IFLAV).EQ.(MAXTRIES+1)) THEN
           call reset_cumulative_variable() ! avoid biais of the initialization
        ENDIF
    IF (ISUM_HEL.NE.0) then
!         We set HEL_PICKED to -HEL_PICKED here so that later on, the call to DS_add_point in dsample.f does not add anything to the grid since it was already done here.
          HEL_PICKED = -iabs(HEL_PICKED)
!         For safety, hardset the helicity sampling jacobian to 0.0d0 to make sure it is not .
          hel_jacobian   = 1.0d0
          IF(DS_get_dim_status('Helicity').eq.1) then 
!           If we finished the initialization we can update the grid so as to start sampling over it.
!           However the grid will now be filled by dsample with different kind of weights (including pdf, flux, etc...) so by setting the grid_mode of the reference grid to 'initialization' we make sure it will be overwritten (as opposed to 'combined') by the running grid at the next update.
            CALL DS_UPDATE_GRID('Helicity')
            CALL DS_SET_GRID_MODE('Helicity','init')
          endif
        ELSE
          IF(NTRY(IFLAV).LE.MAXTRIES)THEN
           DO I=1,NCOMB
              IF (.NOT.GOODHEL(I,IFLAV) .AND. (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB)) THEN
                 GOODHEL(I,IFLAV)=.TRUE.
                 NGOOD = NGOOD +1
                 print *,'Adding good helicity ',I,' flavor ',IFLAV,TS(I)/ANS
              ENDIF
           ENDDO
          ENDIF
          IF(NTRY(IFLAV).EQ.MAXTRIES)THEN
             ISUM_HEL=MIN(ISUM_HEL,NGOOD)
          ENDIF
        endif
    ELSE              !RANDOM HELICITY
       
C           The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
            I = HEL_PICKED

            T=MATRIX%(proc_id)s(P ,NHEL(1,I),IFLAV, IVEC%(blas_matrix_args)s)
            %(beam_polarization)s
c           Always one helicity at a time
            ANS = T
c           Include the Jacobian from helicity sampling
            ANS = ANS * hel_jacobian
            SELECTED_HEL = I
c            WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)  
    ENDIF
    IF (ISUM_HEL .NE. 1.or.(HEL_PICKED.le.0)) THEN
    R=XRAN1(IDUM)*ANS
    SUMHEL=0d0
    DO I=1,NCOMB
       SUMHEL=SUMHEL+DABS(TS(I))
       IF(R.LT.SUMHEL)THEN
           SELECTED_HEL = I
c          WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)
          ANS=DSIGN(ANS,TS(I))        
          GOTO 10
       ENDIF
    ENDDO
 10 CONTINUE   
    ENDIF
    IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO I=1,NDIAGS
            XTOT=XTOT+AMP2(I)
        ENDDO
        IF (XTOT.NE.0D0) THEN
%(set_amp2_line)s
        ELSE IF(ANS.ne.0d0) THEN
            write(*,*) "Problem in the multi-channeling. All amp2 are zero but not the total matrix-element"
            stop 1
        ENDIF
    ENDIF
    IF (IPSEL.GE.1.AND.IPSEL.LE.%(max_flavor_row)s) THEN
       FLAVOR_FOR_SYM(:) = FLAVOR_ROW(:, IPSEL)
    ELSE
       FLAVOR_FOR_SYM(:) = FLAVOR(:)
    ENDIF
    ANS=ANS/DBLE(IDEN)*BROKEN_SYM%(proc_id)s(FLAVOR_FOR_SYM)

    CALL SELECT_COLOR(RCOL, JAMP2, ICONFIG, 1,  SELECTED_COL, IVEC)

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


REAL*8 FUNCTION MATRIX%(proc_id)s(P,NHEL,IFLAV, IVEC%(blas_arg)s)
use model_object
use aloha_object
C  
%(info_lines)s
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
    IMPLICIT NONE
C  
C CONSTANTS
C  
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d)
    include '../../Source/vector.inc'
    include 'genps.inc'
    include 'nexternal.inc'
    include 'maxamps.inc'
    INTEGER    NWAVEFUNCS,     NCOLOR, NCOLORFOLD
    PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
    PARAMETER (NCOLORFOLD=%(ncolorfold)d)
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
    INTEGER FLAVOR(NEXTERNAL)
    INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
    INTEGER IVEC
C  
C LOCAL VARIABLES 
C  
    INTEGER I,J,M,N
    COMPLEX*16 ZTEMP
%(jamp_tmp_decl)s
%(jamp_decl)s
    INTEGER CF(NCOLORFOLD*(NCOLORFOLD+1))
    INTEGER CF_INDEX,DENOM
    COMMON /%(proc_prefix)scolor_matrix%(proc_id)s/ CF,DENOM
    COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)%(blas_decl)s
%(color_fold_decl)s
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
    Double Precision amp2(maxamps), jamp2(0:maxflow)
    common/to_amps/  amp2,       jamp2
    include 'coupl.inc'

    double precision small_width_treatment
    common/narrow_width/small_width_treatment
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
endif

C   Rebuild the FLAVOR(NEXTERNAL) array from the threaded flavor index.
    CALL GET_FLAVOR%(proc_id)s(IFLAV, FLAVOR)
%(flavor_mask_setup)s
AMP(:) = (0d0,0d0)
%(helas_calls)s
%(jamp_lines)s

%(color_fold_gather)s%(blas_gather)s
    MATRIX%(proc_id)s = 0.D0 
    DO M = 1, NAMPSO%(blas_gate)s
      CF_INDEX = 0
      DO I = 1, NCOLORFOLD
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLORFOLD
          CF_INDEX = CF_INDEX +1
          ZTEMP = ZTEMP + CF(CF_INDEX)*%(color_fold_array)s(J,M)
        ENDDO
        DO N = 1, NAMPSO
          IF (CHOSEN_SO_CONFIGS(SQSOINDEX%(proc_id)s(M,N))) THEN
             MATRIX%(proc_id)s = MATRIX%(proc_id)s + ZTEMP*DCONJG(%(color_fold_array)s(I,N))
          ENDIF
        ENDDO
      ENDDO
    ENDDO
     MATRIX%(proc_id)s = MATRIX%(proc_id)s / DENOM

%(amp2_lines)s
    Do I = 1, NCOLOR
      DO M = 1, NAMPSO  
        DO N = 1, NAMPSO
          IF (CHOSEN_SO_CONFIGS(SQSOINDEX%(proc_id)s(M,N))) THEN        
            Jamp2(i)=Jamp2(i)+DABS(DBLE(Jamp(i,m)*dconjg(Jamp(i,n))))
          ENDIF
        enddo
      enddo
    Enddo

    END


%(color_init_routine)s%(blas_routine)s

    %(broken_sym_function)s
