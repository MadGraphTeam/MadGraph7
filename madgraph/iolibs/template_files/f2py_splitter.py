C this is a f2py wrapper for reweight mode at tree level

  subroutine %(f2py_prefix)sf77_smatrixhel(pdgs, procid, npdg, p, ALPHAS, SCALE2, nhel, ANS)
  use model_object
  use aloha_object
  IMPLICIT NONE
C ALPHAS is given at scale2 (SHOULD be different of 0 for loop induced, ignore for LO)  

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
  integer pdgs(*)
  integer npdg, nhel, procid
  double precision p(*)
  double precision ANS, ALPHAS, PI,SCALE2
  integer flavor(%(maxpart)i),lpdgs(%(maxpart)i),I
  logical found
%(flavor_index_decl)s
  include 'coupl.inc'
  
  
  if (scale2.eq.0)then
       PI = 3.141592653589793D0
       G = 2* DSQRT(ALPHAS*PI)
       CALL UPDATE_AS_PARAM()
  else
       CALL UPDATE_AS_PARAM2(scale2, ALPHAS)
  endif

%(flavormapping)s

C     FOUND: as in f77_density (f77_crossed_smatrixhel for a
C     folded crossing); a miss is NaN, not a vanishing |M|^2.
  found = .true.
%(smatrixhel)s
  if (.not.found) call %(f2py_prefix)sf77_no_process_ans('smatrixhel', pdgs, npdg, procid, ans)

      return
      end

%(crossed_smatrixhel)s

  subroutine %(f2py_prefix)sf77_smatrixhel_idx(procindex, flav_idx, npdg, p, ALPHAS, SCALE2, nhel, ANS)
  use model_object
  use aloha_object
  IMPLICIT NONE
C Same as f77_smatrixhel, but selecting the matrix element
C by its slot in get_pdg_order/get_prefix (PROCINDEX,
C 1-based) and taking the extended flavor index (FLAV_IDX =
C K*NFLAV + flav) as given rather than resolving it from the
C PDG codes, for a caller that resolved a FOLDED crossed
C subprocess itself (f77_smatrixhel finds it from the PDG
C codes through f77_crossed_smatrixhel; the FLAVOR array
C cannot express a crossing, see
C matrix_standalone_f2py_flav_idx.inc). The alphas/scale2
C setup is deliberately the same as in f77_smatrixhel.
CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in) :: procindex
CF2PY integer, intent(in) :: flav_idx
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
  integer procindex, flav_idx, npdg, nhel
  double precision p(*)
  double precision ANS, ALPHAS, PI, SCALE2
  include 'coupl.inc'

  if (scale2.eq.0)then
       PI = 3.141592653589793D0
       G = 2* DSQRT(ALPHAS*PI)
       CALL UPDATE_AS_PARAM()
  else
       CALL UPDATE_AS_PARAM2(scale2, ALPHAS)
  endif

  ANS = 0d0
%(smatrixhel_idx)s

      return
      end

  subroutine %(f2py_prefix)sf77_density(pdgs, npdg, procid, P, POS, N_CHANGING, ALLOW_HEL, N_COMB, ALPHAS, SCALE2, INTER)
  IMPLICIT NONE
CF2PY double precision, intent(in) :: p
CF2PY integer, intent(in) :: pdgs
CF2PY integer, intent(in) :: procid
CF2PY integer, intent(in) :: pos
CF2PY integer, INTENT(IN) :: ALLOW_HEL
CF2PY double precision INTENT(IN) :: ALPHAS
CF2PY double precision INTENT(IN) :: SCALE2
CF2PY double complex INTENT(OUT), dimension(N_COMB*(N_COMB+1)/2) :: INTER
CF2PY integer, intent(in) :: N_COMB
CF2PY integer, intent(in) :: N_CHANGING
CF2PY integer, intent(in) :: NPDG
C     scale is a dummy argument added to have the same syntax as in loop-induced
C
C     Some variables seem unused but they are necessary for density_splitter
C

  integer pdgs(*), procid, n_changing, n_comb, npdg
  double precision p(0:3,*)
  double precision ALPHAS, SCALE2
  INTEGER POS(n_changing)
  INTEGER ALLOW_HEL(n_changing*n_comb)
  DOUBLE COMPLEX INTER(n_comb*(n_comb+1)/2)
  integer flavor(%(maxpart)i),lpdgs(%(maxpart)i),I
  logical found
%(flavor_index_decl)s
C     Update is done insider the direct density call functions

%(flavormapping)s

C     FOUND: some matrix element (uncrossed, or a folded
C     crossing through f77_crossed_density) evaluates PDGS.
  found = .true.
%(density_splitter)s
  if (.not.found) call %(f2py_prefix)sf77_no_process('py_get_density', pdgs, npdg, procid, inter, n_comb*(n_comb+1)/2)

            return
            end

%(crossed_density)s

  subroutine %(f2py_prefix)sf77_get_all_inter(pdgs, procid, npdg, P, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, INTER)
  IMPLICIT NONE
C     P momenta
C     NHEL(NPDG) helicity of the legs not in POS (entries
C      at POS ignored)
C     POS(N_CHNGING): position of the changing helicity
C     n_changing: number of changing helicity
C     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to
C      consider (all jamp computed)
C     INTER((NCOMB*NCOMB+1)/2: all interference term (not the
C      symmetric one)

  integer pdgs(*)
  integer npdg, procid, n_changing
  integer nhel(*), lnhel(%(maxpart)i)
  integer n_comb
  double precision p(*)
  double precision ANS,  PI,SCALE2
  INTEGER POS(*)
  INTEGER ALLOW_HEL(*)
  DOUBLE COMPLEX INTER(*)
  integer flavor(%(maxpart)i),lpdgs(%(maxpart)i),I
  logical found
%(flavor_index_decl)s
C     Update is done insider the direct density call functions

C     Update is done insider the direct density call functions

%(flavormapping)s

C     GET_ALL_INTER may overwrite NHEL at POS (the split-order
C     one does): hand it a copy, f2py passes the numpy array.
  lnhel(:) = 0
  do i = 1, min(npdg, %(maxpart)i)
     lnhel(i) = nhel(i)
  enddo

C     FOUND: as in f77_density (f77_crossed_get_all_inter
C     for a folded crossing, NHEL in the caller's legs).
  found = .true.
%(inter_splitter)s
  if (.not.found) call %(f2py_prefix)sf77_no_process('py_get_all_inter', pdgs, npdg, procid, inter, n_comb*(n_comb+1)/2)

            return
            end

%(crossed_get_all_inter)s

      SUBROUTINE %(f2py_prefix)sf77_no_process(FCT, PDGS, NPDG, PROCID, INTER, NINTER)
C     No matrix element of this module, uncrossed or folded
C     crossing, evaluates PDGS (for PROCID): say so and
C     answer NaN, never a silent zero a caller could not tell
C     from a vanishing density matrix. Legs must come in an
C     order a matrix element tabulates (get_pdg_order, or for
C     a folded crossing its GET_PDG_FOR_FLAVOR order).
      IMPLICIT NONE
      CHARACTER*(*) FCT
      INTEGER NPDG, PROCID, NINTER, I
      INTEGER PDGS(NPDG)
      DOUBLE COMPLEX INTER(NINTER)
      DOUBLE PRECISION XNAN
      CALL %(f2py_prefix)sF77_NO_PROCESS_ANS(FCT, PDGS, NPDG, PROCID, XNAN)
      DO I = 1, NINTER
        INTER(I) = DCMPLX(XNAN, XNAN)
      ENDDO
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sF77_NO_PROCESS_ANS(FCT, PDGS, NPDG, PROCID, ANS)
C     F77_NO_PROCESS for a single answer (smatrixhel): the
C     ERROR line, and ANS = NaN.
      USE, INTRINSIC :: IEEE_ARITHMETIC
      IMPLICIT NONE
      CHARACTER*(*) FCT
      INTEGER NPDG, PROCID, I
      INTEGER PDGS(NPDG)
      DOUBLE PRECISION ANS
      WRITE(*,*) 'ERROR: ', FCT, ': no matrix element for PDGS', (PDGS(I), I=1,NPDG), ' PROCID', PROCID, ': answer set to NaN'
      ANS = IEEE_VALUE(ANS, IEEE_QUIET_NAN)
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sf77_INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END


      BLOCK DATA %(f2py_prefix)sBEAMPOL_DEFAULT
C     Unpolarised beams unless PY_SET_BEAMPOL says otherwise. This has
C     to be a default rather than something the caller is trusted to
C     set: the GET_DENSITY routines read /to_beampol/ unconditionally,
C     and a zero-filled common block would be read as a fully polarised
C     beam of the wrong handedness.
      DOUBLE PRECISION BEAMPOL(2)
      COMMON/TO_BEAMPOL/BEAMPOL
      DATA BEAMPOL/1D0,1D0/
      END


      SUBROUTINE %(f2py_prefix)sf77_set_beampol(POL1, POL2)
C     Fill /to_beampol/ on this side of the shared-library boundary --
C     the f2py wrapper and the matrix elements do not share common
C     blocks, which is why the nhel bookkeeping copies rather than
C     aliases them. See PY_SET_BEAMPOL for the convention.
      IMPLICIT NONE
      DOUBLE PRECISION POL1, POL2
      DOUBLE PRECISION BEAMPOL(2)
      COMMON/TO_BEAMPOL/BEAMPOL
      BEAMPOL(1) = POL1
      BEAMPOL(2) = POL2
      RETURN
      END


      SUBROUTINE %(f2py_prefix)sf77_set_bwcutoff(NEW_VALUE)
C     Window of the $-syntax propagators (see the MATRIX routines). Filled on
C     this side of the shared-library boundary, like /to_beampol/: a common
C     block written from the f2py wrapper is not the one the matrix elements
C     read.
      IMPLICIT NONE
      DOUBLE PRECISION NEW_VALUE
      DOUBLE PRECISION STANDALONE_BWCUTOFF
      COMMON/TO_STANDALONE_BWCUTOFF/STANDALONE_BWCUTOFF
      STANDALONE_BWCUTOFF = NEW_VALUE
      RETURN
      END
      
      
      subroutine %(f2py_prefix)sf77_CHANGE_PARA(name, value)
      use model_object
      use aloha_object
      implicit none
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      character*512 name
      double precision value
      
      %(helreset_def)s

      include '../Source/MODEL/input.inc'
      include '../Source/MODEL/coupl.inc'

      %(helreset_setup)s

      SELECT CASE (name)
         %(parameter_setup)s
         CASE DEFAULT
            write(*,*) 'no parameter matching', name, value
      END SELECT

      return
      end
      
    subroutine %(f2py_prefix)sf77_update_all_coup()
    implicit none
     call coup()
    return 
    end
      
      

    subroutine %(f2py_prefix)sf77_get_pdg_order(PDG, ALLPROC)
  IMPLICIT NONE
CF2PY INTEGER, intent(out) :: PDG(%(nb_me)i,%(maxpart)i)  
CF2PY INTEGER, intent(out) :: ALLPROC(%(nb_me)i)
  INTEGER PDG(%(nb_me)i,%(maxpart)i), PDGS(%(nb_me)i,%(maxpart)i)
  INTEGER ALLPROC(%(nb_me)i),PIDs(%(nb_me)i)
  DATA PDGS/ %(pdgs)s /
  DATA PIDS/ %(pids)s /
  PDG = PDGS
  ALLPROC = PIDS
  RETURN
  END 

    subroutine %(f2py_prefix)sf77_get_prefix(prefix)
  IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: PREFIX(%(nb_me)i)
  character*20 PREFIX(%(nb_me)i),PREF(%(nb_me)i)
  DATA PREF / '%(prefix)s'/
  PREFIX = PREF
  RETURN
  END 
 


    subroutine %(f2py_prefix)sf77_set_fixed_extra_scale(new_value)
    implicit none
CF2PY logical, intent(in) :: new_value
    logical new_value
                logical fixed_extra_scale
            integer maxjetflavor
            double precision mue_over_ref
            double precision mue_ref_fixed
            common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
        fixed_extra_scale = new_value
        return 
        end

    subroutine %(f2py_prefix)sf77_set_mue_over_ref(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    mue_over_ref = new_value
        
    return 
    end

    subroutine %(f2py_prefix)sf77_set_mue_ref_fixed(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    mue_ref_fixed = new_value
        
    return 
    end


    subroutine %(f2py_prefix)sf77_set_maxjetflavor(new_value)
    implicit none
CF2PY integer, intent(in) :: new_value
    integer new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    maxjetflavor = new_value
        
    return 
    end


    subroutine %(f2py_prefix)sf77_set_asmz(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
          integer nloop
      double precision asmz
      common/a_block/asmz,nloop
    asmz = new_value
    write(*,*) "asmz is set to ", new_value
        
    return 
    end

    subroutine %(f2py_prefix)sf77_set_nloop(new_value)
    implicit none
CF2PY integer, intent(in) :: new_value
    integer new_value
          integer nloop
      double precision asmz
      common/a_block/asmz,nloop
    nloop = new_value
     write(*,*) "nloop is set to ", new_value
        
    return 
    end
    

    %(nhel)s

%(flavor_repair_function)s
