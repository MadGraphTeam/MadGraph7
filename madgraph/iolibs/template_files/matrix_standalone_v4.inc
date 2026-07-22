      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX

C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
      INTEGER FLAV_IDX
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL %(proc_prefix)sSMATRIX(P,FLAV_IDX,ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
C
%(info_lines)s
C
C MadGraph5_aMC@NLO StandAlone Version
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(nincoming)d)
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=%(hel_avg_factor)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
C
C LOCAL VARIABLES
C
      INTEGER NHEL(NEXTERNAL,NCOMB)
C  put in common block to expose this variable to python interface
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
      REAL*8 T
      REAL*8 %(proc_prefix)sMATRIX
      INTEGER IHEL,IDEN, I, J
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/%(beamone_helavgfactor)d,%(beamtwo_helavgfactor)d/
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER JC(NEXTERNAL)
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER NNTRY_FLAV, NGOODHEL_FLAV
      PARAMETER (NNTRY_FLAV=NFLAV)
      PARAMETER (NGOODHEL_FLAV=NCOMB*NFLAV)
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
C     FLAV_USE is the flavor part of FLAV_IDX.
      INTEGER FLAV_USE
%(smatrix_cross_decl)s
      INTEGER NTRY(NFLAV)
      LOGICAL GOODHEL(NCOMB,NFLAV)
      DATA NTRY/NNTRY_FLAV*0/
      DATA GOODHEL/NGOODHEL_FLAV*.FALSE./

C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
      DATA USERHEL/-1/
      LOGICAL HELRESET
      common/%(proc_prefix)shelreset/HELRESET
      data HELRESET/.true./

%(helicity_lines)s
%(den_factor_line)s

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/%(proc_prefix)sBORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/

C
C     FUNCTIONS
C
      LOGICAL %(proc_prefix)sIS_BORN_HEL_SELECTED
      INTEGER %(proc_prefix)sBROKEN_SYM
c----------
c   Check if helreset mode is on
c---------
if (HELRESET) then
   do i=1,NFLAV
      NTRY(i) = 0
   enddo
   do i=1,NCOMB
      do j=1,NFLAV
         GOODHEL(I,j) = .false.
      enddo
   enddo
   HELRESET = .false.
endif

C ----------
C BEGIN CODE
C ----------
C     FLAV_USE = mod(FLAV_IDX-1, NFLAV) + 1 is the flavor used for masking.
C     FLAV_IDX<1 means GET_FLAVOR_INDEX could not resolve the requested flavor:
C     it is not an allowed combination, so its matrix element is identically
C     zero. Short-circuit before touching the 1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1) THEN
        ANS = 0D0
        RETURN
      ENDIF
      FLAV_USE = MOD(FLAV_IDX-1, NFLAV) + 1
%(smatrix_cross_decode)s
C     The helicity filter is deliberately shared by every crossing of a given
C     flavor, so it is indexed by FLAV_USE rather than by the full FLAV_IDX.
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_USE, FLAVOR)
      IF(USERHEL.EQ.-1) NTRY(FLAV_USE)=NTRY(FLAV_USE)+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
%(smatrix_cross_apply)s
C     When spin-2 particles are involved, the Helicity filtering is dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero. But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the helicity sum is).
C     For this reason, we simply remove the filterin when there is only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        DO IHEL=1,NCOMB
          DO J=1,NFLAV
            GOODHEL(IHEL,J)=.TRUE.
          ENDDO
        ENDDO
      ENDIF
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
%(smatrix_goodhel_gate)s
                    IF(NTRY(FLAV_USE).GE.2.AND.POLARIZATIONS(0,0).ne.-1.and.(.not.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL))) THEN
                        CYCLE
                    ENDIF
C                   MATRIX/GET_AMP get already crossed arrays and the reduced
C                   flavor index: the crossing was applied once, above.
%(smatrix_matrix_call)s
                    IF(POLARIZATIONS(0,0).eq.-1.or.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL)) THEN
                        ANS=ANS+T
                    ENDIF
%(smatrix_goodhel_train)s
                ENDIF
             ENDIF
          ENDDO
%(smatrix_iden_line)s
      IF(USERHEL.NE.-1) THEN
          ANS=ANS*HELAVGFACTOR
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).ne.-1) THEN
            ANS=ANS*BEAMS_HELAVGFACTOR(J)
            ANS=ANS/POLARIZATIONS(J,0)
          ENDIF
        ENDDO
      ENDIF
      END


      REAL*8 FUNCTION %(proc_prefix)sMATRIX(P,NHEL,IC,FLAV_IDX)
      use model_object
C
%(info_lines)s
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c for the point with external lines W(0:6,NEXTERNAL)
C
C CONTRACT: P, NHEL and IC must ALREADY be crossed and FLAV_IDX must ALREADY
C be reduced to [1,NFLAV] (see GET_AMP). SMATRIX applies the crossing once,
C before its helicity loop; this routine never decodes anything.
C
%(process_lines)s
C
      use aloha_object
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
C
C LOCAL VARIABLES
C
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(%(ncolortriang)d)
      INTEGER %(proc_prefix)sDENOM
      common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
%(global_variable)s
C COLOR DATA
%(color_data_lines)s
C
C
C ----------
C BEGIN CODE
C ----------
      call %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
c      WRITE (*,*) '  -> AMP = ', AMP
      call %(proc_prefix)sGET_JAMP(AMP,JAMP)
c      WRITE (*,*) '  -> JAMP = ', JAMP
      call %(proc_prefix)sGET_MATRIX(JAMP,%(proc_prefix)sMATRIX)
c      write (*,*) "  -> col.ave. |M|^2 for HEL=[", NHEL ,"] = ", %(proc_prefix)sMATRIX

%(amp2_lines)s

      END

      SUBROUTINE %(proc_prefix)sGET_NHEL(IDEN_STAR,NHEL_STAR)
C CONSTANTS
C
CF2PY INTENT(OUT) :: NHEL_STAR
CF2PY INTENT(OUT) :: IDEN_STAR
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)

      INTEGER NHEL(NEXTERNAL,NCOMB),NHEL_STAR(NEXTERNAL,NCOMB)
      INTEGER IDEN,IDEN_STAR

%(helicity_lines)s
%(den_factor_line)s
      IDEN_STAR = IDEN
      NHEL_STAR = NHEL
      END

      SUBROUTINE %(proc_prefix)sGET_NHEL_IDX(FLAV_IDX_IN,IDEN_STAR,
     & NHEL_STAR)
C     Same as GET_NHEL, but reporting the denominator SMATRIX ACTUALLY
C     divides by for FLAV_IDX_IN, rather than the static IDEN.
C
C     The static IDEN is the averaging/symmetry factor of the uncrossed
C     *representative* flavor. SMATRIX never uses it bare: it applies
C     IDEN/BROKEN_SYM(FLAVOR) uncrossed (BROKEN_SYM correcting the
C     identical-particle count of the representative to that of the actual
C     flavor) and GET_SPINCOL_CROSS*GET_IDENT_CROSS when crossed. A caller
C     that reads GET_NHEL and multiplies ANS by it to recover the raw
C     helicity/color sum -- the natural thing to do, and what makes two
C     crossings comparable -- is therefore wrong for every crossed flavor
C     and for every non-representative one.
C
C     GET_NHEL is deliberately left alone: its signature and its value are
C     what existing uncrossed callers expect.
%(nhel_idx_decl)s
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER    NFLAV
      PARAMETER (NFLAV=%(nflav)d)
CF2PY INTENT(IN) :: FLAV_IDX_IN
CF2PY INTENT(OUT) :: NHEL_STAR
CF2PY INTENT(OUT) :: IDEN_STAR
      INTEGER FLAV_IDX_IN
      INTEGER NHEL_STAR(NEXTERNAL,NCOMB)
      INTEGER IDEN_STAR
      INTEGER NHI_FLAV, NHI_CROSS
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sBROKEN_SYM

      CALL %(proc_prefix)sGET_NHEL(IDEN_STAR, NHEL_STAR)
C     An index naming no valid flavor gives a zero matrix element; report a
C     0 denominator rather than a plausible-looking one.
      IF (FLAV_IDX_IN .LT. 1) THEN
        IDEN_STAR = 0
        RETURN
      ENDIF
      NHI_FLAV = MOD(FLAV_IDX_IN-1, NFLAV) + 1
      NHI_CROSS = (FLAV_IDX_IN-1) / NFLAV
      CALL %(proc_prefix)sGET_FLAVOR(NHI_FLAV, FLAVOR)
%(nhel_idx_body)s
      RETURN
      END

      SUBROUTINE %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
      use model_object
C
%(process_lines)s
C
C     CONTRACT (this routine is pure: it decodes nothing).
C
C       P(0:3,NEXTERNAL) : momenta, ALREADY crossed.
C       NHEL(NEXTERNAL)  : helicities, ALREADY crossed (permuted, NOT negated:
C                          helas uses nh=nhel*nsf, so the sign follows IC).
C       IC(NEXTERNAL)    : NSF/NSV flag per leg, ALREADY crossed (-1 on a leg
C                          the crossing moved across).
C       FLAV_IDX         : flavor index ALREADY reduced to [1,NFLAV]; it must
C                          NOT be an extended index carrying a crossing.
C
C     When the crossing machinery is written out (see %(proc_prefix)sGET_CROSS_PERM below;
C     it is left out when the process was generated with --use_crossing=False,
C     in which case FLAV_IDX is never extended), the callers inside this file
C     (SMATRIX via MATRIX, GET_ALL_INTER_CROSSED) apply the crossing ONCE per
C     entry point with %(proc_prefix)sAPPLY_CROSSING / %(proc_prefix)sAPPLY_CROSSING_TABLE and then call
C     this routine in their inner loop. An external (f2py) caller holding an
C     extended FLAV_IDX must call the public %(proc_prefix)sAPPLY_CROSSING itself first;
C     passing the extended index here would otherwise silently return the
C     UNCROSSED amplitude, so the range is checked below and violations are
C     reported and return AMP=0.
C
CF2PY INTENT(OUT) :: AMP
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: IC
CF2PY INTENT(IN) :: FLAV_IDX

      use aloha_object
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d)
      INTEGER    NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
      COMPLEX*16 AMP(NGRAPHS)
C
C LOCAL VARIABLES
C
C     FLAVOR is rebuilt from FLAV_IDX below and is NOT permuted by any
C     crossing: each slot keeps its own flavor-group position, which is what
C     the mask indexes.
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER AMP_I
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      double precision bwcutoff
%(flavor_mask_decl)s
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
%(global_variable)s
C
C
      bwcutoff=15 ! use if $ syntax is defined in the process
C     Contract guard: an extended FLAV_IDX (one carrying a crossing) reaching
C     this routine would be silently truncated to its flavor part and give the
C     uncrossed amplitude. Fail loudly and return zero instead.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        WRITE(*,*) 'ERROR: GET_AMP got FLAV_IDX', FLAV_IDX, 'NFLAV', NFLAV
        WRITE(*,*) 'GET_AMP needs a reduced index and crossed P/NHEL/IC.'
        WRITE(*,*) 'Returning AMP=0.'
        DO AMP_I = 1, NGRAPHS
          AMP(AMP_I) = (0D0, 0D0)
        ENDDO
        RETURN
      ENDIF
%(flavor_mask_setup)s
%(helas_calls)s
%(amp2_lines)s
      END

      SUBROUTINE %(proc_prefix)sGET_JAMP(AMP,JAMP)
C
%(process_lines)s
C
CF2PY INTENT(OUT) :: JAMP
CF2PY INTENT(IN) :: AMP

      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER ( NCOLOR=%(ncolor)d)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)

%(jamp_lines)s
      END

      SUBROUTINE %(proc_prefix)sGET_MATRIX(JAMP,MATRIX)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
CF2PY INTENT(OUT) :: MATRIX
CF2PY INTENT(IN) :: JAMP


      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      REAL*8     ZERO,MATRIX
      PARAMETER (ZERO=0D0)
C

C LOCAL VARIABLES
C
      INTEGER I,J
      COMPLEX*16 ZTEMP

      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(proc_prefix)sDENOM
      common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C

C COLOR DATA
C

      MATRIX = 0.D0
      CF_INDEX = 0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX + 1
          ZTEMP = ZTEMP + %(proc_prefix)sCF(CF_INDEX)*JAMP(J)
        ENDDO
        MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))/%(proc_prefix)sDENOM
      ENDDO
      END



      SUBROUTINE %(proc_prefix)sGET_INTER(JAMP_1,JAMP_2, INTER)

CF2PY INTENT(OUT) :: INTER
CF2PY INTENT(IN) :: JAMP_1
CF2PY INTENT(IN) :: JAMP_2

      INTEGER I,J
      INTEGER NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(proc_prefix)sDENOM, IDEN
      %(den_factor_line)s
      common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 JAMP_1(NCOLOR),JAMP_2(NCOLOR),INTER

C COLOR DATA
C

      INTER = (0.D0,0.D0)
      CF_INDEX = 0
      DO I = 1, NCOLOR
C        ZTEMP = DCONJG(JAMP_2(I))
         DO J=I, NCOLOR
          CF_INDEX = CF_INDEX +1
          INTER = INTER + %(proc_prefix)sCF(CF_INDEX) * (JAMP_1(J) * DCONJG(JAMP_2(I)) +JAMP_1(I) * DCONJG(JAMP_2(J)))
        ENDDO
      ENDDO
      INTER = INTER/ (2d0*%(proc_prefix)sDENOM*IDEN)

      END



      SUBROUTINE  %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, SCALE2, INTER)
C     Entry point taking the full FLAVOR(NEXTERNAL) array (back-compat):
C     resolve it to FLAV_IDX and forward to GET_DENSITY_IDX. A crossing can
C     only be requested through GET_DENSITY_IDX: an extended FLAV_IDX carries
C     a crossing code, which no FLAVOR array can express (GET_FLAVOR_INDEX
C     only ever returns 1..NFLAV).
      implicit none
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAVOR(%(nexternal)d)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: SCALE2
CF2PY INTENT(OUT) :: INTER(N_COMB*(N_COMB+1)/2)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE PRECISION ALPHAS, SCALE2
      DOUBLE COMPLEX INTER(*)
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX

      CALL %(proc_prefix)sGET_DENSITY_IDX(P, POS, N_CHANGING, ALLOW_HEL,
     & N_COMB, %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR), ALPHAS, SCALE2,
     & INTER)
      RETURN
      END


      SUBROUTINE  %(proc_prefix)sGET_DENSITY_IDX(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAV_IDX, ALPHAS, SCALE2, INTER)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER(NCOMB*(NCOMB+1)/2): all interference term (not the symmetric one)
c     FLAV_IDX may carry a crossing. It is decoded and applied ONCE here (the
c     whole NHEL table, the momenta and the NSF flags in one go) and only
c     crossed arrays plus the reduced flavor index travel further down. POS and
c     the helicity labels refer to the UNCROSSED (source process) leg ordering
c     and are mapped through the crossing permutation here. No helicity flip is
c     needed on top: helas folds it into nh=nhel*nsf when the NSF flag of a
c     crossed leg is flipped.
      use model_object
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAV_IDX
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: SCALE2
CF2PY INTENT(OUT) :: INTER(N_COMB*(N_COMB+1)/2)
C  SCALE2 is a dummy argument added to have the same syntax as in loop-induced
c
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER THISNHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAV_IDX
      DOUBLE PRECISION ALPHAS, SCALE2
      DOUBLE COMPLEX INTER(*)
      INTEGER NINTER
      INTEGER NB_NHEL
      DOUBLE COMPLEX, ALLOCATABLE :: TMP_INTER(:)
      PARAMETER (NB_NHEL=%(ncomb)d)
c LOCAL
      INTEGER I,IHEL,IPART
      DOUBLE PRECISION PI
C     Crossed copies, built once (see the crossing block below).
      REAL*8 PUSE(0:3,NEXTERNAL)
      INTEGER NHELUSE(NEXTERNAL,NB_NHEL)
      INTEGER IC(NEXTERNAL), ICUSE(NEXTERNAL)
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL), CPOS(NEXTERNAL)
      INTEGER FLAV_USE, DUMFLAV
      DOUBLE PRECISION RESCALE
C
      INTEGER NHEL(NEXTERNAL,NB_NHEL)
C     put in common block to expose this variable to python interface
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
C
C     include coupling definition to update the value of alphas
C
      include 'coupl.inc'

      ninter = N_COMB*(N_COMB+1)/2
      allocate(TMP_INTER(NINTER))
      TMP_INTER(:) = (0d0, 0d0)

      DO I=1, N_COMB*(N_COMB+1)/2
        INTER(I) = 0
      enddo

      IF (ALPHAS.ne.0d0) then
          pi = 3.141592653589793d0
          G = 2* DSQRT(ALPHAS*pi)
          call UPDATE_AS_PARAM()
      ENDIF
C     Unresolved flavor (GET_FLAVOR_INDEX miss): the matrix element and hence
C     every interference term is identically zero. Guarded after the alphas
C     update so that side effect is unchanged.
      IF (FLAV_IDX.LT.1) THEN
        return
      ENDIF
C     Decode and apply the crossing ONCE for the whole density matrix: the
C     permutation is the same for every helicity row, so the NHEL table is
C     permuted in one sweep. RESCALE carries the flavor / crossing dependent
C     part of the normalisation (RESCALE=0 = impossible crossing).
      IC(:) = 1
      CALL %(proc_prefix)sGET_INTER_RESCALE(FLAV_IDX, FLAV_USE, RESCALE)
      IF (RESCALE.EQ.0D0) THEN
        return
      ENDIF
%(density_cross_apply)s
      DO IHEL =1, NB_NHEL
        THISNHEL(:) = NHELUSE(:, IHEL)
        DO IPART=1,N_CHANGING
            if(THISNHEL(CPOS(IPART)).NE.ALLOW_HEL(IPART)) GOTO 10 !BYPASS COMPUTATION FOR HELICITY
        ENDDO
        TMP_INTER(:) = 0
        call  %(proc_prefix)sGET_ALL_INTER_CROSSED(PUSE, THISNHEL, ICUSE, CPOS, N_CHANGING, ALLOW_HEL, N_COMB, FLAV_USE, RESCALE, TMP_INTER)
        do I = 1, N_COMB*(N_COMB+1)/2
           INTER(I) = INTER(I) + TMP_INTER(I)
        enddo
 10   enddo
      return
      deallocate(TMP_INTER)
      end

      SUBROUTINE  %(proc_prefix)sGET_ALL_INTER(P, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, INTER)
C     Entry point taking the full FLAVOR(NEXTERNAL) array (back-compat); see
C     GET_DENSITY. Use GET_ALL_INTER_IDX to request a crossing.
      implicit none
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: NHEL(%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAVOR(%(nexternal)d)
CF2PY INTENT(OUT) :: INTER(NCOMB*(NCOMB+1)/2)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE COMPLEX INTER(*)
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX

      CALL %(proc_prefix)sGET_ALL_INTER_IDX(P, NHEL, POS, N_CHANGING,
     & ALLOW_HEL, N_COMB, %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR), INTER)
      RETURN
      END


      SUBROUTINE  %(proc_prefix)sGET_ALL_INTER_IDX(P, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAV_IDX, INTER)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER((NCOMB*NCOMB+1)/2: all interference term (not the symmetric one)
c     FLAV_IDX may carry a crossing: it is decoded and applied ONCE here, and
c     GET_ALL_INTER_CROSSED below then only sees crossed arrays. POS is given
c     in the UNCROSSED (source process) slot numbering and is mapped through
c     the crossing permutation here.
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: NHEL(%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAV_IDX
CF2PY INTENT(OUT) :: INTER(NCOMB*(NCOMB+1)/2)
c
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAV_IDX
      DOUBLE COMPLEX INTER(*)
c
c     LOCAL
c
      INTEGER I, IPART
      INTEGER IC(NEXTERNAL), ICUSE(NEXTERNAL)
      REAL*8 PUSE(0:3,NEXTERNAL)
      INTEGER NHELUSE(NEXTERNAL)
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL), CPOS(NEXTERNAL)
      INTEGER FLAV_USE, DUMFLAV
      DOUBLE PRECISION RESCALE
C ----------
C BEGIN CODE
C ----------
C     Unresolved flavor (not an allowed combination): the matrix element and
C     therefore all interference terms are zero.
      IF (FLAV_IDX.LT.1) THEN
         DO I = 1, N_COMB*(N_COMB+1)/2
            INTER(I) = (0d0, 0d0)
         ENDDO
         RETURN
      ENDIF
C     RESCALE carries everything flavor / crossing dependent in the
C     normalisation; RESCALE=0 marks a crossing that cannot be applied.
      CALL %(proc_prefix)sGET_INTER_RESCALE(FLAV_IDX, FLAV_USE, RESCALE)
      IF (RESCALE.EQ.0D0) THEN
         DO I = 1, N_COMB*(N_COMB+1)/2
            INTER(I) = (0d0, 0d0)
         ENDDO
         RETURN
      ENDIF
%(allinter_cross_apply)s
      CALL %(proc_prefix)sGET_ALL_INTER_CROSSED(PUSE, NHELUSE, ICUSE, CPOS,
     & N_CHANGING, ALLOW_HEL, N_COMB, FLAV_USE, RESCALE, INTER)
      RETURN
      END


      SUBROUTINE %(proc_prefix)sGET_INTER_RESCALE(FLAV_IDX, FLAV_USE,
     & RESCALE)
C     Split an extended FLAV_IDX and return the factor by which GET_INTER's
C     output must be multiplied.
C
C     GET_INTER only ever sees JAMPs, so it cannot know the flavor: it
C     normalises with the bare static IDEN and everything flavor dependent has
C     to be applied by its caller. That is also what keeps the density matrix
C     consistent with SMATRIX. RESCALE=0 marks a crossing that cannot be
C     applied (zero matrix element).
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER FLAV_IDX, FLAV_USE
      DOUBLE PRECISION RESCALE
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sBROKEN_SYM
%(inter_rescale_decl)s

      FLAV_USE = MOD(FLAV_IDX-1, NFLAV) + 1
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_USE, FLAVOR)
%(inter_rescale_body)s

      RETURN
      END


      SUBROUTINE  %(proc_prefix)sGET_ALL_INTER_CROSSED(P, NHEL, IC, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAV_IDX, RESCALE, INTER)
c     Inner worker of the density machinery.
c
c     CONTRACT: P, NHEL and IC are ALREADY crossed, POS is expressed in the
c     CROSSED slot numbering, FLAV_IDX is ALREADY reduced to [1,NFLAV] and
c     RESCALE already accounts for BROKEN_SYM / the crossed denominator. The
c     callers (GET_ALL_INTER_IDX, GET_DENSITY_IDX) decode and apply the
c     crossing once, so nothing is decoded per GET_AMP call here.
c     NHEL is overwritten at the POS slots.
      implicit none
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER IC(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAV_IDX
      DOUBLE PRECISION RESCALE
      DOUBLE COMPLEX INTER(*)
c
c     Intermediate array
c
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)

      DOUBLE COMPLEX AMP(NGRAPHS)
      DOUBLE COMPLEX, ALLOCATABLE, SAVE :: JAMP(:,:)
      INTEGER, SAVE :: S_NCOMB = 0
c
c     LOCAL
c
      INTEGER I,J,SOL,N

      if (allocated(jamp) .and. S_NCOMB.ne.N_COMB) then
           deallocate(jamp)
      endif

     if (.not.allocated(jamp)) then
        S_NCOMB=N_COMB
        allocate(jamp(ncolor, N_COMB))
      endif
C ----------
C BEGIN CODE
C ----------
      do I = 1, N_COMB
         do N = 1, N_CHANGING
            NHEL(POS(N)) = ALLOW_HEL((I-1)*N_CHANGING+N)
         enddo
         call %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
         call %(proc_prefix)sGET_JAMP(AMP,JAMP(1,I))
      enddo

    SOL = 0
    DO I = 1, N_COMB
       DO J= I, N_COMB
         SOL = SOL +1
         call %(proc_prefix)sGET_INTER(JAMP(1,I), JAMP(1,J), INTER(SOL))
         INTER(SOL) = INTER(SOL)*RESCALE
       ENDDO
    ENDDO


    RETURN
    END




      SUBROUTINE %(proc_prefix)sGET_value(P, ALPHAS, NHEL, FLAVOR ,ANS)
C     f2py interface accepting the full FLAVOR(NEXTERNAL) array (back-compat):
C     resolve it to FLAV_IDX and forward to GET_value_internal. Use
C     GET_value_idx below to pass the flavor index directly.
      use model_object
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
C     ROUTINE FOR F2PY to read the benchmark point.

      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
      CALL  %(proc_prefix)sGET_value_internal(P, ALPHAS, NHEL, FLAV_IDX ,ANS)
      RETURN
      END


      SUBROUTINE %(proc_prefix)sGET_value_idx(P, ALPHAS, NHEL, FLAV_IDX ,ANS)
C     f2py interface accepting the flavor index directly.
      use model_object
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS
      INTEGER FLAV_IDX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAV_IDX

      CALL  %(proc_prefix)sGET_value_internal(P, ALPHAS, NHEL, FLAV_IDX ,ANS)
      RETURN
      END


      SUBROUTINE %(proc_prefix)sGET_value_internal(P, ALPHAS, NHEL, FLAV_IDX ,ANS)
      use model_object
C     This routine is the real value but can not be in the interface
C     due to f2py not knowing how to handle the couplings common block
      use model_object
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS
      real*8 pi
      INTEGER FLAV_IDX
C     ROUTINE FOR F2PY to read the benchmark point.
C     the include file with the values of the parameters and masses
      include "coupl.inc"

      pi = 3.141592653589793d0
      G = 2* DSQRT(ALPHAS*pi)
      call UPDATE_AS_PARAM()
      if (NHEL.ne.0) then
        CALL %(proc_prefix)sSMATRIXHEL(P, NHEL, FLAV_IDX, ANS)
      else
        CALL %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
      endif
      return
      end

      SUBROUTINE %(proc_prefix)sINITIALISEMODEL(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      call setpara(PATH)  !first call to setup the paramaters
      return
      end

      LOGICAL FUNCTION %(proc_prefix)sIS_BORN_HEL_SELECTED(HELID)
      IMPLICIT NONE
C
C     CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C
C     ARGUMENTS
C
      INTEGER HELID
C
C     LOCALS
C
      INTEGER I,J
      LOGICAL FOUNDIT
C
C     GLOBALS
C
      INTEGER HELC(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sPROCESS_NHEL/HELC

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/%(proc_prefix)sBORN_BEAM_POL/POLARIZATIONS
C     ----------
C     BEGIN CODE
C     ----------

      %(proc_prefix)sIS_BORN_HEL_SELECTED = .True.
      if (POLARIZATIONS(0,0).eq.-1) THEN
        RETURN
      ENDIF

      DO I=1,NEXTERNAL
        IF (POLARIZATIONS(I,0).eq.-1) THEN
          CYCLE
        ENDIF
        FOUNDIT = .FALSE.
        DO J=1,POLARIZATIONS(I,0)
          IF (HELC(I,HELID).eq.POLARIZATIONS(I,J)) THEN
            FOUNDIT = .True.
            EXIT
          ENDIF
        ENDDO
        IF(.not.FOUNDIT) THEN
          %(proc_prefix)sIS_BORN_HEL_SELECTED = .False.
          RETURN
        ENDIF
      ENDDO

      RETURN
      END


%(crossing_routines)s


%(broken_sym_function)s


%(flavor_index_function)s


%(flavor_array_function)s


%(flavor_pdg_function)s
