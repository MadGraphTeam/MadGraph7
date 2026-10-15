C     Standalone helicity-recycling driver template. The
C     percent-paren placeholders are filled at generation time
C     by the standalone exporter; the dollar-brace placeholders
C     (helas_calls, helicity_lines, ncomb, nwavefuncs,
C     csym_reuse, csym_dead) are filled by hel_recycle.py when
C     it rewrites the MATRIX body into the recycled form (shared
C     wavefunctions + P1N amplitude split). The rewriter's
C     jamp_lines are deliberately NOT used: the color flows come
C     from the shared GET_JAMP instead, see the color stage of
C     MATRIX below.
C
C     NCOMB below is the RECYCLED count (only the good helicity
C     combinations survive); NHEL(0,K) carries the original
C     helicity id of row K, which is what the polarization
C     filter and SMATRIXHEL select on.
      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P, HEL, FLAV_IDX, ANS)
C     Matrix element restricted to a single helicity
C     configuration, rescaled by HELAVGFACTOR exactly like the
C     standard standalone. HEL is the CANONICAL mixed-radix
C     helicity code (what the standard SMATRIX compares against
C     HELALLOW), not a row index: NHEL(0,K) gives the row K had
C     in the full pre-recycling enumeration, and HELALLOW turns
C     that row into its code. The two coincide for an
C     unpolarized process (HELALLOW is then 1..NCOMBFULL) but
C     not when a polarization restriction selects a subset of
C     the codes.
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=${ncomb})
C     Number of helicity rows BEFORE recycling (the allowed-code
C     list length).
      INTEGER    NCOMBFULL
      PARAMETER (NCOMBFULL=%(ncomb)d)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=%(hel_avg_factor)d)
      INTEGER    NFLAV
      PARAMETER (NFLAV=%(nflav)d)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
      REAL*8 P(0:3,NEXTERNAL), ANS
      INTEGER HEL, FLAV_IDX
      REAL*8 TS(NCOMB)
C     I is the implied-DO variable of the recycled NHEL DATA
C     statements below.
      INTEGER I, K, IDEN
      INTEGER JC(NEXTERNAL)
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sBROKEN_SYM
      INTEGER NHEL(0:NEXTERNAL,NCOMB)
      INTEGER HELALLOW(NCOMBFULL)%(hr_helsel_decl)s
${helicity_lines}
%(hel_allow_data)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      ANS = 0D0
C     HEL=-1: no helicity chosen, sum them (as USERHEL=-1)
      IF (HEL.EQ.-1) THEN
        CALL %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
        RETURN
      ENDIF
%(hr_helcheck)s
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX, FLAVOR)
      DO K = 1, NEXTERNAL
         JC(K) = +1
      ENDDO
      CALL %(proc_prefix)sMATRIX(P, JC, FLAV_IDX, TS)
      DO K = 1, NCOMB
         IF (HELALLOW(NHEL(0,K)).EQ.HEL) ANS = ANS + TS(K)
      ENDDO
      ANS = ANS / DBLE(IDEN) * %(proc_prefix)sBROKEN_SYM(FLAVOR)
      ANS = ANS * HELAVGFACTOR
      END


      SUBROUTINE %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
C
%(process_lines)s
C
C MadGraph7 StandAlone Version - HELICITY RECYCLING
C
C Returns amplitude squared summed/avg over colors and
C helicities for the point in phase space P(0:3,NEXTERNAL).
C
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(nincoming)d)
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=${ncomb})
      INTEGER    NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      REAL*8 P(0:3,NEXTERNAL), ANS
      INTEGER FLAV_IDX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
      REAL*8 TS(NCOMB)
C     I is the implied-DO variable of the recycled NHEL DATA
C     statements below.
      INTEGER I, K, L, M, IDEN
      LOGICAL SELECTED, FOUNDIT
      INTEGER JC(NEXTERNAL)
C     Reduced flavor index: FLAV_IDX with its crossing part
C     stripped.
      INTEGER FLAV_USE
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sBROKEN_SYM
%(hr_cross_decl)s
C     For a 1>N process BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(K),K=1,2)/%(beamone_helavgfactor)d,%(beamtwo_helavgfactor)d/
      INTEGER NHEL(0:NEXTERNAL,NCOMB)
${helicity_lines}
%(den_factor_line)s
      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/%(proc_prefix)sBORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(K,L),K=0,NEXTERNAL),L=0,5)/NPOLENTRIES*-1/
C ----------
C BEGIN CODE
C ----------
C     FLAV_IDX out of range means the requested flavor is not an
C     allowed combination: its matrix element is identically
C     zero.
      ANS = 0D0
%(hr_cross_decode)s
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_USE, FLAVOR)
      DO K = 1, NEXTERNAL
         JC(K) = +1
      ENDDO
%(hr_cross_apply)s
C     The recycled MATRIX returns the color-summed |M|^2 for
C     every (good) helicity combination at once in TS; the dead
C     combinations were dropped at generation time, so there is
C     no runtime good-helicity filter here. Under a crossing the
C     baked helicity rows play the role of the CROSSED
C     configurations, which is why the warm-up measures the good
C     rows of every crossing and the recycled table is their
C     union.
%(hr_matrix_call)s
      DO K = 1, NCOMB
C        Beam polarization: keep only the rows whose per-leg
C        helicities are in the requested set (same test as
C        IS_BORN_HEL_SELECTED, evaluated on the recycled table).
         IF (POLARIZATIONS(0,0).NE.-1) THEN
            SELECTED = .TRUE.
            DO L = 1, NEXTERNAL
               IF (POLARIZATIONS(L,0).EQ.-1) CYCLE
               FOUNDIT = .FALSE.
               DO M = 1, POLARIZATIONS(L,0)
                  IF (NHEL(L,K).EQ.POLARIZATIONS(L,M)) THEN
                     FOUNDIT = .TRUE.
                     EXIT
                  ENDIF
               ENDDO
               IF (.NOT.FOUNDIT) THEN
                  SELECTED = .FALSE.
                  EXIT
               ENDIF
            ENDDO
            IF (.NOT.SELECTED) CYCLE
         ENDIF
         ANS = ANS + TS(K)
      ENDDO
%(hr_iden_line)s
      DO L = 1, NINITIAL
         IF (POLARIZATIONS(L,0).NE.-1) THEN
            ANS = ANS * BEAMS_HELAVGFACTOR(L)
            ANS = ANS / POLARIZATIONS(L,0)
         ENDIF
      ENDDO
      END


      SUBROUTINE %(proc_prefix)sMATRIX(P, IC, FLAV_IDX, TS)
      use model_object
      use aloha_object
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=${nwavefuncs}, NCOLOR=%(ncolor)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=${ncomb})
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C
C ARGUMENTS
C
C     P and IC must ALREADY be crossed and FLAV_IDX ALREADY
C     reduced to 1..NFLAV: SMATRIX applies the crossing once,
C     above.
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER IC(NEXTERNAL)
      INTEGER FLAV_IDX
      REAL*8 TS(NCOMB)
C
C LOCAL VARIABLES
C
      INTEGER I,J,K,KK
C     Raw storage handed to the P1N split-amplitude calls as a
C     type(aloha) scratch wavefunction (see CombineAmp in the
C     DHELAS library).
      COMPLEX*16 TMP(%(wavefunctionsize)d)
C     AMP is HELICITY MAJOR because that is the layout the
C     recycled helas block writes: one CombineAmp call fills the
C     same amplitude for a whole set of helicity rows at once,
C     so a column of AMP is what it produces. It is exactly the
C     wrong layout to READ from, which is why the color stage
C     gathers rows out of it (into AMPK) rather than indexing
C     it. Only NGRAPHS wide: the color flow temporaries used to
C     be written past NGRAPHS in here, and now live in the
C     gather buffer instead.
C
C     On the heap, not in the executable. Helicity major makes
C     it NCOMB times the standard AMP -- 519 MB for g g > 6g
C     even at this width -- and a fixed-size local that size is
C     linked into __DATA. Together with the 825 MB of __TEXT
C     that 3.9M recycled call sites produce, the image runs up
C     against the arm64 dyld shared region, and past it the
C     binary does not START: dyld fails to map its cache and
C     blames the first framework it then cannot find. Allocating
C     keeps __DATA at tens of MB whatever the multiplicity, and
C     costs nothing measurable (g g > 5g steady state
C     0.0219/0.0227 s allocated against 0.0217/0.0218 s linked
C     in).
      COMPLEX*16, ALLOCATABLE, SAVE :: AMP(:,:)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      double precision bwcutoff
      INTEGER FLAVOR(NEXTERNAL)
C     The rows whose helas calls were actually generated. A
C     C-parity partner dropped by the warm-up has none, so its
C     whole AMP row is zero and the color stage would run a full
C     sum over zeros; HRROW lists the rows worth visiting and
C     csym_reuse copies the |M|^2 back into the rest. Built
C     once, from the marks below (none = every row is live).
      LOGICAL HRDEAD(NCOMB)
      INTEGER HRROW(NCOMB), NHRROW
      SAVE HRDEAD, HRROW, NHRROW
      DATA NHRROW/0/
C     The color matrix itself is never read here -- the color
C     sum is done by the shared GET_MATRIX / GET_MATRIX_BATCHV.
C     What this routine still owns is where the common block is
C     FILLED: below a size the entries (and DENOM, always) are
C     written out as DATA, and the standard output puts that
C     DATA in its MATRIX for the same reason. Drop it and DENOM
C     is 0, which turns every |M|^2 into a NaN rather than into
C     a zero.
      INTEGER %(proc_prefix)sCF(%(ncolortriang)d)
      INTEGER %(proc_prefix)sDENOM
      common/%(proc_prefix)scolor_matrix/%(proc_prefix)sCF,%(proc_prefix)sDENOM
%(hr_color_decl)s
C     Recycled helicity table (NHEL(0,k) is the original
C     helicity id).
      INTEGER NHEL(0:NEXTERNAL,NCOMB)
${helicity_lines}
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
%(global_variable)s
C
C COLOR DATA
C
%(color_data_lines)s
C     Per-flavor amplitude/wavefunction mask plus the FLAV_IDX
C     -> FLAVOR rebuild. The mask is helicity independent, so it
C     is resolved once here and the recycled calls carry the
C     IAND(...) guards.
%(flavor_mask_decl)s
C ----------
C BEGIN CODE
C ----------
      bwcutoff=15
      IF (.NOT.ALLOCATED(AMP)) ALLOCATE(AMP(NCOMB,NGRAPHS))
      IF (NHRROW.EQ.0) THEN
        DO K = 1, NCOMB
          HRDEAD(K) = .FALSE.
        ENDDO
${csym_dead}
        DO K = 1, NCOMB
          IF (.NOT.HRDEAD(K)) THEN
            NHRROW = NHRROW + 1
            HRROW(NHRROW) = K
          ENDIF
        ENDDO
      ENDIF
      AMP(:,:) = (0D0,0D0)
%(flavor_mask_setup)s
${helas_calls}
C     END OF RECYCLED HELAS BLOCK
C     That marker is where hel_recycle.split_helas_block cuts
C     when it moves the block into chunk subroutines, so it has
C     to stay right after the calls. The madevent templates need
C     no marker: there the color flows follow the calls
C     directly, and the rewriter cuts at those instead.
C
C     Color stage. Everything below is the SHARED per-helicity
C     code -- the same GET_JAMP the standard output calls, and
C     the same color sum (folded, and batched through BLAS when
C     it is worth it) -- so the recycled build inherits whatever
C     the color side gains instead of carrying its own copy.
%(hr_color_sum)s
C     C-parity de-duplication: a dropped partner's HELAS calls
C     were never generated, so its TS() is 0 here; copy the
C     representative's identical |M|^2 back into it (empty
C     unless the warm-up validated the pairing).
${csym_reuse}
      END


C     Everything below this point -- the per-helicity
C     GET_AMP/GET_JAMP, the density and interference stack,
C     GET_value, the helicity encoder/decoder, the crossing
C     routines and the flavor/BROKEN_SYM helpers -- is appended
C     verbatim from matrix_standalone_v4.inc by
C     _write_hel_recycling_matrix, so the recycled output offers
C     exactly the same entry points as the standard one and the
C     two cannot drift apart.
C
C     Only SMATRIX/SMATRIXHEL take the recycled path. The
C     density machinery evaluates arbitrary helicity
C     configurations (GET_ALL_INTER_CROSSED substitutes
C     helicities at the POS slots), which the recycled table
C     cannot serve: its rows are baked at generation time and
C     the dead ones dropped. It therefore keeps using the plain
C     per-helicity GET_AMP.
