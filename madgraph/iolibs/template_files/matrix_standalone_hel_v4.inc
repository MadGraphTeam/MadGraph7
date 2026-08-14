C     Standalone helicity-recycling driver template.
C     The percent-paren placeholders are filled at generation time by the
C     standalone exporter; the dollar-brace placeholders (helas_calls,
C     helicity_lines, ncomb, nwavefuncs, csym_reuse, csym_dead) are filled by
C     hel_recycle.py when it rewrites the MATRIX body into the recycled form
C     (shared wavefunctions + P1N amplitude split). The rewriter's jamp_lines
C     are deliberately NOT used: the color flows come from the shared GET_JAMP
C     instead, see the color stage of MATRIX below.
C
C     NCOMB below is the RECYCLED count (only the good helicity combinations
C     survive); NHEL(0,K) carries the original helicity id of row K, which is
C     what the polarization filter and SMATRIXHEL select on.
      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P, HEL, FLAV_IDX, ANS)
C     Matrix element restricted to a single helicity configuration, rescaled by
C     HELAVGFACTOR exactly like the standard standalone. HEL is the CANONICAL
C     mixed-radix helicity code (what the standard SMATRIX compares against
C     HELALLOW), not a row index: NHEL(0,K) gives the row K had in the full
C     pre-recycling enumeration, and HELALLOW turns that row into its code. The
C     two coincide for an unpolarized process (HELALLOW is then 1..NCOMBFULL)
C     but not when a polarization restriction selects a subset of the codes.
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=${ncomb})
C     Number of helicity rows BEFORE recycling (the allowed-code list length).
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
C     I is the implied-DO variable of the recycled NHEL DATA statements below.
      INTEGER I, K, IDEN
      INTEGER JC(NEXTERNAL)
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sBROKEN_SYM
      INTEGER NHEL(0:NEXTERNAL,NCOMB)
      INTEGER HELALLOW(NCOMBFULL)
${helicity_lines}
%(hel_allow_data)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      ANS = 0D0
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
C MadGraph5_aMC@NLO StandAlone Version - HELICITY RECYCLING
C
C Returns amplitude squared summed/avg over colors and helicities
c for the point in phase space P(0:3,NEXTERNAL).
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
C     I is the implied-DO variable of the recycled NHEL DATA statements below.
      INTEGER I, K, L, M, IDEN
      LOGICAL SELECTED, FOUNDIT
      INTEGER JC(NEXTERNAL)
C     Reduced flavor index: FLAV_IDX with its crossing part stripped.
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
C     FLAV_IDX out of range means the requested flavor is not an allowed
C     combination: its matrix element is identically zero.
      ANS = 0D0
%(hr_cross_decode)s
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_USE, FLAVOR)
      DO K = 1, NEXTERNAL
         JC(K) = +1
      ENDDO
%(hr_cross_apply)s
C     The recycled MATRIX returns the color-summed |M|^2 for every (good)
C     helicity combination at once in TS; the dead combinations were dropped at
C     generation time, so there is no runtime good-helicity filter here. Under a
C     crossing the baked helicity rows play the role of the CROSSED
C     configurations, which is why the warm-up measures the good rows of every
C     crossing and the recycled table is their union.
%(hr_matrix_call)s
      DO K = 1, NCOMB
C        Beam polarization: keep only the rows whose per-leg helicities are in
C        the requested set (same test as IS_BORN_HEL_SELECTED, evaluated on the
C        recycled table).
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
C     P and IC must ALREADY be crossed and FLAV_IDX ALREADY reduced to
C     1..NFLAV: SMATRIX applies the crossing once, above.
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER IC(NEXTERNAL)
      INTEGER FLAV_IDX
      REAL*8 TS(NCOMB)
C
C LOCAL VARIABLES
C
      INTEGER I,J,K,KK
C     Raw storage handed to the P1N split-amplitude calls as a type(aloha)
C     scratch wavefunction (see CombineAmp in the DHELAS library).
      COMPLEX*16 TMP(%(wavefunctionsize)d)
C     AMP is HELICITY MAJOR because that is the layout the recycled helas
C     block writes: one CombineAmp call fills the same amplitude for a whole
C     set of helicity rows at once, so a column of AMP is what it produces.
C     It is exactly the wrong layout to READ from, which is why the color
C     stage gathers rows out of it (into AMPK) rather than indexing it.
C     Only NGRAPHS wide: the color flow temporaries used to be written past
C     NGRAPHS in here, and now live in the gather buffer instead.
C
C     On the heap, not in the executable. Helicity major makes it NCOMB times
C     the standard AMP -- 519 MB for g g > 6g even at this width -- and a
C     fixed-size local that size is linked into __DATA. Together with the
C     825 MB of __TEXT that 3.9M recycled call sites produce, the image runs
C     up against the arm64 dyld shared region, and past it the binary does not
C     START: dyld fails to map its cache and blames the first framework it
C     then cannot find. Allocating keeps __DATA at tens of MB whatever the
C     multiplicity, and costs nothing measurable (g g > 5g steady state
C     0.0219/0.0227 s allocated against 0.0217/0.0218 s linked in).
      COMPLEX*16, ALLOCATABLE, SAVE :: AMP(:,:)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      double precision bwcutoff
      INTEGER FLAVOR(NEXTERNAL)
C     The rows whose helas calls were actually generated. A C-parity partner
C     dropped by the warm-up has none, so its whole AMP row is zero and the
C     color stage would run a full sum over zeros; HRROW lists the rows worth
C     visiting and csym_reuse copies the |M|^2 back into the rest. Built once,
C     from the marks below (none = every row is live).
      LOGICAL HRDEAD(NCOMB)
      INTEGER HRROW(NCOMB), NHRROW
      SAVE HRDEAD, HRROW, NHRROW
      DATA NHRROW/0/
C     The color matrix itself is never read here -- the color sum is done by
C     the shared GET_MATRIX / GET_MATRIX_BATCHV. What this routine still owns
C     is where the common block is FILLED: below a size the entries (and DENOM,
C     always) are written out as DATA, and the standard output puts that DATA in
C     its MATRIX for the same reason. Drop it and DENOM is 0, which turns every
C     |M|^2 into a NaN rather than into a zero.
      INTEGER %(proc_prefix)sCF(%(ncolortriang)d)
      INTEGER %(proc_prefix)sDENOM
      common/%(proc_prefix)scolor_matrix/%(proc_prefix)sCF,%(proc_prefix)sDENOM
%(hr_color_decl)s
C     Recycled helicity table (NHEL(0,k) is the original helicity id).
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
C     Per-flavor amplitude/wavefunction mask plus the FLAV_IDX -> FLAVOR
C     rebuild. The mask is helicity independent, so it is resolved once here
C     and the recycled calls carry the IAND(...) guards.
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
C     That marker is where hel_recycle.split_helas_block cuts when it moves the
C     block into chunk subroutines, so it has to stay right after the calls.
C     The madevent templates need no marker: there the color flows follow the
C     calls directly, and the rewriter cuts at those instead.
C
C     Color stage. Everything below is the SHARED per-helicity code -- the
C     same GET_JAMP the standard output calls, and the same color sum (folded,
C     and batched through BLAS when it is worth it) -- so the recycled build
C     inherits whatever the color side gains instead of carrying its own copy.
%(hr_color_sum)s
C     C-parity de-duplication: a dropped partner's HELAS calls were never
C     generated, so its TS() is 0 here; copy the representative's identical
C     |M|^2 back into it (empty unless the warm-up validated the pairing).
${csym_reuse}
      END


C     Everything below this point -- the per-helicity GET_AMP/GET_JAMP, the
C     density and interference stack, GET_value, the helicity encoder/decoder,
C     the crossing routines and the flavor/BROKEN_SYM helpers -- is appended
C     verbatim from matrix_standalone_v4.inc by _write_hel_recycling_matrix, so
C     the recycled output offers exactly the same entry points as the standard
C     one and the two cannot drift apart.
C
C     Only SMATRIX/SMATRIXHEL take the recycled path. The density machinery
C     evaluates arbitrary helicity configurations (GET_ALL_INTER_CROSSED
C     substitutes helicities at the POS slots), which the recycled table cannot
C     serve: its rows are baked at generation time and the dead ones dropped.
C     It therefore keeps using the plain per-helicity GET_AMP.
