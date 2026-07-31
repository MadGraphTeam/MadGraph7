C     Standalone helicity-recycling driver template.
C     The percent-paren placeholders are filled at generation time by the
C     standalone exporter; the dollar-brace placeholders (helas_calls,
C     jamp_lines, helicity_lines, ncomb, nwavefuncs, csym_reuse) are filled by
C     hel_recycle.py when it rewrites the MATRIX body into the recycled form
C     (shared wavefunctions + P1N amplitude split).
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
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) RETURN
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX, FLAVOR)
      CALL %(proc_prefix)sMATRIX(P, FLAV_IDX, TS)
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
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sBROKEN_SYM
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
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) RETURN
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX, FLAVOR)
C     The recycled MATRIX returns the color-summed |M|^2 for every (good)
C     helicity combination at once in TS; the dead combinations were dropped at
C     generation time, so there is no runtime good-helicity filter here.
      CALL %(proc_prefix)sMATRIX(P, FLAV_IDX, TS)
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
      ANS = ANS / DBLE(IDEN) * %(proc_prefix)sBROKEN_SYM(FLAVOR)
      DO L = 1, NINITIAL
         IF (POLARIZATIONS(L,0).NE.-1) THEN
            ANS = ANS * BEAMS_HELAVGFACTOR(L)
            ANS = ANS / POLARIZATIONS(L,0)
         ENDIF
      ENDDO
      END


      SUBROUTINE %(proc_prefix)sMATRIX(P, FLAV_IDX, TS)
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
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER FLAV_IDX
      REAL*8 TS(NCOMB)
C
C LOCAL VARIABLES
C
      INTEGER I,J,K
      COMPLEX*16 ZTEMP, TMP_JAMP(%(nb_temp_jamp)i)
C     Raw storage handed to the P1N split-amplitude calls as a type(aloha)
C     scratch wavefunction (see CombineAmp in the DHELAS library).
      COMPLEX*16 TMP(%(wavefunctionsize)d)
      INTEGER %(proc_prefix)sCF(%(ncolortriang)d)
      INTEGER %(proc_prefix)sDENOM, CF_INDEX
      COMPLEX*16 AMP(NCOMB,NGRAPHS), JAMP(NCOLOR)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      double precision bwcutoff
      INTEGER FLAVOR(NEXTERNAL)
C     Recycled helicity table (NHEL(0,k) is the original helicity id).
      INTEGER NHEL(0:NEXTERNAL,NCOMB)
${helicity_lines}
C
C GLOBAL VARIABLES
C
      common/%(proc_prefix)scolor_matrix/%(proc_prefix)sCF,%(proc_prefix)sDENOM
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
      AMP(:,:) = (0D0,0D0)
%(flavor_mask_setup)s
${helas_calls}

C     The recycled jamp block (substituted below) resets JAMP and fills
C     JAMP(icolor) from AMP(K,igraph) for the current helicity K.
      DO K = 1, NCOMB
${jamp_lines}
         TS(K) = 0.D0
         CF_INDEX = 0
         DO I = 1, NCOLOR
            ZTEMP = (0.D0,0.D0)
            DO J = I, NCOLOR
               CF_INDEX = CF_INDEX + 1
               ZTEMP = ZTEMP + %(proc_prefix)sCF(CF_INDEX)*JAMP(J)
            ENDDO
            TS(K) = TS(K) + REAL(ZTEMP*DCONJG(JAMP(I)))
         ENDDO
         TS(K) = TS(K) / %(proc_prefix)sDENOM
      ENDDO
C     C-parity de-duplication: a dropped partner's HELAS calls were never
C     generated, so its TS() is 0 here; copy the representative's identical
C     |M|^2 back into it (empty unless the warm-up validated the pairing).
${csym_reuse}
      END


%(broken_sym_function)s


%(flavor_index_function)s


%(flavor_array_function)s


%(flavor_pdg_function)s


      SUBROUTINE %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL,
     $  N_COMB, FLAVOR, ALPHAS, SCALE2, INTER)
C     Spin-density-matrix mode is not supported with --hel_recycling yet. This
C     stub only satisfies the driver's link-time reference to GET_DENSITY; it
C     aborts if actually called.
      IMPLICIT NONE
      REAL*8 P(*), ALPHAS, SCALE2
      INTEGER POS(*), N_CHANGING, ALLOW_HEL(*), N_COMB, FLAVOR(*)
      COMPLEX*16 INTER(*)
      WRITE(*,*) 'GET_DENSITY is not available with --hel_recycling'
      STOP 1
      END
