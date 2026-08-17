      SUBROUTINE %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS_SUMMED)
C
C     Simple routine wrapper to provide the same interface for
C     backward compatibility for usage without split orders.
C
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
      INTEGER FLAV_IDX
C
C VARIABLES
C
      INTEGER I
      REAL*8 ANS(0:NSQAMPSO)
C
C BEGIN CODE
C
      CALL %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
      ANS_SUMMED=ANS(0)

      END

      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
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
      CALL %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIXHEL_SPLITORDERS(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
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
      CALL %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
C
%(info_lines)s
C
C MadGraph StandAlone Version
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
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=%(hel_avg_factor)d)
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
      COMMON/%(proc_prefix)sCHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
C     FLAV_IDX (the dummy argument) is declared with NFLAV below.
C
C LOCAL VARIABLES
C
      INTEGER FLAVOR(NEXTERNAL)
      REAL*8 T(NSQAMPSO), BUFF
      INTEGER IHEL,IDEN, I, J
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/%(beamone_helavgfactor)d,%(beamtwo_helavgfactor)d/
      INTEGER JC(NEXTERNAL)
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER NNTRY_FLAV, NGOODHEL_FLAV
      PARAMETER (NNTRY_FLAV=NFLAV)
      PARAMETER (NGOODHEL_FLAV=NCOMB*NFLAV)
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
      INTEGER NTRY(NFLAV)
      LOGICAL GOODHEL(NCOMB,NFLAV)
      DATA NTRY/NNTRY_FLAV*0/
      DATA GOODHEL/NGOODHEL_FLAV*.FALSE./
C     C-parity helicity de-duplication (see the helicity loop below):
C     FLIP(IHEL) is the row with every helicity negated (built once by the
C     generator, an involution); TSTORE caches the per-squared-order |M|^2
C     vector within a scan point; DEDUP turns the reuse on in the fast phase
C     only. The reuse is ALL-OR-NOTHING per flavor: CSYM(J) stays true only
C     while EVERY row is in a genuine matched pair -- a self-paired row
C     (FLIP(IHEL)=IHEL) or a single pair whose |M|^2 differ at any scan point
C     clears it for good. Halving the whole loop (rather than an arbitrary
C     subset of rows) is what keeps any per-row accumulation uniformly
C     scaled, so the same rule is used by every backend.
C     Unlike matrix_standalone_v4.inc this needs no separate NTRY_CSYM: that
C     counter exists there only so a CROSSED call does not advance the scan
C     (a crossing permutes/sign-flips the helicities, so a base-row negation
C     is no longer the C-parity partner). This template carries no crossing
C     machinery at all, so NTRY is that counter.
C     The pair test runs per squared-order component, not on their sum: two
C     components could in principle cancel into an equal total, and the
C     doubling below is applied component by component.
      INTEGER FLIP(NCOMB)
      LOGICAL CSYM(NFLAV), DEDUP
      REAL*8 TSTORE(NSQAMPSO,NCOMB)
      REAL*8 TSMAX
%(flip_data)s
      DATA CSYM/NNTRY_FLAV*.TRUE./
%(den_factor_line)s
C
C GLOBAL VARIABLES
C
C     Materialized by FILL_NHEL, which is called below before the loop that
C     reads it (a DATA statement may not initialise a COMMON outside a BLOCK
C     DATA, which is what this template used to rely on).
      INTEGER NHEL(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL

C     DELIBERATE DIVERGENCE from matrix_standalone_v4.inc, which publishes
C     the canonical code list through a common and matches USERHEL against
C     it. (The routine names it and the other entry points this template
C     leaves out are spelled in the commit message and in
C     test_splitorders_template_carries_the_non_crossing_standalone_api, not
C     here: matrix_template_provides answers by searching the template text,
C     so naming a routine in a comment would make it claim the routine
C     exists.)
C     Here the external helicity label stays the ROW NUMBER, because that is
C     what this template's callers pass: MadLoop hands SMATRIXHEL_SPLITORDERS
C     its own USERHEL, which the EW-Sudakov driver fills from
C     `do chosen_hel=1,SDK_GET_NCOMB()` (Template/NLO/SubProcesses/
C     check_sudakov.f) -- a row index into the same enumeration. The two
C     labels coincide for every unpolarized process (the code of row i is
C     exactly i); they differ only under a {0}/{L}-style restriction, and
C     relabelling there would silently move the Born of a polarized NLO run.
      INTEGER USERHEL
      DATA USERHEL/-1/
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
      LOGICAL HELRESET
      common/%(proc_prefix)shelreset/HELRESET
      data HELRESET/.true./

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
      CSYM(i) = .true.
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
C     FLAV_IDX=0 (or out of range): GET_FLAVOR_INDEX could not resolve the
C     requested flavor -- it is not an allowed combination, so its matrix
C     element is identically zero. Short-circuit before touching the
C     1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS(:) = 0d0
        RETURN
      ENDIF
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX, FLAVOR)
      CALL %(proc_prefix)sFILL_NHEL()
      IF(USERHEL.EQ.-1) NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
      ANS(:) = 0d0
C     When spin-2 particles are involved, the Helicity filtering is dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero. But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the helicity sum is).
C     For this reason, we simply remove the filterin when there is only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        GOODHEL(:,:) = .True.
      ENDIF
C     C-parity de-duplication is only safe for the plain unpolarized helicity
C     sum and only once its own scan has settled (NTRY>=20).
      DEDUP = NTRY(FLAV_IDX).GE.20 .AND. CSYM(FLAV_IDX)
     &        .AND. USERHEL.EQ.-1 .AND. POLARIZATIONS(0,0).EQ.-1
      DO IHEL=1,NCOMB
         IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
           IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 20 .OR.USERHEL.NE.-1) THEN
              IF(NTRY(FLAV_IDX).GE.2.AND.POLARIZATIONS(0,0).ne.-1.and.(.not.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL))) THEN
                CYCLE
              ENDIF
C             Fast phase: a row whose fully flipped C-parity partner has an
C             identical |M|^2 (CSYM) is computed once, at the lower index,
C             and counted twice -- skip the higher-index partner here.
              IF (DEDUP.AND.IHEL.GT.FLIP(IHEL)) CYCLE
              CALL %(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX, T)
C             Scan phase: cache |M|^2 to test the C-parity partner below.
              IF (NTRY(FLAV_IDX).LT.20) THEN
                DO I=1,NSQAMPSO
                  TSTORE(I,IHEL)=T(I)
                ENDDO
              ENDIF
C             Fast phase: the representative carries its skipped partner's
C             identical contribution, component by component.
              IF (DEDUP.AND.IHEL.LT.FLIP(IHEL)) THEN
                DO I=1,NSQAMPSO
                  T(I)=T(I)+T(I)
                ENDDO
              ENDIF
              BUFF=0D0
              DO I=1,NSQAMPSO
                IF(POLARIZATIONS(0,0).eq.-1.or.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL)) THEN
                  ANS(I)=ANS(I)+T(I)
                ENDIF
                BUFF=BUFF+T(I)
              ENDDO
              IF (BUFF .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
                GOODHEL(IHEL,FLAV_IDX)=.TRUE.
              ENDIF
            ENDIF
          ENDIF
       ENDDO
C     Scan phase: drop the C-parity pairing of any row whose fully flipped
C     partner gave a different |M|^2 (parity/C violation). One mismatch at any
C     scan point permanently invalidates the pair (robust, like the zero-filter),
C     and because the verdict is all-or-nothing that turns the reuse off for the
C     whole flavor. There is no way to reach the fast phase without having run
C     this block: the two are gated on the same counter (NTRY<20 here,
C     NTRY>=20 for DEDUP), which nothing but this routine advances.
      IF (USERHEL.EQ.-1.AND.NTRY(FLAV_IDX).LT.20
     &    .AND.POLARIZATIONS(0,0).EQ.-1) THEN
C     A pair of rows that are BOTH numerically zero differ only by roundoff,
C     and a purely RELATIVE test on that noise fails at random -- one such pair
C     then vetoes every real pair. So also require the difference to be
C     significant against TSMAX, the largest |M|^2 of this scan point.
        TSMAX=0D0
        DO IHEL=1,NCOMB
          DO I=1,NSQAMPSO
            IF (ABS(TSTORE(I,IHEL)).GT.TSMAX) TSMAX=ABS(TSTORE(I,IHEL))
          ENDDO
        ENDDO
        DO IHEL=1,NCOMB
          IF (FLIP(IHEL).EQ.IHEL) THEN
C           Self-paired row: no distinct partner, so the loop cannot be
C           halved uniformly -- refuse the reuse for this flavor.
            CSYM(FLAV_IDX)=.FALSE.
          ELSE IF (FLIP(IHEL).GT.IHEL) THEN
            DO I=1,NSQAMPSO
              IF (ABS(TSTORE(I,IHEL)-TSTORE(I,FLIP(IHEL))).GT.
     &            1D-6*(ABS(TSTORE(I,IHEL))+ABS(TSTORE(I,FLIP(IHEL))))
     &            .AND.ABS(TSTORE(I,IHEL)-TSTORE(I,FLIP(IHEL))).GT.
     &            1D-12*TSMAX) THEN
                CSYM(FLAV_IDX)=.FALSE.
              ENDIF
            ENDDO
          ENDIF
        ENDDO
      ENDIF
      ANS(0)=0.0d0
      DO I=1,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)*%(proc_prefix)sBROKEN_SYM(FLAVOR)
        IF (CHOSEN_SO_CONFIGS(I)) THEN
          ANS(0)=ANS(0)+ANS(I)
        ENDIF
      ENDDO
      IF(USERHEL.NE.-1) THEN
        DO I=0,NSQAMPSO
          ANS(I)=ANS(I)*HELAVGFACTOR
        ENDDO
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).ne.-1) THEN
            DO I=0,NSQAMPSO
              ANS(I)=ANS(I)*BEAMS_HELAVGFACTOR(J)
              ANS(I)=ANS(I)/POLARIZATIONS(J,0)
            ENDDO
          ENDIF
        ENDDO
      ENDIF
      END


      SUBROUTINE %(proc_prefix)sMATRIX(P,NHEL,IC,FLAV_IDX,RES)
      use model_object
C
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
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
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      INTEGER %(proc_prefix)sDENOM
      INTEGER %(proc_prefix)sCF(%(ncolortriang)d)
      COMMON /%(proc_prefix)sCOLOR_MATRIX/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
      REAL*8 RES(NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N
      COMPLEX*16 ZTEMP
      COMPLEX*16 AMP(NGRAPHS)
      INTEGER CF_INDEX
      COMPLEX*16 JAMP(NCOLOR,NAMPSO)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C FUNCTION
C
      INTEGER %(proc_prefix)sSQSOINDEX
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
%(global_variable)s
C
C COLOR DATA
C
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
       call %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
c      WRITE (*,*) '  -> AMP = ', AMP
       call %(proc_prefix)sGET_JAMP(AMP,JAMP)
c      WRITE (*,*) '  -> JAMP = ', JAMP
       call %(proc_prefix)sGET_MATRIX(JAMP, RES)
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

      INTEGER NHEL(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
      INTEGER NHEL_STAR(NEXTERNAL,NCOMB)
      INTEGER IDEN,IDEN_STAR
%(den_factor_line)s
      CALL %(proc_prefix)sFILL_NHEL()
      IDEN_STAR = IDEN
      NHEL_STAR = NHEL
      END

      SUBROUTINE %(proc_prefix)sGET_NHEL_IDX(FLAV_IDX_IN,IDEN_STAR,
     & NHEL_STAR)
C     Same as GET_NHEL, but reporting the denominator SMATRIX_SPLITORDERS
C     ACTUALLY divides by for FLAV_IDX_IN, rather than the static IDEN.
C
C     The static IDEN is the averaging/symmetry factor of the *representative*
C     flavor. SMATRIX_SPLITORDERS never uses it bare: it applies
C     IDEN/BROKEN_SYM(FLAVOR), BROKEN_SYM correcting the identical-particle
C     count of the representative to that of the actual flavor. A caller that
C     reads GET_NHEL and multiplies ANS by it to recover the raw helicity/color
C     sum -- the natural thing to do -- is therefore wrong for every
C     non-representative flavor of a merged matrix element.
C
C     GET_NHEL is deliberately left alone: its signature and its value are
C     what existing callers expect.
C
C     Unlike matrix_standalone_v4.inc's twin this has no crossing branch:
C     this template carries no crossing machinery, so FLAV_IDX_IN is a plain
C     flavor index in [1,NFLAV] and only BROKEN_SYM can move the denominator.
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
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sBROKEN_SYM

      CALL %(proc_prefix)sGET_NHEL(IDEN_STAR, NHEL_STAR)
C     An index naming no valid flavor gives a zero matrix element (see the
C     guard at the top of SMATRIX_SPLITORDERS); report a 0 denominator rather
C     than a plausible-looking one.
      IF (FLAV_IDX_IN .LT. 1 .OR. FLAV_IDX_IN .GT. NFLAV) THEN
        IDEN_STAR = 0
        RETURN
      ENDIF
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX_IN, FLAVOR)
C     Mirrors SMATRIX_SPLITORDERS exactly: ANS=ANS/IDEN*BROKEN_SYM means the
C     effective denominator is IDEN/BROKEN_SYM. The division is exact --
C     BROKEN_SYM is the ratio of the representative to the actual
C     identical-particle count, and IDEN carries the representative one as a
C     factor.
      IDEN_STAR = IDEN_STAR / %(proc_prefix)sBROKEN_SYM(FLAVOR)
      RETURN
      END


SUBROUTINE %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
      use model_object
C
%(process_lines)s
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
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
      INTEGER FLAVOR(NEXTERNAL)
C
C LOCAL VARIABLES
C
      COMPLEX*16 AMP(NGRAPHS)
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
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      COMPLEX*16 AMP(NGRAPHS), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 JAMP(NCOLOR,NAMPSO)

      JAMP(:,:) = 0d0
%(jamp_lines)s
      END

      SUBROUTINE %(proc_prefix)sGET_MATRIX(JAMP, RES)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
CF2PY INTENT(OUT) :: MATRIX
CF2PY INTENT(IN) :: JAMP(NCOLOR,NAMPSO)


      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      REAL*8     ZERO,RES(NSQAMPSO)
      PARAMETER (ZERO=0D0)
      INTEGER %(proc_prefix)sSQSOINDEX
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N
      COMPLEX*16 ZTEMP

      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(proc_prefix)sDENOM
      common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 JAMP(NCOLOR,NAMPSO), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C COLOR DATA
C

     RES(:) = 0.D0
     DO M = 1, NAMPSO
       CF_INDEX= 0
       DO I = 1, NCOLOR
         ZTEMP = (0.D0,0.D0)
         DO J = I, NCOLOR
             CF_INDEX = CF_INDEX +1
             ZTEMP = ZTEMP + %(proc_prefix)sCF(CF_INDEX)*JAMP(J,M)
         ENDDO
         DO N = 1, NAMPSO
           RES(%(proc_prefix)sSQSOINDEX(M,N)) = RES(%(proc_prefix)sSQSOINDEX(M,N)) + REAL(ZTEMP*DCONJG(JAMP(I,N)))
         ENDDO
       ENDDO
     ENDDO
     RES(:) = RES(:)/%(proc_prefix)sDENOM
     END

      SUBROUTINE %(proc_prefix)sGET_INTER(JAMP_1,JAMP_2, INTER)
      IMPLICIT NONE
CF2PY INTENT(OUT) :: INTER(NSQAMPSO)
CF2PY INTENT(IN) :: JAMP_1(NCOLOR,NAMPSO)
CF2PY INTENT(IN) :: JAMP_2(NCOLOR,NAMPSO)

       INTEGER NAMPSO, NSQAMPSO
       PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
       INTEGER %(proc_prefix)sSQSOINDEX

       INTEGER I,J,M,N
       INTEGER NCOLOR
       PARAMETER (NCOLOR=%(ncolor)d)
       INTEGER CF_INDEX
       INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
       INTEGER %(proc_prefix)sDENOM, IDEN
       %(den_factor_line)s
       common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
       COMPLEX*16 JAMP_1(NCOLOR,NAMPSO),JAMP_2(NCOLOR,NAMPSO),ZTEMP,INTER(NSQAMPSO)
C
C COLOR DATA
C

      INTER(:) = 0.D0
      DO M = 1, NAMPSO
        CF_INDEX= 0
        DO I = 1, NCOLOR
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX +1
              DO N = 1, NAMPSO
                 INTER(%(proc_prefix)sSQSOINDEX(M,N)) =
     $            INTER(%(proc_prefix)sSQSOINDEX(M,N)) +
     $            %(proc_prefix)sCF(CF_INDEX)*(DCONJG(JAMP_2(I,N))*
     $            JAMP_1(J,M)+DCONJG(JAMP_2(J,N))*JAMP_1(I,M))
              ENDDO
           ENDDO
        ENDDO
      ENDDO
      INTER(:) = INTER(:)/(2d0*%(proc_prefix)sDENOM*IDEN)
      RETURN
      END

     SUBROUTINE  %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, MU_R2, INTER_SUM)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER((NCOMB*NCOMB+1)/2i,NSQAMPSO) : all interference term (not the symmetric one)
      use model_object
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAVOR(%(nexternal)d)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(OUT) :: INTER_SUM(N_COMB*(N_COMB+1)/2)
c

      INTEGER NB_NHEL
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NB_NHEL=%(ncomb)d)
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER THISNHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER NINTER
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE COMPLEX, ALLOCATABLE :: TMP_INTER(:,:)
      DOUBLE PRECISION ALPHAS, MU_R2
      DOUBLE COMPLEX INTER_SUM(*)
c LOCAL
      INTEGER I,IHEL,IPART
      INTEGER J
      DOUBLE PRECISION PI
C
C     GLOBAL
C
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      COMMON/%(proc_prefix)sCHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS

      INTEGER NHEL(NEXTERNAL,NB_NHEL),NTRY
C     put in common block to expose this variable to python interface
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
C
C     include coupling definition to update the value of alphas
C
      include 'coupl.inc'

      ninter = N_COMB*(N_COMB+1)/2
      allocate(TMP_INTER(NSQAMPSO, ninter))
      TMP_INTER(:,:) = (0d0, 0d0)

      do I = 1,  N_COMB*(N_COMB+1)/2
        INTER_SUM(I) = 0d0
      enddo

      IF (ALPHAS.ne.0d0) then
          pi = 3.141592653589793d0
          G = 2* DSQRT(ALPHAS*pi)
          call UPDATE_AS_PARAM()
      ENDIF
C     The NHEL table is materialized at runtime (see FILL_NHEL), so it has to
C     be filled here too: a caller reaching the density matrix before any
C     SMATRIX evaluation would otherwise read a table of zeros.
      CALL %(proc_prefix)sFILL_NHEL()

      DO IHEL =1, NB_NHEL
        THISNHEL(:) = NHEL(:, IHEL)
        DO IPART=1,N_CHANGING
            if(THISNHEL(POS(IPART)).NE.ALLOW_HEL(IPART)) GOTO 10 !BYPASS COMPUTATION FOR HELICITY
        ENDDO
        TMP_INTER(:,:) = 0
        call  %(proc_prefix)sGET_ALL_INTER(P, THISNHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, TMP_INTER)
        do J =1, NSQAMPSO
          IF (CHOSEN_SO_CONFIGS(J)) THEN
            do I = 1, N_COMB*(N_COMB+1)/2
              INTER_SUM(I) = INTER_SUM(I) + TMP_INTER(J,I)
            enddo
          endif
        enddo
 10   enddo
      return
      deallocate(TMP_INTER)
      end



      SUBROUTINE  %(proc_prefix)sGET_ALL_INTER(P, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, INTER)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER(NSO, (NCOMB*NCOMB+1)/2): all interference term (not the symmetric one)
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: NHEL(%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAVOR(%(nexternal)d)
CF2PY INTENT(OUT) :: INTER(NSQAMPSO,NCOMB*(NCOMB+1)/2)
c
C
C ARGUMENTS
C
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE COMPLEX INTER(NSQAMPSO,*)
c
c     Intermediate array
c
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      INTEGER IC(NEXTERNAL)

      DOUBLE COMPLEX AMP(NGRAPHS)
      DOUBLE COMPLEX, ALLOCATABLE:: JAMP(:,:,:)
      integer, save :: s_ncomb = -1
c
c     LOCAL
c
      INTEGER I,J,SOL,N
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
      INTEGER %(proc_prefix)sBROKEN_SYM
      DOUBLE PRECISION RESCALE

C ----------
C BEGIN CODE
C ----------
      if (allocated(jamp).and. n_comb.ne.s_ncomb)then
         deallocate(jamp)
      endif
      if (.not.allocated(jamp))then
        allocate(jamp(ncolor, NAMPSO, N_COMB))
        s_ncomb=n_comb
      endif
      IC(:)=1
      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
      do I = 1, N_COMB
         do N = 1, N_CHANGING
            NHEL(POS(N)) = ALLOW_HEL((I-1)*N_CHANGING+N)
         enddo
         call %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
         call %(proc_prefix)sGET_JAMP(AMP,JAMP(1,1,I))
      enddo

C     GET_INTER only sees JAMPs, so it normalises with the bare static IDEN
C     and cannot apply any flavor dependent factor. SMATRIX does
C     ANS/IDEN*BROKEN_SYM(FLAVOR); the density matrix must use the same
C     normalisation or the sum of its diagonal stops matching SMATRIX.
      RESCALE = DBLE(%(proc_prefix)sBROKEN_SYM(FLAVOR))
     SOL = 0
     DO I = 1, N_COMB
        DO J= I, N_COMB
          SOL = SOL +1
          call %(proc_prefix)sGET_INTER(JAMP(1,1,I), JAMP(1,1,J), INTER(1,SOL))
          INTER(:,SOL) = INTER(:,SOL) * RESCALE
        ENDDO
     ENDDO


     RETURN
     END




      SUBROUTINE %(proc_prefix)sGET_value(P, ALPHAS, NHEL, FLAVOR ,ANS)
C     f2py interface accepting the full FLAVOR(NEXTERNAL) array (back-compat):
C     resolve it to FLAV_IDX and forward to GET_value_idx.
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

      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
      CALL %(proc_prefix)sGET_value_idx(P, ALPHAS, NHEL, FLAV_IDX, ANS)
      return
      end

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
      real*8 pi
      INTEGER FLAV_IDX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAV_IDX
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


      SUBROUTINE %(proc_prefix)sDECODE_HEL(CODE, THISNHEL)
C     Decode a canonical mixed-radix helicity CODE (1..NCOMBFULL) into the
C     per-leg helicity values THISNHEL(NEXTERNAL). The last external leg is the
C     least-significant digit, matching the itertools.product ordering used to
C     build the allowed-code list HELALLOW.
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    MAXHEL
      PARAMETER (MAXHEL=%(maxhel)d)
      INTEGER CODE, THISNHEL(NEXTERNAL)
      INTEGER I, K, R, D
      INTEGER NHSTATE(NEXTERNAL), STATES(MAXHEL,NEXTERNAL)
%(nhstate_data)s
%(states_data)s
      R = CODE - 1
      DO K=NEXTERNAL,1,-1
        D = MOD(R, NHSTATE(K))
        THISNHEL(K) = STATES(D+1, K)
        R = R / NHSTATE(K)
      ENDDO
      RETURN
      END

      SUBROUTINE %(proc_prefix)sFILL_NHEL()
C     Materialize the PROCESS_NHEL config table by decoding the list of allowed
C     canonical helicity codes (HELALLOW). Runs once; the table is a runtime
C     cache of the encoder representation, kept for the density-matrix and
C     python (f2py) interfaces.
C
C     The inverse (code from helicity values) that matrix_standalone_v4.inc
C     also carries is deliberately NOT written here: nothing in the tree calls
C     it -- the crossing routines that would are not part of this template, and
C     they inline their own encode anyway. A split-orders matrix.f is written
C     once per MadLoop / FKS born as well as per standalone subprocess, so a
C     dead routine there is paid for many times over.
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER NHEL(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
C     A DATA statement may not initialise a COMMON outside a BLOCK DATA, so
C     the list is a local DATA here and the table it decodes into is filled on
C     the first call -- every reader is ordered behind that call.
C     matrix_standalone_v4.inc additionally publishes this list through a
C     common of its own, because there the external helicity label IS the
C     code; here it is the row number (see SMATRIX_SPLITORDERS), so that
C     common would have no reader.
      INTEGER HELALLOW(NCOMB)
      INTEGER I, K, THIS(NEXTERNAL)
      LOGICAL DONE
      SAVE DONE
%(hel_allow_data)s
      DATA DONE /.FALSE./
      IF (DONE) RETURN
      DO I=1,NCOMB
        CALL %(proc_prefix)sDECODE_HEL(HELALLOW(I), THIS)
        DO K=1,NEXTERNAL
          NHEL(K,I) = THIS(K)
        ENDDO
      ENDDO
      DONE = .TRUE.
      RETURN
      END


%(broken_sym_function)s


%(flavor_index_function)s


%(flavor_array_function)s
