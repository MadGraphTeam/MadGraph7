C     Standalone helicity-recycling ORIG matrix element.
C     This file is NOT the final code: it is the input parsed by
C     madgraph/madevent/hel_recycle.py, which rewrites the MATRIX function
C     body (external/internal wavefunctions, amplitudes and the JAMP sum) into
C     the recycled form and injects it into the standalone _hel driver template
C     via the ${helas_calls} placeholder.  The structure below therefore
C     mirrors the madevent single-MATRIX layout the rewriter expects:
C       AMP(:) = 0  ->  helas calls  ->  JAMP(:) = 0  ->  jamp lines  ->
C       'if(init_mode)' terminator  ->  color sum,
C     plus the helicity (NHEL) table the rewriter reads.
C
C     The helicity table MUST come first: hel_recycle reads it to set the
C     per-leg helicity ranges before it parses the wavefunction calls.
      BLOCK DATA %(proc_prefix)sNHEL_DATA
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER NHEL(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sHEL_TABLE/NHEL
%(helicity_lines)s
      END
C
      DOUBLE PRECISION FUNCTION %(proc_prefix)sMATRIX(P, NHEL, FLAV_IDX)
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
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C     Reduced flavor index (1..NFLAV). FLAVOR(NEXTERNAL) is rebuilt from it by
C     the flavor block below, exactly as in the standard GET_AMP.
      INTEGER FLAV_IDX
      INTEGER FLAVOR(NEXTERNAL)
C
C LOCAL VARIABLES
C
      INTEGER I,J
      COMPLEX*16 ZTEMP, TMP_JAMP(%(nb_temp_jamp)i)
      INTEGER %(proc_prefix)sCF(%(ncolortriang)d)
      INTEGER %(proc_prefix)sDENOM, CF_INDEX
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      TYPE(ALOHA) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      double precision bwcutoff
C     Warm-up instrumentation (only active when init_mode is .true., set by the
C     hel_warmup driver). ZEROAMP(ihel,i) stays .true. while amplitude i has been
C     zero for helicity ihel over every sampled phase-space point; CUR_IHEL is the
C     helicity index the driver is currently evaluating.
      logical init_mode
      common/%(proc_prefix)sto_determine_zero_hel/init_mode
      logical %(proc_prefix)sZEROAMP(NCOMB, NGRAPHS)
      common/%(proc_prefix)sto_zeroamp/%(proc_prefix)sZEROAMP
      integer %(proc_prefix)sCUR_IHEL
      common/%(proc_prefix)sto_cur_ihel/%(proc_prefix)sCUR_IHEL
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
C     rebuild, identical to the standard GET_AMP: the recycled calls carry the
C     same IAND(CURRENT_*_MASK,...) guards.
%(flavor_mask_decl)s
C ----------
C BEGIN CODE
C ----------
      bwcutoff=15
      AMP(:) = (0D0,0D0)
%(flavor_mask_setup)s
%(helas_calls)s

      JAMP(:) = (0D0,0D0)
%(jamp_lines)s

      if(init_mode)then
        DO I=1, NGRAPHS
          if (AMP(I).ne.0) then
            %(proc_prefix)sZEROAMP(%(proc_prefix)sCUR_IHEL, I) = .false.
          endif
        ENDDO
      endif

      %(proc_prefix)sMATRIX = 0.D0
      CF_INDEX = 0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX + 1
          ZTEMP = ZTEMP + %(proc_prefix)sCF(CF_INDEX)*JAMP(J)
        ENDDO
        %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX
     &     + ZTEMP*DCONJG(JAMP(I))
      ENDDO
      %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX/%(proc_prefix)sDENOM
      END
