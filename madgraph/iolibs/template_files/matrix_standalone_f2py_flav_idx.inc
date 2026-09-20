C     ==================================================================
C     f2py entry points taking the flavor index (FLAV_IDX) directly.
C
C     These exist because the FLAVOR(NEXTERNAL) array cannot express a
C     crossing: it holds unsigned group positions and is resolved through
C     GET_FLAVOR_INDEX, which only ever returns 1..NFLAV. An *extended*
C     FLAV_IDX = cross*NFLAV + flav carries both, so every entry point a
C     crossing-aware caller needs must take the index, not the array.
C
C     Only emitted for matrix_standalone_v4.inc, the one template that has
C     GET_DENSITY_IDX / GET_ALL_INTER_IDX / GET_PDG_FOR_FLAVOR at all: the
C     other standalone templates (matchbox, msP/msF, splitOrders) would
C     fail to link against routines they never generate.
C     ==================================================================

      SUBROUTINE PY_%(proc_prefix)sGET_PDG_FOR_FLAVOR(FLAV_IDX, PDGS)
C     Per-leg signed PDG codes of the process FLAV_IDX selects, crossing
C     included (legs permuted, and conjugated where they swapped between
C     the initial and the final state).
C
C     This is what lets a python caller work in PDG codes at all: it can
C     enumerate the candidate FLAV_IDX values, ask each one what process
C     it evaluates, and keep the one matching the request. All-zero means
C     the index names no valid flavor/crossing.
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
CF2PY INTENT(IN) :: FLAV_IDX
CF2PY INTENT(OUT) :: PDGS(NEXTERNAL)
      INTEGER FLAV_IDX
      INTEGER PDGS(NEXTERNAL)
      CALL %(proc_prefix)sGET_PDG_FOR_FLAVOR(FLAV_IDX, PDGS)
      RETURN
      END

      SUBROUTINE PY_%(proc_prefix)sGET_FLAVOR_LAYOUT(NFLAV_OUT,
     &     NEXTERNAL_OUT, NCROSS_OUT)
C     The three constants a caller needs to build an extended FLAV_IDX at
C     all: FLAV_IDX = cross*NFLAV + flav with cross in [0,NCROSS-1], and
C     cross = I*(NEXTERNAL+1)+J. Without NFLAV the encoding is simply not
C     expressible, and parsing it out of matrix.f (as the tests must) is
C     not something a caller should have to do.
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NFLAV
      PARAMETER (NFLAV=%(nflav)d)
CF2PY INTENT(OUT) :: NFLAV_OUT
CF2PY INTENT(OUT) :: NEXTERNAL_OUT
CF2PY INTENT(OUT) :: NCROSS_OUT
      INTEGER NFLAV_OUT, NEXTERNAL_OUT, NCROSS_OUT
      NFLAV_OUT = NFLAV
      NEXTERNAL_OUT = NEXTERNAL
      NCROSS_OUT = %(ncross)d
      RETURN
      END

      SUBROUTINE PY_%(proc_prefix)sGET_NHEL_IDX(FLAV_IDX, IDEN_STAR,
     &     NHEL_STAR)
C     Crossing-aware twin of PY_GET_NHEL.
C
C     GET_NHEL reports the static IDEN, which is the averaging denominator
C     of the *uncrossed* representative flavor only. SMATRIX itself does
C     not use it that way -- it divides by IDEN/BROKEN_SYM(FLAVOR) when
C     uncrossed and by GET_SPINCOL_CROSS*GET_IDENT_CROSS when crossed --
C     so a caller reading GET_NHEL and reconstructing ANS*IDEN would get a
C     wrong answer for any crossed (or merely non-representative) flavor.
C     This entry reports the denominator SMATRIX actually applied for this
C     FLAV_IDX. GET_NHEL keeps its signature and its meaning for existing
C     uncrossed callers.
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
CF2PY INTENT(IN) :: FLAV_IDX
CF2PY INTENT(OUT) :: IDEN_STAR
CF2PY INTENT(OUT) :: NHEL_STAR(NEXTERNAL,NCOMB)
      INTEGER FLAV_IDX
      INTEGER IDEN_STAR
      INTEGER NHEL_STAR(NEXTERNAL,NCOMB)
      CALL %(proc_prefix)sGET_NHEL_IDX(FLAV_IDX, IDEN_STAR, NHEL_STAR)
      RETURN
      END

      SUBROUTINE PY_%(proc_prefix)sGET_DENSITY_IDX(P, POS, N_CHANGING,
     &     ALLOW_HEL, N_COMB, FLAV_IDX, ALPHAS, SCALE2, INTER)
C     Density matrix for an extended FLAV_IDX. PY_GET_DENSITY takes the
C     FLAVOR array and so can only ever ask for an uncrossed density
C     matrix; this is the only way to request a crossed one through f2py.
C     Same CF2PY-before-declarations layout and same explicit INTER sizing
C     as PY_GET_DENSITY -- see the comments there, both matter.
      IMPLICIT NONE
CF2PY double precision, intent(in), dimension(0:3,%(nexternal)d) :: P
CF2PY integer, intent(in), dimension(*) :: POS
CF2PY integer, intent(in) :: N_CHANGING
CF2PY integer, intent(in), dimension(N_CHANGING*N_COMB) :: ALLOW_HEL
CF2PY integer, intent(in) :: N_COMB
CF2PY integer, intent(in) :: FLAV_IDX
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
CF2PY double complex, intent(out), dimension(N_COMB*(N_COMB+1)/2) :: INTER
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAV_IDX
      DOUBLE PRECISION ALPHAS, SCALE2
      DOUBLE COMPLEX INTER(N_COMB*(N_COMB+1)/2)
      CALL %(proc_prefix)sGET_DENSITY_IDX(P, POS, N_CHANGING, ALLOW_HEL,
     &     N_COMB, FLAV_IDX, ALPHAS, SCALE2, INTER)
      RETURN
      END

      SUBROUTINE PY_%(proc_prefix)sGET_ALL_INTER_IDX(P, NHEL, POS,
     &     N_CHANGING, ALLOW_HEL, N_COMB, FLAV_IDX, INTER)
C     The un-normalised interference terms behind GET_DENSITY_IDX, for a
C     caller supplying its own helicity configuration. Exposed for the
C     same reason: its FLAVOR-array twin cannot carry a crossing.
      IMPLICIT NONE
CF2PY double precision, intent(in), dimension(0:3,%(nexternal)d) :: P
CF2PY integer, intent(in), dimension(%(nexternal)d) :: NHEL
CF2PY integer, intent(in), dimension(*) :: POS
CF2PY integer, intent(in) :: N_CHANGING
CF2PY integer, intent(in), dimension(N_CHANGING*N_COMB) :: ALLOW_HEL
CF2PY integer, intent(in) :: N_COMB
CF2PY integer, intent(in) :: FLAV_IDX
CF2PY double complex, intent(out), dimension(N_COMB*(N_COMB+1)/2) :: INTER
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAV_IDX
      DOUBLE COMPLEX INTER(N_COMB*(N_COMB+1)/2)
      CALL %(proc_prefix)sGET_ALL_INTER_IDX(P, NHEL, POS, N_CHANGING,
     &     ALLOW_HEL, N_COMB, FLAV_IDX, INTER)
      RETURN
      END
