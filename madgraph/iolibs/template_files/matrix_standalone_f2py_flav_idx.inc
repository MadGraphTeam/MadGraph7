C     ==================================================================
C     f2py entry points taking the flavor index (FLAV_IDX)
C     directly.
C
C     These exist because the FLAVOR(NEXTERNAL) array cannot
C     express a crossing: it holds unsigned group positions and
C     is resolved through GET_FLAVOR_INDEX, which only ever
C     returns 1..NFLAV. An *extended*
C     FLAV_IDX = K*NFLAV + flav carries both, so every entry
C     point a crossing-aware caller needs must take the index,
C     not the array.
C
C     Only emitted for matrix_standalone_v4.inc, the one
C     template that has GET_DENSITY_IDX / GET_ALL_INTER_IDX /
C     GET_PDG_FOR_FLAVOR at all: the other standalone templates
C     (matchbox, msP/msF, splitOrders) would fail to link
C     against routines they never generate.
C     ==================================================================

      SUBROUTINE PY_%(proc_prefix)sGET_PDG_FOR_FLAVOR(FLAV_IDX, PDGS)
C     Per-leg signed PDG codes of the process FLAV_IDX selects,
C     crossing included (legs permuted, and conjugated where
C     they swapped between the initial and the final state).
C
C     This is what lets a python caller work in PDG codes at
C     all: it can enumerate the candidate FLAV_IDX values, ask
C     each one what process it evaluates, and keep the one
C     matching the request. All-zero means the index names no
C     valid flavor/crossing.
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
C     The three constants a caller needs to build an extended
C     FLAV_IDX at all: FLAV_IDX = K*NFLAV + flav with K in
C     [0,NCROSS-1] a row of the crossing table of this matrix
C     element (K=0: no crossing; see PY_GET_CROSSING for what a
C     row does). Without NFLAV the encoding is simply not
C     expressible, and parsing it out of matrix.f is not
C     something a caller should have to do.
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

%(crossing_wrapper)s      SUBROUTINE PY_%(proc_prefix)sGET_NHEL_IDX(FLAV_IDX, IDEN_STAR,
     &     NHEL_STAR)
C     Crossing-aware twin of PY_GET_NHEL.
C
C     GET_NHEL reports the static IDEN, which is the averaging
C     denominator of the *uncrossed* representative flavor only.
C     SMATRIX itself does not use it that way -- it divides by
C     IDEN/BROKEN_SYM(FLAVOR) when uncrossed and by
C     GET_SPINCOL_CROSS*GET_IDENT_CROSS when crossed -- so a
C     caller reading GET_NHEL and reconstructing ANS*IDEN would
C     get a wrong answer for any crossed (or merely
C     non-representative) flavor. This entry reports the
C     denominator SMATRIX actually applied for this FLAV_IDX.
C     GET_NHEL keeps its signature and its meaning for existing
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
C     Density matrix for an extended FLAV_IDX. PY_GET_DENSITY
C     takes the FLAVOR array and so can only ever ask for an
C     uncrossed density matrix; this is the only way to request
C     a crossed one through f2py. Same CF2PY-before-declarations
C     layout and same explicit INTER sizing as PY_GET_DENSITY --
C     see the comments there, both matter.
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
C     The un-normalised interference terms behind
C     GET_DENSITY_IDX, for a caller supplying its own helicity
C     configuration. Exposed for the same reason: its
C     FLAVOR-array twin cannot carry a crossing.
C     P and POS follow the caller's legs, NHEL the base slots
C     (see GET_ALL_INTER_IDX): for a crossed FLAV_IDX pass
C     NHEL(XK) = helicity of caller leg PERM(XK).
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
