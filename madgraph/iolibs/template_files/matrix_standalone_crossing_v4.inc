      SUBROUTINE %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX_IN, PERM, SGN,
     & FLAV_IDX)
C     Decode the crossing carried by FLAV_IDX_IN into a slot
C     permutation.
C
C       CROSS    = (FLAV_IDX_IN-1) / NFLAV       ! row of the
C       crossing table
C       FLAV_IDX = mod(FLAV_IDX_IN-1, NFLAV) + 1 ! used for
C       masking/...
C
C     The crossing table is generated for this matrix element:
C     row CROSS is a full slot permutation, row 0 the identity,
C     so FLAV_IDX_IN in [1,NFLAV] keeps old callers untouched.
C     It holds only the crossings this matrix element serves
C     (see madgraph/iolibs/crossing_table.py), in the base-slot
C     view the matrix element needs:
C       PERM(L) : the input slot whose momentum lands in base slot
C       L,
C       SGN(L)  : -1 when that leg crosses between the initial and
C       the final
C                 state (the NSF/NSV flag of base slot L), else +1.
C     GET_CROSS_PINV gives the same row in the input-slot view
C     (PDG, denominator). Flipping the NSF/NSV flag is what
C     crosses a leg: helas stores the momentum as p*nsf and uses
C     nhel*nsf, so the momentum sign change and the helicity
C     flip both follow from it. Momenta therefore stay physical
C     (positive energy), which matters because the helas spinors
C     take dsqrt(p(0)+pp).
C
C     FLAV_IDX comes back 0 for an index beyond the table (or
C     for a row that names no crossing); PERM/SGN are then the
C     identity, so a caller that gathers momenta with them never
C     reads out of range; 0 is what GET_PDG_FOR_FLAVOR and
C     GET_AMP already treat as "not a flavor".
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER NCROSS
      PARAMETER (NCROSS=%(ncross)d)
C     ARGUMENTS
      INTEGER FLAV_IDX_IN
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL)
      INTEGER FLAV_IDX
C     LOCAL
      INTEGER CROSS, XK, I
      INTEGER XPERM(0:NCROSS*NEXTERNAL-1), XSGN(0:NCROSS*NEXTERNAL-1)
      INTEGER XVALID(0:NCROSS-1)
%(xperm_data)s
%(xsgn_data)s
%(xvalid_data)s

      FLAV_IDX = MOD(FLAV_IDX_IN-1, NFLAV) + 1
      CROSS = (FLAV_IDX_IN-1) / NFLAV
      DO XK = 1, NEXTERNAL
        PERM(XK) = XK
        SGN(XK) = 1
      ENDDO
      IF (FLAV_IDX_IN .LT. 1 .OR. CROSS .GT. NCROSS-1) THEN
        FLAV_IDX = 0
        RETURN
      ENDIF
      IF (XVALID(CROSS) .EQ. 0) THEN
        FLAV_IDX = 0
        RETURN
      ENDIF
      DO XK = 1, NEXTERNAL
        PERM(XK) = XPERM(CROSS*NEXTERNAL+XK-1)
        SGN(XK) = XSGN(CROSS*NEXTERNAL+XK-1)
      ENDDO

      RETURN
      END


      SUBROUTINE %(proc_prefix)sGET_CROSS_PINV(FLAV_IDX_IN, PINV, SGNI,
     & FLAV_IDX)
C     The crossing of FLAV_IDX_IN in the input-slot view (see
C     GET_CROSS_PERM):
C       PINV(K) : the base slot receiving input slot K (the
C       inverse of PERM),
C       SGNI(K) : -1 when input slot K is a crossed leg, else +1.
C     The crossed process sees base leg PINV(K), charge
C     conjugated when SGNI(K)=-1, in its slot K: this is what
C     the PDG signature, the spin*color average and the
C     identical-particle factor are built from. Both views are
C     needed because a crossing is in general not an involution
C     (u u~ > g g off g g > u u~ is a 4-cycle).
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER NCROSS
      PARAMETER (NCROSS=%(ncross)d)
C     ARGUMENTS
      INTEGER FLAV_IDX_IN
      INTEGER PINV(NEXTERNAL), SGNI(NEXTERNAL)
      INTEGER FLAV_IDX
C     LOCAL
      INTEGER CROSS, XK, I
      INTEGER XPINV(0:NCROSS*NEXTERNAL-1), XSGNI(0:NCROSS*NEXTERNAL-1)
      INTEGER XVALID(0:NCROSS-1)
%(xpinv_data)s
%(xsgni_data)s
%(xvalid_data)s

      FLAV_IDX = MOD(FLAV_IDX_IN-1, NFLAV) + 1
      CROSS = (FLAV_IDX_IN-1) / NFLAV
      DO XK = 1, NEXTERNAL
        PINV(XK) = XK
        SGNI(XK) = 1
      ENDDO
      IF (FLAV_IDX_IN .LT. 1 .OR. CROSS .GT. NCROSS-1) THEN
        FLAV_IDX = 0
        RETURN
      ENDIF
      IF (XVALID(CROSS) .EQ. 0) THEN
        FLAV_IDX = 0
        RETURN
      ENDIF
      DO XK = 1, NEXTERNAL
        PINV(XK) = XPINV(CROSS*NEXTERNAL+XK-1)
        SGNI(XK) = XSGNI(CROSS*NEXTERNAL+XK-1)
      ENDDO

      RETURN
      END


      SUBROUTINE %(proc_prefix)sAPPLY_CROSSING_TABLE(FLAV_IDX_IN, NROW,
     & P_IN, NHEL_IN, IC_IN, P, NHEL, IC, FLAV_IDX)
C     Apply the crossing carried by FLAV_IDX_IN to one set of
C     momenta / NSF flags and to NROW helicity rows at once (see
C     GET_CROSS_PERM).
C
C     SMATRIX uses this to permute its whole NHEL table in a
C     single sweep before the helicity loop: the permutation
C     does not depend on the row, so decoding it once per
C     SMATRIX call rather than once per helicity is both cheaper
C     and the reason MATRIX/GET_AMP can stay pure. P/NHEL/IC
C     must not alias P_IN/NHEL_IN/IC_IN.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
C     ARGUMENTS
      INTEGER FLAV_IDX_IN, NROW
      REAL*8 P_IN(0:3,NEXTERNAL)
      INTEGER NHEL_IN(NEXTERNAL,NROW), IC_IN(NEXTERNAL)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL,NROW), IC(NEXTERNAL)
      INTEGER FLAV_IDX
C     LOCAL
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL)
      INTEGER XK, XR

      CALL %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX_IN, PERM, SGN, FLAV_IDX)

      DO XK = 1, NEXTERNAL
        P(0,XK) = P_IN(0,PERM(XK))
        P(1,XK) = P_IN(1,PERM(XK))
        P(2,XK) = P_IN(2,PERM(XK))
        P(3,XK) = P_IN(3,PERM(XK))
        IC(XK) = SGN(XK)*IC_IN(PERM(XK))
      ENDDO
C     The helicity slots are NOT permuted: IC stays aligned with
C     the MOMENTA -- it is also the NSF direction flag -- so
C     slot XK gets P_IN(PERM(XK)) and NHEL_IN(XK), and a crossed
C     leg is evaluated with the physical helicity NHEL_IN(XK) of
C     the particle it has become. This is the view of a caller
C     holding helicities of the crossed process per base slot
C     (GET_DENSITY_IDX's table, an f2py caller before GET_AMP;
C     GET_ALL_INTER_IDX takes them in the caller's leg order
C     and gives base slot XK NHEL(PERM(XK)) itself). SMATRIX
C     evaluates its own base table instead, through
C     APPLY_CROSSING_ROWS.
      DO XR = 1, NROW
        DO XK = 1, NEXTERNAL
          NHEL(XK,XR) = NHEL_IN(XK,XR)
        ENDDO
      ENDDO

      RETURN
      END


      SUBROUTINE %(proc_prefix)sAPPLY_CROSSING_ROWS(FLAV_IDX_IN, NROW,
     & P_IN, NHEL_IN, IC_IN, P, NHEL, IC, FLAV_IDX)
C     APPLY_CROSSING_TABLE for the BASE helicity table SMATRIX
C     loops over: row H is one helicity configuration of the
C     base process read with every leg outgoing, and each
C     crossing evaluates that SAME configuration. A leg that
C     changes side therefore carries the opposite physical
C     helicity, NHEL_IN(XK,H)*SGN(XK) (SGN also flips its NSF
C     flag, see GET_CROSS_PERM). The crossed amplitude is the
C     analytic continuation of the base one, so a row vanishes
C     for every crossing when it vanishes for the base: one
C     good-helicity filter serves them all. The exception is a
C     massive leg, whose helicity depends on the frame (each
C     crossing is evaluated in its own); each crossing therefore
C     still trains the shared filter itself. CROSS_HELCODE
C     translates the crossed process's own helicity code into
C     the row H that evaluates it.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
C     ARGUMENTS
      INTEGER FLAV_IDX_IN, NROW
      REAL*8 P_IN(0:3,NEXTERNAL)
      INTEGER NHEL_IN(NEXTERNAL,NROW), IC_IN(NEXTERNAL)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL,NROW), IC(NEXTERNAL)
      INTEGER FLAV_IDX
C     LOCAL
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL)
      INTEGER XK, XR, XF

      CALL %(proc_prefix)sAPPLY_CROSSING_TABLE(FLAV_IDX_IN, NROW, P_IN,
     & NHEL_IN, IC_IN, P, NHEL, IC, FLAV_IDX)
      CALL %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX_IN, PERM, SGN, XF)
      DO XR = 1, NROW
        DO XK = 1, NEXTERNAL
          NHEL(XK,XR) = SGN(XK)*NHEL(XK,XR)
        ENDDO
      ENDDO

      RETURN
      END


      SUBROUTINE %(proc_prefix)sAPPLY_CROSSING(FLAV_IDX_IN, P_IN, NHEL_IN,
     & IC_IN, P, NHEL, IC, FLAV_IDX)
C     Single helicity row flavour of APPLY_CROSSING_TABLE; kept
C     public so that an external (f2py) caller holding an
C     extended FLAV_IDX can pre-apply the crossing before
C     calling GET_AMP, which is pure and rejects an extended
C     index (see its contract).
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER FLAV_IDX_IN
      REAL*8 P_IN(0:3,NEXTERNAL)
      INTEGER NHEL_IN(NEXTERNAL), IC_IN(NEXTERNAL)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX

      CALL %(proc_prefix)sAPPLY_CROSSING_TABLE(FLAV_IDX_IN, 1, P_IN,
     & NHEL_IN, IC_IN, P, NHEL, IC, FLAV_IDX)

      RETURN
      END


      INTEGER FUNCTION %(proc_prefix)sGET_SPINCOL_CROSS(CROSS)
C     Initial-state spin*color average of the process
C     crossing-table row CROSS crosses into: the product of the
C     per-leg spin*color (conjugation invariant) of the base
C     legs the row puts in the initial slots, computed once at
C     generation (XSPINCOL). A row beyond the table, or one that
C     names no crossing, returns 0, which SMATRIX /
C     GET_PDG_FOR_FLAVOR / GET_ALL_INTER_CROSSED all map to a
C     null result: this is the single gate that says whether a
C     CROSS row is applicable at all. The flavor-dependent half
C     is GET_IDENT_CROSS.
      IMPLICIT NONE
      INTEGER NCROSS
      PARAMETER (NCROSS=%(ncross)d)
      INTEGER CROSS, I
      INTEGER XSPINCOL(0:NCROSS-1)
%(xspincol_data)s

      IF (CROSS .LT. 0 .OR. CROSS .GT. NCROSS-1) THEN
        %(proc_prefix)sGET_SPINCOL_CROSS = 0
        RETURN
      ENDIF
      %(proc_prefix)sGET_SPINCOL_CROSS = XSPINCOL(CROSS)

      RETURN
      END


      INTEGER FUNCTION %(proc_prefix)sGET_HELAVG_CROSS(CROSS, BEAM)
C     Helicity states of initial leg BEAM (1 or 2) of the
C     process row CROSS crosses into, or of both for BEAM=0:
C     the HELAVGFACTOR / BEAMS_HELAVGFACTOR of that process. A
C     crossing that brings a massive vector or a scalar into
C     the initial state changes it, so a single helicity code
C     (SMATRIXHEL) or a polarised beam undoes this average,
C     not the base process's.
      IMPLICIT NONE
      INTEGER NCROSS
      PARAMETER (NCROSS=%(ncross)d)
      INTEGER CROSS, BEAM, I
      INTEGER XBEAMHEL(0:2*NCROSS-1)
%(xbeamhel_data)s

      %(proc_prefix)sGET_HELAVG_CROSS = 0
      IF (CROSS .LT. 0 .OR. CROSS .GT. NCROSS-1) RETURN
      IF (BEAM .EQ. 0) THEN
        %(proc_prefix)sGET_HELAVG_CROSS = XBEAMHEL(2*CROSS)
     &   * XBEAMHEL(2*CROSS+1)
      ELSE IF (BEAM .EQ. 1 .OR. BEAM .EQ. 2) THEN
        %(proc_prefix)sGET_HELAVG_CROSS = XBEAMHEL(2*CROSS+BEAM-1)
      ENDIF

      RETURN
      END


      INTEGER FUNCTION %(proc_prefix)sGET_IDENT_CROSS(CROSS, FLAVOR)
C     Identical final state factor (product of n!) of the
C     crossed process.
C
C     Flavor dependent, hence computed here rather than
C     tabulated on CROSS: d d~ > g u u~ crossed gives d g > d u
C     u~ with nothing identical, while d d~ > g d d~ crossed
C     gives d g > d d d~ with two identical d. BROKEN_SYM cannot
C     be reused for this: its tables describe the uncrossed
C     final state.
C
C     Two crossed final legs are identical when they carry the
C     same flavor group (same representative PDG, conjugated
C     already when the leg swapped side) and the same position
C     inside it. Neither the per-slot representative PDG nor the
C     FLAVOR source slot is tabulated per crossing: both follow
C     from the table row (GET_CROSS_PINV) applied to two
C     NEXTERNAL-long base tables -- IDS_BASE, the base PDG of
C     each leg, and ANTIPID_BASE, its charge conjugate for a leg
C     that swapped side.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER CROSS
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER IDS_BASE(0:NEXTERNAL-1)
      INTEGER ANTIPID_BASE(0:NEXTERNAL-1)
C     COUNTABLE(leg)=1 for a single external leg, 0 for a leaf
C     locked inside a decay block. A crossing never moves a
C     block leaf (the crossing table has no row that would), so
C     decay products keep their base slots and must be skipped
C     here: their contribution to the identical-final factor is
C     resonance-level, carried whole by IDENT_RESONANCE. For a
C     non-decay process every leg is countable and
C     IDENT_RESONANCE is 1, so this is the plain leaf count it
C     always was.
      INTEGER COUNTABLE(0:NEXTERNAL-1)
      INTEGER IDENT_RESONANCE
      PARAMETER (IDENT_RESONANCE=%(ident_resonance)d)
%(iden_cross_lines)s
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER K, L, N, FACT, I, XIDX
      INTEGER PINV(NEXTERNAL), SGNI(NEXTERNAL), BPID(NEXTERNAL)
      LOGICAL USED(NEXTERNAL)

C     The crossed process sees base leg PINV(K) in its slot K,
C     conjugated when SGNI(K)=-1 (GET_CROSS_PINV): read each
C     slot's representative PDG straight off IDS_BASE
C     (ANTIPID_BASE where the leg changed side).
      CALL %(proc_prefix)sGET_CROSS_PINV(CROSS*NFLAV+1, PINV, SGNI, XIDX)
      IF (XIDX .LT. 1) THEN
        %(proc_prefix)sGET_IDENT_CROSS = 0
        RETURN
      ENDIF
      DO K = 1, NEXTERNAL
        IF (SGNI(K).EQ.1) THEN
          BPID(K) = IDS_BASE(PINV(K)-1)
        ELSE
          BPID(K) = ANTIPID_BASE(PINV(K)-1)
        ENDIF
      ENDDO

C     FLAVOR is not permuted by the crossing, so slot K reads
C     FLAVOR(PINV(K)), the actual flavor of the base leg that
C     moved into it. Start from IDENT_RESONANCE (the
C     resonance-level part of the identical factor, which a
C     crossing never changes) and multiply in the n! of the
C     countable final legs only -- a decay-block leaf is
C     skipped, its symmetry already inside IDENT_RESONANCE.
      DO K = 1, NEXTERNAL
        USED(K) = .FALSE.
      ENDDO
      FACT = IDENT_RESONANCE
      DO K = NINCOMING+1, NEXTERNAL
        IF (USED(K)) CYCLE
        IF (COUNTABLE(PINV(K)-1).EQ.0) CYCLE
        N = 1
        DO L = K+1, NEXTERNAL
          IF (USED(L)) CYCLE
          IF (COUNTABLE(PINV(L)-1).EQ.0) CYCLE
          IF (BPID(K).EQ.BPID(L) .AND.
     $        FLAVOR(PINV(K)).EQ.FLAVOR(PINV(L))) THEN
            USED(L) = .TRUE.
            N = N + 1
            FACT = FACT * N
          ENDIF
        ENDDO
      ENDDO
      %(proc_prefix)sGET_IDENT_CROSS = FACT

      RETURN
      END

      SUBROUTINE %(proc_prefix)sCROSS_HELCODE(FLAV_IDX_IN, CODE, BCODE)
C     BCODE: the base helicity code SMATRIX evaluates for the
C     helicity code CODE of the process FLAV_IDX_IN names. CODE
C     is that process's OWN canonical code -- the one its
C     expanded output takes, the one the madevent output writes
C     for a routed event and the C++ backends report for a
C     crossed one. Input slot K of the crossed process carries
C     base leg B=PINV(K), with the helicity states of that leg
C     in the same order, and SMATRIX evaluates a base row with
C     the side-changing legs' helicities negated
C     (APPLY_CROSSING_ROWS): the crossed leg's helicity H is
C     base value SGNI(K)*H. An uncrossed index, or a CODE < 1
C     (no helicity requested), comes back unchanged; so does a
C     CODE above NCOMB, which names no row and answers 0, as
C     in the expanded output.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER    MAXHEL
      PARAMETER (MAXHEL=%(maxhel)d)
      INTEGER FLAV_IDX_IN, CODE, BCODE
      INTEGER PINV(NEXTERNAL), SGNI(NEXTERNAL), XF, I, K, R, B, V
      INTEGER NC
      INTEGER DIG(NEXTERNAL)
      INTEGER NHSTATE(NEXTERNAL), STATES(MAXHEL,NEXTERNAL)
%(nhstate_data)s
%(states_data)s
      BCODE = CODE
      IF (CODE .LT. 1) RETURN
C     the digit decoding below would wrap a code above NCOMB
      NC = 1
      DO K = 1, NEXTERNAL
        NC = NC*NHSTATE(K)
      ENDDO
      IF (CODE .GT. NC) RETURN
      CALL %(proc_prefix)sGET_CROSS_PINV(FLAV_IDX_IN, PINV, SGNI, XF)
      R = CODE - 1
      DO K = NEXTERNAL, 1, -1
        B = PINV(K)
        V = SGNI(K)*STATES(MOD(R, NHSTATE(B))+1, B)
        R = R / NHSTATE(B)
        DIG(B) = 0
        DO I = 1, NHSTATE(B)
          IF (STATES(I,B).EQ.V) DIG(B) = I-1
        ENDDO
      ENDDO
      R = 0
      DO K = 1, NEXTERNAL
        R = R*NHSTATE(K) + DIG(K)
      ENDDO
      BCODE = R + 1

      RETURN
      END

      SUBROUTINE %(proc_prefix)sCROSS_NHEL_TABLE(FLAV_IDX_IN, NCOMB_IN,
     & NHEL_OUT)
C     The helicity table of the process FLAV_IDX_IN names, in
C     its own leg order: column C holds the helicities of its
C     OWN code C, the code SMATRIXHEL takes (the inverse of
C     CROSS_HELCODE). Slot K carries base leg PINV(K), with the
C     states of that leg in the same order, so an uncrossed
C     index gives back the base table.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER    MAXHEL
      PARAMETER (MAXHEL=%(maxhel)d)
      INTEGER FLAV_IDX_IN, NCOMB_IN
      INTEGER NHEL_OUT(NEXTERNAL,NCOMB_IN)
      INTEGER PINV(NEXTERNAL), SGNI(NEXTERNAL), XF, I, K, R, B, C
      INTEGER NHSTATE(NEXTERNAL), STATES(MAXHEL,NEXTERNAL)
%(nhstate_data)s
%(states_data)s
      CALL %(proc_prefix)sGET_CROSS_PINV(FLAV_IDX_IN, PINV, SGNI, XF)
      DO C = 1, NCOMB_IN
        R = C - 1
        DO K = NEXTERNAL, 1, -1
          B = PINV(K)
          NHEL_OUT(K,C) = STATES(MOD(R, NHSTATE(B))+1, B)
          R = R / NHSTATE(B)
        ENDDO
      ENDDO

      RETURN
      END
