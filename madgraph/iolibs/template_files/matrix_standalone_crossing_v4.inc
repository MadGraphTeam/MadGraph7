      SUBROUTINE %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX_IN, PERM, SGN,
     & FLAV_IDX)
C     Decode the crossing carried by FLAV_IDX_IN into a slot permutation.
C
C       CROSS    = (FLAV_IDX_IN-1) / NFLAV       ! row of the crossing table
C       FLAV_IDX = mod(FLAV_IDX_IN-1, NFLAV) + 1 ! used for masking/...
C
C     The crossing table is generated for this matrix element: row CROSS is a
C     full slot permutation, row 0 the identity, so FLAV_IDX_IN in [1,NFLAV]
C     keeps old callers untouched. It holds only the crossings this matrix
C     element serves (see madgraph/iolibs/crossing_table.py), in the base-slot
C     view the matrix element needs:
C       PERM(L) : the input slot whose momentum lands in base slot L,
C       SGN(L)  : -1 when that leg crosses between the initial and the final
C                 state (the NSF/NSV flag of base slot L), else +1.
C     GET_CROSS_PINV gives the same row in the input-slot view (PDG,
C     denominator). Flipping the NSF/NSV flag is what crosses a leg: helas
C     stores the momentum as p*nsf and uses nhel*nsf, so the momentum sign
C     change and the helicity flip both follow from it. Momenta therefore stay
C     physical (positive energy), which matters because the helas spinors take
C     dsqrt(p(0)+pp).
C
C     FLAV_IDX comes back 0 for an index beyond the table (or for a row that
C     names no crossing); PERM/SGN are then the identity, so a caller that
C     gathers momenta with them never reads out of range; 0 is what
C     GET_PDG_FOR_FLAVOR and GET_AMP already treat as "not a flavor".
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
C     The crossing of FLAV_IDX_IN in the input-slot view (see GET_CROSS_PERM):
C       PINV(K) : the base slot receiving input slot K (the inverse of PERM),
C       SGNI(K) : -1 when input slot K is a crossed leg, else +1.
C     The crossed process sees base leg PINV(K), charge conjugated when
C     SGNI(K)=-1, in its slot K: this is what the PDG signature, the spin*color
C     average and the identical-particle factor are built from. Both views are
C     needed because a crossing is in general not an involution (u u~ > g g off
C     g g > u u~ is a 4-cycle).
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
C     Apply the crossing carried by FLAV_IDX_IN to one set of momenta / NSF
C     flags and to NROW helicity rows at once (see GET_CROSS_PERM).
C
C     SMATRIX uses this to permute its whole NHEL table in a single sweep
C     before the helicity loop: the permutation does not depend on the row, so
C     decoding it once per SMATRIX call rather than once per helicity is both
C     cheaper and the reason MATRIX/GET_AMP can stay pure.
C     P/NHEL/IC must not alias P_IN/NHEL_IN/IC_IN.
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
C     TAU, not sigma: the helicity slots are NOT permuted, only sign-flipped,
C     and the sign is carried by IC above (HELAS builds its spinors from
C     nhel*nsf). IC stays aligned with the MOMENTA -- it is also the NSF
C     direction flag -- so slot XK gets P_IN(PERM(XK)) and NHEL_IN(XK), and the
C     effective helicity is NHEL_IN(XK)*SGN(XK). That is the same map the
C     recycled optim and the madmatrix lanes can realise, so one good-helicity
C     filter now describes every backend (see CROSS_GHIDX).
      DO XR = 1, NROW
        DO XK = 1, NEXTERNAL
          NHEL(XK,XR) = NHEL_IN(XK,XR)
        ENDDO
      ENDDO

      RETURN
      END


      SUBROUTINE %(proc_prefix)sAPPLY_CROSSING(FLAV_IDX_IN, P_IN, NHEL_IN,
     & IC_IN, P, NHEL, IC, FLAV_IDX)
C     Single helicity row flavour of APPLY_CROSSING_TABLE; kept public so that
C     an external (f2py) caller holding an extended FLAV_IDX can pre-apply the
C     crossing before calling GET_AMP, which is pure and rejects an extended
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
C     Initial-state spin*color average of the process crossing-table row CROSS
C     crosses into: the product of the per-leg spin*color (conjugation
C     invariant) of the base legs the row puts in the initial slots, computed
C     once at generation (XSPINCOL). A row beyond the table, or one that names
C     no crossing, returns 0, which SMATRIX / GET_PDG_FOR_FLAVOR /
C     GET_ALL_INTER_CROSSED all map to a null result: this is the single gate
C     that says whether a CROSS row is applicable at all. The flavor-dependent
C     half is GET_IDENT_CROSS.
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


      INTEGER FUNCTION %(proc_prefix)sGET_IDENT_CROSS(CROSS, FLAVOR)
C     Identical final state factor (product of n!) of the crossed process.
C
C     Flavor dependent, hence computed here rather than tabulated on CROSS:
C     d d~ > g u u~ crossed gives d g > d u u~ with nothing identical, while
C     d d~ > g d d~ crossed gives d g > d d d~ with two identical d. BROKEN_SYM
C     cannot be reused for this: its tables describe the uncrossed final state.
C
C     Two crossed final legs are identical when they carry the same flavor
C     group (same representative PDG, conjugated already when the leg swapped
C     side) and the same position inside it. Neither the per-slot representative
C     PDG nor the FLAVOR source slot is tabulated per crossing: both follow from
C     the table row (GET_CROSS_PINV) applied to two NEXTERNAL-long base tables
C     -- IDS_BASE, the base PDG of each leg, and ANTIPID_BASE, its charge
C     conjugate for a leg that swapped side.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER CROSS
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER IDS_BASE(0:NEXTERNAL-1)
      INTEGER ANTIPID_BASE(0:NEXTERNAL-1)
C     COUNTABLE(leg)=1 for a single external leg, 0 for a leaf locked inside a
C     decay block. A crossing never moves a block leaf (the crossing table has
C     no row that would), so decay products keep their base slots and must
C     be skipped here: their contribution to the identical-final factor is
C     resonance-level, carried whole by IDENT_RESONANCE. For a non-decay process
C     every leg is countable and IDENT_RESONANCE is 1, so this is the plain leaf
C     count it always was.
      INTEGER COUNTABLE(0:NEXTERNAL-1)
      INTEGER IDENT_RESONANCE
      PARAMETER (IDENT_RESONANCE=%(ident_resonance)d)
%(iden_cross_lines)s
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER K, L, N, FACT, I, XIDX
      INTEGER PINV(NEXTERNAL), SGNI(NEXTERNAL), BPID(NEXTERNAL)
      LOGICAL USED(NEXTERNAL)

C     The crossed process sees base leg PINV(K) in its slot K, conjugated
C     when SGNI(K)=-1 (GET_CROSS_PINV): read each slot's representative PDG
C     straight off IDS_BASE (ANTIPID_BASE where the leg changed side).
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

C     FLAVOR is not permuted by the crossing, so slot K reads FLAVOR(PINV(K)),
C     the actual flavor of the base leg that moved into it. Start from
C     IDENT_RESONANCE (the resonance-level part of the identical factor, which a
C     crossing never changes) and multiply in the n! of the countable final legs
C     only -- a decay-block leaf is skipped, its symmetry already inside
C     IDENT_RESONANCE.
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

      SUBROUTINE %(proc_prefix)sCROSS_GHIDX(CROSS, PERM, SGN, NHELCOL,
     & GHIDX)
C     Runtime good-helicity remap: send a crossed helicity row (given by its
C     BASE-table config NHELCOL and the crossing's SGN from GET_CROSS_PERM) to
C     the identity row whose shared GOODHEL bit gates it: the base slots are
C     evaluated at NHELCOL(L)*SGN(L) (tau: sign-flip in place, see
C     APPLY_CROSSING_TABLE), so that config re-encoded in the canonical
C     mixed-radix order (the same STATES/NHSTATE the encoder/decoder use) is
C     the row. GHIDX=0 when the row is not filterable (GHFILT flag: a row that
C     names no crossing, or a sign flip that is no bijection on the helicity
C     table); the caller then computes every helicity and never trains. For
C     CROSS=0 (SGN +1) this returns IHEL, so the uncrossed path is unchanged.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER    MAXHEL
      PARAMETER (MAXHEL=%(maxhel)d)
      INTEGER    NCROSS
      PARAMETER (NCROSS=%(ncross)d)
      INTEGER CROSS, PERM(NEXTERNAL), SGN(NEXTERNAL)
      INTEGER NHELCOL(NEXTERNAL), GHIDX
      INTEGER I, K, D, TGT(NEXTERNAL)
      INTEGER GHFILT(0:NCROSS-1)
      INTEGER NHSTATE(NEXTERNAL), STATES(MAXHEL,NEXTERNAL)
%(ghfilt_data)s
%(nhstate_data)s
%(states_data)s
      IF (CROSS .LT. 0 .OR. CROSS .GT. NCROSS-1) THEN
        GHIDX = 0
        RETURN
      ENDIF
      IF (GHFILT(CROSS).EQ.0) THEN
        GHIDX = 0
        RETURN
      ENDIF
C     TAU: sign-flip in place, no slot permutation -- the gate has to live in
C     the same space as what APPLY_CROSSING_TABLE actually evaluates.
      DO K=1,NEXTERNAL
        TGT(K) = SGN(K)*NHELCOL(K)
      ENDDO
      GHIDX = 0
      DO K=1,NEXTERNAL
        DO D=1,NHSTATE(K)
          IF (STATES(D,K).EQ.TGT(K)) GOTO 7
        ENDDO
        D = 1
 7      CONTINUE
        GHIDX = GHIDX*NHSTATE(K) + (D-1)
      ENDDO
      GHIDX = GHIDX + 1

      RETURN
      END
