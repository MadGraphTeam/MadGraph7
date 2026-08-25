      SUBROUTINE %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX_IN, PERM, SGN,
     & FLAV_IDX)
C     Decode the crossing carried by FLAV_IDX_IN into a slot permutation.
C
C       CROSS    = (FLAV_IDX_IN-1) / NFLAV
C       FLAV_IDX = mod(FLAV_IDX_IN-1, NFLAV) + 1   ! used for masking/...
C       I        = CROSS / (NEXTERNAL+1)           ! partner of particle 1
C       J        = mod(CROSS, NEXTERNAL+1)         ! partner of particle 2
C
C     Particle 1 is swapped with particle I and particle 2 with particle J; 0
C     means "leave that particle alone", so FLAV_IDX_IN in [1,NFLAV] gives
C     CROSS=0 and the identity, keeping old callers untouched. The base is
C     NEXTERNAL+1 rather than NEXTERNAL so that I and J run over 0..NEXTERNAL
C     and can designate the last particle as well.
C
C     Swapping moves the momentum and the helicity between the two slots and
C     flips their NSF/NSV flag through IC. Flipping that flag is what actually
C     crosses the leg: helas stores the momentum as p*nsf and uses nhel*nsf,
C     so the momentum sign change and the helicity flip both follow from it.
C     Momenta therefore stay physical (positive energy), which matters because
C     the helas spinors take dsqrt(p(0)+pp).
C
C     The crossing is nothing but a fixed relabelling of slots, so it is
C     decoded once into
C       PERM(K) : the input slot whose content lands in crossed slot K,
C       SGN(K)  : the NSF/NSV sign flip applied to crossed slot K,
C     and the callers reuse it for as many momenta/helicity rows as they like
C     (see APPLY_CROSSING_TABLE) instead of decoding per matrix element call.
C     FLAV_IDX comes back 0 for a code that names no crossing at all (see the
C     two rejections below). PERM/SGN are left a valid permutation whatever
C     happens, so a caller that gathers momenta with them never reads out of
C     range; 0 is what GET_PDG_FOR_FLAVOR and GET_AMP already treat as "not a
C     flavor", so no caller needs a new return convention.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER NCROSS
      PARAMETER (NCROSS=(NEXTERNAL+1)*(NEXTERNAL+1))
C     ARGUMENTS
      INTEGER FLAV_IDX_IN
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL)
      INTEGER FLAV_IDX
C     LOCAL
      INTEGER CROSS, XI, XJ, XK

      FLAV_IDX = MOD(FLAV_IDX_IN-1, NFLAV) + 1
      CROSS = (FLAV_IDX_IN-1) / NFLAV
      XI = CROSS / (NEXTERNAL+1)
      XJ = MOD(CROSS, NEXTERNAL+1)

      DO XK = 1, NEXTERNAL
        PERM(XK) = XK
        SGN(XK) = 1
      ENDDO

C     Out of range, or an overlapping-swap code: both transpositions {1,XI} and
C     {2,XJ} active AND sharing a slot compose into a 3-cycle that the consumers
C     read with opposite orientation, so it is pure redundancy.
      IF (CROSS .LT. 0 .OR. CROSS .GT. NCROSS-1 .OR.
     &    (XI.NE.0 .AND. XI.NE.1 .AND. XJ.NE.0 .AND. XJ.NE.2 .AND.
     &     (XI.EQ.2 .OR. XJ.EQ.1 .OR. XI.EQ.XJ))) THEN
        FLAV_IDX = 0
        RETURN
      ENDIF

C     XI==1 (resp. XJ==2) would swap a particle with itself: degenerate, so
C     treated as "no crossing" just like 0.
      IF (XI.NE.0 .AND. XI.NE.1) THEN
        CALL %(proc_prefix)sSWAP_LEGS(1, XI, PERM, SGN)
      ENDIF
      IF (XJ.NE.0 .AND. XJ.NE.2) THEN
        CALL %(proc_prefix)sSWAP_LEGS(2, XJ, PERM, SGN)
      ENDIF

C     A crossing may only conjugate a leg that CHANGES SIDE. A transposition
C     between two legs on the same side of the initial/final line conjugates
C     both without moving either across, which is no crossing: for a 2 -> N
C     process that is XI==2 / XJ==1, the beam swap, which must not conjugate
C     anything (it would give e.g. u~ g > e+ ve d, not even charge conserving);
C     for a 1 -> N one it is every XJ swap.
      DO XK = 1, NEXTERNAL
        IF (SGN(XK).EQ.-1 .AND.
     &      ((XK.LE.NINCOMING) .EQV. (PERM(XK).LE.NINCOMING))) THEN
          FLAV_IDX = 0
          RETURN
        ENDIF
      ENDDO

      RETURN
      END


      SUBROUTINE %(proc_prefix)sSWAP_LEGS(SLOT_A, SLOT_B, PERM, SGN)
C     Exchange two legs in the permutation being built and flip their NSF/NSV
C     sign (see GET_CROSS_PERM). SGN is swapped along with PERM before being
C     negated, so that two overlapping swaps compose exactly as they would if
C     the momentum/helicity/IC arrays themselves were swapped in turn.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER SLOT_A, SLOT_B
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL)
      INTEGER ITMP

      ITMP = PERM(SLOT_A)
      PERM(SLOT_A) = PERM(SLOT_B)
      PERM(SLOT_B) = ITMP
      ITMP = SGN(SLOT_A)
      SGN(SLOT_A) = SGN(SLOT_B)
      SGN(SLOT_B) = ITMP
      SGN(SLOT_A) = -SGN(SLOT_A)
      SGN(SLOT_B) = -SGN(SLOT_B)

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
C     Initial-state spin*color average of the crossed process.
C
C     Crossing changes which particles sit in the initial state (pulling a
C     gluon in takes the color average from 3 to 8), but every particle of a
C     flavor group shares its spin and color, and conjugation preserves both.
C     So this half of the denominator is just the product of the per-particle
C     spin*color (SPINCOL_PART, one entry per external leg) over the two legs
C     the crossing puts in the initial state -- no per-crossing table needed.
C     A crossing that cannot be applied returns 0, which SMATRIX /
C     GET_PDG_FOR_FLAVOR / GET_ALL_INTER_CROSSED all map to a null result: this
C     is the single gate that says whether a CROSS code is applicable at all.
C     The flavor-dependent half is GET_IDENT_CROSS.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER CROSS, XK, FACTOR, I, XIDX
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL)
C     The DATA tables are emitted together (SPINCOL_PART and COUNTABLE here,
C     plus the IDS_BASE/ANTIPID_BASE tables GET_IDENT_CROSS reads); each routine
C     keeps its own copy rather than sharing a COMMON, which would need a BLOCK
C     DATA unit to be DATA-initialised.
      INTEGER SPINCOL_PART(0:NEXTERNAL-1)
      INTEGER IDS_BASE(0:NEXTERNAL-1)
      INTEGER ANTIPID_BASE(0:NEXTERNAL-1)
      INTEGER COUNTABLE(0:NEXTERNAL-1)
%(iden_cross_lines)s

C     Decode through GET_CROSS_PERM rather than rebuilding the slot map here,
C     so that which codes name a crossing at all is decided in exactly one
C     place (it rejects out-of-range, overlapping-swap and same-side codes).
      CALL %(proc_prefix)sGET_CROSS_PERM(CROSS*NFLAV+1, PERM, SGN, XIDX)
      IF (XIDX .LT. 1) THEN
        %(proc_prefix)sGET_SPINCOL_CROSS = 0
        RETURN
      ENDIF
C     Fail-safe for a decay chain: COUNTABLE is 0 for a leg that cannot be
C     crossed (a leaf locked inside a decay block), and carrying one across the
C     initial/final line would split its resonance. Generation already leaves
C     such a crossing unrecorded, so nothing should ask for it -- this is the
C     net under that, not the thing that makes it correct.
      DO XK = 1, NEXTERNAL
        IF (SGN(XK).EQ.-1 .AND. COUNTABLE(PERM(XK)-1).EQ.0) THEN
          %(proc_prefix)sGET_SPINCOL_CROSS = 0
          RETURN
        ENDIF
      ENDDO
C     Multiply the per-particle spin*color of the legs the crossing puts in the
C     initial slots.
      FACTOR = 1
      DO XK = 1, NINCOMING
        FACTOR = FACTOR * SPINCOL_PART(PERM(XK)-1)
      ENDDO
      %(proc_prefix)sGET_SPINCOL_CROSS = FACTOR

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
C     the crossing PERM/IC (as GET_SPINCOL_CROSS / GET_CROSS_PERM build it)
C     applied to two NEXTERNAL-long base tables -- IDS_BASE, the base PDG of each
C     leg, and ANTIPID_BASE, its charge conjugate for a leg that swapped side.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER CROSS
      INTEGER FLAVOR(NEXTERNAL)
C     SPINCOL_PART is unused here (GET_SPINCOL_CROSS owns it) but must be
C     declared because the shared DATA block below initialises it.
      INTEGER SPINCOL_PART(0:NEXTERNAL-1)
      INTEGER IDS_BASE(0:NEXTERNAL-1)
      INTEGER ANTIPID_BASE(0:NEXTERNAL-1)
C     COUNTABLE(leg)=1 for a single external leg, 0 for a leaf locked inside a
C     decay block. A crossing never moves a block leaf (GET_SPINCOL_CROSS
C     rejects any that would), so decay products keep their base slots and must
C     be skipped here: their contribution to the identical-final factor is
C     resonance-level, carried whole by IDENT_RESONANCE. For a non-decay process
C     every leg is countable and IDENT_RESONANCE is 1, so this is the plain leaf
C     count it always was.
      INTEGER COUNTABLE(0:NEXTERNAL-1)
      INTEGER IDENT_RESONANCE
      PARAMETER (IDENT_RESONANCE=%(ident_resonance)d)
%(iden_cross_lines)s
      INTEGER K, L, N, FACT, XI, XJ, XT, I
      INTEGER PERM(NEXTERNAL), ICS(NEXTERNAL), BPID(NEXTERNAL)
      LOGICAL USED(NEXTERNAL)

C     Rebuild the slot->leg map PERM and its initial/final sign flips ICS from
C     CROSS (identity plus the crossing's two transpositions), exactly as in
C     GET_SPINCOL_CROSS, then read each slot's representative PDG straight off
C     IDS_BASE (ANTIPID_BASE where the leg changed side).
      XI = CROSS / (NEXTERNAL+1)
      XJ = MOD(CROSS, NEXTERNAL+1)
      DO K = 1, NEXTERNAL
        PERM(K) = K
        ICS(K) = 1
      ENDDO
      IF (XI.NE.0 .AND. XI.NE.1) THEN
        XT = PERM(1)
        PERM(1) = PERM(XI)
        PERM(XI) = XT
        ICS(1) = -ICS(1)
        ICS(XI) = -ICS(XI)
      ENDIF
      IF (XJ.NE.0 .AND. XJ.NE.2) THEN
        XT = PERM(2)
        PERM(2) = PERM(XJ)
        PERM(XJ) = XT
        ICS(2) = -ICS(2)
        ICS(XJ) = -ICS(XJ)
      ENDIF
      DO K = 1, NEXTERNAL
        IF (ICS(K).EQ.1) THEN
          BPID(K) = IDS_BASE(PERM(K)-1)
        ELSE
          BPID(K) = ANTIPID_BASE(PERM(K)-1)
        ENDIF
      ENDDO

C     FLAVOR is not permuted by the crossing, so slot K reads FLAVOR(PERM(K)),
C     the actual flavor of the original leg that moved into it. Start from
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
        IF (COUNTABLE(PERM(K)-1).EQ.0) CYCLE
        N = 1
        DO L = K+1, NEXTERNAL
          IF (USED(L)) CYCLE
          IF (COUNTABLE(PERM(L)-1).EQ.0) CYCLE
          IF (BPID(K).EQ.BPID(L) .AND.
     $        FLAVOR(PERM(K)).EQ.FLAVOR(PERM(L))) THEN
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
C     BASE-table config NHELCOL and the crossing's PERM/SGN from GET_CROSS_PERM)
C     to the identity row whose shared GOODHEL bit gates it. This replaces the
C     baked GHREMAP(NCROSS*NCOMB) table: the map is a fixed permutation, so it is
C     cheaper to recompute it from the config than to store it -- permute and
C     sign-flip the config, then re-encode it in the canonical mixed-radix order
C     (the same STATES/NHSTATE the encoder/decoder use).
C     GHIDX=0 when the crossing is not filterable (GHFILT flag: initial-initial
C     swap, inapplicable, or a non-bijection); the caller then computes every
C     helicity and never trains. For CROSS=0 (PERM identity, SGN +1) this returns
C     IHEL, so the uncrossed path is unchanged.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER    MAXHEL
      PARAMETER (MAXHEL=%(maxhel)d)
      INTEGER    NCROSS
      PARAMETER (NCROSS=(NEXTERNAL+1)*(NEXTERNAL+1))
      INTEGER CROSS, PERM(NEXTERNAL), SGN(NEXTERNAL)
      INTEGER NHELCOL(NEXTERNAL), GHIDX
      INTEGER I, K, D, TGT(NEXTERNAL)
      INTEGER GHFILT(0:NCROSS-1)
      INTEGER NHSTATE(NEXTERNAL), STATES(MAXHEL,NEXTERNAL)
%(ghfilt_data)s
%(nhstate_data)s
%(states_data)s
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
