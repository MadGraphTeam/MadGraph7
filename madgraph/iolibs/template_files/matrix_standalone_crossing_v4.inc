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
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
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

C     XI==1 (resp. XJ==2) would swap a particle with itself: degenerate, so
C     treated as "no crossing" just like 0.
      IF (XI.NE.0 .AND. XI.NE.1) THEN
        CALL %(proc_prefix)sSWAP_LEGS(1, XI, PERM, SGN)
      ENDIF
      IF (XJ.NE.0 .AND. XJ.NE.2) THEN
        CALL %(proc_prefix)sSWAP_LEGS(2, XJ, PERM, SGN)
      ENDIF

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
      DO XR = 1, NROW
        DO XK = 1, NEXTERNAL
          NHEL(XK,XR) = NHEL_IN(PERM(XK),XR)
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
C     A crossing that cannot be applied (out of range, or an overlapping swap)
C     returns 0, which SMATRIX / GET_PDG_FOR_FLAVOR map to a null result. The
C     flavor-dependent half is GET_IDENT_CROSS.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NCROSS
      PARAMETER (NCROSS=(NEXTERNAL+1)*(NEXTERNAL+1))
      INTEGER CROSS, XI, XJ, XK, XT, FACTOR, I
      INTEGER PERM(NEXTERNAL)
C     The DATA tables are emitted together (SPINCOL_PART here, plus the
C     IDS_BASE/ANTIPID_BASE tables GET_IDENT_CROSS reads); each routine keeps
C     its own copy rather than sharing a COMMON, which would need a BLOCK DATA
C     unit to be DATA-initialised.
      INTEGER SPINCOL_PART(0:NEXTERNAL-1)
      INTEGER IDS_BASE(0:NEXTERNAL-1)
      INTEGER ANTIPID_BASE(0:NEXTERNAL-1)
%(iden_cross_lines)s

C     CROSS = XI*(NEXTERNAL+1) + XJ, with XI, XJ the crossing partners of
C     particles 1 and 2 (0 = leave alone; XI==1 / XJ==2 swap a particle with
C     itself, also a no-op). Reject out-of-range and the overlapping-swap codes
C     (both transpositions {1,XI} and {2,XJ} active AND sharing a slot -> a
C     3-cycle the consumers read with opposite orientation: pure redundancy).
      XI = CROSS / (NEXTERNAL+1)
      XJ = MOD(CROSS, NEXTERNAL+1)
      IF (CROSS .LT. 0 .OR. CROSS .GT. NCROSS-1 .OR.
     &    (XI.NE.0 .AND. XI.NE.1 .AND. XJ.NE.0 .AND. XJ.NE.2 .AND.
     &     (XI.EQ.2 .OR. XJ.EQ.1 .OR. XI.EQ.XJ))) THEN
        %(proc_prefix)sGET_SPINCOL_CROSS = 0
        RETURN
      ENDIF
C     Build the slot->leg map (identity plus the crossing's two transpositions)
C     and multiply the per-particle spin*color of the legs in the initial slots.
      DO XK = 1, NEXTERNAL
        PERM(XK) = XK
      ENDDO
      IF (XI.NE.0 .AND. XI.NE.1) THEN
        XT = PERM(1)
        PERM(1) = PERM(XI)
        PERM(XI) = XT
      ENDIF
      IF (XJ.NE.0 .AND. XJ.NE.2) THEN
        XT = PERM(2)
        PERM(2) = PERM(XJ)
        PERM(XJ) = XT
      ENDIF
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
C     the actual flavor of the original leg that moved into it.
      DO K = 1, NEXTERNAL
        USED(K) = .FALSE.
      ENDDO
      FACT = 1
      DO K = NINCOMING+1, NEXTERNAL
        IF (USED(K)) CYCLE
        N = 1
        DO L = K+1, NEXTERNAL
          IF (USED(L)) CYCLE
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
      DO K=1,NEXTERNAL
        TGT(PERM(K)) = SGN(K)*NHELCOL(K)
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
