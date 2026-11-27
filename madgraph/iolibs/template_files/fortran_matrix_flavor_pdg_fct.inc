      SUBROUTINE %(func_name)s(FLAV_IDX_IN, PDGS)
C     Return the signed PDG code of every leg of the process
C     FLAV_IDX_IN selects, INCLUDING the crossing it carries.
C
C     This is the bridge between the two vocabularies of this
C     file. Inside matrix.f a flavor is an unsigned group
C     *position*: that is all the matrix element needs, since
C     every member of a flavor group shares the couplings. A
C     caller speaking PDG codes cannot work with that -- a
C     position is meaningless without knowing the group and the
C     leg -- and nothing else generated here maps one back.
C     GET_FLAVOR_INDEX only goes the other way and only accepts
C     positions.
C
C     FLAV_IDX_IN is the *extended* index: it carries a flavor
C     and a crossing (see GET_CROSS_PERM). The PDGs returned are
C     therefore those of the process actually evaluated, i.e.
C     after the crossing has moved the legs around AND
C     conjugated every leg that swapped between the initial and
C     the final state -- an incoming u~ crossed into a final
C     slot comes back as an outgoing u. Handing an f2py caller
C     the uncrossed PDGs would be useless: it is precisely the
C     crossed signature it has to match a request against.
C
C     Conjugation is NOT a sign flip: a self-conjugate particle
C     (the gluon) is its own antiparticle. Both tables are
C     therefore filled at export time with the model's own
C     anti-pdg rule, and this routine only picks the one SGN
C     designates.
C
C     PDGS is set to 0 on every leg when FLAV_IDX_IN names no
C     valid flavor or a crossing that cannot be applied, so a
C     caller can test PDGS(1)==0 rather than having to
C     pre-validate the index.
      IMPLICIT NONE
%(nexternal_decl)s
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
C
C     ARGUMENTS
C
      INTEGER FLAV_IDX_IN
      INTEGER PDGS(NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX_IN
CF2PY INTENT(OUT) :: PDGS(NEXTERNAL)
C
C     LOCAL
C
      INTEGER FP_I, FP_FLAV
      INTEGER FP_PDG_TABLE(NEXTERNAL, NFLAV)
      INTEGER FP_ANTI_TABLE(NEXTERNAL, NFLAV)
      DATA FP_PDG_TABLE /%(pdg_table_data)s/
      DATA FP_ANTI_TABLE /%(antipdg_table_data)s/
%(pdg_cross_decl)s

      DO FP_I = 1, NEXTERNAL
        PDGS(FP_I) = 0
      ENDDO
C     Guard before decoding: a negative index would make the
C     MOD/divide below wrap onto a valid-looking flavor and
C     crossing.
      IF (FLAV_IDX_IN .LT. 1) THEN
        RETURN
      ENDIF

%(pdg_cross_decode)s

      IF (FP_FLAV .LT. 1 .OR. FP_FLAV .GT. NFLAV) THEN
        RETURN
      ENDIF

%(pdg_cross_apply)s

      RETURN
      END
