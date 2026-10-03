      SUBROUTINE %(func_name)s(FLAV_IDX_IN, PDGS)
C     Return the signed PDG code of every leg of the process FLAV_IDX_IN
C     selects, INCLUDING the crossing it carries.
C
C     This is the bridge between the two vocabularies of this file. Inside
C     matrix.f a flavor is an unsigned group *position*: that is all the
C     matrix element needs, since every member of a flavor group shares the
C     couplings. A caller speaking PDG codes cannot work with that -- a
C     position is meaningless without knowing the group and the leg -- and
C     nothing else generated here maps one back. GET_FLAVOR_INDEX only goes
C     the other way and only accepts positions.
C
C     FLAV_IDX_IN is the *extended* index: it carries a flavor and a
C     crossing (see GET_CROSS_PERM). The PDGs returned are therefore those
C     of the process actually evaluated, i.e. after the crossing has moved
C     the legs around AND conjugated every leg that swapped between the
C     initial and the final state -- an incoming u~ crossed into a final
C     slot comes back as an outgoing u. Handing an f2py caller the
C     uncrossed PDGs would be useless: it is precisely the crossed
C     signature it has to match a request against.
C
C     Conjugation is NOT a sign flip: a self-conjugate particle (the gluon)
C     is its own antiparticle. Both tables are therefore filled at export
C     time with the model's own anti-pdg rule, and this routine only picks
C     the one SGN designates.
C
C     PDGS is set to 0 on every leg when FLAV_IDX_IN names no valid flavor
C     or a crossing that cannot be applied, so a caller can test PDGS(1)==0
C     rather than having to pre-validate the index.
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
C     Guard before decoding: a negative index would make the MOD/divide
C     below wrap onto a valid-looking flavor and crossing.
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
