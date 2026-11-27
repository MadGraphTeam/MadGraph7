      SUBROUTINE SMATRIX%(proc_id)s(P, IFLAV, RHEL, RCOL, channel, IVEC, ANS, IHEL, ICOL)
C
%(info_lines)s
C
C MadGraph7 for Madevent Version
C
C Crossing-symmetry router: this subprocess shares its
C matrix element with a base subprocess of the same group.
C Each of its flavors is a crossing of a base flavor, so
C instead of its own (heavy) MATRIX it dispatches to the
C base SMATRIX with the extended FLAV_IDX that reproduces
C the crossed process. The base SMATRIX crosses the momenta
C P (supplied in this subprocess's own leg order) and
C rebuilds the crossed denominator, so ANS is this
C subprocess's matrix element. Only the flavor table
C (GET_FLAVOR) is kept here for the PDF.
C
%(process_lines)s
C
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      REAL*8 P(0:3,NEXTERNAL), ANS
      DOUBLE PRECISION RHEL, RCOL
      INTEGER channel, IVEC, IFLAV, IHEL, ICOL
C     Colour is reselected here, not taken from the base: the
C     base's SELECT_COLOR masked its JAMP2 with the BASE's
C     ICOLAMP row, which at the same ICONFIG can allow a
C     different set of flows than this subprocess's own, so its
C     pick may be a topology this subprocess never emits.
C     XG_SELCOL below permutes the base's published per-flow
C     JAMP2 into this subprocess's flow order and runs
C     SELECT_COLOR with THIS subprocess's IPROC. (The helicity
C     index needs no such treatment: it is a relabel of a choice
C     made in a shared space.)
%(smatrix_router_decl)s
      ANS = 0D0
      IHEL = 1
      ICOL = 1
%(smatrix_router_dispatch)s
      END


      SUBROUTINE GET_FLAVOR%(proc_id)s(IFLAV, FLAVOR_OUT)
C     Returns the flavor array for a given flavor index IFLAV
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER IFLAV, I
      INTEGER FLAVOR_OUT(NEXTERNAL)
      INTEGER FLAVOR(NEXTERNAL,%(max_flavor)s)
      %(get_flavor_matrix)s
      FLAVOR_OUT(:) = FLAVOR(:, IFLAV)
      END


      SUBROUTINE PRINT_ZERO_AMP_%(proc_id)s()
      INTEGER I
      I = 1
      RETURN
      END

%(smatrix_router_helper)s
