      SUBROUTINE SMATRIX%(proc_id)s(P, IFLAV, RHEL, RCOL, channel, IVEC, ANS, IHEL, ICOL)
C
%(info_lines)s
C
C MadGraph5_aMC@NLO for Madevent Version
C
C Crossing-symmetry router: this subprocess shares its matrix element with a
C base subprocess of the same group. Each of its flavors is a crossing of a
C base flavor, so instead of its own (heavy) MATRIX it dispatches to the base
C SMATRIX with the extended FLAV_IDX that reproduces the crossed process. The
C base SMATRIX crosses the momenta P (supplied in this subprocess's own leg
C order) and rebuilds the crossed denominator, so ANS is this subprocess's
C matrix element. Only the flavor table (GET_FLAVOR) is kept here for the PDF.
C
%(process_lines)s
C
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      REAL*8 P(0:3,NEXTERNAL), ANS
      DOUBLE PRECISION RHEL, RCOL
      INTEGER channel, IVEC, IFLAV, IHEL, ICOL
C     Colour is reselected here, not taken from the base: the base's SELECT_COLOR
C     masked its JAMP2 with the BASE's ICOLAMP row, which at the same ICONFIG can
C     allow a different set of flows than this subprocess's own, so its pick may
C     be a topology this subprocess never emits. XG_SELCOL below permutes the
C     base's published per-flow JAMP2 into this subprocess's flow order and runs
C     SELECT_COLOR with THIS subprocess's IPROC. (The helicity index needs no
C     such treatment: it is a relabel of a choice made in a shared space.)
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
