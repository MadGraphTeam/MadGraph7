      SUBROUTINE GPD_DD(IPART, X, XI, T, MU, G)
C     The double-distribution model needs the forward PDFs, which only the
C     madevent output links (Source/GPD/gpd_dd.f).
      IMPLICIT NONE
      INTEGER IPART
      DOUBLE PRECISION X, XI, T, MU, G(4)
      WRITE(*,*) 'GPD: the dd model needs the PDFs of a madevent output;'
      WRITE(*,*) '     use gpd_model = toy or grid in a standalone output.'
      STOP 1
      END
