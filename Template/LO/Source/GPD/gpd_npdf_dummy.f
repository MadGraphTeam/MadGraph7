      SUBROUTINE GPD_NPDF_XF(TEMPLATE, IA, IZ, IPDG, X, MU, XF)
C     Without LHAPDF: nuclear PDF sets (gpd_npdf) are not available.
      IMPLICIT NONE
      CHARACTER*(*) TEMPLATE
      INTEGER IA, IZ, IPDG
      DOUBLE PRECISION X, MU, XF
      WRITE(*,*) 'GPD: gpd_npdf (', TRIM(TEMPLATE), ') needs LHAPDF:',
     &  ' set pdlabel = lhapdf and the lhapdf path of the output'
      XF = 0D0
      STOP 1
      END
