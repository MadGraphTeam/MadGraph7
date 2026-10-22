ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE MP_COUP3( )
      USE MODEL_OBJECT
      IMPLICIT NONE

      INCLUDE 'model_functions.inc'
      REAL*16 MP__PI, MP__ZERO
      PARAMETER (MP__PI=3.1415926535897932384626433832795E0_16)
      PARAMETER (MP__ZERO=0E0_16)
      INCLUDE 'mp_input.inc'
      INCLUDE 'mp_coupl.inc'

      MP__GC_5 = MP__MDL_COMPLEXI*MP__G
      MP__R2_DDZ_V3 = (-(MP__MDL_EE*MP__MDL_COMPLEXI*MP__MDL_SW)
     $ /(6.000000E+00_16*MP__MDL_CW))*MP__MDL_R2MIXEDFACTOR_FIN_
      MP__R2_UUZ_V2 = ((MP__MDL_CW*MP__MDL_EE*MP__MDL_COMPLEXI)
     $ /(2.000000E+00_16*MP__MDL_SW))*MP__MDL_R2MIXEDFACTOR_FIN_
      END
