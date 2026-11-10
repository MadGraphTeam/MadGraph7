ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1( )
      USE MODEL_OBJECT
      IMPLICIT NONE

      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      GC_21 = -(MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D+00*MDL_SW)
      GC_22 = (MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D+00*MDL_SW)
      GC_23 = -(MDL_EE*MDL_COMPLEXI*MDL_SW)/(6.000000D+00*MDL_CW)
      GC_32 = MDL_EE__EXP__2*MDL_COMPLEXI*MDL_V+(MDL_CW__EXP__2
     $ *MDL_EE__EXP__2*MDL_COMPLEXI*MDL_V)/(2.000000D+00
     $ *MDL_SW__EXP__2)+(MDL_EE__EXP__2*MDL_COMPLEXI*MDL_SW__EXP__2
     $ *MDL_V)/(2.000000D+00*MDL_CW__EXP__2)
      GC_33 = -((MDL_COMPLEXI*MDL_YB)/MDL_SQRT__2)
      GC_37 = -((MDL_COMPLEXI*MDL_YT)/MDL_SQRT__2)
      END
