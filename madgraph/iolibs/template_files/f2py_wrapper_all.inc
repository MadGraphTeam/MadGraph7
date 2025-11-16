
C     PY ((21, 21), (-6, 6)) : (21, 21, 6, -6) # M0_ 1
C     PY ((2, -2), (-6, 6)) : (2, -2, 6, -6) # M1_ 1
C     PY ((4, -4), (-6, 6)) : (4, -4, 6, -6) # M1_ 1
C     PY ((1, -1), (-6, 6)) : (1, -1, 6, -6) # M1_ 1
C     PY ((3, -3), (-6, 6)) : (3, -3, 6, -6) # M1_ 1
      SUBROUTINE SMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALE2,
     $  NHEL, ANS)
      IMPLICIT NONE
C     ALPHAS is given at scale2 (SHOULD be different of 0 for loop
C      induced, ignore for LO)  

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
      INTEGER PDGS(*)
      INTEGER NPDG, NHEL, PROCID
      DOUBLE PRECISION P(*)
      DOUBLE PRECISION ANS, ALPHAS,SCALE2

      call f77_smatrixhel(pdgs, procid, npdg, p, 
     & alphas, scale2, nhel, ans)
       
      RETURN
      END

      SUBROUTINE INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END


      SUBROUTINE CHANGE_PARA(NAME, VALUE)
      IMPLICIT NONE
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      CHARACTER*512 NAME
      DOUBLE PRECISION VALUE
      call f77_change_para(name, value)

      RETURN
      END

      SUBROUTINE UPDATE_ALL_COUP()
      IMPLICIT NONE
      CALL f77_update_all_coup()
      RETURN
      END


      SUBROUTINE GET_PDG_ORDER(PDG, ALLPROC)
      IMPLICIT NONE
CF2PY INTEGER, intent(out) :: PDG(%(nb_me)i,%(maxpart)i)
CF2PY INTEGER, intent(out) :: ALLPROC(%(nb_me)i)
  INTEGER PDG(%(nb_me)i,%(maxpart)i)
  INTEGER ALLPROC(%(nb_me)i)
      call f77_get_pdg_order(pdg, allproc)
      RETURN
      END

      SUBROUTINE GET_PREFIX(PREFIX)
      IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: PREFIX(%(nb_me)i)
      CHARACTER*20 PREFIX(%(nb_me)i)
      call f77_get_prefix(prefix)
      RETURN
      END

      SUBROUTINE SET_FIXED_EXTRA_SCALE(NEW_VALUE)
      IMPLICIT NONE
CF2PY logical, intent(in) :: new_value
      LOGICAL NEW_VALUE
      call f77_set_fixed_extra_scale(new_value)
      RETURN
      END

      SUBROUTINE SET_MUE_OVER_REF(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE

      call f77_set_mue_over_ref(new_value)
      RETURN
      END

      SUBROUTINE SET_MUE_REF_FIXED(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE

      call f77_set_mue_ref_fixed(new_value)
      RETURN
      END


      SUBROUTINE SET_MAXJETFLAVOR(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      call f77_set_maxjetflavor(new_value)

      RETURN
      END


      SUBROUTINE SET_ASMZ(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      call f77_set_asmz(new_value)
      RETURN
      END

      SUBROUTINE SET_NLOOP(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      call f77_set_nloop(new_value)
      RETURN
      END


      %(nhel)s

