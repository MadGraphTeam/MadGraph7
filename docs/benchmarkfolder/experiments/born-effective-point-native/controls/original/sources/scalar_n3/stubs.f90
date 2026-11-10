module fx_control
  implicit none
  integer :: fx_calls=0,fx_symmetry=1
  integer :: fx_sequence(32)=0
  real(8) :: fx_values(3,4)=0d0
  logical :: fx_selected(4)=.true.
end module
real(8) function TX_MATRIX(p,nhel,jc,flav_idx)
  use fx_control
  implicit none
  real(8) :: p(0:3,3)
  integer :: nhel(3),jc(3),flav_idx,h
  h=nhel(3)
  fx_calls=fx_calls+1
  fx_sequence(fx_calls)=h
  TX_MATRIX=fx_values(1,h)
end function
subroutine TX_GET_FLAVOR(flav_idx,flavor)
  implicit none
  integer :: flav_idx,flavor(3)
  flavor=flav_idx
end subroutine
logical function TX_IS_BORN_HEL_SELECTED(h)
  use fx_control
  implicit none
  integer :: h
  TX_IS_BORN_HEL_SELECTED=fx_selected(h)
end function
integer function TX_BROKEN_SYM(flavor)
  use fx_control
  implicit none
  integer :: flavor(3)
  TX_BROKEN_SYM=fx_symmetry
end function
real(8) function LEG_MATRIX(p,nhel,jc,flav_idx)
  use fx_control
  implicit none
  real(8) :: p(0:3,3)
  integer :: nhel(3),jc(3),flav_idx,h
  h=nhel(3)
  fx_calls=fx_calls+1
  fx_sequence(fx_calls)=h
  LEG_MATRIX=fx_values(1,h)
end function
subroutine LEG_GET_FLAVOR(flav_idx,flavor)
  implicit none
  integer :: flav_idx,flavor(3)
  flavor=flav_idx
end subroutine
logical function LEG_IS_BORN_HEL_SELECTED(h)
  use fx_control
  implicit none
  integer :: h
  LEG_IS_BORN_HEL_SELECTED=fx_selected(h)
end function
integer function LEG_BROKEN_SYM(flavor)
  use fx_control
  implicit none
  integer :: flavor(3)
  LEG_BROKEN_SYM=fx_symmetry
end function
