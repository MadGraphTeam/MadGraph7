program rounding_probe
  use iso_c_binding
  use ieee_arithmetic
  implicit none
  interface
    function ninjars_abi_version() bind(c) result(version)
      import c_int
      integer(c_int) :: version
    end function
  end interface
  real*16,volatile :: x,y,z
  integer :: i
  integer(c_int64_t) :: bits(2)
  type(ieee_round_type) :: modes(3)
  modes=[ieee_nearest,ieee_up,ieee_down]
  print *, ninjars_abi_version()
  x=1.0_16
  y=2.0_16**(-114)
  do i=1,3
    call ieee_set_rounding_mode(modes(i))
    z=x+y
    bits=transfer(z,bits)
    print '(I2,2Z18.16)',i,bits
  enddo
  call ieee_set_rounding_mode(ieee_nearest)
end program
