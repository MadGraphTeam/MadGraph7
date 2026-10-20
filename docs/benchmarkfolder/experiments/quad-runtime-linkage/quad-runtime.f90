program runtime_probe
  use iso_c_binding
  implicit none
  interface
    function ninjars_abi_version() bind(c) result(version)
      import c_int
      integer(c_int) :: version
    end function
  end interface
  real*16 :: x,y,z
  integer :: i,n
  integer*8 :: t0,t1,rate
  character(32) :: arg
  call get_command_argument(1,arg)
  read(arg,*) n
  x=1.0000000000000001_16
  y=0.9999999999999998_16
  call system_clock(t0,rate)
  do i=1,n
    z=x*y
    x=z+0.0000000000000001_16
  enddo
  call system_clock(t1)
  print '(I4,ES46.36E4,ES26.17E3)',ninjars_abi_version(),x,real(t1-t0,8)/real(rate,8)
end program
