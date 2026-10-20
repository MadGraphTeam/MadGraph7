program helpers
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
  complex*32,volatile :: a,b,c
  real*8,volatile :: d
  integer*8,volatile :: n
  integer :: i,j
  type(ieee_round_type) :: modes(4)
  logical :: flags(size(ieee_all))
  modes=[ieee_nearest,ieee_up,ieee_down,ieee_to_zero]
  print *, ninjars_abi_version()
  do i=1,4
    call ieee_set_rounding_mode(modes(i))
    do j=1,5
      select case(j)
      case(1)
        x=1.0_16; y=2.0_16**(-114)
      case(2)
        x=-1.0_16; y=2.0_16**(-113)
      case(3)
        x=nearest(1.0_16,1.0_16); y=nearest(1.0_16,-1.0_16)
      case(4)
        x=tiny(1.0_16); y=0.5_16
      case(5)
        x=huge(1.0_16); y=nearest(1.0_16,1.0_16)
      end select
      call ieee_set_flag(ieee_all,.false.)
      z=x+y; call emit(z)
      z=x-y; call emit(z)
      z=x*y; call emit(z)
      z=x/y; call emit(z)
      print *, x==y,x/=y,x<y,x<=y,x>y,x>=y
      a=cmplx(x,y,kind=16); b=cmplx(y,-y,kind=16)
      c=a*b; call emit(real(c,16)); call emit(aimag(c))
      c=a/b; call emit(real(c,16)); call emit(aimag(c))
      d=real(x,8); z=real(d,16); call emit(z)
      n=huge(1_8)-j; z=real(n,16); call emit(z)
      call ieee_get_flag(ieee_all,flags)
      print *, flags
    enddo
  enddo
  call ieee_set_rounding_mode(ieee_nearest)
contains
  subroutine emit(v)
    real*16,intent(in) :: v
    integer(c_int64_t) :: bits(2)
    bits=transfer(v,bits)
    print '(2Z18.16)',bits
  end subroutine
end program
