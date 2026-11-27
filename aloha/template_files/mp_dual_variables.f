      module mp_dual_variables
c     First-order dual numbers in quadruple precision, a + d b with d**2=0,
c     for the quadruple-precision routines of a MadLoop library with one
c     P-wave bound state (comp(0) the value, comp(1) the derivative in the
c     relative momentum). The interface is that of dual_variables.f for one
c     P-wave state at first order (NPWAVE=1, DER_ORDER=1).
      implicit none
      private
      public :: MP_Dual
      public :: assignment(=)
      public :: operator(+), operator(-)
      public :: operator(*), operator(/)
      public :: operator(**), sqrt, log
      public :: CONJG, DCONJG
      public :: DBLE, DIMAG
      public :: size, qcmplx, qreal

      integer, parameter :: qp = 16

      type :: MP_Dual
         complex(kind=qp), dimension(0:1) :: comp = (0.0_qp,0.0_qp)
      contains
         procedure :: initZERO
      end type MP_Dual

      interface assignment(=)
         module procedure mp_assign_real, mp_assign_complex,
     &      mp_assign_integer, mp_assign_real8, mp_assign_complex8
      end interface assignment(=)

      interface operator(+)
         module procedure mp_add_dd, mp_add_cd, mp_add_dc, mp_add_rd,
     &      mp_add_dr, mp_add_id, mp_add_di, mp_plus
      end interface operator(+)

      interface operator(-)
         module procedure mp_sub_dd, mp_sub_cd, mp_sub_dc, mp_sub_rd,
     &      mp_sub_dr, mp_sub_id, mp_sub_di, mp_minus
      end interface operator(-)

      interface operator(*)
         module procedure mp_mul_dd, mp_mul_cd, mp_mul_dc, mp_mul_rd,
     &      mp_mul_dr, mp_mul_id, mp_mul_di
      end interface operator(*)

      interface operator(/)
         module procedure mp_div_dd, mp_div_cd, mp_div_dc, mp_div_rd,
     &      mp_div_dr, mp_div_id, mp_div_di
      end interface operator(/)

      interface operator(**)
         module procedure mp_pow_int, mp_pow_real
      end interface operator(**)

      interface sqrt
         module procedure mp_sqrt
      end interface sqrt

      interface log
         module procedure mp_log
      end interface log

      interface DBLE
         module procedure mp_dble
      end interface DBLE

      interface DIMAG
         module procedure mp_dimag
      end interface DIMAG

      interface CONJG
         module procedure mp_conjg
      end interface CONJG

      interface DCONJG
         module procedure mp_conjg
      end interface DCONJG

      interface size
         module procedure mp_dual_length
      end interface size

c     complex*32 and real*16 constructors (dcmplx and dble of the
c     double-precision routines)
      interface qcmplx
         module procedure qcmplx_i, qcmplx_r, qcmplx_ii, qcmplx_rr,
     &      qcmplx_ri, qcmplx_ir, qcmplx_r8, qcmplx_c
      end interface qcmplx

      interface qreal
         module procedure qreal_i, qreal_r, qreal_r8
      end interface qreal

      contains

      subroutine initZERO(self)
         class(MP_Dual) :: self
         self%comp = (0.0_qp,0.0_qp)
      end subroutine initZERO

      pure integer function mp_dual_length(a) result(res)
         type(MP_Dual), intent(in) :: a
         res = size(a%comp) - 1
      end function mp_dual_length

c     assignments
      elemental subroutine mp_assign_real(self, r)
         type(MP_Dual), intent(out) :: self
         real(kind=qp), intent(in) :: r
         self%comp(0) = cmplx(r, 0.0_qp, kind=qp)
         self%comp(1) = (0.0_qp,0.0_qp)
      end subroutine mp_assign_real

      elemental subroutine mp_assign_complex(self, c)
         type(MP_Dual), intent(out) :: self
         complex(kind=qp), intent(in) :: c
         self%comp(0) = c
         self%comp(1) = (0.0_qp,0.0_qp)
      end subroutine mp_assign_complex

      elemental subroutine mp_assign_real8(self, r)
c     (the double-precision constants of the ALOHA templates)
         type(MP_Dual), intent(out) :: self
         real(kind(1d0)), intent(in) :: r
         self%comp(0) = cmplx(real(r, kind=qp), 0.0_qp, kind=qp)
         self%comp(1) = (0.0_qp,0.0_qp)
      end subroutine mp_assign_real8

      elemental subroutine mp_assign_complex8(self, c)
         type(MP_Dual), intent(out) :: self
         complex(kind(1d0)), intent(in) :: c
         self%comp(0) = cmplx(c, kind=qp)
         self%comp(1) = (0.0_qp,0.0_qp)
      end subroutine mp_assign_complex8

      elemental subroutine mp_assign_integer(self, i)
         type(MP_Dual), intent(out) :: self
         integer, intent(in) :: i
         self%comp(0) = cmplx(real(i, kind=qp), 0.0_qp, kind=qp)
         self%comp(1) = (0.0_qp,0.0_qp)
      end subroutine mp_assign_integer

c     sums
      elemental function mp_add_dd(a, b) result(res)
         type(MP_Dual), intent(in) :: a, b
         type(MP_Dual) :: res
         res%comp = a%comp + b%comp
      end function mp_add_dd
      elemental function mp_add_cd(c, a) result(res)
         complex(kind=qp), intent(in) :: c
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = c + a%comp(0)
         res%comp(1) = a%comp(1)
      end function mp_add_cd
      elemental function mp_add_dc(a, c) result(res)
         type(MP_Dual), intent(in) :: a
         complex(kind=qp), intent(in) :: c
         type(MP_Dual) :: res
         res = mp_add_cd(c, a)
      end function mp_add_dc
      elemental function mp_add_rd(r, a) result(res)
         real(kind=qp), intent(in) :: r
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = r + a%comp(0)
         res%comp(1) = a%comp(1)
      end function mp_add_rd
      elemental function mp_add_dr(a, r) result(res)
         type(MP_Dual), intent(in) :: a
         real(kind=qp), intent(in) :: r
         type(MP_Dual) :: res
         res = mp_add_rd(r, a)
      end function mp_add_dr
      elemental function mp_add_id(i, a) result(res)
         integer, intent(in) :: i
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res = mp_add_rd(real(i, kind=qp), a)
      end function mp_add_id
      elemental function mp_add_di(a, i) result(res)
         type(MP_Dual), intent(in) :: a
         integer, intent(in) :: i
         type(MP_Dual) :: res
         res = mp_add_rd(real(i, kind=qp), a)
      end function mp_add_di
      elemental function mp_plus(a) result(res)
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp = a%comp
      end function mp_plus

c     differences
      elemental function mp_sub_dd(a, b) result(res)
         type(MP_Dual), intent(in) :: a, b
         type(MP_Dual) :: res
         res%comp = a%comp - b%comp
      end function mp_sub_dd
      elemental function mp_sub_cd(c, a) result(res)
         complex(kind=qp), intent(in) :: c
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = c - a%comp(0)
         res%comp(1) = -a%comp(1)
      end function mp_sub_cd
      elemental function mp_sub_dc(a, c) result(res)
         type(MP_Dual), intent(in) :: a
         complex(kind=qp), intent(in) :: c
         type(MP_Dual) :: res
         res%comp(0) = a%comp(0) - c
         res%comp(1) = a%comp(1)
      end function mp_sub_dc
      elemental function mp_sub_rd(r, a) result(res)
         real(kind=qp), intent(in) :: r
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = r - a%comp(0)
         res%comp(1) = -a%comp(1)
      end function mp_sub_rd
      elemental function mp_sub_dr(a, r) result(res)
         type(MP_Dual), intent(in) :: a
         real(kind=qp), intent(in) :: r
         type(MP_Dual) :: res
         res%comp(0) = a%comp(0) - r
         res%comp(1) = a%comp(1)
      end function mp_sub_dr
      elemental function mp_sub_id(i, a) result(res)
         integer, intent(in) :: i
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res = mp_sub_rd(real(i, kind=qp), a)
      end function mp_sub_id
      elemental function mp_sub_di(a, i) result(res)
         type(MP_Dual), intent(in) :: a
         integer, intent(in) :: i
         type(MP_Dual) :: res
         res = mp_sub_dr(a, real(i, kind=qp))
      end function mp_sub_di
      elemental function mp_minus(a) result(res)
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp = -a%comp
      end function mp_minus

c     products
      elemental function mp_mul_dd(a, b) result(res)
         type(MP_Dual), intent(in) :: a, b
         type(MP_Dual) :: res
         res%comp(0) = a%comp(0)*b%comp(0)
         res%comp(1) = a%comp(0)*b%comp(1) + a%comp(1)*b%comp(0)
      end function mp_mul_dd
      elemental function mp_mul_cd(c, a) result(res)
         complex(kind=qp), intent(in) :: c
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp = c*a%comp
      end function mp_mul_cd
      elemental function mp_mul_dc(a, c) result(res)
         type(MP_Dual), intent(in) :: a
         complex(kind=qp), intent(in) :: c
         type(MP_Dual) :: res
         res%comp = c*a%comp
      end function mp_mul_dc
      elemental function mp_mul_rd(r, a) result(res)
         real(kind=qp), intent(in) :: r
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp = r*a%comp
      end function mp_mul_rd
      elemental function mp_mul_dr(a, r) result(res)
         type(MP_Dual), intent(in) :: a
         real(kind=qp), intent(in) :: r
         type(MP_Dual) :: res
         res%comp = r*a%comp
      end function mp_mul_dr
      elemental function mp_mul_id(i, a) result(res)
         integer, intent(in) :: i
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp = real(i, kind=qp)*a%comp
      end function mp_mul_id
      elemental function mp_mul_di(a, i) result(res)
         type(MP_Dual), intent(in) :: a
         integer, intent(in) :: i
         type(MP_Dual) :: res
         res%comp = real(i, kind=qp)*a%comp
      end function mp_mul_di

c     quotients
      elemental function mp_div_dd(a, b) result(res)
         type(MP_Dual), intent(in) :: a, b
         type(MP_Dual) :: res
         res%comp(0) = a%comp(0)/b%comp(0)
         res%comp(1) = (a%comp(1)*b%comp(0) - a%comp(0)*b%comp(1))
     &                 /b%comp(0)**2
      end function mp_div_dd
      elemental function mp_div_cd(c, a) result(res)
         complex(kind=qp), intent(in) :: c
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = c/a%comp(0)
         res%comp(1) = -c*a%comp(1)/a%comp(0)**2
      end function mp_div_cd
      elemental function mp_div_dc(a, c) result(res)
         type(MP_Dual), intent(in) :: a
         complex(kind=qp), intent(in) :: c
         type(MP_Dual) :: res
         res%comp = a%comp/c
      end function mp_div_dc
      elemental function mp_div_rd(r, a) result(res)
         real(kind=qp), intent(in) :: r
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res = mp_div_cd(cmplx(r, 0.0_qp, kind=qp), a)
      end function mp_div_rd
      elemental function mp_div_dr(a, r) result(res)
         type(MP_Dual), intent(in) :: a
         real(kind=qp), intent(in) :: r
         type(MP_Dual) :: res
         res%comp = a%comp/r
      end function mp_div_dr
      elemental function mp_div_id(i, a) result(res)
         integer, intent(in) :: i
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res = mp_div_rd(real(i, kind=qp), a)
      end function mp_div_id
      elemental function mp_div_di(a, i) result(res)
         type(MP_Dual), intent(in) :: a
         integer, intent(in) :: i
         type(MP_Dual) :: res
         res%comp = a%comp/real(i, kind=qp)
      end function mp_div_di

c     powers and functions
      elemental function mp_pow_int(a, n) result(res)
         type(MP_Dual), intent(in) :: a
         integer, intent(in) :: n
         type(MP_Dual) :: res
         if (n.eq.0) then
            res%comp(0) = (1.0_qp,0.0_qp)
            res%comp(1) = (0.0_qp,0.0_qp)
         elseif (n.eq.1) then
            res%comp = a%comp
         else
            res%comp(0) = a%comp(0)**n
            res%comp(1) = n*a%comp(0)**(n-1)*a%comp(1)
         endif
      end function mp_pow_int
      elemental function mp_pow_real(a, x) result(res)
         type(MP_Dual), intent(in) :: a
         real(kind=qp), intent(in) :: x
         type(MP_Dual) :: res
         res%comp(0) = a%comp(0)**x
         res%comp(1) = x*a%comp(0)**(x-1.0_qp)*a%comp(1)
      end function mp_pow_real
      elemental function mp_sqrt(a) result(res)
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = sqrt(a%comp(0))
         res%comp(1) = a%comp(1)/(2.0_qp*res%comp(0))
      end function mp_sqrt
      elemental function mp_log(a) result(res)
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = log(a%comp(0))
         res%comp(1) = a%comp(1)/a%comp(0)
      end function mp_log
      elemental function mp_dble(a) result(res)
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = cmplx(real(a%comp(0)), 0.0_qp, kind=qp)
         res%comp(1) = cmplx(real(a%comp(1)), 0.0_qp, kind=qp)
      end function mp_dble
      elemental function mp_dimag(a) result(res)
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp(0) = cmplx(aimag(a%comp(0)), 0.0_qp, kind=qp)
         res%comp(1) = cmplx(aimag(a%comp(1)), 0.0_qp, kind=qp)
      end function mp_dimag
      elemental function mp_conjg(a) result(res)
         type(MP_Dual), intent(in) :: a
         type(MP_Dual) :: res
         res%comp = conjg(a%comp)
      end function mp_conjg

c     constructors
      elemental complex(kind=qp) function qcmplx_i(i)
         integer, intent(in) :: i
         qcmplx_i = cmplx(real(i, kind=qp), 0.0_qp, kind=qp)
      end function qcmplx_i
      elemental complex(kind=qp) function qcmplx_r(r)
         real(kind=qp), intent(in) :: r
         qcmplx_r = cmplx(r, 0.0_qp, kind=qp)
      end function qcmplx_r
      elemental complex(kind=qp) function qcmplx_r8(r)
         real(kind(1d0)), intent(in) :: r
         qcmplx_r8 = cmplx(real(r, kind=qp), 0.0_qp, kind=qp)
      end function qcmplx_r8
      elemental complex(kind=qp) function qcmplx_c(c)
         complex(kind=qp), intent(in) :: c
         qcmplx_c = c
      end function qcmplx_c
      elemental complex(kind=qp) function qcmplx_ii(i, j)
         integer, intent(in) :: i, j
         qcmplx_ii = cmplx(real(i, kind=qp), real(j, kind=qp), kind=qp)
      end function qcmplx_ii
      elemental complex(kind=qp) function qcmplx_rr(r, s)
         real(kind=qp), intent(in) :: r, s
         qcmplx_rr = cmplx(r, s, kind=qp)
      end function qcmplx_rr
      elemental complex(kind=qp) function qcmplx_ri(r, j)
         real(kind=qp), intent(in) :: r
         integer, intent(in) :: j
         qcmplx_ri = cmplx(r, real(j, kind=qp), kind=qp)
      end function qcmplx_ri
      elemental complex(kind=qp) function qcmplx_ir(i, s)
         integer, intent(in) :: i
         real(kind=qp), intent(in) :: s
         qcmplx_ir = cmplx(real(i, kind=qp), s, kind=qp)
      end function qcmplx_ir
      elemental real(kind=qp) function qreal_i(i)
         integer, intent(in) :: i
         qreal_i = real(i, kind=qp)
      end function qreal_i
      elemental real(kind=qp) function qreal_r(r)
         real(kind=qp), intent(in) :: r
         qreal_r = r
      end function qreal_r
      elemental real(kind=qp) function qreal_r8(r)
         real(kind(1d0)), intent(in) :: r
         qreal_r8 = real(r, kind=qp)
      end function qreal_r8

      end module mp_dual_variables
