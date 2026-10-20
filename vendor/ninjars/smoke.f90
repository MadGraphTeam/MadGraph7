program smoke
  use mninjars
  use, intrinsic :: ieee_arithmetic
  implicit none
  complex(KI_NRS) :: c(1),m(1),r(0:2),r1
  real(KI_NRS) :: p(0:3,1),s(1,1)
  complex(KI_QNRS) :: qc(1),qm(1),qr(0:2),qr1
  real(KI_QNRS) :: qp(0:3,1),qs(1,1),delta
  integer :: status
  real(KI_QNRS), volatile :: rounding_x,rounding_y,rounding_sum
  type(ieee_round_type) :: original_rounding
  character(len=512) :: message
  ! Rust staticlib compiler_builtins must not replace the Fortran runtime:
  ! those helpers ignore the caller's directed IEEE rounding mode.
  call ieee_get_rounding_mode(original_rounding)
  rounding_x=1.0_KI_QNRS
  rounding_y=2.0_KI_QNRS**(-114)
  call ieee_set_rounding_mode(ieee_up)
  rounding_sum=rounding_x+rounding_y
  call ieee_set_rounding_mode(original_rounding)
  if (rounding_sum /= nearest(1.0_KI_QNRS,1.0_KI_QNRS)) stop 8
  c = cmplx(1,0,kind=KI_NRS)
  m = cmplx(4,0,kind=KI_NRS)
  p = 0
  s = 0
  call ninjars_tensor_evaluate(c,1,0,s,p,m,4.0_KI_NRS,r,r1,status)
  if (status /= NINJARS_SUCCESS) then
    call ninjars_error_message(message)
    print *, trim(message)
    stop 1
  endif
  if (abs(r(0)-4)>1e-12_KI_NRS .or. abs(r(1)-4)>1e-12_KI_NRS .or. abs(r(2))>1e-12_KI_NRS) stop 2
  delta = 2.0_KI_QNRS**(-100)
  qc = cmplx(1.0_KI_QNRS+delta,0.0_KI_QNRS,kind=KI_QNRS)
  qm = cmplx(4,0,kind=KI_QNRS)
  qp = 0
  qs = 0
  call ninjars_tensor_evaluate(qc,1,0,qs,qp,qm,4.0_KI_QNRS,qr,qr1,status)
  if (status /= NINJARS_SUCCESS) then
    call ninjars_error_message(message)
    print *, trim(message)
    stop 3
  endif
  if (abs(qr(1)-4.0_KI_QNRS*(1.0_KI_QNRS+delta))>1e-32_KI_QNRS) stop 4
  if (real(qr(1),kind=KI_QNRS) == 4.0_KI_QNRS) stop 5
  if (abs(qr(2))>1e-32_KI_QNRS .or. abs(qr1)>1e-32_KI_QNRS) stop 6
  call ninjars_finalize(status)
  if (status /= NINJARS_SUCCESS) stop 7
  print *, 'NinjaRS DP and genuine binary128-input rescue passed'
end program
