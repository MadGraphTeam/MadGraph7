program smoke
  use mninjars
  use, intrinsic :: ieee_arithmetic
  implicit none
  complex(KI_NRS) :: c(1),m(1),r(0:2),r1
  real(KI_NRS) :: p(0:3,1),s(1,1)
  complex(KI_QNRS) :: qc(1),qm(1),qr(0:2),qr1
  real(KI_QNRS) :: qp(0:3,1),qs(1,1),delta
  complex(KI_NRS) :: bm(2)
  real(KI_NRS) :: bp(0:3,2),bs(2,2),pole_sign
  complex(KI_QNRS) :: qbm(2)
  real(KI_QNRS) :: qbp(0:3,2),qbs(2,2)
  integer :: status,pole_mode
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
  call ninjars_tensor_evaluate_compensated(c,1,0,s,p,m,4.0_KI_NRS,r,r1,status)
  if (status /= NINJARS_SUCCESS) stop 9
  if (abs(r(0)-4)>1e-12_KI_NRS .or. abs(r(1)-4)>1e-12_KI_NRS .or. abs(r(2))>1e-12_KI_NRS) stop 10
  ! Returning to the ordinary entry must not retain compensated mode.
  call ninjars_tensor_evaluate(c,1,0,s,p,m,4.0_KI_NRS,r,r1,status)
  if (status /= NINJARS_SUCCESS) stop 11
  if (abs(r(0)-4)>1e-12_KI_NRS .or. abs(r(1)-4)>1e-12_KI_NRS .or. abs(r(2))>1e-12_KI_NRS) stop 12
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
  ! A lightlike scaleless bubble vanishes only after UV and IR cancel.
  ! Exercise the installed optional selector in both numerical domains and
  ! preserve the input's binary128 information in its separated residues.
  bm = cmplx(0,0,kind=KI_NRS)
  bp = 0
  bp(0,2) = 2
  bp(3,2) = 2
  bs = 0
  qbm = cmplx(0,0,kind=KI_QNRS)
  qbp = real(bp,kind=KI_QNRS)
  qbs = 0
  do pole_mode=NINJARS_POLES_UV,NINJARS_POLES_IR
    pole_sign = 3-2*pole_mode
    call ninjars_tensor_evaluate(c,2,0,bs,bp,bm,4.0_KI_NRS,r,r1,status,pole_mode=pole_mode)
    if (status /= NINJARS_SUCCESS) stop 13
    if (abs(r(0))>1e-12_KI_NRS .or. abs(r(1)-pole_sign)>1e-12_KI_NRS .or. abs(r(2))>1e-12_KI_NRS) stop 14
    call ninjars_tensor_evaluate(qc,2,0,qbs,qbp,qbm,4.0_KI_QNRS,qr,qr1,status,pole_mode=pole_mode)
    if (status /= NINJARS_SUCCESS) stop 15
    if (abs(qr(1)-real(pole_sign,KI_QNRS)*(1.0_KI_QNRS+delta))>1e-32_KI_QNRS) stop 16
    if (abs(qr(0))>1e-32_KI_QNRS .or. abs(qr(2))>1e-32_KI_QNRS) stop 17
  enddo
  ! The omitted selector must restore combined poles on a reused context.
  call ninjars_tensor_evaluate(c,2,0,bs,bp,bm,4.0_KI_NRS,r,r1,status)
  if (status /= NINJARS_SUCCESS .or. any(abs(r)>1e-12_KI_NRS)) stop 18
  call ninjars_tensor_evaluate(qc,2,0,qbs,qbp,qbm,4.0_KI_QNRS,qr,qr1,status)
  if (status /= NINJARS_SUCCESS .or. any(abs(qr)>1e-32_KI_QNRS)) stop 19
  call ninjars_finalize(status)
  if (status /= NINJARS_SUCCESS) stop 7
  print *, 'NinjaRS ABI4 DP, binary128-input rescue and separated UV/IR poles passed'
end program
