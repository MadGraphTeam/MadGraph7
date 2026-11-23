program smoke
  use mtirust
  use, intrinsic :: iso_c_binding, only: c_int64_t
  use, intrinsic :: ieee_arithmetic
  implicit none
  real(KI_TIR) :: p(0:3,1),s(1,1),bp(0:3,2),bs(2,2)
  complex(KI_TIR) :: m(1),r(1,4),reference(1,4),bm(2)
  real(KI_QTIR) :: qp(0:3,1),qs(1,1),mass_qp
  complex(KI_QTIR) :: qm(1),qr(1,4),qreference(1,4)
  integer(c_int64_t) :: counts(9),qp_counts(6),qp_before(6),route_counts(20)
  integer :: status,pole_mode,first_generation
  character(len=512) :: message
  character(len=32) :: option

  call get_command_argument(1,option)
  if (tirust_context_generation() /= -1) stop 25
  if (trim(option) == '--require-poles' .and. .not.tirust_supports(1,0,.false.)) stop 24
  if (tirust_context_generation() /= -1) stop 26

  call tirust_dp_route_statistics(route_counts,status)
  call require_success('unused tensor-route statistics')
  if (any(route_counts /= 0_c_int64_t) .or. tirust_context_generation() /= -1) stop 42

  call check_rounding()
  p = 0
  s = 0
  m = cmplx(4,0,kind=KI_TIR)
  call tirust_register_plan(1,0,status)
  call require_success('register tadpole')
  first_generation = tirust_context_generation()
  if (first_generation < 1) stop 27
  call tirust_register_plan(2,0,status)
  call require_success('register bubble')
  if (tirust_context_generation() /= first_generation) stop 28
  if (.not.tirust_supports(1,0,.true.)) stop 22
  if (tirust_supports(7,0,.true.) .or. tirust_supports(1,7,.true.)) stop 23
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status)
  call require_success('DP tadpole')
  call tirust_dp_route_statistics(route_counts,status)
  call require_success('primary tensor-route statistics')
  if (route_counts(1) /= 1_c_int64_t .or. sum(route_counts) /= 1_c_int64_t) stop 43
  if (abs(r(1,1)-4)>1e-12_KI_TIR .or. any(abs(r(1,2:3))>1e-12_KI_TIR) .or. &
      abs(r(1,4)-4)>1e-12_KI_TIR) stop 2
  reference = r
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status)
  call require_success('cached DP tadpole')
  if (any(abs(r-reference)>1e-12_KI_TIR)) stop 3
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status,check_pass=.true.)
  call require_success('independent check pass')
  call tirust_dp_route_statistics(route_counts,status)
  call require_success('independent tensor-route statistics')
  ! The exact repeat returned the stored complete tensor: not an execution.
  if (route_counts(1) /= 1_c_int64_t .or. route_counts(11) /= 1_c_int64_t .or. &
      sum(route_counts) /= 2_c_int64_t) stop 44
  if (any(abs(r-reference)>1e-12_KI_TIR)) stop 4
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status,cache_enabled=.false.)
  call require_success('uncached DP tadpole')
  if (any(abs(r-reference)>1e-12_KI_TIR)) stop 5
  call tirust_cache_statistics(counts,status)
  call require_success('cache statistics')
  if (any(counts<0_c_int64_t)) stop 6
  call tirust_clear_cache(status)
  call require_success('clear cache')
  call tirust_dp_route_statistics(route_counts,status)
  call require_success('tensor routes survive event cache clear')
  if (route_counts(1) /= 2_c_int64_t .or. route_counts(11) /= 1_c_int64_t .or. &
      sum(route_counts) /= 3_c_int64_t) stop 45
  if (tirust_context_generation() /= first_generation) stop 29
  call tirust_set_cache_budget(0_c_int64_t,status)
  call require_success('disable cache insertion')
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status)
  call require_success('zero-budget DP tadpole')
  if (any(abs(r-reference)>1e-12_KI_TIR)) stop 7
  call tirust_set_cache_budget(268435456_c_int64_t,status)
  call require_success('restore cache budget')

  ! Exact last binary128 bit: this cannot survive a double or DD transport.
  mass_qp = 4.0_KI_QTIR*(1.0_KI_QTIR+2.0_KI_QTIR**(-112))
  qm = cmplx(mass_qp,-0.0_KI_QTIR,kind=KI_QTIR)
  qp = 0
  qs = 0
  call tirust_tensor_evaluate(1,0,qp,qs,qm,mass_qp,qr,status)
  call require_success('binary128 tadpole')
  if (real(qr(1,1),kind=KI_QTIR) /= mass_qp .or. real(qr(1,1),kind=KI_QTIR) == 4.0_KI_QTIR) stop 8
  if (abs(aimag(qr(1,1)))>1e-32_KI_QTIR .or. any(abs(qr(1,2:3))>1e-32_KI_QTIR)) stop 9
  if (abs(qr(1,4)-mass_qp)>1e-31_KI_QTIR) stop 10
  qreference = qr
  call tirust_qp_cache_statistics(qp_before,status)
  call require_success('initial QP cache statistics')
  call tirust_tensor_evaluate(1,0,qp,qs,qm,mass_qp,qr,status)
  call require_success('cached binary128 tadpole')
  if (any(abs(qr-qreference)>0.0_KI_QTIR)) stop 36
  call tirust_qp_cache_statistics(qp_counts,status)
  call require_success('repeated QP cache statistics')
  if (qp_counts(1)<=qp_before(1) .or. qp_counts(1)<=0_c_int64_t) stop 37
  if (any(qp_counts(2:6)/=qp_before(2:6))) stop 38
  qp_before = qp_counts
  call tirust_tensor_evaluate(1,0,qp,qs,qm,mass_qp,qr,status,check_pass=.true.)
  call require_success('independent binary128 check pass')
  if (any(abs(qr-qreference)>0.0_KI_QTIR)) stop 39
  call tirust_qp_cache_statistics(qp_counts,status)
  call require_success('independent QP cache statistics')
  ! The check pass has its own QP cache: its first request misses there (never
  ! a hit on the primary entry), is stored, and bypasses nothing.
  if (qp_counts(1)/=qp_before(1) .or. qp_counts(2)/=qp_before(2)+1_c_int64_t) stop 40
  if (qp_counts(3)/=qp_before(3)+1_c_int64_t .or. qp_counts(4)<=qp_before(4)) stop 41
  if (qp_counts(5)/=qp_before(5) .or. qp_counts(6)/=qp_before(6)) stop 46
  qp_before = qp_counts
  call tirust_tensor_evaluate(1,0,qp,qs,qm,mass_qp,qr,status,check_pass=.true.)
  call require_success('repeated binary128 check pass')
  if (any(abs(qr-qreference)>0.0_KI_QTIR)) stop 47
  call tirust_qp_cache_statistics(qp_counts,status)
  call require_success('repeated check QP cache statistics')
  if (qp_counts(1)/=qp_before(1)+1_c_int64_t .or. any(qp_counts(2:6)/=qp_before(2:6))) stop 48

  ! A lightlike scaleless bubble has UV=+1 and IR=-1 separately.
  bp = 0
  bp(0,2) = 2
  bp(3,2) = 2
  bs = 0
  bm = cmplx(0,0,kind=KI_TIR)
  do pole_mode=TIRUST_POLES_BOTH,TIRUST_POLES_NONE
    call tirust_tensor_evaluate(2,0,bp,bs,bm,4.0_KI_TIR,r,status,pole_mode=pole_mode)
    call require_success('separate scaleless bubble poles')
    if (any(abs(r(1,3:4))>1e-12_KI_TIR)) stop 11
    if (pole_mode == TIRUST_POLES_BOTH .or. pole_mode == TIRUST_POLES_UV) then
      if (abs(r(1,1)-1)>1e-12_KI_TIR) stop 12
    else
      if (abs(r(1,1))>1e-12_KI_TIR) stop 13
    endif
    if (pole_mode == TIRUST_POLES_BOTH .or. pole_mode == TIRUST_POLES_IR) then
      if (abs(r(1,2)+1)>1e-12_KI_TIR) stop 14
    else
      if (abs(r(1,2))>1e-12_KI_TIR) stop 15
    endif
  enddo
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status,compute_finite=.false.)
  if (tirust_supports(1,0,.false.)) then
    call require_success('poles-only tadpole')
    if (abs(r(1,1)-4)>1e-12_KI_TIR .or. any(abs(r(1,2:4))>1e-12_KI_TIR)) stop 16
    print *, 'TIRust poles-only evaluation passed'
  else
    if (status /= TIRUST_UNSUPPORTED .or. any(abs(r)>0.0_KI_TIR)) stop 24
    print *, 'TIRust poles-only evaluation is unavailable in this build'
  endif
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status)
  call require_success('restore finite terms')
  if (any(abs(r-reference)>1e-12_KI_TIR)) stop 17
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status,pole_mode=99)
  if (status == TIRUST_SUCCESS .or. any(abs(r)>0.0_KI_TIR)) stop 18
  call tirust_error_message(message)
  if (len_trim(message) == 0) stop 19
  if (tirust_context_generation() /= first_generation) stop 30
  call tirust_finalize(status)
  call require_success('finalize')
  if (tirust_context_generation() /= -1) stop 31
  call tirust_finalize(status)
  call require_success('repeated finalize')
  if (tirust_context_generation() /= -1) stop 32
  if (.not.tirust_supports(1,0,.true.)) stop 22
  if (tirust_context_generation() /= -1) stop 33
  call tirust_tensor_evaluate(1,0,p,s,m,4.0_KI_TIR,r,status)
  call require_success('recreate context')
  if (tirust_context_generation() <= first_generation) stop 34
  if (any(abs(r-reference)>1e-12_KI_TIR)) stop 20
  call tirust_finalize(status)
  call require_success('final teardown')
  if (tirust_context_generation() /= -1) stop 35
  call check_rounding()
  print *, 'TIRust tensor-route statistics passed'
  print *, 'TIRust ABI1: DP, binary128 input, UV/IR poles and cache checks passed'
contains
  subroutine require_success(label)
    character(len=*), intent(in) :: label
    if (status == TIRUST_SUCCESS) return
    call tirust_error_message(message)
    print *, trim(label), ': ', trim(message), ' (status=',status,')'
    stop 1
  end subroutine

  subroutine check_rounding()
    real(KI_QTIR), volatile :: x,y,total
    type(ieee_round_type) :: original
    call ieee_get_rounding_mode(original)
    x = 1.0_KI_QTIR
    y = 2.0_KI_QTIR**(-114)
    call ieee_set_rounding_mode(ieee_up)
    total = x+y
    call ieee_set_rounding_mode(original)
    if (total /= nearest(1.0_KI_QTIR,1.0_KI_QTIR)) stop 21
  end subroutine
end program smoke
