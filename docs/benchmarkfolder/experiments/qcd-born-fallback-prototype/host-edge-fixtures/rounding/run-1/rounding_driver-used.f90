! Policy fixture only: the tested wrapper is copied verbatim from born_rescue.f.
! The two external Born procedures below are explicit recording stubs.
module rounding_fixture
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer :: action, dp_calls, qp_calls, dp_round, qp_round
  integer :: userhel, rescue_count, fail_count
  logical :: helreset, input_ok
  common /ML5_0_HELUSERCHOICE/ userhel
  common /ML5_0_HELRESET/ helreset
  common /AUDIT_BORN_RESCUE_COUNTS/ rescue_count, fail_count
  integer(int64) :: input_bits(0:3,5)
contains
  integer function round_id()
    type(ieee_round_type) :: current
    call ieee_get_rounding_mode(current)
    round_id=0
    if (current==ieee_nearest) round_id=1
    if (current==ieee_to_zero) round_id=2
    if (current==ieee_up) round_id=3
    if (current==ieee_down) round_id=4
    if (current==ieee_away) round_id=5
  end function

  integer function mask(values)
    logical, intent(in) :: values(:)
    integer :: i
    mask=0
    do i=1,size(values)
      if (values(i)) mask=ibset(mask,i-1)
    enddo
  end function

  integer function flag_bit(flag)
    type(ieee_flag_type), intent(in) :: flag
    integer :: i
    flag_bit=0
    do i=1,size(ieee_all)
      if (ieee_all(i)==flag) flag_bit=ibset(0,i-1)
    enddo
  end function
end module

program check_rounding
  use rounding_fixture
  implicit none
  type(ieee_status_type) :: original_status
  type(ieee_round_type) :: modes(5)
  logical :: supported_dp, supported_qp, rescued, ok
  logical :: entry_flags(size(ieee_all)), exit_flags(size(ieee_all))
  logical :: entry_traps(size(ieee_all)), exit_traps(size(ieee_all))
  real*8 :: p(0:3,5), answer(0:1)
  integer(int64) :: answer_bits(0:1), expected_bits(0:1)
  integer :: imode, seed, trap, i, j, cases, failures, errors
  integer :: before_round, after_round, expected_flags, before_rescue, before_fail
  integer :: last_underflow, last_inexact, last_invalid, last_overflow
  integer :: supported_modes

  call ieee_get_status(original_status)
  modes=[ieee_nearest,ieee_to_zero,ieee_up,ieee_down,ieee_away]
  last_underflow=flag_bit(ieee_underflow)
  last_inexact=flag_bit(ieee_inexact)
  last_invalid=flag_bit(ieee_invalid)
  last_overflow=flag_bit(ieee_overflow)
  do i=1,size(ieee_all)
    if (.not.ieee_support_flag(ieee_all(i))) stop 81
    if (.not.ieee_support_halting(ieee_all(i))) stop 82
  enddo
  write(*,'(A,5(1X,I0))') 'FLAG_BITS',size(ieee_all),last_underflow, &
       last_inexact,last_invalid,last_overflow
  do j=1,5
    do i=0,3
      p(i,j)=real(i+7*j,kind=8)/8d0
    enddo
  enddo
  p(0,1)=transfer(int(z'8000000000000000',int64),0d0)
  input_bits=reshape(transfer(p,input_bits),shape(input_bits))
  cases=0
  failures=0
  supported_modes=0
  do imode=1,size(modes)
    supported_dp=ieee_support_rounding(modes(imode),0d0)
    supported_qp=ieee_support_rounding(modes(imode),0q0)
    write(*,'(A,3(1X,I0))') 'SUPPORT',imode,merge(1,0,supported_dp), &
         merge(1,0,supported_qp)
    ! The wrapper's externally visible conversion and mode are DP. Record
    ! QP support separately; the stubs themselves use exact QP constants.
    if (.not.supported_dp) cycle
    supported_modes=supported_modes+1
    do action=1,4
      do seed=0,3
        do trap=0,1
          ! Changing trap masks can clear flags on this compiler. Configure
          ! traps first, then seed/capture flags, with no setters between the
          ! wrapper return and the exit-state snapshot.
          call ieee_set_halting_mode(ieee_all,.false.)
          call ieee_set_flag(ieee_all,.false.)
          call ieee_set_rounding_mode(modes(imode))
          if (trap==1) call ieee_set_halting_mode(ieee_divide_by_zero,.true.)
          if (btest(seed,0)) call ieee_set_flag(ieee_underflow,.true.)
          if (btest(seed,1)) call ieee_set_flag(ieee_inexact,.true.)
          call ieee_get_flag(ieee_all,entry_flags)
          call ieee_get_halting_mode(ieee_all,entry_traps)
          before_round=round_id()
          expected_flags=0
          if (btest(seed,0)) expected_flags=ior(expected_flags,last_underflow)
          if (btest(seed,1)) expected_flags=ior(expected_flags,last_inexact)
          if (mask(entry_flags)/=expected_flags) stop 83
          if (mask(entry_traps)/=trap*flag_bit(ieee_divide_by_zero)) stop 84
          if (before_round/=imode) stop 85
          dp_calls=0
          qp_calls=0
          dp_round=0
          qp_round=0
          input_ok=.true.
          userhel=-1
          helreset=.false.
          before_rescue=rescue_count
          before_fail=fail_count
          answer=0d0
          call ML5_0_BORN_WITH_RESCUE(p,7,1,answer,rescued,ok)
          call ieee_get_flag(ieee_all,exit_flags)
          call ieee_get_halting_mode(ieee_all,exit_traps)
          after_round=round_id()
          answer_bits=transfer(answer,answer_bits)
          ! Snapshot is complete. Make assertions/output independent of traps
          ! and sticky flags; the next case establishes a fresh environment.
          call ieee_set_halting_mode(ieee_all,.false.)
          call ieee_set_flag(ieee_all,.false.)
          errors=0
          if (after_round/=before_round) errors=ibset(errors,0)
          if (any(exit_traps.neqv.entry_traps)) errors=ibset(errors,1)
          if (dp_calls/=1.or.dp_round/=imode) errors=ibset(errors,2)
          if (.not.input_ok.or.any(reshape(transfer(p,input_bits), &
               shape(input_bits))/=input_bits)) errors=ibset(errors,3)
          if (userhel/=-1) errors=ibset(errors,4)
          select case(action)
          case(1)
            expected_flags=ior(expected_flags,last_inexact)
            expected_bits=[int(z'8000000000000000',int64),int(z'3FF0000000000000',int64)]
            if (.not.ok.or.rescued.or.qp_calls/=0.or.helreset) errors=ibset(errors,5)
            if (any(answer_bits/=expected_bits)) errors=ibset(errors,6)
            if (rescue_count/=before_rescue.or.fail_count/=before_fail) errors=ibset(errors,7)
          case(2)
            expected_flags=ior(expected_flags,last_inexact)
            expected_bits=[int(z'3FF0000000000000',int64),int(z'BFF0000000000000',int64)]
            if (imode==3.or.imode==5) expected_bits(0)=int(z'3FF0000000000001',int64)
            if (imode==4.or.imode==5) expected_bits(1)=int(z'BFF0000000000001',int64)
            if (.not.ok.or..not.rescued.or..not.helreset) errors=ibset(errors,5)
            if (any(answer_bits/=expected_bits)) errors=ibset(errors,6)
            if (rescue_count/=before_rescue+1.or.fail_count/=before_fail) errors=ibset(errors,7)
          case(3,4)
            if (action==4) expected_flags=ior(expected_flags,ior(last_overflow,last_inexact))
            if (ok.or..not.rescued.or..not.helreset) errors=ibset(errors,5)
            if (.not.all(ieee_is_nan(answer))) errors=ibset(errors,6)
            if (rescue_count/=before_rescue+1.or.fail_count/=before_fail+1) errors=ibset(errors,7)
          end select
          if (action/=1) then
            if (qp_calls/=1.or.qp_round/=imode) errors=ibset(errors,8)
          endif
          if (mask(exit_flags)/=expected_flags) errors=ibset(errors,9)
          cases=cases+1
          if (errors/=0) failures=failures+1
          write(*,'(A,16(1X,I0),2(1X,Z16.16))') 'CASE',imode,action,seed,trap, &
               before_round,after_round,mask(entry_flags),mask(exit_flags),expected_flags, &
               mask(entry_traps),mask(exit_traps),merge(1,0,rescued),merge(1,0,ok), &
               dp_calls,qp_calls,errors,answer_bits
        enddo
      enddo
    enddo
  enddo
  call ieee_set_status(original_status)
  write(*,'(A,3(1X,I0))') 'RESULT',supported_modes,cases,failures
  if (supported_modes==0) stop 86
  if (failures/=0) stop 91
end program

subroutine ML5_0_SMATRIXHEL_SPLITORDERS(p,hel,flav_idx,answer)
  use rounding_fixture
  implicit none
  real*8 :: p(0:3,5),answer(0:1)
  integer :: hel,flav_idx
  dp_calls=dp_calls+1
  dp_round=round_id()
  input_ok=input_ok.and.hel==7.and.flav_idx==1
  input_ok=input_ok.and.all(reshape(transfer(p,input_bits),shape(input_bits))==input_bits)
  if (action==1) then
    answer=[p(0,1),1d0]
    call ieee_set_flag(ieee_inexact,.true.)
  else
    answer=ieee_value(0d0,ieee_quiet_nan)
    ! Deliberate failed-DP flag: it must not leak into successful QP rescue.
    call ieee_set_flag(ieee_invalid,.true.)
  endif
end subroutine

subroutine AUDIT_QP_SMATRIX_SPLITORDERS(p,flav_idx,answer)
  use rounding_fixture
  implicit none
  real*16 :: p(0:3,5),answer(0:1)
  real*8 :: transported(0:3,5)
  integer :: flav_idx
  qp_calls=qp_calls+1
  qp_round=round_id()
  transported=real(p,kind=8)
  input_ok=input_ok.and.flav_idx==1.and.userhel==7
  input_ok=input_ok.and.all(reshape(transfer(transported,input_bits),shape(input_bits))==input_bits)
  select case(action)
  case(2)
    ! Exactly representable in QP and exactly halfway between two DP values.
    answer(0)=1q0+2q0**(-53)
    answer(1)=-answer(0)
  case(3)
    answer(0)=ieee_value(0q0,ieee_quiet_nan)
    answer(1)=7q0
  case(4)
    ! Finite QP; REAL(...,8) overflows even when directed rounding saturates
    ! at the largest finite DP number. The wrapper must reject either result.
    answer=real(huge(0d0),kind=16)*2q0
  case default
    stop 87
  end select
end subroutine
