program repair_fixture
  use repair_control
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  integer::ImprovePSPoint
  common/FX_REPAIR_MODE/ImprovePSPoint
  real(8)::p(0:3,5),p_before(0:3,5)
  real(16)::c(0:3,5)
  integer::err_primary,err_alternate,outcome
  logical::adopted,sticky,mode_before,mode_after
  type(ieee_round_type)::rounding_before,rounding_after
  p=7.25d0
  p_before=p
  call ieee_get_halting_mode(ieee_invalid,mode_before)
  call ieee_get_rounding_mode(rounding_before)
  call check('psmc_primary',1,0,0,.true.,1,[1,0],101.0_16)
  call check('orig_primary',2,0,0,.true.,1,[2,0],202.0_16)
  call check('mode_zero',0,0,0,.true.,1,[2,0],202.0_16)
  call check('mode_negative',-7,0,0,.true.,1,[2,0],202.0_16)
  call check('psmc_to_orig',1,999,0,.true.,2,[1,2],202.0_16)
  call check('orig_to_psmc',2,999,0,.true.,2,[2,1],101.0_16)
  call check('both_failed',1,999,1111,.true.,-2,[1,2],-99.0_16)
  call check('nonphysical',1,0,0,.false.,-1,[0,0],-99.0_16)
  call check('unsupported',3,0,0,.true.,-3,[0,0],-99.0_16)
  write(*,'(A)')'ALL PASS'
contains
  subroutine require(condition,message)
    logical,intent(in)::condition
    character(*),intent(in)::message
    if(.not.condition)then
      write(*,'(A)')'FAIL '//message
      error stop 1
    endif
  end subroutine
  subroutine check(name,mode,first,second,isphysical,want_outcome,trace,want_value)
    character(*),intent(in)::name
    integer,intent(in)::mode,first,second,want_outcome,trace(2)
    logical,intent(in)::isphysical
    real(16),intent(in)::want_value
    integer::expected_primary,expected_alternate,expected_physical
    ImprovePSPoint=mode
    requested_primary=first
    requested_alternate=second
    physical=isphysical
    repair_calls=0
    physical_calls=0
    method_trace=0
    c=-99.0_16
    adopted=.true.
    outcome=123
    err_primary=-987
    err_alternate=-654
    call ieee_set_flag(ieee_divide_by_zero,.true.)
    call RX_PREPARE_MP_PS_CHECKED(p,c,err_primary,err_alternate,outcome,adopted)
    call require(outcome==want_outcome,'outcome '//name)
    call require(adopted.eqv.(want_outcome>0),'explicit adoption '//name)
    call require(all(c==want_value),'published workspace or preserved sentinel '//name)
    call require(all(p==p_before),'original DP point unchanged '//name)
    call require(all(method_trace==trace),'ordered helper calls '//name)
    call require(repair_calls==count(trace/=0),'helper call count '//name)
    expected_primary=first
    expected_alternate=second
    expected_physical=1
    if(want_outcome==-1)then
      expected_primary=1
      expected_alternate=0
    elseif(want_outcome==-3)then
      expected_primary=0
      expected_alternate=0
      expected_physical=0
    endif
    call require(err_primary==expected_primary,'raw primary error '//name)
    call require(err_alternate==expected_alternate,'raw alternate error '//name)
    call require(physical_calls==expected_physical,'physical check count '//name)
    call ieee_get_flag(ieee_divide_by_zero,sticky)
    call ieee_get_halting_mode(ieee_invalid,mode_after)
    call ieee_get_rounding_mode(rounding_after)
    call require(sticky,'entry sticky flag retained '//name)
    call require(mode_before.eqv.mode_after,'invalid trap preserved '//name)
    call require(rounding_before==rounding_after,'rounding preserved '//name)
    write(*,'(A)')'PASS '//name
  end subroutine
end program
