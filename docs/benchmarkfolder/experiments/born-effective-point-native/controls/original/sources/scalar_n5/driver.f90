program born_control
  use fx_control
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  implicit none
  integer, parameter :: n=5,ns=1
  logical, parameter :: split=.false.
  real(8) :: p(0:3,n),p_saved(0:3,n),ans(0:ns),ans_old(0:ns)
  real(8) :: beampol(2)
  integer :: userhel,old_userhel,nhel(n,4),old_nhel(n,4)
  integer :: pol(0:n,0:5),old_pol(0:n,0:5)
  logical :: helreset,old_helreset,ok
  common /TX_HELUSERCHOICE/userhel
  common /LEG_HELUSERCHOICE/old_userhel
  common /TX_PROCESS_NHEL/nhel
  common /LEG_PROCESS_NHEL/old_nhel
  common /TX_BORN_BEAM_POL/pol
  common /LEG_BORN_BEAM_POL/old_pol
  common /TX_helreset/helreset
  common /LEG_helreset/old_helreset
  common /TO_BEAMPOL/beampol
  integer :: tries(2),old_tries(2),before_tries(2)
  logical :: good(4,2),old_good(4,2),before_good(4,2)
  logical :: reset_observed,old_reset_observed,before_reset
  logical :: modes_before(3),modes_after(3),sticky
  integer :: h,i,calls_old,sequence_old(32)
  type(ieee_round_type) :: rounding_before,rounding_after

  ! This driver controls flags; the generated routine never clears/masks them.
  ! Nonfinite-rejection tests need traps disabled on process entry. Enabled-trap
  ! behavior is deliberately reserved for separate child-process qualification.
  call get_modes(modes_before)
  call require(.not.any(modes_before),'fixture requires default disabled traps')
  call ieee_get_rounding_mode(rounding_before)
  p=0d0
  do i=1,n
    p(0,i)=real(i,8)
  enddo
  p_saved=p
  nhel=1
  do h=1,4
    nhel(n,h)=h
  enddo
  old_nhel=nhel
  pol=-1
  old_pol=-1
  beampol=0d0
  call finite_table()
  old_helreset=.true.
  helreset=.true.
  old_userhel=3
  userhel=3
  fx_calls=0
  fx_sequence=0
  call LEG_SMATRIXHEL(p,-1,1,ans_old(1))
  ans_old(0)=ans_old(1)
  calls_old=fx_calls
  sequence_old=fx_sequence
  fx_calls=0
  fx_sequence=0
  call TX_SMATRIXHEL_SPLITORDERS_CHECKED(p,-1,1,ans,ok)
  call require(ok,'finite accepted')
  call require(all(transfer(ans,[0_int64],ns+1)==transfer(ans_old,[0_int64],ns+1)), &
               'finite bit equality with exact original aggregate')
  call require(all(ans==[2.5d0,2.5d0]),'finite independently tabulated sum')
  call require(fx_calls==calls_old.and.all(fx_sequence==sequence_old),'finite call trace')
  call TX_OBSERVE_LEARNING(tries,good,reset_observed)
  call LEG_OBSERVE_LEARNING(old_tries,old_good,old_reset_observed)
  call require(all(tries==old_tries).and.all(good.eqv.old_good),'finite learning equality')
  call require(reset_observed.eqv.old_reset_observed,'finite reset equality')
  call require(userhel==-1.and.old_userhel==-1,'finite wrapper selection')
  call require(all(p==p_saved),'finite original input unchanged')
  call passed('finite_same_bits_and_learning')

  helreset=.true.
  userhel=3
  fx_calls=0
  call TX_SMATRIXHEL_SPLITORDERS_CHECKED(p,-1,0,ans,ok)
  call TX_OBSERVE_LEARNING(tries,good,reset_observed)
  call require(ok.and.all(ans==0d0).and.fx_calls==0,'invalid flavor zero')
  call require(all(tries==0).and..not.any(good),'invalid flavor reset all arrays')
  call require(.not.reset_observed.and.userhel==-1,'invalid flavor consumes reset')
  call passed('invalid_flavor_zero')

  call finite_table()
  fx_calls=0
  call TX_SMATRIXHEL_SPLITORDERS_CHECKED(p,2,1,ans,ok)
  call require(ok.and.fx_calls==1,'fixed helicity call')
  call require(all(ans==[2d0,2d0]),'fixed helicity values')
  call require(userhel==-1,'fixed success selection')
  call passed('fixed_helicity_success')

  call prepare_rejection()
  if(split)then
    fx_values(2,2)=ieee_value(0d0,ieee_quiet_nan)
  else
    fx_values(1,2)=ieee_value(0d0,ieee_quiet_nan)
  endif
  call reject_case('raw_nan',2)

  call prepare_rejection()
  fx_values(1,2)=ieee_value(0d0,ieee_positive_inf)
  call reject_case('raw_infinity',2)

  call prepare_rejection()
  pol(0,0)=1
  fx_selected(1)=.false.
  fx_values(1,1)=ieee_value(0d0,ieee_quiet_nan)
  call reject_case('excluded_helicity_nan',1)

  call prepare_rejection()
  beampol(1)=2d0
  fx_values(1,1)=huge(1d0)*0.75d0
  call reject_case('beam_weight_overflow',1)

  call prepare_rejection()
  fx_values=0d0
  fx_values(1,1:2)=huge(1d0)*0.75d0
  call reject_case('partial_sum_overflow',2)

  call prepare_rejection()
  fx_values=0d0
  if(split)then
    ! An unchosen order can overflow while selected ANS(0) remains zero.
    fx_values(2,1)=huge(1d0)*0.75d0
  else
    fx_values(1,1)=huge(1d0)*0.75d0
  endif
  fx_symmetry=8
  call reject_case('final_normalization_overflow',4)

  

  call prepare_rejection()
  fx_values=0d0
  call TX_SMATRIXHEL_SPLITORDERS_CHECKED(p,-1,1,ans,ok)
  call require(ok.and.all(ans==0d0),'genuine zero accepted')
  call require(userhel==-1,'zero success selection')
  call passed('genuine_zero')
  call get_modes(modes_after)
  call ieee_get_rounding_mode(rounding_after)
  call require(all(modes_before.eqv.modes_after),'trap modes preserved')
  call require(rounding_before==rounding_after,'rounding preserved')
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
  subroutine passed(name)
    character(*),intent(in)::name
    write(*,'(A)')'PASS '//name
  end subroutine
  subroutine get_modes(modes)
    logical,intent(out)::modes(3)
    call ieee_get_halting_mode(ieee_invalid,modes(1))
    call ieee_get_halting_mode(ieee_divide_by_zero,modes(2))
    call ieee_get_halting_mode(ieee_overflow,modes(3))
  end subroutine
  subroutine finite_table()
    integer::j
    fx_symmetry=1
    fx_selected=.true.
    do j=1,4
      fx_values(:,j)=[real(j,8),2d0*j,3d0*j]
    enddo
  end subroutine
  subroutine prepare_rejection()
    pol=-1
    beampol=0d0
    call finite_table()
    helreset=.true.
    call TX_SMATRIXHEL_SPLITORDERS_CHECKED(p,-1,0,ans,ok)
    call require(ok,'setup invalid flavor reset')
    call TX_SMATRIXHEL_SPLITORDERS_CHECKED(p,-1,2,ans,ok)
    call require(ok,'setup flavor-two learning')
    call TX_OBSERVE_LEARNING(tries,good,reset_observed)
    call require(all(tries==[0,1]).and..not.reset_observed,'setup learned state')
    ! Failure must undo reset/all-flavor changes and earlier helicity learning.
    helreset=.true.
    userhel=3
    fx_calls=0
    fx_sequence=0
  end subroutine
  subroutine reject_case(name,expected_calls)
    character(*),intent(in)::name
    integer,intent(in)::expected_calls
    call TX_OBSERVE_LEARNING(before_tries,before_good,before_reset)
    call ieee_set_flag(ieee_all,.false.)
    call ieee_set_flag(ieee_divide_by_zero,.true.)
    call TX_SMATRIXHEL_SPLITORDERS_CHECKED(p,-1,1,ans,ok)
    call require(.not.ok,'rejected status '//name)
    call require(all(ieee_is_nan(ans)),'all output failure markers '//name)
    call TX_OBSERVE_LEARNING(tries,good,reset_observed)
    call require(all(tries==before_tries),'NTRY rollback '//name)
    call require(all(good.eqv.before_good),'GOODHEL rollback '//name)
    call require(reset_observed.eqv.before_reset,'HELRESET rollback '//name)
    call require(userhel==3,'USERHEL rollback '//name)
    call require(fx_calls==expected_calls,'failure call boundary '//name)
    call require(all(p==p_saved),'original input unchanged '//name)
    call ieee_get_flag(ieee_divide_by_zero,sticky)
    call require(sticky,'entry sticky flag preserved '//name)
    call get_modes(modes_after)
    call ieee_get_rounding_mode(rounding_after)
    call require(all(modes_before.eqv.modes_after),'trap modes preserved '//name)
    call require(rounding_before==rounding_after,'rounding preserved '//name)
    call passed(name)
  end subroutine
end program
