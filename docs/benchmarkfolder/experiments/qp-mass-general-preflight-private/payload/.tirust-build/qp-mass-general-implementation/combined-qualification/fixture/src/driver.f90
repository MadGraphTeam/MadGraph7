program caller_boundary_fixture
  use iso_fortran_env, only: int64,real64,real128
  use ieee_arithmetic
  use ieee_exceptions
  use aloha_object
  use fixture_recording
  implicit none
  include 'coupl.inc'
  include 'mp_coupl.inc'
  real(real64) :: lscale,p(0:3,3)
  integer :: ctmode,nhel(3),ic(3),h,i,j,trap_mode,ios
  common/FX_CT/lscale,ctmode
  logical :: goodamp(1,13),checkphase,heldoublechecked
  integer :: goodhel(1),heloffset,sqso_target,i_so,i_lib
  common/FX_Filters/goodamp,goodhel,heloffset
  common/FX_INIT/checkphase,heldoublechecked
  common/FX_SOCHOICE/sqso_target
  common/FX_I_SO/i_so
  common/FX_I_LIB/i_lib
  logical :: uvct_req_so_done,mp_uvct_req_so_done,ct_req_so_done,mp_ct_req_so_done
  logical :: loop_req_so_done,mp_loop_req_so_done,ctcall_req_so_done,filter_so
  common/FX_SO_REQS/uvct_req_so_done,mp_uvct_req_so_done,ct_req_so_done,mp_ct_req_so_done, &
       loop_req_so_done,mp_loop_req_so_done,ctcall_req_so_done,filter_so
  type(aloha) :: w(20)
  common/FX_W/w
  complex(real64) :: wl(4,0:34,4,-1:1),pl(0:3,-1:1),amp(1),ampl(3,13)
  common/FX_WL/wl,pl
  common/FX_AMPS/amp
  common/FX_AMPL/ampl
  character(32) :: arm,payload,route,arg
  logical :: flags(5),setup_flags(5),before_flags(5),original_halting(5)
  integer(int64) :: one_words(2),complex_words(4),poison_high,expected_words(4)
  integer(int64) :: real_words(2),derived_words(2),cms_words(4)
  type(ieee_status_type) :: original_status,saved_observer
  real(real64),volatile :: invalid_zero,invalid_result

  if(command_argument_count()/=5) error stop 60
  call get_command_argument(1,arm)
  call get_command_argument(2,arg)
  read(arg,*,iostat=ios) ctmode
  if(ios/=0) error stop 61
  call get_command_argument(3,payload)
  call get_command_argument(4,arg)
  read(arg,*,iostat=ios) trap_mode
  if(ios/=0.or.(trap_mode/=0.and.trap_mode/=1)) error stop 62
  call get_command_argument(5,route)
  if(ctmode < -1.or.ctmode > 6) error stop 63
  if(trim(arm)/='legacy'.and.trim(arm)/='guarded'.and.trim(arm)/='trap_control') error stop 64
  if(trim(route)/='grouped'.and.trim(route)/='ungrouped'.and.trim(route)/='loop_induced') error stop 65
  if(ctmode>=4.and.trim(payload)/='valid') error stop 66
  if(.not.ieee_support_datatype(0.0_real128)) error stop 67
  if(.not.ieee_support_halting(ieee_invalid)) error stop 68
  call ieee_get_status(original_status)
  do i=1,5
    original_halting(i)=.false.
    if(ieee_support_halting(tracked(i))) then
      call ieee_get_halting_mode(tracked(i),original_halting(i))
      call ieee_set_halting_mode(tracked(i),.false.)
    endif
  enddo
  if(storage_size(0.0_real64)/=64.or.storage_size(0.0_real128)/=128) error stop 69
  if(storage_size(cmplx(0.0_real128,0.0_real128,real128))/=256.or.storage_size(0)/=32) error stop 70
  one_words=transfer(1.0_real128,one_words)
  if(any(one_words/=[0_int64,int(z'3FFF000000000000',int64)])) error stop 72
  complex_words=transfer(cmplx(1.0_real128,-2.0_real128,real128),complex_words)
  if(any(complex_words/=[0_int64,int(z'3FFF000000000000',int64),0_int64,int(z'C000000000000000',int64)])) error stop 73
  open(newunit=log_unit,file='caller.hex',status='new',action='write')
  write(log_unit,'(A)') 'FORMAT QPMASS_CALLER_FIXTURE_V1'
  write(log_unit,'(A,1X,A,1X,I0,1X,A,1X,I0,1X,A)') 'CASE',trim(arm),ctmode,trim(payload),trap_mode,trim(route)
  write(log_unit,'(A)') 'FLAG_ORDER invalid divide_by_zero overflow underflow inexact'
  write(log_unit,'(A,2(1X,Z16.16))') 'BINARY128_ONE',one_words

  ! Every DP argument and every referenced COMMON control is initialized.
  lscale=1000.0_real64;p=0;nhel=1;ic=1;h=1
  goodamp=.true.;goodhel=1;heloffset=0;checkphase=.false.;heldoublechecked=.false.
  sqso_target=-1;i_so=1;i_lib=1
  uvct_req_so_done=.false.;mp_uvct_req_so_done=.false.;ct_req_so_done=.false.;mp_ct_req_so_done=.false.
  loop_req_so_done=.false.;mp_loop_req_so_done=.false.;ctcall_req_so_done=.false.;filter_so=.false.
  wl=0;pl=0;amp=0;ampl=0
  do i=1,20
    w(i)%w=0;w(i)%p=0;w(i)%flv_index=1
  enddo
  call_count=0;legacy_count=0;qp_count=0;observer_changes=0
  synthetic_status=0;synthetic_result=0
  MDL_REAL=3.0_real64
  MDL_DERIVED=(MDL_REAL+1.0_real64/10.0_real64)/3.0_real64
  MDL_CMS=5.0_real64;MDL_WIDTH=0.125_real64
  CMASS_MDL_CMS=cmplx(5.0_real64,-0.125_real64,real64)
  call ieee_set_flag(tracked,.false.)
  select case(trim(payload))
  case('valid')
    MP__MDL_REAL=3.0_real128
    MP__MDL_DERIVED=(MP__MDL_REAL+1.0_real128/10.0_real128)/3.0_real128
    MP__CMASS_MDL_CMS=cmplx(5.0_real128,-0.125_real128,real128)
  case('finite_poison')
    MP__MDL_REAL=17.5_real128
    MP__MDL_DERIVED=-19.25_real128
    cms_words=[0_int64,int(z'4003580000000000',int64),0_int64,int(z'8000000000000000',int64)]
    MP__CMASS_MDL_CMS=transfer(cms_words,MP__CMASS_MDL_CMS)
  case('qnan','snan')
    ! Controlled absent MP initialization: all writable fields are explicitly
    ! poisoned. This is not literal uninitialized-memory experimentation.
    poison_high=int(z'7FFF000000000000',int64)
    if(trim(payload)=='qnan') poison_high=int(z'7FFF800000000000',int64)
    real_words=[int(z'11',int64),poison_high]
    derived_words=[int(z'22',int64),ibset(poison_high,63)]
    cms_words=[int(z'33',int64),poison_high,int(z'44',int64),ibset(poison_high,63)]
    MP__MDL_REAL=transfer(real_words,MP__MDL_REAL)
    MP__MDL_DERIVED=transfer(derived_words,MP__MDL_DERIVED)
    MP__CMASS_MDL_CMS=transfer(cms_words,MP__CMASS_MDL_CMS)
    one_words=transfer(MP__MDL_REAL,one_words)
    if(any(one_words/=real_words)) error stop 74
    one_words=transfer(MP__MDL_DERIVED,one_words)
    if(any(one_words/=derived_words)) error stop 75
    complex_words=transfer(MP__CMASS_MDL_CMS,complex_words)
    if(any(complex_words/=cms_words)) error stop 76
  case default
    error stop 77
  end select
  call ieee_get_flag(tracked,setup_flags)
  call emit_flags('SETUP',setup_flags)
  call emit_dp_real('ZERO',0.0_real64)
  call emit_dp_real('MDL_REAL',MDL_REAL)
  call emit_dp_real('MDL_DERIVED',MDL_DERIVED)
  call emit_dp_complex('CMASS_MDL_CMS',CMASS_MDL_CMS)
  call emit_mp_real('MP__ZERO',0.0_real128)
  call emit_mp_real('MP__MDL_REAL',MP__MDL_REAL)
  call emit_mp_real('MP__MDL_DERIVED',MP__MDL_DERIVED)
  call emit_mp_complex('MP__CMASS_MDL_CMS',MP__CMASS_MDL_CMS)
  call ieee_set_flag(tracked,.false.)
  call ieee_get_flag(tracked,before_flags)
  call emit_flags('BEFORE',before_flags)
  write(log_unit,'(A)') 'PRECALL'
  flush(log_unit)
  call ieee_set_flag(tracked,.false.)
  if(trap_mode==1) call ieee_set_halting_mode(ieee_invalid,.true.)
  if(trim(arm)=='trap_control') then
    invalid_zero=0.0_real64
    invalid_result=invalid_zero/invalid_zero
  else
    if(trim(arm)=='legacy') then
      select case(trim(route))
      case('grouped')
        CALL FX_legacy_grouped_1(P,NHEL,H,IC)
        CALL FX_legacy_grouped_2(P,NHEL,H,IC)
        CALL FX_legacy_grouped_3(P,NHEL,H,IC)
      case('ungrouped')
        CALL FX_legacy_ungrouped_1(P,NHEL,H,IC)
        CALL FX_legacy_ungrouped_2(P,NHEL,H,IC)
        CALL FX_legacy_ungrouped_3(P,NHEL,H,IC)
      case('loop_induced')
        CALL FX_legacy_loop_induced_1(P,NHEL,H,IC)
        CALL FX_legacy_loop_induced_2(P,NHEL,H,IC)
        CALL FX_legacy_loop_induced_3(P,NHEL,H,IC)
      end select
    else if(trim(arm)=='guarded') then
      select case(trim(route))
      case('grouped')
        CALL FX_guarded_grouped_1(P,NHEL,H,IC)
        CALL FX_guarded_grouped_2(P,NHEL,H,IC)
        CALL FX_guarded_grouped_3(P,NHEL,H,IC)
      case('ungrouped')
        CALL FX_guarded_ungrouped_1(P,NHEL,H,IC)
        CALL FX_guarded_ungrouped_2(P,NHEL,H,IC)
        CALL FX_guarded_ungrouped_3(P,NHEL,H,IC)
      case('loop_induced')
        CALL FX_guarded_loop_induced_1(P,NHEL,H,IC)
        CALL FX_guarded_loop_induced_2(P,NHEL,H,IC)
        CALL FX_guarded_loop_induced_3(P,NHEL,H,IC)
      end select
    endif
  endif
  call ieee_get_flag(tracked,flags)
  call ieee_set_halting_mode(ieee_invalid,.false.)
  call emit_flags('AFTER',flags)
  call observer_begin(saved_observer,before_flags)
  write(log_unit,'(A,3(1X,I0))') 'COUNTS',call_count,legacy_count,qp_count
  call observer_end(saved_observer,before_flags)
  write(log_unit,'(A,2(1X,I0))') 'COMPLETE',call_count,observer_changes
  close(log_unit)
  call ieee_set_status(original_status)
  do i=1,5
    if(ieee_support_halting(tracked(i))) call ieee_set_halting_mode(tracked(i),original_halting(i))
  enddo
end program caller_boundary_fixture
