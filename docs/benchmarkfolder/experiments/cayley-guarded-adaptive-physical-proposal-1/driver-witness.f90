! Private diagnostic main only. All linked numerical objects remain unchanged.
program adaptive_checked_trap
  use iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  use ew_mass_observer, only: obs_close
  implicit none
  ! Exact frozen coupl.inc declarations; this main uses only the scale COMMON.
  double precision :: mu_r, all_mu_r
  common /RSCALE/ mu_r, all_mu_r
  type(ieee_flag_type), parameter :: flags(5) = &
       [ieee_invalid,ieee_divide_by_zero,ieee_overflow,ieee_underflow,ieee_inexact]
  character(16), parameter :: names(5) = &
       [character(16)::'invalid','divide_by_zero','overflow','underflow','inexact']
  integer, parameter :: logunit=917
  integer :: ndim,nso,k,i,code,npoints,initialized,tens,observation_changes=0
  real(8) :: points(0:3,5,2),p(0:3,5),total(0:3),ans(0:3,0:1),acc(0:1)
  real(8), volatile :: x,y,z
  integer(int64) :: accuracy_words(2)
  character(1024) :: argument,policy,fault
  logical :: trapped

  call get_command_argument(1,argument)
  call get_command_argument(2,policy)
  open(logunit,file='checked-control.log',status='new',action='write')
  write(logunit,'(A)') 'FORMAT ADAPTIVE_CHECKED_CONTROL_V1'
  write(logunit,'(A)') 'READY_FOR_MAPPING'
  flush(logunit)
  call wait_mapping()
  call record_state('PROGRAM_START')
  if(trim(argument)=='--fault') then
    fault=policy;trapped=.true.
    write(logunit,'(A,1X,A)') 'MODE fault',trim(fault)
  else
    if(trim(policy)/='masked'.and.trim(policy)/='trapped') stop 90
    trapped=trim(policy)=='trapped'
    write(logunit,'(A,1X,A)') 'MODE physical',trim(policy)
  endif
  do i=1,5
    if(.not.ieee_support_flag(flags(i),0.0_8)) stop 91
    if(.not.ieee_support_halting(flags(i))) stop 92
    ! Underflow and inexact remain nonhalting in every arm. Do not clear flags.
    call ieee_set_halting_mode(flags(i),trapped.and.i<=3)
  enddo
  call record_state('AFTER_POLICY')
  if(trim(argument)=='--fault') then
    write(logunit,'(A,1X,A)') 'FAULT_ENTER',trim(fault)
    flush(logunit)
    select case(trim(fault))
    case('invalid')
      x=-1.0_8;z=sqrt(x)
    case('divide_by_zero')
      x=1.0_8;y=0.0_8;z=x/y
    case('overflow')
      x=huge(1.0_8);y=2.0_8;z=x*y
    case default
      stop 93
    end select
    write(logunit,'(A,1X,A)') 'FAULT_SURVIVED',trim(fault)
    flush(logunit)
    stop 94
  endif
  call record_state('SETPARA_BEFORE')
  call setpara('MadLoop5_resources/param_card.dat')
  call record_state('SETPARA_AFTER')
  call ml5_0_get_answer_dimension(ndim)
  call ml5_0_get_nsqso_loop(nso)
  if(ndim/=1.or.nso/=1) stop 81
  open(73,file=trim(argument),status='old',action='read')
  read(73,*) npoints
  if(npoints/=2) stop 82
  do k=1,2
    do i=1,5
      read(73,*) points(:,i,k)
    enddo
  enddo
  close(73)
  call ml5_0_force_stability_check(.false.)
  ! Identical historical bootstrap and alternating warmup order. T=1/2 marks
  ! initialization finished; it never means physics acceptance (H may be4).
  call evaluate(1,'BOOTSTRAP',0,.false.)
  initialized=0
  do k=1,1002
    call evaluate(1+mod(k-1,2),'WARMUP',k,.false.)
    tens=mod(code,100)/10
    if(tens==1.or.tens==2) then
      initialized=initialized+1
    else
      initialized=0
    endif
    if(initialized>=2) exit
  enddo
  if(initialized<2) stop 83
  write(logunit,'(A)') 'READY_INITIALIZATION_FINISHED_NOT_ACCEPTANCE'
  flush(logunit)
  do k=1,2
    call evaluate(k,'OBSERVED',k,.true.)
  enddo
  call obs_close()
  call record_state('EXIT_BEFORE')
  call ml5_0_exit_madloop()
  call record_state('EXIT_AFTER')
  write(logunit,'(A,1X,I0)') 'COMPLETE_CHECKED_DIAGNOSTIC',observation_changes
  close(logunit)
contains
  subroutine wait_mapping()
    integer(int64)::started,current,rate
    logical::ready
    call system_clock(started,rate)
    if(rate<=0) stop 95
    do
      inquire(file='mapping-ready',exist=ready)
      if(ready) exit
      call system_clock(current)
      if(current-started>20_int64*rate) stop 96
      call sleep(1)
    enddo
  end subroutine

  subroutine evaluate(number,stage,index,observed)
    integer,intent(in)::number,index
    character(*),intent(in)::stage
    logical,intent(in)::observed
    write(logunit,'(A,1X,A,2(1X,I0))') 'CALL_ENTER',trim(stage),index,number
    flush(logunit)
    call record_state(trim(stage)//'_PREPROCESS_BEFORE')
    p=points(:,:,number)
    total=p(:,1)+p(:,2)
    mu_r=sqrt(abs(total(0)**2-sum(total(1:3)**2)))
    call update_as_param()
    ans=0.0_8
    call record_state(trim(stage)//'_PREPROCESS_AFTER')
    call record_state(trim(stage)//'_BEFORE')
    if(observed) call ew_mass_point_begin(number,p)
    call ml5_0_sloopmatrix_thres(p,ans,-1.0_8,acc,code)
    if(observed) call ew_mass_point_end(p,ans,acc,code)
    call record_state(trim(stage)//'_AFTER')
    accuracy_words=transfer(acc,accuracy_words)
    write(logunit,'(A,1X,A,3(1X,I0),2(1X,Z16.16))') &
         'CALL_RETURN',trim(stage),index,number,code,accuracy_words
    flush(logunit)
  end subroutine

  subroutine record_state(phase)
    character(*),intent(in)::phase
    type(ieee_status_type)::saved
    type(ieee_round_type)::rounding
    logical::values(5),after(5),halting(5),supported(5)
    integer::j
    character(16)::round_name
    call ieee_get_status(saved)
    do j=1,5
      call ieee_get_flag(flags(j),values(j))
      call ieee_get_halting_mode(flags(j),halting(j))
      supported(j)=ieee_support_flag(flags(j),0.0_8).and.ieee_support_halting(flags(j))
    enddo
    call ieee_get_rounding_mode(rounding)
    round_name='other'
    if(rounding==ieee_nearest) round_name='nearest'
    if(rounding==ieee_to_zero) round_name='to_zero'
    if(rounding==ieee_up) round_name='up'
    if(rounding==ieee_down) round_name='down'
    write(logunit,'(A,1X,A,1X,A)') 'ROUND',trim(phase),trim(round_name)
    do j=1,5
      write(logunit,'(A,1X,A,1X,A,3(1X,L1))') &
           'STATE',trim(phase),trim(names(j)),values(j),halting(j),supported(j)
    enddo
    do j=1,5
      call ieee_get_flag(flags(j),after(j))
    enddo
    if(any(values.neqv.after)) then
      observation_changes=observation_changes+1
      write(logunit,'(A,1X,A)') 'OBSERVATION_FLAG_CHANGE',trim(phase)
    endif
    ! Restore only observation-local state; never span model/reducer arithmetic.
    call ieee_set_status(saved)
    flush(logunit)
  end subroutine
end program
