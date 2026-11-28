! Validation of stochastic decisions made by the native MC driver. In a
! normal run ntuple supplies every variate directly. During validation the
! draw, fold and stage own their decisions even if two draws interleave.
module mc_decision_ledger
  use iso_fortran_env, only: int64, iostat_end
  implicit none
  private
  public :: decision_open, decision_close, decision_uniform, decision_mode
  public :: decision_begin_draw, decision_context, current_draw, current_fold, current_stage
  public :: stage_setup, stage_coordinates, stage_physics, stage_closing, stage_hit, stage_output

  integer, parameter :: stage_setup=0, stage_coordinates=1, stage_physics=2
  integer, parameter :: stage_closing=3, stage_hit=4, stage_output=5
  integer, protected :: decision_mode=0, current_fold=0, current_stage=stage_setup
  integer(int64), protected :: current_draw=0
  integer(int64) :: last_draw=0, num_used=0, expected_draws=0
  integer :: ledger_unit
  character(*), parameter :: version='MG7 MC decisions v1'

  type :: decision
    integer(int64) :: bits
    integer :: fold, stage, channel, following
  end type
  type(decision), allocatable :: entries(:)
  integer, allocatable :: head(:)

contains
  subroutine decision_context(draw,fold,stage)
    integer(int64), value :: draw
    integer, value :: fold,stage
    current_draw=draw
    current_fold=fold
    current_stage=stage
  end subroutine

  subroutine decision_begin_draw()
    last_draw=last_draw+1
    call decision_context(last_draw,0,stage_coordinates)
  end subroutine

  subroutine decision_open()
    character(16) :: operation
    character(:), allocatable :: path
    integer :: status,length
    last_draw=0
    num_used=0
    call decision_context(0_int64,0,stage_setup)
    call get_environment_variable('MG7_MC_DECISIONS',operation,status=status)
    if (status==1 .or. (status==0 .and. operation==' ')) return
    if (status/=0) error stop 'Invalid MG7_MC_DECISIONS'
    call get_environment_variable('MG7_MC_DECISION_FILE',length=length,status=status)
    if (status/=0 .or. length==0) error stop 'Set MG7_MC_DECISION_FILE separately for each native job'
    allocate(character(length)::path)
    call get_environment_variable('MG7_MC_DECISION_FILE',path)
    select case(operation)
    case('record')
      open(newunit=ledger_unit,file=path,status='new',action='write')
      write(ledger_unit,'(a)') version
      decision_mode=1
    case('replay')
      open(newunit=ledger_unit,file=path,status='old',action='read')
      call load_decisions()
      close(ledger_unit)
      decision_mode=2
    case default
      error stop 'MG7_MC_DECISIONS must be record or replay'
    end select
  end subroutine

  subroutine load_decisions()
    character(160) :: line
    character(8) :: tag
    integer(int64) :: owner,footer_count
    integer :: n,i,status
    integer, allocatable :: tail(:)
    read(ledger_unit,'(a)') line
    if (line/=version) error stop 'Incompatible native decision ledger'
    n=0
    do
      read(ledger_unit,'(a)',iostat=status) line
      if (status/=0) error stop 'Missing end of native decision ledger'
      if (index(line,'END ')==1) exit
      if (n==huge(n)) error stop 'Decision validation fixture too large'
      n=n+1
    enddo
    read(line,*,iostat=status) tag,footer_count,expected_draws
    if (status/=0) error stop 'Malformed end of native decision ledger'
    if (footer_count/=n .or. expected_draws<0 .or. expected_draws>huge(n)-1) &
        error stop 'Invalid native decision ledger counts'
    read(ledger_unit,'(a)',iostat=status) line
    if (status/=iostat_end) error stop 'Unexpected data after native decision ledger'
    rewind(ledger_unit)
    read(ledger_unit,'(a)') line
    allocate(entries(n),head(0:expected_draws),tail(0:expected_draws))
    head=0
    tail=0
    do i=1,n
      read(ledger_unit,*,iostat=status) owner,entries(i)%fold,entries(i)%stage,entries(i)%channel,entries(i)%bits
      if (status/=0) error stop 'Malformed native stochastic decision'
      if (owner<0 .or. owner>expected_draws) error stop 'Invalid native decision owner'
      entries(i)%following=0
      if (tail(owner)==0) then
        head(owner)=i
      else
        entries(tail(owner))%following=i
      endif
      tail(owner)=i
    enddo
  end subroutine

  subroutine decision_uniform(channel,value)
    integer, intent(in) :: channel
    double precision, intent(inout) :: value
    integer :: i
    select case(decision_mode)
    case(1)
      write(ledger_unit,'(4(i0,1x),i0)') current_draw,current_fold,current_stage,channel,transfer(value,0_int64)
    case(2)
      if (current_draw<0 .or. current_draw>ubound(head,1)) error stop 'Unexpected replay draw'
      i=head(current_draw)
      if (i==0) error stop 'Extra stochastic decision on replay'
      if (entries(i)%fold/=current_fold .or. entries(i)%stage/=current_stage .or. entries(i)%channel/=channel) &
          error stop 'Native stochastic decision moved to another fold, stage or channel'
      value=transfer(entries(i)%bits,value)
      head(current_draw)=entries(i)%following
    case default
      return
    end select
    num_used=num_used+1
  end subroutine

  subroutine decision_close()
    select case(decision_mode)
    case(1)
      write(ledger_unit,'(a,2(1x,i0))') 'END',num_used,last_draw
      close(ledger_unit)
    case(2)
      if (num_used/=size(entries,kind=int64) .or. any(head/=0) .or. last_draw/=expected_draws) &
          error stop 'Unconsumed native draw or stochastic decision'
      deallocate(head,entries)
    end select
    decision_mode=0
  end subroutine
end module
