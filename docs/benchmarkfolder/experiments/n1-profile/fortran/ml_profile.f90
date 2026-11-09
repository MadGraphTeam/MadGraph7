! Diagnostic-only attribution support. Not installed or used for acceptance.
module ml_profile_state
  use, intrinsic :: iso_c_binding
  implicit none
  logical :: active = .false.
  integer(c_int64_t) :: ticks(0:7,0:9,0:6,0:6,2) = 0
  integer(c_int64_t) :: calls(0:7,0:9,0:6,0:6,2) = 0
  integer(c_int64_t) :: rate, batch_start, calibration_ticks, calibration_wall
  real(c_double) :: cpu_start, calibration_cpu
  integer :: fortran_level = 0, bridge_level = 0
  integer, parameter :: calibration_iterations = 10000
  interface
    function bridge_control(level) bind(C,name='ml_tirust_profile_control') result(status)
      import
      integer(c_int), value :: level
      integer(c_int) :: status
    end function
    function bridge_json(output,capacity) bind(C,name='ml_tirust_profile_json') result(required)
      import
      type(c_ptr), value :: output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
    end function
  end interface
contains
  subroutine ml_profile_start(start)
    integer(c_int64_t), intent(out) :: start
    start = 0
    if (active) call system_clock(start)
  end subroutine

  subroutine ml_profile_end(stage,ctmode,n,rank,precision,start)
    integer, intent(in) :: stage,ctmode,n,rank,precision
    integer(c_int64_t), intent(in) :: start
    integer(c_int64_t) :: finish
    if (.not.active) return
    call system_clock(finish)
    if (stage<0.or.stage>7.or.ctmode<0.or.ctmode>9.or.n<0.or.n>6.or.rank<0.or.rank>6) stop 91
    ticks(stage,ctmode,n,rank,precision) = ticks(stage,ctmode,n,rank,precision)+finish-start
    calls(stage,ctmode,n,rank,precision) = calls(stage,ctmode,n,rank,precision)+1
  end subroutine

  subroutine ml_profile_reset()
    character(32) :: setting
    integer :: ios,i,status
    integer(c_int64_t) :: start,finish,outer_start
    real(c_double) :: before,after
    setting='0'
    call get_environment_variable('ML_TIRUST_FORTRAN_PROFILE',setting,status=ios)
    if (ios/=0) setting='0'
    read(setting,*,iostat=ios) fortran_level
    if (ios/=0.or.fortran_level<0.or.fortran_level>1) stop 92
    setting='0'
    call get_environment_variable('ML_TIRUST_BRIDGE_PROFILE',setting,status=ios)
    if (ios/=0) setting='0'
    read(setting,*,iostat=ios) bridge_level
    if (ios/=0.or.bridge_level<0.or.bridge_level>2) stop 93
    status=bridge_control(int(bridge_level,c_int))
    if (status/=0) stop 94
    call system_clock(count_rate=rate)
    if (rate<=0) stop 95
    active=.true.
    calls=0
    ticks=0
    call cpu_time(before)
    call system_clock(outer_start)
    do i=1,calibration_iterations
      call ml_profile_start(start)
      call ml_profile_end(0,0,0,0,1,start)
    enddo
    call system_clock(finish)
    call cpu_time(after)
    calibration_ticks=ticks(0,0,0,0,1)
    calibration_wall=finish-outer_start
    calibration_cpu=after-before
    calls=0
    ticks=0
    active=fortran_level==1
    call cpu_time(cpu_start)
    call system_clock(batch_start)
  end subroutine

  subroutine ml_profile_write()
    integer :: a,b,c,d,e
    integer(c_int64_t) :: finish
    real(c_double) :: cpu_finish
    integer(c_size_t) :: required,written
    character(kind=c_char), allocatable, target :: buffer(:)
    call system_clock(finish)
    call cpu_time(cpu_finish)
    active=.false.
    open(81,file='fortran-clock.txt',status='replace')
    write(81,*) 'levels',fortran_level,bridge_level
    write(81,*) 'clock_rate',rate
    write(81,*) 'batch_wall_seconds',real(finish-batch_start,c_double)/real(rate,c_double)
    write(81,*) 'batch_cpu_seconds',cpu_finish-cpu_start
    write(81,*) 'calibration_iterations',calibration_iterations
    write(81,*) 'calibration_internal_seconds',real(calibration_ticks,c_double)/real(rate,c_double)
    write(81,*) 'calibration_wall_seconds',real(calibration_wall,c_double)/real(rate,c_double)
    write(81,*) 'calibration_cpu_seconds',calibration_cpu
    close(81)
    open(82,file='fortran-profile.tsv',status='replace')
    write(82,'(A)') '# stage ctmode propagators rank precision calls seconds'
    do e=1,2
      do d=0,6
        do c=0,6
          do b=0,9
            do a=1,7
              if (calls(a,b,c,d,e)>0) write(82,'(5(I5,1X),I20,1X,ES26.17E3)') &
                a,b,c,d,e,calls(a,b,c,d,e),real(ticks(a,b,c,d,e),c_double)/real(rate,c_double)
            enddo
          enddo
        enddo
      enddo
    enddo
    close(82)
    required=bridge_json(c_null_ptr,0_c_size_t)
    if (required<1.or.required>33554432) stop 96
    allocate(buffer(required))
    written=bridge_json(c_loc(buffer),required)
    if (written/=required.or.buffer(required)/=c_null_char) stop 97
    open(83,file='bridge-profile.json',status='replace',access='stream',form='unformatted')
    write(83) buffer(:required-1)
    close(83)
  end subroutine
end module
