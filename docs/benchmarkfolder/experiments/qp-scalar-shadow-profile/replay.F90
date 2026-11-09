! Private replay only. Compile/run requires a separate reviewed launch.
program qp_scalar_shadow_replay
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  use mtirust
  implicit none
  integer, parameter :: q=KI_QTIR
  type saved_input
    integer :: point,group,n,rank,historical_status
    real(q), allocatable :: p(:,:),s(:,:)
    complex(q), allocatable :: m(:)
    real(q) :: scale
  end type
  type(saved_input) :: inputs(24)
  integer :: i,j,k,c,setup,status,nc,previous_point,index,control,mode,cache_limit
  logical :: partial,check,cache,flags_before(5),flags_after(5)
  type(ieee_flag_type), parameter :: flags(5)=[ieee_invalid,ieee_divide_by_zero,ieee_overflow,ieee_underflow,ieee_inexact]
  real(q) :: re,im
  complex(q), allocatable :: raw(:,:)
  integer(c_int8_t), allocatable :: raw_bytes(:)
  integer(c_int64_t) :: counts(6)
  character(len=1024) :: filename,argument
  character(len=64) :: magic
  character(len=2048) :: error
#ifdef SHADOW
  integer(c_size_t) :: required,written
  integer(c_int) :: mpfr_before,mpfr_after
  character(c_char), allocatable :: json_bytes(:)
  interface
    integer(c_int) function shadow_control(enabled) bind(C,name='ml_tirust_shadow_control')
      import c_int
      integer(c_int), value :: enabled
    end function
    subroutine shadow_label(point,group,synthetic) bind(C,name='ml_tirust_shadow_label')
      import c_int
      integer(c_int), value :: point,group,synthetic
    end subroutine
    integer(c_int) function shadow_flags() bind(C,name='ml_tirust_shadow_mpfr_flags')
      import c_int
    end function
    integer(c_size_t) function shadow_json(out,capacity) bind(C,name='ml_tirust_shadow_json')
      import c_ptr,c_size_t
      type(c_ptr), value :: out
      integer(c_size_t), value :: capacity
    end function
  end interface
#endif
  if(radix(1.0_q)/=2.or.digits(1.0_q)/=113) stop 1
  call get_command_argument(1,filename)
  call get_command_argument(2,argument)
  read(argument,*) mode
  if(mode<0.or.mode>1) stop 2
  open(73,file=trim(filename),status='old',action='read')
  read(73,'(A)') magic
  if(trim(magic)/='QP_SCALAR_SHADOW_V1') stop 3
  read(73,*) c
  if(c/=24) stop 4
  do c=1,24
    read(73,*) inputs(c)%point,inputs(c)%group,inputs(c)%n,inputs(c)%rank,inputs(c)%historical_status
    associate(v=>inputs(c))
      allocate(v%p(0:3,v%n),v%s(v%n,v%n),v%m(v%n))
      read(73,*) v%scale
      do i=1,v%n
        read(73,*) v%p(:,i)
      enddo
      do i=1,v%n
        read(73,*) re,im
        v%m(i)=cmplx(re,im,kind=q)
      enddo
      do i=1,v%n
        read(73,*) v%s(i,:)
      enddo
    end associate
  enddo
  close(73)
#ifdef SHADOW
  setup=shadow_control(int(mode,c_int))
  if(setup/=0) stop 5
  open(77,file='mpfr-flags.dat',status='new',action='write')
#endif
  call tirust_set_cache_budget(268435456_c_int64_t,setup)
  if(setup/=TIRUST_SUCCESS) stop 6
  previous_point=0
  do c=1,36
    partial=.false.;check=.false.;cache=.true.;cache_limit=-1;control=max(0,c-24)
    if(c<=24) then
      index=c
    else
      index=1
      if(control>=11) index=17
      if(control==3.or.control==4.or.control==8) check=.true.
      if(control==5) cache_limit=0
      if(control==12) cache_limit=3
      if(control>=6.and.control<=8) partial=.true.
      if(control==1) then
        call tirust_clear_cache(setup)
        if(setup/=0) stop 7
      endif
      if(control==9.or.control==11) then
        if(control==9) then
          call tirust_set_cache_budget(0_c_int64_t,setup)
        else
          call tirust_set_cache_budget(268435456_c_int64_t,setup)
        endif
        if(setup/=0) stop 8
      endif
    endif
    associate(v=>inputs(index))
      cache=cache_limit==-1.or.v%n<=cache_limit
      if(c<=24.and.v%point/=previous_point) then
        call tirust_clear_cache(setup)
        if(setup/=0) stop 9
        previous_point=v%point
      endif
      call tirust_register_plan(v%n,v%rank,setup)
      if(setup/=0) stop 10
      nc=tirust_coefficient_count(v%rank)
      allocate(raw(nc,4),raw_bytes(nc*4*32))
      ! Capture all reconstructed input bits, including signed zero, without
      ! a floating-point reformat. This is compared across all three runs.
      write(*,'(A,9I6,3L2)') 'BEGIN ',c,control,v%point,v%group,v%n,v%rank,v%historical_status,nc,cache_limit,partial,check,cache
      write(*,'(A,*(Z2.2))') 'INPUT_P ',transfer(v%p,[0_c_int8_t],size(v%p)*16)
      write(*,'(A,*(Z2.2))') 'INPUT_S ',transfer(v%s,[0_c_int8_t],size(v%s)*16)
      write(*,'(A,*(Z2.2))') 'INPUT_M ',transfer(v%m,[0_c_int8_t],size(v%m)*32)
      write(*,'(A,*(Z2.2))') 'INPUT_MU ',transfer(v%scale,[0_c_int8_t],16)
#ifdef SHADOW
      call shadow_label(int(v%point,c_int),int(v%group,c_int),merge(1_c_int,0_c_int,control>0))
      mpfr_before=shadow_flags()
#endif
      call ieee_set_flag(flags,.false.)
      call ieee_get_flag(flags,flags_before)
      call tirust_tensor_evaluate(v%n,v%rank,v%p,v%s,v%m,v%scale,raw,status, &
        pole_mode=0,compute_finite=.not.partial,check_pass=check,cache_enabled=cache)
      call ieee_get_flag(flags,flags_after)
#ifdef SHADOW
      mpfr_after=shadow_flags()
      write(77,'(3I14)') c,mpfr_before,mpfr_after
#endif
      write(*,'(A,I6)') 'STATUS ',status
      write(*,'(A,10L2)') 'HOST_FLAGS ',flags_before,flags_after
      ! mtirust zeros failures; compare those bytes too without interpreting
      ! them as successful tensors. Partial finite slots remain unevaluated.
      raw_bytes=transfer(raw,raw_bytes)
      write(*,'(A,*(Z2.2))') 'OUTPUT ',raw_bytes
      if(status/=TIRUST_SUCCESS) then
        call tirust_error_message(error)
        write(*,'(A,A)') 'ERROR ',trim(error)
      endif
      call tirust_qp_cache_statistics(counts,setup)
      if(setup/=0) stop 11
      write(*,'(A,6I22)') 'CACHE ',counts
      write(*,'(A,I6)') 'END ',c
      flush(6)
      deallocate(raw,raw_bytes)
    end associate
  enddo
  call tirust_finalize(setup)
  if(setup/=0) stop 12
#ifdef SHADOW
  close(77)
  required=shadow_json(c_null_ptr,0_c_size_t)
  if(required>9000000_c_size_t) stop 13
  block
    character(c_char), allocatable, target :: buffer(:)
    allocate(buffer(required))
    written=shadow_json(c_loc(buffer),required)
    if(written/=required) stop 14
    open(78,file='shadow.json',status='new',access='stream',form='unformatted',action='write')
    write(78) buffer(1:required-1)
    close(78)
  end block
#endif
end program
