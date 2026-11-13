! Fortran interface to the TIRust bridge. Tensor moments use IREGI ordering,
! with slots [UV1, IR1, IR2, finite]. Convention::MadLoop is applied in Rust.
! The module retains one context and is intended for serial MadLoop calls.
module mtirust
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  integer, parameter, public :: KI_TIR = c_double
  integer, parameter, public :: KI_QTIR = selected_real_kind(33,4931)
  integer, parameter, public :: TIRUST_SUCCESS = 0, TIRUST_UNSTABLE = 1
  integer, parameter, public :: TIRUST_ERROR = 2, TIRUST_UNSUPPORTED = 3
  integer, parameter, public :: TIRUST_UNSTABLE_KINEMATICS = TIRUST_UNSTABLE
  integer, parameter, public :: TIRUST_POLES_BOTH = 0, TIRUST_POLES_UV = 1
  integer, parameter, public :: TIRUST_POLES_IR = 2, TIRUST_POLES_NONE = 3
  integer, parameter :: record_size = 64
  type(c_ptr), save :: context = c_null_ptr
  integer, save :: context_epoch = 0
  character(len=512), save :: local_error = ''
  public :: tirust_tensor_evaluate, tirust_coefficient_count
  public :: tirust_clear_cache, tirust_finalize, tirust_error_message
  public :: tirust_register_plan, tirust_cache_statistics, tirust_set_cache_budget, tirust_supports
  public :: tirust_context_generation, tirust_qp_cache_statistics
  public :: tirust_dp_route_statistics

  interface tirust_tensor_evaluate
    module procedure tensor_dp, tensor_mp
  end interface

  interface
    function create_context(version,handle) bind(C,name="ml_tirust_context_create") result(status)
      import
      integer(c_int), value :: version
      type(c_ptr), intent(out) :: handle
      integer(c_int) :: status
    end function
    function destroy_context(handle) bind(C,name="ml_tirust_context_destroy") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int) :: status
    end function
    function clear_cache(handle) bind(C,name="ml_tirust_clear_cache") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int) :: status
    end function
    function register_plan(handle,n,rank) bind(C,name="ml_tirust_register_plan") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int), value :: n,rank
      integer(c_int) :: status
    end function
    function cache_statistics(handle,counts,count) bind(C,name="ml_tirust_cache_statistics") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int64_t), intent(out) :: counts(*)
      integer(c_size_t), value :: count
      integer(c_int) :: status
    end function
    function qp_cache_statistics(handle,counts,count) bind(C,name="ml_tirust_qp_cache_statistics") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int64_t), intent(out) :: counts(*)
      integer(c_size_t), value :: count
      integer(c_int) :: status
    end function
    function dp_route_statistics(handle,counts,count) bind(C,name="ml_tirust_dp_route_statistics") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int64_t), intent(out) :: counts(*)
      integer(c_size_t), value :: count
      integer(c_int) :: status
    end function
    function set_cache_budget(handle,bytes) bind(C,name="ml_tirust_set_cache_budget") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int64_t), value :: bytes
      integer(c_int) :: status
    end function
    function supports(n,rank,compute_finite) bind(C,name="ml_tirust_supports") result(status)
      import
      integer(c_int), value :: n,rank,compute_finite
      integer(c_int) :: status
    end function
    function evaluate_dp(handle,n,rank,p,plen,s,slen,m,mlen,mu2,output,olen, &
                         pole_mode,compute_finite,check_pass,cache_enabled) &
        bind(C,name="ml_tirust_evaluate_dp") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int), value :: n,rank,pole_mode,compute_finite,check_pass,cache_enabled
      integer(c_size_t), value :: plen,slen,mlen,olen
      real(c_double), intent(in) :: p(*),s(*)
      complex(c_double_complex), intent(in) :: m(*)
      real(c_double), value :: mu2
      complex(c_double_complex), intent(out) :: output(*)
      integer(c_int) :: status
    end function
    function evaluate_mp(handle,n,rank,p,plen,s,slen,m,mlen,mu2,scale_len,output,olen, &
                         pole_mode,compute_finite,check_pass,cache_enabled) &
        bind(C,name="ml_tirust_evaluate_mp") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int), value :: n,rank,pole_mode,compute_finite,check_pass,cache_enabled
      integer(c_size_t), value :: plen,slen,mlen,scale_len,olen
      character(kind=c_char), intent(in) :: p(*),s(*),m(*),mu2(*)
      character(kind=c_char), intent(out) :: output(*)
      integer(c_int) :: status
    end function
    function last_error(message,capacity) bind(C,name="ml_tirust_last_error") result(required)
      import
      character(kind=c_char), intent(out) :: message(*)
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
    end function
  end interface
contains
  subroutine ensure_context(status)
    integer, intent(out) :: status
    local_error = ''
    status = TIRUST_SUCCESS
    if (.not.c_associated(context)) then
      status = create_context(1_c_int,context)
      if (status == TIRUST_SUCCESS .and. c_associated(context)) context_epoch = context_epoch+1
    endif
    if (status == TIRUST_SUCCESS .and. .not.c_associated(context)) then
      status = TIRUST_ERROR
      local_error = 'TIRust returned a null context.'
    endif
  end subroutine

  integer function tirust_context_generation() result(generation)
    generation = -1
    if (c_associated(context)) generation = context_epoch
  end function

  subroutine tirust_finalize(status)
    integer, intent(out), optional :: status
    integer :: result
    local_error = ''
    result = destroy_context(context)
    if (result == TIRUST_SUCCESS) context = c_null_ptr
    if (present(status)) status = result
  end subroutine

  subroutine tirust_clear_cache(status)
    integer, intent(out), optional :: status
    integer :: result
    call ensure_context(result)
    if (result == TIRUST_SUCCESS) result = clear_cache(context)
    if (present(status)) status = result
  end subroutine

  subroutine tirust_register_plan(n,rank,status)
    integer, intent(in) :: n,rank
    integer, intent(out) :: status
    call ensure_context(status)
    if (status == TIRUST_SUCCESS) status = register_plan(context,int(n,c_int),int(rank,c_int))
  end subroutine

  subroutine tirust_cache_statistics(counts,status)
    ! Subsets reused/computed, masters reused/computed, moments reused/computed,
    ! entries, bytes, and insertions refused by the cache budget (in that order).
    ! The last five counters also include QP whole-tensor cache activity.
    integer(c_int64_t), intent(out) :: counts(9)
    integer, intent(out), optional :: status
    integer :: result
    counts = 0_c_int64_t
    call ensure_context(result)
    if (result == TIRUST_SUCCESS) result = cache_statistics(context,counts,9_c_size_t)
    if (result /= TIRUST_SUCCESS) counts = 0_c_int64_t
    if (present(status)) status = result
  end subroutine

  subroutine tirust_qp_cache_statistics(counts,status)
    ! QP whole-tensor hits, misses, entries, bytes, refused insertions, bypasses.
    integer(c_int64_t), intent(out) :: counts(6)
    integer, intent(out), optional :: status
    integer :: result
    counts = 0_c_int64_t
    call ensure_context(result)
    if (result == TIRUST_SUCCESS) result = qp_cache_statistics(context,counts,6_c_size_t)
    if (result /= TIRUST_SUCCESS) counts = 0_c_int64_t
    if (present(status)) status = result
  end subroutine

  subroutine tirust_dp_route_statistics(counts,status)
    ! Cumulative bridge executions, primary/check blocks of ten:
    ! Direct, MethodChoice, Tier106, Tier256, Tier512, Refused,
    ! unreported/reduction-error, attempted106, attempted256, attempted512.
    ! A nonzero unreported bucket makes tier-attempt counts lower bounds.
    ! Internal tiers use binary64 inputs and a DP numerator, not genuine QP.
    ! A query must not create a context or erase a local reduction diagnostic.
    integer(c_int64_t), intent(out) :: counts(20)
    integer, intent(out), optional :: status
    integer :: result
    counts = 0_c_int64_t
    result = TIRUST_SUCCESS
    if (c_associated(context)) result = dp_route_statistics(context,counts,20_c_size_t)
    if (result /= TIRUST_SUCCESS) counts = 0_c_int64_t
    if (present(status)) status = result
  end subroutine

  subroutine tirust_set_cache_budget(bytes,status)
    integer(c_int64_t), intent(in) :: bytes
    integer, intent(out), optional :: status
    integer :: result
    if (bytes < 0_c_int64_t) then
      local_error = 'TIRust cache budget must be nonnegative.'
      result = TIRUST_ERROR
    else
      call ensure_context(result)
      if (result == TIRUST_SUCCESS) result = set_cache_budget(context,bytes)
    endif
    if (present(status)) status = result
  end subroutine

  logical function tirust_supports(n,rank,compute_finite) result(supported)
    integer, intent(in) :: n,rank
    logical, intent(in) :: compute_finite
    supported = supports(int(n,c_int),int(rank,c_int), &
                         merge(1_c_int,0_c_int,compute_finite)) == TIRUST_SUCCESS
  end function

  pure integer function tirust_coefficient_count(rank) result(count)
    integer, intent(in) :: rank
    integer(c_int64_t) :: r,total
    count = -1
    ! TIRust's packed moment representation has a maximum rank of 564.
    if (rank < 0 .or. rank > 564) return
    r = int(rank,c_int64_t)
    total = (r+1)*(r+2)*(r+3)*(r+4)/24
    if (total > int(huge(count),c_int64_t)) return
    count = int(total)
  end function

  subroutine options(pole_mode,compute_finite,check_pass,cache_enabled,poles,finite,check,cache)
    integer, intent(in), optional :: pole_mode
    logical, intent(in), optional :: compute_finite,check_pass,cache_enabled
    integer(c_int), intent(out) :: poles,finite,check,cache
    poles = 0_c_int
    finite = 1_c_int
    check = 0_c_int
    cache = 1_c_int
    if (present(pole_mode)) poles = int(pole_mode,c_int)
    if (present(compute_finite)) finite = merge(1_c_int,0_c_int,compute_finite)
    if (present(check_pass)) check = merge(1_c_int,0_c_int,check_pass)
    if (present(cache_enabled)) cache = merge(1_c_int,0_c_int,cache_enabled)
  end subroutine

  logical function valid_shapes(n,rank,p_shape,s_shape,m_size,out_shape,status) result(valid)
    integer, intent(in) :: n,rank,p_shape(2),s_shape(2),m_size,out_shape(2)
    integer, intent(out) :: status
    integer :: count
    valid = .false.
    status = TIRUST_ERROR
    local_error = 'TIRust input or output dimensions do not match n and rank.'
    count = tirust_coefficient_count(rank)
    if (n < 1 .or. n > 6 .or. rank < 0 .or. rank > 6 .or. count < 1) then
      status = TIRUST_UNSUPPORTED
      local_error = 'TIRust bridge v1 qualifies n=1..6 and rank=0..6.'
      return
    endif
    if (any(p_shape /= [4,n]) .or. any(s_shape /= [n,n]) .or. m_size /= n) return
    if (any(out_shape /= [count,4])) return
    valid = .true.
    status = TIRUST_SUCCESS
    local_error = ''
  end function

  subroutine tensor_dp(n,rank,shifts,invariants,masses,mu2,coefs,status, &
                       pole_mode,compute_finite,check_pass,cache_enabled)
    integer, intent(in) :: n,rank
    real(KI_TIR), intent(in) :: shifts(0:,:),invariants(:,:),mu2
    complex(KI_TIR), intent(in) :: masses(:)
    complex(KI_TIR), intent(out) :: coefs(:,:)
    integer, intent(out) :: status
    integer, intent(in), optional :: pole_mode
    logical, intent(in), optional :: compute_finite,check_pass,cache_enabled
    integer(c_int) :: poles,finite,check,cache
    coefs = cmplx(0,0,kind=KI_TIR)
    if (.not.valid_shapes(n,rank,shape(shifts),shape(invariants),size(masses),shape(coefs),status)) return
    call options(pole_mode,compute_finite,check_pass,cache_enabled,poles,finite,check,cache)
    call ensure_context(status)
    if (status /= TIRUST_SUCCESS) return
    status = evaluate_dp(context,int(n,c_int),int(rank,c_int),shifts,size(shifts,kind=c_size_t), &
                         invariants,size(invariants,kind=c_size_t),masses,size(masses,kind=c_size_t), &
                         mu2,coefs,size(coefs,kind=c_size_t),poles,finite,check,cache)
    if (status /= TIRUST_SUCCESS) coefs = cmplx(0,0,kind=KI_TIR)
  end subroutine

  subroutine pack(value,record,io_status)
    real(KI_QTIR), intent(in) :: value
    character(kind=c_char), intent(out) :: record(record_size)
    integer, intent(out) :: io_status
    character(len=record_size) :: text
    integer :: k
    io_status = 1
    if (.not.ieee_is_finite(value)) return
    write(text,'(ES64.39E4)',iostat=io_status) value
    if (io_status /= 0) return
    do k=1,record_size
      record(k) = text(k:k)
    enddo
  end subroutine

  subroutine unpack(record,value,io_status)
    character(kind=c_char), intent(in) :: record(record_size)
    real(KI_QTIR), intent(out) :: value
    integer, intent(out) :: io_status
    character(len=record_size) :: text
    integer :: k
    do k=1,record_size
      text(k:k) = record(k)
    enddo
    read(text,*,iostat=io_status) value
    if (io_status == 0) then
      if (.not.ieee_is_finite(value)) io_status = 1
    endif
  end subroutine

  subroutine tensor_mp(n,rank,shifts,invariants,masses,mu2,coefs,status, &
                       pole_mode,compute_finite,check_pass,cache_enabled)
    integer, intent(in) :: n,rank
    real(KI_QTIR), intent(in) :: shifts(0:,:),invariants(:,:),mu2
    complex(KI_QTIR), intent(in) :: masses(:)
    complex(KI_QTIR), intent(out) :: coefs(:,:)
    integer, intent(out) :: status
    integer, intent(in), optional :: pole_mode
    logical, intent(in), optional :: compute_finite,check_pass,cache_enabled
    integer(c_int) :: poles,finite,check,cache
    character(kind=c_char), allocatable :: p_text(:,:),s_text(:,:),m_text(:,:),output(:,:)
    character(kind=c_char) :: scale_text(record_size)
    real(KI_QTIR), allocatable :: values(:)
    integer :: i,j,k,count,ios
    coefs = cmplx(0,0,kind=KI_QTIR)
    if (.not.valid_shapes(n,rank,shape(shifts),shape(invariants),size(masses),shape(coefs),status)) return
    status = TIRUST_UNSUPPORTED
    local_error = 'TIRust quadruple precision requires a binary128 kind with 113 significand bits.'
    if (radix(mu2) /= 2 .or. digits(mu2) /= 113) return
    count = tirust_coefficient_count(rank)
    status = TIRUST_ERROR
    local_error = 'TIRust quadruple-precision transport allocation failed.'
    if (int(count,c_int64_t)*8_c_int64_t > int(huge(count),c_int64_t)) return
    allocate(p_text(record_size,4*n),s_text(record_size,n*n),m_text(record_size,2*n), &
             output(record_size,8*count),values(8*count),stat=ios)
    if (ios /= 0) return
    local_error = 'TIRust could not encode a finite binary128 input.'
    do i=1,n
      call pack(real(masses(i),kind=KI_QTIR),m_text(:,2*i-1),ios)
      if (ios /= 0) return
      call pack(aimag(masses(i)),m_text(:,2*i),ios)
      if (ios /= 0) return
      do j=0,3
        call pack(shifts(j,i),p_text(:,4*(i-1)+j+1),ios)
        if (ios /= 0) return
      enddo
      do j=1,n
        call pack(invariants(j,i),s_text(:,n*(i-1)+j),ios)
        if (ios /= 0) return
      enddo
    enddo
    call pack(mu2,scale_text,ios)
    if (ios /= 0) return
    call options(pole_mode,compute_finite,check_pass,cache_enabled,poles,finite,check,cache)
    call ensure_context(status)
    if (status /= TIRUST_SUCCESS) return
    status = evaluate_mp(context,int(n,c_int),int(rank,c_int),p_text,size(p_text,kind=c_size_t), &
                         s_text,size(s_text,kind=c_size_t),m_text,size(m_text,kind=c_size_t), &
                         scale_text,size(scale_text,kind=c_size_t),output,size(output,kind=c_size_t), &
                         poles,finite,check,cache)
    if (status /= TIRUST_SUCCESS) return
    do k=1,8*count
      call unpack(output(:,k),values(k),ios)
      if (ios /= 0) then
        ! Native insertion precedes this public Fortran transport check.
        ! Reject all numerical cache entries when decoding the result fails.
        ios = clear_cache(context)
        status = TIRUST_ERROR
        local_error = 'TIRust returned an invalid or nonfinite binary128 output record.'
        return
      endif
    enddo
    do j=1,4
      do i=1,count
        k = 2*((j-1)*count+i)
        coefs(i,j) = cmplx(values(k-1),values(k),kind=KI_QTIR)
      enddo
    enddo
  end subroutine

  subroutine tirust_error_message(message)
    character(len=*), intent(out) :: message
    character(kind=c_char) :: buffer(len(message)+1)
    integer(c_size_t) :: required
    integer :: k
    message = local_error
    if (len_trim(local_error) /= 0) return
    buffer = c_null_char
    required = last_error(buffer,int(size(buffer),c_size_t))
    message = ''
    do k=1,len(message)
      if (buffer(k) == c_null_char) exit
      message(k:k) = buffer(k)
    enddo
  end subroutine
end module mtirust
