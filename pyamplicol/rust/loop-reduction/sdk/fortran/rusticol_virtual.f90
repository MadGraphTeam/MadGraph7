! SPDX-License-Identifier: 0BSD
! Linux glibc floor (SDK README, "glibc floor (Linux)"): a client linked
! against the shared librusticol_loop_sdk.so of a release native build needs
! glibc 2.38 or newer (GLIBC_2.38). With gfortran on an older glibc the link
! fails ("librusticol_loop_sdk.so: undefined reference to
! `__isoc23_strtol@GLIBC_2.38'"); a client built on glibc 2.38 or newer and run
! on an older one is stopped by the dynamic loader before the program starts
! ("libc.so.6: version `GLIBC_2.38' not found (required by
! .../librusticol_loop_sdk.so)"). Build and run it with gfortran on glibc 2.38
! or newer, or build the SDK from source on that host.
module rusticol_virtual
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: error_unit
  implicit none
  private
  integer(c_int), parameter, public :: VIRTUAL_OK=0, VIRTUAL_INVALID_ARGUMENT=1
  integer(c_int), parameter, public :: VIRTUAL_BUFFER_TOO_SMALL=2, VIRTUAL_RUNTIME_ERROR=3, VIRTUAL_PANIC=4
  integer(c_int32_t), parameter, public :: VIRTUAL_SUCCESS=0, VIRTUAL_VERIFIED=1, VIRTUAL_RESCUED=2
  integer(c_int32_t), parameter, public :: VIRTUAL_FAILED=3, VIRTUAL_ACCURACY_NOT_ESTABLISHED=4

  type, bind(C), public :: virtual_shape
    integer(c_size_t) :: external_count=0, born_orders=0, virtual_orders=0, helicities=0
    integer(c_size_t) :: born_color_components=0, virtual_color_components=0, epsilon_count=0
    integer(c_int32_t) :: resolved_available=0
  end type
  type, bind(C), public :: virtual_result_shape
    integer(c_size_t) :: requested_points=0, successful_points=0, born_values=0, virtual_values=0
    integer(c_int32_t) :: resolved=0
    type(virtual_shape) :: axes
  end type
  type, bind(C), public :: virtual_correlated_shape
    integer(c_size_t) :: histories=0, born_orders=0, virtual_orders=0, epsilon_count=0
  end type
  type, bind(C), public :: virtual_correlated_result_shape
    integer(c_size_t) :: requested_points=0, successful_points=0, born_values=0, virtual_values=0
    type(virtual_correlated_shape) :: axes
  end type
  type, bind(C) :: complex_pair
    real(c_double) :: real, imag
  end type
  type, public :: virtual_correlated_evaluation
    type(virtual_correlated_result_shape) :: shape
    complex(c_double), allocatable :: born(:,:,:) ! order, history, point
    complex(c_double), allocatable :: virtual_values(:,:,:,:) ! epsilon, order, history, point
    integer(c_int64_t), allocatable :: requested_ids(:), successful_ids(:)
    integer(c_int32_t), allocatable :: statuses(:)
    character(:), allocatable :: diagnostics_json, histories_json, spin_legs_json
  end type
  type, bind(C) :: c_batch_request
    type(c_ptr) :: momenta
    integer(c_size_t) :: momentum_count, point_count
    type(c_ptr) :: point_ids, mu_r2
    integer(c_size_t) :: scale_count
    integer(c_int32_t) :: resolved
    type(c_ptr) :: r2_strategy, verification_json
  end type
  type, bind(C) :: c_spin_sources
    type(c_ptr) :: legs
    integer(c_size_t) :: leg_count
    type(c_ptr) :: values
    integer(c_size_t) :: value_count
  end type
  ! Fortran arrays reverse the C axes without reordering numerical values.
  ! Inclusive results retain singleton colour/helicity dimensions.
  type, public :: virtual_evaluation
    type(virtual_result_shape) :: shape
    real(c_double), allocatable :: born(:,:,:,:) ! colour, helicity, order, successful point
    real(c_double), allocatable :: virtual_values(:,:,:,:,:) ! epsilon, colour, helicity, order, point
    integer(c_int64_t), allocatable :: requested_ids(:), successful_ids(:)
    integer(c_int32_t), allocatable :: statuses(:) ! every requested point
    character(:), allocatable :: diagnostics_json
  end type
  ! Assignment shares a reference-counted runtime. Calls require exclusive access.
  ! Evaluations own copied arrays and never depend on this runtime's lifetime.
  type :: runtime_owner
    type(c_ptr) :: handle=c_null_ptr
    integer(c_size_t) :: references=1
  end type
  type, public :: virtual_runtime
    private
    type(runtime_owner), pointer :: owner=>null()
  contains
    final :: finalize_runtime
    procedure, private :: assign_runtime
    generic, public :: assignment(=) => assign_runtime
    procedure :: load
    procedure :: close
    procedure :: is_loaded
    procedure :: clear
    procedure :: shape => runtime_shape
    procedure :: metadata_json
    procedure :: capabilities_json
    procedure :: update_parameters
    procedure :: set_poles
    procedure :: evaluate
    procedure :: evaluate_correlated
    procedure :: correlation_histories_json
    procedure :: spin_correlation_classes_json
  end type
  public :: virtual_last_error
  interface
    function c_open(path, process, parameters, strategy, output) bind(C, name="rusticol_virtual_open") result(value)
      import
      type(c_ptr), value :: path, process, parameters, strategy
      type(c_ptr) :: output
      integer(c_int) :: value
    end function
    function c_open_policy(path, process, parameters, strategy, policy, output) &
        bind(C, name="rusticol_virtual_open_with_policy") result(value)
      import
      type(c_ptr), value :: path, process, parameters, strategy, policy
      type(c_ptr) :: output
      integer(c_int) :: value
    end function
    function c_free(handle) bind(C, name="rusticol_virtual_free") result(value)
      import
      type(c_ptr), value :: handle
      integer(c_int) :: value
    end function
    function c_clear(handle) bind(C, name="rusticol_virtual_clear") result(value)
      import
      type(c_ptr), value :: handle
      integer(c_int) :: value
    end function
    function c_result_free(handle) bind(C, name="rusticol_virtual_result_free") result(value)
      import
      type(c_ptr), value :: handle
      integer(c_int) :: value
    end function
    function c_shape(handle, output) bind(C, name="rusticol_virtual_shape") result(value)
      import
      type(c_ptr), value :: handle
      type(virtual_shape) :: output
      integer(c_int) :: value
    end function
    function c_result_shape(handle, output) bind(C, name="rusticol_virtual_result_shape") result(value)
      import
      type(c_ptr), value :: handle
      type(virtual_result_shape) :: output
      integer(c_int) :: value
    end function
    function c_metadata_json(handle, output, capacity, needed) bind(C, name="rusticol_virtual_metadata_json") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: needed
      integer(c_int) :: value
    end function
    function c_capabilities_json(handle, output, capacity, needed) bind(C, name="rusticol_virtual_capabilities_json") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: needed
      integer(c_int) :: value
    end function
    function c_result_diagnostics_json(handle, output, capacity, needed) &
        bind(C, name="rusticol_virtual_result_diagnostics_json") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: needed
      integer(c_int) :: value
    end function
    function c_update_parameters(handle, changes, revision) bind(C, name="rusticol_virtual_update_parameters") result(value)
      import
      type(c_ptr), value :: handle, changes
      integer(c_int64_t) :: revision
      integer(c_int) :: value
    end function
    function c_set_poles(handle, poles) bind(C, name="rusticol_virtual_set_poles") result(value)
      import
      type(c_ptr), value :: handle, poles
      integer(c_int) :: value
    end function
    function c_evaluate(handle, request, output) bind(C, name="rusticol_virtual_evaluate") result(value)
      import
      type(c_ptr), value :: handle
      type(c_batch_request), intent(in) :: request
      type(c_ptr) :: output
      integer(c_int) :: value
    end function
    function c_result_copy_born(handle, output, capacity) bind(C, name="rusticol_virtual_result_copy_born") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_result_copy_virtual(handle, output, capacity) bind(C, name="rusticol_virtual_result_copy_virtual") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_result_copy_requested_ids(handle, output, capacity) &
        bind(C, name="rusticol_virtual_result_copy_requested_ids") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_result_copy_successful_ids(handle, output, capacity) &
        bind(C, name="rusticol_virtual_result_copy_successful_ids") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_result_copy_statuses(handle, output, capacity) bind(C, name="rusticol_virtual_result_copy_statuses") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_correlated_evaluate(handle, request, output) &
        bind(C, name="rusticol_virtual_evaluate_correlated") result(value)
      import
      type(c_ptr), value :: handle
      type(c_batch_request), intent(in) :: request
      type(c_ptr) :: output
      integer(c_int) :: value
    end function
    function c_correlated_sources(handle, request, sources, output) &
        bind(C, name="rusticol_virtual_evaluate_correlated_with_sources") result(value)
      import
      type(c_ptr), value :: handle
      type(c_batch_request), intent(in) :: request
      type(c_spin_sources), intent(in) :: sources
      type(c_ptr) :: output
      integer(c_int) :: value
    end function
    function c_spin_legs_json(handle, output, capacity, needed) &
        bind(C, name="rusticol_virtual_correlated_result_spin_legs_json") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: needed
      integer(c_int) :: value
    end function
    function c_spin_classes_json(handle, output, capacity, needed) &
        bind(C, name="rusticol_virtual_spin_correlation_classes_json") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: needed
      integer(c_int) :: value
    end function
    function c_correlated_shape(handle, output) bind(C, name="rusticol_virtual_correlated_result_shape") result(value)
      import
      type(c_ptr), value :: handle
      type(virtual_correlated_result_shape) :: output
      integer(c_int) :: value
    end function
    function c_correlated_free(handle) bind(C, name="rusticol_virtual_correlated_result_free") result(value)
      import
      type(c_ptr), value :: handle
      integer(c_int) :: value
    end function
    function c_correlated_born(handle, output, capacity) &
        bind(C, name="rusticol_virtual_correlated_result_copy_born") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_correlated_virtual(handle, output, capacity) &
        bind(C, name="rusticol_virtual_correlated_result_copy_virtual") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_correlated_requested_ids(handle, output, capacity) &
        bind(C, name="rusticol_virtual_correlated_result_copy_requested_ids") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_correlated_successful_ids(handle, output, capacity) &
        bind(C, name="rusticol_virtual_correlated_result_copy_successful_ids") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_correlated_statuses(handle, output, capacity) &
        bind(C, name="rusticol_virtual_correlated_result_copy_statuses") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_int) :: value
    end function
    function c_correlation_histories(handle, output, capacity, needed) &
        bind(C, name="rusticol_virtual_correlation_histories_json") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: needed
      integer(c_int) :: value
    end function
    function c_correlated_histories(handle, output, capacity, needed) &
        bind(C, name="rusticol_virtual_correlated_result_histories_json") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: needed
      integer(c_int) :: value
    end function
    function c_correlated_diagnostics(handle, output, capacity, needed) &
        bind(C, name="rusticol_virtual_correlated_result_diagnostics_json") result(value)
      import
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: needed
      integer(c_int) :: value
    end function
    function c_last_error_message() bind(C, name="rusticol_virtual_last_error_message") result(value)
      import
      type(c_ptr) :: value
    end function
    function c_strlen(text) bind(C, name="strlen") result(value)
      import
      type(c_ptr), value :: text
      integer(c_size_t) :: value
    end function
  end interface
contains
  function cstring(text) result(buffer)
    character(*), intent(in) :: text
    character(c_char), allocatable :: buffer(:)
    integer(c_size_t) :: i, n
    n=len(text,kind=c_size_t)
    allocate(buffer(n+1))
    do i=1,n
      buffer(i)=text(i:i)
    end do
    buffer(n+1)=c_null_char
  end function

  ! Last C function error. evaluate(error_message=...) also reports local shape errors.
  function virtual_last_error() result(message)
    character(:), allocatable :: message
    character(c_char), pointer :: buffer(:)
    type(c_ptr) :: p
    integer(c_size_t) :: n, i
    p=c_last_error_message()
    n=c_strlen(p)
    call c_f_pointer(p,buffer,[n])
    allocate(character(len=n)::message)
    do i=1,n
      message(i:i)=buffer(i)
    end do
  end function

  subroutine finish(status,ierr,message,error_message)
    integer(c_int), intent(in) :: status
    integer(c_int), optional, intent(out) :: ierr
    character(*), optional, intent(in) :: message
    character(:), allocatable, optional, intent(out) :: error_message
    if (present(error_message)) then
      error_message=''
      if (status/=VIRTUAL_OK) then
        if (present(message)) then
          error_message=message
        else
          error_message=virtual_last_error()
        end if
      end if
    end if
    if (present(ierr)) then
      ierr=status
    else if (status/=VIRTUAL_OK) then
      if (present(message)) then
        write(error_unit,'(a)') message
      else
        write(error_unit,'(a)') virtual_last_error()
      end if
      error stop 'virtual SDK function failed'
    end if
  end subroutine

  function handle_of(self) result(handle)
    class(virtual_runtime), intent(in) :: self
    type(c_ptr) :: handle
    handle=c_null_ptr
    if (associated(self%owner)) handle=self%owner%handle
  end function

  impure elemental subroutine release(self)
    class(virtual_runtime), intent(inout) :: self
    integer(c_int) :: ignored
    if (.not.associated(self%owner)) return
    self%owner%references=self%owner%references-1
    if (self%owner%references==0) then
      ignored=c_free(self%owner%handle)
      deallocate(self%owner)
    end if
    nullify(self%owner)
  end subroutine

  impure elemental subroutine finalize_runtime(self)
    type(virtual_runtime), intent(inout) :: self
    call release(self)
  end subroutine

  impure elemental subroutine assign_runtime(lhs,rhs)
    class(virtual_runtime), intent(inout) :: lhs
    class(virtual_runtime), intent(in) :: rhs
    type(runtime_owner), pointer :: saved
    if (associated(lhs%owner,rhs%owner)) return
    saved=>rhs%owner
    if (associated(saved)) saved%references=saved%references+1
    call release(lhs)
    lhs%owner=>saved
  end subroutine

  subroutine close(self)
    class(virtual_runtime), intent(inout) :: self
    call release(self)
  end subroutine

  logical function is_loaded(self)
    class(virtual_runtime), intent(in) :: self
    is_loaded=c_associated(handle_of(self))
  end function

  subroutine load(self,path,process,parameters_json,r2_strategy,ierr,verification_json)
    class(virtual_runtime), intent(inout) :: self
    character(*), intent(in) :: path
    character(*), optional, intent(in) :: process, parameters_json, r2_strategy, verification_json
    integer(c_int), optional, intent(out) :: ierr
    character(c_char), allocatable, target :: p(:), process_c(:), parameters_c(:), strategy_c(:)
    character(c_char), allocatable, target :: policy_c(:)
    type(c_ptr) :: process_p, parameters_p, strategy_p, policy_p, new_handle
    integer(c_int) :: status
    p=cstring(path)
    process_p=c_null_ptr; parameters_p=c_null_ptr; strategy_p=c_null_ptr
    if (present(process)) then
      process_c=cstring(process); process_p=c_loc(process_c)
    end if
    if (present(parameters_json)) then
      parameters_c=cstring(parameters_json); parameters_p=c_loc(parameters_c)
    end if
    if (present(r2_strategy)) then
      strategy_c=cstring(r2_strategy); strategy_p=c_loc(strategy_c)
    end if
    policy_p=c_null_ptr
    if (present(verification_json)) then
      policy_c=cstring(verification_json); policy_p=c_loc(policy_c)
    end if
    status=c_open_policy(c_loc(p),process_p,parameters_p,strategy_p,policy_p,new_handle)
    if (status==VIRTUAL_OK) then
      call release(self)
      allocate(self%owner)
      self%owner%handle=new_handle
    end if
    call finish(status,ierr)
  end subroutine

  subroutine clear(self,ierr)
    class(virtual_runtime), intent(inout) :: self
    integer(c_int), optional, intent(out) :: ierr
    call finish(c_clear(handle_of(self)),ierr)
  end subroutine

  function runtime_shape(self,ierr) result(value)
    class(virtual_runtime), intent(in) :: self
    integer(c_int), optional, intent(out) :: ierr
    type(virtual_shape) :: value
    call finish(c_shape(handle_of(self),value),ierr)
  end function

  function read_json(handle,kind,status) result(text)
    type(c_ptr), intent(in) :: handle
    integer, intent(in) :: kind
    integer(c_int), intent(out) :: status
    character(:), allocatable :: text
    character(c_char), allocatable, target :: buffer(:)
    integer(c_size_t) :: n, i
    text=''
    select case(kind)
    case(1)
      status=c_metadata_json(handle,c_null_ptr,0_c_size_t,n)
    case(2)
      status=c_capabilities_json(handle,c_null_ptr,0_c_size_t,n)
    case(3)
      status=c_result_diagnostics_json(handle,c_null_ptr,0_c_size_t,n)
    case(4)
      status=c_correlation_histories(handle,c_null_ptr,0_c_size_t,n)
    case(5)
      status=c_correlated_histories(handle,c_null_ptr,0_c_size_t,n)
    case(7)
      status=c_spin_legs_json(handle,c_null_ptr,0_c_size_t,n)
    case(8)
      status=c_spin_classes_json(handle,c_null_ptr,0_c_size_t,n)
    case(6)
      status=c_correlated_diagnostics(handle,c_null_ptr,0_c_size_t,n)
    end select
    if (status/=VIRTUAL_OK) return
    allocate(buffer(n))
    select case(kind)
    case(1)
      status=c_metadata_json(handle,c_loc(buffer),n,n)
    case(2)
      status=c_capabilities_json(handle,c_loc(buffer),n,n)
    case(3)
      status=c_result_diagnostics_json(handle,c_loc(buffer),n,n)
    case(4)
      status=c_correlation_histories(handle,c_loc(buffer),n,n)
    case(5)
      status=c_correlated_histories(handle,c_loc(buffer),n,n)
    case(7)
      status=c_spin_legs_json(handle,c_loc(buffer),n,n)
    case(8)
      status=c_spin_classes_json(handle,c_loc(buffer),n,n)
    case(6)
      status=c_correlated_diagnostics(handle,c_loc(buffer),n,n)
    end select
    if (status/=VIRTUAL_OK) return
    deallocate(text)
    allocate(character(len=n-1)::text)
    do i=1,n-1
      text(i:i)=buffer(i)
    end do
  end function

  function metadata_json(self,ierr) result(text)
    class(virtual_runtime), intent(in) :: self
    integer(c_int), optional, intent(out) :: ierr
    character(:), allocatable :: text
    integer(c_int) :: status
    text=read_json(handle_of(self),1,status)
    call finish(status,ierr)
  end function

  function capabilities_json(self,ierr) result(text)
    class(virtual_runtime), intent(in) :: self
    integer(c_int), optional, intent(out) :: ierr
    character(:), allocatable :: text
    integer(c_int) :: status
    text=read_json(handle_of(self),2,status)
    call finish(status,ierr)
  end function

  function correlation_histories_json(self,ierr) result(text)
    class(virtual_runtime), intent(in) :: self
    integer(c_int), optional, intent(out) :: ierr
    character(:), allocatable :: text
    integer(c_int) :: status
    text=read_json(handle_of(self),4,status)
    call finish(status,ierr)
  end function

  function spin_correlation_classes_json(self,ierr) result(text)
    class(virtual_runtime), intent(in) :: self
    integer(c_int), optional, intent(out) :: ierr
    character(:), allocatable :: text
    integer(c_int) :: status
    text=read_json(handle_of(self),8,status)
    call finish(status,ierr)
  end function

  ! Plan v4: the whole mapping is validated first, as the Python API's
  ! set_model_parameters; a value that is not a finite decimal, an unknown
  ! name, the renormalisation scale's parameter (set per point by mu_r2) or a
  ! parameter fixed at generation at another value sets
  ! ierr = RUSTICOL_VIRTUAL_INVALID_ARGUMENT (1), a nonzero imaginary part
  ! ierr = RUSTICOL_VIRTUAL_RUNTIME_ERROR (3), and changes nothing;
  ! error_message (optional) then holds the refusal.
  subroutine update_parameters(self,exact_json,revision,ierr,error_message)
    class(virtual_runtime), intent(inout) :: self
    character(*), intent(in) :: exact_json
    integer(c_int64_t), intent(out) :: revision
    integer(c_int), optional, intent(out) :: ierr
    character(:), allocatable, optional, intent(out) :: error_message
    character(c_char), allocatable, target :: buffer(:)
    buffer=cstring(exact_json)
    call finish(c_update_parameters(handle_of(self),c_loc(buffer),revision),ierr, &
      error_message=error_message)
  end subroutine

  ! Pole selection of later evaluations: "all" (default), or for plan v4
  ! "none" (the finite part, the pole entries quiet NaNs), "uv" (the bare
  ! loop's UV poles) or "ir" (the renormalised virtual's IR poles).
  subroutine set_poles(self,poles,ierr)
    class(virtual_runtime), intent(inout) :: self
    character(*), intent(in) :: poles
    integer(c_int), optional, intent(out) :: ierr
    character(c_char), allocatable, target :: buffer(:)
    buffer=cstring(poles)
    call finish(c_set_poles(handle_of(self),c_loc(buffer)),ierr)
  end subroutine

  subroutine evaluate(self,momenta,mu_r2,output,point_ids,resolved,r2_strategy,verification_json,ierr,error_message)
    class(virtual_runtime), intent(inout) :: self
    real(c_double), contiguous, target, intent(in) :: momenta(:,:,:), mu_r2(:)
    type(virtual_evaluation), intent(out) :: output
    integer(c_int64_t), contiguous, target, optional, intent(in) :: point_ids(:)
    logical, optional, intent(in) :: resolved
    character(*), optional, intent(in) :: r2_strategy, verification_json
    integer(c_int), optional, intent(out) :: ierr
    character(:), allocatable, optional, intent(out) :: error_message
    type(c_batch_request) :: request
    type(c_ptr) :: result_handle, destination
    type(virtual_shape) :: axes
    type(virtual_evaluation), target :: value
    character(c_char), allocatable, target :: strategy(:), verification(:)
    integer(c_int) :: status, ignored
    integer(c_size_t) :: helicities, colors, points
    axes=self%shape(status)
    if (status/=VIRTUAL_OK) then
      call finish(status,ierr,error_message=error_message); return
    end if
    points=size(momenta,3,kind=c_size_t)
    if (size(momenta,1,kind=c_size_t)/=4 .or. size(momenta,2,kind=c_size_t)/=axes%external_count) then
      call finish(VIRTUAL_INVALID_ARGUMENT,ierr,'momenta must have shape (4,external_leg,point)',error_message)
      return
    end if
    if (present(point_ids)) then
      if (size(point_ids,kind=c_size_t)/=points) then
        call finish(VIRTUAL_INVALID_ARGUMENT,ierr,'point_ids must have one entry per input point',error_message)
        return
      end if
    end if
    request%momenta=c_null_ptr; request%point_ids=c_null_ptr; request%mu_r2=c_null_ptr
    request%momentum_count=size(momenta,kind=c_size_t); request%point_count=points
    request%scale_count=size(mu_r2,kind=c_size_t)
    if (size(momenta,kind=c_size_t)>0) request%momenta=c_loc(momenta)
    if (present(point_ids)) then
      if (size(point_ids,kind=c_size_t)>0) request%point_ids=c_loc(point_ids)
    end if
    if (size(mu_r2,kind=c_size_t)>0) request%mu_r2=c_loc(mu_r2)
    request%resolved=0
    if (present(resolved)) then
      if (resolved) request%resolved=1
    end if
    request%r2_strategy=c_null_ptr; request%verification_json=c_null_ptr
    if (present(r2_strategy)) then
      strategy=cstring(r2_strategy); request%r2_strategy=c_loc(strategy)
    end if
    if (present(verification_json)) then
      verification=cstring(verification_json); request%verification_json=c_loc(verification)
    end if
    status=c_evaluate(handle_of(self),request,result_handle)
    if (status/=VIRTUAL_OK) then
      call finish(status,ierr,error_message=error_message); return
    end if
    status=c_result_shape(result_handle,value%shape)
    if (status==VIRTUAL_OK) then
      axes=value%shape%axes; points=value%shape%successful_points
      helicities=1
      if (value%shape%resolved/=0) helicities=axes%helicities
      colors=1
      if (value%shape%resolved/=0) colors=axes%born_color_components
      allocate(value%born(colors,helicities,axes%born_orders,points))
      colors=1
      if (value%shape%resolved/=0) colors=axes%virtual_color_components
      allocate(value%virtual_values(axes%epsilon_count,colors,helicities,axes%virtual_orders,points))
      allocate(value%requested_ids(value%shape%requested_points),value%successful_ids(points))
      allocate(value%statuses(value%shape%requested_points))
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(value%born,kind=c_size_t)>0) destination=c_loc(value%born)
        status=c_result_copy_born(result_handle,destination,size(value%born,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(value%virtual_values,kind=c_size_t)>0) destination=c_loc(value%virtual_values)
        status=c_result_copy_virtual(result_handle,destination,size(value%virtual_values,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(value%requested_ids,kind=c_size_t)>0) destination=c_loc(value%requested_ids)
        status=c_result_copy_requested_ids(result_handle,destination,size(value%requested_ids,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(value%successful_ids,kind=c_size_t)>0) destination=c_loc(value%successful_ids)
        status=c_result_copy_successful_ids(result_handle,destination,size(value%successful_ids,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(value%statuses,kind=c_size_t)>0) destination=c_loc(value%statuses)
        status=c_result_copy_statuses(result_handle,destination,size(value%statuses,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) value%diagnostics_json=read_json(result_handle,3,status)
    end if
    ! Free regardless of copy/query failures; report the original function status.
    ignored=c_result_free(result_handle)
    if (status==VIRTUAL_OK) then
      output%shape=value%shape
      call move_alloc(value%born,output%born)
      call move_alloc(value%virtual_values,output%virtual_values)
      call move_alloc(value%requested_ids,output%requested_ids)
      call move_alloc(value%successful_ids,output%successful_ids)
      call move_alloc(value%statuses,output%statuses)
      call move_alloc(value%diagnostics_json,output%diagnostics_json)
    end if
    call finish(status,ierr,error_message=error_message)
  end subroutine
  subroutine evaluate_correlated(self,momenta,mu_r2,output,point_ids,r2_strategy,verification_json, &
      ierr,error_message,spin_legs,spin_vectors)
    class(virtual_runtime), intent(inout) :: self
    real(c_double), contiguous, target, intent(in) :: momenta(:,:,:), mu_r2(:)
    type(virtual_correlated_evaluation), intent(out) :: output
    integer(c_int64_t), contiguous, target, optional, intent(in) :: point_ids(:)
    character(*), optional, intent(in) :: r2_strategy, verification_json
    integer(c_int), optional, intent(out) :: ierr
    character(:), allocatable, optional, intent(out) :: error_message
    integer(c_size_t), contiguous, target, optional, intent(in) :: spin_legs(:)
    complex(c_double), contiguous, optional, intent(in) :: spin_vectors(:,:,:) ! component, selected leg, row
    type(complex_pair), allocatable, target :: source_pairs(:,:,:)
    type(c_spin_sources) :: sources
    type(c_batch_request) :: request
    type(c_ptr) :: result_handle, destination
    type(virtual_shape) :: axes
    type(virtual_correlated_evaluation), target :: value
    character(c_char), allocatable, target :: strategy(:), verification(:)
    integer(c_int) :: status, ignored
    integer(c_size_t) :: points
    type(virtual_correlated_shape) :: correlated_axes
    type(complex_pair), allocatable, target :: born_pairs(:,:,:), virtual_pairs(:,:,:,:)
    axes=self%shape(status)
    if (status/=VIRTUAL_OK) then
      call finish(status,ierr,error_message=error_message); return
    end if
    points=size(momenta,3,kind=c_size_t)
    if (size(momenta,1,kind=c_size_t)/=4 .or. size(momenta,2,kind=c_size_t)/=axes%external_count) then
      call finish(VIRTUAL_INVALID_ARGUMENT,ierr,'momenta must have shape (4,external_leg,point)',error_message)
      return
    end if
    if (present(point_ids)) then
      if (size(point_ids,kind=c_size_t)/=points) then
        call finish(VIRTUAL_INVALID_ARGUMENT,ierr,'point_ids must have one entry per input point',error_message)
        return
      end if
    end if
    request%momenta=c_null_ptr; request%point_ids=c_null_ptr; request%mu_r2=c_null_ptr
    request%momentum_count=size(momenta,kind=c_size_t); request%point_count=points
    request%scale_count=size(mu_r2,kind=c_size_t)
    if (size(momenta,kind=c_size_t)>0) request%momenta=c_loc(momenta)
    if (present(point_ids)) then
      if (size(point_ids,kind=c_size_t)>0) request%point_ids=c_loc(point_ids)
    end if
    if (size(mu_r2,kind=c_size_t)>0) request%mu_r2=c_loc(mu_r2)
    request%resolved=0
    request%r2_strategy=c_null_ptr; request%verification_json=c_null_ptr
    if (present(r2_strategy)) then
      strategy=cstring(r2_strategy); request%r2_strategy=c_loc(strategy)
    end if
    if (present(verification_json)) then
      verification=cstring(verification_json); request%verification_json=c_loc(verification)
    end if
    if (present(spin_legs).neqv.present(spin_vectors)) then
      call finish(VIRTUAL_INVALID_ARGUMENT,ierr,'spin_legs and spin_vectors must be supplied together',error_message)
      return
    end if
    if (present(spin_legs)) then
      if (size(spin_vectors,1,kind=c_size_t)/=4 .or. &
          size(spin_vectors,2,kind=c_size_t)/=size(spin_legs,kind=c_size_t)) then
        call finish(VIRTUAL_INVALID_ARGUMENT,ierr,'spin_vectors must have shape (4,selected_leg,row)',error_message)
        return
      end if
      allocate(source_pairs(4,size(spin_legs,kind=c_size_t),size(spin_vectors,3,kind=c_size_t)))
      source_pairs%real=real(spin_vectors,kind=c_double); source_pairs%imag=aimag(spin_vectors)
      sources%legs=c_null_ptr; sources%values=c_null_ptr
      sources%leg_count=size(spin_legs,kind=c_size_t); sources%value_count=size(source_pairs,kind=c_size_t)
      if (sources%leg_count>0) sources%legs=c_loc(spin_legs)
      if (sources%value_count>0) sources%values=c_loc(source_pairs)
      status=c_correlated_sources(handle_of(self),request,sources,result_handle)
    else
      status=c_correlated_evaluate(handle_of(self),request,result_handle)
    end if
    if (status/=VIRTUAL_OK) then
      call finish(status,ierr,error_message=error_message); return
    end if
    status=c_correlated_shape(result_handle,value%shape)
    if (status==VIRTUAL_OK) then
      correlated_axes=value%shape%axes; points=value%shape%successful_points
      allocate(born_pairs(correlated_axes%born_orders,correlated_axes%histories,points))
      allocate(virtual_pairs(correlated_axes%epsilon_count,correlated_axes%virtual_orders, &
          correlated_axes%histories,points))
      allocate(value%requested_ids(value%shape%requested_points),value%successful_ids(points))
      allocate(value%statuses(value%shape%requested_points))
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(born_pairs,kind=c_size_t)>0) destination=c_loc(born_pairs)
        status=c_correlated_born(result_handle,destination,size(born_pairs,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(virtual_pairs,kind=c_size_t)>0) destination=c_loc(virtual_pairs)
        status=c_correlated_virtual(result_handle,destination,size(virtual_pairs,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(value%requested_ids,kind=c_size_t)>0) destination=c_loc(value%requested_ids)
        status=c_correlated_requested_ids(result_handle,destination,size(value%requested_ids,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(value%successful_ids,kind=c_size_t)>0) destination=c_loc(value%successful_ids)
        status=c_correlated_successful_ids(result_handle,destination,size(value%successful_ids,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) then
        destination=c_null_ptr
        if (size(value%statuses,kind=c_size_t)>0) destination=c_loc(value%statuses)
        status=c_correlated_statuses(result_handle,destination,size(value%statuses,kind=c_size_t))
      end if
      if (status==VIRTUAL_OK) value%diagnostics_json=read_json(result_handle,6,status)
      if (status==VIRTUAL_OK) value%histories_json=read_json(result_handle,5,status)
      if (status==VIRTUAL_OK) value%spin_legs_json=read_json(result_handle,7,status)
    end if
    ! Free regardless of copy/query failures; report the original function status.
    ignored=c_correlated_free(result_handle)
    if (status==VIRTUAL_OK) then
      output%shape=value%shape
      output%born=cmplx(born_pairs%real,born_pairs%imag,kind=c_double)
      output%virtual_values=cmplx(virtual_pairs%real,virtual_pairs%imag,kind=c_double)
      call move_alloc(value%requested_ids,output%requested_ids)
      call move_alloc(value%successful_ids,output%successful_ids)
      call move_alloc(value%statuses,output%statuses)
      call move_alloc(value%diagnostics_json,output%diagnostics_json)
      call move_alloc(value%histories_json,output%histories_json)
      call move_alloc(value%spin_legs_json,output%spin_legs_json)
    end if
    call finish(status,ierr,error_message=error_message)
  end subroutine
end module
