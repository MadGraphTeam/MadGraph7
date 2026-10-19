! SPDX-License-Identifier: 0BSD
! ./client-fortran ARTIFACT [gg|uudd]
program client
  use, intrinsic :: iso_c_binding
  use rusticol_virtual
  implicit none
  type :: runtime_box
    type(virtual_runtime) :: value
  end type
  type(virtual_runtime) :: runtime, alias
  type(virtual_evaluation) :: inclusive, resolved, partial, verified, empty, retained
  type(virtual_shape) :: axes
  real(c_double), target :: point(4,4), momenta(4,4,3), scales(3)
  real(c_double), allocatable :: zero_points(:,:,:), zero_scales(:)
  integer(c_int64_t), target :: ids(3)=[91_c_int64_t,4_c_int64_t,70_c_int64_t]
  integer(c_int64_t) :: revision
  integer(c_int) :: status
  integer :: length, p, o, e
  integer, parameter :: born_order(3)=[2,1,3], virtual_order(3)=[3,2,1]
  character(:), allocatable :: path, text
  character(4) :: process
  real(c_double), parameter :: expected_born(3)=[0.8013367806451908d0,-2.0322878122576093d0,3.0512301482522957d0]
  real(c_double), parameter :: expected(3,3)=reshape([ &
    -0.19094989612756078d0,-0.28421984016919477d0,-0.08026310265110026d0, &
     1.313041990604555d0,0.7811563156803705d0,0.20355701776284274d0, &
    -3.018755956775497d0,-1.1476469718544264d0,-0.3056158216076459d0],[3,3])
  character(*), parameter :: policy='{"higher":[{"arithmetic":"double-double","precision":null}],' // &
    '"relative_tolerance":"1e-8","absolute_tolerance":"1e-9","verify_successful":true}'
  call get_command_argument(1,length=length)
  if (length==0) stop 2
  allocate(character(length)::path)
  call get_command_argument(1,path)
  process='gg'
  if (command_argument_count()>1) call get_command_argument(2,process)
  call runtime%load(path,trim(process),r2_strategy='model')
  alias=runtime
  alias=alias
  call runtime%close()
  if (runtime%is_loaded().or..not.alias%is_loaded()) stop 1
  runtime=alias
  call alias%close()
  axes=runtime%shape()
  if (axes%external_count/=4.or.axes%epsilon_count/=3.or.axes%resolved_available/=1) stop 1
  if (trim(process)=='uudd'.and.axes%born_orders/=3) stop 1
  text=runtime%metadata_json()
  if (index(text,'"epsilon_powers":[0,-1,-2]')==0) stop 1
  text=runtime%capabilities_json()
  if (len(text)==0) stop 1
  point=reshape([500d0,0d0,0d0,500d0,500d0,0d0,0d0,-500d0, &
                 500d0,300d0,0d0,400d0,500d0,-300d0,0d0,-400d0],[4,4])
  do p=1,3
    momenta(:,:,p)=point
  end do
  scales=[1d6,9d5,1d6]
  call runtime%evaluate(momenta,scales,inclusive,ids)
  call runtime%evaluate(momenta,scales,resolved,ids,resolved=.true.)
  if (any(inclusive%successful_ids/=ids).or.any(resolved%requested_ids/=ids)) stop 1
  if (any(inclusive%statuses/=VIRTUAL_SUCCESS)) stop 1
  do p=1,3
    do o=1,int(axes%born_orders)
      call close_value(sum(resolved%born(:,:,o,p)),inclusive%born(1,1,o,p))
    end do
    do o=1,int(axes%virtual_orders)
      do e=1,3
        call close_value(sum(resolved%virtual_values(e,:,:,o,p)),inclusive%virtual_values(e,1,1,o,p))
      end do
    end do
  end do
  if (trim(process)=='gg') then
    call close_value(inclusive%virtual_values(3,1,1,1,1),-12d0*0.118d0/(2d0*acos(-1d0))*inclusive%born(1,1,1,1))
  else
    do o=1,3
      call close_value(inclusive%born(1,1,o,1),expected_born(born_order(o)))
      do e=1,3
        call close_value(inclusive%virtual_values(e,1,1,o,1),expected(e,virtual_order(o)))
      end do
    end do
  end if
  momenta(:,:,2)=point*1d200
  call runtime%evaluate(momenta,scales,partial,ids,resolved=.true.)
  if (partial%shape%successful_points/=2.or.any(partial%successful_ids/=[91_c_int64_t,70_c_int64_t])) stop 1
  if (any(partial%statuses/=[VIRTUAL_SUCCESS,VIRTUAL_FAILED,VIRTUAL_SUCCESS])) stop 1
  if (len(partial%diagnostics_json)==0) stop 1
  call runtime%evaluate(momenta(:,:,:1),scales(:1),verified,ids(:1),resolved=.true.,verification_json=policy)
  if (any(verified%statuses/=VIRTUAL_VERIFIED)) stop 1
  allocate(zero_points(4,4,0),zero_scales(0))
  call runtime%evaluate(zero_points,zero_scales,empty,resolved=.true.)
  if (size(empty%born)/=0.or.size(empty%virtual_values)/=0.or.size(empty%statuses)/=0) stop 1
  call runtime%evaluate(momenta,scales,empty,ids(:1),ierr=status,error_message=text)
  if (status/=VIRTUAL_INVALID_ARGUMENT.or.allocated(empty%born)) stop 1
  if (index(text,'point_ids')==0) stop 1
  call runtime%evaluate(momenta(:,:,:1),[0d0],empty,ierr=status)
  if (status==VIRTUAL_OK.or.allocated(empty%born)) stop 1
  text=virtual_last_error()
  if (len(text)==0) stop 1
  call runtime%update_parameters('{"aS":["0.12","0"]}',revision)
  if (revision<=0) stop 1
  call runtime%clear()
  retained=inclusive
  call runtime%close()
  if (any(abs(retained%born-inclusive%born)>0d0)) stop 1
  ! Scalar and array finalization release shared owners without double-free.
  call assignment_scope(path,trim(process))
  print '(a)', 'native Fortran '//trim(process)//' orders/poles/resolved/failures/verification/lifetime controls passed'
contains
  subroutine close_value(a,b)
    real(c_double), intent(in) :: a,b
    if (abs(a-b)>1d-8*max(abs(a),abs(b),1d-20)) then
      print *,a,b
      stop 1
    end if
  end subroutine
  subroutine assignment_scope(path,process)
    character(*), intent(in) :: path,process
    type(virtual_runtime) :: owner, copies(2)
    type(runtime_box) :: a,b
    call owner%load(path,process)
    copies=owner
    call owner%close()
    call copies(1)%clear()
    call copies(1)%close()
    call copies(2)%clear()
    a%value=copies(2)
    b=a
    call a%value%close()
    call copies(2)%close()
    call b%value%clear()
    owner=make_runtime(path,process)
    call owner%clear()
  end subroutine
  function make_runtime(path,process) result(value)
    character(*), intent(in) :: path,process
    type(virtual_runtime) :: value
    call value%load(path,process)
  end function
end program
