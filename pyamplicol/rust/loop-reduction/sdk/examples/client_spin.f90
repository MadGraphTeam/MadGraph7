! SPDX-License-Identifier: 0BSD
program spin_client
  use rusticol_virtual
  use, intrinsic :: iso_c_binding
  implicit none
  type(virtual_runtime) :: runtime
  type(virtual_correlated_evaluation) :: baseline, scaled, rejected
  real(c_double), target :: p(4,4,1), scale(1)
  integer(c_int64_t), target :: ids(1)
  integer(c_size_t), target :: legs(2)
  complex(c_double) :: vectors(4,2,1)
  character(4096) :: path
  character(:), allocatable :: message
  integer(c_int) :: status
  call get_command_argument(1,path)
  call runtime%load(trim(path))
  if(runtime%spin_correlation_classes_json()/='[[3],[4],[3,4]]') error stop 1
  p(:,:,1)=reshape([500d0,0d0,0d0,500d0,500d0,0d0,0d0,-500d0, &
      500d0,300d0,0d0,400d0,500d0,-300d0,0d0,-400d0],[4,4])
  scale=1d6; ids=-1_c_int64_t; legs=[3_c_size_t,4_c_size_t]
  vectors(:,1,1)=[cmplx(0d0,0d0,c_double),cmplx(.8d0,0d0,c_double), &
      cmplx(0d0,1d0,c_double),cmplx(-.6d0,0d0,c_double)]
  vectors(:,2,1)=vectors(:,1,1)
  call runtime%evaluate_correlated(p,scale,baseline,point_ids=ids,spin_legs=legs,spin_vectors=vectors)
  vectors(:,1,:)=vectors(:,1,:)*cmplx(2d0,-3d0,c_double)
  vectors(:,2,:)=vectors(:,2,:)*cmplx(1d0,2d0,c_double)
  call runtime%evaluate_correlated(p,scale,scaled,spin_legs=legs,spin_vectors=vectors)
  if(baseline%successful_ids(1)/=-1 .or. any(shape(baseline%virtual_values)/=[3,1,3,1])) error stop 2
  if(any(abs(scaled%virtual_values-65d0*baseline%virtual_values)> &
      2d-8*max(1d0,abs(scaled%virtual_values)))) error stop 3
  call runtime%evaluate_correlated(p,scale,rejected,spin_legs=legs,ierr=status,error_message=message)
  if(status==VIRTUAL_OK .or. allocated(rejected%born) .or. index(message,'together')==0) error stop 4
  vectors=cmplx(0d0,0d0,c_double); legs=0
  call runtime%clear(); call runtime%close()
  if(baseline%spin_legs_json/='[3,4]' .or. size(baseline%born)/=3) error stop 5
  if(baseline%histories_json/='["identity","ordered","swapped"]') error stop 6
  print *, 'Fortran joint source axes and owned labels passed'
end program
