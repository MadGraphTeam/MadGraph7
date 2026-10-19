! SPDX-License-Identifier: 0BSD
program correlated_client
  use rusticol_virtual
  use, intrinsic :: iso_c_binding
  implicit none
  type(virtual_runtime) :: runtime
  type(virtual_correlated_evaluation) :: result
  real(c_double), target :: p(4,4,1), scale(1)
  integer(c_int64_t), target :: ids(1)
  character(4096) :: path
  integer :: e
  call get_command_argument(1,path)
  call runtime%load(trim(path))
  p(:,:,1)=reshape([500d0,0d0,0d0,500d0,500d0,0d0,0d0,-500d0, &
      500d0,300d0,0d0,400d0,500d0,-300d0,0d0,-400d0],[4,4])
  scale=1d6; ids=-1_c_int64_t ! uint64 max bit pattern
  call runtime%evaluate_correlated(p,scale,result,point_ids=ids)
  if(result%shape%axes%histories/=4 .or. result%successful_ids(1)/=-1) error stop 1
  if(any(shape(result%virtual_values)/=[3,1,4,1])) error stop 2
  if(abs(aimag(result%virtual_values(1,1,3,1)))<1d-8) error stop 3
  do e=1,3
    if(abs(result%virtual_values(e,1,3,1)-conjg(result%virtual_values(e,1,4,1)))>1d-8) error stop 4
  end do
  call runtime%close()
  if(index(result%histories_json,'ordered')==0 .or. size(result%born)/=4) error stop 5
  print *, 'Fortran owned complex histories passed'
end program
