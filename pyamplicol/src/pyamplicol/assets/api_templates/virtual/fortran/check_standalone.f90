! SPDX-License-Identifier: 0BSD
! One native bulk call, with Fortran axes documented by the installed SDK.
program check_standalone
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic
  use rusticol_virtual
  implicit none
  type(virtual_runtime) :: runtime
  type(virtual_evaluation) :: result
  type(virtual_shape) :: axes
  character(:),allocatable :: process,argument,line,text
  real(c_double),allocatable :: momenta(:,:,:),grown(:,:,:),point(:,:)
  real(c_double) :: scale
  integer :: narg,nchar,unit,ios,delimiter
  integer(c_size_t) :: points,nlegs,p,o,h,c,e
  logical :: resolved,eof
  narg=command_argument_count()
  if(narg<2.or.narg>3) error stop 'usage: check_standalone PROCESS MU_R2 [resolved]'
  call get_command_argument(1,length=nchar)
  allocate(character(nchar)::process); call get_command_argument(1,process)
  call get_command_argument(2,length=nchar)
  allocate(character(nchar)::argument); call get_command_argument(2,argument)
  read(argument,*,iostat=ios) scale
  if(ios/=0) error stop 'invalid MU_R2'
  if(.not.ieee_is_finite(scale).or.scale<=0) error stop 'MU_R2 must be positive and finite'
  resolved=narg==3
  if(resolved) then
    deallocate(argument); call get_command_argument(3,length=nchar)
    allocate(character(nchar)::argument); call get_command_argument(3,argument)
    if(argument/='resolved') error stop 'expected resolved'
  end if
  call runtime%load('.',process)
  axes=runtime%shape(); nlegs=axes%external_count
  allocate(point(4,nlegs),momenta(4,nlegs,0)); points=0
  open(newunit=unit,file='API/validation_points.dat',status='old',iostat=ios)
  if(ios/=0) error stop 'cannot open validation points'
  call read_line(unit,line,eof)
  if(eof.or.line/='RUSTICOL_VALIDATION_POINTS_V1') error stop 'invalid point header'
  do
    call read_line(unit,line,eof)
    if(eof) exit
    delimiter=scan(line,' '//achar(9))
    if(delimiter<=1) cycle
    if(line(:delimiter-1)/=process) cycle
    read(line(delimiter+1:),*,iostat=ios) nlegs
    if(ios/=0.or.nlegs/=axes%external_count) error stop 'invalid external count'
    read(line(delimiter+1:),*,iostat=ios) nlegs,point
    if(ios/=0.or.any(.not.ieee_is_finite(point))) error stop 'invalid momenta'
    if(points==huge(points)) error stop 'point count overflow'
    allocate(grown(4,nlegs,points+1))
    grown(:,:,:points)=momenta; grown(:,:,points+1)=point
    call move_alloc(grown,momenta); points=points+1
  end do
  close(unit)
  if(points==0) error stop 'no validation points for selected process'
  call runtime%evaluate(momenta,[scale],result,resolved=resolved)
  text=runtime%metadata_json(); print '(A)',text
  print *, 'shape Born [colour,helicity,order,point]',shape(result%born,kind=c_size_t)
  print *, 'shape virtual [epsilon,colour,helicity,order,point]',shape(result%virtual_values,kind=c_size_t)
  print *, 'requested_ids',result%requested_ids
  print *, 'successful_ids',result%successful_ids
  print *, 'statuses',result%statuses
  do p=1,size(result%successful_ids,kind=c_size_t)
    do o=1,size(result%born,3,kind=c_size_t)
      do h=1,size(result%born,2,kind=c_size_t)
        do c=1,size(result%born,1,kind=c_size_t)
          print '(A,4(1X,I0),1X,ES25.17E3)', 'born',p-1,o-1,h-1,c-1,result%born(c,h,o,p)
        end do
      end do
    end do
    do o=1,size(result%virtual_values,4,kind=c_size_t)
      do h=1,size(result%virtual_values,3,kind=c_size_t)
        do c=1,size(result%virtual_values,2,kind=c_size_t)
          do e=1,3
            print '(A,5(1X,I0),1X,ES25.17E3)', 'virtual',p-1,o-1,h-1,c-1,e-1,result%virtual_values(e,c,h,o,p)
          end do
        end do
      end do
    end do
  end do
  print '(A)',result%diagnostics_json
contains
  subroutine read_line(unit,text,eof)
    integer,intent(in)::unit
    character(:),allocatable,intent(out)::text
    logical,intent(out)::eof
    character(256)::chunk
    integer::status,count
    text=''; eof=.false.
    do
      read(unit,'(A)',advance='no',size=count,iostat=status) chunk
      if(count>0) text=text//chunk(:count)
      if(is_iostat_end(status)) then
        eof=len(text)==0; exit
      end if
      if(is_iostat_eor(status)) exit
      if(status/=0) error stop 'failed reading validation points'
    end do
  end subroutine
end program
