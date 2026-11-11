program caller_bound_probe
  use ml_private_cms_caller_bounds
  use, intrinsic :: iso_c_binding, only: c_int64_t
  use, intrinsic :: ieee_exceptions
  implicit none
  integer :: count_cases,case_id,route,mode,width,n,nw,env,genuine,k,i,j,mu,e,part,nt,ios
  integer,allocatable :: we(:),pairing(:),chord(:,:),classes(:,:),qerror(:,:,:)
  integer(ik) :: external(2,0:3,4)
  integer(ik),allocatable :: shifts(:,:)
  logical,allocatable :: have_chord(:),have_pair(:,:)
  logical :: route_ok,producer_ok,valid,flags_before(5),flags_after(5)
  integer(c_int64_t) :: before,after,token
  character(len=32) :: hexword
  character(len=8192) :: line
  before=read_kinematic_environment_token()
  write(*,'(A,1X,Z16.16,1X,L1)') 'ENV_BEGIN',before,environment_ok(before)
  if(.not.environment_ok(before)) stop 2
  open(10,file='inputs.txt',status='old',action='read')
  read(10,*) count_cases
  do k=1,count_cases
    read(10,*) case_id,route,mode,width,n,nw,env,genuine
    if(case_id/=k.or.n<1.or.n>4.or.nw<1.or.nw>8) stop 3
    allocate(we(nw),pairing(n),chord(0:3,n),classes(n,n),qerror(0:3,n,n))
    allocate(shifts(0:3,n),have_chord(n),have_pair(n,n))
    read(10,*) we
    read(10,*) pairing
    read(10,'(A)') line
    do e=1,4
      do mu=0,3
        do part=1,2
          i=((e-1)*8+mu*2+part-1)*33+1
          hexword=line(i:i+31)
          read(hexword,'(Z32)') external(part,mu,e)
        enddo
      enddo
    enddo
    read(10,'(A)') line
    do e=1,n
      do mu=0,3
        i=((e-1)*4+mu)*33+1
        hexword=line(i:i+31)
        read(hexword,'(Z32)') shifts(mu,e)
      enddo
    enddo
    call ieee_get_flag(ieee_all,flags_before)
    route_ok=route_matches(route,n,nw,we,pairing)
    producer_ok=producer_kind_ok(mode,width,genuine/=0)
    valid=env/=0.and.route_ok.and.producer_ok.and.environment_unchanged(before)
    call build_chord_bounds(route,mode,width,n,external,valid,chord,classes,have_chord)
    call finish_difference_bounds(n,width,shifts,chord,classes,have_chord,valid,qerror,have_pair)
    call ieee_get_flag(ieee_all,flags_after)
    write(*,'(A,3(1X,I0))') 'C',case_id,merge(1,0,route_ok),merge(1,0,producer_ok)
    do j=1,n
      write(*,'(A,7(1X,I0))') 'B',case_id,j,merge(1,0,have_chord(j)),chord(:,j)
    enddo
    do j=1,n
      do i=1,n
        write(*,'(A,9(1X,I0))') 'P',case_id,i,j,classes(i,j),merge(1,0,have_pair(i,j)),qerror(:,i,j)
      enddo
    enddo
    write(*,'(A,1X,I0,2(1X,L1),1X,5L1,1X,5L1)') 'E',case_id, &
      all(flags_before.eqv.flags_after),environment_unchanged(before),flags_before,flags_after
    deallocate(we,pairing,chord,classes,qerror,shifts,have_chord,have_pair)
  enddo
  read(10,'(A)',iostat=ios) line
  if(ios>=0) stop 4
  close(10)
  open(11,file='tokens.txt',status='old',action='read')
  read(11,*) nt
  do k=1,nt
    read(11,'(Z16)') token
    write(*,'(A,1X,I0,1X,L1)') 'T',k,environment_ok(token)
  enddo
  read(11,'(A)',iostat=ios) line
  if(ios>=0) stop 5
  close(11)
  after=read_kinematic_environment_token()
  write(*,'(A,1X,Z16.16,2(1X,L1))') 'ENV_END',after,environment_ok(after),after==before
  write(*,'(A,2(1X,I0))') 'DONE',count_cases,nt
end program
