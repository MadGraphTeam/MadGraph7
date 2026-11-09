! Source-only preparation: compile/run only after explicit lane authorization.
program ew_tensor_replay
  use, intrinsic :: iso_c_binding, only: c_int64_t
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mtirust, only: KI_QTIR, TIRUST_SUCCESS, tirust_tensor_evaluate, &
       tirust_coefficient_count, tirust_register_plan, tirust_set_cache_budget, &
       tirust_clear_cache, tirust_qp_cache_statistics, tirust_error_message, tirust_finalize
  use mninjars, only: KI_QNRS, NINJARS_SUCCESS, ninjars_tensor_evaluate, &
       ninjars_error_message, ninjars_finalize
  implicit none
  integer, parameter :: q=KI_QTIR
  integer, parameter :: mapping(0:34) = [1,2,3,4,5,6,7,10,8,11,13,9,12,14,15, &
       16,17,20,26,18,21,27,23,29,32,19,22,28,24,30,33,25,31,34,35]
  integer, parameter :: cumulative(0:3) = [1,5,15,35]
  character(len=1024) :: filename
  character(len=64) :: magic
  character(len=2048) :: error
  integer :: cases,c,point,group,n,rank,historical_status,nc,i,j,k,degree,ts,ns,setup
  real(q), allocatable :: p(:,:),s(:,:)
  complex(q), allocatable :: masses(:),raw(:,:),unit(:)
  real(q) :: mu2,re,im
  complex(q) :: result(0:2),r1,mapped(0:2)
  integer(c_int64_t) :: counts(6)

  if (q /= KI_QNRS .or. radix(1.0_q) /= 2 .or. digits(1.0_q) /= 113) stop 1
  call get_command_argument(1,filename)
  if (len_trim(filename)==0) stop 2
  open(73,file=trim(filename),status='old',action='read')
  read(73,'(A)') magic
  if (trim(magic)/='EW_TENSOR_REPLAY_V1') stop 3
  read(73,*) cases
  if (cases<1.or.cases>3) stop 4
  write(*,'(A)') 'EW_TENSOR_REPLAY_V1'
  write(*,'(A,I4)') 'CASES ',cases
  call tirust_set_cache_budget(0_c_int64_t,setup)
  if (setup/=TIRUST_SUCCESS) then
    call tirust_error_message(error)
    write(*,'(A,I4,1X,A)') 'SETUP_ERROR ',setup,trim(error)
    stop 5
  endif

  do c=1,cases
    read(73,*) point,group,n,rank,historical_status
    if (n<1.or.n>6.or.rank<0.or.rank>3) stop 6
    nc=tirust_coefficient_count(rank)
    if (nc/=cumulative(rank)) stop 7
    allocate(p(0:3,n),s(n,n),masses(n),raw(nc,4),unit(nc))
    read(73,*) mu2
    do i=1,n
      read(73,*) p(:,i)
    enddo
    do i=1,n
      read(73,*) re,im
      masses(i)=cmplx(re,im,kind=q)
    enddo
    do i=1,n
      read(73,*) s(i,:)
    enddo
    write(*,'(A,6I6)') 'BEGIN ',point,group,n,rank,historical_status,nc
    write(*,'(A,ES45.35E4)') 'SCALE ',mu2
    do i=1,n
      write(*,'(A,I4,4ES45.35E4)') 'SHIFT ',i,p(:,i)
      write(*,'(A,I4,2ES45.35E4)') 'MASS ',i,masses(i)
      do j=1,n
        write(*,'(A,2I4,ES45.35E4)') 'S ',i,j,s(i,j)
      enddo
    enddo
    call tirust_clear_cache(setup)
    if (setup/=TIRUST_SUCCESS) stop 8
    call tirust_register_plan(n,rank,setup)
    if (setup/=TIRUST_SUCCESS) stop 9
    call tirust_qp_cache_statistics(counts,setup)
    if (setup/=TIRUST_SUCCESS) stop 10
    write(*,'(A,6I22)') 'CACHE_BEFORE ',counts
    call tirust_tensor_evaluate(n,rank,p,s,masses,mu2,raw,ts, &
         pole_mode=0,compute_finite=.true.,check_pass=.false.,cache_enabled=.false.)
    write(*,'(A,I5)') 'TIR_STATUS ',ts
    if (ts/=TIRUST_SUCCESS) then
      call tirust_error_message(error)
      write(*,'(A,A)') 'TIR_ERROR ',trim(error)
    else
      do j=1,4
        do i=1,nc
          if (.not.ieee_is_finite(real(raw(i,j),q)).or..not.ieee_is_finite(aimag(raw(i,j)))) stop 11
          write(*,'(A,2I4,2ES45.35E4)') 'TIR_RAW ',i,j,raw(i,j)
        enddo
      enddo
    endif
    call tirust_qp_cache_statistics(counts,setup)
    if (setup/=TIRUST_SUCCESS.or.counts(3)/=0.or.counts(4)/=0) stop 12
    write(*,'(A,6I22)') 'CACHE_AFTER ',counts

    do i=0,nc-1
      degree=0
      do while (i>=cumulative(degree))
        degree=degree+1
      enddo
      if (degree>rank.or.mapping(i)>nc) stop 13
      unit=cmplx(0.0_q,0.0_q,kind=q)
      ! TIRust MadLoop moments already have this sign. Apply it only to NRS.
      unit(i+1)=cmplx(real((-1)**degree,q),0.0_q,kind=q)
      call ninjars_tensor_evaluate(unit,n,rank,s,p,masses,mu2,result,r1,ns,pole_mode=0)
      write(*,'(A,4I6)') 'NRS_UNIT ',i,degree,mapping(i),ns
      if (ns/=NINJARS_SUCCESS) then
        call ninjars_error_message(error)
        write(*,'(A,A)') 'NRS_ERROR ',trim(error)
      else
        ! The finite result already includes R1; never add it a second time.
        do k=0,2
          if (.not.ieee_is_finite(real(result(k),q)).or..not.ieee_is_finite(aimag(result(k)))) stop 14
          write(*,'(A,2I4,2ES45.35E4)') 'NRS_RESULT ',i,k,result(k)
        enddo
        write(*,'(A,I4,2ES45.35E4)') 'NRS_R1 ',i,r1
        if (ts==TIRUST_SUCCESS) then
          mapped=[raw(mapping(i),4),raw(mapping(i),1)+raw(mapping(i),2),raw(mapping(i),3)]
          do k=0,2
            write(*,'(A,2I4,4ES45.35E4)') 'COMPARE ',i,k,mapped(k),result(k)
          enddo
        endif
      endif
      flush(6)
    enddo
    write(*,'(A,2I6)') 'END ',point,group
    flush(6)
    deallocate(p,s,masses,raw,unit)
  enddo
  close(73)
  call tirust_finalize(setup)
  if (setup/=0) stop 15
  call ninjars_finalize(setup)
  if (setup/=0) stop 16
end program ew_tensor_replay
