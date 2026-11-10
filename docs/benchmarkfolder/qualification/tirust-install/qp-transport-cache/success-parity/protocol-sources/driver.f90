! Raw public Fortran API diagnostic; no generated amplitude or timing claim.
program installed_success_parity
  use mtirust
  use, intrinsic :: iso_c_binding, only: c_int64_t
  implicit none
  real(KI_TIR) :: p(0:3,3),s(3,3),delta(0:3),mu,mr,mi
  real(KI_QTIR) :: qp(0:3,3),qs(3,3),qdelta(0:3),qmu,last_mass
  complex(KI_TIR) :: m(3),r(15,4)
  complex(KI_QTIR) :: qm(3),qr(15,4)
  integer(c_int64_t) :: budget,before(15),after(15),words(4)
  integer :: number,id,precision,n,rank,mode,finite,check,cache,control,lastbit
  integer :: nc,status,eval_status,i,j,k,raw_generation,before_generation,after_generation
  character(len=4096) :: input_path,output_path

  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  if (len_trim(input_path)==0 .or. len_trim(output_path)==0) stop 90
  if (storage_size(0.0_KI_TIR)/=64 .or. storage_size(0.0_KI_QTIR)/=128) stop 91
  open(71,file=trim(input_path),status='old',action='read')
  open(72,file=trim(output_path),status='new',action='write')
  read(71,*) number
  do k=1,number
    read(71,*) id,precision,n,rank,mode,finite,check,cache,control,budget,lastbit
    if (id/=k .or. n<1 .or. n>3 .or. rank<0 .or. rank>2) stop 92
    if (precision/=64 .and. precision/=128) stop 93
    nc=tirust_coefficient_count(rank)
    read(71,*) mu
    p=0
    s=0
    m=0
    do i=1,n
      read(71,*) p(:,i)
    enddo
    do i=1,n
      read(71,*) mr,mi
      m(i)=cmplx(mr,mi,kind=KI_TIR)
    enddo
    do i=1,n
      do j=1,n
        delta=p(:,i)-p(:,j)
        s(i,j)=delta(0)**2-sum(delta(1:3)**2)
      enddo
    enddo
    ! Frozen decimal fixture values enter through binary64, as in the earlier
    ! generated-adapter diagnostics. The dedicated tadpole overrides this.
    qp=real(p,kind=KI_QTIR)
    qs=0
    do i=1,n
      do j=1,n
        qdelta=qp(:,i)-qp(:,j)
        qs(i,j)=qdelta(0)**2-sum(qdelta(1:3)**2)
      enddo
    enddo
    qm=cmplx(real(m,kind=KI_QTIR),real(aimag(m),kind=KI_QTIR),kind=KI_QTIR)
    qmu=real(mu,kind=KI_QTIR)
    if (lastbit==1) then
      if (precision/=128 .or. n/=1 .or. rank/=0) stop 94
      last_mass=4.0_KI_QTIR*(1.0_KI_QTIR+2.0_KI_QTIR**(-112))
      qm(1)=cmplx(last_mass,-0.0_KI_QTIR,kind=KI_QTIR)
      qmu=last_mass
    endif
    raw_generation=tirust_context_generation()
    select case(control)
    case(0)
      status=TIRUST_SUCCESS
    case(1)
      call tirust_clear_cache(status)
    case(2)
      call tirust_finalize(status)
    case(3)
      call tirust_set_cache_budget(budget,status)
    case default
      stop 95
    end select
    if (status/=TIRUST_SUCCESS) stop 96
    before_generation=tirust_context_generation()
    before=0
    if (before_generation/=-1) call snapshot(before)
    r=0
    qr=0
    if (precision==64) then
      call tirust_tensor_evaluate(n,rank,p(:,1:n),s(1:n,1:n),m(1:n),mu,r(1:nc,:),eval_status, &
        pole_mode=mode,compute_finite=finite==1,check_pass=check==1,cache_enabled=cache==1)
    else
      call tirust_tensor_evaluate(n,rank,qp(:,1:n),qs(1:n,1:n),qm(1:n),qmu,qr(1:nc,:),eval_status, &
        pole_mode=mode,compute_finite=finite==1,check_pass=check==1,cache_enabled=cache==1)
    endif
    after_generation=tirust_context_generation()
    call snapshot(after)
    write(72,'(A,15(1X,I0))') 'F',id,precision,n,rank,mode,finite,check,cache,control,budget,lastbit, &
      eval_status,raw_generation,before_generation,after_generation
    write(72,'(A,16(1X,I0))') 'B',id,before
    write(72,'(A,16(1X,I0))') 'A',id,after
    do i=1,nc
      do j=1,4
        if (precision==64) then
          words(1:2)=transfer(r(i,j),words(1:2))
          write(72,'(A,3(1X,I0),2(1X,Z16.16))') 'C',id,i,j,words(1:2)
        else
          words=transfer(qr(i,j),words)
          write(72,'(A,3(1X,I0),4(1X,Z16.16))') 'C',id,i,j,words
        endif
      enddo
    enddo
    flush(72)
    if (lastbit==1 .and. eval_status==TIRUST_SUCCESS) then
      if (real(qr(1,1),kind=KI_QTIR)/=last_mass .or. real(qr(1,1),kind=KI_QTIR)==4.0_KI_QTIR) stop 97
      if (abs(aimag(qr(1,1)))>1e-32_KI_QTIR .or. any(abs(qr(1,2:3))>1e-32_KI_QTIR)) stop 98
      if (abs(qr(1,4)-last_mass)>1e-31_KI_QTIR) stop 99
    endif
  enddo
  call tirust_finalize(status)
  if (status/=TIRUST_SUCCESS .or. tirust_context_generation()/=-1) stop 100
  close(71)
  close(72)
contains
  subroutine snapshot(values)
    integer(c_int64_t),intent(out) :: values(15)
    integer :: local_status
    call tirust_cache_statistics(values(1:9),local_status)
    if (local_status/=TIRUST_SUCCESS) stop 101
    call tirust_qp_cache_statistics(values(10:15),local_status)
    if (local_status/=TIRUST_SUCCESS) stop 102
  end subroutine snapshot
end program installed_success_parity
