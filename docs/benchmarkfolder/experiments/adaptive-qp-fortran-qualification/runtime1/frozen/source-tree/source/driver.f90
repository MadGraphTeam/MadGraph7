! Standalone raw tensor/transport qualification, never a physical amplitude.
program adaptive_fortran_transport
  use mtirust
  use qp_tier_records
  use iso_c_binding
  implicit none
  real(KI_QTIR) :: p(0:3,4),s(4,4),scale_value,mre(8),psmall(0:3,1),ssmall(1,1),scale_small
  complex(KI_QTIR) :: masses(4),small_mass(1),coefs(35,4),small_coefs(1,4)
  integer :: status,action,n,rank,poles,finite,check,cache,i,j,arm,finish_status,gen_before,gen_after
  integer :: health
  character(len=16) :: argument
  character(len=512) :: error_before,error_after
  integer(c_int64_t) :: words(2),cache_words(6)
  interface
    subroutine native_observer_open() bind(C)
    end subroutine
    subroutine native_observer_begin(action) bind(C)
      import
      integer(c_int),value :: action
    end subroutine
    subroutine native_observer_flush() bind(C)
    end subroutine
    subroutine native_observer_close() bind(C)
    end subroutine
    integer(c_int) function native_observer_health() bind(C)
      import
    end function
  end interface
  call get_command_argument(1,argument)
  if(argument=='legacy') then
    arm=0
  else if(argument=='adaptive') then
    arm=1
  else
    stop 10
  endif
  open(71,file='records.txt',status='new')
  open(72,file='telemetry.txt',status='new')
  call native_observer_open()
  if(digits(scale_value)/=113.or.radix(scale_value)/=2) stop 11
  write(71,*) 'HEADER',1,arm,digits(scale_value),merge(1,0,tirust_qp_precision_available())
  call read_words('group10-p.words',p,16)
  call read_words('group10-s.words',s,16)
  call read_words('group10-m.words',mre,8)
  do i=1,4
    masses(i)=cmplx(mre(2*i-1),mre(2*i),kind=KI_QTIR)
  enddo
  call read_scale('group10-scale.words',scale_value)
  psmall=0;ssmall=0;scale_small=4
  words=[0_c_int64_t,int(z'8000000000000000',c_int64_t)]
  small_mass(1)=cmplx(real(4,KI_QTIR),transfer(words,scale_small),kind=KI_QTIR)
  call tirust_register_plan(1,0,status)
  write(71,*) 'REGISTER',1,0,status,tirust_context_generation()
  if(status/=0) stop 12
  if(arm==1) then
    call tirust_register_plan(4,3,status)
    write(71,*) 'REGISTER',4,3,status,tirust_context_generation()
    if(status/=0) stop 13
  endif
  do action=1,merge(10,1,arm==1)
    n=1;rank=0;poles=0;finite=1;check=0;cache=1
    if(action>=4.and.action<=9) then
      n=4;rank=3
    endif
    if(action>=7.and.action<=9) finite=0
    if(action==3.or.action==6.or.action==8) check=1
    if(action==10) poles=99
    write(71,*) 'ACTION',action,n,rank,poles,finite,check,cache
    call write_qp_tier_snapshot(72,action,'BEFORE')
    call native_observer_begin(action)
    if(n==1) then
      call tirust_tensor_evaluate(n,rank,psmall,ssmall,small_mass,scale_small,small_coefs,status, &
        pole_mode=poles,compute_finite=finite==1,check_pass=check==1,cache_enabled=cache==1)
      coefs=0;coefs(1,:)=small_coefs(1,:)
    else
      call tirust_tensor_evaluate(n,rank,p,s,masses,scale_value,coefs,status, &
        pole_mode=poles,compute_finite=finite==1,check_pass=check==1,cache_enabled=cache==1)
    endif
    ! Consume numerical status/error before the additive getter clears LAST_ERROR.
    call tirust_error_message(error_before)
    write(71,*) 'STATUS',status
    call write_qp_tier_snapshot(72,action,'AFTER')
    call tirust_error_message(error_after)
    write(71,'(A,1X,1024Z2.2)') 'ERRORS',(iachar(error_before(i:i)),i=1,512),(iachar(error_after(i:i)),i=1,512)
    do j=1,4
      do i=1,tirust_coefficient_count(rank)
        write(71,'(A,2(1X,I0),4(1X,Z16.16))') 'VALUE',i,j,transfer(coefs(i,j),[0_c_int64_t])
      enddo
    enddo
    call native_observer_flush()
    health=native_observer_health()
    call tirust_qp_cache_statistics(cache_words,finish_status)
    write(71,*) 'CACHE',finish_status,cache_words
    write(71,*) 'END',action,health
    flush(71);flush(72)
    if(health/=0.or.finish_status/=0) stop 14
  enddo
  ! Cache/budget changes must preserve precision counters. These are separate
  ! metadata probes, not additional integral calls or phase-space points.
  call write_qp_tier_snapshot(72,101,'BEFORE')
  call tirust_clear_cache(status)
  call tirust_set_cache_budget(0_c_int64_t,finish_status)
  call write_qp_tier_snapshot(72,101,'AFTER')
  write(71,*) 'CACHE_BOUNDARY',status,finish_status
  gen_before=tirust_context_generation()
  call write_qp_tier_snapshot(72,102,'BEFORE')
  call tirust_finalize(status)
  gen_after=tirust_context_generation()
  write(71,*) 'FINALIZE',status,gen_before,gen_after
  call tirust_register_plan(1,0,status)
  call write_qp_tier_snapshot(72,102,'AFTER')
  write(71,*) 'RECREATE',status,tirust_context_generation()
  call tirust_finalize(finish_status)
  call native_observer_close()
  write(71,*) 'DONE',finish_status,native_observer_health(),tirust_context_generation()
  close(71);close(72)
contains
  subroutine read_words(name,values,expected)
    character(len=*),intent(in) :: name
    integer,intent(in) :: expected
    real(KI_QTIR),intent(out) :: values(expected)
    integer :: k,count,ios
    integer(c_int64_t) :: raw(2)
    open(73,file=name,status='old')
    read(73,*) count
    if(count/=expected) stop 16
    do k=1,count
      read(73,'(Z16,1X,Z16)') raw
      values(k)=transfer(raw,scale_value)
    enddo
    read(73,*,iostat=ios) k
    if(ios>=0) stop 15
    close(73)
  end subroutine
  subroutine read_scale(name,value)
    character(len=*),intent(in) :: name
    real(KI_QTIR),intent(out) :: value
    real(KI_QTIR) :: buffer(1)
    call read_words(name,buffer,1)
    value=buffer(1)
  end subroutine
end program
