module fixture_recording
  use iso_fortran_env, only: int64, real64, real128
  use ieee_exceptions
  implicit none
  integer :: log_unit, call_count=0, legacy_count=0, qp_count=0, observer_changes=0
  integer :: synthetic_status(32)=0
  complex(real64) :: synthetic_result(3,32)
  type(ieee_flag_type), parameter :: tracked(5) = &
       [ieee_invalid,ieee_divide_by_zero,ieee_overflow,ieee_underflow,ieee_inexact]
contains
  subroutine observer_begin(saved, before)
    type(ieee_status_type), intent(out) :: saved
    logical, intent(out) :: before(5)
    call ieee_get_status(saved)
    call ieee_get_flag(tracked,before)
  end subroutine
  subroutine observer_end(saved, before)
    type(ieee_status_type), intent(in) :: saved
    logical, intent(in) :: before(5)
    logical :: after(5)
    call ieee_get_flag(tracked,after)
    if(any(before.neqv.after)) observer_changes=observer_changes+1
    call ieee_set_status(saved)
  end subroutine
  subroutine emit_flags(label,flags)
    character(*), intent(in) :: label
    logical, intent(in) :: flags(5)
    type(ieee_status_type) :: saved
    logical :: before(5)
    call observer_begin(saved,before)
    write(log_unit,'(A,1X,A,1X,5L1)') 'FLAGS',label,flags
    call observer_end(saved,before)
  end subroutine
  subroutine emit_dp_real(label,value)
    character(*), intent(in) :: label
    real(real64), intent(in) :: value
    integer(int64) :: word
    type(ieee_status_type) :: saved
    logical :: before(5)
    call observer_begin(saved,before)
    word=transfer(value,word)
    write(log_unit,'(A,1X,A,2(1X,Z16.16))') 'MODEL_DP',label,word,0_int64
    call observer_end(saved,before)
  end subroutine
  subroutine emit_dp_complex(label,value)
    character(*), intent(in) :: label
    complex(real64), intent(in) :: value
    integer(int64) :: words(2)
    type(ieee_status_type) :: saved
    logical :: before(5)
    call observer_begin(saved,before)
    words=transfer(value,words)
    write(log_unit,'(A,1X,A,2(1X,Z16.16))') 'MODEL_DP',label,words
    call observer_end(saved,before)
  end subroutine
  subroutine emit_mp_real(label,value)
    character(*), intent(in) :: label
    real(real128), intent(in) :: value
    integer(int64) :: words(2)
    type(ieee_status_type) :: saved
    logical :: before(5)
    call observer_begin(saved,before)
    words=transfer(value,words)
    write(log_unit,'(A,1X,A,4(1X,Z16.16))') 'MODEL_MP',label,words,0_int64,0_int64
    call observer_end(saved,before)
  end subroutine
  subroutine emit_mp_complex(label,value)
    character(*), intent(in) :: label
    complex(real128), intent(in) :: value
    integer(int64) :: words(4)
    type(ieee_status_type) :: saved
    logical :: before(5)
    call observer_begin(saved,before)
    words=transfer(value,words)
    write(log_unit,'(A,1X,A,4(1X,Z16.16))') 'MODEL_MP',label,words
    call observer_end(saved,before)
  end subroutine
  subroutine record_dp(n,npair,nmother,pairing,mothers,rank,sqso,group,masses)
    integer,intent(in) :: n,npair,nmother,pairing(npair),mothers(nmother),rank,sqso,group
    complex(real64),intent(in) :: masses(n)
    call record_base('LEGACY',n,npair,nmother,pairing,mothers,rank,sqso,group,masses)
  end subroutine
  subroutine record_qp(n,npair,nmother,pairing,mothers,rank,sqso,group,masses,mp_masses)
    integer,intent(in) :: n,npair,nmother,pairing(npair),mothers(nmother),rank,sqso,group
    complex(real64),intent(in) :: masses(n)
    complex(real128),intent(in) :: mp_masses(n)
    call record_base('QPMASS',n,npair,nmother,pairing,mothers,rank,sqso,group,masses,mp_masses)
  end subroutine
  subroutine record_base(kind,n,npair,nmother,pairing,mothers,rank,sqso,group,masses,mp_masses)
    character(*),intent(in) :: kind
    integer,intent(in) :: n,npair,nmother,pairing(npair),mothers(nmother),rank,sqso,group
    complex(real64),intent(in) :: masses(n)
    complex(real128),intent(in),optional :: mp_masses(n)
    integer :: i
    integer(int64) :: dpwords(2),mpwords(4)
    type(ieee_status_type) :: saved
    logical :: before(5)
    call observer_begin(saved,before)
    call_count=call_count+1
    if(call_count>32) error stop 71
    if(present(mp_masses)) then
      qp_count=qp_count+1
    else
      legacy_count=legacy_count+1
    endif
    write(log_unit,'(A,1X,I0,1X,A,6(1X,I0))') 'CALL',call_count,kind,n,npair,nmother,rank,sqso,group
    write(log_unit,'(A,1X,I0,*(1X,I0))') 'PAIRING',call_count,pairing
    write(log_unit,'(A,1X,I0,*(1X,I0))') 'MOTHERS',call_count,mothers
    do i=1,n
      dpwords=transfer(masses(i),dpwords)
      write(log_unit,'(A,2(1X,I0),2(1X,Z16.16))') 'ARG_DP',call_count,i,dpwords
      if(present(mp_masses)) then
        mpwords=transfer(mp_masses(i),mpwords)
        write(log_unit,'(A,2(1X,I0),4(1X,Z16.16))') 'ARG_MP',call_count,i,mpwords
      endif
    enddo
    ! Recording-stub outputs only: these are not amplitudes or reducer statuses.
    synthetic_result(1,call_count)=masses(1)
    synthetic_result(2,call_count)=masses(n)
    synthetic_result(3,call_count)=cmplx(rank,group,kind=real64)
    synthetic_status(call_count)=0
    do i=1,3
      dpwords=transfer(synthetic_result(i,call_count),dpwords)
      write(log_unit,'(A,2(1X,I0),2(1X,Z16.16))') 'RESULT',call_count,i,dpwords
    enddo
    write(log_unit,'(A,2(1X,I0))') 'STATUS',call_count,synthetic_status(call_count)
    call observer_end(saved,before)
  end subroutine
end module fixture_recording
