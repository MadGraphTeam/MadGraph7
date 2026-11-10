! Private matched observer: integer-only bit records, no numerical substitution.
module ew_mass_observer
  use iso_fortran_env, only: int64
  use, intrinsic :: ieee_exceptions
  use mtirust, only: tirust_error_message
  use mninjars, only: ninjars_error_message
  implicit none
  private
  public :: obs_begin, obs_end, obs_close, obs_group, obs_status, obs_boolean
  public :: obs_r8, obs_r16, obs_c16, obs_c32, obs_native, obs_active
  integer, parameter :: unit=916, record_limit=1000000
  integer :: point=0, ordinal=0, group=0, records=0, dropped=0, flag_changes=0
  logical :: active=.false., opened=.false.
contains
  logical function obs_active()
    obs_active=active
  end function
  subroutine enter(saved,flags)
    type(ieee_status_type), intent(out) :: saved
    logical, intent(out) :: flags(size(ieee_all))
    call ieee_get_status(saved)
    call ieee_get_flag(ieee_all,flags)
  end subroutine
  subroutine leave(saved,flags)
    type(ieee_status_type), intent(in) :: saved
    logical, intent(in) :: flags(size(ieee_all))
    logical :: after(size(ieee_all))
    call ieee_get_flag(ieee_all,after)
    if(any(flags.neqv.after)) then
      flag_changes=flag_changes+1
      write(unit,'(A,3(1X,I0),1X,*(L1))') 'OBSERVER_FLAG_CHANGE',point,ordinal,group,flags,after
    end if
    ! Restore only observer-local status; never span a reducer/model operation.
    call ieee_set_status(saved)
  end subroutine
  logical function reserve()
    records=records+1
    reserve=records<=record_limit
    if(.not.reserve) dropped=dropped+1
  end function
  subroutine obs_begin(k)
    integer, intent(in) :: k
    type(ieee_status_type) :: saved
    logical :: flags(size(ieee_all))
    call enter(saved,flags)
    if(.not.opened) then
      open(unit,file='mass-observer.hex',status='new',action='write')
      opened=.true.
      write(unit,'(A)') 'FORMAT EW_QP_MASS_HEX_V1 storage-order-int64-words'
    end if
    point=k;ordinal=0;group=0;active=.true.
    write(unit,'(A,1X,I0,1X,*(L1))') 'BEGIN',point,flags
    call leave(saved,flags)
  end subroutine
  subroutine obs_end()
    type(ieee_status_type) :: saved
    logical :: flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    write(unit,'(A,3(1X,I0),1X,*(L1))') 'END',point,ordinal,group,flags
    flush(unit)
    active=.false.
    call leave(saved,flags)
  end subroutine
  subroutine obs_close()
    type(ieee_status_type) :: saved
    logical :: flags(size(ieee_all))
    if(.not.opened)return
    call enter(saved,flags)
    write(unit,'(A,4(1X,I0))') 'FOOTER',records,dropped,flag_changes,record_limit
    flush(unit)
    call leave(saved,flags)
    close(unit)
    call ieee_set_status(saved)
    opened=.false.
  end subroutine
  subroutine obs_group(id,sqso,mode,n,rank)
    integer,intent(in)::id,sqso,mode,n,rank
    type(ieee_status_type) :: saved
    logical :: flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    ordinal=ordinal+1;group=id
    if(reserve())write(unit,'(A,7(1X,I0),1X,*(L1))') 'GROUP',point,ordinal,id,sqso,mode,n,rank,flags
    call leave(saved,flags)
  end subroutine
  subroutine obs_status(label,value)
    character(*),intent(in)::label
    integer,intent(in)::value
    type(ieee_status_type) :: saved
    logical :: flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    if(reserve())write(unit,'(A,1X,A,1X,I0,1X,*(L1))') 'INTEGER',label,value,flags
    call leave(saved,flags)
  end subroutine
  subroutine obs_boolean(label,value)
    character(*),intent(in)::label
    logical,intent(in)::value
    type(ieee_status_type) :: saved
    logical :: flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    if(reserve())write(unit,'(A,1X,A,1X,L1,1X,*(L1))') 'LOGICAL',label,value,flags
    call leave(saved,flags)
  end subroutine
  subroutine obs_native(backend,status)
    integer,intent(in)::backend,status
    character(2048)::message
    integer::i
    type(ieee_status_type) :: saved
    logical :: flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    if(reserve())write(unit,'(A,2(1X,I0),1X,*(L1))') 'NATIVE',backend,status,flags
    if(status/=0)then
      message=''
      if(backend==9)call tirust_error_message(message)
      if(backend==8)call ninjars_error_message(message)
      if(reserve())write(unit,'(A,*(1X,Z2.2))') 'ERROR_BYTES',(iachar(message(i:i)),i=1,len_trim(message))
    end if
    call leave(saved,flags)
  end subroutine
  subroutine obs_r8(label,n,values)
    character(*),intent(in)::label
    integer,intent(in)::n
    real*8,intent(in)::values(n)
    integer::i
    integer(int64)::bits(1)
    type(ieee_status_type)::saved
    logical::flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    do i=1,n
      bits=transfer(values(i),bits)
      if(reserve())write(unit,'(A,1X,A,1X,I0,1(1X,Z16.16))') 'R8',label,i,bits
    end do
    call leave(saved,flags)
  end subroutine
  subroutine obs_r16(label,n,values)
    character(*),intent(in)::label
    integer,intent(in)::n
    real*16,intent(in)::values(n)
    integer::i
    integer(int64)::bits(2)
    type(ieee_status_type)::saved
    logical::flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    do i=1,n
      bits=transfer(values(i),bits)
      if(reserve())write(unit,'(A,1X,A,1X,I0,2(1X,Z16.16))') 'R16',label,i,bits
    end do
    call leave(saved,flags)
  end subroutine
  subroutine obs_c16(label,n,values)
    character(*),intent(in)::label
    integer,intent(in)::n
    complex*16,intent(in)::values(n)
    integer::i
    integer(int64)::bits(2)
    type(ieee_status_type)::saved
    logical::flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    do i=1,n
      bits=transfer(values(i),bits)
      if(reserve())write(unit,'(A,1X,A,1X,I0,2(1X,Z16.16))') 'C16',label,i,bits
    end do
    call leave(saved,flags)
  end subroutine
  subroutine obs_c32(label,n,values)
    character(*),intent(in)::label
    integer,intent(in)::n
    complex*32,intent(in)::values(n)
    integer::i
    integer(int64)::bits(4)
    type(ieee_status_type)::saved
    logical::flags(size(ieee_all))
    if(.not.active)return
    call enter(saved,flags)
    do i=1,n
      bits=transfer(values(i),bits)
      if(reserve())write(unit,'(A,1X,A,1X,I0,4(1X,Z16.16))') 'C32',label,i,bits
    end do
    call leave(saved,flags)
  end subroutine
end module
