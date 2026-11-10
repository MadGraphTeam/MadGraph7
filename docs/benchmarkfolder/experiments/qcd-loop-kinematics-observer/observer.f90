! Source preparation only: this observer has not been compiled or qualified.
module qcd_kinematics_observer
  use, intrinsic :: iso_fortran_env, only: int8, int64
  use, intrinsic :: ieee_exceptions, only: ieee_status_type, ieee_get_status, ieee_set_status
  implicit none
  private
  public :: qo_open, qo_begin_point, qo_begin_call, qo_put, qo_mark
  public :: qo_native_status, qo_error, qo_end_call, qo_end_point
  public :: qo_is_active, qo_scope_active, qo_get_health

  integer, parameter :: max_events=200000, max_words=2000000
  integer, parameter :: max_errors=4096, error_length=1024, tag_length=64
  integer, parameter :: metadata_length=21
  ! Metadata: kind,seq,point,call,backend,ctmode,group,sqso,n,rank,i,j,value,
  !           word_count,word_first,error_index,native_present,native_status,
  !           stable,fpe,tag_length. Unavailable stable/fpe are -1.
  type :: event_record
    integer :: meta(metadata_length)
    character(len=tag_length) :: tag
  end type
  type(event_record), save :: events(max_events)
  integer(int64), save :: payload(max_words)
  character(len=error_length), save :: errors(max_errors)
  integer, save :: error_used(max_errors), error_original(max_errors)
  integer, save :: nevent=0, nword=0, nerror=0, output_unit=0
  integer, save :: point_seq=0, point_id=0, ordinal=0, call_id=0
  integer, save :: backend_id=0, mode_id=0, group_id=0, sqso_id=0, n_id=0, rank_id=0
  integer, save :: native_value=0, stable_value=-1, fpe_value=-1
  integer(int64), save :: point_faults=0, dropped_events=0, dropped_words=0, dropped_errors=0
  integer(int64), save :: total_faults=0, io_faults=0, truncated_errors=0
  logical, save :: active=.false., in_call=.false., opened=.false., native_present=.false.

contains

  logical function qo_is_active()
    qo_is_active=active
  end function

  logical function qo_scope_active()
    qo_scope_active=active.and.in_call
  end function

  subroutine qo_get_health(faults,io_failures)
    integer(int64), intent(out) :: faults,io_failures
    faults=total_faults
    io_failures=io_faults
  end subroutine

  subroutine note_fault()
    total_faults=total_faults+1_int64
    if (active) point_faults=point_faults+1_int64
  end subroutine

  subroutine reset_call()
    in_call=.false.
    call_id=0
    backend_id=0
    mode_id=0
    group_id=0
    sqso_id=0
    n_id=0
    rank_id=0
    native_present=.false.
    native_value=0
    stable_value=-1
    fpe_value=-1
  end subroutine

  subroutine reserve_event(kind,tag,i,j,value,words,error_index,point_scope,index_out)
    integer, intent(in) :: kind,i,j,value,words,error_index
    character(len=*), intent(in) :: tag
    logical, intent(in) :: point_scope
    integer, intent(out) :: index_out
    integer :: length,k,c
    index_out=0
    if (nevent>=max_events.or.words>max_words-nword) then
      dropped_events=dropped_events+1_int64
      dropped_words=dropped_words+int(words,int64)
      call note_fault()
      return
    endif
    nevent=nevent+1
    index_out=nevent
    events(nevent)%meta=0
    events(nevent)%meta(1:3)=[kind,point_seq,point_id]
    if (.not.point_scope) then
      events(nevent)%meta(4:10)=[call_id,backend_id,mode_id,group_id,sqso_id,n_id,rank_id]
      events(nevent)%meta(17:20)=[merge(1,0,native_present),native_value,stable_value,fpe_value]
    else
      events(nevent)%meta(19:20)=-1
    endif
    events(nevent)%meta(11:16)=[i,j,value,words,merge(nword+1,0,words>0),error_index]
    length=min(len_trim(tag),tag_length)
    if (len_trim(tag)>tag_length) call note_fault()
    events(nevent)%meta(21)=length
    events(nevent)%tag=''
    if (length>0) events(nevent)%tag(1:length)=tag(1:length)
    do k=1,length
      c=iachar(events(nevent)%tag(k:k))
      if (c<32.or.c>126) then
        events(nevent)%tag(k:k)='?'
        call note_fault()
      endif
    enddo
  end subroutine

  subroutine record_fault(reason)
    integer, intent(in) :: reason
    integer :: index_out
    call note_fault()
    if (.not.active) return
    call reserve_event(9,'OBSERVER_ERROR',0,0,reason,0,0,.not.in_call,index_out)
  end subroutine

  subroutine abandon_call()
    integer :: index_out
    call record_fault(1)
    call reserve_event(10,'CALL_ABANDONED',0,0,0,0,0,.false.,index_out)
    call reset_call()
  end subroutine

  subroutine qo_open(path)
    character(len=*), intent(in) :: path
    type(ieee_status_type) :: saved
    integer :: ios,close_status,k
    integer(int8) :: byte_order(8)
    call ieee_get_status(saved)
    if (opened.or.active) then
      call record_fault(6)
    else
      open(newunit=output_unit,file=path,status='new',action='write',form='formatted',iostat=ios)
      if (ios==0) then
        opened=.true.
        byte_order=transfer(int(z'0102030405060708',int64),byte_order)
        write(output_unit,'(*(g0,1x))',iostat=ios) 'QO_HEADER',1,max_events,max_words,max_errors, &
          error_length,tag_length,metadata_length,storage_size(0_int64)
        if (ios==0) write(output_unit,'(*(g0,1x))',iostat=ios) 'QO_BYTE_ORDER',(int(byte_order(k)),k=1,8)
        if (ios/=0) then
          close(output_unit,iostat=close_status)
          opened=.false.
        endif
      endif
      if (ios/=0) then
        io_faults=io_faults+1_int64
        call note_fault()
      endif
    endif
    call ieee_set_status(saved)
  end subroutine

  subroutine qo_begin_point(seq,point)
    integer, intent(in) :: seq,point
    type(ieee_status_type) :: saved
    integer :: index_out
    call ieee_get_status(saved)
    if (active) then
      call record_fault(5)
    else
      active=.true.
      point_seq=seq
      point_id=point
      ordinal=0
      nevent=0
      nword=0
      nerror=0
      point_faults=0_int64
      dropped_events=0_int64
      dropped_words=0_int64
      dropped_errors=0_int64
      truncated_errors=0_int64
      call reset_call()
      if (.not.opened) call record_fault(7)
      call reserve_event(1,'POINT_BEGIN',0,0,0,0,0,.true.,index_out)
    endif
    call ieee_set_status(saved)
  end subroutine

  subroutine qo_begin_call(backend,ctmode,group,sqso,n,rank)
    integer, intent(in) :: backend,ctmode,group,sqso,n,rank
    type(ieee_status_type) :: saved
    integer :: index_out
    call ieee_get_status(saved)
    if (active) then
      if (in_call) call abandon_call()
      ordinal=ordinal+1
      call_id=ordinal
      backend_id=backend
      mode_id=ctmode
      group_id=group
      sqso_id=sqso
      n_id=n
      rank_id=rank
      native_present=.false.
      native_value=0
      stable_value=-1
      fpe_value=-1
      in_call=.true.
      call reserve_event(3,'CALL_BEGIN',0,0,0,0,0,.false.,index_out)
    endif
    call ieee_set_status(saved)
  end subroutine

  subroutine qo_put(tag,i,j,words)
    character(len=*), intent(in) :: tag
    integer, intent(in) :: i,j
    integer(int64), intent(in) :: words(:)
    type(ieee_status_type) :: saved
    integer :: index_out,count
    logical :: point_scope
    call ieee_get_status(saved)
    if (active) then
      point_scope=index(tag,'POINT_')==1
      if (.not.in_call.and..not.point_scope) then
        call record_fault(2)
      else
        count=size(words)
        call reserve_event(4,tag,i,j,0,count,0,point_scope,index_out)
        if (index_out/=0) then
          if (count>0) payload(nword+1:nword+count)=words
          nword=nword+count
        endif
      endif
    endif
    call ieee_set_status(saved)
  end subroutine

  subroutine qo_mark(tag,i,j,value)
    character(len=*), intent(in) :: tag
    integer, intent(in) :: i,j,value
    type(ieee_status_type) :: saved
    integer :: index_out
    logical :: point_scope
    call ieee_get_status(saved)
    if (active) then
      point_scope=index(tag,'POINT_')==1
      if (.not.in_call.and..not.point_scope) then
        call record_fault(2)
      else
        call reserve_event(5,tag,i,j,value,0,0,point_scope,index_out)
      endif
    endif
    call ieee_set_status(saved)
  end subroutine

  subroutine qo_native_status(status)
    integer, intent(in) :: status
    type(ieee_status_type) :: saved
    integer :: index_out
    call ieee_get_status(saved)
    if (active) then
      if (.not.in_call) then
        call record_fault(2)
      else
        if (native_present) call record_fault(3)
        native_present=.true.
        native_value=status
        call reserve_event(6,'NATIVE_STATUS',0,0,status,0,0,.false.,index_out)
      endif
    endif
    call ieee_set_status(saved)
  end subroutine

  subroutine qo_error(message)
    character(len=*), intent(in) :: message
    type(ieee_status_type) :: saved
    integer :: index_out,length
    call ieee_get_status(saved)
    if (active) then
      if (.not.in_call) then
        call record_fault(2)
      elseif (nerror>=max_errors) then
        dropped_errors=dropped_errors+1_int64
        call note_fault()
      else
        length=len_trim(message)
        call reserve_event(7,'BACKEND_ERROR',0,0,length,0,nerror+1,.false.,index_out)
        if (index_out/=0) then
          nerror=nerror+1
          error_original(nerror)=length
          error_used(nerror)=min(length,error_length)
          errors(nerror)=''
          if (length>0) errors(nerror)(1:error_used(nerror))=message(1:error_used(nerror))
          if (length>error_length) then
            truncated_errors=truncated_errors+1_int64
            call note_fault()
          endif
        else
          dropped_errors=dropped_errors+1_int64
        endif
      endif
    endif
    call ieee_set_status(saved)
  end subroutine

  subroutine qo_end_call(stable,fpe)
    logical, intent(in) :: stable,fpe
    type(ieee_status_type) :: saved
    integer :: index_out
    call ieee_get_status(saved)
    if (active) then
      if (.not.in_call) then
        call record_fault(2)
      else
        stable_value=merge(1,0,stable)
        fpe_value=merge(1,0,fpe)
        call reserve_event(8,'CALL_END',0,0,0,0,0,.false.,index_out)
        call reset_call()
      endif
    endif
    call ieee_set_status(saved)
  end subroutine

  subroutine qo_end_point(code)
    integer, intent(in) :: code
    type(ieee_status_type) :: saved
    integer :: index_out,ios,i,j,first,count
    call ieee_get_status(saved)
    if (active) then
      if (in_call) call abandon_call()
      call reserve_event(2,'POINT_END',0,0,code,0,0,.true.,index_out)
      ios=0
      if (opened) then
        write(output_unit,'(*(g0,1x))',iostat=ios) 'QO_POINT',point_seq,point_id,ordinal,nevent,nword,nerror, &
          point_faults,dropped_events,dropped_words,dropped_errors,truncated_errors,total_faults,io_faults
        do i=1,nevent
          if (ios/=0) exit
          write(output_unit,'(*(g0,1x))',iostat=ios) 'QO_EVENT',i,events(i)%meta,trim(events(i)%tag)
          if (ios/=0) exit
          count=events(i)%meta(14)
          first=events(i)%meta(15)
          if (count>0) write(output_unit,'(a,1x,i0,1x,i0,*(1x,z16.16))',iostat=ios) &
            'QO_WORDS',i,count,payload(first:first+count-1)
        enddo
        do i=1,nerror
          if (ios/=0) exit
          write(output_unit,'(*(g0,1x))',iostat=ios) 'QO_ERROR',i,error_used(i),error_original(i)
          if (ios/=0) exit
          write(output_unit,'(*(g0,1x))',iostat=ios) 'QO_ERROR_BYTES',i, &
            (iachar(errors(i)(j:j)),j=1,error_used(i))
        enddo
        if (ios==0) write(output_unit,'(*(g0,1x))',iostat=ios) 'QO_POINT_END',point_seq,point_id,code, &
          nevent,nword,nerror,point_faults,dropped_events,dropped_words,dropped_errors,truncated_errors, &
          total_faults,io_faults
        if (ios==0) flush(output_unit,iostat=ios)
        if (ios/=0) then
          io_faults=io_faults+1_int64
          call note_fault()
        endif
      else
        io_faults=io_faults+1_int64
        call note_fault()
      endif
      active=.false.
      call reset_call()
    endif
    call ieee_set_status(saved)
  end subroutine
end module qcd_kinematics_observer
