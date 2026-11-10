! Additive diagnostic records, emitted outside measured amplitude timing.
module qp_tier_records
  use mtirust
  use iso_c_binding, only: c_int64_t
  implicit none
contains
  subroutine write_qp_tier_snapshot(unit,point,phase)
    integer, intent(in) :: unit,point
    character(len=*), intent(in) :: phase
    integer(c_int64_t) :: words(11)
    integer :: generation,status,available,i
    character(len=512) :: prior,error
    if (phase /= 'BEFORE' .and. phase /= 'AFTER') error stop 'Invalid telemetry phase'
    available=merge(1,0,tirust_qp_precision_available())
    call tirust_qp_precision_snapshot(words,generation,status,prior,error)
    write(unit,'(A,1X,I0,1X,A,4(1X,I0),11(1X,Z16.16))') &
      'TIRUST_QP_TIER_V1',point,phase,available,generation,status,11,words
    write(unit,'(A,1X,I0,1X,A,1X,1024Z2.2)') 'TIRUST_QP_ERROR_V1',point,phase, &
      (iachar(prior(i:i)),i=1,512),(iachar(error(i:i)),i=1,512)
  end subroutine
end module
