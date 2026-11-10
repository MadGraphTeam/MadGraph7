program telemetry_mock_transport
  use mtirust
  use qp_tier_records
  use iso_c_binding
  implicit none
  integer(c_int64_t) :: expected(11),words(11),short(10)
  integer :: status,generation,old_generation,calls
  character(len=512) :: prior,query,message
  character(len=8) :: variant
  interface
    subroutine mock_words(words) bind(C)
      import
      integer(c_int64_t),intent(in) :: words(*)
    end subroutine
    subroutine mock_mode(value) bind(C)
      import
      integer(c_int),value :: value
    end subroutine
    subroutine mock_error() bind(C)
    end subroutine
    integer(c_int) function mock_getters() bind(C)
      import
    end function
    integer(c_int) function mock_creates() bind(C)
      import
    end function
    integer(c_int) function mock_tensors() bind(C)
      import
    end function
    integer(c_int) function raw_query(handle,words,count) bind(C,name='ml_tirust_private_qp_precision_snapshot')
      import
      type(c_ptr),value :: handle
      integer(c_int64_t),intent(out) :: words(*)
      integer(c_size_t),value :: count
    end function
  end interface
  call get_command_argument(1,variant)
  call tirust_qp_precision_snapshot(words,generation,status,prior,query)
  if(generation/=-1.or.mock_creates()/=0.or.any(words/=0)) stop 1
  if(variant=='absent') then
    if(tirust_qp_precision_available().or.status/=TIRUST_UNSUPPORTED) stop 2
    call write_qp_tier_snapshot(6,1,'BEFORE')
    call tirust_register_plan(1,0,status)
    call write_qp_tier_snapshot(6,1,'AFTER')
    if(mock_getters()/=0.or.mock_tensors()/=0) stop 3
    call tirust_finalize(status)
    stop
  endif
  if(.not.tirust_qp_precision_available().or.status/=TIRUST_ERROR) stop 4
  call tirust_register_plan(1,0,status)
  if(status/=0.or.mock_creates()/=1) stop 5
  expected=[1_8,2_8,3_8,4_8,5_8,6_8,7_8,8_8,9_8,512_8,int(z'3FF0000000000001',8)]
  expected(1)=int(z'8000000000000001',8)
  call mock_words(expected)
  call mock_error()
  call tirust_qp_precision_snapshot(words,generation,status,prior,query)
  if(status/=0.or.any(words/=expected).or.generation/=1) stop 6
  if(prior/='retained numerical rejection'.or.query/='') stop 7
  call tirust_error_message(message)
  if(message/=prior) stop 8
  calls=mock_getters()
  call tirust_qp_precision_snapshot(short,generation,status,prior,query)
  if(status/=2.or.any(short/=0).or.mock_getters()/=calls) stop 9
  short=-1
  status=raw_query(c_null_ptr,short,10_c_size_t)
  if(status/=2.or.any(short/=-1).or.mock_getters()/=calls) stop 10
  call mock_mode(1)
  call tirust_qp_precision_snapshot(words,generation,status,prior,query)
  if(status/=2.or.any(words/=0).or.query/='mock getter failure') stop 11
  call tirust_error_message(message)
  if(message/='retained numerical rejection') stop 12
  call mock_mode(2)
  call tirust_qp_precision_snapshot(words,generation,status,prior,query)
  if(status/=2.or.any(words/=0).or.query/='Invalid adaptive QP telemetry response.') stop 13
  call mock_mode(0)
  call tirust_clear_cache(status)
  call tirust_set_cache_budget(0_c_int64_t,status)
  call tirust_qp_precision_snapshot(words,generation,status,prior,query)
  if(status/=0.or.any(words/=expected)) stop 14
  call write_qp_tier_snapshot(6,7,'BEFORE')
  expected(1:9)=expected(1:9)+[3_8,1_8,1_8,1_8,0_8,2_8,1_8,1_8,1_8]
  call mock_words(expected)
  call write_qp_tier_snapshot(6,7,'AFTER')
  call write_qp_tier_snapshot(6,8,'BEFORE')
  old_generation=generation
  call tirust_finalize(status)
  if(tirust_context_generation()/=-1) stop 15
  call tirust_register_plan(1,0,status)
  call tirust_qp_precision_snapshot(words,generation,status,prior,query)
  if(status/=0.or.generation/=old_generation+1.or.any(words/=0)) stop 16
  call write_qp_tier_snapshot(6,8,'AFTER')
  call tirust_finalize(status)
  if(mock_tensors()/=0) stop 17
end program
