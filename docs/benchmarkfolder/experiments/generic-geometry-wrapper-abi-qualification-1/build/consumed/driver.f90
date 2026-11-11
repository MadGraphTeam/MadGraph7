! Direct emitted-wrapper ABI fixture, not full SLOOPMATRIX lifetime coverage.
! argv1=inputs.txt argv2=start.fifo. No scientific reduction is performed.
program geometry_wrapper_abi
  use, intrinsic :: iso_c_binding, only: c_int64_t,c_double,c_double_complex
  use, intrinsic :: iso_fortran_env, only: output_unit
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  use aloha_object
  use ml5_0_invariant_bounds, only: ik,read_kinematic_environment_token, &
       environment_ok,external_slots_for_route
  use gp_recording_state
  implicit none
  interface
    subroutine ml5_0_loop_3_provenance(w1,w2,w3,m1,m2,m3,rank, &
        squaredsoindex,loopnum,gp_route,gp_env,gp_fresh)
      import :: c_int64_t,c_double_complex
      integer :: w1,w2,w3,rank,squaredsoindex,loopnum,gp_route
      complex(c_double_complex) :: m1,m2,m3
      integer(c_int64_t) :: gp_env
      logical :: gp_fresh
    end subroutine
  end interface
  type(ieee_flag_type), parameter :: observed_flags(5) = &
      [ieee_invalid,ieee_divide_by_zero,ieee_overflow,ieee_underflow,ieee_inexact]
  type(ieee_round_type), parameter :: rounds(0:3) = &
      [ieee_nearest,ieee_up,ieee_down,ieee_to_zero]
  type(ieee_status_type) :: original_status
  type(ieee_round_type) :: original_rounding
  logical :: original_halts(size(ieee_all))
  type(aloha) :: w(3)
  type(mp_aloha) :: mp_w(3)
  common /ml5_0_w/ w
  common /ml5_0_mp_w/ mp_w
  real(c_double) :: lscale
  integer :: ctmode
  common /ml5_0_ct/ lscale,ctmode
  integer(c_int64_t), parameter :: mass_bits(2,3) = reshape([ &
      int(z'3FF0000000000000',c_int64_t),int(z'8000000000000000',c_int64_t), &
      int(z'4000000000000000',c_int64_t),int(z'BFF0000000000000',c_int64_t), &
      int(z'4008000000000000',c_int64_t),int(z'3F50000000000000',c_int64_t)], [2,3])
  integer(c_int64_t) :: dp_words(2,0:3,3),post_dp(2,0:3,3)
  integer(ik) :: qp_words(2,0:3,3),post_qp(2,0:3,3)
  complex(c_double_complex) :: masses(3)
  real(kind=16) :: kind_probe
  integer :: unit,fifo_unit,ios,ncases,case_number,case_id,route,we(3)
  integer :: fresh_input,token_mode,sticky_mask,round_index,halt_flag
  integer :: pre_flags(5),post_flags(5),pre_halts(5),post_halts(5)
  integer :: pre_round,post_round,slots(3),i,j,e,mu
  integer :: rank,squaredsoindex,loopnum
  integer(c_int64_t) :: token_before,token_after,supplied_token
  logical :: fresh,slots_ok
  character(4096) :: input_path,fifo_path,trailing
  character(32) :: gate

  call ieee_get_status(original_status)
  call ieee_get_halting_mode(ieee_all,original_halts)
  call ieee_get_rounding_mode(original_rounding)
  call ieee_set_halting_mode(ieee_all,.false.)
  call get_command_argument(1,input_path)
  call get_command_argument(2,fifo_path)
  call require(len_trim(input_path)>0.and.len_trim(fifo_path)>0,'two paths required')
  call require(storage_size(kind_probe)==128.and.digits(kind_probe)==113,'binary128 kind required')
  call require(storage_size(dp_words(1,0,1))==64.and.storage_size(qp_words(1,0,1))==128, &
      'integer transport widths required')
  do i=0,3
    call require(ieee_support_rounding(rounds(i),kind_probe),'requested rounding mode unsupported')
  enddo
  do i=1,5
    call require(ieee_support_flag(observed_flags(i),kind_probe),'requested flag unsupported')
    call require(ieee_support_halting(observed_flags(i)),'requested halting mode unsupported')
  enddo
  call require(environment_ok(read_kinematic_environment_token()),'initial host controls ineligible')
  open(newunit=unit,file=trim(input_path),status='old',action='read',iostat=ios)
  call require(ios==0,'cannot open input')
  read(unit,*,iostat=ios) ncases
  call require(ios==0.and.ncases>0.and.ncases<=4096,'invalid case count')
  ! Runtime capture pause. Opening the FIFO for READ waits for its sole writer.
  write(output_unit,'(A,1X,I0)') 'READY',ncases
  flush(output_unit)
  open(newunit=fifo_unit,file=trim(fifo_path),status='old',action='read',iostat=ios)
  call require(ios==0,'cannot open start FIFO')
  read(fifo_unit,'(A)',iostat=ios) gate
  call require(ios==0.and.trim(gate)=='RUN','missing RUN gate')
  close(fifo_unit)

  do case_number=1,ncases
    call ieee_set_halting_mode(ieee_all,.false.)
    read(unit,*,iostat=ios) case_id,ctmode,route,we,fresh_input,token_mode, &
        sticky_mask,round_index,halt_flag
    call require(ios==0,'invalid case header')
    call require(case_id>0.and.fresh_input>=0.and.fresh_input<=1,'invalid case/freshness')
    call require(token_mode>=0.and.token_mode<=3,'invalid token selection')
    call require(sticky_mask>=0.and.sticky_mask<32.and.round_index>=0.and.round_index<=3 &
        .and.halt_flag>=0.and.halt_flag<=5,'invalid IEEE selection')
    do e=1,3
      do mu=0,3
        read(unit,'(Z16,1X,Z16)',iostat=ios) dp_words(:,mu,e)
        call require(ios==0,'invalid DP word record')
      enddo
    enddo
    do e=1,3
      do mu=0,3
        read(unit,'(Z32,1X,Z32)',iostat=ios) qp_words(:,mu,e)
        call require(ios==0,'invalid QP word record')
      enddo
    enddo
    do e=1,3
      w(e)%w = (0.0_c_double,0.0_c_double)
      mp_w(e)%w = (0.0_16,0.0_16)
      w(e)%flv_index = 0
      mp_w(e)%flv_index = 0
      do mu=0,3
        w(e)%p(mu) = transfer(dp_words(:,mu,e),w(e)%p(mu))
        mp_w(e)%p(mu) = transfer(qp_words(:,mu,e),mp_w(e)%p(mu))
      enddo
      masses(e) = transfer(mass_bits(:,e),masses(e))
    enddo
    rank=2
    squaredsoindex=0
    loopnum=1
    lscale=1.0_c_double
    fresh=fresh_input==1
    call clear_recording()
    ! Only this test driver changes IEEE controls or sticky state.
    call ieee_set_rounding_mode(rounds(round_index))
    call ieee_set_flag(ieee_all,.false.)
    do i=1,5
      if(btest(sticky_mask,i-1)) call ieee_set_flag(observed_flags(i),.true.)
    enddo
    if(halt_flag/=0) call ieee_set_halting_mode(observed_flags(halt_flag),.true.)
    call capture(pre_flags,pre_halts,pre_round)
    token_before=read_kinematic_environment_token()
    select case(token_mode)
    case(0)
      supplied_token=token_before
    case(1)
      supplied_token=0_c_int64_t
    case(2)
      supplied_token=ieor(token_before,1_c_int64_t)
    case(3)
      supplied_token=ieor(token_before,shiftl(1_c_int64_t,40))
    end select
    call ml5_0_loop_3_provenance(we(1),we(2),we(3),masses(1),masses(2),masses(3), &
        rank,squaredsoindex,loopnum,route,supplied_token,fresh)
    call capture(post_flags,post_halts,post_round)
    token_after=read_kinematic_environment_token()
    call ieee_set_halting_mode(ieee_all,.false.)
    ! Word observations and formatting occur after the measured call window.
    do e=1,3
      do mu=0,3
        post_dp(:,mu,e)=transfer(w(e)%p(mu),post_dp(:,mu,e))
        post_qp(:,mu,e)=transfer(mp_w(e)%p(mu),post_qp(:,mu,e))
      enddo
    enddo
    call external_slots_for_route(route,ctmode,slots,slots_ok)
    write(output_unit,'(A,4(1X,I0))') 'CASE',case_id,ctmode,route,fresh_input
    write(output_unit,'(A,3(1X,Z16.16))') 'TOKEN',token_before,token_after,supplied_token
    write(output_unit,'(A,11(1X,I0))') 'PRE',pre_flags,pre_halts,pre_round
    write(output_unit,'(A,11(1X,I0))') 'POST',post_flags,post_halts,post_round
    write(output_unit,'(A,4(1X,I0))') 'SLOTS',merge(1,0,slots_ok),slots
    do j=1,3
      write(output_unit,'(A,6(1X,I0))') 'BOUNDS',j,merge(1,0,recorded_have(j)),recorded_chord(:,j)
    enddo
    do j=1,3
      do i=1,3
        write(output_unit,'(A,3(1X,I0))') 'CLASS',i,j,recorded_class(i,j)
      enddo
    enddo
    write(output_unit,'(A,8(1X,I0))') 'FORWARD',provenance_calls,legacy_calls,recorded_we, &
        recorded_rank,recorded_so,recorded_loop
    do i=1,3
      write(output_unit,'(A,1X,I0,2(1X,Z16.16))') 'MASS',i,recorded_mass_words(:,i)
    enddo
    do e=1,3
      do mu=0,3
        write(output_unit,'(A,2(1X,I0),2(1X,Z16.16))') 'DPMOM',e,mu,post_dp(:,mu,e)
      enddo
    enddo
    do e=1,3
      do mu=0,3
        write(output_unit,'(A,2(1X,I0),2(1X,Z32.32))') 'QPMOM',e,mu,post_qp(:,mu,e)
      enddo
    enddo
    write(output_unit,'(A,1X,Z16.16)') 'RECORDED_ENV',recorded_environment
    write(output_unit,'(A,1X,I0)') 'ENDCASE',case_id
    flush(output_unit)
    call restore_environment()
  enddo
  call ieee_set_halting_mode(ieee_all,.false.)
  read(unit,'(A)',iostat=ios) trailing
  call require(ios<0,'extra input content')
  close(unit)
  write(output_unit,'(A,1X,I0)') 'DONE',ncases
  flush(output_unit)
  call restore_environment()
contains
  subroutine restore_environment()
    call ieee_set_halting_mode(ieee_all,.false.)
    call ieee_set_status(original_status)
    call ieee_set_rounding_mode(original_rounding)
    call ieee_set_halting_mode(ieee_all,original_halts)
  end subroutine

  subroutine require(condition,description)
    logical,intent(in) :: condition
    character(*),intent(in) :: description
    if(.not.condition) then
      call ieee_set_halting_mode(ieee_all,.false.)
      write(output_unit,'(A,1X,A)') 'DRIVER_ERROR',description
      flush(output_unit)
      call restore_environment()
      stop 2
    endif
  end subroutine

  subroutine capture(flags,halts,mode)
    integer,intent(out) :: flags(5),halts(5),mode
    logical :: value
    type(ieee_round_type) :: observed
    integer :: index
    do index=1,5
      call ieee_get_flag(observed_flags(index),value)
      flags(index)=merge(1,0,value)
      call ieee_get_halting_mode(observed_flags(index),value)
      halts(index)=merge(1,0,value)
    enddo
    call ieee_get_rounding_mode(observed)
    mode=-1
    do index=0,3
      if(observed==rounds(index)) mode=index
    enddo
  end subroutine
end program
