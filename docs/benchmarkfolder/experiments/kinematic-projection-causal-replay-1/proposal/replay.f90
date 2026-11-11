! PRIVATE diagnostic proposal: exactly one original MTIRUST tensor call.
! No compiler, loader or execution has been admitted for this source packet.
program kinematic_projection_replay
  use mtirust, only: KI_TIR, KI_QTIR, TIRUST_SUCCESS, TIRUST_TENSOR_EVALUATE, &
       TIRUST_REGISTER_PLAN, TIRUST_SET_CACHE_BUDGET, TIRUST_ERROR_MESSAGE, &
       TIRUST_CONTEXT_GENERATION, TIRUST_FINALIZE, TIRUST_SUPPORTS
  use, intrinsic :: iso_c_binding, only: c_int64_t
  use, intrinsic :: iso_fortran_env, only: output_unit, input_unit
  implicit none
  integer, parameter :: n=4, rank=2, count=15
  integer, parameter :: plan_n(5)=[1,2,2,3,4], plan_rank(5)=[0,0,2,2,2]
  integer :: point, mode, declared_n, declared_rank, nwords, u, ios, i, status, close_status
  integer(c_int64_t) :: words(82), echoed(82), probe(2)
  real(KI_TIR) :: p(0:3,n), s(n,n), mu2
  complex(KI_TIR) :: masses(n), coefs(count,4)
  real(KI_QTIR) :: qp(0:3,n), qs(n,n), qmu2
  complex(KI_QTIR) :: qmasses(n), qcoefs(count,4)
  character(len=4096) :: input_path
  character(len=128) :: line, command
  character(len=8192) :: saved_error

  if (command_argument_count()/=1) error stop 'Exactly one pinned word-input file required'
  if (storage_size(0_c_int64_t)/=64 .or. bit_size(0_c_int64_t)/=64) error stop 'INTEGER*8 ABI differs'
  if (storage_size(0.0_KI_TIR)/=64 .or. digits(0.0_KI_TIR)/=53) error stop 'DP ABI differs'
  if (storage_size(0.0_KI_QTIR)/=128 .or. digits(0.0_KI_QTIR)/=113) error stop 'QP ABI differs'
  if (radix(0.0_KI_TIR)/=2 .or. radix(0.0_KI_QTIR)/=2) error stop 'Binary ABI required'
  probe=transfer(1.0_KI_QTIR,probe)
  if (probe(1)/=0_c_int64_t .or. probe(2)/=int(z'3FFF000000000000',c_int64_t)) &
       error stop 'Qualified low-word-first binary128 ABI required'
  if (TIRUST_CONTEXT_GENERATION()/=-1) error stop 'Context was not fresh at process entry'
  call get_command_argument(1,input_path,status=ios)
  if (ios/=0) error stop 'Input pathname truncated or unavailable'
  open(newunit=u,file=trim(input_path),status='old',action='read',iostat=ios)
  if (ios/=0) error stop 'Pinned input unavailable'
  read(u,'(A)',iostat=ios) line
  if (ios/=0 .or. trim(line)/='KPROJ_WORDS_V1') error stop 'Input schema differs'
  point=0; mode=0; declared_n=0; declared_rank=0; nwords=0
  read(u,*,iostat=ios) point,mode,declared_n,declared_rank,nwords
  if (ios/=0 .or. declared_n/=n .or. declared_rank/=rank) error stop 'Input topology/rank differs'
  if ((mode/=1 .and. mode/=4) .or. nwords/=merge(41,82,mode==1)) error stop 'Input precision/word count differs'
  words=0_c_int64_t
  do i=1,nwords
    read(u,'(A)',iostat=ios) line
    if (ios/=0 .or. len_trim(line)/=16) error stop 'Input integer-word width differs'
    if (verify(line(:16),'0123456789ABCDEF')/=0) error stop 'Input integer word is not canonical hexadecimal'
    read(line(:16),'(Z16)',iostat=ios) words(i)
    if (ios/=0) error stop 'Input integer word could not be read'
  enddo
  read(u,'(A)',iostat=ios) line
  if (ios>=0) error stop 'Unexpected trailing input or I/O failure'
  close(u)

  if (mode==1) then
    p=reshape(transfer(words(1:16),[0.0_KI_TIR],16),shape(p))
    masses=transfer(words(17:24),masses)
    s=reshape(transfer(words(25:40),[0.0_KI_TIR],16),shape(s))
    mu2=transfer(words(41),mu2)
  else
    qp=reshape(transfer(words(1:32),[0.0_KI_QTIR],16),shape(qp))
    qmasses=transfer(words(33:48),qmasses)
    qs=reshape(transfer(words(49:80),[0.0_KI_QTIR],16),shape(qs))
    qmu2=transfer(words(81:82),qmu2)
  endif
  call capture_words()
  if (any(echoed(1:nwords)/=words(1:nwords))) error stop 'Reconstructed Fortran input bits differ'
  write(output_unit,'(A,5(1X,I0))') 'INPUT_ID',point,mode,n,rank,nwords
  write(output_unit,'(A)') 'OPTIONS pole=0 finite=1 check=0 cache=0 budget=268435456'
  write(output_unit,'(A,*(1X,Z16.16))') 'INPUT_WORDS',echoed(1:nwords)
  if (.not.TIRUST_SUPPORTS(n,rank,.true.)) error stop 'Installed backend does not support exact replay topology'
  do i=1,5
    call TIRUST_REGISTER_PLAN(plan_n(i),plan_rank(i),status)
    call structural_status('REGISTER_PLAN',status)
  enddo
  call TIRUST_SET_CACHE_BUDGET(268435456_c_int64_t,status)
  call structural_status('SET_CACHE_BUDGET',status)
  if (TIRUST_CONTEXT_GENERATION()/=1) error stop 'Expected exactly one newly created context'
  write(output_unit,'(A)') 'READY KPROJ_V1'
  flush(output_unit)
  ! Future owned runner verifies live maps at READY before allowing evaluation.
  read(input_unit,'(A)',iostat=ios) command
  if (ios/=0 .or. trim(command)/='EVALUATE') error stop 'Explicit evaluation handshake missing'
  if (mode==1) then
    call TIRUST_TENSOR_EVALUATE(n,rank,p,s,masses,mu2,coefs,status, &
         pole_mode=0,compute_finite=.true.,check_pass=.false.,cache_enabled=.false.)
  else
    call TIRUST_TENSOR_EVALUATE(n,rank,qp,qs,qmasses,qmu2,qcoefs,status, &
         pole_mode=0,compute_finite=.true.,check_pass=.false.,cache_enabled=.false.)
  endif
  ! Error transport must precede any structural call that can clear last_error.
  call TIRUST_ERROR_MESSAGE(saved_error)
  write(output_unit,'(A,1X,I0)') 'FORTRAN_STATUS',status
  write(output_unit,'(A)') 'FORTRAN_EVALUATION_CALLS 1'
  write(output_unit,'(A,1X,I0)') 'CONTEXT_GENERATION',TIRUST_CONTEXT_GENERATION()
  write(output_unit,'(A,1X,I0)') 'ERROR_BUFFER_FULL',merge(1,0,len_trim(saved_error)==len(saved_error))
  write(output_unit,'(A)') 'ERROR_BEGIN'
  write(output_unit,'(A)') trim(saved_error)
  write(output_unit,'(A)') 'ERROR_END'
  call capture_words()
  write(output_unit,'(A,*(1X,Z16.16))') 'POST_INPUT_WORDS',echoed(1:nwords)
  write(output_unit,'(A,1X,I0)') 'INPUTS_UNCHANGED',merge(1,0,all(echoed(1:nwords)==words(1:nwords)))
  if (any(echoed(1:nwords)/=words(1:nwords))) error stop 'Working input arrays changed during evaluation'
  write(output_unit,'(A)') 'COEFFICIENT_BUFFERS_NOT_INSPECTED'
  write(output_unit,'(A)') 'DONE KPROJ_V1'
  flush(output_unit)
  ! Keep the worker alive for the same post-evaluation map verification.
  read(input_unit,'(A)',iostat=ios) command
  if (ios/=0 .or. trim(command)/='EXIT') error stop 'Explicit close handshake missing'
  call TIRUST_FINALIZE(close_status)
  write(output_unit,'(A,1X,I0)') 'FINALIZE_STATUS',close_status
  if (close_status/=TIRUST_SUCCESS .or. TIRUST_CONTEXT_GENERATION()/=-1) error stop 'Context cleanup failed'
  write(output_unit,'(A)') 'CLOSED KPROJ_V1'
contains
  subroutine capture_words()
    if (mode==1) then
    echoed(1:16)=transfer(p,echoed(1:16))
    echoed(17:24)=transfer(masses,echoed(17:24))
    echoed(25:40)=transfer(s,echoed(25:40))
    echoed(41)=transfer(mu2,echoed(41))
    else
    echoed(1:32)=transfer(qp,echoed(1:32))
    echoed(33:48)=transfer(qmasses,echoed(33:48))
    echoed(49:80)=transfer(qs,echoed(49:80))
    echoed(81:82)=transfer(qmu2,echoed(81:82))
    endif
  end subroutine
  subroutine structural_status(label,code)
    character(len=*), intent(in) :: label
    integer, intent(in) :: code
    character(len=8192) :: message
    if (code==TIRUST_SUCCESS) return
    call TIRUST_ERROR_MESSAGE(message)
    write(output_unit,'(A,1X,A,1X,I0)') 'STRUCTURAL_FAILURE',label,code
    write(output_unit,'(A)') trim(message)
    error stop 'No tensor call after structural initialization failure'
  end subroutine
end program kinematic_projection_replay
