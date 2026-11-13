! Exercise the installed Fortran ABI while another process retains its worker.
! The Python harness releases both programs at each READY/ACTIVE barrier.
program oem_lifecycle
  use mninjars
  use, intrinsic :: iso_fortran_env, only: output_unit
  implicit none
  complex(KI_NRS) :: c(1), m(1), r(0:2), r1, bm(2)
  real(KI_NRS) :: p(0:3,1), s(1,1), bp(0:3,2), bs(2,2), mass2, expected
  complex(KI_QNRS) :: qc(1), qm(1), qr(0:2), qr1, qbm(2)
  real(KI_QNRS) :: qp(0:3,1), qs(1,1), qbp(0:3,2), qbs(2,2), delta, qexpected
  integer, parameter :: modes(8) = [0,1,2,3,0,2,1,3]
  integer :: cycle, cycles, status, mode, i, ios
  character(len=32) :: argument, release

  cycles=3
  call get_command_argument(1,argument)
  if (len_trim(argument)>0) then
    read(argument,*,iostat=ios) cycles
    if (ios/=0 .or. cycles<1) error stop 1
  endif
  p=0
  s=0
  qp=0
  qs=0
  bm=cmplx(0,0,kind=KI_NRS)
  bp=0
  bp(0,2)=2
  bp(3,2)=2
  bs=0
  qbm=cmplx(0,0,kind=KI_QNRS)
  qbp=real(bp,kind=KI_QNRS)
  qbs=0
  delta=2.0_KI_QNRS**(-100)

  do cycle=1,cycles
    call barrier('READY',cycle)
    c=cmplx(cycle,0,kind=KI_NRS)
    mass2=4.0_KI_NRS+cycle
    m=cmplx(mass2,0,kind=KI_NRS)
    expected=cycle*mass2
    call ninjars_tensor_evaluate(c,1,0,s,p,m,mass2,r,r1,status)
    call check_status('combined DP tadpole')
    call check(abs(r(0)-expected)<1e-12_KI_NRS .and. abs(r(1)-expected)<1e-12_KI_NRS &
               .and. abs(r(2))<1e-12_KI_NRS,'combined DP tadpole coefficients')
    call ninjars_tensor_evaluate_compensated(c,1,0,s,p,m,mass2,r,r1,status)
    call check_status('compensated DP tadpole')
    call check(abs(r(0)-expected)<1e-12_KI_NRS .and. abs(r(1)-expected)<1e-12_KI_NRS &
               .and. abs(r(2))<1e-12_KI_NRS,'compensated DP tadpole coefficients')

    ! Selected DP constructs Symbolica-backed masters on the shared worker.
    call ninjars_tensor_evaluate(c,2,0,bs,bp,bm,mass2,r,r1,status,pole_mode=NINJARS_POLES_UV)
    call check_status('selected DP bubble')
    call check(abs(r(0))<1e-12_KI_NRS .and. abs(r(1)-cycle)<1e-12_KI_NRS &
               .and. abs(r(2))<1e-12_KI_NRS,'selected DP bubble coefficients')

    qc=cmplx(real(cycle,KI_QNRS)+delta,0.0_KI_QNRS,kind=KI_QNRS)
    qm=cmplx(4,0,kind=KI_QNRS)
    call ninjars_tensor_evaluate(qc,1,0,qs,qp,qm,4.0_KI_QNRS,qr,qr1,status)
    call check_status('MP tadpole')
    qexpected=4.0_KI_QNRS*(real(cycle,KI_QNRS)+delta)
    call check(abs(qr(0)-qexpected)<1e-32_KI_QNRS .and. abs(qr(1)-qexpected)<1e-32_KI_QNRS &
               .and. abs(qr(2))<1e-32_KI_QNRS,'MP tadpole coefficients')
    call check(real(qr(1),KI_QNRS)/=4.0_KI_QNRS*cycle,'MP retains binary128-only input')

    ! Both processes must reach this barrier with a live numerical worker.
    ! An accidental fallback to one restricted thread per user cannot pass it.
    call barrier('ACTIVE',cycle)
    do i=1,size(modes)
      mode=modes(i)
      expected=0
      if (mode==NINJARS_POLES_UV) expected=cycle
      if (mode==NINJARS_POLES_IR) expected=-cycle
      qexpected=0
      if (mode==NINJARS_POLES_UV) qexpected=real(cycle,KI_QNRS)+delta
      if (mode==NINJARS_POLES_IR) qexpected=-real(cycle,KI_QNRS)-delta
      call ninjars_tensor_evaluate(c,2,0,bs,bp,bm,mass2,r,r1,status,pole_mode=mode)
      call check_status('DP pole-mode cycle')
      call check(abs(r(0))<1e-12_KI_NRS .and. abs(r(1)-expected)<1e-12_KI_NRS &
                 .and. abs(r(2))<1e-12_KI_NRS,'DP pole-mode coefficients')
      call ninjars_tensor_evaluate(qc,2,0,qbs,qbp,qbm,4.0_KI_QNRS,qr,qr1,status,pole_mode=mode)
      call check_status('MP pole-mode cycle')
      call check(abs(qr(0))<1e-32_KI_QNRS .and. abs(qr(1)-qexpected)<1e-32_KI_QNRS &
                 .and. abs(qr(2))<1e-32_KI_QNRS,'MP pole-mode coefficients')
    enddo
    call ninjars_finalize(status)
    call check_status('finalize')
    call ninjars_finalize(status)
    call check_status('repeated finalize')
    call barrier('DONE',cycle)
  enddo
  write(output_unit,'(A)') 'OEM_PASSED'
  flush(output_unit)
contains
  subroutine check(condition,description)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: description
    if (.not.condition) then
      write(output_unit,'(A,1X,A)') 'OEM_FAILED',description
      flush(output_unit)
      error stop 1
    endif
  end subroutine

  subroutine check_status(description)
    character(len=*), intent(in) :: description
    character(len=512) :: message
    if (status/=NINJARS_SUCCESS) then
      call ninjars_error_message(message)
      write(output_unit,'(A,1X,I0,1X,A)') 'OEM_STATUS',status,trim(message)
    endif
    call check(status==NINJARS_SUCCESS,description)
  end subroutine

  subroutine barrier(stage,number)
    character(len=*), intent(in) :: stage
    integer, intent(in) :: number
    write(output_unit,'(A,1X,I0)') 'OEM_'//stage,number
    flush(output_unit)
    read(*,'(A)',iostat=ios) release
    call check(ios==0 .and. trim(release)=='GO','barrier release')
  end subroutine
end program
