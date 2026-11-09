program mixed_smoke
  use mtirust
  use mninjars
  use, intrinsic :: ieee_arithmetic
  implicit none
  real(KI_TIR) :: p(0:3,1),s(1,1),scale,bp(0:3,2),bs(2,2)
  complex(KI_TIR) :: m(1),c(1),t(1,4),nr(0:2),r1,bm(2)
  real(KI_QTIR) :: qp(0:3,1),qs(1,1),qscale,qbp(0:3,2),qbs(2,2)
  complex(KI_QTIR) :: qm(1),qc(1),qt(1,4),qnr(0:2),qr1,qbm(2)
  integer :: status,cycle,mode,generation
  logical :: tirust_first
  character(len=32) :: option
  character(len=512) :: message
  call get_command_argument(1,option)
  tirust_first = trim(option) == 'tirust-first'
  p=0; s=0; qp=0; qs=0
  c=cmplx(1,0,kind=KI_TIR); qc=cmplx(1,0,kind=KI_QTIR)
  bp=0; bp(0,2)=2; bp(3,2)=2; bs=0; bm=cmplx(0,0,kind=KI_TIR)
  qbp=real(bp,kind=KI_QTIR); qbs=0; qbm=cmplx(0,0,kind=KI_QTIR)
  call check_rounding()
  do cycle=1,2
    scale=real((cycle+1)**2,kind=KI_TIR)
    m=cmplx(scale,0.0_KI_TIR,kind=KI_TIR)
    qscale=4.0_KI_QTIR*(1.0_KI_QTIR+2.0_KI_QTIR**(-112))
    qm=cmplx(qscale,-0.0_KI_QTIR,kind=KI_QTIR)
    do mode=TIRUST_POLES_BOTH,TIRUST_POLES_NONE
      if (tirust_first) then
        call tir_dp(); call ninja_dp(); call tir_qp(); call ninja_qp()
      else
        call ninja_dp(); call tir_dp(); call ninja_qp(); call tir_qp()
      endif
      if (abs(nr(0)-t(1,4))>1e-12_KI_TIR .or. abs(nr(1)-sum(t(1,1:2)))>1e-12_KI_TIR .or. &
          abs(nr(2)-t(1,3))>1e-12_KI_TIR) stop 2
      if (abs(nr(0)-scale)>1e-12_KI_TIR) stop 3
      if (abs(qnr(0)-qt(1,4))>1e-31_KI_QTIR .or. abs(qnr(1)-sum(qt(1,1:2)))>1e-31_KI_QTIR .or. &
          abs(qnr(2)-qt(1,3))>1e-31_KI_QTIR) stop 4
      if (mode<=TIRUST_POLES_UV) then
        if (real(qnr(1),kind=KI_QTIR)/=qscale .or. real(qt(1,1),kind=KI_QTIR)/=qscale) stop 5
      endif
      call tirust_tensor_evaluate(2,0,bp,bs,bm,scale,t,status,pole_mode=mode)
      call require_tirust('DP bubble'); call check_rounding()
      call ninjars_tensor_evaluate(c,2,0,bs,bp,bm,scale,nr,r1,status,pole_mode=mode)
      call require_ninjars('DP bubble'); call check_rounding()
      if (abs(nr(1)-sum(t(1,1:2)))>1e-12_KI_TIR .or. abs(nr(0)-t(1,4))>1e-12_KI_TIR) stop 6
      call ninjars_tensor_evaluate(qc,2,0,qbs,qbp,qbm,qscale,qnr,qr1,status,pole_mode=mode)
      call require_ninjars('QP bubble'); call check_rounding()
      call tirust_tensor_evaluate(2,0,qbp,qbs,qbm,qscale,qt,status,pole_mode=mode)
      call require_tirust('QP bubble'); call check_rounding()
      if (abs(qnr(1)-sum(qt(1,1:2)))>1e-31_KI_QTIR .or. abs(qnr(0)-qt(1,4))>1e-31_KI_QTIR) stop 7
    enddo
    generation=tirust_context_generation()
    call ninjars_finalize(status); call require_ninjars('finalize'); call check_rounding()
    if (tirust_context_generation()/=generation) stop 8
    call tirust_finalize(status); call require_tirust('finalize'); call check_rounding()
    if (tirust_context_generation()/=-1) stop 9
    tirust_first=.not.tirust_first
  enddo
  print *, 'Mixed TIRust/NinjaRS DP, binary128, poles, lifecycle and rounding passed: ',trim(option)
contains
  subroutine tir_dp()
    call tirust_tensor_evaluate(1,0,p,s,m,scale,t,status,pole_mode=mode)
    call require_tirust('DP tadpole'); call check_rounding()
  end subroutine
  subroutine ninja_dp()
    call ninjars_tensor_evaluate(c,1,0,s,p,m,scale,nr,r1,status,pole_mode=mode)
    call require_ninjars('DP tadpole'); call check_rounding()
  end subroutine
  subroutine tir_qp()
    call tirust_tensor_evaluate(1,0,qp,qs,qm,qscale,qt,status,pole_mode=mode)
    call require_tirust('QP tadpole'); call check_rounding()
  end subroutine
  subroutine ninja_qp()
    call ninjars_tensor_evaluate(qc,1,0,qs,qp,qm,qscale,qnr,qr1,status,pole_mode=mode)
    call require_ninjars('QP tadpole'); call check_rounding()
  end subroutine
  subroutine require_tirust(label)
    character(len=*), intent(in) :: label
    if (status==TIRUST_SUCCESS) return
    call tirust_error_message(message)
    print *, 'TIRust ',trim(label),': ',trim(message),' status=',status
    stop 1
  end subroutine
  subroutine require_ninjars(label)
    character(len=*), intent(in) :: label
    if (status==NINJARS_SUCCESS) return
    call ninjars_error_message(message)
    print *, 'NinjaRS ',trim(label),': ',trim(message),' status=',status
    stop 1
  end subroutine
  subroutine check_rounding()
    real(KI_QTIR), volatile :: x,y,total
    type(ieee_round_type) :: original
    call ieee_get_rounding_mode(original)
    if (original/=ieee_nearest) stop 10
    x=1.0_KI_QTIR; y=2.0_KI_QTIR**(-114)
    call ieee_set_rounding_mode(ieee_up)
    total=x+y
    call ieee_set_rounding_mode(original)
    if (total/=nearest(1.0_KI_QTIR,1.0_KI_QTIR)) stop 11
  end subroutine
end program
