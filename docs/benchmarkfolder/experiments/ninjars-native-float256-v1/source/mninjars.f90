! ISO C interface to the pinned NinjaRS bridge. No compiler-specific quad ABI.
! res(0) contains the complete finite result. r1 is an engine-basis diagnostic;
! the MP generic engine's rational/CC split can differ from the DP Ninja basis.
module mninjars
  use, intrinsic :: iso_c_binding
  implicit none
  private
  integer, parameter, public :: KI_NRS = c_double
  integer, parameter, public :: KI_QNRS = selected_real_kind(33,4931)
  integer, parameter, public :: NINJARS_SUCCESS = 0
  integer, parameter, public :: NINJARS_UNSTABLE_KINEMATICS = 1
  integer, parameter, public :: NINJARS_ERROR = 2
  integer, parameter :: record_size = 64
  type(c_ptr), save :: context = c_null_ptr
  public :: ninjars_tensor_evaluate_compensated
  public :: ninjars_tensor_evaluate, ninjars_error_message, ninjars_finalize

  interface ninjars_tensor_evaluate
    module procedure tensor_dp, tensor_mp
  end interface
  interface
    function create_context(version,handle) bind(C,name="ninjars_context_create") result(status)
      import
      integer(c_int), value :: version
      type(c_ptr), intent(out) :: handle
      integer(c_int) :: status
    end function
    function destroy_context(handle) bind(C,name="ninjars_context_destroy") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int) :: status
    end function
    function evaluate_dp(handle,n,r,count,coeff,s,slen,p,plen,m,mlen,mu2,res,reslen,r1,r1len) &
        bind(C,name="ninjars_evaluate_dp") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int), value :: n,r
      integer(c_size_t), value :: count,slen,plen,mlen,reslen,r1len
      complex(c_double_complex), intent(in) :: coeff(*),m(*)
      real(c_double), intent(in) :: s(*),p(*)
      real(c_double), value :: mu2
      complex(c_double_complex), intent(out) :: res(*),r1
      integer(c_int) :: status
    end function
    function evaluate_dp_compensated(handle,n,r,count,coeff,s,slen,p,plen,m,mlen,mu2,res,reslen,r1,r1len) &
        bind(C,name="ninjars_evaluate_dp_compensated") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int), value :: n,r
      integer(c_size_t), value :: count,slen,plen,mlen,reslen,r1len
      complex(c_double_complex), intent(in) :: coeff(*),m(*)
      real(c_double), intent(in) :: s(*),p(*)
      real(c_double), value :: mu2
      complex(c_double_complex), intent(out) :: res(*),r1
      integer(c_int) :: status
    end function
    function evaluate_mp(handle,n,r,count,coeff,clen,s,slen,p,plen,m,mlen,mu2,scalelen,res,reslen) &
        bind(C,name="ninjars_evaluate_mp") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int), value :: n,r
      integer(c_size_t), value :: count,clen,slen,plen,mlen,scalelen,reslen
      character(c_char), intent(in) :: coeff(*),s(*),p(*),m(*),mu2(*)
      character(c_char), intent(out) :: res(*)
      integer(c_int) :: status
    end function
    function last_error(message,capacity) bind(C,name="ninjars_last_error") result(required)
      import
      character(c_char), intent(out) :: message(*)
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
    end function
  end interface
contains
  subroutine ensure_context(status)
    integer, intent(out) :: status
    status = NINJARS_SUCCESS
    if (.not.c_associated(context)) status = create_context(3_c_int,context)
  end subroutine

  subroutine ninjars_finalize(status)
    integer, intent(out), optional :: status
    integer :: result
    result = destroy_context(context)
    if (result == NINJARS_SUCCESS) context = c_null_ptr
    if (present(status)) status = result
  end subroutine

  pure integer function coefficient_count(rank) result(count)
    integer, intent(in) :: rank
    count = (rank+1)*(rank+2)*(rank+3)*(rank+4)/24
  end function

  subroutine tensor_dp(coeff,n,rank,s,p,m,mu2,res,r1,status)
    integer, intent(in) :: n,rank
    complex(KI_NRS), intent(in) :: coeff(coefficient_count(rank)),m(n)
    real(KI_NRS), intent(in) :: s(n,n),p(0:3,n),mu2
    complex(KI_NRS), intent(out) :: res(0:2),r1
    integer, intent(out) :: status
    res = cmplx(0,0,kind=KI_NRS)
    r1 = cmplx(0,0,kind=KI_NRS)
    call ensure_context(status)
    if (status /= NINJARS_SUCCESS) return
    status = evaluate_dp(context,int(n,c_int),int(rank,c_int), &
                         int(coefficient_count(rank),c_size_t),coeff, &
                         s,size(s,kind=c_size_t),p,size(p,kind=c_size_t), &
                         m,size(m,kind=c_size_t),mu2,res,3_c_size_t,r1,1_c_size_t)
  end subroutine

  subroutine ninjars_tensor_evaluate_compensated(coeff,n,rank,s,p,m,mu2,res,r1,status)
    integer, intent(in) :: n,rank
    complex(KI_NRS), intent(in) :: coeff(coefficient_count(rank)),m(n)
    real(KI_NRS), intent(in) :: s(n,n),p(0:3,n),mu2
    complex(KI_NRS), intent(out) :: res(0:2),r1
    integer, intent(out) :: status
    res = cmplx(0,0,kind=KI_NRS)
    r1 = cmplx(0,0,kind=KI_NRS)
    call ensure_context(status)
    if (status /= NINJARS_SUCCESS) return
    status = evaluate_dp_compensated(context,int(n,c_int),int(rank,c_int), &
                         int(coefficient_count(rank),c_size_t),coeff, &
                         s,size(s,kind=c_size_t),p,size(p,kind=c_size_t), &
                         m,size(m,kind=c_size_t),mu2,res,3_c_size_t,r1,1_c_size_t)
  end subroutine

  subroutine pack(value,record)
    real(KI_QNRS), intent(in) :: value
    character(c_char), intent(out) :: record(record_size)
    character(len=record_size) :: text
    integer :: k
    write(text,'(ES64.39E4)') value
    do k=1,record_size
      record(k) = text(k:k)
    enddo
  end subroutine

  subroutine unpack(record,value,io_status)
    character(c_char), intent(in) :: record(record_size)
    real(KI_QNRS), intent(out) :: value
    integer, intent(out) :: io_status
    character(len=record_size) :: text
    integer :: k
    do k=1,record_size
      text(k:k) = record(k)
    enddo
    read(text,*,iostat=io_status) value
  end subroutine

  subroutine tensor_mp(coeff,n,rank,s,p,m,mu2,res,r1,status)
    integer, intent(in) :: n,rank
    complex(KI_QNRS), intent(in) :: coeff(coefficient_count(rank)),m(n)
    real(KI_QNRS), intent(in) :: s(n,n),p(0:3,n),mu2
    complex(KI_QNRS), intent(out) :: res(0:2),r1
    integer, intent(out) :: status
    character(c_char) :: coeff_text(record_size,2*coefficient_count(rank))
    character(c_char) :: s_text(record_size,n*n),p_text(record_size,4*n)
    character(c_char) :: m_text(record_size,2*n),scale_text(record_size),output(record_size,8)
    real(KI_QNRS) :: values(8)
    integer :: i,j,k,ios
    res = cmplx(0,0,kind=KI_QNRS)
    r1 = cmplx(0,0,kind=KI_QNRS)
    status = NINJARS_ERROR
    if (radix(mu2) /= 2 .or. digits(mu2) /= 113) return
    call ensure_context(status)
    if (status /= NINJARS_SUCCESS) return
    do i=1,coefficient_count(rank)
      call pack(real(coeff(i),kind=KI_QNRS),coeff_text(:,2*i-1))
      call pack(aimag(coeff(i)),coeff_text(:,2*i))
    enddo
    do i=1,n
      call pack(real(m(i),kind=KI_QNRS),m_text(:,2*i-1))
      call pack(aimag(m(i)),m_text(:,2*i))
      do j=0,3
        call pack(p(j,i),p_text(:,4*(i-1)+j+1))
      enddo
      do j=1,n
        call pack(s(j,i),s_text(:,n*(i-1)+j))
      enddo
    enddo
    call pack(mu2,scale_text)
    status = evaluate_mp(context,int(n,c_int),int(rank,c_int),int(coefficient_count(rank),c_size_t), &
                         coeff_text,size(coeff_text,kind=c_size_t),s_text,size(s_text,kind=c_size_t), &
                         p_text,size(p_text,kind=c_size_t),m_text,size(m_text,kind=c_size_t), &
                         scale_text,size(scale_text,kind=c_size_t),output,size(output,kind=c_size_t))
    if (status /= NINJARS_SUCCESS) return
    do k=1,8
      call unpack(output(:,k),values(k),ios)
      if (ios /= 0) then
        status = NINJARS_ERROR
        return
      endif
    enddo
    do k=0,2
      res(k) = cmplx(values(2*k+1),values(2*k+2),kind=KI_QNRS)
    enddo
    r1 = cmplx(values(7),values(8),kind=KI_QNRS)
  end subroutine

  subroutine ninjars_error_message(message)
    character(len=*), intent(out) :: message
    character(c_char) :: buffer(len(message)+1)
    integer(c_size_t) :: required
    integer :: k
    required = last_error(buffer,int(size(buffer),c_size_t))
    message = ''
    do k=1,len(message)
      if (buffer(k) == c_null_char) exit
      message(k:k) = buffer(k)
    enddo
  end subroutine
end module mninjars
