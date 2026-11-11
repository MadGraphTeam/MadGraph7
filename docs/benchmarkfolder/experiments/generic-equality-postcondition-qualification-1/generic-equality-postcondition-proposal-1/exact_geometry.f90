! PRIVATE SOURCE ONLY: not compiled, linked, or runtime qualified.
! Proves bit-level dyadic equality, not snapshot ownership or caller lifetime.
! No floating-point arithmetic, ambient state, memory allocation, or C calls.
module ml_private_exact_geometry
  implicit none
  private
  integer, parameter, public :: geometry_ik=selected_int_kind(33)
  integer, parameter, public :: geometry_max_external=64
  integer, parameter :: format_assert=1/merge(1,0, &
      bit_size(0_geometry_ik)==128.and.bit_size(0)==32)
  integer(geometry_ik), parameter :: max_integer=huge(0_geometry_ik)
  public :: exact_geometry_difference
contains
  pure subroutine normalize(magnitude,exponent)
    integer(geometry_ik), intent(inout) :: magnitude
    integer, intent(inout) :: exponent
    if(magnitude==0_geometry_ik) return
    do while(.not.btest(magnitude,0))
      magnitude=ishft(magnitude,-1)
      exponent=exponent+1
    enddo
  end subroutine

  pure subroutine decode_bits(bits,width,magnitude,exponent,negative,valid)
    integer(geometry_ik), intent(in) :: bits
    integer, intent(in) :: width
    integer(geometry_ik), intent(out) :: magnitude
    integer, intent(out) :: exponent
    logical, intent(out) :: negative,valid
    integer :: fraction_bits,exponent_bits,bias,stored_exponent
    magnitude=0_geometry_ik;exponent=0;negative=.false.;valid=.false.
    select case(width)
    case(64)
      fraction_bits=52;exponent_bits=11;bias=1023
      if(ishft(bits,-64)/=0_geometry_ik) return
    case(128)
      fraction_bits=112;exponent_bits=15;bias=16383
    case default
      return
    end select
    stored_exponent=int(ibits(bits,fraction_bits,exponent_bits))
    if(stored_exponent==2**exponent_bits-1) return
    negative=btest(bits,width-1)
    magnitude=ibits(bits,0,fraction_bits)
    if(stored_exponent==0) then
      exponent=1-bias-fraction_bits
    else
      magnitude=ibset(magnitude,fraction_bits)
      exponent=stored_exponent-bias-fraction_bits
    endif
    call normalize(magnitude,exponent)
    valid=.true.
  end subroutine

  pure logical function exact_geometry_difference(width,reference_width, &
      external_bits,coefficients,q_bits) result(ok)
    ! external_bits(real/imaginary,component,external leg), q_bits(part,component).
    ! Coefficients refer to repaired P, including generated ALOHA signs exactly once.
    ! The caller must separately prove owned/current same-kind reference snapshots,
    ! the exact route and current numerator, and final dot arithmetic. This helper
    ! does not assert any such caller contract or read a freshness/environment token.
    integer, intent(in) :: width,reference_width,coefficients(:)
    integer(geometry_ik), intent(in) :: external_bits(:,0:,:),q_bits(:,0:)
    integer(geometry_ik) :: magnitude(geometry_max_external+1)
    integer(geometry_ik) :: aligned(geometry_max_external+1),absolute_sum,total
    integer(geometry_ik) :: real_magnitude,imag_magnitude,coefficient_magnitude
    integer :: exponent(geometry_max_external+1),sign(geometry_max_external+1)
    integer :: n,mu,leg,terms,e,imag_exponent,coefficient_exponent,common,shift,j
    logical :: negative,imag_negative,valid
    ok=.false.
    if(width/=64.and.width/=128) return
    if(reference_width/=width) return
    n=size(coefficients)
    if(n<1.or.n>geometry_max_external) return
    if(size(external_bits,1)/=2.or.size(external_bits,2)/=4) return
    if(size(external_bits,3)/=n) return
    if(size(q_bits,1)/=2.or.size(q_bits,2)/=4) return
    do mu=0,3
      terms=0
      do leg=1,n
        call decode_bits(external_bits(2,mu,leg),width,imag_magnitude, &
            imag_exponent,imag_negative,valid)
        if(.not.valid.or.imag_magnitude/=0_geometry_ik) return
        call decode_bits(external_bits(1,mu,leg),width,real_magnitude,e,negative,valid)
        if(.not.valid) return
        coefficient_magnitude=abs(int(coefficients(leg),geometry_ik))
        if(real_magnitude==0_geometry_ik.or.coefficient_magnitude==0_geometry_ik) cycle
        coefficient_exponent=0
        call normalize(coefficient_magnitude,coefficient_exponent)
        ! Check before multiplication, including terms that would later cancel.
        if(real_magnitude>max_integer/coefficient_magnitude) return
        terms=terms+1
        magnitude(terms)=real_magnitude*coefficient_magnitude
        exponent(terms)=e+coefficient_exponent
        sign(terms)=1
        if(negative.neqv.(coefficients(leg)<0)) sign(terms)=-1
      enddo
      call decode_bits(q_bits(2,mu),width,imag_magnitude, &
          imag_exponent,imag_negative,valid)
      if(.not.valid.or.imag_magnitude/=0_geometry_ik) return
      call decode_bits(q_bits(1,mu),width,real_magnitude,e,negative,valid)
      if(.not.valid) return
      if(real_magnitude/=0_geometry_ik) then
        terms=terms+1
        magnitude(terms)=real_magnitude;exponent(terms)=e
        sign(terms)=-1
        if(negative) sign(terms)=1
      endif
      if(terms==0) cycle
      common=minval(exponent(:terms))
      absolute_sum=0_geometry_ik
      do j=1,terms
        shift=exponent(j)-common
        if(shift<0.or.shift>126) return
        if(magnitude(j)>ishft(max_integer,-shift)) return
        aligned(j)=ishft(magnitude(j),shift)
        ! This bounds EVERY ordering of the later signed sum before it executes.
        if(aligned(j)>max_integer-absolute_sum) return
        absolute_sum=absolute_sum+aligned(j)
      enddo
      total=0_geometry_ik
      do j=1,terms
        if(sign(j)==1) then
          total=total+aligned(j)
        else
          total=total-aligned(j)
        endif
      enddo
      if(total/=0_geometry_ik) return
    enddo
    ok=.true.
  end function
end module
