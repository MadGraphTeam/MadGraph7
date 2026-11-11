! SOURCE-ONLY mathematical kernel. Not compiled, linked, or runtime qualified.
! Caller must supply proved error bounds and environment/provenance contracts.
! No floating-point operation, external state, point ID, or fixture lookup.
module ml_private_spacelike_certificate
  implicit none
  private
  integer, parameter, public :: ik = selected_int_kind(33)
  integer, parameter, public :: zero_bound = -1000000
  integer, parameter, public :: composite_pair = 2
  public :: certified_spacelike
contains
  integer function ceil_log2(n) result(k)
    integer, intent(in) :: n
    integer :: j
    k=0
    j=n-1
    do while(j>0)
      j=ishft(j,-1)
      k=k+1
    enddo
  end function

  integer function power_sum(v) result(e)
    integer, intent(in) :: v(:)
    integer :: n
    n=count(v/=zero_bound)
    e=zero_bound
    if(n>0) e=maxval(v,mask=v/=zero_bound)+ceil_log2(n)
  end function

  subroutine binary_floor(bits,width,e,finite,negative)
    integer(ik), intent(in) :: bits
    integer, intent(in) :: width
    integer, intent(out) :: e
    logical, intent(out) :: finite,negative
    integer :: nf,ne,bias,ex,j
    e=zero_bound
    finite=.false.
    negative=.false.
    select case(width)
    case(64)
      nf=52; ne=11; bias=1023
      ! DP callers must zero-extend the original64 bits, not sign-extend.
      if(ishft(bits,-64)/=0_ik) return
    case(128)
      nf=112; ne=15; bias=16383
    case default
      return
    end select
    negative=btest(bits,width-1)
    ex=int(ibits(bits,nf,ne))
    if(ex==2**ne-1) return
    finite=.true.
    if(ex/=0) then
      e=ex-bias
    else
      do j=nf-1,0,-1
        if(btest(bits,j)) then
          e=1-bias-nf+j
          return
        endif
      enddo
    endif
  end subroutine

  logical function certified_spacelike(q_bits,raw_bits,error_exp,width, &
      pair_class,mass_snap_seen,have_bounds,environment_valid) result(ok)
    integer(ik), intent(in) :: q_bits(0:3),raw_bits
    integer, intent(in) :: error_exp(0:3),width,pair_class
    logical, intent(in) :: mass_snap_seen,have_bounds,environment_valid
    integer :: p,emin,k,f,qe(0:3),sq(0:3),terms(10),n,b,fr
    logical :: finite,negative
    ok=.false.
    if(pair_class/=composite_pair.or.mass_snap_seen) return
    if(.not.have_bounds.or..not.environment_valid) return
    if(any(error_exp/=zero_bound.and. &
        (error_exp < -100000.or.error_exp > 100000))) return
    select case(width)
    case(64)
      p=53; emin=-1022
    case(128)
      p=113; emin=-16382
    case default
      return
    end select
    call binary_floor(raw_bits,width,fr,finite,negative)
    if(.not.finite.or..not.negative.or.fr==zero_bound) return
    do k=0,3
      call binary_floor(q_bits(k),width,f,finite,negative)
      if(.not.finite) return
      qe(k)=zero_bound
      sq(k)=zero_bound
      if(f/=zero_bound) then
        qe(k)=f+1                 ! |qhat| < 2**qe, integer bound only
        sq(k)=2*qe(k)
      endif
    enddo
    terms=zero_bound
    n=0
    do k=0,3
      if(error_exp(k)==zero_bound) cycle
      n=n+1
      terms(n)=2*error_exp(k)     ! e**2
      if(qe(k)/=zero_bound) then
        n=n+1
        terms(n)=1+qe(k)+error_exp(k) ! 2*|qhat|*e
      endif
    enddo
    b=power_sum(sq)
    if(b/=zero_bound) then
      n=n+1
      terms(n)=3-p+b             ! gamma4 <= 8*u
    endif
    n=n+1
    terms(n)=4+emin-p            ! 16*eta, eta=min_subnormal/2
    b=power_sum(terms(:n))
    ! |raw| >= 2**fr > 2**b >= total forward-error bound.
    ok=(fr>b)
  end function
end module
