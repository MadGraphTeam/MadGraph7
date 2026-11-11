! SOURCE ONLY. No FP arithmetic, implicit context, event IDs, or allowlist.
module ml_private_cms_caller_bounds
  use, intrinsic :: iso_c_binding, only: c_int64_t
  implicit none
  private
  integer, parameter, public :: ik=selected_int_kind(33)
  integer, parameter, public :: zero_bound=-1000000
  public :: read_kinematic_environment_token, environment_ok, environment_unchanged
  public :: route_matches, build_chord_bounds, finish_difference_bounds
  public :: producer_kind_ok
  interface
    function read_kinematic_environment_token() result(token) &
        bind(C,name='ml_private_kinematic_environment_token')
      import :: c_int64_t
      integer(c_int64_t) :: token
    end function
  end interface
  include 'cms_bounds_data.inc'
contains
  logical function producer_kind_ok(mode,width,genuine_qp) result(ok)
    integer,intent(in)::mode,width
    logical,intent(in)::genuine_qp
    ok=.false.
    if((mode==1.or.mode==2).and.width==64) ok=.true.
    if((mode==4.or.mode==5).and.width==128.and.genuine_qp) ok=.true.
  end function

  logical function environment_ok(token) result(ok)
    integer(c_int64_t), intent(in) :: token
    integer(c_int64_t) :: csr,cw
    ok=.false.
    if(iand(token,int(z'FFFFFFFF00000000',c_int64_t))/= &
       int(z'4B45000000000000',c_int64_t)) return
    csr=iand(ishft(token,-16),int(z'FFFF',c_int64_t))
    cw=iand(token,int(z'FFFF',c_int64_t))
    if(iand(csr,int(z'003F',c_int64_t))/=0) return
    if(iand(cw,int(z'E0C0',c_int64_t))/=0) return
    if(iand(csr,int(z'E040',c_int64_t))/=0) return ! nearest, no FTZ/DAZ
    if(iand(cw,int(z'0C00',c_int64_t))/=0) return  ! x87 nearest
    if(iand(cw,int(z'0300',c_int64_t))/=int(z'0300',c_int64_t)) return
    ok=.true.
  end function

  logical function environment_unchanged(token) result(ok)
    integer(c_int64_t), intent(in) :: token
    integer(c_int64_t) :: now
    now=read_kinematic_environment_token()
    ok=environment_ok(token).and.(now==token)
  end function

  integer function clog2(n) result(k)
    integer, intent(in) :: n
    integer :: v
    k=0; v=n-1
    do while(v>0)
      k=k+1; v=ishft(v,-1)
    enddo
  end function

  integer function psum(v) result(e)
    integer, intent(in) :: v(:)
    integer :: count_nonzero
    count_nonzero=count(v/=zero_bound)
    e=zero_bound
    if(count_nonzero>0) e=maxval(v,mask=v/=zero_bound)+clog2(count_nonzero)
  end function

  subroutine upper_exp(bits,width,e,finite,iszero)
    integer(ik), intent(in) :: bits
    integer, intent(in) :: width
    integer, intent(out) :: e
    logical, intent(out) :: finite,iszero
    integer :: nf,ne,bias,ex,j
    finite=.false.; iszero=.false.; e=zero_bound
    select case(width)
    case(64)
      nf=52; ne=11; bias=1023
      if(ishft(bits,-64)/=0_ik) return
    case(128)
      nf=112; ne=15; bias=16383
    case default
      return
    end select
    ex=int(ibits(bits,nf,ne))
    if(ex==2**ne-1) return
    finite=.true.
    if(ex/=0) then
      e=ex-bias+1
    else
      do j=nf-1,0,-1
        if(btest(bits,j)) then
          e=2-bias-nf+j
          return
        endif
      enddo
      iszero=.true.
    endif
  end subroutine

  integer function mode_slot(mode) result(k)
    integer,intent(in):: mode
    k=0
    select case(mode)
    case(1); k=1
    case(2); k=2
    case(4); k=3
    case(5); k=4
    end select
  end function

  logical function route_matches(route,n,nw,we,pairing) result(ok)
    integer,intent(in)::route,n,nw,we(nw),pairing(n)
    ok=.false.
    if(route<1.or.route>26) return
    if(n/=route_n(route).or.nw/=route_nw(route)) return
    if(any(we/=route_we(:nw,route))) return
    if(any(pairing/=route_pairing(:n,route))) return
    ok=.true.
  end function

  subroutine build_chord_bounds(route,mode,width,n,external_bits,environment_valid, &
      chord_error,pair_class,have_chord)
    integer,intent(in)::route,mode,width,n
    integer(ik),intent(in)::external_bits(2,0:3,4)
    logical,intent(in)::environment_valid
    integer,intent(out)::chord_error(0:3,n),pair_class(n,n)
    logical,intent(out)::have_chord(n)
    integer:: slot,p,emin,mu,e,s,L,up(0:3,4),dummy,terms(4),total,err(2),m
    logical::finite,iszero
    chord_error=zero_bound;pair_class=0;have_chord=.false.
    if(.not.environment_valid) return
    if(route<1.or.route>26) return
    if(n/=route_n(route)) return
    slot=mode_slot(mode)
    if(slot==0) return
    if(slot<=2) then
      if(width/=64) return
      p=53;emin=-1022
    else
      if(width/=128) return
      p=113;emin=-16382
    endif
    if(.not.route_supported(route,slot)) return
    do e=1,4
      do mu=0,3
        call upper_exp(external_bits(2,mu,e),width,dummy,finite,iszero)
        if(.not.finite.or..not.iszero) return
        call upper_exp(external_bits(1,mu,e),width,up(mu,e),finite,iszero)
        if(.not.finite) return
      enddo
    enddo
    pair_class=route_class(:n,:n,route,slot)
    do s=1,n
      L=route_ops(s,route,slot)
      if(L<0.or.L>1000000) return ! metadata sanity, L*u <<1/2
      do mu=0,3
        terms=zero_bound
        do e=1,4
          m=route_mult(e,s,route,slot)
          if(m<0.or.m>1000000) return
          if(m>0.and.up(mu,e)/=zero_bound) terms(e)=up(mu,e)+clog2(m)
        enddo
        total=psum(terms)
        err=zero_bound
        if(L>0) then
          if(total/=zero_bound) err(1)=1+clog2(L)-p+total
          err(2)=1+clog2(L)+emin-p
        endif
        chord_error(mu,s)=psum(err)
      enddo
    enddo
    have_chord=.true.
  end subroutine

  subroutine finish_difference_bounds(n,width,shift_bits,chord_error,pair_class, &
      have_chord,environment_valid,q_error,have_pair)
    integer,intent(in)::n,width,chord_error(0:3,n),pair_class(n,n)
    integer(ik),intent(in)::shift_bits(0:3,n)
    logical,intent(in)::have_chord(n),environment_valid
    integer,intent(out)::q_error(0:3,n,n)
    logical,intent(out)::have_pair(n,n)
    integer::p,emin,mu,i,j,s,up(0:3,n),terms(4),mag
    logical::finite,iszero
    q_error=zero_bound;have_pair=.false.
    if(.not.environment_valid) return
    select case(width)
    case(64);p=53;emin=-1022
    case(128);p=113;emin=-16382
    case default;return
    end select
    if(any(chord_error/=zero_bound.and. &
        (chord_error < -100000.or.chord_error > 100000))) return
    do s=1,n
      do mu=0,3
        call upper_exp(shift_bits(mu,s),width,up(mu,s),finite,iszero)
        if(.not.finite) return
      enddo
    enddo
    do j=1,n
      do i=1,n
        if(i==j) cycle
        if(.not.have_chord(i).or..not.have_chord(j)) cycle
        if(pair_class(i,j)<1.or.pair_class(i,j)>2) cycle
        do mu=0,3
          terms=[chord_error(mu,i),chord_error(mu,j),emin-p,zero_bound]
          mag=psum([up(mu,i),up(mu,j)])
          if(mag/=zero_bound) terms(4)=mag-p
          q_error(mu,i,j)=psum(terms)
        enddo
        have_pair(i,j)=.true.
      enddo
    enddo
  end subroutine
end module
