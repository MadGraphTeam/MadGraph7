! Source-only fixture template. 8/dp/1 are mechanical tokens.
! The record stub has no reducer dependency and runs after IEEE observation.
program invariant_fixture
  use iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  implicit none
  integer, parameter :: rk=8, nw=1
  type(ieee_flag_type), parameter :: flags(5) = &
       [ieee_invalid,ieee_divide_by_zero,ieee_overflow,ieee_underflow,ieee_inexact]
  type(ieee_round_type), parameter :: rounds(4) = &
       [ieee_nearest,ieee_to_zero,ieee_up,ieee_down]
  character(32) :: command,arg
  integer :: route,caseid,n,roundid,initial,trapid,arm,k,ni
  integer, parameter :: sizes(4)=[1,2,3,6]
  logical :: supported
  call audit_set_osthres(1.0d-8)
  call get_command_argument(1,command)
  if(trim(command)=='bulk') then
    do roundid=1,4
      supported=ieee_support_rounding(rounds(roundid),0.0_rk)
      write(*,'(A,2(1X,I0))') 'ROUND',roundid,merge(1,0,supported)
      if(.not.supported) cycle
      do initial=0,5
        if(initial>0) then
          if(.not.ieee_support_flag(flags(initial),0.0_rk)) then
            write(*,'(A,2(1X,I0))') 'SKIP_FLAG',roundid,initial
            cycle
          endif
        endif
        do ni=1,size(sizes)
          n=sizes(ni)
          do route=1,2
            do caseid=1,merge(6,8,route==1)
              do arm=0,1
                call evaluate(route,caseid,n,roundid,initial,0,arm)
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
    write(*,'(A)') 'COMPLETE'
  elseif(trim(command)=='trap') then
    call get_command_argument(2,arg); read(arg,*) route
    call get_command_argument(3,arg); read(arg,*) caseid
    call get_command_argument(4,arg); read(arg,*) roundid
    call get_command_argument(5,arg); read(arg,*) trapid
    call get_command_argument(6,arg); read(arg,*) arm
    if(route<1.or.route>2.or.caseid<1.or.caseid>8.or.roundid<1.or.roundid>4 &
         .or.trapid<1.or.trapid>5.or.arm<0.or.arm>1) stop 8
    if(.not.ieee_support_rounding(rounds(roundid),0.0_rk) .or. &
         .not.ieee_support_halting(flags(trapid))) then
      write(*,'(A)') 'UNSUPPORTED_TRAP'
      stop
    endif
    call evaluate(route,caseid,2,roundid,0,trapid,arm)
    write(*,'(A)') 'COMPLETE'
  else
    stop 9
  endif
contains
  function special(kindid) result(x)
    integer,intent(in)::kindid
    real(rk)::x
    integer(int64)::w(nw)
    w=0_int64
    if(nw==1) then
      select case(kindid)
      case(1); w(1)=int(z'8000000000000000',int64) ! negative zero
      case(2); w(1)=int(z'7FF0000000000000',int64) ! positive infinity
      case(3); w(1)=int(z'7FF8000000000001',int64) ! quiet NaN
      case(4); w(1)=int(z'7FF0000000000001',int64) ! signaling NaN
      end select
    else
      select case(kindid)
      case(1); w(nw)=int(z'8000000000000000',int64)
      case(2); w(nw)=int(z'7FFF000000000000',int64)
      case(3); w(nw)=int(z'7FFF800000000000',int64); w(1)=1_int64
      case(4); w(nw)=int(z'7FFF000000000000',int64); w(1)=1_int64
      end select
    endif
    x=transfer(w,x)
  end function

  subroutine setup(route,caseid,n,p,m,a)
    integer,intent(in)::route,caseid,n
    real(rk),intent(out)::p(n,0:3),a(n,n)
    complex(rk),intent(out)::m(n)
    integer::i,j
    real(rk)::mass
    p=0.0_rk
    do i=1,n
      p(i,0)=real(i-1,rk)
      mass=merge(real(173.3d0,rk),real(80.419d0,rk),mod(i,2)==1)
      ! The QP adapter receives DP squared masses promoted to QP.
      if(mod(i,2)==1) then
        m(i)=cmplx(real(173.3d0*173.3d0,rk),0.0_rk,rk)
      else
        m(i)=cmplx(real(80.419d0*80.419d0,rk),0.0_rk,rk)
      endif
      do j=1,n
        a(i,j)=real(i+j,rk)
      enddo
    enddo
    if(route==1) then
      select case(caseid)
      case(1) ! synthetic unequal real mass reconstruction
      case(2) ! exact massive on-shell invariant = 1
        m=cmplx(1.0_rk,0.0_rk,rk)
      case(3) ! exact massless on-shell invariant = 0
        m=cmplx(0.0_rk,0.0_rk,rk); p(:,1)=p(:,0)
      case(4) ! negative complex widths; no real-only rewrite
        m=cmplx(4.0_rk,-0.0625_rk,rk)
      case(5) ! inside existing massive on-shell threshold
        m=cmplx(1.0_rk,0.0_rk,rk)
        p(:,0)=p(:,0)*(1.0_rk+scale(1.0_rk,-30))
      case(6) ! outside existing massive on-shell threshold
        m=cmplx(1.0_rk,0.0_rk,rk)
        p(:,0)=p(:,0)*(1.0_rk+scale(1.0_rk,-20))
      end select
      a=0.0_rk
    elseif(n>=2) then
      select case(caseid)
      case(1) ! already symmetric finite entries
      case(2) ! equal opposite signed zero, including negative-zero diagonal
        a(1,1)=special(1); a(1,2)=0.0_rk; a(2,1)=special(1)
      case(3) ! finite unequal pair
        a(1,2)=1.0_rk; a(2,1)=nearest(1.0_rk,-1.0_rk)
      case(4) ! lower-only infinity
        a(2,1)=special(2)
      case(5) ! lower-only quiet NaN
        a(2,1)=special(3)
      case(6) ! a nonfinite entry must prevent repair of another finite pair
        a(1,1)=special(2); a(1,2)=1.0_rk; a(2,1)=2.0_rk
      case(7) ! upper signaling NaN
        a(1,2)=special(4)
      case(8) ! lower signaling NaN
        a(2,1)=special(4)
      end select
    endif
  end subroutine

  subroutine state(values)
    integer,intent(out)::values(11)
    integer::i
    logical::v
    type(ieee_round_type)::r
    do i=1,5
      call ieee_get_flag(flags(i),v); values(i)=merge(1,0,v)
      call ieee_get_halting_mode(flags(i),v); values(i+5)=merge(1,0,v)
    enddo
    call ieee_get_rounding_mode(r)
    values(11)=0
    do i=1,4
      if(r==rounds(i)) values(11)=i
    enddo
  end subroutine

  subroutine words(tag,i,j,x)
    character(*),intent(in)::tag
    integer,intent(in)::i,j
    real(rk),intent(in)::x
    integer(int64)::w(nw)
    w=transfer(x,w)
    write(*,'(A,2(1X,I0),*(1X,Z16.16))') tag,i,j,w
  end subroutine

  subroutine record_stub(n,a,status)
    ! Mirrors only bridge finite/exact-equality validation, not its computation.
    integer,intent(in)::n
    real(rk),intent(in)::a(n,n)
    integer,intent(out)::status
    integer::i,j
    status=0
    if(.not.all(ieee_is_finite(a))) then
      status=1
      return
    endif
    do j=2,n
      do i=1,j-1
        if(a(i,j)/=a(j,i)) status=2
      enddo
    enddo
  end subroutine

  subroutine evaluate(route,caseid,n,roundid,initial,trapid,arm)
    integer,intent(in)::route,caseid,n,roundid,initial,trapid,arm
    real(rk)::p(n,0:3),a(n,n),ain(n,n),flat(6*n)
    complex(rk)::m(n)
    integer(int64)::before(6*n*nw),after(6*n*nw)
    integer::pre(11),post(11),i,j,status
    do i=1,5
      if(ieee_support_halting(flags(i))) call ieee_set_halting_mode(flags(i),.false.)
    enddo
    call ieee_set_rounding_mode(ieee_nearest)
    call setup(route,caseid,n,p,m,a)
    ain=a
    flat(1:4*n)=reshape(p,[4*n])
    flat(4*n+1:5*n)=real(m,rk); flat(5*n+1:6*n)=aimag(m)
    before=transfer(flat,before)
    ! Integer-only input serialization precedes the measured state boundary.
    write(*,'(A,8(1X,I0))') 'H',rk,route,caseid,n,roundid,initial,trapid,arm
    do j=1,n
      do i=1,n
        call words('A',i,j,ain(i,j))
      enddo
    enddo
    write(*,'(A,*(1X,Z16.16))') 'I',before
    call ieee_set_flag(ieee_all,.false.)
    call ieee_set_rounding_mode(rounds(roundid))
    if(initial>0) call ieee_set_flag(flags(initial),.true.)
    if(trapid>0) call ieee_set_halting_mode(flags(trapid),.true.)
    call state(pre)
    write(*,'(A,*(1X,I0))') 'E',pre
    flush(6)
    if(route==1) then
      if(arm==0) then
        call audit_dp_original(n,p,m,a)
      else
        call audit_dp_candidate(n,p,m,a)
      endif
    elseif(arm==1) then
      call audit_dp_canonical(n,a)
    endif
    call state(post)
    ! Capture before disabling traps, output, or calling the validation stub.
    do i=1,5
      if(ieee_support_halting(flags(i))) call ieee_set_halting_mode(flags(i),.false.)
    enddo
    write(*,'(A,*(1X,I0))') 'X',post
    flat(1:4*n)=reshape(p,[4*n])
    flat(4*n+1:5*n)=real(m,rk); flat(5*n+1:6*n)=aimag(m)
    after=transfer(flat,after)
    write(*,'(A,*(1X,Z16.16))') 'J',after
    do j=1,n
      do i=1,n
        call words('W',i,j,a(i,j))
      enddo
    enddo
    call record_stub(n,a,status)
    write(*,'(A,1X,I0)') 'S',status
    write(*,'(A)') 'END'
    call ieee_set_flag(ieee_all,.false.)
  end subroutine
end program
