program performance
use iso_fortran_env, only:int64
use ieee_arithmetic
use ieee_exceptions
implicit none
include 'params_free.inc'
real*8 p(4,6),out(4,5),reference(4,5),edge(16),first,last,checksum,rr(4,6)
real*16 qp(4,6),qout(4,5),qreference(4,5),qedge(16)
complex*16 coefs_ordered(0:209,1)
common/ORDERED_LCOEFS/coefs_ordered
complex*16 m(6),res(3),previous(3),coefs_base(0:209,1),coefs_current(0:209,1)
complex*16 coefs_early(0:209,1),coefs_linear(0:209,1)
common/BASE_LCOEFS/coefs_base
common/CURRENT_LCOEFS/coefs_current
common/EARLY_LCOEFS/coefs_early
common/LINEAR_LCOEFS/coefs_linear
logical do_(1,0:2)
common/ORDERED_TIRCACHING/do_
logical db(1,0:2),dc(1,0:2),de(1,0:2),dl(1,0:2),stable,init(6)
common/BASE_TIRCACHING/db
common/CURRENT_TIRCACHING/dc
common/EARLY_TIRCACHING/de
common/LINEAR_TIRCACHING/dl
common/REDUCTIONCODEINIT/init
integer libordered
common/ORDERED_I_LIB/libordered
integer libbase,libcurrent,libearly,liblinear
common/BASE_I_LIB/libbase
common/CURRENT_I_LIB/libcurrent
common/EARLY_I_LIB/libearly
common/LINEAR_I_LIB/liblinear
real*8 lsordered
integer ctordered
common/ORDERED_CT/lsordered,ctordered
real*8 lsbase,lscurrent,lsearly,lslinear
integer ctbase,ctcurrent,ctearly,ctlinear
common/BASE_CT/lsbase,ctbase
common/CURRENT_CT/lscurrent,ctcurrent
common/EARLY_CT/lsearly,ctearly
common/LINEAR_CT/lslinear,ctlinear
integer n,rank,miss_period,count,i,j,k,v,mode,caseid,failures,total,c
integer(int64) seed,bits,qbits(2)
character*32 kind,domain,variant,arg
type(ieee_round_type) rounds(4)
logical flags_old(5),flags_new(5)
type(ieee_flag_type) flags(5)
call get_command_argument(1,kind)
call ieee_set_halting_mode(ieee_all,.false.)
if(trim(kind)=='verify') then
  seed=1234567_int64
  edge=[0d0,-0d0,tiny(1d0),-tiny(1d0),nearest(0d0,1d0),-nearest(0d0,1d0), &
        huge(1d0),-huge(1d0),1d0,-1d0,epsilon(1d0),-epsilon(1d0), &
        ieee_value(0d0,ieee_positive_inf),ieee_value(0d0,ieee_negative_inf), &
        ieee_value(0d0,ieee_quiet_nan),ieee_value(0d0,ieee_signaling_nan)]
  qedge=[0.0_16,-0.0_16,tiny(1.0_16),-tiny(1.0_16),nearest(0.0_16,1.0_16),-nearest(0.0_16,1.0_16), &
        huge(1.0_16),-huge(1.0_16),1.0_16,-1.0_16,epsilon(1.0_16),-epsilon(1.0_16), &
        ieee_value(0.0_16,ieee_positive_inf),ieee_value(0.0_16,ieee_negative_inf), &
        ieee_value(0.0_16,ieee_quiet_nan),ieee_value(0.0_16,ieee_signaling_nan)]
  rounds=[ieee_nearest,ieee_up,ieee_down,ieee_to_zero]
  flags=[ieee_invalid,ieee_overflow,ieee_underflow,ieee_inexact,ieee_divide_by_zero]
  failures=0
  total=0
  do mode=1,4
    call ieee_set_rounding_mode(rounds(mode))
    do n=1,6
      do caseid=1,10032
        do j=1,n
          do i=1,4
            if(caseid<=32) then
              p(i,j)=edge(mod(caseid+i+j,16)+1)
              qp(i,j)=qedge(mod(caseid+i+j,16)+1)
            else
              call rng(seed,bits)
              p(i,j)=transfer(bits,p(i,j))
              call rng(seed,qbits(1))
              call rng(seed,qbits(2))
              qp(i,j)=transfer(qbits,qp(i,j))
            endif
          enddo
        enddo
        call ieee_set_flag(ieee_all,.false.)
        call prefix_dp_old(n,p(:,1:n),reference(:,1:n-1))
        call ieee_get_flag(flags,flags_old)
        call ieee_set_flag(ieee_all,.false.)
        call prefix_dp_new(n,p(:,1:n),out(:,1:n-1))
        call ieee_get_flag(flags,flags_new)
        if(any(transfer(reference(:,1:n-1),[0_int64],4*(n-1))/=transfer(out(:,1:n-1),[0_int64],4*(n-1)))) failures=failures+1
        if(any(flags_old.neqv.flags_new)) failures=failures+1
        call ieee_set_flag(ieee_all,.false.)
        call prefix_qp_old(n,qp(:,1:n),qreference(:,1:n-1))
        call ieee_get_flag(flags,flags_old)
        call ieee_set_flag(ieee_all,.false.)
        call prefix_qp_new(n,qp(:,1:n),qout(:,1:n-1))
        call ieee_get_flag(flags,flags_new)
        if(any(transfer(qreference(:,1:n-1),[0_int64],8*(n-1))/=transfer(qout(:,1:n-1),[0_int64],8*(n-1)))) failures=failures+1
        if(any(flags_old.neqv.flags_new)) failures=failures+1
        total=total+2
      enddo
    enddo
  enddo
  call ieee_set_rounding_mode(ieee_nearest)
  print *, 'prefix bit/exception verification cases=',total,'failures=',failures
  if(failures/=0) stop 91
  stop
endif
call get_command_argument(2,arg)
read(arg,*) n
call get_command_argument(3,domain)
call get_command_argument(4,arg)
read(arg,*) miss_period
call get_command_argument(5,variant)
call get_command_argument(6,arg)
read(arg,*) count
rank=n
MLReductionLib=0
MLReductionLib(1)=7
LoopPoleMode=-1
COLLIERComputeUVpoles=.true.
COLLIERComputeIRpoles=.true.
COLLIERUseInternalStabilityTest=.false.
init=.false.
libordered=1
libbase=1
libcurrent=1
libearly=1
liblinear=1
ctordered=1
ctbase=1
ctcurrent=1
ctearly=1
ctlinear=1
coefs_base=cmplx(0.75d0,0.25d0,8)
coefs_ordered=coefs_base
coefs_current=coefs_base
coefs_early=coefs_base
coefs_linear=coefs_base
do_=.false.
db=.false.
dc=.false.
de=.false.
dl=.false.
do j=1,n-1
do i=1,4
p(i,j)=sin(real(i*j,8))*0.1d0
enddo
enddo
p(:,n)=-sum(p(:,1:n-1),dim=2)
qp=real(p,16)
m=cmplx(1d0,0d0,8)
checksum=0d0
call cpu_time(first)
do k=1,count
  if(kind=='prefix') then
    if(domain=='dp') then
      if(variant=='old') then
        call prefix_dp_old(n,p(:,1:n),out(:,1:n-1))
      else
        call prefix_dp_new(n,p(:,1:n),out(:,1:n-1))
      endif
      checksum=checksum+out(1,n-1)
    else
      if(variant=='old') then
        call prefix_qp_old(n,qp(:,1:n),qout(:,1:n-1))
      else
        call prefix_qp_new(n,qp(:,1:n),qout(:,1:n-1))
      endif
      checksum=checksum+real(qout(1,n-1),8)
    endif
  else
    if(miss_period>0) then
      if(mod(k-1,miss_period)==0) then
        do_=.false.
db=.false.
        dc=.false.
        de=.false.
        dl=.false.
      endif
    endif
    select case(variant)
    case('ordered')
      call ORDERED_TIRLOOP(1,1,1,n,p,m,rank,res,stable)
    case('base')
      call BASE_TIRLOOP(1,1,1,n,p,m,rank,res,stable)
    case('current')
      call CURRENT_TIRLOOP(1,1,1,n,p,m,rank,res,stable)
    case('early')
      call EARLY_TIRLOOP(1,1,1,n,p,m,rank,res,stable)
    case('linear')
      call LINEAR_TIRLOOP(1,1,1,n,p,m,rank,res,stable)
    end select
    checksum=checksum+real(res(1),8)
  endif
enddo
call cpu_time(last)
print '(ES24.16E3,1X,ES24.16E3)',last-first,checksum
contains
subroutine rng(state,value)
integer(int64) state,value
state=ieor(state,ishft(state,13))
state=ieor(state,ishft(state,-7))
state=ieor(state,ishft(state,17))
value=state
end subroutine
end program

subroutine ML_LOOP_POLE_FLAGS(uv,ir)
logical uv,ir
uv=.true.
ir=.true.
end
