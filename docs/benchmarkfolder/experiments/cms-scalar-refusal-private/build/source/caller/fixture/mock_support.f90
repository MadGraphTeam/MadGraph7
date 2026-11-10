! Synthetic numerical dependencies only; actual checked CT control flow is separate.
module mpmodule
end module
module mock_state
  use cms_scalar_observer
  implicit none
  integer :: rules(4,2)=0,nrules=0,active_direction=1,top_count(4)=0,forced_status=1,normalization_count=0
  integer :: coefficient_calls(2,2)=0,amplitude_calls(2,2)=0,scalar_calls(2,2)=0
  logical :: force_dp_unstable=.false.
contains
  subroutine mock_report(report,precision,topology)
    type(cms_scalar_report),intent(out) :: report
    integer,intent(in) :: precision,topology
    integer :: i,pi,di
    pi=1;if(precision==16) pi=2
    di=1;if(active_direction==-1) di=2
    scalar_calls(pi,di)=scalar_calls(pi,di)+1
    top_count(topology)=top_count(topology)+1
    call cms_reset(report,precision,topology,1)
    do i=1,nrules
      if(all(rules(:,i)==[precision,active_direction,topology,top_count(topology)])) then
        call cms_record(report,123,CMS_INPUT,CMS_ABS_REAL,1,0,'mock refusal', &
             1.0_16,cmplx(1.0_16,0.0_16,16),cmplx(0.0_16,0.0_16,16),2.0_16,3.0_16)
        report%status=forced_status
        if(forced_status==8) report%native_called=.true.
        return
      endif
    enddo
    report%native_called=.true.;report%result_valid=.true.
  end subroutine
  subroutine mock_count_amp(precision)
    integer,intent(in) :: precision
    integer :: pi,di
    pi=1;if(precision==16) pi=2
    di=1;if(active_direction==-1) di=2
    amplitude_calls(pi,di)=amplitude_calls(pi,di)+1
  end subroutine
end module
module scale
  implicit none
  real(8) :: roots=1,muscale=1,musq=1,limit=1d-8,precstablen=1d-20
  integer :: scaloop=2,n_tot=0,n_mp=0,n_unst=0
  logical :: stablen=.true.
end module
module maxnumden
  integer,parameter :: maxden=4
end module
module mbnvalues
  integer :: bn1(4,4,6)=0,bn2(4,4,6)=0,bn3(4,4,6)=0,bn4(4,4,6)=0
end module
module combinatorics
  integer :: nbn1(4)=0,nbn2(4)=0,nbn3(4)=0,nbn4(4)=0
end module
module dimensions
  integer :: dmns=4,dmns_a=6,dmns_b=6,dmns_c=6,dmns_d=6
  integer :: dmns_1=0,dmns_2=0,dmns_3=0,dmns_4=0
  integer,allocatable :: bbn1(:,:),bbn2(:,:),bbn3(:,:),bbn4(:,:)
end module
module def_propagator
  type propagator
    integer :: i
    complex(8) :: m2
    real(8) :: p(0:3)
  end type
end module
module def_mp_propagator
  type mp_propagator
    integer :: i
    complex(16) :: m2
    real(16) :: qt,p(0:3)
  end type
end module
module denominators
  use def_propagator
  use def_mp_propagator
  use mock_state
  implicit none
  type(propagator) :: den(0:3)
  type(mp_propagator) :: mp_den(0:3)
  interface load_denominators
    module procedure load_dp,load_qp
  end interface
contains
  subroutine load_dp(values,n)
    integer,intent(in) :: n
    type(propagator),intent(in) :: values(0:n-1)
    den(0:n-1)=values
    active_direction=1;if(real(values(0)%m2,8)>2d0) active_direction=-1
    top_count=0
  end subroutine
  subroutine load_qp(values,n)
    integer,intent(in) :: n
    type(mp_propagator),intent(in) :: values(0:n-1)
    mp_den(0:n-1)=values
    active_direction=1;if(real(values(0)%m2,16)>2.0_16) active_direction=-1
    top_count=0
  end subroutine
end module
module tensor_operations
  interface contr
    module procedure contr_dp,contr_qp
  end interface
contains
  subroutine contr_dp(a,b,out)
    real(8),intent(in) :: a(0:3),b(0:3)
    real(8),intent(out) :: out
    out=a(0)*b(0)-sum(a(1:3)*b(1:3))
  end subroutine
  subroutine contr_qp(a,b,out)
    real(16),intent(in) :: a(0:3),b(0:3)
    real(16),intent(out) :: out
    out=a(0)*b(0)-sum(a(1:3)*b(1:3))
  end subroutine
end module
module loopfunctions
  complex(8),allocatable :: aloopfun(:,:),bloopfun(:,:),b1loopfun(:,:),b11loopfun(:,:),cloopfun(:,:),dloopfun(:,:)
end module
module mp_loopfunctions
  complex(16),allocatable :: mp_aloopfun(:,:),mp_bloopfun(:,:),mp_b1loopfun(:,:),mp_b11loopfun(:,:),mp_cloopfun(:,:),mp_dloopfun(:,:)
end module
module avh_olo_forCutTools
contains
  subroutine olo_scale(value)
    real(8),intent(in) :: value
  end subroutine
end module
module coefficients
  use scale
  use mock_state
  implicit none
  complex(8) :: save_acoeff(0:0,6)=1,save_bcoeff(0:6,6)=0,save_ccoeff(0:0,6)=1,save_dcoeff(0:0,6)=1,save_rat1=1
  complex(16) :: save_mp_acoeff(0:0,6)=1,save_mp_bcoeff(0:6,6)=0,save_mp_ccoeff(0:0,6)=1,save_mp_dcoeff(0:0,6)=1,save_mp_rat1=1
  real(8) :: vveck1(6)=0
  real(16) :: mp_vveck1(6)=0
  interface get_coefficients
    module procedure coefficients_dp,coefficients_qp
  end interface
contains
  subroutine coefficients_dp(prec,num,n,dmr,root)
    real(8),intent(out) :: prec
    real(8),intent(in) :: root
    integer,intent(in) :: n,dmr
    external num
    integer :: di
    di=1;if(active_direction==-1) di=2
    coefficient_calls(1,di)=coefficient_calls(1,di)+1
    save_bcoeff(0,:)=1
    stablen=.not.force_dp_unstable;precstablen=1d-20;prec=precstablen
  end subroutine
  subroutine coefficients_qp(prec,num,n,dmr,root)
    real(16),intent(out) :: prec
    real(8),intent(in) :: root
    integer,intent(in) :: n,dmr
    external num
    integer :: di
    di=1;if(active_direction==-1) di=2
    coefficient_calls(2,di)=coefficient_calls(2,di)+1
    save_mp_bcoeff(0,:)=1
    stablen=.true.;precstablen=1d-20;prec=precstablen
  end subroutine
end module
module avh_olo_forCutTools_dp
  use cms_scalar_observer
  use mock_state
  implicit none
contains
  subroutine cms_checked_a0_c(rslt,mm,report)
    complex(8),intent(inout) :: rslt(0:2)
    complex(8),intent(in) :: mm
    type(cms_scalar_report),intent(out) :: report
    call mock_report(report,8,1)
    if(report%status.ne.CMS_OK) return
    rslt=[cmplx(1,0,8),cmplx(2,0,8),cmplx(3,0,8)]
  end subroutine
  subroutine cms_checked_b11rc(b11,b00,b1,b0,pp,m1,m2,report)
    complex(8),intent(inout) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
    complex(8),intent(in) :: m1,m2
    real(8),intent(in) :: pp
    type(cms_scalar_report),intent(out) :: report
    call mock_report(report,8,2)
    if(report%status.ne.CMS_OK) return
    b11=[cmplx(1,0,8),cmplx(2,0,8),cmplx(3,0,8)]
    b00=[cmplx(1,0,8),cmplx(2,0,8),cmplx(3,0,8)]
    b1=[cmplx(1,0,8),cmplx(2,0,8),cmplx(3,0,8)]
    b0=[cmplx(1,0,8),cmplx(2,0,8),cmplx(3,0,8)]
  end subroutine
  subroutine cms_checked_c0rc(rslt,p1,p2,p3,m1,m2,m3,report)
    complex(8),intent(inout) :: rslt(0:2)
    complex(8),intent(in) :: m1,m2,m3
    real(8),intent(in) :: p1,p2,p3
    type(cms_scalar_report),intent(out) :: report
    call mock_report(report,8,3)
    if(report%status.ne.CMS_OK) return
    rslt=[cmplx(1,0,8),cmplx(2,0,8),cmplx(3,0,8)]
  end subroutine
  subroutine cms_checked_d0rc(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,report)
    complex(8),intent(inout) :: rslt(0:2)
    complex(8),intent(in) :: m1,m2,m3,m4
    real(8),intent(in) :: p1,p2,p3,p4,p12,p23
    type(cms_scalar_report),intent(out) :: report
    call mock_report(report,8,4)
    if(report%status.ne.CMS_OK) return
    rslt=[cmplx(1,0,8),cmplx(2,0,8),cmplx(3,0,8)]
  end subroutine
end module
module avh_olo_forCutTools_qp
  use cms_scalar_observer
  use mock_state
  implicit none
contains
  subroutine cms_checked_a0_c(rslt,mm,report)
    complex(16),intent(inout) :: rslt(0:2)
    complex(16),intent(in) :: mm
    type(cms_scalar_report),intent(out) :: report
    call mock_report(report,16,1)
    if(report%status.ne.CMS_OK) return
    rslt=[cmplx(1,0,16),cmplx(2,0,16),cmplx(3,0,16)]
  end subroutine
  subroutine cms_checked_b11rc(b11,b00,b1,b0,pp,m1,m2,report)
    complex(16),intent(inout) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
    complex(16),intent(in) :: m1,m2
    real(16),intent(in) :: pp
    type(cms_scalar_report),intent(out) :: report
    call mock_report(report,16,2)
    if(report%status.ne.CMS_OK) return
    b11=[cmplx(1,0,16),cmplx(2,0,16),cmplx(3,0,16)]
    b00=[cmplx(1,0,16),cmplx(2,0,16),cmplx(3,0,16)]
    b1=[cmplx(1,0,16),cmplx(2,0,16),cmplx(3,0,16)]
    b0=[cmplx(1,0,16),cmplx(2,0,16),cmplx(3,0,16)]
  end subroutine
  subroutine cms_checked_c0rc(rslt,p1,p2,p3,m1,m2,m3,report)
    complex(16),intent(inout) :: rslt(0:2)
    complex(16),intent(in) :: m1,m2,m3
    real(16),intent(in) :: p1,p2,p3
    type(cms_scalar_report),intent(out) :: report
    call mock_report(report,16,3)
    if(report%status.ne.CMS_OK) return
    rslt=[cmplx(1,0,16),cmplx(2,0,16),cmplx(3,0,16)]
  end subroutine
  subroutine cms_checked_d0rc(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,report)
    complex(16),intent(inout) :: rslt(0:2)
    complex(16),intent(in) :: m1,m2,m3,m4
    real(16),intent(in) :: p1,p2,p3,p4,p12,p23
    type(cms_scalar_report),intent(out) :: report
    call mock_report(report,16,4)
    if(report%status.ne.CMS_OK) return
    rslt=[cmplx(1,0,16),cmplx(2,0,16),cmplx(3,0,16)]
  end subroutine
end module
