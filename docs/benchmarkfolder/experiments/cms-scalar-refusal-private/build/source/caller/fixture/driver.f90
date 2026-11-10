program checked_ct_mock
  use iso_fortran_env,only:int64
  use ieee_arithmetic
  use cms_scalar_observer
  use cts_checked_protocol
  use cts_checked_reduction
  use cts_checked_loopfunctions
  use mock_state
  use scale
  use dimensions
  use combinatorics
  use mbnvalues
  use loopfunctions
  use mp_loopfunctions
  implicit none
  type(cts_checked_report) :: report
  complex(8) :: mass(0:3),amp(0:2),acc,r1
  real(8) :: momenta(0:3,0:3)
  integer :: i,model,case_id,mode,expected_status,expected_precision,expected_direction,expected_refusals
  integer :: expected_coeff(2,2),expected_scalar(2,2),first(4),unit
  logical :: stable,valid_expected,stable_expected,rescued_expected
  real(8) :: lscale,pl(0:3,4),mu_r
  integer :: ctmode
  logical :: fpe_dp,fpe_qp,ctinit,tirinit,goleminit,samuraiinit,ninjainit,collierinit
  common/FX_CT/lscale,ctmode
  common/FX_FPE_IN_REDUCTION/fpe_dp,fpe_qp
  common/FX_MODEL/mu_r
  common/REDUCTIONCODEINIT/ctinit,tirinit,goleminit,samuraiinit,ninjainit,collierinit
  interface
    subroutine FX_CTLOOP_CHECKED(n,pl,mass,rank,result,stable,status)
      use cts_checked_protocol,only:cts_checked_report
      integer :: n,rank
      real(8) :: pl(0:3,n)
      complex(8) :: mass(n),result(3)
      logical :: stable
      type(cts_checked_report) :: status
    end subroutine
  end interface
  external numerator,mpnumerator
  allocate(aloopfun(0:2,6),bloopfun(0:2,6),b1loopfun(0:2,6),b11loopfun(0:2,6),cloopfun(0:2,6),dloopfun(0:2,6))
  allocate(mp_aloopfun(0:2,6),mp_bloopfun(0:2,6),mp_b1loopfun(0:2,6),mp_b11loopfun(0:2,6),mp_cloopfun(0:2,6),mp_dloopfun(0:2,6))
  nbn1(1)=1
  bn1(1,1,1)=0
  nbn1(2)=2
  bn1(2,1,1)=0
  bn1(2,1,2)=1
  nbn2(2)=1
  bn2(2,1,1)=0
  bn2(2,2,1)=1
  nbn1(3)=3
  bn1(3,1,1)=0
  bn1(3,1,2)=1
  bn1(3,1,3)=2
  nbn2(3)=3
  bn2(3,1,1)=0
  bn2(3,2,1)=1
  bn2(3,1,2)=0
  bn2(3,2,2)=2
  bn2(3,1,3)=1
  bn2(3,2,3)=2
  nbn3(3)=1
  bn3(3,1,1)=0
  bn3(3,2,1)=1
  bn3(3,3,1)=2
  nbn1(4)=4
  bn1(4,1,1)=0
  bn1(4,1,2)=1
  bn1(4,1,3)=2
  bn1(4,1,4)=3
  nbn2(4)=6
  bn2(4,1,1)=0
  bn2(4,2,1)=1
  bn2(4,1,2)=0
  bn2(4,2,2)=2
  bn2(4,1,3)=0
  bn2(4,2,3)=3
  bn2(4,1,4)=1
  bn2(4,2,4)=2
  bn2(4,1,5)=1
  bn2(4,2,5)=3
  bn2(4,1,6)=2
  bn2(4,2,6)=3
  nbn3(4)=4
  bn3(4,1,1)=0
  bn3(4,2,1)=1
  bn3(4,3,1)=2
  bn3(4,1,2)=0
  bn3(4,2,2)=1
  bn3(4,3,2)=3
  bn3(4,1,3)=0
  bn3(4,2,3)=2
  bn3(4,3,3)=3
  bn3(4,1,4)=1
  bn3(4,2,4)=2
  bn3(4,3,4)=3
  nbn4(4)=1
  bn4(4,1,1)=0
  bn4(4,2,1)=1
  bn4(4,3,1)=2
  bn4(4,4,1)=3
  open(newunit=unit,file='caller-mock.log',status='new',action='write')
  write(unit,'(A)') 'FORMAT CTS_CHECKED_MOCK_V1'
  momenta=0
  do model=0,1
    do i=0,3
      mass(i)=cmplx(i+1,-model*0.125d0,kind=8)
      momenta(1,i)=real(i,8)
    enddo
    case_id=1
    mode=0
    rules=0;nrules=0
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=8
    expected_direction=1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.false.
    first=[0,0,0,0]
    expected_coeff=reshape([1,0,1,0],[2,2])
    expected_scalar=reshape([15,0,15,0],[2,2])
    call run_case()
    case_id=2
    mode=1
    rules=0;nrules=0
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=8
    expected_direction=1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.false.
    first=[0,0,0,0]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,0,0],[2,2])
    call run_case()
    case_id=3
    mode=2
    rules=0;nrules=0
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=8
    expected_direction=-1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.false.
    first=[0,0,0,0]
    expected_coeff=reshape([0,0,1,0],[2,2])
    expected_scalar=reshape([0,0,15,0],[2,2])
    call run_case()
    case_id=4
    mode=3
    rules=0;nrules=0
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=8
    expected_direction=1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.false.
    first=[0,0,0,0]
    expected_coeff=reshape([1,0,1,0],[2,2])
    expected_scalar=reshape([15,0,15,0],[2,2])
    call run_case()
    case_id=5
    mode=4
    rules=0;nrules=0
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.false.
    first=[0,0,0,0]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([0,15,0,0],[2,2])
    call run_case()
    case_id=6
    mode=5
    rules=0;nrules=0
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=-1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.false.
    first=[0,0,0,0]
    expected_coeff=reshape([0,0,0,1],[2,2])
    expected_scalar=reshape([0,0,0,15],[2,2])
    call run_case()
    case_id=7
    mode=6
    rules=0;nrules=0
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.false.
    first=[0,0,0,0]
    expected_coeff=reshape([0,1,0,1],[2,2])
    expected_scalar=reshape([0,15,0,15],[2,2])
    call run_case()
    case_id=8
    mode=1
    rules=0;nrules=1
    rules(:,1)=[8,1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,1,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([1,0,0,0],[2,2])
    call run_case()
    case_id=9
    mode=1
    rules=0;nrules=1
    rules(:,1)=[8,1,1,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,1,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([2,0,0,0],[2,2])
    call run_case()
    case_id=10
    mode=1
    rules=0;nrules=1
    rules(:,1)=[8,1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([5,0,0,0],[2,2])
    call run_case()
    case_id=11
    mode=1
    rules=0;nrules=1
    rules(:,1)=[8,1,2,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,2,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([6,0,0,0],[2,2])
    call run_case()
    case_id=12
    mode=1
    rules=0;nrules=1
    rules(:,1)=[8,1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,3,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([11,0,0,0],[2,2])
    call run_case()
    case_id=13
    mode=1
    rules=0;nrules=1
    rules(:,1)=[8,1,3,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,3,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([12,0,0,0],[2,2])
    call run_case()
    case_id=14
    mode=1
    rules=0;nrules=1
    rules(:,1)=[8,1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,4,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([15,0,0,0],[2,2])
    call run_case()
    case_id=15
    mode=2
    rules=0;nrules=1
    rules(:,1)=[8,-1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,1,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,1,0],[2,2])
    call run_case()
    case_id=16
    mode=2
    rules=0;nrules=1
    rules(:,1)=[8,-1,1,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,1,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,2,0],[2,2])
    call run_case()
    case_id=17
    mode=2
    rules=0;nrules=1
    rules(:,1)=[8,-1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,5,0],[2,2])
    call run_case()
    case_id=18
    mode=2
    rules=0;nrules=1
    rules(:,1)=[8,-1,2,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,2,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,6,0],[2,2])
    call run_case()
    case_id=19
    mode=2
    rules=0;nrules=1
    rules(:,1)=[8,-1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,3,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,11,0],[2,2])
    call run_case()
    case_id=20
    mode=2
    rules=0;nrules=1
    rules(:,1)=[8,-1,3,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,3,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,12,0],[2,2])
    call run_case()
    case_id=21
    mode=2
    rules=0;nrules=1
    rules(:,1)=[8,-1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,4,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,15,0],[2,2])
    call run_case()
    case_id=22
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,1,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([1,0,0,0],[2,2])
    call run_case()
    case_id=23
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,1,1,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,1,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([2,0,0,0],[2,2])
    call run_case()
    case_id=24
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([5,0,0,0],[2,2])
    call run_case()
    case_id=25
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,1,2,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,2,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([6,0,0,0],[2,2])
    call run_case()
    case_id=26
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,3,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([11,0,0,0],[2,2])
    call run_case()
    case_id=27
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,1,3,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,3,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([12,0,0,0],[2,2])
    call run_case()
    case_id=28
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,4,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([15,0,0,0],[2,2])
    call run_case()
    case_id=29
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,-1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,1,1]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,1,0],[2,2])
    call run_case()
    case_id=30
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,-1,1,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,1,2]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,2,0],[2,2])
    call run_case()
    case_id=31
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,-1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,2,1]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,5,0],[2,2])
    call run_case()
    case_id=32
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,-1,2,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,2,2]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,6,0],[2,2])
    call run_case()
    case_id=33
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,-1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,3,1]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,11,0],[2,2])
    call run_case()
    case_id=34
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,-1,3,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,3,2]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,12,0],[2,2])
    call run_case()
    case_id=35
    mode=3
    rules=0;nrules=1
    rules(:,1)=[8,-1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=8
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,-1,4,1]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,15,0],[2,2])
    call run_case()
    case_id=36
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,1,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,1,0,0],[2,2])
    call run_case()
    case_id=37
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,1,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,1,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,2,0,0],[2,2])
    call run_case()
    case_id=38
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=39
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,6,0,0],[2,2])
    call run_case()
    case_id=40
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,3,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,11,0,0],[2,2])
    call run_case()
    case_id=41
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,3,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,3,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,12,0,0],[2,2])
    call run_case()
    case_id=42
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,4,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,15,0,0],[2,2])
    call run_case()
    case_id=43
    mode=5
    rules=0;nrules=1
    rules(:,1)=[16,-1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,1,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,0,1],[2,2])
    call run_case()
    case_id=44
    mode=5
    rules=0;nrules=1
    rules(:,1)=[16,-1,1,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,1,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,0,2],[2,2])
    call run_case()
    case_id=45
    mode=5
    rules=0;nrules=1
    rules(:,1)=[16,-1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,0,5],[2,2])
    call run_case()
    case_id=46
    mode=5
    rules=0;nrules=1
    rules(:,1)=[16,-1,2,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,2,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,0,6],[2,2])
    call run_case()
    case_id=47
    mode=5
    rules=0;nrules=1
    rules(:,1)=[16,-1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,3,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,0,11],[2,2])
    call run_case()
    case_id=48
    mode=5
    rules=0;nrules=1
    rules(:,1)=[16,-1,3,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,3,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,0,12],[2,2])
    call run_case()
    case_id=49
    mode=5
    rules=0;nrules=1
    rules(:,1)=[16,-1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,4,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,0,0,15],[2,2])
    call run_case()
    case_id=50
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,1,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,1,0,0],[2,2])
    call run_case()
    case_id=51
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,1,1,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,1,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,2,0,0],[2,2])
    call run_case()
    case_id=52
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=53
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,1,2,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,6,0,0],[2,2])
    call run_case()
    case_id=54
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,3,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,11,0,0],[2,2])
    call run_case()
    case_id=55
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,1,3,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,3,2]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,12,0,0],[2,2])
    call run_case()
    case_id=56
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,4,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,15,0,0],[2,2])
    call run_case()
    case_id=57
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,-1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,1,1]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([0,15,0,1],[2,2])
    call run_case()
    case_id=58
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,-1,1,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,1,2]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([0,15,0,2],[2,2])
    call run_case()
    case_id=59
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,-1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,2,1]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([0,15,0,5],[2,2])
    call run_case()
    case_id=60
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,-1,2,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,2,2]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([0,15,0,6],[2,2])
    call run_case()
    case_id=61
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,-1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,3,1]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([0,15,0,11],[2,2])
    call run_case()
    case_id=62
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,-1,3,2]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,3,2]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([0,15,0,12],[2,2])
    call run_case()
    case_id=63
    mode=6
    rules=0;nrules=1
    rules(:,1)=[16,-1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,-1,4,1]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([0,15,0,15],[2,2])
    call run_case()
    case_id=64
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,1,1,1]
    expected_coeff=reshape([0,1,0,1],[2,2])
    expected_scalar=reshape([1,15,0,15],[2,2])
    call run_case()
    case_id=65
    mode=0
    rules=0;nrules=2
    rules(:,1)=[8,1,1,1]
    rules(:,2)=[16,1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=2
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,1,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([1,1,0,0],[2,2])
    call run_case()
    case_id=66
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,1,2,1]
    expected_coeff=reshape([0,1,0,1],[2,2])
    expected_scalar=reshape([5,15,0,15],[2,2])
    call run_case()
    case_id=67
    mode=0
    rules=0;nrules=2
    rules(:,1)=[8,1,2,1]
    rules(:,2)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=2
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([5,5,0,0],[2,2])
    call run_case()
    case_id=68
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,1,3,1]
    expected_coeff=reshape([0,1,0,1],[2,2])
    expected_scalar=reshape([11,15,0,15],[2,2])
    call run_case()
    case_id=69
    mode=0
    rules=0;nrules=2
    rules(:,1)=[8,1,3,1]
    rules(:,2)=[16,1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=2
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,3,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([11,11,0,0],[2,2])
    call run_case()
    case_id=70
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,1,4,1]
    expected_coeff=reshape([0,1,0,1],[2,2])
    expected_scalar=reshape([15,15,0,15],[2,2])
    call run_case()
    case_id=71
    mode=0
    rules=0;nrules=2
    rules(:,1)=[8,1,4,1]
    rules(:,2)=[16,1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=2
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,4,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([15,15,0,0],[2,2])
    call run_case()
    case_id=72
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,-1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,-1,1,1]
    expected_coeff=reshape([1,1,0,1],[2,2])
    expected_scalar=reshape([15,15,1,15],[2,2])
    call run_case()
    case_id=73
    mode=0
    rules=0;nrules=2
    rules(:,1)=[8,1,1,1]
    rules(:,2)=[16,-1,1,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=2
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,1,1]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([1,15,0,1],[2,2])
    call run_case()
    case_id=74
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,-1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,-1,2,1]
    expected_coeff=reshape([1,1,0,1],[2,2])
    expected_scalar=reshape([15,15,5,15],[2,2])
    call run_case()
    case_id=75
    mode=0
    rules=0;nrules=2
    rules(:,1)=[8,1,2,1]
    rules(:,2)=[16,-1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=2
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,2,1]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([5,15,0,5],[2,2])
    call run_case()
    case_id=76
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,-1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,-1,3,1]
    expected_coeff=reshape([1,1,0,1],[2,2])
    expected_scalar=reshape([15,15,11,15],[2,2])
    call run_case()
    case_id=77
    mode=0
    rules=0;nrules=2
    rules(:,1)=[8,1,3,1]
    rules(:,2)=[16,-1,3,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=2
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,3,1]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([11,15,0,11],[2,2])
    call run_case()
    case_id=78
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,-1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,-1,4,1]
    expected_coeff=reshape([1,1,0,1],[2,2])
    expected_scalar=reshape([15,15,15,15],[2,2])
    call run_case()
    case_id=79
    mode=0
    rules=0;nrules=2
    rules(:,1)=[8,1,4,1]
    rules(:,2)=[16,-1,4,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=-1
    expected_refusals=2
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[8,1,4,1]
    expected_coeff=reshape([0,1,0,0],[2,2])
    expected_scalar=reshape([15,15,0,15],[2,2])
    call run_case()
    case_id=80
    mode=0
    rules=0;nrules=0
    force_dp_unstable=.true.
    forced_status=1
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[0,0,0,0]
    expected_coeff=reshape([1,1,1,1],[2,2])
    expected_scalar=reshape([15,15,15,15],[2,2])
    call run_case()
    case_id=81
    mode=1
    rules=0;nrules=0
    force_dp_unstable=.true.
    forced_status=1
    expected_status=0
    expected_precision=8
    expected_direction=1
    expected_refusals=0
    valid_expected=.true.
    stable_expected=.false.
    rescued_expected=.false.
    first=[0,0,0,0]
    expected_coeff=reshape([1,0,0,0],[2,2])
    expected_scalar=reshape([15,0,0,0],[2,2])
    call run_case()
    case_id=82
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=0
    expected_status=6
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=83
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=1
    expected_status=1
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=84
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=2
    expected_status=2
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=85
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=3
    expected_status=3
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=86
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=4
    expected_status=4
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=87
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=5
    expected_status=5
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=88
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=6
    expected_status=6
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=89
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=7
    expected_status=7
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=90
    mode=4
    rules=0;nrules=1
    rules(:,1)=[16,1,2,1]
    force_dp_unstable=.false.
    forced_status=8
    expected_status=8
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.false.
    stable_expected=.false.
    rescued_expected=.false.
    first=[16,1,2,1]
    expected_coeff=reshape([0,0,0,0],[2,2])
    expected_scalar=reshape([0,5,0,0],[2,2])
    call run_case()
    case_id=91
    mode=0
    rules=0;nrules=1
    rules(:,1)=[8,1,2,1]
    force_dp_unstable=.false.
    forced_status=8
    expected_status=0
    expected_precision=16
    expected_direction=1
    expected_refusals=1
    valid_expected=.true.
    stable_expected=.true.
    rescued_expected=.true.
    first=[8,1,2,1]
    expected_coeff=reshape([0,1,0,1],[2,2])
    expected_scalar=reshape([5,15,0,15],[2,2])
    call run_case()
  enddo
  write(unit,'(A,1X,I0)') 'COMPLETE',182
  close(unit)
contains
  subroutine run_case()
    coefficient_calls=0;amplitude_calls=0;scalar_calls=0;top_count=0
    aloopfun=777;bloopfun=777;b1loopfun=777;b11loopfun=777;cloopfun=777;dloopfun=777
    mp_aloopfun=777;mp_bloopfun=777;mp_b1loopfun=777;mp_b11loopfun=777;mp_cloopfun=777;mp_dloopfun=777
    amp=888;acc=888;r1=888
    call ctsxcut_checked(mode,1d0,1000d0,4,numerator,mpnumerator,4,momenta,mass,amp,acc,r1,stable,report)
    if(report%status/=expected_status.or.(report%result_valid.neqv.valid_expected)) error stop 1
    if(stable.neqv.stable_expected) error stop 2
    if(report%precision/=expected_precision.or.report%direction/=expected_direction) error stop 3
    if(report%scalar_refusals/=expected_refusals.or.(report%internally_rescued.neqv.rescued_expected)) error stop 4
    if(any(coefficient_calls/=expected_coeff).or.any(amplitude_calls/=expected_coeff)) error stop 5
    if(any(scalar_calls/=expected_scalar)) error stop 6
    if(allocated(bbn1).or.allocated(bbn2).or.allocated(bbn3).or.allocated(bbn4)) error stop 7
    if(valid_expected) then
      if(any(amp/=[cmplx(16d0,0d0,8),cmplx(30d0,0d0,8),cmplx(45d0,0d0,8)])) error stop 8
      if(acc/=cmplx(15d0,0d0,8).or.r1/=cmplx(1d0,0d0,8)) error stop 9
    else
      if(any(.not.ieee_is_nan(real(amp,8)))) error stop 10
      if(.not.ieee_is_nan(real(acc,8)).or..not.ieee_is_nan(real(r1,8))) error stop 11
      if(checked_dp_valid.or.checked_qp_valid) error stop 12
    endif
    if(expected_refusals>0) then
      if(any([report%first_refusal%precision,report%first_refusal%direction, &
              report%first_refusal%topology,report%first_refusal%subtopology]/=first)) error stop 13
      if(report%first_refusal%count/=1.or.report%first_refusal%events(1)%source_line/=123) error stop 14
      if(report%first_refusal%events(1)%original_value/=cmplx(1.0_16,0.0_16,16)) error stop 15
    else
      if(report%first_refusal%count/=0.or.report%first_refusal%native_called) error stop 16
    endif
    write(unit,'(A,2(1X,I0),3(1X,L1),4(1X,I0))') 'CASE',model,case_id, &
      stable,report%result_valid,report%internally_rescued,report%status,report%precision,report%direction,report%scalar_refusals
    write(unit,'(A,12(1X,I0))') 'COUNTS',coefficient_calls,amplitude_calls,scalar_calls
    ! The same fault is passed through the rendered real optimized MadLoop
    ! adapter. Its integer observer proves rejected output skips normalization.
    coefficient_calls=0;amplitude_calls=0;scalar_calls=0;normalization_count=0
    ctmode=mode;lscale=1d0;mu_r=1000d0;pl=0
    ctinit=.false.;tirinit=.false.;goleminit=.false.;samuraiinit=.false.;ninjainit=.false.;collierinit=.false.
    fpe_dp=.false.;fpe_qp=.false.
    call FX_CTLOOP_CHECKED(4,pl,mass,4,amp,stable,report)
    if(report%status/=expected_status.or.(report%result_valid.neqv.valid_expected)) error stop 20
    if(stable.neqv.stable_expected) error stop 21
    if(valid_expected) then
      if(fpe_dp.or.fpe_qp.or.normalization_count/=1) error stop 22
    else
      if(normalization_count/=0) error stop 23
      if(mode<4.and.(.not.fpe_dp.or.fpe_qp)) error stop 24
      if(mode>=4.and.(.not.fpe_qp.or.fpe_dp)) error stop 25
      if(any(.not.ieee_is_nan(real(amp,8)))) error stop 26
    endif
    if(any(coefficient_calls/=expected_coeff).or.any(amplitude_calls/=expected_coeff)) error stop 27
    if(any(scalar_calls/=expected_scalar)) error stop 28
    write(unit,'(A,2(1X,I0),2(1X,L1),1X,I0)') 'ADAPTER',model,case_id,fpe_dp,fpe_qp,normalization_count
    ! Every case is immediately followed by a success using the same process,
    ! modules and allocated scalar storage: stale refusal state must not survive.
    nrules=0;force_dp_unstable=.false.
    call ctsxcut_checked(1,1d0,1000d0,4,numerator,mpnumerator,4,momenta,mass,amp,acc,r1,stable,report)
    if(.not.stable.or..not.report%result_valid.or.report%status/=CMS_OK) error stop 17
    if(report%scalar_refusals/=0.or.report%first_refusal%count/=0) error stop 18
    if(allocated(bbn1).or.allocated(bbn2).or.allocated(bbn3).or.allocated(bbn4)) error stop 19
    write(unit,'(A,2(1X,I0))') 'RESET_OK',model,case_id
  end subroutine
end program
subroutine numerator(q,result)
  complex(8) :: q(0:3),result
  error stop 'Mock coefficient provider must not call numerator'
end subroutine
subroutine mpnumerator(q,result)
  complex(16) :: q(0:3),result
  error stop 'Mock coefficient provider must not call MP numerator'
end subroutine
subroutine FX_INITCT()
  error stop 'Private adapter fixture must use initialized mock state'
end subroutine
subroutine cite(a,b)
  character(*) :: a,b
  error stop 'Private adapter fixture must not initialize CutTools'
end subroutine
subroutine FX_LOOPNUM(q,result)
  complex(8) :: q(0:3),result
  error stop 'Mock coefficient provider must not call numerator'
end subroutine
subroutine FX_MPLOOPNUM(q,result)
  complex(16) :: q(0:3),result
  error stop 'Mock coefficient provider must not call MP numerator'
end subroutine
