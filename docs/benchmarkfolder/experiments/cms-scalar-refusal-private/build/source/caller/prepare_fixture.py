#!/usr/bin/env python3
"""Prepare pure-Fortran mock execution of the full extracted checked CT flow."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
from unittest import mock
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import prepare as p
MOCK=HERE/'fixture'
COMPILER='/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran'
PYTHON='/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12'

SUPPORT='''! Synthetic numerical dependencies only; actual checked CT control flow is separate.
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
'''

def mock_scalar(precision):
    suffix='dp' if precision==8 else 'qp'
    text='module avh_olo_forCutTools_'+suffix+'\n  use cms_scalar_observer\n  use mock_state\n  implicit none\ncontains\n'
    for topology,name,outputs,inputs in [(1,'a0_c',['rslt'],['mm']),
        (2,'b11rc',['b11','b00','b1','b0'],['pp','m1','m2']),
        (3,'c0rc',['rslt'],['p1','p2','p3','m1','m2','m3']),
        (4,'d0rc',['rslt'],['p1','p2','p3','p4','p12','p23','m1','m2','m3','m4'])]:
        text+='  subroutine cms_checked_'+name+'('+','.join(outputs+inputs+['report'])+')\n'
        text+='    complex('+str(precision)+'),intent(inout) :: '+','.join(x+'(0:2)' for x in outputs)+'\n'
        masses=[x for x in inputs if x.startswith('m')];invariants=[x for x in inputs if x.startswith('p')]
        text+='    complex('+str(precision)+'),intent(in) :: '+','.join(masses)+'\n'
        if invariants:text+='    real('+str(precision)+'),intent(in) :: '+','.join(invariants)+'\n'
        text+='    type(cms_scalar_report),intent(out) :: report\n'
        text+='    call mock_report(report,'+str(precision)+','+str(topology)+')\n'
        text+='    if(report%status.ne.CMS_OK) return\n'
        for out in outputs:text+='    '+out+'=[cmplx(1,0,'+str(precision)+'),cmplx(2,0,'+str(precision)+'),cmplx(3,0,'+str(precision)+')]\n'
        text+='  end subroutine\n'
    return text+'end module\n'

def expected(mode,rules,force=False,fault_status=1):
    coeff=[0]*4;scalar=[0]*4;refused=[];rescue=False;valid=True;last_precision=8;last_direction=1
    def stage(precision,direction):
        index=(0 if precision==8 else 1)+(0 if direction==1 else 2)
        for top,count in enumerate((4,6,4,1),1):
            for sub in range(1,count+1):
                scalar[index]+=1
                if [precision,direction,top,sub] in rules:
                    refused.append([precision,direction,top,sub]);return False
        coeff[index]+=1;return True
    if mode<=3:
        dirs={0:[1,-1],1:[1],2:[-1],3:[1,-1]}[mode]
        for direction in dirs:
            last_direction=direction
            if not stage(8,direction):valid=False;break
        if mode==0 and (not valid or force):rescue=True;valid=True
    if mode>=4 or rescue:
        last_precision=16
        dirs={4:[1],5:[-1],6:[1,-1]}.get(mode,[1,-1])
        for direction in dirs:
            last_direction=direction
            if not stage(16,direction):valid=False;break
    stable=valid and not(force and not rescue)
    selected=1 if mode in (0,3,6) and valid else last_direction
    return dict(status=0 if valid else (fault_status if fault_status!=0 else 6),valid=valid,stable=stable,precision=last_precision,
                direction=selected,refusals=len(refused),rescue=rescue and valid,
                coefficient=coeff,amplitude=coeff[:],scalar=scalar,first=refused[0] if refused else [0]*4)

def cases():
    result=[]
    def add(name,mode,rules=[],force=False,fault_status=1):
        result.append(dict(name=name,mode=mode,rules=rules,force=force,fault_status=fault_status,expected=expected(mode,rules,force,fault_status)))
    for mode in range(7):add('success_'+str(mode),mode)
    for mode,precision,dirs in [(1,8,[1]),(2,8,[-1]),(3,8,[1,-1]),(4,16,[1]),(5,16,[-1]),(6,16,[1,-1])]:
        for direction in dirs:
            for topo in range(1,5):
                for sub in ([1,2] if topo<4 else [1]):
                    add('refuse_%s_%s_%s_%s'%(mode,direction,topo,sub),mode,[[precision,direction,topo,sub]])
    for direction in (1,-1):
        for topo in range(1,5):
            add('internal_rescue_%s_%s'%(direction,topo),0,[[8,direction,topo,1]])
            add('internal_failure_%s_%s'%(direction,topo),0,[[8,1,topo,1],[16,direction,topo,1]])
    add('legacy_numerator_rescue',0,[],True)
    add('legacy_numerator_unstable',1,[],True)
    for status in range(9):add('status_'+str(status),4,[[16,1,2,1]],fault_status=status)
    add('nonfinite_dp_internal_rescue',0,[[8,1,2,1]],fault_status=8)
    return result

DRIVER='''program checked_ct_mock
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
@COMBINATORICS@
  open(newunit=unit,file='caller-mock.log',status='new',action='write')
  write(unit,'(A)') 'FORMAT CTS_CHECKED_MOCK_V1'
  momenta=0
  do model=0,1
    do i=0,3
      mass(i)=cmplx(i+1,-model*0.125d0,kind=8)
      momenta(1,i)=real(i,8)
    enddo
@CASES@
  enddo
  write(unit,'(A,1X,I0)') 'COMPLETE',@COUNT@
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
'''

def main():
    assert Path.cwd().resolve()==ROOT
    original={str(path.relative_to(HERE)):p.sha(path) for path in (HERE/'src').glob('*.f90')}
    observer=HERE.parent/'classifier/cms_scalar_observer.f90'
    p.write(MOCK/'cms_scalar_observer.f90',observer.read_text())
    p.write(MOCK/'mock_support.f90',SUPPORT+mock_scalar(8)+mock_scalar(16))
    for name in ('cts_dpr.h','cts_dpc.h','cts_mprec.h','cts_mpr.h','cts_mpc.h'):
        p.write(MOCK/name,(p.CTS/name).read_text())
    reduction=(HERE/'src/cts_checked_reduction.f90').read_text()
    reduction=reduction.replace('  use cts_checked_protocol','  use mock_state,only:mock_count_amp\n  use cts_checked_protocol',1)
    reduction=reduction.replace('    camp = 0.d0','    call mock_count_amp(8)\n    camp = 0.d0',1)
    reduction=reduction.replace('    do k= 0,2; mp_camp(k)= 0.d0; enddo','    call mock_count_amp(16)\n    do k= 0,2; mp_camp(k)= 0.d0; enddo',1)
    p.write(MOCK/'cts_checked_reduction_instrumented.f90',reduction)
    sys.path.insert(0,str(ROOT))
    from madgraph.iolibs import file_writers
    import tempfile
    adapter=(HERE/'templates/loop_optimized/CT_interface_checked.inc').read_text()
    adapter=adapter.replace('      USE CTS_CHECKED_REDUCTION','      USE MOCK_STATE, ONLY: NORMALIZATION_COUNT\n      USE CTS_CHECKED_REDUCTION',1)
    adapter=adapter.replace('## if(AmplitudeReduction) {','      NORMALIZATION_COUNT=NORMALIZATION_COUNT+1\n## if(AmplitudeReduction) {',1)
    mapping={key:1 if fmt=='d' else '' for key,fmt in re.findall(r'%\((\w+)\)([ds])',adapter)}
    mapping.update(proc_prefix='FX_',nexternal=3,real_dp_format='REAL*8',complex_dp_format='COMPLEX*16',mass_dp_format='COMPLEX*16')
    with tempfile.TemporaryDirectory(dir=HERE,prefix='render-') as temporary:
        output=Path(temporary)/'adapter.f'
        with file_writers.FortranWriter(str(output)) as stream:stream.writelines(adapter%mapping,context={'AmplitudeReduction':False})
        p.write(MOCK/'adapter_instrumented.f',output.read_text())
    p.write(MOCK/'coupl.inc','      REAL*8 MU_R\n      COMMON/FX_MODEL/MU_R\n')
    comb=[]
    for n in range(1,5):
        for top in range(1,n+1):
            combos=list(itertools.combinations(range(n),top));comb.append('  nbn%d(%d)=%d'%(top,n,len(combos)))
            for sub,values in enumerate(combos,1):
                for index,value in enumerate(values,1):comb.append('  bn%d(%d,%d,%d)=%d'%(top,n,index,sub,value))
    allcases=cases();body=[];expectedrows=['FORMAT CTS_CHECKED_MOCK_V1']
    logical=lambda value:'.true.' if value else '.false.'
    array=lambda values:'['+','.join(map(str,values))+']'
    for index,case in enumerate(allcases,1):
        e=case['expected']
        body.extend(['    case_id='+str(index),'    mode='+str(case['mode']),'    rules=0;nrules='+str(len(case['rules']))])
        for i,rule in enumerate(case['rules'],1):body.append('    rules(:,%d)='%i+array(rule))
        body.extend(['    force_dp_unstable='+logical(case['force']),
            '    forced_status='+str(case['fault_status']),
            '    expected_status='+str(e['status']), '    expected_precision='+str(e['precision']),
            '    expected_direction='+str(e['direction']), '    expected_refusals='+str(e['refusals']),
            '    valid_expected='+logical(e['valid']), '    stable_expected='+logical(e['stable']),
            '    rescued_expected='+logical(e['rescue']), '    first='+array(e['first']),
            '    expected_coeff=reshape('+array(e['coefficient'])+',[2,2])',
            '    expected_scalar=reshape('+array(e['scalar'])+',[2,2])','    call run_case()'])
    for model in (0,1):
        for index,case in enumerate(allcases,1):
            e=case['expected'];flags=['T' if e[k] else 'F' for k in ('stable','valid','rescue')]
            expectedrows.append(' '.join(map(str,['CASE',model,index,*flags,e['status'],e['precision'],e['direction'],e['refusals']])))
            expectedrows.append(' '.join(map(str,['COUNTS',*e['coefficient'],*e['amplitude'],*e['scalar']])))
            expectedrows.append(' '.join(map(str,['ADAPTER',model,index,'T' if not e['valid'] and case['mode']<4 else 'F',
                'T' if not e['valid'] and case['mode']>=4 else 'F',1 if e['valid'] else 0])))
            expectedrows.append(' '.join(map(str,['RESET_OK',model,index])))
    expectedrows.append('COMPLETE '+str(2*len(allcases)))
    p.write(MOCK/'driver.f90',DRIVER.replace('@COMBINATORICS@','\n'.join(comb)).replace('@CASES@','\n'.join(body)).replace('@COUNT@',str(2*len(allcases))))
    p.write(MOCK/'cases.json',json.dumps(dict(cases=allcases,models=['real_squared_masses_as_complex','complex_squared_masses'],
        expected_journal_rows=expectedrows,scope='Mock scalar/coefficient values; actual checked reduction/control flow'),indent=2,sort_keys=True)+'\n')
    build=MOCK/'build';run=MOCK/'run';build.mkdir(exist_ok=True);run.mkdir(exist_ok=True)
    sources=[MOCK/'cms_scalar_observer.f90',MOCK/'mock_support.f90',HERE/'src/cts_checked_protocol.f90',
             HERE/'src/cts_checked_loopfunctions.f90',MOCK/'cts_checked_reduction_instrumented.f90',MOCK/'adapter_instrumented.f',MOCK/'driver.f90']
    jobs=[];outputs=[]
    for source in sources:
        obj=build/(source.stem+'.o');outputs.append(str(obj))
        jobs.append(dict(name='caller-mock-'+source.stem,argv=[COMPILER,'-O0','-fcheck=all','-fbacktrace',
           '-ffree-line-length-none','-ffixed-line-length-132','-I'+str(MOCK),'-I'+str(build),'-J'+str(build),'-c',str(source),'-o',str(obj)],cwd=p.rel(ROOT)))
    exe=build/'caller_mock'
    jobs.append(dict(name='caller-mock-link',argv=[COMPILER,*outputs,'-o',str(exe)],cwd=p.rel(ROOT)))
    inputs={p.rel(path):p.sha(path) for path in sources+[observer,Path(__file__),MOCK/'cases.json',MOCK/'coupl.inc',HERE/'validate_fixture.py']}
    inputs.update({p.rel(MOCK/name):p.sha(MOCK/name) for name in ('cts_dpr.h','cts_dpc.h','cts_mprec.h','cts_mpr.h','cts_mpc.h')})
    inputs.update({p.rel(HERE/name):digest for name,digest in original.items()})
    manifest=dict(schema='cts-checked-mock-compile-proposal-v1',source_only=True,native_compile_performed=False,native_execution_performed=False,
      inputs=inputs,compiler=COMPILER,build_jobs=jobs,build_outputs=[p.rel(Path(x)) for x in outputs]+[p.rel(exe)],
      build_seconds_max=60,execution_seconds_max=10,cpu=8,retries=0,
      runs=[dict(name='caller-mock-control-flow',argv=[str(exe)],cwd=p.rel(run),timeout_seconds=10,expected_exit=0,expected_outputs=['caller-mock.log'])],
      validation=dict(argv=[PYTHON,'-B',str(HERE/'validate_fixture.py')],cwd=p.rel(ROOT),timeout_seconds=10),
      cases=2*len(allcases),additional_success_reset_calls=2*len(allcases),
      qualification='Mock only; no AVH analytic call, physical integral, full model, real coefficients, runtime or accuracy claim')
    p.write(MOCK/'commands.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    # These three independent objects are compile-only against the actual AVH
    # and CutTools module interfaces, with no mock module directory in search.
    actual=HERE/'actual-build';actual.mkdir(exist_ok=True)
    classifier_actual=HERE.parent/'classifier/actual-build'
    classifier_observer=HERE.parent/'classifier/observer-build'
    real_include=HERE.parent/'classifier/includes'
    actual_jobs=[];actual_outputs=[]
    for source in [HERE/'src/cts_checked_protocol.f90',HERE/'src/cts_checked_loopfunctions.f90',HERE/'src/cts_checked_reduction.f90']:
        output=actual/(source.stem+'.o');actual_outputs.append(str(output))
        actual_jobs.append(dict(name='caller-actual-'+source.stem,argv=[COMPILER,'-O0','-ffree-line-length-none',
           '-J'+str(actual),'-I'+str(actual),'-I'+str(classifier_actual),'-I'+str(classifier_observer),
           '-I'+str(real_include),'-c',str(source),'-o',str(output)],cwd=str(ROOT)))
    classifier_plan_path=HERE.parent/'classifier/compile-plan.json'
    classifier_plan=json.loads(classifier_plan_path.read_text())
    dependencies={str(classifier_plan_path):p.sha(classifier_plan_path)}
    for copy_path,item in classifier_plan['copied_dependencies'].items():
        assert p.sha(Path(copy_path))==item['sha256']
        assert p.sha(Path(item['source']))==item['sha256']
        dependencies[copy_path]=item['sha256'];dependencies[item['source']]=item['sha256']
    dependencies[str(observer)]=p.sha(observer)
    actual_avh=HERE.parent/'classifier/generated/avh_olo_checked.f90'
    dependencies[str(actual_avh)]=p.sha(actual_avh)
    modules=['cms_scalar_observer','mpmodule','mock_state','scale','maxnumden','mbnvalues','combinatorics','dimensions',
      'def_propagator','def_mp_propagator','denominators','tensor_operations','loopfunctions','mp_loopfunctions',
      'avh_olo_forcuttools','coefficients','avh_olo_forcuttools_dp','avh_olo_forcuttools_qp',
      'cts_checked_protocol','cts_checked_loopfunctions','cts_checked_reduction']
    compile_plan=dict(schema='cms-caller-source-only-compile-plan-v1',state='not_authorized_or_executed',
      compiler=dict(path=COMPILER,sha256=p.sha(Path(COMPILER))),cpu=8,jobs_concurrency=1,automatic_retry=False,
      jobs=[dict(job,cwd=str(ROOT/job['cwd'])) for job in jobs]+actual_jobs,
      fixture_sha256={str(ROOT/name):digest for name,digest in inputs.items()},dependency_sha256=dependencies,
      expected_objects=outputs+actual_outputs,expected_executable=str(exe),
      expected_module_outputs=[str(build/(name+'.mod')) for name in modules]+
        [str(actual/(name+'.mod')) for name in ('cts_checked_protocol','cts_checked_loopfunctions','cts_checked_reduction')],
      actual_compile_requires_prior_jobs=['compile-observer','compile-real-AVH-additions'],
      module_search_order=dict(mock=[str(MOCK),str(build)],actual=[str(actual),str(classifier_actual),str(classifier_observer),str(real_include)]),
      actual_objects_linked=False,actual_AVH_native_execution_allowed=False,
      global_compile_link_seconds_max_shared=180,mock_execution_argv=[str(exe)],
      mock_execution_cwd=str(run),mock_execution_seconds_max=10,mock_execution_requires_separate_authorization=True)
    p.write(HERE/'compile-plan.json',json.dumps(compile_plan,indent=2,sort_keys=True)+'\n')
    assert all(p.sha(HERE/name)==digest for name,digest in original.items())
    print(json.dumps(dict(source_only=True,cases=2*len(allcases),commands_sha256=p.sha(MOCK/'commands.json'),compile_plan_sha256=p.sha(HERE/'compile-plan.json'))))

if __name__=='__main__':
    p.REFRESH='--refresh-generated' in sys.argv
    with mock.patch.object(subprocess,'Popen',side_effect=AssertionError('Native launch forbidden')):main()
