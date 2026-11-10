#!/usr/bin/env python3
"""Source-only additive checked CutTools extraction. Never builds or evaluates."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from unittest import mock
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CTS=ROOT/'vendor/CutTools/src/cts'
REFRESH=False

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rel(path):return str(path.relative_to(ROOT))
def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not REFRESH:assert path.read_text()==text,'Changed generated file: '+str(path)
    else:path.write_text(text)

def extract_subroutine(source,name):
    match=re.search(r'(?ims)^\s*subroutine '+name+r'\b.*?^\s*end subroutine '+name+r'\s*$',source)
    assert match,name
    return match.group(0).strip()+'\n'

def loopfunction(source,name,precision,topology):
    text=extract_subroutine(source,name)
    text=text.replace('subroutine '+name,'subroutine '+name+'_checked',2)
    text=text.replace('subroutine '+name+'_checked\n','subroutine '+name+'_checked(direction,report)\n',1)
    at=text.index('   integer :: ib')
    text=text[:at]+('   integer,intent(in) :: direction\n   type(cms_scalar_report),intent(out) :: report\n'+text[at:])
    if precision==16:text=text.replace('   integer :: ib','   integer :: ib,ic',1)
    outer=re.search(r'(?ms)^    if\s*\(.*?^    endif\n',text)
    assert outer,name
    accepted=re.search(r'(?ms)^    (?:else)?if\s*\(scaloop.eq.2\) then\n(.*?)(?=^    elseif \(scaloop.eq.3\) then)',outer.group(0))
    assert accepted,name
    body=accepted.group(1)
    call=re.search(r'(?m)^     call olo\((.*)\)$',body);assert call
    api={1:'a0',2:'b11',3:'c0',4:'d0'}[topology]
    alias=('dp_' if precision==8 else 'qp_')+'checked_'+api
    replacement='     call '+alias+'('+call.group(1)+',report)\n'
    replacement+='     report%direction=direction\n     report%subtopology=ib\n'
    replacement+='     if (.not.cms_report_accepted(report)) then\n'
    replacement+='      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT\n'
    replacement+='      report%result_valid=.false.\n      return\n     endif'
    body=body[:call.start()]+replacement+body[call.end():]
    text=text[:outer.start()]+body+text[outer.end():]
    return text

def loops_module(source):
    text='''! Extracted from the pinned real loop-function routines; checked scaloop=2 only.
module cts_checked_loopfunctions
  include 'cts_mprec.h'
  use denominators
  use scale
  use dimensions
  use maxnumden, only: maxden
  use loopfunctions, only: aloopfun,bloopfun,b1loopfun,b11loopfun,cloopfun,dloopfun
  use mp_loopfunctions, only: mp_aloopfun,mp_bloopfun,mp_b1loopfun,mp_b11loopfun,mp_cloopfun,mp_dloopfun
  use cms_scalar_observer
  use avh_olo_forCutTools_dp, only: dp_checked_a0=>cms_checked_a0_c,dp_checked_b11=>cms_checked_b11rc, &
       dp_checked_c0=>cms_checked_c0rc,dp_checked_d0=>cms_checked_d0rc
  use avh_olo_forCutTools_qp, only: qp_checked_a0=>cms_checked_a0_c,qp_checked_b11=>cms_checked_b11rc, &
       qp_checked_c0=>cms_checked_c0rc,qp_checked_d0=>cms_checked_d0rc
  implicit none
  private
  public :: getloop_checked,get_mp_loop_checked,cts_checked_reset,cms_report_accepted
  public :: checked_dp_valid,checked_qp_valid
  logical,save :: checked_dp_valid=.false.,checked_qp_valid=.false.
contains
  subroutine cts_checked_reset()
    checked_dp_valid=.false.
    checked_qp_valid=.false.
  end subroutine
  logical function cms_report_accepted(report)
    type(cms_scalar_report),intent(in) :: report
    cms_report_accepted=report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called
  end function
'''
    for precision,prefix,entry,valid in [(8,'','getloop_checked','checked_dp_valid'),(16,'get_mp_','get_mp_loop_checked','checked_qp_valid')]:
        names=['get'+t+'loop' for t in 'abcd'] if precision==8 else ['get_mp_'+t+'loop' for t in 'abcd']
        text+='  subroutine '+entry+'(nprop,direction,report)\n'
        text+='    integer,intent(in) :: nprop,direction\n    type(cms_scalar_report),intent(out) :: report\n'
        text+='    report=cms_scalar_report()\n    report%precision='+str(precision)+'\n    report%direction=direction\n'
        text+='    '+valid+'=.false.\n'
        text+='    if(scaloop.ne.2.or.nprop.lt.1.or.nprop.gt.maxden.or.abs(direction).ne.1) then\n      report%status=CMS_OVERLOAD_UNSUPPORTED\n      return\n    endif\n'
        arraynames=(['aloopfun','bloopfun','b1loopfun','b11loopfun','cloopfun','dloopfun'] if precision==8 else
                    ['mp_aloopfun','mp_bloopfun','mp_b1loopfun','mp_b11loopfun','mp_cloopfun','mp_dloopfun'])
        for n,arrays in [(1,arraynames[:1]),(2,arraynames[1:4]),(3,arraynames[4:5]),(4,arraynames[5:])]:
            text+='    if(nprop.ge.'+str(n)+') then\n'
            text+='      if(dmns_'+str(n)+'.lt.1) then\n        report%status=CMS_BAD_INPUT\n        return\n      endif\n'
            for array in arrays:
                text+='      if(.not.allocated('+array+')) then\n        report%status=CMS_BAD_INPUT\n        return\n      endif\n'
                text+='      if(size('+array+',2).lt.dmns_'+str(n)+') then\n        report%status=CMS_BAD_INPUT\n        return\n      endif\n'
            text+='    endif\n'
        for n,name in enumerate(names,1):
            text+='    if(nprop.ge.'+str(n)+') then\n      call '+name+'_checked(direction,report)\n'
            text+='      if(.not.cms_report_accepted(report)) return\n    endif\n'
        text+='    '+valid+'=.true.\n  end subroutine '+entry+'\n'
        for n,name in enumerate(names,1):text+=loopfunction(source,name,precision,n)
    return text+'end module cts_checked_loopfunctions\n'

PROTOCOL='''module cts_checked_protocol
  use cms_scalar_observer
  implicit none
  private
  public :: cts_checked_report
  type cts_checked_report
    integer :: status=CMS_OK,precision=0,direction=0,scalar_refusals=0
    logical :: result_valid=.false.,internally_rescued=.false.
    type(cms_scalar_report) :: first_refusal,last_scalar
  end type
end module cts_checked_protocol
'''

def reduction(source):
    text=source.replace('subroutine ctsxcut(','subroutine ctsxcut_checked(',1)
    text=text.replace('numdummy,mpnumdummy,rnk,p,m2,amp,ampcc,ampr1,stable)',
                      'numdummy,mpnumdummy,rnk,p,m2,amp,ampcc,ampr1,stable,ct_status)',1)
    text=text.replace('end subroutine ctsxcut','end subroutine ctsxcut_checked',1)
    text=text.replace('  use scale','  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan\n  use cts_checked_protocol\n  use cms_scalar_observer\n  use cts_checked_loopfunctions\n  use scale',1)
    text=text.replace('  logical, intent(out) :: stable','  logical, intent(out) :: stable\n  type(cts_checked_report),intent(out) :: ct_status\n  type(cms_scalar_report) :: scalar_report',1)
    anchor='  if (number_propagators.gt.maxden) then'
    initial='''  ct_status=cts_checked_report()
  call cts_checked_reset()
  stable=.false.
  call poison_result()
  if(number_propagators.lt.1.or.number_propagators.gt.maxden.or.imode.lt.0.or.imode.gt.6.or.scaloop.ne.2) then
    ct_status%status=CMS_OVERLOAD_UNSUPPORTED
    return
  endif
'''
    text=text.replace(anchor,initial+anchor,1)
    for precision,entry in [(8,'getloop'),(16,'get_mp_loop')]:
        for direction in (1,-1):
            old='    call '+entry+'(number_propagators)'
            new='    call '+entry+'_checked(number_propagators,'+str(direction)+',scalar_report)\n'
            new+='    ct_status%last_scalar=scalar_report\n'
            valid='checked_dp_valid' if precision==8 else 'checked_qp_valid'
            new+='    if(.not.cms_report_accepted(scalar_report).or..not.'+valid+') then\n      call note_refusal(scalar_report)\n'
            if precision==8:
                new+='      if(imode.eq.0) then\n        passeddp=.false.\n        goto 2000\n      endif\n'
            new+='      goto 9000\n    endif'
            assert old in text;text=text.replace(old,new,1)
    text=text.replace('! The cases when multiprecision is called:',
                      '2000 continue\n! The cases when multiprecision is called:',1)
    anchor='! Deallocating local arrays'
    final='''  ct_status%status=CMS_OK
  ct_status%result_valid=.true.
  ct_status%internally_rescued=(imode.eq.0.and.mpcomputation)
  ct_status%precision=8
  if(mpcomputation) ct_status%precision=16
  ct_status%direction=1
  if(imode.eq.2.or.imode.eq.5) ct_status%direction=-1
  if(imode.eq.0.or.imode.eq.3.or.imode.eq.6) then
    if(.not.dir_stable) ct_status%direction=-1
  endif
  goto 9001
9000 continue
  call tag_as_unstable()
  ct_status%result_valid=.false.
  call poison_result()
  call cts_checked_reset()
9001 continue
'''
    assert text.count(anchor)==1;text=text.replace(anchor,final+anchor,1)
    at='  contains\n'
    helpers='''  subroutine poison_result()
    amp=cmplx(ieee_value(0.d0,ieee_quiet_nan),0.d0,kind=8)
    ampcc=amp(0)
    ampr1=amp(0)
  end subroutine
  subroutine note_refusal(report)
    type(cms_scalar_report),intent(in) :: report
    ct_status%scalar_refusals=ct_status%scalar_refusals+1
    if(ct_status%scalar_refusals.eq.1) ct_status%first_refusal=report
    ct_status%last_scalar=report
    ct_status%status=report%status
    if(ct_status%status.eq.CMS_OK) ct_status%status=CMS_BAD_INPUT
    ct_status%precision=report%precision
    ct_status%direction=report%direction
  end subroutine
'''
    assert text.count(at)==1;text=text.replace(at,at+helpers,1)
    return ('module cts_checked_reduction\n  implicit none\n  private\n  public :: ctsxcut_checked\ncontains\n'+
            text+'\nend module cts_checked_reduction\n')

def adapter(source):
    # Only the real CTLOOP program unit, not unrelated reduction interfaces.
    end=re.search(r'(?m)^\s*END\s*$',source);assert end
    text=source[:end.end()]+'\n'
    text=text.replace('sCTLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)',
                      'sCTLOOP_CHECKED(NLOOPLINE,PL,M2L,RANK,RES,STABLE,CT_STATUS)',1)
    start=text.index('\n',re.search(r'(?m)^\s*SUBROUTINE',text).start())
    text=text[:start]+'''\n      USE CTS_CHECKED_REDUCTION, ONLY: CTSXCUT_CHECKED
      USE CTS_CHECKED_PROTOCOL, ONLY: CTS_CHECKED_REPORT
      USE CMS_SCALAR_OBSERVER, ONLY: CMS_OK
'''+text[start:]
    text=text.replace('C LOCAL VARIABLES', '''C LOCAL VARIABLES''',1)
    declaration='''      TYPE(CTS_CHECKED_REPORT) CT_STATUS
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
'''
    text=text.replace('\t  INTEGER I, J, K','\t  INTEGER I, J, K\n'+declaration,1)
    old=re.search(r'(?m)^      CALL CTSXCUT\(.*$',text);assert old
    call=old.group(0).replace('CALL CTSXCUT(','CALL CTSXCUT_CHECKED(')
    call=call[:-1]+',CT_STATUS)'
    guard='''
C     The existing FPE flag is a hard-refusal latch, not an invented IEEE exception.
C     Select the active MadLoop pass; CT_STATUS records actual internal precision.
      IF(CT_STATUS%STATUS.NE.CMS_OK.OR..NOT.CT_STATUS%RESULT_VALID) THEN
        STABLE=.FALSE.
        IF(CTMODE.GE.4) THEN
          FPE_IN_QP_REDUCTION=.TRUE.
        ELSE
          FPE_IN_DP_REDUCTION=.TRUE.
        ENDIF
        RETURN
      ENDIF
'''
    text=text[:old.start()]+call+guard+text[old.end():]
    return re.sub(r'%(?!\()', '%%', text)

def checked_helas(source):
    text=source.replace('sLOOP_%(nloopline)d%(nwfsargs_header)s(',
                        'sLOOP_%(nloopline)d%(nwfsargs_header)s_CMSCHECKED(',1)
    text=text.replace('      USE ALOHA_OBJECT','      USE CTS_CHECKED_PROTOCOL, ONLY: CTS_CHECKED_REPORT\n      USE CMS_SCALAR_OBSERVER, ONLY: CMS_OK\n      USE ALOHA_OBJECT',1)
    text=text.replace('      LOGICAL complex_mass,doing_qp','      LOGICAL complex_mass,doing_qp\n      TYPE(CTS_CHECKED_REPORT) CT_STATUS',1)
    calls=list(re.finditer(r'(?m)^\s*CALL %\(proc_prefix\)sCTLOOP\(.*$',text))
    assert len(calls)==2
    for call in reversed(calls):
        changed=call.group(0).replace('sCTLOOP(','sCTLOOP_CHECKED(')
        changed=changed[:-1]+',CT_STATUS)\n'
        changed+='      IF(CT_STATUS%STATUS.NE.CMS_OK.OR..NOT.CT_STATUS%RESULT_VALID) RETURN'
        text=text[:call.start()]+changed+text[call.end():]
    # Optimized HELAS helpers use misc.apply_template, which intentionally
    # preserves literal Fortran component access without %% escaping.
    return text

def main():
    assert Path.cwd().resolve()==ROOT
    inputs=[CTS/'cts_loopfunctions.f90',CTS/'cts_cutroutines.f90',CTS/'cts_type.f90',CTS/'cts_combinatorics.f90',
            CTS/'cts_cuttools.f90',*(CTS/name for name in ('cts_mprec.h','cts_mpr.h','cts_mpc.h','cts_dpr.h','cts_dpc.h')),
            ROOT/'madgraph/iolibs/template_files/loop_optimized/CT_interface.inc',
            ROOT/'madgraph/iolibs/template_files/loop/CT_interface.inc',
            ROOT/'madgraph/iolibs/template_files/loop_optimized/helas_loop_amplitude.inc',
            ROOT/'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc']
    hashes={rel(p):sha(p) for p in inputs}
    loop_source=inputs[0].read_text();cut_source=inputs[1].read_text()
    write(HERE/'src/cts_checked_protocol.f90',PROTOCOL)
    write(HERE/'src/cts_checked_loopfunctions.f90',loops_module(loop_source))
    write(HERE/'src/cts_checked_reduction.f90',reduction(cut_source))
    for family in ('loop_optimized','loop'):
        path=ROOT/'madgraph/iolibs/template_files'/family/'CT_interface.inc'
        write(HERE/'templates'/family/'CT_interface_checked.inc',adapter(path.read_text()))
    helper=ROOT/'madgraph/iolibs/template_files/loop_optimized/helas_loop_amplitude.inc'
    write(HERE/'templates/loop_optimized/helas_loop_amplitude_checked.inc',checked_helas(helper.read_text()))
    write(HERE/'source-differences.patch',''.join(difflib.unified_diff(cut_source.splitlines(True),
         (HERE/'src/cts_checked_reduction.f90').read_text().splitlines(True),
         fromfile=rel(inputs[1]),tofile='private/cts_checked_reduction.f90')))
    for p in inputs:assert sha(p)==hashes[rel(p)]
    outputs={str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'src').glob('*'))}
    outputs.update({str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'templates').rglob('*.inc'))})
    outputs['source-differences.patch']=sha(HERE/'source-differences.patch')
    manifest=dict(schema='cms-checked-cuttools-source-v1',source_only=True,native_compile_performed=False,
                  native_execution_performed=False,production_modified=False,inputs=hashes,outputs=outputs,
                  generator_sha256=sha(Path(__file__)),scalar_scope='scaloop=2; complex-mass DP/QP A,B11,C,D; all original cut modes 0..6',
                  numerical_qualification=False)
    write(HERE/'prepared.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(source_only=True,outputs=len(outputs),prepared_sha256=sha(HERE/'prepared.json'))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-generated',action='store_true')
    REFRESH=parser.parse_args().refresh_generated
    with mock.patch.object(subprocess,'Popen',side_effect=AssertionError('Native launch forbidden')):main()
