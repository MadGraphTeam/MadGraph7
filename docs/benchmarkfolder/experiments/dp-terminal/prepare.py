#!/usr/bin/env python3
"""Prepare an isolated rank-1 x rank-1 DP experiment; never edit production."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path.cwd()
REPORT = ROOT / 'docs/benchmarkfolder/experiments/dp-terminal'
WORK = ROOT / '.tirust-build/dp-terminal'
ORIGINAL = ROOT / '.benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z'
SUBPROCESS = ORIGINAL / 'SubProcesses/P0_uux_z'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical_source(text):
    lines = []
    for line in text.splitlines():
        if not line.strip() or line[:1] in ('C', 'c', '*', '!'):
            continue
        if line[5:6] not in (' ', '0'):
            lines[-1] += line[6:]
        else:
            lines.append(line[6:])
    return '\n'.join(lines) + '\n'


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    source = SUBPROCESS / 'polynomial.f'
    compilation = json.loads((ORIGINAL / 'compilation.json').read_text())
    relative = str(source.relative_to(ORIGINAL))
    assert compilation['sources'][relative] == digest(source)
    dense = re.search(r'^      SUBROUTINE ML5_0_UPDATE_WL_1_1\(.*?^      END\s*$',
                      source.read_text(), re.S | re.M).group(0) + '\n'
    dense = logical_source(dense)
    body = dense[dense.index('DO I=1,LCUT_SIZE'):].removesuffix('END\n')
    assert body.count('DO J=1,OUT_SIZE') == 2
    first, second = body.split('DO J=1,OUT_SIZE', 1)
    diagonal = first + 'DO J=1,OUT_SIZE' + second.replace('DO J=1,OUT_SIZE', 'DO J=I,I', 1)
    module = '''module ML5_0_POLYNOMIAL_CONSTANTS
implicit none
integer, parameter :: MAXLWFSIZE=4, LOOPMAXCOEFS=15, VERTEXMAXCOEFS=5
end module
'''
    oracle = dense.replace('ML5_0_UPDATE_WL_1_1', 'DP_DENSE', 1)
    header = '''subroutine DP_TERMINAL(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT,ENTRY_INEXACT,FAST)
use ML5_0_POLYNOMIAL_CONSTANTS
use, intrinsic :: iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: A(4,0:14,4), B(4,0:4,4), OUT(4,0:14,4)
integer :: LCUT_SIZE, IN_SIZE, OUT_SIZE, I, J, K, H
logical, intent(in) :: ENTRY_INEXACT
logical, intent(out) :: FAST
logical :: HALTING(size(IEEE_ALL))
type(IEEE_ROUND_TYPE) :: ROUNDING
FAST=.false.
! The private caller proves diagonal consumption; runtime proof is narrow.
if (.not.ENTRY_INEXACT) goto 900
if (LCUT_SIZE/=4.or.IN_SIZE/=4.or.OUT_SIZE/=4) goto 900
call IEEE_GET_ROUNDING_MODE(ROUNDING)
if (ROUNDING/=IEEE_NEAREST) goto 900
! Conservative admission: no enabled exception trap of any kind.
do H=1,size(IEEE_ALL)
  if (.not.IEEE_SUPPORT_HALTING(IEEE_ALL(H))) goto 900
enddo
call IEEE_GET_HALTING_MODE(IEEE_ALL,HALTING)
if (any(HALTING)) goto 900
! Integer classification cannot signal on exceptional operand bit patterns.
if (.not.all(BOUNDED(real(A(:,0:4,:),kind=8)))) goto 900
if (.not.all(BOUNDED(aimag(A(:,0:4,:))))) goto 900
if (.not.all(BOUNDED(real(B,kind=8)))) goto 900
if (.not.all(BOUNDED(aimag(B)))) goto 900
FAST=.true.
'''
    footer = '''return
900 continue
call DP_DENSE(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
contains
elemental logical function BOUNDED(X)
real(8), intent(in) :: X
integer(int64), parameter :: ONE_WORDS(1)=transfer(1.0_8,(/0_int64/))
BOUNDED=.false.
if (storage_size(X)/=64.or.digits(X)/=53.or.radix(X)/=2.or.maxexponent(X)/=1024) return
if (ONE_WORDS(1)/=shiftl(1023_int64,52)) return
associate (WORDS=>transfer(X,ONE_WORDS))
  if (ibclr(WORDS(1),63)==0_int64) then
    BOUNDED=.true.
  else
    BOUNDED=ibits(WORDS(1),52,11)>=623_int64.and.ibits(WORDS(1),52,11)<=1423_int64
  endif
end associate
end function
end subroutine
'''
    (REPORT / 'oracle.f90').write_text(module + oracle)
    (REPORT / 'terminal.f90').write_text(header + diagonal + footer)
    # Reuse the independent dense/sparse IEEE corpus, frozen into this package.
    test = ROOT / 'tests/unit_tests/various/test_q_polynomial_compiled.py'
    tree = ast.parse(test.read_text())
    driver = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                      t.id == '_SPARSE_DRIVER' for t in node.targets))
    driver = driver.replace('TEST_POLYNOMIAL_CONSTANTS', 'ML5_0_POLYNOMIAL_CONSTANTS')
    driver = driver.replace('0:14', '0:4').replace('0,14', '0,4')
    driver = driver.replace('0:209', '0:14').replace('0,209', '0,14')
    driver = driver.replace('4*4*15*4', '4*4*5*2').replace('4*4*210*4', '4*4*15*2')
    driver = driver.replace('(16)', '(8)').replace('_16', '_8').replace('kind=16', 'kind=8')
    driver = driver.replace('1.e2000', '1.e200').replace('1.e-2000', '1.e-200')
    driver = driver.replace('integer :: i,j,k,ls,ins,outs,trial,rounding,checks',
        'integer :: i,j,k,ls,ins,outs,trial,rounding,checks,flagstate,fast_count,fallback_count,f\n'
        'logical :: entry,took_fast\ntype(ieee_round_type) :: after_rounding')
    driver = driver.replace('checks=0\n', 'checks=0\nfast_count=0\nfallback_count=0\ndo flagstate=0,2**size(ieee_all)-1\n')
    driver = driver.replace('do trial=1,12', 'do trial=1,24')
    driver = driver.replace('if(trial==12) a(i,j,k)=0', '''if(trial==12) a(i,j,k)=0
          if(trial==13) a(i,j,k)=cmplx(2.0_8**400,-2.0_8**400,kind=8)
          if(trial==14) a(i,j,k)=cmplx(2.0_8**(-400),2.0_8**(-400),kind=8)
          if(trial==15) a(i,j,k)=cmplx(2.0_8**401,0.0_8,kind=8)
          if(trial==16) a(i,j,k)=cmplx(2.0_8**(-401),0.0_8,kind=8)
          if(trial==17) a(i,j,k)=cmplx(2.0_8**400,2.0_8**400,kind=8)
          if(trial==18) a(i,j,k)=cmplx(nearest(2.0_8**401,-1.0_8), &
            -nearest(2.0_8**401,-1.0_8),kind=8)
          if(trial==19) a(i,j,k)=cmplx(nearest(2.0_8**(-400),1.0_8),2.0_8**(-400),kind=8)
          if(trial==20) a(i,j,k)=cmplx(nearest(2.0_8**(-400),-1.0_8),0.0_8,kind=8)
          if(trial==21) a(i,j,k)=cmplx(-0.0_8,0.0_8,kind=8)
          if(trial==22.and.i==4) a(i,j,k)=cmplx(small,0.0_8,kind=8)
          if(trial==23.and.i==4) a(i,j,k)=cmplx(inf,0.0_8,kind=8)
          if(trial==24.and.i==4) a(i,j,k)=cmplx(snan,0.0_8,kind=8)''')
    driver = driver.replace('if(trial==12) b(i,j,k)=cmplx(snan,0.0_8,kind=8)', '''if(trial==12) b(i,j,k)=cmplx(snan,0.0_8,kind=8)
          if(trial==13) b(i,j,k)=cmplx(2.0_8**400,2.0_8**400,kind=8)
          if(trial==14.or.trial==17) b(i,j,k)=cmplx(2.0_8**(-400),-2.0_8**(-400),kind=8)
          if(trial==15.or.trial==16.or.trial==20) b(i,j,k)=cmplx(1.0_8,0.0_8,kind=8)
          if(trial==18) b(i,j,k)=cmplx(nearest(2.0_8**401,-1.0_8), &
            nearest(2.0_8**401,-1.0_8),kind=8)
          if(trial==19) b(i,j,k)=cmplx(2.0_8**(-400),-2.0_8**(-400),kind=8)''')
    driver = driver.replace('call ieee_set_flag(ieee_all,.false.)',
        'call ieee_set_flag(ieee_all,.false.)\ndo f=1,size(ieee_all)\n'
        'if(btest(flagstate,f-1)) call ieee_set_flag(ieee_all(f),.true.)\nenddo')
    driver = driver.replace('OLD_ROUTINE', 'DP_DENSE')
    driver = driver.replace('call NEW_ROUTINE(a,ls,b,ins,outs,actual)',
        'call ieee_get_flag(ieee_inexact,entry)\n'
        'call DP_TERMINAL(a,ls,b,ins,outs,actual,entry,took_fast)')
    driver = driver.replace('if(any(old_bits/=new_bits)) then', '''if(took_fast) then
  fast_count=fast_count+1
  if(.not.entry.or.rounding/=1.or.ls/=4.or.ins/=4.or.outs/=4) stop 9
  do i=1,4
    if(any(transfer(expected(i,:,i),old_bits,15*2)/= &
      transfer(actual(i,:,i),new_bits,15*2))) then
      write(*,*) 'CONSUMED BITS',trial,rounding,flagstate,i
      stop 6
    endif
  enddo
else
  fallback_count=fallback_count+1
endif
if(.not.took_fast.and.any(old_bits/=new_bits)) then''')
    driver = driver.replace('checks=checks+1', '''call ieee_get_rounding_mode(after_rounding)
if(after_rounding/=modes(rounding)) stop 8
checks=checks+1''')
    driver = driver.replace('call ieee_set_rounding_mode(ieee_nearest)\nif(checks/=6000)',
        'enddo\ncall ieee_set_rounding_mode(ieee_nearest)\nif(checks/=12000*2**size(ieee_all))')
    driver = driver.replace("write(*,*) 'PASS',checks", "if(fast_count==0.or.fallback_count==0) stop 7\nwrite(*,*) 'PASS',checks,fast_count,fallback_count")
    (REPORT / 'guard-driver.f90').write_text(driver)
    paths = [source, SUBPROCESS / 'coef_construction_1.f', ORIGINAL / 'compilation.json', test, Path(__file__)]
    record = {'scope':'Isolated DP terminal rank1x1 experiment; no production edits or QP changes',
              'inputs':{str(p.relative_to(ROOT)):digest(p) for p in paths},
              'generated':{p.name:digest(p) for p in REPORT.glob('*.f90')},
              'shape':[4,4,4], 'nonzero_component_unbiased_exponents':[-400,400],
              'guard_requires':'nearest rounding, preexisting INEXACT, supported disabled halting for all flags, binary64 layout',
              'candidate_operation_change':'Only second (arithmetic) J loop changes 1:OUT_SIZE to I:I; all zero initialization and dense fallback retained',
              'comparison_scope':'Consumed diagonal bit patterns; full bit patterns on fallback; all input bits, IEEE flags, and rounding modes'}
    (REPORT / 'preparation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print('Prepared isolated sources; no native execution')


if __name__ == '__main__':
    main()
