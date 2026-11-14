"""Contract and IEEE regressions for private generated QP FFV contractions."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines, PolynomialError


def generator(precision='complex*32'):
    return FortranPolynomialRoutines(
        6, updater_max_rank=2, proc_prefix='TEST_', sub_prefix='TEST_',
        coef_format=precision)


def instrument(source):
    source = source.replace('  implicit none', '  implicit none\n'
        '  INTEGER FAST,FALLBACK\n  COMMON/FACTOR_TEST/FAST,FALLBACK', 1)
    source = source.replace('C FACTOR_FAST_PATH',
                            'C FACTOR_FAST_PATH\nFAST=FAST+1', 1)
    source, count = re.subn(
        r'(?m)^(\s*CALL TEST_(?:UPDATE|TERMINAL)_WL_\d_1(?:_DENSE)?\(.*)$',
        r'  FALLBACK=FALLBACK+1\n\1', source)
    assert count >= 5
    return source


class FactorizedWavefunctionUpdateTest(unittest.TestCase):
    def test_supported_private_names_and_rejected_domains(self):
        gen = generator()
        for rank in (2, 3, 4):
            for vertex in ('FFV1L1_2', 'FFV1L2_1'):
                for terminal in (False, True):
                    with self.subTest(rank=rank, vertex=vertex, terminal=terminal):
                        name = 'TEST_%s_WL_%d_1_FACTOR_%s' % (
                            'TERMINAL' if terminal else 'UPDATE', rank, vertex)
                        source = gen.write_factorized_wl_updater(rank, vertex, terminal)
                        self.assertTrue(source.startswith('SUBROUTINE ' + name + '('))
                        self.assertEqual(source, gen.write_factorized_wl_updater(
                            rank, 'MP_' + vertex, terminal))
        for rank, vertex, precision in (
                (1, 'FFV1L1_2', 'complex*32'),
                (5, 'FFV1L2_1', 'complex*32'),
                (2, 'FFS1L1_2', 'complex*32'),
                (2, 'FFV1L1_2', 'complex*16')):
            with self.assertRaises(PolynomialError):
                generator(precision).write_factorized_wl_updater(rank, vertex)
        gen.sub_prefix = 'A' * 63
        with self.assertRaises(PolynomialError):
            gen.write_factorized_wl_updater(2, 'FFV1L1_2')

    @unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
    def test_fast_paths_fallback_bits_flags_and_enabled_traps(self):
        gen = generator()
        sources = []
        old_calls, new_calls, selectors = [], [], []
        index = 0
        for rank in (2, 3, 4):
            sources.append(gen.write_compact_wl_updater(rank, 1))
            sources.append(gen.write_compact_wl_updater(rank, 1, terminal=True))
            for vertex in ('FFV1L1_2', 'FFV1L2_1'):
                for terminal in (False, True):
                    index += 1
                    public = 'TEST_%s_WL_%d_1' % (
                        'TERMINAL' if terminal else 'UPDATE', rank)
                    private = public + '_FACTOR_' + vertex
                    sources.append(instrument(gen.write_factorized_wl_updater(
                        rank, vertex, terminal)))
                    old_calls.append('case(%d)\n  call %s(a,ls,b,ins,outs,old%s)' %
                                     (index, public, ',entry' if terminal else ''))
                    new_calls.append('case(%d)\n  call %s(a,ls,b,ins,outs,new,entry)' %
                                     (index, private))
                    selectors.append('case(%d)\n  flip=%d' %
                                     (index, 1 if vertex == 'FFV1L1_2' else -1))
        driver = _DRIVER.replace('OLD_CALLS', '\n'.join(old_calls))
        driver = driver.replace('NEW_CALLS', '\n'.join(new_calls))
        driver = driver.replace('SELECT_ORIENTATION', '\n'.join(selectors))
        with tempfile.TemporaryDirectory(prefix='madloop-factor-') as tmp:
            work = Path(tmp)
            (work / 'coef_specs.inc').write_text(
                '      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
                '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
            (work / 'loop_max_coefs.inc').write_text(
                '      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n')
            for name, text in (
                    ('constants.f', gen.write_polynomial_constant_module()),
                    ('kernels.f', '\n\n'.join(sources))):
                with FortranWriter(str(work / name)) as writer:
                    writer.writelines(text)
            (work / 'driver.f90').write_text(driver)
            for flags in ([], ['-fno-automatic']):
                with self.subTest(flags=flags):
                    compiled = subprocess.run(
                        ['gfortran', '-O3', '-w', '-fPIC', '-ffixed-line-length-132',
                         *flags, 'constants.f', 'kernels.f', 'driver.f90', '-o', 'check'],
                        cwd=work, capture_output=True, text=True, timeout=180)
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    checked = subprocess.run([str(work / 'check')], cwd=work,
                                             capture_output=True, text=True, timeout=30)
                    self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
                    self.assertEqual(checked.stdout.split(), ['PASS', '204'])


_DRIVER = r"""
program check
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(16) a(4,0:209,4),b(4,0:14,4),old(4,0:209,4),new(4,0:209,4)
complex(16) h(4,4),ci
integer*8 saved_a(4*4*210*4),saved_b(4*4*15*4)
integer*8 oldbits(4*4*210*4),newbits(4*4*210*4)
integer kernel,trial,i,j,k,l,ls,ins,outs,flip,flag,checks,fast,fallback
common/FACTOR_TEST/fast,fallback
logical entry,oldflags(size(ieee_all)),newflags(size(ieee_all)),wantfast
ci=cmplx(0.0_16,1.0_16,16)
checks=0
call ieee_set_halting_mode(ieee_all,.false.)
do kernel=1,12
 select case(kernel)
 SELECT_ORIENTATION
 end select
 do trial=1,12
  call ieee_set_rounding_mode(ieee_nearest)
  a=0
  b=0
  do i=1,4
   do l=0,209
    do k=1,4
     a(k,l,i)=cmplx(real(mod(17*k+7*l+11*i,31)-15,16)/8, &
                          real(mod(13*k+3*l+5*i,23)-11,16)/16,16)
    enddo
   enddo
  enddo
  do k=1,4
   do j=1,4
    h(j,k)=cmplx(real(2*j-k,16)/8,real(j+3*k,16)/16,16)
    b(j,0,k)=cmplx(real(3*j-k,16)/8,real(2*j-k,16)/16,16)
   enddo
  enddo
  if(trial==11) then
   a(:,0:209:2,:)=0
   h(:,2)=0
  endif
  if(trial==12) then
   h(1,:)=cmplx(real(h(1,:),16),-0.0_16,16)
   h(3,:)=cmplx(-0.0_16,aimag(h(3,:)),16)
  endif
  b(:,1,:)=h
  b(1,2,:)=-flip*h(2,:)
  b(2,2,:)=-flip*h(1,:)
  b(3,2,:)=flip*h(4,:)
  b(4,2,:)=flip*h(3,:)
  b(1,3,:)=ci*h(2,:)
  b(2,3,:)=-ci*h(1,:)
  b(3,3,:)=-ci*h(4,:)
  b(4,3,:)=ci*h(3,:)
  b(1,4,:)=-flip*h(1,:)
  b(2,4,:)=flip*h(2,:)
  b(3,4,:)=flip*h(3,:)
  b(4,4,:)=-flip*h(4,:)
  entry=.true.
  ls=4
  ins=4
  outs=4
  select case(trial)
  case(2)
   entry=.false.
  case(3)
   b(1,2,1)=b(1,2,1)+1
  case(4)
   a(1,0,1)=cmplx(scale(1.0_16,1001),0.0_16,16)
  case(5)
   call ieee_set_rounding_mode(ieee_up)
  case(6)
   ls=0
  case(7)
   a(1,0,1)=cmplx(ieee_value(0.0_16,ieee_signaling_nan),0.0_16,16)
  case(8)
   b(1,0,1)=cmplx(ieee_value(0.0_16,ieee_signaling_nan),0.0_16,16)
  case(9)
   outs=3
  case(10)
   b(1,0,1)=cmplx(nearest(0.0_16,1.0_16),0.0_16,16)
  end select
  saved_a=transfer(a,saved_a)
  saved_b=transfer(b,saved_b)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,entry)
  select case(kernel)
  OLD_CALLS
  end select
  call ieee_get_flag(ieee_all,oldflags)
  fast=0
  fallback=0
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,entry)
  select case(kernel)
  NEW_CALLS
  end select
  call ieee_get_flag(ieee_all,newflags)
  wantfast=trial==1.or.trial==6.or.trial==11.or.trial==12
  if(wantfast) then
   if(fast/=1.or.fallback/=0) stop 10
   if(any(abs(old-new)>1.e-30_16*max(1.0_16,maxval(abs(old))))) stop 11
  else
   if(fast/=0.or.fallback/=1) stop 12
   oldbits=transfer(old,oldbits)
   newbits=transfer(new,newbits)
   if(any(oldbits/=newbits)) stop 13
  endif
  if(any(oldflags.neqv.newflags)) stop 14
  if(any(saved_a/=transfer(a,saved_a))) stop 15
  if(any(saved_b/=transfer(b,saved_b))) stop 16
  checks=checks+1
 enddo
 call ieee_set_rounding_mode(ieee_nearest)
 do flag=1,size(ieee_all)
  if(.not.ieee_support_halting(ieee_all(flag))) stop 17
  a=0
  b=0
  ls=4
  ins=4
  outs=4
  entry=.true.
  fast=0
  fallback=0
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_halting_mode(ieee_all(flag),.true.)
  select case(kernel)
  NEW_CALLS
  end select
  call ieee_set_halting_mode(ieee_all,.false.)
  call ieee_get_flag(ieee_all,newflags)
  if(fast/=0.or.fallback/=1.or.any(newflags).or.any(new/=0)) stop 18
  checks=checks+1
 enddo
enddo
if(checks/=204) stop 19
write(*,*) 'PASS',checks
end program
"""
