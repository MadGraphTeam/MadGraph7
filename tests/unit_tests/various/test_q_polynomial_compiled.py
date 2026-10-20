"""Numerical regression for the generated wavefunction update kernels."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import (
    FortranPolynomialRoutines, get_number_of_coefs_for_rank)


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class CompactWavefunctionUpdateTest(unittest.TestCase):
    """Compare optimized kernels with the legacy accumulation order in DP/QP."""

    def test_exact_legacy_results_across_ranks_shapes_and_sparsity(self):
        with tempfile.TemporaryDirectory(prefix='madloop-polynomial-') as tmp:
            work = Path(tmp)
            (work / 'coef_specs.inc').write_text(
                '      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
                '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
            (work / 'loop_max_coefs.inc').write_text(
                '      INTEGER LOOPMAXCOEFS\n'
                '      PARAMETER(LOOPMAXCOEFS=210)\n')
            generator = FortranPolynomialRoutines(
                6, updater_max_rank=2, proc_prefix='TEST_', sub_prefix='TEST_')
            with FortranWriter(str(work / 'constants.f')) as writer:
                writer.writelines(generator.write_polynomial_constant_module())
            for kind in (8, 16):
                with self.subTest(kind=kind):
                    self.check_precision(work, kind)

    def check_precision(self, work, kind):
        generated, calls = [], []
        generator = FortranPolynomialRoutines(
            6, updater_max_rank=2, proc_prefix='TEST_', sub_prefix='TEST_',
            coef_format='complex*%d' % (2 * kind))
        for left in range(5):
            for right in range(3):
                generated.append(generator.write_compact_wl_updater(left, right))
                calls.append("""
call TEST_UPDATE_WL_{left}_{right}(a,ls,b,ins,outs,actual)
call legacy_update(a,ls,b,ins,outs,{nleft},{nright},expected)
if (any(actual/=expected)) then
  write(*,*) 'MISMATCH', {left},{right},ls,ins,outs,trial
  stop 1
endif
checks=checks+1
""".format(left=left, right=right,
           nleft=get_number_of_coefs_for_rank(left),
           nright=get_number_of_coefs_for_rank(right)))
        with FortranWriter(str(work / 'generated.f')) as writer:
            writer.writelines('\n'.join(generated))
        # The reference keeps the old L,I order and accepts the rank counts at
        # runtime, independently of the production generator's loop layout.
        (work / 'driver.f90').write_text("""
program check
use TEST_POLYNOMIAL_CONSTANTS
implicit none
complex({kind}) :: a(4,0:209,4),b(4,0:14,4)
complex({kind}) :: expected(4,0:209,4),actual(4,0:209,4)
real({kind}) :: scale
integer :: i,j,k,ls,ins,outs,trial,checks
checks=0
do trial=1,3
  do k=1,4
    do j=0,209
      do i=1,4
        scale=1.0_{kind}
        if(trial==2.and.mod(j,3)==0) scale=1.e20_{kind}
        if(trial==2.and.mod(j,3)==1) scale=1.e-20_{kind}
        a(i,j,k)=cmplx(real(mod(17*i+31*j+13*k,43)-21,kind={kind})/7, &
          real(mod(37*i+11*j+7*k,31)-15,kind={kind})/13,kind={kind})*scale
      enddo
    enddo
    do j=0,14
      do i=1,4
        b(i,j,k)=cmplx(real(mod(19*i+29*j+3*k,17)-8,kind={kind})/3, &
          real(mod(11*i+17*j+13*k,23)-11,kind={kind})/11,kind={kind})
        if(trial==3.and.mod(i+j+k,3)/=0) b(i,j,k)=0
      enddo
    enddo
  enddo
  do ls=1,4
    do ins=1,4
      do outs=1,4
{calls}
      enddo
    enddo
  enddo
enddo
if(checks/=2880) stop 2
write(*,*) 'PASS', checks
contains
subroutine legacy_update(a,ls,b,ins,outs,nleft,nright,out)
  integer, intent(in) :: ls,ins,outs,nleft,nright
  complex({kind}), intent(in) :: a(4,0:209,4),b(4,0:14,4)
  complex({kind}), intent(out) :: out(4,0:209,4)
  complex({kind}) :: updater
  integer :: i,j,k,l,m,position
  out=0
  do j=1,outs
    do m=0,nright-1
      do k=1,ins
        updater=b(j,m,k)
        if(updater==0) cycle
        do l=0,nleft-1
          position=COMB_COEF_POS(l,m)
          do i=1,ls
            out(j,position,i)=out(j,position,i)+a(k,l,i)*updater
          enddo
        enddo
      enddo
    enddo
  enddo
end subroutine
end program
""".format(kind=kind, calls=''.join(calls)))
        result = subprocess.run(
            ['gfortran', '-O3', '-fcheck=all', '-ffixed-line-length-132',
             'constants.f', 'generated.f', 'driver.f90', '-o', 'check'],
            cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        result = subprocess.run(
            [str(work / 'check')], cwd=work,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('PASS', result.stdout)


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class ExpandedWavefunctionUpdateTest(unittest.TestCase):
    """Check full bit patterns against the legacy per-output accumulation."""

    def test_exact_legacy_results_including_inactive_regions_and_signed_zero(self):
        with tempfile.TemporaryDirectory(prefix='madloop-expanded-') as tmp:
            work = Path(tmp)
            (work / 'coef_specs.inc').write_text(
                '      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
                '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
            (work / 'loop_max_coefs.inc').write_text(
                '      INTEGER LOOPMAXCOEFS\n'
                '      PARAMETER(LOOPMAXCOEFS=210)\n')
            generator = FortranPolynomialRoutines(
                6, updater_max_rank=2, proc_prefix='TEST_', sub_prefix='TEST_')
            with FortranWriter(str(work / 'constants.f')) as writer:
                writer.writelines(generator.write_polynomial_constant_module())
            for kind in (8, 16):
                with self.subTest(kind=kind):
                    self.check_precision(work, kind)

    def check_precision(self, work, kind):
        generator = FortranPolynomialRoutines(
            6, updater_max_rank=2, proc_prefix='TEST_', sub_prefix='TEST_',
            coef_format='complex*%d' % (2 * kind))
        generated, calls = [], []
        for left in range(5):
            for right in range(3):
                generated.append(generator.write_expanded_wl_updater(left, right))
                calls.append("""
expected=sentinel
actual=sentinel
call legacy_update(a,ls,b,ins,outs,{nleft},{nright},{nout},expected)
call TEST_UPDATE_WL_{left}_{right}(a,ls,b,ins,outs,actual)
old_bits=transfer(expected,old_bits)
new_bits=transfer(actual,new_bits)
if(any(old_bits/=new_bits)) then
  write(*,*) 'BIT MISMATCH', {left},{right},ls,ins,outs,trial
  stop 1
endif
if(any(a/=saved_a).or.any(b/=saved_b)) stop 3
checks=checks+1
""".format(left=left, right=right,
           nleft=get_number_of_coefs_for_rank(left),
           nright=get_number_of_coefs_for_rank(right),
           nout=get_number_of_coefs_for_rank(left+right)))
        with FortranWriter(str(work / 'generated.f')) as writer:
            writer.writelines('\n'.join(generated))
        # This runtime convolution is independent of the expanded generator.
        # For each fixed output, the historical routine adds terms in K,L,M
        # order. It clears only active outputs, unlike the compact routines.
        (work / 'driver.f90').write_text("""
program check
use iso_fortran_env, only: int64
use TEST_POLYNOMIAL_CONSTANTS
implicit none
complex({kind}) :: a(4,0:209,4),b(4,0:14,4)
complex({kind}) :: saved_a(4,0:209,4),saved_b(4,0:14,4)
complex({kind}) :: expected(4,0:209,4),actual(4,0:209,4),sentinel
real({kind}) :: scale
integer(int64) :: old_bits({nbits}),new_bits({nbits})
integer :: i,j,k,ls,ins,outs,trial,checks
sentinel=cmplx(-123.25_{kind},456.125_{kind},kind={kind})
checks=0
do trial=1,6
  do k=1,4
    do j=0,209
      do i=1,4
        scale=1.0_{kind}
        if(trial==3.and.mod(j,3)==0) scale=1.e100_{kind}
        if(trial==3.and.mod(j,3)==1) scale=1.e-100_{kind}
        a(i,j,k)=cmplx(real(mod(17*i+31*j+13*k,43)-21,kind={kind})/7, &
          real(mod(37*i+11*j+7*k,31)-15,kind={kind})/13,kind={kind})*scale
        if(trial==2.and.mod(i+j+k,3)/=0) a(i,j,k)=0
        if(trial==4) a(i,j,k)=0
        if(trial==5) a(i,j,k)=cmplx((-1.0_{kind})**i, &
                                  (-1.0_{kind})**(i+1),kind={kind})
        if(trial==6.and.mod(i+j+k,2)==0) a(i,j,k)=cmplx(-0.0_{kind},0.0_{kind},kind={kind})
      enddo
    enddo
    do j=0,14
      do i=1,4
        b(i,j,k)=cmplx(real(mod(19*i+29*j+3*k,17)-8,kind={kind})/3, &
          real(mod(11*i+17*j+13*k,23)-11,kind={kind})/11,kind={kind})
        if(trial==2.and.mod(i+j+k,3)/=0) b(i,j,k)=0
        if(trial==4) b(i,j,k)=0
        if(trial==5) b(i,j,k)=cmplx(1.0_{kind},-1.0_{kind},kind={kind})
        if(trial==6.and.mod(i+j+k,2)==1) b(i,j,k)=cmplx(0.0_{kind},-0.0_{kind},kind={kind})
      enddo
    enddo
  enddo
  saved_a=a
  saved_b=b
  do ls=0,4
    do ins=0,4
      do outs=0,4
{calls}
      enddo
    enddo
  enddo
enddo
if(checks/=11250) stop 2
write(*,*) 'PASS',checks
contains
subroutine legacy_update(a,ls,b,ins,outs,nleft,nright,nout,out)
  integer,intent(in) :: ls,ins,outs,nleft,nright,nout
  complex({kind}),intent(in) :: a(4,0:209,4),b(4,0:14,4)
  complex({kind}),intent(inout) :: out(4,0:209,4)
  integer :: i,j,k,l,m,position
  do i=1,ls
    do j=1,outs
      out(j,0:nout-1,i)=0
      do k=1,ins
        do l=0,nleft-1
          do m=0,nright-1
            position=COMB_COEF_POS(l,m)
            out(j,position,i)=out(j,position,i)+a(k,l,i)*b(j,m,k)
          enddo
        enddo
      enddo
    enddo
  enddo
end subroutine
end program
""".format(kind=kind, nbits=2*kind//8*4*210*4, calls=''.join(calls)))
        compiled = subprocess.run(
            ['gfortran', '-O3', '-fcheck=all', '-ffixed-line-length-132',
             'constants.f', 'generated.f', 'driver.f90', '-o', 'check'],
            cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(compiled.returncode, 0, compiled.stdout)
        checked = subprocess.run([str(work / 'check')], cwd=work,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(checked.returncode, 0, checked.stdout)
        self.assertIn('PASS', checked.stdout)
