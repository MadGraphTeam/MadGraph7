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


# Independent dense routines captured before sparse generation was introduced.
_SPARSE_DENSE_REFERENCES = {
    'expanded': r"""      SUBROUTINE MP_OLD_UPDATE_WL_1_1(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE
     $ ,OUT)
      USE TEST_POLYNOMIAL_CONSTANTS
      INTEGER I,J,K
      COMPLEX*32 A(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      COMPLEX*32 B(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      COMPLEX*32 OUT(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE

      DO I=1,LCUT_SIZE
        DO K=0,14
          DO J=1,OUT_SIZE
            OUT(J,K,I)=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
          ENDDO
        ENDDO
        DO K=1,IN_SIZE
          DO J=1,OUT_SIZE
            OUT(J,0,I)=OUT(J,0,I)+A(K,0,I)*B(J,0,K)
            OUT(J,1,I)=OUT(J,1,I)+A(K,0,I)*B(J,1,K)+A(K,1,I)*B(J,0,K)
            OUT(J,2,I)=OUT(J,2,I)+A(K,0,I)*B(J,2,K)+A(K,2,I)*B(J,0,K)
            OUT(J,3,I)=OUT(J,3,I)+A(K,0,I)*B(J,3,K)+A(K,3,I)*B(J,0,K)
            OUT(J,4,I)=OUT(J,4,I)+A(K,0,I)*B(J,4,K)+A(K,4,I)*B(J,0,K)
            OUT(J,5,I)=OUT(J,5,I)+A(K,1,I)*B(J,1,K)
            OUT(J,6,I)=OUT(J,6,I)+A(K,1,I)*B(J,2,K)+A(K,2,I)*B(J,1,K)
            OUT(J,7,I)=OUT(J,7,I)+A(K,2,I)*B(J,2,K)
            OUT(J,8,I)=OUT(J,8,I)+A(K,1,I)*B(J,3,K)+A(K,3,I)*B(J,1,K)
            OUT(J,9,I)=OUT(J,9,I)+A(K,2,I)*B(J,3,K)+A(K,3,I)*B(J,2,K)
            OUT(J,10,I)=OUT(J,10,I)+A(K,3,I)*B(J,3,K)
            OUT(J,11,I)=OUT(J,11,I)+A(K,1,I)*B(J,4,K)+A(K,4,I)*B(J,1,K)
            OUT(J,12,I)=OUT(J,12,I)+A(K,2,I)*B(J,4,K)+A(K,4,I)*B(J,2,K)
            OUT(J,13,I)=OUT(J,13,I)+A(K,3,I)*B(J,4,K)+A(K,4,I)*B(J,3,K)
            OUT(J,14,I)=OUT(J,14,I)+A(K,4,I)*B(J,4,K)
          ENDDO
        ENDDO
      ENDDO
      END
""",
    'compact': r"""      SUBROUTINE MP_OLD_UPDATE_WL_2_1(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE
     $ ,OUT)
      USE TEST_POLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INTEGER I,J,K,L,M
      COMPLEX*32 A(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      COMPLEX*32 B(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      COMPLEX*32 OUT(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
      INTEGER LCUT_SIZE,IN_SIZE,OUT_SIZE
      INTEGER NEW_POSITION
      COMPLEX*32 UPDATER_COEF

C     Welcome to the computational heart of MadLoop...
      OUT(:,:,:)=CMPLX(0.0E0_16,0.0E0_16,KIND=16)
      DO J=1,OUT_SIZE
        DO M=0,4
          DO K=1,IN_SIZE
            UPDATER_COEF = B(J,M,K)
            IF (UPDATER_COEF.EQ.CMPLX(0.0E0_16,0.0E0_16,KIND=16)) CYCLE
            DO I=1,LCUT_SIZE
              DO L=0,14
                NEW_POSITION = COMB_COEF_POS(L,M)
                OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + A(K,L,I)
     $           *UPDATER_COEF
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      ENDDO

      END
""",
}

_SPARSE_DRIVER = r"""
program exact_sparse
use iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
use TEST_POLYNOMIAL_CONSTANTS
implicit none
complex(16) :: a(4,0:209,4),b(4,0:14,4),expected(4,0:209,4),actual(4,0:209,4)
complex(16) :: sentinel
integer(int64) :: old_bits(4*4*210*4),new_bits(4*4*210*4),saved_a(4*4*210*4)
integer(int64) :: b_bits(4*4*15*4),saved_b(4*4*15*4)
real(16) :: scale,inf,nan,snan,small
logical :: old_flags(size(ieee_all)),new_flags(size(ieee_all))
type(ieee_round_type) :: modes(4)
integer :: i,j,k,ls,ins,outs,trial,rounding,checks
modes=[ieee_nearest,ieee_up,ieee_down,ieee_to_zero]
sentinel=cmplx(-123.25_16,456.125_16,kind=16)
inf=ieee_value(0.0_16,ieee_positive_inf)
nan=ieee_value(0.0_16,ieee_quiet_nan)
snan=ieee_value(0.0_16,ieee_signaling_nan)
small=nearest(0.0_16,1.0_16)
checks=0
do rounding=1,4
  call ieee_set_rounding_mode(modes(rounding))
  do trial=1,12
    do k=1,4
      do j=0,209
        do i=1,4
          scale=1.0_16
          if(trial==3.and.mod(j,3)==0) scale=1.e2000_16
          if(trial==3.and.mod(j,3)==1) scale=1.e-2000_16
          a(i,j,k)=cmplx(real(mod(17*i+31*j+13*k,43)-21,kind=16)/7, &
            real(mod(37*i+11*j+7*k,31)-15,kind=16)/13,kind=16)*scale
          if(trial==2.and.mod(i+j+k,3)/=0) a(i,j,k)=0
          if(trial==4) a(i,j,k)=0
          if(trial==5) a(i,j,k)=cmplx((-1.0_16)**i,(-1.0_16)**(i+1),kind=16)
          if(trial==6.and.mod(i+j+k,2)==0) a(i,j,k)=cmplx(-0.0_16,0.0_16,kind=16)
          if(trial==7) a(i,j,k)=cmplx(0.0_16,-0.0_16,kind=16)
          if(trial==7.and.mod(i+j+k,5)==0) a(i,j,k)=cmplx(inf,nan,kind=16)
          if(trial==8.and.mod(i+j+k,3)/=0) a(i,j,k)=0
          if(trial==9) a(i,j,k)=cmplx(small,-small,kind=16)
          if(trial==10.and.mod(i+j+k,2)==0) a(i,j,k)=0
          if(trial==10.and.mod(i+j+k,2)==1) a(i,j,k)=cmplx(huge(scale),-huge(scale),kind=16)
          if(trial==11) a(i,j,k)=cmplx(snan,0.0_16,kind=16)
          if(trial==12) a(i,j,k)=0
        enddo
      enddo
      do j=0,14
        do i=1,4
          b(i,j,k)=cmplx(real(mod(19*i+29*j+3*k,17)-8,kind=16)/3, &
            real(mod(11*i+17*j+13*k,23)-11,kind=16)/11,kind=16)
          if(trial==2.and.mod(i+j+k,3)/=0) b(i,j,k)=0
          if(trial==4.or.trial==7) b(i,j,k)=0
          if(trial==5) b(i,j,k)=cmplx(1.0_16,-1.0_16,kind=16)
          if(trial==6.and.mod(i+j+k,2)==1) b(i,j,k)=cmplx(0.0_16,-0.0_16,kind=16)
          if(trial==8.and.mod(i+j+k,5)==0) b(i,j,k)=cmplx(inf,nan,kind=16)
          if(trial==9) b(i,j,k)=cmplx(0.25_16,-0.25_16,kind=16)
          if(trial==10.and.mod(i+j+k,2)==1) b(i,j,k)=0
          if(trial==11) b(i,j,k)=0
          if(trial==12) b(i,j,k)=cmplx(snan,0.0_16,kind=16)
        enddo
      enddo
    enddo
    saved_a=transfer(a,saved_a)
    saved_b=transfer(b,saved_b)
    do ls=0,4
      do ins=0,4
        do outs=0,4

expected=sentinel
actual=sentinel
call ieee_set_flag(ieee_all,.false.)
call OLD_ROUTINE(a,ls,b,ins,outs,expected)
call ieee_get_flag(ieee_all,old_flags)
call ieee_set_flag(ieee_all,.false.)
call NEW_ROUTINE(a,ls,b,ins,outs,actual)
call ieee_get_flag(ieee_all,new_flags)
old_bits=transfer(expected,old_bits)
new_bits=transfer(actual,new_bits)
if(any(old_bits/=new_bits)) then
  write(*,*) 'BIT MISMATCH', 1,1,ls,ins,outs,trial,rounding
  stop 1
endif
if(any(old_flags.neqv.new_flags)) then
  write(*,*) 'FLAGS MISMATCH', 1,1,ls,ins,outs,trial,rounding
  write(*,*) old_flags,new_flags
  stop 2
endif
if(any(transfer(a,old_bits)/=saved_a).or.any(transfer(b,b_bits)/=saved_b)) stop 3
checks=checks+1

        enddo
      enddo
    enddo
  enddo
enddo
call ieee_set_rounding_mode(ieee_nearest)
if(checks/=6000) stop 4
write(*,*) 'PASS',checks
end program
"""

@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class SparseQPWavefunctionUpdateTest(unittest.TestCase):
    """Preserve output bits, input bits and IEEE flags, including empty calls."""

    def check_kernel(self, mode, rounding_math=False, reject_layout=False,
                     no_automatic=False):
        left, right = (1, 1) if mode == 'expanded' else (2, 1)
        with tempfile.TemporaryDirectory(prefix='madloop-sparse-qp-') as tmp:
            work = Path(tmp)
            (work / 'coef_specs.inc').write_text(
                '      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
                '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
            (work / 'loop_max_coefs.inc').write_text(
                '      INTEGER LOOPMAXCOEFS\n'
                '      PARAMETER(LOOPMAXCOEFS=210)\n')
            generator = FortranPolynomialRoutines(
                6, updater_max_rank=2, proc_prefix='TEST_', sub_prefix='MP_NEW_',
                coef_format='complex*32')
            constants = FortranPolynomialRoutines(6, updater_max_rank=2,
                                                  sub_prefix='TEST_')
            with FortranWriter(str(work / 'constants.f')) as writer:
                writer.writelines(constants.write_polynomial_constant_module())
            source = getattr(generator, 'write_%s_wl_updater' % mode)(left, right)
            if reject_layout:
                # Simulate an unrecognized representation without floating-point
                # classification: every update must reach the untouched oracle.
                old = 'TRANSFER(1.0_16,(/0_8,0_8/))'
                self.assertEqual(source.count(old), 1)
                source = source.replace(old, '(/0_8,0_8/)')
            with FortranWriter(str(work / 'new.f')) as writer:
                writer.writelines(source)
            (work / 'old.f').write_text(_SPARSE_DENSE_REFERENCES[mode])
            suffix = 'UPDATE_WL_%d_%d' % (left, right)
            (work / 'driver.f90').write_text(
                _SPARSE_DRIVER.replace('OLD_ROUTINE', 'MP_OLD_'+suffix)
                              .replace('NEW_ROUTINE', 'MP_NEW_'+suffix))
            flags = ['-O3', '-fcheck=all', '-ffixed-line-length-132']
            if rounding_math:
                flags.append('-frounding-math')
            if no_automatic:
                flags.append('-fno-automatic')
            build = subprocess.run(
                ['gfortran', *flags, 'constants.f', 'old.f', 'new.f',
                 'driver.f90', '-o', 'check'], cwd=work, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(build.returncode, 0, build.stdout)
            run = subprocess.run([str(work/'check')], cwd=work, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertIn('PASS', run.stdout)
            self.assertIn('6000', run.stdout)

    def test_sparse_binary128_preserves_bits_and_exception_flags(self):
        for mode in ('expanded', 'compact'):
            for rounding_math in (False, True):
                with self.subTest(mode=mode, rounding_math=rounding_math):
                    self.check_kernel(mode, rounding_math)

    def test_unrecognized_binary128_layout_uses_original_arithmetic(self):
        self.check_kernel('expanded', reject_layout=True)

    def test_nlo_static_locals_preserve_bits_and_exception_flags(self):
        # FKS compiles polynomial.f with -fno-automatic. A local scratch array
        # in the implicitly PURE elemental classifier would acquire SAVE and
        # fail compilation; exercise actual updates under those build flags.
        for mode in ('expanded', 'compact'):
            with self.subTest(mode=mode):
                self.check_kernel(mode, rounding_math=True, no_automatic=True)


def _terminal_driver(left, right):
    """Independent full-kernel oracle, including the private guard boundaries."""
    driver = _SPARSE_DRIVER.replace('OLD_ROUTINE', 'MP_OLD_UPDATE_WL_%d_%d' % (left, right))
    name = 'MP_NEW_TERMINAL_WL_%d_%d' % (left, right)
    driver = driver.replace('NEW_ROUTINE', name)
    driver = driver.replace('integer :: i,j,k,ls,ins,outs,trial,rounding,checks',
        'integer :: i,j,k,ls,ins,outs,trial,rounding,checks,flagstate,before_fast\n'
        'integer :: fast,fallback\ncommon/TRACE_COUNTS/fast,fallback')
    driver = driver.replace('checks=0\n', 'checks=0\nfast=0\nfallback=0\ndo flagstate=0,1\n')
    driver = driver.replace('call ieee_set_flag(ieee_all,.false.)',
        'call ieee_set_flag(ieee_all,.false.)\n'
        'if(flagstate==1) call ieee_set_flag(ieee_inexact,.true.)')
    driver = driver.replace('call '+name+'(a,ls,b,ins,outs,actual)',
        'before_fast=fast\ncall '+name+'(a,ls,b,ins,outs,actual,flagstate==1)')
    driver = driver.replace('if(any(old_bits/=new_bits)) then',
        'if(fast>before_fast) then\n'
        '  do i=1,4\n'
        '    if(any(transfer(expected(i,:,i),old_bits,210*4)/= &\n'
        '      transfer(actual(i,:,i),new_bits,210*4))) stop 6\n'
        '  enddo\n'
        'else if(any(old_bits/=new_bits)) then')
    driver = driver.replace('call ieee_set_rounding_mode(ieee_nearest)\nif(checks/=6000)',
        'enddo\ncall ieee_set_rounding_mode(ieee_nearest)\nif(checks/=18000)')
    driver = driver.replace("write(*,*) 'PASS',checks",
        "if(fast==0.or.fallback==0) stop 7\nwrite(*,*) 'PASS',checks,fast,fallback")
    driver = driver.replace('do trial=1,12', 'do trial=1,18')
    driver = driver.replace('if(trial==12) a(i,j,k)=0', """if(trial==12) a(i,j,k)=0
          if(trial==13) a(i,j,k)=cmplx(2.0_16**1000,-2.0_16**1000,kind=16)
          if(trial==14) a(i,j,k)=cmplx(2.0_16**(-1000),2.0_16**(-1000),kind=16)
          if(trial==15) a(i,j,k)=cmplx(2.0_16**1001,0.0_16,kind=16)
          if(trial==16) a(i,j,k)=cmplx(2.0_16**(-1001),0.0_16,kind=16)
          if(trial==17) a(i,j,k)=cmplx(2.0_16**1000,2.0_16**1000,kind=16)
          if(trial==18) a(i,j,k)=cmplx(nearest(2.0_16**1001,-1.0_16), &
            -nearest(2.0_16**1001,-1.0_16),kind=16)""")
    driver = driver.replace('if(trial==12) b(i,j,k)=cmplx(snan,0.0_16,kind=16)',
        """if(trial==12) b(i,j,k)=cmplx(snan,0.0_16,kind=16)
          if(trial==13) b(i,j,k)=cmplx(2.0_16**1000,2.0_16**1000,kind=16)
          if(trial==14.or.trial==17) b(i,j,k)=cmplx(2.0_16**(-1000), &
            -2.0_16**(-1000),kind=16)
          if(trial==15.or.trial==16) b(i,j,k)=cmplx(1.0_16,0.0_16,kind=16)
          if(trial==18) b(i,j,k)=cmplx(nearest(2.0_16**1001,-1.0_16), &
            nearest(2.0_16**1001,-1.0_16),kind=16)""")
    return driver


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TerminalQPWavefunctionUpdateTest(unittest.TestCase):
    def test_consumed_bits_inputs_and_flags_with_full_fallback(self):
        for mode in ('expanded', 'compact'):
            for flags in ([], ['-frounding-math'],
                          ['-frounding-math', '-fno-automatic']):
                with self.subTest(mode=mode, flags=flags):
                    self.check_kernel(mode, flags)

    def check_kernel(self, mode, flags):
        left, right = (1, 1) if mode == 'expanded' else (2, 1)
        with tempfile.TemporaryDirectory(prefix='madloop-terminal-qp-') as tmp:
            work = Path(tmp)
            (work/'coef_specs.inc').write_text(
                '      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
                '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
            (work/'loop_max_coefs.inc').write_text(
                '      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n')
            generator = FortranPolynomialRoutines(6, updater_max_rank=2,
                proc_prefix='TEST_', sub_prefix='MP_NEW_', coef_format='complex*32')
            constants = FortranPolynomialRoutines(6, updater_max_rank=2, sub_prefix='TEST_')
            with FortranWriter(str(work/'constants.f')) as writer:
                writer.writelines(constants.write_polynomial_constant_module())
            method = getattr(generator, 'write_'+mode+'_wl_updater')
            private = method(left, right, terminal=True)
            # Only counters are added; both admission paths must actually run.
            private = private.replace('  LOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT\n',
                '  LOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT\n'
                '  INTEGER FAST,FALLBACK\n  COMMON/TRACE_COUNTS/FAST,FALLBACK\n')
            private = private.replace('  CALL MP_NEW_UPDATE_WL_%d_%d(' % (left, right),
                '  FALLBACK=FALLBACK+1\n  CALL MP_NEW_UPDATE_WL_%d_%d(' % (left, right), 1)
            private = private.replace('  RETURN\nENDIF\n',
                                      '  RETURN\nENDIF\nFAST=FAST+1\n', 1)
            with FortranWriter(str(work/'generated.f')) as writer:
                writer.writelines(method(left, right)+'\n'+private)
            (work/'old.f').write_text(_SPARSE_DENSE_REFERENCES[mode])
            (work/'driver.f90').write_text(_terminal_driver(left, right))
            command = ['gfortran', '-O3', '-fcheck=all', '-ffixed-line-length-132',
                       *flags, 'constants.f', 'old.f', 'generated.f', 'driver.f90', '-o', 'check']
            built = subprocess.run(command, cwd=work, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            result = subprocess.run([str(work/'check')], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('18000', result.stdout)
