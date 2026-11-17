"""Compile exported model helpers and check their binary128 mathematics.

These functions enter electroweak counterterms before loop reduction. A
low-precision intermediate here can invalidate an otherwise successful rescue.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs.export_v4 import UFO_model_to_mg4
from madgraph.iolibs.ufo_expression_parsers import UFOExpressionParserMPFortran


DRIVER = """
program check_helpers
  implicit none
  complex(16), external :: mp_reglog,mp_reglogp,mp_reglogm,mp_regsqrt
  complex(16), external :: mp_grreglog,mp_recms,mp_b0f
  complex(16), external :: mp_sqrt_trajectory
  complex(16) :: z,w,expected,twopii,discriminant
  real(16) :: x,pi,delta
  integer :: i,mode,n
  character(16) :: arg
  call get_command_argument(1,arg)
  read(arg,*) mode
  pi=acos(-1.0_16)
  twopii=cmplx(0.0_16,2*pi,kind=16)
  x=1.234567890123456789012345678901234_16
  select case(mode)
  case(1)
    ! Pure real/imaginary inputs used to be rounded to default COMPLEX.
    do i=1,4
      select case(i)
      case(1)
        z=cmplx(x,0.0_16,kind=16)
      case(2)
        z=cmplx(0.0_16,x,kind=16)
      case(3)
        z=cmplx(-x,0.0_16,kind=16)
      case(4)
        z=cmplx(0.0_16,-x,kind=16)
      end select
      call close(mp_reglog(z),log(z),11)
      call close(mp_reglogp(z),log(z),12)
      call close(mp_reglogm(z),log(z),13)
      call close(mp_regsqrt(z),sqrt(z),14)
      call close(mp_grreglog(1.0_16,z,z),log(z),15)
      ! Exercise normalization of both arguments independently.
      w=cmplx(x,x,kind=16)
      call close(mp_grreglog(1.0_16,w,z),log(w),16)
    enddo
    z=cmplx(0.0_16,0.0_16,kind=16)
    call close(mp_reglog(z),z,17)
    call close(mp_reglogp(z),z,18)
    call close(mp_reglogm(z),z,19)
    call close(mp_grreglog(1.0_16,z,z),z,20)
  case(2)
    ! Continuation is checked against runtime ACOS, not a copied pi literal.
    z=cmplx(-x,-x,kind=16)
    call close(mp_reglogp(z),log(z)+twopii,21)
    call close(mp_reglogm(z),log(z),22)
    call close(mp_grreglog(1.0_16,z,conjg(z)),log(z)+twopii,23)
    call close(mp_grreglog(-1.0_16,z,conjg(z)),log(z)-twopii,24)
    z=conjg(z)
    call close(mp_reglogm(z),log(z)-twopii,25)
    call close(mp_reglogp(z),log(z),26)
    call close(mp_grreglog(1.0_16,z,conjg(z)),log(z)-twopii,27)
    call close(mp_grreglog(0.0_16,z,conjg(z)),log(z),28)
  case(3)
    ! Values identical in double precision must remain distinct in rescue.
    delta=2.0_16**(-100)
    z=cmplx(1.0_16+delta,0.0_16,kind=16)
    expected=log(z)
    if (real(expected,kind=16)==0.0_16) stop 31
    if (abs((mp_reglog(z)-expected)/expected)>1.e-31_16) stop 32
    w=cmplx(1.0_16+delta,x,kind=16)
    if (mp_recms(.false.,w)/=z) stop 33
    if (mp_recms(.true.,w)/=w) stop 34
    if (mp_regsqrt(z)==cmplx(1.0_16,0.0_16,kind=16)) stop 35
    call close(mp_regsqrt(z),sqrt(z),36)
  case(4)
    ! The closed massless B0 branch has an explicit 2*pi*i continuation.
    z=cmplx(9.0_16,-0.9_16,kind=16)
    w=cmplx(4.0_16,-0.2_16,kind=16)
    expected=(w-z)/z*log((w-z)/w)-twopii
    call close(mp_b0f(z,cmplx(0.0_16,kind=16),w),expected,41)
    z=cmplx(1.0_16,-0.1_16,kind=16)
    expected=(w-z)/z*log((w-z)/w)
    call close(mp_b0f(z,cmplx(0.0_16,kind=16),w),expected,42)
  case(5)
    ! A rounded trajectory can move a point across the two-particle threshold.
    ! Below threshold the discriminant crosses its negative-real-axis cut,
    ! so the continued root has the opposite sign to the final principal root.
    w=cmplx(1.0_16,0.0_16,kind=16)
    do i=-1,1,2
      z=cmplx(4.0_16+i*2.0_16**(-30),-1.0_16,kind=16)
      discriminant=1.0_16-4.0_16*(w-cmplx(0.0_16,1.e-24_16,kind=16))/z
      expected=sqrt(discriminant)
      if(i<0) expected=-expected
      do n=100,400,100
        call close(mp_sqrt_trajectory(n,z,w,w),expected,51)
      enddo
    enddo
  case default
    stop 99
  end select
contains
  subroutine close(actual,reference,code)
    complex(16), intent(in) :: actual,reference
    integer, intent(in) :: code
    if (.not.(abs(actual-reference)<=1.e-31_16*max(1.0_16,abs(reference)))) then
      write(*,*) code,actual,reference
      stop 1
    endif
  end subroutine
end program
"""


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class ModelFunctionPrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='model-function-precision-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.work = Path(cls.tmp.name)
        # No UFO expressions are needed: exercise the real exporter and writer.
        exporter = UFO_model_to_mg4.__new__(UFO_model_to_mg4)
        exporter.dir_path = str(cls.work)
        exporter.model = {'modelpath': str(cls.work), 'functions': []}
        exporter.opt = {'mp': True}
        with exporter.open('model_functions.f', format='fortran') as writer:
            exporter.open = lambda *args, **kwargs: writer
            exporter.create_model_functions_def()
        (cls.work / 'check.f90').write_text(DRIVER)
        build = subprocess.run(
            ['gfortran', '-O2', '-ffixed-line-length-132', '-fcheck=all',
             'model_functions.f', 'check.f90', '-o', 'check'],
            cwd=cls.work, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError(build.stdout)

    def check_case(self, mode):
        run = subprocess.run([str(self.work / 'check'), str(mode)],
                             cwd=self.work, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        self.assertEqual(run.returncode, 0, run.stdout)

    def test_real_and_imaginary_axis_precision(self):
        self.check_case(1)

    def test_logarithm_continuation_sheets(self):
        self.check_case(2)

    def test_input_bits_beyond_double_precision(self):
        self.check_case(3)

    def test_massless_b0_continuation(self):
        self.check_case(4)

    def test_threshold_side_controls_continued_square_root(self):
        self.check_case(5)

    def test_parsed_real_casts_preserve_real_and_complex_inputs(self):
        parser = UFOExpressionParserMPFortran(None)
        lines = [
            'program parsed_precision', 'implicit none',
            'real(16) :: mp__x', 'complex(16) :: mp__z',
            'mp__x=1.234567890123456789012345678901234_16',
            'mp__z=cmplx(mp__x,0.25_16,kind=16)',
        ]
        for variable in ('x', 'z'):
            reference = 'mp__x'
            cases = [(f're({variable})', reference),
                     (f'{variable}.real', reference),
                     (f'({variable}*{variable}).real',
                      f'real(mp__{variable}*mp__{variable},kind=16)')]
            for function in ('tan', 'atan'):
                cases += [(f'{prefix}{function}({variable})',
                           f'{function}({reference})')
                          for prefix in ('', 'cmath.')]
            for expression, expected in cases:
                lines.append(f'if ({parser.parse(expression)}/={expected}) stop 1')
        lines += ['mp__x=1.0_16+2.0_16**(-100)',
                  f'if ({parser.parse("x.real")}-1.0_16/=2.0_16**(-100)) stop 2',
                  'end program']
        (self.work / 'parsed.f90').write_text('\n'.join(lines)+'\n')
        build = subprocess.run(
            ['gfortran', '-O2', 'parsed.f90', '-o', 'parsed'],
            cwd=self.work, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        self.assertEqual(build.returncode, 0, build.stdout)
        run = subprocess.run([str(self.work / 'parsed')], cwd=self.work,
                             text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        self.assertEqual(run.returncode, 0, run.stdout)
