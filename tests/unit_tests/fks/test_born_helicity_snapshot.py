"""Sparse colour flows must not read uninitialized helicity contractions."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MG5DIR
from madgraph.iolibs import born_support, file_writers


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestBornHelicitySnapshot(unittest.TestCase):
    def test_sparse_orders_reconstruct_the_same_saved_snapshot(self):
        template = Path(MG5DIR) / 'madgraph/iolibs/template_files/born_hel_splitorders_fks.inc'
        routine = re.search(
            r'^ *SUBROUTINE BORN_HEL_SPLITORDERS\b.*?^ *END *$',
            template.read_text(), flags=re.I | re.M | re.S).group() % {
            'info_lines': '', 'process_lines': '', 'nAmpSplitOrders': 2,
            'nSqAmpSplitOrders': 3, 'ngraphs': 3, 'ncolor': 2,
            'nb_temp_jamp': 0,
            'color_data_lines': '      DATA DENOM/1/\n      DATA CF/9,6,9/',
            # The second order has only one nonzero colour flow, as in the
            # mixed QCD/QED u u~ > t t~ Born. The other entry must be zero.
            'jamp_lines': ('      JAMP(1,1)=AMP(1)/6d0\n'
                           '      JAMP(2,1)=-AMP(1)/2d0\n'
                           '      JAMP(1,2)=-AMP(2)-AMP(3)'),
        }
        with tempfile.TemporaryDirectory(prefix='mg7_born_snapshot_') as tmp:
            work = Path(tmp)
            with file_writers.FortranWriter(str(work / 'kernel.f')) as writer:
                writer.writelines(routine)
            born_support.write_fortran(work / 'nexternal.inc', [
                'integer,parameter::nexternal=3'])
            born_support.write_fortran(work / 'born_nhel.inc', [
                'integer,parameter::max_bhel=2'])
            (work / 'check.f90').write_text('''program check
use, intrinsic :: ieee_arithmetic
implicit none
real(8) p(0:3,2),actual(3),expected(3)
complex(8) saved(3,2),original(3,2)
integer helicity
p=0d0
saved(:,1)=[(6d0,0d0),(1d0,0d0),(2d0,0d0)]
saved(:,2)=2d0*saved(:,1)
original=saved
do helicity=1,2
  call born_hel_splitorders(p,helicity,actual,saved)
  expected=[72d0,0d0,81d0]*real(helicity**2,8)
  if(.not.all(ieee_is_finite(actual)))error stop 'nonfinite colour flow'
  if(any(abs(actual-expected).gt.1d-12))error stop 'wrong colour contraction'
enddo
call born_hel_splitorders(p,1,actual,saved)
if(any(abs(actual-[72d0,0d0,81d0]).gt.1d-12))error stop 'snapshot changed'
if(any(saved.ne.original))error stop 'saved amplitudes mutated'
print *, 'PASS sparse saved helicity snapshot'
end program

integer function sqsoindexb(i,j)
implicit none
integer i,j
sqsoindexb=2
if(i.eq.1.and.j.eq.1)sqsoindexb=1
if(i.eq.2.and.j.eq.2)sqsoindexb=3
end function
''')
            result = subprocess.run([
                shutil.which('gfortran'), '-O0', '-fcheck=all', '-finit-real=snan',
                '-ffixed-line-length-none', 'kernel.f', 'check.f90', '-o', 'check'],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS sparse saved helicity snapshot', result.stdout)
