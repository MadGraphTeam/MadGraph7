"""The accelerated wrapper must honor native per-diagram side effects."""
import ast
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNLORealDiagramRouting(unittest.TestCase):
    def test_diagram_requests_refresh_weights_without_disabling_offload(self):
        source = Path(__file__).resolve().parents[3]/'madmatrix/output.py'
        tree = ast.parse(source.read_text())
        routine = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name == '_write_fortran_routing')
        wrappers = next(n.value.value for n in ast.walk(routine)
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'wrappers'
                                for t in n.targets))
        start = re.search(r'^      SUBROUTINE SMATRIX_REAL\(', wrappers, re.M).start()
        end = re.search(r'^      END\s*$', wrappers[start:], re.M).end()+start
        includes = {
            'nexternal.inc': '      integer nexternal\n      parameter(nexternal=3)\n',
            'orders.inc': '      integer amp_split_size\n      parameter(amp_split_size=1)\n',
            'fks_info.inc': ('      integer REAL_FLAVOR_INDEX_D(2)\n'
                             '      parameter(REAL_FLAVOR_INDEX_D=(/1,2/))\n'),
            'real_matrix_weights.inc': (
                '      double precision real_amp2(2)\n'
                '      common /c_real_diagram_weights/real_amp2\n'
                '      logical get_real_diagram_weights\n'
                '      common /c_get_real_diagram_weights/get_real_diagram_weights\n')}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, text in includes.items():
                (directory/name).write_text(text)
            (directory/'wrapper.f').write_text(wrappers[start:end]+'\n')
            (directory/'check.f90').write_text(DRIVER)
            # The wrapper traces its inputs and results through this
            # self-contained module (inactive unless a trace is requested).
            shutil.copy2(source.parents[1]/'Template/NLO/SubProcesses/mc_batch_trace.f90',
                         directory)
            result = subprocess.run(['gfortran', '-O0', '-fcheck=all',
                '-ffixed-line-length-none', '-ffree-line-length-none',
                'mc_batch_trace.f90', 'wrapper.f', 'check.f90', '-o', 'check'], cwd=directory,
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            result = subprocess.run([str(directory/'check')], cwd=directory,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


DRIVER = '''module route_state
integer :: native_calls=0, offload_calls=0
logical :: allow_offload=.true.
end module

program check_diagrams
use route_state
implicit none
include 'real_matrix_weights.inc'
real(8) p(0:3,3),ret(1),wgt,g,all_g
integer nfksprocess,row
common /c_nfksprocess/nfksprocess
common /strong/g,all_g
p=0d0
g=1d0
nfksprocess=1
get_real_diagram_weights=.false.
real_amp2=-999d0
call smatrix_real(p,ret,wgt)
if(offload_calls/=1.or.native_calls/=0.or.wgt/=99d0) stop 1
if(any(real_amp2/=-999d0)) stop 2
get_real_diagram_weights=.true.
do row=1,2
 nfksprocess=row
 p(0,1)=2*row+1
 real_amp2=-999d0
 call smatrix_real(p,ret,wgt)
 if(native_calls/=row.or.offload_calls/=1) stop 3
 if(any(real_amp2/=[row*p(0,1),(7-2*row)*p(0,1)])) stop 4
 if(wgt/=sum(real_amp2).or.ret(1)/=wgt) stop 5
enddo
get_real_diagram_weights=.false.
allow_offload=.false.
real_amp2=-999d0
call smatrix_real(p,ret,wgt)
if(native_calls/=3.or.offload_calls/=2) stop 6
if(any(real_amp2/=-999d0)) stop 7
allow_offload=.true.
call smatrix_real(p,ret,wgt)
if(native_calls/=3.or.offload_calls/=3.or.wgt/=99d0) stop 8
end program

subroutine mg7_nlo_real_try(p,ret,wgt,nfks,flavour,g,success)
use route_state
implicit none
real(8) p(0:3,3),ret(1),wgt,g
integer nfks,flavour
logical success
offload_calls=offload_calls+1
success=allow_offload
if(success) then
 ret=99d0
 wgt=99d0
endif
end subroutine

logical function mg7_nlo_real_is_fortran()
! The generated nlo_real_offload.f90 answers from the run card backend.
mg7_nlo_real_is_fortran=.false.
end function

subroutine smatrix_real_fortran(p,ret,wgt)
use route_state
implicit none
include 'real_matrix_weights.inc'
real(8) p(0:3,3),ret(1),wgt
integer nfksprocess
common /c_nfksprocess/nfksprocess
native_calls=native_calls+1
if(get_real_diagram_weights) real_amp2=[nfksprocess*p(0,1),(7-2*nfksprocess)*p(0,1)]
wgt=(7-nfksprocess)*p(0,1)
ret(1)=wgt
end subroutine
'''
