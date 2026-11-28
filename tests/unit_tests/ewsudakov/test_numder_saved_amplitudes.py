"""Sudakov parameter derivatives respect scalar Born and Goldstone ABIs."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from madgraph.iolibs import born_support, file_writers
from madgraph.iolibs.export_fks import ProcessExporterFortranFKS


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestSudakovDerivativeSnapshots(unittest.TestCase):
    def test_parameter_derivatives_use_the_complete_matrix_element_interface(self):
        for scheme in ('Gmu', 'alphamz'):
            for goldstone, native in ((False, False), (False, True), (True, False)):
                with self.subTest(scheme=scheme, goldstone=goldstone, native=native):
                    self.check_derivative(scheme, goldstone, native)

    def check_derivative(self, scheme, goldstone, native):
        with tempfile.TemporaryDirectory(prefix='mg7_sudakov_numder_') as temporary:
            root = Path(temporary)
            process = root / 'SubProcesses/P0_test'
            process.mkdir(parents=True)
            model = root / 'Source/MODEL'
            model.mkdir(parents=True)
            parameters = ('real(8) mdl_gf,aewm1,mdl_mz,mdl_mw,mdl_mt,mdl_ymt,mdl_mh,mdl_ntadpole\n'
                          'common/model_parameters/mdl_gf,aewm1,mdl_mz,mdl_mw,mdl_mt,mdl_ymt,mdl_mh,mdl_ntadpole\n')
            born_support.write_fortran(model / 'input.inc', parameters.splitlines())
            for name, lines in {
                    'nexternal.inc': ['integer,parameter::nexternal=4'],
                    'orders.inc': ['integer,parameter::amp_split_size=2'],
                    'genps.inc': ['integer,parameter::ngraphs=3,ncolor=2'],
                    'born_nhel.inc': ['integer,parameter::max_bhel=2'],
                    'coupl.inc': []}.items():
                born_support.write_fortran(process / name, lines)
            exporter = SimpleNamespace(model={'name': 'test_' + scheme})
            with file_writers.FortranWriter(str(process / 'numder.f')) as writer:
                ProcessExporterFortranFKS.write_numder_me(exporter, writer,
                                                        0 if goldstone else None, None)
            if native:
                source = process / 'numder.f'
                source.write_text(born_support.explicit_onehel_calls(source.read_text()))
            name = 'ewsdk_gold_me_1' if goldstone else 'sborn_onehel'
            kernel = name + '_amp' if native else name
            signature = ('p,nhel,ans' if goldstone else 'p,nhel,hell,ans,saved')
            extra = '' if goldstone else 'integer hell\ncomplex(8) saved(3,2)\nsaved(:,:)=(123d0,456d0)\n'
            (process / 'check.f90').write_text('''program check
implicit none
include '../../Source/MODEL/input.inc'
real(8) p(0:3,3),invariants(3,3),expected(6)
complex(8) derivatives(2,6),derivatives2(2,6)
common/to_amp_split_ewsud_der/derivatives
common/to_amp_split_ewsud_der2/derivatives2
integer nhel(3),calls
common/calls/calls
mdl_gf=2d0
aewm1=3d0
mdl_mz=4d0
mdl_mw=5d0
mdl_mt=6d0
mdl_ymt=6d0
mdl_mh=7d0
mdl_ntadpole=1d0
p=0d0
invariants=0d0
nhel=1
calls=0
call par_ren_''' + name + '''(p,nhel,1,invariants)
expected=[1d0,2d0,3d0,8d0,5d0,6d0]
if(calls.ne.12)error stop 'missing parameter variations'
if(any(abs(real(derivatives(1,:))-expected).gt.1d-7))error stop 'wrong LO1 derivative'
if(any(abs(real(derivatives(2,:))-2d0*expected).gt.1d-7))error stop 'wrong split-order derivative'
if(any(abs(derivatives2-3d0*derivatives).gt.1d-7))error stop 'wrong LO2 derivative'
if(mdl_gf.ne.2d0.or.aewm1.ne.3d0.or.mdl_mz.ne.4d0.or.mdl_mw.ne.5d0.or. &
   mdl_mt.ne.6d0.or.mdl_ymt.ne.6d0.or.mdl_mh.ne.7d0.or.mdl_ntadpole.ne.1d0)error stop 'model not restored'
print *, 'PASS complete derivative interface'
end program

subroutine ''' + kernel + '(' + signature + ''')
implicit none
include '../../Source/MODEL/input.inc'
real(8) p(0:3,3),ans,value
integer nhel(3),calls
complex(8) amplitude(2),amplitude2(2)
common/to_amp_split_ewsud/amplitude
common/to_amp_split_ewsud_lo2/amplitude2
common/calls/calls
''' + extra + '''calls=calls+1
value=mdl_gf+aewm1+2d0*mdl_mz+3d0*mdl_mw+4d0*(mdl_mt+mdl_ymt)+5d0*mdl_mh+6d0*mdl_ntadpole
amplitude(1)=cmplx(value,0d0,kind=8)
amplitude(2)=2d0*amplitude(1)
amplitude2=3d0*amplitude
ans=value
end subroutine

subroutine coup()
end subroutine

subroutine get_par_ren_''' + scheme + '''(invariants)
implicit none
real(8) invariants(3,3)
end subroutine
''')
            build = subprocess.run(['gfortran', '-O0', '-fcheck=all',
                                    '-ffixed-line-length-none', 'numder.f', 'check.f90', '-o', 'check'],
                                   cwd=process, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(process / 'check')], cwd=process,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('PASS complete derivative interface', run.stdout)
