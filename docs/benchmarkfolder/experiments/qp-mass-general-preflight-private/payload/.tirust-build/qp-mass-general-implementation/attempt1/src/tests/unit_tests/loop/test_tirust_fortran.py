"""Compile the generated adapter against an independent recording tensor API."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from madgraph import MG5DIR
from madgraph.loop.loop_exporters import LoopProcessExporterFortranSA
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines

ROOT = Path(MG5DIR)
TEMPLATES = ROOT / 'madgraph/iolibs/template_files/loop_optimized'
PARAMS = ROOT / 'Template/loop_material/StandAlone/SubProcesses'


class TensorPermutationTest(unittest.TestCase):
    def test_all_components_through_rank_six(self):
        generator = FortranPolynomialRoutines(6)
        permutation = generator.get_TIRUST_mapping()
        self.assertEqual(sorted(permutation), list(range(1, 211)))
        self.assertEqual(permutation[5:15], [6, 7, 10, 8, 11, 13, 9, 12, 14, 15])
        # Independent lexicographic enumeration of exponent tuples.
        basis = sorted(((a, b, c, d) for a in range(7) for b in range(7)
                        for c in range(7) for d in range(7) if a+b+c+d <= 6),
                       key=lambda x: (sum(x), tuple(-v for v in x)))
        for index, position in enumerate(permutation):
            indices = generator.pq.get_coef_at_position(index)
            self.assertEqual(basis[position-1], tuple(indices.count(k) for k in range(4)))


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TensorAdapterTest(unittest.TestCase):
    def run_command(self, *args):
        result = subprocess.run(args, cwd=self.work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def test_sparse_qp_rescue_list_wraps_at_the_zero_sentinel(self):
        with tempfile.TemporaryDirectory(prefix='tirust-rescue-') as directory:
            self.work = Path(directory)
            source = (TEMPLATES / 'TIR_interface.inc').read_text()
            source = source[source.index('\t  SUBROUTINE %(proc_prefix)sCHOOSE_LOOPLIB'):]
            source = source[:source.index('\t  SUBROUTINE %(proc_prefix)sCLEAR_TIR_CACHE')]
            (self.work / 'process_info.inc').write_text(
                '      INTEGER MAX_SPIN_CONNECTED_TO_LOOP\n'
                '      PARAMETER(MAX_SPIN_CONNECTED_TO_LOOP=3)\n')
            driver = '''
      PROGRAM RESCUE
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER INDEX_QP_TOOLS(@SLOTS@),I_LIB,MODE,START
      LOGICAL QP_TOOLS_AVAILABLE
      CHARACTER*1 ARG
      COMMON/TEST_LOOP_TOOLS/QP_TOOLS_AVAILABLE,INDEX_QP_TOOLS
      CALL DEFAULTPARAM()
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      READ(ARG,*) MODE
      MLReductionLib=0
      MLReductionLib(1)=9
      QP_TOOLS_AVAILABLE=.TRUE.
      INDEX_QP_TOOLS=0
      INDEX_QP_TOOLS(1)=1
      START=1
      IF(MODE.EQ.1) THEN
        MLReductionLib(2)=1
        INDEX_QP_TOOLS(2)=2
      ENDIF
C MODE 2 starts at a DP-only backend and advances to a qualified QP
C backend, then must wrap at the sparse QP-list sentinel to CutTools.
      IF(MODE.EQ.2) THEN
        MLReductionLib(1:3)=(/1,7,9/)
        INDEX_QP_TOOLS(1:2)=(/1,3/)
        START=2
      ENDIF
C MODE 3 starts at the final QP entry and wraps immediately.
      IF(MODE.EQ.3) THEN
        MLReductionLib(1:2)=(/1,9/)
        INDEX_QP_TOOLS(1:2)=(/1,2/)
        START=2
      ENDIF
      CALL TEST_CHOOSE_LOOPLIB(START,7,7,.FALSE.,1,.TRUE.,I_LIB)
      IF(MODE.EQ.0) STOP 99
      IF(MODE.EQ.1.AND.I_LIB.NE.2) STOP 98
      IF(MODE.GE.2.AND.I_LIB.NE.1) STOP 97
      END
      SUBROUTINE DETECT_LOOPLIB(LIB,N,R,CM,HEFT,SPIN,PASS)
      INTEGER LIB,N,R,SPIN
      LOGICAL CM,HEFT,PASS
      PASS=LIB.EQ.1
      END
'''
            # The COMMON block size depends on all available QP backends,
            # including libraries absent from this particular selection list.
            for ninja_qp, ninjars_qp in ((False, False), (True, False),
                                         (False, True), (True, True)):
                with self.subTest(ninja_qp=ninja_qp, ninjars_qp=ninjars_qp):
                    with FortranWriter(str(self.work / 'choose.f')) as writer:
                        writer.writelines(source % dict(proc_prefix='TEST_', nloop_groups=1,
                            has_HEFT_list='DATA HAS_AN_HEFT_VERTEX/.FALSE./'), context={
                                'tirust_supports_quad_prec': True,
                                'ninja_supports_quad_prec': ninja_qp,
                                'ninjars_supports_quad_prec': ninjars_qp})
                    (self.work / 'driver.f').write_text(driver.replace(
                        '@SLOTS@', str(3+int(ninja_qp)+int(ninjars_qp))))
                    self.run_command('gfortran', '-ffixed-line-length-132', '-fcheck=all',
                                     '-I', str(PARAMS), 'choose.f', 'driver.f',
                                     str(PARAMS / 'MadLoopParamReader.f'), '-o', 'check')
                    failed = subprocess.run([str(self.work / 'check'), '0'], cwd=self.work,
                                            text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT)
                    self.assertEqual(failed.returncode, 1, failed.stdout)
                    self.assertIn('No qp loop library can deal', failed.stdout)
                    self.assertNotIn('out of bounds', failed.stdout)
                    for mode in (1, 2, 3):
                        self.run_command(str(self.work / 'check'), str(mode))

    def test_recorded_dp_qp_tensors_and_failure_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix='tirust-adapter-') as directory:
            self.work = Path(directory)
            permutation = FortranPolynomialRoutines(6).get_TIRUST_mapping()
            mapping = '\n'.join('DATA (TIRUST_MAP(I),I=%d,%d)/%s/' % (
                start, min(start+9, 209), ','.join(map(str, permutation[start:start+10])))
                for start in range(0, 210, 10))
            replacements = dict(proc_prefix='TEST_', nloop_groups=1, nSquaredSO=1, maxrank=6,
                                real_dp_format='REAL*8', complex_dp_format='COMPLEX*16',
                                real_mp_format='REAL*16', complex_mp_format='COMPLEX*32',
                                mass_dp_format='COMPLEX*16', include_vector='',
                                tirust_coefmap=mapping,
                                tirust_plan_registration='CALL TIRUST_REGISTER_PLAN(3,6,STATUS)')
            replacements['mass_mp_format'] = 'COMPLEX*32'
            exporter = object.__new__(LoopProcessExporterFortranSA)
            exporter.template_dir = str(TEMPLATES)
            replacements['tirust_qp_tensor_interfaces'] = exporter.qp_mass_interfaces(
                'TIRUST_qp_tensor_interface.inc', replacements)
            replacements['tirust_qp_loop_interfaces'] = exporter.qp_mass_interfaces(
                'TIRUST_qp_loop_interface.inc', replacements)
            helpers = (TEMPLATES / 'CT_interface.inc').read_text()
            helpers = helpers[helpers.index('\t  SUBROUTINE %(proc_prefix)sBUILD_KINEMATIC_MATRIX'):]
            helpers = helpers[:helpers.index('## if(samurai_available)')]
            switch = (TEMPLATES / 'TIR_interface.inc').read_text()
            switch = switch[switch.index('      SUBROUTINE %(proc_prefix)sSWITCH_ORDER'):]
            switch = switch[:switch.index('\t  SUBROUTINE %(proc_prefix)sINITTIR')]
            source = (TEMPLATES / 'TIRUST_interface.inc').read_text() + helpers + switch
            with FortranWriter(str(self.work / 'adapter.f')) as writer:
                writer.writelines(source % replacements, context={
                    'tirust_available': True, 'tirust_supports_quad_prec': True,
                    'AmplitudeReduction': True})
            (self.work / 'loop_max_coefs.inc').write_text(
                '      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n')
            (self.work / 'coupl.inc').write_text(
                '      REAL*8 MU_R\n      COMMON/TEST_COUPL/MU_R\n')
            (self.work / 'module.f90').write_text(recording_module())
            driver = DRIVER.replace('@MAP@', ','.join(map(str, permutation)))
            # The production parameter include is fixed form.
            with FortranWriter(str(self.work / 'driver.f')) as writer:
                writer.writelines(driver)
            self.run_command('gfortran', '-ffixed-line-length-132', '-fcheck=all',
                             '-I', str(PARAMS), 'module.f90', 'adapter.f', 'driver.f',
                             str(PARAMS / 'MadLoopParamReader.f'), '-o', 'check')
            self.run_command(str(self.work / 'check'))
            # A lower-rank generated process must reject diagnostics outside
            # its coefficient capacity even though the library supports them.
            replacements.update(maxrank=1, tirust_coefmap='DATA TIRUST_MAP/1,2,3,4,5/')
            with FortranWriter(str(self.work / 'adapter.f')) as writer:
                writer.writelines(source % replacements, context={
                    'tirust_available': True, 'tirust_supports_quad_prec': True,
                    'AmplitudeReduction': True})
            (self.work / 'loop_max_coefs.inc').write_text(
                '      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=5)\n')
            (self.work / 'module.f90').write_text(recording_module().replace(
                'NCOEF_R(0:6)=[1,5,15,35,70,126,210]', 'NCOEF_R(0:1)=[1,5]'))
            (self.work / 'capacity.f90').write_text('''program capacity
use MTIRUST
implicit none
real(8) :: p(0:3,3)
complex(8) :: m(3),coefs(0:4,3)
logical :: stable
p=0
m=1
call TEST_TIRUST_TENSOR_DIAGNOSTIC(1,3,2,p,m,coefs,stable)
if(stable.or.calls/=0.or.registrations/=0) stop 1
end program
''')
            self.run_command('gfortran', '-ffixed-line-length-132', '-fcheck=all',
                             '-I', str(PARAMS), 'module.f90', 'adapter.f', 'capacity.f90',
                             str(PARAMS / 'MadLoopParamReader.f'), '-o', 'capacity')
            self.run_command(str(self.work / 'capacity'))


def recording_module():
    source = '''module TEST_POLYNOMIAL_CONSTANTS
integer :: NCOEF_R(0:6)=[1,5,15,35,70,126,210]
end module
module MTIRUST
use iso_c_binding
use ieee_arithmetic
implicit none
integer,parameter :: KI_TIR=c_double,KI_QTIR=selected_real_kind(33,4931),TIRUST_SUCCESS=0
integer :: calls=0,registrations=0,fail=0,last_mode=-1,clear_count=0
integer(c_int64_t) :: budget=-1
integer :: epoch=0
logical :: active=.false.,last_check=.false.,last_cache=.false.
interface TIRUST_TENSOR_EVALUATE
module procedure evaluate_dp,evaluate_qp
end interface
contains
logical function TIRUST_SUPPORTS(n,r,finite)
integer :: n,r
logical :: finite
TIRUST_SUPPORTS=n>=1.and.n<=6.and.r>=0.and.r<=6
end function
subroutine TIRUST_REGISTER_PLAN(n,r,status)
integer :: n,r,status
if(.not.active) epoch=epoch+1
active=.true.
registrations=registrations+1
status=0
end subroutine
subroutine TIRUST_SET_CACHE_BUDGET(bytes,status)
integer(c_int64_t) :: bytes
integer :: status
budget=bytes
status=0
end subroutine
subroutine TIRUST_CLEAR_CACHE()
clear_count=clear_count+1
end subroutine
integer function TIRUST_CONTEXT_GENERATION()
TIRUST_CONTEXT_GENERATION=-1
if(active) TIRUST_CONTEXT_GENERATION=epoch
end function
subroutine TIRUST_FINALIZE()
active=.false.
registrations=0
end subroutine
'''
    for suffix, kind in [('dp', 'KI_TIR'), ('qp', 'KI_QTIR')]:
        source += f'''subroutine evaluate_{suffix}(n,r,p,s,m,scale,out,status,pole_mode,finite,check,cache)
integer :: n,r,status,pole_mode,i,j
real({kind}) :: p(0:3,n),s(n,n),scale
complex({kind}) :: m(n),out(:,:)
logical :: finite,check,cache
if(registrations/=1) stop 91
if(scale/=49.or.any(abs(s-transpose(s))>1e-20)) stop 92
calls=calls+1
last_mode=pole_mode
last_check=check
last_cache=cache
status=0
do j=1,4
do i=1,size(out,1)
out(i,j)=cmplx(10*i+j,-j,kind={kind})
enddo
enddo
if(pole_mode==1.or.pole_mode==3) out(:,2:3)=0
if(pole_mode==2.or.pole_mode==3) out(:,1)=0
if(.not.finite) out(:,4)=0
if(fail==1) status=1
if(fail==2) out(1,1)=ieee_value(real(0,kind={kind}),ieee_quiet_nan)
end subroutine
'''
    return source + 'end module\n'


DRIVER = '''program check_adapter
use MTIRUST
use TEST_POLYNOMIAL_CONSTANTS
use ieee_arithmetic
implicit none
include 'MadLoopParams.inc'
real*8 p(0:3,3),mu_r,lscale
real*16 qp(0:3,3)
complex*16 m(3),coefs(0:209,3),res(3)
complex*32 qcoefs(0:209,3),loopcoefs(0:209,1),expected(3)
integer i,j,r,mode,ctmode,id,rr
integer mapping(0:209)
logical stable,fd,fq
common/TEST_COUPL/mu_r
common/TEST_CT/lscale,ctmode
common/TEST_LOOP/id,rr
common/TEST_MP_LCOEFS/loopcoefs
common/TEST_FPE_IN_REDUCTION/fd,fq
data mapping /@MAP@/
call DEFAULTPARAM()
mu_r=7
p(:,1)=[3d0,1d0,2d0,0d0]
p(:,2)=[4d0,-1d0,0d0,2d0]
p(:,3)=-p(:,1)-p(:,2)
qp=p
m=[cmplx(2d0,-0.1d0,8),cmplx(3d0,0d0,8),cmplx(5d0,0d0,8)]
do r=0,6
do mode=0,3
LoopPoleMode=mode
call TEST_TIRUST_TENSOR_DIAGNOSTIC(1,3,r,p,m,coefs,stable)
if(.not.stable.or.last_mode/=mode.or.last_check) stop 1
call TEST_MP_TIRUST_TENSOR_DIAGNOSTIC(4,3,r,qp,m,qcoefs,stable)
if(.not.stable.or.any(coefs(0:NCOEF_R(r)-1,:)/=qcoefs(0:NCOEF_R(r)-1,:))) stop 2
do i=0,NCOEF_R(r)-1
if(coefs(i,1)/=cmplx(10*mapping(i)+4,-4,8)) stop 3
expected=0
if(mode==0.or.mode==1) expected(2)=cmplx(10*mapping(i)+1,-1,16)
if(mode==0.or.mode==2) then
expected(2)=expected(2)+cmplx(10*mapping(i)+2,-2,16)
expected(3)=cmplx(10*mapping(i)+3,-3,16)
endif
if(any(coefs(i,2:3)/=expected(2:3))) stop 4
enddo
enddo
enddo
LoopPoleMode=0
TIRustGlobalCache=2
call TEST_TIRUST_TENSOR_DIAGNOSTIC(2,3,1,p,m,coefs,stable)
if(.not.stable.or..not.last_check.or.last_cache) stop 5
if(budget/=268435456) stop 6
TIRustComputeFinite=.false.
call TEST_TIRUST_TENSOR_DIAGNOSTIC(1,3,1,p,m,coefs,stable)
if(.not.stable.or..not.ieee_is_nan(real(coefs(0,1)))) stop 7
TIRustComputeFinite=.true.
fail=1
call TEST_TIRUST_TENSOR_DIAGNOSTIC(1,3,1,p,m,coefs,stable)
if(stable) stop 8
fail=2
call TEST_TIRUST_TENSOR_DIAGNOSTIC(1,3,1,p,m,coefs,stable)
if(stable) stop 15
call TEST_MP_TIRUST_TENSOR_DIAGNOSTIC(4,3,1,qp,m,qcoefs,stable)
if(stable) stop 9
fail=0
id=1
rr=1
ctmode=4
loopcoefs=0
loopcoefs(3,1)=1
call TEST_MP_TIRUST_LOOP(3,qp,m,1,res,stable)
if(.not.stable) stop 10
expected=[cmplx(44,-4,16),cmplx(83,-3,16),cmplx(43,-3,16)]
expected=expected/(16.0_16*acos(-1.0_16)**2)
if(any(abs(res-expected)>1e-13_16)) stop 11
call TEST_CLEAR_TIRUST_CACHE()
if(clear_count/=1) stop 12
call TIRUST_FINALIZE()
call TEST_TIRUST_TENSOR_DIAGNOSTIC(1,3,0,p,m,coefs,stable)
if(.not.stable.or.registrations/=1) stop 14
call TEST_FINALIZE_TIRUST()
call TEST_TIRUST_TENSOR_DIAGNOSTIC(1,3,0,p,m,coefs,stable)
if(.not.stable.or.registrations/=1) stop 13
end program
'''
