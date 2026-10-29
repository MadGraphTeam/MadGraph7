"""Compile the actual MadLoop adapters without requiring an installed NinjaRS.

A recording scalar backend checks the Fortran ABI, precision and conventions;
end-to-end numerical comparisons use the real libraries in the CLI benchmarks.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from madgraph import MG5DIR
from madgraph.iolibs.file_writers import FortranWriter

ROOT = Path(MG5DIR)
TEMPLATE = ROOT / 'madgraph/iolibs/template_files/loop_optimized'
PARAMS = ROOT / 'Template/loop_material/StandAlone/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class NinjaRSFortranTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ninjars-fortran-')
        self.work = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_command(self, *args):
        result = subprocess.run(args, cwd=self.work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def compile_run(self, sources):
        self.run_command('gfortran', '-ffixed-line-length-132', '-fcheck=all',
                         '-I', str(PARAMS), *sources, '-o', 'check')
        return self.run_command(str(self.work / 'check'))

    def test_legacy_and_ninjars_return_codes(self):
        source = (PARAMS / 'MadLoopCommons.inc').read_text()
        routine = source.split('      INTEGER FUNCTION SET_RET_CODE_U', 1)[1]
        routine = routine.split('      SUBROUTINE DETECT_LOOPLIB', 1)[0]
        (self.work / 'status.f').write_text(
            '      INTEGER FUNCTION SET_RET_CODE_U' + routine + '''
      PROGRAM TEST_STATUS
      IMPLICIT NONE
      INTEGER SET_RET_CODE_U,I
      DO I=1,8
        IF (SET_RET_CODE_U(I,.FALSE.,.TRUE.).NE.I) STOP 1
        IF (SET_RET_CODE_U(I,.FALSE.,.FALSE.).NE.0) STOP 2
        IF (SET_RET_CODE_U(I,.TRUE.,.FALSE.).NE.0) STOP 3
      ENDDO
      IF (SET_RET_CODE_U(1,.TRUE.,.TRUE.).NE.9) STOP 4
      IF (SET_RET_CODE_U(6,.TRUE.,.TRUE.).NE.8) STOP 5
      IF (SET_RET_CODE_U(8,.TRUE.,.TRUE.).NE.2) STOP 6
      END
''')
        self.compile_run(['status.f'])

    def test_card_eight_libraries_and_terminator(self):
        (self.work / 'card.dat').write_text(
            '#MLReductionLib\n8|7|6|5|4|3|2|1|8\n'
            '#UseQPIntegrandForNinjaRS\n.FALSE.\n')
        (self.work / 'card.f').write_text('''
      PROGRAM TEST_CARD
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      CHARACTER*512 FILENAME
      CALL DEFAULTPARAM()
      IF(ANY(MLReductionLib(4:9).NE.0)) STOP 1
      IF(.NOT.UseQPIntegrandForNinjaRS) STOP 2
      FILENAME='card.dat'
      CALL MADLOOPPARAMREADER(FILENAME,.FALSE.)
      IF(ANY(MLReductionLib.NE.(/8,7,6,5,4,3,2,1,0/))) STOP 3
      IF(UseQPIntegrandForNinjaRS) STOP 4
      END
''')
        self.compile_run(['card.f', str(PARAMS / 'MadLoopParamReader.f')])

    def test_unavailable_ninjars_only_is_actionable_and_fallback_is_preserved(self):
        source = (TEMPLATE / 'loop_matrix_standalone.inc').read_text()
        block = source[source.index('IF (LOCAL_ML_INIT) THEN'):source.index(
            'C Setup the file paths')]
        with FortranWriter(str(self.work / 'selection.f')) as writer:
            writer.writelines('''
      PROGRAM TEST_SELECTION
      IMPLICIT NONE
      INTEGER NLOOPLIB,QP_NLOOPLIB,I,J,MODE,MLReductionLib(9)
      PARAMETER (NLOOPLIB=8,QP_NLOOPLIB=3)
      INTEGER INDEX_QP_TOOLS(QP_NLOOPLIB+1)
      LOGICAL LOCAL_ML_INIT,QP_TOOLS_AVAILABLE
      LOGICAL LOOPLIBS_AVAILABLE(NLOOPLIB)
      LOGICAL LOOPLIBS_QPAVAILABLE(0:NLOOPLIB)
      CHARACTER*1 ARG
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      READ(ARG,*) MODE
      LOCAL_ML_INIT=.TRUE.
      LOOPLIBS_AVAILABLE=.FALSE.
      LOOPLIBS_AVAILABLE(1)=.TRUE.
      LOOPLIBS_AVAILABLE(6)=.TRUE.
      LOOPLIBS_QPAVAILABLE=.FALSE.
      LOOPLIBS_QPAVAILABLE(1)=.TRUE.
      LOOPLIBS_QPAVAILABLE(6)=.TRUE.
      MLReductionLib=0
      MLReductionLib(1)=8
      IF(MODE.EQ.2) MLReductionLib(2)=6
      IF(MODE.EQ.3) MLReductionLib(1:3)=(/6,7,1/)
      IF(MODE.EQ.4) THEN
        LOOPLIBS_AVAILABLE(8)=.TRUE.
        LOOPLIBS_QPAVAILABLE(8)=.TRUE.
      ENDIF
''' + block + '''
      ENDIF
      IF(MODE.EQ.1) STOP 71
      IF(MODE.EQ.2) THEN
        IF(ANY(MLReductionLib(1:2).NE.(/6,0/))) STOP 72
        IF(INDEX_QP_TOOLS(1).NE.1) STOP 73
      ELSEIF(MODE.EQ.3) THEN
        IF(ANY(MLReductionLib(1:3).NE.(/6,1,0/))) STOP 74
        IF(ANY(INDEX_QP_TOOLS(1:2).NE.(/1,2/))) STOP 75
      ELSEIF(MODE.EQ.4) THEN
        IF(ANY(MLReductionLib(1:2).NE.(/8,0/))) STOP 76
        IF(INDEX_QP_TOOLS(1).NE.1) STOP 77
      ENDIF
      IF(.NOT.QP_TOOLS_AVAILABLE) STOP 78
      WRITE(*,*) 'SELECTION ACCEPTED'
      END
''')
        self.run_command('gfortran', '-ffixed-line-length-132', '-fcheck=all',
                         'selection.f', '-o', 'selection')
        output = self.run_command(str(self.work / 'selection'), '1')
        self.assertIn("install ninjars", output)
        self.assertIn('regenerate and rebuild', output)
        self.assertNotIn('SELECTION ACCEPTED', output)
        for mode in ('2', '3', '4'):
            with self.subTest(mode=mode):
                output = self.run_command(str(self.work / 'selection'), mode)
                self.assertIn('SELECTION ACCEPTED', output)
                self.assertNotIn('unavailable', output)

    def test_tensor_abi_normalization_precision_and_failures(self):
        for amplitude in (False, True):
            with self.subTest(amplitude_reduction=amplitude):
                self.check_wrapper(amplitude)

    def test_tir_rejects_unsupported_backends_before_cache_access(self):
        """Direct dispatch cannot bypass the normal unavailable-library filter."""
        source = (TEMPLATE / 'TIR_interface.inc').read_text()
        source = source[:source.index(
            '      SUBROUTINE %(proc_prefix)sSWITCH_ORDER')]
        replacements = {
            'proc_prefix': 'TEST_', 'nloop_groups': 1, 'nSquaredSO': 1,
            'nexternal': 2, 'info_lines': '', 'process_lines': '',
            'real_dp_format': 'REAL*8', 'complex_dp_format': 'COMPLEX*16',
            'mass_dp_format': 'COMPLEX*16', 'include_vector': '',
        }
        (self.work / 'loop_max_coefs.inc').write_text(
            '      INTEGER LOOPMAXCOEFS\n      PARAMETER (LOOPMAXCOEFS=1)\n')
        (self.work / 'tir_cache_size.inc').write_text(
            '      PARAMETER (TIR_CACHE_SIZE=1)\n')
        (self.work / 'coupl.inc').write_text(
            '      REAL*8 MU_R\n      COMMON/TEST_COUPL/MU_R\n')
        driver = '''
      PROGRAM TEST_DISPATCH
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER MODE,CTMODE,LIBINDEX
      REAL*8 PL(0:3,1),LSCALE,MU_R,NORMALIZATION
      COMPLEX*16 MASS(1),RES(3),LOOPCOEFS(1)
      LOGICAL STABLE,INIT_FLAGS(6),TIR_DONE(1,0:1)
      CHARACTER*2 ARG
      COMMON/TEST_CT/LSCALE,CTMODE
      COMMON/TEST_I_LIB/LIBINDEX
      COMMON/TEST_LCOEFS/LOOPCOEFS
      COMMON/TEST_COUPL/MU_R
      COMMON/TEST_TIRCACHING/TIR_DONE
      COMMON/REDUCTIONCODEINIT/INIT_FLAGS
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      READ(ARG,*) MODE
      PL=0D0
      MASS=1D0
      LOOPCOEFS=1D0
      INIT_FLAGS=.FALSE.
      TIR_DONE=.FALSE.
      STABLE=.TRUE.
      LIBINDEX=1
      CTMODE=1
      LoopPoleMode=0
      COLLIERUseInternalStabilityTest=.FALSE.
      MU_R=91D0
      MLReductionLib=0
      MLReductionLib(1)=7
      IF(MODE.GE.10) THEN
C Fill a real coefficient cache with a valid reduction before switching IDs.
        CALL TEST_TIRLOOP(1,1,1,1,PL,MASS,0,RES,STABLE)
      ENDIF
      MLReductionLib(1)=MOD(MODE,10)
      CALL TEST_TIRLOOP(1,1,1,1,PL,MASS,0,RES,STABLE)
      NORMALIZATION=%(factor)dD0/(16D0*3.14159265358979323846D0**2)
      IF(ANY(ABS(RES-NORMALIZATION).GT.1D-15)) STOP 81
      WRITE(*,*) 'DISPATCH RETURNED'
      END
      SUBROUTINE TEST_INITTIR()
      END
      SUBROUTINE ML_LOOP_POLE_FLAGS(UV,IR)
      LOGICAL UV,IR
      UV=.TRUE.
      IR=.TRUE.
      END
      SUBROUTINE TEST_CLEAR_TIR_CACHE()
      LOGICAL TIR_DONE(1,0:1)
      COMMON/TEST_TIRCACHING/TIR_DONE
      TIR_DONE=.FALSE.
      END
      INTEGER FUNCTION TEST_TIRCACHE_INDEX(MODE)
      INTEGER MODE
      TEST_TIRCACHE_INDEX=1
      END
      SUBROUTINE TEST_COLLIERLOOP(MODE,N,R,PL,PDEN,MASS,COEFS,ERRORS)
      INTEGER MODE,N,R
      REAL*8 PL(0:3,N),PDEN(0:3,N)
      COMPLEX*16 MASS(N),COEFS(1,3),ERRORS(1,3)
      COEFS=1D0
      ERRORS=0D0
      END
'''
        for amplitude in (False, True):
            for caching in (False, True):
                with self.subTest(amplitude=amplitude, caching=caching):
                    with FortranWriter(str(self.work / 'tir.f')) as writer:
                        writer.writelines(source % replacements, context={
                            'AmplitudeReduction': amplitude, 'TIRCaching': caching,
                            'collier_available': True, 'golem_available': False,
                            'iregi_available': False, 'pjfry_available': False})
                    (self.work / 'dispatch.f').write_text(
                        driver % {'factor': 1 if amplitude else 2})
                    self.run_command('gfortran', '-ffixed-line-length-132',
                                     '-fcheck=all', '-I', str(PARAMS),
                                     'tir.f', 'dispatch.f', '-o', 'dispatch')
                    for mode in (7, 17, 8, 18, 6, 16):
                        result = subprocess.run(
                            [str(self.work / 'dispatch'), str(mode)],
                            cwd=self.work, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
                        with self.subTest(mode=mode):
                            if mode % 10 == 7:
                                self.assertEqual(result.returncode, 0, result.stdout)
                                self.assertIn('DISPATCH RETURNED', result.stdout)
                            else:
                                self.assertNotEqual(result.returncode, 0, result.stdout)
                                self.assertNotIn('DISPATCH RETURNED', result.stdout)
                                if mode % 10 == 8:
                                    self.assertIn('install ninjars', result.stdout)
                                    self.assertIn('regenerate and rebuild', result.stdout)
                                else:
                                    self.assertIn('Unsupported TIR reduction library',
                                                  result.stdout)

    def test_exit_cleanup_with_and_without_ninjars(self):
        source = (TEMPLATE / 'loop_matrix_standalone.inc').read_text()
        source = source[source.index(
            '\t  SUBROUTINE %(proc_prefix)sEXIT_MADLOOP()'):]
        (self.work / 'cleanup.f90').write_text('''
module mninjars
contains
subroutine ninjars_finalize(status)
integer, optional, intent(out) :: status
integer :: finalizations,color_cleanups
common /cleanup_counts/ finalizations,color_cleanups
finalizations=finalizations+1
if (present(status)) status=0
end subroutine
end module
''')
        for available in (False, True):
            for color_flows in (False, True):
                with self.subTest(ninjars_available=available,
                                  color_flows=color_flows):
                    with FortranWriter(str(self.work / 'exit.f')) as writer:
                        writer.writelines(source % {'proc_prefix': 'TEST_'},
                                          context={
                                              'ninjars_available': available,
                                              'ComputeColorFlows': color_flows})
                    (self.work / 'exit_driver.f90').write_text(f'''
program check_exit
integer :: finalizations,color_cleanups
common /cleanup_counts/ finalizations,color_cleanups
finalizations=0
color_cleanups=0
call TEST_EXIT_MADLOOP()
if (finalizations/={int(available)}) stop 1
if (color_cleanups/={int(color_flows)}) stop 2
end program
subroutine TEST_DEALLOCATE_COLOR_FLOWS()
integer :: finalizations,color_cleanups
common /cleanup_counts/ finalizations,color_cleanups
color_cleanups=color_cleanups+1
end subroutine
''')
                    sources = ['cleanup.f90'] if available else []
                    self.compile_run(sources + ['exit.f', 'exit_driver.f90'])

    def test_public_wrapper_resets_retry_on_entry_and_early_return(self):
        source = (TEMPLATE / 'loop_matrix_standalone.inc').read_text()
        source = source[:source.index(
            '      SUBROUTINE %(proc_prefix)sSLOOPMATRIX_RETRY_IMPL(P_USER,ANS)')]
        source += '## }\n'
        with FortranWriter(str(self.work / 'entry.f')) as writer:
            writer.writelines(source % {
                'proc_prefix': 'TEST_', 'nexternal': 3, 'real_dp_format': 'REAL*8'},
                context={'ninjars_available': True})
        (self.work / 'entry_driver.f90').write_text('''
program entry_check
implicit none
logical :: mode,used
common /TEST_NRS_RETRY_FLAGS/ mode,used
real(8) :: p(0:3,3),ans(0:3,0:0)
integer :: i
p=0
do i=1,2
  mode=.true.
  used=.true.
  call TEST_SLOOPMATRIX(p,ans)
  if(mode.or..not.used.or.any(ans/=777d0)) stop 61
enddo
end program
subroutine TEST_SLOOPMATRIX_RETRY_IMPL(p,ans)
implicit none
logical :: mode,used
common /TEST_NRS_RETRY_FLAGS/ mode,used
integer :: count,passed,failed,hard
common /TEST_NRS_RETRY_STATS/ count,passed,failed,hard
real(8) :: native_acc,comp_acc
common /TEST_NRS_RETRY_ACC/ native_acc,comp_acc
real(8) :: p(0:3,3),ans(0:3,0:0)
if(mode.or.used) stop 62
if(count/=0.or.passed/=0.or.failed/=0.or.hard/=0) stop 63
if(native_acc/=-1d0.or.comp_acc/=-1d0) stop 64
ans=777
mode=.true.
used=.true.
count=1
failed=1
return
end subroutine
''')
        self.compile_run(['entry.f', 'entry_driver.f90'])

    def check_wrapper(self, amplitude):
        replacements = {
            'proc_prefix': 'TEST_', 'nloop_groups': 2, 'nSquaredSO': 2,
            'real_dp_format': 'REAL*8', 'complex_dp_format': 'COMPLEX*16',
            'mass_dp_format': 'COMPLEX*16', 'real_mp_format': 'REAL*16',
            'complex_mp_format': 'COMPLEX*32', 'include_vector': '',
        }
        source = (TEMPLATE / 'NINJARS_interface.inc').read_text()
        # Use the production momentum reversal and on-shell threshold helpers.
        helpers = (TEMPLATE / 'CT_interface.inc').read_text()
        helpers = helpers[helpers.index('\t  SUBROUTINE %(proc_prefix)sBUILD_KINEMATIC_MATRIX'):]
        helpers = helpers[:helpers.index('## if(samurai_available)')]
        switch = (TEMPLATE / 'TIR_interface.inc').read_text()
        switch = switch[switch.index('      SUBROUTINE %(proc_prefix)sSWITCH_ORDER'):]
        switch = switch[:switch.index('\t  SUBROUTINE %(proc_prefix)sINITTIR')]
        matrix = (TEMPLATE / 'loop_matrix_standalone.inc').read_text()
        setter = matrix[matrix.index(
            '      SUBROUTINE %(proc_prefix)sSET_LOOP_POLE_MODE(MODE)'):]
        setter = setter[:setter.index(
            '\t  SUBROUTINE %(proc_prefix)sCOLLIER_COMPUTE_UV_POLES')]
        with FortranWriter(str(self.work / 'wrapper.f')) as writer:
            writer.writelines((source + helpers + switch + setter) % replacements,
                              context={'ninjars_available': True,
                                       'ninjars_supports_quad_prec': True,
                                       'AmplitudeReduction': amplitude})
        (self.work / 'loop_max_coefs.inc').write_text(
            '      INTEGER LOOPMAXCOEFS\n      PARAMETER (LOOPMAXCOEFS=15)\n')
        (self.work / 'coupl.inc').write_text(
            '      REAL*8 MU_R\n      COMMON/TEST_COUPL/MU_R\n')
        module = '''module TEST_POLYNOMIAL_CONSTANTS
integer :: NCOEF_R(0:2)=[1,5,15]
integer :: COEFTORANK_MAP(0:14)=[0,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
end module
module mninjars
use, intrinsic :: ieee_arithmetic
integer,parameter :: KI_NRS=kind(1d0),KI_QNRS=selected_real_kind(33,4931)
integer,parameter :: NINJARS_SUCCESS=0
integer :: failure_mode=0
logical :: seen_mp_extra=.false.
integer :: compensated_calls=0,last_pole_mode=-99
real(8) :: last_shift(0:3),last_s
complex(8) :: last_mass
interface ninjars_tensor_evaluate
module procedure eval_dp,eval_mp
end interface
contains
'''
        for precision, kind in [('dp', 'KI_NRS'), ('mp', 'KI_QNRS')]:
            module += f'''subroutine eval_{precision}(c,n,r,s,p,m,mu,res,r1,status,pole_mode)
integer :: n,r,status,i
integer,optional :: pole_mode
complex({kind}) :: c(:),m(n),res(0:2),r1
real({kind}) :: s(n,n),p(0:3,n),mu
if (.not.present(pole_mode)) stop 58
last_pole_mode=pole_mode
if (size(c)/=(r+1)*(r+2)*(r+3)*(r+4)/24) stop 31
if (abs(mu-49)>1.e-12_{kind}) stop 32
last_shift=real(p(:,2),kind=8)
last_mass=cmplx(m(2),kind=8)
last_s=real(s(1,2),kind=8)
status=0
r1=0
'''
            if precision == 'mp':
                module += 'seen_mp_extra=real(c(1),kind=16)>1.0_16\n'
            module += f'''do i=0,2
res(i)=(i+1)*sum(c)*cmplx(1,1,kind={kind})
enddo
if (failure_mode==1) status=2
if (failure_mode==2) res(0)=ieee_value(0.0_{kind},ieee_quiet_nan)
end subroutine
'''
        module += '''subroutine ninjars_tensor_evaluate_compensated(c,n,r,s,p,m,mu,res,r1,status,pole_mode)
integer :: n,r,status
integer,optional :: pole_mode
complex(KI_NRS) :: c(:),m(n),res(0:2),r1
real(KI_NRS) :: s(n,n),p(0:3,n),mu
compensated_calls=compensated_calls+1
call eval_dp(c,n,r,s,p,m,mu,res,r1,status,pole_mode)
end subroutine
end module
'''
        (self.work / 'modules.f90').write_text(module)
        shape = '(0:14,2)' if amplitude else '(0:14,2,2)'
        loop_common = 'ID,R' if amplitude else 'ID,SQSOINDEX,R'
        coef = 'COEF(:,1)' if amplitude else 'COEF(:,1,1)'
        mpcoef = 'MPCOEF(:,1)' if amplitude else 'MPCOEF(:,1,1)'
        expected = 'cmplx(-13d0,-13d0,kind=8)' if amplitude else 'cmplx(-26d0,0d0,kind=8)'
        expected_mp = 'cmplx(1d0,1d0,kind=8)' if amplitude else 'cmplx(2d0,0d0,kind=8)'
        driver = f'''program check_wrapper
use mninjars
implicit none
integer :: ID,SQSOINDEX,R,CTMODE,I
common/TEST_LOOP/{loop_common}
real(8) :: LSCALE,MU_R,P(0:3,3),NOR
real(16) :: MP(0:3,3)
common/TEST_CT/LSCALE,CTMODE
common/TEST_COUPL/MU_R
complex(8) :: COEF{shape},M(3),RES(3)
complex(16) :: MPCOEF{shape}
common/TEST_LCOEFS/COEF
common/TEST_MP_LCOEFS/MPCOEF
logical :: FDP,FQP,STABLE
common/TEST_FPE_IN_REDUCTION/FDP,FQP
logical :: NRS_RETRY_MODE,NRS_RETRY_USED
common/TEST_NRS_RETRY_FLAGS/NRS_RETRY_MODE,NRS_RETRY_USED
! BLOCK DATA must initialize a direct wrapper call without SLOOPMATRIX.
if (NRS_RETRY_MODE.or.NRS_RETRY_USED) stop 53
call DEFAULTPARAM()
ID=1
SQSOINDEX=1
R=1
MU_R=7
CTMODE=1
FDP=.false.
FQP=.false.
COEF=0
{coef}(0:4)=[(cmplx(I+1,0,kind=8),I=0,4)]
P(:,1)=[4d0,1d0,2d0,3d0]
P(:,2)=[2d0,0d0,0d0,0d0]
P(:,3)=-P(:,1)-P(:,2)
M=[cmplx(3d0,-.5d0,8),cmplx(5d0,-.7d0,8),cmplx(7d0,-.9d0,8)]
NOR=1d0/(16d0*3.14159265358979323846d0**2)
call TEST_NINJARS_LOOP(3,P,M,R,RES,STABLE)
if (.not.STABLE.or.FDP) stop 40
if (abs(RES(1)/NOR-{expected})>1d-12) stop 41
if (abs(RES(2)-2*RES(1))>1d-12) stop 42
if (abs(RES(3)-3*RES(1))>1d-12) stop 43
if (any(abs(last_shift-P(:,1))>1d-12)) stop 44
if (abs(last_s-2d0)>1d-12) stop 45
if (abs(last_mass-M(2))>1d-12) stop 46
if (compensated_calls/=0) stop 54
NRS_RETRY_MODE=.true.
call TEST_NINJARS_LOOP(3,P,M,R,RES,STABLE)
if (compensated_calls/=1.or..not.STABLE) stop 55
if (abs(RES(1)/NOR-{expected})>1d-12) stop 56
NRS_RETRY_MODE=.false.
CTMODE=2
call TEST_NINJARS_LOOP(3,P,M,R,RES,STABLE)
if (.not.STABLE.or.abs(last_shift(0)-6d0)>1d-12) stop 47
if (abs(last_mass-cmplx(7d0,-.9d0,8))>1d-12) stop 48
if (compensated_calls/=1) stop 57
CTMODE=4
MP=real(P,kind=16)
MPCOEF=0
{mpcoef}(0:0)=cmplx(1.0_16+2.0_16**(-100),0.0_16,kind=16)
call TEST_MP_NINJARS_LOOP(3,MP,M,R,RES,STABLE)
if (.not.STABLE.or.FQP.or..not.seen_mp_extra) stop 49
if (abs(RES(1)/NOR-{expected_mp})>1d-12) stop 50
! Check the actual generic setter and all three precision paths.
if(last_pole_mode/=0) stop 59
do I=-1,3
  call TEST_SET_LOOP_POLE_MODE(I)
  CTMODE=1
  call TEST_NINJARS_LOOP(3,P,M,R,RES,STABLE)
  if(last_pole_mode/=max(0,I).or..not.STABLE) stop 60
  NRS_RETRY_MODE=.true.
  call TEST_NINJARS_LOOP(3,P,M,R,RES,STABLE)
  if(last_pole_mode/=max(0,I).or..not.STABLE) stop 61
  NRS_RETRY_MODE=.false.
  CTMODE=4
  call TEST_MP_NINJARS_LOOP(3,MP,M,R,RES,STABLE)
  if(last_pole_mode/=max(0,I).or..not.STABLE) stop 62
enddo
failure_mode=1
call TEST_NINJARS_LOOP(3,P,M,R,RES,STABLE)
if (STABLE.or..not.FDP.or.any(RES/=0)) stop 51
failure_mode=2
call TEST_MP_NINJARS_LOOP(3,MP,M,R,RES,STABLE)
if (STABLE.or..not.FQP.or.any(RES/=0)) stop 52
end program
'''
        # Fortran permits a section of a whole array, but not two chained sections.
        driver = driver.replace(coef + '(0:4)', 'COEF(0:4,1)' if amplitude else 'COEF(0:4,1,1)')
        driver = driver.replace(mpcoef + '(0:0)', 'MPCOEF(0,1)' if amplitude else 'MPCOEF(0,1,1)')
        (self.work / 'driver.f90').write_text(driver)
        self.compile_run(['modules.f90', 'wrapper.f', 'driver.f90',
                          str(PARAMS / 'MadLoopParamReader.f')])
