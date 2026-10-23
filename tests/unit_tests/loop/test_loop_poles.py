"""Bare-loop pole selection, legacy compatibility and cache invalidation."""
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from madgraph import MG5DIR
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.banner import MadLoopParam

ROOT = Path(MG5DIR)
PARAMS = ROOT / 'Template/loop_material/StandAlone/SubProcesses'
TEMPLATES = ROOT / 'madgraph/iolibs/template_files/loop_optimized'


class LoopPoleCardTest(unittest.TestCase):
    def test_card_roundtrip_and_legacy_default(self):
        card = MadLoopParam()
        self.assertEqual(card['LoopPoleMode'], -1)
        for mode in (-1, 0, 1, 2, 3):
            with self.subTest(mode=mode):
                card['LoopPoleMode'] = mode
                stream = io.StringIO()
                card.write(stream)
                parsed = MadLoopParam(stream.getvalue())
                self.assertEqual(parsed['LoopPoleMode'], mode)
                self.assertTrue(parsed['COLLIERComputeUVpoles'])
                self.assertTrue(parsed['COLLIERComputeIRpoles'])


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class LoopPoleFortranTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='madloop-poles-')
        self.work = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_command(self, *args, success=True):
        result = subprocess.run(args, cwd=self.work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def compile(self, *sources):
        self.run_command('gfortran', '-ffixed-line-length-132', '-fcheck=all',
                         '-I', str(PARAMS), *sources,
                         str(PARAMS / 'MadLoopParamReader.f'), '-o', 'check')

    def test_fortran_card_flags_and_rescue_chain_validation(self):
        (self.work / 'card.dat').write_text('#LoopPoleMode\n2\n#MLReductionLib\n8|7\n')
        (self.work / 'check.f').write_text('''
      PROGRAM CHECK
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER MODE,I,J,K
      LOGICAL UV,IR
      CHARACTER*1 ARG
      CHARACTER*512 FILENAME
      CALL DEFAULTPARAM()
      IF(LoopPoleMode.NE.-1) STOP 11
      FILENAME='card.dat'
      CALL MADLOOPPARAMREADER(FILENAME,.FALSE.)
      IF(LoopPoleMode.NE.2) STOP 12
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      READ(ARG,*) MODE
      IF(MODE.EQ.1) THEN
        MLReductionLib(2)=6
        CALL ML_CHECK_LOOP_POLE_MODE()
        STOP 13
      ENDIF
      DO I=0,1
        DO J=0,1
          COLLIERComputeUVpoles=I.EQ.1
          COLLIERComputeIRpoles=J.EQ.1
          DO K=-1,3
            LoopPoleMode=K
            CALL ML_CHECK_LOOP_POLE_MODE()
            CALL ML_LOOP_POLE_FLAGS(UV,IR)
            IF(K.EQ.-1) THEN
              IF(UV.NEQV.(I.EQ.1)) STOP 14
              IF(IR.NEQV.(J.EQ.1)) STOP 15
            ELSE
              IF(UV.NEQV.(K.EQ.0.OR.K.EQ.1)) STOP 16
              IF(IR.NEQV.(K.EQ.0.OR.K.EQ.2)) STOP 17
            ENDIF
            IF(COLLIERComputeUVpoles.NEQV.(I.EQ.1)) STOP 18
            IF(COLLIERComputeIRpoles.NEQV.(J.EQ.1)) STOP 19
          ENDDO
        ENDDO
      ENDDO
      DO I=-1,0
        LoopPoleMode=I
        MLReductionLib(1:3)=(/6,7,1/)
        CALL ML_CHECK_LOOP_POLE_MODE()
      ENDDO
      WRITE(*,*) 'POLE SELECTOR PASSED'
      END
''')
        self.compile('check.f')
        result = self.run_command(str(self.work / 'check'), '0')
        self.assertIn('POLE SELECTOR PASSED', result.stdout)
        result = self.run_command(str(self.work / 'check'), '1', success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Set MLReductionLib to 8, 7 or 8|7', result.stdout)
        self.assertNotIn('POLE SELECTOR PASSED', result.stdout)

    def test_collier_projection_and_mode_changes_invalidate_cached_tensors(self):
        # Compile the production wrapper against a deterministic COLLIER double.
        # Real-library numerical comparisons are separate integration tests.
        source = (TEMPLATES / 'COLLIER_interface.inc').read_text().split(
            '      SUBROUTINE %(proc_prefix)sSET_COLLIER_GLOBAL_CACHE', 1)[0]
        replacements = {
            'proc_prefix': 'TEST_', 'info_lines': '', 'process_lines': '',
            'real_dp_format': 'REAL*8', 'complex_dp_format': 'COMPLEX*16',
            'mass_dp_format': 'COMPLEX*16', 'maxrank': 0, 'nloop_groups': 1,
            'include_vector': '',
            'collier_coefmap': '\n'.join(
                '      DATA COEFMAP_%s/0/' % name
                for name in ('ZERO', 'ONE', 'TWO', 'THREE')),
        }
        with FortranWriter(str(self.work / 'wrapper.f')) as writer:
            writer.writelines(source % replacements,
                              context={'AmplitudeReduction': True})
        (self.work / 'loop_max_coefs.inc').write_text(
            '      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=1)\n')
        (self.work / 'coupl.inc').write_text(
            '      REAL*8 MU_R\n      COMMON/TEST_COUPL/MU_R\n')
        (self.work / 'unique_id.inc').write_text(
            '      INTEGER UNIQUE_ID\n      PARAMETER(UNIQUE_ID=1)\n')
        (self.work / 'global_specs.inc').write_text(
            '      INTEGER NPROCS,MAXNEXTERNAL\n'
            '      PARAMETER(NPROCS=1,MAXNEXTERNAL=3)\n')
        module = '''module collier
implicit none
real(8) :: uv=0,ir1=0,ir2=0
integer :: calls=0,resets=0
interface TNten_cll
module procedure tensor,tadpole
end interface
contains
subroutine SetDeltaUV_cll(v)
real(8) :: v
uv=v
end subroutine
subroutine SetDeltaIR_cll(v,w)
real(8) :: v,w
ir1=v
ir2=w
end subroutine
subroutine tadpole(t,u,m,n,r,e)
integer :: n,r
complex(8) :: t(0:r,0:r,0:r,0:r),u(0:r,0:r,0:r,0:r),m(n)
real(8) :: e(0:r)
t=10d0+2d0*uv+3d0*ir1+5d0*(ir2-acos(-1d0)**2/6d0)
u=2d0
e=0d0
calls=calls+1
end subroutine
subroutine tensor(t,u,p,s,m,n,r,e)
integer :: n,r
complex(8) :: t(0:r,0:r,0:r,0:r),u(0:r,0:r,0:r,0:r),m(n)
complex(8) :: p(0:3,n-1),s(n*(n-1)/2)
real(8) :: e(0:r)
call tadpole(t,u,m,n,r,e)
end subroutine
subroutine INITCACHESYSTEM_CLL(n,r)
integer :: n,r
resets=resets+1
end subroutine
'''
        for name in ('SwitchOffCache_cll', 'SwitchOnCache_cll', 'InitEvent_cll'):
            module += ('subroutine %s(i)\ninteger,optional :: i\n'
                       'end subroutine\n') % name
        for name in ('SetMuUV2_cll', 'SetMuIR2_cll'):
            module += ('subroutine %s(v)\nreal(8) :: v\nend subroutine\n') % name
        module += ('subroutine SwitchOffCacheSystem_cll()\nend subroutine\n'
                   'end module\nsubroutine INITCOLLIER()\nend subroutine\n'
                   'subroutine cite(a,b)\ncharacter(*) :: a,b\nend subroutine\n'
                   'subroutine TEST_SWITCH_ORDER()\nend subroutine\n')
        (self.work / 'collier.f90').write_text(module)
        (self.work / 'driver.f').write_text('''
      PROGRAM CHECK
      USE COLLIER
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      REAL*8 MU_R,P(0:3,1),PDEN(0:3,0),EXPECTED(3,0:3)
      COMMON/TEST_COUPL/MU_R
      COMPLEX*16 M(1),T(0:0,3),E(0:0,3)
      INTEGER ID,R,I,J,PREVIOUS
      COMMON/TEST_LOOP/ID,R
      LOGICAL CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT,CLINIT
      COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,
     &SAMURAIINIT,NINJAINIT,CLINIT
      INTEGER ACTIVE,COUNTS(4),CALLS_IN_EVENT(4)
      LOGICAL INIT_EVENT
      COMMON/TEST_COLLIER_CACHE_STATUS/ACTIVE,COUNTS,CALLS_IN_EVENT,
     &INIT_EVENT
      CALL DEFAULTPARAM()
      MU_R=1D0
      P=0D0
      M=1D0
      ID=1
      R=0
      CLINIT=.FALSE.
      ACTIVE=1
      COUNTS=-1
      CALLS_IN_EVENT=0
      INIT_EVENT=.TRUE.
      COLLIERUseCacheForPoles=.TRUE.
      EXPECTED(:,0)=(/10D0,5D0,5D0/)
      EXPECTED(:,1)=(/10D0,2D0,0D0/)
      EXPECTED(:,2)=(/10D0,3D0,5D0/)
      EXPECTED(:,3)=(/10D0,0D0,0D0/)
C     Deliberately retain all caches between mode changes and CT modes.
      DO J=1,2
        DO I=0,3
          LoopPoleMode=I
          CALL TEST_COLLIERLOOP(1,1,0,P,PDEN,M,T,E)
          IF(ANY(ABS(T(0,:)-EXPECTED(:,I)).GT.1D-12)) STOP 21
          PREVIOUS=CALLS
          CALL TEST_COLLIERLOOP(2,1,0,P,PDEN,M,T,E)
          IF(ANY(ABS(T(0,:)-EXPECTED(:,I)).GT.1D-12)) STOP 22
          IF(CALLS.NE.PREVIOUS) STOP 23
        ENDDO
      ENDDO
      IF(RESETS.NE.7) STOP 24
C     Legacy choices remain active after returning from an explicit mode.
      LoopPoleMode=-1
      COLLIERComputeUVpoles=.FALSE.
      CALL TEST_COLLIERLOOP(1,1,0,P,PDEN,M,T,E)
      IF(ANY(ABS(T(0,:)-EXPECTED(:,2)).GT.1D-12)) STOP 25
      COLLIERComputeIRpoles=.FALSE.
      CALL TEST_COLLIERLOOP(1,1,0,P,PDEN,M,T,E)
      IF(ANY(ABS(T(0,:)-EXPECTED(:,3)).GT.1D-12)) STOP 26
      WRITE(*,*) 'COLLIER SELECTOR AND CACHE PASSED'
      END
''')
        self.compile('collier.f90', 'wrapper.f', 'driver.f')
        result = self.run_command(str(self.work / 'check'))
        self.assertIn('COLLIER SELECTOR AND CACHE PASSED', result.stdout)

    def test_invalid_mode_rejected_by_card_and_public_setter(self):
        source = (TEMPLATES / 'loop_matrix_standalone.inc').read_text()
        source = source[source.index(
            '      SUBROUTINE %(proc_prefix)sSET_LOOP_POLE_MODE(MODE)'):]
        source = source[:source.index(
            '\t  SUBROUTINE %(proc_prefix)sCOLLIER_COMPUTE_UV_POLES')]
        with FortranWriter(str(self.work / 'setter.f')) as writer:
            writer.writelines(source % {'proc_prefix': 'TEST_'})
        (self.work / 'invalid.f').write_text('''
      PROGRAM CHECK
      IMPLICIT NONE
      INTEGER MODE
      CHARACTER*2 ARG
      CHARACTER*512 FILENAME
      CALL DEFAULTPARAM()
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      READ(ARG,*) MODE
      IF(MODE.EQ.0) THEN
        FILENAME='card.dat'
        CALL MADLOOPPARAMREADER(FILENAME,.FALSE.)
      ELSE
        CALL TEST_SET_LOOP_POLE_MODE(MODE)
      ENDIF
      WRITE(*,*) 'INVALID MODE ACCEPTED'
      END
''')
        self.compile('setter.f', 'invalid.f')
        for mode in (-2, 4):
            for route in ('card', 'setter'):
                with self.subTest(mode=mode, route=route):
                    (self.work / 'card.dat').write_text(
                        '#LoopPoleMode\n%d\n' % mode)
                    result = self.run_command(str(self.work / 'check'),
                                              '0' if route == 'card' else str(mode),
                                              success=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('LoopPoleMode must be', result.stdout)
                    self.assertNotIn('INVALID MODE ACCEPTED', result.stdout)

    def test_nonoptimized_output_rejects_selected_poles(self):
        source = (TEMPLATES.parent / 'loop/loop_matrix_standalone.inc').read_text()
        guard = source[source.index(
            'C The non-optimized output uses CutTools'):]
        guard = guard[:guard.index('IF (LOCAL_ML_INIT) THEN')]
        with FortranWriter(str(self.work / 'nonoptimized.f')) as writer:
            writer.writelines('''
      PROGRAM CHECK
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      CHARACTER*2 ARG
      CALL DEFAULTPARAM()
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      READ(ARG,*) LoopPoleMode
''' + guard + '''
      WRITE(*,*) 'ACCEPTED'
      END
''')
        self.compile('nonoptimized.f')
        for mode in (-1, 0):
            self.assertIn('ACCEPTED', self.run_command(
                str(self.work / 'check'), str(mode)).stdout)
        for mode in (1, 2, 3):
            result = self.run_command(str(self.work / 'check'), str(mode),
                                      success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('loop_optimized_output enabled', result.stdout)

    def test_diagnostic_modes_bypass_and_restore_learned_zero_filters(self):
        source = (TEMPLATES / 'loop_matrix_standalone.inc').read_text()
        start = source.index('IF (BYPASS_FILTERS) THEN')
        bypass = source[start:source.index(
            'IF(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN', start)]
        start = source.index('IF (BYPASS_FILTERS) THEN', start + 1)
        restore = source[start:source.index(
            'C Make sure that we finish by emptying caches', start)]
        start = source.index('IF (WriteOutFilters.AND.')
        write_filters = source[start:source.index(
            'IF (BYPASS_FILTERS) THEN', start)]
        assignment = next(line for line in source.splitlines()
                          if line.startswith('BYPASS_FILTERS='))
        start = source.index('## if(LoopInduced) {\nC For loop-induced processes')
        count_training_points = source[start:source.index(
            'IF (USER_STAB_PREC.GT.0.0d0) THEN', start)]
        # Represent a saved combined-pole zero: UV and IR separately survive.
        # Reading deliberately restrictive files reproduces the stale-filter
        # case, without relying on a numerical library to infer a zero.
        (self.work / 'saved_loops.dat').write_text('F\n')
        (self.work / 'saved_helicities.dat').write_text('0\n0\n')
        with FortranWriter(str(self.work / 'filters.f')) as writer:
            writer.writelines('''
      PROGRAM CHECK
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER NCOMB,NSQUAREDSO,NLOOPGROUPS
      PARAMETER(NCOMB=2,NSQUAREDSO=1,NLOOPGROUPS=1)
      LOGICAL GOODAMP(1,1),OLD_GOODAMP(1,1)
      INTEGER GOODHEL(2),OLD_GOODHEL(2),I,J,MODE,MODELIST(6),NTRY
      LOGICAL CHECKPHASE,HELDOUBLECHECKED,BYPASS_CHECK,BYPASS_FILTERS
      LOGICAL OLD_CHECKPHASE,OLD_HELDOUBLECHECKED
      LOGICAL FOUNDHELFILTER,FOUNDLOOPFILTER,EXISTS
      CHARACTER*32 HelFilterFN,LoopFilterFN
      REAL*8 UV,IR
      CALL DEFAULTPARAM()
      USELOOPFILTER=.TRUE.
      HelFilterFN='new_helicities.dat'
      LoopFilterFN='new_loops.dat'
      OPEN(1,FILE='saved_loops.dat',STATUS='OLD')
      READ(1,*) GOODAMP
      CLOSE(1)
      OPEN(1,FILE='saved_helicities.dat',STATUS='OLD')
      READ(1,*) GOODHEL(1)
      READ(1,*) GOODHEL(2)
      CLOSE(1)
      MODELIST=(/1,2,0,3,1,2/)
      NTRY=2
      BYPASS_CHECK=.FALSE.
      CHECKPHASE=.FALSE.
      HELDOUBLECHECKED=.TRUE.
      FOUNDHELFILTER=.FALSE.
      FOUNDLOOPFILTER=.FALSE.
      DO MODE=1,6
        LoopPoleMode=MODELIST(MODE)
''' + assignment + '\n' + count_training_points + write_filters + bypass + '''
        IF(NTRY.NE.2) STOP 30
        IF(ANY(GOODHEL.NE.1).OR..NOT.ALL(GOODAMP)) STOP 31
        IF(CHECKPHASE.OR..NOT.HELDOUBLECHECKED) STOP 32
        UV=0D0
        IF(GOODAMP(1,1)) THEN
          IF(GOODHEL(1).GT.0) UV=UV+3D0
          IF(GOODHEL(2).GT.0) UV=UV+5D0
        ENDIF
        IR=-UV
        IF(UV.NE.8D0.OR.IR.NE.-8D0) STOP 33
''' + restore + '''
        IF(ANY(GOODHEL.NE.0).OR.ANY(GOODAMP)) STOP 34
        IF(CHECKPHASE.OR..NOT.HELDOUBLECHECKED) STOP 35
        IF(BYPASS_CHECK) STOP 36
        INQUIRE(FILE=HelFilterFN,EXIST=EXISTS)
        IF(EXISTS) STOP 37
        INQUIRE(FILE=LoopFilterFN,EXIST=EXISTS)
        IF(EXISTS) STOP 38
      ENDDO
C     Returning to legacy mode restores the caller's learned filters.
      LoopPoleMode=-1
''' + assignment + '\n' + bypass + '''
      IF(BYPASS_FILTERS.OR.ANY(GOODHEL.NE.0).OR.ANY(GOODAMP)) STOP 39
      WRITE(*,*) 'DIAGNOSTIC FILTER BYPASS PASSED'
      END
''', context={'LoopInduced': False})
        self.compile('filters.f')
        result = self.run_command(str(self.work / 'check'))
        self.assertIn('DIAGNOSTIC FILTER BYPASS PASSED', result.stdout)
