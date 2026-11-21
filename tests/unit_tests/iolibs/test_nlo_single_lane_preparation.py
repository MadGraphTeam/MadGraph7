"""Single-lane preparation must leave unsupported execution paths unchanged."""
import itertools
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import tests.unit_tests as unittest
from madgraph import MG5DIR
from madgraph.iolibs import export_fks, file_writers
from madmatrix import output as madmatrix_output


class TestSingleLanePreparation(unittest.TestCase):
    def run_fixed_probe(self, program):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('requires gfortran')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.f'
            executable = Path(tmp) / 'probe'
            path.write_text(program)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                '-ffixed-line-length-none', path, '-o', executable], cwd=tmp,
                capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            run = subprocess.run([executable], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            self.assertEqual(run.stdout.strip(), 'PASS')

    def test_event_writer_uses_selected_hard_colours_or_selected_born_flow(self):
        source = (Path(MG5DIR)/'Template/NLO/SubProcesses/add_write_info.f').read_text()
        start = source.index('      iflow=abs(born_flow_picked)')
        end = source.index('      do i=1,nexpart', start)
        self.run_fixed_probe('''      program probe
      implicit none
      integer nexternal
      parameter(nexternal=4)
      integer jpart(7,nexternal),colour_connections(2,nexternal)
      integer iflow,born_flow_picked,idum,i
      logical Hevents
      do i=1,nexternal
         colour_connections(:,i)=(/100+i,200+i/)
      enddo
      Hevents=.true.
      born_flow_picked=-2
      do
         jpart=-999
''' + source[start:end] + '''
         if(Hevents)then
            if(any(jpart(4:5,:)/=colour_connections))stop 1
            Hevents=.false.
            born_flow_picked=3
         else
            if(any(jpart(4:5,:)/=3))stop 2
            exit
         endif
      enddo
      print *, 'PASS'
      end
      subroutine fill_icolor_S(flow,jpart,idum)
      integer flow,jpart(7,4),idum
      jpart(4:5,:)=flow
      end
''')

    def test_fold_resets_do_not_discard_the_parked_born_sample(self):
        source = (Path(MG5DIR)/'Template/NLO/SubProcesses/driver_mintMC.f').read_text()
        start = source.index('      if(point%status.ne.mc_wait_real)then')
        end = source.index('      have_sum_born=.false.', start)
        self.run_fixed_probe('''      program probe
      implicit none
      type born_sample
         logical valid
      end type
      type phase_space
         type(born_sample) born
      end type
      type native_point
         integer status
         type(phase_space) outer
         logical has_chart
      end type
      type(native_point) point
      integer ifl,status,mc_wait_real
      parameter(mc_wait_real=1)
      logical expected
      do status=0,1
      do ifl=0,2
         point%status=status
         point%has_chart=.true.
         point%outer%born%valid=.true.
''' + source[start:end] + '''      endif
         expected=status==mc_wait_real.or.ifl==2
         if(point%has_chart.neqv.expected)stop 1
         if(point%outer%born%valid.neqv.expected)stop 2
      enddo
      enddo
      print *, 'PASS'
      end
''')

    def test_emitted_backend_query_ignores_bridge_initialization_and_fallback(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the emitted-query probe')
        manifest = dict(real_matrix_elements=[dict(id=1,
            library_process_id='PMM_B1_R1', local_squared_orders=[[2]],
            backend_capabilities=dict(cuda=True, hip=True))],
            fks_rows=[dict(fks_row=1, real_me_id=1, local_to_global=[1])])
        with tempfile.TemporaryDirectory(prefix='mg7_backend_query_') as tmp:
            chooser_path = os.path.join(tmp, 'real_me_chooser.f')
            with file_writers.FortranWriter(chooser_path) as chooser, \
                    file_writers.FortranWriter(os.path.join(tmp, 'lum.f')) as lum:
                export_fks.ProcessExporterFortranFKS.write_real_wrappers(
                    None, chooser, lum, SimpleNamespace(real_processes=[]), None, None)
            with open(chooser_path) as stream:
                self.assertIn('MG7_NLO_REAL_IS_FORTRAN', stream.read())
            exporter = object.__new__(madmatrix_output.ProcessExporterMadMatrixNLOReal)
            exporter._write_fortran_routing(manifest, tmp)
            with open(chooser_path) as stream:
                chooser = stream.read()
            default = re.search(r'(?is)(logical function '
                r'mg7_nlo_real_is_fortran_default\(\).*?\n\s*end\s*\n)', chooser)
            self.assertIsNotNone(default)
            with open(os.path.join(tmp, 'nlo_real_offload.f90')) as stream:
                routing = stream.read()
            query = re.search(r'(?is)(logical function mg7_nlo_real_is_fortran'
                              r'\(\).*?end function)', routing)
            self.assertIsNotNone(query)
            program = '''module mg7_nlo_real_offload_state
implicit none
character(len=16) :: default_backend
logical :: bridge_available,initialization_attempted
end module
program probe
use mg7_nlo_real_offload_state
implicit none
logical, external :: mg7_nlo_real_is_fortran
logical, external :: mg7_nlo_real_is_fortran_default
'''
            for backend in ('fortran', 'scalar', 'avx2', 'cuda', 'hip'):
                for attempted, available in ((False, False), (True, False), (True, True)):
                    program += "default_backend='%s'\n" % backend
                    program += 'initialization_attempted=.%s.\n' % str(attempted).lower()
                    program += 'bridge_available=.%s.\n' % str(available).lower()
                    program += 'print *, mg7_nlo_real_is_fortran(),mg7_nlo_real_is_fortran_default()\n'
            program += 'end program\n' + query.group(1) + '\n' + default.group(1)
            path = os.path.join(tmp, 'query.f90')
            binary = os.path.join(tmp, 'query')
            with open(path, 'w') as stream:
                stream.write(program)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all', path,
                '-o', binary], cwd=tmp, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            rows = [line.split() for line in subprocess.check_output([binary], text=True).splitlines()]
        self.assertEqual(rows, [['T', 'T']]*3 + [['F', 'T']]*12)

    def test_cache_preparation_separates_born_and_selected_sector(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the preparation probe')
        directory = os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses')
        with open(os.path.join(directory, 'driver_vec.f')) as stream:
            source = stream.read()
        anchor = '      subroutine prepare_single_lane_cache('
        self.assertIn(anchor, source)
        helper = source[source.index(anchor):]
        helper = helper[:helper.index('\n      end\n')+len('\n      end\n')]
        program = '''      module driver_vec
      implicit none
      double precision spb(0:3,3,0:2,1)
      double precision sp1_cnt(0:3,4,-2:2,0:2,1)
      logical real_active(2,1),passcuts_nbody_vec(2,1)
      logical passcuts_n1body_vec(2,1)
      integer lane_nbody(1),lane_group(1)
      end module
      program probe
      use driver_vec
      implicit none
      double precision p(0:3,4),p_born(0:3,3)
      common /pborn/p_born
      double precision pc(0:3,4,-2:2),wc(-2:2),pw(-2:2),jc(-2:2)
      common/counterevnts/pc,wc,pw,jc
      integer sector,calls(4),slot
      common/c_nFKSprocess/sector
      common/counts/calls,slot
      include 'timing_variables.inc'
      calls=0
      slot=0
      t_vecamp=0
      t_vecreal=0
      lane_nbody=2
      lane_group=2
      p=7d0
      p_born=3d0
      pc=5d0
      sector=2
      call prepare_single_lane_cache(p,.true.,.true.,.true.)
      if (any(calls.ne.(/1,1,0,0/)).or.slot.ne.9) stop 1
      if (any(spb(:,:,0,1).ne.3d0)) stop 2
      if (any(sp1_cnt(:,:,:,0,1).ne.5d0)) stop 3
c Poison the previous sector's dispatch flag; it must not be replayed.
      real_active=.true.
      call prepare_single_lane_cache(p,.false.,.true.,.false.)
      if (any(calls.ne.(/1,2,1,1/))) stop 4
      if (.not.passcuts_nbody_vec(2,1)) stop 5
      if (passcuts_n1body_vec(2,1)) stop 6
      if (t_vecamp.lt.0.or.t_vecreal.lt.0) stop 7
      print *, 'PASS'
      end
      subroutine set_alphaS_and_vec(p,value)
      double precision p(0:3,4)
      integer value
      call set_alphaS(p)
      call set_alphaS_vec_from_scalar(value)
      end
      subroutine set_alphaS(p)
      double precision p(0:3,4)
      integer calls(4),slot
      common/counts/calls,slot
      if (any(p.ne.5d0)) stop 8
      calls(1)=calls(1)+1
      end
      subroutine set_alphaS_vec_from_scalar(value)
      integer calls(4),slot,value
      common/counts/calls,slot
      slot=value
      end
      subroutine prepare_sector_cache(p,lane)
      double precision p(0:3,4)
      integer lane,calls(4),slot
      common/counts/calls,slot
      if (lane.ne.1.or.any(p.ne.7d0)) stop 9
      calls(3)=calls(3)+1
      end
      subroutine amplitudes_ivec(map,rwgt,width,nbody,lane)
      use driver_vec
      integer map(0:2,0:2),width,nbody,lane,calls(4),slot
      double precision rwgt
      common/counts/calls,slot
      if (width.ne.1.or.lane.ne.1) stop 10
      if (nbody.eq.2) then
         if (any(map.ne.0)) stop 11
      else if (nbody.eq.0) then
         if (map(2,0).ne.1.or.map(2,1).ne.2) stop 12
         if (any(real_active)) stop 13
         real_active(2,1)=.true.
      else
         stop 14
      endif
      calls(2)=calls(2)+1
      end
      subroutine amplitudes_real_vec(map,width)
      use driver_vec
      integer map(0:2,0:2),width,calls(4),slot
      common/counts/calls,slot
      if (width.ne.1.or.count(real_active).ne.1) stop 15
      if (.not.real_active(2,1)) stop 16
      calls(4)=calls(4)+1
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_single_lane_cache_') as tmp:
            for name, content in [('nexternal.inc', '      integer nexternal\n      parameter(nexternal=4)\n'),
                                  ('nFKSconfigs.inc', '      integer fks_configs\n      parameter(fks_configs=2)\n')]:
                with open(os.path.join(tmp, name), 'w') as stream:
                    stream.write(content)
            shutil.copy2(os.path.join(directory, 'timing_variables.inc'), tmp)
            path = os.path.join(tmp, 'probe.f')
            binary = os.path.join(tmp, 'probe')
            with open(path, 'w') as stream:
                stream.write(program + helper)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                '-ffixed-line-length-132', path, '-o', binary], cwd=tmp,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            self.assertEqual(subprocess.check_output([binary], text=True).strip(), 'PASS')

    def test_preparation_scope_uses_actual_backend_and_configured_width(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the preparation probe')
        with open(os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses',
                               'driver_vec.f')) as stream:
            source = stream.read()
        anchor = '      logical function use_single_lane_preparation('
        self.assertIn(anchor, source)
        function = source[source.index(anchor):]
        function = function[:function.index('\n      end\n')+len('\n      end\n')]
        cases = list(itertools.product((1, 256), (1, 256), (-1, 0, 1, 2, 3),
                                       (0, 3), ('PYTHIA8', 'HERWIG6'),
                                       (False, True)))
        program = '''      program probe
      implicit none
      logical backend,use_single_lane_preparation
      external use_single_lane_preparation
      common /test_backend/ backend
'''
        for configured, active, mode, matching, shower, backend in cases:
            program += '      backend=.%s.\n' % str(backend).lower()
            program += "      write(*,*) use_single_lane_preparation(%d,%d,\n     $ %d,%d,'%s')\n" % (
                configured, active, mode, matching, shower)
        program += '''      end
      logical function mg7_nlo_real_is_fortran()
      implicit none
      logical backend
      common /test_backend/ backend
      mg7_nlo_real_is_fortran=backend
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_single_lane_') as tmp:
            path = os.path.join(tmp, 'probe.f')
            binary = os.path.join(tmp, 'probe')
            with open(path, 'w') as stream:
                stream.write(program + function)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                '-ffixed-line-length-132', path, '-o', binary],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            actual = subprocess.check_output([binary], text=True).split()
        self.assertEqual(len(actual), len(cases))
        for case, value in zip(cases, actual):
            configured, active, mode, matching, shower, backend = case
            expected = (configured == active == 1 and mode in (0, 1, 2)
                        and matching != 3 and shower == 'PYTHIA8' and backend)
            with self.subTest(case=case):
                self.assertEqual(value == 'T', expected)
