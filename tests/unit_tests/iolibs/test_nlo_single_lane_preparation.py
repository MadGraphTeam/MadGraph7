"""Single-lane preparation must leave unsupported execution paths unchanged."""
import itertools
import os
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
    def test_rejected_last_fold_keeps_born_colour_fallback(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the colour lifecycle probe')
        directory = os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses')
        with open(os.path.join(directory, 'driver_mintMC.f')) as stream:
            source = stream.read()
        start = source.index('         call generate_momenta(nndim,iconfig,jac,x,p)',
                             source.index('      function sigintF_vec(ifl)'))
        end = source.index('         if (passcuts_nbody) then', start)
        preparation = source[start:end]
        with open(os.path.join(directory, 'add_write_info.f')) as stream:
            source = stream.read()
        start = source.index('      if (colour_connections(1,1).lt.0) then')
        end = source.index('         xtarget=ran2()*sumborn', start)
        fallback = source[start:end]
        program = '''      module state
      implicit none
      double precision sborn_jamp2(2,1)
      integer fold,prepared
      end module
      program probe
      use state
      implicit none
      integer nndim,iconfig,izero,ivec,nFKS_picked_nbody,ickkw
      integer nFKSprocess,i,max_bcol,iBornGraph
      parameter(max_bcol=2)
      integer colour_connections(2,2)
      logical single_lane_preparation,passcuts_nbody,passcuts
      logical icolamp(2,1,1)
      double precision jac,x(1),p(0:3,4),p_born(0:3,3)
      double precision vegas_wgt,p1_cnt(0:3,4,-2:2),rwgt
      double precision born_amp2(1),sborn_amp2(1,1)
      double precision born_jamp2(2),born_amp_split(1)
      double precision sborn_amp_split(1,1)
      double precision born_amp_split_cnt(1,1,1)
      double precision sborn_amp_split_cnt(1,1,1,1)
      double precision born_ans_cnt(1,1),sborn_ans_cnt(1,1,1)
      double precision born_saveamp(1,1),sborn_saveamp(1,1,1)
      double precision wgt_born,swgt_born(1),jamp2(2),sumborn
      external passcuts
      single_lane_preparation=.true.
      nndim=1
      iconfig=1
      izero=0
      ivec=1
      nFKS_picked_nbody=1
      nFKSprocess=1
      ickkw=0
      iBornGraph=1
      icolamp=.true.
      p_born=1d0
      prepared=0
      do fold=1,2
c Poison the previous fold, then reject the last fold's Born cuts.
         sborn_jamp2=-999d0
''' + preparation + '''
         if (passcuts_nbody.neqv.(fold.eq.1)) stop 21
         if (prepared.ne.fold) stop 22
         colour_connections=-1
''' + fallback + '''
         if (sumborn.ne.3d0*fold) stop 23
         if (any(jamp2.ne.(/1d0,2d0/)*fold)) stop 24
      endif
      enddo
      print *, 'PASS'
      stop
 12   stop 25
      end
      subroutine prepare_single_lane_cache(p,nbody,cut1,cut2)
      use state
      double precision p(0:3,4)
      logical nbody,cut1,cut2
      if (.not.(nbody.and.cut1.and.cut2)) stop 26
      sborn_jamp2(:,1)=(/1d0,2d0/)*fold
      prepared=prepared+1
      end
      logical function passcuts(p,rwgt)
      use state
      double precision p(*),rwgt
      passcuts=fold.eq.1
      end
      subroutine generate_momenta(n,c,j,x,p)
      integer n,c
      double precision j,x(*),p(*)
      end
      subroutine compute_prefactors_nbody(w)
      double precision w
      end
      subroutine set_cms_stuff(i)
      integer i
      end
      subroutine set_shower_scale_noshape(p,i)
      double precision p(*)
      integer i
      end
      subroutine set_FxFx_scale(i,p,j)
      integer i,j
      double precision p(*)
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_generation_colour_') as tmp:
            path = os.path.join(tmp, 'probe.f')
            binary = os.path.join(tmp, 'probe')
            with open(path, 'w') as stream:
                stream.write(program)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                '-ffixed-line-length-132', path, '-o', binary], cwd=tmp,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            run = subprocess.run([binary], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertEqual(run.stdout.strip(), 'PASS')

    def test_scalar_cache_defaults_preserve_generation_fold_resets(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the cache lifecycle probe')
        with open(os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses',
                               'driver_mintMC.f')) as stream:
            source = stream.read()
        source = source[source.index('      function sigintF_vec(ifl)'):]
        start = source.index('      if (firsttime) then')
        first = source[start:source.index('      if (ifl.eq.0) then', start)]
        start = source.index('      if (ifl.eq.0 .or. ifl.eq.1) then')
        reset = source[start:source.index('         if (ifl.eq.0) then', start)]
        program = '''      program probe
      implicit none
      logical firsttime,single_lane_preparation,goodhel_set
      integer policy,callback,ifl,i,counts,proc_map(0:1,0:1)
      integer sum,ini_fin_fks,born_class_map(1)
      integer imode
      integer BORN_FLAVOR_INDEX_D(1),goodhel_calls(1)
      common /reset_calls/counts
      BORN_FLAVOR_INDEX_D=1
      do imode=0,2
      do policy=0,1
         firsttime=.true.
         single_lane_preparation=policy.eq.1
         counts=0
         do callback=1,12
            ifl=mod(callback-1,3)
''' + first + reset + '''         endif
         enddo
         if (policy.eq.0.and.counts.ne.8) stop 1
         if (policy.eq.1.and.imode.ne.2.and.counts.ne.1) stop 2
         if (policy.eq.1.and.imode.eq.2.and.counts.ne.8) stop 3
      enddo
      enddo
      print *, 'PASS'
      end
      subroutine setup_proc_map(sum,map,ini)
      integer sum,map(0:1,0:1),ini
      map=1
      end
      subroutine find_iproc_map()
      end
      subroutine setup_event_attributes()
      end
      subroutine reset_storage()
      integer counts
      common /reset_calls/counts
      counts=counts+1
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_cache_lifecycle_') as tmp:
            path = os.path.join(tmp, 'probe.f')
            binary = os.path.join(tmp, 'probe')
            with open(path, 'w') as stream:
                stream.write(program)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                '-ffixed-line-length-132', path, '-o', binary], cwd=tmp,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            run = subprocess.run([binary], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertEqual(run.stdout.strip(), 'PASS')

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
            madmatrix_output.ProcessExporterMadMatrixNLOReal._write_fortran_routing(
                None, manifest, tmp)
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
