"""Coupling timer fusion preserves model state and momentum-update ordering."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import tests.unit_tests as unittest
from madgraph import MG5DIR



# The production routines open sampled timing categories (timing.f); the
# probes never sample (timing_sampled stays false), so no-op entries suffice.
TIMING_STUBS = """      subroutine timing_enter(category)
      integer category
      end
      subroutine timing_leave(category)
      integer category
      end
"""

class TestNLOCouplingTimer(unittest.TestCase):
    def test_combined_preparation_preserves_scalar_and_vector_state(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the coupling state probe')
        directory = Path(MG5DIR) / 'Template/NLO/SubProcesses'
        source = (directory / 'setscales.f').read_text()
        self.assertIn('      subroutine set_alphaS_and_vec(', source)
        source = source[source.index('      subroutine set_alphaS(xp)'):
                        source.index('      subroutine set_alphaS_vec(xp,vecid)')]
        # Instrument only timestamps; execute the production state transitions.
        source = source.replace('call cpu_time(', 'call probe_time(')
        program = '''      module couplings
      double precision g_vec(4),result_vec(4)
      end module
      program probe
      use couplings
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      double precision p(0:3,nexternal),pp(0:3,max_particles)
      common/momenta_pp/pp
      integer counts(4),ticks,k,mode,slot
      common/probe_counts/counts,ticks
      character*8 arg
      call get_command_argument(1,arg)
      read(arg,*) mode
      pp=0d0
      counts=0
      ticks=0
      g_vec=-1d0
      result_vec=-2d0
      muR_over_ref=1d0
      muF1_over_ref=1d0
      muF2_over_ref=1d0
      QES_over_ref=1d0
      fixed_ren_scale=.false.
      fixed_QES_scale=.false.
      t_coupl=0
      do k=1,4
         p=dble(k)
         fixed_couplings=k.eq.3
         slot=mod(k,3)+1
         if (mode.eq.0) then
            call set_alphaS(p)
            call set_alphaS_vec_from_scalar(slot)
         else
            call set_alphaS_and_vec(p,slot)
         endif
         write(*,*) 'STATE',g,g_vec,result_vec,pp,counts
      enddo
c set_alphaS times itself through sampled categories (timing.f); only
c set_alphaS_vec_from_scalar still reads the clock, twice per call. The
c fused set_alphaS_and_vec reads it never.
      if (mode.eq.0.and.ticks.ne.8) stop 1
      if (mode.eq.1.and.ticks.ne.0) stop 2
      end
      subroutine probe_time(t)
      real t
      integer counts(4),ticks
      common/probe_counts/counts,ticks
      ticks=ticks+1
      t=real(ticks)
      end
      subroutine set_ren_scale(p,dummy)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'coupl.inc'
      double precision p(0:3,nexternal),dummy,pp(0:3,max_particles)
      common/momenta_pp/pp
      integer counts(4),ticks
      common/probe_counts/counts,ticks
c A momentum-dependent model distinguishes old and newly copied PP.
      g=sum(p)+sum(pp)
      dummy=5d0
      counts(1)=counts(1)+1
      end
      subroutine set_fac_scale(p,dummy)
      double precision p(0:3,4),dummy(2)
      integer counts(4),ticks
      common/probe_counts/counts,ticks
      dummy=7d0
      counts(2)=counts(2)+1
      end
      subroutine set_QES_scale(p,dummy)
      implicit none
      include 'run.inc'
      double precision p(0:3,4),dummy
      integer counts(4),ticks
      common/probe_counts/counts,ticks
      dummy=9d0
      scale=9d0
      counts(3)=counts(3)+1
      end
      subroutine update_as_param_vec(slot)
      use couplings
      implicit none
      include 'genps.inc'
      integer slot,counts(4),ticks
      common/probe_counts/counts,ticks
      double precision pp(0:3,max_particles)
      common/momenta_pp/pp
      result_vec(slot)=g_vec(slot)+sum(pp)
      counts(4)=counts(4)+1
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_coupling_timer_') as tmp:
            tmp = Path(tmp)
            includes = {
                'genps.inc': '      integer max_particles\n      parameter(max_particles=6)\n',
                'nexternal.inc': '      integer nexternal\n      parameter(nexternal=4)\n',
                'coupl.inc': '      double precision g\n      common/strong/g\n',
                'run.inc': '''      logical fixed_ren_scale,fixed_couplings,fixed_QES_scale
      double precision scale,muR_over_ref,muF1_over_ref
      double precision muF2_over_ref,QES_over_ref
      common/probe_run/scale,muR_over_ref,muF1_over_ref,
     $ muF2_over_ref,QES_over_ref,fixed_ren_scale,fixed_couplings,
     $ fixed_QES_scale
'''}
            for name, text in includes.items():
                (tmp / name).write_text(text)
            shutil.copy2(directory / 'timing_variables.inc', tmp)
            (tmp / 'probe.f').write_text(program + source + TIMING_STUBS)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                '-ffixed-line-length-132', 'probe.f', '-o', 'probe'], cwd=tmp,
                capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            observed = []
            for mode in ('0', '1'):
                result = subprocess.run([str(tmp / 'probe'), mode], cwd=tmp,
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                observed.append([line for line in result.stdout.splitlines()
                                 if line.lstrip().startswith('STATE')])
            self.assertEqual(len(observed[0]), 4)
            self.assertEqual(observed[0], observed[1])
