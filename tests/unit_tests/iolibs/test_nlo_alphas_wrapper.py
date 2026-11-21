"""The LHAPDF alpha-s wrapper forwards each query without unused profiling."""
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

class TestNLOAlphasWrapper(unittest.TestCase):
    def test_active_pdf_set_scale_and_call_counts(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the alpha-s wrapper probe')
        root = Path(MG5DIR) / 'Template/NLO'
        source = (root / 'Source/alfas_functions_lhapdf6.f').read_text()
        start = source.index('      DOUBLE PRECISION FUNCTION ALPHAS(Q)')
        end = source.index('\n      END', start) + len('\n      END')
        # Count timestamp calls while executing the actual production wrapper.
        wrapper = source[start:end].replace('call cpu_time(', 'call probe_time(')
        program = '''      program probe
      implicit none
      integer active,queries,lookups,ticks,k,j
      common/probe_state/active,queries,lookups,ticks
      double precision q,alphas,got,want
      external alphas
      queries=0
      lookups=0
      ticks=0
      do k=1,10
         do j=0,3
            active=mod(k+j,4)
            q=10d0**(j-2)*dble(k)
            got=alphas(q)
            want=dble(active)+q/1024d0
            if (got.ne.want) stop 1
         enddo
      enddo
      if (queries.ne.40.or.lookups.ne.40) stop 2
      if (ticks.ne.0) stop 3
      write(*,*) 'ALPHAS_WRAPPER_PASS',queries,lookups,ticks
      end
      subroutine getnset(iset)
      implicit none
      integer iset,active,queries,lookups,ticks
      common/probe_state/active,queries,lookups,ticks
      queries=queries+1
      iset=active
      end
      double precision function alphasPDFm(iset,q)
      implicit none
      integer iset,active,queries,lookups,ticks
      common/probe_state/active,queries,lookups,ticks
      double precision q
      if (iset.ne.active) stop 4
      lookups=lookups+1
      alphasPDFm=dble(iset)+q/1024d0
      end
      subroutine probe_time(t)
      implicit none
      real t
      integer active,queries,lookups,ticks
      common/probe_state/active,queries,lookups,ticks
      ticks=ticks+1
      t=real(ticks)
      end
'''
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'probe.f').write_text(program + wrapper + '\n' + TIMING_STUBS)
            (directory / 'alfas.inc').write_text('')
            shutil.copy2(root / 'SubProcesses/timing_variables.inc', directory)
            build = subprocess.run(
                [compiler, '-O2', '-ffixed-line-length-132', 'probe.f', '-o', 'probe'],
                cwd=directory, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(directory / 'probe')], cwd=directory,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('ALPHAS_WRAPPER_PASS', result.stdout)
