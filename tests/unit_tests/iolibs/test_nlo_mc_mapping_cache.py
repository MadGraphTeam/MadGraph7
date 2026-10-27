"""Unused resonance/event-projection Born caches must not be prepared."""
import os
import shutil
import subprocess
import tempfile

import tests.unit_tests as unittest
from madgraph import MG5DIR


class TestMCMappingCache(unittest.TestCase):
    def test_mapping_setup_keeps_resonances_and_nonstandard_beams(self):
        """Compile the actual serial prepass block and count prepared slots.

        The event mapping is consumed for resonances. The no-radiation
        mapping must remain available for beams without guaranteed event
        projection, including dressed leptons. Ordinary pp, photon, and
        fixed-energy beam pairs need no no-radiation cache.
        """
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the mapping-cache probe')
        with open(os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses',
                               'driver_vec.f')) as stream:
            source = stream.read()
        start = source.index('      subroutine generate_momenta_vector(')
        anchor = '     $          .not.(passcuts_nbody.or.passcuts_n1body)) cycle\n'
        start = source.index(anchor, start) + len(anchor)
        stop = source.index('            call set_alphaS(p1_cnt(0,1,0))', start)
        block = source[start:stop]
        probe = '''      program probe
      implicit none
      integer lpp(2),beams(2,6),i,r,iFKS,ivec,icoup,ncalls,slots
      logical granny_is_res,need_mapping_born(2,1,1)
      double precision p_born_ev(0:3,4),p_born_norad(0:3,4)
      double precision spb_ev(0:3,4,1,1),spb_norad(0:3,4,1,1)
      common /calls/ ncalls,slots
      beams=reshape((/1,1,2,2,0,0,1,-1,3,-3,4,4/),(/2,6/))
      iFKS=1
      ivec=1
      icoup=1
      p_born_ev=1d0
      p_born_norad=2d0
      do i=1,6
         lpp=beams(:,i)
         do r=0,1
            granny_is_res=r.eq.1
            need_mapping_born=.true.
            ncalls=0
            slots=0
''' + block + '''
            write(*,*) i,r,need_mapping_born(:,1,1),ncalls,slots
         enddo
      enddo
      end
      subroutine set_alphaS(p)
      implicit none
      double precision p(0:3,4)
      integer ncalls,slots
      common /calls/ ncalls,slots
      ncalls=ncalls+1
      end
      subroutine set_alphaS_vec_from_scalar(slot)
      implicit none
      integer slot,ncalls,slots
      common /calls/ ncalls,slots
      slots=slots+2**(slot-1)
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_mc_mapping_cache_') as tmp:
            path = os.path.join(tmp, 'probe.f')
            with open(path, 'w') as stream:
                stream.write(probe)
            binary = os.path.join(tmp, 'probe')
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                                       '-ffixed-line-length-132', path, '-o', binary],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            output = subprocess.check_output([binary], text=True)
        rows = [line.split() for line in output.splitlines()]
        self.assertEqual(len(rows), 12)
        for beam, resonance, ev, norad, calls, slots in rows:
            needs_ev = int(resonance) == 1
            needs_norad = int(beam) > 3
            with self.subTest(beam=beam, resonance=resonance):
                self.assertEqual(ev == 'T', needs_ev)
                self.assertEqual(norad == 'T', needs_norad)
                self.assertEqual(int(calls), int(needs_ev) + int(needs_norad))
                self.assertEqual(int(slots), int(needs_ev) + 2 * int(needs_norad))
