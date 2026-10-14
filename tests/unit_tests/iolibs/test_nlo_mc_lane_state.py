"""Every lane of the batched MC@NLO integrand must start like a new point."""

import os
import re
import shutil
import subprocess
import tempfile

import tests.unit_tests as unittest
from madgraph import MG5DIR


class TestMCLaneState(unittest.TestCase):

    def test_every_lane_starts_its_nbody_part_fresh(self):
        """Compile the head of the production lane loop of sigintF_vec.

        The n+1-body loop of a lane sets nbody=.false.; setfksfactor (called
        by update_fks_dir) then gives the Born the symmetry factor of the
        real emission, 2 when the real has two final-state gluons. When the
        next lane inherited that state, its Born, virtual and integrated
        counterterms were doubled: q q~ > e+ e- g and g g > t t~ g came out
        ~x1.9 at vector_size > 1. Each lane must reach the n-body
        update_fks_dir with nbody=.true. and no cached Born, like every point
        of the scalar driver.
        """
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the lane-state probe')
        with open(os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses',
                               'driver_mintMC.f')) as stream:
            source = stream.read()
        anchor = source.index('nFKS_picked_nbody=lane_nbody(ivec)')
        start = source.rindex('      do ivec=1,vector_size\n', 0, anchor)
        stop = source.index('call update_fks_dir(nFKS_picked_nbody)', anchor)
        stop = source.index('\n', stop) + 1
        block = source[start:stop]
        self.assertEqual(len(re.findall(r'^\s+do ivec=', block, re.MULTILINE)),
                         1, block)
        probe = '''      program probe
      implicit none
      integer nexternal
      parameter (nexternal=5)
      integer vector_size,ivec,nFKS_picked_nbody,lane_nbody(4)
      integer, allocatable :: itype_vec(:,:)
      double precision vol1,born_vol1,lane_vol1(4),lane_born_vol1(4)
      logical nbody
      common/cnbody/nbody
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      vector_size=4
      lane_nbody=(/3,1,4,2/)
      lane_vol1=0.25d0
      lane_born_vol1=1d0
c State left by the batched momenta and amplitudes of all lanes
      nbody=.true.
      calculatedBorn=.true.
''' + block + '''c The n+1-body part of this lane
         nbody=.false.
         calculatedBorn=.true.
      enddo
      end

      subroutine update_fks_dir(nFKS)
      implicit none
      integer nFKS
      logical nbody
      common/cnbody/nbody
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      write(*,*) nFKS,nbody,calculatedBorn
      end

      subroutine retrieve_weight_lines(n,ivec)
      implicit none
      integer n,ivec
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_mc_lane_state_') as tmp:
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
        calls = [line.split() for line in output.splitlines() if line.strip()]
        self.assertEqual([int(call[0]) for call in calls], [3, 1, 4, 2])
        for lane, (nfks, nbody, cached) in enumerate(calls, 1):
            with self.subTest(lane=lane):
                self.assertEqual(nbody, 'T')
                self.assertEqual(cached, 'F')
