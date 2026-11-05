"""Every lane of the batched MC@NLO integrand must start like a new point."""

import os
import re
import shutil
import subprocess
import tempfile

import tests.unit_tests as unittest
from madgraph import MG5DIR


class TestMCLaneState(unittest.TestCase):

    def test_single_lane_integration_keeps_weight_lines_scalar(self):
        """Exercise the production reset/final-copy decisions in all modes.

        Single-lane integration already retains scalar contributions across
        folds. Copying their full capacities to vector storage was unused
        work. Generation must still supply that storage to event picking,
        and multiple lanes need it for their separate fold histories.
        """
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the weight-storage probe')
        with open(os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses',
                               'driver_mintMC.f')) as stream:
            source = stream.read()
        start = source.index('            icontr=0\n')
        reset = source[start:source.index('            virt_wgt_mint', start)]
        start = source.rindex('            f_vec(:,ivec) = f(:)\n')
        start = source.index('\n', start) + 1
        append = source[start:source.index('            MCcnt_vec', start)]
        probe = '''      program probe
      implicit none
      integer nexternal,ivec,icontr,vector_size,imode,nreset,nappend
      integer, allocatable :: itype(:)
      common /calls/ nreset,nappend
      nexternal=5
      ivec=1
      allocate(itype(20))
      do imode=0,2
         do vector_size=1,4,3
            nreset=0
            nappend=0
''' + reset + append + '''
            write(*,*) imode,vector_size,nreset,nappend
         enddo
      enddo
      end
      subroutine reset_weight_lines_vec(n)
      implicit none
      integer n,nreset,nappend
      common /calls/ nreset,nappend
      nreset=nreset+1
      end
      subroutine append_weight_lines(n,i)
      implicit none
      integer n,i,nreset,nappend
      common /calls/ nreset,nappend
      nappend=nappend+1
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_mc_weight_storage_') as tmp:
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
        rows = [tuple(map(int, line.split())) for line in output.splitlines()]
        self.assertEqual(rows, [(0, 1, 0, 0), (0, 4, 1, 1),
                                (1, 1, 0, 0), (1, 4, 1, 1),
                                (2, 1, 1, 1), (2, 4, 1, 1)])

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
      logical check_born_amps,born_checks_amps
      common/c_born_amps_check/check_born_amps,born_checks_amps
      vector_size=4
      check_born_amps=.false.
      born_checks_amps=.false.
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
