"""The MC Born partition must not depend on adaptive FKS sampling probabilities."""

import os
import re
import shutil
import subprocess
import tempfile

import tests.unit_tests as unittest
from madgraph import MG5DIR


class TestMCBornPartition(unittest.TestCase):

    def test_nbody_shares_preserve_the_mg5_partition(self):
        """Compile the production Jacobian block and vary the sampling grid.

        Signed normalization alone cannot detect the old class-probability
        allocation: it changes S-event cancellation and shower-group populations
        while leaving the signed integral unbiased. Each active group must carry
        an equal share of its physical Born class, as in ungrouped MG5.
        """
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the MC partition probe')
        with open(os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses',
                               'driver_mintMC.f')) as stream:
            source = stream.read()
        start = source.index('c Also the Born needs to be included')
        stop = start + re.search(r'^ +call generate_momenta\(', source[start:],
                                 re.MULTILINE).start()
        block = source[start:stop]
        probe = '''      program probe
      implicit none
      integer n,g,labels(8)
      logical physical
      double precision probs(8),initial,jac,pclass
      read(*,*) n,physical,initial
      read(*,*) labels
      read(*,*) probs
      do g=1,n
         pclass=sum(probs(1:n),mask=labels(1:n).eq.labels(g))
         jac=initial
         call normalize(n,physical,labels,g,probs(g),pclass,jac)
         write(*,'(es26.17)') jac
      enddo
      end

      subroutine normalize(n,physical,labels,g,p,pclass,jac)
      implicit none
      integer n,g,labels(8),proc_map(0:8,0:8),born_class_map(8)
      integer lane_group(4),ivec
      logical physical,HAS_PHYSICAL_FKS_CLASSES
      double precision p,pclass,jac,vol1,born_vol1
      proc_map=0
      proc_map(0,0)=n
      born_class_map=labels
      lane_group=1
      ivec=3
      lane_group(ivec)=g
      HAS_PHYSICAL_FKS_CLASSES=physical
      vol1=p
      born_vol1=pclass
''' + block + '''      end
'''
        cases = [
            # One physical gg Born, with four very unevenly sampled groups.
            (True, [7, 7, 7, 7], [.1, .4, .2, .3], [.25]*4),
            # Sparse labels, unequal multiplicities and a singleton class.
            (True, [3, 11, 3, 11, 11, 90],
             [.03, .10, .17, .20, .35, .15],
             [.5, 1./3, .5, 1./3, 1./3, 1.]),
            (True, [31], [1.], [1.]),
            # Ungrouped luminosities span all groups, regardless of labels.
            (False, [3, 11, 3, 11], [.1, .4, .2, .3], [.25]*4),
        ]
        with tempfile.TemporaryDirectory(prefix='mg7_mc_partition_') as tmp:
            path = os.path.join(tmp, 'probe.f')
            with open(path, 'w') as stream:
                stream.write(probe)
            binary = os.path.join(tmp, 'probe')
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                                       '-ffixed-line-length-132', path, '-o', binary],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            for physical, labels, probabilities, shares in cases:
                for initial in (1., .5):
                    with self.subTest(physical=physical, labels=labels,
                                      initial=initial):
                        n = len(labels)
                        # Inactive process-map entries must not add multiplicity.
                        padded = labels + [labels[0]]*(8-n)
                        data = '%d %s %s\n%s\n%s\n' % (
                            n, '.true.' if physical else '.false.', initial,
                            ' '.join(map(str, padded)),
                            ' '.join(map(str, probabilities + [0.]*(8-n))))
                        output = subprocess.check_output([binary], input=data,
                                                         text=True)
                        jacobians = [float(line) for line in output.splitlines()]
                        self.assertEqual(len(jacobians), n)
                        weights = [p*j/initial for p, j in zip(probabilities, jacobians)]
                        for weight, share in zip(weights, shares):
                            self.assertAlmostEqual(weight, share, places=14)
                        classes = set(labels) if physical else {None}
                        for label in classes:
                            total = sum(w for w, b in zip(weights, labels)
                                        if label is None or b == label)
                            self.assertAlmostEqual(total, 1., places=14)
