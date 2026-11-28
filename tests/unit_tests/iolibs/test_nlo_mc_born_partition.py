"""The MC Born partition is independent of adaptive FKS sampling probabilities."""

import subprocess

from tests.unit_tests.fks.test_born_flow_rejection import NativeIntegrandFixture


class TestMCBornPartition(NativeIntegrandFixture):
    def test_nbody_shares_preserve_the_mg5_partition(self):
        # Read the Jacobian entering real sigintF_advance's Born sampler for
        # each group. Labels are deliberately non-contiguous, and the fixture
        # maps groups to sectors in reverse order. Inactive labels also match.
        cases = [
            (True, [7] * 4, [.1, .4, .2, .3]),
            (True, [3, 11, 3, 11, 11, 90], [.03, .10, .17, .20, .35, .15]),
            (True, [31], [1.]),
            (False, [3, 11, 3, 11], [.1, .4, .2, .3]),
        ]
        for physical, labels, probabilities in cases:
            for orientation in (0, 1):
                with self.subTest(physical=physical, labels=labels, orientation=orientation):
                    count = len(labels)
                    padded = labels + [labels[0]] * (8-count)
                    probs = probabilities + [0.] * (8-count)
                    data = (f"{count} {'T' if physical else 'F'} {orientation}\n" +
                            ' '.join(map(str, padded)) + '\n' +
                            ' '.join(map(str, probs)) + '\n')
                    result = subprocess.run([self.executable, 'partition'], input=data,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    rows = [r.split() for r in result.stdout.splitlines() if r.startswith('JAC ')]
                    self.assertEqual(len(rows), count)
                    initial = 1. if orientation == 0 else .5
                    for probability, (_, sector, jac) in zip(probabilities, rows):
                        label = labels[int(sector)-1]
                        share = 1. / (labels.count(label) if physical else count)
                        self.assertAlmostEqual(probability * float(jac) / initial, share, places=14)
