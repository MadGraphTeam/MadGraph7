"""The optimised Sudakov-table generator (vendor/SudGen/sudgen_fast.cc).

Set PYTHIA8 to a Pythia 8.3 installation to run these tests, for example:

  PYTHIA8=/path/to/pythia8 python3 -m unittest tests.unit_tests.fks.test_sudgen_fast

They are not part of the CI job lists, which do not provide Pythia.
"""
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SUDGEN = ROOT / 'vendor/SudGen'


@unittest.skipUnless(os.environ.get('PYTHIA8') and shutil.which('g++'),
                     'requires g++ and a Pythia installation in PYTHIA8')
class TestSudgenFast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_sudgen_fast_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = Path(cls.tempdir.name)
        pythia = Path(os.environ['PYTHIA8']).resolve()
        sys.path.insert(0, str(SUDGEN))
        import sudgen_settings
        lines, cls.metadata = sudgen_settings.table_settings(str(pythia))
        cls.cmd = cls.work / 'sudgen.cmd'
        cls.cmd.write_text('\n'.join(lines) + '\n')
        flags = subprocess.check_output([str(pythia / 'bin/pythia8-config'),
                                         '--cxxflags', '--libs']).decode().split()
        cls.exe = cls.work / 'sudgen_fast'
        subprocess.run(['g++', '-O2', '-std=c++17', str(SUDGEN / 'sudgen_fast.cc'),
                        '-o', str(cls.exe)] + flags + ['-ldl'], check=True,
                       capture_output=True, text=True)

    def run_row(self, name, *args):
        out = self.work / name
        # type, id, ixm_first, ixm_last, nrep, seed, then the output file and
        # the optional direct-node arguments.
        subprocess.run([str(self.exe), str(self.cmd)] + [str(a) for a in args[:6]]
                       + [str(out)] + [str(a) for a in args[6:]], check=True,
                       capture_output=True, text=True, timeout=600)
        headers, rows = [], {}
        for line in out.read_text().splitlines():
            if line.startswith('#'):
                headers.append(line[1:].split())
                continue
            ixm, ist, st, xm, delta, err = line.split()
            rows[int(ist)] = (float(st), float(delta), float(err))
        return headers, rows

    def test_settings_come_from_the_shower_interface(self):
        settings = self.metadata['settings']
        for key, value in (('TimeShower:alphaSvalue', '0.118'),
                           ('SpaceShower:alphaSvalue', '0.118'),
                           ('TimeShower:weightGluonToQuark', '1'),
                           ('TimeShower:dampenBeamRecoil', 'off'),
                           ('TimeShower:globalRecoil', 'on'),
                           ('SpaceShower:pdfMode', '1'),
                           ('TimeShower:MEcorrections', 'off'),
                           ('1:m0', '0.33')):
            with self.subTest(key=key):
                self.assertEqual(str(settings[key]), value)
        headers, _ = self.run_row('isr.dat', 1, 21, 20, 20, 1, 3, 1, 2, 'direct')
        initialised = dict(zip(headers[1][0::2], headers[1][1::2]))
        self.assertEqual(float(initialised['alphaSvalue']), .118)
        self.assertEqual(int(initialised['alphaSorder']), 1)
        self.assertEqual(int(initialised['pdfMode']), 1)

    def test_product_of_intervals_matches_direct_runs(self):
        # Final-state gluon dipole of 466 GeV: one direct run of 10^4 trials
        # from 7 TeV per node against the product row with 10x10^4 trials
        # per interval.
        _, direct = self.run_row('direct.dat', 2, 21, 35, 35, 1, 7, 1, 100, 'direct')
        _, product = self.run_row('product.dat', 2, 21, 35, 35, 10, 8)
        pulls = []
        for ist, (_, value, _) in direct.items():
            p, rel = product[ist][1], product[ist][2]
            if value < 1.:
                sigma = math.hypot(math.sqrt(max(p * (1. - p), 1e-12) / 1e4), p * rel)
                pulls.append((value - p) / sigma)
        self.assertGreater(len(pulls), 40)
        mean = sum(pulls) / len(pulls)
        rms = math.sqrt(sum(x * x for x in pulls) / len(pulls))
        self.assertLess(abs(mean), .5)
        self.assertLess(rms, 1.6)
        values = [product[ist][1] for ist in sorted(product)]
        self.assertTrue(all(b >= a for a, b in zip(values[:-1], values[1:])))
        self.assertEqual(values[-1], 1.)
        # Deep in the Sudakov region the product keeps a few-percent error
        # where a single run of 10^4 trials returns zero.
        self.assertGreater(product[1][1], 0.)
        self.assertLess(product[1][2], .1)


if __name__ == '__main__':
    unittest.main()
