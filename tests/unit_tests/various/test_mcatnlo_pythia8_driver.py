"""Live driver tests with MG7_TEST_PYTHIA8 and optional MG7_TEST_HEPMC3.

The Pythia8 prefix can also be found through pythia8-config on PATH.
"""

import gzip
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph.various import mcatnlo_shower
from madgraph.various.shower_card import ShowerCard
from tests.unit_tests.various.test_lhe_utilities import L


ROOT = Path(__file__).resolve().parents[3]
CONFIG = shutil.which('pythia8-config')
PYTHIA8 = os.environ.get('MG7_TEST_PYTHIA8') or (
    str(Path(CONFIG).resolve().parents[1]) if CONFIG else None)
HEPMC3 = os.environ.get('MG7_TEST_HEPMC3')


@unittest.skipUnless(PYTHIA8 and shutil.which('make'), 'requires Pythia8 and make')
class TestPythia8Driver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='mg7_live_py8_')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.pythia = mcatnlo_shower.Pythia8Installation(PYTHIA8)
        cls.executables = {}

    def build(self, hepmc=0):
        if hepmc in self.executables:
            return self.executables[hepmc]
        flags = ''
        if hepmc == 3:
            if HEPMC3:
                prefix = Path(HEPMC3).resolve()
                libdir = prefix / 'lib'
                if not libdir.is_dir():
                    libdir = prefix / 'lib64'
                flags = '-I%s -L%s -lHepMC3' % (prefix / 'include', libdir)
            elif self.pythia.hepmc()[0] == 3:
                flags = self.pythia.hepmc()[1]
            else:
                self.skipTest('requires HepMC3 (MG7_TEST_HEPMC3)')
        builddir = self.root / ('build_%d' % hepmc)
        builddir.mkdir()
        result = subprocess.run([
            'make', '-j2', '-f', str(ROOT / 'Template/NLO/MCatNLO/pythia8/Makefile'),
            'PYTHIA8_CONFIG=' + self.pythia.config,
            'CXX=' + self.pythia.cxx_compiler(),
            'HEPMC=%d' % hepmc, 'HEPMC_FLAGS=' + flags],
            cwd=builddir, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.executables[hepmc] = builddir / 'Pythia8.exe'
        return self.executables[hepmc]

    def shower(self, sample, limit, fxfx=False, hepmc=0, event_norm='average'):
        executable = self.build(hepmc)
        directory = tempfile.TemporaryDirectory(prefix='run_', dir=self.root)
        self.addCleanup(directory.cleanup)
        work = Path(directory.name)
        with gzip.open(ROOT / 'tests/input_files' / sample, 'rt') as stream:
            text = stream.read()
        event_file = work / 'events.lhe'
        event_file.write_text(text)
        header = L.LHEHeader(text[:text.index('<event')])
        card = ShowerCard()
        card['hadronize'] = False
        card['qcut'] = 30.
        card['njmax'] = 1
        contract = mcatnlo_shower.MatchingContract(
            parton_shower='PYTHIA8', ickkw=3 if fxfx else 0,
            ptj=20., born_coloured_partons=-1 if fxfx else 2)
        run = dict(nevents=limit, nevents_total=header.nevents,
                   event_norm=event_norm, seed=17,
                   mc_masses={pdg: mass for pdg, mass in header.mc_masses().items()
                              if pdg > 0},
                   masses={6: 173., 25: 125.}, widths={6: 1.49},
                   pdf=('internal', None, False))
        settings = mcatnlo_shower.pythia8_settings(contract, card, run, self.pythia)
        (work / 'Pythia8.cmd').write_text(settings.render())
        (work / 'locked.dat').write_text(settings.render_locked())
        environment = dict(os.environ)
        environment['PYTHIA8DATA'] = self.pythia.xmldoc
        environment.pop('LD_LIBRARY_PATH', None)
        environment.pop('DYLD_LIBRARY_PATH', None)
        result = subprocess.run([str(executable), 'Pythia8.cmd', 'locked.dat'],
                                cwd=work, env=environment, capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (work / 'run.log').write_text(result.stdout + result.stderr)
        summary = mcatnlo_shower.check_shower_log(str(work / 'run.log'))
        self.assertTrue(summary['contract'], result.stdout + result.stderr)
        return work, summary

    def test_fxfx_vetoes_respect_the_requested_lhe_prefix(self):
        for norm in ('average', 'sum'):
            for limit in (1, 3, 10):
                with self.subTest(event_norm=norm, limit=limit):
                    _, summary = self.shower('hj_fxfx.lhe.gz', limit,
                                             fxfx=True, event_norm=norm)
                    self.assertEqual(int(summary['read']), limit)
                    self.assertLess(int(summary['showered']), limit)
                    if int(summary['showered']) == 0:
                        self.assertEqual(float(summary['sigma']), 0.)

    def test_ordinary_shower_preserves_prefix_and_reweighted_input(self):
        _, summary = self.shower('ttbar_nlo.lhe.gz', 2)
        self.assertEqual(int(summary['read']), 2)
        self.assertEqual(int(summary['showered']), 2)

    def test_hepmc3_weight_names_survive_round_trip(self):
        work, summary = self.shower('ttbar_nlo.lhe.gz', 2, hepmc=3)
        source = work / 'read.cc'
        source.write_text(r'''
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/GenEvent.h"
#include <iostream>
int main() {
  HepMC3::ReaderAscii reader("Pythia8.hep");
  for (int i = 0; i < 2; ++i) {
    HepMC3::GenEvent event;
    reader.read_event(event);
    if (reader.failed() || !event.run_info()) return 1;
    const auto& names = event.run_info()->weight_names();
    if (names.size() != event.weights().size() || names.size() != 28) return 2;
    for (size_t j = 0; j < names.size(); ++j) {
      if (names[j].empty() || event.run_info()->weight_index(names[j]) != int(j))
        return 3;
    }
  }
  return 0;
}
''')
        flags = self.pythia.hepmc()[1]
        if HEPMC3:
            prefix = Path(HEPMC3).resolve()
            libdir = prefix / 'lib' if (prefix / 'lib').is_dir() else prefix / 'lib64'
            flags = '-I%s -L%s -Wl,-rpath,%s -lHepMC3' % (
                prefix / 'include', libdir, libdir)
        import shlex
        result = subprocess.run([self.pythia.cxx_compiler(), '-std=c++17',
                                 str(source), '-o', str(work / 'read')] +
                                shlex.split(flags), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run([str(work / 'read')], cwd=work,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(int(summary['showered']), 2)
