"""Live driver tests with MG7_TEST_PYTHIA8.

The Pythia8 prefix can also be found through pythia8-config on PATH.
MG7_TEST_HEPMC2 and MG7_TEST_HEPMC3 optionally select the HepMC prefixes.
"""

import gzip
import os
from pathlib import Path
import re
import shlex
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
HEPMC2 = os.environ.get('MG7_TEST_HEPMC2')


@unittest.skipUnless(PYTHIA8 and shutil.which('make'), 'requires Pythia8 and make')
class TestPythia8Driver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='mg7_live_py8_')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.pythia = mcatnlo_shower.Pythia8Installation(PYTHIA8)
        cls.executables = {}

    def hepmc_flags(self, major):
        if major == 0:
            return ''
        prefix = HEPMC3 if major == 3 else HEPMC2
        if prefix:
            prefix = Path(prefix).resolve()
            libdir = prefix / 'lib' if (prefix / 'lib').is_dir() else prefix / 'lib64'
            library = 'HepMC3' if major == 3 else 'HepMC'
            return '-I%s -L%s -Wl,-rpath,%s -l%s' % (
                prefix / 'include', libdir, libdir, library)
        version, flags = self.pythia.hepmc()
        if version != major:
            self.skipTest('requires HepMC%d (MG7_TEST_HEPMC%d)' % (major, major))
        return flags

    def build(self, hepmc=0):
        if hepmc in self.executables:
            return self.executables[hepmc]
        flags = self.hepmc_flags(hepmc)
        builddir = self.root / ('build_%d' % hepmc)
        builddir.mkdir(exist_ok=True)
        result = subprocess.run([
            'make', '-j2', '-f', str(ROOT / 'Template/NLO/MCatNLO/pythia8/Makefile'),
            'PYTHIA8_CONFIG=' + self.pythia.config,
            'CXX=' + self.pythia.cxx_compiler(),
            'HEPMC=%d' % hepmc, 'HEPMC_FLAGS=' + flags],
            cwd=builddir, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.executables[hepmc] = builddir / 'Pythia8.exe'
        return self.executables[hepmc]

    def shower(self, sample, limit, fxfx=False, hepmc=0, event_norm='average',
               command_suffix='', expected_status=0):
        executable = self.build(hepmc)
        directory = tempfile.TemporaryDirectory(prefix='run_', dir=self.root)
        self.addCleanup(directory.cleanup)
        work = Path(directory.name)
        with gzip.open(ROOT / 'tests/input_files' / sample, 'rt') as stream:
            text = stream.read()
        if fxfx:
            # This legacy fixture predates the npNLO attribute of current hooks.
            def add_npnlo(match):
                raw = match.group(0)
                event = L.LHEEvent(raw)
                born_jets = sum(bool(any(event.icolup[i])) for i in event.outgoing())
                born_jets -= event.amcatnlo_info()['iSorH'] == 2
                return raw.replace('<event>', "<event npNLO='%d'>" % born_jets, 1)
            text = re.sub(r'<event>.*?</event>', add_npnlo, text, flags=re.S)
        event_file = work / 'events.lhe'
        event_file.write_text(text)
        header = L.LHEHeader(text[:text.index('<event')])
        card = ShowerCard()
        card['hadronize'] = False
        card['qcut'] = 30.
        card['njmax'] = 1
        contract = mcatnlo_shower.MatchingContract(
            parton_shower='PYTHIA8', ickkw=3 if fxfx else 0,
            ptj=20., born_coloured_partons=-1 if fxfx else 2,
            max_n_matched_jets=1 if fxfx else -1)
        run = dict(nevents=limit, nevents_total=header.nevents,
                   event_norm=event_norm, seed=17,
                   mc_masses={pdg: mass for pdg, mass in header.mc_masses().items()
                              if pdg > 0},
                   masses={6: 173., 25: 125.}, widths={6: 1.49},
                   pdf=('internal', None, False))
        settings = mcatnlo_shower.pythia8_settings(contract, card, run, self.pythia)
        (work / 'Pythia8.cmd').write_text(settings.render() + command_suffix)
        (work / 'locked.dat').write_text(settings.render_locked())
        environment = dict(os.environ)
        environment['PYTHIA8DATA'] = self.pythia.xmldoc
        environment.pop('LD_LIBRARY_PATH', None)
        environment.pop('DYLD_LIBRARY_PATH', None)
        result = subprocess.run([str(executable), 'Pythia8.cmd', 'locked.dat'],
                                cwd=work, env=environment, capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)
        (work / 'run.log').write_text(result.stdout + result.stderr)
        summary = mcatnlo_shower.check_shower_log(str(work / 'run.log'))
        self.assertIs(summary['contract'], expected_status == 0, result.stdout + result.stderr)
        return work, summary

    def test_late_radiation_changes_fail_the_runtime_matching_check(self):
        for key in ('PartonLevel:FSR', 'PartonLevel:ISR'):
            with self.subTest(setting=key):
                _, summary = self.shower(
                    'ttbar_nlo.lhe.gz', 2, command_suffix='%s = off\n' % key,
                    expected_status=3)
                self.assertTrue(any('mismatch: ' + key in line
                                    for line in summary['mismatches']))

    def test_fxfx_vetoes_respect_the_requested_lhe_prefix(self):
        for norm in ('average', 'sum'):
            for limit in (1, 3, 10):
                with self.subTest(event_norm=norm, limit=limit):
                    _, summary = self.shower('hj_fxfx.lhe.gz', limit,
                                             fxfx=True, event_norm=norm)
                    self.assertEqual(int(summary['read']), limit)
                    self.assertLessEqual(int(summary['showered']), limit)
                    if limit > 1:
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
        flags = self.hepmc_flags(3)
        result = subprocess.run([self.pythia.cxx_compiler(), '-std=c++17',
                                 str(source), '-o', str(work / 'read')] +
                                shlex.split(flags), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run([str(work / 'read')], cwd=work,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(int(summary['showered']), 2)

    def test_final_hepmc_cross_section_includes_trailing_fxfx_vetoes(self):
        for major in (2, 3):
            for norm in ('average', 'sum'):
                with self.subTest(hepmc=major, event_norm=norm):
                    work, summary = self.shower('hj_fxfx.lhe.gz', 12,
                                                 fxfx=True, hepmc=major, event_norm=norm)
                    source = work / 'read_cross_section.cc'
                    source.write_text(r'''
#include <iomanip>
#include <iostream>
#if HEPMC == 3
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenCrossSection.h"
#else
#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenCrossSection.h"
#endif
int main() {
  int count = 0;
  double weights = 0., cross = 0.;
#if HEPMC == 3
  HepMC3::ReaderAscii reader("Pythia8.hep");
  for (;;) {
    HepMC3::GenEvent event;
    reader.read_event(event);
    if (reader.failed()) break;
    weights += event.weights()[0];
    cross = event.cross_section()->xsec();
    ++count;
  }
#else
  HepMC::IO_GenEvent reader("Pythia8.hep", std::ios::in);
  while (HepMC::GenEvent* event = reader.read_next_event()) {
    weights += event->weights()[0];
    cross = event->cross_section()->cross_section();
    ++count;
    delete event;
  }
#endif
  std::cout << std::setprecision(17) << count << ' ' << weights << ' ' << cross;
}
''')
                    result = subprocess.run([
                        self.pythia.cxx_compiler(), '-std=c++17', '-DHEPMC=%d' % major,
                        str(source), '-o', str(work / 'read_cross_section')] +
                        shlex.split(self.hepmc_flags(major)),
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    result = subprocess.run([str(work / 'read_cross_section')],
                                            cwd=work, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    count, weights, cross = map(float, result.stdout.split())
                    self.assertGreater(count, 0)
                    self.assertEqual(count, int(summary['showered']))
                    self.assertEqual(int(summary['read']), 12)
                    expected = weights if norm == 'sum' else weights / 12.
                    self.assertAlmostEqual(cross, expected, delta=abs(expected) * 1e-7)
                    self.assertAlmostEqual(float(summary['sigma']), expected,
                                           delta=abs(expected) * 5e-4)
