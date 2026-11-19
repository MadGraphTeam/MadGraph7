"""Exercise the Pythia shower input and analysis reader without Pythia."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from madgraph.various.shower_card import ShowerCard


ROOT = Path(__file__).resolve().parents[3]
MCATNLO = ROOT / 'Template/NLO/MCatNLO'


@unittest.skipUnless(shutil.which('g++'), 'requires a C++ compiler')
class TestPythiaWeightReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix='mg5_py8_reader_')
        cls.addClassCleanup(cls.work.cleanup)
        work = Path(cls.work.name)
        source = work / 'reader.cc'
        source.write_text(r'''
#include "LHEFRead.h"
#ifdef TEST_HWU
extern "C" {
  void pyabeg_(int&, char(*)[80]);
  void pyanal_(int&, double*);
  void pyaend_(double&);
  void setup_event_(int&);
}
#else
extern "C" int get_wgts_info_len(void) { return 80; }
#endif
int main(int argc, char** argv) {
  MyReader reader(argv[1]);
  char labels[1024][wgts_info_len_used];
  int count = 0;
  reader.lhef_read_wgtsinfo_(count, labels);
  std::cout << "count " << count << '\n';
  if (argc > 2) {
    std::cout << "central " << std::string(labels[0], 80) << '\n';
    std::cout << "label " << std::string(labels[1], 80) << '\n';
  }
#ifdef TEST_HWU
  int first = 1;
  setup_event_(first);
  pyabeg_(count, labels);
#endif
  double weights[1024] = {0.};
  for (int event = 0; event < 2; ++event) {
    reader.lhef_read_wgts_(weights);
    std::cout << "event " << event << ' ' << weights[1] << '\n';
#ifdef TEST_HWU
    int index = event + 1;
    setup_event_(index);
    weights[0] = 1.;
    pyanal_(count, weights);
#endif
  }
#ifdef TEST_HWU
  double norm = 1.;
  pyaend_(norm);
#endif
}
''')
        cls.executable = work / 'reader'
        subprocess.run([shutil.which('g++'), '-O0', '-I',
                        str(MCATNLO / 'include'), str(source), '-o',
                        str(cls.executable)], check=True, capture_output=True)

    def read_events(self, reweight, jwgtinfo=0, header=True,
                    label='scale variation', show_label=False, executable=None):
        weights = ("<initrwgt>\n<weightgroup name='test'>\n"
                   f"<weight id='1001'> {label} </weight>\n"
                   "</weightgroup>\n</initrwgt>\n") if reweight else ''
        contents = '<LesHouchesEvents version="3.0">\n'
        if header:
            contents += '<header>\n' + weights + '</header>\n'
        contents += ('<init>\n2212 2212 6500 6500 0 0 0 0 -4 1\n'
                     '1 0 1 1\n</init>\n')
        for value in (2.5, -3.25):
            contents += ('<event>\n1 1 1 10 0.01 0.1\n'
                         '25 1 0 0 0 0 0 0 0 125 125 0 9\n'
                         '#aMCatNLO 1 0 0 0 0 0 0 %d 0 0 0 0 0 0 0\n'
                         % jwgtinfo)
            if reweight:
                contents += "<rwgt>\n<wgt id='1001'> %s </wgt>\n</rwgt>\n" % value
            contents += ("<scales scalup_1_2='20' scalup_2_1='30'>\n"
                         '</scales>\n</event>\n')
        contents += '</LesHouchesEvents>\n'
        path = Path(self.work.name) / 'events.lhe'
        path.write_text(contents)
        result = subprocess.run([str(executable or self.executable), str(path)] +
                                (['labels'] if show_label else []),
                                cwd=self.work.name, text=True,
                                capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_delta_events_without_reweighting(self):
        self.assertEqual(self.read_events(False),
                         'count 1\nevent 0 0\nevent 1 0\n')

    def test_events_without_header(self):
        self.assertEqual(self.read_events(False, header=False),
                         'count 1\nevent 0 0\nevent 1 0\n')

    def test_reweighting_with_supported_weight_formats(self):
        for jwgtinfo in (0, 9, -9):
            with self.subTest(jwgtinfo=jwgtinfo):
                self.assertEqual(self.read_events(True, jwgtinfo),
                                 'count 2\nevent 0 2.5\nevent 1 -3.25\n')

    def test_inclusive_scale_labels_are_recognised_and_order_tags_retained(self):
        for tag in (0, 400, 402):
            label = f'tag= {tag} dyn= -1 muR=0.20000E+01 muF=0.10000E+01'
            result = self.read_events(True, -9, label=label, show_label=True)
            output = next(line for line in result.splitlines() if line.startswith('label '))
            self.assertIn('dyn= -1 muR=0.20000E+01 muF=0.10000E+01', output)
            self.assertEqual('tag=' in output, tag != 0)

    def test_fortran_labels_are_space_padded_without_nuls(self):
        for label in ('scale variation', 'x' * 79, 'x' * 100):
            with self.subTest(label=label):
                result = self.read_events(True, label=label, show_label=True)
                central = next(line[8:] for line in result.splitlines()
                               if line.startswith('central '))
                variation = next(line[6:] for line in result.splitlines()
                                 if line.startswith('label '))
                self.assertEqual(central, 'central value'.ljust(80))
                self.assertEqual(variation, (' ' + label + ' ')[:80].ljust(80))

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_rates_analysis_output_can_be_read_as_hwu(self):
        from madgraph.various.histograms import HwUList

        work = Path(self.work.name)
        (work / 'orders.inc').write_text(
            '      integer amp_split_size\n      parameter(amp_split_size=1)\n')
        (work / 'pineappl_maxproc.inc').write_text(
            '      integer max_nproc,mxpdflumi\n'
            '      parameter(max_nproc=1,mxpdflumi=1)\n')
        (work / 'event.f').write_text('''      subroutine setup_event(iev)
      include 'HEPMC.INC'
      integer iev
      logical pineappl
      common /for_pineappl/ pineappl
      pineappl=.false.
      nevhep=iev
      nhep=2
      phep(3,1)=1d0
      phep(3,2)=-1d0
      evwgt=1d0
      end
''')
        result = subprocess.run([
            'gfortran', '-O0', '-fcheck=all', '-ffixed-line-length-none',
            '-I', str(work), '-I', str(MCATNLO / 'include'), '-c',
            str(ROOT / 'Template/NLO/FixedOrderAnalysis/HwU.f'),
            str(MCATNLO / 'PY8Analyzer/py8an_HwU_rates.f'),
            str(ROOT / 'Template/NLO/SubProcesses/pineappl_interface_dummy.f'),
            'event.f'], cwd=work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        executable = work / 'hwu_reader'
        runtime = subprocess.check_output([
            'gfortran', '-print-file-name=libgfortran' +
            ('.dylib' if sys.platform == 'darwin' else '.so')], text=True).strip()
        runtime_directory = str(Path(runtime).parent)
        result = subprocess.run([
            'g++', '-O0', '-DTEST_HWU', '-I', str(MCATNLO / 'include'),
            'reader.cc', 'HwU.o', 'py8an_HwU_rates.o',
            'pineappl_interface_dummy.o', 'event.o', '-L' + runtime_directory,
            '-Wl,-rpath,' + runtime_directory, '-lgfortran', '-o',
            str(executable)], cwd=work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.read_events(True, label='dyn=-1 muR=2.0 muF=1.0', executable=executable)
        output = work / 'MADatNLO.HwU'
        self.assertNotIn(b'\0', output.read_bytes())
        histograms = HwUList(str(output))
        self.assertEqual(len(histograms), 1)
        self.assertIn('central', histograms[0].bins.weight_labels)
        self.assertAlmostEqual(histograms[0].bins[0].wgts['central'], 1.)


@unittest.skipUnless(shutil.which('bash'), 'requires bash')
class TestPythiaDeltaSteering(unittest.TestCase):
    """Starting scales of the Pythia8 command file written by MG5_aMC."""

    def steering(self, delta, dipole_setting):
        from madgraph.various import mcatnlo_shower

        class Installation(object):
            version = '8.312' if dipole_setting else '8.244'
            path = '/fake'
            def has_setting(self, name):
                return dipole_setting or \
                    name != 'Beams:setDipoleShowerStartingScalesFromLHEF'
            def has_lhapdf6(self):
                return True

        contract = mcatnlo_shower.MatchingContract(parton_shower='PYTHIA8',
                                                   mcatnlo_delta=delta)
        run = {'nevents': 10, 'nevents_total': 10, 'event_norm': 'average',
               'mc_masses': {5: 4.8}, 'pdf': ('internal', None, False), 'seed': 1}
        return mcatnlo_shower.pythia8_settings(
            contract, ShowerCard(), run, Installation()).render()

    def test_delta_keeps_particle_and_dipole_scales(self):
        result = self.steering(True, True)
        self.assertIn('Beams:setProductionScalesFromLHEF = on', result)
        self.assertIn('Beams:setDipoleShowerStartingScalesFromLHEF = on', result)

    def test_legacy_pythia_is_refused_for_delta(self):
        # Without per-dipole starting scales MC@NLO-Delta cannot be showered
        # consistently: this is an error, no longer a silent downgrade.
        from madgraph.various import mcatnlo_shower
        with self.assertRaises(mcatnlo_shower.ShowerInterfaceError):
            self.steering(True, False)

    def test_ordinary_mcatnlo_keeps_scalar_scale(self):
        result = self.steering(False, True)
        self.assertIn('Beams:setProductionScalesFromLHEF = off', result)
        self.assertNotIn('Beams:setDipoleShowerStartingScalesFromLHEF', result)
        result = self.steering(False, False)
        self.assertIn('Beams:setProductionScalesFromLHEF = off', result)
