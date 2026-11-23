"""The precision check of an mg7 run (madgraph/mg7/precision_check.py).

The default precision, denom64 (FPTYPE=v), evaluates the amplitudes in single
precision. [run] precision_check re-evaluates a fraction of the events with an
all64 (FPTYPE=d) build of the same matrix elements:

  * ``test_precision_check_mg7`` -- ``p p > e+ e- j`` with LHE output: the run
    builds both FPTYPEs side by side (build.<backend>_v and build.<backend>_d),
    writes precision_check.json for the requested fraction of the events, all
    matched to a library, finite, and within single-precision rounding of all64
    without being identical to it; then a second run in color32 rebuilds in its
    own directory without a 'make clean', and a third, in all64, checks nothing.
  * ``test_precision_check_reproduces_the_run_mg7`` -- the same process with
    lhe_npy output and the |M|^2 of every event stored by madspace: the check's
    own evaluation in the run's precision reproduces it, mirrored flavours
    included, so the check compares the very matrix elements the run used.

Self-skips if the mg7 runtime stack / PDF is unavailable (as
test_mg7_reproducibility.py). Run locally with e.g.::

    ./tests/test_manager.py test_precision_check.*_mg7 -pA -t0 -l INFO
"""

from __future__ import absolute_import
from __future__ import division

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import madgraph.interface.master_interface as MGCmd
from madgraph.various.banner import RunCardMG7
from tests.acceptance_tests.test_mg7_reproducibility import (
    _mg7_datadir_or_skip, _run)

pjoin = os.path.join

_PROCESS = 'p p > e+ e- j'
_EVENTS = 2000


class MG7PrecisionCheckTest(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='mg7_precision_check_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _output_process(self, name, **settings):
        run_dir = pjoin(self.path, name)
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        mg.exec_cmd('generate %s' % _PROCESS)
        mg.exec_cmd('output mg7 %s' % run_dir)
        defaults = {
            'generation.events': _EVENTS,
            'postprocessing.systematics': False,
            'run.seed': 4242,
        }
        defaults.update(settings)
        self._set_run_card(run_dir, **defaults)
        return run_dir

    @staticmethod
    def _set_run_card(run_dir, **settings):
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        card = RunCardMG7(toml)
        for key, value in settings.items():
            card.set(key, value, user=True)
        card.write(toml)

    def _generate(self, run_dir, datadir, tag):
        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        _run([sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
             run_dir, pjoin(run_dir, 'gen_%s.log' % tag), env,
             'mg7 generate_events')
        runs = sorted(glob.glob(pjoin(run_dir, 'Events', 'run_*')))
        self.assertTrue(runs, 'no run directory under %s' % run_dir)
        return runs[-1]

    def test_precision_check_mg7(self):
        datadir = _mg7_datadir_or_skip(self)
        run_dir = self._output_process(
            'lhe', **{'run.output_format': 'lhe', 'run.precision_check': 0.05})
        card = RunCardMG7(pjoin(run_dir, 'Cards', 'run_card.toml'))
        self.assertEqual(card['run']['precision'], 'denom64')

        run_path = self._generate(run_dir, datadir, 'denom64')
        with open(pjoin(run_path, 'precision_check.json')) as f:
            summary = json.load(f)
        self.assertEqual(summary['fptype'], 'v')
        self.assertEqual(summary['reference_fptype'], 'd')
        self.assertEqual(summary['events'], 100)    # 5% of 2000
        self.assertEqual(summary['unmatched_events'], 0)
        self.assertEqual(summary['non_finite'], 0)
        self.assertEqual(summary['above_threshold'], 0)
        # single precision amplitudes: not all64, but close to it
        self.assertGreater(summary['max_relative_deviation'], 0.)
        self.assertLess(summary['max_relative_deviation'], 1e-4)
        self.assertLess(summary['mean_relative_deviation'], 1e-5)

        # the two FPTYPEs were built side by side
        subprocess_dir = sorted(glob.glob(pjoin(run_dir, 'SubProcesses', 'P*')))[0]
        builds = sorted(os.path.basename(path) for path in
                        glob.glob(pjoin(subprocess_dir, 'build.*')))
        self.assertEqual(len(builds), 2, builds)
        self.assertTrue(builds[0].endswith('_d') and builds[1].endswith('_v'), builds)

        # another precision: built in its own directory, no 'make clean' needed
        self._set_run_card(run_dir, **{'run.precision': 'color32',
                                       'run.precision_check': 0.0})
        run_path = self._generate(run_dir, datadir, 'color32')
        self.assertFalse(os.path.exists(pjoin(run_path, 'precision_check.json')))
        self.assertTrue(glob.glob(pjoin(subprocess_dir, 'build.*_m')))

        # all64 is its own reference: nothing to check
        self._set_run_card(run_dir, **{'run.precision': 'all64',
                                       'run.precision_check': 0.05})
        run_path = self._generate(run_dir, datadir, 'all64')
        self.assertFalse(os.path.exists(pjoin(run_path, 'precision_check.json')))

    def test_precision_check_reproduces_the_run_mg7(self):
        datadir = _mg7_datadir_or_skip(self)
        run_dir = self._output_process(
            'stored', **{'run.output_format': 'lhe_npy', 'run.precision_check': 0.01})
        script = pjoin(run_dir, 'check_mapping.py')
        with open(script, 'w') as f:
            f.write(textwrap.dedent('''
                import json, os, sys
                sys.path.insert(0, %r)
                import numpy as np
                from madgraph.mg7.madgraph_process import MadgraphProcess
                from madgraph.mg7 import precision_check as pc
                from madgraph.mg7.common import me_parameters

                process = MadgraphProcess()
                process.store_matrix_elements = True
                process.survey()
                process.train_madnis()
                process.generate_events()
                events = np.load(process.events_path, mmap_mode="r",
                                 max_header_size=1 << 30)
                legs = 0
                while "part%%d_energy" %% (legs + 1) in events.dtype.names:
                    legs += 1
                table = pc.flavor_table(process)
                groups = {}
                for event in events[:1000]:
                    kept = [i for i in range(1, legs + 1)
                            if event["part%%d_status_code" %% i] in (-1, 1)]
                    pdgs = tuple(int(event["part%%d_pdg_id" %% i]) for i in kept)
                    path_format, flavor, swap = table[pdgs]
                    momenta = [[event["part%%d_%%s" %% (i, c)]
                                for c in ("energy", "px", "py", "pz")] for i in kept]
                    if swap:
                        momenta = [momenta[1], momenta[0]] + momenta[2:]
                    group = groups.setdefault(path_format, [[], [], [], [], 0])
                    group[0].append(momenta)
                    group[1].append(float(event["alpha_qcd"]))
                    group[2].append(flavor)
                    group[3].append(float(event["matrix_element"]))
                    group[4] += int(swap)
                names = {p: "g%%d" %% i for i, p in enumerate(groups)}
                inputs = {}
                for p, g in groups.items():
                    inputs[names[p] + "_momenta"] = np.array(g[0])
                    inputs[names[p] + "_alpha_s"] = np.array(g[1])
                    inputs[names[p] + "_flavors"] = np.array(g[2], dtype=np.uint32)
                backend, fptype = process.precision_check_builds[0]
                libraries = {names[p]: {
                    "library": p.format(device=backend, fptype=fptype),
                    "helicity_choice": process.helicity_choices.get(p)}
                    for p in groups}
                values = pc.evaluate(process.run_path, "test", libraries, inputs,
                                     process.param_card_path,
                                     me_parameters(process.run_card))
                worst = max(float(np.max(np.abs(values[names[p]] / np.array(g[3]) - 1)))
                            for p, g in groups.items())
                with open("mapping.json", "w") as f:
                    json.dump({"worst": worst, "fptype": fptype,
                               "events": sum(len(g[3]) for g in groups.values()),
                               "mirrored": sum(g[4] for g in groups.values()),
                               "summary": os.path.isfile(os.path.join(
                                   process.run_path, pc.RESULT_FILE))}, f)
                ''' % os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))))))
        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        _run([sys.executable, script], run_dir, pjoin(run_dir, 'mapping.log'), env,
             'stored matrix element comparison')
        with open(pjoin(run_dir, 'mapping.json')) as f:
            result = json.load(f)
        self.assertEqual(result['fptype'], 'v')
        self.assertEqual(result['events'], 1000)
        self.assertGreater(result['mirrored'], 0)
        self.assertTrue(result['summary'])
        # the same library at the same inputs: the same |M|^2 as in the run
        self.assertLess(result['worst'], 1e-12)


if __name__ == '__main__':
    unittest.main()
