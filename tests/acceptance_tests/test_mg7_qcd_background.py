"""Exercise the mixed two-, three- and four-jet CPU run reported in #199.

Run with ``./tests/test_manager.py test_qcd_background_mg7 -pA -t0``.
``MG7_QCD_CPU_MODE`` can select a matrix-element backend (default: auto),
including avx512y on machines that support it. The survey keeps the default
cuts, scales, PDF, precision and batch sizes; only the final event count and
thread count are bounded, and optional training/postprocessing are disabled.
"""

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
from madgraph.various.banner import RunCardMG7
from tests.acceptance_tests.test_check_xsec_processes_mg7 import _tail


class MG7QCDBackgroundTest(unittest.TestCase):

    def test_qcd_background_mg7(self):
        """Every subprocess must survive survey and produce finite weights."""
        import madspace
        if not hasattr(madspace, 'ChannelEventGenerator'):
            self.skipTest('mg7 runtime stack (madspace) unavailable')
        pdf = RunCardMG7()['beam']['pdf1']
        if not misc.resolve_lhapdf().find_set(pdf):
            self.skipTest('%s LHAPDF data not found' % pdf)

        path = tempfile.mkdtemp(prefix='mg7_qcd_background_')
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        run_dir = os.path.join(path, 'process')
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        mg.exec_cmd('generate p p > j j')
        mg.exec_cmd('add process p p > j j j')
        mg.exec_cmd('add process p p > j j j j')
        mg.exec_cmd('output mg7 %s -f' % run_dir)

        card_path = os.path.join(run_dir, 'Cards', 'run_card.toml')
        card = RunCardMG7(card_path)
        for key, value in {
                'run.seed': 199,
                'run.cpu_mode': os.environ.get('MG7_QCD_CPU_MODE', 'auto'),
                'run.cpu_thread_pool_size': 2,
                'generation.events': 100,
                'madnis.enable': False,
                'systematics.enable': False,
                'postprocessing.systematics': False,
                'run.postprocessing_histograms': False,
                'run.make_plots': False}.items():
            card.set(key, value, user=True)
        card.write(card_path)

        log_path = os.path.join(path, 'generate.log')
        with open(log_path, 'w') as log:
            result = subprocess.run(
                [sys.executable, os.path.join(run_dir, 'bin', 'generate_events'),
                 '-f'], cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                timeout=600)
        self.assertEqual(result.returncode, 0, _tail(log_path))
        infos = glob.glob(os.path.join(run_dir, 'Events', '*', 'info.json'))
        self.assertEqual(len(infos), 1, _tail(log_path))
        with open(infos[0]) as stream:
            info = json.load(stream)
        # The launcher can report an error with exit status zero. Require a
        # completed run, including sampling of every generated subprocess.
        self.assertEqual(info['status'], 'done', _tail(log_path))
        with open(os.path.join(run_dir, 'SubProcesses', 'subprocesses.json')) as stream:
            subprocesses = json.load(stream)
        self.assertEqual(len(subprocesses), 32)
        self.assertEqual({channel['subprocess'] for channel in info['channels']},
                         set(range(len(subprocesses))))
        for channel in [info['process']] + info['channels']:
            self.assertGreater(channel['count'], 0, channel['name'])
            self.assertTrue(math.isfinite(channel['mean']), channel['name'])
            self.assertTrue(math.isfinite(channel['error']), channel['name'])
        self.assertGreater(info['process']['mean'], 0.)
        self.assertTrue(info['process']['done'])
