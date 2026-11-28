################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""An mg7 gridpack that decays its events with MadSpin ([gridpack]
include_madspin).

``p p > t t~`` with ``t > b l+ vl`` / ``t~ > b~ l- vl~``: the gridpack is made
with the madspin card in place (run_mode minimal: MadSpin is prepared on a
sample the new gridpack generates), then run as a user would -- twice with one
seed (identical), once with another (different), once moved elsewhere with
MadGraph off the PYTHONPATH (the gridpack is self-contained), and the cluster
mode (local_parallel) decays the events of each job. The decayed <init> cross
section is sigma x BR. Self-skips if the mg7 runtime stack / PDF is
unavailable.

Run locally with e.g.::

    ./tests/test_manager.py 'test_gridpack_madspin.*' -pA -t0 -l INFO
"""

import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
from madgraph.various.banner import RunCardMG7
from tests.acceptance_tests.test_mg7_reproducibility import (
    _mg7_datadir_or_skip, _run)

pjoin = os.path.join

_SEED = 2718
_MADSPIN_CARD = """\
set seed 99
decay t > w+ b, w+ > l+ vl
decay t~ > w- b~, w- > l- vl~
launch
"""


def _events(path):
    """The <event> blocks of an LHE file."""
    with gzip.open(path, 'rt') as f:
        return re.findall(r'<event>.*?</event>', f.read(), flags=re.S)


def _init(path):
    """sigma, error of the <init> block of an LHE file."""
    with gzip.open(path, 'rt') as f:
        text = f.read()
    lines = text[text.index('<init>'):text.index('</init>')].splitlines()
    return [float(v) for v in lines[2].split()[:2]]


class MG7GridpackMadSpinTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix='mg7_gp_madspin_')
        cls.process_dir = None
        cls.gridpack = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def setUp(self):
        self.datadir = _mg7_datadir_or_skip(self)
        self.env = dict(os.environ, LHAPDF_DATA_PATH=self.datadir)

    # ------------------------------------------------------------------
    def _process(self):
        cls = MG7GridpackMadSpinTest
        if cls.process_dir is None:
            run_dir = pjoin(self.path, 'proc')
            mg = MGCmd.MasterCmd()
            mg.no_notification()
            # run_cmd, not exec_cmd: only it records the commands in the
            # proc_card, whose generate line MadSpin reads from the banner
            mg.run_cmd('set automatic_html_opening False --no_save')
            mg.run_cmd('import model sm')
            mg.run_cmd('generate p p > t t~')
            mg.run_cmd('output mg7 %s' % run_dir)
            with open(pjoin(run_dir, 'Cards', 'madspin_card.dat'), 'w') as f:
                f.write(_MADSPIN_CARD)
            cls.process_dir = run_dir
        return cls.process_dir

    def _card(self, **settings):
        settings = dict({
            'run.seed': _SEED,
            'run.cpu_thread_pool_size': 4,
            'output.output_format': 'lhe',
            'histograms.postprocessing_histograms': False,
            'histograms.make_plots': False,
            'beam.pdf': 'NNPDF23_lo_as_0130_qed',
            'generation.events': 500,
            'systematics.enable': False,
            'madnis.enable': False,
            'gridpack.save_gridpack': False,
            # bin/generate_events -f runs no tool, so 'auto' would not ship
            # MadSpin: ask for it
            'gridpack.include_madspin': True,
            'cluster.enable': False,
        }, **settings)
        toml = pjoin(self._process(), 'Cards', 'run_card.toml')
        card = RunCardMG7(toml)
        for key, value in settings.items():
            card.set(key, value, user=True)
        card.write(toml)

    def _launch(self, name):
        """bin/generate_events -f in the process directory; its new run."""
        run_dir = self._process()
        before = set(glob.glob(pjoin(run_dir, 'Events', '*')))
        _run([sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
             run_dir, pjoin(self.path, name + '.log'), self.env, name)
        new = sorted(d for d in set(glob.glob(pjoin(run_dir, 'Events', '*'))) - before
                     if not os.path.basename(d).startswith('scan'))
        self.assertTrue(new, 'no new run in %s' % pjoin(run_dir, 'Events'))
        return new[0]

    def _gridpack(self):
        """The gridpack, made once: run_mode minimal, MadSpin on."""
        cls = MG7GridpackMadSpinTest
        if cls.gridpack is None:
            self._card(**{'gridpack.save_gridpack': True,
                          'gridpack.run_mode': 'minimal'})
            run = self._launch('make_gridpack')
            gridpack = pjoin(run, 'gridpack')
            self.assertTrue(os.path.isdir(gridpack), os.listdir(run))
            cls.gridpack = gridpack
        return cls.gridpack

    def _run_gridpack(self, gridpack, name, seed, *options, env=None, cwd=None,
                      expect_failure=False):
        output = pjoin(self.path, name)
        command = [sys.executable, pjoin(gridpack, 'bin', 'generate_events'),
                   '--events', '200', '--seed', str(seed), '--output_dir', output,
                   '--cpu_thread_pool_size', '4'] + list(options)
        log = pjoin(self.path, name + '.log')
        if expect_failure:
            with open(log, 'w') as f:
                code = subprocess.call(command, cwd=cwd or self.path,
                                       env=env or self.env, stdout=f,
                                       stderr=subprocess.STDOUT)
            with open(log) as f:
                return code, f.read()
        _run(command, cwd or self.path, log, env or self.env, name)
        with open(log) as f:
            return output, f.read()

    # ------------------------------------------------------------------
    def test_gridpack_madspin_carries_a_prepared_madspin(self):
        gridpack = self._gridpack()
        ms_dir = pjoin(gridpack, 'madspin', 'ms_dir')
        for name in ('madspin_card.dat', 'max_wgt', 'ms_me_cache.pkl',
                     'ms_version.dat', 'ms_param_card.dat'):
            self.assertTrue(os.path.exists(pjoin(ms_dir, name)), name)
        decay_dirs = sorted(glob.glob(pjoin(ms_dir, 'decay_*')))
        self.assertEqual(len(decay_dirs), 2, decay_dirs)
        for decay_dir in decay_dirs:
            # an mg7 gridpack, without the pools of the preparation
            self.assertTrue(os.path.exists(pjoin(decay_dir, 'data', 'data.json')))
            self.assertTrue(os.path.exists(pjoin(decay_dir, 'ms_partial_width.dat')))
            self.assertEqual(os.listdir(pjoin(decay_dir, 'Events')), [])
        for package in ('madgraph', 'MadSpin', 'madmatrix', 'aloha',
                        pjoin('models', 'sm')):
            self.assertTrue(os.path.isdir(pjoin(gridpack, 'madspin', 'python',
                                                package)), package)
        # the card is shipped without the seed (the run gives it), for the record
        with open(pjoin(gridpack, 'Cards', 'madspin_card.dat')) as f:
            card = f.read()
        self.assertIn('decay t > w+ b, w+ > l+ vl', card)
        self.assertNotIn('set seed', card)
        with open(pjoin(ms_dir, 'madspin_card.dat')) as f:
            self.assertNotIn('set seed', f.read())
        with open(pjoin(gridpack, 'Cards', 'grid_run_card.toml')) as f:
            self.assertRegex(f.read(), r'\[madspin\]\s*\nenable = true')

    def test_gridpack_madspin_seeds(self):
        """Same seed: identical decayed events; another seed: different."""
        gridpack = self._gridpack()
        out_a, log_a = self._run_gridpack(gridpack, 'seed_a', 11)
        out_b, _ = self._run_gridpack(gridpack, 'seed_b', 11)
        out_c, _ = self._run_gridpack(gridpack, 'seed_c', 12)
        decayed = 'events_decayed.lhe.gz'
        self.assertIn('Events written to %s' % pjoin(out_a, decayed), log_a)
        # the undecayed events are not kept by default
        self.assertFalse(os.path.exists(pjoin(out_a, 'events.lhe.gz')))
        events_a = _events(pjoin(out_a, decayed))
        self.assertEqual(len(events_a), 200)
        self.assertEqual(events_a, _events(pjoin(out_b, decayed)))
        self.assertEqual(_init(pjoin(out_a, decayed)), _init(pjoin(out_b, decayed)))
        self.assertNotEqual(events_a, _events(pjoin(out_c, decayed)))
        # the top quarks are decayed: intermediate (status 2), with leptons
        for event in events_a[:20]:
            tops = [line.split() for line in event.splitlines()[2:]
                    if line.split()[:1] in (['6'], ['-6'])]
            self.assertEqual(sorted(t[1] for t in tops), ['2', '2'], event)
            leptons = [line.split()[0] for line in event.splitlines()[2:]
                       if line.split()[:1] and line.split()[0] in
                       ('11', '-11', '13', '-13') and line.split()[1] == '1']
            self.assertEqual(len(leptons), 2, event)

    def test_gridpack_madspin_cross_section_is_sigma_br(self):
        gridpack = self._gridpack()
        out, log = self._run_gridpack(gridpack, 'keep', 21, '--keep_undecayed')
        self.assertIn('Undecayed events kept in', log)
        sigma, _ = _init(pjoin(out, 'events.lhe.gz'))
        decayed, _ = _init(pjoin(out, 'events_decayed.lhe.gz'))
        # BR = product of the partial widths measured when the gridpack was
        # made, over the total widths, times the Breit-Wigner truncation
        widths = []
        for decay_dir in sorted(glob.glob(pjoin(gridpack, 'madspin', 'ms_dir',
                                                'decay_*'))):
            with open(pjoin(decay_dir, 'ms_partial_width.dat')) as f:
                widths.append(float(f.read().split()[0]))
        with open(pjoin(gridpack, 'Cards', 'param_card.dat')) as f:
            total = float(re.search(r'^DECAY\s+6\s+(\S+)', f.read(), re.M).group(1))
        with open(pjoin(out, 'madspin.log')) as f:
            truncation = float(re.search(r'keeps ([0-9.eE+-]+) of the cross',
                                         f.read()).group(1))
        branching_ratio = widths[0] * widths[1] / total ** 2 * truncation
        self.assertGreater(branching_ratio, 0.03)
        self.assertLess(branching_ratio, 0.06)
        self.assertAlmostEqual(decayed / sigma, branching_ratio, delta=1e-4 * branching_ratio)

    def test_gridpack_madspin_runs_without_madgraph(self):
        """Moved elsewhere, run with MadGraph nowhere on the path: the
        gridpack decays with its own Python, and gives the same events as in
        place."""
        gridpack = self._gridpack()
        moved = pjoin(self.path, 'elsewhere', 'my_gridpack')
        shutil.copytree(gridpack, moved, symlinks=True)
        env = dict(self.env)
        env.pop('PYTHONPATH', None)
        # a directory that holds no MadGraph: neither the cwd nor the path
        cwd = tempfile.mkdtemp(dir=self.path)
        out, _ = self._run_gridpack(moved, 'moved', 31, env=env, cwd=cwd)
        with open(pjoin(out, 'madspin.log')) as f:
            log = f.read()
        self.assertIn('MadSpin runs with the MadGraph Python in %s'
                      % os.path.realpath(pjoin(moved, 'madspin', 'python')), log)
        self.assertNotIn('Compiling code', log)
        in_place, _ = self._run_gridpack(gridpack, 'in_place', 31)
        self.assertEqual(_events(pjoin(out, 'events_decayed.lhe.gz')),
                         _events(pjoin(in_place, 'events_decayed.lhe.gz')))
        # nothing was written into the gridpack
        self.assertEqual(os.listdir(pjoin(moved, 'madspin', 'ms_dir', 'decay_6_0',
                                          'Events')), [])

    def test_gridpack_madspin_options(self):
        gridpack = self._gridpack()
        out, log = self._run_gridpack(gridpack, 'no_madspin', 41, '--no_madspin')
        self.assertTrue(os.path.exists(pjoin(out, 'events.lhe.gz')))
        self.assertFalse(os.path.exists(pjoin(out, 'events_decayed.lhe.gz')))
        self.assertNotIn('MadSpin', log)
        code, log = self._run_gridpack(gridpack, 'npy', 42, '--output_format',
                                       'lhe_npy', expect_failure=True)
        self.assertNotEqual(code, 0)
        self.assertIn('MadSpin decays LHE events', log)

    def test_gridpack_madspin_cluster_mode(self):
        """local_parallel cluster mode: each generation job decays its own
        events with the merged gridpack's MadSpin."""
        self._process()
        self._card(**{'generation.events': 400,
                      'cluster.enable': True,
                      'cluster.type': 'local_parallel',
                      'cluster.parallel_jobs': 2,
                      'cluster.events_per_job': 200,
                      'cluster.keep_job_files': True})
        try:
            run = self._launch('cluster')
        finally:
            self._card(**{'cluster.enable': False})
        decayed = pjoin(run, 'events_decayed.lhe.gz')
        self.assertTrue(os.path.exists(decayed), os.listdir(run))
        self.assertEqual(len(_events(decayed)), 400)
        jobs = sorted(glob.glob(pjoin(run, 'cluster', 'deploy', 'job_*')))
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertTrue(os.path.exists(pjoin(job, 'madspin.log')), job)
            self.assertTrue(os.path.exists(pjoin(job, 'events_decayed.lhe.gz')), job)
        with open(pjoin(run, 'info.json')) as f:
            info = json.load(f)
        ratio = info['decayed']['mean'] / info['process']['mean']
        self.assertGreater(ratio, 0.03)
        self.assertLess(ratio, 0.06)
        self.assertAlmostEqual(_init(decayed)[0], info['decayed']['mean'],
                               delta=1e-6 * info['decayed']['mean'])


if __name__ == '__main__':
    unittest.main()
