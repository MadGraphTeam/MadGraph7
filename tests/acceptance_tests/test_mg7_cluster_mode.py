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
"""The cluster mode of mg7 ([cluster] in the run card), on this machine.

The subprocesses are trained in parallel jobs, their gridpacks merged, and the
events generated in parallel jobs from the merged gridpack. With the batch
system "multicore" this runs without a cluster, so it checks everything but the
submission to a scheduler: the cross section has to agree with the one of an
ordinary run, and every option has to leave the directory in the state a user
expects. Self-skips if the mg7 runtime stack / PDF is unavailable.

Process: ``p p > t t~ j``, which has four subprocesses.

Run locally with e.g.::

    ./tests/test_manager.py test_.*cluster_mode -pA -t0 -l INFO
"""

from __future__ import absolute_import
from __future__ import division

import glob
import gzip
import json
import math
import os
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

_PROCESS = 'p p > t t~ j'
_EVENTS = 4000
_SEED = 4242
_SIGMAS = 5.


class MG7ClusterModeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = tempfile.mkdtemp(prefix='mg7_cluster_')
        cls.run_dir = None
        cls.reference = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def setUp(self):
        self.datadir = _mg7_datadir_or_skip(self)
        self.env = dict(os.environ, LHAPDF_DATA_PATH=self.datadir)

    def _process(self):
        """The output directory, made once for all tests."""
        if MG7ClusterModeTest.run_dir is None:
            run_dir = pjoin(self.path, 'proc')
            mg = MGCmd.MasterCmd()
            mg.no_notification()
            mg.exec_cmd('set automatic_html_opening False --no_save')
            mg.exec_cmd('generate %s' % _PROCESS)
            mg.exec_cmd('output mg7 %s' % run_dir)
            MG7ClusterModeTest.run_dir = run_dir
        return MG7ClusterModeTest.run_dir

    def _card(self, **settings):
        settings = dict({
            'run.seed': _SEED,
            'run.output_format': 'lhe',
            'run.postprocessing_histograms': False,
            'run.make_plots': False,
            'beam.pdf': 'NNPDF23_lo_as_0130_qed',
            'generation.events': _EVENTS,
            'systematics.enable': False,
            'postprocessing.systematics': False,
            'madnis.enable': False,
            'cluster.enable': False,
            'cluster.type': 'multicore',
            'cluster.nb_core': 2,
            'cluster.subprocesses_per_job': 2,
            'cluster.events_per_job': _EVENTS // 2,
        }, **settings)
        toml = pjoin(self._process(), 'Cards', 'run_card.toml')
        card = RunCardMG7(toml)
        for key, value in settings.items():
            card.set(key, value, user=True)
        card.write(toml)

    def _run(self, name):
        """Run the process directory, and return its new Events directory."""
        run_dir = self._process()
        before = set(glob.glob(pjoin(run_dir, 'Events', '*')))
        _run([sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
             run_dir, pjoin(self.path, name + '.log'), self.env, name)
        new = sorted(set(glob.glob(pjoin(run_dir, 'Events', '*'))) - before)
        self.assertEqual(len(new), 1, new)
        return new[0]

    def _cross_section(self, events_dir):
        with open(pjoin(events_dir, 'info.json')) as f:
            process = json.load(f)['process']
        return process['mean'], process['error'], process['count_unweighted']

    def _reference(self):
        """The cross section of a run without the cluster mode."""
        if MG7ClusterModeTest.reference is None:
            self._card()
            events_dir = self._run('reference')
            MG7ClusterModeTest.reference = self._cross_section(events_dir)[:2]
        return MG7ClusterModeTest.reference

    def _assert_compatible(self, mean, error):
        ref_mean, ref_error = self._reference()
        sigma = math.sqrt(ref_error ** 2 + error ** 2)
        self.assertLessEqual(
            abs(mean - ref_mean), _SIGMAS * sigma,
            'cross section %g +- %g is not compatible with the one of an '
            'ordinary run %g +- %g' % (mean, error, ref_mean, ref_error))

    def _count_lhe_events(self, events_dir):
        with gzip.open(pjoin(events_dir, 'events.lhe.gz'), 'rt') as f:
            return sum(1 for line in f if line.startswith('<event>'))

    def test_cluster_run(self):
        """Training in parallel, a merged gridpack, generation in parallel."""
        self._reference()
        self._card(**{'cluster.enable': True, 'cluster.fix_max_weight': True})
        events_dir = self._run('cluster')
        mean, error, count = self._cross_section(events_dir)
        self.assertEqual(count, _EVENTS)
        self.assertEqual(self._count_lhe_events(events_dir), _EVENTS)
        self._assert_compatible(mean, error)
        # two training jobs for the four subprocesses, two generation jobs
        self.assertEqual(
            len(glob.glob(pjoin(events_dir, 'cluster', 'train', 'job_*', 'done'))), 2)
        self.assertEqual(
            len(glob.glob(pjoin(events_dir, 'cluster', 'deploy', 'job_*', 'done'))), 2)
        # the events of the jobs are only kept on request
        self.assertFalse(glob.glob(pjoin(events_dir, 'cluster', 'deploy', 'job_*',
                                         'events.lhe.gz')))

    def test_cluster_run_estimating_the_max_weights(self):
        """Every generation job finds the max weights itself; merge groups and
        npy events."""
        self._reference()
        self._card(**{'cluster.enable': True, 'cluster.fix_max_weight': False,
                      'phasespace.merge_subprocesses': True,
                      'cluster.subprocesses_per_job': 1,
                      'run.output_format': 'lhe_npy'})
        try:
            events_dir = self._run('cluster_minimal')
        finally:
            self._card()
        import numpy
        events = numpy.load(pjoin(events_dir, 'events.npy'), mmap_mode='r')
        self.assertEqual(len(events), _EVENTS)
        mean, error, _ = self._cross_section(events_dir)
        self._assert_compatible(mean, error)

    def test_gridpack_only(self):
        """The merged gridpack is the result, and it generates events."""
        self._reference()
        self._card(**{'cluster.enable': True, 'cluster.mode': 'gridpack_only'})
        events_dir = self._run('gridpack_only')
        self.assertFalse(os.path.exists(pjoin(events_dir, 'events.lhe.gz')))
        gridpack = pjoin(events_dir, 'gridpack')
        self.assertTrue(os.path.isdir(gridpack))
        out = pjoin(self.path, 'gridpack_events')
        _run([sys.executable, pjoin(gridpack, 'bin', 'generate_events'),
              '--events', str(_EVENTS), '--output_dir', out, '--verbosity', 'log'],
             self.path, pjoin(self.path, 'gridpack_run.log'), self.env, 'gridpack')
        mean, error, count = self._cross_section(out)
        self.assertEqual(count, _EVENTS)
        self._assert_compatible(mean, error)
