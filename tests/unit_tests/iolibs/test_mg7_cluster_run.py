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
"""The steering of the cluster mode: how the work is split and which batch
system is used. Running it is covered by the acceptance test."""
from __future__ import absolute_import
import json
import os
import tempfile
import types
import unittest

from madgraph.mg7 import bootstrap as mg7_bootstrap
from madgraph.various.banner import RunCardMG7

# Importing launch without a compiled madspace starts the madspace build, and
# test_manager imports every unit test module while collecting: check first.
cluster_run = None
if mg7_bootstrap.madspace_is_installed():
    try:
        import madgraph.iolibs.template_files.mg7.cluster_run as cluster_run
    except Exception as error:
        cluster_run = None
        import_error = error


def card(**settings):
    run_card = RunCardMG7()
    for key, value in settings.items():
        run_card.set(key.replace('__', '.'), value, user=True)
    return run_card


@unittest.skipIf(cluster_run is None, 'madspace not available')
class TestSplitting(unittest.TestCase):

    def test_events_are_split_evenly(self):
        self.assertEqual(cluster_run.split_events(100000, 100000), [100000])
        self.assertEqual(cluster_run.split_events(250000, 100000),
                         [83334, 83333, 83333])
        self.assertEqual(cluster_run.split_events(5, 100000), [5])
        for total, per_job in ((1, 1), (10, 3), (1000003, 1000)):
            jobs = cluster_run.split_events(total, per_job)
            self.assertEqual(sum(jobs), total)
            self.assertLessEqual(max(jobs), per_job)
            self.assertLessEqual(max(jobs) - min(jobs), 1)

    def test_subprocesses_are_split_in_order(self):
        self.assertEqual(cluster_run.split_subprocesses(5, 2),
                         [[0, 1], [2, 3], [4]])
        self.assertEqual(cluster_run.split_subprocesses(3, 1), [[0], [1], [2]])
        self.assertEqual(cluster_run.split_subprocesses(3, 10), [[0, 1, 2]])

    def test_seeds_are_reproducible_and_distinct(self):
        seeds = {cluster_run.derived_seed(42, pass_, i)
                 for pass_ in ('train', 'deploy') for i in range(50)}
        self.assertEqual(len(seeds), 100)
        self.assertEqual(cluster_run.derived_seed(42, 'train', 3),
                         cluster_run.derived_seed(42, 'train', 3))
        self.assertNotEqual(cluster_run.derived_seed(42, 'train', 3),
                            cluster_run.derived_seed(43, 'train', 3))
        self.assertLess(max(seeds), 2 ** 63)


@unittest.skipIf(cluster_run is None, 'madspace not available')
class TestMakeCluster(unittest.TestCase):

    def test_options_of_the_pass(self):
        cluster, name = cluster_run.make_cluster(
            card(cluster__type='slurm'), {}, 'gpu', '04:00:00', '--gres=gpu:1')
        self.assertEqual(name, 'slurm')
        self.assertEqual(cluster.cluster_queue, 'gpu')
        self.assertEqual(cluster.options['cluster_walltime'], '04:00:00')
        self.assertEqual(cluster.extra_submit_args(), ['--gres=gpu:1'])
        self.assertEqual(cluster.nb_retry, 1)

    def test_auto_takes_the_configuration(self):
        options = {'cluster_type': 'pbs', 'cluster_queue': 'short',
                   'cluster_walltime': '1:00'}
        cluster, name = cluster_run.make_cluster(card(), options, '', '', '')
        self.assertEqual(name, 'pbs')
        self.assertEqual(cluster.cluster_queue, 'short')
        self.assertEqual(cluster.options['cluster_walltime'], '1:00')
        # what the run card names wins
        cluster, _ = cluster_run.make_cluster(card(), options, 'long', '', '')
        self.assertEqual(cluster.cluster_queue, 'long')

    def test_auto_without_a_configured_batch_system(self):
        for options in ({}, {'cluster_type': None}, {'cluster_type': 'None'}):
            with self.assertRaisesRegex(cluster_run.ClusterRunError, 'batch system'):
                cluster_run.make_cluster(card(), options, '', '', '')

    def test_local_parallel(self):
        cluster, name = cluster_run.make_cluster(
            card(cluster__type='local_parallel', cluster__parallel_jobs=3), {}, '', '', '')
        self.assertEqual(name, 'local_parallel')
        self.assertEqual(cluster.nb_core, 3)
        cluster, _ = cluster_run.make_cluster(
            card(cluster__type='local_parallel'), {}, '', '', '')
        self.assertEqual(cluster.nb_core, os.cpu_count())


@unittest.skipIf(cluster_run is None, 'madspace not available')
class TestUnfinishedRun(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        os.makedirs('Cards')
        for name in ('run_card.toml', 'param_card.dat'):
            with open(os.path.join('Cards', name), 'w') as f:
                f.write('# %s\n' % name)
        run_card = card()
        self.run = cluster_run.ClusterRun(
            types.SimpleNamespace(run_card=run_card), {})

    def make_run(self, number, stage, digest=None):
        path = os.path.join('Events', 'run_%02d' % number, 'cluster')
        os.makedirs(path)
        with open(os.path.join(path, 'state.json'), 'w') as f:
            json.dump({'hash': digest or cluster_run.card_hash(), 'stage': stage}, f)
        return os.path.dirname(path)

    def test_nothing_to_continue(self):
        self.assertIsNone(self.run.find_unfinished_run(cluster_run.card_hash()))

    def test_unfinished_run_with_the_same_cards(self):
        self.make_run(1, 'done')
        unfinished = self.make_run(2, 'deploy')
        self.assertEqual(self.run.find_unfinished_run(cluster_run.card_hash()),
                         unfinished)

    def test_only_the_newest_run_is_continued(self):
        self.make_run(1, 'train')
        self.make_run(2, 'done')
        self.assertIsNone(self.run.find_unfinished_run(cluster_run.card_hash()))

    def test_runs_sorted_by_number(self):
        self.make_run(99, 'done')
        unfinished = self.make_run(100, 'deploy')
        self.assertEqual(self.run.find_unfinished_run(cluster_run.card_hash()),
                         unfinished)

    def test_changed_cards(self):
        self.make_run(1, 'train', digest='other')
        self.assertIsNone(self.run.find_unfinished_run(cluster_run.card_hash()))
        with open(os.path.join('Cards', 'run_card.toml'), 'a') as f:
            f.write('# edited\n')
        self.make_run(2, 'train')
        self.assertIsNotNone(self.run.find_unfinished_run(cluster_run.card_hash()))
        with open(os.path.join('Cards', 'run_card.toml'), 'a') as f:
            f.write('# edited again\n')
        self.assertIsNone(self.run.find_unfinished_run(cluster_run.card_hash()))


@unittest.skipIf(cluster_run is None, 'madspace not available')
class TestCleanTrainingData(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make_run(self, keep):
        run = cluster_run.ClusterRun(types.SimpleNamespace(
            run_card=card(cluster__keep_job_files=keep)), {})
        run.cluster_path = self.tmp.name
        run.state = {'train_jobs': [{'index': 0}, {'index': 1}]}
        for index in (0, 1):
            job_dir = run.job_dir('train', index)
            os.makedirs(os.path.join(job_dir, 'gridpack', 'data'))
            for name in ('log', 'info.json', 'done', 'gridpack/data/globals.tar',
                         'gridpack/data/channels.json', 'gridpack/data/fixed_alphas.info'):
                with open(os.path.join(job_dir, name), 'w') as f:
                    f.write('x')
        return run

    def files(self, run):
        return sorted(os.path.relpath(os.path.join(root, name), run.job_dir('train', 0))
                      for root, _, names in os.walk(run.job_dir('train', 0))
                      for name in names)

    def test_gridpacks_dropped_logs_and_info_kept(self):
        run = self.make_run(False)
        run.clean_training_data()
        self.assertEqual(self.files(run), ['done', 'info.json', 'log'])

    def test_keep_job_files(self):
        run = self.make_run(True)
        run.clean_training_data()
        self.assertIn('gridpack/data/globals.tar', self.files(run))


@unittest.skipIf(cluster_run is None, 'madspace not available')
class TestCleanDeployData(unittest.TestCase):

    def test_events_dropped_logs_and_info_kept(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for keep, expected in ((True, ['done', 'events.lhe.gz', 'info.json', 'log',
                                       'weights.npy']),
                               (False, ['done', 'info.json', 'log'])):
            run = cluster_run.ClusterRun(types.SimpleNamespace(
                run_card=card(cluster__keep_job_files=keep)), {})
            run.cluster_path = os.path.join(tmp.name, str(keep))
            run.state = {'deploy_jobs': [{'index': 0}]}
            job_dir = run.job_dir('deploy', 0)
            os.makedirs(job_dir)
            for name in ('log', 'info.json', 'done', 'events.lhe.gz', 'weights.npy'):
                with open(os.path.join(job_dir, name), 'w') as f:
                    f.write('x')
            run.clean_deploy_data()
            self.assertEqual(sorted(os.listdir(job_dir)), expected)


@unittest.skipIf(cluster_run is None, 'madspace not available')
class TestPassOverrides(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        os.makedirs('Cards')
        run_card = card(run__device=['cpu'], run__cpu_thread_pool_size=-1)
        run_card.dynamic_sections['cluster_train'] = {
            'run': {'device': ['cuda:0'], 'gpu_thread_pool_size': 4}}
        run_card.dynamic_sections['cluster_deploy'] = {
            'run': {'cpu_thread_pool_size': 6}}
        run_card.write(os.path.join('Cards', 'run_card.toml'))
        self.run = cluster_run.ClusterRun(types.SimpleNamespace(
            run_card=RunCardMG7(os.path.join('Cards', 'run_card.toml'))), {})
        self.run.cluster_objects = {
            'train': (types.SimpleNamespace(nb_core=2), 'local_parallel')}

    def test_devices(self):
        self.assertEqual(self.run.devices, {'train': ['cuda:0'], 'deploy': ['cpu']})

    def test_threads(self):
        # an override is taken as it is, -1 shares the cores among the jobs
        self.assertEqual(self.run.thread_count('deploy', 'local_parallel'), 6)
        self.assertEqual(self.run.thread_count('train', 'local_parallel'),
                         max(1, (os.cpu_count() or 1) // 2))

    def test_train_card(self):
        self.run.cluster_type = 'slurm'
        self.run.backends = {'train': ['cuda'], 'deploy': ['cpp512y']}
        os.makedirs('job')
        self.run.write_train_card({'selection': [0], 'seed': 1}, 'job')
        job_card = RunCardMG7(os.path.join('job', 'run_card.toml'))
        self.assertEqual(job_card['run']['device'], ['cuda:0'])
        self.assertEqual(job_card['run']['gpu_thread_pool_size'], 4)
        self.assertEqual(job_card['run']['cpu_thread_pool_size'], -1)
        self.assertFalse(job_card['cluster']['enable'])


@unittest.skipIf(cluster_run is None, 'madspace not available')
class TestSupported(unittest.TestCase):

    def test_post_processing_tools_are_refused(self):
        cluster_run.check_supported({}, None)
        cluster_run.check_supported({'shower': 'OFF', 'madspin': 'OFF'}, None)
        with self.assertRaisesRegex(cluster_run.ClusterRunError, 'Pythia8'):
            cluster_run.check_supported({'shower': 'Pythia8'}, None)

    def test_reweighting_is_refused(self):
        with self.assertRaisesRegex(cluster_run.ClusterRunError, 'reweight'):
            cluster_run.check_supported({}, object())
