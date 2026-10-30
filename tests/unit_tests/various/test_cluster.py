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
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit tests for madgraph.various.cluster"""

import threading
from unittest import mock

import madgraph.various.cluster as cluster
import tests.unit_tests as unittest


class TestMultiCore(unittest.TestCase):
    """Test the MultiCore cluster"""

    def test_failure_with_queued_jobs(self):
        """a failing job stops wait(), even with jobs still in the queue"""

        mycluster = cluster.MultiCore(nb_core=1)
        mycluster.submit('/bin/sh', ['-c', 'exit 1'])
        for i in range(2):
            mycluster.submit('/bin/sh', ['-c', 'sleep 30'])

        raised = []
        def wait():
            try:
                mycluster.wait('.', lambda idle, run, done: None)
            except Exception as error:
                raised.append(error)

        # wait() used to wait forever for the queued jobs, which the stopped
        # workers never start: do not let that block the test suite
        thread = threading.Thread(target=wait, daemon=True)
        thread.start()
        thread.join(60)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(raised), 1)
        self.assertIn('non zero status', str(raised[0]))


class TestExtraSubmitArgs(unittest.TestCase):
    """cluster_extra_args reaches the submit command of the batch systems"""

    def submit(self, name, extra, output):
        """The command line (and stdin) the cluster ``name`` submits with."""
        calls = []

        def popen(command, **opts):
            calls.append(command)
            process = mock.Mock(returncode=0)
            process.communicate.side_effect = lambda stdin=None: (
                calls.append(stdin) or (output, None))
            return process

        options = {'cluster_queue': 'gpu', 'cluster_type': name}
        if extra is not None:
            options['cluster_extra_args'] = extra
        # a failing submission is retried after a sleep: fail fast instead
        with mock.patch.object(cluster.misc, 'Popen', side_effect=popen), \
                mock.patch.object(cluster.misc.time, 'sleep', side_effect=RuntimeError):
            cluster.from_name[name](**options).submit(
                '/bin/echo', ['hello'], cwd='.', stdout='out', stderr='err')
        return calls

    def test_slurm(self):
        command = self.submit('slurm', '--gres=gpu:1 --mem="8 G"',
                              b'Submitted batch job 12')[0]
        self.assertEqual(command[0], 'sbatch')
        self.assertIn('--gres=gpu:1', command)
        self.assertIn('--mem=8 G', command)
        # options, not arguments of the job
        self.assertLess(command.index('--gres=gpu:1'), command.index('/bin/echo'))
        self.assertIn('gpu', command)

    def test_slurm_without_extra_args(self):
        command = self.submit('slurm', None, b'Submitted batch job 12')[0]
        self.assertFalse([arg for arg in command if arg.startswith('--gres')])

    def test_pbs(self):
        command = self.submit('pbs', '-l select=1:ngpus=1', b'123.server')[0]
        self.assertEqual(command[0], 'qsub')
        self.assertEqual(command[command.index('-q') + 1], 'gpu')
        self.assertIn('select=1:ngpus=1', command)

    def test_lsf(self):
        command = self.submit('lsf', '-gpu num=1', b'Job <12> is submitted')[0]
        self.assertEqual(command[0], 'bsub')
        self.assertEqual(command[command.index('-gpu') + 1], 'num=1')

    def test_condor_lines(self):
        calls = self.submit('condor', 'request_gpus = 1; request_memory = 8GB',
                            b'1 job(s) submitted to cluster 7.')
        text = calls[1].decode()
        self.assertIn('\nrequest_gpus = 1\n', text)
        self.assertIn('\nrequest_memory = 8GB\n', text)
        self.assertLess(text.index('request_gpus'), text.index('queue 1'))
