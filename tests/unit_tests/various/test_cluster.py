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
