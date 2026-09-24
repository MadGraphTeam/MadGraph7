################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Test multicore job submission."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import madgraph.various.cluster as cluster


class TestMultiCore(unittest.TestCase):

    def test_gpu_environment_not_passed_to_python_job(self):
        """GPU subprocess settings must not become Python function kwargs."""
        calls = []

        def python_job(value):
            calls.append(value)
            return 0

        with mock.patch.dict(os.environ,
                             {'NVIDIA_VISIBLE_DEVICES': 'GPU-test'},
                             clear=True):
            multicore = cluster.MultiCore(nb_core=1,
                                          cluster_temp_path=None)
            multicore.submit(python_job, argument=['done'])
            multicore.wait('.', lambda *args: None)

        self.assertEqual(calls, ['done'])

    def test_gpu_environment_is_passed_to_executable(self):
        """GPU selection remains part of a submitted process environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'gpu.txt')
            code = ('import os; open(%r, "w").write('
                    'os.environ["CUDA_VISIBLE_DEVICES"])' % output)
            with mock.patch.dict(os.environ,
                                 {'NVIDIA_VISIBLE_DEVICES': 'GPU-test'},
                                 clear=True):
                multicore = cluster.MultiCore(nb_core=1,
                                              cluster_temp_path=None)
                multicore.submit(sys.executable, argument=['-c', code])
                multicore.wait('.', lambda *args: None)

            with open(output) as stream:
                self.assertEqual(stream.read(), 'GPU-test')
