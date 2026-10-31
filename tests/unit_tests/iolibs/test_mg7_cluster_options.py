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
"""The batch system of the cluster mode comes from the MadGraph configuration
where the [cluster] section does not name one."""
from __future__ import absolute_import
import os
import tempfile
import unittest

from madgraph.iolibs.template_files.mg7 import bootstrap as mg7_bootstrap

# Importing launch without a compiled madspace starts the madspace build, and
# test_manager imports every unit test module while collecting: check first.
launch = None
if mg7_bootstrap.madspace_is_installed():
    try:
        import madgraph.iolibs.template_files.mg7.launch as launch
    except Exception as error:
        launch = None
        import_error = error


@unittest.skipIf(launch is None, 'madspace not available')
class TestClusterOptions(unittest.TestCase):

    def test_cluster_options_are_read(self):
        with tempfile.TemporaryDirectory() as me_dir:
            os.makedirs(os.path.join(me_dir, 'Cards'))
            with open(os.path.join(me_dir, 'Cards', 'me5_configuration.txt'), 'w') as f:
                f.write('cluster_type = slurm\ncluster_queue = gpu # the queue\n'
                        'cluster_walltime = None\n')
            options = launch.load_mg5_options(me_dir, with_cluster=True)
            plain = launch.load_mg5_options(me_dir)
        self.assertEqual(options['cluster_type'], 'slurm')
        self.assertEqual(options['cluster_queue'], 'gpu')
        # a value of None leaves the default
        self.assertNotIn('cluster_walltime', options)
        # the madevent interface applies every option it is given
        self.assertFalse([key for key in plain if key.startswith('cluster')])
