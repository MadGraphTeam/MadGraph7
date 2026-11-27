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

from madgraph.mg7 import bootstrap as mg7_bootstrap

# Importing launch without a compiled madspace starts the madspace build, and
# test_manager imports every unit test module while collecting: check first.
launch = None
if mg7_bootstrap.madspace_is_installed():
    try:
        import madgraph.mg7.launch as launch
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
                        'cluster_walltime = None\ncluster_local_path = /cvmfs/mg\n'
                        'enforce_shared_disk = True\n')
            options = launch.load_mg5_options(me_dir, with_cluster=True)
            plain = launch.load_mg5_options(me_dir)
        self.assertEqual(options['cluster_type'], 'slurm')
        self.assertEqual(options['cluster_queue'], 'gpu')
        # a value of None leaves the default
        self.assertNotIn('cluster_walltime', options)
        # the batch system classes get the whole configuration, as from madevent
        self.assertEqual(options['cluster_local_path'], '/cvmfs/mg')
        self.assertIs(options['enforce_shared_disk'], True)
        self.assertNotIn('enforce_shared_disk', plain)
        # the madevent interface applies every option it is given
        self.assertFalse([key for key in plain if key.startswith('cluster')])

    def test_cluster_enabled_takes_the_site_default(self):
        """A card without cluster.enable takes the site default, as RunCardMG7
        does; the setting of the card wins."""
        from unittest import mock
        from madgraph.various.banner import RunCardMG7
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            self.addCleanup(os.chdir, cwd)
            os.makedirs('Cards')
            site = os.path.join(tmp, 'default_run_card_mg7.toml')

            def enabled(card, site_text):
                with open(os.path.join('Cards', 'run_card.toml'), 'w') as f:
                    f.write(card)
                with open(site, 'w') as f:
                    f.write(site_text)
                with mock.patch.object(RunCardMG7, 'default_run_card', site):
                    return launch.cluster_enabled()

            on = '[cluster]\nenable = true\n'
            off = '[cluster]\nenable = false\n'
            self.assertTrue(enabled('[run]\n', on))
            self.assertFalse(enabled('[run]\n', ''))
            self.assertFalse(enabled(off, on))
            self.assertTrue(enabled(on, off))
