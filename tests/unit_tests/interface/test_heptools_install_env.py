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
"""`install ninja` must not pass an inherited `make -jN` to Ninja's build.

Ninja's Automake rules install its Fortran .mod files as headers without a rule
that makes them; they only appear as a side effect of compiling the .F90
sources. installNinja.sh runs a plain `make`, so with MAKEFLAGS=-j32 in the
environment (e.g. a batch script exporting it) the build stopped with
"No rule to make target 'quadninjago_module.mod'". Other tools keep the
caller's environment."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import madgraph.interface.madgraph_interface as mg_interface

PARALLEL_MAKE = {'MAKEFLAGS': '-j32', 'MFLAGS': '-j32', 'GNUMAKEFLAGS': '-j32'}


class HEPToolsInstallerEnvironmentTest(unittest.TestCase):

    def setUp(self):
        self.mg_dir = tempfile.mkdtemp(prefix='heptools_env_')
        os.makedirs(os.path.join(self.mg_dir, 'HEPTools', 'HEPToolsInstallers'))

    def tearDown(self):
        shutil.rmtree(self.mg_dir)

    def installer_env(self, tool):
        """Run advanced_install(tool) with the installer replaced by a stub
        and return the env passed to it (None means inherited)."""
        cmd = mock.MagicMock()
        cmd.options = {'cpp_compiler': None, 'fortran_compiler': None,
                       'heptools_install_dir': None}
        cmd.heptools_install_target.return_value = (
            os.path.join(self.mg_dir, 'HEPTools'), 'mg7_configuration.txt')
        cmd.install_plugin = []
        cmd.SERIAL_MAKE_HEPTOOLS = mg_interface.MadGraphCmd.SERIAL_MAKE_HEPTOOLS
        cmd.environment_without_parallel_make = \
            mg_interface.MadGraphCmd.environment_without_parallel_make
        calls = []

        def installer(args, *pos, **opts):
            calls.append(opts)
            return 0

        misc = mg_interface.misc
        with mock.patch.object(mg_interface, 'MG5DIR', self.mg_dir), \
             mock.patch.object(misc, 'call', side_effect=installer), \
             mock.patch.object(misc, 'detect_cpp_std_lib_dependence',
                               return_value='-lstdc++'), \
             mock.patch.object(misc, 'get_ninja_quad_prec_support',
                               return_value=True), \
             mock.patch.dict(os.environ, PARALLEL_MAKE):
            mg_interface.MadGraphCmd.advanced_install(cmd, tool)
        self.assertEqual(len(calls), 1)
        return calls[0].get('env')

    def test_ninja_installer_runs_without_parallel_make_flags(self):
        env = self.installer_env('ninja')
        self.assertIsNotNone(env)
        for key in PARALLEL_MAKE:
            self.assertNotIn(key, env)
        self.assertEqual(env.get('PATH'), os.environ.get('PATH'))

    def test_other_tools_inherit_the_environment(self):
        self.assertIsNone(self.installer_env('collier'))

    def test_helper_leaves_the_given_environment_untouched(self):
        given = dict(PARALLEL_MAKE, PATH='/bin')
        cleaned = mg_interface.MadGraphCmd.environment_without_parallel_make(given)
        self.assertEqual(cleaned, {'PATH': '/bin'})
        self.assertEqual(given['MAKEFLAGS'], '-j32')


if __name__ == '__main__':
    unittest.main()
