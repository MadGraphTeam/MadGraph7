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
"""`set eMELA` validates the eMELA-config it is given by running it with
--version. eMELA-config is a bash script built on egrep, and grep >= 3.8 (e.g.
AlmaLinux 10, Fedora) warns on stderr that egrep is obsolescent: treating any
stderr output as a failure silently dropped a working eMELA there, so every
eMELA acceptance test skipped."""

from __future__ import absolute_import

import os
import shutil
import stat
import tempfile
import unittest

import madgraph.interface.madgraph_interface as mg_interface

pjoin = os.path.join


class _Cmd(object):
    """the state set2_eMELA touches"""

    set2_eMELA = mg_interface.MadGraphCmd.set2_eMELA

    def __init__(self):
        self.options = {'eMELA': None}
        self.history = []

    def check_set(self, args):
        pass


class TestSetEMELA(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='set_emela_')
        self.cmd = _Cmd()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def script(self, body):
        path = pjoin(self.tmpdir, 'eMELA-config')
        with open(path, 'w') as fsock:
            fsock.write('#!/bin/sh\n%s\n' % body)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def test_stderr_warning_is_accepted(self):
        path = self.script('echo "egrep: warning: egrep is obsolescent" >&2\n'
                           'echo 1.1')
        self.cmd.set2_eMELA([path])
        self.assertEqual(self.cmd.options['eMELA'], path)

    def test_failing_executable_is_rejected(self):
        self.cmd.options['eMELA'] = 'previous'
        self.cmd.set2_eMELA([self.script('echo 1.1\nexit 1')])
        self.assertIsNone(self.cmd.options['eMELA'])

    def test_no_version_is_rejected(self):
        self.cmd.set2_eMELA([self.script('true')])
        self.assertIsNone(self.cmd.options['eMELA'])

    def test_missing_executable_is_rejected(self):
        self.cmd.set2_eMELA([pjoin(self.tmpdir, 'nope')])
        self.assertIsNone(self.cmd.options['eMELA'])


if __name__ == '__main__':
    unittest.main()
