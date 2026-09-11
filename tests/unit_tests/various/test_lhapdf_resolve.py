################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
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
"""Test the configuration-file locations and the LHAPDF path resolution."""

from __future__ import absolute_import
import os
import shutil
import stat
import tempfile
import unittest

import madgraph.various.misc as misc

pjoin = os.path.join


class TestUserConfigLocation(unittest.TestCase):
    """MadGraph7 must keep its per-user configuration to itself: a file shared
    with an MadGraph7 installation is what issue #94 is about."""

    def setUp(self):
        self.saved = {key: os.environ.get(key)
                      for key in ('HOME', 'XDG_CONFIG_HOME')}

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_xdg_config_home_wins(self):
        os.environ['HOME'] = '/home/someone'
        os.environ['XDG_CONFIG_HOME'] = '/xdg'
        self.assertEqual(misc.user_config_dir(), pjoin('/xdg', 'mg7'))
        self.assertEqual(misc.user_config_file(),
                         pjoin('/xdg', 'mg7', 'mg7_configuration.txt'))

    def test_home_fallback(self):
        os.environ['HOME'] = '/home/someone'
        os.environ.pop('XDG_CONFIG_HOME', None)
        self.assertEqual(misc.user_config_dir(), pjoin('/home/someone', '.mg7'))
        self.assertEqual(misc.user_config_file(),
                         pjoin('/home/someone', '.mg7', 'mg7_configuration.txt'))

    def test_never_mg5(self):
        """No location MadGraph7 uses may live under MadGraph7's ~/.mg5."""
        os.environ['HOME'] = '/home/someone'
        os.environ.pop('XDG_CONFIG_HOME', None)
        self.assertNotIn('.mg5', misc.user_config_file())
        self.assertNotIn('mg5_configuration', misc.user_config_file())
        self.assertNotIn('mg5_configuration', misc.install_config_file('/mg'))

    def test_no_home(self):
        os.environ.pop('HOME', None)
        os.environ.pop('XDG_CONFIG_HOME', None)
        self.assertIsNone(misc.user_config_dir())
        self.assertIsNone(misc.user_config_file())


class TestResolveLhapdf(unittest.TestCase):
    """misc.resolve_lhapdf is the single place both 'launch' and a standalone
    bin/generate_events use to locate LHAPDF, so it carries all the awkward
    cases: relative option values, a '--python=' filter, a path list, and a
    broken or absent lhapdf-config."""

    # what a lhapdf-config answers, keyed by the flag it supports
    SCRIPT = ('#!/bin/sh\n'
              'case "$1" in\n'
              '  %(flag)s) echo "%(datadir)s" ;;\n'
              '  --version) echo 6.5.4 ;;\n'
              '  *) exit 1 ;;\n'
              'esac\n')

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='mg7_lhapdf_test')
        self.root = pjoin(self.tmpdir, 'MG')
        self.datadir = pjoin(self.tmpdir, 'lha6', 'share', 'LHAPDF')
        os.makedirs(pjoin(self.datadir, 'MYSET'))
        os.makedirs(pjoin(self.root, 'input'))
        self.exe = self.write_config('lha6', self.datadir)
        self.saved = {key: os.environ.get(key)
                      for key in ('LHAPDF_DATA_PATH', 'MADGRAPH_LHAPDF_CONFIG',
                                  'PATH', 'HOME', 'XDG_CONFIG_HOME')}
        os.environ.pop('LHAPDF_DATA_PATH', None)
        os.environ.pop('MADGRAPH_LHAPDF_CONFIG', None)
        # the resolver caches per executable path, and every test writes its
        # own, but a stale entry from another test module would still leak in
        misc._lhapdf_datadirs_cache.clear()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        misc._lhapdf_datadirs_cache.clear()

    def write_config(self, name, datadir, flag='--datadir'):
        """Write a fake lhapdf-config under <tmpdir>/<name>/bin and return it."""
        bindir = pjoin(self.tmpdir, name, 'bin')
        if not os.path.isdir(bindir):
            os.makedirs(bindir)
        exe = pjoin(bindir, 'lhapdf-config')
        with open(exe, 'w') as fsock:
            fsock.write(self.SCRIPT % {'flag': flag, 'datadir': datadir})
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)
        return exe

    def resolve(self, **options):
        # the CVMFS mirror is a real path which may be mounted on the machine
        # running the tests: opt out unless the test is about it
        options.setdefault('cvmfs_lhapdf_path', None)
        return misc.resolve_lhapdf(options, root=self.root)

    def test_absolute_lhapdf(self):
        paths = self.resolve(lhapdf=self.exe)
        self.assertEqual(paths.config, self.exe)
        self.assertEqual(paths.data_paths, [self.datadir])
        self.assertEqual(paths.find_set('MYSET'), self.datadir)
        self.assertIsNone(paths.find_set('OTHERSET'))

    def test_python_filter_is_not_part_of_the_path(self):
        paths = self.resolve(lhapdf='%s --python=3.11' % self.exe)
        self.assertEqual(paths.config, self.exe)
        self.assertEqual(paths.data_paths, [self.datadir])

    def test_relative_value_roots_at_the_install(self):
        """'./HEPTools/...' means relative to the MadGraph root, not to cwd."""
        exe = self.write_config(pjoin('MG', 'HEPTools', 'lhapdf6'), self.datadir)
        relative = os.path.join('.', 'HEPTools', 'lhapdf6', 'bin', 'lhapdf-config')
        paths = self.resolve(lhapdf=relative)
        self.assertEqual(paths.config, exe)
        self.assertEqual(paths.data_paths, [self.datadir])

    def test_bare_name_from_path(self):
        os.environ['PATH'] = os.path.dirname(self.exe)
        paths = self.resolve(lhapdf='lhapdf-config')
        self.assertEqual(paths.config, self.exe)

    def test_missing_executable_does_not_raise(self):
        os.environ['PATH'] = pjoin(self.tmpdir, 'empty')
        paths = self.resolve(lhapdf=pjoin(self.tmpdir, 'nope', 'lhapdf-config'))
        self.assertIsNone(paths.config)
        self.assertEqual(paths.data_paths, [])
        self.assertEqual(paths.download_path,
                         pjoin(self.root, 'HEPTools', 'lhapdf_pdfsets'))

    def test_no_lhapdf_at_all(self):
        os.environ['PATH'] = pjoin(self.tmpdir, 'empty')
        paths = self.resolve()
        self.assertIsNone(paths.config)
        self.assertEqual(paths.data_paths, [])

    def test_lhapdf5_pdfsets_path(self):
        exe = self.write_config('lha5', self.datadir, flag='--pdfsets-path')
        paths = self.resolve(lhapdf=exe)
        self.assertEqual(paths.config, exe)
        self.assertEqual(paths.data_paths, [self.datadir])

    def test_datadir_list_is_split(self):
        """A path list must be split; joining it into a filename is the bug
        that made a multi-entry LHAPDF_DATA_PATH silently unusable."""
        other = pjoin(self.tmpdir, 'other')
        os.makedirs(other)
        exe = self.write_config('lhamulti',
                                os.pathsep.join([other, self.datadir]))
        paths = self.resolve(lhapdf=exe)
        self.assertEqual(paths.data_paths, [other, self.datadir])
        self.assertEqual(paths.find_set('MYSET'), self.datadir)

    def test_env_data_path_comes_first(self):
        other = pjoin(self.tmpdir, 'other')
        os.makedirs(pjoin(other, 'MYSET'))
        os.environ['LHAPDF_DATA_PATH'] = os.pathsep.join([other, self.datadir])
        paths = misc.resolve_lhapdf({'lhapdf': self.exe,
                                     'cvmfs_lhapdf_path': None}, root=self.root)
        self.assertEqual(paths.data_paths[0], other)
        self.assertEqual(paths.find_set('MYSET'), other)
        # ... and use_env=False ignores the environment entirely
        paths = misc.resolve_lhapdf({'lhapdf': self.exe,
                                     'cvmfs_lhapdf_path': None}, root=self.root,
                                    use_env=False)
        self.assertEqual(paths.data_paths, [self.datadir])

    def test_env_config_wins_over_options(self):
        other = pjoin(self.tmpdir, 'other')
        os.makedirs(other)
        exe = self.write_config('lhaenv', other)
        os.environ['MADGRAPH_LHAPDF_CONFIG'] = exe
        paths = misc.resolve_lhapdf({'lhapdf': self.exe,
                                     'cvmfs_lhapdf_path': None}, root=self.root)
        self.assertEqual(paths.config, exe)

    def test_lhapdf_py3_fallback(self):
        """set2_lhapdf leaves 'lhapdf' at its default when a --python= filter
        does not match the running interpreter, so lhapdf_py3 must be tried."""
        os.environ['PATH'] = pjoin(self.tmpdir, 'empty')
        paths = self.resolve(lhapdf=pjoin(self.tmpdir, 'nope', 'lhapdf-config'),
                             lhapdf_py3=self.exe)
        self.assertEqual(paths.config, self.exe)
        self.assertEqual(paths.data_paths, [self.datadir])

    def test_none_and_auto_are_not_paths(self):
        os.environ['PATH'] = pjoin(self.tmpdir, 'empty')
        self.assertIsNone(self.resolve(lhapdf='None').config)
        self.assertIsNone(self.resolve(lhapdf='auto').config)

    def test_heptools_fallback_when_datadir_is_read_only(self):
        if os.geteuid() == 0:
            self.skipTest('root can write to a read-only directory')
        mode = os.stat(self.datadir).st_mode
        os.chmod(self.datadir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            paths = self.resolve(lhapdf=self.exe)
            self.assertEqual(paths.download_path,
                             pjoin(self.root, 'HEPTools', 'lhapdf_pdfsets'))
        finally:
            os.chmod(self.datadir, mode)

    def test_create_makes_the_download_directory(self):
        os.environ['PATH'] = pjoin(self.tmpdir, 'empty')
        target = pjoin(self.root, 'HEPTools', 'lhapdf_pdfsets')
        paths = misc.resolve_lhapdf({'cvmfs_lhapdf_path': None}, root=self.root)
        self.assertEqual(paths.download_path, target)
        self.assertFalse(os.path.isdir(target))
        paths = misc.resolve_lhapdf({'cvmfs_lhapdf_path': None}, root=self.root,
                                    create=True)
        self.assertTrue(os.path.isdir(target))

    def test_cvmfs_mirror_is_searched_last_and_never_downloaded_into(self):
        """a set present on the CVMFS mirror is found there instead of being
        downloaded, but the mirror is read-only so it is never a download
        target"""

        mirror = pjoin(self.tmpdir, 'cvmfs')
        os.makedirs(pjoin(mirror, 'CVMFSSET'))
        paths = self.resolve(lhapdf=self.exe, cvmfs_lhapdf_path=mirror)
        self.assertEqual(paths.data_paths, [self.datadir, mirror])
        self.assertEqual(paths.find_set('CVMFSSET'), mirror)
        # the local data directory still wins for a set it holds
        self.assertEqual(paths.find_set('MYSET'), self.datadir)
        self.assertNotEqual(paths.download_path, mirror)
        # a mirror which is not mounted is simply ignored
        paths = self.resolve(lhapdf=self.exe,
                             cvmfs_lhapdf_path=pjoin(self.tmpdir, 'nomount'))
        self.assertEqual(paths.data_paths, [self.datadir])

    def test_with_data_path_promotes_a_directory(self):
        paths = self.resolve(lhapdf=self.exe)
        promoted = paths.with_data_path('/somewhere')
        self.assertEqual(promoted.data_paths[0], '/somewhere')
        self.assertEqual(promoted.config, paths.config)
