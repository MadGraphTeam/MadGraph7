"""Installer lifecycle, source isolation, CLI dispatch and exported ABI checks."""

import io
import json
from pathlib import Path
import shlex
import shutil
import tempfile
import types
import unittest
from unittest import mock

from madgraph import MadGraph5Error
from madgraph.interface.madgraph_interface import MadGraphCmd
from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA
from madgraph.various import banner, ninjars_install


class TestNinjaRSInstaller(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / 'HEPTools'
        self.bridge = self.root / 'vendor' / 'ninjars'
        self.bridge.mkdir(parents=True)
        (self.bridge / 'Cargo.lock').write_text('# pinned test fixture\n')
        (self.bridge / 'Cargo.toml').write_text('[package]\nname = "bridge"\n')
        (self.bridge / 'smoke.f90').write_text('end\n')

    def fake_build(self, command, cwd, log, environment):
        """Emulate compiler output while exercising staging/publish for real."""
        log.write_text('build and smoke log\n')
        if command[0] == 'make':
            prefix = Path(next(item[7:] for item in command
                               if item.startswith('PREFIX=')))
            lib = prefix / 'lib'
            lib.mkdir(parents=True)
            (lib / 'libninjars.a').write_text('new working archive')
            (lib / 'mninjars.mod').write_text('Fortran module')
            (lib / 'ninjars-native-libs.txt').write_text('-lgmp -lmpfr -lpthread\n')

    def test_publish_and_idempotent_install(self):
        with mock.patch.object(ninjars_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(ninjars_install, '_run', side_effect=self.fake_build) as run:
            libdir = ninjars_install.install(self.root, self.prefix, jobs=2)
            config = ninjars_install.read_config(libdir)
            self.assertTrue(config['quad_precision'])
            self.assertEqual(config['native_libs'], ['-lgmp', '-lmpfr', '-lpthread'])
            self.assertTrue((self.prefix / 'lib' / 'libninjars.a').is_file())
            self.assertTrue((self.prefix / 'ninjars' / 'Cargo.lock').is_file())
            self.assertTrue((self.prefix / 'ninjars' / 'install.log').is_file())
            self.assertEqual(len(run.call_args_list), 3)
            self.assertEqual(run.call_args_list[0].args[3]['CARGO_BUILD_JOBS'], '2')
            self.assertEqual(ninjars_install.install(self.root, self.prefix), libdir)
            self.assertEqual(len(run.call_args_list), 3)

    def test_failed_rebuild_preserves_previous_installation(self):
        lib = self.prefix / 'ninjars' / 'lib'
        lib.mkdir(parents=True)
        (lib / 'libninjars.a').write_text('previous working archive')

        def fail_smoke(command, cwd, log, environment):
            self.fake_build(command, cwd, log, environment)
            if command[0].endswith('/smoke'):
                raise MadGraph5Error('quad-precision smoke failed')

        with mock.patch.object(ninjars_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(ninjars_install, '_run', side_effect=fail_smoke):
            with self.assertRaisesRegex(MadGraph5Error, 'quad-precision'):
                ninjars_install.install(self.root, self.prefix, ['--force'])
        self.assertEqual((lib / 'libninjars.a').read_text(), 'previous working archive')

    def test_local_sources_are_copied_and_original_is_unchanged(self):
        source = self.root / 'reference'
        for relative in ('Cargo.toml', 'crates/ninjars/Cargo.toml',
                         'crates/ninjars-symbolica/Cargo.toml'):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# original\n')
        (source / 'target').mkdir()
        (source / 'target' / 'cached.bin').write_text('do not copy')
        stage = self.root / 'stage'
        stage.mkdir()
        bridge = stage / 'bridge'
        bridge.mkdir()
        options = ninjars_install.build_parser().parse_args(['--source', str(source)])
        with mock.patch.object(ninjars_install.subprocess, 'check_output',
                               return_value='1234abcd\n'):
            provenance = ninjars_install._prepare_overrides(bridge, stage, options)
        config = (bridge / '.cargo' / 'config.toml').read_text()
        self.assertIn(str(stage / 'sources' / 'ninjars'), config)
        self.assertNotIn(str(source), config)
        copied = stage / 'sources' / 'ninjars'
        self.assertFalse((copied / 'target').exists())
        (copied / 'Cargo.toml').write_text('# changed in build\n')
        self.assertEqual((source / 'Cargo.toml').read_text(), '# original\n')
        self.assertEqual(provenance['ninjars']['revision'], '1234abcd')

    def test_missing_tool_and_bad_job_count_fail_before_build(self):
        with mock.patch.object(ninjars_install.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(MadGraph5Error, 'requires cargo'):
                ninjars_install.install(self.root, self.prefix)
        for jobs in (0, -2):
            with self.subTest(jobs=jobs), self.assertRaisesRegex(
                    MadGraph5Error, 'positive'):
                ninjars_install.install(self.root, self.prefix,
                                        ['--jobs=%s' % jobs])
        self.assertFalse(self.prefix.exists())

    def test_metadata_requires_compatible_abi(self):
        lib = self.root / 'lib'
        lib.mkdir()
        config = lib / ninjars_install.CONFIG_NAME
        for version in (1, 99):
            config.write_text(json.dumps({'abi_version': version, 'native_libs': []}))
            with self.subTest(version=version), self.assertRaisesRegex(
                    MadGraph5Error, 'ABI'):
                ninjars_install.read_config(lib)
        config.write_text(json.dumps({'abi_version': ninjars_install.ABI_VERSION,
                                     'native_libs': 'bad'}))
        with self.assertRaisesRegex(MadGraph5Error, 'link metadata'):
            ninjars_install.read_config(lib)
        self.assertFalse(ninjars_install.supports_quad_precision(lib))

    def test_cli_dispatch_is_local_and_saves_only_after_success(self):
        cmd = types.SimpleNamespace(
            options={'heptools_install_dir': str(self.prefix),
                     'fortran_compiler': 'gfortran', 'nb_core': 2},
            split_arg=shlex.split,
            check_install=lambda args: {'options_for_HEPToolsInstaller': args[1:]},
            heptools_install_target=lambda prefix: (prefix, 'config.txt'),
            exec_cmd=mock.Mock())
        with mock.patch.object(ninjars_install, 'install',
                               return_value='/installed/lib') as install:
            MadGraphCmd.do_install(cmd, 'ninjars --jobs=3')
        self.assertEqual(cmd.options['ninjars'], '/installed/lib')
        self.assertEqual(install.call_args.kwargs['arguments'], ['--jobs=3'])
        cmd.exec_cmd.assert_called_once_with('save options config.txt ninjars',
                                            printcmd=False, log=False)
        cmd.exec_cmd.reset_mock()
        with mock.patch.object(ninjars_install, 'install',
                               side_effect=MadGraph5Error('failed')):
            with self.assertRaises(MadGraph5Error):
                MadGraphCmd.do_install(cmd, 'ninjars --force')
        cmd.exec_cmd.assert_not_called()

    def test_exporter_rejects_library_without_module(self):
        lib = self.root / 'lib'
        lib.mkdir()
        (lib / 'libninjars.a').write_text('archive')
        (lib / ninjars_install.CONFIG_NAME).write_text(json.dumps(
            {'abi_version': ninjars_install.ABI_VERSION, 'native_libs': [],
             'quad_precision': True}))
        exporter = object.__new__(LoopProcessOptimizedExporterFortranSA)
        exporter.tir_available_dict = {'ninjars': True}
        exporter.dependencies = 'external'
        with self.assertRaisesRegex(MadGraph5Error, 'Fortran module'):
            exporter.link_TIR(str(self.root), str(lib), 'libninjars.a', 'ninjars')
        (lib / 'mninjars.mod').write_text('module')
        self.assertEqual(exporter.link_TIR(
            str(self.root), str(lib), 'libninjars.a', 'ninjars'), str(lib))
        exporter.dependencies = 'internal'
        (lib / 'ninjars-native-libs.txt').write_text('-lgmp\n')
        include = self.root / 'include'
        include.mkdir()
        (include / 'ninjars.h').write_text('/* header */\n')
        copied = self.root / 'export' / 'lib'
        self.assertEqual(exporter.link_TIR(
            str(copied), str(lib), 'libninjars.a', 'ninjars'), str(copied))
        shutil.rmtree(lib)
        shutil.rmtree(include)
        self.assertTrue(ninjars_install.supports_quad_precision(copied))
        for name in ('libninjars.a', 'mninjars.mod', 'ninjars-config.json',
                     'ninjars-native-libs.txt', 'ninjars.h'):
            self.assertTrue((copied / name).is_file(), name)
            self.assertFalse((copied / name).is_symlink(), name)

    def test_madloop_card_round_trip_keeps_id8_and_qp_option(self):
        card = banner.MadLoopParam()
        card['MLReductionLib'] = '8|6|1'
        card['UseQPIntegrandForNinjaRS'] = False
        stream = io.StringIO()
        card.write(stream)
        reread = banner.MadLoopParam(stream.getvalue())
        self.assertEqual(reread['MLReductionLib'], '8|6|1')
        self.assertFalse(reread['UseQPIntegrandForNinjaRS'])
        self.assertEqual(card._ID_reduction_tool_map[8], 'NinjaRS')


if __name__ == '__main__':
    unittest.main()
