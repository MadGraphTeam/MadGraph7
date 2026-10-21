"""Installer lifecycle, source isolation, CLI dispatch and exported ABI checks."""

import io
import hashlib
import json
import os
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
        (self.bridge / 'Cargo.lock').write_text('''version = 4
[[package]]
name = "ninjars"
version = "0.1.0"
source = "git+https://github.com/alphal00p/ninjars?rev=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
[[package]]
name = "oneloop"
version = "0.1.0"
source = "git+https://github.com/alphal00p/oneloopmaster.git?rev=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb#bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
''')
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
            (lib / 'ninjars-link-before.txt').write_text('-lgcc_s\n')

    @staticmethod
    def snapshot(directory):
        """Compare symlink text and regular bytes without following links."""
        result = {}
        for path in directory.rglob('*'):
            name = str(path.relative_to(directory))
            if path.is_symlink():
                result[name] = ('symlink', os.readlink(path))
            elif path.is_file():
                result[name] = ('file', path.read_bytes())
            elif path.is_dir():
                result[name] = ('directory', None)
        return result

    def previous_install(self):
        previous = self.prefix / 'ninjars'
        (previous / 'lib').mkdir(parents=True)
        (previous / 'lib/libninjars.a').write_text('previous working archive')
        (previous / 'lib/mninjars.mod').write_text('previous module')
        (previous / 'install.log').write_text('previous validated smoke')
        (previous / 'lib' / ninjars_install.CONFIG_NAME).write_text(json.dumps({
            'abi_version':ninjars_install.ABI_VERSION, 'quad_precision':True,
            'native_libs':['-lgmp'], 'link_before':['-lgcc_s'], 'link_order_version':1}))
        return previous

    def test_publish_and_idempotent_install(self):
        with mock.patch.object(ninjars_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(ninjars_install, '_run', side_effect=self.fake_build) as run:
            libdir = ninjars_install.install(self.root, self.prefix, jobs=2)
            config = ninjars_install.read_config(libdir)
            self.assertTrue(config['quad_precision'])
            self.assertEqual(config['native_libs'], ['-lgmp', '-lmpfr', '-lpthread'])
            self.assertEqual(config['link_before'], ['-lgcc_s'])
            provenance = config['build_provenance']
            self.assertEqual(provenance['lock_sha256'], hashlib.sha256(
                (self.bridge/'Cargo.lock').read_bytes()).hexdigest())
            self.assertEqual(provenance['manifest_sha256'], hashlib.sha256(
                (self.bridge/'Cargo.toml').read_bytes()).hexdigest())
            self.assertEqual([item['name'] for item in provenance['git_dependencies']],
                             ['ninjars', 'oneloop'])
            self.assertIn('#' + 'b'*40,
                          provenance['git_dependencies'][1]['source'])
            smoke_link = run.call_args_list[1].args[0]
            self.assertLess(smoke_link.index('-lgcc_s'), next(
                i for i, arg in enumerate(smoke_link) if arg.endswith('/libninjars.a')))
            self.assertTrue((self.prefix / 'lib' / 'libninjars.a').is_file())
            self.assertTrue((self.prefix / 'ninjars' / 'Cargo.lock').is_file())
            self.assertTrue((self.prefix / 'ninjars' / 'install.log').is_file())
            self.assertEqual(len(run.call_args_list), 3)
            self.assertEqual(run.call_args_list[0].args[3]['CARGO_BUILD_JOBS'], '2')
            self.assertEqual(ninjars_install.install(self.root, self.prefix), libdir)
            self.assertEqual(len(run.call_args_list), 3)

    def test_update_rebuilds_installed_backend_with_current_checkout_pins(self):
        previous = self.previous_install()
        # Simulate a git pull selecting a newer immutable dependency revision.
        lock = self.bridge/'Cargo.lock'
        lock.write_text(lock.read_text().replace('a'*40, 'c'*40))
        with mock.patch.object(ninjars_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(ninjars_install, 'read_config',
                                  side_effect=AssertionError('already-installed fast path')), \
                mock.patch.object(ninjars_install, '_run', side_effect=self.fake_build) as run:
            libdir = Path(ninjars_install.install(self.root, self.prefix, ['--update']))
        self.assertEqual(libdir/'libninjars.a', previous/'lib/libninjars.a')
        self.assertEqual((libdir/'libninjars.a').read_text(), 'new working archive')
        self.assertEqual((previous/'Cargo.lock').read_bytes(), lock.read_bytes())
        provenance = ninjars_install.read_config(libdir)['build_provenance']
        self.assertEqual(provenance['lock_sha256'], hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertIn('#'+'c'*40, provenance['git_dependencies'][0]['source'])
        self.assertEqual(len(run.call_args_list), 3)
        parser = ninjars_install.build_parser()
        self.assertTrue(parser.parse_args(['--update']).force)
        self.assertTrue(parser.parse_args(['--force']).force)

    def test_failed_update_build_or_smoke_preserves_previous_installation(self):
        previous = self.previous_install()
        common = self.prefix/'lib'
        common.mkdir()
        (common/'libninjars.a').symlink_to('../ninjars/lib/libninjars.a')
        before_tree, before_aliases = self.snapshot(previous), self.snapshot(common)
        for failing_step in ('make', 'smoke'):
            def fail_update(command, cwd, log, environment):
                if (command[0] == 'make' and failing_step == 'make') or (
                        command[0].endswith('/smoke') and failing_step == 'smoke'):
                    raise MadGraph5Error('failed update '+failing_step)
                self.fake_build(command, cwd, log, environment)
            with self.subTest(failing_step=failing_step), \
                    mock.patch.object(ninjars_install.shutil, 'which', return_value='/bin/tool'), \
                    mock.patch.object(ninjars_install, '_run', side_effect=fail_update):
                with self.assertRaisesRegex(MadGraph5Error, 'failed update '+failing_step):
                    ninjars_install.install(self.root, self.prefix, ['--update'])
            self.assertEqual(before_tree, self.snapshot(previous))
            self.assertEqual(before_aliases, self.snapshot(common))

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

    def test_alias_publication_failure_rolls_back_previous_install_and_retry(self):
        previous = self.previous_install()
        common = self.prefix / 'lib'
        common.mkdir()
        (common / 'libninjars.a').symlink_to('../ninjars/lib/libninjars.a')
        (common / 'mninjars.mod').symlink_to('../custom/old-module.mod')
        (common / ninjars_install.CONFIG_NAME).symlink_to('../missing/old-metadata.json')
        (common / 'ninjars-native-libs.txt').write_text('user-owned regular flags')
        (common / 'ninjars-link-before.txt').symlink_to('../custom/old-runtime.txt')
        sentinel = common / '.ninjars-link-before.txt.ninjars'
        sentinel.write_text('user-owned hidden temporary sentinel')
        before_tree, before_aliases = self.snapshot(previous), self.snapshot(common)
        replace = ninjars_install.os.replace
        injected = []
        def fail_alias(source, target):
            if Path(target) == common / ninjars_install.CONFIG_NAME and not injected:
                injected.append(True)
                raise OSError('simulated alias publication failure')
            return replace(source, target)
        with mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                mock.patch.object(ninjars_install,'_run',side_effect=self.fake_build):
            with mock.patch.object(ninjars_install.os,'replace',side_effect=fail_alias):
                with self.assertRaisesRegex(OSError,'alias publication failure'):
                    ninjars_install.install(self.root,self.prefix,['--force'])
            self.assertEqual(before_tree,self.snapshot(previous))
            self.assertEqual(before_aliases,self.snapshot(common))
            libdir = ninjars_install.install(self.root,self.prefix,['--force'])
        self.assertEqual(injected,[True])
        self.assertEqual((Path(libdir)/'libninjars.a').read_text(),'new working archive')
        self.assertEqual((common/'ninjars-native-libs.txt').read_text(),'user-owned regular flags')
        self.assertEqual(sentinel.read_text(),'user-owned hidden temporary sentinel')

    def test_first_install_alias_failure_removes_publication_and_retry_succeeds(self):
        common = self.prefix/'lib'
        replace = ninjars_install.os.replace
        injected = []
        def fail_alias(source,target):
            if Path(target) == common/'mninjars.mod' and not injected:
                injected.append(True)
                raise OSError('first install alias publication failure')
            return replace(source,target)
        with mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                mock.patch.object(ninjars_install,'_run',side_effect=self.fake_build):
            with mock.patch.object(ninjars_install.os,'replace',side_effect=fail_alias):
                with self.assertRaisesRegex(OSError,'alias publication failure'):
                    ninjars_install.install(self.root,self.prefix)
            self.assertFalse((self.prefix/'ninjars').exists())
            self.assertFalse(common.exists())
            self.assertTrue(list(self.prefix.glob('.ninjars-build-*/install.log')))
            libdir = ninjars_install.install(self.root,self.prefix)
        self.assertTrue(ninjars_install.supports_quad_precision(libdir))
        self.assertEqual((common/'libninjars.a').read_text(),'new working archive')

    def test_common_lib_mkdir_failure_restores_previous_install_and_retry(self):
        previous = self.previous_install()
        before = self.snapshot(previous)
        common = self.prefix/'lib'
        mkdir = Path.mkdir
        def fail_common(path,*args,**kwargs):
            if path == common:
                raise OSError('simulated common library mkdir failure')
            return mkdir(path,*args,**kwargs)
        with mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                mock.patch.object(ninjars_install,'_run',side_effect=self.fake_build):
            with mock.patch.object(Path,'mkdir',autospec=True,side_effect=fail_common):
                with self.assertRaisesRegex(OSError,'common library mkdir failure'):
                    ninjars_install.install(self.root,self.prefix,['--force'])
            self.assertEqual(before,self.snapshot(previous))
            self.assertFalse(common.exists())
            libdir = ninjars_install.install(self.root,self.prefix,['--force'])
        self.assertEqual((Path(libdir)/'libninjars.a').read_text(),'new working archive')

    def test_post_commit_cleanup_failure_keeps_success_and_recovery_diagnostic(self):
        self.previous_install()
        with mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                mock.patch.object(ninjars_install,'_run',side_effect=self.fake_build), \
                mock.patch.object(ninjars_install.shutil,'rmtree',side_effect=OSError('cleanup denied')):
            with self.assertLogs('madgraph.ninjars_install',level='WARNING') as messages:
                libdir = ninjars_install.install(self.root,self.prefix,['--force'])
        self.assertTrue(ninjars_install.supports_quad_precision(libdir))
        self.assertEqual((self.prefix/'lib/libninjars.a').read_text(),'new working archive')
        self.assertTrue(list(self.prefix.glob('.ninjars-build-*/previous/lib/libninjars.a')))
        self.assertIn('installed successfully','\n'.join(messages.output))
        self.assertIn('cleanup denied','\n'.join(messages.output))

    def test_rollback_failure_retains_recovery_files_and_raises_clear_error(self):
        previous = self.previous_install()
        common = self.prefix/'lib'
        common.mkdir()
        (common/'libninjars.a').symlink_to('../ninjars/lib/libninjars.a')
        (common/'mninjars.mod').symlink_to('../custom/previous.mod')
        replace = ninjars_install.os.replace
        def fail_publication_and_module_restore(source,target):
            if Path(target) == common/ninjars_install.CONFIG_NAME:
                raise OSError('simulated alias publication failure')
            if Path(target) == common/'mninjars.mod' and Path(source).parent.name == 'previous-links':
                raise OSError('simulated module restore failure')
            return replace(source,target)
        with mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                mock.patch.object(ninjars_install,'_run',side_effect=self.fake_build), \
                mock.patch.object(ninjars_install.os,'replace',side_effect=fail_publication_and_module_restore):
            with self.assertRaisesRegex(MadGraph5Error,'rollback.*retained'):
                ninjars_install.install(self.root,self.prefix,['--force'])
        self.assertEqual((previous/'lib/libninjars.a').read_text(),'previous working archive')
        recovery = list(common.glob('.ninjars-links-*/previous-links/mninjars.mod'))
        self.assertEqual(len(recovery),1)
        self.assertEqual(os.readlink(recovery[0]),'../custom/previous.mod')

    def test_symlinked_common_directory_keeps_user_files_and_working_aliases(self):
        shared = self.root/'shared-libraries'
        shared.mkdir()
        self.prefix.mkdir()
        common = self.prefix/'lib'
        common.symlink_to('../shared-libraries')
        (shared/'ninjars-native-libs.txt').write_text('user-owned link flags')
        sentinel = shared/'.libninjars.a.ninjars'
        sentinel.write_text('user-owned hidden file')
        with mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                mock.patch.object(ninjars_install,'_run',side_effect=self.fake_build):
            libdir = Path(ninjars_install.install(self.root,self.prefix))
        self.assertEqual(os.readlink(common),'../shared-libraries')
        self.assertEqual((common/'libninjars.a').resolve(),libdir/'libninjars.a')
        self.assertEqual((common/'libninjars.a').read_text(),'new working archive')
        self.assertEqual((shared/'ninjars-native-libs.txt').read_text(),'user-owned link flags')
        self.assertEqual(sentinel.read_text(),'user-owned hidden file')
        self.assertFalse(list(shared.glob('.ninjars-links-*')))

    def test_interrupted_alias_publication_restores_previous_state(self):
        previous = self.previous_install()
        common = self.prefix/'lib'
        common.mkdir()
        (common/'libninjars.a').symlink_to('../ninjars/lib/libninjars.a')
        (common/'mninjars.mod').symlink_to('../custom/old-module.mod')
        before_tree, before_aliases = self.snapshot(previous), self.snapshot(common)
        replace = ninjars_install.os.replace
        for interrupt_at in (previous,common/ninjars_install.CONFIG_NAME):
            interrupted = []
            def interrupt_after_replace(source,target):
                result = replace(source,target)
                if Path(target) == interrupt_at and not interrupted:
                    interrupted.append(True)
                    raise KeyboardInterrupt('interrupted after atomic publication')
                return result
            with self.subTest(interrupt_at=interrupt_at), \
                    mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                    mock.patch.object(ninjars_install,'_run',side_effect=self.fake_build), \
                    mock.patch.object(ninjars_install.os,'replace',side_effect=interrupt_after_replace):
                with self.assertRaises(KeyboardInterrupt):
                    ninjars_install.install(self.root,self.prefix,['--force'])
            self.assertEqual(interrupted,[True])
            self.assertEqual(before_tree,self.snapshot(previous))
            self.assertEqual(before_aliases,self.snapshot(common))

    def test_local_sources_are_copied_and_original_is_unchanged(self):
        source = self.root / 'reference'
        for relative in ('Cargo.toml', 'crates/ninjars/Cargo.toml',
                         'crates/ninjars-symbolica/Cargo.toml'):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# original\n')
        (source / 'target').mkdir()
        (source / 'target' / 'cached.bin').write_text('do not copy')
        (source / '.git').mkdir()
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
        self.assertEqual(provenance['ninjars']['source_files'], 3)
        original_hash = provenance['ninjars']['source_tree_sha256']
        other = self.root/'other-copy'
        with mock.patch.object(ninjars_install.subprocess, 'check_output',
                               return_value='1234abcd\n'):
            copied_provenance = ninjars_install._copy_source(source, other, ['Cargo.toml'])
        self.assertEqual(copied_provenance['source_tree_sha256'], original_hash)
        (source/'Cargo.toml').write_text('# same revision, modified source\n')
        with mock.patch.object(ninjars_install.subprocess, 'check_output',
                               return_value='1234abcd\n'):
            modified = ninjars_install._copy_source(source, self.root/'modified', ['Cargo.toml'])
        self.assertNotEqual(modified['source_tree_sha256'], original_hash)

    def test_source_tarball_never_inherits_enclosing_repository_revision(self):
        source = self.root/'unpacked-dependency'
        source.mkdir()
        (source/'Cargo.toml').write_text('[package]\nname = "oneloop"\n')
        with mock.patch.object(ninjars_install.subprocess, 'check_output',
                               return_value='a'*40+'\n') as git:
            provenance = ninjars_install._copy_source(
                source, self.root/'copied-tarball', ['Cargo.toml'])
        git.assert_not_called()
        self.assertIsNone(provenance['revision'])
        self.assertEqual(provenance['source_files'], 1)
        self.assertEqual(len(provenance['source_tree_sha256']), 64)

    def test_invalid_resolved_lock_never_replaces_previous_install(self):
        previous = self.previous_install()
        before = self.snapshot(previous)
        (self.bridge/'Cargo.lock').write_text('invalid resolved lock\n')
        with mock.patch.object(ninjars_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(ninjars_install, '_run', side_effect=self.fake_build):
            with self.assertRaisesRegex(MadGraph5Error, 'Invalid resolved NinjaRS Cargo.lock'):
                ninjars_install.install(self.root, self.prefix, ['--force'])
        self.assertEqual(self.snapshot(previous), before)

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
        for version in (1, 2, 99):
            config.write_text(json.dumps({'abi_version': version, 'native_libs': []}))
            with self.subTest(version=version), self.assertRaisesRegex(
                    MadGraph5Error, 'ABI'):
                ninjars_install.read_config(lib)
        config.write_text(json.dumps({'abi_version': ninjars_install.ABI_VERSION,
                                     'native_libs': 'bad'}))
        with self.assertRaisesRegex(MadGraph5Error, 'link metadata'):
            ninjars_install.read_config(lib)
        config.write_text(json.dumps({'abi_version': ninjars_install.ABI_VERSION,
                                     'native_libs': []}))
        with self.assertRaisesRegex(MadGraph5Error, 'compiler-runtime.*metadata'):
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
            MadGraphCmd.do_install(cmd, 'ninjars --update --jobs=3')
        self.assertEqual(cmd.options['ninjars'], '/installed/lib')
        self.assertEqual(install.call_args.kwargs['arguments'], ['--update', '--jobs=3'])
        cmd.exec_cmd.assert_called_once_with('save options config.txt ninjars',
                                            printcmd=False, log=False)
        cmd.exec_cmd.reset_mock()
        with mock.patch.object(ninjars_install, 'install',
                               side_effect=MadGraph5Error('failed')):
            with self.assertRaises(MadGraph5Error):
                MadGraphCmd.do_install(cmd, 'ninjars --update')
        cmd.exec_cmd.assert_not_called()

    def test_exporter_skips_incomplete_optional_library_and_copies_valid_install(self):
        lib = self.root / 'lib'
        lib.mkdir()
        (lib / 'libninjars.a').write_bytes(b'!<arch>\nsynthetic archive metadata')
        (lib / ninjars_install.CONFIG_NAME).write_text(json.dumps(
            {'abi_version': ninjars_install.ABI_VERSION, 'native_libs': ['-lgmp'],
             'link_before': ['-lgcc_s'], 'link_order_version': 1,
             'quad_precision': True}))
        exporter = object.__new__(LoopProcessOptimizedExporterFortranSA)
        exporter.tir_available_dict = {'ninjars': True}
        exporter.dependencies = 'external'
        (lib / 'ninjars-native-libs.txt').write_text('-lgmp\n')
        (lib / 'ninjars-link-before.txt').write_text('-lgcc_s\n')
        with self.assertLogs('madgraph.loop_exporter', level='WARNING') as logged:
            self.assertEqual(exporter.link_TIR(str(self.root), str(lib), 'libninjars.a', 'ninjars'), '')
        self.assertFalse(exporter.tir_available_dict['ninjars'])
        self.assertIn('install ninjars --update', '\n'.join(logged.output))
        (lib / 'mninjars.mod').write_text('module')
        exporter.tir_available_dict['ninjars'] = True
        self.assertEqual(exporter.link_TIR(
            str(self.root), str(lib), 'libninjars.a', 'ninjars'), str(lib))
        exporter.dependencies = 'internal'
        (lib / 'ninjars-native-libs.txt').write_text('-lgmp\n')
        (lib / 'ninjars-link-before.txt').write_text('-lgcc_s\n')
        include = self.root / 'include'
        include.mkdir()
        (include / 'ninjars.h').write_text('/* header */\n')
        (include / 'mninjars.f90').write_text('module mninjars\nend module\n')
        copied = self.root / 'export' / 'lib'
        self.assertEqual(exporter.link_TIR(
            str(copied), str(lib), 'libninjars.a', 'ninjars'), str(copied))
        shutil.rmtree(lib)
        shutil.rmtree(include)
        self.assertTrue(ninjars_install.supports_quad_precision(copied))
        for name in ('libninjars.a', 'mninjars.mod', 'ninjars-config.json',
                     'ninjars-native-libs.txt', 'ninjars-link-before.txt',
                     'ninjars.h', 'mninjars.f90'):
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
