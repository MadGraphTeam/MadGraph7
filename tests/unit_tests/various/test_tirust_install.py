"""TIRust publication, source isolation, CLI dispatch and partial-result guards."""

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph import MadGraph5Error
from madgraph.interface.madgraph_interface import MadGraphCmd
from madgraph.various import banner, rust_dependency_notices, tirust_install


class TestTIRustInstaller(unittest.TestCase):
    def setUp(self):
        notice_patch = mock.patch.object(rust_dependency_notices, 'package', side_effect=self.fake_notices)
        notice_patch.start()
        self.addCleanup(notice_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / 'HEPTools'
        self.bridge = self.root / 'vendor/tirust'
        self.bridge.mkdir(parents=True)
        (self.bridge / 'Cargo.toml').write_text('[package]\nname="bridge"\n')
        (self.bridge / 'Cargo.lock').write_text('''version = 4
[[package]]
name = "oneloop"
version = "0.1.0"
source = "git+https://github.com/alphal00p/oneloopmaster.git?rev=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
''')
        (self.bridge / 'smoke.f90').write_text('end\n')
        (self.bridge / 'patches').mkdir()
        (self.bridge / 'patches/0001-direct-poles-only.patch').write_text('fixture patch\n')
        (self.bridge / 'pole-kernels').mkdir()
        (self.bridge / 'pole-kernels/Cargo.toml').write_text('[package]\nname="madloop-pole-kernels"\n')
        self.source = self.root / 'reference-tirust'
        self.make_source(self.source)
        (self.bridge / 'patches/manifest.json').write_text(json.dumps({
            'schema_version': 1, 'upstream_revision': tirust_install.UPSTREAM_REVISION,
            'patch_sha256': hashlib.sha256(b'fixture patch\n').hexdigest(),
            'source_sha256': {'Cargo.toml': hashlib.sha256(b'# immutable reference\n').hexdigest()}}))
        (self.bridge / 'pole-kernels/src').mkdir()
        for filename in ['lib.rs', 'generated_manifest.txt'] + [
                'generated_%s.rs' % family for family in ('a0', 'b0', 'c0', 'd0', 'constants')]:
            (self.bridge / 'pole-kernels/src' / filename).write_text('fixture generated source\n')
        self.arguments = ['--source', str(self.source)]

    @staticmethod
    def fake_notices(bridge, libdir, backend, environment):
        names = [backend + suffix for suffix in (
            '-dependency-notices.tar.gz', '-dependency-provenance.json', '-NOTICES.txt')]
        for name in names:
            (Path(libdir) / name).write_text('fixture dependency notice\n')
        return {'files': names, 'sha256': {name: hashlib.sha256(
            (Path(libdir) / name).read_bytes()).hexdigest() for name in names}, 'package_count': 1}

    @staticmethod
    def make_source(source):
        for filename in ('Cargo.toml', 'crates/iregi-rust/Cargo.toml', 'crates/iregi-masters/Cargo.toml'):
            path = source / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# immutable reference\n')

    @staticmethod
    def snapshot(path):
        return {str(p.relative_to(path)): (('link', os.readlink(p)) if p.is_symlink()
                else ('file', p.read_bytes())) for p in path.rglob('*') if p.is_symlink() or p.is_file()}

    def fake_run(self, command, cwd, log, environment):
        log.write_text('fixture compiler log\n')
        if command[:2] == ['git', 'checkout']:
            self.make_source(cwd)
        if command[0] == 'make':
            lib = Path(next(x[7:] for x in command if x.startswith('PREFIX='))) / 'lib'
            lib.mkdir(parents=True)
            (lib / 'libtirust.a').write_bytes(b'!<arch>\nfixture archive')
            (lib / 'libmadloop_tirust.so').write_bytes(b'fixture isolated runtime')
            (lib / 'mtirust.mod').write_text('fixture module')
            (lib / 'tirust-native-libs.txt').write_text('-lmadloop_tirust\n')
            (lib / 'tirust-link-before.txt').write_text('-lgcc_s\n')

    def install(self, arguments=None):
        return Path(tirust_install.install(self.root, self.prefix,
            self.arguments if arguments is None else arguments, jobs=2))

    def test_publication_idempotence_metadata_and_smoke_link(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
            lib = self.install()
            config = tirust_install.read_config(lib)
            self.assertTrue(tirust_install.is_available(lib))
            self.assertTrue(tirust_install.supports_quad_precision(lib))
            self.assertIn('libmadloop_tirust.so', tirust_install.export_files(lib))
            self.assertIn('libtirust.a', tirust_install.export_files(lib))
            self.assertIn('tirust-config.json', tirust_install.export_files(lib))
            self.assertIn('tirust-NOTICES.txt', tirust_install.export_files(lib))
            self.assertEqual(config['native_libs'], ['-lmadloop_tirust'])
            self.assertEqual(config['local_sources']['tirust']['patch_sha256'], hashlib.sha256(b'fixture patch\n').hexdigest())
            self.assertTrue((self.prefix / 'lib/libmadloop_tirust.so').is_symlink())
            smoke = next(c.args[0] for c in run.call_args_list if c.args[0][0] == 'gfortran')
            self.assertTrue(any(arg.startswith('-Wl,-rpath,') for arg in smoke))
            self.assertIn('-lmadloop_tirust', smoke)
            count = run.call_count
            self.assertEqual(self.install(), lib)
            self.assertEqual(run.call_count, count)

    def test_default_source_fetch_is_pinned_and_then_patched(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
            self.install([])
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(['git', 'fetch', '--depth=1', tirust_install.UPSTREAM_URL,
                       tirust_install.UPSTREAM_REVISION], commands)
        self.assertEqual(sum(command[0] == 'patch' for command in commands), 1)

    def test_local_overrides_never_change_reference_sources(self):
        oneloop = self.root / 'reference-oneloop'
        oneloop.mkdir()
        (oneloop / 'Cargo.toml').write_text('# scalar source\n')
        before = self.snapshot(self.source), self.snapshot(oneloop)
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
            self.install(self.arguments + ['--oneloop-source', str(oneloop)])
            metadata_call = next(c for c in run.call_args_list if c.args[0][:2] == ['cargo', 'metadata'])
            self.assertEqual(metadata_call.args[3]['CARGO_BUILD_JOBS'], '2')
        self.assertEqual(before, (self.snapshot(self.source), self.snapshot(oneloop)))

    def test_failed_smoke_update_preserves_installation_and_aliases(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run):
            self.install()
        before = self.snapshot(self.prefix)
        def fail(command, *args):
            if command[0] == 'gfortran':
                raise MadGraph5Error('smoke link failed')
            self.fake_run(command, *args)
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=fail):
            with self.assertRaisesRegex(MadGraph5Error, 'smoke link failed'):
                self.install(self.arguments + ['--update'])
        for name, value in before.items():
            self.assertEqual(self.snapshot(self.prefix)[name], value)

    def test_alias_failure_rolls_back_shared_runtime(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run):
            self.install()
        previous = self.snapshot(self.prefix / 'tirust')
        aliases = self.snapshot(self.prefix / 'lib')
        original = os.replace
        def fail(source, target):
            if Path(target) == self.prefix / 'lib/libmadloop_tirust.so' and Path(source).parent.name == 'links':
                raise OSError('shared alias publication failed')
            original(source, target)
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run), \
                mock.patch.object(tirust_install.os, 'replace', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'shared alias'):
                self.install(self.arguments + ['--update'])
        self.assertEqual(previous, self.snapshot(self.prefix / 'tirust'))
        self.assertEqual(aliases, self.snapshot(self.prefix / 'lib'))

    def test_incomplete_shared_runtime_is_unavailable(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run):
            lib = self.install()
        (lib / 'libmadloop_tirust.so').unlink()
        self.assertFalse(tirust_install.is_available(lib))
        with self.assertRaisesRegex(MadGraph5Error, 'Incomplete'):
            self.install()

    def test_missing_patch_fails_before_build(self):
        (self.bridge / 'patches/0001-direct-poles-only.patch').unlink()
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
            with self.assertRaisesRegex(MadGraph5Error, 'poles-only patch'):
                self.install()
        self.assertFalse(any(c.args[0][0] == 'make' for c in run.call_args_list))

    def test_internal_export_copies_complete_shared_runtime(self):
        from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run):
            lib = self.install()
        exporter = LoopProcessOptimizedExporterFortranSA.__new__(LoopProcessOptimizedExporterFortranSA)
        exporter.dependencies = 'internal'
        exporter.tir_available_dict = {'tirust': True}
        copied = self.root / 'export/lib'
        self.assertEqual(exporter.link_TIR(str(copied), str(lib), 'libtirust.a', 'tirust'), str(copied))
        self.assertTrue(tirust_install.is_available(copied))
        for artifact in tirust_install.export_files(lib):
            self.assertFalse((copied / Path(artifact).name).is_symlink())
        tirust_install.shutil.rmtree(self.prefix)
        self.assertTrue(tirust_install.is_available(copied))

    def test_corrupt_existing_archive_is_not_reported_as_installed(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run):
            lib = self.install()
        (lib / 'libtirust.a').write_bytes(b'broken')
        with self.assertRaisesRegex(MadGraph5Error, 'Incomplete'):
            self.install()

    def test_replaced_shared_runtime_loses_smoke_qualification(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run):
            lib = self.install()
        (lib / 'libmadloop_tirust.so').write_bytes(b'another nonempty shared runtime')
        self.assertFalse(tirust_install.is_available(lib))
        self.assertFalse(tirust_install.supports_quad_precision(lib))

    def assert_incompatible_metadata_rejected(self, cases):
        """Exercise the real metadata readers with mocked installation artifacts."""
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
            lib = self.install()
            path = lib / tirust_install.CONFIG_NAME
            original = path.read_bytes()
            config = json.loads(original)
            calls = run.call_count
            for field, value, message in cases:
                with self.subTest(field=field, value=value):
                    path.write_text(json.dumps({**config, field: value}))
                    try:
                        with self.assertRaisesRegex(MadGraph5Error, message):
                            tirust_install.read_config(lib)
                        self.assertFalse(tirust_install.is_available(lib))
                        self.assertFalse(tirust_install.supports_quad_precision(lib))
                        self.assertEqual(run.call_count, calls)
                    finally:
                        path.write_bytes(original)
            self.assertTrue(tirust_install.is_available(lib))
            self.assertTrue(tirust_install.supports_quad_precision(lib))

    def test_incompatible_abi_metadata_is_unavailable(self):
        self.assert_incompatible_metadata_rejected([
            ('abi_version', value, 'Unsupported TIRust bridge ABI')
            for value in (None, 0, tirust_install.ABI_VERSION + 1,
                          str(tirust_install.ABI_VERSION), True,
                          float(tirust_install.ABI_VERSION))])

    def test_invalid_native_link_metadata_is_unavailable(self):
        self.assert_incompatible_metadata_rejected([
            ('native_libs', value, 'Invalid TIRust native link metadata')
            for value in (None, '-lmadloop_tirust', [1],
                          ['-lmadloop_tirust\n-lunexpected'],
                          ['-lmadloop_tirust\r-lunexpected'])])

    def test_invalid_compiler_runtime_link_metadata_is_unavailable(self):
        message = 'Missing or invalid TIRust compiler-runtime link metadata'
        cases = [('link_order_version', value, message)
                 for value in (None, 0, 2, '1', True, 1.0)]
        cases += [('link_before', value, message)
                  for value in (None, '-lgcc_s', [], [1],
                                ['-lgcc_s\n-lunexpected'],
                                ['-lgcc_s\r-lunexpected'])]
        self.assert_incompatible_metadata_rejected(cases)

    def test_incompatible_shared_runtime_metadata_is_unavailable(self):
        self.assert_incompatible_metadata_rejected([
            ('runtime_artifacts', value, 'Missing TIRust shared-runtime metadata')
            for value in (None, [], 'libmadloop_tirust.so', ['libother.so'])])

    def test_patch_or_pristine_source_change_is_rejected(self):
        for kind in ('patch', 'source'):
            with self.subTest(kind=kind):
                target = (self.bridge / 'patches/0001-direct-poles-only.patch' if kind == 'patch'
                          else self.source / 'Cargo.toml')
                original = target.read_bytes()
                target.write_bytes(original + b'changed\n')
                with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                        mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
                    with self.assertRaisesRegex(MadGraph5Error, 'pinned TIRust poles-only patch'):
                        self.install()
                self.assertFalse(any(c.args[0][0] == 'make' for c in run.call_args_list))
                target.write_bytes(original)

    def test_install_prefix_inside_either_source_is_rejected_before_writes(self):
        for option in ('--source', '--oneloop-source'):
            with self.subTest(option=option):
                prefix = self.source / 'install'
                with self.assertRaisesRegex(MadGraph5Error, 'inside a source checkout'):
                    self.install([option, str(self.source), '--prefix', str(prefix)])
                self.assertFalse(prefix.exists())
        with self.assertRaisesRegex(MadGraph5Error, 'inside its source checkout'):
            tirust_install._copy_source(self.source, self.source / 'copy', ['Cargo.toml'])
        self.assertFalse((self.source / 'copy').exists())

    def test_update_cannot_replace_an_original_source_checkout(self):
        for option in ('--source', '--oneloop-source'):
            with self.subTest(option=option):
                source = self.prefix / 'tirust/source'
                self.make_source(source)
                before = self.snapshot(source)
                with self.assertRaisesRegex(MadGraph5Error, 'replace a source checkout'):
                    self.install([option, str(source), '--update'])
                self.assertEqual(self.snapshot(source), before)

    def test_invalid_jobs_and_missing_tool_fail_without_publication(self):
        with self.assertRaisesRegex(MadGraph5Error, 'positive integer'):
            self.install(self.arguments + ['--jobs=0'])
        with mock.patch.object(tirust_install.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(MadGraph5Error, 'requires cargo'):
                self.install()
        self.assertFalse((self.prefix / 'tirust').exists())

    def test_cli_dispatch_saves_only_successful_install(self):
        cmd = SimpleNamespace(options={'heptools_install_dir':str(self.prefix),
            'fortran_compiler':'gfortran', 'nb_core':2}, split_arg=lambda value:value.split(),
            check_install=lambda args:{'options_for_HEPToolsInstaller':args[1:]},
            heptools_install_target=lambda prefix:(prefix, 'config.txt'), exec_cmd=mock.Mock())
        def installed(*args, **kwargs):
            kwargs['on_publish']('/installed/tirust/lib')
            return '/installed/tirust/lib'
        with mock.patch.object(tirust_install, 'install', side_effect=installed) as install, \
                mock.patch.object(tirust_install, 'save_configuration') as save:
            MadGraphCmd.do_install(cmd, 'tirust --update --jobs=3')
        self.assertEqual(cmd.options['tirust'], '/installed/tirust/lib')
        self.assertEqual(install.call_args.kwargs['arguments'], ['--update', '--jobs=3'])
        save.assert_called_once_with(cmd, 'config.txt', '/installed/tirust/lib')
        with mock.patch.object(tirust_install, 'install', side_effect=MadGraph5Error('failed')):
            with self.assertRaises(MadGraph5Error):
                MadGraphCmd.do_install(cmd, 'tirust --update')
        cmd.exec_cmd.assert_not_called()

    def test_cli_configuration_failure_preserves_previous_session_option(self):
        cmd = SimpleNamespace(options={'heptools_install_dir': str(self.prefix),
            'fortran_compiler': 'gfortran', 'nb_core': 2, 'tirust': '/previous/lib'},
            split_arg=lambda value: value.split(),
            check_install=lambda args: {'options_for_HEPToolsInstaller': args[1:]},
            heptools_install_target=lambda prefix: (prefix, 'config.txt'),
            exec_cmd=mock.Mock(side_effect=OSError('read-only config')))
        def installed(*args, **kwargs):
            kwargs['on_publish']('/new/tirust/lib')
            return '/new/tirust/lib'
        with mock.patch.object(tirust_install, 'install', side_effect=installed), \
                mock.patch.object(tirust_install, 'save_configuration', side_effect=OSError('read-only config')):
            with self.assertRaisesRegex(MadGraph5Error, 'configuration could not be saved'):
                MadGraphCmd.do_install(cmd, 'tirust --update')
        self.assertEqual(cmd.options['tirust'], '/previous/lib')

    def test_cli_configuration_failure_restores_installation_aliases_and_configuration(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run):
            self.install()
        previous = self.prefix / 'tirust'
        (previous / 'previous-installation-marker').write_text('must survive failed update\n')
        installed_before = self.snapshot(previous)
        aliases_before = self.snapshot(self.prefix / 'lib')
        config = self.root / 'configuration.txt'
        config.write_text('tirust = /previous/configured/lib\nother = preserved\n')
        config_before = config.read_bytes()
        def fail_config_write(path, *args):
            # The new candidate has passed smoke and is visible when saving.
            self.assertTrue(tirust_install.is_available(previous / 'lib'))
            self.assertFalse((previous / 'previous-installation-marker').exists())
            self.assertTrue(list(self.prefix.glob('.tirust-build-*/previous')))
            Path(path).write_text('partial new configuration')
            raise OSError('simulated configuration write failure')
        cmd = SimpleNamespace(options={'heptools_install_dir': str(self.prefix),
            'fortran_compiler': 'gfortran', 'nb_core': 2, 'tirust': '/previous/configured/lib'},
            split_arg=lambda value: value.split(),
            check_install=lambda args: {'options_for_HEPToolsInstaller': args[1:]},
            heptools_install_target=lambda prefix: (prefix, str(config)),
            write_configuration=fail_config_write)
        with mock.patch('madgraph.interface.madgraph_interface.MG5DIR', str(self.root)), \
                mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
            with self.assertRaisesRegex(MadGraph5Error, 'configuration could not be saved'):
                MadGraphCmd.do_install(cmd, 'tirust --source=%s --update' % self.source)
        self.assertTrue(any(Path(call.args[0][0]).name == 'smoke' for call in run.call_args_list))
        self.assertEqual(self.snapshot(previous), installed_before)
        self.assertEqual(self.snapshot(self.prefix / 'lib'), aliases_before)
        self.assertEqual(config.read_bytes(), config_before)
        self.assertEqual(cmd.options['tirust'], '/previous/configured/lib')
        self.assertEqual(list(self.root.glob('.tirust_configuration_*')), [])

    def test_configuration_callback_runs_for_new_and_existing_install(self):
        callback = mock.Mock()
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
            lib = tirust_install.install(self.root, self.prefix, self.arguments, on_publish=callback)
            callback.assert_called_once_with(lib)
            callback.reset_mock()
            count = run.call_count
            self.assertEqual(tirust_install.install(
                self.root, self.prefix, self.arguments, on_publish=callback), lib)
            callback.assert_called_once_with(lib)
            self.assertEqual(run.call_count, count)

    def test_first_install_configuration_failure_removes_candidate_and_aliases(self):
        with mock.patch.object(tirust_install.shutil, 'which', return_value='/bin/tool'), \
                mock.patch.object(tirust_install, '_run', side_effect=self.fake_run) as run:
            with self.assertRaisesRegex(OSError, 'configuration save failed'):
                tirust_install.install(self.root, self.prefix, self.arguments,
                    on_publish=mock.Mock(side_effect=OSError('configuration save failed')))
        self.assertTrue(any(Path(call.args[0][0]).name == 'smoke' for call in run.call_args_list))
        self.assertFalse((self.prefix / 'tirust').exists())
        self.assertFalse((self.prefix / 'lib').exists())

    def test_configuration_failure_does_not_truncate_original(self):
        config = self.root / 'configuration.txt'
        config.write_text('tirust = /previous/lib\nother = preserved\n')
        before = config.read_bytes()
        def fail(path, *args):
            Path(path).write_text('partial write')
            raise OSError('disk write failed')
        command = SimpleNamespace(write_configuration=fail)
        with self.assertRaisesRegex(OSError, 'disk write failed'):
            tirust_install.save_configuration(command, str(config), '/new/lib')
        self.assertEqual(config.read_bytes(), before)
        self.assertEqual(list(self.root.glob('.tirust_configuration_*')), [])

    def test_configuration_publication_preserves_other_settings(self):
        from madgraph.interface.extended_cmd import Cmd
        config = self.root / 'configuration.txt'
        config.write_text('tirust = /previous/lib\nother = preserved\n')
        command = SimpleNamespace(write_configuration=lambda *args: Cmd.write_configuration(None, *args))
        tirust_install.save_configuration(command, str(config), '/new/lib')
        content = config.read_text()
        self.assertIn('tirust = /new/lib', content)
        self.assertIn('other = preserved', content)


class TestTIRustParameters(unittest.TestCase):
    def test_opt_in_defaults_and_card_roundtrip(self):
        card = banner.MadLoopParam()
        self.assertEqual(card['MLReductionLib'], '6|7|1')
        self.assertEqual(card._ID_reduction_tool_map[9], 'TIRust')
        self.assertTrue(card['TIRustComputeFinite'])
        self.assertTrue(card['UseQPIntegrandForTIRust'])
        self.assertEqual(card['TIRustGlobalCache'], -1)
        self.assertEqual(card['TIRustCacheMaxBytes'], 268435456)
        card['MLReductionLib'] = '9'
        card['TIRustCacheMaxBytes'] = 4294967296
        card.validate_tirust()
        card['TIRustComputeFinite'] = False
        card['LoopPoleMode'] = 1
        template = '#MLReductionLib\n9\n#TIRustComputeFinite\n.false.\n#LoopPoleMode\n1\n'
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.dat') as source:
            source.write(template)
            source.flush()
            output = io.StringIO()
            card.write(output, template=source.name)
        reread = banner.MadLoopParam(output.getvalue())
        self.assertFalse(reread['TIRustComputeFinite'])
        self.assertEqual(reread['LoopPoleMode'], 1)

    def test_invalid_partial_and_cache_requests(self):
        for updates, message in [
                ({'TIRustComputeFinite':False, 'LoopPoleMode':3}, 'requires UV'),
                ({'TIRustComputeFinite':False, 'MLReductionLib':'9|7'}, 'without fallback'),
                ({'TIRustGlobalCache':-2}, 'GlobalCache'),
                ({'TIRustCacheMaxBytes':-1}, 'CacheMaxBytes')]:
            with self.subTest(updates=updates):
                card = banner.MadLoopParam()
                card['MLReductionLib'] = '9'
                for name, value in updates.items():
                    card[name] = value
                with self.assertRaisesRegex(MadGraph5Error, message):
                    card.validate_tirust()
        card = banner.MadLoopParam()
        card['MLReductionLib'] = '9'
        card['TIRustComputeFinite'] = False
        with self.assertRaisesRegex(MadGraph5Error, 'production'):
            card.validate_tirust(production=True)
