"""Temporary-file preparation tests; every external process is forbidden."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


NO_SPAWN = patch.object(subprocess, 'Popen', side_effect=AssertionError('External processes forbidden'))
NO_SPAWN.start()
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cms_preparation_under_test', HERE / 'run.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
P = R.prerequisites


def tearDownModule():
    NO_SPAWN.stop()


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


class TemporaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)


class EntrypointTests(TemporaryTests):
    def setUp(self):
        super().setUp()
        self.mg = self.root / 'mg'
        self.report = self.root / 'report'
        self.report.mkdir()
        self.entry = write(self.mg / 'bin/madgraph', (R.ROOT / 'bin/madgraph').read_text())
        self.cli = write(self.root / 'commands/candidate.mg5', 'quit\n')
        self.receipt = self.report / 'materialized-candidate.json'
        self.record = {'revision': R.PLAN['candidate_revision'],
                       'files': {'bin/madgraph': {'kind': 'file', 'sha256': checksum(self.entry)}}}
        self.save_receipt()
        self.stack.enter_context(patch.object(R, 'REPORT', self.report))

    def save_receipt(self):
        self.receipt.write_text(json.dumps(self.record))

    def test_exact_tracked_entrypoint_and_plain_bytecode_disabled_argv(self):
        result = R.cli_admission(self.mg, 'candidate', self.cli)
        self.assertEqual(result['entrypoint'], 'bin/madgraph')
        self.assertEqual(result['sha256'], checksum(self.entry))
        self.assertEqual(result['materialized_receipt_sha256'], checksum(self.receipt))
        self.assertEqual(result['argv'], [sys.executable, '-B', './bin/madgraph', '--plain', str(self.cli)])

    def test_missing_entrypoint_does_not_fall_back_to_mg5_alias(self):
        self.entry.unlink()
        write(self.mg / 'bin/mg5_aMC', '# unavailable spelling is not a fallback\n')
        with self.assertRaisesRegex(RuntimeError, 'Tracked CLI entrypoint missing'):
            R.cli_admission(self.mg, 'candidate', self.cli)

    def test_rewritten_materialized_receipt_cannot_replace_frozen_entrypoint_bytes(self):
        self.entry.write_text('# modified CLI and matching local receipt\n')
        self.record['files']['bin/madgraph']['sha256'] = checksum(self.entry)
        self.save_receipt()
        with self.assertRaisesRegex(RuntimeError, 'Tracked CLI entrypoint bytes differ'):
            R.cli_admission(self.mg, 'candidate', self.cli)

    def test_untracked_changed_symlink_or_wrong_revision_entrypoint_is_denied(self):
        original = self.entry.read_text()
        self.entry.write_text('changed\n')
        with self.assertRaisesRegex(RuntimeError, 'bytes differ'):
            R.cli_admission(self.mg, 'candidate', self.cli)
        self.entry.unlink()
        donor = write(self.root / 'same-byte-entry', original)
        self.entry.symlink_to(donor)
        with self.assertRaisesRegex(RuntimeError, 'entrypoint missing'):
            R.cli_admission(self.mg, 'candidate', self.cli)
        self.entry.unlink(); self.entry.write_text(original)
        self.record['files'] = {}; self.save_receipt()
        with self.assertRaisesRegex(RuntimeError, 'entrypoint missing'):
            R.cli_admission(self.mg, 'candidate', self.cli)
        self.record['revision'] = 'incorrect'; self.save_receipt()
        with self.assertRaisesRegex(RuntimeError, 'revision mismatch'):
            R.cli_admission(self.mg, 'candidate', self.cli)

    def test_only_output_parent_is_created(self):
        output = self.root / 'new/exports/candidate/cms'
        R.prepare_output_parent(output)
        self.assertTrue(output.parent.is_dir())
        self.assertFalse(output.exists())

    def test_existing_output_file_directory_and_dangling_symlink_are_denied(self):
        for kind in ('file', 'directory', 'dangling-symlink'):
            with self.subTest(kind=kind):
                output = self.root / ('existing-' + kind)
                if kind == 'file': output.write_text('keep')
                elif kind == 'directory': output.mkdir()
                else: output.symlink_to(self.root / 'absent')
                with self.assertRaisesRegex(RuntimeError, 'Fresh CLI output required'):
                    R.prepare_output_parent(output)


class PrerequisiteTests(TemporaryTests):
    def setUp(self):
        super().setUp()
        module = write(self.root / 'external/mninja.mod', 'external module\n')
        helper = write(self.root / 'external/ninja-config', '#!/bin/sh\nexit 0\n')
        helper.chmod(0o755)
        self.external_module, self.external_helper = module, helper
        self.spec = {'CT_version': '15.2.0', 'external': {}, 'sources': {}}
        for path in (module, helper):
            self.spec['external'][str(path)] = {
                'resolved': str(path.resolve()), 'sha256': checksum(path),
                'size': path.stat().st_size, 'executable': path == helper}
        for name, abi in (('ninjars', 4), ('tirust', 1)):
            lib = self.root / ('external/' + name)
            artifact = write(lib / ('lib' + name + '.a'), 'bound bridge archive\n')
            config = {'abi_version': abi, 'quad_precision': True, 'link_order_version': 1,
                      'link_before': ['-lgfortran'], 'native_libs': ['-ldl'],
                      'artifact_sha256': {artifact.name: checksum(artifact)}}
            paths = [artifact, write(lib / (name + '-config.json'), json.dumps(config)),
                     write(lib / (name + '-link-before.txt'), '-lgfortran\n'),
                     write(lib / (name + '-native-libs.txt'), '-ldl\n')]
            for path in paths:
                self.spec['external'][str(path)] = {
                    'resolved': str(path.resolve()), 'sha256': checksum(path),
                    'size': path.stat().st_size, 'executable': False}
        shared = self.root / 'shared-cuttools'
        write(shared / 'includects/libcts.a', 'sealed CT archive\n')
        write(shared / 'includects/mpmodule.mod', 'sealed CT module\n')
        write(shared / 'includects/compiler_version.log', '15.2.0\n')
        for source in ('baseline', 'candidate'):
            donor = self.root / ('donor-' + source)
            (donor / 'vendor').mkdir(parents=True)
            (donor / 'vendor/CutTools').symlink_to(shared, target_is_directory=True)
            write(donor / 'vendor/IREGI/src/libiregi.a', source + ' IREGI archive\n')
            names = ['vendor/CutTools/includects/' + name for name in
                     ('libcts.a', 'mpmodule.mod', 'compiler_version.log')]
            names.append('vendor/IREGI/src/libiregi.a')
            self.spec['sources'][source] = {'required': names, 'copies': {
                name: {'donor': str(donor / name),
                       'resolved_donor': str((donor / name).resolve()),
                       'sha256': checksum(donor / name),
                       'size': (donor / name).stat().st_size} for name in names}}
        self.stack.enter_context(patch.object(P, 'DATA', self.spec))

    def donor(self, source, name):
        return Path(self.spec['sources'][source]['copies'][name]['donor'])

    def private(self, source='candidate'):
        tree = self.root / ('private-' + source)
        tree.mkdir()
        return tree

    def test_symlink_donors_become_private_regular_files_at_logical_paths(self):
        P.check_donors()
        for source in ('baseline', 'candidate'):
            tree = self.private(source)
            tracked = write(tree / 'tracked-source.f', 'do not change\n')
            receipt = P.copy_private(source, tree)
            self.assertFalse((tree / 'vendor/CutTools').is_symlink())
            self.assertEqual(set(receipt), set(self.spec['sources'][source]['required']))
            for name, item in self.spec['sources'][source]['copies'].items():
                target = tree / name
                self.assertTrue(target.is_file())
                self.assertFalse(target.is_symlink())
                self.assertTrue(target.resolve().is_relative_to(tree.resolve()))
                self.assertEqual(checksum(target), item['sha256'])
            self.assertEqual((tree / 'vendor/IREGI/src/libiregi.a').read_text(), source + ' IREGI archive\n')
            self.assertEqual(tracked.read_text(), 'do not change\n')
            P.check_private(source, tree)

    def test_bridge_config_abi_qp_and_link_metadata_are_checked_independently(self):
        config_path = self.root / 'external/tirust/tirust-config.json'
        original = config_path.read_bytes()
        record = self.spec['external'][str(config_path)]
        for key, value in (('abi_version', 99), ('quad_precision', False),
                           ('link_order_version', 99), ('link_before', [])):
            with self.subTest(key=key):
                config = json.loads(original); config[key] = value
                config_path.write_text(json.dumps(config))
                record.update(sha256=checksum(config_path), size=config_path.stat().st_size)
                with self.assertRaisesRegex(RuntimeError, 'ABI/QP configuration differs|Empty compiler-runtime link metadata'):
                    P.check_donors()
        config_path.write_bytes(original)
        record.update(sha256=checksum(config_path), size=config_path.stat().st_size)
        metadata = config_path.parent / 'tirust-link-before.txt'
        metadata.write_text('-lwrong-runtime\n')
        self.spec['external'][str(metadata)].update(sha256=checksum(metadata), size=metadata.stat().st_size)
        with self.assertRaisesRegex(RuntimeError, 'Bridge link metadata mismatch'):
            P.check_donors()

    def test_missing_required_binding_or_donor_is_rejected(self):
        name = 'vendor/CutTools/includects/mpmodule.mod'
        record = self.spec['sources']['candidate']['copies'].pop(name)
        with self.assertRaisesRegex(RuntimeError, 'Required prebuilt not declared'):
            P.check_donors()
        self.spec['sources']['candidate']['copies'][name] = record
        Path(record['resolved_donor']).unlink()
        with self.assertRaisesRegex(RuntimeError, 'Prebuilt dependency changed/missing'):
            P.copy_private('candidate', self.private())

    def test_stale_library_is_rejected(self):
        path = self.donor('candidate', 'vendor/IREGI/src/libiregi.a')
        path.write_text('stale archive\n')
        with self.assertRaisesRegex(RuntimeError, 'Prebuilt dependency changed/missing'):
            P.check_donors()

    def test_wrong_compiler_log_is_rejected_even_when_its_hash_was_updated(self):
        name = 'vendor/CutTools/includects/compiler_version.log'
        path = self.donor('candidate', name)
        path.write_text('15.2.1\n')
        for source in ('baseline', 'candidate'):
            self.spec['sources'][source]['copies'][name]['sha256'] = checksum(path)
        with self.assertRaisesRegex(RuntimeError, 'CutTools compiler version differs'):
            P.check_donors()

    def test_external_module_helper_or_helper_permission_changes_are_rejected(self):
        for path in (self.external_module, self.external_helper):
            with self.subTest(path=path.name):
                original = path.read_bytes(); path.write_bytes(original + b'changed')
                with self.assertRaises(RuntimeError): P.check_donors()
                path.write_bytes(original)
        self.external_helper.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, 'helper is not executable'):
            P.external_hashes()

    def test_same_byte_external_module_retarget_is_rejected(self):
        other = write(self.root / 'other-module.mod', self.external_module.read_text())
        self.external_module.unlink(); self.external_module.symlink_to(other)
        with self.assertRaisesRegex(RuntimeError, 'Prebuilt dependency target changed'):
            P.external_hashes()

    def test_existing_matching_private_file_is_preserved_but_changed_or_symlink_file_is_denied(self):
        tree = self.private(); P.copy_private('candidate', tree)
        P.copy_private('candidate', tree)
        path = tree / 'vendor/CutTools/includects/libcts.a'
        path.write_bytes(b'stale')
        with self.assertRaisesRegex(RuntimeError, 'Prebuilt would overwrite tracked bytes'):
            P.copy_private('candidate', tree)
        path.unlink(); path.symlink_to(self.donor('candidate', 'vendor/CutTools/includects/libcts.a'))
        with self.assertRaisesRegex(RuntimeError, 'escapes clone|must be a real file'):
            P.copy_private('candidate', tree)

    def test_private_ancestor_escape_is_denied_before_creating_any_outside_directory(self):
        tree = self.private(); outside = self.root / 'outside'; outside.mkdir()
        (tree / 'vendor').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'path escapes clone'):
            P.copy_private('candidate', tree)
        self.assertEqual(list(outside.iterdir()), [])

    def test_deleted_private_archive_module_log_or_source_iregi_blocks_cli(self):
        for name in self.spec['sources']['candidate']['required']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary); tree = work / 'mg-candidate'; tree.mkdir()
                P.copy_private('candidate', tree); (tree / name).unlink()
                command = Mock(side_effect=AssertionError('CLI must not be reached'))
                with patch.object(R, 'WORK', work), patch.object(R, 'registered', return_value={}), \
                        patch.object(R.L, 'Deadline', side_effect=lambda seconds: nullcontext()), patch.object(R, 'cmd', command):
                    with self.assertRaisesRegex(RuntimeError, 'Private prebuilt changed/missing'):
                        R.prepare_process('candidate', R.PLAN['cases'][0])
                command.assert_not_called()

    def test_stale_private_module_blocks_cli(self):
        work = self.root / 'work'; tree = work / 'mg-candidate'; tree.mkdir(parents=True)
        P.copy_private('candidate', tree)
        (tree / 'vendor/CutTools/includects/mpmodule.mod').write_text('stale module\n')
        command = Mock(side_effect=AssertionError('CLI must not be reached'))
        with patch.object(R, 'WORK', work), patch.object(R, 'registered', return_value={}), \
                patch.object(R.L, 'Deadline', side_effect=lambda seconds: nullcontext()), patch.object(R, 'cmd', command):
            with self.assertRaisesRegex(RuntimeError, 'Private prebuilt changed/missing'):
                R.prepare_process('candidate', R.PLAN['cases'][0])
        command.assert_not_called()

    def test_active_compiler_version_is_checked_without_running_a_compiler(self):
        capture = Mock(return_value=' 15.2.0\n')
        R.check_compiler(capture)
        capture.assert_called_once_with(['gfortran', '-dumpfullversion', '-dumpversion'])
        for value in ('15.2.1\n', '', 'gfortran 15.2.0'):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'Active compiler differs'):
                R.check_compiler(Mock(return_value=value))

    def test_compiler_mismatch_stops_before_cli_or_output_creation(self):
        work = self.root / 'work'; tree = work / 'mg-candidate'; tree.mkdir(parents=True)
        P.copy_private('candidate', tree)
        command = Mock(return_value='15.2.1\n')
        with patch.object(R, 'WORK', work), patch.object(R, 'registered', return_value={}), \
                patch.object(R.L, 'Deadline', side_effect=lambda seconds: nullcontext()), patch.object(R, 'cmd', command):
            with self.assertRaisesRegex(RuntimeError, 'Active compiler differs'):
                R.prepare_process('candidate', R.PLAN['cases'][0])
        self.assertEqual(command.call_count, 1)
        self.assertEqual(command.call_args.args[0], ['gfortran', '-dumpfullversion', '-dumpversion'])
        self.assertFalse((work / 'exports').exists())


class GeneratedCapabilitiesTests(TemporaryTests):
    subdirectory = 'SubProcesses/P0_epem_mupmum'

    def generated(self, flags=None, qp=(6, 8, 9), source='candidate'):
        flags = flags or [True, False, True, False, False, True, True, True] + ([True] if source == 'candidate' else [])
        text = '      DATA LOOPLIBS_AVAILABLE /' + ','.join('.TRUE.' if flag else '.FALSE.' for flag in flags) + '/\n'
        text += '      DATA LOOPLIBS_QPAVAILABLE(0:5) /.FALSE.,.TRUE.,4*.FALSE./\n'
        text += ''.join('      DATA LOOPLIBS_QPAVAILABLE(%d) /.TRUE./\n' % backend for backend in qp)
        text += '      DATA LOOPLIBS_QPAVAILABLE(7) /.FALSE./\n'
        write(self.root / 'SubProcesses/MadLoop_makefile_definitions', 'LOOP_INCLUDE = -I/pinned/modules\n')
        return write(self.root / self.subdirectory / 'loop_matrix.f', text)

    def check(self, source='candidate'):
        return P.check_generated(source, self.root, self.subdirectory)

    def test_current_and_baseline_generated_dp_and_qp_availability(self):
        self.generated()
        current = self.check()
        self.assertEqual(current['available'], [True, False, True, False, False, True, True, True, True])
        self.assertEqual(current['qp_available'], {index: index in (1, 6, 8, 9) for index in range(10)})
        self.generated(qp=(6, 8), source='baseline')
        baseline = self.check('baseline')
        self.assertEqual(baseline['available'], [True, False, True, False, False, True, True, True])
        self.assertEqual(baseline['qp_available'], {index: index in (1, 6, 8) for index in range(9)})

    def test_each_requested_dp_backend_must_be_available(self):
        expected = [True, False, True, False, False, True, True, True, True]
        for backend in range(1, 10):
            flags = list(expected); flags[backend - 1] = not flags[backend - 1]; self.generated(flags)
            with self.subTest(backend=backend), self.assertRaisesRegex(RuntimeError, 'Generated reducer availability differs'):
                self.check()

    def test_explicit_required_qp_flags_cannot_be_missing_or_false(self):
        for backend in (6, 8, 9):
            for kind in ('missing', 'false'):
                path = self.generated(qp=tuple(value for value in (6, 8, 9) if value != backend))
                if kind == 'false':
                    with path.open('a') as stream:
                        stream.write('      DATA LOOPLIBS_QPAVAILABLE(%d) /.FALSE./\n' % backend)
                with self.subTest(backend=backend, kind=kind), self.assertRaisesRegex(RuntimeError, 'QP capability differs'):
                    self.check()

    def test_cuttools_initial_qp_block_cannot_be_missing_false_or_wrong_length(self):
        for initial in ('', '.FALSE.,.FALSE.,4*.FALSE.', '.FALSE.,.TRUE.,3*.FALSE.', '.FALSE.,.TRUE.,5*.FALSE.'):
            path = self.generated(); text = path.read_text()
            original = '      DATA LOOPLIBS_QPAVAILABLE(0:5) /.FALSE.,.TRUE.,4*.FALSE./\n'
            replacement = '      DATA LOOPLIBS_QPAVAILABLE(0:5) /' + initial + '/\n' if initial else ''
            path.write_text(text.replace(original, replacement))
            with self.subTest(initial=initial), self.assertRaisesRegex(RuntimeError, 'QP capability differs|Malformed QP'):
                self.check()

    def test_repeat_logical_syntax_and_malformed_or_short_availability(self):
        self.assertEqual(P.logical_values('.TRUE.,4*.FALSE.,4*.TRUE.'), [True] + [False] * 4 + [True] * 4)
        for text in ('true', '.TRUE.,', '2*.BAD.', '.TRUE. garbage'):
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, 'Malformed generated availability'):
                P.logical_values(text)
        self.generated(flags=[True])
        with self.assertRaisesRegex(RuntimeError, 'Generated reducer availability differs'):
            self.check()
        write(self.root / self.subdirectory / 'loop_matrix.f', 'C no availability DATA\n')
        with self.assertRaisesRegex(RuntimeError, 'Missing generated reducer availability'):
            self.check()

    def test_external_definition_missing_or_placeholder_fails_and_internal_dependency_trees_are_denied(self):
        self.generated(); definitions = self.root / 'SubProcesses/MadLoop_makefile_definitions'
        for text in (None, 'LOOP_INCLUDE = <Not_found_define_it_yourself>\n'):
            if text is None: definitions.unlink()
            else: definitions.write_text(text)
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, 'external include directory'):
                self.check()
        for name in ('CutTools', 'IREGI'):
            self.generated(); path = self.root / 'Source' / name; path.mkdir(parents=True)
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'internal dependency tree'):
                self.check()
            path.rmdir()


class EnvironmentAndDefaultTests(TemporaryTests):
    def test_private_state_pwd_loader_override_removal_and_one_thread_environment(self):
        base = {'PATH': '/pinned/bin', 'HOME': '/unchanged/home', 'LD_PRELOAD': 'remove',
                'LD_LIBRARY_PATH': 'remove', 'LD_AUDIT': 'remove', 'PYTHONPATH': 'remove'}
        bash = write(self.root / 'bin/bash', '# shell fixture only\n')
        with patch.object(R.q, 'env', side_effect=lambda: copy.deepcopy(base)), patch.object(R.shutil, 'which', return_value=str(bash)):
            for stage, jobs in (('prepare', '-j4'), ('preflight', '-j1'), ('run', '-j1')):
                with self.subTest(stage=stage), patch.object(R.L, 'CURRENT_STAGE', stage):
                    env = R.environment(self.root / 'actual-child-cwd')
                    self.assertEqual(env['PWD'], str(self.root / 'actual-child-cwd'))
                    self.assertEqual(env['HOME'], '/unchanged/home')
                    self.assertEqual(env['MAKEFLAGS'], jobs + ' SHELL=' + str(bash.resolve()))
                    self.assertEqual(env['XDG_STATE_HOME'], R.PLAN['environment']['set']['XDG_STATE_HOME'])
                    for name in ('XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'MPLCONFIGDIR'):
                        self.assertTrue(Path(env[name]).is_relative_to(R.WORK))
                    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE'):
                        self.assertEqual(env[name], '1')
                    self.assertEqual(env['PYTHONHASHSEED'], '0')
                    for name in R.PLAN['environment']['remove']: self.assertNotIn(name, env)

    def test_default_or_stage_without_execute_is_source_only_and_never_seals_or_launches(self):
        for argv in (['run.py'], ['run.py', 'prepare'], ['run.py', 'preflight'], ['run.py', 'run']):
            stream = io.StringIO()
            with self.subTest(argv=argv), patch.object(sys, 'argv', argv), patch.object(Path, 'cwd', return_value=R.ROOT), \
                    patch.object(R, 'seal', side_effect=AssertionError('Must not seal by default')), \
                    patch.object(R, 'cmd', side_effect=AssertionError('Must not launch by default')), redirect_stdout(stream):
                R.main()
            result = json.loads(stream.getvalue())
            self.assertTrue(result['source_only'])
            self.assertEqual(result['commands_executed'], 0)
            self.assertEqual(result['requests'], 6)


if __name__ == '__main__':
    unittest.main()
