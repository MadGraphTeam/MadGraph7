"""Pure mocks and temporary files; never read live qualification5 inputs."""
import contextlib
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('stress_suffix_capture_under_test', HERE / 'capture.py')
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='stress-suffix-capture-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def confirmation(self, exit_code=1):
        return {'kind': 'root-reported-tool-terminal-observation', 'session_id': 7248,
                'exit_code': exit_code, 'observed_utc': '2026-11-10T15:10:00Z',
                'source_index_sha256': capture.SOURCE5_INDEX,
                'inherited_checkpoint_sha256': capture.INHERITED_CHECKPOINT}

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(capture.json_bytes(value))

    def terminal_fixture(self, *, exit_code=1):
        report = self.root / 'closed-report'
        report.mkdir()
        checkpoint = report / 'inherited-checkpoint.json'
        checkpoint.write_text('{}\n')
        checksum = capture.file_sha(checkpoint)
        stage = {'stage': 'run', 'status': 'passed' if exit_code == 0 else 'failed',
                 'exception': None if exit_code == 0 else "RuntimeError('preserved')",
                 'source_index_sha256': capture.SOURCE5_INDEX,
                 'inherited_checkpoint_sha256': checksum,
                 'native_execution': True, 'performance_eligible': False}
        inherited = {'stage': 'inherit', 'status': 'passed', 'source_index_sha256': capture.SOURCE5_INDEX,
                     'inherited_checkpoint_sha256': checksum}
        self.write_json(report / 'stage-run.json', stage)
        self.write_json(report / 'stage-inherit.json', inherited)
        return report, checksum, stage

    def private_worker(self, label='partial'):
        report = self.root / 'report'
        work = self.root / 'work'
        output = work / 'workers' / 'case' / label
        process = output / 'SubProcesses/P0_case'
        process.mkdir(parents=True)
        (output / 'Cards').mkdir()
        (output / 'Cards/param_card.dat').write_text('physical\n')
        (output / 'Cards/MadLoopParams.dat').write_text('numerical\n')
        (process / 'check').write_bytes(b'\x7fELF' + b'fake-not-executed' * 4)
        (process / 'a.f').write_text('source\n')
        return report, work, output, process

    def test_default_reads_no_active_inputs(self):
        with patch('sys.argv', ['capture.py']), contextlib.redirect_stdout(io.StringIO()) as stdout, \
                patch.object(capture, 'read_json', side_effect=AssertionError('active JSON read')), \
                patch.object(capture, 'terminal_gate', side_effect=AssertionError('terminal read')), \
                patch.object(capture, 'postclosure_bindings', side_effect=AssertionError('worker read')):
            capture.main()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result['required_terminal_session'], 7248)
        self.assertFalse(result['capture'])
        self.assertFalse(result['native_execution'])
        self.assertFalse(result['report_inputs_read'])

    def test_execute_requires_both_arguments_before_read(self):
        for argv in (['capture.py', '--execute'],
                     ['capture.py', '--execute', '--confirmation', 'unused.json'],
                     ['capture.py', '--execute', '--destination', 'unused-destination']):
            with self.subTest(argv=argv), patch('sys.argv', argv), \
                    patch.object(capture, 'read_json', side_effect=AssertionError('premature read')):
                with self.assertRaises(ValueError):
                    capture.main()

    def test_invalid_confirmation_precedes_terminal_and_output(self):
        variants = [{}, {**self.confirmation(), 'extra': True}]
        for key, value in (('kind', 'synthetic'), ('session_id', True), ('session_id', 7249),
                           ('exit_code', False), ('exit_code', -9), ('observed_utc', 'invented'),
                           ('source_index_sha256', '0' * 64), ('inherited_checkpoint_sha256', '0' * 64)):
            variants.append({**self.confirmation(), key: value})
        destination = self.root / 'must-not-exist'
        for variant in variants:
            with self.subTest(variant=variant), \
                    patch.object(capture.Path, 'cwd', return_value=capture.ROOT), \
                    patch.object(capture, 'terminal_gate', side_effect=AssertionError('premature report read')):
                with self.assertRaises(ValueError):
                    capture.capture(variant, destination)
                self.assertFalse(destination.exists())

    def test_terminal_gate_accepts_closed_pass_and_failure(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code):
                report, checksum, stage = self.terminal_fixture(exit_code=exit_code)
                with patch.object(capture, 'INHERITED_CHECKPOINT', checksum):
                    self.assertEqual(capture.terminal_gate(report, self.confirmation(exit_code)), stage)
                for path in report.iterdir():
                    path.unlink()
                report.rmdir()

    def test_terminal_gate_rejects_status_exit_or_exception_mismatch(self):
        report, checksum, stage = self.terminal_fixture()
        with patch.object(capture, 'INHERITED_CHECKPOINT', checksum):
            with self.assertRaises(ValueError):
                capture.terminal_gate(report, self.confirmation(0))
            self.write_json(report / 'stage-run.json', {**stage, 'exception': None})
            with self.assertRaises(ValueError):
                capture.terminal_gate(report, self.confirmation(1))

    def test_terminal_gate_rejects_source_checkpoint_and_inherit_changes(self):
        report, checksum, stage = self.terminal_fixture()
        with patch.object(capture, 'INHERITED_CHECKPOINT', checksum):
            for key in ('source_index_sha256', 'inherited_checkpoint_sha256'):
                self.write_json(report / 'stage-run.json', {**stage, key: '0' * 64})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    capture.terminal_gate(report, self.confirmation())
            self.write_json(report / 'stage-run.json', stage)
            (report / 'inherited-checkpoint.json').write_text('{"changed":true}\n')
            with self.assertRaises(ValueError):
                capture.terminal_gate(report, self.confirmation())
            (report / 'inherited-checkpoint.json').write_text('{}\n')
            inherited = capture.read_json(report / 'stage-inherit.json')
            self.write_json(report / 'stage-inherit.json', {**inherited, 'status': 'failed'})
            with self.assertRaises(ValueError):
                capture.terminal_gate(report, self.confirmation())

    def test_gzip_is_deterministic_append_once_and_text_only(self):
        first = capture.Capsule(self.root / 'capsule-a', '1' * 64)
        second = capture.Capsule(self.root / 'capsule-b', '1' * 64)
        raw = (b'large retained failed console\n' * 4000)
        for packet in (first, second):
            packet.append('closed/console.txt', raw)
            packet.save('retention.json', {'large': raw.decode()})
        name = first.index['compressed_text']['closed/console.txt']['path']
        self.assertEqual((first.root / name).read_bytes(), (second.root / name).read_bytes())
        self.assertEqual(gzip.decompress((first.root / name).read_bytes()), raw)
        self.assertEqual((first.root / name).read_bytes()[4:8], b'\0' * 4)
        self.assertIn('retention.json', first.index['files'])
        with self.assertRaises(ValueError):
            first.append('closed/console.txt', b'duplicate')
        with self.assertRaises(ValueError):
            first.append('binary', b'\0')
        with self.assertRaises(FileExistsError):
            capture.Capsule(first.root, '1' * 64)

    def test_reader_paths_cover_own_and_two_dependencies(self):
        ancestor, preparation = self.root / 'ancestor', self.root / 'preparation'
        for dependency in (ancestor, preparation):
            dependency.mkdir()
            (dependency / 'payload.txt').write_text(dependency.name)
            self.write_json(dependency / 'evidence-index.json', {'files': {'payload.txt': {}}})
        with patch.object(capture, 'ANCESTOR', ancestor), patch.object(capture, 'PREPARATION', preparation):
            packet = capture.Capsule(self.root / 'capsule', '1' * 64)
            packet.append('payload.txt', b'own')
            reader = packet.reader()
            self.assertEqual(reader.paths, {'evidence-index.json', 'payload.txt',
                                           'ancestor/evidence-index.json', 'ancestor/payload.txt',
                                           'preparation/evidence-index.json', 'preparation/payload.txt'})
            self.assertEqual(reader('payload.txt'), b'own')
            self.assertEqual(reader('ancestor/payload.txt'), b'ancestor')
            self.assertEqual(reader('preparation/payload.txt'), b'preparation')
            self.assertEqual(json.loads(reader('evidence-index.json')), packet.index)
            (ancestor / 'unindexed.txt').write_text('must not be read')
            with self.assertRaises(ValueError):
                reader('ancestor/unindexed.txt')

    def test_partial_worker_and_journals_survive_missing_specification(self):
        report, work, output, process = self.private_worker()
        raw = process / 'worker-raw/session-abcdefgh/startup'
        raw.mkdir(parents=True)
        (raw / 'startup.json').write_text('{}\n')
        (raw / 'binary.dat').write_bytes(b'\0')
        retained = {}
        post = capture.postclosure_bindings(report, work, lambda name, data: retained.__setitem__(name, data))
        worker = post['workers']['case/partial']
        self.assertEqual(post['worker_directories'], ['case/partial'])
        self.assertFalse(worker['specification_present'])
        self.assertEqual(worker['scientific_sources']['SubProcesses/P0_case/a.f'], capture.file_sha(process / 'a.f'))
        self.assertEqual(worker['physical_card_sha256'], capture.file_sha(output / 'Cards/param_card.dat'))
        self.assertEqual(len(worker['journals']), 1)
        self.assertEqual(len(retained), 1)
        self.assertTrue(any('No readable specification' in error for error in worker['binding_errors']))
        self.assertTrue(any('Unretained journal' in error for error in worker['binding_errors']))

    def test_missing_changed_and_unexpected_runtime_bindings_are_errors(self):
        report, work, output, process = self.private_worker('specified')
        evidence = report / 'runs/case/specified'
        specification = {'cwd': str(process), 'scientific_sources': {'SubProcesses/P0_case/a.f': '0' * 64},
                         'numerical_cards': {'Cards/MadLoopParams.dat': '0' * 64},
                         'physical_card_sha256': '0' * 64, 'build': {'executable_sha256': '0' * 64},
                         'expected_runtime': {}}
        self.write_json(evidence / 'specification.json', specification)
        post = capture.postclosure_bindings(report, work, lambda *_: None)
        self.assertTrue(any('No retained mapped-runtime' in error for error in post['workers']['case/specified']['binding_errors']))
        binary = process / 'check'
        stat = binary.stat()
        dev = f'{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}'
        missing = self.root / 'missing-library.so'
        (evidence / 'maps-before.txt').write_text(
            f'00000000-00001000 r-xp 00000000 {dev} {stat.st_ino + 1} {binary}\n'
            f'00002000-00003000 r-xp 00000000 00:00 1 {missing}\n')
        post = capture.postclosure_bindings(report, work, lambda *_: None)
        errors = post['workers']['case/specified']['binding_errors']
        self.assertTrue(any('Changed scientific_sources' in error for error in errors))
        self.assertTrue(any('Changed numerical_cards' in error for error in errors))
        self.assertTrue(any('Changed physical parameter card' in error for error in errors))
        self.assertTrue(any('Changed worker executable' in error for error in errors))
        self.assertTrue(any('Mapped inode/device changed' in error for error in errors))
        self.assertIn('error', post['mapped_files'][str(missing)])
        self.assertEqual(post['mapped_files'][str(binary)]['sha256'], capture.file_sha(binary))

    def test_external_source_card_and_executable_links_are_not_read(self):
        report, work, output, process = self.private_worker()
        outside = self.root / 'outside'
        outside.mkdir()
        secret = outside / 'secret.f'
        secret.write_text('unrelated input\n')
        (output / 'Source').symlink_to(outside, target_is_directory=True)
        for path in (output / 'Cards/param_card.dat', process / 'check'):
            path.unlink()
            path.symlink_to(secret)
        external_reads = []
        original = capture.file_sha
        def guarded_sha(path):
            if Path(path).resolve().is_relative_to(outside):
                external_reads.append(str(path))
                raise AssertionError('External input opened')
            return original(path)
        with patch.object(capture, 'file_sha', guarded_sha):
            post = capture.postclosure_bindings(report, work, lambda *_: None)
        worker = post['workers']['case/partial']
        self.assertEqual(external_reads, [])
        self.assertIsNone(worker['physical_card_sha256'])
        self.assertIsNone(worker['executable_sha256'])
        self.assertTrue(any('Unsafe scientific source root' in error for error in worker['binding_errors']))
        self.assertTrue(any('Unsafe input path' in error for error in worker['binding_errors']))

    def test_malformed_specification_records_error_without_aborting(self):
        for number, invalid in enumerate(([], {'cwd': 12}, {'cwd': '/unused', 'scientific_sources': []})):
            with self.subTest(invalid=invalid):
                label = 'malformed-' + str(number)
                report, work, output, process = self.private_worker(label)
                self.write_json(report / 'runs' / 'case' / label / 'specification.json', invalid)
                post = capture.postclosure_bindings(report, work, lambda *_: None)
                worker = post['workers']['case/' + label]
                self.assertTrue(worker['specification_present'])
                self.assertTrue(any('Unreadable specification' in error for error in worker['binding_errors']))
                self.assertTrue(any('No readable specification' in error for error in worker['binding_errors']))


if __name__ == '__main__':
    unittest.main()
