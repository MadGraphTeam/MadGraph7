#!/usr/bin/env python3
"""Source/orchestration tests only; no subprocesses, compilers or reducers."""
import contextlib
import difflib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('current_lifecycle_tested', HERE / 'run.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class ControllerTests(unittest.TestCase):
    def test_default_plan_creates_nothing_and_starts_nothing(self):
        with patch.object(R.sys, 'argv', ['run.py']), patch.object(R, 'source_seal') as seal, \
                patch.object(R.subprocess, 'Popen') as spawn, contextlib.redirect_stdout(io.StringIO()) as out:
            R.main()
        self.assertFalse(seal.called)
        self.assertFalse(spawn.called)
        value = json.loads(out.getvalue())
        self.assertFalse(value['execution_performed'])
        self.assertEqual(value['physical_calls_per_worker'], 19)
        self.assertEqual(value['key_calls_per_worker'], 28)
        self.assertEqual(value['cpu'], 12)
        self.assertEqual(value['jobs'], 1)

    def test_execute_requires_explicit_stage(self):
        with patch.object(R.sys, 'argv', ['run.py', '--execute']), \
                patch.object(R, 'source_seal') as seal, self.assertRaisesRegex(RuntimeError, 'explicit stage'):
            R.main()
        self.assertFalse(seal.called)

    def test_query_names_are_unique_across_separate_stages(self):
        names = []
        with patch.object(R, 'command', side_effect=lambda args, name, *rest: names.append(name)):
            for stage in ('build', 'smoke', 'run'):
                with patch.object(R, 'CURRENT_STAGE', stage), patch.object(R, 'QUERY', 0):
                    R.capture(['dummy'])
                    R.capture(['dummy'])
        self.assertEqual(len(set(names)), 6)
        self.assertEqual(names[2], 'query-smoke-0001')

    def test_unstaged_query_is_rejected_without_spawn(self):
        with patch.object(R, 'CURRENT_STAGE', None), patch.object(R, 'command') as command, \
                self.assertRaisesRegex(RuntimeError, 'stage namespace'):
            R.capture(['dummy'])
        self.assertFalse(command.called)

    def test_rendered_drivers_are_sealed_separately_from_generic_inventory(self):
        names = ('lifecycle_driver.f', 'lifecycle_card_smoke.f', 'observer.c', 'tirust.h', 'lifecycle.mk')
        with tempfile.TemporaryDirectory() as temp, patch.object(R, 'OUTPUT', Path(temp)):
            sub = Path(temp) / 'SubProcesses'
            sub.mkdir()
            for name in names:
                (sub / name).write_text(name)
            state = {'rendered': {name: R.sha(sub / name) for name in names}}
            R.validate_rendered(state)
            self.assertNotIn('SubProcesses/lifecycle_driver.f', R.q.inventory(Path(temp), generated=True))
            for name in names:
                old = (sub / name).read_bytes()
                (sub / name).write_bytes(old + b'changed')
                with self.assertRaisesRegex(RuntimeError, 'Rendered source changed'):
                    R.validate_rendered(state)
                (sub / name).write_bytes(old)
            del state['rendered']['lifecycle_card_smoke.f']
            with self.assertRaisesRegex(RuntimeError, 'Incomplete rendered'):
                R.validate_rendered(state)

    def test_recorded_environment_omits_unrelated_credentials(self):
        value = R.recorded_environment({'PATH': '/tool', 'OMP_NUM_THREADS': '1',
                                        'PRIVATE_TOKEN': 'do-not-retain', 'SYMBOLICA_LICENSE': 'secret'})
        self.assertEqual(value, {'PATH': '/tool', 'OMP_NUM_THREADS': '1'})

    def test_runtime_admission_rejects_changed_prelink_closure(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(R, 'REPORT', Path(temp)), \
                patch.object(R, 'frozen'):
            root = Path(temp)
            for name in ('source-freeze.json', 'generated-source-freeze.json', 'linker-prebinding.json'):
                (root / name).write_text('{}\n')
            (root / 'build.json').write_text(json.dumps({
                'source_freeze_sha256': R.sha(root / 'source-freeze.json'),
                'generated_source_freeze_sha256': R.sha(root / 'generated-source-freeze.json'),
                'linker_prebinding_sha256': 'incorrect'}))
            with self.assertRaisesRegex(RuntimeError, 'Pre-link closure evidence changed'):
                R.validate_build()

    def test_incomplete_warning_rejects_unchecked_120(self):
        R.require_complete_console('RESULT 120 finite=1.0\n')
        with self.assertRaisesRegex(RuntimeError, 'Incomplete amplitude'):
            R.require_complete_console('##W03: omitted contribution\nRESULT 120 finite=1.0\n')

    def test_short_equivalent_alternate_mass_token(self):
        original = 'BLOCK MASS\n 23 9.118760e+01 # MZ\n 6 1.730000e+02\nDECAY 23 0.0\n'
        value = R.alternate_card(original)
        self.assertIn('23  9.200000e+01', value)
        self.assertIn('6 1.730000e+02', value)
        self.assertLessEqual(R.smoke_parser.numeric_prevalidation(value)['maximum_width'], 20)
        with self.assertRaisesRegex(RuntimeError, 'Unsupported numeric token'):
            R.smoke_parser.numeric_prevalidation('23 9.20000000000000000000000e+01\n')

    def test_alternate_mass_requires_unique_parameter(self):
        for text in ('BLOCK MASS\n 6 173\n', 'BLOCK MASS\n23 91\n23 91\n'):
            with self.assertRaisesRegex(RuntimeError, 'Unique MZ'):
                R.alternate_card(text)

    def test_nested_deadline_restores_parent_without_extending_it(self):
        with patch.object(R, 'ACTIVE', None), patch.object(R.time, 'monotonic', return_value=100), \
                patch.object(R.signal, 'getsignal', return_value='old'), \
                patch.object(R.signal, 'signal'), patch.object(R.signal, 'setitimer') as timer:
            with R.Deadline(100) as parent:
                with R.Deadline(200) as child:
                    self.assertEqual(child.end, 200)
                    self.assertIs(child.parent, parent)
                self.assertIs(R.ACTIVE, parent)
                self.assertEqual(parent.end, 200)
                self.assertEqual(timer.call_args.args[1], 90)
            self.assertIsNone(R.ACTIVE)

    def test_nested_worker_budget_is_shorter_than_stage(self):
        with patch.object(R, 'ACTIVE', None), patch.object(R.time, 'monotonic', return_value=100), \
                patch.object(R.signal, 'getsignal'), patch.object(R.signal, 'signal'), \
                patch.object(R.signal, 'setitimer'):
            with R.Deadline(750):
                with R.Deadline(180) as child:
                    self.assertEqual(child.end, 280)

    def test_expiry_during_handoff_is_deferred_until_child_owned(self):
        deadline = R.Deadline(20)
        deadline.handoff = True
        deadline.expired()
        self.assertTrue(deadline.deferred)
        deadline.handoff = False
        with self.assertRaises(TimeoutError):
            deadline.expired()

    def test_deadline_exit_rejects_elapsed_overrun(self):
        with patch.object(R, 'ACTIVE', None), patch.object(R.time, 'monotonic', side_effect=[0, 0, 21]), \
                patch.object(R.signal, 'getsignal'), patch.object(R.signal, 'signal'), \
                patch.object(R.signal, 'setitimer'), self.assertRaisesRegex(TimeoutError, 'exceeded'):
            with R.Deadline(20):
                pass

    def test_cleanup_failure_is_retained_before_raise(self):
        process = Mock(returncode=0)
        process.wait.return_value = 0
        owned = Mock()
        owned.stop.return_value = {'survivors': [{'pid': 123}]}
        active = types.SimpleNamespace(end=10000, handoff=False, deferred=False)
        with tempfile.TemporaryDirectory() as temp, patch.object(R, 'REPORT', Path(temp)), \
                patch.object(R, 'ACTIVE', active), patch.object(R, 'REQUESTS', Mock()), \
                patch.object(R, 'environment', return_value={'PATH': '/tools', 'PRIVATE_TOKEN': 'secret'}), \
                patch.object(R, 'remaining', return_value=20), patch.object(R.time, 'monotonic', return_value=100), \
                patch.object(R, 'now', return_value='start-stamp'), patch.object(R.subprocess, 'Popen', return_value=process), \
                patch.object(R.cleanup, 'OwnedChild', return_value=owned), patch.object(R.signal, 'getsignal'), \
                patch.object(R.signal, 'signal'), patch.object(R.signal, 'setitimer'):
            with self.assertRaisesRegex(RuntimeError, 'survived cleanup'):
                R.command(['not-executed'], 'synthetic-cleanup')
            value = json.loads((Path(temp) / 'execution/synthetic-cleanup.json').read_text())
            self.assertIn('survived cleanup', value['cleanup_exception'])
            self.assertEqual(value['started_utc'], 'start-stamp')
            self.assertEqual(value['environment'], {'PATH': '/tools'})

    def test_observer_diff_is_only_declared_maps_hook(self):
        old = (HERE / 'observer-original.c').read_text()
        new = (HERE / 'observer.c').read_text()
        diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                           fromfile='observer-original.c', tofile='observer.c'))
        self.assertEqual(diff, (HERE / 'observer-maps.patch').read_text())
        start = new.index('/* Runtime-map capture occurs')
        end = new.index('void lc_open(void)', start)
        restored = new[:start] + new[end:]
        for name in ('open', 'close'):
            restored = restored.replace(f'    lc_capture_maps("runtime-maps-{name}.txt");\n', '')
        self.assertEqual(restored, old)

    def test_copied_sources_match_frozen_lineage(self):
        for name, record in json.loads((HERE / 'lineage.json').read_text())['copied_files'].items():
            self.assertEqual(R.sha(HERE / name), record['sha256'], name)
        manifest = R.read_manifest()
        self.assertEqual(manifest['id'], 'entry-latch-cache-guard-1')

    def test_successful_historical_card_smoke_replays_offline(self):
        path = R.ROOT / 'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle/attempt5/runs/card-smoke'
        result = R.smoke_parser.check_smoke(path / 'records.txt', path / 'trace.txt')
        self.assertTrue(result['passed'])
        self.assertEqual(result['native_exports_observed'], 0)

    def test_actual_loader_used_and_missing_tirust_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            loader, library = path / 'ld.so', path / 'libmadloop_tirust.so'
            loader.write_bytes(b'loader')
            library.write_bytes(b'library')
            with patch.object(R, 'interpreter', return_value=loader), \
                    patch.object(R.q, 'INSTALL', path), patch.object(R, 'remaining', return_value=10), \
                    patch.object(R, 'capture', return_value=f'libmadloop_tirust.so => {library} (0x1)\n{loader} (0x2)') as capture:
                value = R.runtime(path / 'exe')
                self.assertEqual(capture.call_args.args[0], [loader, '--list', path / 'exe'])
                self.assertEqual(set(value['files']), {str(loader), str(library)})
            with patch.object(R, 'interpreter', return_value=loader), \
                    patch.object(R.q, 'INSTALL', path), patch.object(R, 'remaining', return_value=10), \
                    patch.object(R, 'capture', return_value=f'{loader} (0x2)'), \
                    self.assertRaisesRegex(RuntimeError, 'TIRust DSO absent'):
                R.runtime(path / 'exe')


if __name__ == '__main__':
    unittest.main()
