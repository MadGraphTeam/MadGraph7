"""Fault tests use temporary text providers only; no library/process execution."""
from pathlib import Path
from unittest import mock
import copy
import hashlib
import json
import tempfile
import unittest
import donor_closure as dc
import run as control


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qp-donor-closure-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.directory = self.base / 'private-runtime'
        self.runtime, self.manifest = {}, {'schema': 'exact-donor-soname-aliases-v1', 'entries': []}
        for number in range(15):
            p = self.base / ('provider-%d' % number)
            p.write_text('synthetic text provider %d\n' % number)
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            self.runtime[str(p)] = sha
            self.manifest['entries'].append({'source_path': str(p), 'resolved_path': str(p),
                                             'soname': 'libfixture%d.so' % number, 'sha256': sha})

    def create(self):
        return dc.create(self.directory, self.manifest, self.runtime)

    def test_exact_15_aliases_verified_without_subprocess(self):
        with mock.patch.object(control.supervise, 'run_captured', side_effect=AssertionError('No native')):
            result = self.create()
            self.assertTrue(result['passed'])
            self.assertEqual(result['library_count'], 15)
            self.assertEqual(result, dc.verify(self.directory, self.manifest, self.runtime))

    def test_existing_directory_is_one_shot(self):
        self.directory.mkdir()
        with self.assertRaises(FileExistsError):
            self.create()

    def test_original_provider_hash_change_rejected_before_directory_creation(self):
        Path(next(iter(self.runtime))).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Original provider changed'):
            self.create()
        self.assertFalse(self.directory.exists())

    def test_missing_or_extra_library_rejected(self):
        for action in ('remove', 'extra'):
            with self.subTest(action=action):
                m = copy.deepcopy(self.manifest)
                if action == 'remove':
                    m['entries'].pop()
                else:
                    m['entries'].append(copy.deepcopy(m['entries'][0]))
                with self.assertRaisesRegex(ValueError, 'fifteen'):
                    dc.validate_manifest(m, self.runtime)

    def test_duplicate_soname_rejected(self):
        self.manifest['entries'][1]['soname'] = self.manifest['entries'][0]['soname']
        with self.assertRaisesRegex(ValueError, 'Duplicate SONAME'):
            self.create()

    def test_unsafe_soname_rejected(self):
        for name in ('', '.', '..', '../bad', '/absolute'):
            with self.subTest(name=name):
                m = copy.deepcopy(self.manifest)
                m['entries'][0]['soname'] = name
                with self.assertRaisesRegex(ValueError, 'Unsafe SONAME'):
                    dc.validate_manifest(m, self.runtime)

    def test_missing_private_alias_rejected(self):
        self.create()
        next(self.directory.iterdir()).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or extra'):
            dc.verify(self.directory, self.manifest, self.runtime)

    def test_extra_private_file_rejected(self):
        self.create()
        (self.directory / 'extra').write_text('shadow')
        with self.assertRaisesRegex(ValueError, 'Missing or extra'):
            dc.verify(self.directory, self.manifest, self.runtime)

    def test_retargeted_alias_rejected_even_same_bytes(self):
        self.create()
        e = self.manifest['entries'][0]
        other = self.base / 'same-bytes-different-path'
        other.write_bytes(Path(e['source_path']).read_bytes())
        alias = self.directory / e['soname']
        alias.unlink(); alias.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'Alias target changed'):
            dc.verify(self.directory, self.manifest, self.runtime)

    def test_regular_file_alias_rejected(self):
        self.create()
        alias = self.directory / self.manifest['entries'][0]['soname']
        raw = alias.read_bytes(); alias.unlink(); alias.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'Alias target changed'):
            dc.verify(self.directory, self.manifest, self.runtime)

    def test_mutated_provider_rejected_after_creation(self):
        self.create()
        Path(next(iter(self.runtime))).write_text('changed after aliases')
        with self.assertRaisesRegex(ValueError, 'provider bytes changed'):
            dc.verify(self.directory, self.manifest, self.runtime)

    def test_alias_directory_symlink_rejected(self):
        self.create()
        link = self.base / 'symlink-directory'; link.symlink_to(self.directory)
        with self.assertRaisesRegex(ValueError, 'not a directory'):
            dc.verify(link, self.manifest, self.runtime)

    def test_bound_runtime_is_exact_original_source2_bytes(self):
        old = json.loads((control.HERE / 'prior-source2-bindings.json').read_bytes())
        new = json.loads((control.HERE / 'bindings.json').read_bytes())
        self.assertEqual(new['runtime'], old['runtime'])
        self.assertEqual(new['target_interpreter'], old['target_interpreter'])
        control.validate_target_closure(new)

    def test_substituted_compatible_provider_still_rejected(self):
        b = json.loads((control.HERE / 'bindings.json').read_bytes())
        name = next(iter(b['runtime']))
        b['runtime'][name] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'closure changed'):
            control.validate_target_closure(b)

    def test_scientific_payloads_identical_to_source2(self):
        old = control.HERE.parent / 'source-proposal2'
        names = ['analyze.py', 'driver.f', 'observer.c', 'observer.f90', 'prepare.py', 'shapes.py', 'supervise.py',
                 'exec_target.py']
        names += [p.relative_to(control.HERE).as_posix() for p in (control.HERE / 'fixtures').rglob('*') if p.is_file()]
        for name in names:
            with self.subTest(name=name):
                self.assertEqual((control.HERE / name).read_bytes(), (old / name).read_bytes())

    def test_current_inherited_build_passes_readonly_hash_checks(self):
        binding = json.loads((control.HERE / 'bindings.json').read_bytes())
        path, proof, exe = control.inherited_build(binding)
        self.assertEqual(path.name, 'build.json')
        self.assertEqual(proof['executable_sha256'],
                         'ed6a51fff0312b06a45cd97a409aafa7d2447eef5ae7800579b44c0c904b8474')
        self.assertEqual(str(exe), proof['executable'])

    def test_wrong_reuse_source_index_rejected_without_launch(self):
        original = control.HERE
        temporary = self.base / 'fake-proposal'; temporary.mkdir()
        for name in ('build2-reuse.json', 'inherited-build2.json', 'prior-source2-index.json'):
            (temporary / name).write_bytes((original / name).read_bytes())
        record = json.loads((temporary / 'build2-reuse.json').read_bytes())
        record['source2_index_sha256'] = '0' * 64
        (temporary / 'build2-reuse.json').write_text(json.dumps(record))
        binding = json.loads((original / 'bindings.json').read_bytes())
        with mock.patch.object(control, 'HERE', temporary), mock.patch.object(control, 'run_command') as launch:
            with self.assertRaisesRegex(ValueError, 'matching source2 build2'):
                control.inherited_build(binding)
            launch.assert_not_called()

    def test_wrong_reuse_executable_hash_rejected_without_launch(self):
        original = control.HERE
        temporary = self.base / 'fake-proposal'; temporary.mkdir()
        for name in ('build2-reuse.json', 'inherited-build2.json', 'prior-source2-index.json'):
            (temporary / name).write_bytes((original / name).read_bytes())
        record = json.loads((temporary / 'build2-reuse.json').read_bytes())
        record['build2_executable_sha256'] = '0' * 64
        (temporary / 'build2-reuse.json').write_text(json.dumps(record))
        binding = json.loads((original / 'bindings.json').read_bytes())
        with mock.patch.object(control, 'HERE', temporary), mock.patch.object(control, 'run_command') as launch:
            with self.assertRaisesRegex(ValueError, 'executable or link map identity'):
                control.inherited_build(binding)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
