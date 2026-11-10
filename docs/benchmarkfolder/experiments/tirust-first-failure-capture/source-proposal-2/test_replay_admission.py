import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('additive_replay', Path(__file__).with_name('replay_admission.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {}
        self.changes = {}
        for i, name in enumerate(m.CHANGES):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = ('approved%d' % i).encode()
            path.write_bytes(blob)
            self.files[name] = 'old%d' % i
            self.changes[name] = {'old': self.files[name], 'new': m.sha(blob)}
        other = self.root / 'other.f'
        other.write_bytes(b'unchanged')
        self.files['other.f'] = m.sha(b'unchanged')

    def admit(self, files=None, original=None):
        files = self.files if files is None else files
        if original is None:
            def original(root, inventory):
                for name, checksum in inventory.items():
                    m.require(m.digest(Path(root) / name) == checksum, 'other source differs')
                return inventory
        with patch.object(m, 'ROOT', self.root), patch.object(m, 'CHANGES', self.changes):
            return m.adjusted_inventory(self.root, files, original)

    def test_only_two_successors_substituted_without_mutating_original(self):
        before = dict(self.files)
        adjusted = self.admit()
        self.assertEqual(self.files, before)
        self.assertEqual(sum(adjusted[k] != before[k] for k in before), 2)
        self.assertEqual(adjusted['other.f'], before['other.f'])

    def test_wrong_original_hash_rejected(self):
        changed = dict(self.files)
        changed[next(iter(self.changes))] = 'unregistered-old'
        with self.assertRaisesRegex(ValueError, 'Frozen template identity'):
            self.admit(changed)

    def test_missing_original_key_rejected(self):
        changed = dict(self.files)
        del changed[next(iter(self.changes))]
        with self.assertRaisesRegex(ValueError, 'Frozen template identity'):
            self.admit(changed)

    def test_third_source_change_still_rejected(self):
        (self.root / 'other.f').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'other source differs'):
            self.admit()

    def test_unapproved_successor_rejected(self):
        (self.root / next(iter(self.changes))).write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'Approved template successor'):
            self.admit()

    def test_original_null_semantics_retained(self):
        marker = object()
        def original(root, inventory):
            self.assertIn('dangling', inventory)
            self.assertIsNone(inventory['dangling'])
            return marker
        self.files['dangling'] = None
        self.assertIs(self.admit(original=original), marker)

    def test_original_native_or_source_guard_failure_propagates(self):
        def original(root, inventory):
            raise RuntimeError('original guard remains active')
        with self.assertRaisesRegex(RuntimeError, 'original guard remains active'):
            self.admit(original=original)

    def test_wrong_production_root_rejected(self):
        with patch.object(m, 'ROOT', self.root / 'other'):
            with self.assertRaisesRegex(ValueError, 'Unexpected production source root'):
                m.adjusted_inventory(self.root, self.files, lambda *args: None)

    def test_default_cli_does_not_execute_native(self):
        import ast
        tree = ast.parse(Path(m.__file__).read_text())
        main = tree.body[-1]
        self.assertIsInstance(main, ast.If)
        self.assertIn('if args.execute:', ast.unparse(main))
        self.assertEqual(len(m.CHANGES), 2)


if __name__ == '__main__':
    unittest.main()
