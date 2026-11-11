"""Pure package replay, relocation and rejection tests; outputs stay outside it."""
import contextlib
import copy
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
CAPSULE = HERE.parent
SPEC = importlib.util.spec_from_file_location('portable_wrapper_abi_verifier', HERE / 'verify.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
ACTIVE_GUARDS = []


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def digest_index(root):
    return identity(Path(root) / 'evidence-index.json')['sha256']


def reseal(root):
    path = Path(root) / 'evidence-index.json'
    index = json.loads(path.read_bytes())
    index['files'] = {name: identity(Path(root) / name) for name in index['files']}
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    return digest_index(root)


def allowed_path(path, roots):
    if isinstance(path, int):
        return True
    name = os.path.abspath(os.fsdecode(path))
    return any(name == root or name.startswith(root + os.sep) for root in roots)


def audit(event, args):
    if not ACTIVE_GUARDS:
        return
    state = ACTIVE_GUARDS[-1]
    if event == 'open':
        name, mode, flags = args
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing or not allowed_path(name, state['roots']):
            state['forbidden_attempts'] += 1
            raise AssertionError('forbidden package replay open: ' + str(name))
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'ctypes.dlopen'):
        state['forbidden_attempts'] += 1
        raise AssertionError('forbidden replay API: ' + event)


sys.addaudithook(audit)


@contextlib.contextmanager
def blocked(root):
    roots = [str(Path(root).resolve()), str(Path(sys.prefix).resolve()),
             str(Path(sys.base_prefix).resolve())]
    state = dict(roots=roots, forbidden_attempts=0, external_resolve_attempts=0)
    original_resolve = Path.resolve
    def checked_resolve(path, *args, **kwargs):
        if not allowed_path(path, roots):
            state['external_resolve_attempts'] += 1
            raise AssertionError('recorded external path resolved: ' + str(path))
        return original_resolve(path, *args, **kwargs)
    def deny(*args, **kwargs):
        state['forbidden_attempts'] += 1
        raise AssertionError('native/subprocess call forbidden during replay')
    ACTIVE_GUARDS.append(state)
    try:
        with patch.object(Path, 'resolve', checked_resolve), \
             patch.object(subprocess, 'Popen', deny), \
             patch.object(os, 'system', deny), patch.object(ctypes, 'CDLL', deny):
            yield state
    finally:
        ACTIVE_GUARDS.pop()


class PortableReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (CAPSULE / 'evidence-index.json').is_file():
            raise RuntimeError('seal the parent package before running portable tests')
        cls.index_sha = digest_index(CAPSULE)

    @contextlib.contextmanager
    def relocated(self):
        with tempfile.TemporaryDirectory(prefix='generic-wrapper-abi-portable-') as temp:
            root = Path(temp) / 'isolated-copy'
            shutil.copytree(CAPSULE, root)
            yield root

    def test_exact_replay_and_no_forbidden_calls(self):
        before = VERIFY.Package(CAPSULE).inventory()
        with blocked(CAPSULE) as guard:
            result = VERIFY.verify(CAPSULE, self.index_sha)
        self.assertEqual(guard['forbidden_attempts'], 0)
        self.assertEqual(guard['external_resolve_attempts'], 0)
        self.assertEqual(result['source_bound_available'], 180)
        self.assertEqual(result['conservative_rejections'], 60)
        self.assertTrue(result['independent_text_audit_bound'])
        self.assertTrue(result['source_parent_interval_gap_reproduced'])
        self.assertFalse(result['full_geometry_window_qualified'])
        self.assertEqual(before, VERIFY.Package(CAPSULE).inventory())

    def test_relocated_replay_with_original_tree_and_native_calls_blocked(self):
        original = VERIFY.verify(CAPSULE, self.index_sha)
        with self.relocated() as root:
            with blocked(root) as guard:
                result = VERIFY.verify(root, self.index_sha)
            self.assertEqual(result, original)
            self.assertEqual(guard['forbidden_attempts'], 0)
            self.assertEqual(guard['external_resolve_attempts'], 0)

    def test_wrong_evidence_digest_rejected(self):
        with self.assertRaises(ValueError):
            VERIFY.verify(CAPSULE, '0' * 64)

    def test_changed_raw_protocol_rejected(self):
        with self.relocated() as root:
            path = root / 'run/native.log'
            path.write_bytes(path.read_bytes().replace(b'ENDCASE 1\n', b'ENDCASE 2\n', 1))
            with self.assertRaises(ValueError):
                VERIFY.verify(root, self.index_sha)

    def test_resealed_raw_protocol_still_fails_closed_receipt_join(self):
        with self.relocated() as root:
            path = root / 'run/native.log'
            path.write_bytes(path.read_bytes() + b'EXTRA 0\n')
            digest = reseal(root)
            with self.assertRaises(ValueError):
                VERIFY.verify(root, digest)

    def test_resealed_failed_stage_rejected(self):
        with self.relocated() as root:
            path = root / 'build/result.json'
            row = json.loads(path.read_bytes())
            row['passed'] = False
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                VERIFY.verify(root, reseal(root))

    def test_resealed_source_edit_rejected_by_original_seal(self):
        with self.relocated() as root:
            path = root / 'source/analyze.py'
            path.write_bytes(path.read_bytes() + b'\n# changed\n')
            with self.assertRaises(ValueError):
                VERIFY.verify(root, reseal(root))

    def test_resealed_source3_edit_rejected_by_original_seal(self):
        with self.relocated() as root:
            path = root / 'source3/src/madgraph/loop/invariant_render.py'
            path.write_bytes(path.read_bytes() + b'\n# changed\n')
            with self.assertRaises(ValueError):
                VERIFY.verify(root, reseal(root))

    def test_resealed_parent_interval_overclaim_rejected(self):
        with self.relocated() as root:
            path = root / 'scope.json'
            scope = json.loads(path.read_bytes())
            scope['parent_instruction_intervals_qualified'] = True
            path.write_text(json.dumps(scope) + '\n')
            with self.assertRaises(ValueError):
                VERIFY.verify(root, reseal(root))

    def test_extra_file_rejected(self):
        with self.relocated() as root:
            (root / 'unexpected.txt').write_text('not sealed\n')
            with self.assertRaises(ValueError):
                VERIFY.verify(root, self.index_sha)

    def test_binary_payload_rejected_even_with_new_inventory(self):
        with self.relocated() as root:
            path = root / 'probe'
            path.write_bytes(b'\x7fELF\x00not a text payload')
            index_path = root / 'evidence-index.json'
            index = json.loads(index_path.read_bytes())
            index['files']['probe'] = identity(path)
            index_path.write_text(json.dumps(index) + '\n')
            with self.assertRaises(ValueError):
                VERIFY.verify(root, digest_index(root))

    def test_symlink_payload_rejected_without_target_open(self):
        with self.relocated() as root:
            outside = root.parent / 'outside.txt'
            outside.write_text('unrelated tree\n')
            readme = root / 'README.md'
            readme.unlink()
            readme.symlink_to(outside)
            with self.assertRaises(ValueError):
                VERIFY.verify(root, self.index_sha)

    def test_unsafe_paths_rejected(self):
        for path in ('../outside', '/absolute', 'a/../b', 'a//b', 'a\\b', '.', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                VERIFY.relative(path)

    def test_resealed_diagnostic_scope_change_rejected(self):
        with self.relocated() as root:
            path = root / 'diagnostics/gap-result.json'
            row = json.loads(path.read_bytes())
            row['full_geometry_window_qualified'] = True
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                VERIFY.verify(root, reseal(root))


if __name__ == '__main__':
    unittest.main(verbosity=2)
