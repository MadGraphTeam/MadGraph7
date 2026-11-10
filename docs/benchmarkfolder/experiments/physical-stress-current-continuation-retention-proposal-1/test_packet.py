"""Pure packet-format tests; no campaign reads or native execution."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('continuation_packet_tested', HERE / 'verify_evidence.py')
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)


def fixture(compressed=False):
    raw = b'{"a":1}\n'; files = {}; index = {'schema_version':1, 'files':{}, 'compressed_text':{}}
    name = 'compressed/value.json.gz' if compressed else 'value.json'
    physical = gzip.compress(raw, mtime=0) if compressed else raw
    files[name] = physical
    index['files'][name] = {'kind':'file', 'size':len(physical), 'sha256':hashlib.sha256(physical).hexdigest()}
    if compressed:
        index['compressed_text']['value.json'] = {'path':name, 'raw_bytes':len(raw), 'raw_sha256':hashlib.sha256(raw).hexdigest()}
    files['evidence-index.json'] = json.dumps(index).encode()
    return files


class Packet(unittest.TestCase):
    def test_plain_and_deterministic_gzip_roundtrip(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                files = fixture(compressed)
                self.assertEqual(V.Packet(files.__getitem__).load('value.json'), {'a':1})

    def test_changed_physical_payload_rejected(self):
        files = fixture(); files['value.json'] = b'changed'
        with self.assertRaisesRegex(ValueError, 'Changed evidence'): V.Packet(files.__getitem__)

    def test_wrong_gzip_raw_binding_rejected(self):
        files = fixture(True); index = json.loads(files['evidence-index.json'])
        index['compressed_text']['value.json']['raw_sha256'] = '0' * 64
        files['evidence-index.json'] = json.dumps(index).encode()
        with self.assertRaisesRegex(ValueError, 'Changed compressed'): V.Packet(files.__getitem__)

    def test_nonfinite_json_rejected(self):
        files = fixture(); raw = b'{"a":NaN}'
        index = json.loads(files['evidence-index.json']); files['value.json'] = raw
        index['files']['value.json'].update(size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        files['evidence-index.json'] = json.dumps(index).encode()
        with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'): V.Packet(files.__getitem__).load('value.json')

    def test_unportable_names_rejected(self):
        for name in ('../x', '/x', 'a/../x', 'a//x', 'a\\x'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError): V.safe(name)

    def test_dependency_indices_are_explicit(self):
        declaration = V.dependency_declaration()
        self.assertEqual(set(declaration), {'ancestor','preparation'})
        self.assertEqual(declaration['ancestor']['index_sha256'], V.ANCESTOR_INDEX)
        self.assertEqual(declaration['preparation']['index_sha256'], V.PREPARATION_INDEX)

    def test_callback_without_physical_paths_rejected_before_any_read(self):
        def reader(name): raise AssertionError('must reject before reading')
        with self.assertRaisesRegex(ValueError, 'must declare its physical paths'): V.verify(read_bytes=reader)

    def test_explicit_reader_closure_rejects_extra_or_missing_paths(self):
        def reader(name): raise AssertionError('closure test never reads')
        reader.paths = {'a', 'evidence-index.json'}
        self.assertEqual(V.reader_closure(reader, {'a', 'evidence-index.json'}), reader.paths)
        for expected in ({'evidence-index.json'}, {'a', 'b', 'evidence-index.json'}):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, 'Reader closure differs'): V.reader_closure(reader, expected)


if __name__ == '__main__': unittest.main()
