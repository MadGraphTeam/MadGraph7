"""Pure compact-stream and inventory tests; no compiler, library or native work."""
import gzip
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('compact_capture_verify', Path(__file__).with_name('verify.py'))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


class GzipTests(unittest.TestCase):
    def setUp(self):
        self.raw = b'complete GNU link map\n' * 100
        self.packed = gzip.compress(self.raw, mtime=0)

    def test_one_complete_deterministic_member_is_lossless(self):
        self.assertEqual(V.decode_gzip(self.packed, len(self.raw), V.sha(self.raw)), self.raw)

    def test_concatenated_and_trailing_members_are_rejected(self):
        for bad in (self.packed + b'junk', self.packed + gzip.compress(b'extra', mtime=0)):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                V.decode_gzip(bad, len(self.raw), V.sha(self.raw))

    def test_truncated_header_payload_and_trailer_are_rejected(self):
        for bad in (self.packed[:8], self.packed[:-1], self.packed[:-8], self.packed[:20]):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                V.decode_gzip(bad, len(self.raw), V.sha(self.raw))

    def test_exact_raw_size_and_hash_are_required(self):
        for size, checksum in ((len(self.raw) - 1, V.sha(self.raw)),
                               (len(self.raw) + 1, V.sha(self.raw)), (len(self.raw), '0' * 64)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                V.decode_gzip(self.packed, size, checksum)

    def test_crc_corruption_is_rejected(self):
        bad = bytearray(self.packed); bad[-8] ^= 1
        with self.assertRaises(ValueError): V.decode_gzip(bytes(bad), len(self.raw), V.sha(self.raw))

    def test_expansion_budget_and_header_metadata_are_bounded(self):
        for size in (-1, True, V.MAX_MAP_BYTES + 1):
            with self.assertRaises(ValueError): V.decode_gzip(self.packed, size, V.sha(self.raw))
        bad = bytearray(self.packed); bad[4] = 1
        with self.assertRaises(ValueError): V.decode_gzip(bytes(bad), len(self.raw), V.sha(self.raw))
        bad = bytearray(self.packed); bad[3] = 8
        with self.assertRaises(ValueError): V.decode_gzip(bytes(bad), len(self.raw), V.sha(self.raw))


class InventoryTests(unittest.TestCase):
    def fixture(self):
        blobs = {'index.json': b'index', 'retained.txt': b'abc'}
        def read(name):
            self.assertFalse(name.startswith('/'))
            return blobs[name]
        read.paths = set(blobs)
        index = dict(schema='madloop-tirust-first-failure-capture-compact-v1',
                     original_index_sha256=V.ORIGINAL_INDEX_SHA256, native_evaluations=0,
                     logical_data_changes=0, files={'retained.txt': dict(sha256=V.sha(b'abc'), bytes=3)})
        return blobs, read, index

    def test_physical_hashes_sizes_and_complete_inventory(self):
        blobs, read, index = self.fixture()
        self.assertEqual(V.physical_payloads(index, read), {'retained.txt': b'abc'})
        read.paths.add('unindexed')
        with self.assertRaises(ValueError): V.physical_payloads(index, read)
        read.paths.remove('unindexed'); blobs['retained.txt'] = b'abd'
        with self.assertRaises(ValueError): V.physical_payloads(index, read)

    def test_original_identity_and_zero_work_claims_are_strict(self):
        _, read, index = self.fixture()
        for updates in (dict(original_index_sha256='0' * 64), dict(native_evaluations=1),
                        dict(logical_data_changes=1), dict(native_evaluations=False)):
            with self.assertRaises(ValueError): V.physical_payloads(index | updates, read)

    def test_original_paths_traversal_and_duplicate_json_are_rejected(self):
        for name in ('/original/host/path', '../map', 'a/../map', 'a//map', 'a\\map'):
            with self.assertRaises(ValueError): V.logical_path(name)
        for text in ('{"x":1,"x":2}', '{"x":Infinity}', '{"x":NaN}'):
            with self.assertRaises(ValueError): V.json_value(text)

    def test_unchecked_inherited_code_is_never_executed(self):
        with self.assertRaises(ValueError): V.inherited_verifier(b'raise AssertionError("executed")')


if __name__ == '__main__': unittest.main()
