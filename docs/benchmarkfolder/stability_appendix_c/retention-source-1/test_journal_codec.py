"""Synthetic byte fixtures only: no native evaluation or campaign data reads."""
import copy
import gzip
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('journal_codec', Path(__file__).with_name('journal_codec.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class JournalCodecTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'synthetic-journal.txt'
        self.source.write_bytes(b'group 1\x00\n' * 11 + bytes(range(256)))
        self.output = self.root / 'chunks'

    def encode(self, **kwargs):
        return M.encode_journal(self.source, self.output, **kwargs)

    def replace_stream(self, descriptor, data):
        """Bind the actual compressed corruption, so gzip validation must catch it."""
        chunk = descriptor['chunks'][0]
        (self.output / chunk['path']).write_bytes(data)
        chunk['gzip_sha256'] = hashlib.sha256(data).hexdigest()
        chunk['gzip_bytes'] = len(data)

    def test_multichunk_round_trip_and_complete_descriptor(self):
        expected = self.source.read_bytes()
        descriptor = self.encode(raw_chunk_bytes=37)
        self.assertGreater(len(descriptor['chunks']), 1)
        self.assertEqual(descriptor['raw_bytes'], len(expected))
        self.assertEqual(descriptor['raw_sha256'], hashlib.sha256(expected).hexdigest())
        self.assertEqual(M.decode_journal(descriptor, self.output), expected)
        self.assertEqual([c['path'] for c in descriptor['chunks']],
                         [f'chunk-{i:06d}.gz' for i in range(len(descriptor['chunks']))])
        self.assertNotIn(str(self.source), str(descriptor))

    def test_deterministic_gzip_headers_and_exact_chunks(self):
        first = self.encode(raw_chunk_bytes=37)
        second_root = self.root / 'second'
        second = M.encode_journal(self.source, second_root, raw_chunk_bytes=37)
        self.assertEqual(first, second)
        for chunk in first['chunks']:
            a = (self.output / chunk['path']).read_bytes()
            self.assertEqual(a, (second_root / chunk['path']).read_bytes())
            self.assertEqual(a[:3], b'\x1f\x8b\x08')
            self.assertEqual(a[3], 0)  # No original filename or other optional fields.
            self.assertEqual(a[4:8], bytes(4))  # mtime=0.
            self.assertEqual(a[8], 2)  # Maximum compression (level9).

    def test_empty_journal_is_one_valid_empty_member(self):
        self.source.write_bytes(b'')
        descriptor = self.encode(total_raw_budget=0)
        self.assertEqual(len(descriptor['chunks']), 1)
        self.assertEqual(descriptor['raw_bytes'], 0)
        self.assertEqual(M.decode_journal(descriptor, self.output, total_raw_budget=0), b'')

    def test_source_is_not_needed_after_capture(self):
        expected = self.source.read_bytes()
        descriptor = self.encode(raw_chunk_bytes=37)
        self.source.unlink()
        self.assertEqual(M.decode_journal(descriptor, self.output), expected)

    def test_existing_chunk_is_preserved_and_created_chunks_are_rolled_back(self):
        self.output.mkdir()
        collision = self.output / 'chunk-000001.gz'
        collision.write_bytes(b'existing')
        with self.assertRaises(FileExistsError):
            self.encode(raw_chunk_bytes=37)
        self.assertEqual(collision.read_bytes(), b'existing')
        self.assertFalse((self.output / 'chunk-000000.gz').exists())
        self.assertEqual(self.source.read_bytes(), b'group 1\x00\n' * 11 + bytes(range(256)))

    def test_source_and_destination_symlinks_are_rejected(self):
        source_link = self.root / 'source-link'
        source_link.symlink_to(self.source)
        with self.assertRaises((OSError, ValueError)):
            M.encode_journal(source_link, self.output)
        real = self.root / 'real'
        real.mkdir()
        self.output.symlink_to(real, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)):
            self.encode()
        self.assertEqual(list(real.iterdir()), [])

    def test_encode_budget_is_checked_before_destination_or_reads(self):
        with mock.patch.object(M, '_directory_fd', wraps=M._directory_fd) as opened:
            with self.assertRaises(ValueError):
                self.encode(total_raw_budget=1)
            self.assertEqual(opened.call_count, 1)  # Source parent only.
        self.assertFalse(self.output.exists())

    def test_decode_budget_bombs_are_rejected_before_any_file_open(self):
        descriptor = self.encode(raw_chunk_bytes=37)
        variants = []
        over_total = copy.deepcopy(descriptor)
        over_total['raw_bytes'] = M.DEFAULT_RAW_BUDGET + 1
        variants.append(over_total)
        over_raw_chunk = copy.deepcopy(descriptor)
        over_raw_chunk['chunks'][0]['raw_bytes'] = M.MAX_RAW_CHUNK + 1
        variants.append(over_raw_chunk)
        over_gzip_chunk = copy.deepcopy(descriptor)
        over_gzip_chunk['chunks'][0]['gzip_bytes'] = M.MAX_GZIP_CHUNK + 1
        variants.append(over_gzip_chunk)
        for value in variants:
            with self.subTest(value=value), mock.patch.object(M, '_directory_fd') as opened:
                with self.assertRaises(ValueError):
                    M.decode_journal(value, self.output)
                opened.assert_not_called()
        with mock.patch.object(M, '_directory_fd') as opened:
            with self.assertRaises(ValueError):
                M.decode_journal(descriptor, self.output, total_raw_budget=1)
            opened.assert_not_called()

    def test_unsafe_duplicate_and_noncanonical_paths_are_rejected_before_open(self):
        descriptor = self.encode(raw_chunk_bytes=37)
        for path in ('../outside.gz', '/absolute.gz', 'a/../../outside.gz', './chunk.gz',
                     'a//chunk.gz', 'a\\chunk.gz', 'C:chunk.gz', '.', '', 'a\0b'):
            changed = copy.deepcopy(descriptor)
            changed['chunks'][0]['path'] = path
            with self.subTest(path=path), mock.patch.object(M, '_directory_fd') as opened:
                with self.assertRaises(ValueError):
                    M.decode_journal(changed, self.output)
                opened.assert_not_called()
        descriptor['chunks'][1]['path'] = descriptor['chunks'][0]['path']
        with self.assertRaises(ValueError):
            M.decode_journal(descriptor, self.output)

    def test_chunk_root_and_intermediate_symlinks_are_rejected(self):
        descriptor = self.encode()
        chunk = descriptor['chunks'][0]
        actual = self.output / chunk['path']
        original = self.root / 'stored.gz'
        actual.rename(original)
        actual.symlink_to(original)
        with self.assertRaises((OSError, ValueError)):
            M.decode_journal(descriptor, self.output)
        actual.unlink()
        original.rename(actual)
        root_link = self.root / 'root-link'
        root_link.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)):
            M.decode_journal(descriptor, root_link)
        (self.output / 'nested').symlink_to(self.output, target_is_directory=True)
        chunk['path'] = 'nested/' + chunk['path']
        with self.assertRaises((OSError, ValueError)):
            M.decode_journal(descriptor, self.output)

    def test_nonregular_chunk_is_rejected_without_blocking(self):
        descriptor = self.encode()
        path = self.output / descriptor['chunks'][0]['path']
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(ValueError):
            M.decode_journal(descriptor, self.output)

    def test_crc_truncation_trailing_and_concatenated_members_are_rejected(self):
        original = self.encode()
        stream = (self.output / original['chunks'][0]['path']).read_bytes()
        corrupt_crc = bytearray(stream)
        corrupt_crc[-8] ^= 1
        variants = (stream[:-1], bytes(corrupt_crc), stream + b'trailing',
                    stream + gzip.compress(b'', mtime=0), b'not gzip')
        for encoded in variants:
            descriptor = copy.deepcopy(original)
            self.replace_stream(descriptor, encoded)
            with self.subTest(size=len(encoded)), self.assertRaises(ValueError):
                M.decode_journal(descriptor, self.output)

    def test_all_checksum_and_size_bindings_are_enforced(self):
        original = self.encode()
        for where, field, value in (('chunk', 'raw_sha256', '0' * 64),
                                    ('chunk', 'gzip_sha256', '0' * 64),
                                    ('full', 'raw_sha256', '0' * 64),
                                    ('chunk', 'gzip_bytes', original['chunks'][0]['gzip_bytes'] + 1)):
            descriptor = copy.deepcopy(original)
            target = descriptor['chunks'][0] if where == 'chunk' else descriptor
            target[field] = value
            with self.subTest(field=field, where=where), self.assertRaises(ValueError):
                M.decode_journal(descriptor, self.output)
        for difference in (-1, 1):
            descriptor = copy.deepcopy(original)
            descriptor['raw_bytes'] += difference
            descriptor['chunks'][0]['raw_bytes'] += difference
            with self.subTest(difference=difference), self.assertRaises(ValueError):
                M.decode_journal(descriptor, self.output)

    def test_small_declared_raw_size_cannot_expand_a_compression_bomb(self):
        descriptor = self.encode()
        self.replace_stream(descriptor, gzip.compress(b'x' * (2 * 1024 * 1024), mtime=0))
        descriptor['raw_bytes'] = descriptor['chunks'][0]['raw_bytes'] = 1
        with self.assertRaisesRegex(ValueError, 'exceeds declaration'):
            M.decode_journal(descriptor, self.output)

    def test_exact_limit_footer_round_trip(self):
        expected = bytes(range(256)) * 1024
        self.source.write_bytes(expected)
        descriptor = self.encode(raw_chunk_bytes=M._BLOCK, total_raw_budget=len(expected))
        self.assertEqual(M.decode_journal(descriptor, self.output, total_raw_budget=len(expected)), expected)
        self.assertTrue(all(c['raw_bytes'] == M._BLOCK for c in descriptor['chunks']))

    def test_compressed_stream_is_read_with_explicit_bounds_across_blocks(self):
        expected = random.Random(20261110).randbytes(3 * M._BLOCK)
        self.source.write_bytes(expected)
        descriptor = self.encode()
        self.assertGreater(descriptor['chunks'][0]['gzip_bytes'], 2 * M._BLOCK)
        original_open = M._regular_file
        requests = []

        class BoundedFile:
            def __init__(self, file):
                self.file = file

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.file.close()

            def fileno(self):
                return self.file.fileno()

            def read(self, count):
                self_outer.assertGreater(count, 0)
                self_outer.assertLessEqual(count, M._BLOCK)
                requests.append(count)
                return self.file.read(count)

        self_outer = self
        with mock.patch.object(M, '_regular_file', side_effect=lambda *args: BoundedFile(original_open(*args))):
            self.assertEqual(M.decode_journal(descriptor, self.output), expected)
        self.assertGreater(len(requests), 3)

    def test_actual_oversized_compressed_file_is_rejected_before_reads(self):
        descriptor = self.encode()
        path = self.output / descriptor['chunks'][0]['path']
        with path.open('ab') as output:
            output.truncate(M.MAX_GZIP_CHUNK + 1)
        with self.assertRaisesRegex(ValueError, 'Compressed size mismatch'):
            M.decode_journal(descriptor, self.output)

    def test_malformed_descriptor_and_boolean_sizes_are_rejected(self):
        original = self.encode()
        variants = [None, {}, dict(original, schema='other'), dict(original, chunks=[]),
                    dict(original, raw_bytes=True), dict(original, raw_sha256='z' * 64)]
        for field in ('raw_bytes', 'gzip_bytes'):
            descriptor = copy.deepcopy(original)
            descriptor['chunks'][0][field] = True
            variants.append(descriptor)
        for descriptor in variants:
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                M.decode_journal(descriptor, self.output)
        for kwargs in ({'raw_chunk_bytes': 0}, {'raw_chunk_bytes': True},
                       {'raw_chunk_bytes': M.MAX_RAW_CHUNK + 1}, {'total_raw_budget': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.encode(**kwargs)


if __name__ == '__main__':
    unittest.main()
