import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('capture_under_test', Path(__file__).with_name('capture.py'))
capture = importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)


class CaptureTests(unittest.TestCase):
    def test_regular_rejects_symlink_and_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / 'file').write_bytes(b'evidence')
            self.assertEqual(capture.regular(root / 'file'), root / 'file')
            (root / 'link').symlink_to('file')
            for path in (root, root / 'link'):
                with self.assertRaises(ValueError): capture.regular(path)

    def test_regular_rejects_ancestor_alias(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / 'real').mkdir(); (root / 'real/file').write_bytes(b'evidence')
            (root / 'alias').symlink_to('real', target_is_directory=True)
            with self.assertRaises(ValueError): capture.regular(root / 'alias/file')

    def test_packet_requires_pinned_index_and_payload(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); data = b'exact'; (root / 'source.py').write_bytes(data)
            index = json.dumps({'files': {'source.py': {'bytes': len(data), 'sha256': capture.sha(data)}}}).encode()
            (root / 'source-index.json').write_bytes(index)
            self.assertEqual(capture.packet(root, capture.sha(index))['files']['source.py']['bytes'], 5)
            with self.assertRaises(ValueError): capture.packet(root, '0' * 64)
            (root / 'source.py').write_bytes(b'wrong')
            with self.assertRaises(ValueError): capture.packet(root, capture.sha(index))

    def test_wrong_admission_fails_before_closed_reads(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with patch.object(capture, 'BASE', root), patch.object(capture, 'packet', side_effect=AssertionError('must not open')):
                with self.assertRaises(ValueError):
                    capture.capture(root / 'fresh', expected_terminal_sha256='0' * 64, root_reported_exit_code=0)
                with self.assertRaises(ValueError):
                    capture.capture(root / 'fresh', expected_terminal_sha256=capture.RUN_SHA, root_reported_exit_code=True)
                self.assertFalse((root / 'fresh').exists())

    def test_destination_alias_or_existing_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); existing = root / 'already'; existing.mkdir()
            with patch.object(capture, 'BASE', root), patch.object(capture, 'packet', side_effect=AssertionError('must not open')):
                with self.assertRaises(ValueError):
                    capture.capture(existing, expected_terminal_sha256=capture.RUN_SHA, root_reported_exit_code=0)
                with self.assertRaises(ValueError):
                    capture.capture(root / 'other' / 'fresh', expected_terminal_sha256=capture.RUN_SHA, root_reported_exit_code=0)


if __name__ == '__main__': unittest.main()
