"""Portable evidence replay and corruption rejection; no native execution."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("adaptive_qp_portable_replay", HERE / "replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class TestReplay(unittest.TestCase):
    def copied(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(shutil.copytree(HERE, Path(temporary.name) / "runtime1"))

    def test_exact_result_without_native_dependencies(self):
        result = replay.replay(self.copied())
        self.assertTrue(result["passed"])
        self.assertFalse(result["native_execution"])
        self.assertEqual(result["genuine_binary128_input_components"], 13)
        self.assertEqual(result["runtime_files"], 36)

    def test_modified_raw_rejected(self):
        root = self.copied()
        file = root / "raw/adaptive/records.txt.gz"
        file.write_bytes(file.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "inventory or SHA-256"):
            replay.replay(root)

    def test_modified_analyzer_rejected(self):
        root = self.copied()
        file = root / "frozen/source-tree/source/analyze.py"
        file.write_text(file.read_text() + "\n# changed analyzer\n")
        with self.assertRaisesRegex(ValueError, "inventory or SHA-256"):
            replay.replay(root)

    def test_missing_or_extra_stream_rejected(self):
        root = self.copied()
        (root / "raw/legacy/observer.jsonl.gz").unlink()
        with self.assertRaisesRegex(ValueError, "inventory or SHA-256"):
            replay.replay(root)
        root = self.copied()
        (root / "extra.txt").write_text("extra retained stream")
        with self.assertRaisesRegex(ValueError, "inventory or SHA-256"):
            replay.replay(root)


if __name__ == "__main__":
    unittest.main()
