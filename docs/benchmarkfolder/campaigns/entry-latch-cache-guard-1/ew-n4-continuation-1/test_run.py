"""Pure tests of the narrow continuation's admission and schedule."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("test_ew_continuation_source", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class ContinuationTests(unittest.TestCase):
    def test_single_existing_worker_child(self):
        item = run.item()
        self.assertEqual(item["cap_seconds"], 28800.)
        self.assertEqual(item["case"], "ew_n4_epem_mupmumaa")
        self.assertEqual(item["command"][-1], "--lane")
        self.assertNotIn("--backends", item["command"])
        self.assertNotIn("generate", item["command"])

    def test_exact_original_five(self):
        self.assertEqual(run.LANES, [("baseline", 1), ("candidate", 1),
                                    ("candidate", 6), ("candidate", 7), ("candidate", 8)])

    def test_valid_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "x").write_text("frozen")
            run.checked_files({"x": {"sha256": run.digest(root / "x"), "bytes": 6}}, root)

    def test_changed_bytes_and_size_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "x").write_text("frozen")
            for record in ({"sha256": "0" * 64, "bytes": 6},
                           {"sha256": run.digest(root / "x"), "bytes": 7}):
                with self.subTest(record=record), self.assertRaises(ValueError):
                    run.checked_files({"x": record}, root)

    def test_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "x").write_text("frozen")
            (root / "link").symlink_to(root / "x")
            record = {"sha256": run.digest(root / "x"), "bytes": 6}
            for name in ("../escape", "/absolute", "link"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    run.checked_files({name: record}, root)

    def test_unchanged_supervision_pins(self):
        self.assertEqual(run.digest(run.REPORT / "n1-qualification-1/run.py"), run.ADAPTER_SHA)
        self.assertEqual(run.digest(run.REPORT / "remaining-qualification-1/run.py"), run.REMAINING_SHA)


if __name__ == "__main__":
    unittest.main()
