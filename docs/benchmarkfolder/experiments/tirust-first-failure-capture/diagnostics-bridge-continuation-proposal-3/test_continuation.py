"""Pure source and temporary-file tests; no compiler/process/library loading."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import continue_build as c


class ContinuationTests(unittest.TestCase):
    def test_only_private_cargo_compilation_is_allowed(self):
        for line in ("Fresh symbolica v3", "   Compiling madloop-tirust v0.1.0 (/original/bridge)",
                     "Dirty madloop-tirust v0.1.0: unfinished", "Running `/rustc --crate-name madloop_tirust_private_diag --edition=2024 ...`"):
            self.assertIsNone(c.unexpected_activity(line))
        for line in ("Compiling symbolica v3", "Dirty iregi-rust v0.1: config changed", "Compiling madloop-tirust-other v1",
                     "Running `/path/build-script-build`", "Running `/rustc --crate-name oneloop ...`"):
            self.assertIsNotNone(c.unexpected_activity(line))

    def test_real_independent_copy_preserves_bytes_mode_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);old = root / "old";new = root / "new"
            (old / "target").mkdir(parents=True);source = old / "target/a"
            source.write_bytes(b"completed dependency artifact");source.chmod(0o640)
            os.utime(source, ns=(1000000000, 2000000000))
            before = c.tree_state(old)
            with mock.patch.object(c, "OLD", old):c.import_outputs({"target/a": c.digest(source)}, new)
            self.assertEqual(before, c.tree_state(old))
            self.assertEqual(source.read_bytes(), (new / "target/a").read_bytes())
            self.assertNotEqual(source.stat().st_ino, (new / "target/a").stat().st_ino)
            self.assertEqual(source.stat().st_mtime_ns, (new / "target/a").stat().st_mtime_ns)
            self.assertEqual(source.stat().st_mode, (new / "target/a").stat().st_mode)

    def test_hash_mismatch_symlink_unsafe_or_existing_import_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);old = root / "old";new = root / "new"
            (old / "target").mkdir(parents=True);q = old / "target/a";q.write_bytes(b"A")
            (old / "target/link").symlink_to(q)
            with mock.patch.object(c, "OLD", old):
                for inventory in ({"target/a": "0"*64}, {"target/link": c.digest(q)}, {"../target/a": c.digest(q)}, {"": "0"*64}):
                    with self.assertRaises(ValueError):c.import_outputs(inventory, new)
                c.import_outputs({"target/a": c.digest(q)}, new)
                with self.assertRaises(ValueError):c.import_outputs({"target/a": c.digest(q)}, new)

    def test_guard_detects_content_changes_with_preserved_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);q = root / "a";q.write_bytes(b"A");before = c.tree_state(root)
            q.read_bytes();self.assertEqual(before, c.tree_state(root))
            stamp = q.stat().st_mtime_ns;q.write_bytes(b"B");os.utime(q, ns=(q.stat().st_atime_ns, stamp))
            self.assertNotEqual(before, c.tree_state(root))

    def test_exact_source_references_and_resource_scope(self):
        config = json.loads((c.HERE / "bindings.json").read_text())
        self.assertEqual(config["v2_source_index_sha256"], c.V2_INDEX)
        self.assertEqual(config["imported_inventory"]["files"], 1161)
        self.assertEqual(c.CAP, 17179869184);self.assertEqual(c.CPUS, {16, 17})
        source = Path(c.__file__).read_text()
        self.assertIn('OLD / "bridge", cargo=True', source)
        self.assertIn('"-vv", "--color", "never"', source)
        self.assertNotIn("ctypes", source);self.assertNotIn("ld-linux", source)
        self.assertNotIn("os.link(", source)


if __name__ == "__main__":
    def deny(event, args):
        if event.startswith(("subprocess.", "ctypes.dlopen", "os.exec", "os.spawn", "os.posix_spawn")) or event in ("os.system", "os.fork", "os.forkpty"):
            raise RuntimeError("Native/process execution forbidden in pure tests")
    sys.addaudithook(deny)
    unittest.main(verbosity=2)
