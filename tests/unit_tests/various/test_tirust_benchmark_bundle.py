"""The benchmark bundle is byte-identical to its source and restores safely."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark_bundle as bundle


@unittest.skipUnless(shutil.which("git"), "git is required")
class TestBenchmarkBundle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        workspace = self.source / bundle.WORKSPACE
        (workspace / "campaigns/c-1").mkdir(parents=True)
        (workspace / "README.md").write_text("benchmark\n")
        (workspace / "campaigns/c-1/summary.json").write_text('{"passed": true}\n')
        (workspace / "run.sh").write_text("#!/bin/sh\n")
        (workspace / "run.sh").chmod(0o755)
        (self.source / "other.txt").write_text("not bundled\n")
        for command in (["init", "-q"], ["add", "."],
                        ["-c", "user.name=t", "-c", "user.email=t@example.invalid",
                         "commit", "-q", "-m", "evidence"]):
            subprocess.run(["git", "-C", str(self.source), *command], check=True)
        self.include = self.base / "include"
        (self.include / "tests").mkdir(parents=True)
        (self.include / "tests/test_relocated.py").write_text("# relocated\n")

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, output="out"):
        return bundle.build(self.source, "HEAD", self.base / output, self.include, "b.tar.xz")

    def test_build_is_deterministic_and_byte_identical(self):
        path, digest = self.build()
        self.assertEqual(self.build("again")[1], digest)
        with tarfile.open(path) as archive:
            names = archive.getnames()
            self.assertEqual(archive.extractfile(f"{bundle.WORKSPACE}/README.md").read(),
                             b"benchmark\n")
            self.assertTrue(archive.getmember(f"{bundle.WORKSPACE}/run.sh").mode & 0o111)
            provenance = json.load(archive.extractfile(f"{bundle.WORKSPACE}/{bundle.PROVENANCE}"))
        self.assertNotIn("other.txt", names)
        self.assertIn(f"{bundle.WORKSPACE}/tests/test_relocated.py", names)
        self.assertEqual(provenance["archived_files"], 3)
        self.assertEqual(list(provenance["added_files"]),
                         [f"{bundle.WORKSPACE}/tests/test_relocated.py"])

    def test_include_cannot_replace_archived_files(self):
        (self.include / "README.md").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "must not replace"):
            self.build()

    def test_restore_verifies_keeps_identical_and_refuses_changed_files(self):
        path, digest = self.build()
        target = self.base / "checkout"
        target.mkdir()
        with self.assertRaisesRegex(ValueError, "expected"):
            bundle.restore(target, path, "0" * 64)
        counts = bundle.restore(target, path, digest)
        self.assertEqual(counts, {"written": 6, "replaced": 0, "unchanged": 0})
        self.assertEqual(bundle.verify(target), 5)
        self.assertEqual(bundle.restore(target, path, digest)["unchanged"], 6)
        summary = target / bundle.WORKSPACE / "campaigns/c-1/summary.json"
        summary.write_text("local edit\n")
        with self.assertRaisesRegex(ValueError, "1 existing files differ"):
            bundle.restore(target, path, digest)
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            bundle.verify(target)
        self.assertEqual(bundle.restore(target, path, digest, force=True)["replaced"], 1)
        self.assertEqual(summary.read_text(), '{"passed": true}\n')

    def test_restore_rejects_members_outside_the_workspace(self):
        for name in ("../escape.txt", "docs/other/file.txt", "/abs/file.txt"):
            path = self.base / "bad.tar.xz"
            with tarfile.open(path, "w:xz") as archive:
                info = tarfile.TarInfo(name)
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "unsafe|outside"):
                bundle.restore(self.base, path, bundle.file_sha256(path))
            self.assertFalse((self.base.parent / "escape.txt").exists())

    def restored_campaign(self):
        """Restore a revision bundle in place, then run a "campaign" on top of it."""
        path, digest = self.build()
        bundle.restore(self.source, path, digest)
        workspace = self.source / bundle.WORKSPACE
        (workspace / "README.md").write_text("benchmark, updated\n")
        (workspace / "campaigns/c-2/raw").mkdir(parents=True)
        (workspace / "campaigns/c-2/summary.json").write_text('{"passed": false}\n')
        (workspace / "campaigns/c-2/raw/log.txt").write_text("log\n")
        (workspace / "campaigns/stray.txt").write_text("not listed, not added\n")
        return workspace

    def test_workspace_build_takes_the_base_and_additions_only(self):
        self.restored_campaign()
        path, digest = bundle.build_workspace(self.source, ["campaigns/c-2"], self.base / "w1", "v2.tar.xz")
        self.assertEqual(bundle.build_workspace(self.source, ["campaigns/c-2"], self.base / "w2",
                                                "v2.tar.xz")[1], digest)
        prefix = bundle.WORKSPACE + "/"
        with tarfile.open(path) as archive:
            names = set(archive.getnames())
            provenance = json.load(archive.extractfile(prefix + bundle.PROVENANCE))
            self.assertEqual(archive.extractfile(prefix + "README.md").read(), b"benchmark, updated\n")
        self.assertIn(prefix + "tests/test_relocated.py", names)
        self.assertIn(prefix + "campaigns/c-2/raw/log.txt", names)
        self.assertNotIn(prefix + "campaigns/stray.txt", names)
        self.assertEqual(provenance["source"], "workspace")
        self.assertEqual(provenance["base_bundle"], "b.tar.xz")
        self.assertEqual(list(provenance["changed_files"]), [prefix + "README.md"])
        self.assertEqual(sorted(provenance["added_files"]),
                         [prefix + "campaigns/c-2/raw/log.txt", prefix + "campaigns/c-2/summary.json"])
        target = self.base / "fresh"
        target.mkdir()
        bundle.restore(target, path, digest)
        self.assertEqual(bundle.verify(target), 7)

    def test_workspace_build_refuses_missing_base_files_links_and_outside_paths(self):
        workspace = self.restored_campaign()
        with self.assertRaisesRegex(ValueError, "not a path"):
            bundle.build_workspace(self.source, ["../../other.txt"], self.base / "w")
        with self.assertRaisesRegex(ValueError, "at least one"):
            bundle.build_workspace(self.source, [], self.base / "w")
        (workspace / "campaigns/c-2/link").symlink_to(self.source / "other.txt")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            bundle.build_workspace(self.source, ["campaigns/c-2"], self.base / "w")
        (workspace / "campaigns/c-2/link").unlink()
        (workspace / "run.sh").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            bundle.build_workspace(self.source, ["campaigns/c-2"], self.base / "w")


if __name__ == "__main__":
    unittest.main()
