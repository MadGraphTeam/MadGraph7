"""The benchmark must reject changed EW model sources without repinning them."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "ninjars_setup_references", ROOT / "tools/ninjars_setup_references.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class ReferenceModelVerificationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.model = self.directory / "model"
        self.model.mkdir()
        (self.model / "particles.py").write_bytes(b"particles = []\n")
        (self.model / "__init__.py").write_bytes(b"")
        # Construct the expected byte stream independently of the verifier.
        self.digest = hashlib.sha256(
            b"__init__.py\0particles.py\0particles = []\n").hexdigest()
        self.provenance = self.directory / "provenance.json"
        self.provenance.write_text(json.dumps({"ew_model": {
            "name": "model", "source": "https://example.invalid/pinned-model.tgz",
            "hashed_files": ["__init__.py", "particles.py"],
            "python_sources_sha256": self.digest}}))
        patch = mock.patch.object(setup, "PROVENANCE", self.provenance)
        patch.start()
        self.addCleanup(patch.stop)

    def test_original_digest_and_import_caches(self):
        cache = self.model / "__pycache__"
        cache.mkdir()
        (cache / "particles.cpython-test.pyc").write_bytes(b"generated cache")
        self.assertEqual(setup.verify_ew_model(self.model), self.digest)

    def test_modified_source_does_not_change_provenance(self):
        retained = self.provenance.read_bytes()
        (self.model / "particles.py").write_bytes(b"particles = ['changed']\n")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            setup.verify_ew_model(self.model)
        self.assertEqual(self.provenance.read_bytes(), retained)

    def test_added_and_missing_modules_are_rejected(self):
        (self.model / "parameters.py").write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "added=.*parameters.py"):
            setup.verify_ew_model(self.model)
        (self.model / "parameters.py").unlink()
        (self.model / "particles.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing=.*particles.py"):
            setup.verify_ew_model(self.model)

    def test_verify_only_never_downloads_or_builds(self):
        with mock.patch.object(setup, "BASELINE", self.directory), \
                mock.patch.object(setup, "verify_ew_model", return_value=self.digest) as verify, \
                mock.patch.object(setup, "download") as download, \
                mock.patch.object(setup.subprocess, "run") as run, \
                mock.patch.object(setup.sys, "argv", ["setup", "--verify-model-only"]), \
                mock.patch("builtins.print"):
            setup.main()
        verify.assert_called_once_with()
        download.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
