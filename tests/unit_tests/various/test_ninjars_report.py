"""The optional report download must never install unverified binary inputs."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "ninjars_report_artifacts", ROOT / "doc/benchmarkfolder/fetch_artifacts.py")
ARTIFACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARTIFACTS)


class TestReportArtifacts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.name = "final-report-v1/fonts/example.otf"
        self.data = b"frozen report font"
        self.manifest = {"files": {self.name: {
            "bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest()}}}
        self.patch = mock.patch.object(ARTIFACTS, "ARTIFACTS", (self.name,))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_verified_download_and_offline_retry(self):
        with mock.patch.object(ARTIFACTS.urllib.request, "urlopen",
                               return_value=io.BytesIO(self.data)) as download:
            result = ARTIFACTS.fetch_artifacts(self.root, self.manifest)
        self.assertEqual(result["downloaded"], 1)
        self.assertEqual((self.root / self.name).read_bytes(), self.data)
        download.assert_called_once_with(ARTIFACTS.BASE_URL + self.name, timeout=60)
        with mock.patch.object(ARTIFACTS.urllib.request, "urlopen") as download:
            result = ARTIFACTS.fetch_artifacts(self.root, self.manifest)
        self.assertEqual(result["downloaded"], 0)
        download.assert_not_called()

    def test_wrong_hash_and_partial_download_are_not_installed(self):
        for data in (b"x" * len(self.data), self.data[:-1], self.data + b"extra"):
            with self.subTest(data=data):
                with mock.patch.object(ARTIFACTS.urllib.request, "urlopen",
                                       return_value=io.BytesIO(data)):
                    with self.assertRaisesRegex(ValueError, "failed checksum"):
                        ARTIFACTS.fetch_artifacts(self.root, self.manifest)
                self.assertEqual(list(self.root.rglob("*")), [])

    def test_changed_file_is_not_overwritten(self):
        target = self.root / self.name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user-modified")
        with mock.patch.object(ARTIFACTS.urllib.request, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "move it aside first"):
                ARTIFACTS.fetch_artifacts(self.root, self.manifest)
        download.assert_not_called()
        self.assertEqual(target.read_bytes(), b"user-modified")

    def test_failed_atomic_install_cleans_temporary_file(self):
        with mock.patch.object(ARTIFACTS.urllib.request, "urlopen",
                               return_value=io.BytesIO(self.data)):
            with mock.patch.object(Path, "replace", side_effect=OSError("full")):
                with self.assertRaisesRegex(OSError, "full"):
                    ARTIFACTS.fetch_artifacts(self.root, self.manifest)
        self.assertEqual(list((self.root / self.name).parent.iterdir()), [])

    def test_symlink_cannot_escape_output_root(self):
        outside = self.root / "outside"
        output = self.root / "output"
        outside.mkdir()
        output.mkdir()
        (output / "final-report-v1").symlink_to(outside, target_is_directory=True)
        with mock.patch.object(ARTIFACTS.urllib.request, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "escapes output directory"):
                ARTIFACTS.fetch_artifacts(output, self.manifest)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
