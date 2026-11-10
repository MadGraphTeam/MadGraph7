"""Filesystem-only adversarial tests; fixture bytes are not native programs."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from validate_shim import validate_shim


class ShimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.provider = self.base / "provider"
        self.provider.write_bytes(b"fixture provider, not executable")
        self.shim = self.base / "shim"
        self.shim.mkdir()
        self.alias = self.shim / "libfixture.so.1"
        self.alias.symlink_to(self.provider)
        self.pins = {self.alias.name: {"canonical_path": str(self.provider),
                                    "sha256": hashlib.sha256(self.provider.read_bytes()).hexdigest()}}

    def reject(self):
        with self.assertRaises((ValueError, OSError)):
            validate_shim(self.shim, self.pins)

    def test_positive(self):
        result = validate_shim(self.shim, self.pins)
        self.assertTrue(result["passed"])
        self.assertEqual(result["provider_count"], 1)

    def test_missing(self):
        self.alias.unlink()
        self.reject()

    def test_extra_file(self):
        (self.shim / "extra").write_bytes(b"x")
        self.reject()

    def test_extra_hwcaps_directory(self):
        (self.shim / "glibc-hwcaps").mkdir()
        self.reject()

    def test_regular_copy_even_with_correct_hash(self):
        self.alias.unlink()
        self.alias.write_bytes(self.provider.read_bytes())
        self.reject()

    def test_relative_alias_even_to_correct_provider(self):
        self.alias.unlink()
        self.alias.symlink_to("../provider")
        self.reject()

    def test_dangling_alias(self):
        self.provider.unlink()
        self.reject()

    def test_wrong_provider(self):
        wrong = self.base / "wrong"
        wrong.write_bytes(self.provider.read_bytes())
        self.alias.unlink()
        self.alias.symlink_to(wrong)
        self.reject()

    def test_mutated_provider(self):
        self.provider.write_bytes(b"changed")
        self.reject()

    def test_changed_hash_pin(self):
        self.pins[self.alias.name]["sha256"] = "0" * 64
        self.reject()

    def test_invalid_hash(self):
        self.pins[self.alias.name]["sha256"] = "Z" * 64
        self.reject()

    def test_path_traversal_soname(self):
        self.pins["../libfixture.so.1"] = self.pins.pop(self.alias.name)
        self.reject()

    def test_canonical_path_symlink_rejected(self):
        indirect = self.base / "indirect"
        indirect.symlink_to(self.provider)
        self.pins[self.alias.name]["canonical_path"] = str(indirect)
        self.reject()

    def test_shim_root_symlink_rejected(self):
        indirect = self.base / "shim-indirect"
        indirect.symlink_to(self.shim)
        with self.assertRaises(ValueError):
            validate_shim(indirect, self.pins)

    def test_duplicate_provider_rejected(self):
        (self.shim / "libother.so.1").symlink_to(self.provider)
        self.pins["libother.so.1"] = dict(self.pins[self.alias.name])
        self.reject()

    def test_empty_closure_rejected(self):
        with self.assertRaises(ValueError):
            validate_shim(self.shim, {})

    def test_unknown_metadata_rejected(self):
        self.pins[self.alias.name]["unexpected"] = True
        self.reject()


if __name__ == "__main__":
    unittest.main()
