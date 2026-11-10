"""Pure replay-admission tests; deny child processes and explicit DSO loading."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import replay
import static_audit as audit


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.observation = json.loads((audit.HERE / "static-observation.json").read_text())

    def test_exact_retained_static_closure(self):
        actual = audit.qualify(self.observation)
        self.assertEqual(actual, json.loads((audit.HERE / "static-qualification.json").read_text()))
        self.assertEqual(len(actual["providers"]), 14)
        self.assertNotIn(self.observation["old_bridge"], actual["providers"])

    def test_wrong_architecture_or_new_dependency_rejected(self):
        name = self.observation["private_dso"]
        for old, new in (("Class:                             ELF64", "Class:                             ELF32"),
                         ("Shared library: [libm.so.6]", "Shared library: [libunreviewed.so]")):
            changed = copy.deepcopy(self.observation)
            self.assertIn(old, changed["objects"][name]["stdout"])
            changed["objects"][name]["stdout"] = changed["objects"][name]["stdout"].replace(old, new, 1)
            with self.assertRaises(ValueError):
                audit.qualify(changed)

    def test_private_symbol_interposition_rejected(self):
        changed = copy.deepcopy(self.observation)
        target = changed["loader"]
        old = changed["objects"][target]["stdout"]
        marker = "Version symbols section"
        self.assertIn(marker, old)
        changed["objects"][target]["stdout"] = old.replace(marker,
            "  999: 0000000000001000 10 FUNC GLOBAL DEFAULT 1 ml_tirust_evaluate_dp\n\n" + marker, 1)
        with self.assertRaises(ValueError):
            audit.qualify(changed)

    def test_only_explicit_helper_adaptations(self):
        source = replay.PILOT.read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), replay.PILOT_SHA)
        module = replay.adapted_pilot()
        self.assertEqual(module.command.__code__.co_filename, str(replay.HERE / "replay.py") + ":adapted:command")
        self.assertEqual(module.runtime_binding.__code__.co_filename, str(replay.HERE / "replay.py") + ":adapted:runtime_binding")
        self.assertIn("16", module.command.__code__.co_consts)
        self.assertIn(16, module.command.__code__.co_consts)
        self.assertNotIn(12, module.command.__code__.co_consts)
        def constants(code):
            for value in code.co_consts:
                if isinstance(value, types.CodeType):
                    yield from constants(value)
                else:
                    yield value
        self.assertIn("libmadloop_tirust_private_diag.so", list(constants(module.runtime_binding.__code__)))

    def test_same_key_alternate_path_and_symlink_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "inputs").mkdir()
            path = root / "inputs/request.dat"
            path.write_bytes(b"same frozen key and data\n")
            files = {str(path): audit.digest(path)}
            manifest = root / "request-manifest.txt"
            manifest.write_text(str(path) + "\n")
            expected = audit.digest(manifest)
            replay.verify_copied_inputs(files, manifest, expected)
            alternate = root / "alternate.dat"
            alternate.write_bytes(path.read_bytes())
            manifest.write_text(str(alternate) + "\n")
            with self.assertRaises(ValueError):
                replay.verify_copied_inputs(files, manifest, expected)
            manifest.unlink()
            target = root / "manifest-target"
            target.write_text(str(path) + "\n")
            manifest.symlink_to(target)
            with self.assertRaises(ValueError):
                replay.verify_copied_inputs(files, manifest, expected)

    def test_input_symlink_and_alias_membership_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            directory = root / "inputs"
            directory.mkdir()
            target = root / "target"
            target.write_bytes(b"unchanged")
            path = directory / "request.dat"
            path.symlink_to(target)
            manifest = root / "request-manifest.txt"
            manifest.write_text(str(path) + "\n")
            with self.assertRaises(ValueError):
                replay.verify_copied_inputs({str(path): audit.digest(target)}, manifest, audit.digest(manifest))
            aliases = root / "providers"
            aliases.mkdir()
            (aliases / "libx.so").symlink_to(target)
            replay.verify_aliases(aliases, {"libx.so": str(target)})
            (aliases / "libextra.so").symlink_to(target)
            with self.assertRaises(ValueError):
                replay.verify_aliases(aliases, {"libx.so": str(target)})

    def test_loader_output_requires_classified_resolved_paths(self):
        target = self.observation["loader"]
        self.assertEqual(replay.loader_paths("linux-vdso.so.1 (0x1)\nlibx.so => " + target + " (0x2)\n"), {str(Path(target).resolve())})
        for text in ("libx.so => not found\n", "unexpected loader warning\n"):
            with self.assertRaises(ValueError):
                replay.loader_paths(text)


if __name__ == "__main__":
    def deny(event, args):
        if event.startswith(("subprocess.", "ctypes.dlopen", "os.exec", "os.spawn", "os.posix_spawn")) or event in ("os.system", "os.fork", "os.forkpty"):
            raise RuntimeError("Native/process execution forbidden in pure tests")
    sys.addaudithook(deny)
    unittest.main(verbosity=2)
