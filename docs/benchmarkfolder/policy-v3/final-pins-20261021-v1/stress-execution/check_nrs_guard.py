#!/usr/bin/env python3
"""NinjaRS stress orchestration guard checks; no numerical engine or compiler."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import collect_nrs_first as controller


class GuardChecks(unittest.TestCase):
    def spec(self, directory):
        path = Path(directory) / "bound-helper.py"
        path.write_text("# frozen harmless helper fixture\n")
        final = controller.adapter.final
        return {"numerical_revision": final.REVISION, "generation_revision": final.REVISION,
            "generator_identity_sha256": final.GENERATOR, "installed_archive_sha256": final.ARCHIVE,
            "sources": {"helper": controller.supplement.bound(path)}}, path

    def test_unrelated_docs_and_tool_commits_preserve_exact_guard(self):
        with tempfile.TemporaryDirectory(dir=controller.adapter.ROOT / ".ninjars-build/tmp") as directory:
            spec, _ = self.spec(directory)
            original = subprocess.check_output
            def git_view(command, **kwargs):
                if command == ["git", "rev-parse", "HEAD"]:
                    return "f" * 40 + "\n"
                if command == ["git", "diff", "--name-only", "-z", controller.adapter.final.REVISION, "HEAD"]:
                    return "docs/unrelated.md\0tools/unimported.py\0"
                return original(command, **kwargs)
            with patch.object(controller.subprocess, "check_output", side_effect=git_view):
                self.assertEqual(controller.stable_orchestration(spec, record=False), "f" * 40)

    def test_bound_helper_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=controller.adapter.ROOT / ".ninjars-build/tmp") as directory:
            spec, path = self.spec(directory)
            path.write_text("# changed helper fixture\n")
            with self.assertRaises((ValueError, RuntimeError)):
                controller.stable_orchestration(spec, record=False)

    def test_GEN_vendor_diff_failure_is_not_ignored(self):
        with tempfile.TemporaryDirectory(dir=controller.adapter.ROOT / ".ninjars-build/tmp") as directory:
            spec, _ = self.spec(directory)
            def dirty(command, **kwargs):
                self.assertIn("madgraph", command)
                self.assertIn("vendor", command)
                raise subprocess.CalledProcessError(1, command)
            with patch.object(controller.subprocess, "run", side_effect=dirty):
                with self.assertRaises(subprocess.CalledProcessError):
                    controller.stable_orchestration(spec, record=False)

    def test_archive_relabeling_is_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory(dir=controller.adapter.ROOT / ".ninjars-build/tmp") as directory:
            spec, _ = self.spec(directory)
            spec["installed_archive_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                controller.stable_orchestration(spec, record=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
