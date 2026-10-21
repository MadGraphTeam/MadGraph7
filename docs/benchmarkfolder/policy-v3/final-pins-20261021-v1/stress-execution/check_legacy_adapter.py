#!/usr/bin/env python3
"""Legacy selector/scope checks only; no compiler or amplitude call."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import collect_legacy_cpu12 as legacy


class LegacyScopeChecks(unittest.TestCase):
    def guard_spec(self, directory):
        helper = Path(directory) / "helper.py"
        helper.write_text("# frozen harmless test helper\n")
        return {"numerical_revision": legacy.adapter.final.REVISION,
            "generation_revision": legacy.adapter.final.REVISION,
            "generator_identity_sha256": legacy.adapter.final.GENERATOR,
            "installed_archive_sha256": legacy.adapter.final.ARCHIVE,
            "sources": {"helper.py": legacy.supplement.bound(helper)}}, helper

    def test_exact_missing_legacy_targets(self):
        self.assertEqual(legacy.selection("ordinary"),
                         {"cpu": 12, "cases": [legacy.EW], "backends": [1, 6, 7]})
        for variant in ("rotation", "scale"):
            self.assertEqual(legacy.selection(variant), {"cpu": 12,
                "cases": [legacy.QCD + "-" + variant, legacy.EW + "-" + variant], "backends": [6]})
        with self.assertRaises(ValueError):
            legacy.selection("unregistered")

    def test_case_scope_preserves_every_other_physical_setting(self):
        for variant in legacy.adapter.final.stress.VARIANTS:
            complete = legacy.ORIGINAL_CONFIGURATION(variant)
            scoped = legacy.scoped_configuration(variant)
            self.assertEqual({k: v for k, v in scoped.items() if k != "cases"},
                             {k: v for k, v in complete.items() if k != "cases"})
            self.assertEqual(scoped["cases"], [case for case in complete["cases"]
                if case["id"] in legacy.selection(variant)["cases"]])

    def test_resealed_ninjars_dispatch_is_rejected(self):
        root = legacy.adapter.ROOT
        with tempfile.TemporaryDirectory(dir=root / ".ninjars-build/tmp") as directory:
            path = Path(directory) / "base.json"
            base = legacy.policy.seal({"schema": legacy.raw.SCHEMA, "variant": "ordinary",
                "configuration": legacy.scoped_configuration("ordinary"), "cpu": 12})
            path.write_text(json.dumps(base))
            selected = legacy.selected_spec(base, legacy.supplement.bound(path))
            self.assertEqual(legacy.validate_selection(selected)["backends"], [1, 6, 7])
            selected["explicit_legacy_only_scope"]["backends"] = [8]
            with self.assertRaises(ValueError):
                legacy.validate_selection(legacy.policy.seal(selected))

    def test_full_cohort_validation_is_restored_even_on_failure(self):
        def full_validator(_path):
            self.assertIs(legacy.raw.stress.configuration, legacy.ORIGINAL_CONFIGURATION)
            raise RuntimeError("expected full-cohort validation control")
        with patch.object(legacy.raw, "cohort", full_validator):
            worker, run_case = legacy.raw.workers.Worker, legacy.raw.workers.run_case
            legacy.configure("ordinary")
            self.assertIs(legacy.raw.workers.Worker, worker)
            self.assertIs(legacy.raw.workers.run_case, run_case)
            with self.assertRaisesRegex(RuntimeError, "expected full-cohort"):
                legacy.raw.cohort("unused")
            self.assertIs(legacy.raw.stress.configuration, legacy.scoped_configuration)
            self.assertEqual(legacy.raw.CPU, 12)
            self.assertEqual(legacy.raw.BACKENDS, (1, 6, 7))

    def test_unrelated_document_and_tool_commits_keep_exact_source_guard(self):
        with tempfile.TemporaryDirectory(dir=legacy.adapter.ROOT / ".ninjars-build/tmp") as directory:
            spec, _ = self.guard_spec(directory)
            original = subprocess.check_output
            def git_view(command, **kwargs):
                if command == ["git", "rev-parse", "HEAD"]:
                    return "f" * 40 + "\n"
                if command == ["git", "diff", "--name-only", "-z", legacy.adapter.final.REVISION, "HEAD"]:
                    return "docs/unrelated-report.md\0tools/unimported-tool.py\0"
                return original(command, **kwargs)
            with patch.object(legacy.subprocess, "check_output", side_effect=git_view):
                self.assertEqual(legacy.stable_orchestration(spec, record=False), "f" * 40)

    def test_bound_helper_mutation_fails_even_if_commit_is_only_documentation(self):
        with tempfile.TemporaryDirectory(dir=legacy.adapter.ROOT / ".ninjars-build/tmp") as directory:
            spec, helper = self.guard_spec(directory)
            helper.write_text("# unexpected replacement\n")
            with self.assertRaises((ValueError, RuntimeError)):
                legacy.stable_orchestration(spec, record=False)

    def test_GEN_or_vendor_diff_failure_is_not_ignored(self):
        with tempfile.TemporaryDirectory(dir=legacy.adapter.ROOT / ".ninjars-build/tmp") as directory:
            spec, _ = self.guard_spec(directory)
            def dirty(command, **kwargs):
                self.assertIn("madgraph", command)
                self.assertIn("vendor", command)
                raise subprocess.CalledProcessError(1, command)
            with patch.object(legacy.subprocess, "run", side_effect=dirty):
                with self.assertRaises(subprocess.CalledProcessError):
                    legacy.stable_orchestration(spec, record=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
