"""Synthetic source-only report-candidate tests; no actual campaign ingestion."""
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REPORT = module(HERE / "campaign_report.py", "candidate_report_test")
CHECK = module(HERE / "campaign_report_checkpoint.py", "candidate_checkpoint_test")
REFRESH = module(HERE / "campaign_report_refresh.py", "candidate_refresh_test")
RETAIN_TEST = module(DOCS / "campaigns/invariant-symmetry-1/numerical-retention-1/test_retain.py", "synthetic_retainer_fixture")


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


class SyntheticFullCheckpointFixture(RETAIN_TEST.CheckpointFixture):
    """All eleven cases, retaining the real supervisor/config order difference."""
    def _supervisor(self):
        original = self.case
        records = {}
        by_id = {case["id"]: case for case in self.fixture.config["cases"]}
        for case_id in RETAIN_TEST.RET.CASE_ORDER:
            self.case = by_id[case_id]
            if case_id != original["id"]:
                self.fixture.complete_case(self.case)
            super()._supervisor()
            records.update(self.specification["compiled_inputs"])
        self.case = original
        self.specification["compiled_inputs"] = records
        RETAIN_TEST.FIX.dump(self.supervisor / "specification.json", self.specification)

    def __init__(self, parent, terminal_exit=0):
        super().__init__(parent)
        original = self.case
        by_id = {case["id"]: case for case in self.fixture.config["cases"]}
        closed = []
        for case_id in RETAIN_TEST.RET.CASE_ORDER:
            self.case = by_id[case_id]
            closed.append(self._closed_entry())
        self.case = original
        self.confirmation["closed_cases"] = closed
        self.confirmation["terminal_confirmation"] = {
            "session_id": RETAIN_TEST.RET.SESSION, "observed_terminal": True, "exit_code": terminal_exit}
        RETAIN_TEST.FIX.dump(self.confirmation_file, self.confirmation)


class SyntheticFullCheckpointTests(unittest.TestCase):
    def ingest_full(self, parent, terminal_exit=0):
        fixture = SyntheticFullCheckpointFixture(parent, terminal_exit)
        source_index = fixture.package / "source-index.json"
        source = json.loads(source_index.read_bytes())
        source["schema"] = "madloop-tirust-source-only-numerical-retention-proposal-v1"
        write(source_index, source)
        source_sha = hashlib.sha256(source_index.read_bytes()).hexdigest()
        docs = parent / "detached-docs"
        package = docs / "packets/synthetic-full-closed"
        fixture.seal(package)
        for name in ("manifest.json", "cases.json"):
            target = docs / "campaigns/final" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture.campaign / name, target)
        manifest = json.loads((docs / "campaigns/final/manifest.json").read_bytes())
        config = fixture.fixture.config
        expected_order = [case["id"] for case in config["cases"]]
        self.assertNotEqual(expected_order, RETAIN_TEST.RET.CASE_ORDER)
        self.assertEqual(set(expected_order), set(RETAIN_TEST.RET.CASE_ORDER))
        pin = {"index": "packets/synthetic-full-closed/index.json",
               "sha256": hashlib.sha256((package / "index.json").read_bytes()).hexdigest()}
        shutil.rmtree(fixture.root)
        evidence = REPORT.Evidence(docs)
        with patch.object(CHECK, "SOURCE_INDEX_SHA256", source_sha), \
                patch("subprocess.Popen", side_effect=AssertionError("native process forbidden")), \
                patch("ctypes.CDLL", side_effect=AssertionError("native load forbidden")):
            result = CHECK.collect(evidence, pin, manifest, config, require_full=True)
        self.assertFalse(any("experiments/numerical/" in name for name in evidence.files))
        return result

    def test_legitimate_full_eleven_checkpoint_accepts_distinct_configuration_order(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.ingest_full(Path(directory))
        summary = result["summary"]
        self.assertEqual(summary["case_ids"], RETAIN_TEST.RET.CASE_ORDER)
        self.assertEqual(summary["selected_lane_slots"], 66)
        self.assertTrue(summary["full_matrix_closed"])
        self.assertTrue(summary["full_matrix_regression_passed"])
        self.assertEqual(result["numerical"]["status"], "passed")
        self.assertEqual(len(result["numerical"]["cases"]), 11)
        self.assertTrue(all(case["all_six_complete"] and case["regression_pass"]
                            for case in result["numerical"]["cases"]))
        self.assertFalse(summary["performance_acceptance_eligible"])
        self.assertFalse(summary["overall_goal_qualified"])

    def test_full_eleven_passed_lanes_with_failed_supervisor_remain_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.ingest_full(Path(directory), terminal_exit=1)
        self.assertTrue(result["summary"]["full_matrix_closed"])
        self.assertFalse(result["summary"]["full_matrix_regression_passed"])
        self.assertEqual(result["numerical"]["status"], "failed")
        self.assertTrue(all(case["all_six_complete"] and case["regression_pass"]
                            for case in result["numerical"]["cases"]))


class SyntheticCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.fixture = RETAIN_TEST.CheckpointFixture(self.temp)
        source_index = self.fixture.package / "source-index.json"
        source = json.loads(source_index.read_bytes())
        source["schema"] = "madloop-tirust-source-only-numerical-retention-proposal-v1"
        write(source_index, source)
        self.source_sha = hashlib.sha256(source_index.read_bytes()).hexdigest()
        self.docs = self.temp / "detached-docs"
        self.package = self.docs / "packets/synthetic-closed"
        self.fixture.seal(self.package)
        for name in ("manifest.json", "cases.json"):
            target = self.docs / "campaigns/final" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.fixture.campaign / name, target)
        self.manifest = json.loads((self.docs / "campaigns/final/manifest.json").read_bytes())
        self.config = self.fixture.fixture.config
        self.pin = self.index_pin()
        # Erase the original synthetic raw workspace before report ingestion.
        shutil.rmtree(self.fixture.root)
        self.stack.enter_context(patch.object(CHECK, "SOURCE_INDEX_SHA256", self.source_sha))
        self.stack.enter_context(patch("subprocess.Popen", side_effect=AssertionError("native process forbidden")))
        self.stack.enter_context(patch("ctypes.CDLL", side_effect=AssertionError("native load forbidden")))

    def index_pin(self):
        return {"index": "packets/synthetic-closed/index.json",
                "sha256": hashlib.sha256((self.package / "index.json").read_bytes()).hexdigest()}

    def ingest(self, require_full=False):
        evidence = REPORT.Evidence(self.docs)
        result = CHECK.collect(evidence, self.pin, self.manifest, self.config, require_full=require_full)
        self.assertFalse(any("experiments/numerical/" in name for name in evidence.files))
        return result

    def mutate_index(self, change):
        index = json.loads((self.package / "index.json").read_bytes())
        change(index)
        write(self.package / "index.json", index)
        self.pin = self.index_pin()

    def test_relocated_synthetic_checkpoint_replays_without_original_raw_tree(self):
        result = self.ingest()
        self.assertEqual(result["summary"]["selected_case_count"], 1)
        self.assertEqual(result["numerical"]["cases"][0]["complete_lanes"], 6)
        self.assertFalse(result["summary"]["performance_acceptance_eligible"])

    def test_synthetic_checkpoint_replays_with_bytecode_enabled(self):
        with patch.object(sys, "dont_write_bytecode", False):
            result = self.ingest()
        self.assertEqual(result["summary"]["selected_case_count"], 1)

    def test_full_current_report_rejects_partial_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "terminal eleven-case"):
            self.ingest(require_full=True)

    def test_checkpoint_rejects_missing_or_malformed_index_hash_before_read(self):
        for checksum in (None, "", "x" * 64, "A" * 64, "0" * 63):
            with self.subTest(checksum=checksum):
                self.pin["sha256"] = checksum
                with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                    self.ingest()

    def test_unclosed_package_rejected(self):
        self.mutate_index(lambda value: value.update(closed=False))
        with self.assertRaisesRegex(ValueError, "not closed"):
            self.ingest()

    def test_old_campaign_checkpoint_rejected(self):
        self.mutate_index(lambda value: value["campaign_binding"].update(id="old"))
        with self.assertRaisesRegex(ValueError, "another campaign"):
            self.ingest()

    def test_source_pin_substitution_rejected_before_import(self):
        self.mutate_index(lambda value: value.update(source_index_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "qualified source-only"):
            self.ingest()

    def test_index_cannot_escape_package(self):
        self.mutate_index(lambda value: value["files"].update({"../outside": {"sha256": "0"*64, "bytes": 0}}))
        with self.assertRaisesRegex(ValueError, "Unsafe checkpoint"):
            self.ingest()

    def test_rehashed_summary_drift_fails_independent_replay(self):
        summary = json.loads((self.package / "summary.json").read_bytes())
        summary["selected_status"] = "invented-success"
        identity = write(self.package / "summary.json", summary)
        self.mutate_index(lambda value: value["files"].update({"summary.json": identity}))
        with self.assertRaisesRegex(ValueError, "summary differs"):
            self.ingest()

    def test_changed_source_member_rejected(self):
        (self.package / "retain.py").write_bytes(b"raise AssertionError('must never execute substituted source')\n")
        self.mutate_index(lambda value: value["files"].update({"retain.py": {
            "sha256": hashlib.sha256((self.package / "retain.py").read_bytes()).hexdigest(),
            "bytes": (self.package / "retain.py").stat().st_size}}))
        with self.assertRaisesRegex(ValueError, "source closure differs"):
            self.ingest()


class SyntheticSelectionTests(unittest.TestCase):
    def selection(self):
        return {"schema": REFRESH.CURRENT_SCHEMA, "campaign": "synthetic-current",
            "numerical_checkpoint": {"index": "synthetic/index.json", "sha256": "0"*64},
            "closed_cases": ["synthetic"], "documents": {}, "packages": {}, "implementation_packages": {},
            "historical_archive": {"index": REFRESH.ARCHIVE, "sha256": REFRESH.ARCHIVE_SHA},
            "local_handoff_path": "campaigns/invariant-symmetry-1/timing-handoff-3/LOCAL_TIMING_TODO.md",
            "snapshot_date": "2026-11-10", "performance_status": "deferred to Hua-Sheng",
            "final_performance_accepted": False, "current_lifecycle": None}

    def test_old_provisional_or_next_campaign_join_rejected(self):
        for key in ("closed_provisional", "next_campaign", "timings", "paired"):
            selected = self.selection()
            selected[key] = {"synthetic": True}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "no old timing"):
                REFRESH.current_selection(None, selected, {"id": selected["campaign"]}, {})

    def test_failed_terminal_cannot_turn_passed_lanes_into_matrix_success(self):
        summary = {"numerical": {"status": "passed", "cases": ["synthetic only"]},
                   "selected_status": "passed", "full_matrix_closed": True,
                   "supervisor_terminal": {"exit_code": 1}}
        result = CHECK.numerical_outcome(summary)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["cases"], ["synthetic only"])
        self.assertEqual(summary["numerical"]["status"], "passed")

    def test_absent_current_lifecycle_pin_is_pending_without_discovery(self):
        current = module(HERE / "campaign_report_current_lifecycle.py", "synthetic_current_lifecycle")
        evidence = SimpleNamespace(add=lambda *args: self.fail("No selected proof may be read"))
        result = current.collect(evidence, {}, None)
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["evidence"])

    def test_current_lifecycle_rejects_missing_or_malformed_index_hash_before_read(self):
        current = module(HERE / "campaign_report_current_lifecycle.py", "synthetic_lifecycle_hash")
        for checksum in (None, "", "x" * 64, "A" * 64, "0" * 63):
            with self.subTest(checksum=checksum), self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                current.collect(None, {}, {"index": current.INDEX, "sha256": checksum})

    def test_historical_index_cannot_be_selected_as_current_lifecycle(self):
        current = module(HERE / "campaign_report_current_lifecycle.py", "synthetic_current_lifecycle_bad")
        with self.assertRaisesRegex(ValueError, "successor2"):
            current.collect(None, {}, {"index": "experiments/subprocess-cache-lifecycle/evidence-index.json", "sha256": "0"*64})

    def test_current_report_routes_only_to_closed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cases": [{"id": "synthetic_" + str(i)} for i in range(11)]}
            config_identity = write(root / "campaigns/synthetic-current/cases.json", config)
            manifest = {"id": "synthetic-current", "schema": "madloop-tirust-campaign-v1",
                "shared_inputs": {"files": {"cases.json": config_identity["sha256"]}},
                "candidate": {"exporter_identity": {"revision": "synthetic-revision", "files_sha256": "synthetic-source"},
                              "numerical_identity_sha256": "synthetic-numerical"}}
            write(root / "campaigns/synthetic-current/manifest.json", manifest)
            selected = {"schema": REFRESH.CURRENT_SCHEMA, "closed_cases": [r["id"] for r in config["cases"]],
                        "numerical_checkpoint": {"synthetic": True}, "current_lifecycle": None}
            checkpoint = SimpleNamespace(collect=lambda *args: {"numerical": {"status": "failed", "cases": []}})
            refresh = SimpleNamespace(selection=lambda *args: selected,
                current_supplement=lambda *args: {"implementation_supplement": {"strict_installer_promotion": "pending"}})
            modules = {
                "campaign_report_refresh": refresh, "campaign_report_checkpoint": checkpoint,
                "campaign_report_pairs": SimpleNamespace(collect_pairs=lambda *args: {
                    "main_status": "pending", "regression_status": "pending", "status": "pending"}),
                "campaign_report_cohorts": SimpleNamespace(collect_cohorts=lambda *args: {"status": "pending"}),
                "campaign_report_lifecycle": SimpleNamespace(),
                "campaign_report_current_lifecycle": SimpleNamespace(collect=lambda *args: {"status": "pending"})}
            def document(name):
                if name == "campaign_report_numerical":
                    self.fail("Live numerical collector must not be selected")
                return modules[name]
            with patch.object(REPORT, "document_module", side_effect=document), \
                    patch.object(REPORT, "implementation", return_value={"status": "passed"}), \
                    patch.object(REPORT, "statistics_qualification", return_value={"status": "passed"}), \
                    patch.object(REPORT, "historical_stress_limitations", return_value=[
                        {"title": title, "required_for_stress_acceptance": True, "status": "passed"}
                        for title in ("EW threshold mass transport", "QCD soft/collinear prototype", "Complex-mass extremes")]):
                result = REPORT.collect(REPORT.Evidence(root), "synthetic-current", "synthetic-selection.json")
            self.assertEqual(result["criteria"]["all_eleven_five_backend_numerical"], "failed")
            self.assertEqual(result["criteria"]["eight_main_runtime_gates"], "deferred to Hua-Sheng")
            self.assertEqual(result["criteria"]["subprocess_event_cache_lifecycle"], "pending")
            self.assertEqual(result["criteria"]["stress_accuracy"], "open")
            self.assertFalse(result["delivery_complete"])

    def test_historical_stress_uses_pinned_old_manifest_and_stays_historical(self):
        observed = []
        old = {"id": "split-dp-terminal-1", "synthetic": True}
        evidence = SimpleNamespace(json=lambda name, checksum: observed.append((name, checksum)) or old)
        rows = [{"title": "QCD n3 target", "status": "diagnostic", "text": "synthetic retained failure"}]
        with patch.object(REPORT, "stress_limitations", side_effect=lambda reader, manifest: rows if manifest is old else self.fail("Wrong stress manifest")):
            actual = REPORT.historical_stress_limitations(evidence)
        self.assertEqual(observed, [(REPORT.HISTORICAL_STRESS_MANIFEST, REPORT.HISTORICAL_STRESS_MANIFEST_SHA)])
        self.assertEqual(actual[0]["historical_report_campaign"], "split-dp-terminal-1")
        self.assertFalse(actual[0]["current_campaign_stress_qualified"])

    def test_completed_historical_disclosures_are_preserved_without_promotion(self):
        stability = {"cases": 8, "default_dp_failures_retained": 4, "tighter_policy_qp_rescues": 4}
        runtime = {"verification": "pass", "group_calls": 232, "runtime_capture_complete": True,
                   "ordinary_parity_exact": True, "physics_campaign_accepted": False, "production_promoted": False}
        evidence = SimpleNamespace(json=lambda name: stability if "adjudication.json" in name else runtime)
        report = {"implementation": {"library_sha256": "synthetic-installed"}, "stress": [
            {"title": "Default stability estimator", "text": "old not replayed", "current_campaign_stress_qualified": False},
            {"title": "QCD higher precision", "text": "old pending", "current_campaign_stress_qualified": False}]}
        REFRESH.preserve_historical_disclosures(evidence, report)
        self.assertEqual(report["stress"][0]["detail"]["current_native_replay"], "completed")
        self.assertEqual(report["stress"][0]["native_library_sha256"], "synthetic-installed")
        self.assertEqual(report["stress"][1]["status"], "diagnostic")
        self.assertTrue(all(not row["current_campaign_stress_qualified"] for row in report["stress"]))

    def test_current_lifecycle_reuses_exact_source_validator(self):
        identity_tests = module(DOCS / "test_campaign_report_lifecycle.py", "synthetic_lifecycle_identities")
        identities = identity_tests.identities.__wrapped__()
        current = module(HERE / "campaign_report_current_lifecycle.py", "synthetic_lifecycle_identity_check")
        strict = current.legacy()
        self.assertEqual(strict.validate_candidate(*identities)["implementation_revision"], "implementation")
        identities[0]["files"]["Template/cache.f"]["sha256"] = "synthetically changed source"
        with self.assertRaisesRegex(ValueError, "registered production source"):
            strict.validate_candidate(*identities)

    def test_explicit_adaptive_role_routes_before_generic_fortran_parser(self):
        implementation = module(HERE / "campaign_report_implementation.py", "synthetic_role_dispatch")
        diagnostic_result = {"packages": {"adaptive_fixed_replay": {"synthetic": True}},
                             "primitives": {"adaptive_fixed_replay": {"unchecked": True}},
                             "notes": [{"title": "synthetic only"}]}
        diagnostic = SimpleNamespace(ROLES={"adaptive_fixed_replay"}, collect=lambda *args: diagnostic_result)
        with patch.object(implementation, "closed_diagnostics", return_value=diagnostic), \
                patch.object(implementation, "guarded_diagnostic_modules", return_value=[]), \
                patch.object(implementation, "_collect_legacy", side_effect=AssertionError("generic Fortran parser forbidden")):
            result = implementation.collect(None, {"adaptive_fixed_replay": {"synthetic": True}})
        self.assertEqual(result["primitives"]["adaptive_fixed_replay"], {"unchecked": True})
        self.assertFalse(result["performance_acceptance_eligible"])

    def test_guarded_roles_route_before_legacy_and_keep_prior_diagnostics(self):
        implementation = module(HERE / "campaign_report_implementation.py", "synthetic_guarded_route")
        old = SimpleNamespace(ROLES={"adaptive_fixed_replay"}, collect=lambda *args: {
            "packages": {"adaptive_fixed_replay": {"synthetic": "old private"}},
            "primitives": {"adaptive_fixed_replay": {"unchecked": True}},
            "notes": [{"title": "synthetic old adaptive retained"}]})
        def guarded(role):
            return SimpleNamespace(ROLES={role}, collect=lambda *args: {
                "packages": {role: {"synthetic": role}}, "primitives": {role: {"promotion": False}},
                "notes": [{"title": "synthetic " + role}]})
        selected = {role: {"synthetic": True} for role in
                    ("adaptive_fixed_replay", "guarded_adaptive_build", "guarded_adaptive_physical")}
        with patch.object(implementation, "closed_diagnostics", return_value=old), \
                patch.object(implementation, "guarded_diagnostic_modules", return_value=[
                    guarded("guarded_adaptive_build"), guarded("guarded_adaptive_physical")]), \
                patch.object(implementation, "_collect_legacy", side_effect=AssertionError("legacy role parser forbidden")):
            result = implementation.collect(None, selected)
        self.assertEqual(set(result["packages"]), set(selected))
        self.assertEqual(result["notes"][0]["title"], "synthetic old adaptive retained")
        self.assertFalse(result["main_campaign_modified"])
        self.assertFalse(result["performance_acceptance_eligible"])

    def test_false_local_performance_pass_rejected(self):
        selected = self.selection()
        selected["final_performance_accepted"] = True
        with self.assertRaisesRegex(ValueError, "local performance"):
            REFRESH.current_selection(None, selected, {"id": selected["campaign"]}, {})

    def test_canonical_style_and_strict_historical_validators_preserved(self):
        self.assertEqual((HERE / "style.typ").read_bytes(), (DOCS / "style.typ").read_bytes())
        for name in ("campaign_report_lifecycle.py", "campaign_report_provisional.py"):
            self.assertEqual((HERE / name).read_bytes(), (DOCS / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
